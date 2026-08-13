"""
Regression tests for the spec-compliance audit round (see
docs/SPEC_COMPLIANCE_AUDIT.md). Covers:

* P0 ADX sign fix (downtrends no longer read as NaN/flipped DI)
* P0 confirmed-pivot gating in divergence + pattern swing detection
  (no right-edge repainting / lookahead)
* P0 honest pattern labeling (``confidence`` is canonical, ``prob`` is
  a documented legacy alias)
* Regime slope normalization (slope_pct_20, cross-asset comparable)
* P1 backtest statistical significance (bootstrap CIs, PSR, deflated
  Sharpe, Sortino/Calmar, tail loss)
* P1 per-regime performance breakdown
* P2 robustness framework (threshold ablation, parameter sensitivity)
* P2 model governance registry + save hook
* P2 live dedup signal expiry
"""

import tempfile
import unittest
from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from src.features.indicators import adx, add_all_indicators
from src.features.regime import (
    detect_regime,
    get_current_regime_summary,
    multi_timeframe_regime,
)
from src.features.levels import detect_swings
from src.features.divergence import _swing_points as div_swings
from src.features.patterns import detect_patterns, patterns_summary


def _ohlc(closes, vol=1000.0):
    closes = np.asarray(closes, dtype=float)
    n = len(closes)
    df = pd.DataFrame(
        {
            "open": closes,
            "high": closes + 0.2,
            "low": closes - 0.2,
            "close": closes,
            "volume": np.full(n, vol),
        }
    )
    df.index = pd.date_range("2020-01-01", periods=n, freq="D")
    return df


class TestAdxRegression(unittest.TestCase):
    """P0: negative-DM sign bug - ADX must be direction-blind and +DI/-DI
    must point the right way."""

    def test_downtrend_di(self):
        n = 300
        idx = pd.date_range("2020-01-01", periods=n, freq="D")
        closes = pd.Series(np.linspace(100, 60, n), index=idx)
        op = closes.shift(1).fillna(100)
        df = pd.DataFrame(
            {
                "open": op,
                "high": pd.concat([op, closes], axis=1).max(axis=1),
                "low": pd.concat([op, closes], axis=1).min(axis=1),
                "close": closes,
            }
        )
        a = adx(df["high"], df["low"], df["close"])
        self.assertGreater(float(a["adx"].iloc[-1]), 50)  # trend detected
        self.assertLess(float(a["plus_di"].iloc[-1]), 5)  # no up-moves
        self.assertGreater(float(a["minus_di"].iloc[-1]), 50)

    def test_uptrend_di(self):
        n = 300
        idx = pd.date_range("2020-01-01", periods=n, freq="D")
        closes = pd.Series(np.linspace(60, 120, n), index=idx)
        op = closes.shift(1).fillna(60)
        df = pd.DataFrame(
            {
                "open": op,
                "high": pd.concat([op, closes], axis=1).max(axis=1),
                "low": pd.concat([op, closes], axis=1).min(axis=1),
                "close": closes,
            }
        )
        a = adx(df["high"], df["low"], df["close"])
        self.assertGreater(float(a["adx"].iloc[-1]), 50)
        self.assertGreater(float(a["plus_di"].iloc[-1]), 50)
        self.assertLess(float(a["minus_di"].iloc[-1]), 5)


def _zigzag():
    """up to a peak, down to a trough, up again - a clean interior swing
    pair (peak at bar 15, trough at bar 31) plus a fresh up-leg."""
    c = []
    c += list(np.linspace(100, 105, 16))  # 0..15  (peak at 15)
    c += list(np.linspace(105, 97, 16))  # 16..31  (trough at 31)
    c += list(np.linspace(97, 104, 8))  # 32..39
    return c


class TestConfirmedPivotGating(unittest.TestCase):
    """P0: signal generation must only use pivots whose centred detection
    window is fully inside the data - no trailing unconfirmed pivots, so
    a reported divergence/pattern never depends on a swing that could
    repaint when the next bar prints."""

    def test_divergence_excludes_trailing_unconfirmed_pivot(self):
        df = _ohlc(_zigzag())
        pts = div_swings(df, left=2, right=2)
        self.assertGreater(len(pts), 0)
        kinds = [k for _, _, k in pts]
        self.assertIn("high", kinds)  # the interior peak is found
        for i, _, kind in pts:
            self.assertLess(i, len(df) - 2, f"unconfirmed {kind} at bar {i}")

    def test_patterns_exclude_trailing_unconfirmed_pivot(self):
        df = _ohlc(_zigzag())
        from src.features.patterns import _swing_points

        pts = _swing_points(df, left=3, right=3)
        self.assertGreater(len(pts), 0)
        for i, _, kind in pts:
            self.assertLess(i, len(df) - 3, f"unconfirmed {kind} at bar {i}")

    def test_detect_swings_marks_interior_pivot(self):
        # the gating lives in the consumers; detect_swings itself still
        # marks the (interior) peak. The trailing bars are NaN by the
        # centered-window semantics (never pivots) - the consumer gating
        # makes that boundary explicit and future-proof.
        closes = _zigzag()
        df = _ohlc(closes)
        marked = detect_swings(df.reset_index(drop=True), 2, 2)
        self.assertTrue(bool(marked["swing_high"].iloc[15]))
        self.assertFalse(bool(marked["swing_high"].iloc[-1]))


class TestPatternConfidenceHonesty(unittest.TestCase):
    """P0: the pattern score is a structural confidence, not a calibrated
    probability - it must be labeled ``confidence`` (with ``prob`` kept as
    a documented alias) and never change value between the two keys."""

    def _hss(self):
        c = []
        c += list(np.linspace(95, 100, 8))
        c += list(np.linspace(100, 97, 4))
        c += list(np.linspace(97, 105, 5))
        c += list(np.linspace(105, 97.5, 5))
        c += list(np.linspace(97.5, 100, 4))
        c += list(np.linspace(100, 95, 6))
        return c

    def test_confidence_key_is_canonical_and_matches_prob(self):
        pats = detect_patterns(_ohlc(self._hss()))
        self.assertTrue(any(p["name"] == "Head & Shoulders" for p in pats))
        for p in pats:
            self.assertIn("confidence", p)
            self.assertIn("prob", p)  # legacy alias still present
            self.assertEqual(p["confidence"], p["prob"])
            self.assertGreaterEqual(p["confidence"], 65)

    def test_summary_exposes_confidence(self):
        s = patterns_summary(_ohlc(self._hss()))
        if s["patterns"]:
            self.assertIn("confidence", s["best"])


class TestRegimeSlopeNormalization(unittest.TestCase):
    """Regression slope must be comparable across assets (per-bar % of
    price), not raw points/bar."""

    @classmethod
    def setUpClass(cls):
        rng = np.random.default_rng(3)
        n = 300
        idx = pd.date_range("2019-01-01", periods=n, freq="D")
        closes = 100 + np.cumsum(rng.normal(0.02, 0.4, n))
        df = pd.DataFrame(
            {
                "open": closes,
                "high": closes + 0.3,
                "low": closes - 0.3,
                "close": closes,
                "volume": np.full(n, 1000.0),
            }
        )
        df.index = idx
        cls.df = add_all_indicators(df)

    def test_slope_pct_column_present(self):
        out = detect_regime(self.df)
        self.assertIn("slope_pct_20", out.columns)
        ratio = out["slope_pct_20"] / (out["slope_20"] / out["close"] * 100)
        self.assertTrue(np.allclose(ratio.dropna(), 1.0, atol=1e-6))

    def test_summary_has_normalized_slope(self):
        out = detect_regime(self.df)
        s = get_current_regime_summary(out)
        self.assertIn("slope_pct_20", s)
        self.assertAlmostEqual(
            s["slope_pct_20"], float(out["slope_pct_20"].iloc[-1]), places=4
        )

    def test_mtf_rows_include_normalized_slope(self):
        out = detect_regime(self.df)
        mtf = multi_timeframe_regime(out)
        self.assertTrue(all("slope_pct_20" in r for r in mtf["rows"]))


class TestBacktestSignificance(unittest.TestCase):
    """P1: the backtest must report whether its edge is distinguishable
    from noise (bootstrap CIs, PSR, deflated Sharpe, Sortino/Calmar)."""

    def _run(self, n=400, every=10):
        from src.backtest.engine import BacktestParams, run_backtest

        rng = np.random.default_rng(5)
        idx = pd.date_range("2015-01-01", periods=n, freq="D")
        closes = 100 + np.cumsum(rng.normal(0.0, 0.3, n))
        df = pd.DataFrame(
            {
                "open": closes,
                "high": closes * 1.02,
                "low": closes * 0.98,
                "close": closes,
                "volume": np.full(n, 1000.0),
            }
        )
        df.index = idx
        df = add_all_indicators(df)

        confirmed = np.zeros(n, dtype=bool)
        entry_lo = np.full(n, np.nan)
        entry_hi = np.full(n, np.nan)
        invalidation = np.full(n, np.nan)
        resistance = np.full(n, np.nan)
        score = np.zeros(n)
        for i in range(50, n - 1, every):
            confirmed[i] = True
            entry_lo[i] = closes[i] * 0.995
            entry_hi[i] = closes[i] * 1.005
            invalidation[i] = closes[i] * 0.99
            resistance[i] = closes[i] * 1.01
            score[i] = 6
        signal = pd.DataFrame(
            {
                "confirmed": confirmed,
                "entry_lo": entry_lo,
                "entry_hi": entry_hi,
                "invalidation": invalidation,
                "resistance": resistance,
                "score": score,
            },
            index=idx,
        )
        params = BacktestParams(entry_type="market", max_hold=15, n_trials=20)
        return run_backtest(signal, df, params, symbol="T"), df

    def test_stats_include_significance_block(self):
        res, _ = self._run()
        s = res.stats
        self.assertGreaterEqual(s["n_trades"], 10)
        self.assertIn("sortino", s)
        self.assertIn("calmar", s)
        self.assertIn("tail_loss_pct", s)
        self.assertIn("expectancy_ci95", s)
        self.assertIn("win_rate_ci95", s)
        for ci in (s["expectancy_ci95"], s["win_rate_ci95"]):
            self.assertTrue(ci[0] is None or ci[0] <= ci[1])
        if s["psr_positive"] is not None:
            self.assertTrue(0.0 <= s["psr_positive"] <= 1.0)
        if s["deflated_sharpe"] is not None:
            self.assertTrue(0.0 <= s["deflated_sharpe"] <= 1.0)

    def test_few_trades_yields_null_ci(self):
        from src.backtest.engine import BacktestParams, run_backtest

        n = 60
        idx = pd.date_range("2020-01-01", periods=n, freq="D")
        closes = np.linspace(100, 103, n)
        df = pd.DataFrame(
            {
                "open": closes,
                "high": closes + 0.1,
                "low": closes - 0.1,
                "close": closes,
                "volume": np.full(n, 1000.0),
            }
        )
        df.index = idx
        df = add_all_indicators(df)
        confirmed = np.zeros(n, dtype=bool)
        confirmed[30] = True
        entry_lo = np.full(n, np.nan)
        entry_lo[30] = 99.0
        entry_hi = np.full(n, np.nan)
        entry_hi[30] = 101.0
        invalidation = np.full(n, np.nan)
        invalidation[30] = 98.0
        resistance = np.full(n, np.nan)
        resistance[30] = 102.0
        signal = pd.DataFrame(
            {
                "confirmed": confirmed,
                "entry_lo": entry_lo,
                "entry_hi": entry_hi,
                "invalidation": invalidation,
                "resistance": resistance,
                "score": np.zeros(n),
            },
            index=idx,
        )
        res = run_backtest(signal, df, BacktestParams(entry_type="market"), symbol="T")
        self.assertLess(res.stats["n_trades"], 10)
        self.assertTrue(res.stats["expectancy_ci95"][0] is None)

    def test_regime_breakdown(self):
        from src.backtest.engine import regime_breakdown

        res, df = self._run()
        reg = detect_regime(df)
        rows = regime_breakdown(res, reg["regime"])
        self.assertGreater(len(rows), 0)
        total = sum(r["n"] for r in rows.values())
        self.assertEqual(total, res.stats["n_trades"])
        for r in rows.values():
            self.assertIn("win_rate", r)
            self.assertIn("avg_r", r)


class TestRobustnessFramework(unittest.TestCase):
    """P2: threshold ablation + parameter sensitivity."""

    def _frame(self, n=300):
        rng = np.random.default_rng(11)
        idx = pd.date_range("2016-01-01", periods=n, freq="D")
        closes = 100 + np.cumsum(rng.normal(0.0, 0.3, n))
        df = pd.DataFrame(
            {
                "open": closes,
                "high": closes * 1.02,
                "low": closes * 0.98,
                "close": closes,
                "volume": np.full(n, 1000.0),
            }
        )
        df.index = idx
        df = add_all_indicators(df)
        confirmed = np.zeros(n, dtype=bool)
        entry_lo = np.full(n, np.nan)
        entry_hi = np.full(n, np.nan)
        invalidation = np.full(n, np.nan)
        resistance = np.full(n, np.nan)
        score = np.zeros(n)
        for i in range(50, n - 1, 10):
            confirmed[i] = True
            entry_lo[i] = closes[i] * 0.995
            entry_hi[i] = closes[i] * 1.005
            invalidation[i] = closes[i] * 0.99
            resistance[i] = closes[i] * 1.01
            score[i] = 7 if i % 20 == 0 else 5
        signal = pd.DataFrame(
            {
                "confirmed": confirmed,
                "entry_lo": entry_lo,
                "entry_hi": entry_hi,
                "invalidation": invalidation,
                "resistance": resistance,
                "score": score,
            },
            index=idx,
        )
        return signal, df

    def test_ablate_threshold_table(self):
        from src.backtest.engine import BacktestParams
        from src.backtest.robustness import ablate_threshold

        signal, df = self._frame()
        rows = ablate_threshold(
            signal, df, BacktestParams(entry_type="market"), thresholds=(0, 5, 7)
        )
        self.assertEqual(len(rows), 3)
        self.assertTrue(
            {"threshold", "n_trades", "win_rate", "expectancy_r"}.issubset(rows.columns)
        )
        # higher floors must not increase trade count
        self.assertEqual(list(rows["n_trades"]), sorted(rows["n_trades"], reverse=True))

    def test_param_sensitivity_table(self):
        from src.backtest.engine import BacktestParams
        from src.backtest.robustness import param_sensitivity

        signal, df = self._frame()
        rows = param_sensitivity(
            signal,
            df,
            BacktestParams(entry_type="market"),
            param="slippage",
            values=(0.0, 0.001, 0.002),
        )
        self.assertEqual(len(rows), 3)
        self.assertIn("param_value", rows.columns)

    def test_param_sensitivity_rejects_unknown_field(self):
        from src.backtest.engine import BacktestParams
        from src.backtest.robustness import param_sensitivity

        signal, df = self._frame()
        with self.assertRaises(ValueError):
            param_sensitivity(signal, df, BacktestParams(), param="not_a_field")


class TestClusterLabelAlignment(unittest.TestCase):
    """The KMeans fallback must return FULL-LENGTH labels in row order
    (the HMM twin was fixed in the audit round; the cluster path fell
    through to the reduced-index alignment and mislabeled warm-up rows).
    Every code path must satisfy the len(df) contract."""

    def _frame(self, n=300):
        df = _ohlc([100 + 0.15 * i for i in range(n)])
        return add_all_indicators(df)

    def test_short_history_fallback_is_full_length_row_order(self):
        from src.features.regime import cluster_regime_labels

        # too short for clustering -> deterministic fallback path
        df = detect_regime(self._frame(n=50))
        labels = cluster_regime_labels(df, n_clusters=4)
        self.assertEqual(len(labels), len(df))
        self.assertEqual(labels, df["regime"].tolist())  # row order kept

    def test_no_regime_column_fallback_is_neutral_full_length(self):
        from src.features.regime import cluster_regime_labels

        # indicators present but no ``regime`` column (detect_regime was
        # never called) -> neutral full-length fallback, not a crash
        df = self._frame(n=50)
        labels = cluster_regime_labels(df, n_clusters=4)
        self.assertEqual(len(labels), len(df))
        self.assertTrue(all(lab == "Range / Chop" for lab in labels))

    def test_fitted_path_still_full_length(self):
        from src.features.regime import REGIME_LEVELS, cluster_regime_labels

        df = self._frame(n=300)
        labels = cluster_regime_labels(df, n_clusters=4)
        self.assertEqual(len(labels), len(df))
        self.assertTrue(all(lab in REGIME_LEVELS for lab in labels))


class TestModelRegistry(unittest.TestCase):
    """P2: model governance - every saved model lands in the ledger."""

    def test_record_and_latest(self):
        from src.model.registry import entries_for, latest, record

        with tempfile.TemporaryDirectory() as tmp:
            reg = f"{tmp}/registry.json"
            meta = {
                "auc_oos": 0.55,
                "n_samples": 1000,
                "symbols": 13,
                "best_params": {"lr": 0.04},
            }
            rec = record("models/x.joblib", meta, registry_path=reg)
            self.assertIsNotNone(rec)
            self.assertEqual(rec["auc_oos"], 0.55)
            record("models/y.joblib", {"auc_oos": 0.6}, registry_path=reg)
            self.assertEqual(len(entries_for("models/x.joblib", reg)), 1)
            self.assertEqual(latest("models/y.joblib", reg)["auc_oos"], 0.6)

    def test_save_model_writes_registry(self):
        from src.model.model import save_model

        with tempfile.TemporaryDirectory() as tmp:
            path = f"{tmp}/m.joblib"
            save_model({"dummy": 1}, ["a", "b"], {"auc_oos": 0.51}, path)
            from src.model.registry import entries_for

            recs = entries_for(path)
            self.assertEqual(len(recs), 1)
            self.assertEqual(recs[0]["auc_oos"], 0.51)

    def test_prune_tmp_entries(self):
        from src.model.registry import all_entries, prune_tmp_entries, record

        with tempfile.TemporaryDirectory() as tmp:
            reg = f"{tmp}/registry.json"
            record("/tmp/tmpabc/m.joblib", {"auc_oos": 0.4}, registry_path=reg)
            record("/tmp/tmpdef/m.joblib", {"auc_oos": 0.5}, registry_path=reg)
            record("models/real.joblib", {"auc_oos": 0.6}, registry_path=reg)
            removed = prune_tmp_entries(reg)
            self.assertEqual(removed, 2)
            remaining = all_entries(reg)
            self.assertEqual(len(remaining), 1)
            self.assertEqual(remaining[0]["model"], "models/real.joblib")

    def test_prune_tmp_noop_when_clean(self):
        from src.model.registry import prune_tmp_entries, record

        with tempfile.TemporaryDirectory() as tmp:
            reg = f"{tmp}/registry.json"
            record("models/real.joblib", {"auc_oos": 0.6}, registry_path=reg)
            self.assertEqual(prune_tmp_entries(reg), 0)


class TestSignalExpiry(unittest.TestCase):
    """P2: dedup state must expire so a stale setup becomes re-eligible."""

    def test_purge_expired(self):
        from src.live.signals import purge_expired

        now = datetime.now(timezone.utc)
        fresh = (now - timedelta(days=1)).strftime("%Y-%m-%dT%H:%M:%SZ")
        old = (now - timedelta(days=60)).strftime("%Y-%m-%dT%H:%M:%SZ")
        seen = {"FRESH", "OLD", "NO_TS"}
        sent_at = {"FRESH": fresh, "OLD": old, "NO_TS": None}
        out_seen, out_ts = purge_expired(seen, sent_at, max_age_days=30)
        self.assertIn("FRESH", out_seen)
        self.assertIn("NO_TS", out_seen)  # unknown age never expires
        self.assertNotIn("OLD", out_seen)

    def test_state_roundtrip_keeps_timestamps(self):
        from src.live.signals import load_state, load_state_with_meta, save_state

        with tempfile.TemporaryDirectory() as tmp:
            path = f"{tmp}/alerts.json"
            save_state({"K1"}, path, sent_at={"K1": "2026-01-01T00:00:00Z"})
            self.assertEqual(load_state(path), {"K1"})
            seen, ts = load_state_with_meta(path)
            self.assertEqual(ts.get("K1"), "2026-01-01T00:00:00Z")
            # legacy format (no sent_at) still loads
            Path(path).write_text('{"seen": ["K2"]}')
            self.assertEqual(load_state(path), {"K2"})


if __name__ == "__main__":
    unittest.main()
