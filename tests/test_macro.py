"""
Tests for the Macro Overlay (src/macro/overlay.py + src/macro/run.py):
factor scoring, causality of the align, per-symbol bias mapping, the
gate logic, the causal gate series, and the with-vs-without backtest.

Most tests use synthetic daily frames so they are deterministic and
offline; a few use the local candidates group for integration.
"""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.append(str(Path(__file__).parent.parent))

from src.macro.overlay import (
    _symbol_class,
    align_scores,
    factor_scores,
    gate_series,
    macro_bias_for_symbol,
    macro_gate,
    macro_regime,
)
from src.macro.run import compare_gate

TEST_GROUP, TEST_TF = "candidates", "D1"


def _daily_series(values, start="2024-01-01"):
    return pd.DataFrame(
        {"close": values},
        index=pd.date_range(start, periods=len(values), freq="B"),
    )


def _scores_frame(n=300, dxy_trend=1.0, vix_trend=-1.0, tnx_trend=0.5):
    """Synthetic macro frame: trending series producing known scores."""
    idx = pd.date_range("2024-01-01", periods=n, freq="B")
    # Slopes are strong enough that the 20-day change trips the +/-2
    # thresholds (e.g. 20d change of 0.12*20 = 2.4 > 2).
    dxy = 100 + np.arange(n) * dxy_trend * 0.12
    vix = 25 - np.arange(n) * vix_trend * 0.12
    tnx = 4.0 + np.arange(n) * tnx_trend * 0.005
    frame = pd.DataFrame({"dxy": dxy, "vix": vix, "tnx": tnx}, index=idx)
    return factor_scores(frame)


class TestFactorScores(unittest.TestCase):
    def test_uptrending_dxy_is_positive(self):
        s = _scores_frame(dxy_trend=1.0)
        self.assertTrue(s["dxy_score"].tail(50).mean() > 0)

    def test_downtrending_dxy_is_negative(self):
        s = _scores_frame(dxy_trend=-1.0)
        self.assertTrue(s["dxy_score"].tail(50).mean() < 0)

    def test_scores_in_range(self):
        s = _scores_frame()
        for col in ["dxy_score", "vix_score", "tnx_score"]:
            valid = s[col].dropna()
            self.assertTrue(valid.between(-2, 2).all(), col)

    def test_rising_vix_risk_off(self):
        # vix = 25 - i*vix_trend*0.01 -> rising VIX needs vix_trend=-1.
        s = _scores_frame(vix_trend=-1.0)
        self.assertTrue(s["vix_score"].tail(50).mean() < 0)

    def test_falling_vix_risk_on(self):
        s = _scores_frame(vix_trend=1.0)
        self.assertTrue(s["vix_score"].tail(50).mean() > 0)

    def test_warmup_rows_dropped(self):
        s = _scores_frame(n=400)
        # The dxy score (SMA200 + RSI) is only defined after the 200-bar
        # warm-up; the VIX/TNX scores start earlier and keep those rows.
        first_dxy = s["dxy_score"].dropna().index[0]
        self.assertGreaterEqual((first_dxy - pd.Timestamp("2024-01-01")).days, 250)
        # dxy rows before that point are NaN, not spurious zeros.
        early = s.loc[s.index < first_dxy, "dxy_score"]
        self.assertTrue(early.isna().all())


class TestCausality(unittest.TestCase):
    def test_align_shifts_by_one_day(self):
        # Scores known on day D become actionable only on day D+1.
        scores = pd.DataFrame(
            {"dxy_score": [1.0, 2.0]},
            index=pd.to_datetime(["2024-01-01", "2024-01-02"]),
        )
        idx = pd.to_datetime(["2024-01-01", "2024-01-02", "2024-01-03"])
        aligned = align_scores(scores, idx, shift_days=1)
        self.assertTrue(np.isnan(aligned.loc["2024-01-01", "dxy_score"]))
        self.assertEqual(aligned.loc["2024-01-02", "dxy_score"], 1.0)
        self.assertEqual(aligned.loc["2024-01-03", "dxy_score"], 2.0)

    def test_gate_series_is_boolean_and_length_matches(self):
        scores = _scores_frame(n=300)
        idx = pd.date_range("2024-06-01", periods=100, freq="B")
        g = gate_series("EURUSD", scores, idx)
        self.assertEqual(len(g), len(idx))
        self.assertTrue(g.dtype == bool)

    def test_gate_series_blocks_strong_headwind_symbol(self):
        # DXY strongly up -> EURUSD (USD-quoted) should be mostly blocked.
        scores = _scores_frame(n=300, dxy_trend=1.0, vix_trend=-1.0)
        idx = pd.date_range("2025-01-01", periods=60, freq="B")
        g = gate_series("EURUSD", scores, idx)
        # Note: early rows are pre-macro (NaN -> allowed True); the tail
        # of the sample must contain blocked bars when USD is strong.
        tail = g.tail(20)
        self.assertFalse(
            tail.all(),
            "EURUSD should be blocked at some point with a strongly rising DXY",
        )

    def test_empty_scores_means_all_allowed(self):
        idx = pd.date_range("2024-01-01", periods=20, freq="B")
        g = gate_series("EURUSD", pd.DataFrame(), idx)
        self.assertTrue(g.all())


class TestSymbolBias(unittest.TestCase):
    def _row(self, dxy=1.0, vix=-1.0, tnx=1.0):
        return {"dxy_score": dxy, "vix_score": vix, "tnx_score": tnx}

    def test_usd_quoted_pair_gets_headwind(self):
        b = macro_bias_for_symbol("EURUSD", self._row(dxy=2))
        self.assertLess(b["bias"], 0)
        self.assertIn("headwind", b["note"].lower())

    def test_usd_based_pair_gets_tailwind(self):
        b = macro_bias_for_symbol("USDJPY", self._row(dxy=2))
        self.assertGreater(b["bias"], 0)
        self.assertIn("tailwind", b["note"].lower())

    def test_metal_anti_dollar(self):
        b = macro_bias_for_symbol("XAUUSD", self._row(dxy=2))
        self.assertLess(b["bias"], 0)
        # Weak dollar -> gold tailwind
        b2 = macro_bias_for_symbol("XAUUSD", self._row(dxy=-2))
        self.assertGreater(b2["bias"], 0)

    def test_crypto_risk_sensitive(self):
        # Positive vix_score = risk-on -> crypto tailwind.
        b = macro_bias_for_symbol("BTCUSD", self._row(vix=2))
        self.assertGreater(b["bias"], 0)
        b2 = macro_bias_for_symbol("BTCUSD", self._row(vix=-2))
        self.assertLess(b2["bias"], 0)

    def test_index_risk_on_positive(self):
        b = macro_bias_for_symbol("US500", self._row(vix=2))
        self.assertGreater(b["bias"], 0)
        b2 = macro_bias_for_symbol("US500", self._row(vix=-2))
        self.assertLess(b2["bias"], 0)

    def test_commodity_is_not_crypto(self):
        # Regression: XALUSD/XNGUSD must classify as commodities (they were
        # being swallowed by the crypto catch-all), and copper is not
        # strongly risk-on like BTC.
        self.assertEqual(_symbol_class("XALUSD")[0], "commodity")
        self.assertEqual(_symbol_class("XNGUSD")[0], "commodity")
        self.assertEqual(_symbol_class("XAUUSD")[0], "metal")
        self.assertEqual(_symbol_class("XRPUSD")[0], "crypto")
        b = macro_bias_for_symbol("XALUSD", self._row(vix=2))
        self.assertGreater(b["bias"], 0)
        self.assertIn("pro-cyclical", b["note"])

    def test_bias_in_range(self):
        for sym in ["EURUSD", "USDJPY", "XAUUSD", "BTCUSD", "US500", "AAPL"]:
            b = macro_bias_for_symbol(sym, self._row())
            self.assertTrue(-2 <= b["bias"] <= 2, sym)
            self.assertIn("label", b)

    def test_symbol_class(self):
        self.assertEqual(_symbol_class("EURUSD")[0], "fx")
        self.assertEqual(_symbol_class("USDJPY")[0], "fx")
        self.assertEqual(_symbol_class("EURGBP")[0], "fx")
        self.assertEqual(_symbol_class("XAUUSD")[0], "metal")
        self.assertEqual(_symbol_class("BTCUSD")[0], "crypto")
        self.assertEqual(_symbol_class("US500")[0], "index")
        self.assertEqual(_symbol_class("AAPL")[0], "equity")


class TestGate(unittest.TestCase):
    def test_strong_headwind_blocked(self):
        g = macro_gate(
            "EURUSD", {"dxy_score": 2, "vix_score": 0, "tnx_score": 0}, min_bias=-0.5
        )
        self.assertFalse(g["allowed"])

    def test_neutral_allowed(self):
        g = macro_gate("EURUSD", {"dxy_score": 0, "vix_score": 0, "tnx_score": 0})
        self.assertTrue(g["allowed"])

    def test_tailwind_allowed(self):
        g = macro_gate("USDJPY", {"dxy_score": 2, "vix_score": 0, "tnx_score": 0})
        self.assertTrue(g["allowed"])

    def test_threshold_tightens(self):
        row = {"dxy_score": 1, "vix_score": 0, "tnx_score": 0}
        # EURUSD bias -1.0 passes at -1.5, blocked at -0.5.
        self.assertTrue(macro_gate("EURUSD", row, min_bias=-1.5)["allowed"])
        self.assertFalse(macro_gate("EURUSD", row, min_bias=-0.5)["allowed"])

    def test_regime_labels(self):
        r = macro_regime({"dxy_score": 1, "vix_score": -1, "tnx_score": 0})
        self.assertEqual(r["usd"], "USD Bullish")
        self.assertEqual(r["risk"], "Risk-Off")
        self.assertEqual(r["rates"], "Neutral")

    def test_short_gate_allows_headwind(self):
        # A strong dollar is a headwind for a EURUSD LONG but a tailwind
        # for a EURUSD SHORT - the short-direction gate must allow it.
        row = {"dxy_score": 2, "vix_score": 0, "tnx_score": 0}
        self.assertFalse(macro_gate("EURUSD", row, direction="long")["allowed"])
        self.assertTrue(macro_gate("EURUSD", row, direction="short")["allowed"])

    def test_short_gate_blocks_tailwind(self):
        # A strong dollar is a tailwind for a USDJPY LONG and argues
        # against fading it - the short-direction gate must block it.
        row = {"dxy_score": 2, "vix_score": 0, "tnx_score": 0}
        self.assertTrue(macro_gate("USDJPY", row, direction="long")["allowed"])
        self.assertFalse(macro_gate("USDJPY", row, direction="short")["allowed"])

    def test_long_gate_default_is_long(self):
        row = {"dxy_score": 2, "vix_score": 0, "tnx_score": 0}
        self.assertEqual(
            macro_gate("EURUSD", row)["allowed"],
            macro_gate("EURUSD", row, direction="long")["allowed"],
        )


class TestBacktestCompare(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.res = compare_gate("US500", TEST_GROUP, TEST_TF, start="2020-01-01")

    def test_gate_available(self):
        self.assertTrue(self.res.get("gate_available"))

    def test_gated_trades_never_exceed_raw(self):
        # Masking entries cannot create trades.
        self.assertLessEqual(self.res["gated"]["n_trades"], self.res["raw"]["n_trades"])

    def test_stats_shape(self):
        for side in ("raw", "gated"):
            for k in [
                "n_trades",
                "win_rate",
                "profit_factor",
                "total_return_pct",
                "sharpe",
            ]:
                self.assertIn(k, self.res[side])

    def test_delta_keys(self):
        self.assertIn("total_return_pct", self.res["delta_pct"])

    def test_pre_macro_coverage_reported(self):
        # Backtests starting before the macro history must report the
        # ungated fraction honestly instead of pretending those bars are
        # gate-decided.
        res = compare_gate("US500", TEST_GROUP, TEST_TF, start="2018-01-01")
        self.assertIn("pre_macro_pct", res["gate"])
        self.assertGreaterEqual(res["gate"]["pre_macro_pct"], 0.0)
        self.assertLessEqual(res["gate"]["pre_macro_pct"], 100.0)


class TestSnapshotOnRealData(unittest.TestCase):
    def test_report_has_macro_section(self):
        from src.analysis.dashboard_data import load_symbol_report

        _, report = load_symbol_report("US500", TEST_GROUP, TEST_TF)
        m = report.get("macro")
        self.assertIsNotNone(m)
        self.assertIn("regime", m)
        self.assertIn("bias", m)
        self.assertIn("gate", m)

    def test_scanner_has_macro_columns(self):
        from src.analysis.scanner import scan_universe

        table = scan_universe(
            data_dir="data/raw", group=TEST_GROUP, timeframe=TEST_TF, fetch_mt5=False
        )
        for col in ["macro_bias", "macro_label", "macro_gate"]:
            self.assertIn(col, table.columns)


if __name__ == "__main__":
    unittest.main()
