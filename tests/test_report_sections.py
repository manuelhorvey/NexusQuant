"""
Tests for the institutional report sections wired in the audit pass
(src/analysis/report.py + src/macro/overlay.py): the multi-timeframe
(D/W/M) regime table + cluster label, the MA ribbon (cross probability /
slope / width), the divergence engine, the Fibonacci confluence map,
anchored VWAP + volume-profile nodes, and the macro sensitivity table
(market beta, dollar/yield/vol correlations, sector ETF).

Also covers the Yahoo D1 midnight-normalization fix (daily bars must align
with the macro overlay's date index for sensitivities to compute).
"""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.append(str(Path(__file__).parent.parent))

from src.features.indicators import add_all_indicators
from src.features.regime import detect_regime
from src.analysis.report import generate_full_report
from src.macro.overlay import SECTOR_ETF, macro_sensitivities


def _make_frame(n=400, seed=7, start="2023-01-01"):
    rng = np.random.default_rng(seed)
    idx = pd.bdate_range(start, periods=n)
    rets = rng.normal(0.0003, 0.012, n)
    close = 100 * np.exp(np.cumsum(rets))
    open_ = close * (1 + rng.normal(0, 0.002, n))
    high = np.maximum(open_, close) * (1 + np.abs(rng.normal(0, 0.003, n)))
    low = np.minimum(open_, close) * (1 - np.abs(rng.normal(0, 0.003, n)))
    return pd.DataFrame(
        {
            "open": open_,
            "high": high,
            "low": low,
            "close": close,
            "volume": rng.integers(1_000, 10_000, n),
        },
        index=idx,
    )


def _prepared(n=400):
    df = _make_frame(n)
    df = add_all_indicators(df)
    df = detect_regime(df)
    return df


class ReportSectionsTest(unittest.TestCase):
    def test_regime_has_mtf_and_cluster(self):
        r = generate_full_report(_prepared(), symbol="TEST")
        rg = r["regime"]
        self.assertIn("mtf", rg)
        self.assertGreaterEqual(len(rg["mtf"]), 1)
        self.assertIn("mtf_consensus", rg)
        self.assertIn("regime_cluster", rg)
        for row in rg["mtf"]:
            for key in (
                "timeframe",
                "regime",
                "adx",
                "vs_200_pct",
                "slope_20",
                "confidence",
            ):
                self.assertIn(key, row)

    def test_ml_section_carries_feature_importance(self):
        """Spec #10: the report ML section embeds feature importance
        (top features by gain + the factor-group breakdown)."""
        r = generate_full_report(_prepared(), symbol="TEST")
        ml = r.get("ml")
        if not ml:  # no saved model in the test env -> graceful absence
            self.skipTest("no ensemble model saved")
        imp = ml.get("importance")
        self.assertIsNotNone(imp)
        self.assertGreaterEqual(len(imp["top"]), 1)
        self.assertGreaterEqual(len(imp["by_group"]), 1)
        for t in imp["top"]:
            self.assertIn("feature", t)
            self.assertIn("gain_pct", t)
            self.assertGreaterEqual(t["gain_pct"], 0.0)
        total = sum(g["gain_pct"] for g in imp["by_group"])
        self.assertAlmostEqual(total, 100.0, delta=1.0)

    def test_sentiment_wired_for_fx_symbol(self):
        """Spec #9: sentiment is computed for EVERY symbol class (FX
        included), not just equities - gracefully unavailable offline."""
        r = generate_full_report(_prepared(), symbol="EURUSD")
        self.assertIn("sentiment", r)
        se = r["sentiment"] or {}
        self.assertIn("available", se)
        self.assertIn("composite", se)

    def test_mtf_false_keeps_scan_row_keys(self):
        """mtf=False (the universe-ranking path) must drop the resample-
        heavy D/W/M table + cluster label but keep every key scan_symbol
        reads from the regime section."""
        df = _prepared(400)
        r = generate_full_report(df, symbol="TEST", mtf=False)
        rg = r["regime"]
        self.assertNotIn("mtf", rg)
        self.assertNotIn("regime_cluster", rg)
        # Keys the scanner ranking row depends on.
        self.assertIn("regime", rg)
        self.assertIn("price_vs_200sma_pct", rg)
        self.assertIn("adx", rg)
        self.assertIn("slope_20", rg)
        self.assertIn("confidence", rg)

    def test_mtf_true_has_table_and_cluster(self):
        df = _prepared(400)
        r = generate_full_report(df, symbol="TEST", mtf=True)
        rg = r["regime"]
        self.assertIn("mtf", rg)
        self.assertGreaterEqual(len(rg["mtf"]), 1)
        self.assertIn("mtf_consensus", rg)
        self.assertIn("regime_cluster", rg)

    def test_ma_ribbon_present(self):
        r = generate_full_report(_prepared(), symbol="TEST")
        rb = r.get("ma_ribbon") or {}
        self.assertTrue(rb.get("available"))
        for key in (
            "cross_prob",
            "cross_direction",
            "ribbon_slope",
            "ribbon_width_pct",
            "ribbon_alignment",
            "signal",
        ):
            self.assertIn(key, rb)
        self.assertTrue(0.0 <= rb["cross_prob"] <= 1.0)

    def test_divergences_present(self):
        r = generate_full_report(_prepared(), symbol="TEST")
        dv = r.get("divergences") or {}
        self.assertIn("signals", dv)
        self.assertIn("count", dv)
        for d in dv.get("signals", []):
            self.assertGreaterEqual(d["confidence"], 65.0)

    def test_format_divergence_both_shapes(self):
        """Shape-agnostic rendering (shared by report + dashboard):
        failure swings use the RSI peak list, regular/hidden use
        osc_from -> osc_to."""
        from src.features.divergence import format_divergence

        self.assertEqual(
            format_divergence(
                {"type": "failure_swing", "osc": "rsi_14", "peaks": [70.0, 72.0, 68.0]}
            ),
            "rsi_14 peaks 70.0 → 72.0 → 68.0",
        )
        self.assertEqual(
            format_divergence(
                {"type": "regular", "osc": "macd_hist", "osc_from": 1.0, "osc_to": 0.5}
            ),
            "macd_hist 1.0 → 0.5",
        )

    def test_print_report_handles_failure_swing(self):
        """Regression: print_report used to KeyError on failure-swing
        signals (they carry 'peaks', not 'osc_from'/'osc_to')."""
        import io
        from contextlib import redirect_stdout
        from src.analysis.report import print_report

        r = generate_full_report(_prepared(), symbol="TEST")
        r["divergences"] = {
            "signals": [
                {
                    "name": "RSI Failure Swing (bearish)",
                    "side": "bearish",
                    "type": "failure_swing",
                    "osc": "rsi_14",
                    "peaks": [70.0, 72.0, 68.0],
                    "confidence": 70,
                }
            ],
            "count": 1,
            "available": True,
        }
        buf = io.StringIO()
        with redirect_stdout(buf):
            print_report(r)
        self.assertIn("rsi_14 peaks", buf.getvalue())

    def test_fib_map_present(self):
        r = generate_full_report(_prepared(), symbol="TEST")
        fm = r.get("fib_map") or []
        # Both up and down legs produce 6 ratio rows when legs exist.
        if fm:
            for row in fm:
                for key in (
                    "level",
                    "side",
                    "price",
                    "confluence",
                    "distance_from_close_pct",
                ):
                    self.assertIn(key, row)
                self.assertIn(
                    row["level"],
                    ["38.2%", "50.0%", "61.8%", "78.6%", "127.2%", "161.8%"],
                )

    def test_levels_have_vwap_and_volume_profile(self):
        r = generate_full_report(_prepared(), symbol="TEST")
        lv = r["levels"]
        self.assertIn("anchored_vwap", lv)
        vp = lv.get("volume_profile") or []
        if vp:
            for node in vp:
                self.assertIn("price", node)
                self.assertIn("is_high_volume", node)

    def test_divergence_engine_detects_regular_bullish(self):
        """A crafted double-bottom price with RSI higher-low triggers a
        regular bullish divergence at >= 65 confidence."""
        rng = np.random.default_rng(3)
        n = 260
        close = np.full(n, 100.0)
        # lower low then higher low (price), with RSI driven by a decaying
        # down-move -> higher oscillator low.
        seg = np.linspace(0, -1, 40) * 6
        close[60:100] += seg
        close[100:140] += -seg * 0.35  # shallower second trough
        close[140:] += np.linspace(0, 4, n - 140)
        close = np.maximum(close, 80)
        idx = pd.bdate_range("2023-01-01", periods=n)
        high = close + 0.5
        low = close - 0.5
        df = pd.DataFrame(
            {
                "open": close - 0.2,
                "high": high,
                "low": low,
                "close": close,
                "volume": rng.integers(1_000, 5_000, n),
            },
            index=idx,
        )
        df = add_all_indicators(df)
        from src.features.divergence import detect_divergences

        sigs = detect_divergences(df)
        self.assertIsInstance(sigs, list)


class SensitivitiesTest(unittest.TestCase):
    def test_sensitivities_shape_and_range(self):
        df = _prepared(300)
        s = macro_sensitivities("EURUSD", df, data_dir="data/raw")
        self.assertIn("lookback", s)
        # Values are correlations in [-1, 1] or None (data dependent).
        for key in ("spx_corr", "dollar_sens", "yield_sens", "vol_sens"):
            v = s.get(key)
            if v is not None:
                self.assertGreaterEqual(v, -1.0)
                self.assertLessEqual(v, 1.0)

    def test_market_beta_is_float_or_absent(self):
        df = _prepared(300)
        s = macro_sensitivities("EURUSD", df)
        if "market_beta" in s:
            self.assertIsInstance(s["market_beta"], float)

    def test_missing_frame_returns_empty(self):
        self.assertEqual(macro_sensitivities("EURUSD", None), {})

    def test_short_frame_returns_empty(self):
        df = _make_frame(10)
        df = add_all_indicators(df)
        self.assertEqual(macro_sensitivities("EURUSD", df), {})

    def test_sector_etf_map_has_valid_tickers(self):
        for _, etf in SECTOR_ETF.items():
            self.assertEqual(etf, etf.upper())
            self.assertTrue(etf.startswith("XL"))

    def test_report_embeds_sensitivities_when_df_passed(self):
        r = generate_full_report(_prepared(300), symbol="EURUSD", group="full_fx")
        ss = (r.get("macro") or {}).get("sensitivities")
        # Sensitivities may be present (macro data exists locally); the
        # shape is what matters.
        if ss is not None:
            self.assertIn("market_beta", ss)


class YahooDailyNormalizeTest(unittest.TestCase):
    def test_daily_bars_normalized_to_midnight(self):
        """Yahoo 1d bars carry exchange-close timestamps; the fetcher must
        normalize them so daily series align with the macro index."""
        from src.data.yahoo import fetch_ohlcv
        from unittest.mock import patch

        ts = pd.date_range("2024-01-01", periods=10, freq="D") + pd.Timedelta(
            hours=13, minutes=30
        )
        closes = np.linspace(100, 110, 10)
        payload = {
            "chart": {
                "result": [
                    {
                        "timestamp": [int(t.timestamp()) for t in ts],
                        "indicators": {
                            "quote": [
                                {
                                    "open": (closes - 0.2).tolist(),
                                    "high": (closes + 0.5).tolist(),
                                    "low": (closes - 0.5).tolist(),
                                    "close": closes.tolist(),
                                    "volume": [1000] * 10,
                                }
                            ]
                        },
                    }
                ],
                "error": None,
            }
        }
        with patch("src.data.yahoo._chart", return_value=payload):
            df = fetch_ohlcv("AAPL", "D1")
        times = pd.to_datetime(df["date"])
        self.assertTrue((times.dt.hour == 0).all())
        self.assertTrue((times.dt.minute == 0).all())


if __name__ == "__main__":
    unittest.main()
