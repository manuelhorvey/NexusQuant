"""
Tests for the Sell-the-Rally (short-side) engine and its downstream
wiring: rally detection (mirror of dip.py), the short target ladder,
risk_plan_from_report_short, the causal rally signal series, short
labels (1R + meta), the dual dataset, and graceful short-model predict.
"""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

import pandas as pd

sys.path.append(str(Path(__file__).parent.parent))

from tests.test_dip import make_downtrend, make_uptrend_dip, with_indicators
from src.features.regime import detect_regime
from src.features.rally import detect_rally
from src.risk.targets import build_short_target_ladder
from src.risk.run import risk_plan_from_report_short
from src.risk.sizing import fractional_qty
from src.backtest.signals import rally_signal_series
from src.model.features import (
    build_dataset,
    build_dataset_dual,
    build_labels_short,
    make_meta_labels_short,
)


def _prep(df: pd.DataFrame) -> pd.DataFrame:
    return detect_regime(with_indicators(df))


def _report_with_rally(df, **overrides) -> dict:
    """Minimal full-report-shaped dict the short risk plan reads."""
    latest = df.iloc[-1]
    report = {
        "last_date": str(df.index[-1].date()),
        "last_close": float(latest["close"]),
        "volatility": {"atr_14": float(latest["atr_14"])},
        "rally": detect_rally(df),
    }
    report.update(overrides)
    return report


class RallyEngineTest(unittest.TestCase):
    def test_uptrend_gives_no_downtrend(self):
        df = _prep(make_uptrend_dip())
        rl = detect_rally(df)
        self.assertEqual(rl["rally_stage"], "No Downtrend")
        self.assertFalse(rl["rally_confirmed"])

    def test_output_shape(self):
        df = _prep(make_downtrend())
        rl = detect_rally(df)
        for key in (
            "rally_score",
            "rally_confirmed",
            "rally_stage",
            "entry_zone",
            "invalidation_level",
            "target",
            "rally_depth_pct",
            "trigger",
            "components",
        ):
            self.assertIn(key, rl)
        self.assertTrue(0 <= rl["rally_score"] <= 8)

    def test_components_are_8_or_fewer(self):
        df = _prep(make_downtrend())
        comp = detect_rally(df)["components"]
        # score = sum of the boolean component flags + trigger.
        n_flags = sum(
            1
            for k in comp
            if k not in ("bias_score", "rally_depth_pct", "bearish_structure")
        )
        self.assertGreaterEqual(n_flags, 7)  # 7 named + trigger inside


class ShortTargetsTest(unittest.TestCase):
    def test_ladder_below_entry(self):
        levels = {
            "clusters": [
                {"price": 95.0, "score": 5, "tags": ["swing_low"]},
                {"price": 90.0, "score": 3, "tags": ["fib"]},
            ],
            "last_down_leg": [90.0, 105.0],
        }
        ladder = build_short_target_ladder(entry=100.0, stop=102.0, levels=levels)
        self.assertEqual(len(ladder["targets"]), 3)
        for t in ladder["targets"]:
            self.assertLess(t["price"], 100.0)
            self.assertGreater(t["rr"], 0.0)
        self.assertGreaterEqual(ladder["best_rr"], ladder["targets"][0]["rr"])

    def test_fallback_when_no_levels(self):
        ladder = build_short_target_ladder(entry=100.0, stop=101.0, levels={})
        self.assertEqual([t["rr"] for t in ladder["targets"]], [1.0, 2.0, 3.0])

    def test_invalid_geometry_returns_empty(self):
        ladder = build_short_target_ladder(entry=100.0, stop=99.0, levels={})
        self.assertEqual(ladder["targets"], [])


class ShortRiskPlanTest(unittest.TestCase):
    def test_short_plan_computes(self):
        df = _prep(make_downtrend())
        report = _report_with_rally(df)
        plan = risk_plan_from_report_short(report, "TEST")
        s = plan.get("setup")
        if s is None:  # rally may not be actionable on the fixture
            self.assertEqual(
                plan["reason"], "no actionable rally setup (entry zone/invalidation)"
            )
            return
        self.assertGreater(s["stop"], s["entry"])  # stop above for shorts
        self.assertIn("rr_ok", s)
        self.assertEqual(len(plan["sizes"]), 3)

    def test_short_sizing_direction(self):
        # stop above entry must still size a positive quantity.
        qty = fractional_qty(
            equity=100_000, entry=100.0, stop=101.0, risk_pct=0.01, direction="short"
        )
        self.assertGreater(qty, 0)
        self.assertAlmostEqual(qty, 100_000 * 0.01 / 1.0)


class RallySignalSeriesTest(unittest.TestCase):
    def test_series_shape_and_causality(self):
        df = _prep(make_downtrend())
        sig = rally_signal_series(df)
        for key in (
            "score",
            "confirmed",
            "stage",
            "below_sma200",
            "ma_stack",
            "trend",
            "rally",
            "stretched",
            "at_resistance",
            "fib_zone",
            "trigger",
            "rally_depth_pct",
            "entry_lo",
            "entry_hi",
            "invalidation",
            "target",
        ):
            self.assertIn(key, sig.columns)
        # Causal: corrupting future bars must not change early rows.
        df2 = df.copy()
        df2.loc[df2.index[100] :, ["close", "high", "low", "open"]] *= 2.0
        sig2 = rally_signal_series(df2)
        self.assertTrue(sig.iloc[:50].equals(sig2.iloc[:50]))


class ShortLabelsTest(unittest.TestCase):
    def test_short_labels_mostly_wins_on_downtrend(self):
        df = _prep(make_downtrend())
        y = build_labels_short(df, horizon=10)["label"].dropna()
        self.assertGreater(y.mean(), 0.5)

    def test_short_labels_mirror_geometry(self):
        df = _prep(make_uptrend_dip())
        y = build_labels_short(df, horizon=10)["label"].dropna()
        self.assertLess(y.mean(), 0.5)  # uptrend is mostly short losses

    def test_meta_short_returns_01_or_nan(self):
        df = _prep(make_downtrend())
        sig = rally_signal_series(df)
        lbl = make_meta_labels_short(df, sig)
        vals = lbl.dropna()
        self.assertTrue(set(vals.unique()).issubset({0.0, 1.0}))

    def test_dataset_side_short(self):
        df = _prep(make_downtrend())
        ds = build_dataset(df, symbol="TEST", meta=True, side="short")
        self.assertIn("X", ds)
        self.assertIn("confirmed", ds)
        self.assertEqual(len(ds["X"]), len(ds["y"]))

    def test_dataset_dual_has_both_labels(self):
        df = _prep(make_uptrend_dip())
        d2 = build_dataset_dual(df, symbol="TEST", meta=True)
        self.assertIn("y_long", d2)
        self.assertIn("y_short", d2)
        self.assertIn("confirmed_short", d2)
        self.assertGreaterEqual(len(d2["X_short"]), 0)

    def test_predict_short_graceful_without_model(self):
        from src.model.model import predict_short_series

        df = _prep(make_uptrend_dip())
        prob = predict_short_series(df, symbol="TEST")
        # No rally model saved in the test env -> None (graceful).
        self.assertTrue(prob is None or prob.notna().any())


if __name__ == "__main__":
    unittest.main()
