"""
Tests for the direction-neutral setup classifier (src/features/setups.py).

Covers the taxonomy, the independent long/short evidence scores, the
direction verdict, the "no universal 200-SMA gate" property, and the
expected-value helpers.
"""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent))

import numpy as np
import pandas as pd

from src.features.setups import (
    ALL_FAMILIES,
    LONG_FAMILIES,
    SHORT_FAMILIES,
    classify_setup,
    expected_value,
    probability_weighted_rr,
)


def _trending_frame(n: int = 300, up: bool = True, vol: float = 0.25) -> pd.DataFrame:
    """Synthetic trending OHLCV with indicators for testing.

    Small per-bar drift with modest noise keeps the price path non-
    degenerate (a drift of -0.3/bar over 300 bars collapses the series to
    the float floor and destroys every indicator, so the tests would fail
    for the wrong reason).
    """
    rng = np.random.default_rng(7)
    # LINEAR price path (100 -> ~180 or ~55): compounding returns over 300
    # bars collapse the series to the float floor and destroy every
    # indicator, so linear is the honest way to get a clean trend.
    drift = 0.28 if up else -0.15
    base = 100.0 + np.arange(n) * drift
    close = np.maximum(base + rng.normal(0, vol * 8, n), 5.0)
    df = pd.DataFrame(
        {
            "open": close * (1 + rng.normal(0, 0.001, n)),
            "high": close * (1 + abs(rng.normal(0, 0.002, n))),
            "low": close * (1 - abs(rng.normal(0, 0.002, n))),
            "close": close,
            "volume": rng.integers(800, 1200, n).astype(float),
        }
    )
    from src.features.indicators import add_all_indicators

    return add_all_indicators(df)


class TestTaxonomy(unittest.TestCase):
    def test_families_complete(self):
        self.assertEqual(
            set(LONG_FAMILIES),
            {
                "LONG_TREND_CONTINUATION",
                "LONG_BUY_DIP",
                "LONG_BREAKOUT",
                "LONG_BREAKOUT_RETEST",
                "LONG_REVERSAL",
                "LONG_MEAN_REVERSION",
            },
        )
        self.assertEqual(
            set(SHORT_FAMILIES),
            {
                "SHORT_TREND_CONTINUATION",
                "SHORT_SELL_RALLY",
                "SHORT_BREAKDOWN",
                "SHORT_BREAKDOWN_RETEST",
                "SHORT_REVERSAL",
                "SHORT_MEAN_REVERSION",
            },
        )
        self.assertEqual(len(ALL_FAMILIES), 12)


class TestDirectionVerdict(unittest.TestCase):
    def test_bull_frame_trends_long(self):
        df = _trending_frame(up=True)
        s = classify_setup(df)
        self.assertIn(s["direction"], {"long", "flat"})
        self.assertGreaterEqual(s["long_score"], s["short_score"])
        self.assertGreaterEqual(s["long_score"], 0.0)
        self.assertLessEqual(s["long_score"], 1.0)

    def test_bear_frame_trends_short(self):
        df = _trending_frame(up=False)
        s = classify_setup(df)
        self.assertIn(s["direction"], {"short", "flat"})
        self.assertGreaterEqual(s["short_score"], s["long_score"])

    def test_evidence_scores_independent(self):
        # Long and short scores must be independently estimated: a strong
        # uptrend has HIGH long evidence and LOW short evidence, not
        # symmetric 0.5/0.5 values.
        df = _trending_frame(up=True)
        s = classify_setup(df)
        self.assertGreater(s["long_score"], s["short_score"])
        self.assertGreater(s["long_score"], 0.5)

    def test_no_200sma_universal_gate(self):
        # A breakdown below the 200-SMA must be able to produce a SHORT
        # verdict, and - critically - the classifier must allow LONG
        # families below the 200-SMA (reversal/retest) and SHORT families
        # above it. Construct: strong downtrend (below SMA200) -> the short
        # side wins.
        df = _trending_frame(up=False)
        s = classify_setup(df)
        self.assertEqual(s["direction"], "short")
        # The long side must still have a scored family even in the
        # downtrend (reversal / buy-dip families exist), never zero by gate.
        self.assertIn("LONG_REVERSAL", s["long_families"])

    def test_family_is_from_taxonomy(self):
        df = _trending_frame(up=True)
        s = classify_setup(df)
        if s["setup_family"] is not None:
            self.assertIn(s["setup_family"], ALL_FAMILIES)

    def test_engine_veto_rejects_buy_dip_without_uptrend(self):
        # The dip engine explicitly reads "No Uptrend" (below SMA200 / no
        # bullish structure) - the classifier must NOT let LONG_BUY_DIP win
        # anyway. The pullback family is vetoed; other long families may
        # still score but the dip family itself is suppressed.
        df = _trending_frame(up=False)
        s = classify_setup(
            df,
            dip={
                "dip_stage": "No Uptrend",
                "dip_score": 2,
                "dip_confirmed": False,
            },
        )
        self.assertLessEqual(s["long_families"]["LONG_BUY_DIP"], 0.25)

    def test_engine_veto_rejects_sell_rally_without_downtrend(self):
        df = _trending_frame(up=True)
        s = classify_setup(
            df,
            rally={
                "rally_stage": "No Downtrend",
                "rally_score": 2,
                "rally_confirmed": False,
            },
        )
        self.assertLessEqual(s["short_families"]["SHORT_SELL_RALLY"], 0.25)

    def test_engine_confirm_boosts_pullback_family(self):
        df = _trending_frame(up=True)
        s = classify_setup(
            df,
            dip={"dip_stage": "Confirmed", "dip_score": 7, "dip_confirmed": True},
        )
        self.assertGreaterEqual(s["long_families"]["LONG_BUY_DIP"], 0.99)

    def test_continuation_requires_momentum_alignment(self):
        # A symbol below the SMA200 with POSITIVE MACD momentum must not
        # score SHORT_TREND_CONTINUATION highly (momentum contradicts the
        # short side even though price is below the SMA). This is the
        # "200-SMA is context, not a gate" property on the continuation
        # family.
        df = _trending_frame(up=True)
        # Force price below its SMA200 but keep MACD positive: flip the
        # close series to a lower level while leaving indicator columns.
        df2 = df.copy()
        df2["close"] = df["sma_200"] * 0.985
        s = classify_setup(df2)
        self.assertLessEqual(s["short_families"]["SHORT_TREND_CONTINUATION"], 0.4)


class TestExpectedValue(unittest.TestCase):
    def test_ev_positive_with_edge(self):
        ev = expected_value(0.6, avg_win_r=2.5, avg_loss_r=-1.0)
        self.assertIsNotNone(ev)
        self.assertGreater(ev, 0.0)  # 0.6*2.5 - 0.4*1 = 1.1

    def test_ev_negative_with_no_edge(self):
        ev = expected_value(0.3, avg_win_r=2.0, avg_loss_r=-1.0)
        self.assertIsNotNone(ev)
        self.assertLess(ev, 0.0)  # 0.3*2 - 0.7*1 = -0.1

    def test_ev_none_without_probability(self):
        # An EV must NEVER be fabricated when no calibrated probability
        # exists (the "fake probability" red-team requirement).
        self.assertIsNone(expected_value(None, avg_win_r=2.0))
        self.assertIsNone(probability_weighted_rr([], prob_win=None))

    def test_no_fabricated_probability_without_ml(self):
        # Without a calibrated ML probability the classifier must NOT
        # invent prob_long/prob_short from evidence scores (the "no fake
        # probability" rule). EV consumers must see None, never a
        # fabricated value.
        df = _trending_frame(up=True)
        s = classify_setup(df, ml=None)
        self.assertIsNone(s["prob_long"])
        self.assertIsNone(s["prob_short"])
        self.assertIsNotNone(s["long_score"])
        self.assertIsNotNone(s["short_score"])

    def test_zero_probability_not_treated_as_missing(self):
        # A legitimate 0.0 calibrated probability must survive the
        # prob_long fallback (explicit None checks, not truthiness).
        df = _trending_frame(up=True)
        s = classify_setup(
            df, ml={"prob": 0.9, "prob_long": 0.0, "prob_short": 0.8}
        )
        self.assertEqual(s["prob_long"], 0.0)
        self.assertEqual(s["prob_short"], 0.8)

    def test_pwrr_single_target_equals_ev(self):
        ladder = [{"target": "T1", "rr": 2.5}]
        pw = probability_weighted_rr(ladder, prob_win=0.6)
        self.assertIsNotNone(pw)
        # 0.6*2.5*1.0 - 0.4*1 = 1.1 for the first (fully reachable) target.
        self.assertAlmostEqual(pw, 1.1, places=2)

    def test_pwrr_further_targets_discount(self):
        ladder = [{"target": "T1", "rr": 1.0}, {"target": "T2", "rr": 3.0}]
        pw1 = probability_weighted_rr(ladder[:1], prob_win=0.5)
        pw2 = probability_weighted_rr(ladder, prob_win=0.5)
        self.assertIsNotNone(pw1)
        self.assertIsNotNone(pw2)
        # Adding a further target with a discount still increases EV but by
        # less than its nominal R:R would imply.
        self.assertGreater(pw2, pw1)
        self.assertLess(pw2 - pw1, 3.0 * 0.5)


if __name__ == "__main__":
    unittest.main()
