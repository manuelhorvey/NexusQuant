"""Tests for the Final Quant Rating (src/analysis/rating.py) and the
Volume & Flow indicators (src/features/indicators.py)."""

import unittest

import pandas as pd

from src.analysis.rating import (
    quant_rating,
    factor_contributions,
    final_rating,
    RECOMMENDATIONS,
)
from src.features.indicators import ad_line, volume_flow_summary


def fake_report(prob=None):
    rep = {
        "moving_averages": {"price_vs_sma200": "Above"},
        "trend_strength": {"adx": 28.0, "plus_di": 22.0, "minus_di": 15.0},
        "momentum": {"rsi_14": 62.0, "macd_hist": 0.001, "bb_pct_b": 0.7},
        "volume_flow": {"buyer_seller_score": 30.0},
        "macro": {"bias": {"bias": 1.0}},
    }
    if prob is not None:
        rep["ml"] = {"prob_pct": prob}
    return rep


class TestQuantRating(unittest.TestCase):
    def test_thresholds(self):
        self.assertEqual(quant_rating(85.0), "Strong Buy")
        self.assertEqual(quant_rating(84.9), "Buy")
        self.assertEqual(quant_rating(70.0), "Buy")
        self.assertEqual(quant_rating(69.9), "Neutral")
        self.assertEqual(quant_rating(50.0), "Neutral")
        self.assertEqual(quant_rating(49.9), "Sell")
        self.assertEqual(quant_rating(30.0), "Sell")
        self.assertEqual(quant_rating(29.9), "Strong Sell")
        self.assertEqual(quant_rating(0.0), "Strong Sell")

    def test_contributions_sum_to_net_tilt(self):
        # ML present -> blended probability (0.6 * ML + 0.4 * factor stack)
        fd = factor_contributions(fake_report(prob=70.0))
        self.assertAlmostEqual(
            sum(c["contribution"] for c in fd["contributions"]),
            fd["prob_pct"] - 50.0,
            delta=0.35,  # contributions are rounded to 0.1
        )
        self.assertTrue(fd["ml_based"])
        self.assertEqual(fd["source"], "ml+factors")
        # the blend damps the pure ML number toward the factor stack
        self.assertNotEqual(fd["prob_pct"], 70.0)
        factors = [c["factor"] for c in fd["contributions"]]
        self.assertIn("Trend", factors)
        self.assertIn("Momentum", factors)
        self.assertIn("Macro", factors)

    def test_rule_based_fallback_without_model(self):
        fd = factor_contributions(fake_report())
        self.assertFalse(fd["ml_based"])
        self.assertEqual(fd["source"], "rule")
        self.assertGreaterEqual(fd["prob_pct"], 1.0)
        self.assertLessEqual(fd["prob_pct"], 99.0)

    def test_final_rating_consistency(self):
        # prob is blended 0.6*ML + 0.4*factor-stack, so use extreme inputs
        r = final_rating(fake_report(prob=99.0))
        self.assertEqual(r["rating"], "Strong Buy")
        self.assertIn(r["recommendation"], RECOMMENDATIONS.values())
        r = final_rating(fake_report(prob=1.0))
        self.assertEqual(r["rating"], "Strong Sell")

    def test_bearish_factor_shows_negative_contribution(self):
        # below-SMA200 trend (bearish) must contribute negatively even when
        # the caller overrides the probability (sign follows the factor)
        rep = fake_report()
        rep["moving_averages"]["price_vs_sma200"] = "Below"
        fd = factor_contributions(rep, prob_pct=30.0)
        trend = next(c for c in fd["contributions"] if c["factor"] == "Trend")
        self.assertLess(trend["contribution"], 0.0)
        # the caller's bearish probability conflicts with the bullish factor
        # stack -> the difference is reported as unexplained (model-driven)
        self.assertLess(fd["unexplained"], 0.0)


class TestVolumeFlow(unittest.TestCase):
    def test_ad_line_single_bar(self):
        # CLV = ((105-90) - (110-105)) / 20 = 0.5 -> contribution 500
        df = pd.DataFrame(
            {
                "high": [110.0],
                "low": [90.0],
                "close": [105.0],
                "volume": [1000.0],
            }
        )
        ad = ad_line(df["high"], df["low"], df["close"], df["volume"])
        self.assertAlmostEqual(ad.iloc[-1], 500.0, places=4)

    def test_buyer_seller_score_direction(self):
        n = 25
        # rising closes hugging the highs -> accumulation
        up = pd.DataFrame(
            {
                "date": pd.date_range("2024-01-01", periods=n, freq="D"),
                "open": 100.0,
                "high": 100.0 + 0.5,
                "low": 100.0 - 2.0,
                "close": [100.0 + 0.5 * i for i in range(n)],
                "volume": [1000.0] * n,
            }
        )
        s_up = volume_flow_summary(up)
        self.assertTrue(s_up["available"])
        self.assertGreater(s_up["buyer_seller_score"], 0)
        self.assertIn(
            s_up["buyer_seller_label"], ["Accumulation", "Strong Accumulation"]
        )

        # falling closes hugging the lows -> distribution
        down = pd.DataFrame(
            {
                "date": pd.date_range("2024-01-01", periods=n, freq="D"),
                "open": 100.0,
                "high": 100.0 + 2.0,
                "low": 100.0 - 0.5,
                "close": [100.0 - 0.5 * i for i in range(n)],
                "volume": [1000.0] * n,
            }
        )
        s_down = volume_flow_summary(down)
        self.assertLess(s_down["buyer_seller_score"], 0)
        self.assertIn(
            s_down["buyer_seller_label"], ["Distribution", "Strong Distribution"]
        )

    def test_not_available_without_volume(self):
        n = 30
        df = pd.DataFrame(
            {
                "date": pd.date_range("2024-01-01", periods=n, freq="D"),
                "open": 1.0,
                "high": 1.01,
                "low": 0.99,
                "close": 1.0,
            }
        )
        s = volume_flow_summary(df)
        self.assertFalse(s["available"])


if __name__ == "__main__":
    unittest.main()
