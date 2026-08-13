"""Tests for the Pattern Recognition Engine (src/features/patterns.py)."""

import unittest

import numpy as np
import pandas as pd

from src.features.patterns import detect_patterns, patterns_summary


def make_frame(closes, vol_pattern=None):
    closes = np.asarray(closes, dtype=float)
    n = len(closes)
    df = pd.DataFrame(
        {
            "date": pd.date_range("2020-01-01", periods=n, freq="D"),
            "open": closes,
            "high": closes + 0.15,
            "low": closes - 0.15,
            "close": closes,
            "volume": np.full(n, 1000.0),
        }
    )
    if vol_pattern is not None:
        df["volume"] = vol_pattern
    return df


def double_top_series():
    c = []
    c += list(np.linspace(95.0, 99.0, 10))  # approach
    c += list(np.linspace(99.0, 100.0, 4))  # peak 1
    c += list(np.linspace(100.0, 98.0, 5))  # dip (neckline)
    c += list(np.linspace(98.0, 100.0, 5))  # peak 2
    c += list(np.linspace(100.0, 96.5, 6))  # breakdown
    return c


def double_bottom_series():
    c = []
    c += list(np.linspace(103.0, 99.0, 10))  # approach
    c += list(np.linspace(99.0, 98.0, 4))  # trough 1
    c += list(np.linspace(98.0, 103.0, 5))  # neckline
    c += list(np.linspace(103.0, 98.0, 5))  # trough 2
    c += list(np.linspace(98.0, 105.0, 6))  # breakout
    return c


def hs_series():
    c = []
    c += list(np.linspace(95.0, 100.0, 8))  # left shoulder
    c += list(np.linspace(100.0, 97.0, 4))  # neck 1
    c += list(np.linspace(97.0, 105.0, 5))  # head
    c += list(np.linspace(105.0, 97.5, 5))  # neck 2
    c += list(np.linspace(97.5, 100.0, 4))  # right shoulder
    c += list(np.linspace(100.0, 95.0, 6))  # breakdown
    return c


class TestPatterns(unittest.TestCase):
    def test_double_top_detected(self):
        # volume fades on the second peak (classic confirmation)
        closes = double_top_series()
        vol = np.full(len(closes), 1000.0)
        vol[len(closes) // 2 :] = 600.0
        pats = detect_patterns(make_frame(closes, vol))
        names = [p["name"] for p in pats]
        self.assertIn("Double Top", names)
        dt = next(p for p in pats if p["name"] == "Double Top")
        self.assertEqual(dt["side"], "bearish")
        self.assertEqual(dt["status"], "Confirmed")
        self.assertGreaterEqual(dt["prob"], 65)
        self.assertLess(dt["breakout"], 100.0)

    def test_double_bottom_detected(self):
        closes = double_bottom_series()
        vol = np.full(len(closes), 1000.0)
        vol[: len(closes) // 2] = 600.0  # volume expands at trough 2
        pats = detect_patterns(make_frame(closes, vol))
        names = [p["name"] for p in pats]
        self.assertIn("Double Bottom", names)
        db = next(p for p in pats if p["name"] == "Double Bottom")
        self.assertEqual(db["side"], "bullish")
        self.assertEqual(db["status"], "Confirmed")
        self.assertGreater(db["breakout"], 100.0)

    def test_head_and_shoulders_detected(self):
        pats = detect_patterns(make_frame(hs_series()))
        names = [p["name"] for p in pats]
        self.assertIn("Head & Shoulders", names)
        hs = next(p for p in pats if p["name"] == "Head & Shoulders")
        self.assertEqual(hs["side"], "bearish")
        self.assertEqual(hs["status"], "Confirmed")
        self.assertGreaterEqual(hs["prob"], 65)

    def test_no_patterns_on_random_walk(self):
        rng = np.random.default_rng(7)
        closes = 100 + np.cumsum(rng.normal(0, 0.4, 140))
        pats = detect_patterns(make_frame(closes))
        self.assertEqual(pats, [])

    def test_patterns_summary_shape(self):
        s = patterns_summary(make_frame(hs_series()))
        self.assertEqual(s["count"], len(s["patterns"]))
        if s["patterns"]:
            self.assertEqual(s["best"]["name"], s["patterns"][0]["name"])
            self.assertIn("prob", s["best"])
            self.assertIn("breakout", s["best"])
            self.assertIn("status", s["best"])

    def test_short_series_returns_empty(self):
        self.assertEqual(detect_patterns(make_frame([1.0, 1.1, 1.2])), [])


if __name__ == "__main__":
    unittest.main()
