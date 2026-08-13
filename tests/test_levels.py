"""
NexusQuant - Levels engine unit tests.
Run from project root:
    python -m unittest discover -s tests -v
"""

import sys
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.features.levels import (
    cluster_levels,
    compute_levels,
    detect_swings,
    fibonacci_levels,
    pivot_levels,
)


def zigzag_frame() -> pd.DataFrame:
    """Synthetic zig-zag with known swing points."""
    close = [100, 102, 105, 103, 101, 104, 107, 105, 102, 106, 110, 108, 106, 109, 112]
    high = [v + 0.5 for v in close]
    low = [v - 0.5 for v in close]
    dates = pd.date_range("2025-01-01", periods=len(close), freq="D")
    return pd.DataFrame(
        {"open": close, "high": high, "low": low, "close": close},
        index=dates,
    )


class TestDetectSwings(unittest.TestCase):
    def test_swing_points_found(self):
        df = detect_swings(zigzag_frame(), left=2, right=2)
        # 110.5 (index 10) is the last *confirmed* swing high
        # (the final bar has an incomplete right window, so it is not confirmed)
        self.assertIn(110.5, df.loc[df["swing_high"], "high"].tolist())
        # 101.5 (index 8) is a confirmed swing low
        self.assertIn(101.5, df.loc[df["swing_low"], "low"].tolist())
        # no swing points on the unconfirmed trailing bars
        self.assertFalse(df["swing_high"].iloc[-1])

    def test_no_false_positives_in_trend(self):
        # Monotonic uptrend -> no swing highs
        n = 30
        close = np.arange(1, n + 1, dtype=float)
        df = pd.DataFrame(
            {"open": close, "high": close + 0.1, "low": close - 0.1, "close": close},
            index=pd.date_range("2025-01-01", periods=n, freq="D"),
        )
        out = detect_swings(df)
        self.assertEqual(out["swing_high"].sum(), 0)


class TestPivots(unittest.TestCase):
    def test_pivot_math(self):
        df = pd.DataFrame(
            {"open": [1.1], "high": [1.2], "low": [1.0], "close": [1.1]},
        )
        df = pd.concat([df, df])  # 2 rows so iloc[-2] exists
        levels = dict((tag, price) for price, tag in pivot_levels(df))
        self.assertAlmostEqual(levels["pivot"], (1.2 + 1.0 + 1.1) / 3)
        self.assertAlmostEqual(levels["pivot_r1"], 2 * levels["pivot"] - 1.0)
        self.assertAlmostEqual(levels["pivot_s1"], 2 * levels["pivot"] - 1.2)
        self.assertEqual(len(levels), 7)

    def test_short_frame_returns_empty(self):
        df = zigzag_frame().iloc[:1]
        self.assertEqual(pivot_levels(df), [])


class TestFibonacci(unittest.TestCase):
    def test_up_leg_levels(self):
        df = zigzag_frame()
        levels = dict((tag, price) for price, tag in fibonacci_levels(df))
        # Last confirmed up leg: 101.5 -> 110.5 (indices 8 -> 10)
        rng = 110.5 - 101.5
        self.assertAlmostEqual(levels["fib_up_0.382"], 110.5 - 0.382 * rng)
        self.assertAlmostEqual(levels["fib_up_ext_1.618"], 110.5 + 1.618 * rng)
        # Last confirmed down leg: 110.5 -> 105.5 (indices 10 -> 12)
        self.assertAlmostEqual(levels["fib_down_0.500"], 105.5 + 0.5 * (110.5 - 105.5))


class TestClustering(unittest.TestCase):
    def test_merges_nearby_levels(self):
        levels = [(100.0, "swing_low"), (100.2, "fib_up_0.5"), (104.0, "pivot_r1")]
        clusters = cluster_levels(levels, tolerance=0.5)
        self.assertEqual(len(clusters), 2)
        merged = clusters[0]
        self.assertEqual(merged["score"], 2)
        self.assertEqual(merged["strength"], "Medium")
        self.assertAlmostEqual(merged["price"], 100.1)

    def test_strong_cluster(self):
        levels = [(100, "a"), (100.1, "b"), (100.2, "c"), (101.5, "d")]
        clusters = cluster_levels(levels, tolerance=0.3)
        self.assertEqual(clusters[0]["score"], 3)
        self.assertEqual(clusters[0]["strength"], "Strong")


class TestComputeLevels(unittest.TestCase):
    def test_nearest_support_below_close(self):
        df = zigzag_frame()
        # make a long frame so ATR exists
        big = pd.concat([df] * 20, ignore_index=True)
        big.index = pd.date_range("2024-01-01", periods=len(big), freq="D")
        info = compute_levels(big)
        close = info["close"]
        if info["nearest_support"]:
            self.assertLess(info["nearest_support"]["price"], close)
        if info["nearest_resistance"]:
            self.assertGreater(info["nearest_resistance"]["price"], close)
        self.assertIn("clusters", info)
        self.assertIn("pivots", info)

    def test_clusters_sorted(self):
        df = zigzag_frame()
        big = pd.concat([df] * 20, ignore_index=True)
        big.index = pd.date_range("2024-01-01", periods=len(big), freq="D")
        info = compute_levels(big)
        prices = [c["price"] for c in info["clusters"]]
        self.assertEqual(prices, sorted(prices))


if __name__ == "__main__":
    unittest.main()
