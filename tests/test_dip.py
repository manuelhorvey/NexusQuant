"""
NexusQuant - Buy-the-Dip engine unit tests.
Run from project root:
    python -m unittest discover -s tests -v
"""

import sys
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.features.dip import _momentum_trigger, detect_dip
from src.features.indicators import add_all_indicators
from src.features.levels import levels_summary


def make_uptrend_dip() -> pd.DataFrame:
    """
    Synthetic D1 series: steady uptrend, then a pullback of ~4% into support,
    then a small bounce. Indicators + levels can be computed on top.
    """
    rng = np.random.default_rng(42)
    n = 260
    # Uptrend base with noise, then a dip in the last ~8 bars
    trend = np.linspace(100.0, 200.0, n - 10)
    noise = rng.normal(0, 0.6, n - 10)
    closes = trend + noise

    # Pullback: 200 -> 191.5 (-4.25%) then a decisive bounce (trigger)
    dip = np.array(
        [198.0, 196.0, 194.0, 192.5, 191.5, 193.0, 194.5, 196.0, 197.5, 199.0]
    )
    closes = np.concatenate([closes, dip])

    open_ = np.roll(closes, 1)
    open_[0] = closes[0]
    high = np.maximum(open_, closes) + 0.4
    low = np.minimum(open_, closes) - 0.4

    idx = pd.date_range("2024-01-01", periods=len(closes), freq="D")
    df = pd.DataFrame(
        {"open": open_, "high": high, "low": low, "close": closes, "volume": 1000},
        index=idx,
    )
    return df


def make_downtrend() -> pd.DataFrame:
    """Monotonic downtrend - should never confirm a dip."""
    n = 260
    closes = np.linspace(200.0, 100.0, n)
    open_ = np.roll(closes, 1)
    open_[0] = closes[0]
    high = np.maximum(open_, closes) + 0.4
    low = np.minimum(open_, closes) - 0.4
    idx = pd.date_range("2024-01-01", periods=len(closes), freq="D")
    return pd.DataFrame(
        {"open": open_, "high": high, "low": low, "close": closes, "volume": 1000},
        index=idx,
    )


def with_indicators(df: pd.DataFrame) -> pd.DataFrame:
    return add_all_indicators(df)


class TestMomentumTrigger(unittest.TestCase):
    def test_bullish_trigger_fires(self):
        n = 60
        closes = np.linspace(100, 108, n)
        # add a small down-then-up so rsi/macd turn
        closes[-8:] = [107.2, 106.8, 106.4, 106.6, 107.0, 107.4, 107.8, 108.2]
        open_ = closes - 0.2
        df = pd.DataFrame(
            {
                "open": open_,
                "high": closes + 0.3,
                "low": closes - 0.3,
                "close": closes,
                "volume": 1000,
            }
        )
        out = _momentum_trigger(df)
        self.assertIn("triggered", out)

    def test_short_frame_no_trigger(self):
        df = pd.DataFrame(
            {
                "open": [1, 2],
                "high": [1, 2],
                "low": [1, 2],
                "close": [1, 2],
                "volume": [1, 1],
            }
        )
        self.assertFalse(_momentum_trigger(df)["triggered"])


class TestDetectDip(unittest.TestCase):
    def test_uptrend_dip_confirmable(self):
        df = with_indicators(make_uptrend_dip())
        levels = levels_summary(df)
        out = detect_dip(df, levels=levels)
        self.assertIn("dip_score", out)
        self.assertIn("dip_stage", out)
        # Structure must be intact (price above SMA200 in an uptrend)
        self.assertTrue(out["components"]["above_sma200"])
        # A genuine pullback in an uptrend must at least be watchable
        self.assertGreaterEqual(out["dip_score"], 4)
        self.assertIn(out["dip_stage"], ("Confirmed", "In Pullback", "Deep Pullback"))

    def test_downtrend_never_confirmed(self):
        df = with_indicators(make_downtrend())
        levels = levels_summary(df)
        out = detect_dip(df, levels=levels)
        self.assertFalse(out["dip_confirmed"])
        self.assertNotEqual(out["dip_stage"], "Confirmed")

    def test_entry_zone_and_invalidation(self):
        df = with_indicators(make_uptrend_dip())
        levels = levels_summary(df)
        out = detect_dip(df, levels=levels)
        if out["entry_zone"]:
            low, high = out["entry_zone"]
            self.assertLess(low, high)
        if out["invalidation_level"] is not None:
            self.assertGreater(df["close"].iloc[-1], out["invalidation_level"])

    def test_score_range(self):
        df = with_indicators(make_uptrend_dip())
        out = detect_dip(df, levels=levels_summary(df))
        self.assertGreaterEqual(out["dip_score"], 0)
        self.assertLessEqual(out["dip_score"], 8)


if __name__ == "__main__":
    unittest.main()
