"""
NexusQuant - MT5 provider unit tests (no live MT5 connection required).
Run from project root:
    python -m unittest discover -s tests -v
"""

import sys
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.data.mt5 import SCHEMA, MT5Error, rates_to_frame, timeframe_id


class TestTimeframeId(unittest.TestCase):
    def test_common_timeframes(self):
        self.assertEqual(timeframe_id("D1"), 16408)
        self.assertEqual(timeframe_id("H1"), 16385)
        self.assertEqual(timeframe_id("H4"), 16388)
        self.assertEqual(timeframe_id("M15"), 15)

    def test_case_insensitive(self):
        self.assertEqual(timeframe_id("d1"), 16408)
        self.assertEqual(timeframe_id("h1"), 16385)

    def test_invalid_timeframe(self):
        with self.assertRaises(ValueError):
            timeframe_id("XYZ")
        with self.assertRaises(ValueError):
            timeframe_id("")


class TestRatesToFrame(unittest.TestCase):
    def test_normalise_named_array(self):
        raw = np.array(
            [
                (1786428000, 1.15, 1.16, 1.14, 1.15, 100, 6, 0),
                (1786431600, 1.15, 1.16, 1.14, 1.15, 200, 7, 0),
            ],
            dtype=[
                ("time", "i8"),
                ("open", "f8"),
                ("high", "f8"),
                ("low", "f8"),
                ("close", "f8"),
                ("tick_volume", "i8"),
                ("spread", "i8"),
                ("real_volume", "i8"),
            ],
        )
        df = rates_to_frame(raw, "EURUSD")
        self.assertEqual(list(df.columns), SCHEMA)
        self.assertEqual(len(df), 2)
        self.assertEqual(list(df["symbol"]), ["EURUSD", "EURUSD"])
        self.assertEqual(list(df["volume"]), [100, 200])
        self.assertEqual(list(df["spread_points"]), [6, 7])
        self.assertEqual(df["date"].min().year, 2026)

    def test_empty_input(self):
        df = rates_to_frame(None, "EURUSD")
        self.assertTrue(df.empty)
        self.assertEqual(list(df.columns), SCHEMA)

    def test_missing_ohlc_raises(self):
        raw = np.array([(1, 2)], dtype=[("time", "i8"), ("close", "f8")])
        with self.assertRaises(MT5Error):
            rates_to_frame(raw, "EURUSD")


if __name__ == "__main__":
    unittest.main()
