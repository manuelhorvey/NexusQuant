"""
NexusQuant - Scanner unit tests (stdlib unittest, run from project root):
    python -m unittest discover -s tests -v
"""

import sys
import unittest
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.analysis.scanner import (
    _directional_score,
    discover_symbols,
    scan_symbol,
    scan_universe,
)


def make_frame(
    close: float,
    sma200: float,
    rsi: float,
    macd_hist: float,
    adx: float,
    plus_di: float,
    minus_di: float,
) -> pd.DataFrame:
    """One-bar frame with the indicator columns _directional_score reads."""
    return pd.DataFrame(
        [
            {
                "close": close,
                "sma_200": sma200,
                "rsi_14": rsi,
                "macd_hist": macd_hist,
                "adx": adx,
                "plus_di": plus_di,
                "minus_di": minus_di,
            }
        ]
    )


class TestDirectionalScore(unittest.TestCase):
    def test_all_bullish(self):
        df = make_frame(
            close=110, sma200=100, rsi=60, macd_hist=1, adx=30, plus_di=30, minus_di=20
        )
        self.assertEqual(_directional_score(df)["score"], 4)
        self.assertEqual(_directional_score(df)["label"], "Strong Bullish")

    def test_all_bearish(self):
        df = make_frame(
            close=90, sma200=100, rsi=40, macd_hist=-1, adx=30, plus_di=20, minus_di=30
        )
        self.assertEqual(_directional_score(df)["score"], -4)
        self.assertEqual(_directional_score(df)["label"], "Strong Bearish")

    def test_neutral(self):
        df = make_frame(
            close=100, sma200=100, rsi=50, macd_hist=0, adx=20, plus_di=25, minus_di=25
        )
        self.assertEqual(_directional_score(df)["score"], 0)
        self.assertEqual(_directional_score(df)["label"], "Neutral")

    def test_label_boundaries(self):
        # +2 -> Bullish, +1 -> Mild Bullish, -1 -> Mild Bearish, -2 -> Bearish
        df = make_frame(
            close=105, sma200=100, rsi=55, macd_hist=0, adx=15, plus_di=1, minus_di=1
        )
        self.assertEqual(_directional_score(df)["score"], 2)
        self.assertEqual(_directional_score(df)["label"], "Bullish")

        # Only RSI above 50 -> +1
        df = make_frame(
            close=100, sma200=100, rsi=55, macd_hist=0, adx=15, plus_di=1, minus_di=1
        )
        self.assertEqual(_directional_score(df)["score"], 1)
        self.assertEqual(_directional_score(df)["label"], "Mild Bullish")

        # Only RSI below 50 -> -1
        df = make_frame(
            close=100, sma200=100, rsi=45, macd_hist=0, adx=15, plus_di=1, minus_di=1
        )
        self.assertEqual(_directional_score(df)["score"], -1)
        self.assertEqual(_directional_score(df)["label"], "Mild Bearish")

        # close and RSI both bearish -> -2
        df = make_frame(
            close=95, sma200=100, rsi=45, macd_hist=0, adx=15, plus_di=1, minus_di=1
        )
        self.assertEqual(_directional_score(df)["score"], -2)
        self.assertEqual(_directional_score(df)["label"], "Bearish")

    def test_low_adx_skips_di_component(self):
        # adx <= 25: DI direction must not contribute
        df = make_frame(
            close=100, sma200=100, rsi=50, macd_hist=0, adx=24, plus_di=99, minus_di=1
        )
        self.assertEqual(_directional_score(df)["score"], 0)


class TestDiscovery(unittest.TestCase):
    def test_discovers_group_symbols_d1(self):
        # after the folder cleanup, FX lives in full_fx/ (no top-level files)
        symbols = discover_symbols("data/raw", group="full_fx")
        self.assertIn("EURUSD", symbols)
        self.assertIn("GBPUSD", symbols)

    def test_discovers_group_symbols(self):
        symbols = discover_symbols("data/raw", group="candidates")
        self.assertTrue({"US30", "US500", "USTEC", "BTCUSD"} <= set(symbols))

    def test_group_timeframe_flat_layout(self):
        # after the backfill regroup, asset-class folders hold all
        # timeframes flat: full_fx/EURUSD_H1.parquet etc.
        symbols = discover_symbols("data/raw", group="full_fx", timeframe="H1")
        self.assertIn("EURUSD", symbols)
        symbols = discover_symbols("data/raw", group="full_fx", timeframe="H4")
        self.assertIn("EURUSD", symbols)


class TestScan(unittest.TestCase):
    def test_scan_symbol_structure(self):
        # Structural integrity of a real-data scan (directional-score *logic*
        # is covered by TestDirectionalScore with synthetic frames; this test
        # must not freeze a live market snapshot, which drifts over time).
        row = scan_symbol("EURJPY", group="full_fx")
        self.assertEqual(row["symbol"], "EURJPY")
        self.assertGreaterEqual(row["bias_score"], -4)
        self.assertLessEqual(row["bias_score"], 4)
        self.assertGreater(row["adx"], 0)
        self.assertIn(
            row["bias"],
            {
                "Strong Bullish",
                "Bullish",
                "Mild Bullish",
                "Neutral",
                "Mild Bearish",
                "Bearish",
                "Strong Bearish",
            },
        )

    def test_scan_universe_ranking_order(self):
        table = scan_universe("data/raw", group="candidates")
        scores = table["bias_score"].tolist()
        self.assertEqual(scores, sorted(scores, reverse=True))
        self.assertEqual(table["rank"].tolist(), list(range(1, len(table) + 1)))
        self.assertEqual(len(table), 4)


if __name__ == "__main__":
    unittest.main()
