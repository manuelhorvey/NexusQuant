"""Tests for the equity factor scanner (src/equity/run.py)."""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent))

from src.equity.run import equity_read, equity_scan


class TestEquityRead(unittest.TestCase):
    def test_real_symbol(self):
        rec = equity_read("AAPL", fetch_news=False)
        self.assertEqual(rec["symbol"], "AAPL")
        self.assertIsNotNone(rec["close"])
        self.assertIsNotNone(rec["momentum"])
        # value/quality may be None without a fundamentals CSV, but the
        # technical fields must be present
        for key in (
            "composite",
            "rsi_14",
            "adx",
            "vs_sma200_pct",
            "bias_score",
            "news_score",
            "date",
        ):
            self.assertIn(key, rec)

    def test_missing_symbol_raises(self):
        with self.assertRaises(FileNotFoundError):
            equity_read("ZZZZZZ_NOT_A_SYMBOL", fetch_news=False)


class TestEquityScan(unittest.TestCase):
    def test_scan_small(self):
        table = equity_scan(symbols=["AAPL", "MSFT", "TSLA"])
        self.assertEqual(len(table), 3)
        self.assertIn("rank", table.columns)
        # sorted by composite descending
        comps = table["composite"].tolist()
        self.assertEqual(comps, sorted(comps, reverse=True))
        for col in ("value", "quality", "momentum", "composite", "symbol"):
            self.assertIn(col, table.columns)

    def test_scan_unknown_group(self):
        with self.assertRaises(RuntimeError):
            equity_scan(symbols=None, group="no_such_group_xyz")


if __name__ == "__main__":
    unittest.main()
