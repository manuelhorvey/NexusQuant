"""
Tests for the Yahoo OHLCV provider (src/data/yahoo.py): symbol -> ticker
mapping (indices / metals / commodities / FX / crypto / stocks), chart
payload parsing (null padding dropped, OHLC integrity), H4 resampling,
and the classified-folder parquet cache.
"""

from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import numpy as np
import pandas as pd

sys.path.append(str(Path(__file__).parent.parent))

from src.data.yahoo import (
    YahooError,
    ensure_yahoo_parquet,
    fetch_ohlcv,
    yahoo_ticker,
)


class YahooTickerTest(unittest.TestCase):
    def test_indices(self):
        self.assertEqual(yahoo_ticker("US500"), "^GSPC")
        self.assertEqual(yahoo_ticker("USTEC"), "^NDX")
        self.assertEqual(yahoo_ticker("US30"), "^DJI")
        self.assertEqual(yahoo_ticker("DXY"), "DX-Y.NYB")

    def test_metals_and_commodities(self):
        self.assertEqual(yahoo_ticker("XAUUSD"), "GC=F")
        self.assertEqual(yahoo_ticker("XAGUSD"), "SI=F")
        self.assertEqual(yahoo_ticker("USOIL"), "CL=F")
        self.assertEqual(yahoo_ticker("XBRUSD"), "BZ=F")

    def test_fx_pairs(self):
        self.assertEqual(yahoo_ticker("EURUSD"), "EURUSD=X")
        self.assertEqual(yahoo_ticker("GBPJPY"), "GBPJPY=X")

    def test_crypto(self):
        self.assertEqual(yahoo_ticker("BTCUSD"), "BTC-USD")
        self.assertEqual(yahoo_ticker("ETHUSD"), "ETH-USD")

    def test_stocks_etfs_passthrough(self):
        self.assertEqual(yahoo_ticker("AAPL"), "AAPL")
        self.assertEqual(yahoo_ticker("GLD"), "GLD")

    def test_case_insensitive(self):
        self.assertEqual(yahoo_ticker("aapl"), "AAPL")
        self.assertEqual(yahoo_ticker("eurusd"), "EURUSD=X")


def _fake_payload(n=60, freq="D"):
    """Synthetic chart payload: n bars, starting 2024-01-01."""
    ts = pd.date_range("2024-01-01", periods=n, freq=freq)
    closes = np.linspace(100.0, 110.0, n)
    return {
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
                                "volume": [1000] * n,
                            }
                        ]
                    },
                }
            ],
            "error": None,
        }
    }


class FetchOhlcvTest(unittest.TestCase):
    def test_daily_parse(self):
        with mock.patch("src.data.yahoo._chart", return_value=_fake_payload()):
            df = fetch_ohlcv("AAPL", "D1")
        self.assertIsNotNone(df)
        self.assertEqual(len(df), 60)
        for col in ("date", "open", "high", "low", "close", "volume"):
            self.assertIn(col, df.columns)
        self.assertTrue((df[["open", "high", "low", "close"]] > 0).all().all())

    def test_null_padding_dropped(self):
        payload = _fake_payload()
        quote = payload["chart"]["result"][0]["indicators"]["quote"][0]
        quote["close"][:5] = [None] * 5  # Yahoo pads early series with nulls
        with mock.patch("src.data.yahoo._chart", return_value=payload):
            df = fetch_ohlcv("AAPL", "D1")
        self.assertEqual(len(df), 55)

    def test_h4_resampled_to_4h(self):
        """H4 = 1h payload resampled to UTC-anchored 4h bars (24 -> 6)."""
        with mock.patch(
            "src.data.yahoo._chart", return_value=_fake_payload(n=24, freq="1h")
        ):
            df = fetch_ohlcv("AAPL", "H4")
        self.assertIsNotNone(df)
        self.assertEqual(len(df), 6)
        # First bucket open = first 1h bar's open; last bucket close =
        # last 1h bar's close (24 hourly bars end at 110.0).
        self.assertAlmostEqual(df["open"].iloc[0], 100.0 - 0.2, places=4)
        self.assertAlmostEqual(df["close"].iloc[-1], 110.0, places=4)

    def test_unknown_ticker_returns_none(self):
        with mock.patch("src.data.yahoo._chart", return_value=None):
            self.assertIsNone(fetch_ohlcv("ZZZZ", "D1"))


class EnsureParquetTest(unittest.TestCase):
    def test_writes_classified_group(self):
        tmp = Path(tempfile.mkdtemp())
        with mock.patch("src.data.yahoo.fetch_ohlcv", return_value=_frame_for_test()):
            path = ensure_yahoo_parquet("AAPL", "D1", str(tmp))
        self.assertEqual(path.name, "AAPL_D1.parquet")
        self.assertEqual(path.parent.name, "equity")
        df = pd.read_parquet(path)
        self.assertEqual(len(df), 60)
        for col in (
            "date",
            "symbol",
            "open",
            "high",
            "low",
            "close",
            "volume",
            "spread_points",
        ):
            self.assertIn(col, df.columns)

    def test_raises_when_no_data(self):
        tmp = Path(tempfile.mkdtemp())
        with mock.patch("src.data.yahoo.fetch_ohlcv", return_value=None):
            with self.assertRaises(YahooError):
                ensure_yahoo_parquet("NOPE", "D1", str(tmp))


def _frame_for_test():
    n = 60
    closes = np.linspace(100.0, 110.0, n)
    return pd.DataFrame(
        {
            "date": pd.date_range("2024-01-01", periods=n),
            "open": closes - 0.2,
            "high": closes + 0.5,
            "low": closes - 0.5,
            "close": closes,
            "volume": 1000,
        }
    )


if __name__ == "__main__":
    unittest.main()
