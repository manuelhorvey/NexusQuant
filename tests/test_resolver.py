"""
Tests for the on-demand data resolver (src/data/resolver.py): the
local-exact-group fast path, cross-group search, longest-history
preference, MT5 fallback, Yahoo fallback, and the FileNotFoundError
escalation with the cascade disabled.
"""

from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path

import pandas as pd

sys.path.append(str(Path(__file__).parent.parent))

from src.data.resolver import effective_group, find_local, resolve_symbol_data

SCHEMA = ["date", "symbol", "open", "high", "low", "close", "volume", "spread_points"]


def _write_frame(path: Path, dates, symbol="TEST") -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(
        {
            "date": pd.to_datetime(dates),
            "symbol": symbol,
            "open": 1.0,
            "high": 1.1,
            "low": 0.9,
            "close": 1.05,
            "volume": 100,
            "spread_points": 5,
        }
    )
    df.to_parquet(path, compression="zstd")


class FindLocalTest(unittest.TestCase):
    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp())
        (self.tmp / "full_fx").mkdir()

    def test_finds_in_any_group_folder(self):
        """A symbol in a non-requested group folder is still found."""
        _write_frame(
            self.tmp / "equity" / "AAPL_D1.parquet",
            pd.date_range("2024-01-01", periods=50),
        )
        got = find_local("AAPL", "D1", str(self.tmp))
        self.assertIsNotNone(got)
        self.assertEqual(got.name, "AAPL_D1.parquet")
        self.assertEqual(got.parent.name, "equity")

    def test_prefers_exact_group(self):
        _write_frame(
            self.tmp / "full_fx" / "EURUSD_D1.parquet",
            pd.date_range("2024-01-01", periods=100),
        )
        got = find_local("EURUSD", "D1", str(self.tmp), group="full_fx")
        self.assertEqual(got.parent.name, "full_fx")

    def test_prefers_longest_history(self):
        _write_frame(
            self.tmp / "a" / "X_D1.parquet", pd.date_range("2020-01-01", periods=500)
        )
        _write_frame(
            self.tmp / "b" / "X_D1.parquet", pd.date_range("2022-01-01", periods=50)
        )
        got = find_local("X", "D1", str(self.tmp))
        self.assertEqual(got.parent.name, "a")

    def test_nested_layout(self):
        (self.tmp / "mt5" / "D1").mkdir(parents=True)
        _write_frame(
            self.tmp / "mt5" / "D1" / "GBPUSD_D1.parquet",
            pd.date_range("2024-01-01", periods=50),
        )
        got = find_local("GBPUSD", "D1", str(self.tmp))
        self.assertIsNotNone(got)
        self.assertEqual(got.parent.name, "D1")

    def test_missing_returns_none(self):
        self.assertIsNone(find_local("NOPE", "D1", str(self.tmp)))


class ResolveTest(unittest.TestCase):
    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp())
        (self.tmp / "full_fx").mkdir()

    def test_local_exact_group(self):
        _write_frame(
            self.tmp / "full_fx" / "EURUSD_D1.parquet",
            pd.date_range("2024-01-01", periods=50),
        )
        path = resolve_symbol_data(
            "EURUSD",
            "D1",
            str(self.tmp),
            group="full_fx",
            allow_mt5=False,
            allow_yahoo=False,
        )
        self.assertEqual(path.name, "EURUSD_D1.parquet")

    def test_cross_group_local_without_network(self):
        """Local file in any group resolves even with both fetchers off."""
        _write_frame(
            self.tmp / "equity" / "AAPL_D1.parquet",
            pd.date_range("2024-01-01", periods=50),
        )
        path = resolve_symbol_data(
            "AAPL",
            "D1",
            str(self.tmp),
            group="full_fx",
            allow_mt5=False,
            allow_yahoo=False,
        )
        self.assertEqual(path.parent.name, "equity")

    def test_mt5_fallback(self):
        """Falls back to the MT5 bridge, cached into the classified group."""
        from unittest.mock import patch

        # The resolver imports the providers lazily, so patch the source
        # modules - not the resolver namespace.
        with patch("src.data.mt5.ensure_parquet") as mock:
            mock.return_value = self.tmp / "full_fx" / "EURUSD_D1.parquet"
            path = resolve_symbol_data(
                "EURUSD",
                "D1",
                str(self.tmp),
                group=None,
                allow_mt5=True,
                allow_yahoo=False,
            )
        self.assertEqual(path.name, "EURUSD_D1.parquet")
        mock.assert_called_once()
        # The destination group must be classified, never the raw root.
        args, kwargs = mock.call_args
        self.assertEqual(kwargs.get("group"), "full_fx")

    def test_yahoo_fallback_when_mt5_fails(self):
        """MT5 failure cascades to Yahoo, cached into the classified group."""
        from unittest.mock import patch
        from src.data.mt5 import MT5Error

        with (
            patch("src.data.mt5.ensure_parquet", side_effect=MT5Error("bridge down")),
            patch("src.data.yahoo.ensure_yahoo_parquet") as mock_y,
        ):
            mock_y.return_value = self.tmp / "equity" / "AAPL_D1.parquet"
            path = resolve_symbol_data(
                "AAPL",
                "D1",
                str(self.tmp),
                group=None,
                allow_mt5=True,
                allow_yahoo=True,
            )
        self.assertEqual(path.parent.name, "equity")
        args, kwargs = mock_y.call_args
        self.assertEqual(kwargs.get("group"), "equity")

    def test_raises_when_all_disabled(self):
        with self.assertRaises(FileNotFoundError) as ctx:
            resolve_symbol_data(
                "NOPE",
                "D1",
                str(self.tmp),
                group=None,
                allow_mt5=False,
                allow_yahoo=False,
            )
        self.assertIn("NOPE", str(ctx.exception))

    def test_raises_when_all_sources_fail(self):
        from unittest.mock import patch
        from src.data.mt5 import MT5Error
        from src.data.yahoo import YahooError

        with (
            patch("src.data.mt5.ensure_parquet", side_effect=MT5Error("bridge down")),
            patch(
                "src.data.yahoo.ensure_yahoo_parquet",
                side_effect=YahooError("no ticker"),
            ),
        ):
            with self.assertRaises(FileNotFoundError) as ctx:
                resolve_symbol_data(
                    "NOPE",
                    "D1",
                    str(self.tmp),
                    group=None,
                    allow_mt5=True,
                    allow_yahoo=True,
                )
        self.assertIn("bridge down", str(ctx.exception))
        self.assertIn("no ticker", str(ctx.exception))

    def test_equity_membership_classifies_universe(self):
        """S&P 500 constituents fetch into equity_universe/, not equity/."""
        from unittest.mock import patch
        from src.data.mt5 import MT5Error

        mem_dir = self.tmp / "equity_universe"
        mem_dir.mkdir(exist_ok=True)
        pd.DataFrame({"symbol": ["NVDA", "AAPL"]}).to_csv(
            mem_dir / "_membership.csv", index=False
        )
        with (
            patch("src.data.mt5.ensure_parquet", side_effect=MT5Error("bridge down")),
            patch("src.data.yahoo.ensure_yahoo_parquet") as mock_y,
        ):
            mock_y.return_value = self.tmp / "equity_universe" / "NVDA_D1.parquet"
            resolve_symbol_data(
                "NVDA",
                "D1",
                str(self.tmp),
                group=None,
                allow_mt5=True,
                allow_yahoo=True,
            )
        args, kwargs = mock_y.call_args
        self.assertEqual(kwargs.get("group"), "equity_universe")


class EffectiveGroupTest(unittest.TestCase):
    def test_flat_group(self):
        self.assertEqual(
            effective_group(Path("/d/equity/AAPL_D1.parquet"), "/d"), "equity"
        )

    def test_nested_layout_uses_grandparent(self):
        self.assertEqual(effective_group(Path("/d/mt5/D1/SYM_D1.parquet"), "/d"), "mt5")

    def test_root_file_falls_back(self):
        self.assertEqual(
            effective_group(Path("/d/SYM_D1.parquet"), "/d", fallback="full_fx"),
            "full_fx",
        )


if __name__ == "__main__":
    unittest.main()
