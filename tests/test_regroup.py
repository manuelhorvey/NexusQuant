"""
Tests for the backfill regroup tool (src/data/regroup.py): symbol
classification and the keep-longest duplicate policy.
"""

from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.data.regroup import classify_symbol, plan_merge


class TestClassifySymbol(unittest.TestCase):
    def test_fx_pairs(self):
        for s in ["EURUSD", "GBPCZK", "USDAED", "AUDTRY", "USDTRY", "CHFDKK"]:
            self.assertEqual(classify_symbol(s, set()), "full_fx", s)

    def test_crypto(self):
        for s in [
            "ADAUSD",
            "XRPUSD",
            "XTZUSD",
            "BTCUSDT",
            "BTCUSD",
            "BTCAUD",
            "ETHBTC",
            "LINKUSD",
        ]:
            self.assertEqual(classify_symbol(s, set()), "crypto", s)

    def test_metals(self):
        for s in ["XAUUSD", "XAGJPY", "XAUUSD247", "XAGAUD", "XPTUSD"]:
            self.assertEqual(classify_symbol(s, set()), "metals", s)

    def test_commodities(self):
        for s in ["XALUSD", "XCUUSD", "XNGUSD", "XZNUSD", "USOIL"]:
            self.assertEqual(classify_symbol(s, set()), "commodities", s)

    def test_indices(self):
        for s in ["US30_x10", "US500_x100", "AUS200", "DXY", "DE30", "JP225", "IN50"]:
            self.assertEqual(classify_symbol(s, set()), "indices", s)

    def test_equity_universe_membership_wins(self):
        self.assertEqual(classify_symbol("AAPL", {"AAPL", "MSFT"}), "equity_universe")
        # not a member -> generic equity bucket
        self.assertEqual(classify_symbol("AAPL", set()), "equity")

    def test_generic_equity_catchall(self):
        for s in ["AMC", "BABA", "NIO", "TSM"]:
            self.assertEqual(classify_symbol(s, set()), "equity", s)


class TestMergePolicy(unittest.TestCase):
    def _write(self, path: Path, n_bars: int) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        idx = pd.date_range("2020-01-01", periods=n_bars, freq="D")
        df = pd.DataFrame(
            {
                "date": idx,
                "open": 1.0,
                "high": 1.0,
                "low": 1.0,
                "close": 1.0,
                "volume": 1,
            }
        )
        df.to_parquet(path)

    def test_keep_longest_wins(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            src = root / "staging" / "D1"
            (root / "full_fx").mkdir(parents=True)
            self._write(src / "EURUSD_D1.parquet", 400)  # source shorter
            self._write(root / "full_fx" / "EURUSD_D1.parquet", 2500)

            plan, n_move, n_skip = plan_merge(
                [str(src)],
                str(root),
                curated={"full_fx": {"EURUSD"}},
                equity_membership=set(),
            )
            self.assertEqual(n_skip, 1)  # shorter duplicate dropped
            self.assertEqual(plan[0][2], "skip-shorter")

    def test_source_longer_overwrites(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            src = root / "staging" / "D1"
            (root / "full_fx").mkdir(parents=True)
            self._write(src / "EURUSD_D1.parquet", 3000)  # source longer
            self._write(root / "full_fx" / "EURUSD_D1.parquet", 2500)

            plan, n_move, n_skip = plan_merge(
                [str(src)],
                str(root),
                curated={"full_fx": {"EURUSD"}},
                equity_membership=set(),
            )
            self.assertEqual(n_move, 1)
            self.assertEqual(plan[0][2], "move")

    def test_new_file_moves(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            src = root / "staging" / "D1"
            self._write(src / "USOIL_D1.parquet", 100)
            plan, n_move, n_skip = plan_merge(
                [str(src)], str(root), curated={}, equity_membership=set()
            )
            self.assertEqual(n_move, 1)
            dst = Path(plan[0][1])
            self.assertEqual(dst.parent.name, "commodities")

    def test_unreadable_source_kept_not_deleted(self):
        # a source that cannot be read must never be deleted on an
        # unconfirmed "shorter" - keep-both for manual review.
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            src = root / "staging" / "D1"
            src.mkdir(parents=True)
            (root / "full_fx").mkdir(parents=True)
            (src / "EURUSD_D1.parquet").write_bytes(b"not a parquet file")
            self._write(root / "full_fx" / "EURUSD_D1.parquet", 2500)

            plan, n_move, n_skip = plan_merge(
                [str(src)],
                str(root),
                curated={"full_fx": {"EURUSD"}},
                equity_membership=set(),
            )
            self.assertEqual(len(plan), 1)
            self.assertEqual(plan[0][2], "keep-both")
            self.assertTrue(Path(plan[0][0]).exists())  # source untouched

    def test_archive_short_moves_and_manifests(self):
        from src.data.regroup import archive_short

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "full_fx").mkdir(parents=True)
            self._write(root / "full_fx" / "USDAOA_D1.parquet", 120)
            self._write(root / "full_fx" / "EURUSD_D1.parquet", 2500)
            moved = archive_short(str(root), str(root / "archive"), min_bars=250)
            self.assertEqual(len(moved), 1)
            self.assertTrue((root / "archive" / "USDAOA_D1.parquet").exists())
            self.assertTrue((root / "archive" / "_manifest.txt").exists())
            # the usable file is untouched
            self.assertTrue((root / "full_fx" / "EURUSD_D1.parquet").exists())

    def test_prune_empty_ancestors(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            src = root / "staging" / "D1"
            self._write(src / "USOIL_D1.parquet", 100)
            plan, _, _ = plan_merge(
                [str(src)], str(root), curated={}, equity_membership=set()
            )
            from src.data.regroup import execute_merge

            execute_merge(plan, prune=True)
            # the empty staging/D1 and staging ancestors are pruned
            self.assertFalse((root / "staging" / "D1").exists())
            self.assertFalse((root / "staging").exists())
            self.assertTrue((root / "commodities" / "USOIL_D1.parquet").exists())


if __name__ == "__main__":
    unittest.main()
