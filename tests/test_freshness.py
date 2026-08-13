"""
Tests for the data freshness layer (src/data/freshness.py + the merge
logic in src/data/update.py): staleness thresholds (weekend absorption),
missing files, the freshness report, and the incremental merge/dedupe.
"""

from __future__ import annotations

import sys
import tempfile
import unittest
from datetime import datetime
from pathlib import Path

import pandas as pd

sys.path.append(str(Path(__file__).parent.parent))

from src.data.freshness import (
    DEFAULT_MAX_STALE_DAYS,
    freshness_report,
    is_stale,
    last_bar_date,
    staleness_days,
    summary,
)
from src.data.update import fetch_missing

SCHEMA = ["date", "symbol", "open", "high", "low", "close", "volume", "spread_points"]


def _write_frame(path: Path, dates, symbol="TEST") -> None:
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


class _FakeProvider:
    """MT5Provider stub: returns the queued frames for copy_rates_range."""

    def __init__(self, frame):
        self.frame = frame
        self.calls = []

    def copy_rates_range(self, symbol, timeframe, date_from, date_to=None):
        self.calls.append((symbol, timeframe, date_from, date_to))
        return self.frame

    def copy_rates_from_pos(self, symbol, timeframe, start=0, count=None):
        # ensure_parquet (the created-file path) only checks .empty and
        # writes; return an empty-but-valid frame so it can succeed.
        return pd.DataFrame(columns=SCHEMA)

    def close(self):
        pass

    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False


class TestStaleness(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.dir = Path(self.tmp.name)
        # Friday 2026-08-07
        self.friday = datetime(2026, 8, 7)

    def tearDown(self):
        self.tmp.cleanup()

    def _path(self, days_ago: int) -> Path:
        p = self.dir / f"S_{days_ago}.parquet"
        date = pd.Timestamp(self.friday) - pd.Timedelta(days=days_ago)
        _write_frame(p, [date])
        return p

    def test_last_bar_date(self):
        p = self._path(0)
        self.assertEqual(last_bar_date(p), pd.Timestamp("2026-08-07"))

    def test_missing_file_is_none(self):
        self.assertIsNone(last_bar_date(self.dir / "nope.parquet"))

    def test_fresh_on_monday(self):
        # Friday bar on Monday = 3 calendar days old - not stale (weekend).
        p = self._path(0)
        self.assertFalse(is_stale(p, "D1", today=datetime(2026, 8, 10)))
        self.assertEqual(staleness_days(p, datetime(2026, 8, 10)), 3)

    def test_stale_by_thursday(self):
        # Friday bar on Thursday = 6 days old - stale (> 4 for D1).
        p = self._path(0)
        self.assertTrue(is_stale(p, "D1", today=datetime(2026, 8, 13)))

    def test_missing_file_is_stale(self):
        self.assertTrue(is_stale(self.dir / "nope.parquet", "D1"))

    def test_threshold_customizable(self):
        p = self._path(0)
        self.assertTrue(
            is_stale(p, "D1", max_stale_days={"D1": 2}, today=datetime(2026, 8, 10))
        )
        self.assertFalse(
            is_stale(p, "D1", max_stale_days={"D1": 10}, today=datetime(2026, 8, 10))
        )

    def test_default_thresholds(self):
        self.assertEqual(DEFAULT_MAX_STALE_DAYS["D1"], 4)
        self.assertLess(DEFAULT_MAX_STALE_DAYS["H1"], DEFAULT_MAX_STALE_DAYS["D1"])


class TestFreshnessReport(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.dir = Path(self.tmp.name)
        self.group = self.dir / "fx"
        self.group.mkdir()
        _write_frame(
            self.group / "AAA_D1.parquet", ["2026-08-10", "2026-08-11"], symbol="AAA"
        )
        _write_frame(self.group / "BBB_D1.parquet", ["2026-07-20"], symbol="BBB")

    def tearDown(self):
        self.tmp.cleanup()

    def test_report_flags_fresh_and_stale(self):
        rep = freshness_report(str(self.dir), "fx", "D1", today=datetime(2026, 8, 11))
        by = {r["symbol"]: r["status"] for r in rep.to_dict(orient="records")}
        self.assertEqual(by["AAA"], "FRESH")
        self.assertEqual(by["BBB"], "STALE")

    def test_summary_counts(self):
        rep = freshness_report(str(self.dir), "fx", "D1", today=datetime(2026, 8, 11))
        s = summary(rep)
        self.assertEqual(s["total"], 2)
        self.assertEqual(s["fresh"], 1)
        self.assertEqual(s["stale"], 1)
        self.assertEqual(s["missing"], 0)


class TestIncrementalUpdate(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.dir = Path(self.tmp.name)
        self.group = self.dir / "fx"
        self.group.mkdir()

    def tearDown(self):
        self.tmp.cleanup()

    def test_append_new_bars(self):
        path = self.group / "AAA_D1.parquet"
        _write_frame(path, ["2026-08-08", "2026-08-09"], symbol="AAA")
        new = pd.DataFrame(
            {
                "date": pd.to_datetime(["2026-08-09", "2026-08-10", "2026-08-11"]),
                "symbol": "AAA",
                "open": 1.0,
                "high": 1.1,
                "low": 0.9,
                "close": 1.05,
                "volume": 100,
                "spread_points": 5,
            }
        )
        status, added, total, err = fetch_missing(
            _FakeProvider(new), "AAA", "D1", "fx", str(self.dir)
        )
        self.assertEqual(status, "updated")
        self.assertEqual(added, 2)  # 08-10, 08-11 (08-09 deduped)
        self.assertEqual(total, 4)
        merged = pd.read_parquet(path)
        self.assertEqual(len(merged), 4)
        # No duplicate dates.
        self.assertEqual(merged["date"].nunique(), 4)

    def test_overlap_is_deduped(self):
        # Provider returns bars that fully overlap the existing file.
        path = self.group / "AAA_D1.parquet"
        _write_frame(path, ["2026-08-08", "2026-08-09"], symbol="AAA")
        new = pd.DataFrame(
            {
                "date": pd.to_datetime(["2026-08-08", "2026-08-09"]),
                "symbol": "AAA",
                "open": 1.0,
                "high": 1.1,
                "low": 0.9,
                "close": 1.05,
                "volume": 100,
                "spread_points": 5,
            }
        )
        status, added, total, _ = fetch_missing(
            _FakeProvider(new), "AAA", "D1", "fx", str(self.dir)
        )
        self.assertEqual(status, "current")
        self.assertEqual(added, 0)

    def test_created_when_missing(self):
        # No existing file -> full download via ensure_parquet. The fake
        # returns an empty schema frame, so ensure_parquet raises MT5Error
        # ("no data"); either way it must not crash with a None deref.
        status, added, total, err = fetch_missing(
            _FakeProvider(pd.DataFrame(columns=SCHEMA)),
            "ZZZ",
            "D1",
            "fx",
            str(self.dir),
        )
        self.assertIn(status, ("created", "error"))


if __name__ == "__main__":
    unittest.main()
