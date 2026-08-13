"""Tests for the database seeder (database/seed_data/seed.py).

The seeder talks to the api.db singleton, so the SQLite test URL is set
before importing anything that touches the DB (same pattern as test_api.py).
"""

from __future__ import annotations

import os
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent))

_tmp = tempfile.mkdtemp(prefix="nexus_seed_test_")
os.environ["NEXUS_DATABASE_URL"] = f"sqlite:///{_tmp}/seed.db"

from database.seed_data.seed import seed  # noqa: E402
from api.db import ReportSnapshot, SignalEvent, db  # noqa: E402


class TestSeeder(unittest.TestCase):
    @classmethod
    def tearDownClass(cls):
        # Release the sqlite connection pool so the process exits cleanly
        # (avoids a ResourceWarning about an unclosed database).
        db.engine.dispose()

    def setUp(self):
        # The api.db singleton persists across test methods; clear both
        # tables so each test starts clean.
        with db.session() as s:
            s.query(ReportSnapshot).delete()
            s.query(SignalEvent).delete()
            s.commit()

    def test_seed_creates_rows(self):
        result = seed(symbol="XAUUSD", group="full_fx", timeframe="D1")
        self.assertEqual(result["snapshot"], "upserted")
        self.assertEqual(result["signal_event"], "upserted")
        self.assertEqual(result["backend"], "sqlite")
        with db.session() as s:
            self.assertEqual(s.query(ReportSnapshot).count(), 1)
            self.assertEqual(s.query(SignalEvent).count(), 1)
            snap = s.query(ReportSnapshot).first()
            self.assertEqual(snap.symbol, "XAUUSD")
            self.assertEqual(snap.group, "full_fx")
            self.assertEqual(snap.timeframe, "D1")
            self.assertEqual(snap.payload["symbol"], "XAUUSD")

    def test_seed_is_idempotent(self):
        seed(symbol="GBPUSD")
        seed(symbol="GBPUSD")  # re-run must not duplicate
        with db.session() as s:
            self.assertEqual(s.query(ReportSnapshot).count(), 1)
            self.assertEqual(s.query(SignalEvent).count(), 1)

    def test_seed_different_symbols_coexist(self):
        seed(symbol="EURUSD")
        seed(symbol="GBPUSD")
        with db.session() as s:
            symbols = {r.symbol for r in s.query(ReportSnapshot).all()}
            self.assertEqual(symbols, {"EURUSD", "GBPUSD"})
            # signal events are keyed per symbol -> one each
            self.assertEqual(s.query(SignalEvent).count(), 2)

    def test_seed_uses_live_upsert_not_stale_row(self):
        # After re-seeding the same symbol, the payload reflects the newer
        # as_of (upsert by the API's unique key, not a second row).
        seed(symbol="EURUSD")
        seed(symbol="EURUSD")
        with db.session() as s:
            rows = (
                s.query(ReportSnapshot).filter(ReportSnapshot.symbol == "EURUSD").all()
            )
            self.assertEqual(len(rows), 1)


if __name__ == "__main__":
    unittest.main()
