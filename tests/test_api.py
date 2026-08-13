"""
Tests for the FastAPI service (api/app.py + api/db.py).

The DB singleton is created at import time, so the SQLite test URL is set
before importing ``api.app``. Endpoint tests use the real local data (a
small symbol set) so the integration is honest; the persistence tests verify
snapshot + signal-event storage round-trips.
"""

from __future__ import annotations

import os
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent))

_tmp = tempfile.mkdtemp(prefix="nexus_api_test_")
os.environ["NEXUS_DATABASE_URL"] = f"sqlite:///{_tmp}/test.db"

import pandas as pd  # noqa: E402
from fastapi.testclient import TestClient  # noqa: E402

from api.app import app, _records, _clean  # noqa: E402

client = TestClient(app)

TEST_SYMBOLS = ["EURUSD", "GBPUSD"]


class TestHealthAndDiscovery(unittest.TestCase):
    def test_health(self):
        r = client.get("/health")
        self.assertEqual(r.status_code, 200)
        body = r.json()
        self.assertEqual(body["status"], "ok")
        self.assertEqual(body["db_backend"], "sqlite")  # test env fallback

    def test_groups(self):
        r = client.get("/api/v1/groups")
        self.assertEqual(r.status_code, 200)
        groups = r.json()
        self.assertIsInstance(groups, list)
        self.assertTrue(any(g.get("group") == "full_fx" for g in groups))

    def test_scan_small(self):
        r = client.get(
            "/api/v1/scan",
            params={
                "group": "full_fx",
                "timeframe": "D1",
                "symbols": ",".join(TEST_SYMBOLS),
            },
        )
        self.assertEqual(r.status_code, 200, r.text)
        body = r.json()
        self.assertEqual(body["n"], 2)
        syms = {row["symbol"] for row in body["rows"]}
        self.assertEqual(syms, set(TEST_SYMBOLS))
        row = body["rows"][0]
        for key in (
            "symbol",
            "regime",
            "bias",
            "bias_score",
            "dip_score",
            "rating",
            "pattern",
            "support",
            "resistance",
        ):
            self.assertIn(key, row)

    def test_scan_unknown_group_422(self):
        r = client.get(
            "/api/v1/scan", params={"group": "no_such_group", "timeframe": "D1"}
        )
        self.assertEqual(r.status_code, 422)


class TestSymbolEndpoints(unittest.TestCase):
    def test_report(self):
        r = client.get(
            "/api/v1/symbol/EURUSD", params={"group": "full_fx", "timeframe": "D1"}
        )
        self.assertEqual(r.status_code, 200, r.text)
        rep = r.json()
        for section in (
            "regime",
            "levels",
            "dip",
            "moving_averages",
            "volume_flow",
            "patterns",
            "stress",
            "rating",
        ):
            self.assertIn(section, rep)
        self.assertEqual(rep["symbol"], "EURUSD")
        self.assertIsNotNone(rep["last_date"])

    def test_report_404(self):
        r = client.get("/api/v1/symbol/NOSUCHSYMBOL", params={"group": "full_fx"})
        self.assertEqual(r.status_code, 404)

    def test_risk(self):
        r = client.get(
            "/api/v1/symbol/EURUSD/risk", params={"group": "full_fx", "equity": 250000}
        )
        self.assertEqual(r.status_code, 200, r.text)
        plan = r.json()
        self.assertEqual(plan["symbol"], "EURUSD")
        # Either an actionable setup with sizes, or a graceful no-setup reason
        if plan.get("setup"):
            self.assertEqual(len(plan["sizes"]), 3)
        else:
            self.assertIn("reason", plan)

    def test_stress(self):
        r = client.get("/api/v1/symbol/EURUSD/stress", params={"group": "full_fx"})
        self.assertEqual(r.status_code, 200, r.text)
        st = r.json()
        self.assertIn("available", st)
        if st["available"]:
            self.assertEqual(len(st["scenarios"]), 3)
            for row in st["scenarios"]:
                self.assertIn("loss_usd", row)
                self.assertIn("daily_limit_breach", row)

    def test_backtest(self):
        r = client.get("/api/v1/symbol/EURUSD/backtest", params={"group": "full_fx"})
        self.assertEqual(r.status_code, 200, r.text)
        bt = r.json()
        self.assertIn("stats", bt)
        self.assertIn("win_rate", bt["stats"])


class TestPersistence(unittest.TestCase):
    def test_snapshot_roundtrip(self):
        r = client.post(
            "/api/v1/snapshots", params={"symbol": "GBPUSD", "group": "full_fx"}
        )
        self.assertEqual(r.status_code, 200, r.text)
        body = r.json()
        self.assertEqual(body["status"], "stored")
        self.assertIn("as_of", body)

        r = client.get(
            "/api/v1/snapshots", params={"symbol": "GBPUSD", "group": "full_fx"}
        )
        self.assertEqual(r.status_code, 200)
        snaps = r.json()
        self.assertTrue(len(snaps) >= 1)
        self.assertEqual(snaps[0]["symbol"], "GBPUSD")
        self.assertIn("payload", snaps[0])

    def test_snapshot_dedup_by_date(self):
        # Same symbol/date stored twice -> unique constraint keeps one row.
        client.post(
            "/api/v1/snapshots", params={"symbol": "GBPUSD", "group": "full_fx"}
        )
        client.post(
            "/api/v1/snapshots", params={"symbol": "GBPUSD", "group": "full_fx"}
        )
        r = client.get(
            "/api/v1/snapshots", params={"symbol": "GBPUSD", "group": "full_fx"}
        )
        snaps = r.json()
        dates = [s["as_of"] for s in snaps]
        self.assertEqual(len(dates), len(set(dates)))  # no duplicate as_of

    def test_report_persist_flag(self):
        r = client.get(
            "/api/v1/symbol/EURUSD", params={"group": "full_fx", "persist": "true"}
        )
        self.assertEqual(r.status_code, 200, r.text)
        r = client.get(
            "/api/v1/snapshots", params={"symbol": "EURUSD", "group": "full_fx"}
        )
        self.assertTrue(len(r.json()) >= 1)


class TestLivePass(unittest.TestCase):
    def test_live_pass_endpoint(self):
        r = client.get(
            "/api/v1/live/pass", params={"group": "full_fx", "symbols": "EURUSD,GBPUSD"}
        )
        self.assertEqual(r.status_code, 200, r.text)
        body = r.json()
        self.assertIn("scanned", body)
        self.assertIn("candidates", body)
        self.assertIn("data_age_days", body)
        self.assertIn("new_alerts", body)

    def test_live_pass_dry_run_does_not_write_state(self):
        state = Path("data/live/alerts.json")
        before = state.read_text() if state.exists() else None
        r = client.get(
            "/api/v1/live/pass", params={"group": "full_fx", "symbols": "EURUSD,GBPUSD"}
        )
        self.assertEqual(r.status_code, 200, r.text)
        after = state.read_text() if state.exists() else None
        self.assertEqual(before, after)  # dry_run default leaves no trace

    def test_live_pass_persist_flag(self):
        # persist=true stores events in the DB (dry_run still default)
        r = client.get(
            "/api/v1/live/pass",
            params={"group": "full_fx", "symbols": "EURUSD,GBPUSD", "persist": "true"},
        )
        self.assertEqual(r.status_code, 200, r.text)
        r = client.get("/api/v1/signals")
        self.assertEqual(r.status_code, 200)

    def test_signals_content_roundtrip(self):
        # Store an event directly, then verify its content round-trips.
        from datetime import datetime
        from api.db import SignalEvent, db

        with db.session() as s:
            s.add(
                SignalEvent(
                    symbol="EURUSD",
                    key="EURUSD:test",
                    payload={"text": "test alert"},
                    created_at=datetime(2026, 8, 12, 9, 0),
                )
            )
            s.commit()
        r = client.get("/api/v1/signals", params={"symbol": "EURUSD"})
        self.assertEqual(r.status_code, 200)
        events = r.json()
        self.assertTrue(any(e["key"] == "EURUSD:test" for e in events))
        found = [e for e in events if e["key"] == "EURUSD:test"][0]
        self.assertEqual(found["payload"]["text"], "test alert")
        self.assertEqual(found["created_at"], "2026-08-12T09:00:00")


class TestHelpers(unittest.TestCase):
    def test_clean_handles_numpy_and_nan(self):
        import numpy as np

        d = {
            "a": np.float64(1.5),
            "b": np.nan,
            "c": float("inf"),
            "d": np.int64(3),
            "e": {"f": np.float64(-0.0)},
            "g": [np.float64(2.0), None],
            "h": True,
        }
        out = _clean(d)
        self.assertEqual(out["a"], 1.5)
        self.assertIsNone(out["b"])
        self.assertIsNone(out["c"])
        self.assertEqual(out["d"], 3)
        self.assertEqual(out["g"], [2.0, None])
        self.assertIs(out["h"], True)
        # list-of-dicts path
        df = __import__("pandas").DataFrame(
            {"symbol": ["A"], "v": [np.nan], "s": [1.0]}
        )
        recs = _records(df)
        self.assertIsNone(recs[0]["v"])
        self.assertEqual(recs[0]["s"], 1.0)

    def test_clean_datetime_and_nested(self):
        from datetime import date, datetime
        import numpy as np

        d = {
            "dt": datetime(2026, 8, 12, 9, 30),
            "day": date(2026, 8, 12),
            "ts": pd.Timestamp("2026-08-12"),
            "nested": {"arr": np.array([1.0, np.nan, 2.0])},
        }
        out = _clean(d)
        self.assertEqual(out["dt"], "2026-08-12T09:30:00")
        self.assertEqual(out["day"], "2026-08-12")
        self.assertEqual(out["ts"], "2026-08-12T00:00:00")
        self.assertEqual(out["nested"]["arr"], [1.0, None, 2.0])


if __name__ == "__main__":
    unittest.main()
