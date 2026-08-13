"""
NexusQuant - database seeder.

Inserts a sample report snapshot + signal event so the persistence
endpoints (/api/v1/snapshots, /api/v1/signals) return data before any
live run. Idempotent: existing rows are updated in place (upsert by the
same unique key the API uses), so re-running is safe.

Usage:
    ./venv/bin/python database/seed_data/seed.py
    NEXUS_DATABASE_URL=postgresql+psycopg://nexus:nexus@localhost:5432/nexus \
        ./venv/bin/python database/seed_data/seed.py
"""

from __future__ import annotations

import sys
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from api.db import ReportSnapshot, SignalEvent, db


def seed(symbol: str = "XAUUSD", group: str = "full_fx", timeframe: str = "D1") -> dict:
    """Upsert one demo snapshot + one demo signal event; returns counts."""
    now = datetime.now(timezone.utc)
    as_of = str(now.date())

    snapshot_payload = {
        "symbol": symbol,
        "last_date": as_of,
        "last_close": 4417.3,
        "regime": {"regime": "Range / Chop", "adx": 16.8, "price_vs_200sma_pct": -3.41},
        "dip": {"dip_score": 1, "dip_stage": "No Uptrend", "dip_confirmed": False},
        "rating": {"prob_pct": 56.0, "rating": "Neutral"},
    }
    signal_payload = {"text": f"🎯 {symbol} — seed signal (demo row)"}

    with db.session() as s:
        snap = (
            s.query(ReportSnapshot)
            .filter_by(symbol=symbol, group=group, timeframe=timeframe, as_of=as_of)
            .first()
        )
        if snap:
            snap.payload = snapshot_payload
            snap.created_at = now
        else:
            s.add(
                ReportSnapshot(
                    symbol=symbol,
                    group=group,
                    timeframe=timeframe,
                    as_of=as_of,
                    payload=snapshot_payload,
                )
            )
        key = f"{symbol}:seed-{as_of}"
        ev = s.query(SignalEvent).filter_by(key=key).first()
        if ev:
            ev.payload = signal_payload
            ev.created_at = now
        else:
            s.add(SignalEvent(symbol=symbol, key=key, payload=signal_payload))
        s.commit()
    return {
        "backend": db.backend,
        "symbol": symbol,
        "as_of": as_of,
        "snapshot": "upserted",
        "signal_event": "upserted",
    }


if __name__ == "__main__":
    result = seed()
    print(
        f"seeded [{result['backend']}] {result['symbol']} "
        f"({result['as_of']}): snapshot {result['snapshot']}, "
        f"signal {result['signal_event']}"
    )
