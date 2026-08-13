"""
NexusQuant - Persistence layer (SQLAlchemy).

Stores report snapshots and live signal events so the API serves a queryable
history, not just live computations.

Backend selection (in priority order):

1. ``NEXUS_DATABASE_URL`` env var - anything SQLAlchemy can reach; the
   docker-compose stack sets it to the Postgres service, e.g.
   ``postgresql+psycopg://nexus:nexus@db:5432/nexus``.
2. **Graceful SQLite fallback** - if no env var is set (or Postgres is
   unreachable), a local SQLite file ``data/db/nexus.db`` is used so the API
   works with zero Docker/Postgres setup. ``/health`` reports which backend
   is active.

Usage:
    from api.db import db

    with db.session() as s:
        snap = ReportSnapshot(symbol="XAUUSD", ...)
        s.add(snap)
        s.commit()
"""

from __future__ import annotations

import os
from datetime import datetime
from pathlib import Path
from typing import Optional

from sqlalchemy import (
    JSON,
    Column,
    DateTime,
    Integer,
    String,
    UniqueConstraint,
    create_engine,
    text,
)
from sqlalchemy.orm import declarative_base, sessionmaker

Base = declarative_base()

DEFAULT_SQLITE_PATH = "data/db/nexus.db"
SQLITE_URL = f"sqlite:///{Path(DEFAULT_SQLITE_PATH).resolve()}"


class ReportSnapshot(Base):
    """One persisted institutional report (payload = the full report dict)."""

    __tablename__ = "report_snapshots"
    __table_args__ = (
        UniqueConstraint(
            "symbol", "group", "timeframe", "as_of", name="uq_snapshot_symbol_date"
        ),
    )

    id = Column(Integer, primary_key=True)
    symbol = Column(String, nullable=False, index=True)
    group = Column(String, default="", index=True)
    timeframe = Column(String, default="D1")
    as_of = Column(String, nullable=False)  # report last_date
    payload = Column(JSON, nullable=False)
    created_at = Column(DateTime, default=datetime.utcnow)


class SignalEvent(Base):
    """One live-signal alert (dedup key + payload), for signal history."""

    __tablename__ = "signal_events"
    __table_args__ = (UniqueConstraint("key", name="uq_signal_key"),)

    id = Column(Integer, primary_key=True)
    symbol = Column(String, nullable=False, index=True)
    key = Column(String, nullable=False)
    payload = Column(JSON, nullable=False)
    created_at = Column(DateTime, default=datetime.utcnow)


class Database:
    """Engine + session factory with the Postgres→SQLite graceful fallback.

    Initialised lazily on first use (``session()`` / ``backend``), so merely
    importing ``api.db`` never touches disk or connects anywhere - important
    for CLI/test processes that set ``NEXUS_DATABASE_URL`` afterwards.
    """

    def __init__(self) -> None:
        self._engine = None
        self._session_factory = None
        self._backend: Optional[str] = None

    def _ensure(self) -> None:
        if self._engine is not None:
            return
        url = os.environ.get("NEXUS_DATABASE_URL", "").strip()
        if url:
            try:
                eng = create_engine(url, pool_pre_ping=True)
                with eng.connect() as conn:
                    conn.execute(text("SELECT 1"))
                self._engine = eng
                if "postgres" in url:
                    self._backend = "postgres"
                elif "sqlite" in url:
                    self._backend = "sqlite"
                else:
                    self._backend = "configured"
            except Exception:
                self._engine = None

        if self._engine is None:
            # SQLite fallback: file under data/db/ (gitignored).
            Path(DEFAULT_SQLITE_PATH).parent.mkdir(parents=True, exist_ok=True)
            self._engine = create_engine(SQLITE_URL)
            self._backend = "sqlite"

        self._session_factory = sessionmaker(bind=self._engine, expire_on_commit=False)
        Base.metadata.create_all(self._engine)

    @property
    def engine(self):
        self._ensure()
        return self._engine

    @property
    def backend(self) -> str:
        self._ensure()
        return self._backend or "none"

    def session(self):
        """A SQLAlchemy ``Session`` (itself a context manager)::

        with db.session() as s:
            s.add(...)
            s.commit()
        """
        self._ensure()
        return self._session_factory()


db = Database()
