-- NexusQuant — 001_initial.sql
-- Initial schema for the persistence layer (report snapshots + signal events).
-- Mirrors api/db.py (SQLAlchemy models).
--
-- **SQLite dialect** (AUTOINCREMENT + DATETIME). For PostgreSQL use the
-- matching 001_initial.postgres.sql (BIGSERIAL + TIMESTAMP). JSON columns
-- map to native JSON on Postgres and TEXT storage on SQLite.
--
-- The API creates this schema automatically via Base.metadata.create_all on
-- first use; these files exist so a production database can be brought up
-- from raw SQL and so the schema is reviewable outside the code.

BEGIN;

CREATE TABLE IF NOT EXISTS report_snapshots (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    symbol      VARCHAR   NOT NULL,
    "group"     VARCHAR   NOT NULL DEFAULT '',
    timeframe   VARCHAR   NOT NULL DEFAULT 'D1',
    as_of       VARCHAR   NOT NULL,
    payload     JSON      NOT NULL,
    created_at  DATETIME
);

CREATE INDEX IF NOT EXISTS ix_report_snapshots_symbol  ON report_snapshots (symbol);
CREATE INDEX IF NOT EXISTS ix_report_snapshots_group   ON report_snapshots ("group");
CREATE UNIQUE INDEX IF NOT EXISTS uq_snapshot_symbol_date
    ON report_snapshots (symbol, "group", timeframe, as_of);

CREATE TABLE IF NOT EXISTS signal_events (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    symbol      VARCHAR   NOT NULL,
    key         VARCHAR   NOT NULL,
    payload     JSON      NOT NULL,
    created_at  DATETIME
);

CREATE INDEX IF NOT EXISTS ix_signal_events_symbol ON signal_events (symbol);
CREATE UNIQUE INDEX IF NOT EXISTS uq_signal_key     ON signal_events (key);

COMMIT;
