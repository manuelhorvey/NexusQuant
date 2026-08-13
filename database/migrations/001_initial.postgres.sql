-- NexusQuant — 001_initial.postgres.sql
-- PostgreSQL dialect of the initial schema (report snapshots + signal
-- events). The API creates this schema automatically via
-- Base.metadata.create_all; these files exist so a production Postgres can
-- be brought up from raw SQL.
--
-- SQLite reference: 001_initial.sql (same columns/constraints, AUTOINCREMENT
-- + DATETIME). Choose the file matching your backend.

BEGIN;

CREATE TABLE IF NOT EXISTS report_snapshots (
    id          BIGSERIAL PRIMARY KEY,
    symbol      VARCHAR   NOT NULL,
    "group"     VARCHAR   NOT NULL DEFAULT '',
    timeframe   VARCHAR   NOT NULL DEFAULT 'D1',
    as_of       VARCHAR   NOT NULL,
    payload     JSON      NOT NULL,
    created_at  TIMESTAMP
);

CREATE INDEX IF NOT EXISTS ix_report_snapshots_symbol
    ON report_snapshots (symbol);
CREATE INDEX IF NOT EXISTS ix_report_snapshots_group
    ON report_snapshots ("group");
CREATE UNIQUE INDEX IF NOT EXISTS uq_snapshot_symbol_date
    ON report_snapshots (symbol, "group", timeframe, as_of);

CREATE TABLE IF NOT EXISTS signal_events (
    id          BIGSERIAL PRIMARY KEY,
    symbol      VARCHAR   NOT NULL,
    key         VARCHAR   NOT NULL,
    payload     JSON      NOT NULL,
    created_at  TIMESTAMP
);

CREATE INDEX IF NOT EXISTS ix_signal_events_symbol ON signal_events (symbol);
CREATE UNIQUE INDEX IF NOT EXISTS uq_signal_key     ON signal_events (key);

COMMIT;
