"""
NexusQuant - FastAPI service.

Serves the whole research stack as a REST API. Missing symbols are
resolved on demand (local -> MT5 terminal -> Yahoo) so any ticker works,
not just the ones already backfilled:

    GET  /health                       backend + data status
    GET  /api/v1/groups                discoverable datasets
    GET  /api/v1/scan                  ranked universe table
    GET  /api/v1/symbol/{s}/report     full institutional report
    GET  /api/v1/symbol/{s}/risk       risk & position sizing plan
    GET  /api/v1/symbol/{s}/stress     stress test (2008/COVID/2022)
    GET  /api/v1/symbol/{s}/backtest   causal dip backtest stats
    GET  /api/v1/live/pass             one live signal pass
    POST /api/v1/snapshots             persist a report snapshot
    GET  /api/v1/snapshots             list persisted snapshots
    GET  /api/v1/signals               signal-event history

Run locally (SQLite fallback persistence):

    uvicorn api.app:app --reload --port 8000

Run the full stack (Postgres via docker-compose):

    docker compose up --build
"""

from __future__ import annotations

import math
import sys
from datetime import date, datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware

sys.path.insert(0, str(Path(__file__).parent.parent))

from api.db import ReportSnapshot, SignalEvent, db

from src.analysis.dashboard_data import discover_groups, load_symbol_report
from src.analysis.scanner import scan_universe
from src.risk.run import risk_plan_from_report
from src.risk.stress import stress_table_from_report
from src.backtest.engine import BacktestParams
from src.backtest.signals import dip_signal_series
from src.live.signals import live_signal_pass

app = FastAPI(
    title="NexusQuant API",
    description="Institutional multi-asset quant research & signal service",
    version="0.1.0",
)

# The dashboard runs on a different origin in dev; keep the API open.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

DATA_DIR = "data/raw"


# ---------------------------------------------------------------------------
# JSON-safe helpers (numpy scalars, NaN/Inf -> null, DataFrame -> records)
# ---------------------------------------------------------------------------


def _clean(value: Any) -> Any:
    """Recursively convert numpy/pandas scalars and NaN/Inf to JSON-safe."""
    if value is None:
        return None
    if isinstance(value, bool):
        return bool(value)
    if isinstance(value, (int, np.integer)):
        return int(value)
    if isinstance(value, (float, np.floating)):
        v = float(value)
        return None if math.isnan(v) or math.isinf(v) else v
    if isinstance(value, np.ndarray):
        return [_clean(x) for x in value.tolist()]
    if isinstance(value, (pd.Timestamp, datetime, date)):
        return value.isoformat()
    if isinstance(value, dict):
        return {str(k): _clean(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_clean(v) for v in value]
    return value


def _records(table: pd.DataFrame) -> List[Dict]:
    return [_clean(r) for r in table.to_dict(orient="records")]


def _group_or_default(group: Optional[str]) -> Optional[str]:
    return group or None


# ---------------------------------------------------------------------------
# TTL-aware report cache
#
# Keyed on (symbol, group, timeframe, parquet mtime) so a data refresh via
# ``python -m src.data.update`` is picked up on the *next* request instead of
# serving stale reports until the server restarts. Entries are evicted when
# the cache grows past ``CACHE_MAX`` (simple dict + FIFO by insertion).
# ---------------------------------------------------------------------------

CACHE_MAX = 128
_report_cache: Dict[Tuple[str, str, str, float], Any] = {}
_cache_order: List[Tuple[str, str, str, float]] = []


def _parquet_mtime(symbol: str, group: str, timeframe: str) -> float:
    from src.data.resolver import find_local

    p = find_local(symbol, timeframe, DATA_DIR, _group_or_default(group) or None)
    if p is None:
        from src.analysis.scanner import _data_path

        p = _data_path(symbol, DATA_DIR, _group_or_default(group) or None, timeframe)
    try:
        return p.stat().st_mtime
    except OSError:
        return 0.0


def _cached_report(symbol: str, group: str, timeframe: str):
    mtime = _parquet_mtime(symbol, group, timeframe)
    key = (symbol, group, timeframe, mtime)
    hit = _report_cache.get(key)
    if hit is not None:
        return hit
    df, report = load_symbol_report(
        symbol,
        _group_or_default(group) or None,
        timeframe,
        data_dir=DATA_DIR,
        allow_fetch=True,
    )
    _report_cache[key] = (df, report)
    _cache_order.append(key)
    if len(_report_cache) > CACHE_MAX:
        # Evict oldest keys (files updated are naturally invalidated by mtime
        # already; this only bounds memory for long-running servers).
        while len(_report_cache) > CACHE_MAX and _cache_order:
            old = _cache_order.pop(0)
            _report_cache.pop(old, None)
    return df, report


def _load_symbol(symbol: str, group: Optional[str], timeframe: str):
    try:
        return _cached_report(symbol, group or "", timeframe)
    except FileNotFoundError:
        raise HTTPException(
            404, f"No data for {symbol} ({group or 'any'} · {timeframe})."
        ) from None
    except ValueError as exc:  # no valid rows / too short history
        raise HTTPException(422, f"{symbol}: {exc}") from None


# ---------------------------------------------------------------------------
# Health & discovery
# ---------------------------------------------------------------------------


@app.get("/health")
def health():
    return {
        "status": "ok",
        "service": "NexusQuant API",
        "db_backend": db.backend,
        "data_dir": DATA_DIR,
    }


@app.get("/api/v1/groups")
def groups():
    return _clean(discover_groups(DATA_DIR))


# ---------------------------------------------------------------------------
# Universe scan
# ---------------------------------------------------------------------------


@app.get("/api/v1/scan")
def scan(
    group: Optional[str] = Query(None),
    timeframe: str = Query("D1"),
    symbols: Optional[str] = Query(None, description="comma-separated"),
    top: Optional[int] = Query(None),
):
    sym_list = [s.strip() for s in symbols.split(",") if s.strip()] if symbols else None
    try:
        table = scan_universe(
            data_dir=DATA_DIR,
            group=_group_or_default(group),
            timeframe=timeframe,
            symbols=sym_list,
            fetch_mt5=True,
        )  # on-demand MT5/Yahoo fallback
    except Exception as exc:
        raise HTTPException(422, str(exc)) from None
    if top:
        table = table.head(top)
    return {
        "group": group or "default",
        "timeframe": timeframe,
        "n": len(table),
        "rows": _records(table),
    }


# ---------------------------------------------------------------------------
# Symbol analysis
# ---------------------------------------------------------------------------


@app.get("/api/v1/symbol/{symbol}")
@app.get("/api/v1/symbol/{symbol}/report")
def symbol_report(
    symbol: str,
    group: Optional[str] = None,
    timeframe: str = "D1",
    persist: bool = Query(False),
):
    df, report = _load_symbol(symbol, group, timeframe)
    if persist:
        _save_snapshot(symbol, group or "", timeframe, report)
    return _clean(report)


@app.get("/api/v1/symbol/{symbol}/risk")
def symbol_risk(
    symbol: str,
    group: Optional[str] = None,
    timeframe: str = "D1",
    equity: float = 100_000.0,
):
    df, report = _load_symbol(symbol, group, timeframe)
    plan = risk_plan_from_report(report, symbol, equity=equity)
    return _clean(plan)


@app.get("/api/v1/symbol/{symbol}/stress")
def symbol_stress(
    symbol: str,
    group: Optional[str] = None,
    timeframe: str = "D1",
    equity: float = 100_000.0,
):
    df, report = _load_symbol(symbol, group, timeframe)
    table = stress_table_from_report(report, symbol, equity=equity, df=df)
    return _clean(table)


@app.get("/api/v1/symbol/{symbol}/backtest")
def symbol_backtest(
    symbol: str,
    group: Optional[str] = None,
    timeframe: str = "D1",
    risk_pct: float = 0.01,
    rr_fallback: float = 2.0,
    max_hold: int = 20,
    entry_type: str = "limit",
):
    df, report = _load_symbol(symbol, group, timeframe)
    if len(df) < 60:
        raise HTTPException(422, f"{symbol}: only {len(df)} bars")
    signal = dip_signal_series(df)
    bars_per_year = {"D1": 252, "H4": 1512, "H1": 6048}.get(timeframe.upper(), 252)
    params = BacktestParams(
        risk_pct=risk_pct,
        rr_fallback=rr_fallback,
        max_hold=max_hold,
        entry_type=entry_type,
        bars_per_year=bars_per_year,
    )
    from dataclasses import asdict
    from src.backtest.engine import run_backtest

    result = run_backtest(signal, df, params, symbol=symbol)
    return _clean(
        {
            "symbol": symbol,
            "timeframe": timeframe,
            "params": asdict(params),
            "stats": result.stats,
            "n_trades": len(result.trades),
        }
    )


# ---------------------------------------------------------------------------
# Live signal pass
# ---------------------------------------------------------------------------


@app.get("/api/v1/live/pass")
def live_pass(
    group: Optional[str] = "full_fx",
    timeframe: str = "D1",
    symbols: Optional[str] = None,
    persist: bool = Query(
        False, description="also store any new setups as signal events in the database"
    ),
    dry_run: bool = Query(
        True,
        description="when true, do not write the alert dedup state "
        "file (safe read-only inspection)",
    ),
):
    """Run one live signal pass.

    NOTE: by default (``dry_run=true``) this is read-only - it scans and
    reports candidates without writing the dedup state file, so repeated
    calls behave like an idempotent GET. Set ``dry_run=false`` to actually
    consume the pass (mark setups as alerted), as the cron scheduler does.
    """
    sym_list = [s.strip() for s in symbols.split(",") if s.strip()] if symbols else None
    result = live_signal_pass(
        group=_group_or_default(group),
        timeframe=timeframe,
        symbols=sym_list,
        dry_run=dry_run,
    )
    out = {k: v for k, v in result.items() if k != "new_alerts"}
    out["new_alerts"] = [
        _clean(
            {
                "symbol": a["symbol"],
                "key": a["key"],
                "text": a["text"],
            }
        )
        for a in result.get("new_alerts", [])
    ]

    # Persist signal events for history (dedup on key).
    if persist:
        _store_signal_events(result.get("new_alerts", []))
    return _clean(out)


# ---------------------------------------------------------------------------
# Persistence endpoints
# ---------------------------------------------------------------------------


def _save_snapshot(symbol: str, group: str, timeframe: str, report: Dict) -> None:
    """Upsert one snapshot per (symbol, group, timeframe, as_of)."""
    from datetime import datetime

    as_of = str(report.get("last_date", ""))
    payload = _clean(report)
    with db.session() as s:
        row = (
            s.query(ReportSnapshot)
            .filter_by(symbol=symbol, group=group, timeframe=timeframe, as_of=as_of)
            .first()
        )
        if row:
            row.payload = payload
            row.created_at = datetime.utcnow()
        else:
            s.add(
                ReportSnapshot(
                    symbol=symbol,
                    group=group,
                    timeframe=timeframe,
                    as_of=as_of,
                    payload=payload,
                )
            )
        s.commit()


def _store_signal_events(alerts: List[Dict]) -> None:
    """Upsert signal events by their dedup key."""
    from datetime import datetime

    with db.session() as s:
        for a in alerts:
            row = s.query(SignalEvent).filter_by(key=a["key"]).first()
            if row:
                row.payload = {"text": a["text"]}
                row.created_at = datetime.utcnow()
            else:
                s.add(
                    SignalEvent(
                        symbol=a["symbol"], key=a["key"], payload={"text": a["text"]}
                    )
                )
        s.commit()


@app.get("/api/v1/snapshots")
def list_snapshots(
    symbol: str, group: Optional[str] = None, limit: int = Query(20, le=200)
):
    with db.session() as s:
        q = s.query(ReportSnapshot).filter(ReportSnapshot.symbol == symbol)
        if group:
            q = q.filter(ReportSnapshot.group == group)
        rows = q.order_by(ReportSnapshot.created_at.desc()).limit(limit).all()
        return _clean(
            [
                {
                    "symbol": r.symbol,
                    "group": r.group,
                    "timeframe": r.timeframe,
                    "as_of": r.as_of,
                    "created_at": r.created_at,
                    "payload": r.payload,
                }
                for r in rows
            ]
        )


@app.post("/api/v1/snapshots")
def store_snapshot(symbol: str, group: Optional[str] = None, timeframe: str = "D1"):
    """Generate + persist today's report snapshot for a symbol."""
    df, report = _load_symbol(symbol, group, timeframe)
    _save_snapshot(symbol, group or "", timeframe, report)
    return {
        "status": "stored",
        "symbol": symbol,
        "group": group or "",
        "timeframe": timeframe,
        "as_of": report.get("last_date"),
    }


@app.get("/api/v1/signals")
def list_signals(symbol: Optional[str] = None, limit: int = Query(50, le=500)):
    with db.session() as s:
        q = s.query(SignalEvent)
        if symbol:
            q = q.filter(SignalEvent.symbol == symbol)
        rows = q.order_by(SignalEvent.created_at.desc()).limit(limit).all()
        return _clean(
            [
                {
                    "symbol": r.symbol,
                    "key": r.key,
                    "created_at": r.created_at,
                    "payload": r.payload,
                }
                for r in rows
            ]
        )


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("api.app:app", host="0.0.0.0", port=8000, reload=True)
