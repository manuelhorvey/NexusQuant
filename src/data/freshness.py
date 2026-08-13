"""
NexusQuant - Data Freshness Checks.

Answers the question *"is the data I'm making decisions on up to date?"*
for the local parquet files under ``data/raw``.

* ``staleness`` - how many calendar days since the last bar for a symbol.
* ``is_stale``  - a single bool using per-timeframe thresholds that absorb
  weekends/holidays (a Friday D1 bar is still fresh on Monday).
* ``freshness_report`` - a DataFrame over a whole group, ready for the CLI
  and the live-signal gate.

Pure and offline (reads only parquet) - the update itself lives in
``src/data/update.py`` (needs the MT5 bridge).
"""

from __future__ import annotations

import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

import pandas as pd

sys.path.append(str(Path(__file__).parent.parent.parent))

# Calendar-day thresholds per timeframe: absorb weekends (Fri->Mon is 3d)
# plus one extra day of slack. A D1 file whose last bar is older than this
# is considered stale.
DEFAULT_MAX_STALE_DAYS: Dict[str, int] = {
    "D1": 4,
    "H4": 2,
    "H1": 2,
    "M1": 1,
    "M5": 1,
    "M15": 1,
    "M30": 1,
}


def last_bar_date(path: Path) -> Optional[pd.Timestamp]:
    """Last bar timestamp of a parquet file (None if unreadable/empty)."""
    try:
        df = pd.read_parquet(path, columns=["date"])
        if df.empty:
            return None
        return pd.to_datetime(df["date"]).max()
    except Exception:
        return None


def staleness_days(path: Path, today: Optional[datetime] = None) -> Optional[float]:
    """
    Calendar days between today and the last bar. None when the file is
    missing or unreadable.
    """
    last = last_bar_date(path)
    if last is None:
        return None
    today = pd.Timestamp(today or datetime.now()).normalize()
    return float((today - last.normalize()).days)


def is_stale(
    path: Path,
    timeframe: str = "D1",
    max_stale_days: Optional[Dict[str, int]] = None,
    today: Optional[datetime] = None,
) -> bool:
    """True when the file is missing or older than the timeframe threshold."""
    thresholds = {**(DEFAULT_MAX_STALE_DAYS or {}), **(max_stale_days or {})}
    age = staleness_days(path, today)
    if age is None:
        return True  # missing/unreadable = stale
    return age > thresholds.get(timeframe.upper(), 4)


def freshness_report(
    data_dir: str = "data/raw",
    group: Optional[str] = None,
    timeframe: str = "D1",
    symbols: Optional[List[str]] = None,
    max_stale_days: Optional[Dict[str, int]] = None,
    today: Optional[datetime] = None,
) -> pd.DataFrame:
    """
    Report over every ``{SYMBOL}_{TF}.parquet`` in a group:

        symbol, last_date, age_days, bars, status (FRESH / STALE / MISSING)
    """
    from src.analysis.scanner import _data_path

    if symbols is None:
        from src.analysis.scanner import discover_symbols

        symbols = discover_symbols(data_dir, group, timeframe)

    rows = []
    for sym in symbols:
        path = _data_path(sym, data_dir, group, timeframe)
        # Single read per symbol: derive last date, bar count and age
        # together (the report may run over a 350-symbol universe).
        n_bars = 0
        last = None
        if path.exists():
            try:
                dates = pd.to_datetime(pd.read_parquet(path, columns=["date"])["date"])
                n_bars = len(dates)
                last = dates.max()
            except Exception:
                last = None
        if last is not None:
            today_ts = pd.Timestamp(today or datetime.now()).normalize()
            age = float((today_ts - last.normalize()).days)
        else:
            age = None
        status = (
            "MISSING"
            if age is None
            else (
                "STALE" if is_stale(path, timeframe, max_stale_days, today) else "FRESH"
            )
        )
        rows.append(
            {
                "symbol": sym,
                "last_date": last.date() if last is not None else None,
                "age_days": age,
                "bars": n_bars,
                "status": status,
            }
        )
    return pd.DataFrame(rows)


def summary(report: pd.DataFrame) -> Dict:
    """Counts by status + the max age, for the CLI one-liner."""
    counts = report["status"].value_counts().to_dict()
    ages = report["age_days"].dropna()
    return {
        "total": len(report),
        "fresh": int(counts.get("FRESH", 0)),
        "stale": int(counts.get("STALE", 0)),
        "missing": int(counts.get("MISSING", 0)),
        "max_age_days": float(ages.max()) if len(ages) else None,
    }
