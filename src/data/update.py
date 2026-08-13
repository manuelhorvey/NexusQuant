"""
NexusQuant - Incremental Data Update.

Keeps the local parquet data fresh so decisions are made on current bars.
Two modes:

    python -m src.data.update --group full_fx --check
        # freshness report only - no MT5 needed, safe for cron

    python -m src.data.update --group full_fx --timeframe D1
        # connect to MT5, append the bars missing since each file's last bar

The update is **incremental**: for every file we read its last bar date and
call ``copy_rates_range(last_date - overlap, now)``, merge with the existing
frame (dedupe on date, keep the newest) and write back. A full universe
refresh is a few MB and takes seconds - not a re-download.

Usage:
    python -m src.data.update --check                          # whole tree
    python -m src.data.update --group full_fx --check
    python -m src.data.update --group full_fx                  # update D1
    python -m src.data.update --group full_fx --timeframes D1,H4
    python -m src.data.update --symbols EURUSD,GBPUSD,XAUUSD --group full_fx
    python -m src.data.update --group full_fx --dry-run        # preview
    python -m src.data.update --group full_fx --json
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import logging
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import pandas as pd

sys.path.append(str(Path(__file__).parent.parent.parent))

from src.analysis.scanner import _data_path, discover_symbols
from src.data.freshness import (
    freshness_report,
    last_bar_date,
    summary,
)
from src.data.mt5 import MT5Error, MT5Provider

log = logging.getLogger("nexus.data.update")

# Re-fetch this many calendar days before the last bar so the merge always
# has overlap (partial bars, timezone drift) and dedupe keeps the newest.
OVERLAP_DAYS = 5


# ---------------------------------------------------------------------------
# Core update
# ---------------------------------------------------------------------------


def fetch_missing(
    provider: MT5Provider,
    symbol: str,
    timeframe: str,
    group: Optional[str],
    data_dir: str = "data/raw",
    overlap_days: int = OVERLAP_DAYS,
) -> Tuple[str, int, int, Optional[str]]:
    """
    Append the missing bars for one symbol@timeframe.

    Returns ``(status, added, total, error)`` where status is
    ``updated`` / ``current`` / ``created`` / ``error``.
    """
    path = _data_path(symbol, data_dir, group, timeframe)
    last = last_bar_date(path) if path.exists() else None

    # The whole per-symbol body is isolated so a corrupt file, dtype
    # coercion error, disk failure or mid-call rpyc/network error (not an
    # MT5Error) never stops the rest of the update run.
    try:
        if last is None:
            # No file yet - do a full download (reuse the MT5 bridge).
            from src.data.mt5 import ensure_parquet

            p = ensure_parquet(symbol, timeframe, data_dir, group, provider=provider)
            n = len(pd.read_parquet(p, columns=["date"]))
            return "created", n, n, None

        date_from = last - pd.Timedelta(days=overlap_days)
        new = provider.copy_rates_range(
            symbol,
            timeframe,
            date_from=date_from.to_pydatetime(),
            date_to=dt.datetime.now(),
        )
        if new is None or new.empty:
            return "current", 0, 0, None

        old = pd.read_parquet(path)
        merged = (
            pd.concat([old, new], ignore_index=True)
            .drop_duplicates(subset=["date"], keep="last")
            .sort_values("date")
            .reset_index(drop=True)
        )
        added = len(merged) - len(old)
        if added > 0:
            merged.to_parquet(path, compression="zstd")
            return "updated", added, len(merged), None
        return "current", 0, len(merged), None
    except Exception as exc:
        log.warning("update failed for %s@%s: %s", symbol, timeframe, exc)
        return "error", 0, 0, str(exc)


def update_group(
    provider: MT5Provider,
    group: Optional[str],
    timeframe: str,
    data_dir: str = "data/raw",
    symbols: Optional[List[str]] = None,
    dry_run: bool = False,
    overlap_days: int = OVERLAP_DAYS,
) -> Dict:
    """
    Update every symbol in a group; returns a result dict for the CLI.
    """
    if symbols is None:
        symbols = discover_symbols(data_dir, group, timeframe)
    if not symbols:
        return {
            "group": group,
            "timeframe": timeframe,
            "symbols": [],
            "added": 0,
            "updated": 0,
            "current": 0,
            "created": 0,
            "errors": [],
            "dry_run": dry_run,
        }

    results = []
    for sym in symbols:
        if dry_run:
            path = _data_path(sym, data_dir, group, timeframe)
            last = last_bar_date(path)
            results.append(
                {
                    "symbol": sym,
                    "status": "would_update"
                    if (last is None or last.date() < dt.date.today())
                    else "current",
                    "last_date": str(last.date()) if last is not None else None,
                }
            )
            continue
        status, added, total, err = fetch_missing(
            provider, sym, timeframe, group, data_dir, overlap_days
        )
        results.append(
            {
                "symbol": sym,
                "status": status,
                "added": added,
                "total": total,
                "error": err,
            }
        )

    def _count(status):
        return sum(1 for r in results if r["status"] == status)

    return {
        "group": group,
        "timeframe": timeframe,
        "symbols": results,
        "added": sum(r.get("added", 0) for r in results),
        "updated": _count("updated"),
        "created": _count("created"),
        "current": _count("current"),
        "errors": [r["error"] for r in results if r.get("error")],
        "dry_run": dry_run,
    }


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        description="Incremental data update for NexusQuant (MT5 bridge)",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--group", default=None, help="data group (full_fx, metals, ...); default: all"
    )
    parser.add_argument(
        "--timeframes", default="D1", help="comma-separated timeframes to update"
    )
    parser.add_argument(
        "--symbols",
        default=None,
        help="comma-separated symbol list (default: all in group)",
    )
    parser.add_argument("--data-dir", default="data/raw")
    parser.add_argument(
        "--check", action="store_true", help="freshness report only (no MT5 connection)"
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="show what would be fetched without fetching",
    )
    parser.add_argument("--json", action="store_true")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8001)
    parser.add_argument(
        "--max-stale",
        type=int,
        default=None,
        help="override the stale threshold (calendar days)",
    )
    args = parser.parse_args(argv)

    symbols = None
    if args.symbols:
        symbols = [s.strip().upper() for s in args.symbols.split(",") if s.strip()]
    timeframes = [tf.strip().upper() for tf in args.timeframes.split(",") if tf.strip()]
    max_stale = {tf: args.max_stale for tf in timeframes} if args.max_stale else None

    # ---- check mode (offline) -------------------------------------------
    if args.check:
        out = {"checked_at": str(dt.datetime.now()), "groups": []}
        groups = (
            [args.group]
            if args.group
            else [
                p.name
                for p in sorted(Path(args.data_dir).iterdir())
                if p.is_dir() and any(p.glob(f"*_{tf}.parquet") for tf in timeframes)
            ]
        )
        for grp in groups:
            for tf in timeframes:
                rep = freshness_report(
                    args.data_dir, grp, tf, symbols, max_stale_days=max_stale
                )
                out["groups"].append(
                    {
                        "group": grp,
                        "timeframe": tf,
                        "report": rep.to_dict(orient="records"),
                        **summary(rep),
                    }
                )
        if args.json:
            print(json.dumps(out, indent=2, default=str))
            return 0
        print("\n" + "=" * 70)
        print("DATA FRESHNESS REPORT")
        print("=" * 70)
        for g in out["groups"]:
            print(
                f"\n{g['group']} · {g['timeframe']} — "
                f"{g['fresh']} fresh · {g['stale']} stale · "
                f"{g['missing']} missing · max age {g['max_age_days']}d"
            )
            if g["report"]:
                rep = pd.DataFrame(g["report"])
                show = rep[["symbol", "last_date", "age_days", "bars", "status"]]
                print(show.to_string(index=False))
        print("=" * 70)
        return 0

    # ---- update mode (needs MT5) ----------------------------------------
    try:
        with MT5Provider(host=args.host, port=args.port) as prov:
            print(f"MT5 terminal: {prov.version()}")
            results = []
            for tf in timeframes:
                r = update_group(
                    prov, args.group, tf, args.data_dir, symbols, dry_run=args.dry_run
                )
                results.append(r)
    except MT5Error as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        print(
            "Hint: run 'python -m src.data.update --check' for an "
            "offline freshness report.",
            file=sys.stderr,
        )
        return 1

    if args.json:
        print(json.dumps(results, indent=2, default=str))
        return 0

    total_added = sum(r["added"] for r in results)
    for r in results:
        verb = "would update" if args.dry_run else "updated"
        print(
            f"\n[{r['group'] or 'all'} · {r['timeframe']}] "
            f"{len(r['symbols'])} symbols · "
            f"{r['updated']} {verb} · {r['created']} created · "
            f"{r['current']} current · {len(r['errors'])} errors"
        )
        if r["errors"]:
            print(f"  errors: {r['errors'][:5]}")
    print(f"\nTotal new bars: {total_added}" + (" (dry run)" if args.dry_run else ""))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
