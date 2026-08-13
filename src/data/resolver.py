"""
NexusQuant - On-demand data resolver.

The single entry point the whole stack uses when a symbol's bars are asked
for: ``resolve_symbol_data`` returns a local parquet path for any symbol /
timeframe, fetching it on demand if needed. Resolution cascade:

1. **Local, exact group** - ``{data_dir}/{group}/{SYMBOL}_{TF}.parquet``
   (the fast path; zero I/O beyond the stat).
2. **Local, any group** - search every group folder (flat and nested
   layouts) for the file. A symbol that lives in ``equity/`` resolves even
   when the caller asked for ``full_fx``.
3. **MT5** - pull from the running Wine terminal (covers its 357-symbol
   universe: FX, metals, indices, stocks, crypto, commodities), written
   into the **classified** asset-class folder (``classify_symbol``), never
   the ``data/raw`` root.
4. **Yahoo** - no-auth chart API (stocks / ETFs / anything MT5 does not
   list, or when the bridge is down), same classified-folder caching.

``allow_mt5`` / ``allow_yahoo`` gate the network sources so callers can
stay local-only (the dashboard, unit tests) or fully on-demand (scanner
CLI, live pass, API).

Every failure raises ``FileNotFoundError`` with a message listing what was
tried - callers either surface it (scanner keeps scanning the rest of the
universe) or degrade gracefully (report shows no data).
"""

from __future__ import annotations

from pathlib import Path
from typing import List, Optional

import pandas as pd


def find_local(
    symbol: str, timeframe: str, data_dir: str = "data/raw", group: Optional[str] = None
) -> Optional[Path]:
    """
    Local-only lookup: ``{group}/{SYMBOL}_{TF}.parquet`` anywhere under
    ``data_dir``. Returns the path with the longest history when several
    copies exist (same policy as the regroup tool), or None.
    """
    tf = timeframe.upper()
    base = Path(data_dir)
    candidates: List[Path] = []

    if group:
        p = base / group / f"{symbol}_{tf}.parquet"
        if p.exists():
            candidates.append(p)

    # Flat layout: each group folder holds {SYMBOL}_{TF}.parquet.
    for sub in base.iterdir():
        if not sub.is_dir():
            continue
        p = sub / f"{symbol}_{tf}.parquet"
        if p.exists():
            candidates.append(p)
        # Nested layout (e.g. data/raw/mt5/D1/{SYMBOL}_D1.parquet).
        for nested in sub.iterdir():
            if nested.is_dir():
                q = nested / f"{symbol}_{tf}.parquet"
                if q.exists():
                    candidates.append(q)

    if not candidates:
        return None
    # Prefer the longest history (drop files are garbage-free by design).
    return max(candidates, key=_row_count)


def _row_count(path: Path) -> int:
    try:
        return len(pd.read_parquet(path, columns=["date"]))
    except Exception:
        return 0


def effective_group(
    path: Path, data_dir: str, fallback: Optional[str] = None
) -> Optional[str]:
    """
    Group folder a resolved parquet path actually lives in.

    Handles both layouts: flat groups (``equity/AAPL_D1.parquet`` ->
    ``equity``) and nested (``data/raw/mt5/D1/SYM_D1.parquet`` -> ``mt5``,
    not the timeframe dir name). Falls back to ``fallback`` for files
    directly in the data root.
    """
    root = Path(data_dir)
    parent = path.parent
    if parent == root:
        return fallback
    # Nested layout: the immediate parent is a timeframe dir (D1/H4/H1...).
    if parent.parent != root and parent.name.upper() in {
        "M1",
        "M5",
        "M15",
        "M30",
        "H1",
        "H4",
        "D1",
        "W1",
        "MN1",
    }:
        return parent.parent.name
    return parent.name


def _classified_group(symbol: str, data_dir: str) -> str:
    """Asset-class group folder for a symbol (regroup's convention).

    Loads the curated S&P 500 membership (``equity_universe/_membership.csv``)
    so constituents land in ``equity_universe/`` like the regroup tool, not
    the generic ``equity/`` fallback.
    """
    from src.data.regroup import classify_symbol

    mem = set()
    p = Path(data_dir) / "equity_universe" / "_membership.csv"
    if p.exists():
        try:
            mem = set(pd.read_csv(p)["symbol"].str.upper())
        except Exception:
            mem = set()
    return classify_symbol(symbol, mem)


def resolve_symbol_data(
    symbol: str,
    timeframe: str,
    data_dir: str = "data/raw",
    group: Optional[str] = None,
    allow_mt5: bool = True,
    allow_yahoo: bool = True,
) -> Path:
    """
    Return a local parquet path for ``symbol``@``timeframe``, fetching on
    demand (MT5 -> Yahoo) when missing. Never writes to the ``data/raw``
    root: fetched files land in the classified asset-class folder.

    ``group`` is a hint for the fast path only - if the file lives in a
    different group folder it is still found and used.
    """
    tf = timeframe.upper()

    # 1) Local fast path: exact group.
    if group:
        exact = Path(data_dir) / group / f"{symbol}_{tf}.parquet"
        if exact.exists():
            return exact

    # 2) Local search across every group folder.
    local = find_local(symbol, tf, data_dir)
    if local is not None:
        return local

    dest_group = _classified_group(symbol, data_dir)

    # 3) MT5 terminal (Wine rpyc bridge).
    if allow_mt5:
        try:
            from src.data.mt5 import ensure_parquet

            return ensure_parquet(symbol, tf, data_dir, group=dest_group)
        except Exception as exc:
            mt5_err = str(exc)
    else:
        mt5_err = "disabled"

    # 4) Yahoo public chart API (no auth).
    if allow_yahoo:
        try:
            from src.data.yahoo import ensure_yahoo_parquet

            return ensure_yahoo_parquet(symbol, tf, data_dir, group=dest_group)
        except Exception as exc:
            yahoo_err = str(exc)
    else:
        yahoo_err = "disabled"

    raise FileNotFoundError(
        f"No data for {symbol}@{tf} anywhere under {data_dir} "
        f"(MT5: {mt5_err}; Yahoo: {yahoo_err})"
    )


# ---------------------------------------------------------------------------
# CLI (handy for scripting / debugging the cascade)
# ---------------------------------------------------------------------------


def main(argv: Optional[List[str]] = None) -> int:
    import argparse
    import sys

    parser = argparse.ArgumentParser(
        description="Resolve a symbol to a local parquet path (fetch on demand)"
    )
    parser.add_argument("symbol")
    parser.add_argument("--timeframe", default="D1")
    parser.add_argument("--group", default=None)
    parser.add_argument("--data-dir", default="data/raw")
    parser.add_argument("--no-mt5", action="store_true")
    parser.add_argument("--no-yahoo", action="store_true")
    args = parser.parse_args(argv)
    try:
        path = resolve_symbol_data(
            args.symbol.upper(),
            args.timeframe,
            args.data_dir,
            args.group,
            allow_mt5=not args.no_mt5,
            allow_yahoo=not args.no_yahoo,
        )
    except FileNotFoundError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1
    print(path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
