"""
NexusQuant - Equity fundamentals data provider (institutional spec #8).

Fetches real fundamentals via yfinance and writes them into the local
``data/fundamentals/universe.csv`` that ``src.equity.fundamentals`` scores
against. This closes the "Value/Quality read None without a local CSV" gap:
one refresh pass and every equity report has P/E, EV/EBITDA, P/B, ROE and
Debt/Equity.

Columns written (all optional, matching the scorer):
    symbol, pe, ev_ebitda, pb, roe_pct, debt_to_equity,
    earnings_surprise_pct, analyst_revisions

Usage:
    python -m src.equity.data_provider --symbols AAPL,MSFT,SPY
    python -m src.equity.data_provider --universe sp500      # auto list
    python -m src.equity.data_provider --symbols AAPL --json

Fully graceful: a symbol with no yfinance read (or offline mode) is
skipped with a warning, never a crash.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional

import pandas as pd

sys.path.append(str(Path(__file__).parent.parent.parent))

FUNDAMENTALS_DIR = "data/fundamentals"
_UNIVERSE_CSV = Path(FUNDAMENTALS_DIR) / "universe.csv"


def fetch_symbol_fundamentals(symbol: str) -> Optional[Dict]:
    """
    Pull one symbol's fundamentals through yfinance ``Ticker.info``.

    Returns the scorer-friendly dict (keys as above, ``source: "yahoo"``)
    or None when yfinance is unavailable / the ticker has no data.
    """
    try:
        import yfinance as yf
    except Exception:
        return None
    try:
        info = yf.Ticker(symbol).info or {}
    except Exception:
        return None
    if not info or "quoteType" not in info:
        return None

    def num(key: str) -> Optional[float]:
        v = info.get(key)
        try:
            return float(v)
        except (TypeError, ValueError):
            return None

    # yfinance returns ROE as a fraction (0.25 = 25%); convert to percent.
    roe = num("returnOnEquity")
    roe_pct = round(roe * 100.0, 1) if roe is not None else None
    # analyst_revisions is unavailable directly; recommendationMean (1=strong
    # buy .. 5=sell) is directionally close - invert so + = positive revisions.
    rec = num("recommendationMean")
    rev = round(4.0 - rec, 2) if rec is not None else None

    out = {
        "symbol": symbol,
        "pe": num("trailingPE"),
        "ev_ebitda": num("enterpriseToEbitda"),
        "pb": num("priceToBook"),
        "roe_pct": roe_pct,
        "debt_to_equity": num("debtToEquity"),
        "earnings_surprise_pct": num("earningsQuarterlyGrowth"),
        "analyst_revisions": rev,
        "source": "yahoo",
    }
    # Require at least one real fundamental to consider the fetch a success.
    if all(v is None for k, v in out.items() if k not in ("symbol", "source")):
        return None
    return out


def refresh_fundamentals(
    symbols: List[str],
    data_dir: str = FUNDAMENTALS_DIR,
    delay: float = 0.2,
) -> Dict:
    """
    Fetch + write ``universe.csv`` for ``symbols`` (upserts by symbol).

    Returns ``{fetched, skipped, failed, path}``. Existing rows for
    symbols NOT in the list are preserved, so repeated passes accumulate
    coverage without re-fetching.
    """
    out_dir = Path(data_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    existing = {}
    if _UNIVERSE_CSV.exists():
        try:
            df = pd.read_csv(_UNIVERSE_CSV, dtype=str)
            for _, row in df.iterrows():
                existing[row["symbol"].upper()] = row.to_dict()
        except Exception:
            existing = {}

    fetched, failed = 0, []
    for sym in symbols:
        rec = fetch_symbol_fundamentals(sym)
        if rec is None:
            failed.append(sym)
            continue
        row = {k: ("" if v is None else v) for k, v in rec.items() if k != "source"}
        existing[sym.upper()] = row
        fetched += 1
        if delay:
            time.sleep(delay)  # be polite to Yahoo rate limits

    out = pd.DataFrame(list(existing.values()))
    cols = [
        "symbol",
        "pe",
        "ev_ebitda",
        "pb",
        "roe_pct",
        "debt_to_equity",
        "earnings_surprise_pct",
        "analyst_revisions",
    ]
    out = out[[c for c in cols if c in out.columns]]
    out.to_csv(_UNIVERSE_CSV, index=False)

    return {
        "fetched": fetched,
        "skipped": len(symbols) - fetched - len(failed),
        "failed": failed,
        "path": str(_UNIVERSE_CSV),
        "total_rows": len(out),
    }


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        description="NexusQuant equity fundamentals provider (yfinance)"
    )
    parser.add_argument("--symbols", default=None, help="comma-separated symbols")
    parser.add_argument(
        "--universe",
        choices=["sp500"],
        default=None,
        help="auto-fetch the S&P 500 list (first 200)",
    )
    parser.add_argument(
        "--limit", type=int, default=200, help="cap on auto-universe size"
    )
    parser.add_argument("--json", action="store_true")
    parser.add_argument(
        "--delay", type=float, default=0.2, help="seconds between fetches (rate limit)"
    )
    args = parser.parse_args(argv)

    symbols: List[str] = []
    if args.symbols:
        symbols = [s.strip().upper() for s in args.symbols.split(",") if s.strip()]
    if args.universe == "sp500":
        from src.equity.universe import sp500_tickers

        symbols = [s for s in sp500_tickers() if s][: args.limit]

    if not symbols:
        print(
            "No symbols to fetch (pass --symbols or --universe sp500).", file=sys.stderr
        )
        return 1

    res = refresh_fundamentals(symbols, delay=args.delay)
    if args.json:
        print(json.dumps(res, indent=2, default=str))
    else:
        print(
            f"Fetched {res['fetched']} symbols → {res['path']} "
            f"({res['total_rows']} rows)"
        )
        if res["failed"]:
            print(
                f"  skipped/failed ({len(res['failed'])}): "
                f"{', '.join(res['failed'][:10])}"
            )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
