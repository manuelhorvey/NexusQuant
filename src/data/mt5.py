"""
NexusQuant - MetaTrader 5 data provider (via the Wine rpyc bridge)

The MetaTrader 5 terminal runs under Wine on this machine
(WINEPREFIX=~/.wine_mt5). A Wine Python 3.12 process runs an rpyc classic
server (default port 8001) that exposes the official MetaTrader5 module.

This module talks to that bridge directly over rpyc — no mt5linux dependency
(the PyPI `mt5linux` package changed API and is Docker-oriented; we only need
the raw rpyc link, which is all the legacy mt5linux client did anyway).

Usage (download to data/raw):
    python -m src.data.mt5 --symbol EURUSD --timeframe D1 --bars 4000
    python -m src.data.mt5 --symbol EURUSD --timeframe H1 --group h1 --bars 20000
    python -m src.data.mt5 --symbols EURUSD,GBPUSD --timeframe D1 --group full_fx
    python -m src.data.mt5 --list-symbols
    python -m src.data.mt5 --backfill                     # full universe D1/H4/H1
    python -m src.data.mt5 --backfill --timeframes D1    # just D1
    python -m src.data.mt5 --backfill --group-filter '*XAU*'
"""

from __future__ import annotations

import argparse
import datetime as dt
import sys
from pathlib import Path
from typing import List, Optional, Sequence, Tuple

import pandas as pd
from tqdm import tqdm

DEFAULT_HOST = "127.0.0.1"
DEFAULT_PORT = 8001

# MT5 timeframe ids (match the official MetaTrader5 module constants).
TIMEFRAME_IDS = {
    "M1": 1,
    "M2": 2,
    "M3": 3,
    "M4": 4,
    "M5": 5,
    "M6": 6,
    "M10": 10,
    "M12": 12,
    "M15": 15,
    "M20": 20,
    "M30": 30,
    "H1": 16385,
    "H2": 16386,
    "H3": 16387,
    "H4": 16388,
    "H6": 16390,
    "H8": 16392,
    "H12": 16396,
    "D1": 16408,
    "W1": 32769,
    "MN1": 49153,
}

# Our parquet schema (same as the existing data/raw files).
SCHEMA = ["date", "symbol", "open", "high", "low", "close", "volume", "spread_points"]

# Sensible history windows per timeframe (D1 ~15y, H4 ~2y, H1 ~1y).
DEFAULT_BARS = {
    "M1": 20000,
    "M5": 20000,
    "M15": 20000,
    "M30": 20000,
    "H1": 20000,
    "H4": 10000,
    "D1": 4000,
    "W1": 1500,
    "MN1": 500,
}


class MT5Error(RuntimeError):
    """Raised when the MT5 bridge or terminal cannot be used."""


def timeframe_id(timeframe: str) -> int:
    """Map a timeframe string ('D1', 'H1', ...) to the MT5 constant id."""
    tf = timeframe.strip().upper()
    if tf not in TIMEFRAME_IDS:
        raise ValueError(
            f"Unsupported timeframe {timeframe!r}; use one of {sorted(TIMEFRAME_IDS)}"
        )
    return TIMEFRAME_IDS[tf]


def rates_to_frame(rates, symbol: str) -> pd.DataFrame:
    """
    Normalise raw MT5 copy_rates output to the NexusQuant parquet schema:
        date, symbol, open, high, low, close, volume, spread_points
    """
    if rates is None or len(rates) == 0:
        return pd.DataFrame(columns=SCHEMA)

    df = pd.DataFrame(rates)
    rename = {
        "time": "date",
        "tick_volume": "volume",
        "spread": "spread_points",
    }
    df = df.rename(columns=rename)

    for col in ["open", "high", "low", "close"]:
        if col not in df.columns:
            raise MT5Error(f"MT5 rates missing required column: {col}")

    df["date"] = pd.to_datetime(df["date"], unit="s")
    df["symbol"] = symbol
    if "volume" not in df.columns:
        df["volume"] = 0
    if "spread_points" not in df.columns:
        df["spread_points"] = 0

    df = df[SCHEMA].sort_values("date").reset_index(drop=True)
    return df


class MT5Provider:
    """Thin rpyc client for the MetaTrader 5 bridge running under Wine."""

    def __init__(self, host: str = DEFAULT_HOST, port: int = DEFAULT_PORT):
        try:
            import rpyc
        except ImportError as exc:
            raise MT5Error("rpyc is not installed — run: pip install rpyc") from exc

        try:
            self._conn = rpyc.classic.connect(host, port)
        except Exception as exc:
            raise MT5Error(
                f"Cannot reach the MT5 bridge at {host}:{port}. Is the Wine "
                f"rpyc server running? ({exc})"
            ) from exc

        self._conn._config["sync_request_timeout"] = 300  # 5 min for big pulls
        self._conn.execute("import MetaTrader5 as mt5")
        self._conn.execute("import datetime")

        if not self._conn.eval("mt5.initialize()"):
            err = self._conn.eval("mt5.last_error()")
            self.close()
            raise MT5Error(f"mt5.initialize() failed: {err}")

    # -- lifecycle ----------------------------------------------------------

    def __enter__(self) -> "MT5Provider":
        return self

    def __exit__(self, *exc) -> None:
        self.close()

    def close(self) -> None:
        try:
            self._conn.execute("mt5.shutdown()")
        except Exception:
            pass
        try:
            self._conn.close()
        except Exception:
            pass

    # -- metadata -----------------------------------------------------------

    def version(self):
        return self._conn.eval("mt5.version()")

    def terminal_info(self):
        return self._conn.eval("mt5.terminal_info()")

    def symbols_total(self) -> int:
        return int(self._conn.eval("mt5.symbols_total()"))

    def list_symbols(self, group: Optional[str] = None) -> List[str]:
        if group:
            raw = self._conn.eval(f"mt5.symbols_get(group={group!r})")
        else:
            raw = self._conn.eval("mt5.symbols_get()")
        if not raw:
            return []
        return sorted(s.name for s in raw)

    # -- rate fetching ------------------------------------------------------

    def copy_rates_from_pos(
        self,
        symbol: str,
        timeframe: str,
        start: int = 0,
        count: Optional[int] = None,
    ) -> pd.DataFrame:
        """Fetch the latest `count` bars for `symbol`@`timeframe`."""
        tf = timeframe_id(timeframe)
        if count is None:
            count = DEFAULT_BARS[timeframe.strip().upper()]
        # Activate the symbol in the terminal's Market Watch first; history
        # requests for unlisted symbols otherwise fail with "Call failed".
        self._conn.eval(f"mt5.symbol_select({symbol!r}, True)")
        code = f"mt5.copy_rates_from_pos({symbol!r}, {tf}, {int(start)}, {int(count)})"
        rates = self._conn.eval(code)
        if rates is None:
            err = self._conn.eval("mt5.last_error()")
            raise MT5Error(f"copy_rates_from_pos({symbol}, {timeframe}) failed: {err}")
        return rates_to_frame(rates, symbol)

    def copy_rates_range(
        self,
        symbol: str,
        timeframe: str,
        date_from: dt.datetime,
        date_to: Optional[dt.datetime] = None,
    ) -> pd.DataFrame:
        """Fetch bars in [date_from, date_to] (defaults to now)."""
        tf = timeframe_id(timeframe)
        if date_to is None:
            date_to = dt.datetime.now()
        code = (
            f"mt5.copy_rates_range({symbol!r}, {tf}, "
            f"datetime.datetime({date_from.year}, {date_from.month}, {date_from.day}), "
            f"datetime.datetime({date_to.year}, {date_to.month}, {date_to.day}))"
        )
        rates = self._conn.eval(code)
        if rates is None:
            err = self._conn.eval("mt5.last_error()")
            raise MT5Error(f"copy_rates_range({symbol}, {timeframe}) failed: {err}")
        return rates_to_frame(rates, symbol)


# ---------------------------------------------------------------------------
# Helpers for the rest of the codebase
# ---------------------------------------------------------------------------


def ensure_parquet(
    symbol: str,
    timeframe: str,
    data_dir: str = "data/raw",
    group: Optional[str] = None,
    bars: Optional[int] = None,
    provider: Optional[MT5Provider] = None,
) -> Path:
    """
    Return the parquet path for `symbol`@`timeframe`, downloading from MT5
    and caching to disk if it does not already exist.
    """
    tf = timeframe.strip().upper()
    base = Path(data_dir)
    if group:
        base = base / group
    path = base / f"{symbol}_{tf}.parquet"
    if path.exists():
        return path

    if bars is None:
        bars = DEFAULT_BARS.get(tf, 4000)

    own = provider is None
    prov = provider or MT5Provider()
    try:
        df = prov.copy_rates_from_pos(symbol, timeframe, 0, bars)
        if df.empty:
            raise MT5Error(f"MT5 returned no data for {symbol}@{tf}")
        path.parent.mkdir(parents=True, exist_ok=True)
        df.to_parquet(path, compression="zstd")
        print(f"[mt5] cached {symbol}@{tf} -> {path} ({len(df)} bars)")
    finally:
        if own:
            prov.close()
    return path


# ---------------------------------------------------------------------------
# Bulk backfill of the full MT5 universe
# ---------------------------------------------------------------------------


def backfill(
    provider: MT5Provider,
    timeframes: Sequence[str] = ("D1", "H4", "H1"),
    target_dir: str = "data/raw/mt5",
    skip_existing: bool = True,
    symbols: Optional[List[str]] = None,
    group_filter: Optional[str] = None,
    bars: Optional[int] = None,
) -> Tuple[int, int, List[Tuple[str, str, str]]]:
    """
    Download the full MT5 universe across several timeframes.

    Files are saved as ``{target_dir}/{TIMEFRAME}/{SYMBOL}_{TIMEFRAME}.parquet``
    so the bulk data lives apart from the curated groups (full_fx, h1, ...).

    Returns ``(downloaded, skipped, failures)`` where failures is a list of
    ``(symbol, timeframe, error)`` tuples.
    """
    if symbols is None:
        symbols = provider.list_symbols(group_filter)
        if not symbols:
            raise MT5Error("MT5 returned no symbols to backfill")

    target = Path(target_dir)
    total = len(symbols) * len(timeframes)
    downloaded = 0
    skipped = 0
    failures: List[Tuple[str, str, str]] = []

    pbar = tqdm(
        total=total, desc="backfill", unit="file", unit_scale=True, dynamic_ncols=True
    )
    try:
        for tf in timeframes:
            tf_name = tf.upper()
            for sym in symbols:
                path = target / tf_name / f"{sym}_{tf_name}.parquet"
                if skip_existing and path.exists():
                    skipped += 1
                    pbar.update(1)
                    continue
                try:
                    ensure_parquet(
                        sym,
                        tf,
                        data_dir=str(target),
                        group=tf_name,
                        bars=bars,
                        provider=provider,
                    )
                    downloaded += 1
                except Exception as exc:  # keep the backfill going
                    failures.append((sym, tf_name, str(exc)))
                    print(f"[backfill] FAIL {sym}@{tf_name}: {exc}", file=sys.stderr)
                pbar.update(1)
                pbar.set_postfix(ok=downloaded, skip=skipped, fail=len(failures))
    finally:
        pbar.close()

    if failures:
        print(f"\n[backfill] {len(failures)} failure(s) (first 10):", file=sys.stderr)
        for sym, tf, err in failures[:10]:
            print(f"  {sym}@{tf}: {err}", file=sys.stderr)
        if len(failures) > 10:
            print(f"  ... and {len(failures) - 10} more", file=sys.stderr)

    return downloaded, skipped, failures


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        description="Download market data from the running MetaTrader 5 "
        "terminal (Wine rpyc bridge) into data/raw.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--symbol", help="single symbol to download")
    parser.add_argument("--symbols", help="comma-separated list of symbols")
    parser.add_argument("--timeframe", default="D1", help="M1..MN1, H1, H4, D1...")
    parser.add_argument(
        "--bars",
        type=int,
        default=None,
        help="number of most recent bars to fetch (default depends on timeframe)",
    )
    parser.add_argument(
        "--group", default=None, help="subdirectory to save into (full_fx, h1, h4, ...)"
    )
    parser.add_argument("--data-dir", default="data/raw")
    parser.add_argument("--host", default=DEFAULT_HOST)
    parser.add_argument("--port", type=int, default=DEFAULT_PORT)
    parser.add_argument(
        "--list-symbols",
        action="store_true",
        help="list symbols available in MT5 and exit",
    )
    parser.add_argument(
        "--group-filter", default=None, help="filter for --list-symbols, e.g. '*USD*'"
    )
    parser.add_argument(
        "--backfill",
        action="store_true",
        help="download the full MT5 universe across --timeframes into --target-dir",
    )
    parser.add_argument(
        "--timeframes",
        default="D1,H4,H1",
        help="comma-separated timeframes for --backfill",
    )
    parser.add_argument(
        "--target-dir",
        default="data/raw/mt5",
        help="root directory for --backfill data",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="re-download even if the parquet file exists",
    )
    args = parser.parse_args(argv)

    try:
        with MT5Provider(host=args.host, port=args.port) as prov:
            print(f"MT5 terminal: {prov.version()} | {prov.symbols_total()} symbols")

            if args.list_symbols:
                symbols = prov.list_symbols(args.group_filter)
                print(f"{len(symbols)} symbols:")
                for s in symbols:
                    print(f"  {s}")
                return 0

            symbols = []
            if args.symbols:
                symbols = [
                    s.strip().upper() for s in args.symbols.split(",") if s.strip()
                ]
            if args.symbol:
                symbols.append(args.symbol.upper())
            symbols = list(dict.fromkeys(symbols))

            if args.backfill:
                timeframes = [
                    tf.strip().upper()
                    for tf in args.timeframes.split(",")
                    if tf.strip()
                ]
                for tf in timeframes:
                    timeframe_id(tf)  # validate early
                downloaded, skipped, failures = backfill(
                    prov,
                    timeframes=timeframes,
                    target_dir=args.target_dir,
                    skip_existing=not args.force,
                    symbols=symbols or None,
                    group_filter=args.group_filter,
                    bars=args.bars,
                )
                print(
                    f"\n[backfill] done: {downloaded} downloaded, "
                    f"{skipped} skipped, {len(failures)} failed "
                    f"-> {args.target_dir}"
                )
                return 1 if failures else 0

            if not symbols:
                parser.error(
                    "provide --symbol or --symbols (or --list-symbols / --backfill)"
                )

            for sym in symbols:
                ensure_parquet(
                    sym,
                    args.timeframe,
                    data_dir=args.data_dir,
                    group=args.group,
                    bars=args.bars,
                    provider=prov,
                )
    except MT5Error as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
