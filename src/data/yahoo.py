"""
NexusQuant - Yahoo Finance OHLCV provider (no auth).

Public chart API fallback for symbols the MT5 terminal does not list, or
when the Wine bridge is down: US stocks, ETFs and exotic indices resolve
to their Yahoo ticker and are cached as standard NexusQuant parquet files
(same schema as the MT5 exports: date, symbol, open, high, low, close,
volume, spread_points), ready for the loader / scanner / model pipeline.

Mapping rules (see ``yahoo_ticker``):

* indices (US500 -> ^GSPC, USTEC -> ^NDX, US30 -> ^DJI, DXY, ...)
* metals (XAUUSD -> GC=F, XAGUSD -> SI=F, ...)
* commodities (USOIL/WTI -> CL=F, XBRUSD/BRENT -> BZ=F, ...)
* FX pairs (EURUSD -> EURUSD=X)
* crypto (BTCUSD -> BTC-USD)
* anything else is used as-is (stocks / ETFs: AAPL, GLD, ...)

Timeframes: D1 uses ``interval=1d`` (range 10y); H1 uses ``interval=1h``
(range 730d, the intraday cap); H4 resamples the 1h series to 4-hour
bars (UTC-anchored 00/04/08/12/16/20, matching the MT5 H4 convention).
Every failure returns None / raises ``YahooError`` - the resolver treats
Yahoo as the last-resort source, never a blocker.

CLI:

    python -m src.data.yahoo --symbol AAPL --timeframe D1
    python -m src.data.yahoo --symbols AAPL,MSFT,GLD --timeframe D1
    python -m src.data.yahoo --symbol EURUSD=X --ticker --dry-run  # map only
"""

from __future__ import annotations

import argparse
import json
import sys
import urllib.parse
import urllib.request
from pathlib import Path
from typing import List, Optional

import pandas as pd

from src.data.mt5 import SCHEMA
from src.data.regroup import CRYPTO_BASES, CURRENCY_CODES

_UA = {"User-Agent": "Mozilla/5.0 (NexusQuant research)"}
CHART_URL = "https://query1.finance.yahoo.com/v8/finance/chart/"

# Internal symbol -> Yahoo ticker for indices / metals / commodities.
# FX pairs and crypto follow generic rules in ``yahoo_ticker``; stocks and
# ETFs use the symbol as-is (AAPL -> AAPL).
INDEX_YAHOO = {
    "US500": "^GSPC",
    "SPX": "^GSPC",
    "USTEC": "^NDX",
    "NDX": "^NDX",
    "US30": "^DJI",
    "DJIA": "^DJI",
    "RUT": "^RUT",
    "GER30": "^GDAXI",
    "DAX": "^GDAXI",
    "UK100": "^FTSE",
    "JPN225": "^N225",
    "AUS200": "^AXJO",
    "FRA40": "^FCHI",
    "SWI20": "^SSMI",
    "HKG50": "^HSI",
    "SPA35": "^IBEX",
    "IN50": "^NSEI",
    "DXY": "DX-Y.NYB",
}
METAL_YAHOO = {
    "XAUUSD": "GC=F",
    "XAGUSD": "SI=F",
    "XPTUSD": "PL=F",
    "XPDUSD": "PA=F",
}
COMMODITY_YAHOO = {
    "USOIL": "CL=F",
    "WTI": "CL=F",
    "XTIUSD": "CL=F",
    "BRENT": "BZ=F",
    "UKOIL": "BZ=F",
    "XBRUSD": "BZ=F",
    "XNGUSD": "NG=F",
    "XCUUSD": "HG=F",
}

# Intraday data is capped at 730 days by the chart API; daily goes further.
RANGE_INTRADAY = "730d"
RANGE_DAILY = "10y"


class YahooError(RuntimeError):
    """Raised when a Yahoo fetch fails (network, bad ticker, empty payload)."""


def yahoo_ticker(symbol: str) -> str:
    """Map an internal NexusQuant symbol to its Yahoo chart ticker."""
    s = symbol.upper().strip()
    if s in INDEX_YAHOO:
        return INDEX_YAHOO[s]
    if s in METAL_YAHOO:
        return METAL_YAHOO[s]
    if s in COMMODITY_YAHOO:
        return COMMODITY_YAHOO[s]
    if len(s) == 6 and s[:3] in CURRENCY_CODES and s[3:] in CURRENCY_CODES:
        return f"{s}=X"
    for base in CRYPTO_BASES:
        if s == base + "USD" or s == base + "USDT":
            return f"{base}-USD"
    return s  # stocks / ETFs / unknown ticker as-is


def _chart(
    ticker: str, interval: str, range_: str, timeout: int = 25
) -> Optional[dict]:
    """GET the Yahoo chart payload; None on any failure (incl. 404)."""
    url = (
        CHART_URL + urllib.parse.quote(ticker) + f"?range={range_}&interval={interval}"
    )
    try:
        req = urllib.request.Request(url, headers=_UA)
        with urllib.request.urlopen(req, timeout=timeout) as r:
            payload = json.load(r)
    except Exception:
        return None
    if not payload or payload.get("chart", {}).get("error"):
        return None
    return payload


def _parse_payload(payload: dict) -> Optional[pd.DataFrame]:
    """Normalize a chart payload to date/open/high/low/close/volume.

    Yahoo pads the early series with null closes - those rows are dropped,
    along with any row where OHLC is not strictly positive.
    """
    try:
        result = payload["chart"]["result"][0]
        ts = result.get("timestamp") or []
        quote = (result.get("indicators", {}).get("quote") or [{}])[0]
        if not ts or not quote:
            return None
        frame = pd.DataFrame(
            {
                "date": pd.to_datetime(ts, unit="s"),
                "open": quote.get("open"),
                "high": quote.get("high"),
                "low": quote.get("low"),
                "close": quote.get("close"),
                "volume": quote.get("volume"),
            }
        )
        frame["volume"] = pd.to_numeric(frame["volume"], errors="coerce").fillna(0)
        for col in ("open", "high", "low", "close"):
            frame[col] = pd.to_numeric(frame[col], errors="coerce")
        frame = frame.dropna(subset=["open", "high", "low", "close"])
        frame = frame[(frame[["open", "high", "low", "close"]] > 0).all(axis=1)]
        return (
            frame.sort_values("date").reset_index(drop=True)
            if not frame.empty
            else None
        )
    except Exception:
        return None


def fetch_ohlcv(
    ticker: str, timeframe: str = "D1", range_: Optional[str] = None
) -> Optional[pd.DataFrame]:
    """
    Fetch OHLCV bars for a Yahoo ticker.

    D1 -> 1d bars (10y default); H1 -> 1h bars (730d cap); H4 -> 1h bars
    resampled to 4h (UTC 00/04/08/12/16/20 anchors). W1/MN1 fall back to
    daily with ``range_`` untouched. Returns None on any failure.
    """
    tf = timeframe.upper()
    if tf not in ("D1", "H4", "H1", "W1", "MN1"):
        raise YahooError(
            f"Unsupported timeframe {timeframe!r} for Yahoo; "
            f"use D1 / H4 / H1 (W1/MN1 fall back to daily)"
        )
    if tf == "H4":
        hourly = _chart_daily(ticker, "1h", range_ or RANGE_INTRADAY)
        if hourly is None or hourly.empty:
            return None
        res = (
            hourly.set_index("date")
            .resample("4h")
            .agg(
                {
                    "open": "first",
                    "high": "max",
                    "low": "min",
                    "close": "last",
                    "volume": "sum",
                }
            )
            .dropna(subset=["open", "high", "low", "close"])
        )
        res = res[(res[["open", "high", "low", "close"]] > 0).all(axis=1)]
        return res.reset_index() if not res.empty else None
    if tf == "H1":
        return _chart_daily(ticker, "1h", range_ or RANGE_INTRADAY)
    # D1 / W1 / MN1: daily bars (the best-effort fallback for weekly/monthly).
    return _chart_daily(ticker, "1d", range_ or RANGE_DAILY)


def _chart_daily(ticker: str, interval: str, range_: str):
    payload = _chart(ticker, interval, range_)
    frame = _parse_payload(payload) if payload else None
    if frame is None or interval != "1d":
        return frame
    # Daily bars come with exchange-close timestamps (e.g. 13:30 UTC).
    # Normalize to midnight so daily series align with the macro overlay's
    # date-indexed frames (DXY/VIX/TNX) and other D1 parquet files.
    frame = frame.copy()
    frame["date"] = pd.to_datetime(frame["date"]).dt.normalize()
    return frame


def ensure_yahoo_parquet(
    symbol: str,
    timeframe: str = "D1",
    data_dir: str = "data/raw",
    group: Optional[str] = None,
) -> Path:
    """
    Fetch ``symbol`` from Yahoo and cache it as a NexusQuant parquet file.

    ``group`` defaults to the classified asset-class folder (via
    ``classify_symbol``); the file is written as ``{group}/{SYMBOL}_{TF}.parquet``
    with the standard MT5 schema. Raises ``YahooError`` when the ticker is
    unknown, the fetch fails, or the payload is too short to be useful.
    """
    tf = timeframe.upper()
    ticker = yahoo_ticker(symbol)
    if ticker != symbol.upper().strip():
        print(f"[yahoo] {symbol} -> {ticker} ({tf})")
    frame = fetch_ohlcv(ticker, tf)
    if frame is None or len(frame) < 30:
        raise YahooError(
            f"Yahoo returned no usable data for {symbol} ({ticker}@{tf}) - "
            f"unknown ticker or offline?"
        )

    if group is None:
        from src.data.regroup import classify_symbol

        group = classify_symbol(symbol, set())
    frame = frame.copy()
    frame["symbol"] = symbol
    frame["spread_points"] = 0
    frame = frame[SCHEMA]
    path = Path(data_dir) / group / f"{symbol}_{tf}.parquet"
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_parquet(path, compression="zstd")
    print(f"[yahoo] cached {symbol}@{tf} -> {path} ({len(frame)} bars)")
    return path


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        description="Fetch OHLCV from Yahoo's public chart API (no auth)",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--symbol", help="single symbol")
    parser.add_argument("--symbols", help="comma-separated list")
    parser.add_argument("--timeframe", default="D1")
    parser.add_argument(
        "--group", default=None, help="group folder to save into (default: classified)"
    )
    parser.add_argument("--data-dir", default="data/raw")
    parser.add_argument(
        "--ticker",
        action="store_true",
        help="only print the symbol -> Yahoo ticker mapping",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="fetch + validate but do not write the file",
    )
    args = parser.parse_args(argv)

    symbols = []
    if args.symbols:
        symbols += [s.strip().upper() for s in args.symbols.split(",") if s.strip()]
    if args.symbol:
        symbols.append(args.symbol.upper())
    symbols = list(dict.fromkeys(symbols))

    if args.ticker:
        for s in symbols or ["AAPL", "EURUSD", "XAUUSD", "BTCUSD", "US500"]:
            print(f"  {s:10s} -> {yahoo_ticker(s)}")
        return 0

    if not symbols:
        parser.error("provide --symbol or --symbols")

    failed = 0
    for sym in symbols:
        try:
            if args.dry_run:
                frame = fetch_ohlcv(yahoo_ticker(sym), args.timeframe)
                print(
                    f"[dry-run] {sym} ({yahoo_ticker(sym)}): "
                    f"{0 if frame is None else len(frame)} bars"
                )
            else:
                ensure_yahoo_parquet(sym, args.timeframe, args.data_dir, args.group)
        except Exception as exc:
            failed += 1
            print(f"ERROR {sym}: {exc}", file=sys.stderr)
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
