"""
NexusQuant - COT positioning: downloader, cache and loader (graceful offline).

Commitments of Traders net-position percentiles give a positioning-context
feature: when speculators are stretched long (percentile ~100) the bar is
crowded and reversals get more likely; extreme shorts (~0) can mark
exhaustion. The module has three layers:

* **Downloader** - pulls non-commercial (speculative) futures-only positions
  for 15 markets (8 FX/metals currencies + silver, WTI, Brent and the
  S&P/NASDAQ/DOW/Russell equity-index futures) from the CFTC Socrata API
  (``6dca-aqww`` - Disaggregated Futures Only, no auth) and converts each
  market's weekly net positioning into a **causal percentile** (0-100)
  of its own history (expanding window - each point ranked only against
  data available at that time).
* **Cache** - writes ``data/raw/cot/{KEY}_cot.csv`` (``date,percentile``)
  per market and refreshes them weekly (COT publishes every Tuesday; a
  report older than ``DEFAULT_STALE_DAYS`` is stale). ``--force`` overrides.
* **Loader** - the model's feature pipeline reads those CSVs via ``load_cot``
  (lru-cached); anything missing degrades to the neutral 50 baseline, so
  training / scanning / the API never depend on COT availability.

CLI:

    python -m src.model.cot --status            # offline cache report
    python -m src.model.cot --fetch             # refresh when stale
    python -m src.model.cot --fetch --force     # always refetch
    python -m src.model.cot --fetch --json
"""

from __future__ import annotations

import argparse
import functools
import json
import urllib.parse
import urllib.request
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

# Symbol -> currency whose COT positioning drives it (for JPY crosses the
# dollar/risk leg is the driver; gold uses its own futures positioning).
CCY_MAP = {
    "EURUSD": "EUR",
    "GBPUSD": "GBP",
    "USDJPY": "JPY",
    "USDCHF": "CHF",
    "AUDUSD": "AUD",
    "NZDUSD": "NZD",
    "USDCAD": "CAD",
    "XAUUSD": "GOLD",
    "EURJPY": "EUR",
    "GBPJPY": "GBP",
    "NZDJPY": "NZD",
    "AUDJPY": "AUD",
    "CADJPY": "CAD",
    "EURGBP": "EUR",
    "EURAUD": "EUR",
    "EURCAD": "EUR",
    "EURCHF": "EUR",
    "EURNZD": "EUR",
    "GBPAUD": "GBP",
    "GBPCAD": "GBP",
    "GBPCHF": "GBP",
    "GBPNZD": "GBP",
    "AUDCAD": "AUD",
    "AUDCHF": "AUD",
    "AUDNZD": "AUD",
    "NZDCAD": "NZD",
    "NZDCHF": "NZD",
    "CADCHF": "CAD",
    "CHFJPY": "CHF",
}

# Currency -> exact CFTC contract market name (primary contract only, so the
# exact-name match excludes the cross-rate markets like "EURO FX/JAPANESE
# YEN XRATE"). Verified against the live API on 2026-08-12.
CCY_MARKET = {
    "EUR": "EURO FX",
    "GBP": "BRITISH POUND",
    "JPY": "JAPANESE YEN",
    "CHF": "SWISS FRANC",
    "AUD": "AUSTRALIAN DOLLAR",
    "NZD": "NZ DOLLAR",
    "CAD": "CANADIAN DOLLAR",
    "GOLD": "GOLD",
}

# Metals / energy / index markets (same source, verified 2026-08-12). The
# ICE U.S. Dollar Index futures are NOT in the CFTC disaggregated report, so
# there is no DXY series - USD positioning is already covered by the seven
# currency futures above.
ASSET_MARKET = {
    "XAG": "SILVER",
    "WTI": "WTI CRUDE OIL 1ST LINE",
    "BRENT": "BRENT LAST DAY",
    "SP500": "E-MINI S&P 500",
    "NASDAQ": "NASDAQ MINI",
    "DOW": "DJIA x $5",
    "RUSSELL": "RUSSELL E-MINI",
}

# Every market the downloader tracks (used for the fetch query, cache
# freshness and pruning).
ALL_MARKETS = {**CCY_MARKET, **ASSET_MARKET}

# Instrument symbols -> positioning market key. The downloader/cache live
# under ``data/raw/cot/{KEY}_cot.csv``; ``cot_features`` resolves a symbol
# through this map first, then through ``CCY_MAP``.
SYMBOL_MARKET = {
    "XAGUSD": "XAG",
    "XAGJPY": "XAG",
    "XAGEUR": "XAG",
    "XAGGBP": "XAG",
    "XAGNZD": "XAG",
    "US500": "SP500",
    "SPX": "SP500",
    "USTEC": "NASDAQ",
    "NDX": "NASDAQ",
    "US30": "DOW",
    "DJIA": "DOW",
    "RUT": "RUSSELL",
    "XTIUSD": "WTI",
    "USOIL": "WTI",
    "WTI": "WTI",
    "XBRUSD": "BRENT",
    "UKOIL": "BRENT",
    "BRENT": "BRENT",
}

# CFTC Socrata endpoint: Disaggregated Futures-Only report (free, no auth).
COT_API_URL = "https://publicreporting.cftc.gov/resource/6dca-aqww.json"

DEFAULT_COT_DIR = "data/raw/cot"
# COT publishes weekly on Tuesdays. Two weekly cycles are tolerated before
# the cache counts as stale: some contracts (e.g. "WTI CRUDE OIL 1ST LINE")
# consistently report one week behind the rest, so a single-missed-week
# window would mark an otherwise-current cache stale forever.
DEFAULT_STALE_DAYS = 16
# At least a year of weekly reports before a currency is usable.
MIN_ROWS = 52  # Socrata row cap - 20+ years of weekly reports across 15 markets is
# ~20k rows today, well under the cap; pagination still guards growth.
FETCH_LIMIT = 50000

_UA = {"User-Agent": "Mozilla/5.0 (NexusQuant research)"}


# ---------------------------------------------------------------------------
# Loader (used by the model feature pipeline)
# ---------------------------------------------------------------------------


@functools.lru_cache(maxsize=4)
def load_cot(data_dir: str = DEFAULT_COT_DIR) -> Optional[Dict[str, pd.Series]]:
    """
    Load all available COT percentile Series (``{KEY}_cot.csv`` per market).

    Returns ``{market: Series(percentile)}`` or ``None`` when the directory
    or files are missing. Cached so universe scans don't re-read the CSVs on
    every symbol. The Series index is parsed as dates and sorted ascending.
    """
    d = Path(data_dir)
    if not d.is_dir():
        return None
    out: Dict[str, pd.Series] = {}
    for p in sorted(d.glob("*_cot.csv")):
        ccy = p.name.replace("_cot.csv", "").upper()
        try:
            df = pd.read_csv(p)
            df.columns = [str(c).strip().lower() for c in df.columns]
            if "date" not in df.columns or "percentile" not in df.columns:
                continue
            s = pd.to_numeric(df["percentile"], errors="coerce")
            s.index = pd.to_datetime(df["date"], errors="coerce")
            s = s[~s.index.isna()].sort_index().dropna()
            if len(s) < 10:
                continue
            out[ccy] = s
        except Exception:
            continue
    return out if out else None


def cot_for_symbol(symbol: str, data_dir: str = DEFAULT_COT_DIR) -> Optional[pd.Series]:
    """
    COT percentile Series for a symbol's positioning market, or None.

    Resolves through ``SYMBOL_MARKET`` first (metals / energy / indices,
    e.g. US500 -> SP500), then through ``CCY_MAP`` for FX symbols.
    """
    cot = load_cot(data_dir)
    if not cot:
        return None
    key = SYMBOL_MARKET.get(symbol) or CCY_MAP.get(symbol)
    if key is None:
        return None
    return cot.get(key)


# ---------------------------------------------------------------------------
# Downloader
# ---------------------------------------------------------------------------


def _fetch_json(url: str, timeout: int) -> list:
    """GET a JSON array from ``url``; raises on any failure."""
    req = urllib.request.Request(url, headers=_UA)
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.load(r)


def fetch_cot_raw(timeout: int = 60) -> Optional[Dict[str, pd.DataFrame]]:
    """
    Fetch non-commercial futures-only net positioning per market.

    Queries the CFTC Disaggregated Futures-Only report for the exact primary
    contract of each market (excludes cross-rate markets), keeps only
    futures-only rows, and returns ``{key: DataFrame(date, net, oi)}`` with
    ``net = noncommercial long - short``. Markets with less than a year of
    reports are dropped. Returns None on any network/parse failure.
    """
    markets = "','".join(ALL_MARKETS.values())
    # ~8 markets x ~1,930 weekly reports ≈ 15k rows today - well under the
    # Socrata cap, but paginate anyway so growth (more markets) can never
    # silently truncate history.
    base = {
        "$select": (
            "report_date_as_yyyy_mm_dd,contract_market_name,"
            "noncomm_positions_long_all,"
            "noncomm_positions_short_all,open_interest_all"
        ),
        "$where": (
            f"contract_market_name in ('{markets}') and futonly_or_combined='FutOnly'"
        ),
        "$order": "report_date_as_yyyy_mm_dd",
        "$limit": FETCH_LIMIT,
    }
    all_rows: List[dict] = []
    try:
        offset = 0
        while True:
            params = urllib.parse.urlencode({**base, "$offset": offset})
            page = _fetch_json(f"{COT_API_URL}?{params}", timeout)
            if not isinstance(page, list) or not page:
                break
            all_rows.extend(page)
            if len(page) < FETCH_LIMIT:
                break
            offset += FETCH_LIMIT
    except Exception:
        return None
    rows = all_rows
    if not rows:
        return None

    df = pd.DataFrame(rows)
    rev = {v: k for k, v in ALL_MARKETS.items()}
    # Exact-name match: every contract name must be unique, otherwise a
    # future edit could silently drop a market from the reverse map.
    assert len(rev) == len(ALL_MARKETS), (
        f"duplicate CFTC contract names in ALL_MARKETS: {ALL_MARKETS}"
    )
    df["ccy"] = df.get("contract_market_name", pd.Series(dtype=str)).map(rev)
    df = df[df["ccy"].notna()].copy()
    if df.empty:
        return None
    df["date"] = pd.to_datetime(df.get("report_date_as_yyyy_mm_dd"), errors="coerce")
    for col in (
        "noncomm_positions_long_all",
        "noncomm_positions_short_all",
        "open_interest_all",
    ):
        df[col] = pd.to_numeric(df.get(col), errors="coerce")

    out: Dict[str, pd.DataFrame] = {}
    for ccy, g in df.groupby("ccy"):
        g = g.sort_values("date").drop_duplicates("date", keep="last")
        g = g[
            [
                "date",
                "noncomm_positions_long_all",
                "noncomm_positions_short_all",
                "open_interest_all",
            ]
        ].dropna()
        if len(g) < MIN_ROWS:
            continue
        g = g.copy()
        g["net"] = g["noncomm_positions_long_all"] - g["noncomm_positions_short_all"]
        out[ccy] = g.reset_index(drop=True)
    return out or None


def expanding_percentile(values: pd.Series) -> pd.Series:
    """
    Causal percentile (0-100): each point ranked against its own history
    only (expanding window), so no future information leaks into the value.
    """
    x = np.asarray(values, dtype=float)
    n = len(x)
    out = np.empty(n)
    for i in range(n):
        out[i] = (x[: i + 1] <= x[i]).mean() * 100.0
    return pd.Series(out, index=values.index)


def percentile_series(g: pd.DataFrame) -> pd.Series:
    """Weekly net-position percentile Series (indexed by report date)."""
    pct = expanding_percentile(g["net"])
    pct.index = pd.DatetimeIndex(g["date"].values)
    return pct.sort_index()


# ---------------------------------------------------------------------------
# Cache
# ---------------------------------------------------------------------------


def _cache_summary(cot_dir: Path) -> Dict[str, dict]:
    """{ccy: {rows, last_date, last_percentile}} from the cached CSVs."""
    out: Dict[str, dict] = {}
    if not cot_dir.is_dir():
        return out
    for p in sorted(cot_dir.glob("*_cot.csv")):
        ccy = p.name.replace("_cot.csv", "").upper()
        try:
            df = pd.read_csv(p)
            s = pd.to_numeric(df.get("percentile"), errors="coerce").dropna()
            d = pd.to_datetime(df.get("date"), errors="coerce").dropna()
            if s.empty or d.empty:
                continue
            out[ccy] = {
                "rows": int(len(s)),
                "last_date": str(d.max().date()),
                "last_percentile": round(float(s.iloc[-1]), 1),
            }
        except Exception:
            continue
    return out


def _cache_fresh(cot_dir: Path, max_stale_days: int = DEFAULT_STALE_DAYS) -> bool:
    """True when every expected market file exists and its last report
    date is within ``max_stale_days`` of today."""
    if not cot_dir.is_dir():
        return False
    today = pd.Timestamp.now(tz=None).normalize()
    for key in ALL_MARKETS:
        p = cot_dir / f"{key}_cot.csv"
        if not p.exists():
            return False
        try:
            df = pd.read_csv(p)
            last = pd.to_datetime(df.get("date"), errors="coerce").max()
            if pd.isna(last) or (today - last).days > max_stale_days:
                return False
        except Exception:
            return False
    return True


def _write_cot_files(
    cot_dir: Path, series_map: Dict[str, pd.Series]
) -> Dict[str, dict]:
    """Write ``{KEY}_cot.csv`` (date,percentile) per market."""
    cot_dir.mkdir(parents=True, exist_ok=True)
    for ccy, s in series_map.items():
        frame = pd.DataFrame(
            {"date": s.index.strftime("%Y-%m-%d"), "percentile": s.round(2).values}
        )
        frame.to_csv(cot_dir / f"{ccy}_cot.csv", index=False)
    return _cache_summary(cot_dir)


def update_cot(
    data_dir: str = DEFAULT_COT_DIR, force: bool = False, timeout: int = 60
) -> Dict:
    """
    Refresh the cached COT CSVs when stale; graceful on failure.

    Returns a summary dict: ``fetched`` (bool), ``error`` (str|None),
    ``reason`` and per-market ``currencies`` (see ``_cache_summary``).
    Never raises - a failed fetch leaves the existing cache in place.
    """
    cot_dir = Path(data_dir) / "cot"
    if not force and _cache_fresh(cot_dir):
        return {
            "fetched": False,
            "reason": "cache fresh",
            "error": None,
            "currencies": _cache_summary(cot_dir),
        }

    raw = fetch_cot_raw(timeout)
    if not raw:
        return {
            "fetched": False,
            "reason": "fetch attempted",
            "error": "CFTC fetch failed (offline?)",
            "currencies": _cache_summary(cot_dir),
        }

    series_map = {ccy: percentile_series(g) for ccy, g in raw.items()}
    written = _write_cot_files(cot_dir, series_map)
    # Prune files for markets absent from this fetch: otherwise a stale
    # CSV would be served by load_cot and would keep _cache_fresh failing.
    for key in ALL_MARKETS:
        if key not in series_map:
            stale = cot_dir / f"{key}_cot.csv"
            if stale.exists():
                stale.unlink()
    # The loader is lru-cached; a fresh process would be fine, but clear so
    # a long-lived server picks up the new files immediately.
    load_cot.cache_clear()
    return {"fetched": True, "reason": "fetched", "error": None, "currencies": written}


def cot_status(data_dir: str = DEFAULT_COT_DIR) -> Dict:
    """Offline report of the COT cache state (never touches the network)."""
    cot_dir = Path(data_dir) / "cot"
    cur = _cache_summary(cot_dir)
    fresh = _cache_fresh(cot_dir)
    missing = [key for key in ALL_MARKETS if key not in cur]
    return {"fresh": fresh, "missing": missing, "currencies": cur, "dir": str(cot_dir)}


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        description="CFTC COT positioning cache (download / status)",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--fetch", action="store_true", help="refresh the COT cache when stale"
    )
    parser.add_argument(
        "--force", action="store_true", help="always refetch (ignore freshness)"
    )
    parser.add_argument(
        "--status", action="store_true", help="offline cache report (no network)"
    )
    parser.add_argument("--data-dir", default="data/raw")
    parser.add_argument("--json", action="store_true", help="emit JSON")
    args = parser.parse_args(argv)

    if args.status:
        res = cot_status(args.data_dir)
        if args.json:
            print(json.dumps(res, indent=2))
        else:
            _print_status(res)
        return 0

    if args.fetch:
        res = update_cot(args.data_dir, force=args.force)
        if args.json:
            print(json.dumps(res, indent=2))
        else:
            if res["error"]:
                print(f"COT: {res['error']} - using existing cache")
            elif res["fetched"]:
                print(
                    f"COT: fetched {len(res['currencies'])} currencies "
                    f"→ {Path(args.data_dir) / 'cot'}"
                )
            else:
                print(f"COT: {res['reason']}")
            _print_status(cot_status(args.data_dir))
        return 0 if not res["error"] else 1

    # No action: default to status.
    _print_status(cot_status(args.data_dir))
    return 0


def _print_status(res: Dict) -> None:
    if res["fresh"]:
        print("COT cache: FRESH (all markets, weekly)")
    else:
        missing = res.get("missing", [])
        print(
            f"COT cache: STALE / INCOMPLETE"
            f"{f' (missing: {missing})' if missing else ''}"
        )
        print("  run `python -m src.model.cot --fetch` to refresh")
    for ccy, info in sorted(res.get("currencies", {}).items()):
        print(
            f"  {ccy:6s} {info['rows']:>5} rows · last {info['last_date']} "
            f"· pct {info['last_percentile']:.0f}"
        )


if __name__ == "__main__":
    raise SystemExit(main())
