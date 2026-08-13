"""
NexusQuant - Macro Overlay.

Adds a top-down context layer on top of the technical engines:

* **USD strength** - from the DXY (already in ``data/raw/indices/``; H4 is
  resampled to daily when no D1 file exists).
* **Risk sentiment** - VIX level + 20-day change (best-effort Yahoo fetch,
  cached under ``data/raw/macro/``; the overlay works without it).
* **Rates pressure** - US10Y (``^TNX``) trend, same best-effort fetch.

Each factor is scored causally in ``[-2, +2]`` on daily bars, combined into
a macro regime, and translated into a per-symbol *macro bias* (how the top
down macro is tilting that instrument - e.g. a strong dollar is a headwind
for EURUSD but a tailwind for USDJPY, and gold likes a weak dollar).

The bias powers a **gate**: a Buy-the-Dip signal can be filtered out when
the macro backdrop for that symbol is a strong headwind. Everything is
graceful - if only DXY exists locally, VIX/TNX are simply omitted.
"""

from __future__ import annotations

import json
import sys
import urllib.request
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

sys.path.append(str(Path(__file__).parent.parent.parent))

from src.data.regroup import (
    COMMODITY_BASES,
    CRYPTO_BASES,
    CURRENCY_CODES,
    INDEX_SYMBOLS,
    METAL_BASES,
)

MACRO_CACHE = "data/raw/macro"

# Yahoo tickers -> internal names.
YAHOO_TICKERS = {"vix": "^VIX", "tnx": "^TNX"}

# Sector ETFs (spec #9): symbol -> sector ETF ticker for the most common
# mega-caps. Symbols not in the map report sector_etf=None (graceful).
SECTOR_ETF = {
    "AAPL": "XLK",
    "MSFT": "XLK",
    "NVDA": "XLK",
    "AVGO": "XLK",
    "CRM": "XLK",
    "ORCL": "XLK",
    "ADBE": "XLK",
    "AMD": "XLK",
    "INTC": "XLK",
    "QCOM": "XLK",
    "TXN": "XLK",
    "CSCO": "XLK",
    "IBM": "XLK",
    "META": "XLC",
    "GOOGL": "XLC",
    "GOOG": "XLC",
    "NFLX": "XLC",
    "DIS": "XLC",
    "CMCSA": "XLC",
    "T": "XLC",
    "VZ": "XLC",
    "JPM": "XLF",
    "BAC": "XLF",
    "WFC": "XLF",
    "GS": "XLF",
    "MS": "XLF",
    "V": "XLF",
    "MA": "XLF",
    "AXP": "XLF",
    "C": "XLF",
    "UNH": "XLV",
    "JNJ": "XLV",
    "LLY": "XLV",
    "PFE": "XLV",
    "MRK": "XLV",
    "ABBV": "XLV",
    "TMO": "XLV",
    "AMZN": "XLY",
    "TSLA": "XLY",
    "HD": "XLY",
    "MCD": "XLY",
    "NKE": "XLY",
    "SBUX": "XLY",
    "LOW": "XLY",
    "WMT": "XLP",
    "COST": "XLP",
    "PG": "XLP",
    "KO": "XLP",
    "PEP": "XLP",
    "XOM": "XLE",
    "CVX": "XLE",
    "COP": "XLE",
    "SLB": "XLE",
    "BA": "XLI",
    "GE": "XLI",
    "CAT": "XLI",
    "HON": "XLI",
    "UPS": "XLI",
    "LIN": "XLB",
    "SHW": "XLB",
    "FCX": "XLB",
    "NEE": "XLU",
    "DUK": "XLU",
    "SO": "XLU",
    "PLD": "XLRE",
    "AMT": "XLRE",
    "EQIX": "XLRE",
}


# ---------------------------------------------------------------------------
# Series loading (local-first)
# ---------------------------------------------------------------------------


def load_dxy_daily(data_dir: str = "data/raw") -> Optional[pd.Series]:
    """
    Daily DXY close series (indexed by date).

    Prefers ``indices/DXY_D1.parquet``; otherwise resamples the H4 file to
    daily (last H4 close per day).
    """
    indices = Path(data_dir) / "indices"
    for path in (indices / "DXY_D1.parquet", indices / "DXY_H4.parquet"):
        if path.exists():
            try:
                df = pd.read_parquet(path)
                s = df.set_index("date")["close"].sort_index()
                s.index = pd.to_datetime(s.index)
                daily = s.resample("D").last().dropna()
                if len(daily) > 50:
                    return daily
            except Exception:
                continue
    return None


def load_yahoo_daily(
    name: str, cache_dir: str = MACRO_CACHE, range_: str = "10y"
) -> Optional[pd.Series]:
    """
    Daily close for a Yahoo ticker (``vix``/``tnx``), cached as parquet.

    Uses the public chart API (no auth); returns None on any failure so the
    overlay degrades gracefully. ``range_`` can be e.g. '5y'/'10y'/'max'.
    """
    cache = Path(cache_dir) / f"{name}_D1.parquet"
    if cache.exists():
        try:
            s = pd.read_parquet(cache).set_index("date")["close"]
            s.index = pd.to_datetime(s.index)
            return s.sort_index()
        except Exception:
            pass
    # Names not in YAHOO_TICKERS are used as raw tickers (e.g. a sector
    # ETF like 'XLK'), which is also how the sector-ETF cache is keyed.
    ticker = YAHOO_TICKERS.get(name, name)
    url = (
        f"https://query1.finance.yahoo.com/v8/finance/chart/{ticker}?"
        f"range={range_}&interval=1d"
    )
    try:
        req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
        with urllib.request.urlopen(req, timeout=15) as r:
            payload = json.load(r)
        result = payload["chart"]["result"][0]
        ts = result.get("timestamp") or []
        close = result.get("indicators", {}).get("quote", [{}])[0].get("close") or []
        if not ts or len(close) != len(ts):
            return None
        # Yahoo pads the early series with null closes -> let pandas turn
        # them into NaN instead of crashing on float(None).
        s = pd.Series(
            pd.to_numeric(pd.Series(close, dtype="object"), errors="coerce").values,
            index=pd.to_datetime(ts, unit="s").normalize(),
        )
        s = s[~s.isna()]
        cache.parent.mkdir(parents=True, exist_ok=True)
        pd.DataFrame({"date": s.index, "close": s.values}).to_parquet(cache)
        return s.sort_index()
    except Exception:
        return None


def load_spx_daily(
    data_dir: str = "data/raw", cache_dir: str = MACRO_CACHE
) -> Optional[pd.Series]:
    """
    Daily S&P 500 close series (market beta / correlation anchor, spec #9).

    Prefers local parquet (``indices/US500_D1.parquet`` or the Yahoo D1
    under ``candidates/``), then the cached Yahoo fetch (``^GSPC``). Never
    fetches from here - the scanner/report run over the whole universe.
    """
    # Reuse the resolver's cross-group search: covers indices/, candidates/,
    # equity/ and nested layouts with longest-history preference, instead of
    # a hardcoded path list that misses folders.
    try:
        from src.data.resolver import find_local

        path = find_local("US500", "D1", data_dir)
        if path is None:
            path = find_local("SPX", "D1", data_dir)
        if path is not None and path.exists():
            s = pd.read_parquet(path).set_index("date")["close"]
            s.index = pd.to_datetime(s.index)
            daily = s.resample("D").last().dropna()
            if len(daily) > 50:
                return daily
    except Exception:
        pass
    # Cached Yahoo ^GSPC (populated manually / by a future --fetch-spx).
    p = Path(cache_dir) / "spx_D1.parquet"
    if p.exists():
        try:
            s = pd.read_parquet(p).set_index("date")["close"]
            s.index = pd.to_datetime(s.index)
            daily = s.resample("D").last().dropna()
            if len(daily) > 50:
                return daily
        except Exception:
            pass
    return None


_MACRO_FRAME_CACHE: Dict[tuple, pd.DataFrame] = {}


def _macro_frame_cached(
    data_dir: str, cache_dir: str, fetch: bool = False
) -> pd.DataFrame:
    """``load_macro_frame`` with an in-process cache.

    Universe scans call ``macro_report_for_symbol`` once per symbol; without
    caching each call re-reads the same DXY/VIX/TNX parquet files and
    re-derives the 200-SMA/RSI series (seconds per symbol). The cache key is
    (data_dir, cache_dir, fetch) - safe because data freshness is managed by
    the update pipeline (a cache invalidation is a process restart, which the
    daily cron does naturally).
    """
    key = (data_dir, cache_dir, fetch)
    if key not in _MACRO_FRAME_CACHE:
        _MACRO_FRAME_CACHE[key] = load_macro_frame(data_dir, cache_dir, fetch)
    return _MACRO_FRAME_CACHE[key]


def load_macro_frame(
    data_dir: str = "data/raw", cache_dir: str = MACRO_CACHE, fetch: bool = False
) -> pd.DataFrame:
    """
    Daily macro frame with any of ``dxy`` / ``vix`` / ``tnx`` columns present.

    ``fetch=True`` attempts a Yahoo download for VIX/TNX (offline-safe:
    failures just leave the columns absent).
    """
    series: Dict[str, pd.Series] = {}
    dxy = load_dxy_daily(data_dir)
    if dxy is not None:
        series["dxy"] = dxy
    for name in ("vix", "tnx"):
        s = load_yahoo_daily(name, cache_dir)
        if s is not None:
            series[name] = s
        elif fetch:
            series[name] = load_yahoo_daily(name, cache_dir)  # retry w/ cache
    if not series:
        return pd.DataFrame()
    frame = pd.concat(series, axis=1, sort=True)
    # Daily closes: carry each series across non-trading days (weekends /
    # holidays) so indicator windows (e.g. diff(20)) never see internal NaN.
    frame = frame.ffill()
    frame.index.name = "date"
    return frame.sort_index().dropna(how="all")


# ---------------------------------------------------------------------------
# Causal factor scores ([-2, +2] per factor, daily)
# ---------------------------------------------------------------------------


def _score_from_components(parts: List[int]) -> int:
    return int(np.clip(sum(parts), -2, 2))


def factor_scores(daily: pd.DataFrame) -> pd.DataFrame:
    """
    Per-day factor scores in ``[-2, +2]``.

    * ``dxy_score`` - USD strength: price vs SMA200 (+/-1) plus RSI(14) vs
      50 (+/-1); positive = strong dollar.
    * ``vix_score``  - risk-on/off: level bands (+/-1) plus 20-day change
      (+/-1); positive = risk-on (low or falling VIX).
    * ``tnx_score``  - rates pressure: close vs SMA200 (+/-1) plus 20-day
      slope (+/-1); positive = easing (falling yields).

    All indicators are causal (rolling windows only), and rows with NaN
    warm-up are dropped.
    """
    out = pd.DataFrame(index=daily.index)
    if "dxy" in daily:
        c = daily["dxy"]
        sma200 = c.rolling(200).mean()
        rsi = _rsi(c, 14)
        out["dxy_score"] = [
            _score_from_components(
                [
                    1 if close > m else -1 if close < m else 0,
                    1
                    if (not np.isnan(r) and r > 50)
                    else -1
                    if (not np.isnan(r) and r < 50)
                    else 0,
                ]
            )
            for close, m, r in zip(c, sma200, rsi, strict=True)
        ]
        # NaN during warm-up (rsi/sma not yet defined) -> score 0 rows NaN
        out.loc[~sma200.notna(), "dxy_score"] = np.nan
    if "vix" in daily:
        c = daily["vix"]
        chg = c.diff(20)
        level = np.select([c < 15, c > 25], [1, -1], default=0)
        delta = np.select([chg < -2, chg > 2], [1, -1], default=0)
        out["vix_score"] = np.clip(level.astype(int) + delta.astype(int), -2, 2).astype(
            float
        )
        out.loc[chg.isna(), "vix_score"] = np.nan
    if "tnx" in daily:
        c = daily["tnx"]
        sma200 = c.rolling(200).mean()
        slope = c.diff(20)
        trend = (c > sma200).astype(int) * 2 - 1
        trend = trend.where(sma200.notna(), np.nan)
        slope_sig = np.select([slope < -0.1, slope > 0.1], [1, -1], default=0)
        out["tnx_score"] = np.clip(trend + slope_sig, -2, 2)
    return out.dropna(how="all")


def _rsi(close: pd.Series, n: int = 14) -> pd.Series:
    delta = close.diff()
    gain = delta.clip(lower=0).rolling(n).mean()
    loss = (-delta.clip(upper=0)).rolling(n).mean()
    rs = gain / loss.replace(0, np.nan)
    rsi = 100 - 100 / (1 + rs)
    return rsi


def macro_regime(scores_row: Dict) -> Dict:
    """Human labels + composite from one row of factor scores."""
    dxy = scores_row.get("dxy_score")
    vix = scores_row.get("vix_score")
    tnx = scores_row.get("tnx_score")

    def band(v, up, down):
        if v is None or np.isnan(v):
            return None
        return up if v >= 1 else down if v <= -1 else "Neutral"

    usd = band(dxy, "USD Bullish", "USD Bearish") or "USD Neutral"
    risk = band(vix, "Risk-On", "Risk-Off") or "Neutral"
    rates = band(tnx, "Easing", "Tightening") or "Neutral"
    composite = float(
        np.nanmean([v for v in (dxy, vix, tnx) if v is not None and not np.isnan(v)])
    )

    # USD strength is not inherently risk-positive; the composite here is a
    # pure average of the factors for display (bias mapping handles signs).
    return {
        "usd": usd,
        "risk": risk,
        "rates": rates,
        "composite": round(composite, 2),
        "dxy_score": None if dxy is None or np.isnan(dxy) else int(dxy),
        "vix_score": None if vix is None or np.isnan(vix) else int(vix),
        "tnx_score": None if tnx is None or np.isnan(tnx) else int(tnx),
    }


# ---------------------------------------------------------------------------
# Symbol -> macro bias
# ---------------------------------------------------------------------------


def _symbol_class(symbol: str) -> tuple:
    s = symbol.upper()
    if s[:3] in METAL_BASES:
        return ("metal", s[:3])
    if len(s) == 6 and s[:3] in CURRENCY_CODES and s[3:] in CURRENCY_CODES:
        return ("fx", (s[:3], s[3:]))
    if s in COMMODITY_BASES or s[:3] in COMMODITY_BASES:
        return ("commodity", s[:3])
    if (
        s in CRYPTO_BASES
        or s[:3] in CRYPTO_BASES
        or (s.endswith("USD") and s[:-3] not in CURRENCY_CODES)
    ):
        return ("crypto", s)
    if s in INDEX_SYMBOLS or (any(ch.isdigit() for ch in s) and not s.endswith("USD")):
        return ("index", s)
    return ("equity", s)


def macro_bias_for_symbol(symbol: str, scores_row: Dict) -> Dict:
    """
    How the macro backdrop tilts a symbol: ``{bias, label, factors}``.

    bias in ``[-2, +2]`` (positive = macro tailwind). Class rules:

    * FX pair quoted in USD (EURUSD) - dollar strength is a headwind.
    * FX pair based in USD (USDJPY)  - dollar strength is a tailwind.
    * FX cross (EURGBP)              - mild composite of risk and USD.
    * Metals (XAUUSD...)             - anti-dollar / anti-real-yield.
    * Indices & equities             - risk-on positive, tightening negative.
    * Crypto                         - strongly risk-sensitive.
    """
    dxy = scores_row.get("dxy_score")
    vix = scores_row.get("vix_score")
    tnx = scores_row.get("tnx_score")

    def num(v, default=0.0):
        if v is None or np.isnan(v):
            return default
        return float(v)

    kind, detail = _symbol_class(symbol)
    if kind == "fx":
        base, quote = detail
        if quote == "USD" and base != "USD":
            raw = -num(dxy)
            note = f"USD strength is a headwind for {base}/{quote}"
        elif base == "USD":
            raw = num(dxy)
            note = f"USD strength is a tailwind for {base}/{quote}"
        else:
            raw = 0.5 * num(vix) - 0.5 * num(dxy)
            note = f"cross pair - mild {base}/{quote} composite"
    elif kind == "metal":
        raw = -0.6 * num(dxy) - 0.4 * num(tnx) + 0.3 * num(vix)
        note = f"{detail} is anti-dollar / anti-real-yield"
    elif kind == "crypto":
        raw = 1.5 * num(vix) - 0.3 * num(dxy)
        note = f"{detail} is strongly risk-sensitive"
    elif kind == "commodity":
        # Industrial/energy commodities: pro-cyclical, mildly anti-dollar.
        raw = num(vix) - 0.3 * num(dxy) - 0.2 * num(tnx)
        note = f"{detail} is pro-cyclical, mildly anti-dollar"
    elif kind == "index":
        raw = num(vix) - 0.5 * num(tnx)
        note = f"{detail} rallies on risk-on, suffers tightening"
    else:  # equity
        raw = num(vix) - 0.3 * num(tnx)
        note = f"{detail} is risk-sensitive"

    bias = float(np.clip(raw, -2, 2))
    if bias >= 1.5:
        label = "Strong Tailwind"
    elif bias >= 0.5:
        label = "Tailwind"
    elif bias > -0.5:
        label = "Neutral"
    elif bias > -1.5:
        label = "Headwind"
    else:
        label = "Strong Headwind"
    return {
        "bias": round(bias, 2),
        "label": label,
        "note": note,
        "factors": {
            "dxy": round(num(dxy), 1),
            "vix": round(num(vix), 1),
            "tnx": round(num(tnx), 1),
        },
    }


def macro_bias_series(symbol: str, aligned: pd.DataFrame) -> pd.Series:
    """
    Vectorized continuous macro bias in ``[-2, +2]`` for a full aligned frame.

    Same class rules as ``macro_bias_for_symbol`` (kept in one place so they
    cannot drift), but computed over every row at once - used to add macro
    bias as a feature to the ML model without a per-row dict round-trip.
    """
    kind, detail = _symbol_class(symbol)

    def _num(col: str) -> pd.Series:
        if col in aligned.columns:
            return pd.to_numeric(aligned[col], errors="coerce").fillna(0.0)
        return pd.Series(0.0, index=aligned.index)

    d = _num("dxy_score")
    v = _num("vix_score")
    t = _num("tnx_score")
    if kind == "fx":
        base, quote = detail
        if quote == "USD" and base != "USD":
            raw = -d
        elif base == "USD":
            raw = d
        else:
            raw = 0.5 * v - 0.5 * d
    elif kind == "metal":
        raw = -0.6 * d - 0.4 * t + 0.3 * v
    elif kind == "crypto":
        raw = 1.5 * v - 0.3 * d
    elif kind == "commodity":
        raw = v - 0.3 * d - 0.2 * t
    elif kind == "index":
        raw = v - 0.5 * t
    else:  # equity
        raw = v - 0.3 * t
    return pd.Series(np.clip(raw, -2, 2), index=aligned.index)


def macro_gate(
    symbol: str,
    scores_row: Dict,
    min_bias: float = -0.5,
    direction: str = "long",
) -> Dict:
    """
    Gate a signal: allowed unless the macro backdrop strongly opposes the
    trade's direction.

    ``bias`` is the symbol's macro *tailwind* (positive = bullish backdrop).
    For a LONG the gate blocks strong headwinds (``bias < min_bias``); for
    a SHORT the gate blocks strong tailwinds (``bias > -min_bias``) - a
    strong bullish macro backdrop argues against fading it. Symmetric by
    default (``min_bias=-0.5`` -> longs blocked below -0.5, shorts blocked
    above +0.5).
    """
    bias = macro_bias_for_symbol(symbol, scores_row)
    if direction == "short":
        allowed = bias["bias"] <= -min_bias
        reason = (
            "ok"
            if allowed
            else f"{bias['label']} (bias {bias['bias']:+.2f} > {-min_bias})"
        )
    else:
        allowed = bias["bias"] >= min_bias
        reason = (
            "ok"
            if allowed
            else f"{bias['label']} (bias {bias['bias']:+.2f} < {min_bias})"
        )
    return {
        "allowed": allowed,
        "bias": bias["bias"],
        "label": bias["label"],
        "reason": reason,
    }


def align_scores(
    scores: pd.DataFrame, index: pd.DatetimeIndex, shift_days: int = 1
) -> pd.DataFrame:
    """
    Align daily factor scores onto a symbol's bar index, strictly causal.

    Daily macro closes are known only after their day ends, so bars on day D
    use the macro state as of day D-1 (``shift_days=1``) via forward-fill.
    """
    shifted = scores.copy()
    if shift_days:
        shifted.index = shifted.index + pd.Timedelta(days=shift_days)
    return shifted.reindex(index, method="ffill")


def gate_series(
    symbol: str,
    scores: pd.DataFrame,
    index: pd.DatetimeIndex,
    min_bias: float = -0.5,
    shift_days: int = 1,
    direction: str = "long",
) -> pd.Series:
    """
    Causal boolean series: is the macro backdrop acceptable for ``symbol``
    in the given ``direction`` (long = block strong headwinds; short =
    block strong tailwinds)?

    One bool per bar in ``index`` (aligned + forward-filled, macro state as
    of the prior day). Bars before the first usable macro row default to
    ``True`` (no macro data yet -> no filter applied).
    """
    if scores is None or scores.empty:
        return pd.Series(True, index=index)
    aligned = align_scores(scores, index, shift_days=shift_days)
    allowed = aligned.apply(
        lambda row: macro_gate(
            symbol, row.to_dict(), min_bias, direction=direction
        )["allowed"],
        axis=1,
    )
    return allowed.fillna(True)


# ---------------------------------------------------------------------------
# Snapshot helpers (used by the report / scanner / CLI)
# ---------------------------------------------------------------------------

_MACRO_CACHE: Dict[str, Optional[pd.DataFrame]] = {}


def full_macro_scores(
    data_dir: str = "data/raw", cache_dir: str = MACRO_CACHE, fetch: bool = False
) -> Optional[pd.DataFrame]:
    """
    Full daily factor-score history, memoized per ``data_dir``.

    A universe scan calls this once and reuses it for every symbol; the
    macro backdrop is global (top-down), only the symbol mapping differs.
    Returns None (uncached) when no macro source is available.
    """
    # fetch is part of the key so a later --fetch in the same process is
    # never poisoned by an earlier fetch=False call's memoized frame.
    key = f"{data_dir}|{cache_dir}|{int(fetch)}"
    if key not in _MACRO_CACHE:
        frame = _macro_frame_cached(data_dir, cache_dir, fetch)
        _MACRO_CACHE[key] = factor_scores(frame) if not frame.empty else None
    return _MACRO_CACHE[key]


def macro_for_model(
    data_dir: str = "data/raw", cache_dir: str = MACRO_CACHE, fetch: bool = False
) -> Optional[pd.DataFrame]:
    """
    Memoized daily macro frame for the ML model: factor scores plus a raw
    ``dxy`` column (for 20-day USD momentum). Returns None when no macro
    source is available so callers degrade to neutral features.
    """
    key = f"model|{data_dir}|{cache_dir}|{int(fetch)}"
    if key not in _MACRO_CACHE:
        raw = _macro_frame_cached(data_dir, cache_dir, fetch)
        if raw is not None and not raw.empty:
            scores = factor_scores(raw)
            if scores is not None and not scores.empty:
                out = scores.copy()
                if "dxy" in raw.columns:
                    out["dxy"] = raw["dxy"]
                _MACRO_CACHE[key] = out
            else:
                _MACRO_CACHE[key] = None
        else:
            _MACRO_CACHE[key] = None
    return _MACRO_CACHE[key]


def latest_macro_scores(
    data_dir: str = "data/raw", cache_dir: str = MACRO_CACHE, fetch: bool = False
) -> Optional[pd.DataFrame]:
    """
    Last row of daily factor scores (the current macro backdrop).
    """
    full = full_macro_scores(data_dir, cache_dir, fetch=fetch)
    return None if full is None else full.tail(1)


def macro_sensitivities(
    symbol: str,
    df: Optional[pd.DataFrame] = None,
    data_dir: str = "data/raw",
    cache_dir: str = MACRO_CACHE,
    lookback: int = 90,
) -> Dict:
    """
    Macro sensitivity table (institutional spec #9): how the symbol's daily
    returns co-move with the market (S&P 500), the dollar (DXY), yields
    (US10Y) and volatility (VIX).

    * ``market_beta`` / ``spx_corr`` - trailing ``lookback``-day beta and
      correlation vs the S&P 500 (local US500, else cached Yahoo ^GSPC).
    * ``dollar_sens`` / ``yield_sens`` / ``vol_sens`` - same-window
      correlations vs DXY, TNX and VIX (the macro overlay's factor series).
    * ``sector_etf`` - correlation vs the symbol's sector ETF (``SECTOR_ETF``
      map, cached Yahoo fetch); None when unmapped or not cached.

    All series are aligned on the symbol's own bars with a 1-day lag
    (strictly causal). Returns ``{}`` when the symbol frame is missing or
    too short.
    """
    if df is None or len(df) < 40 or "close" not in df.columns:
        return {}
    sym = df["close"].astype(float).dropna()
    if len(sym) < lookback + 10:
        lookback = max(30, len(sym) // 2)
    sym_ret = sym.pct_change().reindex(sym.index)
    # Defensive: daily frames must align with the macro overlay's midnight
    # index regardless of source (MT5 server-time bars vs Yahoo closes). If
    # every bar shares the same time-of-day, normalize to midnight; intraday
    # frames (H4/H1) are left untouched (macro correlation is a daily
    # concept and correctly yields None there).
    times = sym.index.normalize()
    if len(times) and (sym.index - times).nunique() == 1:
        sym_ret.index = times
        sym = sym.copy()
        sym.index = times

    # Shift each macro series back 1 day so only *yesterday's* macro state
    # is compared to today's move (causal alignment, same as the gate).
    def _corr(macro: Optional[pd.Series], name: str) -> Optional[float]:
        if macro is None or len(macro) < 30:
            return None
        m = macro.copy()
        m.index = m.index + pd.Timedelta(days=1)
        aligned = pd.concat(
            [sym_ret.rename("sym"), m.reindex(sym.index).rename(name)], axis=1
        )
        aligned = aligned.tail(lookback).dropna()
        if len(aligned) < 25 or aligned[name].std() == 0 or aligned["sym"].std() == 0:
            return None
        return round(float(aligned["sym"].corr(aligned[name])), 3)

    out: Dict = {"lookback": lookback}

    spx = load_spx_daily(data_dir, cache_dir)
    out["spx_corr"] = _corr(spx, "spx")
    spx_ret = spx.pct_change().dropna() if spx is not None else None
    if spx_ret is not None and len(spx_ret) >= 25:
        m = spx_ret.copy()
        m.index = m.index + pd.Timedelta(days=1)
        aligned = (
            pd.concat(
                [sym_ret.rename("sym"), m.reindex(sym.index).rename("spx")], axis=1
            )
            .tail(lookback)
            .dropna()
        )
        if len(aligned) >= 25 and aligned["spx"].std() > 0:
            out["market_beta"] = round(
                float(aligned["sym"].cov(aligned["spx"]) / aligned["spx"].var()), 2
            )

    frame = _macro_frame_cached(data_dir, cache_dir, fetch=False)
    if frame is not None and not frame.empty:
        if "dxy" in frame.columns:
            out["dollar_sens"] = _corr(frame["dxy"], "dxy")
        if "tnx" in frame.columns:
            out["yield_sens"] = _corr(frame["tnx"], "tnx")
        if "vix" in frame.columns:
            out["vol_sens"] = _corr(frame["vix"], "vix")

    # Sector ETF - cache-only here (the report/scanner never fire network
    # calls). Warm the cache with ``python -m src.macro.run --fetch``.
    # The key is always emitted (corr None when unmapped / not cached) so
    # API consumers see one stable shape.
    etf = SECTOR_ETF.get(symbol.upper())
    c = None
    if etf:
        p = Path(cache_dir) / f"{etf}_D1.parquet"
        if p.exists():
            try:
                s = pd.read_parquet(p).set_index("date")["close"]
                s.index = pd.to_datetime(s.index)
                c = _corr(s, "etf")
            except Exception:
                c = None
    out["sector_etf"] = {"ticker": etf, "corr": c}
    return out


def macro_report_for_symbol(
    symbol: str,
    data_dir: str = "data/raw",
    cache_dir: str = MACRO_CACHE,
    fetch: bool = False,
    df: Optional[pd.DataFrame] = None,
) -> Optional[Dict]:
    """
    One-shot top-down snapshot for a symbol: ``{regime, bias, gate,
    sensitivities}`` or None when no macro source is available. The
    canonical shape consumed by the report generator, scanner and
    dashboard. ``df`` is optional - when provided, the sensitivity table
    (spec #9) is computed; without it (cheap universe scans) it is omitted.
    """
    row = latest_macro_scores(data_dir, cache_dir, fetch=fetch)
    if row is None:
        return None
    row_dict = row.iloc[-1].to_dict()
    out = {
        "regime": macro_regime(row_dict),
        "bias": macro_bias_for_symbol(symbol, row_dict),
        "gate": macro_gate(symbol, row_dict, direction="long"),
        "gate_short": macro_gate(symbol, row_dict, direction="short"),
    }
    if df is not None:
        out["sensitivities"] = macro_sensitivities(symbol, df, data_dir, cache_dir)
    return out
