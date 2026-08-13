"""
NexusQuant - Fundamental Factor Model (institutional spec #8).

Scores each equity on the three classic style factors, each 0-100:

* **Value**  - cheaper is better: low P/E, low EV/EBITDA, low P/B.
* **Quality** - strong profitability with balance-sheet discipline: high
  ROE / ROA, low Debt/Equity.
* **Momentum** - price + earnings momentum: 1/3/6/12-month returns and
  RSI from local price data (always available), plus earnings surprise and
  analyst revisions when fundamentals are provided.

Data sources, in priority order (all graceful):

1. **Local CSV** - ``data/fundamentals/{SYMBOL}.csv`` (single) or
   ``data/fundamentals/universe.csv`` (one row per symbol). Columns:
   ``symbol, pe, ev_ebitda, pb, roe_pct, debt_to_equity, earnings_surprise_pct,
   analyst_revisions`` (all optional except symbol; missing -> that factor
   falls back to its price-based component).
2. **Yahoo attempt** - ``fetch_yahoo_fundamentals`` tries the v7 quote /
   quoteSummary endpoints. Yahoo now requires an authenticated crumb, so
   this normally fails cleanly (returns None) - the module keeps working
   on price momentum alone.
3. **Price-only** - Momentum is always scored from the local OHLCV frame;
   Value/Quality report ``None`` when no fundamentals source exists.

Composites are documented weights; every score carries its ``source`` so
downstream code can distinguish fundamentals-driven from price-only reads.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Dict, Optional

import numpy as np
import pandas as pd

sys.path.append(str(Path(__file__).parent.parent.parent))

FUNDAMENTALS_DIR = "data/fundamentals"
PRICE_RETURN_LOOKBACKS = (21, 63, 126, 252)  # ~1m / 3m / 6m / 12m


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------


def _find_csv(symbol: str, data_dir: str) -> Optional[Path]:
    base = Path(data_dir)
    single = base / f"{symbol}.csv"
    if single.exists():
        return single
    universe = base / "universe.csv"
    if universe.exists():
        return universe
    return None


def load_fundamentals(symbol: str, data_dir: str = FUNDAMENTALS_DIR) -> Optional[Dict]:
    """Per-symbol fundamentals from a local CSV (single-file or universe)."""
    path = _find_csv(symbol, data_dir)
    if path is None:
        return None
    try:
        if path.name == "universe.csv":
            df = pd.read_csv(path, dtype=str)
            row = df[df["symbol"].str.upper() == symbol.upper()]
            if row.empty:
                return None
            rec = row.iloc[0].to_dict()
        else:
            rec = pd.read_csv(path, dtype=str).iloc[0].to_dict()
    except Exception:
        return None
    if not rec:
        return None

    def num(key: str) -> Optional[float]:
        v = rec.get(key)
        try:
            return float(v)
        except (TypeError, ValueError):
            return None

    out = {
        k: num(k)
        for k in (
            "pe",
            "ev_ebitda",
            "pb",
            "roe_pct",
            "debt_to_equity",
            "earnings_surprise_pct",
            "analyst_revisions",
        )
    }
    out["source"] = "csv"
    return out


def fetch_yahoo_fundamentals(symbol: str) -> Optional[Dict]:
    """
    Best-effort Yahoo fundamentals. Yahoo now requires an authenticated
    crumb/cookie for v7 quote and quoteSummary, so this normally returns
    None (documented degradation). Kept as a source hook: if Yahoo reopens
    these endpoints, wire it in here without touching the scoring layer.
    """
    try:
        import urllib.request

        url = f"https://query1.finance.yahoo.com/v7/finance/quote?symbols={symbol}"
        req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
        with urllib.request.urlopen(req, timeout=10) as r:
            import json

            payload = json.load(r)
        results = payload.get("quoteResponse", {}).get("result", [])
        if not results:
            return None
        q = results[0]
        out = {
            "pe": q.get("trailingPE"),
            "pb": q.get("priceToBook"),
            "source": "yahoo",
        }
        if all(v is None for v in (out["pe"], out["pb"])):
            return None
        return out
    except Exception:
        return None


# ---------------------------------------------------------------------------
# Scoring (0-100)
# ---------------------------------------------------------------------------


def _clip01(v: float) -> float:
    return float(np.clip(v, 0.0, 1.0))


def value_score(fundamentals: Optional[Dict]) -> Optional[Dict]:
    """Value 0-100 from P/E + EV/EBITDA + P/B (lower is better)."""
    if not fundamentals:
        return None
    pe, ev, pb = (
        fundamentals.get("pe"),
        fundamentals.get("ev_ebitda"),
        fundamentals.get("pb"),
    )
    parts, names = [], []
    # Band anchors: PE 8 = excellent value ... 40 = expensive.
    if pe is not None and pe > 0:
        parts.append(100.0 - (min(max(pe, 8.0), 40.0) - 8.0) / 32.0 * 100.0)
        names.append("P/E")
    if ev is not None and ev > 0:
        parts.append(100.0 - (min(max(ev, 4.0), 25.0) - 4.0) / 21.0 * 100.0)
        names.append("EV/EBITDA")
    if pb is not None and pb > 0:
        parts.append(100.0 - (min(max(pb, 0.5), 12.0) - 0.5) / 11.5 * 100.0)
        names.append("P/B")
    if not parts:
        return None
    return {"score": round(float(np.mean(parts)), 1), "parts": names}


def quality_score(fundamentals: Optional[Dict]) -> Optional[Dict]:
    """Quality 0-100 from ROE + low Debt/Equity (higher is better)."""
    if not fundamentals:
        return None
    roe, dte = (fundamentals.get("roe_pct"), fundamentals.get("debt_to_equity"))
    parts, names = [], []
    # ROE: 5% weak ... 35%+ excellent.
    if roe is not None:
        parts.append((min(max(roe, 5.0), 35.0) - 5.0) / 30.0 * 100.0)
        names.append("ROE")
    # Debt/Equity: >1.5 heavily levered ... 0 clean.
    if dte is not None and dte >= 0:
        parts.append(100.0 - min(max(dte, 0.0), 1.5) / 1.5 * 100.0)
        names.append("Debt/Equity")
    if not parts:
        return None
    return {"score": round(float(np.mean(parts)), 1), "parts": names}


def momentum_score(df: pd.DataFrame, fundamentals: Optional[Dict] = None) -> Dict:
    """
    Momentum 0-100: blended price momentum (1/3/6/12m returns + RSI,
    always available) and, when provided, earnings surprise + analyst
    revisions. Weighted 70% price / 30% fundamentals.
    """
    close = df["close"].astype(float)
    last = close.iloc[-1]
    rets = {}
    for n in PRICE_RETURN_LOOKBACKS:
        if len(close) > n and close.iloc[-1 - n] > 0:
            rets[n] = (last / close.iloc[-1 - n] - 1.0) * 100.0
    if rets:
        # Returns across horizons -> momentum, symmetric around 50:
        # +30% 12m move maps near 90, -30% near 10 (unlike a clip at 0,
        # which would leave a crashing stock reading identical to a flat
        # one - momentum must rank the full spectrum).
        base = 50.0
        for _, r in rets.items():
            base += np.clip(r / 30.0, -1.0, 1.0) * 40.0 / len(rets)
        price_comp = float(np.clip(base, 0.0, 100.0))
    else:
        price_comp = 50.0

    rsi = 50.0
    if "rsi_14" in df.columns:
        v = df["rsi_14"].iloc[-1]
        if not np.isnan(v):
            rsi = float(v)
    # RSI tilts the score slightly (mean-reversion aware, 20% weight).
    price_mom = 0.8 * price_comp + 0.2 * rsi

    names = ["price"]
    if fundamentals:
        est = fundamentals.get("earnings_surprise_pct")
        rev = fundamentals.get("analyst_revisions")
        parts = []
        if est is not None:
            parts.append(_clip01((est + 10.0) / 20.0) * 100.0)
            names.append("earnings surprise")
        if rev is not None:
            # analyst_revisions: +N net upgrades (could be float count) or
            # a -1..+1 style score; treat magnitude as direction-scaled.
            parts.append(_clip01((float(rev) + 1.0) / 2.0) * 100.0)
            names.append("analyst revisions")
        if parts:
            fund_comp = float(np.mean(parts))
            score = 0.7 * price_mom + 0.3 * fund_comp
            names.insert(0, "fundamentals")
            return {"score": round(score, 1), "parts": names}
    return {"score": round(price_mom, 1), "parts": names}


# ---------------------------------------------------------------------------
# Composite
# ---------------------------------------------------------------------------


def factor_scores(
    symbol: str, df: pd.DataFrame, fundamentals: Optional[Dict] = None
) -> Dict:
    """
    Full factor-model read for one symbol:

    ``{symbol, value, quality, momentum, composite, factors, sources}``
    where ``composite`` weights Value 25% / Quality 25% / Momentum 50%
    over whatever factors are available (renormalised when one is missing).
    """
    val = value_score(fundamentals)
    qual = quality_score(fundamentals)
    mom = momentum_score(df, fundamentals)

    avail = []
    weights = {"value": 0.25, "quality": 0.25, "momentum": 0.50}
    if val:
        avail.append(("value", val["score"], weights["value"]))
    if qual:
        avail.append(("quality", qual["score"], weights["quality"]))
    avail.append(("momentum", mom["score"], weights["momentum"]))  # always

    wsum = sum(w for _, _, w in avail)
    composite = sum(s * w for _, s, w in avail) / wsum if wsum else 0.0

    sources = []
    if val:
        sources.append("value:fundamentals")
    else:
        sources.append("value:none")
    if qual:
        sources.append("quality:fundamentals")
    else:
        sources.append("quality:none")
    sources.append("momentum:price")

    return {
        "symbol": symbol,
        "value": val["score"] if val else None,
        "quality": qual["score"] if qual else None,
        "momentum": mom["score"],
        "composite": round(composite, 1),
        "factors": {
            "value": val["parts"] if val else [],
            "quality": qual["parts"] if qual else [],
            "momentum": mom["parts"],
        },
        "sources": sources,
        "fundamentals_source": (fundamentals or {}).get("source", "none"),
    }


if __name__ == "__main__":
    print("NexusQuant Fundamental Factor Model ready.")
