"""
NexusQuant - Equity universe management (S&P 500).

Provides the S&P 500 ticker list for backfills / scans / fundamentals
refresh. ``sp500_tickers()`` tries the live Wikipedia table first, then a
curated static fallback of the largest names, so it always returns
something usable.

Usage:
    from src.equity.universe import sp500_tickers
    tickers = sp500_tickers()
"""

from __future__ import annotations

from typing import List

# Static fallback: the largest / most-liquid S&P 500 names (used when the
# live Wikipedia table cannot be read, e.g. offline).
_STATIC_FALLBACK = [
    "AAPL",
    "MSFT",
    "NVDA",
    "AMZN",
    "GOOGL",
    "META",
    "BRK-B",
    "LLY",
    "AVGO",
    "TSLA",
    "JPM",
    "WMT",
    "V",
    "UNH",
    "XOM",
    "MA",
    "COST",
    "PG",
    "HD",
    "JNJ",
    "ORCL",
    "ABBV",
    "NFLX",
    "BAC",
    "CRM",
    "KO",
    "CVX",
    "MRK",
    "AMD",
    "PEP",
    "ADBE",
    "TMO",
    "WFC",
    "LIN",
    "CSCO",
    "ACN",
    "MCD",
    "ABT",
    "GE",
    "IBM",
    "TXN",
    "CAT",
    "PM",
    "QCOM",
    "INTU",
    "AMGN",
    "VZ",
    "DHR",
    "NEE",
    "DIS",
    "CMCSA",
    "RTX",
    "SPGI",
    "GS",
    "BKNG",
    "AXP",
    "UBER",
    "MS",
    "NOW",
    "AMAT",
    "ISRG",
    "PGR",
    "BLK",
    "LOW",
    "SYK",
    "ETN",
    "TJX",
    "HON",
    "ELV",
    "COP",
    "BSX",
    "UNP",
    "SCHW",
    "C",
    "PLD",
    "ADP",
    "GILD",
    "LMT",
    "DE",
    "MDT",
    "MU",
    "TMUS",
    "AMT",
    "CB",
    "MMC",
    "PYPL",
    "KLAC",
    "BMY",
    "ADI",
    "CI",
    "MDLZ",
    "UPS",
    "SHW",
    "NKE",
    "SO",
    "TT",
    "EQIX",
    "DUK",
    "SBUX",
    "ICE",
    "PNC",
    "TGT",
    "MO",
    "AON",
    "WELL",
    "REGN",
    "EW",
    "CME",
    "FDX",
    "ITW",
    "USB",
    "APH",
    "ZTS",
    "APH",
    "MMM",
    "APD",
]


def sp500_tickers() -> List[str]:
    """
    The S&P 500 constituent list.

    Tries the live Wikipedia table (constituents change over time), then
    falls back to the static list. Returns uppercase tickers, deduplicated,
    in table order.
    """
    try:
        import pandas as pd

        tables = pd.read_html(
            "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"
        )
        df = tables[0]
        col = "Symbol" if "Symbol" in df.columns else df.columns[0]
        tickers = [str(t).upper().strip() for t in df[col].tolist()]
        tickers = [t for t in tickers if t and t != "NAN"]
        if len(tickers) >= 100:
            return _dedup(tickers)
    except Exception:
        pass
    return _dedup(_STATIC_FALLBACK)


def _dedup(tickers: List[str]) -> List[str]:
    seen = set()
    out = []
    for t in tickers:
        if t not in seen:
            seen.add(t)
            out.append(t)
    return out


if __name__ == "__main__":
    print(f"NexusQuant equity universe ready: {len(sp500_tickers())} tickers")
