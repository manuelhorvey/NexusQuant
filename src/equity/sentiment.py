"""
NexusQuant - News & Social Sentiment (institutional spec #9, sentiment half).

Two scores in ``[-1, +1]`` (positive = bullish sentiment):

* **News** - fetched from the public Yahoo Finance search endpoint (no auth,
  cached under ``data/raw/sentiment/``) and scored with a financial lexicon
  over headline + summary text.
* **Social** - best-effort; StockTwits' public API now returns 403 without
  an API key, so this is normally ``None`` (documented degradation). The
  hook is kept so a keyed provider can be wired in without touching the
  scoring layer.

Everything is graceful: offline / blocked sources yield ``None`` scores and
the caller sees ``available=False`` rather than an exception.
"""

from __future__ import annotations

import json
import re
import sys
import urllib.request
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np

sys.path.append(str(Path(__file__).parent.parent.parent))

# Generic cache dir - the module serves FX/metals/indices as well as
# equities, so the cache lives outside any asset-class group folder.
SENTIMENT_CACHE = "data/raw/sentiment"

POSITIVE = {
    "beat",
    "beats",
    "surge",
    "surges",
    "surged",
    "rally",
    "rallies",
    "rallied",
    "record",
    "strong",
    "stronger",
    "strength",
    "growth",
    "growing",
    "upgrade",
    "upgraded",
    "upgrades",
    "raise",
    "raises",
    "raised",
    "jump",
    "jumps",
    "jumped",
    "gain",
    "gains",
    "gained",
    "outperform",
    "outperforms",
    "outperformed",
    "bullish",
    "buy",
    "profit",
    "profits",
    "profitability",
    "expand",
    "expands",
    "expanding",
    "exceed",
    "exceeds",
    "exceeded",
    "accelerate",
    "accelerating",
    "momentum",
    "optimistic",
    "recovery",
    "recovers",
    "opportunity",
    "breakout",
    "breakthrough",
    "innovate",
    "innovation",
    "dominant",
    "leader",
    "leading",
    "impressive",
    "soar",
    "soars",
    "soared",
    "skyrocket",
    "positive",
    "boost",
    "boosts",
    "boosted",
    "demand",
}

NEGATIVE = {
    "miss",
    "misses",
    "missed",
    "drop",
    "drops",
    "dropped",
    "decline",
    "declines",
    "declined",
    "fall",
    "falls",
    "fell",
    "plunge",
    "plunges",
    "plunged",
    "crash",
    "crashes",
    "crashed",
    "downgrade",
    "downgraded",
    "downgrades",
    "cut",
    "cuts",
    "cutting",
    "weak",
    "weaker",
    "weakness",
    "lawsuit",
    "lawsuits",
    "investigation",
    "probe",
    "fraud",
    "scandal",
    "layoff",
    "layoffs",
    "recession",
    "recessions",
    "warning",
    "warns",
    "warned",
    "bearish",
    "sell",
    "selling",
    "slump",
    "slumps",
    "slumped",
    "loss",
    "losses",
    "losing",
    "underperform",
    "underperforms",
    "risk",
    "risks",
    "volatility",
    "uncertainty",
    "headwind",
    "headwinds",
    "shrink",
    "shrinking",
    "delay",
    "delays",
    "delayed",
    "recall",
    "recalls",
    "guilty",
    "criminal",
    "fine",
    "penalty",
    "debt",
    "bankruptcy",
    "insolvent",
    "trouble",
    "struggle",
    "struggles",
    "struggling",
    "disappoint",
    "disappointing",
    "negative",
    "concern",
    "concerns",
}

_NEG_WORD = re.compile(r"\b[a-z'-]+\b")


def _tokenize(text: str) -> List[str]:
    return _NEG_WORD.findall(text.lower())


def lexicon_score(texts: List[str]) -> float:
    """Financial-lexicon score in ``[-1, +1]`` over a list of texts.

    Raw (pos-neg)/(pos+neg) with a **neutral band**: a tie or near-tie
    (|raw| < 0.15) reads 0.0, because headline lexicons otherwise flip on
    a single word. Sample-size dampening lives in ``news_sentiment``
    (keyed on article count, where the anecdote-vs-signal line lives).
    """
    pos = neg = 0
    for t in texts:
        for w in _tokenize(t):
            if w in POSITIVE:
                pos += 1
            elif w in NEGATIVE:
                neg += 1
    total = pos + neg
    if total == 0:
        return 0.0
    raw = float(np.clip((pos - neg) / total, -1.0, 1.0))
    if abs(raw) < 0.15:
        return 0.0
    return round(raw, 3)


def _article_relevance(symbol: str, item: Dict) -> float:
    """How relevant an article is to ``symbol``: title mentions the ticker
    (1.0) > the ticker is the article's primary subject (0.8) > a secondary
    mention (0.3). Prevents generic market stories (which Yahoo lists under
    every queried symbol) from dominating the sentiment read."""
    title = item.get("title") or ""
    tickers = [str(t).upper() for t in (item.get("relatedTickers") or [])]
    sym = symbol.upper()
    if sym in title.upper():
        return 1.0
    if tickers and tickers[0] == sym:
        return 0.8
    return 0.3


def _fetch_yahoo_news(symbol: str, news_count: int = 20) -> Optional[tuple]:
    """Fetch news -> ``(articles, n_articles)`` or None, where ``articles``
    is ``[{texts, weight}]`` (title + summary texts, relevance weight)."""
    url = (
        f"https://query1.finance.yahoo.com/v1/finance/search?"
        f"q={symbol}&quotesCount=0&newsCount={news_count}"
    )
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
    try:
        with urllib.request.urlopen(req, timeout=12) as r:
            payload = json.load(r)
    except Exception:
        return None
    news = payload.get("news", [])
    if not news:
        return None
    articles = []
    for item in news[:news_count]:
        title = item.get("title") or ""
        summary = item.get("summary") or ""
        texts = [title]
        if summary:
            texts.append(summary)
        articles.append({"texts": texts, "weight": _article_relevance(symbol, item)})
    return articles, len(articles)


def _weighted_score(articles: List[Dict]) -> float:
    """Lexicon score in ``[-1, +1]`` weighted by per-article relevance,
    with the same neutral band as ``lexicon_score``."""
    pos = neg = 0.0
    for a in articles:
        w = float(a.get("weight", 1.0))
        for t in a.get("texts", []):
            for word in _tokenize(t):
                if word in POSITIVE:
                    pos += w
                elif word in NEGATIVE:
                    neg += w
    total = pos + neg
    if total == 0:
        return 0.0
    raw = float(np.clip((pos - neg) / total, -1.0, 1.0))
    if abs(raw) < 0.15:
        return 0.0
    return round(raw, 3)


def _cache_path(symbol: str) -> Path:
    return Path(SENTIMENT_CACHE) / f"{symbol.upper()}_news.json"


def _load_cache(symbol: str, max_age_days: float = 1.0) -> Optional[tuple]:
    """``(articles, n_articles)`` from the 1-day cache, else None."""
    p = _cache_path(symbol)
    if not p.exists():
        return None
    import time

    age = (time.time() - p.stat().st_mtime) / 86400.0
    if age > max_age_days:
        return None
    try:
        data = json.loads(p.read_text())
        articles = data.get("articles")
        n_articles = data.get("n_articles")
        # Pre-v3 cache entries (flat texts, no weights) cannot be scored
        # honestly; treat as a miss so the next call refetches.
        if not articles or n_articles is None:
            return None
        return (articles, int(n_articles))
    except Exception:
        return None


def _save_cache(symbol: str, articles: List[Dict], n_articles: int) -> None:
    try:
        p = _cache_path(symbol)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(
            json.dumps(
                {
                    "symbol": symbol.upper(),
                    "articles": articles,
                    "n_articles": n_articles,
                }
            )
        )
    except Exception:
        pass


def news_sentiment(symbol: str, fetch: bool = True, max_age_days: float = 1.0) -> Dict:
    """News sentiment ``{score, n_articles, available, source}`` in [-1,+1]."""
    cached = _load_cache(symbol, max_age_days)
    if cached is not None:
        articles, n_articles = cached
    else:
        fetched = _fetch_yahoo_news(symbol) if fetch else None
        if fetched is None:
            return {
                "score": None,
                "n_articles": 0,
                "available": False,
                "source": "none",
            }
        articles, n_articles = fetched
        _save_cache(symbol, articles, n_articles)

    raw = _weighted_score(articles)
    # Confidence from the RELEVANT article count (weight >= 0.8, i.e.
    # actually about the symbol), not the total: Yahoo lists generic market
    # stories under every queried ticker, so 10 headlines where only 2 are
    # symbol-specific is a 0.4-confidence read (2/5), not a full one.
    # A handful of relevant headlines is anecdote, not signal.
    relevant = sum(1 for a in articles if float(a.get("weight", 0)) >= 0.8)
    confidence = min(1.0, relevant / 5.0)
    return {
        "score": round(raw * confidence, 3),
        "n_articles": n_articles,
        "relevant": relevant,
        "available": True,
        "source": "yahoo-news",
    }


def social_sentiment(symbol: str) -> Dict:
    """Social sentiment - normally unavailable (providers require keys)."""
    return {
        "score": None,
        "available": False,
        "source": "none",
        "note": "social providers (StockTwits etc.) now require API keys",
    }


def sentiment_report(symbol: str, fetch_news: bool = True) -> Dict:
    """Combined sentiment read: ``{news, social, composite, available}``."""
    news = news_sentiment(symbol, fetch=fetch_news)
    social = social_sentiment(symbol)

    scores = [s for s in (news.get("score"), social.get("score")) if s is not None]
    composite = float(np.mean(scores)) if scores else None
    return {
        "symbol": symbol,
        "news": news,
        "social": social,
        "composite": round(composite, 3) if composite is not None else None,
        "available": news.get("available") or social.get("available"),
    }


if __name__ == "__main__":
    print("NexusQuant News & Social Sentiment module ready.")
