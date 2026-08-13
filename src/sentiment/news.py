"""
NexusQuant - News sentiment sources.

Primary source: the public Yahoo Finance search endpoint (no auth, 1-day
cache under ``data/raw/sentiment/``), scored with a financial lexicon -
implemented in ``src.equity.sentiment.news_sentiment`` and re-exported
here so the whole sentiment package shares one scorer.

Hooks for keyed providers (NewsAPI / Alpha Vantage) can be added here
without touching the scoring layer - they just need to return the same
``{score, n_articles, available, source}`` shape.
"""

from __future__ import annotations

from typing import Dict

from src.equity.sentiment import news_sentiment  # noqa: F401  (re-export)


def news_read(symbol: str, fetch: bool = True, max_age_days: float = 1.0) -> Dict:
    """News sentiment read in ``[-1, +1]`` (None when unavailable)."""
    return news_sentiment(symbol, fetch=fetch, max_age_days=max_age_days)


if __name__ == "__main__":
    print("NexusQuant sentiment.news ready.")
