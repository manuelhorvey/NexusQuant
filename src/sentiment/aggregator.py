"""
NexusQuant - Composite sentiment aggregator.

Blends news (60%) and social (40%) into one daily ``[-1, +1]`` score per
symbol, renormalising when one leg is unavailable so the available signal
is never silently halved:

    composite = (w_news * news + w_social * social) / (w_news + w_social)

when both exist; the available single leg alone otherwise; ``None`` when
nothing is available.
"""

from __future__ import annotations

from typing import Dict, Optional

from src.sentiment.news import news_read
from src.sentiment.social import social_read

NEWS_WEIGHT = 0.6
SOCIAL_WEIGHT = 0.4


def composite_sentiment(
    symbol: str,
    news_weight: float = NEWS_WEIGHT,
    social_weight: float = SOCIAL_WEIGHT,
    fetch_news: bool = False,
) -> Dict:
    """
    Weighted composite sentiment for ``symbol``.

    ``fetch_news=False`` (default) reads the 1-day news cache only - safe
    inside universe scans / reports that must never fire network calls.

    Returns ``{symbol, news, social, composite, available, weights}``.
    """
    news = news_read(symbol, fetch=fetch_news)
    social = social_read(symbol)

    parts = []
    weights = []
    if news.get("score") is not None:
        parts.append(float(news["score"]))
        weights.append(news_weight)
    if social.get("score") is not None:
        parts.append(float(social["score"]))
        weights.append(social_weight)

    wsum = sum(weights)
    composite: Optional[float] = None
    if wsum > 0:
        composite = round(
            sum(p * w for p, w in zip(parts, weights, strict=True)) / wsum, 3
        )

    return {
        "symbol": symbol,
        "news": news,
        "social": social,
        "composite": composite,
        "available": news.get("available") or social.get("available"),
        "weights": {"news": news_weight, "social": social_weight},
    }


if __name__ == "__main__":
    print("NexusQuant sentiment aggregator ready.")
