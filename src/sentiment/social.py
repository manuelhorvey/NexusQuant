"""
NexusQuant - Social sentiment sources.

Graceful by design: public social feeds (StockTwits, Reddit) now require
API keys, so ``social_read`` returns ``None`` unless a keyed provider is
configured. Wired providers (documented hooks, all optional):

* ``STOCKTWITS_API_KEY`` - StockTwits API (symbol mentions + sentiment)
* ``REDDIT_*`` (praw) - r/investing etc. mention counts

Both stay behind the ``available=False`` contract so the aggregator never
has to branch on provider presence.
"""

from __future__ import annotations

import os
from typing import Dict

from src.equity.sentiment import social_sentiment  # noqa: F401  (re-export)


def social_read(symbol: str) -> Dict:
    """Social sentiment read (None/False unless a keyed provider exists)."""
    key = os.environ.get("STOCKTWITS_API_KEY")
    if key:
        try:
            from src.equity.sentiment import social_sentiment as _impl

            return _impl(symbol)
        except Exception:
            pass
    return {
        "score": None,
        "available": False,
        "source": "none",
        "note": "social providers (StockTwits etc.) now require API "
        "keys - set STOCKTWITS_API_KEY to enable",
    }


if __name__ == "__main__":
    print("NexusQuant sentiment.social ready.")
