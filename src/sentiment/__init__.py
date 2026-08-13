"""
NexusQuant - News & Social Sentiment package (institutional spec #9).

Thin, dependency-light layer on top of ``src.equity.sentiment`` (the
Yahoo-news lexicon scorer, which serves every symbol class) plus a
weighted composite aggregator:

    from src.sentiment.aggregator import composite_sentiment
    read = composite_sentiment("XAUUSD")   # {-1..+1} or None

Structure mirrors the plan: ``news.py`` (news sources), ``social.py``
(graceful - providers require API keys), ``aggregator.py`` (weighted
composite). All providers degrade gracefully offline.
"""
