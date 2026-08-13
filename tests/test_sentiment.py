"""Tests for News & Social Sentiment (src/equity/sentiment.py)."""

from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent))

from src.equity.sentiment import (
    POSITIVE,
    NEGATIVE,
    _article_relevance,
    _tokenize,
    _weighted_score,
    lexicon_score,
    news_sentiment,
    sentiment_report,
    social_sentiment,
)
from src.equity import sentiment as sent_mod


class TestLexicon(unittest.TestCase):
    def test_bullish_text(self):
        score = lexicon_score(
            [
                "Apple beats expectations, rallies to record high, "
                "upgrades and strong growth"
            ]
        )
        self.assertGreater(score, 0.0)

    def test_bearish_text(self):
        score = lexicon_score(
            ["Company misses estimates, slumps, downgraded, layoffs, recession warning"]
        )
        self.assertLess(score, 0.0)

    def test_neutral_text(self):
        self.assertEqual(lexicon_score(["the quarterly report was filed"]), 0.0)

    def test_bounds(self):
        for t in (
            ["great", "great", "great", "great"],
            ["miss", "miss", "miss", "miss"],
        ):
            s = lexicon_score(t)
            self.assertLessEqual(s, 1.0)
            self.assertGreaterEqual(s, -1.0)

    def test_tokenize_lowercases(self):
        self.assertIn("beat", _tokenize("BEAT, expectations!"))
        self.assertNotIn("", _tokenize("a b"))


class TestNewsSentiment(unittest.TestCase):
    def test_cache_path_uses_symbol(self):
        with tempfile.TemporaryDirectory() as tmp:
            old = sent_mod.SENTIMENT_CACHE
            sent_mod.SENTIMENT_CACHE = tmp
            try:
                Path(tmp).mkdir(exist_ok=True)
                Path(tmp, "AAPL_news.json").write_text(
                    json.dumps(
                        {
                            "symbol": "AAPL",
                            "articles": [
                                {"texts": ["beats expectations"], "weight": 1.0}
                            ],
                            "n_articles": 10,
                        }
                    )
                )
                out = news_sentiment("AAPL", fetch=False)
                self.assertTrue(out["available"])
                self.assertGreater(out["score"], 0.0)
                self.assertEqual(out["n_articles"], 10)
            finally:
                sent_mod.SENTIMENT_CACHE = old

    def test_sample_size_dampening(self):
        # 1 article vs 10 articles: same bullish text, but the small sample
        # is pulled toward 0 (anecdote vs signal).
        texts = ["beats expectations and rallies on strong growth"]
        raw = lexicon_score(texts)
        with tempfile.TemporaryDirectory() as tmp:
            old = sent_mod.SENTIMENT_CACHE
            sent_mod.SENTIMENT_CACHE = tmp
            try:
                Path(tmp).mkdir(exist_ok=True)
                for n, label in ((1, "few"), (10, "many")):
                    p = Path(tmp, f"S{n}_news.json")
                    p.write_text(
                        json.dumps(
                            {
                                "symbol": f"S{n}",
                                "articles": [{"texts": texts, "weight": 1.0}] * n,
                                "n_articles": n,
                            }
                        )
                    )
                    out = news_sentiment(f"S{n}", fetch=False)
                    self.assertTrue(out["available"])
                    if label == "few":
                        self.assertLess(out["score"], raw)
                    else:
                        self.assertEqual(out["score"], raw)
            finally:
                sent_mod.SENTIMENT_CACHE = old

    def test_neutral_band_zero(self):
        # an exact tie reads 0.0, not a coin-flip sign
        self.assertEqual(lexicon_score(["growth and decline"]), 0.0)
        # and a near-tie (|raw| < 0.15) also reads 0.0
        score = lexicon_score(["growth decline"])  # 1 vs 1
        self.assertEqual(score, 0.0)

    def test_relevance_weighting_damps_generic_news(self):
        # A generic bullish headline (secondary mention) contributes far
        # less than one actually about the symbol.
        articles = [
            {"texts": ["beats expectations, record rally"], "weight": 1.0},
            {
                "texts": ["strong growth, booming, best quarter"],
                "weight": 0.3,
            },  # generic market story
        ]
        full = _weighted_score(articles)
        generic_only = _weighted_score([articles[1]])
        self.assertEqual(full, 1.0)
        self.assertEqual(generic_only, 1.0)
        # The key property is _article_relevance returns the ladder.
        self.assertEqual(
            _article_relevance(
                "AAPL", {"title": "AAPL beats estimates", "relatedTickers": ["AAPL"]}
            ),
            1.0,
        )
        self.assertEqual(
            _article_relevance(
                "AAPL",
                {"title": "big tech roundup", "relatedTickers": ["MSFT", "AAPL"]},
            ),
            0.3,
        )
        self.assertEqual(
            _article_relevance(
                "AAPL", {"title": "apple roundup", "relatedTickers": ["AAPL", "MSFT"]}
            ),
            0.8,
        )

    def test_missing_cache_graceful(self):
        with tempfile.TemporaryDirectory() as tmp:
            old = sent_mod.SENTIMENT_CACHE
            sent_mod.SENTIMENT_CACHE = tmp
            try:
                # fetch=False + empty cache dir -> unavailable, no crash
                out = news_sentiment("ZZZZZZ", fetch=False)
                self.assertFalse(out["available"])
                self.assertIsNone(out["score"])
            finally:
                sent_mod.SENTIMENT_CACHE = old

    def test_cache_respects_age(self):
        import time

        with tempfile.TemporaryDirectory() as tmp:
            old = sent_mod.SENTIMENT_CACHE
            sent_mod.SENTIMENT_CACHE = tmp
            try:
                Path(tmp).mkdir(exist_ok=True)
                p = Path(tmp, "OLD_news.json")
                p.write_text(json.dumps({"symbol": "OLD", "texts": ["beats"]}))
                # backdate the mtime 3 days
                t = time.time() - 3 * 86400
                import os

                os.utime(p, (t, t))
                out = news_sentiment("OLD", fetch=False, max_age_days=1.0)
                self.assertFalse(out["available"])
            finally:
                sent_mod.SENTIMENT_CACHE = old


class TestSocialSentiment(unittest.TestCase):
    def test_graceful_unavailable(self):
        out = social_sentiment("AAPL")
        self.assertFalse(out["available"])
        self.assertIsNone(out["score"])


class TestSentimentReport(unittest.TestCase):
    def test_composite_only_from_available(self):
        with tempfile.TemporaryDirectory() as tmp:
            old = sent_mod.SENTIMENT_CACHE
            sent_mod.SENTIMENT_CACHE = tmp
            try:
                Path(tmp).mkdir(exist_ok=True)
                Path(tmp, "AAPL_news.json").write_text(
                    json.dumps(
                        {
                            "symbol": "AAPL",
                            "articles": [
                                {"texts": ["beats and rallies"], "weight": 1.0}
                            ],
                            "n_articles": 10,
                        }
                    )
                )
                rep = sentiment_report("AAPL", fetch_news=False)
                self.assertTrue(rep["available"])
                self.assertIsNotNone(rep["composite"])
                self.assertGreater(rep["composite"], 0.0)
                # social absent -> composite == news score
                self.assertEqual(rep["composite"], round(rep["news"]["score"], 3))
            finally:
                sent_mod.SENTIMENT_CACHE = old

    def test_offline_report(self):
        with tempfile.TemporaryDirectory() as tmp:
            old = sent_mod.SENTIMENT_CACHE
            sent_mod.SENTIMENT_CACHE = tmp
            try:
                rep = sentiment_report("ZZZZZZ", fetch_news=False)
                self.assertFalse(rep["available"])
                self.assertIsNone(rep["composite"])
            finally:
                sent_mod.SENTIMENT_CACHE = old

    def test_lexicons_are_nonempty(self):
        self.assertTrue(POSITIVE)
        self.assertTrue(NEGATIVE)


if __name__ == "__main__":
    unittest.main()
