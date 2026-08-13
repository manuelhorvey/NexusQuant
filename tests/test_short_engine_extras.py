"""
Tests for the gap-closing modules: HMM regime, the yfinance fundamentals
provider, the sentiment aggregator package, and the live short filter.
"""

from __future__ import annotations

import sys
import unittest
from pathlib import Path
from unittest.mock import patch

import pandas as pd

sys.path.append(str(Path(__file__).parent.parent))

from tests.test_dip import make_uptrend_dip, with_indicators
from src.features.regime import detect_regime, detect_regime_hmm
from src.equity.data_provider import fetch_symbol_fundamentals, refresh_fundamentals
from src.equity.universe import sp500_tickers
from src.sentiment.aggregator import composite_sentiment


def _prep(df: pd.DataFrame) -> pd.DataFrame:
    return detect_regime(with_indicators(df))


class HmmRegimeTest(unittest.TestCase):
    def test_hmm_column_added(self):
        df = _prep(make_uptrend_dip())
        out = detect_regime_hmm(df)
        self.assertIn("regime_hmm", out.columns)
        labels = set(out["regime_hmm"].dropna().unique())
        self.assertTrue(
            labels.issubset(
                {"Bull Trend", "Bear Trend", "Range / Chop", "High Volatility"}
            )
        )

    def test_hmm_graceful_on_short_frame(self):
        # Too little history -> deterministic labels (no crash).
        df = _prep(make_uptrend_dip()).head(80)
        out = detect_regime_hmm(df)
        self.assertEqual(len(out), 80)
        self.assertIn("regime_hmm", out.columns)


class FundamentalsProviderTest(unittest.TestCase):
    def test_fetch_graceful_offline(self):
        # No network / no yfinance -> None, never a crash.
        try:
            rec = fetch_symbol_fundamentals("ZZZZ-INVALID")
        except Exception as exc:  # pragma: no cover
            self.fail(f"fetch_symbol_fundamentals raised: {exc}")
        self.assertTrue(rec is None or isinstance(rec, dict))

    def test_refresh_writes_csv(self, tmp_path=None):
        def _fake_fetch(symbol: str):
            if symbol == "FAILSYM":
                return None
            return {
                "symbol": symbol,
                "pe": 15.0,
                "ev_ebitda": 10.0,
                "pb": 2.0,
                "roe_pct": 20.0,
                "debt_to_equity": 0.4,
                "earnings_surprise_pct": 3.0,
                "analyst_revisions": 0.5,
                "source": "yahoo",
            }

        with (
            patch(
                "src.equity.data_provider.fetch_symbol_fundamentals",
                side_effect=_fake_fetch,
            ),
            patch(
                "src.equity.data_provider._UNIVERSE_CSV",
                new=Path("/tmp/_nx_test_universe.csv"),
            ),
            patch("src.equity.data_provider.FUNDAMENTALS_DIR", new="/tmp"),
        ):
            res = refresh_fundamentals(["TEST", "FAILSYM"], delay=0.0)
            self.assertEqual(res["fetched"], 1)
            self.assertEqual(res["failed"], ["FAILSYM"])
            self.assertTrue(Path(res["path"]).exists())
        Path("/tmp/_nx_test_universe.csv").unlink(missing_ok=True)

    def test_sp500_tickers_nonempty(self):
        tickers = sp500_tickers()
        self.assertGreater(len(tickers), 50)
        self.assertEqual(tickers, [t.upper() for t in tickers])


class SentimentAggregatorTest(unittest.TestCase):
    def test_aggregator_renormalises_single_leg(self):
        with patch(
            "src.sentiment.aggregator.news_read",
            return_value={"score": 0.5, "available": True, "n_articles": 5},
        ):
            read = composite_sentiment("TEST", fetch_news=True)
            self.assertAlmostEqual(read["composite"], 0.5)
            self.assertTrue(read["available"])

    def test_aggregator_both_legs_weighted(self):
        with (
            patch(
                "src.sentiment.aggregator.news_read",
                return_value={"score": 0.6, "available": True, "n_articles": 5},
            ),
            patch(
                "src.sentiment.aggregator.social_read",
                return_value={"score": 0.2, "available": True},
            ),
        ):
            read = composite_sentiment("TEST", fetch_news=True)
            # 0.6 * 0.6 + 0.4 * 0.2 = 0.44
            self.assertAlmostEqual(read["composite"], 0.44, places=3)

    def test_aggregator_none_when_nothing_available(self):
        with (
            patch(
                "src.sentiment.aggregator.news_read",
                return_value={"score": None, "available": False, "n_articles": 0},
            ),
            patch(
                "src.sentiment.aggregator.social_read",
                return_value={"score": None, "available": False},
            ),
        ):
            read = composite_sentiment("TEST", fetch_news=True)
            self.assertIsNone(read["composite"])
            self.assertFalse(read["available"])


class LiveShortFilterTest(unittest.TestCase):
    def _table(self):
        return pd.DataFrame(
            [
                {
                    "symbol": "USDJPY",
                    "rally_score": 7,
                    "rally_confirmed": "Yes",
                    "rally_stage": "Confirmed",
                    "bias_score": -3,
                    "macro_gate": "PASS",
                    "ml_prob": 30.0,
                    "short_best_rr": 2.5,
                    "short_entry_zone": "150.0-150.5",
                    "short_invalidation": 151.0,
                    "support": 148.0,
                },
                {
                    "symbol": "EURUSD",
                    "rally_score": 2,
                    "rally_confirmed": "No",
                    "rally_stage": "No Downtrend",
                    "bias_score": 1,
                    "macro_gate": "PASS",
                    "ml_prob": 55.0,
                    "short_best_rr": 1.0,
                    "short_entry_zone": None,
                    "short_invalidation": None,
                    "support": None,
                },
            ]
        )

    def test_filter_keeps_only_high_conviction_shorts(self):
        from src.live.signals import filter_short_signals

        f = filter_short_signals(
            self._table(), min_rally_score=5, require_confirmed=True
        )
        self.assertEqual(f["symbol"].tolist(), ["USDJPY"])

    def test_filter_ml_threshold_inverts_for_shorts(self):
        from src.live.signals import filter_short_signals

        f = filter_short_signals(
            self._table(), min_ml_prob=40.0, require_confirmed=False
        )
        # USDJPY has ml 30 <= 60 (100-40) -> kept; EURUSD 55 > 60 -> dropped.
        self.assertEqual(f["symbol"].tolist(), ["USDJPY"])

    def test_short_rr_series_prefers_ladder(self):
        from src.live.signals import _short_rr_series

        rr = _short_rr_series(self._table())
        self.assertEqual(rr.iloc[0], 2.5)


if __name__ == "__main__":
    unittest.main()
