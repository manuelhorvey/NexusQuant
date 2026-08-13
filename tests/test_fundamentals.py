"""Tests for the Fundamental Factor Model (src/equity/fundamentals.py)."""

from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.append(str(Path(__file__).parent.parent))

from src.equity.fundamentals import (
    factor_scores,
    load_fundamentals,
    momentum_score,
    quality_score,
    value_score,
)


def make_frame(n: int = 300, drift: float = 0.05) -> pd.DataFrame:
    rng = np.random.default_rng(1)
    closes = 100 * np.cumprod(1 + rng.normal(drift / 250, 0.01, n))
    return pd.DataFrame(
        {
            "close": closes,
            "high": closes * 1.005,
            "low": closes * 0.995,
            "open": closes,
            "volume": 1_000_000,
        }
    )


class TestValueScore(unittest.TestCase):
    def test_cheaper_is_better(self):
        cheap = value_score({"pe": 10.0, "ev_ebitda": 6.0, "pb": 1.0})
        rich = value_score({"pe": 35.0, "ev_ebitda": 20.0, "pb": 10.0})
        self.assertIsNotNone(cheap)
        self.assertGreater(cheap["score"], rich["score"])

    def test_none_when_no_fundamentals(self):
        self.assertIsNone(value_score(None))
        self.assertIsNone(value_score({}))

    def test_invalid_values_skipped(self):
        out = value_score({"pe": -5.0, "ev_ebitda": 10.0})
        self.assertIsNotNone(out)  # negative PE skipped, EV/EBITDA used

    def test_bounds(self):
        out = value_score({"pe": 1000.0})
        self.assertGreaterEqual(out["score"], 0.0)
        self.assertLessEqual(out["score"], 100.0)


class TestQualityScore(unittest.TestCase):
    def test_high_roe_low_debt_better(self):
        good = quality_score({"roe_pct": 30.0, "debt_to_equity": 0.2})
        bad = quality_score({"roe_pct": 7.0, "debt_to_equity": 1.4})
        self.assertGreater(good["score"], bad["score"])

    def test_none_when_missing(self):
        self.assertIsNone(quality_score(None))
        self.assertIsNone(quality_score({}))


class TestMomentumScore(unittest.TestCase):
    def test_uptrend_scores_high(self):
        up = make_frame(drift=0.10)
        down = make_frame(drift=-0.10)
        up["rsi_14"] = 70.0
        down["rsi_14"] = 30.0
        self.assertGreater(momentum_score(up)["score"], momentum_score(down)["score"])

    def test_decline_scores_below_neutral(self):
        # Bearish momentum must rank below the 50 neutral, not sit on it
        # (a clip-at-0 bug previously left crashing stocks reading 50).
        down = make_frame(drift=-0.15)
        down["rsi_14"] = 30.0
        self.assertLess(momentum_score(down)["score"], 50.0)

    def test_fundamentals_blend(self):
        df = make_frame(drift=0.0)
        df["rsi_14"] = 50.0
        base = momentum_score(df)["score"]
        boosted = momentum_score(
            df, {"earnings_surprise_pct": 8.0, "analyst_revisions": 1.0}
        )["score"]
        self.assertGreater(boosted, base)
        parts = momentum_score(
            df, {"earnings_surprise_pct": 8.0, "analyst_revisions": 1.0}
        )["parts"]
        self.assertIn("fundamentals", parts)

    def test_short_history_graceful(self):
        df = make_frame(n=10)
        out = momentum_score(df)
        self.assertIsNotNone(out["score"])  # no crash, neutral-ish


class TestLoadFundamentals(unittest.TestCase):
    def test_universe_csv(self):
        with tempfile.TemporaryDirectory() as tmp:
            Path(tmp, "universe.csv").write_text(
                "symbol,pe,ev_ebitda,pb,roe_pct,debt_to_equity,"
                "earnings_surprise_pct,analyst_revisions\n"
                "AAPL,28.0,24.0,7.0,35.0,1.1,5.0,0.4\n"
                "MSFT,32.0,26.0,12.0,38.0,0.6,2.0,0.2\n"
            )
            rec = load_fundamentals("aapl", data_dir=tmp)
            self.assertIsNotNone(rec)
            self.assertAlmostEqual(rec["pe"], 28.0)
            self.assertEqual(rec["source"], "csv")
            self.assertIsNone(load_fundamentals("NOSUCH", data_dir=tmp))

    def test_single_csv(self):
        with tempfile.TemporaryDirectory() as tmp:
            Path(tmp, "TSLA.csv").write_text(
                "pe,ev_ebitda,pb,roe_pct,debt_to_equity\n60.0,50.0,15.0,22.0,0.8\n"
            )
            rec = load_fundamentals("TSLA", data_dir=tmp)
            self.assertIsNotNone(rec)
            self.assertAlmostEqual(rec["pe"], 60.0)

    def test_missing_dir(self):
        self.assertIsNone(load_fundamentals("AAPL", data_dir="/nonexistent"))


class TestFactorScores(unittest.TestCase):
    def test_price_only(self):
        df = make_frame(drift=0.05)
        df["rsi_14"] = 55.0
        out = factor_scores("TEST", df)
        self.assertIsNone(out["value"])
        self.assertIsNone(out["quality"])
        self.assertIsNotNone(out["momentum"])
        self.assertIn("momentum:price", out["sources"])
        self.assertEqual(out["fundamentals_source"], "none")

    def test_full_model(self):
        df = make_frame(drift=0.05)
        df["rsi_14"] = 55.0
        fund = {
            "pe": 12.0,
            "ev_ebitda": 8.0,
            "pb": 1.5,
            "roe_pct": 25.0,
            "debt_to_equity": 0.3,
            "earnings_surprise_pct": 4.0,
            "analyst_revisions": 0.3,
            "source": "csv",
        }
        out = factor_scores("AAPL", df, fund)
        self.assertIsNotNone(out["value"])
        self.assertIsNotNone(out["quality"])
        self.assertGreaterEqual(out["composite"], 0.0)
        self.assertLessEqual(out["composite"], 100.0)
        self.assertEqual(out["fundamentals_source"], "csv")
        self.assertIn("value:fundamentals", out["sources"])

    def test_composite_weights(self):
        df = make_frame(drift=0.0)
        df["rsi_14"] = 50.0
        out = factor_scores(
            "X",
            df,
            {
                "pe": 10.0,
                "ev_ebitda": 6.0,
                "pb": 1.0,
                "roe_pct": 30.0,
                "debt_to_equity": 0.2,
                "source": "csv",
            },
        )
        # value+quality+all momentum present -> composite is a weighted avg
        self.assertGreater(out["composite"], 40.0)


if __name__ == "__main__":
    unittest.main()
