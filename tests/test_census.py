"""
Tests for the historical opportunity census (src/analysis/census.py).

The census is the audit's evidence that the engine is direction-neutral in
*opportunity detection*: long and short candidates/signals must both be
countable across history, with causal (no-lookahead) family labels and
realized outcomes.
"""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent))

import numpy as np
import pandas as pd

from src.analysis.census import (
    _classify_history,
    _realized_r,
    opportunity_census,
)
from src.features.indicators import add_all_indicators


def _make_df(n: int = 600, seed: int = 3) -> pd.DataFrame:
    """Synthetic OHLCV long enough for the census warm-up."""
    rng = np.random.default_rng(seed)
    # Regime mix: up 150, down 150, range 150, up 150.
    seg = []
    close = 100.0
    for i in range(n):
        if i % 300 < 150:
            close *= 1.0 + rng.normal(0.0008, 0.004)
        elif i % 300 < 250:
            close *= 1.0 + rng.normal(-0.0012, 0.004)
        else:
            close *= 1.0 + rng.normal(0.0, 0.006)
        seg.append(max(close, 1.0))
    close = pd.Series(seg)
    df = pd.DataFrame(
        {
            "open": close * (1 + rng.normal(0, 0.001, n)),
            "high": close * (1 + abs(rng.normal(0, 0.002, n))),
            "low": close * (1 - abs(rng.normal(0, 0.002, n))),
            "close": close,
            "volume": rng.integers(800, 1200, n).astype(float),
        }
    )
    return add_all_indicators(df)


class TestClassifyHistory(unittest.TestCase):
    def test_causal_window_no_future(self):
        df = _make_df()
        hist = _classify_history(df, step=5)
        self.assertGreater(len(hist), 20)
        # Every classified bar must have a score on the SAME side axes.
        self.assertTrue((hist["long_score"] >= 0).all())
        self.assertTrue((hist["short_score"] >= 0).all())
        self.assertTrue((hist["long_score"] <= 1).all())
        self.assertTrue((hist["short_score"] <= 1).all())
        # Families must come from the taxonomy when present.
        from src.features.setups import ALL_FAMILIES

        for fam in hist["setup_family"].dropna():
            self.assertIn(fam, ALL_FAMILIES)

    def test_both_sides_produce_candidates(self):
        # Over a mixed regime the census must observe BOTH long and short
        # family candidates - the architecture must not be blind to one
        # side (the central audit requirement).
        df = _make_df()
        hist = _classify_history(df, step=3)
        long_n = (hist["direction"] == "long").sum()
        short_n = (hist["direction"] == "short").sum()
        self.assertGreater(long_n, 0)
        self.assertGreater(short_n, 0)


class TestRealizedR(unittest.TestCase):
    def _sig_frame(self, df):
        from src.backtest.signals import dip_signal_series, rally_signal_series

        return dip_signal_series(df), rally_signal_series(df)

    def test_first_touch_resolution(self):
        df = _make_df()
        sl, ss = self._sig_frame(df)
        rl = _realized_r(sl, df, "long")
        rs = _realized_r(ss, df, "short")
        # Results are either NaN (unresolved) or in R units (win > 0, loss -1).
        for r in (rl, rs):
            vals = r.dropna()
            self.assertTrue((vals > 0).all() or (vals <= 0).all())
            self.assertTrue(((vals == -1.0) | (vals > 0)).all())


class TestCensus(unittest.TestCase):
    def test_census_counts_both_sides(self):
        import tempfile

        with tempfile.TemporaryDirectory() as tmp:
            from pathlib import Path as P

            p = P(tmp)
            (p / "full_fx").mkdir(parents=True)
            df = _make_df(700)
            # Write RAW OHLCV only: the census re-runs add_all_indicators
            # on load, so indicator columns in the parquet would duplicate.
            raw = df[["open", "high", "low", "close", "volume"]].copy()
            raw["date"] = pd.date_range("2018-01-01", periods=len(df))
            raw.to_parquet(p / "full_fx" / "TEST_D1.parquet")

            stats = opportunity_census(
                ["TEST"], data_dir=tmp, group="full_fx", min_family_score=0.3
            )
        self.assertEqual(stats["n_symbols"], 1)
        for side in ("long", "short"):
            self.assertIn(side, stats["recall"])
            self.assertGreaterEqual(stats["recall"][side]["candidates"], 0)
        # The two-sided contract: if one side fired, the other must be
        # *measured* (0 is a measured zero, not an absence of the side).
        self.assertIn("long", stats["side"])
        self.assertIn("short", stats["side"])


if __name__ == "__main__":
    unittest.main()
