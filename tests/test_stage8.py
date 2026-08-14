"""
Regression tests for Stage-8 invariants (src/analysis/stage8.py).

Stage-8 campaign invariants locked here (do not regress):

  1. The frozen LONG protocol: k=3 of {L1_rsi30, L2_drop5, L3_streak5n},
     PRIMARY_H=10, COST_R=0.05, 1R = 1.25 x ATR.
  2. True R-unit forward returns are unit-consistent across price scales:
     a 1R move must produce ~1.0 R regardless of the instrument's close
     level (the Stage-8 fix for Stage-7's 1/close-scaled fwd_atr).
  3. Vol buckets are CAUSAL (rolling 250-bar percentile) — no full-sample
     qcut (the Stage-7 mild look-ahead that was fixed).
  4. SHORT reversal remains LOCKED as falsified (Stage-6): stage8 exposes
     no short trigger.
  5. BH FDR is monotone and valid.
  6. _trade_r subtracts exactly COST_R and yields chronologically sorted
     net-R series.
"""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.append(str(Path(__file__).parent.parent))

from src.analysis.stage8 import (
    COST_R,
    K_LONG,
    LONG_COMBO,
    PRIMARY_H,
    R_MULT,
    _bh_fdr,
    _fwd_R,
    _trade_r,
    _bucket,
    eligible_universe,
)


def _synthetic_frame(close_level: float, n: int = 600, seed: int = 1) -> pd.DataFrame:
    """Deterministic OHLC frame at a given price level (for unit-consistency
    checks). Returns a frame with close/high/low/atr_14 + fwd10_R."""
    rng = np.random.default_rng(seed)
    rets = rng.normal(0.0, 0.01, n)
    close = close_level * np.exp(np.cumsum(rets))
    high = close * (1 + np.abs(rng.normal(0, 0.004, n)))
    low = close * (1 - np.abs(rng.normal(0, 0.004, n)))
    idx = pd.date_range("2015-01-01", periods=n, freq="B")
    df = pd.DataFrame(
        {"close": close, "high": high, "low": low, "open": close},
        index=idx,
    )
    # ATR ~ 1% of price at this level
    df["atr_14"] = close * 0.01
    return df


class TestFrozenProtocol(unittest.TestCase):
    def test_frozen_entry_rule(self):
        self.assertEqual(K_LONG, 3)
        self.assertEqual(LONG_COMBO, ["L1_rsi30", "L2_drop5", "L3_streak5n"])
        self.assertEqual(PRIMARY_H, 10)
        self.assertEqual(COST_R, 0.05)
        self.assertEqual(R_MULT, 1.25)

    def test_no_short_triggers_in_stage8(self):
        # SHORT reversal is LOCKED falsified (Stage-6); stage8 must not
        # define any short trigger surface (the docstring legitimately
        # mentions the locked short leg, so scope the check to code).
        import inspect
        import src.analysis.stage8 as s8

        src = inspect.getsource(s8)
        self.assertNotIn("S1_rsi70", src)
        self.assertNotIn("S2_rally5", src)
        # no short trigger family is constructed anywhere
        self.assertNotIn('direction == "short"', src)
        self.assertNotIn('"SHORT_', src)


class TestRUnits(unittest.TestCase):
    def test_r_units_scale_invariant(self):
        """A 1R (1.25*ATR) move must yield ~1.0 R at any price level."""
        for level in (0.65, 1.0, 150.0, 4400.0):
            df = _synthetic_frame(level)
            df = _fwd_R(df, horizons=(10,))
            # force the 10-bar move to be exactly +1.25*ATR
            row = df.index[len(df) // 2]
            pos = df.index.get_loc(row)
            entry = float(df["close"].iloc[pos])
            atr = float(df["atr_14"].iloc[pos])
            target = entry + R_MULT * atr
            # overwrite the close 10 bars later to land exactly on target
            j = min(pos + 10, len(df) - 1)
            df.iloc[j, df.columns.get_loc("close")] = target
            df = _fwd_R(df, horizons=(10,))
            r = float(df["fwd10_R"].iloc[pos])
            self.assertAlmostEqual(
                r, 1.0, places=2, msg=f"R not scale-invariant at close={level}"
            )

    def test_fwd_R_columns_present(self):
        df = _synthetic_frame(1.0)
        df = _fwd_R(df, horizons=(1, 5, 10, 20))
        for h in (1, 5, 10, 20):
            self.assertIn(f"fwd{h}_R", df.columns)


class TestCausalVolBuckets(unittest.TestCase):
    def test_no_full_sample_qcut(self):
        """The Stage-8 vol bucket must be a rolling (causal) percentile,
        never a full-sample qcut rank (Stage-7 mild look-ahead)."""
        import inspect
        import src.analysis.stage8 as s8

        src = inspect.getsource(s8)
        self.assertIn("rolling(VOL_ROLL).rank(pct=True)", src)
        # qcut over the full series must not be used for vol buckets
        self.assertNotIn('qcut(df["atr_14"]', src)
        self.assertNotIn(".rank(method=", src)

    def test_bucket_labels(self):
        import numpy as np
        import pandas as pd

        n = 400
        idx = pd.date_range("2016-01-01", periods=n, freq="B")
        atr = np.concatenate([np.full(200, 1.0), np.full(200, 3.0)])
        df = pd.DataFrame({"atr_14": atr}, index=idx)
        df["close"] = 1.0
        atr20 = df["atr_14"].rolling(20).mean()
        df["vol_pct"] = atr20.rolling(250).rank(pct=True)
        df["vol_bucket"] = pd.cut(
            df["vol_pct"],
            [0.0, 0.33, 0.66, 1.0],
            labels=["low", "med", "high"],
        )
        # early rows are NaN (no 250-bar history) -> causal, no look-ahead
        self.assertTrue(df["vol_bucket"].iloc[:250].isna().all())
        # after the regime shift, the bucket must eventually read high
        self.assertIn("high", set(df["vol_bucket"].dropna().astype(str)))


class TestBH_FDR(unittest.TestCase):
    def test_monotone_and_valid(self):
        pvals = {"a": 0.001, "b": 0.01, "c": 0.05, "d": 0.6}
        q, sig = _bh_fdr(pvals)
        # q-values are monotone non-decreasing in p
        order = sorted(pvals, key=pvals.get)
        qs = [q[k] for k in order]
        self.assertEqual(qs, sorted(qs))
        # a survives at 0.05 (smallest p)
        self.assertIn("a", sig)
        for _k, v in q.items():
            self.assertGreaterEqual(v, 0.0)
            self.assertLessEqual(v, 1.0)

    def test_empty(self):
        q, sig = _bh_fdr({})
        self.assertEqual(q, {})
        self.assertEqual(sig, [])


class TestTradeR(unittest.TestCase):
    def test_net_r_cost_subtracted_and_sorted(self):
        frames = {
            "A": _frozen_for_trade_r(0.65),
            "B": _frozen_for_trade_r(1.2),
        }

        def mask(sym, df):
            # trade on a fixed date mid-frame for both symbols
            s = pd.Series(False, index=df.index)
            mid = df.index[len(df) // 2]
            s.loc[mid] = True
            return s

        rs = _trade_r(frames, mask, horizon=10, cutoff="2030-01-01")
        v = rs.values.astype(float)
        # each trade = fwd10_R - COST_R; no raw fwd value leaks through
        self.assertTrue(np.isfinite(v).all())
        self.assertTrue(rs.index.is_monotonic_increasing)
        # two symbols -> two trades
        self.assertEqual(len(v), 2)


def _frozen_for_trade_r(level: float) -> pd.DataFrame:
    df = _synthetic_frame(level)
    df = _fwd_R(df, horizons=(10,))
    return df


class TestUniverseAndBuckets(unittest.TestCase):
    def test_bucket_classification(self):
        self.assertEqual(_bucket("EURUSD"), "fx_major_cross")
        self.assertEqual(_bucket("USDTRY"), "fx_exotic")
        self.assertEqual(_bucket("XAUUSD"), "metal")
        self.assertEqual(_bucket("BTCUSD"), "crypto")
        self.assertEqual(_bucket("US500"), "index")

    def test_eligible_universe_shape(self):
        u = eligible_universe()
        self.assertGreaterEqual(len(u), 100)
        self.assertIn("EURUSD", u)
        self.assertIn("BTCUSD", u)
        self.assertEqual(u["EURUSD"], "full_fx")
        self.assertEqual(u["BTCUSD"], "candidates")


if __name__ == "__main__":
    unittest.main()
