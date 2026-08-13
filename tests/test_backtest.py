"""
Tests for the backtesting package: engine trade mechanics, signal causality
(no lookahead) and end-to-end sanity on real data.
"""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.append(str(Path(__file__).parent.parent))

from tests.test_dip import make_downtrend, make_uptrend_dip, with_indicators

from src.backtest.engine import BacktestParams, run_backtest
from src.backtest.signals import dip_signal_series, _causal_swings


def make_ohlcv(prices: list, start="2020-01-01") -> pd.DataFrame:
    idx = pd.date_range(start, periods=len(prices), freq="D")
    close = pd.Series(prices, index=idx, dtype=float)
    open_ = close.shift(1).fillna(close.iloc[0])
    high = pd.concat([open_, close], axis=1).max(axis=1)
    low = pd.concat([open_, close], axis=1).min(axis=1)
    return pd.DataFrame(
        {
            "open": open_,
            "high": high,
            "low": low,
            "close": close,
            "volume": 1_000_000,
        }
    )


# ---------------------------------------------------------------------------
# Signal causality
# ---------------------------------------------------------------------------


class TestSignalCausality(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.df = with_indicators(make_uptrend_dip())
        cls.sig = dip_signal_series(cls.df)

    def test_columns(self):
        for col in [
            "score",
            "confirmed",
            "stage",
            "bias_score",
            "bullish_structure",
            "entry_lo",
            "entry_hi",
            "invalidation",
            "resistance",
            "dip_depth_pct",
        ]:
            self.assertIn(col, self.sig.columns)

    def test_no_signal_before_history(self):
        # SMA200 needs ~200 bars; nothing can confirm early.
        self.assertFalse(self.sig["confirmed"].iloc[:120].any())

    def test_dip_confirmable_at_end(self):
        self.assertIn(
            self.sig["stage"].iloc[-1], ("Confirmed", "In Pullback", "Deep Pullback")
        )
        self.assertGreaterEqual(self.sig["score"].iloc[-1], 4)

    def test_downtrend_never_confirmed(self):
        sig = dip_signal_series(with_indicators(make_downtrend()))
        self.assertFalse(sig["confirmed"].any())
        self.assertIn(sig["stage"].iloc[-1], ("No Uptrend", "Not a Dip"))

    def test_entry_zone_sanity(self):
        last = self.sig.dropna(subset=["entry_lo"]).iloc[-1]
        close = self.df["close"].loc[last.name]
        self.assertLess(last["entry_lo"], close)
        self.assertLess(last["invalidation"], last["entry_lo"])

    def test_swings_are_lagged(self):
        # A swing at bar i must not be usable before bar i+right.
        high = self.df["high"]
        sh_raw = high == high.rolling(5, center=True).max()
        first_raw = sh_raw.idxmax()
        sh, _ = _causal_swings(self.df)
        first_use = sh[sh].index.min()
        self.assertGreaterEqual(first_use, first_raw + pd.Timedelta(days=2))


# ---------------------------------------------------------------------------
# Engine mechanics (hand-crafted signals)
# ---------------------------------------------------------------------------


class TestEngine(unittest.TestCase):
    def _signal(self, df, confirmed, entry_lo, inv, res, score=6):
        """Signal frame with the trade levels set at the confirmed bars."""
        n = len(df)
        sig = pd.DataFrame(
            {
                "confirmed": np.array(confirmed, dtype=bool),
                "entry_lo": np.nan,
                "entry_hi": np.nan,
                "invalidation": np.nan,
                "resistance": np.nan,
                "score": np.full(n, float(score)),
            },
            index=df.index,
        )
        for i, c in enumerate(confirmed):
            if c:
                sig.iloc[i, sig.columns.get_loc("entry_lo")] = entry_lo
                sig.iloc[i, sig.columns.get_loc("invalidation")] = inv
                sig.iloc[i, sig.columns.get_loc("resistance")] = res
        return sig

    def test_stop_loss_exit(self):
        df = make_ohlcv([100, 101, 102, 100, 99, 98, 97, 96, 95, 94])
        sig = self._signal(
            df, [0, 0, 1, 0, 0, 0, 0, 0, 0, 0], entry_lo=99.0, inv=96.0, res=110.0
        )
        res = run_backtest(sig, df, BacktestParams(max_hold=100), symbol="TEST")
        self.assertEqual(len(res.trades), 1)
        t = res.trades[0]
        self.assertEqual(t.reason, "stop")
        self.assertEqual(t.exit_price, 96.0)
        self.assertLess(t.pnl, 0)

    def test_target_exit(self):
        df = make_ohlcv([100, 101, 102, 100, 101, 102, 104, 106, 108, 110])
        sig = self._signal(
            df, [0, 0, 1, 0, 0, 0, 0, 0, 0, 0], entry_lo=100.5, inv=98.0, res=108.0
        )
        res = run_backtest(sig, df, BacktestParams(max_hold=100), symbol="TEST")
        t = res.trades[0]
        self.assertEqual(t.reason, "target")
        self.assertEqual(t.exit_price, 108.0)
        self.assertGreater(t.pnl, 0)

    def test_time_stop(self):
        df = make_ohlcv([100, 101] + [100] * 28)
        sig = self._signal(df, [0, 1] + [0] * 28, entry_lo=100.0, inv=95.0, res=None)
        res = run_backtest(
            sig, df, BacktestParams(max_hold=10, entry_valid_bars=3), symbol="TEST"
        )
        t = res.trades[0]
        self.assertEqual(t.reason, "time")
        self.assertEqual(t.bars_held, 10)

    def test_position_sizing(self):
        df = make_ohlcv([100, 101, 102, 100, 101, 102, 104, 106, 108, 110])
        sig = self._signal(
            df, [0, 0, 1, 0, 0, 0, 0, 0, 0, 0], entry_lo=100.0, inv=95.0, res=200.0
        )
        res = run_backtest(
            sig,
            df,
            BacktestParams(initial_capital=100_000, risk_pct=0.01, max_hold=3),
            symbol="TEST",
        )
        # qty = 100000 * 0.01 / (100 - 95) = 200 (closed by time stop)
        self.assertAlmostEqual(res.trades[0].qty, 200.0)

    def test_limit_not_filled_cancelled(self):
        df = make_ohlcv([100, 101, 102, 103, 104, 105, 106, 107, 108, 109])
        sig = self._signal(
            df, [0, 0, 1, 0, 0, 0, 0, 0, 0, 0], entry_lo=90.0, inv=85.0, res=None
        )
        res = run_backtest(
            sig, df, BacktestParams(entry_valid_bars=2, max_hold=100), symbol="TEST"
        )
        self.assertEqual(len(res.trades), 0)  # never touched -> cancelled

    def test_market_entry(self):
        df = make_ohlcv([100, 101, 102, 103, 104, 105, 106, 107, 108, 109])
        sig = self._signal(
            df, [0, 0, 1, 0, 0, 0, 0, 0, 0, 0], entry_lo=101.0, inv=98.0, res=200.0
        )
        res = run_backtest(
            sig, df, BacktestParams(entry_type="market", max_hold=3), symbol="TEST"
        )
        # market fills at the open of the bar AFTER the signal bar
        self.assertEqual(res.trades[0].entry_price, 102.0)

    def test_stats_correct(self):
        df = make_ohlcv(
            [
                100,
                101,
                102,
                100,
                99,
                102,
                104,
                106,
                108,
                106,
                103,
                102,
                101,
                100,
                103,
                106,
                109,
                112,
            ]
        )
        confirmed = [0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0]
        sig = self._signal(df, confirmed, entry_lo=100.5, inv=98.0, res=107.0)
        res = run_backtest(sig, df, BacktestParams(max_hold=100), symbol="TEST")
        s = res.stats
        self.assertEqual(s["n_trades"], 2)
        self.assertEqual(s["wins"] + s["losses"], s["n_trades"])
        self.assertGreaterEqual(s["win_rate"], 0.0)
        self.assertLessEqual(s["win_rate"], 1.0)
        self.assertGreaterEqual(s["max_drawdown_pct"], -100.0)
        self.assertGreater(s["gross_profit"], 0)

    def test_equity_curve_length(self):
        df = make_ohlcv([100] * 40)
        sig = self._signal(df, [0, 1] + [0] * 38, entry_lo=99.0, inv=95.0, res=None)
        res = run_backtest(sig, df, BacktestParams(max_hold=5), symbol="TEST")
        self.assertEqual(len(res.equity), len(df))
        self.assertGreater(res.equity.iloc[-1], 0)


# ---------------------------------------------------------------------------
# End-to-end on real data
# ---------------------------------------------------------------------------


class TestEndToEnd(unittest.TestCase):
    def test_real_symbol_backtest(self):
        path = Path("data/raw/candidates/US500_D1.parquet")
        if not path.exists():
            self.skipTest("US500 data not available")
        from src.data.loader import clean_data, load_data

        df = with_indicators(clean_data(load_data(path, symbol="US500")))
        sig = dip_signal_series(df)
        res = run_backtest(
            sig, df, BacktestParams(max_hold=20, bars_per_year=252), symbol="US500"
        )
        s = res.stats
        self.assertGreaterEqual(s["n_trades"], 0)
        self.assertGreaterEqual(s["win_rate"], 0.0)
        self.assertGreaterEqual(s["max_drawdown_pct"], -100.0)
        if s["n_trades"] > 0:
            self.assertEqual(s["wins"] + s["losses"], s["n_trades"])

    def test_signal_produces_confirmations(self):
        path = Path("data/raw/candidates/US500_D1.parquet")
        if not path.exists():
            self.skipTest("US500 data not available")
        from src.data.loader import clean_data, load_data

        df = with_indicators(clean_data(load_data(path, symbol="US500")))
        sig = dip_signal_series(df)
        self.assertGreater(int(sig["confirmed"].sum()), 0)


if __name__ == "__main__":
    unittest.main()
