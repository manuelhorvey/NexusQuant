"""
Tests for the risk & position sizing package: sizing math, VaR, RiskManager
gating, correlation-aware limits, and the backtest engine sizing modes.
"""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.append(str(Path(__file__).parent.parent))

from src.backtest.engine import BacktestParams, run_backtest
from src.risk.limits import RiskManager
from src.risk.metrics import (
    check_correlation_limit,
    portfolio_heat,
    portfolio_var,
    returns_correlation,
    trade_var,
    trade_var_pct,
)
from src.risk.sizing import (
    fractional_qty,
    kelly_fraction,
    kelly_qty,
    risk_dollars,
    risk_pct_of_equity,
    size_position,
    vol_target_qty,
)


def make_ohlcv(prices: list, start="2020-01-01", atr=None) -> pd.DataFrame:
    idx = pd.date_range(start, periods=len(prices), freq="D")
    close = pd.Series(prices, index=idx, dtype=float)
    open_ = close.shift(1).fillna(close.iloc[0])
    high = pd.concat([open_, close], axis=1).max(axis=1)
    low = pd.concat([open_, close], axis=1).min(axis=1)
    df = pd.DataFrame(
        {"open": open_, "high": high, "low": low, "close": close, "volume": 1_000_000}
    )
    if atr is not None:
        df["atr_14"] = float(atr)
    return df


def make_signal(df, confirmed, entry_lo, inv, res, score=6):
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


# ---------------------------------------------------------------------------
# Position sizing math
# ---------------------------------------------------------------------------


class TestSizing(unittest.TestCase):
    def test_fractional_qty(self):
        # qty = equity * risk / (entry - stop) = 100000 * 0.01 / 5 = 200
        self.assertAlmostEqual(fractional_qty(100_000, 100, 95, 0.01), 200.0)

    def test_fractional_qty_guards(self):
        self.assertEqual(fractional_qty(100_000, 100, 100, 0.01), 0.0)
        self.assertEqual(fractional_qty(100_000, 95, 100, 0.01), 0.0)
        self.assertEqual(fractional_qty(0, 100, 95, 0.01), 0.0)

    def test_vol_target_qty_uncapped(self):
        # sigma = 0.01 * 1 = 0.01; notional = 100000*0.02/0.01 = 200000 -> qty 2000
        qty = vol_target_qty(100_000, 100, 95, 0.01, 0.02, hold_bars=1)
        self.assertAlmostEqual(qty, 2000.0)

    def test_vol_target_qty_cap(self):
        # uncapped 2000, but cap risk 1% -> fractional qty 200
        qty = vol_target_qty(
            100_000, 100, 95, 0.01, 0.02, hold_bars=1, cap_risk_pct=0.01
        )
        self.assertAlmostEqual(qty, 200.0)

    def test_vol_target_scales_with_hold(self):
        # hold 4 bars -> sigma doubles (sqrt(4)) -> qty halves
        q1 = vol_target_qty(100_000, 100, 95, 0.01, 0.02, hold_bars=1)
        q4 = vol_target_qty(100_000, 100, 95, 0.01, 0.02, hold_bars=4)
        self.assertAlmostEqual(q4, q1 / 2.0)

    def test_kelly_fraction(self):
        # f = 0.5 * (0.6 - 0.4/1.0) = 0.1
        self.assertAlmostEqual(kelly_fraction(0.6, 1.0, fraction=0.5), 0.1)
        # negative edge -> 0
        self.assertEqual(kelly_fraction(0.3, 1.0, fraction=0.5), 0.0)
        # invalid p -> 0
        self.assertEqual(kelly_fraction(1.0, 1.0, fraction=0.5), 0.0)

    def test_kelly_qty_capped(self):
        # f = 0.1 -> uncapped qty 2000, cap 1% -> 200
        qty = kelly_qty(100_000, 100, 95, 0.6, 1.0, fraction=0.5, cap_risk_pct=0.01)
        self.assertAlmostEqual(qty, 200.0)

    def test_kelly_qty_no_edge(self):
        self.assertEqual(kelly_qty(100_000, 100, 95, 0.4, 1.0), 0.0)

    def test_size_position_dispatch(self):
        self.assertAlmostEqual(
            size_position(100_000, 100, 95, mode="fractional", risk_pct=0.01), 200.0
        )
        self.assertAlmostEqual(
            size_position(
                100_000,
                100,
                95,
                mode="kelly",
                p=0.6,
                payoff=1.0,
                kelly_fraction_ratio=0.5,
                cap_risk_pct=0.01,
            ),
            200.0,
        )
        # voltarget without atr -> falls back to fractional
        self.assertAlmostEqual(
            size_position(100_000, 100, 95, mode="voltarget", risk_pct=0.01), 200.0
        )

    def test_risk_dollars_and_pct(self):
        self.assertAlmostEqual(risk_dollars(200, 100, 95), 1000.0)
        self.assertAlmostEqual(risk_pct_of_equity(200, 100, 95, 100_000), 0.01)


# ---------------------------------------------------------------------------
# VaR
# ---------------------------------------------------------------------------


class TestVaR(unittest.TestCase):
    def test_trade_var(self):
        # VaR95 = 1.645 * 2.0 * 1000 * sqrt(1) = 3290
        self.assertAlmostEqual(trade_var(1000, 2.0, hold_bars=1), 3290.0)
        # sqrt scaling with horizon
        self.assertAlmostEqual(trade_var(1000, 2.0, hold_bars=4), 6580.0)
        self.assertEqual(trade_var(1000, -2.0, hold_bars=1), 0.0)

    def test_trade_var_pct(self):
        self.assertAlmostEqual(trade_var_pct(1000, 100, 2.0, 100_000), 0.0329)

    def test_portfolio_var_uncorrelated(self):
        # w = [0.5, 0.5], sigma = 0.01; sigma_p = 0.005*sqrt(2)
        corr = np.eye(2)
        var = portfolio_var([1000, 1000], [0.01, 0.01], corr, z=1.645)
        expected = 1.645 * 0.005 * np.sqrt(2) * 2000
        self.assertAlmostEqual(var, float(expected), places=3)

    def test_portfolio_var_correlated_is_higher(self):
        id_var = portfolio_var([1000, 1000], [0.01, 0.01], np.eye(2))
        corr_var = portfolio_var([1000, 1000], [0.01, 0.01], np.ones((2, 2)))
        self.assertGreater(corr_var, id_var)

    def test_portfolio_var_edges(self):
        self.assertEqual(portfolio_var([], [], np.empty((0, 0))), 0.0)
        self.assertEqual(portfolio_var([0, 0], [0.01, 0.01], np.eye(2)), 0.0)


# ---------------------------------------------------------------------------
# Portfolio heat & correlation limits
# ---------------------------------------------------------------------------


class TestPortfolioRisk(unittest.TestCase):
    def test_portfolio_heat(self):
        positions = [
            {"entry": 100, "stop": 95, "qty": 200},
            {"entry": 50, "stop": 49, "qty": 1000},
        ]
        # risk = 1000 + 1000 = 2000 / 100000 = 0.02
        self.assertAlmostEqual(portfolio_heat(positions, 100_000), 0.02)
        self.assertEqual(portfolio_heat(positions, 0), 0.0)

    def test_correlation_gate(self):
        corr = pd.DataFrame(
            [[1.0, 0.9, 0.3], [0.9, 1.0, 0.5], [0.3, 0.5, 1.0]],
            index=["A", "B", "C"],
            columns=["A", "B", "C"],
        )
        # A vs existing B: 0.9 > 0.6 -> blocked
        chk = check_correlation_limit("A", ["B"], corr, max_corr=0.6)
        self.assertFalse(chk["allowed"])
        self.assertAlmostEqual(chk["avg_corr"], 0.9)
        # A vs existing C: 0.3 -> allowed
        chk = check_correlation_limit("A", ["C"], corr, max_corr=0.6)
        self.assertTrue(chk["allowed"])
        # no holdings -> allowed with None avg
        chk = check_correlation_limit("A", [], corr)
        self.assertTrue(chk["allowed"])
        self.assertIsNone(chk["avg_corr"])

    def test_returns_correlation(self):
        rets = pd.DataFrame({"A": [0.01, -0.01, 0.02], "B": [-0.01, 0.01, -0.02]})
        corr = returns_correlation(rets)
        self.assertAlmostEqual(corr.loc["A", "B"], -1.0)

    def test_top_pairs_no_nan_no_dupes(self):
        # pandas >= 3 stack() keeps NaN on the diagonal: pairs must be
        # strictly upper-triangle, distinct and finite.
        from src.risk.metrics import top_correlated_pairs

        corr = pd.DataFrame(
            [[1.0, 0.9, 0.3], [0.9, 1.0, 0.5], [0.3, 0.5, 1.0]],
            index=["A", "B", "C"],
            columns=["A", "B", "C"],
        )
        pairs = top_correlated_pairs(corr)
        self.assertEqual(len(pairs), 3)
        self.assertEqual(pairs[0]["pair"], "A / B")
        self.assertAlmostEqual(pairs[0]["corr"], 0.9)
        for p in pairs:
            i, j = p["pair"].split(" / ")
            self.assertNotEqual(i, j)
            self.assertTrue(p["corr"] == p["corr"])  # not NaN
        self.assertEqual(top_correlated_pairs(corr.iloc[:1, :1]), [])


# ---------------------------------------------------------------------------
# RiskManager trading limits
# ---------------------------------------------------------------------------


class TestPortfolioReport(unittest.TestCase):
    """portfolio_report with controlled inputs (no real data)."""

    def test_corr_reindexed_by_position_symbols(self):
        # Regression: if a symbol has returns but NO setup, the old code
        # sliced corr.values[:n, :n] and misaligned rows into the VaR.
        import src.risk.run as rr
        from src.risk.run import portfolio_report

        real_pf, real_plan = rr.prepare_frame, rr.symbol_risk_plan

        def fake_prepare(sym, group, timeframe, data_dir="data/raw"):
            idx = pd.date_range("2020-01-01", periods=300, freq="D")
            rng = np.random.default_rng(ord(sym[0]))
            base = {"A": 100.0, "B": 100.0, "C": 100.0}[sym]
            close = pd.Series(base + np.cumsum(rng.normal(0, 0.1, 300)), index=idx)
            df = pd.DataFrame(
                {
                    "open": close,
                    "high": close + 0.2,
                    "low": close - 0.2,
                    "close": close,
                    "volume": 1_000_000,
                }
            )
            df["returns"] = df["close"].pct_change()
            df["atr_14"] = 1.0
            return df

        def fake_plan(sym, group, timeframe, data_dir="data/raw", **kw):
            if sym == "B":
                return {"symbol": sym, "setup": None, "reason": "no setup"}
            return {
                "symbol": sym,
                "setup": {"entry": 100.0, "stop": 99.0},
                "date": "2020-12-31",
                "close": 100.0,
                "inputs": {},
                "sizes": [],
            }

        rr.prepare_frame, rr.symbol_risk_plan = fake_prepare, fake_plan
        try:
            rep = portfolio_report(
                ["A", "B", "C"], "x", "D1", equity=100_000, risk_pct=0.01
            )
        finally:
            rr.prepare_frame, rr.symbol_risk_plan = real_pf, real_plan

        self.assertEqual(rep["n_setups"], 2)
        self.assertEqual([p["symbol"] for p in rep["positions"]], ["A", "C"])
        # VaR must use the A/C block of the correlation matrix, not a
        # positional slice (which would grab the A/B block).
        expected = portfolio_var(
            [100_000, 100_000],
            [rep["vols"]["A"] / 100, rep["vols"]["C"] / 100],
            rep["correlation"].loc[["A", "C"], ["A", "C"]].values,
        )
        # vols are stored rounded to 3 dp, so allow a small tolerance while
        # still failing on a genuinely misaligned correlation block.
        self.assertAlmostEqual(rep["portfolio_var_95_1bar"], expected, delta=0.5)

    def test_degenerate_overlap_graceful(self):
        import src.risk.run as rr
        from src.risk.run import portfolio_report

        real_pf, real_plan = rr.prepare_frame, rr.symbol_risk_plan

        def fake_prepare(sym, group, timeframe, data_dir="data/raw"):
            # disjoint date ranges -> zero overlapping rows after dropna
            start = "2020-01-01" if sym == "A" else "2030-01-01"
            idx = pd.date_range(start, periods=300, freq="D")
            close = pd.Series(100.0 + np.arange(300) * 0.1, index=idx)
            df = pd.DataFrame(
                {
                    "open": close,
                    "high": close + 0.2,
                    "low": close - 0.2,
                    "close": close,
                    "volume": 1_000_000,
                }
            )
            df["returns"] = df["close"].pct_change()
            df["atr_14"] = 1.0
            return df

        def fake_plan(sym, group, timeframe, data_dir="data/raw", **kw):
            return {
                "symbol": sym,
                "setup": {"entry": 100.0, "stop": 99.0},
                "date": "2020-12-31",
                "close": 100.0,
                "inputs": {},
                "sizes": [],
            }

        rr.prepare_frame, rr.symbol_risk_plan = fake_prepare, fake_plan
        try:
            rep = portfolio_report(["A", "B"], "x", "D1", equity=100_000, risk_pct=0.01)
        finally:
            rr.prepare_frame, rr.symbol_risk_plan = real_pf, real_plan
        self.assertEqual(rep["symbols"], [])
        self.assertIn("overlap", rep["reason"])


class TestRiskManager(unittest.TestCase):
    def test_daily_loss_halt(self):
        rm = RiskManager(
            equity=100_000, max_daily_loss_pct=0.02, max_weekly_loss_pct=0.10
        )
        rm.record_pnl(-1500)
        self.assertFalse(rm.daily_halted)
        rm.record_pnl(-600)  # -2100 < -2000 -> halted
        self.assertTrue(rm.daily_halted)
        self.assertFalse(rm.status()["trading_enabled"])
        self.assertIn("daily", rm.can_open(0.01)["reason"])

    def test_weekly_loss_halt(self):
        rm = RiskManager(equity=100_000, max_weekly_loss_pct=0.04)
        rm.record_pnl(-1000)
        rm.record_pnl(-1000)
        rm.record_pnl(-2100)  # -4100 <= -4000 -> weekly halt
        self.assertTrue(rm.weekly_halted)

    def test_roll_day_resets(self):
        rm = RiskManager(equity=100_000, max_daily_loss_pct=0.02)
        rm.record_pnl(-2100)
        self.assertTrue(rm.daily_halted)
        rm.roll_day()
        self.assertFalse(rm.daily_halted)
        self.assertEqual(rm.daily_pnl, 0.0)

    def test_max_concurrent(self):
        rm = RiskManager(equity=100_000, max_concurrent=2)
        self.assertTrue(rm.can_open(0.01)["allowed"])
        rm.open_position(0.01)
        rm.open_position(0.01)
        chk = rm.can_open(0.01)
        self.assertFalse(chk["allowed"])
        self.assertIn("concurrent", chk["reason"])
        rm.close_position(0.01)
        self.assertTrue(rm.can_open(0.01)["allowed"])

    def test_heat_gate(self):
        rm = RiskManager(equity=100_000, max_heat_pct=0.04)
        rm.open_position(0.03)
        chk = rm.can_open(0.02)  # 0.03 + 0.02 > 0.04 -> blocked
        self.assertFalse(chk["allowed"])
        self.assertIn("heat", chk["reason"])
        # small trade fits under the limit
        self.assertTrue(rm.can_open(0.005)["allowed"])

    def test_open_close_heat_consistency(self):
        rm = RiskManager(equity=100_000)
        rm.open_position(0.01)
        rm.open_position(0.02)
        self.assertEqual(rm.n_open, 2)
        self.assertAlmostEqual(rm.heat, 0.03)
        rm.close_position(0.02)
        self.assertAlmostEqual(rm.heat, 0.01)


# ---------------------------------------------------------------------------
# Engine sizing modes
# ---------------------------------------------------------------------------


class TestReportRiskSection(unittest.TestCase):
    """The full report carries a risk section (graceful when no setup)."""

    def _report(self, maker):
        from tests.test_dip import with_indicators
        from src.features.regime import detect_regime
        from src.analysis.report import generate_full_report

        df = detect_regime(with_indicators(maker()))
        return generate_full_report(df, symbol="TEST")

    def test_risk_section_on_uptrend(self):
        from tests.test_dip import make_uptrend_dip

        report = self._report(make_uptrend_dip)
        self.assertIn("risk", report)
        rk = report["risk"]
        self.assertEqual(rk["symbol"], "TEST")
        if rk.get("setup"):
            self.assertEqual(len(rk["sizes"]), 3)
            methods = {row["method"] for row in rk["sizes"]}
            self.assertEqual(methods, {"fractional", "voltarget", "kelly"})
            # fractional qty matches risk_dollars math
            frac = rk["sizes"][0]
            self.assertAlmostEqual(frac["risk_pct_equity"], 1.0, places=1)
        else:
            self.assertIn("reason", rk)

    def test_risk_kelly_uses_ml_prob_when_present(self):
        # regression: the risk section must be built AFTER the ML section,
        # otherwise Kelly silently falls back to the default p.
        from tests.test_dip import make_uptrend_dip

        report = self._report(make_uptrend_dip)
        ml = report.get("ml")
        rk = report.get("risk")
        if ml and rk and rk.get("setup"):
            self.assertAlmostEqual(
                rk["inputs"]["kelly_p"], ml["prob_pct"] / 100.0, places=3
            )
            self.assertAlmostEqual(rk["setup"]["ml_prob"], ml["prob_pct"], places=1)

    def test_risk_section_on_downtrend_graceful(self):
        from tests.test_dip import make_downtrend

        report = self._report(make_downtrend)
        rk = report.get("risk")
        if rk is not None:
            if rk.get("setup"):
                # a degenerate/no-uptrend setup must not masquerade as a
                # confirmed dip - the stage travels with the plan
                self.assertIn(
                    rk["setup"]["dip_stage"],
                    ("No Uptrend", "Not a Dip", "Support Broken", "Deep Pullback"),
                )
            else:
                self.assertIn("reason", rk)

    def test_risk_plan_reports_rr_floor_and_ladder(self):
        """Spec #11: the plan distinguishes the nearest single-target R:R
        (~1R) from the ladder's achievable R:R (scaling out) and reports
        whether the 2.5:1 floor is met."""
        from tests.test_dip import make_uptrend_dip

        report = self._report(make_uptrend_dip)
        from src.risk.run import risk_plan_from_report

        plan = risk_plan_from_report(report, "TEST", min_rr=2.5)
        if not plan.get("setup"):
            self.skipTest("no actionable dip setup in the fixture")
        s = plan["setup"]
        for key in ("rr", "rr_nearest", "rr_ok", "min_rr", "best_rr", "min_rr_tp"):
            self.assertIn(key, s)
        self.assertEqual(s["min_rr"], 2.5)
        # The ladder (when present) must make the floor reachable via TP3;
        # best_rr >= floor implies rr_ok.
        if s["best_rr"] is not None:
            self.assertGreaterEqual(s["best_rr"], 2.5)
            self.assertTrue(s["rr_ok"])
            self.assertIsNotNone(s["min_rr_tp"])
            self.assertGreaterEqual(s["rr"], s["rr_nearest"])

    def test_risk_plan_honours_custom_min_rr(self):
        """A floor no ladder target can reach must be reported as not met
        (honest negative, not a silent pass)."""
        from tests.test_dip import make_uptrend_dip

        report = self._report(make_uptrend_dip)
        from src.risk.run import risk_plan_from_report

        plan = risk_plan_from_report(report, "TEST", min_rr=10.0)
        if not plan.get("setup"):
            self.skipTest("no actionable dip setup in the fixture")
        s = plan["setup"]
        self.assertFalse(s["rr_ok"])
        self.assertEqual(s["rr"], s["rr_nearest"])


class TestEngineSizingModes(unittest.TestCase):
    def _run(self, params):
        df = make_ohlcv([100, 101, 102, 100, 101, 102, 104, 106, 108, 110], atr=2.0)
        sig = make_signal(
            df, [0, 0, 1, 0, 0, 0, 0, 0, 0, 0], entry_lo=100.0, inv=95.0, res=200.0
        )
        return run_backtest(sig, df, params, symbol="TEST")

    def test_fractional_mode(self):
        res = self._run(
            BacktestParams(initial_capital=100_000, risk_pct=0.01, max_hold=3)
        )
        self.assertAlmostEqual(res.trades[0].qty, 200.0)

    def test_voltarget_mode_differs(self):
        # vol_target 0.3% with ATR/price ~2% -> qty well under the 1% risk cap
        res = self._run(
            BacktestParams(
                sizing_mode="voltarget", vol_target=0.003, risk_pct=0.01, max_hold=3
            )
        )
        q = res.trades[0].qty
        self.assertGreater(q, 0)
        self.assertLess(q, 200.0)  # smaller than fractional 1%

    def test_kelly_mode_no_edge_no_trades(self):
        res = self._run(
            BacktestParams(
                sizing_mode="kelly",
                kelly_p=0.4,
                payoff=1.0,
                kelly_fraction=0.5,
                max_hold=3,
            )
        )
        self.assertEqual(len(res.trades), 0)

    def test_kelly_mode_with_edge(self):
        # f = 0.5*(0.51 - 0.49/1) = 0.01 -> qty 200; cap 2% would be 400
        res = self._run(
            BacktestParams(
                sizing_mode="kelly",
                kelly_p=0.51,
                payoff=1.0,
                kelly_fraction=0.5,
                risk_pct=0.02,
                max_hold=3,
            )
        )
        self.assertAlmostEqual(res.trades[0].qty, 200.0)

    def test_engine_accepts_all_modes(self):
        for mode in ("fractional", "voltarget", "kelly"):
            res = self._run(BacktestParams(sizing_mode=mode, max_hold=3))
            self.assertIn(res.stats["n_trades"], (0, 1))


if __name__ == "__main__":
    unittest.main()
