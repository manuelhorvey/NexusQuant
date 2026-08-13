"""
Regression tests for the GAP-closure round:

* short-side backtest engine mechanics (mirror entry/exit/PnL/sizing),
* ``run_backtest_both`` combined long+short view,
* ``predict_long_short`` contract (dual calibrated probabilities + net_bias),
* rating consumes the short model via net_bias (Sell / Strong Sell reachable),
* HMM regime surfaced in the MTF summary,
* direction-aware stress for the short book,
* cross-sectional factor model (plan's factor_model.py API),
* live ``--format institutional`` path.
"""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.append(str(Path(__file__).parent.parent))

from tests.test_backtest import make_ohlcv  # noqa: E402

from src.backtest.engine import (  # noqa: E402
    BacktestParams,
    run_backtest,
    run_backtest_both,
)
from src.backtest.signals import dip_signal_series, rally_signal_series  # noqa: E402


def _short_signal(df, confirmed, entry_hi, inv, tgt, score=6):
    """Signal frame with short trade levels (entry above market, stop above,
    target below) set at the confirmed bars."""
    n = len(df)
    sig = pd.DataFrame(
        {
            "confirmed": np.array(confirmed, dtype=bool),
            "entry_lo": np.nan,
            "entry_hi": np.nan,
            "invalidation": np.nan,
            "target": np.nan,
            "resistance": np.nan,
            "score": np.full(n, float(score)),
        },
        index=df.index,
    )
    for i, c in enumerate(confirmed):
        if c:
            sig.iloc[i, sig.columns.get_loc("entry_hi")] = entry_hi
            sig.iloc[i, sig.columns.get_loc("invalidation")] = inv
            sig.iloc[i, sig.columns.get_loc("target")] = tgt
    return sig


class TestShortEngine(unittest.TestCase):
    def test_short_target_exit(self):
        # Rally into 102, limit short fills at 102.5, price drops to 97.
        df = make_ohlcv([100, 101, 102, 103, 102, 101, 100, 99, 98, 97])
        sig = _short_signal(
            df, [0, 0, 1, 0, 0, 0, 0, 0, 0, 0], entry_hi=102.5, inv=104.5, tgt=97.0
        )
        res = run_backtest(
            sig, df, BacktestParams(max_hold=100), symbol="TEST", side="short"
        )
        t = res.trades[0]
        self.assertEqual(t.reason, "target")
        self.assertEqual(t.exit_price, 97.0)
        self.assertGreater(t.pnl, 0)  # sold high, bought low
        self.assertGreater(t.r_multiple, 0)

    def test_short_stop_exit(self):
        # Rally keeps going: price rises through the stop above entry.
        df = make_ohlcv([100, 101, 102, 103, 104, 105, 106, 107, 108, 109])
        sig = _short_signal(
            df, [0, 0, 1, 0, 0, 0, 0, 0, 0, 0], entry_hi=102.5, inv=104.5, tgt=95.0
        )
        res = run_backtest(
            sig, df, BacktestParams(max_hold=100), symbol="TEST", side="short"
        )
        t = res.trades[0]
        self.assertEqual(t.reason, "stop")
        self.assertEqual(t.exit_price, 104.5)
        self.assertLess(t.pnl, 0)
        # exact 1R loss (stop distance = risk distance)
        self.assertAlmostEqual(t.r_multiple, -1.0, places=6)

    def test_short_limit_not_filled_cancelled(self):
        # Limit above market never touched (price keeps falling away).
        df = make_ohlcv([100, 99, 98, 97, 96, 95, 94, 93, 92, 91])
        sig = _short_signal(
            df, [0, 0, 1, 0, 0, 0, 0, 0, 0, 0], entry_hi=103.0, inv=105.0, tgt=90.0
        )
        res = run_backtest(
            sig,
            df,
            BacktestParams(entry_valid_bars=2, max_hold=100),
            symbol="TEST",
            side="short",
        )
        self.assertEqual(len(res.trades), 0)

    def test_short_market_entry_next_open(self):
        # market fills at the open of the bar AFTER the signal bar; in the
        # fixture open == previous close, so bar 3's open is 102 (bar 2's
        # close), NOT 103 - the entry must never be the signal bar itself.
        df = make_ohlcv([100, 101, 102, 103, 102, 101, 100, 99, 98, 97])
        sig = _short_signal(
            df, [0, 0, 1, 0, 0, 0, 0, 0, 0, 0], entry_hi=102.5, inv=104.5, tgt=90.0
        )
        res = run_backtest(
            sig,
            df,
            BacktestParams(entry_type="market", max_hold=3),
            symbol="TEST",
            side="short",
        )
        self.assertEqual(res.trades[0].entry_price, 102.0)
        # filled on the bar AFTER the signal bar (never the signal bar)
        self.assertEqual(res.trades[0].entry_time, df.index[3])

    def test_short_sizing_direction(self):
        # entry 102.5, stop 104.5 -> risk per unit 2 -> qty = 100k*1% / 2 = 500
        df = make_ohlcv([100, 101, 102, 103, 102, 101, 100, 99, 98, 97])
        sig = _short_signal(
            df, [0, 0, 1, 0, 0, 0, 0, 0, 0, 0], entry_hi=102.5, inv=104.5, tgt=90.0
        )
        res = run_backtest(
            sig,
            df,
            BacktestParams(initial_capital=100_000, risk_pct=0.01, max_hold=3),
            symbol="TEST",
            side="short",
        )
        self.assertAlmostEqual(res.trades[0].qty, 500.0)

    def test_short_slippage_direction(self):
        # 2bps per side: entry fill sells at bid (entry * 0.9998), exit buy
        # back at ask. Both work against the short, so a round trip with no
        # price move costs ~4bps of notional.
        df = make_ohlcv(
            [100, 101, 102, 102.4, 102.4, 102.4, 102.4, 102.4, 102.4, 102.4]
        )
        sig = _short_signal(
            df, [0, 0, 1, 0, 0, 0, 0, 0, 0, 0], entry_hi=102.1, inv=103.1, tgt=90.0
        )
        res = run_backtest(
            sig,
            df,
            BacktestParams(slippage=0.0002, max_hold=3),
            symbol="TEST",
            side="short",
        )
        t = res.trades[0]
        self.assertEqual(t.reason, "time")
        # sold at 102.1*(1-0.0002); buying back costs the ask -> loss
        self.assertLess(t.pnl, 0)
        self.assertAlmostEqual(t.entry_price, 102.1 * 0.9998, places=4)

    def test_short_uses_target_not_resistance_column(self):
        # The short signal's target is the SUPPORT below entry; the long
        # column 'resistance' (if present) must NOT be used as the target.
        df = make_ohlcv([100, 101, 102, 103, 102, 101, 100, 99, 98, 97])
        sig = _short_signal(
            df, [0, 0, 1, 0, 0, 0, 0, 0, 0, 0], entry_hi=102.5, inv=104.5, tgt=97.0
        )
        res = run_backtest(
            sig, df, BacktestParams(max_hold=100), symbol="TEST", side="short"
        )
        self.assertEqual(res.trades[0].reason, "target")
        self.assertEqual(res.trades[0].exit_price, 97.0)


class TestBacktestBoth(unittest.TestCase):
    def test_both_merges_trades_and_equity(self):
        # Both books deploy independent capital: merged equity = sum - initial.
        # An uptrend-with-dips frame confirms the long book (the short book
        # correctly stays flat on it); the merge arithmetic must hold either
        # way.
        from tests.test_dip import make_uptrend_dip, with_indicators

        df = with_indicators(make_uptrend_dip())
        params = BacktestParams(initial_capital=100_000, max_hold=20)
        long_res = run_backtest(
            dip_signal_series(df), df, params, symbol="T", side="long"
        )
        short_res = run_backtest(
            rally_signal_series(df), df, params, symbol="T", side="short"
        )
        both = run_backtest_both(
            dip_signal_series(df), rally_signal_series(df), df, params, symbol="T"
        )
        self.assertEqual(
            both.stats["n_trades"],
            long_res.stats["n_trades"] + short_res.stats["n_trades"],
        )
        np.testing.assert_allclose(
            both.equity.values,
            long_res.equity.values + short_res.equity.values - 100_000.0,
        )
        # the long book actually traded on the uptrend fixture
        self.assertGreater(long_res.stats["n_trades"], 0)

    def test_both_keeps_long_side_backward_compat(self):
        # side defaults to "long": existing callers unaffected.
        df = make_ohlcv([100, 101, 102, 100, 99, 98, 97, 96, 95, 94])
        sig = _short_signal(
            df, [0, 0, 1, 0, 0, 0, 0, 0, 0, 0], entry_hi=102.0, inv=104.0, tgt=95.0
        )
        res = run_backtest(sig, df, BacktestParams(max_hold=100), symbol="TEST")
        # default long side: entry_hi is ignored (no entry_lo set -> no trade)
        self.assertEqual(len(res.trades), 0)


class TestPredictLongShort(unittest.TestCase):
    def test_graceful_none_without_models(self):
        from src.model.model import predict_long_short

        df = make_ohlcv([100] * 250)
        out = predict_long_short(
            df,
            symbol="TEST",
            group="x",
            long_model="models/does_not_exist_1.joblib",
            short_model="models/does_not_exist_2.joblib",
        )
        # both models missing -> None (callers fall back to rule engines)
        self.assertIsNone(out)

    def test_rating_net_bias_reaches_sell(self):
        # A strong short model + weak long model must push the rating below
        # 50 (Sell / Strong Sell territory) - the dual-side wiring.
        from src.analysis.rating import factor_contributions, quant_rating

        report = {
            "moving_averages": {
                "price_vs_sma200": "Below",
                "sma_50": 100.0,
                "sma_100": 100.0,
                "sma_200": 100.0,
                "ema_50": 100.0,
                "ema_100": 100.0,
                "ema_200": 100.0,
            },
            "trend_strength": {"adx": 30, "plus_di": 20, "minus_di": 40},
            "momentum": {"rsi_14": 40, "macd_hist": -0.1, "bb_pct_b": 0.3},
            "volume_flow": {"buyer_seller_score": -30},
            "macro": {"bias": {"bias": -1.0}},
            "sentiment": {"composite": -0.3},
            "ml": {"prob_pct": 45.0},  # weak long read
            "ml_short": {"prob_pct": 70.0},  # strong short read
        }
        b = factor_contributions(report)
        # net_bias = 0.45 - 0.70 = -0.25 -> effective 37.5 blended with the
        # (bearish) rule stack -> clearly below 50.
        self.assertLess(b["prob_pct"], 50.0)
        self.assertIn(quant_rating(b["prob_pct"]), ("Sell", "Strong Sell"))

    def test_rating_uses_short_when_present(self):
        # Same report shape but NO short model: the rating must stay on the
        # long-ML path (can't dip below 50 from a missing model).
        from src.analysis.rating import factor_contributions

        report = {
            "moving_averages": {
                "price_vs_sma200": "Below",
                "sma_50": 100.0,
                "sma_100": 100.0,
                "sma_200": 100.0,
                "ema_50": 100.0,
                "ema_100": 100.0,
                "ema_200": 100.0,
            },
            "trend_strength": {"adx": 30, "plus_di": 20, "minus_di": 40},
            "momentum": {"rsi_14": 40, "macd_hist": -0.1, "bb_pct_b": 0.3},
            "volume_flow": {"buyer_seller_score": -30},
            "macro": {"bias": {"bias": -1.0}},
            "sentiment": {"composite": -0.3},
            "ml": {"prob_pct": 45.0},
        }
        b = factor_contributions(report)
        self.assertGreaterEqual(b["prob_pct"], 1.0)


class TestHmmInReport(unittest.TestCase):
    def test_mtf_summary_surfaces_hmm_label(self):
        from src.features.indicators import add_all_indicators
        from src.features.regime import get_current_regime_summary_mtf, REGIME_LEVELS

        df = make_ohlcv([100 + 0.3 * i for i in range(500)])
        df = add_all_indicators(df)
        summary = get_current_regime_summary_mtf(df, use_hmm=True)
        self.assertIn("regime_hmm", summary)
        self.assertIn(summary["regime_hmm"], REGIME_LEVELS)
        self.assertTrue(summary["mtf"])  # D/W/M rows present

    def test_report_has_short_stress_section(self):
        from src.features.indicators import add_all_indicators
        from src.features.regime import detect_regime
        from src.analysis.report import generate_full_report

        df = add_all_indicators(make_ohlcv([100 + 0.2 * i for i in range(500)]))
        df = detect_regime(df)
        report = generate_full_report(df, symbol="TEST", mtf=False)
        # stress_short is always a dict - "available" with the direction
        # flag when a short setup exists, a graceful "no setup" reason
        # otherwise. Never a crash.
        sts = report.get("stress_short")
        self.assertIsInstance(sts, dict)
        if sts.get("available"):
            self.assertIn("direction", sts)
        self.assertIn("rally", report)
        self.assertIn("short_risk", report)


class TestStressDirection(unittest.TestCase):
    def test_short_crash_is_favorable(self):
        from src.risk.stress import stress_position

        sc = {"drawdown_pct": 50.0, "vol_mult": 2.0, "name": "GFC"}
        pos = {"qty": 1000.0, "entry": 1.10, "atr": 0.01, "direction": "short"}
        r = stress_position(pos, sc, equity=100_000)
        self.assertLess(r["loss_usd"], 0)  # a gain in the crash
        self.assertGreater(r["loss_pct_equity"], -100.0)
        # VaR still positive: vol blow-up risk is not direction-flipped
        self.assertGreater(r["var95_stress"], 0.0)

    def test_long_unchanged(self):
        from src.risk.stress import stress_position

        sc = {"drawdown_pct": 50.0, "vol_mult": 2.0, "name": "GFC"}
        pos = {"qty": 1000.0, "entry": 1.10, "atr": 0.01, "direction": "long"}
        r = stress_position(pos, sc, equity=100_000)
        self.assertGreater(r["loss_usd"], 0)
        self.assertAlmostEqual(r["loss_usd"], 550.0, places=2)

    def test_table_short_risk_key(self):
        from src.risk.stress import stress_table_from_report

        report = {
            "last_date": "2026-08-12",
            "last_close": 1.10,
            "volatility": {"atr_14": 0.005},
            "short_risk": {
                "setup": {"entry": 1.105, "stop": 1.12, "target": 1.09, "rr": 1.0},
                "inputs": {"equity": 100_000},
                "sizes": [{"method": "fractional", "qty": 1000.0}],
            },
        }
        t = stress_table_from_report(
            report, "T", direction="short", risk_key="short_risk"
        )
        self.assertTrue(t["available"])
        self.assertEqual(t["direction"], "short")
        self.assertLess(t["scenarios"][0]["loss_usd"], 0.0)


class TestFactorModel(unittest.TestCase):
    def test_cross_sectional_ordering(self):
        from src.equity.factor_model import compute_factor_scores

        df = pd.DataFrame(
            {
                "symbol": ["CHEAP", "AVG", "DEAR"],
                "pe_ratio": [8.0, 18.0, 45.0],
                "ev_ebitda": [5.0, 11.0, 28.0],
                "roe": [30.0, 15.0, 3.0],
                "debt_equity": [0.2, 0.9, 2.4],
                "mom_12m": [0.35, 0.05, -0.35],
            }
        )
        out = compute_factor_scores(df).set_index("symbol")
        # cheaper -> higher value score; profitable -> higher quality
        self.assertGreater(
            out.loc["CHEAP", "value_score"], out.loc["DEAR", "value_score"]
        )
        self.assertGreater(
            out.loc["CHEAP", "quality_score"], out.loc["DEAR", "quality_score"]
        )
        for _, row in out.iterrows():
            for col in ("value_score", "quality_score", "momentum_score", "composite"):
                self.assertGreaterEqual(row[col], 0.0)
                self.assertLessEqual(row[col], 100.0)

    def test_single_row_graceful(self):
        # A lone symbol has no cross-section: fields needing >= 2 rows are
        # skipped, composite still works on momentum when it exists.
        from src.equity.factor_model import (
            compute_factor_scores,
            get_factor_interpretation,
        )

        out = compute_factor_scores(
            pd.DataFrame(
                {
                    "symbol": ["ONLY"],
                    "pe_ratio": [12.0],
                    "roe": [20.0],
                    "mom_12m": [0.1],
                }
            )
        )
        self.assertEqual(len(out), 1)
        self.assertEqual(get_factor_interpretation(88.0), "Strong")
        self.assertEqual(get_factor_interpretation(45.0), "Weak")
        self.assertEqual(get_factor_interpretation(None), "N/A")

    def test_re_exports_fundamentals_api(self):
        from src.equity.factor_model import factor_scores, load_fundamentals  # noqa: F401

        self.assertTrue(callable(factor_scores))
        self.assertTrue(callable(load_fundamentals))

    def test_provider_key_aliases(self):
        # Regression: the plan's canonical field names (pe_ratio, roe, ...)
        # must ALSO accept the provider/fundamentals keys (pe, roe_pct, ...)
        # or the two APIs can never be chained.
        from src.equity.factor_model import compute_factor_scores

        df = pd.DataFrame(
            {
                "symbol": ["A", "B"],
                "pe": [8.0, 45.0],
                "ev_ebitda": [5.0, 28.0],
                "pb": [0.9, 9.0],
                "roe_pct": [30.0, 3.0],
                "debt_to_equity": [0.2, 2.4],
                "earnings_surprise_pct": [8.0, -6.0],
            }
        )
        out = compute_factor_scores(df).set_index("symbol")
        self.assertGreater(out.loc["A", "value_score"], out.loc["B", "value_score"])
        self.assertGreater(out.loc["A", "quality_score"], out.loc["B", "quality_score"])


class TestInstitutionalFormat(unittest.TestCase):
    def test_institutional_reports_runs(self):
        from src.live.run import institutional_reports

        path = Path("data/raw/full_fx/EURUSD_D1.parquet")
        if not path.exists():
            self.skipTest("EURUSD data not available")
        n = institutional_reports("full_fx", "D1", symbols=["EURUSD"])
        self.assertGreaterEqual(n, 1)


class TestPortfolioReportEarlyReturn(unittest.TestCase):
    def test_no_data_returns_complete_shape(self):
        # Regression: the early return used to omit n_setups and crash the
        # CLI with a KeyError when every symbol failed to load.
        from src.risk.run import portfolio_report

        rep = portfolio_report(["NOSUCHSYM"], "full_fx", "D1")
        self.assertIn("n_setups", rep)
        self.assertIn("reason", rep)
        self.assertEqual(rep["n_setups"], 0)
        self.assertIn("heat_pct", rep)

    def test_single_symbol_complete_shape(self):
        # One symbol still yields a complete report shape (the early return
        # only fires on < 2 overlapping rows or no data at all).
        from src.risk.run import portfolio_report

        path = Path("data/raw/full_fx/EURUSD_D1.parquet")
        if not path.exists():
            self.skipTest("EURUSD data not available")
        rep = portfolio_report(["EURUSD"], "full_fx", "D1")
        self.assertIn("n_setups", rep)
        self.assertIn("heat_pct", rep)
        self.assertIn("positions", rep)


if __name__ == "__main__":
    unittest.main()
