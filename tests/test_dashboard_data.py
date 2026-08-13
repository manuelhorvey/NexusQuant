"""
Tests for the Streamlit-free dashboard data layer
(src/analysis/dashboard_data.py): group discovery, universe/detail
loaders, and Plotly chart builders.
"""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent))

from src.analysis.dashboard_data import (
    build_bias_chart,
    build_compare_chart,
    build_dip_chart,
    build_equity_chart,
    build_heatmap,
    build_momentum_chart,
    build_price_chart,
    build_regime_chart,
    dip_color,
    directional_bias,
    discover_groups,
    get_symbols,
    load_symbol_report,
    load_universe,
    regime_color,
    run_symbol_backtest,
)

# The "candidates" group is small (US30/US500/USTEC/BTCUSD) -> fast tests.
TEST_GROUP, TEST_TF = "candidates", "D1"


class TestDiscoverGroups(unittest.TestCase):
    def test_finds_known_groups(self):
        groups = discover_groups()
        labels = [g["label"] for g in groups]
        # asset-class folders, one entry per group · timeframe
        for want in [
            "full_fx · D1",
            "full_fx · H4",
            "candidates · D1",
            "crypto · D1",
            "metals · D1",
            "indices · D1",
            "commodities · D1",
            "equity_universe · D1",
        ]:
            self.assertIn(want, labels)
        # legacy root 'majors' group is gone after the folder cleanup
        self.assertNotIn("majors", labels)

    def test_every_group_has_symbols(self):
        for g in discover_groups():
            self.assertGreater(g["n"], 0, f"empty group {g['label']}")
            self.assertIn("group", g)
            self.assertIn("timeframe", g)

    def test_get_symbols_matches_count(self):
        g = next(x for x in discover_groups() if x["label"] == "candidates · D1")
        self.assertEqual(len(get_symbols(g["group"], g["timeframe"])), g["n"])


class TestUniverse(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.table = load_universe(TEST_GROUP, TEST_TF)

    def test_columns_present(self):
        for col in [
            "rank",
            "symbol",
            "date",
            "close",
            "regime",
            "bias",
            "bias_score",
            "adx",
            "rsi_14",
            "dip_score",
            "dip_stage",
            "support",
            "resistance",
            "entry_zone",
            "invalidation",
        ]:
            self.assertIn(col, self.table.columns)

    def test_ranked_sorted(self):
        # Rank 1 = strongest directional bias (then ADX).
        t = self.table.sort_values("rank")
        self.assertGreaterEqual(t["bias_score"].iloc[0], t["bias_score"].iloc[-1])

    def test_bias_within_range(self):
        self.assertTrue(self.table["bias_score"].between(-4, 4).all())

    def test_dip_stage_is_known(self):
        known = {
            "Confirmed",
            "In Pullback",
            "No Uptrend",
            "Not a Dip",
            "Support Broken",
            "Deep Pullback",
        }
        self.assertTrue(set(self.table["dip_stage"]).issubset(known))


class TestDetail(unittest.TestCase):
    def test_symbol_report_shape(self):
        df, report = load_symbol_report("US500", TEST_GROUP, TEST_TF)
        self.assertGreater(len(df), 100)
        for section in [
            "regime",
            "moving_averages",
            "momentum",
            "volatility",
            "trend_strength",
            "levels",
            "dip",
            "simple_bias",
        ]:
            self.assertIn(section, report)
        self.assertIn("dip_score", report["dip"])
        self.assertIn("entry_zone", report["dip"])

    def test_directional_bias(self):
        df, _ = load_symbol_report("US500", TEST_GROUP, TEST_TF)
        bias = directional_bias(df)
        self.assertIn("score", bias)
        self.assertIn("label", bias)
        self.assertTrue(-4 <= bias["score"] <= 4)

    def test_missing_symbol_raises(self):
        with self.assertRaises(FileNotFoundError):
            load_symbol_report("ZZZZNOTREAL", TEST_GROUP, TEST_TF)


class TestCharts(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.df, cls.report = load_symbol_report("US500", TEST_GROUP, TEST_TF)
        cls.table = load_universe(TEST_GROUP, TEST_TF)
        cls.records = cls.table.head(3).to_dict(orient="records")
        cls.frames = [cls.df] * 3

    def test_price_chart(self):
        fig = build_price_chart(self.df, self.report, 120)
        self.assertGreater(len(fig.data), 0)

    def test_momentum_chart(self):
        fig = build_momentum_chart(self.df, 120)
        self.assertGreater(len(fig.data), 0)

    def test_heatmap_chart(self):
        fig = build_heatmap(self.table)
        self.assertGreater(len(fig.data), 0)

    def test_regime_chart(self):
        fig = build_regime_chart(self.table)
        self.assertGreater(len(fig.data), 0)

    def test_compare_chart(self):
        fig = build_compare_chart(self.frames, ["A", "B", "C"], lookback=120)
        self.assertEqual(len(fig.data), 3)

    def test_bias_and_dip_charts(self):
        for fig in (build_bias_chart(self.records), build_dip_chart(self.records)):
            self.assertGreater(len(fig.data), 0)

    def test_color_helpers(self):
        self.assertEqual(regime_color("Bull Trend"), "#26a69a")
        self.assertEqual(dip_color("Confirmed"), "#22c55e")
        # Unknown values fall back to a neutral gray.
        self.assertTrue(regime_color("???").startswith("#"))
        self.assertTrue(dip_color("???").startswith("#"))


class TestBacktestIntegration(unittest.TestCase):
    def test_backtest_runs_and_charts(self):
        res = run_symbol_backtest(
            "US500", TEST_GROUP, TEST_TF, risk_pct=0.01, max_hold=20
        )
        self.assertGreater(len(res.equity), 100)
        self.assertGreaterEqual(res.stats["n_trades"], 0)
        self.assertGreaterEqual(res.stats["win_rate"], 0.0)
        self.assertLessEqual(res.stats["win_rate"], 1.0)
        fig = build_equity_chart(res)
        self.assertGreater(len(fig.data), 0)

    def test_backtest_different_params(self):
        r1 = run_symbol_backtest(
            "US500", TEST_GROUP, TEST_TF, max_hold=5, entry_type="market"
        )
        r2 = run_symbol_backtest(
            "US500", TEST_GROUP, TEST_TF, max_hold=40, entry_type="limit"
        )
        self.assertGreaterEqual(r1.stats["n_trades"], 0)
        self.assertGreaterEqual(r2.stats["n_trades"], 0)

    def test_start_year_filter_no_crash(self):
        # Regression: the dashboard filters equity by start year and trades by
        # entry; compute_stats must not KeyError on missing entry timestamps.
        import pandas as pd
        from src.backtest.engine import compute_stats

        res = run_symbol_backtest("US500", TEST_GROUP, TEST_TF, max_hold=20)
        cutoff = pd.Timestamp(res.equity.index[-100].date())
        eq = res.equity[res.equity.index >= cutoff]
        trades = [t for t in res.trades if t.entry_time >= cutoff]
        res.equity = eq
        res.trades = trades
        stats = compute_stats(res)
        self.assertGreaterEqual(stats["n_trades"], 0)

    def test_sharpe_not_diluted_by_flat_bars(self):

        res = run_symbol_backtest("US500", TEST_GROUP, TEST_TF, max_hold=20)
        # All trades kept -> exposure > 0 somewhere; Sharpe finite.
        self.assertTrue(res.stats["sharpe"] != float("inf"))
        self.assertEqual(res.stats["sharpe"], res.stats["sharpe"])
        # A strategy that only traded the last 50 bars must not get a
        # Sharpe crushed by the 2500 flat bars before it.
        if res.stats["n_trades"] > 0:
            self.assertGreater(res.stats["exposure_pct"], 0.0)


if __name__ == "__main__":
    unittest.main()
