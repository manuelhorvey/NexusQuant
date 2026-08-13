"""Tests for the Stress Testing module (src/risk/stress.py)."""

import unittest

import numpy as np
import pandas as pd

from src.risk.stress import (
    SCENARIOS,
    historical_crash_stats,
    scenario_var,
    stress_portfolio,
    stress_position,
    stress_table_from_report,
)


class TestStressPosition(unittest.TestCase):
    def test_loss_magnitude(self):
        sc = {"drawdown_pct": 50.0, "vol_mult": 2.0, "name": "T"}
        pos = {"qty": 1000.0, "entry": 1.10, "atr": 0.005, "direction": "long"}
        r = stress_position(pos, sc, equity=100_000)
        # 1000 * 1.10 * 0.50 = 550
        self.assertAlmostEqual(r["loss_usd"], 550.0, places=2)
        self.assertAlmostEqual(r["loss_pct_equity"], 0.55, places=2)

    def test_daily_limit_breach_uses_1d_var(self):
        sc = {"drawdown_pct": 50.0, "vol_mult": 2.0, "name": "T", "horizon_days": 200}
        # 1-day shocked VaR = 1.645 * (0.01*2) * qty
        pos = {"qty": 100000.0, "entry": 1.00, "atr": 0.01}
        r = stress_position(pos, sc, equity=50_000)  # 1d VaR 6.58% eq
        self.assertTrue(r["daily_limit_breach"])
        self.assertTrue(r["scenario_cap_breach"])  # loss 100% of eq
        # equity sized so 1-day shocked VaR (3290) is under the 2% limit
        # but the gap-through loss (50k) still exceeds the 20% scenario cap
        r2 = stress_position(pos, sc, equity=200_000)  # 1d VaR 1.65% eq
        self.assertFalse(r2["daily_limit_breach"])
        self.assertTrue(r2["scenario_cap_breach"])  # loss 25% of equity

    def test_loss_scales_with_notional_and_days_of_limit(self):
        sc = {"drawdown_pct": 50.0, "vol_mult": 2.0, "name": "T", "horizon_days": 200}
        pos = {"qty": 1000.0, "entry": 1.10, "atr": 0.005}
        r = stress_position(pos, sc, equity=100_000)
        self.assertAlmostEqual(r["loss_usd"], 550.0, places=2)
        self.assertAlmostEqual(r["days_of_daily_limit"], 0.275, places=3)

    def test_scenario_var_formula_and_horizon_scale(self):
        v = scenario_var(qty=1000.0, atr=0.01, z=1.645, hold_bars=10, vol_mult=2.0)
        self.assertAlmostEqual(v, 1.645 * 0.02 * 1000.0 * np.sqrt(10), places=2)
        v200 = scenario_var(qty=1000.0, atr=0.01, z=1.645, hold_bars=200, vol_mult=2.0)
        self.assertAlmostEqual(v200 / v, np.sqrt(20), places=4)  # sqrt(200/10)


class TestStressPortfolio(unittest.TestCase):
    def test_aggregation(self):
        sc = {"drawdown_pct": 50.0, "vol_mult": 2.0, "name": "T", "horizon_days": 200}
        positions = [
            {"qty": 1000.0, "entry": 1.10, "atr": 0.005},
            {"qty": 500.0, "entry": 2.00, "atr": 0.01},
        ]
        r = stress_portfolio(positions, equity=100_000, scenario=sc)
        self.assertAlmostEqual(r["total_loss_usd"], 550.0 + 500.0, places=2)
        self.assertAlmostEqual(r["total_loss_pct_equity"], 1.05, places=2)
        self.assertFalse(r["daily_limit_breach"])

    def test_breach_propagates(self):
        sc = {"drawdown_pct": 50.0, "vol_mult": 2.0, "name": "T", "horizon_days": 200}
        positions = [
            {"qty": 1000.0, "entry": 1.10, "atr": 0.005},
            {"qty": 100000.0, "entry": 1.00, "atr": 0.01},
        ]
        r = stress_portfolio(positions, equity=50_000, scenario=sc)
        self.assertTrue(r["daily_limit_breach"])

    def test_cap_breach_aggregates(self):
        sc = {"drawdown_pct": 50.0, "vol_mult": 1.0, "name": "T", "horizon_days": 200}
        # small 1-day VaR (low ATR) but gap-through loss = 50% of equity
        positions = [{"qty": 1000.0, "entry": 100.0, "atr": 0.0001}]
        r = stress_portfolio(positions, equity=100_000, scenario=sc)
        self.assertFalse(r["daily_limit_breach"])
        self.assertTrue(r["scenario_cap_breach"])


class TestHistoricalCrashStats(unittest.TestCase):
    def _crisis_frame(self, n=900):
        # 2019 sideways, then a sharp 2020-03 crash, then recovery; long
        # enough (~3.5y) to also cover the 2022 window
        dates = pd.date_range("2019-01-01", periods=n, freq="B")
        rng = np.random.default_rng(3)
        closes = 100 + np.cumsum(rng.normal(0, 0.2, len(dates)))
        crash_start = np.where(dates >= pd.Timestamp("2020-03-01"))[0][0]
        closes[crash_start : crash_start + 10] *= np.linspace(1.0, 0.75, 10)
        return pd.DataFrame(
            {
                "date": dates,
                "close": closes,
                "high": closes + 1.0,
                "low": closes - 1.0,
                "volume": 1000,
            }
        )

    def test_covid_realized(self):
        stats = historical_crash_stats(self._crisis_frame())
        self.assertIsNotNone(stats["covid_dd_pct"])
        self.assertLess(stats["covid_dd_pct"], 0)  # a drawdown, not a gain
        self.assertGreaterEqual(stats["covid_dd_pct"], -40.0)
        self.assertIsNotNone(stats["covid_vol_mult"])
        self.assertGreater(stats["covid_vol_mult"], 1.0)  # vol elevated in crisis

    def test_dd_2022_computed(self):
        stats = historical_crash_stats(self._crisis_frame())
        self.assertIsNotNone(stats["dd_2022_pct"])
        self.assertLess(stats["dd_2022_pct"], 0)

    def test_datetime_index_path(self):
        df = self._crisis_frame().set_index("date")
        stats = historical_crash_stats(df)  # no 'date' column: uses the index
        self.assertIsNotNone(stats["covid_dd_pct"])
        self.assertIsNotNone(stats["covid_vol_mult"])

    def test_short_history_returns_none(self):
        df = pd.DataFrame(
            {
                "close": [1.0, 1.01, 1.02],
                "date": pd.date_range("2024-01-01", periods=3, freq="D"),
            }
        )
        stats = historical_crash_stats(df)
        self.assertIsNone(stats["covid_dd_pct"])
        self.assertIsNone(stats["covid_vol_mult"])

    def test_covid_dd_is_windowed_not_global(self):
        # A deep crash outside the COVID window must NOT be reported as
        # the COVID drawdown: the measure is windowed to Mar-Jun 2020.
        dates = pd.date_range("2019-01-01", periods=400, freq="B")
        closes = 100 + np.cumsum(np.random.default_rng(7).normal(0, 0.1, len(dates)))
        # deep 2019 crash (-40%), then flat into 2020
        closes[50:70] *= np.linspace(1.0, 0.60, 20)
        closes[70:] = closes[70] + np.linspace(0, 2, len(dates) - 70)
        df = pd.DataFrame({"date": dates, "close": closes})
        stats = historical_crash_stats(df)
        # the -40% 2019 crash is OUTSIDE the COVID window, so the reported
        # COVID drawdown must be a small windowed number, not -40
        self.assertIsNotNone(stats["covid_dd_pct"])
        self.assertGreater(stats["covid_dd_pct"], -10.0)

    def test_dd_2022_requires_40_bars(self):
        # just under the 40-bar minimum -> None; crossing it -> a value
        dates = pd.date_range("2021-10-01", periods=39 + 30, freq="B")
        closes = 100 + np.cumsum(np.random.default_rng(2).normal(0, 0.15, len(dates)))
        closes[-20:] *= np.linspace(1.0, 0.90, 20)
        df = pd.DataFrame({"date": dates, "close": closes})
        stats = historical_crash_stats(df)
        self.assertIsNone(stats["dd_2022_pct"])
        df2 = df.iloc[:39].copy()
        df2["date"] = df["date"].iloc[:39].values
        stats2 = historical_crash_stats(df2)
        self.assertIsNone(stats2["dd_2022_pct"])


class TestStressFromReport(unittest.TestCase):
    def _fake_report(self, with_setup=True):
        report = {
            "last_date": "2026-08-12",
            "last_close": 1.10,
            "volatility": {"atr_14": 0.005},
            "risk": {
                "setup": {"entry": 1.095, "stop": 1.08, "target": 1.12, "rr": 1.0},
                "inputs": {"equity": 100_000},
                "sizes": [
                    {"method": "fractional", "qty": 1000.0},
                    {"method": "voltarget", "qty": 500.0},
                    {"method": "kelly", "qty": 700.0},
                ],
            },
        }
        if not with_setup:
            report["risk"] = {"setup": None, "reason": "no actionable setup"}
        return report

    def test_table_from_report(self):
        t = stress_table_from_report(self._fake_report(), "TEST")
        self.assertTrue(t["available"])
        self.assertEqual(len(t["scenarios"]), 3)
        names = [r["scenario"] for r in t["scenarios"]]
        self.assertEqual(names, list(SCENARIOS.keys()))
        self.assertFalse(t["any_breach"])
        # fractional qty used (1000 units); rows carry the new fields
        self.assertEqual(t["qty"], 1000.0)
        row = t["scenarios"][0]
        for key in (
            "loss_usd",
            "var95_stress",
            "var95_1d_pct_equity",
            "daily_limit_breach",
            "scenario_cap_breach",
            "days_of_daily_limit",
        ):
            self.assertIn(key, row)

    def test_no_setup_graceful(self):
        t = stress_table_from_report(self._fake_report(with_setup=False), "T")
        self.assertFalse(t["available"])

    def test_historical_attached_when_df_present(self):
        dates = pd.date_range("2019-01-01", periods=900, freq="B")
        closes = 100 + np.cumsum(np.random.default_rng(4).normal(0, 0.1, len(dates)))
        crash = np.where(dates >= pd.Timestamp("2020-03-01"))[0][0]
        closes[crash : crash + 10] *= np.linspace(1.0, 0.78, 10)
        df = pd.DataFrame({"date": dates, "close": closes})
        t = stress_table_from_report(self._fake_report(), "T", df=df)
        self.assertIn("historical", t)
        # values actually populate, not just a present-but-empty dict
        self.assertIsNotNone(t["historical"]["covid_dd_pct"])
        self.assertIsNotNone(t["historical"]["covid_vol_mult"])
        self.assertIsNotNone(t["historical"]["dd_2022_pct"])


if __name__ == "__main__":
    unittest.main()
