"""
Tests for the unified Trade Plan / Decision layer (src/analysis/plan.py).

The plan must answer for ANY symbol: which side (long/short/neutral),
what status (CONFIRMED/WATCH/NO_SETUP), the exact limit zone to act on,
and what would change the read - never silence.
"""

import io
import sys
import unittest
from contextlib import redirect_stdout
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import pandas as pd

from src.analysis.plan import (
    format_plan,
    scanner_action,
    trade_plan,
)


def _ohlc(closes, vol=1000.0):
    closes = pd.Series(list(closes), dtype=float)
    n = len(closes)
    df = pd.DataFrame(
        {
            "open": closes,
            "high": closes + 0.2,
            "low": closes - 0.2,
            "close": closes,
            "volume": pd.Series([vol] * n),
        }
    )
    df.index = pd.date_range("2020-01-01", periods=n, freq="D")
    return df


def _mk_report(**over):
    """A minimal report dict with every key trade_plan reads."""
    report = {
        "symbol": "TEST",
        "last_date": "2026-08-12",
        "last_close": 1.1000,
        "dip": {
            "dip_stage": "No Uptrend",
            "dip_score": 2,
            "dip_confirmed": False,
            "entry_zone": None,
            "invalidation_level": None,
            "target": None,
        },
        "rally": {
            "rally_stage": "No Downtrend",
            "rally_score": 2,
            "rally_confirmed": False,
            "entry_zone": None,
            "invalidation_level": None,
            "target": None,
        },
        "ml": {"prob_pct": 50.0, "label": "Neutral"},
        "ml_short": {"prob_pct": 50.0, "label": "Neutral"},
        "rating": {"prob_pct": 55.0, "rating": "Neutral"},
        "macro": {"gate": {"allowed": True}, "bias": {"bias": 0.0, "label": "Neutral"}},
        "levels": {
            "nearest_support": {"price": 1.0800, "score": 3},
            "nearest_resistance": {"price": 1.1200, "score": 3},
        },
        "risk": {"setup": None},
        "short_risk": {"setup": None},
        "targets": {},
        "short_targets": {},
    }
    report.update(over)
    return report


class TestDecidePlan(unittest.TestCase):
    def test_confirmed_long_gives_buy_limit(self):
        report = _mk_report(
            dip={
                "dip_stage": "Confirmed",
                "dip_score": 7,
                "dip_confirmed": True,
                "entry_zone": (1.1000, 1.1050),
                "invalidation_level": 1.0950,
                "target": 1.1200,
            }
        )
        plan = trade_plan(report)
        self.assertEqual(plan["direction"], "long")
        self.assertEqual(plan["status"], "CONFIRMED")
        self.assertEqual(plan["action"], "BUY-LIMIT 1.10000-1.10500")
        self.assertTrue(plan["long"]["active"])
        self.assertEqual(plan["long"]["stop"], 1.0950)
        self.assertIn("trigger", plan["what_changes"])

    def test_confirmed_short_gives_sell_limit(self):
        report = _mk_report(
            rally={
                "rally_stage": "Confirmed",
                "rally_score": 7,
                "rally_confirmed": True,
                "entry_zone": (1.1150, 1.1200),
                "invalidation_level": 1.1250,
                "target": 1.0900,
            }
        )
        plan = trade_plan(report)
        self.assertEqual(plan["direction"], "short")
        self.assertEqual(plan["status"], "CONFIRMED")
        self.assertEqual(plan["action"], "SELL-LIMIT 1.11500-1.12000")
        self.assertTrue(plan["short"]["active"])

    def test_watch_long_when_in_pullback(self):
        report = _mk_report(
            dip={
                "dip_stage": "In Pullback",
                "dip_score": 5,
                "dip_confirmed": False,
                "entry_zone": (1.1020, 1.1040),
                "invalidation_level": 1.0980,
                "target": 1.1150,
            }
        )
        plan = trade_plan(report)
        self.assertEqual(plan["status"], "WATCH")
        self.assertEqual(plan["direction"], "long")
        self.assertEqual(plan["action"], "WAIT-LONG 1.10200-1.10400")
        self.assertFalse(plan["long"]["active"])
        self.assertTrue(plan["long"]["watch"])

    def test_watch_short_when_in_rally(self):
        report = _mk_report(
            rally={
                "rally_stage": "In Rally",
                "rally_score": 5,
                "rally_confirmed": False,
                "entry_zone": (1.1160, 1.1190),
                "invalidation_level": 1.1230,
                "target": 1.0950,
            }
        )
        plan = trade_plan(report)
        self.assertEqual(plan["status"], "WATCH")
        self.assertEqual(plan["direction"], "short")
        self.assertEqual(plan["action"], "WAIT-SHORT 1.11600-1.11900")

    def test_no_setup_when_nothing_qualifies(self):
        plan = trade_plan(_mk_report())
        self.assertEqual(plan["status"], "NO_SETUP")
        self.assertEqual(plan["direction"], "neutral")
        self.assertEqual(plan["action"], "NO-SETUP")
        self.assertIn("stand aside", plan["summary"])
        self.assertIn("200-SMA", plan["what_changes"])

    def test_macro_block_flags_gate(self):
        report = _mk_report(
            macro={"gate": {"allowed": False}, "bias": {"bias": -1.0, "label": "Bear"}},
            dip={
                "dip_stage": "Confirmed",
                "dip_score": 7,
                "dip_confirmed": True,
                "entry_zone": (1.1000, 1.1050),
                "invalidation_level": 1.0950,
                "target": 1.1200,
            },
        )
        plan = trade_plan(report)
        self.assertEqual(plan["gate"], "BLOCKED")
        self.assertIn("macro BLOCKED", plan["action"])

    def test_low_score_confirmed_is_not_confirmed(self):
        report = _mk_report(
            dip={
                "dip_stage": "In Pullback",
                "dip_score": 3,
                "dip_confirmed": False,
                "entry_zone": (1.1020, 1.1040),
                "invalidation_level": 1.0980,
                "target": 1.1150,
            }
        )
        plan = trade_plan(report)
        self.assertEqual(plan["status"], "NO_SETUP")  # below watch floor

    def test_both_watch_tie_break_by_rating(self):
        report = _mk_report(
            dip={
                "dip_stage": "In Pullback",
                "dip_score": 5,
                "dip_confirmed": False,
                "entry_zone": (1.1020, 1.1040),
                "invalidation_level": 1.0980,
                "target": 1.1150,
            },
            rally={
                "rally_stage": "In Rally",
                "rally_score": 5,
                "rally_confirmed": False,
                "entry_zone": (1.1160, 1.1190),
                "invalidation_level": 1.1230,
                "target": 1.0950,
            },
            rating={"prob_pct": 35.0, "rating": "Sell"},
        )
        plan = trade_plan(report)
        self.assertEqual(plan["status"], "WATCH")
        self.assertEqual(plan["direction"], "short")  # rating leans bearish


class TestScannerAction(unittest.TestCase):
    def test_flat_row_confirmed_long(self):
        row = {
            "symbol": "T",
            "dip_stage": "Confirmed",
            "dip_score": 7,
            "dip_confirmed": "Yes",
            "entry_zone": "1.10000-1.10500",
            "invalidation": 1.0950,
            "resistance": 1.1200,
            "best_rr": 2.5,
            "rally_stage": "No Downtrend",
            "rally_score": 2,
            "rally_confirmed": "No",
            "short_entry_zone": None,
            "short_invalidation": None,
            "support": 1.0900,
            "short_best_rr": None,
            "ml_prob": 62.0,
            "ml_short_prob": 30.0,
            "rating": "Buy",
            "macro_gate": "PASS",
        }
        self.assertEqual(scanner_action(row), "BUY-LIMIT 1.10000-1.10500")

    def test_flat_row_no_setup(self):
        row = {
            "symbol": "T",
            "dip_stage": "No Uptrend",
            "dip_score": 2,
            "dip_confirmed": "No",
            "entry_zone": None,
            "invalidation": None,
            "resistance": 1.1200,
            "best_rr": None,
            "rally_stage": "No Downtrend",
            "rally_score": 1,
            "rally_confirmed": "No",
            "short_entry_zone": None,
            "short_invalidation": None,
            "support": 1.0900,
            "short_best_rr": None,
            "ml_prob": 50.0,
            "ml_short_prob": 50.0,
            "rating": "Neutral",
            "macro_gate": None,
        }
        self.assertEqual(scanner_action(row), "NO-SETUP")


class TestReportIntegration(unittest.TestCase):
    def test_generate_full_report_embeds_plan(self):
        from src.analysis.report import generate_full_report, print_report
        from src.features.indicators import add_all_indicators
        from src.features.regime import detect_regime

        df = add_all_indicators(_ohlc([100 + 0.15 * i for i in range(400)]))
        df = detect_regime(df)
        report = generate_full_report(df, symbol="T", mtf=False)
        self.assertIn("plan", report)
        plan = report["plan"]
        self.assertIn(plan["status"], {"CONFIRMED", "WATCH", "NO_SETUP"})
        self.assertIn(plan["direction"], {"long", "short", "neutral"})
        self.assertTrue(plan["action"])
        self.assertTrue(plan["what_changes"])

        buf = io.StringIO()
        with redirect_stdout(buf):
            print_report(report)
        self.assertIn("11c. TRADE PLAN / ACTION", buf.getvalue())
        self.assertIn(plan["action"], buf.getvalue())

    def test_format_plan_renders(self):
        plan = trade_plan(
            _mk_report(
                dip={
                    "dip_stage": "In Pullback",
                    "dip_score": 5,
                    "dip_confirmed": False,
                    "entry_zone": (1.1020, 1.1040),
                    "invalidation_level": 1.0980,
                    "target": 1.1150,
                }
            )
        )
        text = format_plan(plan)
        self.assertIn(plan["action"], text)
        self.assertIn("What changes", text)


if __name__ == "__main__":
    unittest.main()
