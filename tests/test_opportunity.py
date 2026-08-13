"""
Tests for the unified Opportunity Book + EV-aware decision engine
(src/analysis/opportunity.py) and the portfolio currency-leg exposure
(src/risk/run.currency_exposure).

Covers the campaign acceptance criteria: explicit direction, EV-driven
LONG/SHORT/FLAT, no fabricated probabilities, explainable rejections,
cost-aware EV, and the JPY-cross directional concentration view.
"""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent))

from src.analysis.opportunity import (
    build_opportunity_book,
    format_opportunity_book,
    roundtrip_cost_r,
)
from src.risk.run import currency_exposure


def _report(symbol: str = "EURUSD", **overrides) -> dict:
    """Minimal report dict with the keys the opportunity book reads."""
    rep = {
        "symbol": symbol,
        "setup_classification": {
            "direction": "long",
            "long_families": {"LONG_BREAKOUT": 0.4, "LONG_BUY_DIP": 0.2},
            "short_families": {"SHORT_BREAKDOWN": 0.3, "SHORT_SELL_RALLY": 0.1},
            "prob_long": 0.6,
            "prob_short": 0.3,
            "confidence": 0.4,
        },
        "regime": {"regime": "Bull Trend"},
        "dip": {"dip_confirmed": False, "dip_score": 3},
        "rally": {"rally_confirmed": False, "rally_score": 2},
        "risk": {
            "setup": {"entry": 1.10, "stop": 1.09, "target": 1.15, "rr": 1.0},
        },
        "short_risk": {
            "setup": {"entry": 1.12, "stop": 1.13, "target": 1.08, "rr": 1.0},
        },
        "targets": {
            "best_rr": 3.0,
            "min_rr_tp": "TP3",
            "targets": [
                {"target": "TP1", "rr": 1.0},
                {"target": "TP2", "rr": 2.0},
                {"target": "TP3", "rr": 3.0},
            ],
        },
        "short_targets": {
            "best_rr": 2.0,
            "min_rr_tp": "TP2",
            "targets": [
                {"target": "TP1", "rr": 1.0},
                {"target": "TP2", "rr": 2.0},
            ],
        },
        "macro": {"gate": {"allowed": True, "reason": "ok"}},
    }
    rep.update(overrides)
    return rep


class TestOpportunityBook(unittest.TestCase):
    def test_ev_path_picks_long(self):
        rep = _report()  # P(long)=0.6 vs P(short)=0.3
        book = build_opportunity_book(rep)
        self.assertEqual(book["verdict"]["direction"], "long")
        self.assertIn(book["verdict"]["status"], {"TRADE", "RULE"})
        self.assertTrue(book["long"]["taken"])
        self.assertFalse(book["short"]["taken"])

    def test_ev_path_picks_short_when_higher(self):
        rep = _report(
            **{
                "setup_classification": {
                    "direction": "short",
                    "long_families": {"LONG_BREAKOUT": 0.2},
                    "short_families": {"SHORT_BREAKDOWN": 0.5},
                    "prob_long": 0.3,
                    "prob_short": 0.7,
                    "confidence": 0.5,
                }
            }
        )
        book = build_opportunity_book(rep)
        self.assertEqual(book["verdict"]["direction"], "short")
        self.assertTrue(book["short"]["taken"])

    def test_flat_when_neither_ev_clears_floor(self):
        rep = _report(
            **{
                "setup_classification": {
                    "direction": "flat",
                    "long_families": {"LONG_BREAKOUT": 0.1},
                    "short_families": {"SHORT_BREAKDOWN": 0.1},
                    "prob_long": 0.4,
                    "prob_short": 0.4,
                    "confidence": 0.1,
                },
                "targets": {"best_rr": 0.8, "min_rr_tp": None, "targets": []},
                "short_targets": {"best_rr": 0.8, "min_rr_tp": None, "targets": []},
            }
        )
        book = build_opportunity_book(rep)
        self.assertEqual(book["verdict"]["direction"], "flat")
        self.assertEqual(book["verdict"]["status"], "FLAT")

    def test_no_fabricated_probability(self):
        # No calibrated ML probability -> EV must be None and the decision
        # must NOT invent a probability or an EV.
        rep = _report(
            **{
                "setup_classification": {
                    "direction": "long",
                    "long_families": {"LONG_BREAKOUT": 0.4},
                    "short_families": {"SHORT_BREAKDOWN": 0.3},
                    "prob_long": None,
                    "prob_short": None,
                    "confidence": 0.4,
                }
            }
        )
        book = build_opportunity_book(rep)
        self.assertIsNone(book["long"]["probability"])
        self.assertIsNone(book["long"]["expected_r"])
        # Rule path: no engine confirmed -> FLAT, not a fake trade.
        self.assertEqual(book["verdict"]["direction"], "flat")

    def test_explainable_rejections(self):
        rep = _report()  # long wins
        book = build_opportunity_book(rep)
        # The winner must have NO contradictory rejection reasons.
        self.assertEqual(book["long"]["rejection_reasons"], [])
        # The loser must explain why it lost.
        self.assertTrue(book["short"]["rejection_reasons"])

    def test_macro_blocked_flips_to_flat(self):
        rep = _report(**{"macro": {"gate": {"allowed": False, "reason": "headwind"}}})
        book = build_opportunity_book(rep)
        self.assertEqual(book["verdict"]["direction"], "flat")
        self.assertIn("macro", book["verdict"]["reason"])

    def test_short_uses_short_direction_macro_gate(self):
        # The SHORT opportunity must be gated by the short-direction macro
        # gate (gate_short), not the long one: a strong-dollar backdrop
        # blocks the EURUSD long but must NOT block the EURUSD short.
        rep = _report(
            **{
                "setup_classification": {
                    "direction": "short",
                    "long_families": {"LONG_BREAKOUT": 0.2},
                    "short_families": {"SHORT_BREAKDOWN": 0.5},
                    "prob_long": 0.2,
                    "prob_short": 0.7,
                    "confidence": 0.5,
                },
                "macro": {
                    "gate": {"allowed": False, "reason": "headwind"},
                    "gate_short": {"allowed": True, "reason": "ok"},
                },
            }
        )
        book = build_opportunity_book(rep)
        self.assertEqual(book["verdict"]["direction"], "short")
        self.assertNotIn("macro", book["verdict"]["reason"].lower())

    def test_short_macro_blocked_flips_to_flat(self):
        # When the short-direction gate is blocked the short must fall to
        # FLAT, exactly as the long side does under a blocked long gate.
        rep = _report(
            **{
                "setup_classification": {
                    "direction": "short",
                    "long_families": {"LONG_BREAKOUT": 0.2},
                    "short_families": {"SHORT_BREAKDOWN": 0.5},
                    "prob_long": 0.2,
                    "prob_short": 0.7,
                    "confidence": 0.5,
                },
                "macro": {
                    "gate": {"allowed": True, "reason": "ok"},
                    "gate_short": {"allowed": False, "reason": "tailwind"},
                },
            }
        )
        book = build_opportunity_book(rep)
        self.assertEqual(book["verdict"]["direction"], "flat")
        self.assertIn("macro", book["verdict"]["reason"])

    def test_missing_families_rejected(self):
        rep = _report(
            **{
                "setup_classification": {
                    "direction": "flat",
                    "long_families": {},
                    "short_families": {},
                    "prob_long": None,
                    "prob_short": None,
                    "confidence": 0.0,
                }
            }
        )
        book = build_opportunity_book(rep)
        self.assertIn("no setup family", book["long"]["rejection_reasons"][0])

    def test_format_renders_verdict(self):
        book = build_opportunity_book(_report())
        text = format_opportunity_book(book)
        self.assertIn("VERDICT: LONG", text)
        self.assertIn("EV", text)

    def test_rule_path_picks_higher_engine_score(self):
        # No calibrated probabilities anywhere -> rule path. BOTH engines
        # confirmed: the higher engine score must decide, never a long-first
        # list order (dip 6 vs rally 8 -> SHORT wins).
        rep = _report(
            **{
                "setup_classification": {
                    "direction": "flat",
                    "long_families": {"LONG_BUY_DIP": 0.4},
                    "short_families": {"SHORT_SELL_RALLY": 0.4},
                    "prob_long": None,
                    "prob_short": None,
                    "confidence": 0.4,
                },
                "dip": {"dip_confirmed": True, "dip_score": 6},
                "rally": {"rally_confirmed": True, "rally_score": 8},
            }
        )
        book = build_opportunity_book(rep)
        self.assertEqual(book["verdict"]["direction"], "short")
        self.assertTrue(book["short"]["taken"])

    def test_no_payoff_basis_means_no_ev(self):
        # Calibrated probability but NO target ladder and NO achieved R:R:
        # EV must be None (assuming a 1.0R payoff would be a silent
        # fabrication), and the side carries an explicit rejection reason.
        rep = _report(
            **{
                "setup_classification": {
                    "direction": "long",
                    "long_families": {"LONG_BREAKOUT": 0.5},
                    "short_families": {"SHORT_BREAKDOWN": 0.2},
                    "prob_long": 0.6,
                    "prob_short": 0.3,
                    "confidence": 0.5,
                },
                "risk": {"setup": None},
                "targets": {"best_rr": None, "min_rr_tp": None, "targets": []},
            }
        )
        book = build_opportunity_book(rep)
        self.assertIsNone(book["long"]["expected_r"])
        self.assertIn("payoff basis", book["long"]["rejection_reasons"][0])

    def test_min_ev_parameterizable(self):
        # A caller may raise the EV floor. With a 1.0R payoff basis and
        # P=0.6: EV = 0.6*1 - 0.4 - 0.05 cost = +0.15R, which clears the
        # 0.1R floor but NOT a 0.5R floor -> FLAT under the stricter gate.
        rep = _report(
            **{
                "setup_classification": {
                    "direction": "long",
                    "long_families": {"LONG_BREAKOUT": 0.4},
                    "short_families": {"SHORT_BREAKDOWN": 0.2},
                    "prob_long": 0.6,
                    "prob_short": 0.3,
                    "confidence": 0.4,
                },
                "targets": {"best_rr": 1.0, "min_rr_tp": None, "targets": []},
                "short_targets": {
                    "best_rr": 1.0,
                    "min_rr_tp": None,
                    "targets": [],
                },
            }
        )
        book = build_opportunity_book(rep, min_ev=0.5)
        self.assertEqual(book["verdict"]["direction"], "flat")
        book2 = build_opportunity_book(rep, min_ev=0.1)
        self.assertEqual(book2["verdict"]["direction"], "long")


class TestCostModel(unittest.TestCase):
    def test_cost_in_r_scales_with_stop(self):
        # Same spread/slippage, wider stop -> smaller cost in R.
        c1 = roundtrip_cost_r(1.10, 1.09, spread_points=0.0001, slippage_pips=0.5)
        c2 = roundtrip_cost_r(1.10, 1.05, spread_points=0.0001, slippage_pips=0.5)
        self.assertGreater(c1, c2)

    def test_cost_defaults_when_no_stop(self):
        self.assertGreater(roundtrip_cost_r(None, None), 0.0)

    def test_cost_never_negative(self):
        c = roundtrip_cost_r(1.10, 1.09, spread_points=0.0, slippage_pips=0.0)
        self.assertGreaterEqual(c, 0.0)

    def test_jpy_pip_scaling(self):
        # The cost model must convert pip slippage through the CORRECT pip
        # size: the same 50-pip stop costs the same in R on a JPY pair
        # (stop 0.50, pip 0.01) and a major (stop 0.005, pip 0.0001) -
        # equality across conventions is the honest invariant, because
        # cost in R depends only on the slippage:stop *pip ratio*.
        c_jpy = roundtrip_cost_r(150.0, 149.50, slippage_pips=0.5, pip_size=0.01)
        c_maj = roundtrip_cost_r(1.10, 1.095, slippage_pips=0.5, pip_size=0.0001)
        self.assertAlmostEqual(c_jpy, c_maj, places=4)
        # 0.5 pip slippage each way on a 1-pip stop = 1R of cost, on any
        # pair (pip normalization makes the convention irrelevant).
        c_jpy_1pip = roundtrip_cost_r(150.0, 149.99, slippage_pips=0.5, pip_size=0.01)
        c_maj_1pip = roundtrip_cost_r(1.10, 1.0999, slippage_pips=0.5, pip_size=0.0001)
        self.assertAlmostEqual(c_jpy_1pip, c_maj_1pip, places=4)
        self.assertAlmostEqual(c_jpy_1pip, 1.0, places=2)


class TestCurrencyExposure(unittest.TestCase):
    def test_jpy_crosses_are_concentrated_jpy_short(self):
        positions = [
            {"symbol": "EURJPY", "direction": "long", "notional": 100000},
            {"symbol": "GBPJPY", "direction": "long", "notional": 100000},
            {"symbol": "CADJPY", "direction": "long", "notional": 100000},
        ]
        ce = currency_exposure(positions)
        # Three JPY-cross longs = one big JPY-short leg.
        self.assertAlmostEqual(ce["exposure"]["JPY"], -300000.0)
        self.assertEqual(ce["largest"][0], "JPY")
        self.assertTrue(any("JPY" in w for w in ce["warnings"]))

    def test_long_short_netting(self):
        positions = [
            {"symbol": "EURUSD", "direction": "long", "notional": 100000},
            {"symbol": "EURUSD", "direction": "short", "notional": 100000},
        ]
        ce = currency_exposure(positions)
        self.assertAlmostEqual(ce["exposure"]["EUR"], 0.0)
        self.assertAlmostEqual(ce["exposure"]["USD"], 0.0)
        self.assertAlmostEqual(ce["net"], 0.0)

    def test_single_asset_class(self):
        ce = currency_exposure(
            [{"symbol": "AAPL", "direction": "long", "notional": 50000}]
        )
        self.assertAlmostEqual(ce["exposure"]["AAPL"], 50000.0)

    def test_gold_counts_as_single_leg(self):
        ce = currency_exposure(
            [{"symbol": "XAUUSD", "direction": "long", "notional": 50000}]
        )
        self.assertAlmostEqual(ce["exposure"]["XAUUSD"], 50000.0)


if __name__ == "__main__":
    unittest.main()
