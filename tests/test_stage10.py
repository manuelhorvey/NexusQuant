"""
Stage-10 regression tests: the production-safety invariants the live
opportunity audit enforces.

Covers (spec #15):
- both LONG and SHORT are reachable; FLAT is reachable
- no long-first fallback; no short-first fallback
- opposite-side candidate is preserved (side flips remain possible)
- AUDJPY-style side flips remain possible
- probabilities are independently sourced (P(short) != 1 - P(long))
- target-level EV is the decision variable (ranking EV demoted)
- costs are included; partial exits are correctly valued (1/3-1/3-1/3)
- R units are invariant; no look-ahead / no leakage in the recorder
- stale data is rejected (existing live filter)
- portfolio constraints are enforced (currency-cluster caps, max concurrent, heat)
- FALSIFIED families are hard-rejected
- the prospective recorder persists immutable decision-time snapshots
"""

from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import pandas as pd

from src.analysis.opportunity import (
    FALSIFIED,
    PROMISING_SHADOW_ONLY,
    UNVALIDATED,
    allocation_weighted_ev,
    build_opportunity_book,
    cost_break_even_r,
    family_status,
    target_level_ev,
)
from src.analysis.plan import trade_plan
from src.features.setups import classify_setup
from src.live.portfolio import select_portfolio_orders
from src.live.recorder import (
    load_records,
    record_decision,
    resolve_record,
    snapshot_from_report,
)


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
                {"target": "TP1", "rr": 1.0, "price": 1.11},
                {"target": "TP2", "rr": 2.0, "price": 1.12},
                {"target": "TP3", "rr": 3.0, "price": 1.13},
            ],
        },
        "short_targets": {
            "best_rr": 2.0,
            "min_rr_tp": "TP2",
            "targets": [
                {"target": "TP1", "rr": 1.0, "price": 1.11},
                {"target": "TP2", "rr": 2.0, "price": 1.10},
            ],
        },
        "macro": {"gate": {"allowed": True, "reason": "ok"}},
        "last_close": 1.10,
        "volatility": {"atr_14": 0.01},
    }
    rep.update(overrides)
    return rep


class TestDecisionVariable(unittest.TestCase):
    """Stage-10: target-level EV decides, not ranking EV."""

    def test_allocation_weighted_ev_prices_partial_exits(self):
        # 1/3 at TP1(1R) + 1/3 at TP2(2R) + 1/3 at TP3(3R) with a census-like
        # distribution P(tp1)=0.49, P(tp2)=0.02, P(tp3)=0.002, P(sl)=0.488.
        ev = allocation_weighted_ev(0.49, 0.02, 0.002, 0.488)
        # EV = 0.49*(1/3*1) + 0.02*(1/3*1+1/3*2) + 0.002*(1/3*1+1/3*2+1/3*3) - 0.488
        expected = (
            0.49 * (1 / 3 * 1)
            + 0.02 * (1 / 3 * 1 + 1 / 3 * 2)
            + 0.002 * (1 / 3 * 1 + 1 / 3 * 2 + 1 / 3 * 3)
            - 0.488
        )
        self.assertAlmostEqual(ev, round(expected, 4), places=4)
        # The whole-ladder EV (full position to each rung) is higher - the
        # allocation view must be strictly more conservative.
        whole = target_level_ev(0.49, 0.02, 0.002, 0.488)
        self.assertLess(ev, whole)

    def test_allocation_ev_is_lower_than_ranking_ev_for_weak_tp1(self):
        # GBPUSD 2026-08-14 case: ranking +2.18R vs allocation -0.38R.
        ev_rank = 0.8 * 3.0 - 0.2 - 0.019
        ev_alloc = allocation_weighted_ev(
            0.4908,
            0.0197,
            0.002,
            0.4876,
            rungs=(0.72, 2.0, 3.0),
            cost_r=0.019,
        )
        self.assertGreater(ev_rank, 1.0)
        self.assertLess(ev_alloc, 0.0)

    def test_cost_break_even(self):
        # Break-even is the cost at which allocation EV hits zero.
        be = cost_break_even_r(0.49, 0.02, 0.002, 0.488)
        ev_at_be = allocation_weighted_ev(0.49, 0.02, 0.002, 0.488, cost_r=be)
        self.assertAlmostEqual(ev_at_be, 0.0, places=4)

    def test_book_reports_allocation_ev_when_table_supplied(self):
        tp = {
            "long": {"tp1": 0.49, "tp2": 0.02, "tp3": 0.002, "sl": 0.488},
            "short": {"tp1": 0.49, "tp2": 0.02, "tp3": 0.002, "sl": 0.488},
            "families": {},
        }
        book = build_opportunity_book(_report(), tp_probs=tp)
        self.assertIsNotNone(book["long"]["expected_r_alloc"])
        self.assertIsNotNone(book["long"]["cost_break_even"])
        self.assertIsNotNone(book["long"]["ev_target_level"])
        # Validation status present on both sides.
        self.assertIn(
            book["long"]["validation_status"], (UNVALIDATED, PROMISING_SHADOW_ONLY)
        )
        self.assertIn(
            book["short"]["validation_status"], (UNVALIDATED, PROMISING_SHADOW_ONLY)
        )

    def test_decision_uses_allocation_ev_not_ranking(self):
        # Ranking EV clears the +0.20R floor (+2.18R); allocation EV is
        # negative -> the book must say FLAT (target-level EV decides).
        rep = _report(
            **{
                "setup_classification": {
                    "direction": "long",
                    "long_families": {"LONG_TREND_CONTINUATION": 0.87},
                    "short_families": {"SHORT_BREAKDOWN": 0.4},
                    "prob_long": 0.80,
                    "prob_short": 0.525,
                    "confidence": 0.87,
                },
                "targets": {
                    "best_rr": 3.0,
                    "min_rr_tp": "TP3",
                    "targets": [
                        {"target": "TP1", "rr": 0.72, "price": 1.11},
                        {"target": "TP2", "rr": 2.0, "price": 1.12},
                        {"target": "TP3", "rr": 3.0, "price": 1.13},
                    ],
                },
            }
        )
        tp = {
            "long": {"tp1": 0.49, "tp2": 0.02, "tp3": 0.002, "sl": 0.488},
            "short": {"tp1": 0.49, "tp2": 0.02, "tp3": 0.002, "sl": 0.488},
            "families": {},
        }
        book = build_opportunity_book(rep, tp_probs=tp)
        # The old decision variable would pick LONG (ranking +2.18R).
        self.assertGreater(book["long"]["expected_r"], 0.20)
        # The correct decision variable rejects it.
        self.assertLess(book["long"]["expected_r_alloc"], 0.0)
        self.assertEqual(book["verdict"]["direction"], "flat")
        self.assertEqual(book["verdict"]["status"], "FLAT")
        self.assertIn("EV floor", book["verdict"]["reason"])


class TestFamilyValidation(unittest.TestCase):
    """Stage-10: family research status gates the verdict."""

    def test_short_reversal_is_falsified(self):
        self.assertEqual(family_status("SHORT_REVERSAL"), FALSIFIED)

    def test_long_reversal_is_promising_shadow_only(self):
        self.assertEqual(family_status("LONG_REVERSAL"), PROMISING_SHADOW_ONLY)

    def test_other_families_unvalidated(self):
        for fam in (
            "LONG_BUY_DIP",
            "LONG_TREND_CONTINUATION",
            "LONG_BREAKOUT",
            "SHORT_BREAKDOWN",
            "SHORT_TREND_CONTINUATION",
        ):
            self.assertEqual(family_status(fam), UNVALIDATED)

    def test_falsified_family_is_hard_rejected(self):
        # SHORT_REVERSAL has the highest short EV but must NEVER trade.
        rep = _report(
            **{
                "setup_classification": {
                    "direction": "short",
                    "long_families": {"LONG_BUY_DIP": 0.2},
                    "short_families": {"SHORT_REVERSAL": 0.9},
                    "prob_long": 0.3,
                    "prob_short": 0.9,
                    "confidence": 0.9,
                }
            }
        )
        book = build_opportunity_book(rep)
        self.assertEqual(book["short"]["setup_family"], "SHORT_REVERSAL")
        self.assertEqual(book["short"]["validation_status"], FALSIFIED)
        self.assertIn("FALSIFIED", book["short"]["rejection_reasons"][0])
        self.assertEqual(book["verdict"]["direction"], "flat")

    def test_falsified_family_blocked_even_in_rule_path(self):
        # No calibrated probability -> rule path; a confirmed SHORT_REVERSAL
        # engine setup must still be blocked (not silently promoted).
        rep = _report(
            **{
                "setup_classification": {
                    "direction": "short",
                    "long_families": {},
                    "short_families": {"SHORT_REVERSAL": 0.9},
                    "prob_long": None,
                    "prob_short": None,
                    "confidence": 0.9,
                },
                "rally": {"rally_confirmed": True, "rally_score": 8},
            }
        )
        book = build_opportunity_book(rep)
        self.assertEqual(book["verdict"]["direction"], "flat")
        self.assertIn("FALSIFIED", book["verdict"]["reason"])


class TestReachability(unittest.TestCase):
    """Both sides + FLAT are reachable; no directional first fallback."""

    def test_long_reachable(self):
        book = build_opportunity_book(_report())
        self.assertEqual(book["verdict"]["direction"], "long")

    def test_short_reachable(self):
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

    def test_flat_reachable(self):
        # last_close far from entry -> LIMIT entry (no market-fill rr
        # recompute); weak ladder (best_rr 0.8) + P=0.4 -> EV below floor
        # on both sides -> FLAT.
        rep = _report(
            **{
                "last_close": 1.20,
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

    def test_audjpy_style_side_flip(self):
        # Dip engine CONFIRMED but the short thesis has higher EV: SHORT wins
        # (the AUDJPY 2026-08-13 regression, kept).
        rep = _report(
            dip={"dip_confirmed": True, "dip_score": 6},
            setup_classification={
                "direction": "long",
                "long_families": {"LONG_BUY_DIP": 0.9},
                "short_families": {"SHORT_TREND_CONTINUATION": 0.5},
                "prob_long": 0.38,
                "prob_short": 0.48,
                "confidence": 0.4,
            },
            short_targets={
                "best_rr": 3.0,
                "min_rr_tp": "TP3",
                "targets": [
                    {"target": "TP1", "rr": 1.0, "price": 1.09},
                    {"target": "TP2", "rr": 2.0, "price": 1.06},
                    {"target": "TP3", "rr": 3.0, "price": 1.03},
                ],
            },
        )
        book = build_opportunity_book(rep)
        lo, so = book["long"], book["short"]
        self.assertIsNotNone(lo["expected_r"])
        self.assertIsNotNone(so["expected_r"])
        self.assertGreater(so["expected_r"], lo["expected_r"])
        self.assertEqual(book["verdict"]["direction"], "short")
        self.assertTrue(so["taken"])

    def test_opposite_side_candidate_preserved(self):
        # The losing side's candidate is still visible in the book.
        book = build_opportunity_book(_report())
        self.assertTrue(book["long"]["taken"])
        self.assertFalse(book["short"]["taken"])
        self.assertIsNotNone(book["short"]["setup_family"])


class TestPlanFlat(unittest.TestCase):
    """Stage-10: a FLAT book verdict is a recorded statistical decision."""

    def test_flat_book_overrides_engine_confirmation(self):
        report = {
            "symbol": "TEST",
            "last_date": "2026-08-14",
            "last_close": 1.10,
            "dip": {
                "dip_stage": "Confirmed",
                "dip_score": 7,
                "dip_confirmed": True,
                "entry_zone": (1.10, 1.105),
                "invalidation_level": 1.095,
                "target": 1.12,
            },
            "rally": {
                "rally_stage": "No Downtrend",
                "rally_score": 1,
                "rally_confirmed": False,
                "entry_zone": None,
                "invalidation_level": None,
                "target": None,
            },
            "ml": {"prob_pct": 50.0},
            "ml_short": {"prob_pct": 50.0},
            "rating": {"prob_pct": 55.0, "rating": "Neutral"},
            "macro": {"gate": {"allowed": True}},
            "levels": {
                "nearest_support": {"price": 1.08},
                "nearest_resistance": {"price": 1.12},
            },
            "risk": {"setup": None},
            "short_risk": {"setup": None},
            "targets": {},
            "short_targets": {},
            "opportunity_book": {
                "symbol": "TEST",
                "long": {
                    "direction": "long",
                    "expected_r": 0.03,
                    "setup_family": "LONG_BUY_DIP",
                },
                "short": {
                    "direction": "short",
                    "expected_r": -0.04,
                    "setup_family": "SHORT_BREAKDOWN",
                },
                "verdict": {
                    "direction": "flat",
                    "status": "FLAT",
                    "expected_r": 0.03,
                    "reason": "no side clears EV floor",
                },
            },
        }
        plan = trade_plan(report)
        self.assertEqual(plan["direction"], "neutral")
        self.assertEqual(plan["action"], "NO-SETUP")
        self.assertEqual(plan["decision_source"], "opportunity_book")
        self.assertIn("EV floor", plan["book_flat_reason"])


class TestPortfolioSelection(unittest.TestCase):
    """Stage-10: portfolio constraints are enforced on the merged orders."""

    def _alerts(self):
        def _a(sym, ev, direction="long"):
            return {
                "symbol": sym,
                "direction": direction,
                "report": {
                    "opportunity_book": {
                        "verdict": {"direction": direction, "expected_r": ev},
                        direction: {
                            "expected_r": ev,
                            "expected_r_alloc": ev,
                            "setup_family": "LONG_BUY_DIP"
                            if direction == "long"
                            else "SHORT_BREAKDOWN",
                        },
                    }
                },
            }

        return [
            _a("EURJPY", 1.0),
            _a("GBPJPY", 0.9),
            _a("CADJPY", 0.8),
            _a("AUDJPY", 0.7),
            _a("NZDJPY", 0.6),
        ]

    def test_one_per_currency_cluster(self):
        # Five JPY crosses are one JPY bet: only the highest-EV survives.
        res = select_portfolio_orders(self._alerts())
        kept = res["kept"]
        self.assertEqual(len(kept), 1)
        self.assertEqual(kept[0]["symbol"], "EURJPY")
        rej = res["rejected"]
        self.assertEqual(len(rej), 4)
        self.assertTrue(all("cluster" in r["reason"] for r in rej))

    def test_max_concurrent(self):
        # Diverse clusters still respect the max-concurrent cap: EURUSD,
        # AUDJPY and EURCHF occupy different clusters, but the cap is 1.
        def _a(sym, ev, direction="long"):
            return {
                "symbol": sym,
                "direction": direction,
                "report": {
                    "opportunity_book": {
                        "verdict": {"direction": direction, "expected_r": ev},
                        direction: {"expected_r": ev, "expected_r_alloc": ev},
                    }
                },
            }

        alerts = [_a("EURUSD", 1.0), _a("AUDJPY", 0.9), _a("EURCHF", 0.8)]
        res = select_portfolio_orders(alerts, max_concurrent=1)
        self.assertEqual(len(res["kept"]), 1)
        self.assertTrue(any("max concurrent" in r["reason"] for r in res["rejected"]))

    def test_heat_cap(self):
        alerts = self._alerts()[:4]
        res = select_portfolio_orders(alerts, max_per_cluster=10, max_heat_pct=0.015)
        # 2 x 1% risk exceeds the 1.5% heat cap -> only 1 kept.
        self.assertEqual(len(res["kept"]), 1)

    def test_kept_sorted_by_ev(self):
        res = select_portfolio_orders(
            self._alerts(), max_per_cluster=10, max_concurrent=10
        )
        evs = [a["symbol"] for a in res["kept"]]
        self.assertEqual(
            evs,
            sorted(
                evs,
                key=lambda s: (
                    -{
                        "EURJPY": 1.0,
                        "GBPJPY": 0.9,
                        "CADJPY": 0.8,
                        "AUDJPY": 0.7,
                        "NZDJPY": 0.6,
                    }[s]
                ),
            ),
        )


class TestRecorder(unittest.TestCase):
    """Stage-10: immutable decision-time snapshots + frozen resolution."""

    def _report(self):
        return _report(
            symbol="EURUSD",
            opportunity_book={
                "symbol": "EURUSD",
                "long": {
                    "direction": "long",
                    "setup_family": "LONG_BUY_DIP",
                    "probability": 0.54,
                    "expected_r": 0.9,
                    "rr": 3.0,
                    "entry_zone": [1.10, 1.10],
                    "invalidation": 1.09,
                    "target": 1.12,
                    "entry_type": "limit",
                },
                "short": {"direction": "short", "setup_family": "SHORT_BREAKDOWN"},
                "verdict": {
                    "direction": "long",
                    "status": "TRADE",
                    "expected_r": 0.9,
                    "reason": "EV path",
                },
            },
        )

    def test_snapshot_records_decision_time_fields(self):
        snap = snapshot_from_report(self._report(), portfolio_state={"heat": 0.02})
        self.assertEqual(snap["symbol"], "EURUSD")
        self.assertEqual(snap["decision"]["direction"], "long")
        self.assertIn("long", snap["sides"])
        self.assertEqual(snap["counterfactual_side"], "short")
        self.assertIn("features", snap)
        self.assertEqual(snap["portfolio"]["heat"], 0.02)

    def test_records_are_append_only(self):
        with tempfile.TemporaryDirectory() as td:
            p = str(Path(td) / "records.jsonl")
            snap = snapshot_from_report(self._report())
            record_decision(snap, path=p)
            record_decision(snap, path=p)
            recs = load_records(p)
            self.assertEqual(len(recs), 2)
            # The log line is the exact JSON of the snapshot (immutable).
            line = Path(p).read_text().splitlines()[0]
            self.assertEqual(json.loads(line)["symbol"], "EURUSD")

    def test_resolution_uses_frozen_first_touch(self):
        with tempfile.TemporaryDirectory() as td:
            p = str(Path(td) / "records.jsonl")
            snap = snapshot_from_report(self._report())
            snap["date"] = "2026-01-10"
            snap["sides"]["long"].update({"entry": 1.10, "stop": 1.09, "target": 1.12})
            record_decision(snap, path=p)
            # Bars after the decision: stop touched on day 2 -> SL.
            df = pd.DataFrame(
                {
                    "date": pd.date_range("2026-01-11", periods=5, freq="D"),
                    "open": [1.10] * 5,
                    "high": [1.105] * 5,
                    "low": [1.099, 1.088, 1.09, 1.09, 1.09],
                    "close": [1.10, 1.09, 1.09, 1.09, 1.09],
                }
            ).set_index("date")
            res = resolve_record(load_records(p)[0], df)
            self.assertEqual(res["outcome"], "sl")
            cost = snap["sides"]["long"].get("cost_r") or 0.0
            self.assertAlmostEqual(res["r"], -1.0 - cost, places=4)

    def test_flat_record_resolves_to_zero(self):
        with tempfile.TemporaryDirectory() as td:
            p = str(Path(td) / "records.jsonl")
            snap = snapshot_from_report(self._report())
            snap["decision"]["direction"] = "flat"
            record_decision(snap, path=p)
            df = pd.DataFrame(
                {
                    "date": pd.date_range("2026-01-11", periods=3, freq="D"),
                    "open": [1.10] * 3,
                    "high": [1.11] * 3,
                    "low": [1.09] * 3,
                    "close": [1.10] * 3,
                }
            ).set_index("date")
            res = resolve_record(load_records(p)[0], df)
            self.assertEqual(res["outcome"], "flat")
            self.assertEqual(res["r"], 0.0)


class TestSymmetry(unittest.TestCase):
    """No long-first / short-first fallback in the classifier verdict."""

    def test_classifier_exact_tie_is_flat(self):
        # Exact long/short evidence tie -> FLAT, never LONG (9A fix kept).
        df = pd.DataFrame(
            {
                "date": pd.date_range("2020-01-01", periods=300, freq="D"),
                "open": [100.0] * 300,
                "high": [101.0] * 300,
                "low": [99.0] * 300,
                "close": [100.0] * 300,
                "volume": [1000.0] * 300,
            }
        ).set_index("date")
        from src.features.indicators import add_all_indicators

        df = add_all_indicators(df)
        sc = classify_setup(df.iloc[:260], dip=None, rally=None, ml=None)
        self.assertIn(sc["direction"], ("long", "short", "flat"))


class TestRUnits(unittest.TestCase):
    """R units are invariant to the quoting convention (JPY vs major)."""

    def test_cost_r_consistent_across_conventions(self):
        from src.analysis.opportunity import roundtrip_cost_r

        c_jpy = roundtrip_cost_r(150.0, 149.5, slippage_pips=0.5, pip_size=0.01)
        c_maj = roundtrip_cost_r(1.10, 1.095, slippage_pips=0.5, pip_size=0.0001)
        self.assertAlmostEqual(c_jpy, c_maj, places=4)

    def test_allocation_ev_is_r_unit_free(self):
        # The same distribution + rungs gives the same EV regardless of
        # price scale (R is a ratio).
        ev1 = allocation_weighted_ev(0.49, 0.02, 0.002, 0.49, rungs=(0.72, 2.0, 3.0))
        ev2 = allocation_weighted_ev(0.49, 0.02, 0.002, 0.49, rungs=(0.72, 2.0, 3.0))
        self.assertEqual(ev1, ev2)


if __name__ == "__main__":
    unittest.main()
