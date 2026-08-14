"""
Stage-11 regression tests: the prospective alpha accumulation campaign.

Covers:
- the frozen protocol manifest is stable and its hash changes with ANY
  parameter change (a parameter change = a NEW protocol)
- snapshots embed the protocol hash; protocol-mismatched snapshots are
  excluded from evaluation (old observations never leak into a new
  protocol)
- counterfactual resolution prices the full ladder with the frozen
  1/3-1/3-1/3 allocation: TP1/TP2/TP3 hits, SL hit, time exit, MFE/MAE,
  stop-first within a bar, realized R net of costs - for FLAT-rejected
  candidates too
- the accumulation evaluator is read-only and produces per-family taken
  + counterfactual means, filter effectiveness and over-restriction
  diagnostics
- the promotion ladder: L0/L1/L2/L3 gating with exact failed-gate reasons
- no long-first / no short-first in the counterfactual path (both sides
  resolved symmetrically)
"""

from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import pandas as pd

from src.analysis.promotion import (
    L0_UNVALIDATED,
    L1_RESEARCH_CANDIDATE,
    L2_PROSPECTIVE_VALIDATION,
    L3_VALIDATED_ALPHA,
    check_l3_gates,
    level_from_status,
    promote,
)
from src.analysis.protocol import (
    freeze_manifest,
    matches_protocol,
    protocol_hash,
)
from src.analysis.stage11 import (
    accumulate_prospective,
    build_stage11_doc,
)
from src.live.recorder import (
    record_decision,
    resolve_counterfactual,
    resolve_record,
)


def _snapshot(
    direction: str = "flat",
    protocol_sha: str = None,
    **overrides,
) -> dict:
    """Minimal decision-time snapshot with a resolvable long + short side.

    Long: entry 100 / stop 99 (1R risk) / ladder 101-102-103 (1R/2R/3R).
    Short: entry 100 / stop 101 / ladder 99-98-97 (1R/2R/3R).
    Cost 0.01R each side.
    """
    from src.analysis.protocol import protocol_hash as _ph

    snap = {
        "recorded_at": "2026-08-14T00:00:00Z",
        "protocol": {"version": "stage11.1", "sha256": protocol_sha or _ph()},
        "date": "2026-08-14",
        "symbol": "TESTFX",
        "decision": {"direction": direction, "status": "TRADE", "reason": "test"},
        "sides": {
            "long": {
                "family": "LONG_TREND_CONTINUATION",
                "entry": 100.0,
                "stop": 99.0,
                "target": 101.0,
                "cost_r": 0.01,
                "entry_type": "limit",
            },
            "short": {
                "family": "SHORT_BREAKDOWN",
                "entry": 100.0,
                "stop": 101.0,
                "target": 99.0,
                "cost_r": 0.01,
                "entry_type": "limit",
            },
        },
        "ladders": {
            "long": [
                {"tp": 1, "price": 101.0, "rr": 1.0, "source": "test"},
                {"tp": 2, "price": 102.0, "rr": 2.0, "source": "test"},
                {"tp": 3, "price": 103.0, "rr": 3.0, "source": "test"},
            ],
            "short": [
                {"tp": 1, "price": 99.0, "rr": 1.0, "source": "test"},
                {"tp": 2, "price": 98.0, "rr": 2.0, "source": "test"},
                {"tp": 3, "price": 97.0, "rr": 3.0, "source": "test"},
            ],
        },
    }
    snap.update(overrides)
    return snap


def _bars(days: list) -> pd.DataFrame:
    """Build a D1 frame from [(high, low)] tuples starting 2026-08-15."""
    rows = []
    for i, (hi, lo) in enumerate(days):
        rows.append(
            {
                "date": pd.Timestamp("2026-08-15") + pd.Timedelta(days=i),
                "open": (hi + lo) / 2,
                "high": hi,
                "low": lo,
                "close": (hi + lo) / 2,
            }
        )
    return pd.DataFrame(rows).set_index("date")


class TestProtocolFreeze(unittest.TestCase):
    def test_hash_stable_across_calls(self):
        self.assertEqual(protocol_hash(), protocol_hash())

    def test_manifest_hash_matches_protocol_hash(self):
        m = freeze_manifest()
        self.assertEqual(m["sha256"], protocol_hash())
        self.assertEqual(m["protocol_version"], "stage11.1")
        # frozen_at is metadata - changing it must NOT change the hash
        m2 = freeze_manifest()
        self.assertEqual(m["sha256"], m2["sha256"])

    def test_parameter_change_changes_hash(self):
        """A parameter change is a NEW protocol - the core freeze mechanic."""
        h0 = protocol_hash()
        import src.analysis.protocol as proto

        orig = proto.DECISION_RULES["min_ev_r"]
        try:
            proto.DECISION_RULES["min_ev_r"] = 0.99
            self.assertNotEqual(protocol_hash(), h0)
        finally:
            proto.DECISION_RULES["min_ev_r"] = orig
        self.assertEqual(protocol_hash(), h0)

    def test_matches_protocol(self):
        self.assertTrue(matches_protocol(_snapshot()))
        self.assertFalse(matches_protocol(_snapshot(protocol_sha="deadbeef")))
        self.assertFalse(matches_protocol({"no": "protocol"}))


class TestCounterfactualResolution(unittest.TestCase):
    def test_all_rungs_hit_long(self):
        rec = _snapshot()
        df = _bars([(101.5, 99.5), (102.5, 101.0), (103.5, 102.0)])
        cf = resolve_counterfactual(rec, df, "long")
        self.assertTrue(cf["tp1_hit"] and cf["tp2_hit"] and cf["tp3_hit"])
        self.assertFalse(cf["sl_hit"])
        self.assertFalse(cf["time_exit"])
        # 1/3*1 + 1/3*2 + 1/3*3 - cost = 2.00 - 0.01
        self.assertAlmostEqual(cf["realized_r"], 1.99, places=4)
        self.assertAlmostEqual(cf["mfe_r"], 3.5, places=4)  # (103.5-100)/1
        self.assertAlmostEqual(cf["mae_r"], 0.5, places=4)  # (100-99.5)/1

    def test_sl_first_long(self):
        rec = _snapshot()
        df = _bars([(100.5, 98.5)])
        cf = resolve_counterfactual(rec, df, "long")
        self.assertTrue(cf["sl_hit"])
        self.assertFalse(any([cf["tp1_hit"], cf["tp2_hit"], cf["tp3_hit"]]))
        self.assertAlmostEqual(cf["realized_r"], -1.01, places=4)

    def test_tp1_then_sl_long(self):
        """TP1 exits 1/3; the remaining 2/3 stops at -1R."""
        rec = _snapshot()
        df = _bars([(101.5, 99.5), (100.0, 98.0)])
        cf = resolve_counterfactual(rec, df, "long")
        self.assertTrue(cf["tp1_hit"])
        self.assertTrue(cf["sl_hit"])
        self.assertFalse(cf["tp2_hit"])
        # 1/3*1 - (1 - 1/3)*1 - cost = -1/3 - 0.01
        self.assertAlmostEqual(cf["realized_r"], -0.3433, places=3)

    def test_time_exit_long(self):
        rec = _snapshot()
        df = _bars([(100.5, 99.5)] * 5)  # no rung, no stop
        cf = resolve_counterfactual(rec, df, "long")
        self.assertTrue(cf["time_exit"])
        self.assertFalse(cf["sl_hit"])
        self.assertAlmostEqual(cf["realized_r"], -0.01, places=4)  # cost only

    def test_stop_first_within_bar(self):
        """Stop and TP1 in the SAME bar -> stop wins (frozen semantics)."""
        rec = _snapshot()
        df = _bars([(101.5, 98.5)])  # touches both 101 and 99 in one bar
        cf = resolve_counterfactual(rec, df, "long")
        self.assertTrue(cf["sl_hit"])
        self.assertFalse(cf["tp1_hit"])
        self.assertAlmostEqual(cf["realized_r"], -1.01, places=4)

    def test_short_side_symmetric(self):
        rec = _snapshot()
        df = _bars([(100.5, 98.5), (99.0, 97.0), (98.0, 96.0)])
        cf = resolve_counterfactual(rec, df, "short")
        self.assertTrue(cf["tp1_hit"] and cf["tp2_hit"] and cf["tp3_hit"])
        self.assertFalse(cf["sl_hit"])
        self.assertAlmostEqual(cf["realized_r"], 1.99, places=4)
        self.assertAlmostEqual(cf["mfe_r"], 4.0, places=4)  # (100-96)/1
        self.assertAlmostEqual(cf["mae_r"], 0.5, places=4)  # (100.5-100)/1

    def test_both_sides_resolved_for_flat(self):
        """FLAT records resolve BOTH candidate sides counterfactually -
        the whole point of the accumulation window."""
        rec = _snapshot(direction="flat")
        df = _bars([(101.5, 99.5), (102.5, 101.0), (103.5, 102.0)])
        res = resolve_record(rec, df)
        self.assertEqual(res["outcome"], "flat")
        self.assertEqual(res["r"], 0.0)
        self.assertIn("long", res["counterfactual"])
        self.assertIn("short", res["counterfactual"])
        self.assertAlmostEqual(
            res["counterfactual"]["long"]["realized_r"], 1.99, places=4
        )

    def test_ladder_fallback_to_single_target(self):
        rec = _snapshot()
        rec["ladders"] = {"long": [], "short": []}  # no ladder -> primary target
        df = _bars([(101.5, 99.5)])
        cf = resolve_counterfactual(rec, df, "long")
        self.assertTrue(cf["tp1_hit"])
        self.assertEqual(cf["rungs_rr"], [1.0])
        # 1/1 fraction at 1R - cost
        self.assertAlmostEqual(cf["realized_r"], 0.99, places=4)

    def test_no_levels_returns_none(self):
        rec = _snapshot()
        rec["sides"]["long"]["entry"] = None
        df = _bars([(101.5, 99.5)])
        self.assertIsNone(resolve_counterfactual(rec, df, "long"))


class TestAccumulation(unittest.TestCase):
    def _run(self, snaps, days_list):
        with tempfile.TemporaryDirectory() as td:
            rp = str(Path(td) / "records.jsonl")
            resp = str(Path(td) / "resolutions.jsonl")
            for s in snaps:
                record_decision(s, path=rp)
            from src.live.recorder import load_records, record_resolution

            recs = load_records(rp)
            dfs = [
                pd.concat(days) if isinstance(days, list) else days
                for days in days_list
            ]
            for rec, df in zip(recs, dfs, strict=True):
                res = resolve_record(rec, df)
                if res:
                    record_resolution(res, path=resp)
            from src.live.recorder import load_resolutions

            return accumulate_prospective(
                records=load_records(rp), resolutions=load_resolutions(resp)
            )

    def test_taken_mean_r_per_family(self):
        # Two LONG decisions on LONG_TREND_CONTINUATION: all-rungs (+1.99)
        # and SL-first (-1.01).
        long1 = _snapshot(direction="long", recorded_at="t1")
        long1["sides"]["long"]["family"] = "LONG_TREND_CONTINUATION"
        long2 = _snapshot(direction="long", recorded_at="t2")
        long2["sides"]["long"]["family"] = "LONG_TREND_CONTINUATION"
        stats = self._run(
            [long1, long2],
            [
                _bars([(101.5, 99.5), (102.5, 101.0), (103.5, 102.0)]),
                _bars([(100.5, 98.5)]),
            ],
        )
        f = stats["families"]["LONG_TREND_CONTINUATION"]
        self.assertEqual(f["taken_n"], 2)
        self.assertAlmostEqual(f["taken_mean_r"], (1.99 - 1.01) / 2, places=4)
        self.assertEqual(stats["n_resolved"], 2)

    def test_flat_counterfactual_means_and_over_restriction(self):
        # FLAT record: long side would have made +1.99 (>= +0.30R ->
        # over-restriction evidence), short side would have lost -1.01.
        flat = _snapshot(direction="flat")
        stats = self._run(
            [flat],
            [_bars([(101.5, 99.5), (102.5, 101.0), (103.5, 102.0)])],
        )
        lf = stats["families"]["LONG_TREND_CONTINUATION"]
        sf = stats["families"]["SHORT_BREAKDOWN"]
        self.assertEqual(lf["flat_n"], 1)
        self.assertAlmostEqual(lf["cf_long_mean_r"], 1.99, places=4)
        self.assertAlmostEqual(sf["cf_short_mean_r"], -1.01, places=4)
        self.assertEqual(lf["over_restriction_n"], 1)
        d = stats["diagnostics"]
        self.assertEqual(d["over_restriction_count"], 1)
        # filter effectiveness: only one side negative -> not both
        self.assertEqual(d["flat_decisions_with_counterfactuals"], 1)
        self.assertEqual(d["flat_decisions_both_sides_negative"], 0)
        self.assertAlmostEqual(d["filter_effectiveness"], 0.0, places=4)

    def test_filter_effectiveness_both_sides_negative(self):
        # FLAT where both sides would have stopped out.
        flat = _snapshot(direction="flat")
        stats = self._run(
            [flat],
            [_bars([(101.5, 98.5)])],  # both sides SL in one bar
        )
        d = stats["diagnostics"]
        self.assertEqual(d["flat_decisions_with_counterfactuals"], 1)
        self.assertEqual(d["flat_decisions_both_sides_negative"], 1)
        self.assertEqual(d["filter_effectiveness"], 1.0)

    def test_protocol_mismatch_excluded(self):
        old = _snapshot(direction="long", recorded_at="old", protocol_sha="deadbeef")
        cur = _snapshot(direction="long", recorded_at="cur")
        stats = self._run(
            [old, cur],
            [_bars([(101.5, 99.5), (102.5, 101.0), (103.5, 102.0)])] * 2,
        )
        self.assertEqual(stats["n_records_total"], 2)
        self.assertEqual(stats["n_eligible"], 1)
        self.assertEqual(stats["n_excluded_protocol_mismatch"], 1)
        self.assertEqual(stats["n_resolved"], 1)

    def test_accumulate_is_read_only(self):
        """The evaluator never writes - no files appear, no state changes."""
        with tempfile.TemporaryDirectory() as td:
            rp = str(Path(td) / "records.jsonl")
            record_decision(_snapshot(), path=rp)
            before = Path(rp).read_text()
            from src.live.recorder import load_records

            stats = accumulate_prospective(records=load_records(rp), resolutions=[])
            self.assertEqual(Path(rp).read_text(), before)
            self.assertEqual(stats["n_resolved"], 0)

    def test_doc_builder_writes_only_doc(self):
        with tempfile.TemporaryDirectory() as td:
            out = str(Path(td) / "doc.md")
            recp = str(Path(td) / "records.jsonl")
            record_decision(_snapshot(direction="flat"), path=recp)
            from src.live.recorder import load_records

            stats = accumulate_prospective(records=load_records(recp), resolutions=[])
            build_stage11_doc(stats, out_path=out)
            self.assertTrue(Path(out).exists())
            text = Path(out).read_text()
            self.assertIn("Stage-11", text)
            self.assertIn("L1_RESEARCH_CANDIDATE", text)


class TestNonFillSemantics(unittest.TestCase):
    """Limit orders whose zone is never touched never fill - 0R, no cost.

    A trade that never happened costs nothing: a non-fill must NOT deduct
    the side's cost assumption.
    """

    def test_long_limit_zone_never_touched(self):
        # Long limit entry 100; every forward bar stays ABOVE 100 -> the
        # buy zone is never traded through -> non-fill.
        rec = _snapshot()
        df = _bars([(110.0, 105.0)] * 3)
        cf = resolve_counterfactual(rec, df, "long")
        self.assertIsNotNone(cf)
        self.assertFalse(cf["filled"])
        self.assertEqual(cf["outcome"], "non_fill")
        self.assertEqual(cf["realized_r"], 0.0)  # no cost on a non-fill
        self.assertFalse(cf["sl_hit"])
        self.assertFalse(cf["time_exit"])
        self.assertEqual(cf["mfe_r"], 0.0)

    def test_short_limit_zone_never_touched(self):
        # Short limit entry 100; every forward bar stays BELOW 100 -> the
        # sell zone is never traded through -> non-fill.
        rec = _snapshot()
        df = _bars([(90.0, 85.0)] * 3)
        cf = resolve_counterfactual(rec, df, "short")
        self.assertFalse(cf["filled"])
        self.assertEqual(cf["outcome"], "non_fill")
        self.assertEqual(cf["realized_r"], 0.0)

    def test_limit_touched_after_delay_fills_from_that_bar(self):
        # Zone untouched for one bar, then traded through -> fills and the
        # full ladder resolves from the fill bar onward.
        rec = _snapshot()
        df = _bars([(110.0, 105.0), (101.5, 99.5), (102.5, 101.0), (103.5, 102.0)])
        cf = resolve_counterfactual(rec, df, "long")
        self.assertTrue(cf["filled"])
        self.assertTrue(cf["tp1_hit"] and cf["tp2_hit"] and cf["tp3_hit"])
        self.assertAlmostEqual(cf["realized_r"], 1.99, places=4)

    def test_taken_limit_that_never_fills_resolves_non_fill(self):
        # A TAKEN limit decision whose zone is never touched is a non-fill
        # at 0R - the outcome label and r must both say so.
        rec = _snapshot(direction="long")
        df = _bars([(110.0, 105.0)] * 3)
        res = resolve_record(rec, df)
        self.assertIsNotNone(res)
        self.assertEqual(res["outcome"], "non_fill")
        self.assertEqual(res["r"], 0.0)

    def test_market_entry_always_fills(self):
        # A market entry fills immediately regardless of where bars go -
        # no non-fill path for market orders.
        rec = _snapshot()
        rec["sides"]["long"]["entry_type"] = "market"
        df = _bars([(110.0, 105.0)] * 3)
        cf = resolve_counterfactual(rec, df, "long")
        self.assertTrue(cf["filled"])
        self.assertNotEqual(cf["outcome"], "non_fill")


class TestFlatDecisionClassification(unittest.TestCase):
    """The most important Stage-11 measurement: is NexusQuant good at
    knowing when NOT to trade? Every FLAT decision is classified by what
    its counterfactuals subsequently did:

      A. Correct FLAT - both candidate sides subsequently lose
      B. Over-restriction - a rejected side would have made >= +0.30R
      C. Correct directional rejection - the decision-time higher-ranked
         side won and the other lost
    """

    def _flat_stats(self, bars, **side_overrides):
        flat = _snapshot(direction="flat")
        for side, vals in side_overrides.items():
            flat["sides"][side].update(vals)
        with tempfile.TemporaryDirectory() as td:
            rp = str(Path(td) / "records.jsonl")
            resp = str(Path(td) / "resolutions.jsonl")
            record_decision(flat, path=rp)
            from src.live.recorder import (
                load_records,
                load_resolutions,
                record_resolution,
            )

            recs = load_records(rp)
            res = resolve_record(recs[0], bars)
            record_resolution(res, path=resp)
            return accumulate_prospective(
                records=load_records(rp), resolutions=load_resolutions(resp)
            )

    def test_a_correct_flat_both_sides_lose(self):
        # Both candidate sides stop out in the same bar -> the FLAT call
        # saved both losses.
        stats = self._flat_stats(_bars([(101.5, 98.5)]))
        cls = stats["diagnostics"]["flat_classification"]
        self.assertEqual(cls["correct_flat"], 1)
        self.assertEqual(cls["over_restriction"], 0)
        self.assertEqual(cls["correct_directional_rejection"], 0)
        self.assertEqual(stats["diagnostics"]["flat_decisions_both_sides_negative"], 1)

    def test_b_over_restriction_rejected_side_would_have_won(self):
        # Long side would have made +1.99R, short side lost. No decision-
        # time ranking recorded -> cannot be C -> over-restriction.
        stats = self._flat_stats(_bars([(101.5, 99.5), (102.5, 101.0), (103.5, 102.0)]))
        cls = stats["diagnostics"]["flat_classification"]
        self.assertEqual(cls["over_restriction"], 1)
        self.assertEqual(cls["correct_flat"], 0)
        self.assertEqual(cls["correct_directional_rejection"], 0)
        self.assertEqual(stats["diagnostics"]["over_restriction_count"], 1)

    def test_c_correct_directional_rejection(self):
        # Long +1.99R, short -1.01R, and the decision-time allocation EV
        # ranked long higher -> the ranking identified the superior side.
        stats = self._flat_stats(
            _bars([(101.5, 99.5), (102.5, 101.0), (103.5, 102.0)]),
            long={"expected_r_alloc": 0.50},
            short={"expected_r_alloc": -0.40},
        )
        cls = stats["diagnostics"]["flat_classification"]
        self.assertEqual(cls["correct_directional_rejection"], 1)
        self.assertEqual(cls["over_restriction"], 0)

    def test_c_denied_when_ranking_contradicts(self):
        # Same outcome, but the decision-time ranking favored the LOSING
        # side -> not a correct rejection, it is over-restriction with a
        # mis-ranked side.
        stats = self._flat_stats(
            _bars([(101.5, 99.5), (102.5, 101.0), (103.5, 102.0)]),
            long={"expected_r_alloc": -0.40},
            short={"expected_r_alloc": 0.50},
        )
        cls = stats["diagnostics"]["flat_classification"]
        self.assertEqual(cls["correct_directional_rejection"], 0)
        self.assertEqual(cls["over_restriction"], 1)

    def test_ambiguous_below_over_restriction_bar(self):
        # Long non-fill (0.0R - zone never touched), short stopped (-1.01R):
        # best outcome 0.0 is below the +0.30R bar -> ambiguous.
        stats = self._flat_stats(_bars([(110.0, 105.0)] * 3))
        cls = stats["diagnostics"]["flat_classification"]
        self.assertEqual(cls["ambiguous"], 1)
        self.assertEqual(cls["correct_flat"], 0)

    def test_unresolved_when_no_forward_bars(self):
        stats = self._flat_stats(pd.DataFrame())  # no bars -> nothing resolves
        cls = stats["diagnostics"]["flat_classification"]
        self.assertEqual(cls["unresolved"], 1)

    def test_cf_counts_captured_before_pop(self):
        """cf_long_n / cf_short_n must reflect the window (regression for
        the pop-before-read bug)."""
        stats = self._flat_stats(_bars([(101.5, 99.5), (102.5, 101.0), (103.5, 102.0)]))
        f = stats["families"]["LONG_TREND_CONTINUATION"]
        self.assertEqual(f["cf_long_n"], 1)
        self.assertEqual(f["cf_short_n"], 0)  # short side is SHORT_BREAKDOWN


class TestPromotionLadder(unittest.TestCase):
    def test_level_from_status(self):
        self.assertEqual(
            level_from_status("PROMISING-SHADOW-ONLY"), L1_RESEARCH_CANDIDATE
        )
        self.assertEqual(level_from_status("FALSIFIED"), L0_UNVALIDATED)
        self.assertEqual(level_from_status("UNVALIDATED"), L0_UNVALIDATED)
        self.assertEqual(level_from_status("PRODUCTION-VALIDATED"), L3_VALIDATED_ALPHA)
        self.assertEqual(level_from_status(None), L0_UNVALIDATED)

    def test_no_records_stays_shadow(self):
        r = promote({"status": "PROMISING-SHADOW-ONLY"})
        self.assertEqual(r["level"], L1_RESEARCH_CANDIDATE)
        self.assertEqual(r["action"], "SHADOW_ONLY")
        r2 = promote({"status": "UNVALIDATED"})
        self.assertEqual(r2["level"], L0_UNVALIDATED)
        self.assertEqual(r2["action"], "FLAT")

    def test_accruing_but_under_min_n_is_l2(self):
        r = promote(
            {
                "status": "PROMISING-SHADOW-ONLY",
                "prospective_records": 10,
                "resolved": 10,
                "min_effective_n": 50,
            }
        )
        self.assertEqual(r["level"], L2_PROSPECTIVE_VALIDATION)
        self.assertEqual(r["action"], "SHADOW_ONLY")

    def test_falsified_hard_flat_even_with_records(self):
        r = promote({"status": "FALSIFIED", "prospective_records": 99, "resolved": 99})
        self.assertEqual(r["level"], L0_UNVALIDATED)
        self.assertEqual(r["action"], "FLAT")

    def test_l3_gates_all_pass(self):
        stats = {
            "min_effective_n": 60,
            "independent_window_days": 120,
            "walk_forward_ok": True,
            "net_r_after_costs": 0.20,
            "net_r_after_stress": 0.05,
            "permutation_p": 0.01,
            "bootstrap_ci_lower": 0.10,
            "loso_ok": True,
            "n_regimes_positive": 3,
            "max_symbol_conc": 0.2,
            "max_cluster_conc": 0.4,
            "max_drawdown_r": 8.0,
            "portfolio_contribution_ok": True,
            "ece": 0.05,
        }
        r = check_l3_gates(stats)
        self.assertFalse(r["failed"])
        self.assertEqual(r["level"], L3_VALIDATED_ALPHA)

    def test_l3_gates_enumerate_failures(self):
        r = check_l3_gates({"min_effective_n": 3, "permutation_p": 0.9})
        self.assertIn("min_effective_n", r["failed"])
        self.assertIn("independent_window", r["failed"])
        self.assertIn("permutation_test", r["failed"])
        self.assertIn("cost_robustness", r["failed"])
        self.assertEqual(r["level"], L2_PROSPECTIVE_VALIDATION)
        self.assertIn("11", r["reason"])  # count of failed gates

    def test_promote_reaches_l3_when_gates_pass(self):
        stats = {
            "status": "PROMISING-SHADOW-ONLY",
            "prospective_records": 100,
            "resolved": 60,
            "min_effective_n": 50,
            "independent_window_days": 120,
            "walk_forward_ok": True,
            "net_r_after_costs": 0.20,
            "net_r_after_stress": 0.05,
            "permutation_p": 0.01,
            "bootstrap_ci_lower": 0.10,
            "loso_ok": True,
            "n_regimes_positive": 3,
            "max_symbol_conc": 0.2,
            "max_cluster_conc": 0.4,
            "max_drawdown_r": 8.0,
            "portfolio_contribution_ok": True,
            "ece": 0.05,
        }
        r = promote(stats)
        self.assertEqual(r["level"], L3_VALIDATED_ALPHA)
        self.assertEqual(r["action"], "TINY_CONTROLLED_CAPITAL")


if __name__ == "__main__":
    unittest.main()
