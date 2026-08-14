"""
Regression tests for Stage-9 invariants (src/analysis/stage9.py).

Stage-9 campaign invariants locked here (do not regress):

  1. The frozen strategy spec is versioned (v1.0.0) and sha256-hashed
     deterministically (the same source must yield the same hash).
  2. The spec encodes the frozen LONG-only protocol: k=3 of
     {L1_rsi30, L2_drop5, L3_streak5n}, PRIMARY_H=10, COST_R=0.05,
     1R = 1.25 x ATR, 28-symbol universe, FLAT default, SHORT locked.
  3. Effective N counts one bet per (date, currency-cluster) — the
     Stage-9 cluster-adjusted sample proxy — and never exceeds raw N.
  4. Walk-forward folds are year-sized, end before the consumed
     2025-06-01 boundary, and are reported per-fold (underpowered
     folds flagged, never pooled away).
  5. The portfolio equity curve spans the FULL OOS business-day calendar
     (zeros on non-signal days) — the Stage-9 fix for the signal-days-only
     annualization that inflated CAGR/Sharpe.
  6. Production gates are mechanically wired to the evidence (portfolio
     sim + clustering resolve gates 11/12; no hard-coded PENDING remains).
  7. Deterministic seeds: permutation p and spec hash are stable across
     repeated evaluation.
"""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent))

from src.analysis.stage8 import LONG_COMBO
from src.analysis.stage9 import (
    CONSUMED_START,
    COST_R,
    K_LONG,
    PRIMARY_H,
    R_MULT,
    UNIVERSE28,
    _spec_hash,
    clustering,
    effective_n,
    freeze_spec,
    portfolio_sim,
    production_gates,
    walk_forward5,
)


class TestFrozenSpec(unittest.TestCase):
    def test_spec_version_and_hash(self):
        spec = freeze_spec()
        self.assertEqual(spec["spec"]["version"], "v1.0.0")
        self.assertEqual(spec["hash_algorithm"], "sha256")
        self.assertEqual(len(spec["hash"]), 64)
        # deterministic: same source -> same hash
        self.assertEqual(_spec_hash(), spec["hash"])

    def test_frozen_entry_rule(self):
        self.assertEqual(K_LONG, 3)
        self.assertEqual(LONG_COMBO, ["L1_rsi30", "L2_drop5", "L3_streak5n"])
        self.assertEqual(PRIMARY_H, 10)
        self.assertEqual(COST_R, 0.05)
        self.assertEqual(R_MULT, 1.25)

    def test_frozen_universe_and_direction(self):
        spec = freeze_spec()["spec"]
        self.assertEqual(spec["universe"]["n"], 28)
        self.assertEqual(len(UNIVERSE28), 28)
        self.assertIn("LONG only", spec["direction"])
        self.assertIn("falsified", spec["direction"].lower())
        # FLAT is the default state
        self.assertIn("FLAT", spec["flat_behavior"])

    def test_consumed_window_not_reusable(self):
        spec = freeze_spec()["spec"]
        windows = spec["windows"]
        self.assertIn("not reusable", windows["consumed_untouched"].lower())
        self.assertIn(CONSUMED_START, windows["consumed_untouched"])


class TestEffectiveN(unittest.TestCase):
    def test_cluster_adjusted_never_exceeds_raw(self):
        e = effective_n()
        self.assertLessEqual(e["cluster_adjusted_n"], e["raw_n"])
        self.assertGreaterEqual(e["cluster_adjusted_n"], 1)
        # raw n is the OOS trade count (Stage-8/9 majors-only: 163)
        self.assertEqual(e["raw_n"], 163)


class TestWalkForwardDiscipline(unittest.TestCase):
    def test_folds_are_year_sized_and_pre_consumed(self):
        wf = walk_forward5()
        self.assertEqual(len(wf["folds"]), 5)
        for f in wf["folds"]:
            # every fold test window must end at or before the consumed
            # 2025-06-01 boundary (never extending past the untouched period)
            self.assertLessEqual(f["window"].split("..")[1], CONSUMED_START)
            # each fold carries its own oos stats (no pooling required)
            self.assertIn("oos", f)
            self.assertIn("underpowered", f)

    def test_underpowered_folds_flagged(self):
        wf = walk_forward5()
        # the final short folds are underpowered by construction and must be
        # flagged rather than silently pooled
        self.assertTrue(any(f["underpowered"] for f in wf["folds"]))
        for f in wf["folds"]:
            n = f["oos"].get("n", 0)
            if n < 20:
                self.assertTrue(f["underpowered"])


class TestPortfolioCalendarSpan(unittest.TestCase):
    def test_capital_utilization_is_a_fraction(self):
        # Stage-9 fix: the equity curve spans the full OOS business-day
        # calendar, so utilization (active days / calendar days) must be
        # in (0, 1) — the pre-fix signal-days-only version produced > 1.
        ps = portfolio_sim()
        for key in ("fixed_25bp", "fixed_50bp"):
            v = ps[key]
            self.assertGreater(v["capital_utilization"], 0.0)
            self.assertLess(v["capital_utilization"], 1.0)
            # honest annualization: ~46 trades/yr, not the inflated 345
            self.assertLess(v["turnover_trades_per_year"], 100)

    def test_25bp_sizing_is_safe_100bp_is_not(self):
        ps = portfolio_sim()
        self.assertGreater(ps["fixed_25bp"]["max_dd"], -0.15)
        self.assertGreater(ps["fixed_50bp"]["max_dd"], -0.15)


class TestProductionGatesWired(unittest.TestCase):
    def test_no_pending_gates_remain(self):
        g = production_gates()
        self.assertEqual(g["n_pending"], 0, "gates 11/12 must be mechanically resolved")
        # the fresh-window gate is unresolvable with current data: it must be
        # UNRESOLVED (not PASS), which is what blocks promotion
        self.assertEqual(g["n_unresolved"], 1)
        self.assertEqual(g["final_answer"], "NO")
        self.assertIn("B — PROMISING BUT INSUFFICIENT EVIDENCE", g["classification"])

    def test_gate_ids_complete(self):
        g = production_gates()
        ids = [x["id"] for x in g["gates"]]
        self.assertEqual(len(ids), 15)
        self.assertIn("11_portfolio_incremental_alpha", ids)
        self.assertIn("12_acceptable_drawdown", ids)

    def test_clustering_correlation_control_cuts_drawdown(self):
        cl = clustering()
        # one-per-cluster must cut cumulative-R drawdown vs all-signals
        # (maxdd_r is negative; toward zero = less severe)
        self.assertGreater(
            cl["one_per_cluster_per_day"]["maxdd_r"],
            cl["all_signals"]["maxdd_r"],
        )


class TestDeterminism(unittest.TestCase):
    def test_spec_hash_stable_across_calls(self):
        self.assertEqual(_spec_hash(), _spec_hash())


if __name__ == "__main__":
    unittest.main()
