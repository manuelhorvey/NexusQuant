"""Tests for the Multi-Target Trade Ladder (src/risk/targets.py)."""

import unittest

from src.risk.targets import build_target_ladder, best_rr, MIN_RR


def levels_fixture():
    return {
        "clusters": [
            {"price": 1.115, "score": 2, "tags": ["fib_up_0.382", "pivot_r1"]},
            {"price": 1.150, "score": 3, "tags": ["swing_high", "pivot_r2"]},
            {"price": 1.190, "score": 1, "tags": ["fib_up_ext_2.618"]},
        ],
        "last_up_leg": (1.08, 1.16),
    }


class TestTargetLadder(unittest.TestCase):
    def test_ladder_from_levels(self):
        ladder = build_target_ladder(
            entry=1.10, stop=1.08, close=1.095, levels=levels_fixture()
        )
        ts = ladder["targets"]
        self.assertEqual([t["target"] for t in ts], ["TP1", "TP2", "TP3"])
        prices = [t["price"] for t in ts]
        self.assertEqual(prices, sorted(prices))
        # TP1 = nearest confluence, R:R = (1.115-1.10)/0.02
        self.assertEqual(ts[0]["price"], 1.115)
        self.assertEqual(ts[0]["rr"], 0.75)
        self.assertIn("fib_up_0.382", ts[0]["source"])
        # TP2 at the swing-high confluence clears the 2.5 floor
        tp2 = next(t for t in ts if t["target"] == "TP2")
        self.assertGreaterEqual(tp2["rr"], 2.0)
        self.assertEqual(ladder["min_rr_tp"], "TP2")
        self.assertGreaterEqual(ladder["best_rr"], 4.0)

    def test_r_multiple_fallbacks_without_levels(self):
        ladder = build_target_ladder(entry=1.10, stop=1.08, close=1.095, levels={})
        ts = ladder["targets"]
        self.assertEqual(len(ts), 3)
        self.assertEqual([t["rr"] for t in ts], [1.0, 2.0, 3.0])
        self.assertEqual(ladder["best_rr"], 3.0)
        # first TP that clears the 2.5 floor is TP3
        self.assertEqual(ladder["min_rr_tp"], "TP3")

    def test_2_5_rr_reachable_when_single_target_is_not(self):
        # single-target-to-nearest-resistance R:R is only 0.75 here, but the
        # ladder still offers an exit at >= 2.5 (spec #11 requirement)
        ladder = build_target_ladder(
            entry=1.10, stop=1.08, close=1.095, levels=levels_fixture()
        )
        self.assertLess(ladder["targets"][0]["rr"], 1.0)
        self.assertIsNotNone(ladder["min_rr_tp"])
        tp = best_rr(ladder)
        self.assertIsNotNone(tp)
        self.assertGreaterEqual(tp["rr"], MIN_RR)

    def test_degenerate_inputs(self):
        # stop above entry -> no risk, no ladder
        ladder = build_target_ladder(entry=1.10, stop=1.12, levels={})
        self.assertEqual(ladder["targets"], [])
        self.assertEqual(ladder["best_rr"], 0.0)
        self.assertIsNone(ladder["min_rr_tp"])
        # no stop
        ladder = build_target_ladder(entry=1.10, stop=None, levels={})
        self.assertEqual(ladder["targets"], [])

    def test_best_rr_none_when_floor_unreachable(self):
        ladder = build_target_ladder(entry=1.10, stop=1.08, close=1.095, levels={})
        tp = best_rr(ladder, min_rr=99.0)
        self.assertIsNone(tp)


if __name__ == "__main__":
    unittest.main()
