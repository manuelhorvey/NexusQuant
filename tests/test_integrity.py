"""
Stage-11 Monitoring & Integrity battery tests.

The prospective experiment only means something if the machinery that
records, freezes and resolves stays correct under real passage of time.
These tests prove the integrity battery itself detects drift: malformed
protocol hashes, unparseable timestamps, stale recordings, duplicate
replay, missing counterfactual sides, reconciliation breaks, missing
costs, out-of-range R units and corrupt logs - and that a clean window
passes every check.

Covers:
- every check fires FAIL/WARN on its specific violation
- a clean records+resolutions pair passes everything
- the battery is read-only (never writes, never mutates inputs)
"""

from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.analysis.integrity import run_integrity
from src.analysis.protocol import protocol_hash
from src.live.recorder import (
    record_decision,
    resolve_record,
)


def _record(**overrides) -> dict:
    """Minimal well-formed snapshot (both candidate sides, flat decision)."""
    rec = {
        "recorded_at": "2026-08-14T08:00:00Z",
        "protocol": {"version": "stage11.1", "sha256": protocol_hash()},
        "date": "2026-08-14",
        "symbol": "EURUSD",
        "features": {"rsi_14": 55.0, "adx": 20.0, "atr_14": 0.004},
        "sides": {
            "long": {
                "family": "LONG_TREND_CONTINUATION",
                "entry": 1.10,
                "stop": 1.09,
                "target": 1.11,
                "cost_r": 0.01,
                "entry_type": "limit",
            },
            "short": {
                "family": "SHORT_BREAKDOWN",
                "entry": 1.10,
                "stop": 1.11,
                "target": 1.09,
                "cost_r": 0.01,
                "entry_type": "limit",
            },
        },
        "portfolio": {"n_positions": 0},
        "decision": {"direction": "flat", "status": "TRADE", "reason": "test"},
    }
    rec.update(overrides)
    return rec


def _resolution(**overrides) -> dict:
    res = {
        "date": "2026-08-14",
        "symbol": "EURUSD",
        "recorded_at": "2026-08-14T08:00:00Z",
        "protocol": protocol_hash(),
        "direction": "flat",
        "outcome": "flat",
        "r": 0.0,
        "counterfactual": {
            "long": {"realized_r": -1.01, "rungs_rr": [1.0, 2.0, 3.0]},
            "short": {"realized_r": -1.01, "rungs_rr": [1.0, 2.0, 3.0]},
        },
    }
    res.update(overrides)
    return res


def _find(result: dict, name: str) -> dict:
    for c in result["checks"]:
        if c["name"] == name:
            return c
    raise KeyError(name)


class TestIntegrityBattery(unittest.TestCase):
    def _run(self, records=None, resolutions=None):
        """Run the battery against temp paths so crash_recovery never reads
        the real log."""
        with tempfile.TemporaryDirectory() as td:
            rp = str(Path(td) / "records.jsonl")
            return run_integrity(
                records=records or [],
                resolutions=resolutions or [],
                record_path=rp,
                resolution_path=rp,
            )

    def test_clean_window_passes_everything(self):
        result = self._run(records=[_record()], resolutions=[_resolution()])
        s = result["summary"]
        self.assertEqual(s["n_fail"], 0, result["checks"])
        for c in result["checks"]:
            self.assertIn(c["status"], ("PASS", "SKIP"), c)

    def test_empty_log_skips_protocol_check(self):
        result = self._run()
        self.assertEqual(_find(result, "protocol_hash_immutability")["status"], "SKIP")
        self.assertEqual(result["summary"]["n_fail"], 0)

    def test_malformed_protocol_hash_fails(self):
        bad = _record(protocol={"version": "stage11.1", "sha256": "zzz"})
        result = self._run(records=[bad])
        c = _find(result, "protocol_hash_immutability")
        self.assertEqual(c["status"], "FAIL")
        self.assertIn("malformed", c["detail"])

    def test_mismatched_protocol_hash_still_pass_but_excluded(self):
        old = _record(protocol={"version": "stage11.1", "sha256": "0" * 64})
        result = self._run(records=[old])
        c = _find(result, "protocol_hash_immutability")
        self.assertEqual(c["status"], "PASS")  # well-formed, belongs elsewhere
        self.assertIn("excluded", c["detail"])

    def test_unparseable_timestamp_fails(self):
        result = self._run(records=[_record(recorded_at="not-a-timestamp")])
        self.assertEqual(_find(result, "timestamp_correctness")["status"], "FAIL")

    def test_future_decision_date_fails(self):
        # decision date after recorded_at -> future-data violation
        rec = _record(date="2026-09-01")
        result = self._run(records=[rec])
        self.assertEqual(_find(result, "timestamp_correctness")["status"], "FAIL")

    def test_stale_recording_warns(self):
        rec = _record(recorded_at="2026-09-01T08:00:00Z", date="2026-08-14")
        result = self._run(records=[rec])
        self.assertEqual(_find(result, "fresh_data_enforcement")["status"], "WARN")

    def test_duplicate_recorded_at_fails(self):
        a = _record()
        b = _record(symbol="GBPUSD")  # same recorded_at key -> duplicate
        result = self._run(records=[a, b])
        self.assertEqual(_find(result, "no_duplicate_observations")["status"], "FAIL")

    def test_repeated_date_symbol_warns(self):
        a = _record(recorded_at="2026-08-14T08:00:00Z")
        b = _record(recorded_at="2026-08-14T08:01:00Z")  # distinct, same pair
        result = self._run(records=[a, b])
        self.assertEqual(_find(result, "no_duplicate_observations")["status"], "WARN")

    def test_missing_counterfactual_side_fails(self):
        res = _resolution()
        del res["counterfactual"]["short"]
        result = self._run(records=[_record()], resolutions=[res])
        self.assertEqual(_find(result, "counterfactual_symmetry")["status"], "FAIL")

    def test_reconciliation_mismatch_fails(self):
        res = _resolution(
            direction="long",
            r=0.99,
            counterfactual={
                "long": {"realized_r": -1.01, "rungs_rr": [1.0]},
                "short": {"realized_r": -1.01, "rungs_rr": [1.0]},
            },
        )
        result = self._run(records=[_record()], resolutions=[res])
        c = _find(result, "realized_counterfactual_reconciliation")
        self.assertEqual(c["status"], "FAIL")
        self.assertIn("1/1", c["detail"])

    def test_missing_cost_fails(self):
        rec = _record()
        del rec["sides"]["long"]["cost_r"]
        result = self._run(records=[rec])
        self.assertEqual(_find(result, "cost_accounting")["status"], "FAIL")

    def test_negative_cost_fails(self):
        rec = _record()
        rec["sides"]["short"]["cost_r"] = -0.5
        result = self._run(records=[rec])
        self.assertEqual(_find(result, "cost_accounting")["status"], "FAIL")

    def test_out_of_range_r_units_fails(self):
        res = _resolution(
            counterfactual={
                "long": {"realized_r": 99.0, "rungs_rr": [99.0]},
                "short": {"realized_r": -1.01, "rungs_rr": [1.0]},
            }
        )
        result = self._run(records=[_record()], resolutions=[res])
        c = _find(result, "r_unit_invariance")
        self.assertEqual(c["status"], "FAIL")

    def test_corrupt_log_warns(self):
        with tempfile.TemporaryDirectory() as td:
            rp = str(Path(td) / "records.jsonl")
            Path(rp).write_text("{not json}\n" + json.dumps(_record()) + "\n")
            result = run_integrity(
                records=[_record()], resolutions=[], record_path=rp, resolution_path=rp
            )
            self.assertEqual(_find(result, "crash_recovery")["status"], "WARN")

    def test_over_concurrent_portfolio_warns(self):
        rec = _record(portfolio={"n_positions": 9})
        result = self._run(records=[rec])
        self.assertEqual(_find(result, "portfolio_caps_structural")["status"], "WARN")

    def test_battery_is_read_only(self):
        recs = [_record()]
        res = [_resolution()]
        with tempfile.TemporaryDirectory() as td:
            rp = str(Path(td) / "records.jsonl")
            before = json.dumps(recs[0], sort_keys=True)
            run_integrity(
                records=recs, resolutions=res, record_path=rp, resolution_path=rp
            )
            self.assertEqual(json.dumps(recs[0], sort_keys=True), before)
            self.assertFalse(Path(rp).exists())  # never writes

    def test_end_to_end_taken_decision_reconciles(self):
        """A real recorded + resolved LONG decision must pass the battery -
        the taken r equals its own counterfactual realized r."""
        from src.live.recorder import load_resolutions, record_resolution

        rec = _record()
        rec["decision"] = {"direction": "long", "status": "TRADE", "reason": "test"}
        rec["ladders"] = {
            "long": [
                {"tp": 1, "price": 1.11, "rr": 1.0, "source": "test"},
                {"tp": 2, "price": 1.12, "rr": 2.0, "source": "test"},
                {"tp": 3, "price": 1.13, "rr": 3.0, "source": "test"},
            ],
            "short": [],
        }
        rec["sides"]["long"]["entry_type"] = "market"
        rec["sides"]["long"]["entry"] = 1.10
        rec["sides"]["long"]["stop"] = 1.09
        rec["sides"]["long"]["target"] = 1.11
        rec["sides"]["short"]["entry"] = 1.10

        with tempfile.TemporaryDirectory() as td:
            rp = str(Path(td) / "records.jsonl")
            resp = str(Path(td) / "resolutions.jsonl")
            record_decision(rec, path=rp)
            from src.live.recorder import load_records

            df = _bars([(1.115, 1.095), (1.125, 1.105), (1.135, 1.115)])
            resolved = resolve_record(load_records(rp)[0], df)
            self.assertIsNotNone(resolved)
            record_resolution(resolved, path=resp)
            result = run_integrity(
                records=load_records(rp),
                resolutions=load_resolutions(resp),
                record_path=rp,
                resolution_path=resp,
            )
            self.assertEqual(result["summary"]["n_fail"], 0, result["checks"])
            c = _find(result, "realized_counterfactual_reconciliation")
            self.assertEqual(c["status"], "PASS")
            self.assertIn("1", c["detail"])


def _bars(days) -> pd.DataFrame:
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


if __name__ == "__main__":
    unittest.main()
