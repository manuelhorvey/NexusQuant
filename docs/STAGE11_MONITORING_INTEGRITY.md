# NexusQuant - Stage-11: Monitoring & Integrity Audit

**Date:** 2026-08-14 - **Purpose:** prove that the prospective recorder, protocol freeze, counterfactual engine and promotion machinery remain correct under real passage of time. This battery runs on every audit pass; a FAIL here blocks any promotion review until resolved.

Audited: **16 snapshots, 0 resolutions** - summary: **14 PASS, 0 FAIL, 0 WARN, 0 SKIP**.

## Check results

| Check | Status | Detail |
|---|---|---|
| protocol_hash_immutability | PASS | 16 snapshots match the current frozen protocol; 0 belong to another protocol and are correctly excluded from evaluation |
| timestamp_correctness | PASS | all 16 snapshots have parseable ISO UTC timestamps with decision date <= recorded_at |
| fresh_data_enforcement | PASS | all snapshots within 7 days of their bar date |
| no_duplicate_observations | PASS | 16 unique observations, no repeats |
| no_future_data_contamination | PASS | 0 resolution(s) reference their decision date; resolution is structurally forward-only (searchsorted right, strict post-decision horizon) |
| decision_time_features | PASS | all snapshots carry decision-time features |
| candidate_side_symmetry | PASS | every snapshot carries both LONG and SHORT candidate sides |
| counterfactual_symmetry | PASS | every resolution carries both counterfactual sides |
| realized_counterfactual_reconciliation | PASS | 0 taken decisions reconcile exactly with their counterfactual realized R |
| cost_accounting | PASS | every leveled side carries a non-negative cost assumption |
| limit_non_fill_semantics | PASS | limit entries fill only on a zone touch; untouched zones resolve non_fill at 0R (no cost). Non-fills observed: 0 |
| r_unit_invariance | PASS | all rungs and realized R within R-unit bounds |
| crash_recovery | PASS | log is append-only and fully parseable; a partial write is tolerated (skipped, never fatal) |
| portfolio_caps_structural | PASS | live selection enforces cluster/concurrent/heat caps by code; snapshot portfolio states are within bounds |

## Rules of the road

- **A FAIL means the prospective evidence cannot be trusted as-is** - investigate before any promotion review. A WARN means evidence may be incomplete (stale recording, damaged log, possible replay) - quantify and document. SKIP means the log is empty or unreadable (nothing to audit yet).
- **The freeze is mechanical**: any snapshot whose protocol hash differs from the current manifest is excluded from evaluation by `matches_protocol` - it belongs to a different experiment, not this one.
- **FLAT is a successful system outcome.** The measurement that matters is whether the system correctly distinguishes LONG opportunity, SHORT opportunity, and no economically valid opportunity - not how many trades it produces.

## Reproduce

```bash
./venv/bin/python -m src.analysis.integrity            # run the battery
./venv/bin/python -m src.analysis.integrity --all      # + rewrite this doc
```

