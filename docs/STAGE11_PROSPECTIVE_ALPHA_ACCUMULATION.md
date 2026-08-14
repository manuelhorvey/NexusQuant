# NexusQuant - Stage-11: Prospective Alpha Accumulation Campaign

**Date:** 2026-08-14 - **Status:** the Stage-10 decision engine is FROZEN and every new D1 bar is prospective evidence. Nothing is tuned against the accruing observations; the only writes are immutable decision-time snapshots and their frozen first-touch resolutions.

## 1. The question this campaign answers

> When NexusQuant says FLAT today, does the frozen research process eventually identify genuinely cost-positive opportunities as new data arrives?

Stage-10 established the architecture: search the entire opportunity space, let evidence decide LONG / SHORT / FLAT - and the correct answer on 2026-08-14 was **FLAT x 16**. That is a successful validation of the decision architecture, not a failure to trade. Stage-11 freezes that architecture and starts the clock on the **genuinely unseen window**.

## 2. What is frozen (preregistered protocol)

Protocol version **stage11.1** - hash ``8958b971e0ecb8e3...``. The full manifest (universe, features, labels, engines, TP allocation, cost model, decision rules, portfolio rules, validation gates, exit semantics) lives in `src/analysis/protocol.py` and is quoted by every snapshot at record time.

The freeze is enforced mechanically: a snapshot is eligible for evaluation only when its embedded protocol hash equals the current one (`matches_protocol`). Changing any threshold, allocation, cost, cap or gate changes the hash - which makes it a NEW protocol that must be preregistered separately, and excludes every old snapshot from its evaluation.

## 3. Window status

| Metric | Value |
|---|---|
| Snapshots recorded (all) | 16 |
| Eligible under current frozen protocol | 16 |
| Excluded (protocol mismatch) | 0 |
| Resolved with frozen first-touch semantics | 0 |

The window opened with Stage-11. As new D1 bars accrue, `--record` passes append snapshots and each subsequent pass resolves them. Nothing below is a conclusion - it is a readout of an open window.

## 4. Family accumulation + promotion ladder

Ladder: L0 UNVALIDATED (FLAT) -> L1 RESEARCH CANDIDATE (shadow) -> L2 PROSPECTIVE VALIDATION (shadow) -> L3 VALIDATED ALPHA (tiny controlled capital) -> L4 PRODUCTION CANDIDATE (controlled deployment) -> L5 PRODUCTION (monitored). The L3 gate battery is the research firewall: min effective N, independent window, walk-forward, cost robustness, permutation test, bootstrap CI, LOSO, regime stability, concentration limits, portfolio contribution, drawdown limits (`src/analysis/promotion.py`).

| Family | Research status | Ladder level | Action | Records | Resolved | Taken mean R | CF long mean R | CF short mean R | Over-restrict n |
|---|---|---|---|---|---|---|---|---|---|
| LONG_BREAKOUT | UNVALIDATED | L0_UNVALIDATED | FLAT | 0 | 0 | - | - | - | 0 |
| LONG_BREAKOUT_RETEST | UNVALIDATED | L0_UNVALIDATED | FLAT | 0 | 0 | - | - | - | 0 |
| LONG_BUY_DIP | UNVALIDATED | L0_UNVALIDATED | FLAT | 0 | 0 | - | - | - | 0 |
| LONG_MEAN_REVERSION | UNVALIDATED | L0_UNVALIDATED | FLAT | 0 | 0 | - | - | - | 0 |
| LONG_REVERSAL | PROMISING-SHADOW-ONLY | L1_RESEARCH_CANDIDATE | SHADOW_ONLY | 0 | 0 | - | - | - | 0 |
| LONG_TREND_CONTINUATION | UNVALIDATED | L0_UNVALIDATED | FLAT | 0 | 0 | - | - | - | 0 |
| SHORT_BREAKDOWN | UNVALIDATED | L0_UNVALIDATED | FLAT | 0 | 0 | - | - | - | 0 |
| SHORT_BREAKDOWN_RETEST | UNVALIDATED | L0_UNVALIDATED | FLAT | 0 | 0 | - | - | - | 0 |
| SHORT_MEAN_REVERSION | UNVALIDATED | L0_UNVALIDATED | FLAT | 0 | 0 | - | - | - | 0 |
| SHORT_REVERSAL | FALSIFIED | L0_UNVALIDATED | FLAT | 0 | 0 | - | - | - | 0 |
| SHORT_SELL_RALLY | UNVALIDATED | L0_UNVALIDATED | FLAT | 0 | 0 | - | - | - | 0 |
| SHORT_TREND_CONTINUATION | UNVALIDATED | L0_UNVALIDATED | FLAT | 0 | 0 | - | - | - | 0 |

*Taken mean R* = mean realized R of taken decisions in the window; *CF mean R* = mean counterfactual realized R of the side had it been taken (FLAT-rejected candidates included, 1/3-1/3-1/3 allocation, costs deducted). `None` = no observations yet.

## 5. Self-diagnosing threshold evidence

The accumulation window is designed to detect BOTH failure modes of a filter:

**Filter effectiveness** - among FLAT decisions with resolvable counterfactuals, how often would BOTH candidate sides have lost money? A high share means the filter is correctly rejecting losers:

- FLAT decisions with counterfactuals: 0
- Both candidate sides negative (filter saved the loss): 0
- Filter effectiveness: **n/a (no observations)**

**Over-restriction evidence** - how many rejected candidates would have cleared +0.3R realized after costs? A material count means the threshold may be too strict - a claim that can only be evaluated by the preregistered protocol on the completed window, never by acting on the interim readout:

- Over-restriction count: **0**
- Detail: none

**FLAT-decision classification** - the most important Stage-11 measurement: is NexusQuant actually good at knowing when NOT to trade? Each FLAT decision is classified by what its counterfactuals subsequently did:

| Class | Meaning | Count |
|---|---|---|
| A. Correct FLAT | both candidate sides subsequently lost | 0 |
| B. Over-restriction | a rejected side would have made >= +0.30R after costs | 0 |
| C. Correct directional rejection | the decision-time higher-ranked side won; the other lost | 0 |
| ambiguous | positive but below the over-restriction bar | 0 |
| unresolved | no forward bars to resolve | 0 |

A preponderance of A means the gates are correctly keeping you out of losers. A growing B count means the system is too conservative (evaluate ONLY by the preregistered protocol on the completed window). C means the opportunity-ranking mechanism is correctly identifying the superior side even when it declines to trade. **The measurement itself is the deliverable - do not act on interim counts.**

## 6. Promotion discipline

- **Nothing is promoted on interim observations.** A level change requires the frozen protocol's evaluation of the COMPLETED window (min effective N, independent window, and the full L3 battery).
- **LONG_REVERSAL stays L1 RESEARCH CANDIDATE (shadow)** until the fresh-window gate closes with genuinely new observations - it is never silently promoted to production.
- **SHORT_REVERSAL stays L0/FALSIFIED (hard FLAT)** - Stage-6 falsification is not revisited by the window; only a new preregistered hypothesis can be.
- **A protocol change is a new campaign**, not an amendment: new hash, old snapshots excluded, and the new protocol must itself be frozen before it accrues.

## 7. How to run

```bash
./venv/bin/python -m src.live.run --format diagnostics --record   # record every decision (incl. FLAT) each pass
./venv/bin/python -m src.live.run --format institutional --record # richer snapshots (mtf)
./venv/bin/python -m src.analysis.stage11                         # accumulate + readout
./venv/bin/python -m src.analysis.stage11 --all                   # + rewrite this doc
```

Snapshots: `data/validation/prospective_records.jsonl` (immutable, append-only). Resolutions: `data/validation/prospective_resolutions.jsonl` (separate file - snapshots never mutate).

## 8. What a completed window looks like

When enough untouched bars have accrued (Stage-9 protocol: single-shot evaluation of the frozen strategy on the fresh window, no tuning), the readout becomes decidable: either a family clears the L3 battery and moves to tiny controlled capital, or it fails a gate and stays put / is re-classified. **The correct output may remain FLAT for a long time - that is a feature, not a bug.** A systematic trading system has no psychological requirement to trade every day.
