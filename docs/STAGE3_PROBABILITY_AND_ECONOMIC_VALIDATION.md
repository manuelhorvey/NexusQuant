# NexusQuant — Stage-3: Probability & Economic Validation

**Date:** 2026-08-13
**Status:** Research report — verdict **NOT PRODUCTION READY**
**Module:** `src/analysis/stage3.py` (rerunnable, see §16)
**Data:** 16 symbols × D1 (EURUSD, GBPUSD, USDJPY, USDCHF, AUDUSD, AUDCHF, NZDUSD,
USDCAD, EURJPY, GBPJPY, NZDJPY, AUDJPY, CADJPY, CHFJPY, GBPCAD, XAUUSD);
`data/raw/full_fx`, ~40k bars, all fresh through 2026-08-12/13.

This is the second-stage forensic validation of the two-sided engine. Stage-2
proved the **architecture** can see both sides (LONG/SHORT/FLAT from evidence,
EV-driven competition, 200-SMA contextual, AUDJPY flip regression-tested).
Stage-3 asks the economic question: **after honest probability calibration,
target modeling, costs and out-of-sample validation, does genuine tradable edge
remain?**

The one-line answer: **calibration is fixable and the target-level economics are
now precisely measured — but the directional signal itself is weak, and no
exit architecture or cost assumption examined here converts it into positive
net expectancy.** The system remains NOT PRODUCTION READY, with the evidence
narrowed to two candidate families (LONG_MEAN_REVERSION, SHORT_BREAKDOWN) and a
handful of regime×vol cells that warrant focused follow-up.

---

## 1. Baseline (frozen)

| Item | Value |
|---|---|
| HEAD | `4410096` (phase-8 campaign commit) + uncommitted two-sided fixes in working tree |
| Models | `models/dip_lgbm.joblib` (long), `models/rally_lgbm.joblib` (short), trained 2026-08-12 |
| Model split | 2022-01-01 (train < split ≤ test), both bundles carry an isotonic calibrator |
| Model config | LGBM n_est=400, lr=0.04, leaves=31, depth=7, min_child=20, l2=1.0 |
| Label geometry | stop = 1.25×ATR(14), target = 0.75×ATR(14), horizon 10 bars (win pays **0.6R**) |
| Ladder geometry | risk = 1.25×ATR, rungs at 1R/2R/3R, stop −1R, first-touch, horizon 20 |
| Decision gates | EV floor +0.2R (ranking EV), R:R ≥ 2.5 (ladder-best), macro gate, entry ≤ 0.25×ATR → MARKET else LIMIT |
| Cost assumption | 0.05R round trip (reference), grid 0 → 0.20R for the sweep |
| Stage-2 census | LONG 5,764 candidates / 222 signals / 57.2% win / **+0.847R** raw exp.; SHORT 3,599 / 79 / 59.5% / **+0.720R**; L:S candidate ratio Bear 0.10, Bull 11.42 |
| Stage-2 calibration | LONG Brier 0.281 / ECE 0.202 (predicts 44%, actual 63%); SHORT Brier 0.259 / ECE 0.124 |
| Stage-2 target EV | LONG +0.049 → −0.002 @0.05R; SHORT +0.023 → −0.027 @0.05R |

**Reconciliation note (important).** Stage-2's "+0.847R / +0.720R signal
expectancy" measured *signal-confirmed* bars with the engines' own entry/target
geometry and R values (ladder R:R up to 3.0) on small samples (222 / 79
signals). Stage-3 measures every classified candidate under the uniform ML-label
geometry (win 0.6R / loss 1R, break-even win rate **62.5%**) and the ladder
first-touch geometry (1R/2R/3R). Under those honest geometries the same
classifier output shows expectancy ≈ 0. The Stage-2 number was favorable
selection + geometry, not a robust edge.

---

## 2. Calibration experiments (Phases 2/3)

Method: strictly out-of-sample. Calibrators are fit **only on pre-split bars**
(before the models' own 2022-01-01 training boundary) and evaluated **only on
post-split bars** — out-of-sample for both the model and the calibrator. Every
bar is pooled per side across all 16 symbols. Long side uses the dip signal
context, short side the rally signal context (identical to live inference).

| Method (LONG, n_tr=12,257 / n_te=13,584) | Brier | ECE | MCE | LogLoss | slope | mean→actual |
|---|---|---|---|---|---|---|
| raw model output | 0.3184 | 0.2583 | 0.5255 | 0.8593 | 0.010 | 0.401→0.642 |
| **saved bundle calibrator** | 0.2864 | **0.2223** | 0.6660 | 0.8141 | 0.012 | 0.433→0.642 |
| isotonic (OOS-fitted) | 0.2316 | **0.0367** | 0.8655 | 0.6579 | 0.089 | 0.605→0.642 |
| Platt (OOS-fitted) | 0.2313 | **0.0361** | 0.1431 | 0.6554 | 0.231 | 0.605→0.642 |
| beta (OOS-fitted) | 0.2313 | **0.0361** | 0.1431 | 0.6554 | 0.231 | 0.605→0.642 |

| Method (SHORT, n_tr=12,178 / n_te=13,461) | Brier | ECE | MCE | LogLoss | slope | mean→actual |
|---|---|---|---|---|---|---|
| raw model output | 0.3284 | 0.2480 | 0.5352 | 0.9525 | 0.025 | 0.493→0.587 |
| **saved bundle calibrator** | 0.2575 | **0.1023** | 0.4737 | 0.7978 | 0.113 | 0.491→0.587 |
| isotonic (OOS-fitted) | 0.2440 | **0.0403** | 0.1857 | 0.6850 | 0.411 | 0.624→0.587 |
| Platt (OOS-fitted) | 0.2438 | **0.0375** | 0.0807 | 0.6809 | 0.371 | 0.622→0.587 |
| beta (OOS-fitted) | 0.2438 | **0.0374** | 0.0808 | 0.6808 | 0.372 | 0.622→0.587 |

**Findings**

1. **The saved bundle calibrators are effectively broken** — LONG bundle ECE
   0.222 is barely better than raw (0.258); SHORT bundle 0.102. The model
   factory's calibrator does not generalize to the post-split window (or was
   fit/misapplied in a way that fails out-of-sample). The Stage-2 ECE 0.202/0.124
   was therefore a *conservative* estimate of the production defect, not the
   model's ceiling.
2. **Properly OOS-fitted calibrators meet the aspirational ECE < 0.05 target**:
   ECE 0.036 (long) / 0.038 (short) via Platt or beta; isotonic 0.037/0.040.
   Platt/beta additionally give the best MCE (0.08–0.14) and slope (0.23–0.37),
   i.e. a usable probability scale across the whole range rather than isotonic's
   flat-tail artifacts. **Recommendation: replace the bundle calibrators with
   OOS-fitted Platt (or beta) recalibration, refit on pre-split data, re-validate.**
3. **Per-regime calibrators do not beat the global one** (long 0.041–0.067,
   short 0.035–0.067 vs global 0.036/0.038). Given sample-size requirements per
   regime and no material gain, the global calibrator is preferred — per-regime
   adds complexity without measured benefit.
4. **Calibration cannot fix ranking.** Raw slopes are 0.01 (long) / 0.025
   (short): the raw scores are nearly flat, so the models are weak *rankers*.
   Calibration makes the probabilities *honest*, not *stronger*. Brier after
   calibration (0.231/0.244) is the residual model quality and is unchanged by
   any calibrator.
5. Both models are demonstrably **independent** (mean raw 0.401 vs 0.493;
   different calibration curves) — P(short) = 1 − P(long) remains rejected.

---

## 3. Target probability model (Phase 4)

First-touch multi-barrier resolution (stop −1R, rungs +1R/+2R/+3R, horizon 20)
over every classified candidate (score ≥ 0.45). Probabilities are mutually
consistent by construction: P(TP2) ≤ P(TP1) because TP2 requires passing TP1.

| Side | P(TP1) | P(TP2) | P(TP3) | P(SL) | EV@0R | EV@0.05R |
|---|---|---|---|---|---|---|
| LONG (all) | 0.479 | 0.016 | 0.002 | 0.503 | +0.009 | −0.041 |
| SHORT (all) | 0.453 | 0.031 | 0.006 | 0.510 | +0.029 | −0.021 |

The probability of reaching a far target *before the stop* is fundamentally
different from directional accuracy — the model predicts direction (≈58–62%),
but only ~48% of candidates touch even +1R before −1R, and 2–3R is almost never
reached within 20 bars. **The ladder-best × P(direction) ranking EV (the
+1.1R-style number still shown live) materially overstates the payoff
distribution and must not be used as a decision probability.**

The current implementation keeps `ev_target_level` in the opportunity book
(computed from the cached TP table) alongside the ranking EV, with both printed
in `--format diagnostics`. A full survival/competing-risks model (per-symbol,
per-family) is the documented next step (§14); the empirical first-touch table
is the honest substitute available today and is used for every number in this
report.

---

## 4. Exit analysis (Phase 6)

Identical entries (every bar, uniform geometry), six exit policies, horizon 20:

| Policy | LONG meanR | LONG win% | LONG breakeven | SHORT meanR | SHORT win% | SHORT breakeven |
|---|---|---|---|---|---|---|
| TP1 only | +0.017 | 51% | 0.025R | −0.081 | 46% | 0.0R |
| TP2 only | −0.893 | 4% | 0.0R | −0.843 | 5% | 0.0R |
| TP3 only | −0.984 | 0% | 0.0R | −0.953 | 1% | 0.0R |
| Partial (½@TP1, ½@SL) | −0.203 | 52% | 0.0R | −0.247 | 48% | 0.0R |
| Trailing (BE@TP1, +1R@TP2) | −0.016 | 26% | 0.0R | −0.120 | 23% | 0.0R |
| Time-stop (mark to 20-bar close) | **+0.145** | 54% | **0.15R** | −0.145 | 46% | 0.0R |

**Findings**

1. **The 1R/2R/3R ladder itself destroys expectancy.** TP2/TP3 rungs win 0–5%
   of the time; any policy that routes the rider half through them is negative.
   This directly explains the Phase 7 result (§6): the ladder is the wrong exit
   architecture for this horizon.
2. **No exit policy rescues the short side** — every short policy is negative
   before costs. The short signal's fixed-horizon edge is absent (§5), so the
   exit problem is secondary on that side.
3. Long **time-stop** (+0.145R, 54% win, survives ~0.10–0.15R costs) is the
   least-bad policy and beats TP1-only — consistent with the (weak) positive
   drift in the universe rather than with ladder geometry. TP1-only survives
   only ~0.025R of costs.
4. Exit policy choice is a second-order lever given the signal weakness: the
   spread between best and worst *defensible* policies is ~0.15R, smaller than
   the gap between signal edge and zero.

---

## 5. Signal edge vs exit edge (Phase 8) — the critical separation

Fixed-horizon ATR-normalized forward returns (no SL/TP at all) per family ×
regime. * = |t| > 2. Mean returns in ATR units.

| Family \| Regime | h1 | h3 | h5 | h10 | h20 |
|---|---|---|---|---|---|
| LONG_BUY_DIP \| Bull Trend | −0.019 | −0.017 | −0.037 | **−0.251\*** | −0.311 |
| LONG_BUY_DIP \| Range/Chop | −0.013 | −0.032 | −0.040 | **−0.082\*** | −0.093 |
| LONG_TREND_CONTINUATION \| Bear Trend | **+0.130\*** | **+0.134\*** | **+0.224\*** | +0.221 | +0.112 |
| LONG_TREND_CONTINUATION \| Bull Trend | +0.008 | −0.004 | −0.024 | **−0.064\*** | **−0.137\*** |
| LONG_MEAN_REVERSION \| Range/Chop | +0.044 | **+0.420\*** | **+0.488\*** | **+0.760\*** | **+0.693\*** |
| SHORT_SELL_RALLY \| Range/Chop | −0.039 | −0.002 | −0.029 | −0.078 | −0.060 |
| SHORT_TREND_CONTINUATION \| Bear Trend | −0.017 | −0.020 | −0.049 | −0.069 | −0.114 |
| SHORT_TREND_CONTINUATION \| Bull Trend | −0.018 | −0.043 | −0.018 | +0.059 | +0.043 |
| SHORT_BREAKDOWN \| Range/Chop | +0.032 | −0.017 | +0.016 | +0.011 | −0.183 |
| SHORT_MEAN_REVERSION \| Range/Chop | −0.075 | −0.124 | −0.130 | −0.035 | +0.039 |

(Blank cells = n < 30.)

**Findings**

1. **The entry signal itself is weak at fixed horizons.** The two significant
   positives are LONG_MEAN_REVERSION in Range (+0.42 to +0.76 ATR at h3–h20,
   t ≈ 2.2–2.5, but n=40) and LONG_TREND_CONTINUATION in Bear Trend at h1–h5
   (+0.13 to +0.22, t > 2) — a counter-trend short-horizon effect.
2. **The core long family is not supported at horizon.** LONG_BUY_DIP is
   *negative* at h10/h20 in both Bull Trend and Range (−0.25/−0.31, −0.08).
   The original system's flagship setup shows no fixed-horizon directional edge.
3. **Short families are uniformly weak**: SELL_RALLY, TREND_CONTINUATION and
   MEAN_REVERSION shorts show no positive fixed-horizon edge in any regime;
   breakdown shorts are flat. The short side's positive census expectancy was
   geometry/selection, not a fixed-horizon effect.
4. Therefore the ≈0 target-level EV is **not primarily an exit-architecture
   problem**: the directional information at the entry point is itself ~zero for
   most families. (Long time-stop's +0.145R on *all* bars is unconditional
   drift, present without any signal.)

---

## 6. Regime × volatility target-level EV (Phases 7/10) — why TP2/TP3 collapse

| Side \| Regime \| Vol | n | P(TP1) | P(TP2) | P(TP3) | P(SL) | EV@0 | EV@0.05 |
|---|---|---|---|---|---|---|---|---|
| LONG \| ALL \| HIGH | 1,290 | 0.533 | 0.016 | 0.002 | 0.449 | +0.122 | +0.072 |
| LONG \| ALL \| LOW | 1,362 | 0.476 | 0.035 | 0.002 | 0.487 | +0.066 | +0.016 |
| LONG \| ALL \| MED | 2,941 | 0.479 | 0.014 | 0.002 | 0.505 | +0.008 | −0.042 |
| LONG \| **Bear Trend** \| ALL | 192 | 0.578 | 0.021 | 0.000 | 0.401 | **+0.219** | **+0.169** |
| LONG \| Bear Trend \| HIGH | 52 | 0.635 | 0.000 | 0.000 | 0.365 | **+0.269** | **+0.219** |
| LONG \| Bull Trend \| ALL | 2,544 | 0.456 | 0.024 | 0.004 | 0.517 | −0.003 | −0.053 |
| LONG \| Range/Chop \| HIGH | 539 | 0.570 | 0.011 | 0.000 | 0.419 | **+0.172** | **+0.122** |
| SHORT \| ALL \| HIGH | 877 | 0.414 | 0.022 | 0.008 | 0.556 | −0.075 | −0.125 |
| SHORT \| ALL \| LOW | 682 | 0.474 | 0.034 | 0.006 | 0.487 | +0.072 | +0.022 |
| SHORT \| ALL \| MED | 1,955 | 0.463 | 0.035 | 0.005 | 0.497 | +0.050 | −0.000 |
| SHORT \| **Bear Trend** \| LOW | 324 | 0.528 | 0.031 | 0.003 | 0.438 | **+0.161** | **+0.111** |
| SHORT \| Bear Trend \| MED | 1,015 | 0.464 | 0.032 | 0.006 | 0.499 | +0.046 | −0.004 |
| SHORT \| Range/Chop \| HIGH | 244 | 0.426 | 0.016 | 0.000 | 0.557 | −0.098 | −0.148 |

**Findings**

1. **TP2/TP3 collapse is structural, not a tuning artifact**: at 2–3R away with
   1.25×ATR risk inside a 20-bar horizon, P(TP2) ≈ 2–3% and P(TP3) ≈ 0.2–0.8%
   everywhere. The nearest-target R:R (0.9–1.25) the risk layer reports is the
   economically honest one; the ladder-best 3R is a theoretical ceiling, not an
   expected payoff.
2. **The aggregate ≈0 hides real regime structure on the LONG side**: longs in
   Bear Trend (+0.169–0.219 @0.05R) and Range HIGH vol (+0.122 @0.05R) are
   consistently positive — this is the counter-trend/vol-expansion long effect
   also visible in §5. Bull Trend longs are the worst cell (−0.053).
3. **Short side**: only Bear Trend LOW vol (+0.111 @0.05R) clears costs; HIGH
   vol shorts are the worst cell (−0.125 to −0.148). The short edge, where it
   exists, is a *quiet* bear-market mean-reversion/breakdown effect — exactly
   the opposite of "sell into high-volatility momentum".

---

## 7. Setup-family attribution (Phase 11)

| Family | n | win% | expR (0.6R geometry) | tpEV@0 (ladder) | fwd5 (t) | fwd10 (t) |
|---|---|---|---|---|---|---|
| LONG_TREND_CONTINUATION | 3,835 | 60% | −0.003 | +0.040 | −0.012 (−0.7) | −0.043 (−1.8) |
| SHORT_TREND_CONTINUATION | 3,226 | 58% | −0.027 | +0.022 | −0.054 (−2.2) | −0.054 (−1.6) |
| LONG_BUY_DIP | 1,639 | 60% | +0.002 | +0.053 | −0.041 (−1.6) | −0.093 (−2.8) |
| LONG_BREAKOUT | 217 | 59% | −0.010 | +0.090 | +0.002 (0.0) | −0.055 (−0.5) |
| SHORT_SELL_RALLY | 176 | 58% | −0.016 | +0.006 | −0.030 (−0.3) | −0.080 (−0.5) |
| SHORT_BREAKDOWN | 121 | 65% | **+0.069** | **+0.075** | −0.122 (−1.1) | −0.122 (−0.7) |
| SHORT_MEAN_REVERSION | 60 | 50% | −0.183 | +0.035 | −0.117 (−0.8) | −0.059 (−0.3) |
| LONG_MEAN_REVERSION | 40 | **70%** | **+0.145** | **+0.342** | **+0.451 (2.2)** | **+0.703 (2.5)** |

**Findings**

1. **Only LONG_MEAN_REVERSION shows a statistically significant, cost-surviving
   edge on every metric** (70% win, +0.145R exp., +0.342R tpEV, fwd10 +0.70 ATR
   with t = 2.5). It is the smallest family (n=40) — a real candidate, but
   sample size is a hard limit until more data/families confirm it.
2. SHORT_BREAKDOWN is the only short family with positive expectancy (+0.069R
   exp., +0.075 tpEV, 65% win) — but its fixed-horizon returns are negative,
   so the edge is fragile (dependent on stop/target geometry and small n=121).
3. **Every other family fails the 62.5% break-even bar under the model's own
   label geometry** (58–60% wins × 0.6R ≠ positive expectancy) and shows no
   positive fixed-horizon edge. LONG_BUY_DIP and SHORT_TREND_CONTINUATION — the
   two most-traded families — are both ≈0 or negative.
4. Families that lack a demonstrable edge should be **removed or de-ranked**
   from production per the Stage-2 mandate ("remove setup families that have no
   demonstrable edge"), with the caveat that sample sizes are small for the
   rare families.

---

## 8. Symbol robustness (Phase 12)

| Symbol | n | win% | expR | tpEV | fwd5 (t) | fwd10 (t) |
|---|---|---|---|---|---|---|
| XAUUSD | 700 | 66% | +0.073 | **+0.195** | +0.000 (4.1) | +0.000 (5.5) |
| CHFJPY | 699 | 60% | −0.018 | +0.056 | +0.002 (4.7) | +0.004 (6.2) |
| EURJPY | 546 | 62% | +0.033 | +0.101 | +0.001 (1.4) | +0.000 (0.8) |
| EURUSD | 548 | 62% | +0.017 | +0.067 | −0.022 (−0.4) | −0.048 (−0.7) |
| GBPCAD | 705 | 61% | −0.007 | +0.030 | +0.011 (0.3) | +0.024 (0.5) |
| GBPJPY | 562 | 59% | −0.006 | +0.065 | +0.000 (0.8) | +0.001 (1.7) |
| NZDJPY | 534 | 58% | −0.025 | +0.033 | +0.000 (0.1) | +0.001 (0.5) |
| USDJPY | 503 | 58% | −0.042 | +0.045 | +0.001 (1.0) | +0.001 (1.7) |
| USDCAD | 562 | 58% | −0.023 | −0.018 | +0.038 (0.9) | +0.076 (1.2) |
| AUDJPY | 562 | 59% | −0.012 | +0.015 | −0.000 (−0.3) | +0.001 (0.9) |
| AUDUSD | 543 | 60% | −0.002 | +0.043 | −0.074 (−0.9) | −0.129 (−1.1) |
| NZDUSD | 526 | 60% | −0.010 | +0.052 | −0.160 (−1.8) | −0.297 (−2.4) |
| AUDCHF | 691 | 60% | −0.001 | +0.021 | −0.187 (−1.9) | −0.273 (−1.9) |
| GBPUSD | 563 | 57% | −0.053 | −0.062 | −0.017 (−0.4) | −0.043 (−0.7) |
| USDCHF | 540 | 55% | −0.062 | −0.034 | −0.096 (−1.5) | −0.196 (−2.0) |
| CADJPY | 530 | 55% | −0.051 | −0.039 | −0.000 (−0.4) | +0.000 (0.3) |

**Findings**

1. **XAUUSD and CHFJPY show persistent positive drift** (t ≈ 4–6 on tiny
   positive means — small per-bar drift, very consistent). XAUUSD also has the
   best tpEV (+0.195). These two are the only symbols whose *unconditional*
   direction is statistically reliable in-sample.
2. **NZDUSD, AUDCHF, USDCHF have the worst forward returns** (t ≈ −1.8 to −2.4)
   — consistent negative drift for the candidates tested; the short side on
   these names deserves the follow-up, not the long side.
3. Per-symbol samples (500–700) are too small to declare per-symbol edges; no
   per-symbol thresholds are justified today (and none are used).

---

## 9. Cost threshold / break-even (Phase 9)

Derived from the exit sweep (mean R per policy, cost grid 0 → 0.20R):

| Side \| Policy | Gross meanR | Break-even cost |
|---|---|---|
| LONG \| TP1 only | +0.017 | **0.025R** |
| LONG \| time-stop | +0.145 | **0.15R** |
| LONG \| trailing | −0.016 | 0 |
| SHORT \| any policy | ≤ −0.081 | **0** |

**Findings**

- The long TP1 policy breaks even at 0.025R — below any realistic FX
  round-trip (0.05R reference). The long time-stop is the only policy with an
  economic margin, and only if costs stay under ~0.15R.
- **No short policy is profitable even at zero cost** under the uniform
  geometry; costs are not the reason the short side fails.
- The targeted family cells (§6) — LONG Bear Trend / Range HIGH, SHORT Bear
  Trend LOW — have break-even headroom of 0.11–0.22R, which *is* cost-viable;
  they are the only cost-robust cells found.

---

## 10. Walk-forward validation (Phase 13) — protocol, not yet run

**Not run by design.** The models and calibrators are not yet frozen: the §2
result shows the saved calibrators must be replaced (Platt/beta OOS-fitted) and
the model stack re-validated. Running a purged walk-forward against the current
calibrators would validate a defect. The protocol is locked:

- Chronological folds (e.g. 5 folds, 2-year test each), purge = 20 bars,
  embargo = 10 bars after each fold boundary.
- Calibrators fit only inside each training fold; never on test.
- Thresholds (EV floor, R:R floor, family score) fixed *before* the study; no
  test-set tuning.
- Per fold, per side and combined: expectancy, net expectancy, Sharpe, Sortino,
  profit factor, max DD, win rate, Brier, ECE, and PSR/DSR computed from the
  pooled out-of-sample trades.

Every Stage-3 analysis above is causal and deterministic, so the walk-forward
can consume the same pipelines unchanged.

## 11. Untouched final test (Phase 14) — reserved

The most recent period (e.g. last 12 months) will be **reserved untouched** and
run exactly once after calibration/model decisions are frozen. No Stage-3
number in this report used it.

## 12. Ablation (Phase 15) — partial evidence

A full feature-level ablation is deferred to the walk-forward stage, but the
regime×family decomposition above is already a meaningful ablation of the
*decision structure*:

- **Removing the family layer** (e.g. treating all setups alike): expectancy ≈
  0 — the aggregate hides the two candidate families.
- **Removing the regime layer**: the LONG Bear Trend / Range-HIGH and SHORT
  Bear-Trend-LOW cells are the only cost-surviving regions; a regime-agnostic
  deployment would trade the negative cells too.
- **Removing the exit ladder** (time-stop instead): +0.13R swing on long —
  exit structure matters, but less than the signal's absence.
- The strongest *component* evidence: LONG_MEAN_REVERSION's edge is driven by
  Range/Chop + high-vol conditions; the short edge, where present, is
  Bear-Trend LOW-vol. These are the two independent information sources worth
  keeping for the next iteration.

## 13. Multiple-testing assessment (Phase 16)

Honest experiment census for this campaign (all documented):

| Experiment family | Count |
|---|---|
| Setup families evaluated | 12 (6 long + 6 short) |
| Calibration methods compared | 5 × 2 sides (raw, bundle, isotonic, Platt, beta) |
| Exit policies | 6 × 2 sides |
| Regime × vol cells | 44 reported |
| Cost levels | 8 |
| Horizons | 5 |
| Thresholds tuned | 0 (family score 0.45, EV +0.2R, R:R 2.5 all pre-existing) |
| Per-symbol thresholds | 0 |

Selection risk is material: the two surviving families (LONG_MEAN_REVERSION
n=40, SHORT_BREAKDOWN n=121) were identified from this same data. **They must
be treated as hypotheses, not confirmed edges** — PBO/DSR are only meaningful
after the walk-forward produces out-of-sample trades, which is exactly why the
verdict below does not rest on them. No thresholds were moved to improve any
number in this report.

## 14. Remaining risks

1. **Small samples on the only positive cells**: LONG_MEAN_REVERSION (n=40),
   SHORT_BREAKDOWN (n=121), LONG Bear-Trend cells (n=192). All could be
   noise; the t-stats on mean reversion are the only ones above 2.
2. **Weak rankers**: raw model slopes 0.01–0.025 — calibration fixes the scale,
   not the information. Model improvement (features/labels/horizon) is the
   real bottleneck, not calibration plumbing.
3. **Label-geometry mismatch with the live ladder**: models train on
   target 0.75×ATR (0.6R) while the live ladder uses 1R/2R/3R rungs; the
   bridge is the empirical first-touch table, which is exactly what shows the
   ladder is unreachable. A per-target probability model (survival/
   competing-risks) is the correct next build.
4. **Universe drift**: XAUUSD/CHFJPY positive drift and NZDUSD/AUDCHF/USDCHF
   negative drift are in-sample; both could reverse.
5. **No walk-forward / untouched test yet**: by design (§10/§11), but it means
   every number above is single-split OOS, not multi-fold OOS.
6. **Cost model is a flat R charge**: no spread-widening stress, financing,
   or impact; the 0.05R reference is standard but not stressed beyond the grid.
7. **The live `ev_target_level` is a cached table** keyed by the 16-symbol
   universe — it must be regenerated (`--write-probs` equivalent) whenever
   symbols or data extend beyond the cached universe.

## 15. Final production gates

| Gate | Criterion | Status today |
|---|---|---|
| CALIBRATION | ECE < 0.05 OOS on refit calibrators | ✅ achievable (0.036/0.038); ❌ not yet deployed |
| TARGET MODEL | stable OOS per-target probabilities | ❌ empirical table only; survival model not built |
| ECONOMICS | positive net EV after 0.05R costs | ❌ aggregate ≈ 0; only 2 family / 3 regime×vol cells positive |
| ROBUSTNESS | positive across walk-forward folds | ❌ not run |
| DIRECTION | LONG and SHORT independently validated | ✅ architecture; ❌ short economics absent |
| RISK | acceptable DD / tail | ❌ no position-level study in this stage |
| OVERFITTING | no material data-mining evidence | ⚠️ two small-sample survivors; PBO/DSR pending walk-forward |

## 16. Final verdict

> **NOT PRODUCTION READY.**

The architecture is two-sided and honest — that work stands. But Stage-3's
economic evidence is unambiguous: after strictly OOS calibration, first-touch
target probabilities, realistic costs, and fixed-horizon signal testing, the
system's net expectancy is ≈ 0. The old +1.1R ranking EV and the Stage-2
+0.85R signal expectancy were artifacts of geometry and selection, and the
report now says so with the receipts:

- Directional ranking at fixed horizons exists for **LONG_MEAN_REVERSION**
  (Range, t≈2.5) and short-horizon **LONG_TREND_CONTINUATION** (Bear) only.
- The core LONG_BUY_DIP family is negative at 10–20 bars in bull/range.
- No short family has fixed-horizon edge; no short exit policy is positive
  even at zero cost.
- The 1R/2R/3R ladder is structurally unreachable (P(TP2)≈2–3%, P(TP3)≈0.5%).
- The saved bundle calibrators are broken; refitting them OOS is a *fixable*
  defect (ECE 0.036–0.038 vs 0.222/0.102) — the single actionable engineering
  item from this stage.

**What would change the verdict** (in order): (1) refit calibrators and re-run
the §2 experiment; (2) build a per-target probability model; (3) run the locked
walk-forward + untouched test; (4) validate LONG_MEAN_REVERSION /
SHORT_BREAKDOWN / Bear-Trend-long hypotheses on the untouched fold. If the
walk-forward confirms positive net expectancy after 0.05R costs with ECE < 0.05,
the appropriate verdict is CONDITIONALLY READY for the surviving families only.
Nothing in this stage was tuned to improve any number; the next stage must hold
itself to the same standard.

---

## Appendix: how to reproduce

```bash
# Calibration experiment (strictly OOS, fit pre-split / eval post-split)
./venv/bin/python -m src.analysis.stage3 \
  --symbols EURUSD,GBPUSD,USDJPY,USDCHF,AUDUSD,AUDCHF,NZDUSD,USDCAD,\
EURJPY,GBPJPY,NZDJPY,AUDJPY,CADJPY,CHFJPY,GBPCAD,XAUUSD --calibrate

# Fixed-horizon signal edge (Phase 8) + TP tables (7/10) + attribution (11/12)
./venv/bin/python -m src.analysis.stage3 --symbols <list> --edge --tp --attrib

# Exit-path sweep + cost break-even (6/9)
./venv/bin/python -m src.analysis.stage3 --symbols <list> --exits

# Everything, written to data/validation/stage3_results.json
./venv/bin/python -m src.analysis.stage3 --symbols <list> --all
```

All analyses are causal (trailing windows, forward-only resolution), deterministic, and
run on the same pipeline the live system uses.
