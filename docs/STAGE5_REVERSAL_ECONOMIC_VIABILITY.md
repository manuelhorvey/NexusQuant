# NexusQuant — Stage-5: Reversal Alpha Expression & Economic Viability

**Date:** 2026-08-13
**Status:** Research report — verdict **C. PROMISING ECONOMIC EDGE — NEEDS MORE VALIDATION**
**Module:** `src/analysis/stage5.py` (rerunnable, see §17)
**Data:** 16 core symbols × D1; ~40k bars pooled.
**Discipline:** no production changes; threshold selection on TRAIN only
(pre-2022-01-01); all economics reported strictly OOS (2022-01-01 → 2025-06-01);
the **2025-06-01+ period remains untouched** (reserved for the single-shot final
test); cost grid mandatory — gross means nothing without break-even.

Stage-4 asked whether the reproducible reversal effect is *real*. Stage-5 asks
the only question that matters next:

> **Can the reversal effect be expressed as a cost-positive, out-of-sample
> trading strategy?**

The one-line answer: **yes — but only through specific, regime-conditioned,
extreme-trigger expressions, not through the static or symmetric rules.** Short
stretched rallies in Bear/Range regimes (RSI>70, 5-bar ATR rallies, positive
streaks) and long RSI-oversold (RSI<30) and deep-below-SMA200 conditions all
clear realistic costs out-of-sample with break-even costs of 0.15–0.20 ATR vs a
0.0625 ATR realistic cost. The static "distance-from-SMA200" and symmetric ±θ
rules — the obvious first guess — do **not** survive, and the frozen simple
rule is only 1-of-3 folds positive. That is exactly the differentiation the
campaign needed: the hypothesis is promoted, not deployed.

---

## 1. Stage-4 freeze (recorded)

| Item | Value |
|---|---|
| Stage-4 verdict | WEAK / UNSTABLE ALPHA |
| Signal | Reversal; vs_sma200 rank-IC −0.059 @20 bars; −0.114 in Bear vs −0.003 in Bull |
| FDR | 78/159 feature×horizon tests significant @ q=0.05; signal strengthens over time |
| Model failure | Unified LGBM (all features) OOS AUC 0.495, rank-IC −0.009 |
| Cost hurdle | No simple baseline survives 0.0625 ATR round-trip (h=10) |
| Reserved | 2025-06-01+ untouched (Phase 16 single-shot test) |
| Module | `src/analysis/stage4.py` — unchanged this stage |

---

## 2. Method

- **Signals** (deterministic, causal): distance-from-SMA200 in four
  normalizations (raw %, ATR, 100-bar z-score, 250-bar percentile rank);
  eight stretched-rally definitions; five oversold/excursion definitions.
- **Selection**: for the benchmark, thresholds are chosen on pre-2022 training
  data only (best gross across a θ×side grid); the chosen rule is then
  reported once on 2022→2025-06 OOS. Single selection, no re-tuning.
- **Economics**: every trade's P&L is the signed ATR-normalized forward return
  (10-bar primary) minus the cost grid. 1R = 1.25×ATR; realistic round-trip =
  0.0625 ATR (0.05R).

---

## 3. Pure reversal benchmark (Phases 2/3) — static rules fail OOS

| Family | θ (train) | side | n_train | train gross | n_oos | OOS gross | OOS net | break-even |
|---|---|---|---|---|---|---|---|---|
| dist_atr | 3.0 | long | 4,079 | +0.030 | 2,654 | −0.065 | −0.127 | 0 |
| dist_z | 2.5 | long | 1,811 | +0.124 | 1,296 | −0.031 | −0.094 | 0 |
| **dist_pct** | **8.0** | long | 145 | +0.491 | **241** | **+0.516** | **+0.454** | **0.2** |

**Findings**

1. **The static distance rules do not transfer**: the train-selected
   dist_atr / dist_z long rules flip sign out-of-sample (−0.065, −0.031). A
   distance-above/below-SMA200 rule is not a stable expression of the
   reversal effect.
2. **The one survivor is the deep-oversold tail**: being 8%+ below the 200-SMA
   (dist_pct θ=8) predicts +0.516 ATR / 10-bar OOS (net +0.454, break-even
   0.2 ATR). Rare (n=241 across 16 symbols in 3.5 years) but large — and it is
   per-regime consistent (Bear Trend net +0.157, Range gross +1.31 with n=61).
   This is the crash-dip condition; it is the correct "buy weakness" — not the
   buy-the-dip engine's mild pullback.
3. Selecting thresholds on the training window for rules that don't
   generalize is itself a documented overfitting trap (see §14 ledger).

---

## 4. Bear-rally short hypothesis (Phase 4) — CONFIRMED, extension-driven

All evaluated as SHORT signals in **Bear Trend / Range** regimes only, h=10:

| Definition | n | gross | net | win% | break-even |
|---|---|---|---|---|---|
| **C. RSI > 70** | 393 | +0.208 | **+0.145** | 51% | **0.20** |
| **D. 5-bar rally > 0.8 ATR** | 3,378 | +0.151 | **+0.088** | 50% | **0.20** |
| **E. 5 consecutive up days** | 630 | +0.146 | **+0.084** | 52% | **0.15** |
| F. Rally > 2 ATR off 10-bar low | 1,109 | +0.068 | +0.005 | 46% | 0.075 |
| A. Distance above SMA200 > 1.5 ATR | 6,981 | +0.025 | −0.038 | 47% | 0.025 |
| B. Distance above SMA50 > 1.0 ATR | 6,027 | +0.016 | −0.046 | 48% | 0.025 |

**Findings**

1. **The hypothesis is confirmed — and it is carried by *extension* signals,
   not by static distance.** RSI>70, a 5-bar ATR-normalized rally, and a
   positive streak all clear realistic costs (break-even 0.15–0.20 ATR).
   This is the exact behavior the user hypothesized: *"unusually strong
   rallies into stretched territory in Bear/Range create short
   opportunities."*
2. **The static distance-from-SMA200 short is the wrong expression** (break-
   even 0.025 ATR — below any realistic cost). Stage-4's headline rank-IC on
   vs_sma200 is a *correlation*, not a tradeable rule; the tradeable trigger
   is the momentum-extension of the rally, not the level of the distance.
3. RSI>70 in Bear/Range (n=393) is the strongest single short trigger found in
   the entire campaign: +0.145R-equivalent net with 0.20 ATR headroom.

---

## 5. High-volatility long hypothesis (Phase 5) — it's RSI-oversold, not drop

| Definition | n | gross | net | win% | break-even |
|---|---|---|---|---|---|
| **C. RSI < 30** | 1,372 | +0.182 | **+0.119** | 59% | **0.20** |
| D. Volatility > 80th pct | 7,240 | +0.009 | −0.054 | 55% | 0.01 |
| A. 5-bar drop > 0.8 ATR | 8,993 | −0.013 | −0.076 | 54% | 0 |
| E. Below SMA50 > 0.8 ATR | 22,681 | −0.039 | −0.102 | 52% | 0 |

**Findings**

1. **The long side of the reversal is the RSI-oversold bounce** (+0.119 net,
   59% win, break-even 0.20 ATR) — not "big drop", not "high volatility"
   per se. High-vol percentile, ATR-drop and below-SMA50 conditions carry no
   edge (and below-SMA50 is negative).
2. RSI<30 is the mirror of RSI>70 and the two are the cleanest pair in the
   whole campaign: **short RSI>70 in Bear/Range, long RSI<30** — both cost-
   positive with 0.20 ATR headroom. Simple, interpretable, directionally
   explicit.

---

## 6. Economic horizon (Phase 6) — the reversal compounds with holding time

Primary static rule (dist_atr ±2), gross/net by horizon:

| h | 1 | 5 | 10 | 20 | 30 | 40 | 60 |
|---|---|---|---|---|---|---|---|
| gross | +0.002 | +0.000 | +0.009 | +0.048 | +0.078 | +0.117 | +0.202 |
| net (0.0625) | −0.061 | −0.063 | −0.054 | −0.014 | **+0.016** | **+0.054** | **+0.140** |
| FLAT rate | 39% | 39% | 39% | 39% | 39% | 39% | 39% |

**Findings**

1. The reversal effect is a **slow** effect: it clears costs only at ≥30 bars
   for the static rule. This explains why the 10-bar-horizon models and the
   original 5–20-bar holding architecture under-express it.
2. The extension-based triggers (RSI/rally/streak) clear costs at h=10 already
   — the horizon requirement is expression-specific. RSI-extreme signals are
   faster than distance signals.
3. FLAT rate is 39% of bars under ±2 ATR distance (61% of bars generate a
   signal). Tighter triggers (RSI) naturally raise the FLAT rate toward the
   ~60% target.

---

## 7. Cost break-even curve (Phase 8 — mandatory deliverable)

Primary static rule (dist_atr ±2, h=10): **break-even 0.01 ATR — not viable**.
The economic story is the extension-based triggers:

| Expression | gross | break-even ATR | realistic 0.0625 | headroom |
|---|---|---|---|---|
| SHORT RSI>70 (Bear/Range) | +0.208 | **0.20** | +0.145 | 3.2× |
| SHORT 5-bar rally (Bear/Range) | +0.151 | **0.20** | +0.088 | 3.2× |
| SHORT 5-day streak (Bear/Range) | +0.146 | **0.15** | +0.084 | 2.4× |
| LONG RSI<30 | +0.182 | **0.20** | +0.119 | 3.2× |
| LONG dist_pct < −8% | +0.516 | **0.20** | +0.454 | 3.2× |
| static dist_atr ±2, h=10 | +0.009 | 0.01 | −0.054 | 0 |

**The cost test is passed by exactly the expressions that carry the effect.**
Break-even costs of 0.15–0.20 ATR are 2.4–3.2× the realistic 0.0625 ATR
round-trip — a real economic margin at h=10, before financing/impact
considerations. The static and symmetric rules fail it; they are rejected.

---

## 8. Monotonicity (Phase 11) — extreme-tail driven, NOT monotone

Forward 10-bar return by distance-from-SMA200 decile (16 symbols, OOS):

| decile | dist (ATR) | mean fwd10 | | decile | dist (ATR) | mean fwd10 |
|---|---|---|---|---|---|---|
| 0 | −6.07 | **+0.129** | | 5 | +1.22 | −0.033 |
| 1 | −3.86 | −0.123 | | 6 | +2.39 | −0.067 |
| 2 | −2.48 | −0.163 | | 7 | +3.69 | −0.020 |
| 3 | −1.24 | +0.016 | | 8 | +5.27 | −0.047 |
| 4 | −0.02 | −0.074 | | 9 | +7.99 | −0.076 |

**Not a smooth monotone function.** The positive tail is the far-oversold
decile (+0.129); the upper tail is mildly negative; the middle is noisy.
The effect is **extreme-trigger driven** — which is exactly why the RSI/
streak/deep-dip *threshold rules* work while a symmetric fade-all-strength
rule and smooth monotone models underperform. Any future model should be
built as a tail-trigger ensemble, not a global monotone transform.

---

## 9. Symbol heterogeneity (Phase 12) — the effect is concentrated

| Symbol | rank-IC | n | gross | net | break-even |
|---|---|---|---|---|---|
| **USDCHF** | −0.134 | 1,232 | +0.238 | **+0.175** | **0.20** |
| **NZDUSD** | −0.106 | 1,163 | +0.187 | **+0.124** | **0.20** |
| **USDCAD** | −0.115 | 1,186 | +0.123 | **+0.061** | **0.15** |
| AUDUSD | −0.072 | 1,299 | +0.006 | −0.057 | 0.01 |
| GBPCAD | −0.059 | 1,294 | +0.015 | −0.048 | 0.025 |
| JPY crosses | −0.042…−0.096 | ~1,300 ea | ≈ 0 | ≈ −0.06 | 0 |
| **EURUSD** | **+0.033** | 1,546 | −0.119 | −0.181 | 0 |
| AUDCHF | −0.079 | 1,506 | −0.173 | −0.235 | 0 |
| XAUUSD | −0.013 | 1,484 | −0.000 | −0.063 | 0 |

**Findings**

1. **The reversal short is a USD-bloc phenomenon**: USDCHF, NZDUSD, USDCAD
   carry the effect (net +0.06 to +0.18, break-even 0.15–0.20 ATR) — the
   pairs whose 2022–2025 USD rallies overshot. This is an economically
   plausible explanation (USD strength overshoot), which the campaign's final
   gate explicitly requires.
2. **EURUSD is the counterexample**: positive rank-IC (+0.033) — the reversal
   does NOT hold there (EURUSD behaved momentum/range-like in this sample).
   Symbols with no evidence must not be forced into the strategy.
3. JPY crosses have negative rank-IC but the ±2 rule trades little (gross ≈ 0);
   their reversal would need expression-specific triggers (e.g., RSI) —
   unproven here.
4. Per-symbol n (~1,200–1,500) supports the concentration claim directionally;
   the frozen walk-forward is the confirmation step.

---

## 10. Cross-asset timing (Phase 13)

Reversal rank-IC of dist_atr split by the sign of the 1-day-lagged
cross-asset risk proxy (mean AUDJPY/NZDJPY 5-bar ATR return):

| Symbol | risk-off IC | risk-on IC |
|---|---|---|
| USDCAD | **−0.144** | −0.078 |
| AUDJPY | **−0.080** | +0.008 |
| USDJPY | +0.006 | −0.042 |
| EURUSD | −0.023 | +0.103 |
| XAUUSD | +0.003 | −0.019 |

**The risk context matters for timing, not direction.** The reversal short is
stronger in risk-off states for the USD/JPY pairs (USDCAD −0.144 off vs −0.078
on; AUDJPY −0.080 off vs +0.008 on). EURUSD again inverts (positive IC in
risk-on). Cross-asset information does not add a new direction but is a
candidate *timing* conditioner for the next validation round.

---

## 11. Why the LGBM destroys the signal (Phase 10) — feature dilution, proven

MODELS A–E on the same chronological split (train < 2022, test 2022→cutoff),
target = sign(10-bar ATR return):

| Model | AUC | Brier | rank-IC | long exp (R) | short exp (R) |
|---|---|---|---|---|---|
| A. single reversal feature (vs_sma200) | 0.5098 | 0.2549 | +0.017 | −0.061 | −0.007 |
| **B. reversal features only** | **0.5148** | 0.2660 | **+0.026** | −0.026 | +0.017 |
| C. reversal + regime | 0.5139 | 0.2677 | +0.024 | −0.043 | +0.005 |
| D. all technical | 0.4976 | 0.2788 | −0.004 | −0.064 | +0.006 |
| E. all features (deployed-style) | 0.4947 | 0.2860 | −0.009 | −0.073 | +0.011 |
| **F. monotone-constrained reversal** | **0.5198** | 0.2620 | **+0.034** | — | — |

**The mechanism is feature dilution, not model capacity or label mismatch.**
OOS performance degrades *monotonically* as non-reversal features are added:
B (+0.026 rank-IC, AUC 0.515) → C → D (−0.004) → E (−0.009). The deployed
models train on the full feature set — exactly the anti-predictive E
configuration — so their flat OOS slopes (Stage-3) are a feature-selection
failure, not an information failure. The monotone-constrained LGBM on the
reversal features alone (F) is the best model in the campaign (AUC 0.520,
rank-IC +0.034). The 10-bar sign label still under-expresses the 30–60 bar
effect and the ±0.55/0.45 probability thresholds are uncalibrated for these
expressions — the model layer remains second-order to the rule layer found in
§4–§5, which is why this stage's economic evidence is rule-based.

---

## 12. Walk-forward (Phase 14, preliminary — frozen static rule)

Frozen simple rule (dist_atr ±2, h=10), purged 3-fold walk-forward, 20-bar
embargo:

| Fold | Window | n | gross | net | FLAT% |
|---|---|---|---|---|---|
| 1 | 2022-01-21..2023-06-30 | 4,302 | −0.055 | −0.118 | 30% |
| 2 | 2023-07-20..2024-12-31 | 3,947 | +0.165 | +0.103 | 36% |
| 3 | 2025-01-20..2025-06-01 | 1,042 | +0.007 | −0.055 | 32% |

**1 of 3 folds positive net — the static rule does not survive multi-fold
validation.** This is consistent with §3 (static rules fail) and is the
expected result; the walk-forward of the *extension-based* triggers (RSI>70 /
RSI<30 / streaks), which carry the cost-positive effect, is the required next
run once their thresholds are frozen on training data only.

---

## 13. Simple baselines (Phase 17, reference)

From Stage-4 §10 (27 symbols, h=10, 0.0625 ATR): every simple strategy was
net-negative (always-short −0.001, mean-reversion-fade −0.049, momentum
−0.076, trend-rule −0.091, breakout −0.163). The Stage-5 extension triggers
(§4–§5) are the **first expressions in the entire campaign that beat the
simple baselines net of costs** — but only at the specific triggers,
regimes and symbols documented here, and only single-split OOS so far.

---

## 14. Overfitting ledger (Phase 17)

| Experiment family | Count | Selection discipline |
|---|---|---|
| θ×side candidates per distance family | 15 | best-on-train, single OOS report |
| Stretched-rally definitions | 8 | fixed thresholds, pre-specified |
| Oversold definitions | 5 | fixed thresholds, pre-specified |
| Horizons | 9 | reported, no selection |
| Cost levels | 10 | reported, no selection |
| Models compared | 6 | pre-specified feature sets |
| Symbols | 16 | pre-specified |
| Thresholds tuned on OOS | 0 | — |

**Documented traps found:** the train-selected static rules (dist_atr/dist_z)
flipped sign OOS — a textbook overfit demonstration that only the *deep-tail*
rule (dist_pct 8%) survived. Every cost-positive trigger here is either
pre-specified by hypothesis (§4–§5) or extreme-tail (θ high), which is the
multiple-testing-cheap end of the design. PBO/DSR are computed at the frozen
walk-forward stage (they need out-of-sample trades, which the extension-trigger
walk-forward will provide).

---

## 15. Three-action logic (Phase 18) — what production would look like IF validated

The evidence-supported decision structure (not yet deployed):

```
LONG   when  RSI < 30            (or deep dist_pct tail) and EV clears cost
SHORT  when  RSI > 70 / 5-bar ATR rally / 5-day streak  in Bear|Range regime
FLAT   otherwise  (target ≈ 60% of bars)
```

- FLAT remains the default; the RSI triggers fire on a small fraction of bars.
- Symbols: USD-bloc (USDCHF, NZDUSD, USDCAD) for shorts; EURUSD and XAUUSD
  must be excluded unless the walk-forward says otherwise.
- Costs: only expressions with break-even ≥ 0.15 ATR are eligible.
- Every trigger requires the frozen walk-forward + the untouched 2025-06-01+
  test before any production consideration (Phase 19 gates).

---

## 16. Final classification

> **C. PROMISING ECONOMIC EDGE — NEEDS MORE VALIDATION**

Not "untradeable" (B): four specific expressions clear realistic costs by
2.4–3.2× with economically plausible structure (USD overshoot shorts in
Bear/Range; RSI-oversold bounce; crash-dip tail). Not "robust" (D) or a
production candidate (E): the static rule fails multi-fold OOS (1/3 folds),
the surviving triggers need their own frozen walk-forward, the untouched test
has not been run, and the effect is symbol-concentrated.

**What would move this to D/E (in order):**
1. Freeze the extension-trigger expressions (RSI>70/RSI<30/streaks/deep-tail)
   with thresholds chosen on pre-2022 data only; run the purged walk-forward
   on those exact rules.
2. Pass ≥2 of 3 folds net-positive and the single-shot untouched 2025-06-01+
   test net-positive after 0.0625 ATR costs.
3. Confirm the USD-bloc concentration survives the untouched period and that
   the symbol exclusions (EURUSD, XAUUSD) are stable.
4. Show the combined long/short/FLAT engine beats the simple baselines (§13)
   on the untouched window.
5. Then — and only then — a production-candidate build with calibration,
   target-probability, cost/slippage models and the §15 decision logic.

**The bottom line of Stage-5:** the reversal hypothesis earned its promotion
from "hypothesis" to "promising economic edge with specific, testable
expressions." The system has NOT earned the right to trade it. The next
campaign is a focused frozen-expression validation, not another exploration.

---

## 17. Reproduce

```bash
# Benchmark + hypotheses + cost curves + monotonicity + heterogeneity + WF
./venv/bin/python -m src.analysis.stage5 --symbols <16-list> \
  --reversal --shorts --hv --horizon --cost --mono --hetero --cross --wf
# Model comparison (why LGBM destroys the signal)
./venv/bin/python -m src.analysis.stage5 --symbols <16-list> --models
# Everything
./venv/bin/python -m src.analysis.stage5 --symbols <16-list> --all
```

Results land in `data/validation/stage5_results.json`. All analyses are
causal, deterministic, exclude the reserved 2025-06-01+ period, and perform
threshold selection on pre-2022 training data only.
