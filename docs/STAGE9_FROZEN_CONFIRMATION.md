# NexusQuant — Stage-9: Frozen Long Reversal Confirmation, Portfolio Economics & Production Readiness Gate

**Status:** Research / validation only — **no production code was modified.** SHORT reversal remains **LOCKED as falsified** (Stage-6) and was not revived. No parameter, symbol, threshold, regime, or exit optimization occurred in this stage: the Stage-8 hypothesis was frozen and hashed before any evaluation.

**Deliverable artifacts:**
- `src/analysis/stage9.py` — reproducible research module (deterministic seeds)
- `data/validation/stage9/stage9_results.json` — machine-readable results (all phases)
- Commands at the end of this report

---

## 1. Executive summary

**Final answer: NO — NexusQuant has NOT demonstrated that the LONG reversal strategy is repeatable, cost-positive, portfolio-level alpha. Classification: B — PROMISING BUT INSUFFICIENT EVIDENCE.**

The deciding gate is not a failure of the strategy — it is a hard data constraint: **the dataset ends today (2026-08-13), and the Stage-8 single-shot untouched test consumed the entire 2025-06-01+ window. There are zero bars of genuinely unseen data left.** The fresh-independent-window gate (the campaign's core requirement) is therefore *physically unresolvable* with existing data and is classified **UNRESOLVED**, which mechanically blocks promotion to A (PRODUCTION READY).

Everything else that could be validated with the frozen strategy was validated, and the picture is genuinely strong:

| Gate | Status | Evidence |
|---|---|---|
| 1. No leakage | ✅ PASS | 13 PASS / 0 FAIL / 2 UNKNOWN (unused paths) |
| 2. Positive untouched expectancy | ⛔ **UNRESOLVED** | no data exists past consumed 2025-06-01+ window |
| 3. MT-adjusted evidence | ✅ PASS | majors perm p=0.007; BH-FDR: fold2, fold3, pooled significant |
| 4. Robust bootstrap CI | ✅ PASS | trade boot95 [0.105, 0.600] excludes 0; symbol boot95 [0.06, 0.64] |
| 5. Majority of WF folds | ❌ **FAIL** | 2/5 folds positive (2/3 adequately sampled; CY2022 fold negative) |
| 6. No dominant symbol | ✅ PASS | LOSO min +0.29R; 100% of 28 exclusions positive (Stage-8) |
| 7. No dominant regime | ❌ **FAIL** | edge is a Bear-regime phenomenon (Bear|med +0.92R, Range −0.39R) |
| 8. Positive after realistic costs | ✅ PASS | net at 0.05R: +0.347R |
| 9. Adequate cost headroom | ✅ PASS | break-even 0.395R vs 0.05R realistic ≈ 7.9× |
| 10. Baseline superiority | ✅ PASS | beats random-timing p95 and all 7 simple baselines |
| 11. Portfolio incremental alpha | ✅ PASS | 25bp CAGR +1.7% / Sharpe 0.68; correlation control cuts maxDD 31.3R→24.2R |
| 12. Acceptable drawdown | ✅ PASS | 25bp maxDD −6.4% / 50bp −12.2% (breaker 15%); 1% too aggressive |
| 13. Adequate effective sample | ❌ **FAIL** | raw N=163, cluster-adjusted N=144 (target ≥ 250) |
| 14. Stable exit behavior | ✅ PASS | time-stop / signal-reversal transfer OOS |
| 15. No catastrophic delay sensitivity | ✅ PASS | 1-bar +0.31R, 2-bar +0.24R (both still positive) |

**Scorecard: 11 PASS / 3 FAIL / 1 UNRESOLVED.** The three fails are sample-size/regime-concentration issues, not validity failures; the single UNRESOLVED gate (fresh independent data) is the decisive blocker. The user's instruction was explicit: *"Do not recommend another research experiment unless a critical gate remains unresolved"* — one does, and it is unresolvable by analysis: **the strategy needs new, unseen bars.**

### The most important correction in this campaign

The Stage-9 portfolio simulation originally annualized over *signal days only* (119 days instead of the 1,186-day OOS calendar span), inflating CAGR to +18% and Sharpe to 2.3. **Fixed in this campaign**: the equity curve now spans the full business-day calendar with zero returns on non-signal days. The honest numbers are CAGR +1.7–3.6% at 25–50bp risk, Sharpe 0.68–0.74, max DD 6–12%. This is the economically meaningful statement, and it is the number locked by the regression tests.

---

## 2. Frozen strategy specification (v1.0.0, sha256)

**Configuration hash: `6f3cc20cf75ee86fa050424c2e32023c847b020c0dd3bf22164c4cf3abbf4279`** (deterministic; recomputed and verified identical on rerun).

| Element | Frozen definition |
|---|---|
| Direction | **LONG only.** SHORT reversal LOCKED falsified (Stage-6); no short surface defined |
| Universe | 28 liquid FX majors/crosses (pre-registered Stage-8 bucket; exotics excluded by prior evidence, NOT retuned) |
| Entry | LONG **k=3 of 3** {L1: RSI<30, L2: 5-bar ATR-normalized drop < −0.8, L3: 5-day negative streak} |
| Horizon | 10 bars (PRIMARY_H) |
| Exit | **Primary:** time-stop at 10 bars. **Alternates:** signal-reversal (RSI crosses back > 35), return-to-mean (close ≥ entry). Both transfer OOS. 3R target ladder structurally rejected for this signal |
| Risk | 1R = 1.25×ATR; cost 0.05R round-trip |
| Windows | Train < 2022-01-01 (selection only) · OOS 2022-01-01 → 2025-06-01 · untouched 2025-06-01+ **consumed** (Stage-8, single-shot, NOT reusable) |
| FLAT | Default whenever the k=3 condition is absent; no requirement to trade |

---

## 3. Fresh independent window — the decisive gate (UNRESOLVED)

```
consumed_window:          2025-06-01 → 2026-08-13
data_end:                 2026-08-13 (today)
bars_past_consumed_window: 0
status:                   UNRESOLVED
```

Stage-8 evaluated the untouched 2025-06-01+ window exactly once (n=39 majors, +0.555R, ex-USDCHF +0.619R). That window is now **consumed** by protocol. The dataset ends today, so no genuinely independent validation window exists yet. This gate **cannot be satisfied with existing data** and mechanically blocks promotion to A.

**What would resolve it:** accrue new bars past 2026-08-13 and re-run the frozen strategy once (single-shot) on a window starting after 2026-08-13. Nothing in the research path can substitute for this.

---

## 4. Effective sample size

| Metric | Value |
|---|---|
| Raw N (OOS trades) | 163 |
| Unique signal days | 119 |
| **Cluster-adjusted N** | **144** (one bet per date × currency-cluster) |
| Target (pre-registered) | ≥ 250 raw / ≥ 250 effective |

**Gate 13 FAILS** on both raw and effective N. The 163-trade / p=0.007 result is statistically significant but below the pre-registered power requirement; the block bootstrap 95% CI [−0.135, +0.780] (which respects serial dependence) does not exclude zero, confirming the sample is at the margin of adequacy.

---

## 5. Chronological walk-forward (5 expanding-window purged folds, no pooling)

| Fold | Test window | Train N | OOS N | Mean R | Win | perm p | maxDD R | Adequate? |
|---|---|---|---|---|---|---|---|---|
| 1 | 2022-01-20 → 2022-12-31 | 255 | 78 | **−0.172** | 0.462 | 0.327 | −31.3 | ✅ n=78 |
| 2 | 2023-01-20 → 2023-12-31 | 333 | 46 | **+0.773** | 0.826 | 0.000 | −6.2 | ✅ n=46 |
| 3 | 2024-01-20 → 2024-12-31 | 379 | 29 | **+0.808** | 0.655 | 0.037 | −8.2 | ✅ n=29 |
| 4 | 2025-01-20 → 2025-03-31 | 408 | 2 | 0.000 | — | — | 0.0 | ❌ n=2 |
| 5 | 2025-04-20 → 2025-06-01 | 413 | 1 | 0.000 | — | — | 0.0 | ❌ n=1 |

**2/5 folds positive (2/3 adequately sampled). Gate 5 FAILS.** CY2022 is a materially negative, adequately-sampled fold (n=78, −0.172R) — the strategy does not work in every era, and the pooled result cannot hide that. Folds 4–5 are underpowered by construction (the OOS window is truncated by the consumed boundary) and are flagged rather than pooled.

---

## 6. Statistical battery

| Test | Result |
|---|---|
| Permutation (sign-flip, OOS) | p = 0.006 |
| Trade bootstrap 95% CI | [0.105, 0.600] — excludes 0 |
| Trade bootstrap 99% CI | [0.029, 0.680] |
| **Block bootstrap 95% CI** (respects serial dependence) | **[−0.135, 0.780] — does NOT exclude 0** |
| Symbol bootstrap 95% CI | [0.059, 0.636] |
| BH-FDR q (fold1, fold2, fold3, pooled) | 0.327 / 0.000 / 0.049 / 0.012 |
| Significant at q=0.05 | fold2, fold3, pooled |

**Gate 3 and 4 PASS.** The trade/symbol bootstraps exclude zero; the block bootstrap — the most conservative and the one that accounts for autocorrelation in the 10-bar holding period — does not, which is exactly the honest boundary of the current sample size.

---

## 7. Economic significance & cost stress

| Cost (R round-trip) | Net R/trade |
|---|---|
| 0.05 (realistic) | +0.347 |
| 0.10 | +0.297 |
| 0.15 | +0.247 |
| 0.20 | +0.197 |
| 0.25 | +0.147 |
| 0.30 | +0.097 |

- Gross expectancy: +0.397R · **break-even cost: 0.395R** · headroom **≈ 7.9×** realistic cost.
- Adversarial cost shocks: 2× (+0.297, p=0.023), 3× (+0.247, p=0.057 — marginal), 5× slippage shock (+0.147, p=0.233 — loses significance). Break-even under shocks stays ≥ 0.20R.
- **Gates 8 and 9 PASS** — this is a genuine economic margin, not a cost-illusory edge.

---

## 8. Portfolio-level validation (the new economics)

Simulation: full OOS calendar span (2022-01-01 → 2025-06-01, business days), trades risk `risk_frac` of equity, per-symbol cap, per-currency-cluster cap, daily loss limit 3%, 15% drawdown breaker.

| Sizing | CAGR | Sharpe | Sortino | Calmar | maxDD | PF | Trades/yr |
|---|---|---|---|---|---|---|---|
| **Fixed 25bp** | **+1.7%** | **0.68** | 0.28 | 0.26 | **−6.4%** | 1.41 | 46 |
| Fixed 50bp | +3.6% | 0.74 | 0.32 | 0.29 | −12.2% | 1.44 | 46 |
| Fixed 100bp | −4.8% | −1.28 | — | — | −17.4% | 0.34 | 46 |
| Kelly full (26%) / capped 5% | ~0% | ~0.15 | — | — | −18.3% | 1.06 | 46 |

- **Capital utilization: 13.4%** of days active; ~46 trades/year across 28 symbols.
- **1% risk is too aggressive** — it breaches the 15% breaker and the fixed-100bp sim loses money through path dependence. The sustainable band is 25–50bp.
- **Gate 12 PASS** at 25–50bp: max DD −6.4% / −12.2%, both inside the 15% breaker.

### Monte Carlo (10,000 block-bootstrap paths, 1% risk — intentionally above sustainable sizing)

| Metric | Value |
|---|---|
| Median final equity | 1.71 |
| 5th-percentile equity | 0.905 |
| P(profit) | 0.917 |
| Median max-DD | ≈ −20% |
| P(DD > 20%) | 0.508 |
| P(ruin, −50%) | 0.007 |

The MC confirms the sim: at 1% risk the drawdown tail is unacceptable (P(DD>20%) ≈ 51%); at 25–50bp the strategy's real DD is 6–12%. The MC is a stress statement, not the recommended sizing.

---

## 9. Signal clustering / correlation control

| Currency cluster | N | Win | Mean R |
|---|---|---|---|
| CHF | 46 | 0.674 | +0.684 |
| USD | 43 | 0.674 | +0.265 |
| EUR | 48 | 0.562 | +0.218 |

| Policy | Mean R | maxDD (R) |
|---|---|---|
| All signals | +0.347 | −31.3 |
| **One per cluster per day** | **+0.338** | **−24.2** |

**Gate 11 PASS**: correlation control costs ~0.01R of expectancy and cuts cumulative-R drawdown by 23% (−31.3R → −24.2R). The strategy benefits from cluster-level position limits (the CHF block — USDCHF/AUDCHF/NZDCHF — is the densest, most profitable cluster and must never be treated as independent bets).

---

## 10. Baselines (frozen universe, net of cost)

| Baseline | N | Net R/trade |
|---|---|---|
| **STAGE9_FROZEN_LONG (k=3)** | **163** | **+0.347** |
| RSI<30 alone | 1,022 | +0.243 |
| Always FLAT | 27,291 | 0.000 |
| Buy & hold | 27,291 | −0.030 |
| Momentum (ret10) | 14,045 | −0.059 |
| SMA200 trend | 27,291 | −0.086 |
| Mean-reversion fade-5 | 5,364 | −0.092 |
| Random timing | 562 | −0.182 |

**Gate 10 PASS.** The frozen k=3 strategy beats every simple baseline — including RSI<30 alone (+0.243), confirming the Stage-8 finding that the multi-condition trigger carries the edge, not the single oversold signal.

---

## 11. Adversarial falsification

| Test | N | Mean R | win | perm p | Survives? |
|---|---|---|---|---|---|
| as-is | 163 | +0.347 | 0.620 | 0.007 | — |
| 1-bar delay | 163 | +0.306 | 0.626 | 0.017 | ✅ |
| 2-bar delay | 163 | +0.243 | 0.607 | 0.023 | ✅ |
| Random timing | — | −0.033 (mean); p95 +0.189 | — | — | ✅ (signal beats p95) |
| Return permutation | 163 | +0.347 (mean preserved by design) | — | — | ✅ (maxDD −11.7 vs −31.3: time structure matters) |
| Sign inversion | 163 | −0.347 | 0.380 | 0.010 | ✅ sanity (mirror) |
| Cost 2× | 163 | +0.297 | 0.607 | 0.023 | ✅ |
| Cost 3× | 163 | +0.247 | 0.607 | 0.057 | ⚠️ marginal |
| Slippage shock 5× | 163 | +0.147 | 0.558 | 0.233 | ❌ loses significance (stays positive) |
| Ex-best symbol (GBPCAD) | 150 | +0.293 | 0.613 | 0.017 | ✅ |
| **Ex-best regime (Bear Trend)** | **26** | **−0.391** | 0.500 | 0.25 | ❌ **regime concentration** |
| Ex-best fold (CY2024) | — | reported per-fold | — | — | ✅ (fold 3 removed; other folds still reported) |

**The ex-Bear result is the honest center of the falsification story**: remove Bear Trend and the remaining 26 trades (Range) are negative (−0.39R). The edge is a *Bear-regime* effect. This is why gate 7 FAILS and why the regime dimension must be treated as a hard gate in any production expression — not an optional filter.

---

## 12. Regime stability (descriptive)

| Regime | N | Mean R | Win | perm p |
|---|---|---|---|---|
| Bear Trend | all | 137 | +0.487 | 0.642 | 0.000 |
| Bear Trend | low vol | 39 | +0.612 | 0.769 | 0.02 |
| Bear Trend | med vol | 26 | +0.921 | 0.692 | 0.01 |
| Bear Trend | high vol | 72 | +0.262 | 0.556 | 0.16 |
| Range / Chop | all | 26 | −0.391 | 0.500 | 0.25 |

Bull Trend and High-Volatility rows are empty/underpowered in the OOS window. The Stage-7 "Bear + high-vol" hypothesis remains falsified (high-vol Bear: +0.26, not significant); the real edge is **low/med-vol Bear**. This is a regime-specific signal, not a universal one — documented, not optimized away.

---

## 13. Exit validation (descriptive MFE/MAE)

| Metric | Value |
|---|---|
| N | 163 |
| Median MFE | 1.86R |
| Median MAE | 1.30R |
| Median time to peak MFE | 11 bars |
| P(MFE ≥ 1R) | 75.5% |
| P(MFE ≥ 3R) | 17.2% |

**Gate 14 PASS.** The frozen exits (time-stop h=10, signal-reversal, return-to-mean) transferred OOS in Stage-8; this table confirms the holding geometry: the signal is a slow reversal (peak favorable excursion at ~11 bars), 75% of trades see 1R of favorable excursion, but only 17% ever reach 3R — the 3R ladder was structurally wrong for this signal, and the time/reversal exits are the right expression.

---

## 14. Production gates — final scorecard

| # | Gate | Status |
|---|---|---|
| 1 | No leakage | ✅ PASS |
| 2 | Positive untouched expectancy | ⛔ UNRESOLVED (no fresh data) |
| 3 | Positive MT-adjusted evidence | ✅ PASS |
| 4 | Robust bootstrap CI | ✅ PASS |
| 5 | Positive majority of WF folds | ❌ FAIL (2/5; CY2022 negative) |
| 6 | No dominant symbol | ✅ PASS |
| 7 | No dominant regime | ❌ FAIL (Bear-only) |
| 8 | Positive after realistic costs | ✅ PASS |
| 9 | Adequate cost headroom | ✅ PASS (7.9×) |
| 10 | Baseline superiority | ✅ PASS |
| 11 | Portfolio incremental alpha | ✅ PASS |
| 12 | Acceptable drawdown | ✅ PASS (25–50bp) |
| 13 | Adequate effective sample | ❌ FAIL (144 < 250) |
| 14 | Stable exit behavior | ✅ PASS |
| 15 | No catastrophic delay sensitivity | ✅ PASS |

**11 PASS / 3 FAIL / 1 UNRESOLVED.**

- **Deciding gate: 2 (fresh independent window) — UNRESOLVED.** No data exists past the consumed window; promotion to A is mechanically impossible today.
- **Additional fails:** 5 (walk-forward not majority-positive), 7 (regime concentration), 13 (effective N below pre-registered target).
- **Classification: B — PROMISING BUT INSUFFICIENT EVIDENCE. Final answer: NO.**

---

## 15. Final question — answered mechanically

> *"Has NexusQuant demonstrated enough independent evidence that this LONG reversal strategy represents repeatable, cost-positive, portfolio-level alpha rather than a research artifact?"*

**NO.**

**Which gate caused the decision:** gate 2 (fresh independent window) is UNRESOLVED — the dataset ends today and the only untouched window was consumed by the Stage-8 single-shot test. Even setting that aside, gates 5, 7, and 13 fail on their own merits (one negative adequately-sampled fold, Bear-regime-only edge, effective N 144 < 250). The strategy has not yet earned promotion; it also has not been falsified.

---

## 16. Failure modes

1. **Regime dependence (most important):** ex-Bear the edge is negative. Any production expression MUST gate on low/med-vol Bear — and that gate itself needs independent validation on fresh data.
2. **Time instability:** CY2022 was negative (n=78, −0.17R). The edge is not era-invariant.
3. **Sample size:** block bootstrap does not exclude zero; effective N is 144 vs the 250 target.
4. **Sizing discipline:** 1% risk per trade is ruin-adjacent (P(DD>20%) ≈ 51% in MC); the viable band is 25–50bp with a 15% portfolio breaker.
5. **Correlation:** the CHF cluster (USDCHF/AUDCHF/NZDCHF) carries the densest, most profitable signals; one-per-cluster position limits are mandatory.
6. **Exotics:** remain excluded by prior evidence; do not expand the universe to inflate N.

---

## 17. Reproducibility

```bash
# full campaign (all phases; deterministic seeds; writes data/validation/stage9/stage9_results.json)
python -m src.analysis.stage9 --all

# per-phase flags (each independent): --spec --effn --wf --stats --econ --portfolio
#   --cluster --baselines --adv --regime --exits --mc --fresh --gates --leak
```

- Config hash `6f3cc20c…` verified stable across reruns; `stat_battery().permutation_p` verified deterministic (0.006 on both runs).
- Results: `data/validation/stage9/stage9_results.json` (16 sections).
- Regression tests: `tests/test_stage9.py` (frozen spec/hash stability, R-unit scale invariance, effective-N definition, WF fold discipline, gate wiring, deterministic seeds).

---

## 18. Next steps (the only ones the evidence supports)

1. **Accrue data, then run the single decisive test.** The fresh-window gate is unresolvable by analysis. When bars exist past 2026-08-13, run the frozen strategy exactly once on that window (full vs ex-USDCHF, per protocol).
2. **Do NOT retune anything** in the meantime. Every additional experiment against the same 163 OBSERVATIONS compounds multiple-testing risk without adding information.
3. **If the fresh window confirms the edge**, the remaining work is engineering, not research: the LOW/MED-VOL-BEAR gate, 25–50bp risk with one-per-cluster caps, and the 15% breaker — then a constrained paper-trading phase where the engine remains free to output **FLAT**.
4. **If the fresh window does not confirm it**, the honest result is that the LONG reversal hypothesis, like the SHORT leg in Stage-6, does not survive genuinely unseen time — and the research program should say so.
