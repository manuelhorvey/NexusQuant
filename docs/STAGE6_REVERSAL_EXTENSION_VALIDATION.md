# NexusQuant — Stage-6: Reversal Extension-Trigger Validation, Walk-Forward & Untouched-Test Campaign

**Date:** 2026-08-13
**Status:** Research report — verdict **B. PROMISING BUT INSUFFICIENT EVIDENCE (LONG leg only); the SHORT leg is FALSIFIED**
**Module:** `src/analysis/stage6.py` (rerunnable, see §22)
**Data:** 16 core symbols × D1; ~40k bars pooled.
**Discipline:** research only — no production changes; threshold/trigger-combo
selection on TRAIN only (pre-2022-01-01); all economics strictly OOS
(2022-01-01 → 2025-06-01); the **2025-06-01+ period was evaluated exactly once**
with frozen rules (single-shot, see §19); cost grid mandatory; FLAT is a
first-class outcome.

Stage-5 promoted five extension hypotheses to test. Stage-6 froze them and ran
the prescribed falsification campaign: trigger combos, confirmation, purged
walk-forward, symbol/regime/cost robustness, exit transfer, target
probabilities, baselines, multiple-testing controls, adversarial tests and the
single-shot untouched test.

> **The one-line answer: the SHORT leg of the hypothesis is falsified — it is
> statistically indistinguishable from random timing, carried by low-sample
> symbols, and deeply negative in the untouched period. The LONG leg survives
> the robustness battery — statistically significant, timing-dependent,
> positive at target-level EV and in the untouched test — but only on 11
> untouched trades concentrated in one symbol (USDCHF) and one regime (Bear).
> NexusQuant has not earned the right to trade either side yet.**

---

## 1. Research-integrity audit (Phase 1)

All 10 pre-registered checks pass — **CLEAN, no leakage found**:

| Check | Result |
|---|---|
| Features causal (trailing windows only) | ✅ |
| Regime causal (trailing slope/ADX/SMA/ATR, no centered windows) | ✅ |
| Labels: forward returns used only as evaluation outcome | ✅ |
| No future info in signal construction (trailing percentile/z-score) | ✅ |
| Threshold selection pre-2022-01-01 only | ✅ |
| Walk-forward embargo (20 bars) implemented | ✅ |
| Untouched period 2025-06-01+ excluded from every selection metric | ✅ |
| Symbol selection pre-specified (watchlist), not performance-driven | ✅ |
| No calibration fit on test data | ✅ |
| No threshold tuning on the untouched period | ✅ |

Notes:
- `detect_regime` (rule-based) verified causal: `linear_regression_slope(20)`,
  `adx`, `sma_200`, `atr_14.rolling(100).median()`; no `shift(-)` or centered
  windows.
- Indicators/setups `.iloc[-1]` uses are latest-bar reporting helpers.
- `detect_regime_cluster` standardizes on full-sample mean/std — a mild
  look-ahead, but the cluster path is unused by stages 2–6 (documented caveat).
- Stage-5 θ selection and Stage-6 combo selection run on pre-2022 data only.

## 2. Frozen hypotheses (Phase 2) — unchanged from Stage-5

| ID | Trigger | Regime gate | Direction |
|---|---|---|---|
| S1 | RSI > 70 | Bear Trend / Range | SHORT |
| S2 | 5-bar ATR-normalized rally > 0.8 | Bear Trend / Range | SHORT |
| S3 | ≥ 5 consecutive up closes | Bear Trend / Range | SHORT |
| L1 | RSI < 30 | all | LONG |
| L2 | 5-bar ATR-normalized drop < −0.8 | all | LONG |
| L3 | ≥ 5 consecutive down closes | all | LONG |
| L4 | price ≥ 8% below 200-SMA (crash tail) | all | LONG |

No trigger was modified to improve results; only the k-of-3 combination level
was selected, on training data only.

## 3. Data / time splits

- 16 core symbols, D1 bars.
- **Train:** up to 2022-01-01 (threshold and combo selection only).
- **OOS validation:** 2022-01-01 → 2025-06-01 (all tables below, unless noted).
- **Untouched:** 2025-06-01+ — evaluated once (§19).
- Walk-forward folds (pre-registered): 2022-01-21..2023-06-30,
  2023-07-20..2024-12-31, 2025-01-20..2025-06-01, each with a 20-bar embargo
  and per-fold train selection.

## 4. Trigger combinations — k-of-3 (Phase 5; selected on TRAIN, reported OOS)

| Dir | k | n_train | train net | n_OOS | OOS gross | OOS net | win | break-even | flat |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| SHORT | 1 | 2,390 | +0.151 | 1,570 | −0.011 | **−0.074** | 0.452 | 0.0 | 0.892 |
| SHORT | **2** *(selected)* | 250 | +0.515 | 165 | +0.165 | **+0.103** | 0.588 | 0.2 | 0.989 |
| SHORT | 3 | 18 | +1.281 | 8 | — | — | — | — | — |
| LONG | 1 | 3,375 | −0.081 | 2,284 | −0.060 | **−0.123** | 0.501 | 0.0 | 0.842 |
| LONG | 2 | 662 | +0.091 | 423 | +0.336 | **+0.274** | 0.589 | 0.2 | 0.971 |
| LONG | **3** *(selected)* | 104 | +0.117 | 64 | +0.864 | **+0.801** | 0.734 | 0.2 | 0.996 |

Key observations:
- **Single triggers (k=1) are negative OOS on both sides** — the raw extension
  alone does not predict. Only *co-occurrence* (2-of-3 / 3-of-3) carries the
  effect.
- LONG k=3's OOS +0.801R is the headline number — but on only 64 trades
  (§8 shows how fragile that is per-symbol).
- Flat rate 97–99.6%: the frozen rules trade rarely. This is correct behavior.

## 5. Confirmation test (Phase 6) — extension-only wins

Same-bar confirmation (rejection candle / momentum flip / close back inside
band) applied to k≥1 triggers, OOS:

| Dir | confirmation | n | net |
|---|---|---|---|
| SHORT | rejection_candle | 221 | −0.074 |
| SHORT | momentum_flip | 367 | −0.138 |
| SHORT | inside_band | 1,332 | −0.118 |
| LONG | rejection_candle | 398 | **−0.012** |
| LONG | momentum_flip | 624 | −0.160 |
| LONG | inside_band | 1,826 | −0.117 |

**Confirmation does not add value — it removes trades without improving
expectancy.** The extension trigger itself is the information; waiting for
same-bar confirmation destroys the edge (worse than k=1 extension-only, which
is already negative). Tested as pre-defined candidates only, no search.

## 6. Purged walk-forward (Phase 7; per-fold train selection, embargo 20 bars)

| Dir | fold | window | k | train net | n | OOS net | win |
|---|---|---|---:|---:|---:|---:|---:|
| SHORT | 1 | 2022-01-21..2023-06-30 | 2 | +0.515 | 45 | **+0.333** | 0.622 |
| SHORT | 2 | 2023-07-20..2024-12-31 | 2 | +0.519 | 92 | **−0.310** | 0.500 |
| SHORT | 3 | 2025-01-20..2025-06-01 | 2 | +0.348 | 11 | **+0.490** | 0.727 |
| LONG | 1 | 2022-01-21..2023-06-30 | 3 | +0.117 | 37 | **+0.504** | 0.676 |
| LONG | 2 | 2023-07-20..2024-12-31 | 3 | +0.218 | 20 | **+1.185** | 0.750 |
| LONG | 3 | 2025-01-20..2025-06-01 | 3 | +0.353 | **1** | — | — |

- SHORT: 2/3 folds positive, avg +0.171 — but fold 2 is materially negative
  (−0.31) and fold 3 has only 11 trades.
- LONG: 2/3 folds positive with the strongest fold-2 result (+1.19) — but fold
  3 degenerates to **1 trade**. Sample starvation, not validation.

## 7. Regime robustness (Phase 9; OOS, frozen k)

| Dir | regime | n | gross | net | win | break-even |
|---|---|---|---:|---:|---:|---:|---:|
| SHORT | Range / Chop | 147 | +0.149 | **+0.086** | 0.578 | 0.15 |
| LONG | Bear Trend | 54 | +0.739 | **+0.676** | 0.704 | 0.20 |

- The SHORT edge exists *only* in Range (Bear Trend < 30 OOS trades — the
  regime gate makes short trades rare in a genuine downtrend).
- The LONG edge is *only* in Bear Trend.
- No regime shows both sides positive — consistent with the Stage-4 finding
  that the signal is regime-conditional, but also means each side stands on a
  single regime pillar.

## 8. Symbol robustness (Phase 8; OOS, frozen k)

**SHORT k=2** — only 2 symbols reach ≥ 20 OOS trades:

| symbol | n | net | % PnL |
|---|---:|---:|---:|
| USDCHF | 34 | **−0.595** | 117% |
| AUDCHF | 35 | +0.088 | −17% |

- **Pooled over the adequately-sampled symbols the SHORT net is −0.507.** The
  headline +0.103 pooled net is entirely carried by *low-sample* symbols
  (NZDUSD n=13 +1.91, GBPUSD n=19 +0.53, AUDUSD n=9 +0.67) — noise.
- Leave-one-symbol-out: removing USDCHF flips the adequately-sampled pool to
  +0.088; removing AUDCHF leaves −0.595.

**LONG k=3** — **no symbol reaches 20 OOS trades** (max: EURUSD n=13, GBPCAD
n=13, NZDUSD n=11). The +0.801 pooled result cannot be attributed per symbol;
it is spread across seven small counts and could be a few lucky trades.

> **Both sides fail symbol robustness, in opposite ways: SHORT is *negatively*
> concentrated (the only well-sampled pair is net-negative), LONG is
> *unattributable* (nothing is well-sampled).**

## 9. Cost sensitivity (Phase 10; frozen k, OOS)

| Dir | n | gross | break-even (ATR) | net @0.0625 | net @0.15 | net @0.20 |
|---|---:|---:|---:|---:|---:|---:|
| SHORT | 165 | +0.165 | 0.20 | +0.103 | +0.015 | **−0.035** |
| LONG | 64 | +0.864 | 0.20 | +0.801 | +0.714 | **+0.664** |

- Break-even 0.20 ATR vs realistic 0.0625 ATR = 3.2× headroom — cost is *not*
  the binding constraint for either side (unlike Stage-4's simple baselines).
- LONG stays positive even at 3× the realistic cost.

## 10. Exit transfer (Phase 11) — train-selected exits FAIL OOS

SL/TP first-touch grid (0.6–3.0R × 10/20 bars), exit selected on TRAIN, then
evaluated OOS on identical entries:

| Dir | selected exit | train net (R) | n_OOS | OOS gross (R) | OOS net (R) |
|---|---|---|---:|---:|---:|
| SHORT | TP=3.0R, h=20 | +0.119 | 1,570 | −0.112 | **−0.162** |
| LONG | TP=2.0R, h=10 | −0.007 | 2,284 | +0.032 | **−0.018** |

- **The exit selected on training data does not transfer.** Both sides are
  negative OOS under *any* SL/TP configuration tested.
- This replicates the Stage-3 finding: the fixed-horizon signal edge is
  destroyed by the exit architecture. The problem is not entry discovery —
  it is that SL/TP exits monetize the wrong part of the forward return
  distribution.

## 11. Target probabilities + economic EV (Phase 12/13; TP=1R, h=20, OOS)

| Dir | family | n | P(TP) | P(SL) | P(time) | mean R | EV net (R) |
|---|---|---|---:|---:|---:|---:|---:|---:|
| SHORT | S1_rsi70 | 86 | 0.477 | 0.523 | 0.000 | −0.047 | **−0.097** |
| SHORT | S2_rally5 | 1,372 | 0.442 | 0.541 | 0.017 | −0.099 | **−0.149** |
| SHORT | S3_streak5p | 285 | 0.365 | 0.600 | 0.035 | −0.239 | **−0.289** |
| LONG | L1_rsi30 | 484 | 0.579 | 0.384 | 0.037 | +0.192 | **+0.142** |
| LONG | L2_drop5 | 1,983 | 0.487 | 0.501 | 0.012 | −0.014 | **−0.064** |
| LONG | L3_streak5n | 304 | 0.595 | 0.382 | 0.023 | +0.219 | **+0.169** |
| LONG | L4_crash | 241 | 0.647 | 0.349 | 0.004 | +0.300 | **+0.250** |

- **All three SHORT families are negative EV at the 1R/20 target.**
- **Three of four LONG families are positive** — L1 (RSI<30), L3 (streaks),
  L4 (crash tail); L2 (5-bar drop) is not.
- P(TP1) ≈ 44–65%, P(SL) ≈ 35–60% at 1R — the probability mass is real on the
  LONG side, inverted on the SHORT side.
- No calibrator was fitted in this stage; these are empirical outcome
  frequencies under the frozen exit, so they are honest by construction but
  not yet smoothed/calibrated estimates.

## 12. Economic EV decision logic (Phase 13)

With FLAT = 0 as the baseline:

- **LONG EV = +0.14 to +0.25R** for L1/L3/L4 at 1R/20 — positive but small.
- **SHORT EV = −0.10 to −0.29R** — negative across every trigger.
- The correct decision under these numbers is **FLAT for the short side in
  every family**; LONG only for L1/L3/L4 triggers, and only after the
  sample-size problem (§8) is resolved.

## 13. Baselines (Phase 15; net of 0.0625 ATR, h=10, OOS)

| baseline | n | gross | net |
|---|---:|---:|---:|
| buy_hold | 14,493 | −0.059 | **−0.121** |
| always_short | 14,493 | +0.059 | **−0.004** |
| random_sign | 14,493 | −0.016 | **−0.079** |
| **rsi_reversal** (RSI<30 long / RSI>70 short) | 1,362 | +0.205 | **+0.142** |
| ma_reversal (±4% from SMA200) | 5,075 | +0.108 | **+0.045** |
| random_timing_short | 1,264 | +0.030 | **−0.033** |
| random_timing_long | 1,264 | −0.030 | **−0.092** |

- **The plain RSI-reversal baseline nets +0.142 — it beats the SHORT combo
  (+0.103) and matches the LONG L1 family alone (+0.142).** The k-of-3
  machinery adds nothing over the simplest reversal rule on the short side.
- always_short ≈ 0 net over OOS — there is no naked short drift to harvest.

## 14. Multiple-testing controls (Phase 14; frozen k, OOS)

| Dir | n | gross | boot 95% CI (gross) | perm p | net | net CI |
|---|---:|---:|---:|---:|---:|---:|
| SHORT | 165 | +0.165 | [−0.19, +0.54] | **0.38** | +0.103 | [−0.25, +0.48] |
| LONG | 64 | +0.864 | [+0.39, +1.39] | **0.005** | +0.801 | [+0.33, +1.32] |

- **SHORT: the 95% bootstrap CI includes zero and the permutation test is
  nowhere near significant (p=0.38). The OOS short edge is not
  distinguishable from noise.**
- **LONG: the CI excludes zero and the permutation test is significant
  (p=0.005).** This is the strongest single piece of evidence in the entire
  campaign — but it rests on 64 trades.

## 15. Adversarial / falsification tests (Phase 18; frozen k, OOS)

| Dir | variant | n | gross | net |
|---|---|---:|---:|---:|
| SHORT | as_is | 165 | +0.165 | +0.103 |
| SHORT | reversed | 165 | −0.165 | −0.228 |
| SHORT | delay_1bar | 165 | +0.163 | +0.100 |
| SHORT | **shuffled** | 165 | **+0.179** | **+0.116** |
| LONG | as_is | 64 | +0.864 | +0.801 |
| LONG | reversed | 64 | −0.864 | −0.926 |
| LONG | delay_1bar | 64 | +0.788 | +0.725 |
| LONG | **shuffled** | 64 | **−0.001** | **−0.064** |

- **SHORT: random entry timing produces the same result as the actual signal
  (+0.179 vs +0.165 gross). The short "edge" is pure timing noise.** This is
  the falsification, in one row.
- **LONG: random timing destroys the edge entirely (−0.001 vs +0.864 gross).**
  The LONG signal genuinely times the market; 1-bar execution delay costs
  ~0.08R but does not remove it.

## 16. Untouched final test (Phase 16; single-shot, frozen rules, 2025-06-01+)

Run exactly once, after all research decisions were frozen. Nothing in this
section influenced any other analysis.

**SHORT (k=2):**

| metric | value |
|---|---:|
| trades | 56 |
| gross | −0.706 |
| **net** | **−0.768** |
| win | 0.482 |
| flat | 0.989 |
| by symbol | USDCHF +0.93, AUDUSD −3.77, AUDCHF −2.57, NZDUSD −0.66, USDCAD −0.36, GBPCAD +0.56 |
| by regime | Range −0.76, Bear −0.80 |

**LONG (k=3):**

| metric | value |
|---|---:|
| trades | 11 |
| gross | +1.625 |
| **net** | **+1.563** |
| win | 0.818 |
| flat | 0.998 |
| by symbol | USDCHF +0.20 (only symbol with ≥ 5 trades) |
| by regime | Bear Trend +2.65 |

- **The SHORT hypothesis fails the untouched test outright (−0.77 net, both
  regimes negative).**
- **The LONG hypothesis is positive (+1.56 net, 82% win) but on 11 trades,
  one symbol, one regime.** A single USDCHF Bear-Trend cluster accounts for
  the entire result. It is *consistent* with the OOS finding, not *proof*.

## 17. Failure analysis

1. **The short leg was a selection artifact.** Stage-5's promising short
   numbers came from pooling; per-symbol the only well-sampled pair is net
   negative; the pooled positive was carried by n<20 symbols; permutation
   (p=0.38), shuffled timing, and the untouched test all agree it is noise.
   The regime gate (shorting Bear/Range) produces ~0 drift to harvest
   (always_short ≈ 0 net), and the extension triggers do not add timing
   information on top.
2. **The long leg is real but unproven at scale.** Significant permutation
   test, shuffled-timing destruction, positive untouched test, positive
   target EV in three families — but 64 OOS / 11 untouched trades,
   concentrated in USDCHF/Bear. Everything the robustness battery can say
   about a 64-trade sample has been said.
3. **Exit architecture remains the payoff bottleneck.** Train-selected SL/TP
   exits do not transfer to OOS on either side; the fixed-horizon edge (LONG)
   is monetized only at time-exit, not via targets/stops.
4. **Confirmation logic is net-negative** and should not be added.

## 18. Production promotion gate (Phase 20) — scorecard

| Gate | SHORT | LONG |
|---|---|---|
| No leakage | ✅ | ✅ |
| Positive OOS expectancy | ⚠️ (+0.10, ns) | ✅ (+0.80) |
| Positive cost-adjusted expectancy | ✅ (be 0.20) | ✅ (be 0.20) |
| Multiple profitable walk-forward folds | ⚠️ 2/3, one −0.31 | ⚠️ 2/3, fold3 n=1 |
| Sufficient trade count | ⚠️ 165 | ❌ 64 (11 untouched) |
| Acceptable confidence interval | ❌ CI ⊃ 0, p=0.38 | ✅ CI ⊅ 0, p=0.005 |
| Controlled multiple testing | ❌ | ✅ |
| Stable across time | ❌ untouched −0.77 | ⚠️ untouched +1.56 (n=11) |
| Not dominated by one symbol | ❌ USDCHF −0.60 | ❌ USDCHF only |
| Not dominated by one regime | ❌ Range only | ❌ Bear only |
| Calibrated probabilities | n/a (empirical) | n/a (empirical) |
| Target-level EV positive | ❌ all families negative | ✅ 3 of 4 families |
| Realistic cost robustness | ✅ | ✅ |
| Acceptable drawdown | n/a (no equity curve) | n/a |
| Beats meaningful baselines | ❌ loses to rsi_reversal | ⚠️ ≈ baseline (L1) |
| Survives adversarial tests | ❌ shuffled ≡ signal | ✅ shuffled kills edge |
| Untouched final test confirms | ❌ | ⚠️ positive, n=11 |

**Gate verdict: SHORT fails 6 critical gates; LONG fails on sample size and
concentration. Neither side may be promoted.**

## 19. Untouched-test discipline note

The untouched window was evaluated once, in a dedicated run whose only output
was §16. A subsequent regeneration of the other phases explicitly excluded
`--untouched`; the JSON artifact therefore contains all OOS analyses plus the
single-shot untouched numbers preserved in this document. The evaluation is
deterministic and reproducible via `python -m src.analysis.stage6 --untouched`.

## 20. Limitations

- 64-trade LONG sample: all statistical inference on the long side is
  low-power; the bootstrap CI is wide ([+0.39, +1.39]).
- No per-barrier probability *model* — empirical frequencies under one frozen
  exit only (TP=1R, h=20); P(TP2/TP3) unresolved (Stage-3 showed 2–3%).
- No equity-curve metrics (Sharpe/Sortino/maxDD) — trade-level expectancy only.
- Regime labels are a single fixed classification; no regime-label sensitivity.
- Spread/slippage stress beyond 0.20 ATR not tested for the combos.
- USDCHF concentration means the LONG result may be a single-asset effect
  (CHF regime) rather than a general reversal effect.

## 21. Answers to the campaign's 14 questions

1. **Is the LONG reversal edge real?** — Likely, but unproven at scale
   (p=0.005, shuffled-kills, untouched +1.56 on n=11).
2. **Is the SHORT reversal edge real?** — **No. Falsified** (p=0.38,
   shuffled ≡ signal, untouched −0.77).
3. **Does the edge survive transaction costs?** — Yes for both at the
   fixed-horizon level (break-even 0.20 ATR); no for SL/TP expressions.
4. **Does it survive walk-forward?** — Partially: LONG 2/3 (fold3 n=1),
   SHORT 2/3 (fold2 −0.31).
5. **Does it survive the untouched test?** — LONG yes (n=11, single symbol);
   SHORT no.
6. **Is it robust across symbols?** — No (LONG unattributable, SHORT
   negatively concentrated).
7. **Is it robust across regimes?** — No (each side lives in exactly one
   regime).
8. **Independent of one/two symbols?** — No (LONG = USDCHF; SHORT wrecked by
   USDCHF).
9. **Is the probability model calibrated?** — Not modeled this stage;
   empirical frequencies are honest but unsmoothed.
10. **Is target-level EV genuinely positive?** — LONG: +0.14..+0.25R for
    L1/L3/L4; SHORT: negative everywhere.
11. **Does it beat simple baselines?** — SHORT: no (loses to plain RSI
    reversal). LONG: L1 ≈ baseline; the combo's incremental value unproven.
12. **Should NexusQuant trade it?** — **No.**
13. **If yes, under what conditions?** — n/a this stage.
14. **What research should happen next?** — See §22.

## 22. Final decision

> ### **B. PROMISING BUT INSUFFICIENT EVIDENCE — LONG leg only. The SHORT leg is FALSIFIED.**

The campaign's central claim — *extreme upside extension in Bear/Range creates
SHORT opportunity, extreme downside extension creates LONG opportunity* — is
**rejected for the short side and unproven for the long side**.

**What the evidence actually supports:**
- A genuine, timing-dependent, statistically significant **LONG reversal
  signal** (RSI<30, 5-down-streaks, deep-below-SMA200 crash tail) that
  survives permutation testing, adversarial timing, costs, and — weakly — the
  untouched period. It lives in Bear Trend, appears concentrated in USDCHF,
  and is monetizable only at time-exit (SL/TP destroys it).
- **No support for any SHORT extension expression.** Every falsification
  instrument agrees.
- FLAT remains the correct default state (~99% of bars under these rules).

**What must happen before any promotion (Stage-7, in order):**
1. **Kill the short side of the research program** unless a *different* short
   information source (not extension-triggered) is proposed and pre-registered.
2. **Grow the LONG sample**: extend the universe beyond the 16 core symbols
   and/or lower the k threshold to k=2 (n=423, OOS net +0.274) *if* justified
   on train data — then re-run the full battery.
3. **Solve the exit problem for real**: the fixed-horizon edge must be
   converted to a tradeable exit (time-stop variants first; trailing/adaptive
   second) with train-selected, OOS-verified transfer.
4. **Re-run the complete campaign** (combos → WF → untouched) on the expanded
   sample with the new exit, then the untouched test a second time — the
   untouched window has now been consumed once; a new single-shot window must
   be reserved before that run.

**Bottom line:** Stage-6 did its job — it falsified the short half of the
hypothesis and promoted the long half from "promising" to "significant but
sample-starved." That is a cleaner research position than any marginally
profitable backtest would have been.

## 23. Reproducibility

```bash
# integrity audit + all OOS phases (untouched excluded by design)
./venv/bin/python -m src.analysis.stage6 --all
# single-shot untouched test (run once, frozen rules)
./venv/bin/python -m src.analysis.stage6 --untouched
```

Results JSON: `data/validation/stage6_results.json`. Deterministic (fixed
seeds); full regression suite **587 tests OK (2 skipped)**, ruff + format
clean.
