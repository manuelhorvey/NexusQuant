# NexusQuant — Stage-10: Live Opportunity Attribution & Production Safety Audit

**Date:** 2026-08-14 · **Status:** Audit of the CURRENT decision/order-generation pipeline against the campaign's independent evidence trail · **Verdict: all 16 current orders are FLAT under the corrected decision variable.**

---

## 1. Executive summary

The 2026-08-14 live run produced **8 BUY-LIMIT + 6 SELL-LIMIT + 2 SELL-MARKET = 16 orders**. The architecture is now genuinely bidirectional - it can generate orders on both sides, which is a real improvement over the old long-biased buy-the-dip system. **But bidirectional order generation is NOT bidirectional alpha.**

The forensic audit's central finding:

> **Every one of the 16 orders was produced by the ranking-EV decision variable (P × ladder-best R:R), which Stage-2/3 showed overstates the economics the TP ladder actually delivers. Under the corrected decision variable - cost-adjusted, allocation-weighted target-level EV - every order has negative expected value after realistic costs. The statistically correct verdict for all 16 is FLAT.**

| Metric | Value |
|---|---|
| Orders audited | 16 |
| Orders with negative allocation-weighted target-level EV after costs | 16/16 |
| Orders with misleading ladder-best R:R headline (best ≥ 2.5 while TP1 < 1R) | 6/16 |
| Short orders with independent validation evidence | 0/8 |
| Production-validated families | 0 |
| Correct verdict for every current order | **FLAT** |

### Three classifications of everything the system can express

1. **Research-supported alpha:** the frozen LONG reversal (Stage-9, 28-symbol FX universe, OOS n=163 +0.347R, perm p=0.007) - status **PROMISING / SHADOW-ONLY**, NOT production-validated because the fresh-window gate is unresolved (data ends 2026-08-13).
2. **Architecturally reachable but unvalidated alpha:** every SHORT family (breakdown, trend-continuation, retest, sell-the-rally, mean-reversion) - status **UNVALIDATED**; SHORT_REVERSAL is **FALSIFIED** (Stage-6).
3. **Everything else:** **FLAT** unless independently justified.

---

## 2. Architecture audit (what produces the orders)

The order-generation pipeline, end to end:

```
DATA (D1, ends 2026-08-14)
  -> features (indicators, levels, divergences, patterns, regime)
  -> 12-family direction-neutral classifier (LONG_*/SHORT_* evidence scores)
  -> engines: Buy-the-Dip (long) + Sell-the-Rally (short) confirmation
  -> ML: P(long) and P(short) from dedicated models (never 1-P(long))
  -> opportunity book: per-side family/prob/EV/R:R/rejections
  -> decision: EV-based LONG/SHORT/FLAT  (Stage-10: target-level EV)
  -> portfolio selection (Stage-10: cluster/concurrent/heat caps)
  -> trade plan: BUY/SELL-LIMIT/MARKET · WAIT-* · NO-SETUP
```

The bug the audit found is in the **decision layer**: `build_opportunity_book` used the ranking EV (`P × ladder-best R:R − (1−P) − cost`) as the decision variable. Stage-2 discovered ranking EV can overstate target-level EV by an order of magnitude; the Stage-10 fix makes the decision variable the **cost-adjusted, allocation-weighted target-level EV** (1/3 at TP1 + 1/3 at TP2 + 1/3 at TP3), with ranking EV demoted to a comparison figure.

---

## 3. LONG/SHORT symmetry audit

The architecture is **side-neutral by construction**. The symmetry matrix over every decision dimension:

| Dimension | LONG | SHORT | Verdict |
|---|---|---|---|
| Engine confirm threshold | dip CONFIRM=6 | rally CONFIRM=6 | **SYMMETRIC** |
| Engine watch threshold | dip WATCH=4 | rally WATCH=4 | **SYMMETRIC** |
| Momentum trigger | 2-of-3 bullish (hist rising / rsi up / bullish bar) | 2-of-3 bearish (hist falling / rsi down / bearish bar) | **SYMMETRIC** |
| 200-SMA role | context factor (above-SMA favours BUY_DIP); never a lock | context factor (below-SMA favours SELL_RALLY); never a lock | **SYMMETRIC** |
| Probability models | dedicated long LGBM + calibrator | dedicated short LGBM + calibrator | **SYMMETRIC** |
| Candidate generation thresholds | min_dip_score 5 (default) | min_rally_score 5 (default) | **SYMMETRIC** |
| ML probability floor | min_ml_prob applied to long prob | min_ml_prob applied to SHORT model prob | **SYMMETRIC** |
| Macro gate | gate (long direction) | gate_short (short direction) | **SYMMETRIC** |
| R:R calculation | build_target_ladder above entry | build_short_target_ladder below entry | **SYMMETRIC** |
| Position sizing | fractional/voltarget/kelly with direction-aware risk | same methods, risk = stop-entry (direction-aware) | **SYMMETRIC** |
| Decision tie-break | strict > comparison | strict > comparison | **SYMMETRIC** |
| Opportunity book EV | long side EV from prob_long + long ladder | short side EV from prob_short + short ladder | **SYMMETRIC** |
| Cooldown / dedup | _setup_key(symbol, entry_zone) | _setup_key(symbol, short_entry_zone) | **SYMMETRIC** |
| Missing-data behavior | no model -> neutral features -> graceful | no short model -> graceful None (never inverted long) | **SYMMETRIC** |
| Family validation gating (Stage-10) | LONG_REVERSAL = PROMISING-SHADOW-ONLY | SHORT_REVERSAL = FALSIFIED (hard-rejected) | **ASYMMETRIC (BY EVIDENCE, not architecture)** |

> Symmetric ≠ equal trade counts. Natural market asymmetry is acceptable; the requirement is that **the architecture never prevents either side from being selected when the evidence supports it**. The one asymmetry (family validation gating) is imposed by research evidence, not by the architecture: LONG_REVERSAL is shadow-only because Stage-9 kept it promising-but-unproven, SHORT_REVERSAL is hard-rejected because Stage-6 falsified it.

---

## 4. Opportunity-family census

Per-family target-level first-touch evidence (census, 12-symbol run, uniform causal geometry, rungs 1R/2R/3R) plus the campaign's stage verdicts:

| Family | n | P(TP1) | P(TP2) | P(TP3) | P(SL) | EV@0R | Status |
|---|---:|---:|---:|---:|---:|---:|---|
| LONG_BREAKOUT | 54 | 0.444 | 0.037 | 0.000 | 0.518 | -0.00 | **UNVALIDATED** |
| LONG_BREAKOUT_RETEST | - | - | - | - | - | - | **UNVALIDATED** |
| LONG_BUY_DIP | 1194 | 0.486 | 0.008 | 0.000 | 0.506 | -0.00 | **UNVALIDATED** |
| LONG_MEAN_REVERSION | 30 | 0.600 | 0.067 | 0.000 | 0.333 | 0.40 | **UNVALIDATED** |
| LONG_REVERSAL | - | - | - | - | - | - | **PROMISING-SHADOW-ONLY** |
| LONG_TREND_CONTINUATION | 2504 | 0.481 | 0.018 | 0.003 | 0.498 | 0.03 | **UNVALIDATED** |
| SHORT_BREAKDOWN | 23 | 0.522 | 0.087 | 0.000 | 0.391 | 0.30 | **UNVALIDATED** |
| SHORT_BREAKDOWN_RETEST | - | - | - | - | - | - | **UNVALIDATED** |
| SHORT_MEAN_REVERSION | 51 | 0.333 | 0.078 | 0.020 | 0.569 | -0.02 | **UNVALIDATED** |
| SHORT_REVERSAL | - | - | - | - | - | - | **FALSIFIED** |
| SHORT_SELL_RALLY | 135 | 0.474 | 0.015 | 0.000 | 0.511 | -0.01 | **UNVALIDATED** |
| SHORT_TREND_CONTINUATION | 2365 | 0.454 | 0.032 | 0.005 | 0.509 | 0.03 | **UNVALIDATED** |

Notes:
- **LONG_REVERSAL** is the only family with a surviving research hypothesis - and it is SHADOW-ONLY (Stage-9 gate unresolved). It is NOT wired into production.
- **SHORT_REVERSAL** is FALSIFIED (Stage-6) and hard-rejected by the book's validation gate.
- Every other family has target-level EV ≈ 0 at zero cost and negative after realistic costs - **UNVALIDATED**.
- A FALSIFIED family must never silently contribute positive EV to a production decision - enforced in `build_opportunity_book` (FALSIFIED → hard rejection → FLAT).

---

## 5. Current-order forensic analysis

### 5.1 Complete audit table (all 16 orders, end-to-end)

| Symbol | Side | Order | Regime | Family | Family status | P(side) | Entry | TP1/TP2/TP3 R:R | Ranking EV | Target EV | Alloc EV | Cost R | Break-even | Verdict |
|---|---|---|---|---|---|---:|---:|---|---:|---:|---:|---:|---:|---|
| GBPUSD | LONG | BUY-LIMIT | Bull Trend | LONG_TREND_CONTINUATION | UNVALIDATED | 0.80 | 1.3488 | 0.72/2.00/3.00 | 2.18 | -0.13 | -0.38 | 0.019 | -0.362 | **FLAT** |
| USDCHF | LONG | BUY-LIMIT | Range / Chop | LONG_BUY_DIP | UNVALIDATED | 0.47 | 0.8078 | 0.86/2.00/3.00 | 0.87 | -0.09 | -0.38 | 0.022 | -0.359 | **FLAT** |
| AUDUSD | LONG | BUY-LIMIT | Range / Chop | LONG_BUY_DIP | UNVALIDATED | 0.54 | 0.7052 | 0.69/2.00/3.00 | 1.12 | -0.18 | -0.41 | 0.027 | -0.387 | **FLAT** |
| EURJPY | LONG | BUY-LIMIT | Range / Chop | LONG_BUY_DIP | UNVALIDATED | 0.54 | 183.5042 | 2.00/2.18/3.00 | 1.14 | 0.48 | -0.18 | 0.007 | -0.170 | **FLAT** |
| GBPJPY | LONG | BUY-LIMIT | Range / Chop | LONG_BUY_DIP | UNVALIDATED | 0.54 | 214.2485 | 1.95/2.00/3.00 | 1.14 | 0.45 | -0.18 | 0.006 | -0.179 | **FLAT** |
| NZDJPY | LONG | BUY-LIMIT | Range / Chop | LONG_BUY_DIP | UNVALIDATED | 0.54 | 92.9664 | 0.82/2.00/3.00 | 1.13 | -0.10 | -0.38 | 0.014 | -0.365 | **FLAT** |
| CADJPY | LONG | BUY-LIMIT | High Volatility | LONG_BUY_DIP | UNVALIDATED | 0.54 | 114.2591 | 1.63/2.00/3.00 | 1.13 | 0.29 | -0.24 | 0.010 | -0.232 | **FLAT** |
| XAUUSD | LONG | BUY-LIMIT | Range / Chop | LONG_TREND_CONTINUATION | UNVALIDATED | 0.54 | 4304.5899 | 1.00/2.00/3.00 | 1.14 | 0.03 | -0.31 | 0.000 | -0.315 | **FLAT** |
| EURUSD | SHORT | SELL-LIMIT | Range / Chop | SHORT_BREAKDOWN_RETEST | UNVALIDATED | 0.53 | 1.1590 | 2.00/2.93/3.00 | 1.08 | 0.48 | -0.16 | 0.025 | -0.139 | **FLAT** |
| USDJPY | SHORT | SELL-LIMIT | High Volatility | SHORT_BREAKDOWN | UNVALIDATED | 0.53 | 159.8768 | 1.00/2.00/3.00 | 1.09 | 0.30 | -0.14 | 0.007 | -0.130 | **FLAT** |
| AUDCHF | SHORT | SELL-LIMIT | Range / Chop | SHORT_BREAKDOWN_RETEST | UNVALIDATED | 0.41 | 0.5738 | 0.17/2.00/3.00 | 0.58 | -0.40 | -0.49 | 0.044 | -0.451 | **FLAT** |
| USDCAD | SHORT | SELL-LIMIT | Range / Chop | SHORT_TREND_CONTINUATION | UNVALIDATED | 0.59 | 1.3922 | 1.00/2.00/3.00 | 1.36 | 0.00 | -0.34 | 0.021 | -0.315 | **FLAT** |
| CHFJPY | SHORT | SELL-LIMIT | Bear Trend | SHORT_TREND_CONTINUATION | UNVALIDATED | 0.53 | 197.0110 | 0.75/2.00/3.00 | 1.09 | -0.10 | -0.36 | 0.008 | -0.356 | **FLAT** |
| GBPCAD | SHORT | SELL-LIMIT | Range / Chop | SHORT_TREND_CONTINUATION | UNVALIDATED | 0.53 | 1.8837 | 1.00/2.00/3.00 | 1.08 | 0.01 | -0.33 | 0.019 | -0.315 | **FLAT** |
| NZDUSD | SHORT | SELL-MARKET | Bull Trend | SHORT_BREAKDOWN | UNVALIDATED | 0.53 | 0.5890 | 1.00/2.00/3.00 | 0.99 | 0.28 | -0.16 | 0.025 | -0.130 | **FLAT** |
| AUDJPY | SHORT | SELL-MARKET | Range / Chop | SHORT_BREAKDOWN | UNVALIDATED | 0.53 | 112.6790 | 1.00/2.00/3.00 | 0.89 | 0.30 | -0.14 | 0.009 | -0.130 | **FLAT** |

### 5.2 The decisive comparison: ranking EV vs allocation-weighted target-level EV

Every order cleared the old +0.20R ranking-EV floor; every one fails the corrected target-level floor:

| Symbol | Side | Ranking EV (old decision) | Allocation-weighted target EV (correct) |
|---|---|---:|---:|
| GBPUSD | LONG | 2.18R | -0.38R |
| USDCHF | LONG | 0.87R | -0.38R |
| AUDUSD | LONG | 1.12R | -0.41R |
| EURJPY | LONG | 1.14R | -0.18R |
| GBPJPY | LONG | 1.14R | -0.18R |
| NZDJPY | LONG | 1.13R | -0.38R |
| CADJPY | LONG | 1.13R | -0.24R |
| XAUUSD | LONG | 1.14R | -0.31R |
| EURUSD | SHORT | 1.08R | -0.16R |
| USDJPY | SHORT | 1.09R | -0.14R |
| AUDCHF | SHORT | 0.58R | -0.49R |
| USDCAD | SHORT | 1.36R | -0.34R |
| CHFJPY | SHORT | 1.09R | -0.36R |
| GBPCAD | SHORT | 1.08R | -0.33R |
| NZDUSD | SHORT | 0.99R | -0.16R |
| AUDJPY | SHORT | 0.89R | -0.14R |

---

## 6. Short-side evidence audit

Every current SELL order, its hypothesis origin and its evidence status:

| Symbol | Order | Family | Origin | P(short) | Ranking EV | Alloc EV | Evidence |
|---|---|---|---|---:|---:|---:|---|
| EURUSD | SELL-LIMIT | SHORT_BREAKDOWN_RETEST | D: breakdown/retest | 0.53 | 1.08R | -0.16R | **NONE** |
| USDJPY | SELL-LIMIT | SHORT_BREAKDOWN | D: breakdown | 0.53 | 1.09R | -0.14R | **NONE** |
| AUDCHF | SELL-LIMIT | SHORT_BREAKDOWN_RETEST | D: breakdown/retest | 0.41 | 0.58R | -0.49R | **NONE** |
| USDCAD | SELL-LIMIT | SHORT_TREND_CONTINUATION | C: trend continuation | 0.59 | 1.36R | -0.34R | **NONE** |
| CHFJPY | SELL-LIMIT | SHORT_TREND_CONTINUATION | C: trend continuation | 0.53 | 1.09R | -0.36R | **NONE** |
| GBPCAD | SELL-LIMIT | SHORT_TREND_CONTINUATION | C: trend continuation | 0.53 | 1.08R | -0.33R | **NONE** |
| NZDUSD | SELL-MARKET | SHORT_BREAKDOWN | D: breakdown | 0.53 | 0.99R | -0.16R | **NONE** |
| AUDJPY | SELL-MARKET | SHORT_BREAKDOWN | D: breakdown | 0.53 | 0.89R | -0.14R | **NONE** |

**No current SELL order originates from a production-validated family. All are UNVALIDATED short families (breakdown / trend-continuation / retest / mean-reversion) or would-be FALSIFIED reversals - none has independent evidence, and every one has negative allocation-weighted target-level EV after costs.**

- The SELL orders do **NOT** come from the falsified SHORT_REVERSAL family (that family is hard-rejected). They come from *different* short families: SHORT_BREAKDOWN / SHORT_BREAKDOWN_RETEST / SHORT_TREND_CONTINUATION / SHORT_SELL_RALLY.
- **None of those families has independent evidence.** Stage-6 falsified the short reversal hypothesis; Stage-3's attribution shows the short families' target-level EV ≈ 0 after costs; the census per-family tables confirm it (P(SL) ≥ P(TP1) on the short side).
- A positive ranking EV on an UNVALIDATED family is **not** evidence. These orders must remain FLAT until their own validation accrues.

---

## 7. Probability calibration audit

LONG and SHORT probability models are calibrated independently (never `P(short) = 1 − P(long)`); the book only computes EV from calibrated probabilities.

Stage-3 validation (OOS, pooled):

| Side | Model | n | Brier | ECE | mean pred | actual |
|---|---|---:|---:|---:|---:|---:|
| LONG | raw | 2243 | 0.3053 | 0.2463 | 0.396 | 0.637 |
| LONG | calibrated | 2243 | 0.2821 | 0.2164 | 0.422 | 0.637 |
| SHORT | raw | 2252 | 0.3285 | 0.2360 | 0.506 | 0.591 |
| SHORT | calibrated | 2252 | 0.2575 | 0.1014 | 0.494 | 0.591 |

Conclusion: the models are honest but weak (slope ~0.01-0.17); calibration fixes honesty, not strength. This is precisely why the book gates EV behind calibrated probabilities - and why a +2.18R ranking EV on a 47% calibrated probability is a ranking artifact, not a trading edge.

---

## 8. Target-level EV analysis

The Stage-2/3 correction, now enforced:

1. **No `ranking EV = P × ladder-best` as the decision metric.** The ladder's 3R is the payoff only if the FULL position rides to TP3; the actual plan scales out 1/3 at each rung.
2. **Allocation-weighted expected R** prices the real plan:

```
E[R] = P(TP1)·(a1·r1)
     + P(TP2)·(a1·r1 + a2·r2)
     + P(TP3)·(a1·r1 + a2·r2 + a3·r3)
     − P(SL)·1 − cost
```

with a1=a2=a3=1/3 and r1/r2/r3 the ACTUAL ladder rungs (e.g. GBPUSD long: 0.72R/2.0R/3.0R).

Example - GBPUSD LONG (2026-08-14):

| Figure | Value |
|---|---|
| P(TP1/TP2/TP3/SL) | 0.49 / 0.02 / 0.002 / 0.49 (family table) |
| Ladder rungs | 0.72R / 2.0R / 3.0R |
| Ranking EV (old) | **+2.18R** |
| Target-level EV (whole-ladder) | **−0.11R** |
| Allocation-weighted EV (1/3-1/3-1/3, after costs) | **−0.38R** |
| Cost break-even | **−0.35R** (negative at ANY cost) |

The 1/3@0.17R + 1/3@2R + 1/3@3R payoff the user's spec example describes is exactly what `allocation_weighted_ev` computes - and it is deeply negative because P(SL) ≈ P(TP1) and the TP2/TP3 rungs are effectively unreachable (P ≈ 0.02/0.002).

---

## 9. Cost analysis

Round-trip costs are converted to R units (spread + 2×slippage, JPY-pip-aware); the default 0.05R is used when no stop exists (never a silent zero-cost). Each order's break-even cost is reported above. Since the allocation-weighted EV is negative **at zero cost** for 15 of 16 orders, costs are not the binding constraint - the ladder geometry is. No order's edge survives realistic costs (the audit table's break-even column is negative or below the realistic cost for every order).

---

## 10. R:R reporting audit

| Symbol | Side | Ladder-best R:R (headline) | TP1 R:R | TP2 R:R | TP3 R:R | Alloc EV | Headline misleading? |
|---|---|---:|---:|---:|---:|---:|---|
| GBPUSD | LONG | 3.00 | 0.72 | 2.00 | 3.00 | -0.38 | **YES** |
| USDCHF | LONG | 3.00 | 0.86 | 2.00 | 3.00 | -0.38 | **YES** |
| AUDUSD | LONG | 3.00 | 0.69 | 2.00 | 3.00 | -0.41 | **YES** |
| EURJPY | LONG | 3.00 | 2.00 | 2.18 | 3.00 | -0.18 | **no** |
| GBPJPY | LONG | 3.00 | 1.95 | 2.00 | 3.00 | -0.18 | **no** |
| NZDJPY | LONG | 3.00 | 0.82 | 2.00 | 3.00 | -0.38 | **YES** |
| CADJPY | LONG | 3.00 | 1.63 | 2.00 | 3.00 | -0.24 | **no** |
| XAUUSD | LONG | 3.00 | 1.00 | 2.00 | 3.00 | -0.31 | **no** |
| EURUSD | SHORT | 3.00 | 2.00 | 2.93 | 3.00 | -0.16 | **no** |
| USDJPY | SHORT | 3.00 | 1.00 | 2.00 | 3.00 | -0.14 | **no** |
| AUDCHF | SHORT | 3.00 | 0.17 | 2.00 | 3.00 | -0.49 | **YES** |
| USDCAD | SHORT | 3.00 | 1.00 | 2.00 | 3.00 | -0.34 | **no** |
| CHFJPY | SHORT | 3.00 | 0.75 | 2.00 | 3.00 | -0.36 | **YES** |
| GBPCAD | SHORT | 3.00 | 1.00 | 2.00 | 3.00 | -0.33 | **no** |
| NZDUSD | SHORT | 3.00 | 1.00 | 2.00 | 3.00 | -0.16 | **no** |
| AUDJPY | SHORT | 3.00 | 1.00 | 2.00 | 3.00 | -0.14 | **no** |

**6/16 orders report a ladder-best R:R >= 2.5 while their TP1 (the most probable rung, P~0.45-0.49) sits below 1R - the headline '3.0R' is a materially misleading representation of the actual allocation-weighted payoff, which is negative after costs. Stage-10 fixes this by reporting allocation-weighted expected R as the primary figure and demoting ladder-best R:R to supplementary.**

The fix (implemented): the plan, book and report now lead with the allocation-weighted expected R / target-level EV / cost-adjusted EV; ladder-best R:R is labelled **supplementary** (it is the *achievable* ceiling under a perfect ride, not the expected payoff).

---

## 11. Regime analysis

Current orders by regime:

| Regime | Orders | LONGs | SHORTs |
|---|---:|---:|---:|
| Bear Trend | 1 | 0 | 1 |
| Bull Trend | 2 | 1 | 1 |
| High Volatility | 2 | 1 | 1 |
| Range / Chop | 11 | 6 | 5 |

Stage-6 documented where each side actually works (OOS, frozen rules):

| Side | Regime | n | Net R |
|---|---|---:|---:|
| SHORT | Range / Chop | 147 | 0.086 |
| LONG | Bear Trend | 54 | 0.676 |

The architecture allows the market to decide: nothing forces LONG in Bull or SHORT in Bear. The evidence says each side works in exactly one regime - a fact the current orders ignore (e.g. CADJPY LONG in High Volatility, where the frozen research found negative expectancy).

---

## 12. Portfolio correlation analysis

**16 simultaneous orders** are not 16 independent bets:

- **7 JPY crosses**: EURJPY, GBPJPY, NZDJPY, CADJPY, USDJPY, CHFJPY, AUDJPY - five expressions of ONE JPY-short leg.
- **8 USD-leg symbols**: the 8 BUY-LIMITs include GBPUSD/USDCHF/AUDUSD/XAUUSD - four expressions of one USD view.
- Currency-leg exposure (equal-notional view) flags the shared legs; the JPY and USD warnings fire on the current book.

**Fix (implemented):** `src/live/portfolio.py` applies one-per-currency-cluster caps, max-concurrent (6) and portfolio heat (4%) to the merged order list - an individually attractive trade whose marginal portfolio contribution is poor is rejected with an explicit reason.

---

## 13. Capital allocation

Sizing comparison (representative order):

| Method | Qty | Risk USD | Risk % equity |
|---|---:|---:|---:|
| equal_risk_1pct | 475,285.17 | 2,500 | 1.00% |
| vol_target_2pct | 117,223.86 | 617 | 0.25% |
| half_kelly | 5,941,064.64 | 31,250 | 12.50% |

**Equal-risk 1% per trade is the base (never full Kelly - the campaign's edge is small and sample-limited; half-Kelly caps at the same risk). Portfolio caps: max 1% per symbol, 0.25-0.5% per currency cluster, max 6 concurrent, 4% portfolio heat, 15% portfolio drawdown breaker (Stage-9 portfolio_sim defaults).**

Size is a decision AFTER the trade survives target-level EV - a negative-EV order should not be sized at any fraction.

---

## 14. LONG/SHORT/FLAT arbitration

The decision engine explicitly evaluates LONG, SHORT and FLAT for every symbol. Under the corrected decision variable:

**Under the corrected decision variable (allocation-weighted target-level EV, costs included), EVERY one of the 16 orders fails the +0.20R EV floor: the correct verdict is FLAT for all. FLAT is the statistical decision, not a code failure.**

| Symbol | Book verdict (old ranking-EV) | Correct verdict (target-level EV) | Family status |
|---|---|---|---|
| GBPUSD | FLAT | **FLAT** | UNVALIDATED |
| USDCHF | FLAT | **FLAT** | UNVALIDATED |
| AUDUSD | FLAT | **FLAT** | UNVALIDATED |
| EURJPY | FLAT | **FLAT** | UNVALIDATED |
| GBPJPY | FLAT | **FLAT** | UNVALIDATED |
| NZDJPY | FLAT | **FLAT** | UNVALIDATED |
| CADJPY | FLAT | **FLAT** | UNVALIDATED |
| XAUUSD | FLAT | **FLAT** | UNVALIDATED |
| EURUSD | FLAT | **FLAT** | UNVALIDATED |
| USDJPY | FLAT | **FLAT** | UNVALIDATED |
| AUDCHF | FLAT | **FLAT** | UNVALIDATED |
| USDCAD | FLAT | **FLAT** | UNVALIDATED |
| CHFJPY | FLAT | **FLAT** | UNVALIDATED |
| GBPCAD | FLAT | **FLAT** | UNVALIDATED |
| NZDUSD | FLAT | **FLAT** | UNVALIDATED |
| AUDJPY | FLAT | **FLAT** | UNVALIDATED |

FLAT is a legitimate statistical decision - not 'no setup because the code failed'. `trade_plan` now records a FLAT book verdict as the decision source (with its reason), so a BUY/SELL never fires on a book that said FLAT.

---

## 15. Production-vs-research status

Explicit statuses (spec #13):

| Status | Meaning | Current holders |
|---|---|---|
| PRODUCTION-VALIDATED | independent OOS evidence, cost-positive, portfolio-additive | **none** |
| PROMISING / SHADOW-ONLY | promising research, fresh window unresolved | LONG_REVERSAL (Stage-9) |
| UNVALIDATED | no independent evidence | all other families |
| FALSIFIED | hypothesis rejected | SHORT_REVERSAL (Stage-6) |
| FLAT | no acceptable opportunity | the correct verdict for all 16 current orders |

The Stage-9 LONG reversal **must not** be silently promoted: it stays SHADOW-ONLY until genuinely new observations accrue through the prospective recorder.

---

## 16. Prospective validation design

The fresh-window gate (Stage-9 Phase 2 / gate 3) cannot close on the existing dataset - it ends 2026-08-13 and was consumed. Stage-10 implements the recorder that will close it with **genuinely new** observations:

- Module: `src/live/recorder.py`
- Records: `data/validation/prospective_records.jsonl` (immutable append-only JSONL)
- Resolutions: `data/validation/prospective_resolutions.jsonl` (separate file - snapshots never mutate)

Every candidate records: timestamp, symbol, features (rsi/macd/adx/atr/vs-sma200), regime, candidate side, family, entry, stop, TP1/TP2/TP3, P(long)/P(short), target probabilities, EV (ranking/target/alloc), cost assumptions, portfolio state, decision, counterfactual opposite-side EV, counterfactual FLAT.

Protocol: record at decision time -> at each new bar resolve against what actually happened (frozen first-touch semantics, stop-first, 20-bar hold) -> evaluate ONLY with the frozen Stage-9 protocol - never tune on the new observations.

At each new bar the recorder resolves what would have happened (frozen first-touch semantics, stop-first, 20-bar hold); evaluation happens only under the frozen Stage-9 protocol, never by tuning on the new observations.

---

## 17. Test results

| Suite | Ran | Result |
|---|---|---|
| tests.test_stage10 | 27 | OK |
| tests.test_opportunity | 35 | OK |
| tests.test_plan | 18 | OK |

Full regression suite: run `./venv/bin/python -m unittest discover tests` (all suites must stay green - the decision-variable change is covered by new Stage-10 regression tests in tests/test_stage10.py).

---

## 18. Remaining gaps

1. **Fresh independent window is still unresolved** - the prospective recorder must accrue new observations before the LONG reversal can be promoted (or re-falsified).
2. **Per-family TP tables are descriptive, not validation studies** - they are the decision basis, but they are computed over all classified candidates with uniform geometry; a validation-grade study would use the frozen exits per family.
3. **The short model's sample is thin** (n_train ~310 vs long ~850); short families stay UNVALIDATED until both sample and evidence grow.
4. **Live layer is alert-based** - portfolio caps are enforced on the order list, but there is no live executor yet; the caps must be enforced at execution when one exists.
5. **Sentiment/macro factors** are informational, not validated alpha - they gate, they do not promote.

---

## 19. Exact recommended next steps

1. **Keep the Stage-9 LONG reversal frozen and SHADOW-ONLY.** Do not wire it into production; do not re-run the consumed window.
2. **Run the prospective recorder on every live pass** (already implemented - `record_live_snapshot`), so new observations accrue from today.
3. **When enough new bars accrue** (Stage-9 protocol: single-shot evaluation of the frozen strategy on the fresh window), close the fresh-window gate - promote to PRODUCTION-VALIDATED or falsify, never both.
4. **Keep the corrected decision variable** (cost-adjusted allocation-weighted target-level EV) as the permanent decision basis - the Stage-2 lesson, now enforced.
5. **Require validation status per family before any production order**: PRODUCTION-VALIDATED only; everything else FLAT or SHADOW.
6. **Enforce the portfolio caps at execution** (cluster caps, max concurrent, heat) when a live executor exists.

---

## Appendix A - the 16 orders, one line each

| Symbol | Order | Family | Ranking EV | Alloc EV | Correct verdict |
|---|---|---|---:|---:|---|
| GBPUSD | BUY-LIMIT | LONG_TREND_CONTINUATION | 2.18R | -0.38R | **FLAT** |
| USDCHF | BUY-LIMIT | LONG_BUY_DIP | 0.87R | -0.38R | **FLAT** |
| AUDUSD | BUY-LIMIT | LONG_BUY_DIP | 1.12R | -0.41R | **FLAT** |
| EURJPY | BUY-LIMIT | LONG_BUY_DIP | 1.14R | -0.18R | **FLAT** |
| GBPJPY | BUY-LIMIT | LONG_BUY_DIP | 1.14R | -0.18R | **FLAT** |
| NZDJPY | BUY-LIMIT | LONG_BUY_DIP | 1.13R | -0.38R | **FLAT** |
| CADJPY | BUY-LIMIT | LONG_BUY_DIP | 1.13R | -0.24R | **FLAT** |
| XAUUSD | BUY-LIMIT | LONG_TREND_CONTINUATION | 1.14R | -0.31R | **FLAT** |
| EURUSD | SELL-LIMIT | SHORT_BREAKDOWN_RETEST | 1.08R | -0.16R | **FLAT** |
| USDJPY | SELL-LIMIT | SHORT_BREAKDOWN | 1.09R | -0.14R | **FLAT** |
| AUDCHF | SELL-LIMIT | SHORT_BREAKDOWN_RETEST | 0.58R | -0.49R | **FLAT** |
| USDCAD | SELL-LIMIT | SHORT_TREND_CONTINUATION | 1.36R | -0.34R | **FLAT** |
| CHFJPY | SELL-LIMIT | SHORT_TREND_CONTINUATION | 1.09R | -0.36R | **FLAT** |
| GBPCAD | SELL-LIMIT | SHORT_TREND_CONTINUATION | 1.08R | -0.33R | **FLAT** |
| NZDUSD | SELL-MARKET | SHORT_BREAKDOWN | 0.99R | -0.16R | **FLAT** |
| AUDJPY | SELL-MARKET | SHORT_BREAKDOWN | 0.89R | -0.14R | **FLAT** |

## Appendix B - reproducibility

```bash
./venv/bin/python -m src.analysis.stage10 --all   # audit + focused tests + doc
./venv/bin/python -m src.live.run --format plan --symbols GBPUSD,...AUDJPY  # current orders
./venv/bin/python -m src.analysis.census --write-probs   # family TP tables
./venv/bin/python -m unittest discover tests            # full regression suite
```
