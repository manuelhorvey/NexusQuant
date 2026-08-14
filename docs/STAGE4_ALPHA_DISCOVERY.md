# NexusQuant — Stage-4: Alpha Discovery & Signal-Architecture Forensics

**Date:** 2026-08-13
**Status:** Research report — verdict **WEAK / UNSTABLE ALPHA**
**Module:** `src/analysis/stage4.py` (rerunnable, see §16)
**Data:** 16 core symbols × D1 (the watchlist) for the model-level phases; 27-symbol
universe (16 + EURGBP, GBPCHF, EURCHF, EURNZD, GBPNZD, EURAUD, GBPAUD, NZDCHF,
NZDCAD, AUDCAD, CADCHF) for the market-wide phases; ~58k bars, all fresh.
**Reservation:** every number below excludes **2025-06-01+** — that period is
reserved untouched for the eventual single-shot final test (Phase 16).

Stage-3 established that calibration is fixable (ECE 0.036/0.038 OOS) but that
the trading architecture does not demonstrate economic alpha, and that the raw
models are near-flat rankers (slope 0.01–0.025). Stage-4 removes the trading
architecture entirely and asks the prior question:

> **Does the existing feature / information set contain statistically
> defensible directional alpha — and if so, what is the correct expression of
> it (LONG / SHORT / FLAT)?**

The one-line answer: **yes — the information set contains a strong, stable,
multiple-testing-controlled MEAN-REVERSION signal, and it is the opposite of
the current architecture.** Every momentum/trend feature predicts *reversal*
(fading distance-from-SMA200, RSI extremes, MA-stack, VIX), strongest in Bear
and Range regimes; high volatility predicts *bounces*. But no simple expression
of that signal survives realistic costs, the multivariate model fails to
extract it out-of-sample (OOS AUC ≈ 0.49), and the preliminary walk-forward is
positive only in the most recent fold. The correct architecture is therefore a
**regime-conditional mean-reversion engine with FLAT as the default state** —
not the buy-the-dip/trend-continuation framework currently deployed.

---

## 1. Baseline (frozen)

| Item | Value |
|---|---|
| HEAD | `4410096` + uncommitted two-sided fixes + Stage-3 module in working tree |
| Models | `dip_lgbm.joblib` / `rally_lgbm.joblib`, trained 2026-08-12, split 2022-01-01 |
| Label geometry | sign of 10-bar ATR-normalized forward return (this stage) |
| Feature set | `FEATURE_COLUMNS` (55 numeric + symbol), dip-signal context |
| Cost assumption | 0.05R round trip = **0.0625 ATR** (1R = 1.25×ATR) |
| Thresholds | family score 0.45, EV floor +0.2R, R:R ≥ 2.5 — untouched this stage |
| Calibration | Stage-3: OOS-fitted Platt/beta achieve ECE 0.036/0.038 |
| Stage-3 summary | target-level EV ≈ 0 after costs; LONG_BUY_DIP negative at 10–20 bars; only LONG_MEAN_REVERSION (n=40) and SHORT_BREAKDOWN (n=121) positive |

---

## 2. Unconditional forward-return edge — no architecture at all (Phases 2/5/8)

Every bar, no gates, no setup families, no SL/TP. Pooled across 27 symbols:

| h | n | mean ATR | median | P(up) | t |
|---|---|---|---|---|---|
| 1 | 58,315 | −0.0056 | 0.000 | 0.506 | −2.25 |
| 5 | 58,315 | −0.0315 | 0.000 | 0.508 | −5.70 |
| 10 | 58,315 | −0.0614 | 0.000 | 0.508 | −7.78 |
| 20 | 58,315 | −0.1150 | 0.001 | 0.512 | −10.42 |
| 40 | 58,315 | −0.2178 | 0.002 | 0.513 | −14.67 |

**Findings**

1. **No unconditional drift edge.** P(up) is 0.506–0.513 (entropy ≈ 1.0 bit):
   direction at any horizon is a near coin flip. "Always long" accuracy ceiling
   is 51.3%.
2. The *negative* pooled mean is a universe-composition artifact of this
   sample: CHF crosses crashed (AUDCHF −0.435 ATR / 10-bar, NZDCHF −0.429,
   CADCHF −0.320, EURCHF −0.214), while the majors are flat-to-positive
   (USDJPY +0.002, t=4.2; EURJPY +0.001, t=3.9; XAUUSD ~0, t=9.1; USDCAD
   +0.078, t=2.4). Symbol selection *is* the drift — a cross-sectional fact,
   not a timing edge.
3. **Per-regime unconditional returns reveal the trend-reversal structure**:
   after "Bull Trend" bars the next 10 bars average −0.123 ATR (t = −8.1);
   after "High Volatility" bars +0.103 ATR (t = +2.4); Range −0.056 (t = −5.3);
   Bear −0.018 (ns). The market *fades* detected trends — including the
   uptrends the buy-the-dip engine hunts inside.

**FLAT base rates (the decision frame, h=10):**

| k ATR | P(up > k) | P(down < −k) | P(flat) |
|---|---|---|---|
| 0.25 | 0.303 | 0.314 | 0.383 |
| 0.50 | 0.264 | 0.272 | **0.465** |
| 1.00 | 0.191 | 0.199 | **0.610** |
| 1.50 | 0.132 | 0.143 | **0.725** |

**FLAT is the correct default.** At a ±1 ATR bar there is a 61% chance of
neither side being reached within 10 bars; at ±1.5 ATR, 72.5%. Any
LONG/SHORT/FLAT engine that trades more than ~40% of bars is structurally
over-trading this market.

---

## 3. Feature-level alpha audit (Phase 3) — FDR-controlled

Univariate rank-IC of every model feature vs forward ATR returns, pooled over
the 16 core symbols, horizons 5/10/20. **159 tests, 78 significant at
Benjamini–Hochberg q = 0.05.** Stability = rank-IC in the first vs second
chronological half.

**Strongest features (h=20 unless noted):**

| Feature | rank IC | hit spread | IC 1st half | IC 2nd half |
|---|---|---|---|---|
| vix_score | −0.0717 | −0.085 | −0.078 | −0.091 |
| vs_sma200_pct | −0.0593 | −0.012 | −0.068 | −0.085 |
| vol_x_trend | −0.0580 | −0.011 | −0.064 | −0.086 |
| ma_stack | −0.0457 | −0.020 | −0.053 | −0.041 |
| volatility_20 | **+0.0438** | +0.020 | +0.054 | +0.014 |
| rsi_14 | −0.0430 | −0.014 | −0.025 | −0.096 |
| risk_mom20 | −0.0426 | −0.020 | −0.045 | −0.052 |
| atr_pct | **+0.0408** | +0.018 | +0.050 | +0.023 |
| sma50_gap | −0.0412 | −0.015 | −0.026 | −0.096 |
| dip_depth_pct | **+0.0310** | +0.015 | +0.015 | +0.067 |
| h4_mom5 (h=5) | **+0.0310** | +0.015 | +0.041 | +0.022 |

**Findings**

1. **The information set contains a real, persistent signal — and it is
   REVERSAL.** Distance above the 200-SMA, RSI extremes, MA-stack, VIX and
   risk-on momentum all predict *negative* future returns. The magnitudes
   (−0.04 to −0.07 rank IC) are economically meaningful for daily FX, and the
   signal **strengthens over time** (rsi_14: −0.025 → −0.096; sma50_gap:
   −0.026 → −0.096; sma20_gap: −0.009 → −0.083 between halves) — not a decayed
   or dead artifact.
2. **Volatility is positively predictive** (volatility_20 +0.044, atr_pct
   +0.041, bb_width +0.029): high vol → positive forward returns (bounce).
   This is the one component that aligns with the current system's "LONG in
   High Volatility" cells and is a genuinely independent information source.
3. **Only one dip-family feature has positive IC — dip_depth_pct (+0.031)**:
   *deeper* dips predict up-moves (a pure mean-reversion effect). The rest of
   the dip componentry (dip_score −0.016, bias_score −0.028, above_sma200
   −0.031, ma_stack −0.046) is negative — the dip engine's *confirmation*
   layer predicts the wrong direction.
4. The only momentum-positive cells: h4_mom5 at h=5 (+0.031) and risk_mom20 in
   Bull Trend (+0.057, §7) — short-horizon H4 momentum and cross-asset
   risk-on *continuation inside bull regimes*.

---

## 4. Regime-conditional alpha (Phase 8) — where the signal lives

Rank-IC at h=10 of the strongest features, by regime (16 core symbols):

| Feature | Bull Trend | Bear Trend | Range / Chop |
|---|---|---|---|
| vs_sma200_pct | −0.0033 | **−0.1135** | −0.0284 |
| vol_x_trend | +0.0027 | **−0.1114** | −0.0287 |
| risk_mom20 | **+0.0569** | −0.0934 | −0.0435 |
| vix_score | −0.0089 | −0.0615 | −0.0460 |
| ma_stack | −0.0181 | +0.0002 | −0.0366 |
| sma50_gap | −0.0202 | −0.0559 | −0.0265 |
| rsi_14 | −0.0380 | −0.0351 | −0.0222 |
| atr_pct | −0.0032 | **+0.0562** | — |

**Findings**

1. **The reversal signal is overwhelmingly a Bear/Range phenomenon.** Fading
   distance-above-SMA200 has rank-IC −0.114 in Bear Trend vs −0.003 in Bull:
   rallies into resistance are the short opportunity the architecture has been
   structurally unable to see (its shorts require price *below* the 200-SMA).
   This is the quantitative confirmation of the Stage-2 architectural fix.
2. **Continuation exists only at the cross-asset level inside bull regimes**
   (risk_mom20 +0.057 in Bull) and short-horizon H4 momentum — narrow and
   specific.
3. **rsi_14 reversal is regime-uniform** (−0.022 to −0.038) — the most stable
   single predictor across all states.
4. Consequence: the correct participation policy is regime-conditional —
   reversal-heavy in Bear/Range, mostly FLAT in Bull, with volatility-bounce
   longs wherever vol spikes.

---

## 5. Horizon discovery / signal decay (Phases 7/12) — why BUY_DIP fails

Rank-IC by horizon, pooled over 27 symbols:

| Feature | h1 | h3 | h5 | h10 | h20 | h40 |
|---|---|---|---|---|---|---|
| vs_sma200_pct | −0.013 | −0.031 | −0.033 | −0.054 | −0.068 | −0.074 |
| rsi_14 | −0.019 | −0.027 | −0.026 | −0.043 | −0.049 | −0.069 |
| bb_pct_b | −0.020 | −0.022 | −0.017 | −0.030 | −0.035 | −0.050 |
| sma20_gap | −0.019 | −0.023 | −0.019 | −0.031 | −0.034 | −0.046 |
| slope20 | −0.013 | −0.019 | −0.020 | −0.036 | −0.043 | −0.055 |
| ret_5 | −0.022 | −0.020 | −0.011 | −0.019 | −0.022 | −0.031 |

**The signal does not decay — it compounds with horizon.** The reversal effect
is monotonically stronger at 10–40 bars. This directly explains the Stage-3
finding that LONG_BUY_DIP is negative at 10–20 bars: the engine buys strength
(dips inside uptrends with bullish confirmation) precisely where the data says
returns are most negative, and its 5–20-bar holding period is exactly the
horizon at which the reversal effect is strongest. It is not a tuning problem;
the setup family expresses the wrong sign of the available information.

---

## 6. Label audit (Phase 11)

| h | n | P(up) | entropy (bits) | always-long accuracy |
|---|---|---|---|---|
| 1 | 58,315 | 0.506 | 1.000 | 0.506 |
| 10 | 58,315 | 0.508 | 1.000 | 0.508 |
| 40 | 58,315 | 0.513 | 0.999 | 0.513 |

Sign labels are near-maximally balanced — no class-imbalance artifact inflates
accuracy. The label problem is *information*, not balance: any label scheme
built on these features must exploit the reversal effect (§3–§4) or it is
predicting noise.

---

## 7. Cross-sectional test (Phase 9)

Daily tercile long-short on 10-bar momentum across 27 symbols (2,629 days,
avg 22.2 symbols/day):

- Cross-sectional rank-IC: **−0.0116 (t = −1.52)** — momentum does not rank
  assets correctly; the sign points to reversal.
- Tercile long-short spread: −0.009 ATR gross → **−0.134 ATR net** of
  turnover×cost. Cross-sectional momentum is dead money after costs; a
  reversal-flavored cross-section (fade top tercile, buy bottom tercile)
  would be the direction to test next, but nothing in this sample supports
  momentum cross-sectionality.

---

## 8. Basket relative-value (Phase 10)

Basket-relative 5-bar momentum vs forward 10-bar return, per basket
(own momentum minus same-day basket mean):

| Basket | Members | Mean rank-IC |
|---|---|---|
| JPY | 6 crosses | −0.0273 |
| USD | 6 | −0.0185 |
| EUR | 3 | −0.0314 |
| CHF | 3 | −0.0271 |

**Basket-relative momentum also reverses** — the effect is not an
idiosyncrasy of individual symbols but operates within currency families too.
No triangular/basket relationship in this feature set survives as a long
momentum signal; the same reversal sign dominates everywhere.

---

## 9. Grouped ablation (Phase 4) — unified model, chronological split

Unified LightGBM (same hyperparameters as the deployed models) on all 16 core
symbols, target = sign(10-bar ATR return), train < 2022-01-01 (n=17,273),
test ≥ 2022 (n=14,493). dAUC = OOS AUC after removing the group minus baseline.

| Baseline | AUC | Brier | rank IC |
|---|---|---|---|
| full feature set | **0.4947** | 0.286 | −0.0091 |

| Remove group | AUC | dAUC | read |
|---|---|---|---|
| cross (risk/gold) | 0.5068 | **+0.0123** | harmful — removing helps most |
| macro | 0.4996 | +0.0049 | harmful |
| momentum | 0.4979 | +0.0032 | harmful |
| time | 0.4977 | +0.0030 | harmful |
| cot | 0.4968 | +0.0021 | harmful |
| interactions | 0.4958 | +0.0011 | neutral |
| mtf (H4) | 0.4928 | −0.0019 | mildly informative |
| volume | 0.4923 | −0.0024 | mildly informative |
| regime | 0.4915 | −0.0032 | informative |
| volatility | 0.4910 | −0.0037 | informative |
| dip | 0.4884 | **−0.0063** | informative |
| trend | 0.4874 | **−0.0073** | most informative |

**Findings**

1. **The unified model does not extract the univariate signal OOS** — baseline
   AUC 0.495 (below random), rank-IC −0.009. This is the multivariate
   counterpart of Stage-3's flat raw slopes: the LGBM trained on the sign
   label with these features fails to capture the reversal structure that the
   univariate audit shows is present. The mismatch (strong univariate reversal
   IC, anti-predictive multivariate model) is itself a finding: the model
   config/label/feature interaction is destroying signal, not adding it.
2. **The trend and dip groups are the most informative** (removing them hurts
   most) — but *informative in the wrong direction*: they are the carriers of
   the reversal relationship the model fails to exploit (and which the
   deployed engines express with the wrong sign).
3. **Cross-asset and macro proxies are actively harmful OOS** (removing them
   improves AUC by +0.012/+0.005). These enter the deployed models as
   features; on this sample they add noise beyond the reversal core.
4. The small dAUC magnitudes (±0.012) and a below-random baseline mean the
   ablations are second-order: the primary problem is model/label expression,
   not feature selection.

---

## 10. Simple baselines (Phase 17) — the cost hurdle

Net ATR per 10-bar hold, 27 symbols, 0.0625 ATR round-trip cost:

| Strategy | gross ATR | net ATR | hit |
|---|---|---|---|
| always short | +0.0614 | **−0.0011** | 0.492 |
| mean reversion (fade 5-bar) | +0.0140 | −0.0485 | 0.504 |
| always long | −0.0614 | −0.1239 | 0.508 |
| 200-SMA trend rule | −0.0282 | −0.0907 | 0.487 |
| 5-bar momentum | −0.0136 | −0.0761 | 0.496 |
| Donchian breakout | −0.1006 | −0.1631 | 0.461 |
| random | −0.0004 | −0.0629 | 0.499 |

**Every simple strategy is negative after 0.0625 ATR costs** on this universe
and sample. Two facts stand out: (a) *always short* is gross-positive (+0.061
ATR/10 bars) — the sample's CHF-cross crash carries it — yet still loses −0.001
net, so even the strongest drift of the period does not clear the cost hurdle;
(b) the only *systematic* (non-drift) gross-positive strategy is fading 5-bar
moves (+0.014) — the mean-reversion family again — and it nets −0.049. The
complex system must beat these numbers, and it currently does not.

---

## 11. Preliminary purged walk-forward (Phase 15, 3 folds, embargo 20 bars)

Unified model, thresholds p ≥ 0.55 long / p ≤ 0.45 short, R = fwd10 ATR / 1.25
minus 0.05R cost:

| Fold | Test window | n | AUC | nL | nS | L expR | S expR |
|---|---|---|---|---|---|---|---|
| 1 | 2022-01..2023-06 | 6,135 | 0.481 | 2,787 | 2,239 | −0.114 | −0.021 |
| 2 | 2023-07..2024-12 | 6,157 | 0.501 | 2,181 | 2,182 | −0.096 | −0.023 |
| 3 | 2025-01..2025-06 | 1,524 | 0.543 | 842 | 282 | +0.136 | +0.412 |

**Not robust across folds.** Folds 1–2 are negative on both sides; fold 3 is
positive (the smallest, single-period window). Directional accuracy improves
monotonically (0.481 → 0.543) — consistent with the reversal signal
*strengthening* over time (§3) — but as of now the strategy does not survive
chronological validation. The frozen walk-forward (Phase 15) and the untouched
2025-06-01+ test (Phase 16) remain the decisive gates.

---

## 12. Multiple-testing ledger (Phase 14)

| Experiment family | Count |
|---|---|
| Univariate feature × horizon tests | 159 (78 FDR-significant @ q=0.05) |
| Regime × feature IC cells | 24 |
| Simple baseline strategies | 7 |
| Basket tests | 4 |
| Ablation groups | 12 |
| Walk-forward folds | 3 |
| Thresholds tuned this stage | 0 (all pre-existing, untouched) |

FDR correction is applied to the feature audit. The *direction* of the
surviving effects is overwhelmingly consistent (reversal across every
independent test family: horizon curve, per-regime, cross-sectional,
basket-relative, ablation) — the result is not a single lucky cell but a
reproducible sign structure. Caveat: the 78 significant tests share
overlapping information (same bars, correlated features), so the effective
test count is lower; the consistency argument matters more than the count.
PBO/DSR remain deferred to the frozen walk-forward stage as specified.

---

## 13. Architecture determination (Phase 18)

The evidence supports a specific, narrow architecture:

**E. Regime-switching mean reversion** (reversal core, volatility-bounce long,
cross-asset continuation inside Bull regimes only) — **not** trend following,
momentum, breakout, the current buy-the-dip framework, and **not** a symmetric
long/short momentum engine.

- Direction: the signal is real and two-sided in *principle* (short the
  rallies in Bear/Range via vs_sma200/RSI/MA-stack; long the vol-bounce and
  deep-dip exhaustion), but its economic expression has not yet survived costs
  or multi-fold OOS.
- The correct default state is **FLAT** (46–61% of bars at ±0.5–1.0 ATR).
- **Do not rebuild trend/momentum expressions of this feature set.** The data
  says the current LONG_BUY_DIP / trend-continuation families trade the wrong
  sign at their holding horizon; no threshold tuning can fix that.

---

## 14. Final classification (Phase 20)

> **2. WEAK / UNSTABLE ALPHA**

Not "no evidence": the mean-reversion signal is statistically strong, stable
across time halves, FDR-controlled, and consistent across every independent
test family. Not "promising but unverified" in the optimistic sense: no
expression of the signal has yet produced positive net expectancy after costs,
and the walk-forward is negative in 2 of 3 folds.

What would move the classification, in order:
1. A mean-reversion expression (fade vs_sma200/RSI/MA-stack extremes; long
   vol-bounce/deep-dip) with the regime participation policy of §4 — tested
   for cost break-even first (Stage-3 §9 showed the break-even cost level is
   the binding constraint).
2. A model/label redesign that can actually extract the univariate reversal
   IC OOS (the current LGBM/sign-label combination is anti-predictive).
3. The frozen purged walk-forward, then the single-shot untouched test on
   2025-06-01+.
4. Explicit FLAT-rate reporting: the engine should trade < 40% of bars.

---

## 15. Remaining risks

1. **The untouched period is reserved but untested** — every conclusion above
   is pre-2025-06-01; the reversal signal's recent strength (§3 stability)
   could be regime-specific to 2022–2025 (USD strength, CHF carry unwind,
   rate cycle).
2. **Cost is a flat 0.0625 ATR charge** — no spread widening, financing, or
   impact; reversal strategies are typically cost-sensitive at daily
   frequency.
3. **Universe drift is material** — the pooled negative drift is CHF-cross
   driven; per-symbol drift (USDJPY, XAUUSD positive) is in-sample.
4. **The univariate → multivariate gap is unexplained** — why the LGBM fails
   to extract the reversal IC OOS (label margin, feature interaction, model
   regularization) is the key open research question.
5. **Small recent-fold sample** — fold 3 (the only positive one) is 1,524
   bars.
6. **The dip componentry's negative IC** means the deployed engines are
   trading a systematically wrong-signed confirmation layer; removing it is
   the first safe engineering step, but is a production change and therefore
   outside this research stage.

---

## 16. Reproduce

```bash
# Market-wide: unconditional edge, baselines, labels, cross-sectional, basket, horizon curve (27 symbols)
./venv/bin/python -m src.analysis.stage4 --symbols <27-symbol list> --uncond --baselines --labels --cross --basket --curve
# Feature audit (16 core symbols), FDR-controlled
./venv/bin/python -m src.analysis.stage4 --symbols <16-symbol list> --features
# Grouped ablation + preliminary walk-forward (16 core symbols)
./venv/bin/python -m src.analysis.stage4 --symbols <16-symbol list> --ablate --wf
# Everything
./venv/bin/python -m src.analysis.stage4 --symbols <list> --all
```

Results land in `data/validation/stage4_results.json`. All analyses are causal
(features at t vs realized returns t..t+h), deterministic, and exclude the
reserved 2025-06-01+ test period.
