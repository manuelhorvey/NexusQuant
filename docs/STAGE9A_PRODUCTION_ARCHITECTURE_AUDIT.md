# NexusQuant — Production Architecture Audit: Dip-Architecture Remnants & Side-Neutrality (pre-live-validation)

**Status:** Audit of the **production codebase** (not the research modules) conducted after Stage-9 froze the LONG reversal hypothesis. No production behavior was changed except one verified tie-break fix (below). Stage-9 remains frozen; the LONG reversal signal is **NOT** wired into production — by design, pending the fresh-window gate.

**Scope:** the exact audit list from the Stage-9 follow-up — signal generation, candidate discovery, long/short symmetry, regime gating, entry/SL/TP construction, exit logic, position sizing, portfolio correlation, signal deduplication, cooldowns, capital utilization, transaction costs, walk-forward model fitting, probability calibration, live-vs-research feature parity — plus the core requirement: **the opportunity discovery layer must remain side-neutral**, free to say LONG / SHORT / FLAT on the evidence.

**Bottom line: the production pipeline is side-neutral and two-sided. The original buy-the-dip architecture survives only as a named engine (Buy-the-Dip) that now competes on equal footing with Sell-the-Rally, the classifier, calibrated EV, and the macro gate. One genuine long-priority remnant was found and fixed; one cosmetic ordering remains (documented).**

---

## 1. Executive summary

| # | Audit item | Verdict | Evidence |
|---|---|---|---|
| 1 | Candidate discovery | ✅ PASS | `build_opportunity_book` evaluates BOTH sides independently → EV-driven LONG/SHORT/FLAT verdict |
| 2 | Signal generation | ✅ PASS | Scanner computes dip AND rally, long AND short ML probs, both macro gates |
| 3 | Long/short symmetry | ✅ PASS (1 fix) | Dip/rally engines exact mirrors; tie-break fixed to FLAT |
| 4 | Regime gating | ✅ PASS | 200-SMA contextual, not a directional lock (Stage-6 verified) |
| 5 | Entry construction | ✅ PASS | Per-side entry zones; market/limit classification honest |
| 6 | SL construction | ✅ PASS | Long stop below / short stop above (mirrored) |
| 7 | TP construction | ✅ PASS | `build_target_ladder` (above) / `build_short_target_ladder` (below) |
| 8 | Exit logic | ✅ PASS | Backtest exits mirrored; research exit-transfer documented separately |
| 9 | Position sizing | ✅ PASS | `fractional_qty` / `vol_target_qty` / `kelly_qty` direction-aware |
| 10 | Portfolio correlation | ⚠️ PARTIAL | Cluster caps exist in research (one-per-cluster); live pass is alert-only (no live book portfolio yet) |
| 11 | Signal deduplication | ✅ PASS | Direction-keyed dedup + expiry re-eligibilization |
| 12 | Cooldowns | ✅ PASS | Signal-expiry cooldown implemented |
| 13 | Capital utilization | ⚠️ N/A | Live layer is alert/pass-based; no live execution yet |
| 14 | Transaction costs | ✅ PASS | `roundtrip_cost_r` in R units; `target_probs.json` cost-adjusted EV |
| 15 | Walk-forward fitting | ✅ PASS | Purged walk-forward CV + embargo + chronological val tail |
| 16 | Probability calibration | ✅ PASS | Per-side isotonic calibrators saved with model bundles; census validates Brier/ECE per side |
| 17 | Live-vs-research parity | ⚠️ PARTIAL | Models served with same feature context; but the FROZEN reversal signal is deliberately NOT in production (correct) |
| 18 | **Discovery-layer neutrality** | ✅ PASS | Functional proof below: 3/6 symbols verdict SHORT, 3/6 LONG, on the same day |

**The tie-break fix (the only code change):** `classify_setup` resolved exact long/short evidence-score ties to LONG (`>=` vs `>`). Now an exact tie → FLAT (side-neutral; the market decides). Regression test added (`test_verdict_matches_strict_score_comparison`). 18/18 setup tests pass.

---

## 2. Candidate discovery — ✅ PASS (side-neutral)

`src/analysis/opportunity.py::build_opportunity_book` is the decision layer. It builds a **full opportunity book per symbol** — both a LONG and a SHORT `Opportunity`, each with its own setup family, calibrated probability, expected value, risk and explicit rejection reasons — then picks LONG / SHORT / FLAT from expected value.

- Path 1 (no calibrated probability either side): engine-confirmed side, **higher engine score decides** — never a long-first list order.
- Path 2 (EV-aware): LONG wins when `EV_L > min_ev` **and** `EV_L >= EV_S`; SHORT wins on the mirror; **neither clearing the floor → FLAT with explicit reasons**.
- Hard gates on the EV winner: R:R floor + macro gate. A winner failing a hard gate → FLAT.
- `FLAT` is the *absence* of an acceptable opportunity, never a forced third choice.

Functional proof (2026-08-13 data, same day, both sides evaluated):

| Symbol | Verdict | EV_long | EV_short | P_long | P_short | Long family | Short family |
|---|---|---|---|---|---|---|---|
| EURUSD | **LONG** | +1.12 | +0.47 | 0.536 | 0.406 | LONG_BREAKOUT | SHORT_BREAKDOWN_RETEST |
| GBPUSD | **LONG** | +2.18 | +1.00 | 0.800 | 0.525 | LONG_TREND_CONT | SHORT_BREAKDOWN_RETEST |
| USDJPY | **SHORT** | +0.89 | +1.09 | 0.473 | 0.525 | LONG_BREAKOUT_RETEST | SHORT_TREND_CONT |
| AUDUSD | **SHORT** | +0.96 | +1.07 | 0.536 | 0.525 | LONG_TREND_CONT | SHORT_BREAKDOWN_RETEST |
| USDCAD | **SHORT** | +0.87 | +1.36 | 0.473 | 0.595 | LONG_BREAKOUT_RETEST | SHORT_TREND_CONT |
| XAUUSD | **LONG** | +1.14 | +0.97 | 0.536 | 0.404 | LONG_TREND_CONT | SHORT_BREAKDOWN_RETEST |

The market decides the direction; both sides are always evaluated with calibrated probabilities and honest EV.

---

## 3. Signal generation — ✅ PASS

`src/analysis/scanner.py::scan_symbol` computes both sides in one pass:

- `detect_dip(df, trigger_df, levels)` — the long engine.
- `detect_rally(df, trigger_df, levels)` — the short engine (exact mirror).
- `predict_series(...)` (long LGBM) and `predict_short_series(...)` (dedicated short LGBM, mirror labels) — independent per-side probabilities (never `1 - P(long)`).
- Macro gate **and** macro gate-short — independent per-side top-down filters.
- Direction-neutral 12-family classifier (`setup_classification`) surfaces `LONG_*` / `SHORT_*` / FLAT regardless of engine confirmation.

The scanner's ranked table sorts by `bias_score` descending — a cosmetic long-first display ordering. It is **display-only**: the live passes re-sort per side (`dip_score` / `rally_score`) before taking the top N, so it never biases candidate selection. Documented, not a decision remnant.

---

## 4. Long/short symmetry — ✅ PASS (one fix applied)

- **Engines:** `src/features/dip.py` / `src/features/rally.py` are structural mirrors — uptrend→downtrend, support→resistance, cooled-RSI 30-55→stretched-RSI >60, bullish→bearish momentum trigger (2-of-3), same 0-8 scoring, same confirm/watch thresholds (6/4).
- **Filters:** `filter_signals` (long, `min_bias=-4` default = anything) vs `filter_short_signals` (short, `max_bias=+4` default = anything) — no directional lock by default; both require engine confirmation + macro pass.
- **Tie-break fix:** `classify_setup` previously used `long_score >= short_score` (→ LONG on exact tie) vs `short_score > long_score`. **Fixed to strict comparisons; exact tie → FLAT.** (Commit-sized change, covered by a new regression test.)

---

## 5. Regime gating — ✅ PASS (200-SMA contextual, not a lock)

Verified in the Stage-6 integrity audit and re-confirmed here:

- `detect_regime` is causal (trailing slope/ADX/SMA/ATR-ratio; no centered windows).
- The 200-SMA is an *input* to the classifier and engines, not an absolute directional gate: LONG reversal/retest families can fire below the 200-SMA, SHORT families above it (locked by `test_no_200sma_universal_gate`).
- The only full-sample standardization lives in `detect_regime_cluster` (clustering path), which the live pass does not use — documented UNKNOWN in the leakage audit.

---

## 6. Entry / SL / TP construction — ✅ PASS (mirrored)

- **Entry:** long zone from the dip engine (pullback into support), short zone from the rally engine (rally into resistance). `entry_type_for` classifies immediate (market) vs pending (limit) by proximity to close within a fraction of ATR; market fills are re-priced at the close (never claim the limit-level R:R at a market fill).
- **Stop:** long invalidation below entry / short invalidation above (mirrored).
- **Targets:** `src/risk/targets.py` — `build_target_ladder` builds TP1/TP2/TP3 **above** entry; `build_short_target_ladder` mirrors **below**. `best_rr` enforces the 2.5:1 scaling-out floor. Documented: Stage-7/9 research shows the 3R ladder is structurally wrong for the *reversal* signal specifically — but that signal is not yet in production, and the ladder serves the existing dip/rally architecture's targets.

---

## 7. Exit logic — ✅ PASS

- Backtest engine (`src/backtest/engine.py::run_backtest`) takes `side="long"|"short"` and mirrors every mechanical detail: target column (resistance for long / target for short), fills (long buys ask / short sells bid), stop touch (low for long / high for short).
- Research finding (Stage-7/9): for the *reversal* signal, time-stop / signal-reversal / return-to-mean exits transfer OOS while SL/TP ladders do not. This is **documented, not deployed** — the exit architecture is frozen with the Stage-9 hypothesis.

---

## 8. Position sizing — ✅ PASS

`src/risk/sizing.py`: `fractional_qty`, `vol_target_qty`, `kelly_qty` all take `direction` and compute risk-per-unit symmetrically (`stop - entry` for short, `entry - stop` for long). Kelly uses calibrated probabilities and a conservative half-Kelly default in settings.

---

## 9. Portfolio correlation / dedup / cooldowns — ⚠️ PARTIAL / ✅ PASS

- **Correlation:** research has one-per-currency-cluster position limits (cuts cumulative-R drawdown 31.3R → 24.2R). The **live layer is alert/pass-based** — there is no live portfolio book yet, so cluster caps are not (and cannot yet be) enforced at execution. This is a documented gap to close when a live portfolio executor is built, not a dip-architecture remnant.
- **Dedup:** `_setup_key(symbol, entry_zone)` — direction-aware (short uses `short_entry_zone`); long/short passes share the state file but keys are side-specific.
- **Cooldowns:** `purge_expired` re-eligibilizes setups older than the expiry window (default 30d).

---

## 10. Transaction costs — ✅ PASS

`roundtrip_cost_r` converts spread + 2×slippage into R units (JPY 0.01 pip / else 0.0001); defaults to `DEFAULT_COST_R = 0.05R` when no stop exists (never silently zero-costs). The live book consumes `data/validation/target_probs.json` (written by the census `--write-probs`) for honest payoff-distribution EV: P(TP1)=0.49 / P(SL)=0.49 long, 0.45/0.51 short — the same ~0-EV-after-costs numbers Stage-3 exposed. **The target-level EV is reported next to the ranking EV, and the ranking EV is the decision variable until the TP table is symbol-specific and models recalibrated (documented in the book code).**

---

## 11. Walk-forward model fitting — ✅ PASS

`src/model/run.py`: chronological splits with a one-label-horizon embargo at every boundary (`split_chronological`); purged walk-forward CV (`--cv N`); early stopping on a chronological validation tail (`_chrono_val_split`). No shuffled CV. This matches the research discipline (Stage-4..9).

---

## 12. Probability calibration — ✅ PASS (per-side, honest)

- `fit_calibrator` fits an isotonic regression on a **chronological validation slice**; the calibrator is saved in the model bundle and applied at predict time (`apply_calibrator`).
- Separate LONG and SHORT models with mirror labels (`build_labels` / `build_labels_short`), calibrated independently — never `P(short) = 1 - P(long)`.
- `census.model_calibration` scores both sides independently (n, mean predicted vs actual, Brier, ECE, reliability table).
- Caveat (Stage-3/4 finding, unchanged): the *raw* models are weak rankers (slope 0.01–0.025), so calibration makes probabilities honest but not stronger. This is why the book gates EV behind calibrated probabilities and why the fresh-window test is decisive.

---

## 13. Live-vs-research feature parity — ⚠️ PARTIAL (by design)

- The live models are served with the same H4 / cross-asset / COT context they were trained on (`predict_series` with group/data_dir context; missing inputs → neutral features, never crashes).
- **The frozen Stage-9 reversal signal (k=3 of {RSI<30, 5-bar ATR drop, 5-day streak}, time-stop exits) is NOT in the production code** — this is deliberate: Stage-9's verdict was NO (insufficient evidence), and the fresh-window gate is unresolved. Wiring it now would violate the freeze.
- Parity gap to close **when the fresh window confirms**: the research pipeline and the live pipeline currently share indicator/regime primitives but not a single frozen-signal module. When promoted, the reversal trigger + time-stop must be implemented as a first-class `Opportunity` family so it competes in the book alongside the existing engines — not as a side-channel.

---

## 14. Recommendations (when Stage-10 / the fresh window arrives)

1. **Do not touch the frozen Stage-9 strategy.** Run it exactly once on data past 2026-08-13 (single-shot, per protocol).
2. **Close the correlation gap:** build the portfolio book (one-per-cluster caps, 25–50bp risk, 15% breaker) before any live execution — research already quantified the benefit (−31.3R → −24.2R maxDD).
3. **If promoted:** implement the reversal trigger + time-stop exits as a book family (side-neutral by construction), with the per-symbol calibrator path already in place.
4. **Keep the discovery layer side-neutral:** the tie-break fix stays; the scanner's cosmetic long-first ordering may be neutralized (sort by |bias| or keep as display-only — no decision impact either way).
5. **Reserve the fresh window:** the 2025-06-01+ period is consumed; do not re-run anything against it.

---

## 15. Reproducibility

- Tie-break fix: `src/features/setups.py` (strict comparison → FLAT on exact tie); regression test `tests/test_setups.py::test_verdict_matches_strict_score_comparison`.
- Audit evidence was generated live: `./venv/bin/python` dry-run long+short passes and per-symbol opportunity books on 2026-08-13 data (commands inline in the audit worklog).
- Full suite: 612 tests (18 setup incl. new invariant; 13 stage9; 11 stage8; rest production) — see validation run.
