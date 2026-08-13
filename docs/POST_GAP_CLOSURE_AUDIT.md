# Post-Gap-Closure Audit (independent second pass)

> Performed after the documented gaps were reconciled. This audit pretended
> the original gap document did not exist and reviewed the codebase as a
> production system. Every issue below was either fixed in this round or is
> explicitly documented as residual risk.

**Audit date:** 2026-08-12 · **Full suite:** 465 tests OK (2 data-dependent
skips) after this round's 24 new tests.

---

## P0 — CRITICAL (must fix before production)

**None remaining.** All P0-class findings from the reconciliation pass were
fixed and regression-tested this round:

| Finding | Fixed as |
|---------|----------|
| Backtest engine could not simulate shorts at all (`rally_signal_series` orphaned) | `run_backtest(side=...)` + `run_backtest_both` + `--side` CLI + 7 engine tests |
| Short risk plan sized Kelly with the **long** model probability | uses `ml_short` |
| `portfolio_report` crashed with `KeyError: n_setups` on the no-data path (found by running the real CLI) | complete early-return shape + CLI guard + 2 regression tests |

## P1 — HIGH (fixed this round)

| Finding | Fix |
|---------|-----|
| `predict_long_short` (prob_long/prob_short/net_bias) documented but **absent from code** | implemented + tested |
| Rating could never reach Sell/Strong Sell (short model ignored) | `net_bias` blend wired into `factor_contributions` + tested (strong short read ⇒ < 50%) |
| HMM regime label never surfaced in the MTF summary/report | `use_hmm` threading through `multi_timeframe_regime` + `get_current_regime_summary_mtf` + report opt-in |
| Report stress hardcoded `direction="long"` (short book unstressable) | direction/risk_key params + report section 18b + direction-aware loss sign |
| `load_model` lru_cache keyed on path only ⇒ retrained model invisible to long-lived processes until restart | cache keyed on file mtime |

## P2 — MEDIUM (documented decisions / external dependencies)

1. **No saved short model** (`models/rally_lgbm.joblib` absent) — short ML
   probability degrades to the rule engine. Unblocked: `python -m
   src.model.run --group full_fx --side short`.
2. **Yahoo fundamentals are auth-blocked** — `data_provider.py` returns
   `None` fields gracefully; Value/Quality need the local CSV until a keyed
   source is provided. Documented in the module + README.
3. **Social sentiment (Twitter/X, StockTwits) requires API keys** — scored
   as gracefully unavailable; aggregator is ready for a keyed provider.
4. **`both` backtest = independent capital per book** — a research view of
   combined edge, not a shared-margin portfolio sim (documented in
   `run_backtest_both`).
5. **Sentiment intentionally excluded from the macro gate** — news
   timestamps are too coarse for a causal gate; rating already carries a
   Sentiment factor. Deliberate deviation from plan item 4.4, documented.
6. **Factor scores are not ML features** — the pooled model trains on FX
   (no fundamentals); sparse equity-only features would hurt it. Documented
   deviation from plan item 3.4.
7. **Alert dedup state (`data/live/alerts.json`) is a plain JSON file** —
   read-modify-write is not atomic under concurrent passes. Acceptable for
   cron (single instance) + `--watch`; a lock or atomic rename is the
   hardening item if multi-process scheduling is introduced.

## P3 — LOW (nice-to-have, scheduled)

- Combined long+short **portfolio** stress CLI (`stress_portfolio` exists
  as a function but no CLI wires both books through it).
- API `/symbol/{s}/stress` is long-only (per-symbol CLI has `--include-short`
  at portfolio level; the REST endpoint could take `?side=`).
- HMM is opt-in for report speed; a precomputed cache would let it default
  on.
- No structured model-drift metrics (OOS AUC over time is logged in the
  training run, not tracked).
- `equity_universe` scanner `--fetch-news` warm-up is a manual step.

## QUANT FORENSIC TRACE (data → … → reporting)

Traced end-to-end for both sides. Verified no-lookahead / leakage guards:

| Stage | Finding |
|-------|---------|
| Data → features | indicators causal by construction (rolling windows end at bar); swings lagged by their confirmation window (`_causal_swings`, tested) |
| Features → model | chronological train/val/test with embargo + early stopping on a chronological tail; features served at predict time with the SAME context (macro 1-day lag, H4 same-day closes, COT weekly + lag) |
| Model → signal | isotonic calibration fit on OOS only; `ml_short` uses the rally signal series so features match training |
| Signal → filters | R:R floor evaluated on the ladder (2.5:1) for both sides; macro gate uses prior-day state |
| Filters → risk | Kelly p side-correct (fixed this round); sizing risk-per-unit `abs(entry-stop)`; VaR parametric |
| Risk → position | direction-aware sizing + portfolio heat/correlation gates |
| Backtest | market entry never on the signal bar; stop checked before target; slippage direction-correct per side; `both` books documented as separate capital |
| Stress | direction-aware sign; historical realizations windowed (COVID = Mar–Jun 2020 only, tested) |
| Reporting | decisions printable end-to-end (`--format institutional`), freshness gating in the live pass |

**Residual quant risks (accepted, documented):**
- Backtest costs are slippage-only (no spread/swap/commission model).
- `both` mode double-deploys capital by design.
- Stress VaR is parametric (ATR × vol-mult), not a historical replay.
- Sentiment/news and COT caching degrade gracefully but are stale-day
  granularity by nature.

## VALIDATION EVIDENCE (this round)

- **Tests before:** 441 OK (2 skips). **After:** 465 OK (2 skips) — 24 new
  (short engine mechanics, both-merge, predict_long_short, rating net_bias,
  HMM MTF, direction-aware stress, factor model + provider-key aliases,
  institutional format, portfolio early-return regression).
- **End-to-end:** `--format institutional` printed all 18 sections for
  EURUSD; `backtest --side short/both` ran on USDJPY (5 / 15 trades);
  `risk --portfolio --include-short` printed LONG+SHORT positions with
  correlation gates; the previously-crashing no-group portfolio path now
  prints a graceful reason.
- **Adversarial:** the portfolio KeyError was found by executing the CLI on
  real data, not by reading code — the pattern (run every CLI path) caught
  a bug the suite missed and is now covered by a regression test.

## PRODUCTION READINESS VERDICT

### PRODUCTION READY WITH CONDITIONS

The research/signal system is production-shaped for **research and paper
use**:

1. Train the short-side model (`--side short`) to enable short ML
   probabilities (rules-only today).
2. Provide fundamentals via the local CSV (or a keyed vendor) for
   Value/Quality factors.
3. Live execution / broker bridge remains explicitly out of scope — this is
   a signal + risk system, not an order router. No external execution path
   exists, so no execution-safety exposure.

Everything else in the audit's P0/P1 set is closed with regression
coverage, and the full suite is green.
