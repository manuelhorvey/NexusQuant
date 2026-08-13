# NexusQuant Forensic Audit — Phase 1 (Two-Sided Campaign)

Date: 2026-08-13 · Author: audit pass · Status: findings, no code changes

Independent verification of the repo's two-sided claims (prior artifacts:
`docs/TWO_SIDED_ENGINE_AUDIT.md`, `docs/POST_GAP_CLOSURE_AUDIT.md`). Every
claim below was re-derived from source and re-run on live data, not taken
from the docs.

---

## 1. Headline verdict

The architecture is genuinely **two-sided at the analysis layer** (scanner,
report, plan, opportunity book, dual models) — the prior "PARTIALLY
TWO-SIDED" verdict holds **for analysis**. But the **live alert path is NOT
wired through the decision engine**, and the short live filter contains two
confirmed defects (inverted ML column, direction-agnostic macro gate) that
systematically distort the short side. These are the central Phase-2/4/5
work items.

---

## 2. Verified-against-source inventory

| Artifact | Location | Audit result |
|---|---|---|
| Direction-neutral 12-family classifier | `src/features/setups.py` | Present, used by report/scanner/plan/census |
| Long engine (Buy-the-Dip) | `src/features/dip.py:148` | Hard gate `close > SMA200 AND bias >= 0` |
| Short engine (Sell-the-Rally) | `src/features/rally.py:148` | Hard gate `close < SMA200 AND bias <= 0` |
| Dual models | `src/model/model.py` | `predict_long_short` + `predict_short_series`; `dip_lgbm.joblib` + `rally_lgbm.joblib` exist |
| Opportunity book / EV decision | `src/analysis/opportunity.py:288` | Full LONG/SHORT/FLAT + EV + rejections; **consumed by report + diagnostics only** |
| Live long pass | `src/live/signals.py:485` | Engine filter, NO opportunity-book consultation |
| Live short pass | `src/live/signals.py:599` | Engine filter, NO opportunity-book consultation |
| Live merge | `src/live/run.py:362` | `_merge_long_short` = concat of long+short alerts; no conflict resolution, no FLAT |
| Backtest | `src/backtest/engine.py` + `signals.py` | Side-aware, cost inversion correct |

## 3. Census — headline claim REPRODUCES, but is symbol-dependent

Full default run (12 symbols, 30,246 bars):

    [LONG]  candidates 4,469 · confirmed 214 · win 59.8% · exp 0.766R
    [SHORT] candidates 3,793 · confirmed 216 · win 82.9% · exp 1.135R
    Long/short signal ratio: 0.99

This matches the prior audit's "~1.0 for every 1.0" claim. **However** a
6-symbol subset (EURUSD, GBPUSD, USDJPY, USDCAD, NZDJPY, USDCHF) gave
**2.36** (66 vs 28), and per-symbol mixes swing hard (AUDSEK 30L/83S vs
USDCAD 10L/2S). Conclusion: **equal detection is a universe-level average,
not a per-market property** — the audit must not claim directional neutrality
from the aggregate alone.

## 4. Methodology gap in the census itself

- `src/analysis/census.py:74` calls `classify_setup(window, levels=None,
  divergence=None, pattern=None)` — strips the structural enrichments
  (S/R levels, divergence, patterns) that the production/report path feeds.
- `src/analysis/census.py:208-213` validates a family candidate only when a
  **confirmed dip/rally engine signal** fires on the same bar. The 10
  non-pullback families (BREAKOUT/BREAKDOWN/RETEST/REVERSAL/MEAN_REVERSION)
  are counted as candidates but never outcome-validated.
- Consequence: the census measures the two engines' pullback confirmation,
  not the 12-family classifier's broader opportunity detection. A SHORT
  BREAKDOWN or SHORT_REVERSAL with no rally engine confirmation is invisible
  to the "confirmed signal" metric.

## 5. CONFIRMED BUG — short live filter never uses the short ML model

`src/live/signals.py:340-342`:

```python
if min_ml_prob is not None and "ml_prob" in f:
    has_ml = f["ml_prob"].notna()
    f = f[~has_ml | (f["ml_prob"] <= (100.0 - min_ml_prob))]
```

- It reads the **long** model's `ml_prob` (bullish) column and inverts it.
- The scanner computes `ml_short_prob` via the dedicated short model
  (`src/analysis/scanner.py:258-271`, `predict_short_series`) and emits it
  in the table (scanner.py:356), but `filter_short_signals` never reads it.
- Empirical proof: a symbol with long prob 90 / short prob 70 is **dropped**
  from shorts (90 > 60), even though the short model says short edge is 70%.
  The dedicated `rally_lgbm.joblib` is dead weight in the live path.
- The regression test `test_filter_ml_threshold_inverts_for_shorts`
  (tests/test_short_engine_extras.py:178) **codifies this defect** rather
  than guarding against it.

## 6. CONFIRMED BUG — macro gate is direction-agnostic, suppresses shorts

`src/live/signals.py:343-345` requires `macro_gate == "PASS"` for shorts.
`macro_gate` (`src/macro/overlay.py:526`) = `bias >= -0.5`, where `bias` is
the symbol's macro **tailwind** (positive = tailwind).

Empirical proof: strong-USD backdrop (dxy_score +2) gives EURUSD
`bias -2.0, "Strong Headwind"` → gate **BLOCKED**. For a EURUSD **short**,
a strong dollar is a tailwind — exactly the regime that favors the short —
yet the same gate blocks it. The gate has no direction parameter and is
applied identically to both sides. This is a directional asymmetry in the
live path (not in the analysis layer, where the book checks macro only on
the EV winner).

## 7. CONFIRMED GAP — live path bypasses the decision engine

- `live_signal_pass` / `live_short_pass` call `filter_signals` /
  `filter_short_signals` and never call `build_opportunity_book`.
- `report.py:432-435` builds `report["opportunity_book"]`, surfaced by
  `run.py --format diagnostics` and the report — not by the alert path.
- `run.py:362 _merge_long_short` concatenates long+short alerts with no
  conflict resolution: no EV comparison, no FLAT, no "both sides forming"
  arbitration. (In practice simultaneous long+short for one symbol is
  prevented by the engines' opposing SMA200 gates, but the merge does not
  enforce it and gives no explanation if it ever happens.)
- Phase-4 core work: route live alerts through the opportunity-book verdict.

## 8. Residues from prior "fixed" claims

- `src/features/setups.py:596` still uses the `or` fallthrough
  (`prob_long = ml.get("prob_long") or ml.get("prob")`). Harmless today only
  because report.py:373-382 sets `prob` and `prob_long` to the same value;
  POST_GAP_CLOSURE_AUDIT claimed "explicit None checks at all sites" — not
  true. A legitimate 0.0 prob_long with a nonzero `prob` key would silently
  cross-feed.
- `models/registry.json` contains **106** `/tmp/tmp*.joblib` entries from
  tests — registry pollution, not a correctness issue, but makes
  reproducibility claims fragile.
- Uncommitted working-tree changes touch `opportunity.py`, `report.py`,
  `run.py`, `signals.py`, `risk/run.py` + untracked `tests/test_opportunity.py`
  — the two-sided work is partially uncommitted (commit `c408cb4` is in,
  later edits are not).

## 9. Test suite status

538 tests OK, 2 skipped (`./venv/bin/python -m unittest discover tests`,
~140s). No live-path test asserts mutual exclusion or opportunity-book
routing; `test_filter_ml_threshold_inverts_for_shorts` encodes the inverted-
ML behavior instead of the short model. pytest is not installed.

## 10. Phase-2+ directives derived from this audit

1. **Opportunity-first architecture**: make `build_opportunity_book` the
   single decision entry point consumed by report, live alerts, and API.
2. **Fix short ML wiring**: `filter_short_signals` must filter on the short
   model's `ml_short_prob`, with the inverted-long fallback removed; replace
   the codifying test.
3. **Direction-aware macro gate**: gate short opportunities on the short-
   favorable direction (e.g. `bias <= +0.5`), keeping the long gate as-is.
4. **Census v2**: validate all 12 families' outcomes (not just engine
   confirmations) and include structural enrichments without lookahead.
5. **Census reporting**: report per-market ratio dispersion, not only the
   universe average.
6. **Cleanup**: remove the `or` fallthrough, purge registry /tmp entries,
   commit the pending working tree after review.

## 11. Phase 3 addendum: dual-model training parity (2026-08-13)

Audited the training + serve path of the two calibrated models
(`models/dip_lgbm.joblib` long, `models/rally_lgbm.joblib` short).

**Verified sound:**
- **Feature parity**: both bundles carry the *identical* 54 `FEATURE_COLUMNS`
  (list equality). The rally signal is mapped onto the dip feature slots
  (`_normalize_signal` in `src/model/features.py`) with the direction flag
  flipped, so one feature builder serves both sides.
- **Label parity**: `build_labels_short` / `make_meta_labels_short` mirror the
  long builders with barriers inverted (stop above, target below); `side`
  flows through `build_dataset` to select rally signal + mirrored labels.
- **Meta-label default**: both models trained on the actual engine outcome
  (confirmed setup fill / stop / resistance target), not the generic 1R.
- **Calibration**: both are isotonic-calibrated on pooled OOS folds; serve
  time applies `apply_calibrator`.
- **Serve parity**: `predict_short_series` builds `rally_signal_series` so
  features match training distribution; `predict_long_short` returns
  `{prob_long, prob_short, net_bias}` and never fabricates `net_bias` when a
  side's model is missing.
- **Consumption parity**: `filter_short_signals` gates on `ml_short_prob`;
  the opportunity book's short side reads `sc["prob_short"]`; the macro gate
  uses `gate_short` (all fixed in Phase 2).

**Fix applied (this phase):**
- `report.py` `sc["ev"]` / `sc["pw_rr"]` always read `prob_long` regardless of
  the classifier's `direction` — a short setup's EV was computed from the LONG
  model's probability. The opportunity book's per-side EV was already correct;
  this only misreported the ranking/plan/print surface. Extracted
  `_direction_win_prob` (direction-matching prob, 0.0 valid, explicit None
  fallback) and wired it in. +6 regression tests.

**Residual asymmetry (training-time, not a code defect):**
- Long model `num_leaves=127` (hyperparameter search run), short model
  `num_leaves=31` (defaults — trained without `--search`). Short model also
  has far fewer training samples (n_train 310 vs 853; n_test 686 vs 1606),
  though its OOS AUC (0.5836) slightly beats the long (0.5775). Retraining the
  short model with `--search` when more rally data accumulates is recommended,
  but no pipeline change is required.

## 12. Phase 4 addendum: EV-driven live alert merge (2026-08-13)

Closed the phase-1 gap "live path bypasses the opportunity book": in `--mode
both`, `_merge_long_short` was a bare concatenation (`long["new_alerts"] +
short["new_alerts"]`), so a symbol could in principle fire BOTH a LONG and a
SHORT alert in the same pass (the opposing SMA200 gates structurally prevent
it, but nothing enforced it).

**Fix:** added `merge_pass_alerts` in `src/live/run.py`. When a symbol appears
in both passes, the EV-driven `opportunity_book.verdict.direction` (embedded
in every alert's report via `generate_full_report`) arbitrates: keep the
verdict's side, drop the other; a FLAT verdict drops both (FLAT = no
acceptable opportunity). Symbols that cleared only one side pass through
unchanged. When an alert's report carries no book, it degrades to the prior
concatenation (graceful).

Surfaced in the merged result as `conflicts_resolved` and rendered in the
briefing footer / JSON output. `format_briefing` counts conflicts arbitrated
to FLAT in the no-alert line. +5 regression tests. 558 tests OK, 2 skipped.