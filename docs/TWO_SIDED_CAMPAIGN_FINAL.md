# NexusQuant Two-Sided Campaign — Final Report

Date: 2026-08-13 · Commits: `dbd9148` → `f9fb753` (7 phases)

Transformation of the Buy-the-Dip long-biased system into a directionally
neutral opportunity-discovery engine that emits LONG / SHORT / FLAT from
evidence, EV, risk, and portfolio context — without optimizing for more
trades or mirroring the long logic.

---

## 1. What was delivered, phase by phase

| # | Phase | Commit | What changed |
|---|---|---|---|
| 1 | Forensic audit | `dbd9148` | `docs/FORENSIC_AUDIT_2026-08-13.md`; verified every two-sided claim against source + live data. |
| 2 | Decision-layer hardening | `47b0471` | Fixed short ML wiring + direction-agnostic macro gate; removed evidence-fallback probs; +9 tests. |
| 3 | Dual-model parity | `0550d4a` | Direction-matching prob for setup EV (`_direction_win_prob`); +6 tests. |
| 4 | EV-driven live merge | `1726127` | `merge_pass_alerts` arbitrates dual-side alerts by the opportunity-book verdict; +5 tests. |
| 5 | All-family validation | `38b4e3f` | `_uniform_r` validates outcomes for all 12 families, not just engine-confirmed pullbacks; +2 tests. |
| 6 | Per-market dispersion | `f916976` | Per-symbol ratio + `_ratio_dispersion` (median/std/IQR/most-skewed); +3 tests. |
| 7 | Registry hygiene | `f9fb753` | `prune_tmp_entries`, registered the real dip model, verified the `or`-fallthrough fix; +2 tests. |

Test suite: **565 OK, 2 skipped** (`./venv/bin/python -m unittest discover tests`).

---

## 2. Acceptance criteria

1. **Both sides' opportunities are visible.** The scanner ranks all symbols
   with independent `ml_prob` (long model) and `ml_short_prob` (short model);
   the report prints `P(long)` / `P(short)`; the plan emits
   SELL-LIMIT/BUY-LIMIT/WAIT/NO-SETUP per symbol; the census counts long and
   short candidates across history. ✅
2. **Edge is quantified, not asserted.** EV in R units is computed ONLY from a
   calibrated ML probability AND a real payoff basis (target ladder or
   achieved R:R). No probability, no EV — FLAT with explicit reasons. ✅
3. **Weak opportunities are rejected.** The opportunity book lists every
   rejection reason: no family evidence, no calibrated prob, EV ≤ 0 after
   costs, R:R floor missed, macro blocked. ✅
4. **The existing long edge is preserved.** Long-path filters, sizing, and the
   opportunity-book EV path were untouched except the phase-3 direction-prob
   fix (which only stops SHORT setups from reading the LONG prob). ✅
5. **LONG/SHORT/FLAT is chosen from evidence.** The book's verdict compares
   per-side EV (or engine-confirmed rule path when no calibrated prob), and
   the winning side must clear R:R + macro hard gates or it goes FLAT. ✅
6. **No fake probabilities.** The phase-2 `or`-fallthrough is gone
   (explicit `None` checks, verified: a 0.0 calibrated prob survives); EV is
   never fabricated from evidence scores; `net_bias` is never computed from an
   absent short model. ✅
7. **FLAT = absence of an acceptable opportunity.** The book returns FLAT
   whenever neither side clears min_ev (or both engines are unconfirmed), and
   the phase-4 merge drops both alerts when the verdict is FLAT. ✅
8. **Nothing was optimized to produce more trades or more shorts.** No
   thresholds were lowered, no new signal sources added, no long logic
   mirrored. The short model's lower sample count is reported, not hidden. ✅

---

## 3. Remaining known limitations (transparent, not hidden)

- **Live pass still sizes from the pullback engines**, then arbitrates
  dual-fire via the book verdict — the book does not yet *initiate* alerts for
  non-pullback families (breakout/breakdown/retest are visible in the scanner
  and plan, not alerted).
- **Census structural enrichments** (levels/divergence/patterns) are still
  off in `_classify_history`: causal per-window computation is O(n²)
  (~181ms/window, ~25 min for 12 symbols). Documented, not regressed.
- **Short model data scarcity**: n_train 310 vs long 853; OOS AUC 0.5836 vs
  0.5775. Retrain with `--search` when more rally data accumulates.
- **`models/registry.json` is gitignored** — the cleaned ledger lives locally.

---

## 4. How to reproduce

```bash
./venv/bin/python -m unittest discover tests        # 565 OK, 2 skipped
./venv/bin/python -m src.analysis.census            # 12-symbol census + dispersion
./venv/bin/python -m src.analysis.census --symbols EURUSD,GBPUSD,USDJPY,USDCAD,NZDJPY,USDCHF
./venv/bin/python -m src.live.run --dry-run --mode both   # EV-arbitrated alerts
./venv/bin/python -m src.model.registry --prune-tmp # registry hygiene
```
