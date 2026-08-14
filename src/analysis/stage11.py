"""
NexusQuant - Stage-11: Prospective Alpha Accumulation Campaign.

Stage-10's answer on 2026-08-14 was FLAT x 16: the architecture now
searches the entire opportunity space and lets evidence decide LONG /
SHORT / FLAT. Stage-11 freezes that decision engine (see
``src/analysis/protocol.py``) and answers the one question that matters:

    "When NexusQuant says FLAT today, does the frozen research process
     eventually identify genuinely cost-positive opportunities as new
     data arrives?"

This module is the READ-ONLY evaluator over the accrued window:

* ``accumulate_prospective(records, resolutions)`` - per-family
  accumulation stats computed ONLY from snapshots recorded under the
  current frozen protocol (protocol-mismatched snapshots are excluded) and
  ONLY with the frozen first-touch resolution semantics. It never fits,
  never tunes, never writes history.

* ``threshold_diagnostics(...)`` - the two self-diagnosing views the
  campaign needs:
    - filter effectiveness: among FLAT decisions, how often would BOTH
      candidate sides have lost money? (high share = the filter works)
    - over-restriction evidence: how many rejected candidates would have
      been cost-positive (realized R >= +0.30R after costs)? (a material
      count = the threshold may be too strict - to be evaluated ONLY by
      the preregistered protocol, never by eyeballing the result)

* ``promotion_status(...)`` - each family's current ladder level
  (src/analysis/promotion.py) from its accumulated evidence.

* ``build_stage11_doc(...)`` - writes
  ``docs/STAGE11_PROSPECTIVE_ALPHA_ACCUMULATION.md``.

Discipline (spec #11 "Freeze"): while the window is open the protocol hash
is pinned; recording and resolution are the only writes (to the immutable
JSONL logs); this module changes nothing about how decisions are made.

CLI::

    python -m src.analysis.stage11            # accumulate + print summary
    python -m src.analysis.stage11 --all      # + write the doc
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Dict, List, Optional

sys.path.append(str(Path(__file__).parent.parent.parent))

from src.analysis.opportunity import FAMILY_STATUS
from src.analysis.promotion import ACTIONS, level_from_status, promote
from src.analysis.protocol import matches_protocol, protocol_hash
from src.live.recorder import (
    DEFAULT_RECORD_FILE,
    DEFAULT_RESOLUTION_FILE,
    load_records,
    load_resolutions,
)

# A rejected candidate is "over-restriction evidence" when it would have
# cleared this realized-R bar after costs (documented diagnostic; part of
# the preregistered protocol, not a tuning lever).
OVER_RESTRICTION_R = 0.30

TAKEN_THRESHOLD_R = 0.20  # the frozen +0.20R EV floor, for diagnostics


def _join_resolutions(records: List[Dict], resolutions: List[Dict]) -> List[Dict]:
    """Attach each resolution to its record (by recorded_at; falls back to
    date+symbol). Resolutions without a matching record are dropped -
    they cannot be attributed to a decision-time snapshot."""
    by_key: Dict[str, Dict] = {}
    for rec in resolutions:
        key = (rec.get("recorded_at"),)
        if rec.get("recorded_at") is None:
            key = (rec.get("date"), rec.get("symbol"))
        by_key.setdefault(key, rec)
    out = []
    for rec in records:
        key = (rec.get("recorded_at"),)
        if key not in by_key:
            key = (rec.get("date"), rec.get("symbol"))
        res = by_key.get(key)
        out.append({"record": rec, "resolution": res})
    return out


def accumulate_prospective(
    records: Optional[List[Dict]] = None,
    resolutions: Optional[List[Dict]] = None,
    record_path: str = DEFAULT_RECORD_FILE,
    resolution_path: str = DEFAULT_RESOLUTION_FILE,
) -> Dict:
    """Per-family accumulation stats over the frozen prospective window.

    Only snapshots whose embedded protocol hash equals the CURRENT frozen
    hash are eligible - anything recorded under a different protocol is
    generated under different rules and is excluded (a protocol change is
    a new preregistration, and old observations never silently leak into
    the new protocol's evaluation).

    Returns a stats dict (never writes anything except via the caller's
    explicit doc builder):

      protocol_hash, n_records, n_eligible, n_resolved,
      families: {family: {records, resolved, taken_n, taken_mean_r,
                          flat_n, cf_long_mean_r, cf_short_mean_r,
                          over_restriction_n, level, action}},
      diagnostics: {filter_effectiveness, over_restriction_count,
                    over_restriction_detail}
    """
    if records is None:
        records = load_records(record_path)
    if resolutions is None:
        resolutions = load_resolutions(resolution_path)

    n_total = len(records)
    eligible = [r for r in records if matches_protocol(r)]
    joined = _join_resolutions(eligible, resolutions)
    resolved = [j for j in joined if j["resolution"] is not None]

    fam: Dict[str, Dict] = {}
    for fam_name in sorted(FAMILY_STATUS):
        fam[fam_name] = {
            "records": 0,
            "resolved": 0,
            "taken_n": 0,
            "taken_mean_r": None,
            "flat_n": 0,
            "cf_long_n": 0,
            "cf_long_mean_r": None,
            "cf_short_n": 0,
            "cf_short_mean_r": None,
            "over_restriction_n": 0,
        }

    for j in resolved:
        rec = j["record"]
        res = j["resolution"]
        decision_dir = (rec.get("decision") or {}).get("direction")
        sides = rec.get("sides") or {}

        def _family_of(side: str, sides_: Dict = sides) -> Optional[str]:
            return (sides_.get(side) or {}).get("family")

        # The decision's own family (taken side, or none when FLAT).
        taken_fam = (
            _family_of(decision_dir) if decision_dir in ("long", "short") else None
        )

        cf = res.get("counterfactual") or {}

        if decision_dir in ("long", "short") and taken_fam and fam.get(taken_fam):
            f = fam[taken_fam]
            f["records"] += 1
            f["resolved"] += 1
            f["taken_n"] += 1
            rr = res.get("r")
            vals = f.setdefault("_taken_r", [])
            vals.append(rr)
        # FLAT: both candidate sides are counterfactual - count each side
        # under its own family.
        elif decision_dir == "flat":
            for side in ("long", "short"):
                cfs = cf.get(side)
                fname = _family_of(side)
                if cfs is None or cfs.get("realized_r") is None or not fam.get(fname):
                    continue
                f = fam[fname]
                f["records"] += 1
                f["resolved"] += 1
                f["flat_n"] += 1
                key = f"_cf_{side}_r"
                f.setdefault(key, []).append(cfs["realized_r"])
                if cfs["realized_r"] >= OVER_RESTRICTION_R:
                    f["over_restriction_n"] += 1
        else:
            continue

    # Means + ladder levels. The _n counts are captured BEFORE the lists
    # are popped (taken_n/cf_long_n/cf_short_n must reflect the window).
    for fname, f in fam.items():
        for key, out_key in (
            ("_taken_r", "taken_mean_r"),
            ("_cf_long_r", "cf_long_mean_r"),
            ("_cf_short_r", "cf_short_mean_r"),
        ):
            vals = f.pop(key, None)
            n_key = out_key.replace("_mean_r", "_n")
            f[n_key] = len(vals) if vals else 0
            if vals:
                f[out_key] = round(sum(vals) / len(vals), 4)
        lvl = level_from_status(FAMILY_STATUS.get(fname))
        f["level"] = lvl
        f["action"] = ACTIONS.get(lvl)

    # --- FLAT-decision classification (Stage-11 monitoring campaign) ---
    # For every FLAT decision, the counterfactuals answer: is the system
    # actually good at knowing when NOT to trade?
    #   A. Correct FLAT         - both sides subsequently lose
    #   B. Over-restriction     - a rejected side would have made
    #                             >= +OVER_RESTRICTION_R after costs
    #   C. Correct directional rejection - the side the system ranked
    #                             higher (decision-time allocation EV) is
    #                             the one that won; the other lost
    flat_pairs = 0
    both_negative = 0
    cls = {
        "correct_flat": 0,
        "over_restriction": 0,
        "correct_directional_rejection": 0,
        "ambiguous": 0,
        "unresolved": 0,
    }
    flat_records = [
        j
        for j in resolved
        if (j["record"].get("decision") or {}).get("direction") == "flat"
    ]
    for j in flat_records:
        rec = j["record"]
        cf = (j["resolution"] or {}).get("counterfactual") or {}
        lng = (cf.get("long") or {}).get("realized_r")
        sht = (cf.get("short") or {}).get("realized_r")
        sides = rec.get("sides") or {}
        if lng is None and sht is None:
            cls["unresolved"] += 1
            continue
        flat_pairs += 1
        lng_neg = lng is None or lng < 0
        sht_neg = sht is None or sht < 0
        if lng_neg and sht_neg:
            both_negative += 1
            cls["correct_flat"] += 1
            continue
        best = max(r for r in (lng, sht) if r is not None)
        if best >= OVER_RESTRICTION_R:
            pos = [
                s
                for s, r in (("long", lng), ("short", sht))
                if r is not None and r >= OVER_RESTRICTION_R
            ]
            neg = [
                s for s, r in (("long", lng), ("short", sht)) if r is not None and r < 0
            ]
            if len(pos) == 1 and len(neg) == 1:
                pos_s, neg_s = pos[0], neg[0]
                rank_pos = (sides.get(pos_s) or {}).get("expected_r_alloc")
                rank_neg = (sides.get(neg_s) or {}).get("expected_r_alloc")
                if (
                    rank_pos is not None
                    and rank_neg is not None
                    and rank_pos > rank_neg
                ):
                    cls["correct_directional_rejection"] += 1
                    continue
            cls["over_restriction"] += 1
            continue
        cls["ambiguous"] += 1

    over_restriction = 0
    detail = []
    for fname, f in fam.items():
        if f["over_restriction_n"]:
            over_restriction += f["over_restriction_n"]
            detail.append(f"{fname}: {f['over_restriction_n']}")

    return {
        "protocol_hash": protocol_hash(),
        "n_records_total": n_total,
        "n_eligible": len(eligible),
        "n_excluded_protocol_mismatch": n_total - len(eligible),
        "n_resolved": len(resolved),
        "families": fam,
        "diagnostics": {
            "flat_decisions_with_counterfactuals": flat_pairs,
            "flat_decisions_both_sides_negative": both_negative,
            "filter_effectiveness": (
                round(both_negative / flat_pairs, 4) if flat_pairs else None
            ),
            "over_restriction_count": over_restriction,
            "over_restriction_detail": detail,
            "over_restriction_r": OVER_RESTRICTION_R,
            "flat_classification": cls,
        },
    }


def promotion_status(stats: Dict) -> Dict:
    """Each family's ladder level + promotion gates from accumulated stats.

    Purely diagnostic: reports where each family sits and what the next
    gate is. No level changes here - promotion is decided by the frozen
    protocol's evaluation, never by this readout.
    """
    out = {}
    for fname, f in (stats.get("families") or {}).items():
        promo = promote(
            {
                "status": FAMILY_STATUS.get(fname),
                "prospective_records": f["records"],
                "resolved": f["resolved"],
                "min_effective_n": 50,
                # Evidence fields stay empty until the window is large
                # enough to measure them - a gate that cannot be checked
                # cannot be passed.
                "net_r_after_costs": (f["taken_mean_r"] if f["taken_n"] else None),
            }
        )
        out[fname] = {
            "level": promo["level"],
            "action": promo["action"],
            "reason": promo["reason"],
            "records": f["records"],
            "resolved": f["resolved"],
            "taken_mean_r": f["taken_mean_r"],
        }
    return out


def _md_table(headers: List[str], rows: List[List]) -> str:
    hdr = "| " + " | ".join(headers) + " |"
    sep = "|" + "|".join(["---"] * len(headers)) + "|"
    body = ["| " + " | ".join(str(c) for c in r) + " |" for r in rows]
    return "\n".join([hdr, sep] + body)


def build_stage11_doc(
    stats: Optional[Dict] = None,
    out_path: str = "docs/STAGE11_PROSPECTIVE_ALPHA_ACCUMULATION.md",
    record_path: str = DEFAULT_RECORD_FILE,
    resolution_path: str = DEFAULT_RESOLUTION_FILE,
) -> Path:
    """Write the Stage-11 campaign doc from the accumulated stats.

    The doc is a status readout of the frozen window - what has accrued,
    what each family's ladder level is, what the diagnostics say, and the
    exact protocol that governs any future promotion. It changes nothing.
    """
    if stats is None:
        stats = accumulate_prospective(
            record_path=record_path, resolution_path=resolution_path
        )
    promo = promotion_status(stats)
    diag = stats["diagnostics"]

    cls = diag.get("flat_classification") or {}
    fam_rows = []
    for fname, f in sorted(stats["families"].items()):
        p = promo[fname]
        fam_rows.append(
            [
                fname,
                FAMILY_STATUS.get(fname),
                p["level"],
                p["action"],
                f["records"],
                f["resolved"],
                _fmt(f["taken_mean_r"]),
                _fmt(f["cf_long_mean_r"]),
                _fmt(f["cf_short_mean_r"]),
                f["over_restriction_n"],
            ]
        )

    lines = [
        "# NexusQuant - Stage-11: Prospective Alpha Accumulation Campaign",
        "",
        "**Date:** 2026-08-14 - **Status:** the Stage-10 decision engine is FROZEN and every "
        "new D1 bar is prospective evidence. Nothing is tuned against the accruing "
        "observations; the only writes are immutable decision-time snapshots and their "
        "frozen first-touch resolutions.",
        "",
        "## 1. The question this campaign answers",
        "",
        "> When NexusQuant says FLAT today, does the frozen research process eventually "
        "identify genuinely cost-positive opportunities as new data arrives?",
        "",
        "Stage-10 established the architecture: search the entire opportunity space, let "
        "evidence decide LONG / SHORT / FLAT - and the correct answer on 2026-08-14 was "
        "**FLAT x 16**. That is a successful validation of the decision architecture, not a "
        "failure to trade. Stage-11 freezes that architecture and starts the clock on the "
        "**genuinely unseen window**.",
        "",
        "## 2. What is frozen (preregistered protocol)",
        "",
        f"Protocol version **stage11.1** - hash ``{stats['protocol_hash'][:16]}...``. The "
        "full manifest (universe, features, labels, engines, TP allocation, cost model, "
        "decision rules, portfolio rules, validation gates, exit semantics) lives in "
        "`src/analysis/protocol.py` and is quoted by every snapshot at record time.",
        "",
        "The freeze is enforced mechanically: a snapshot is eligible for evaluation only "
        "when its embedded protocol hash equals the current one "
        "(`matches_protocol`). Changing any threshold, allocation, cost, cap or gate "
        "changes the hash - which makes it a NEW protocol that must be preregistered "
        "separately, and excludes every old snapshot from its evaluation.",
        "",
        "## 3. Window status",
        "",
        "| Metric | Value |",
        "|---|---|",
        f"| Snapshots recorded (all) | {stats['n_records_total']} |",
        f"| Eligible under current frozen protocol | {stats['n_eligible']} |",
        f"| Excluded (protocol mismatch) | {stats['n_excluded_protocol_mismatch']} |",
        f"| Resolved with frozen first-touch semantics | {stats['n_resolved']} |",
        "",
        "The window opened with Stage-11. As new D1 bars accrue, `--record` passes append "
        "snapshots and each subsequent pass resolves them. Nothing below is a conclusion - "
        "it is a readout of an open window.",
        "",
        "## 4. Family accumulation + promotion ladder",
        "",
        "Ladder: L0 UNVALIDATED (FLAT) -> L1 RESEARCH CANDIDATE (shadow) -> L2 PROSPECTIVE "
        "VALIDATION (shadow) -> L3 VALIDATED ALPHA (tiny controlled capital) -> L4 "
        "PRODUCTION CANDIDATE (controlled deployment) -> L5 PRODUCTION (monitored). The L3 "
        "gate battery is the research firewall: min effective N, independent window, "
        "walk-forward, cost robustness, permutation test, bootstrap CI, LOSO, regime "
        "stability, concentration limits, portfolio contribution, drawdown limits "
        "(`src/analysis/promotion.py`).",
        "",
        _md_table(
            [
                "Family",
                "Research status",
                "Ladder level",
                "Action",
                "Records",
                "Resolved",
                "Taken mean R",
                "CF long mean R",
                "CF short mean R",
                "Over-restrict n",
            ],
            fam_rows,
        ),
        "",
        "*Taken mean R* = mean realized R of taken decisions in the window; *CF mean R* = "
        "mean counterfactual realized R of the side had it been taken (FLAT-rejected "
        "candidates included, 1/3-1/3-1/3 allocation, costs deducted). `None` = no "
        "observations yet.",
        "",
        "## 5. Self-diagnosing threshold evidence",
        "",
        "The accumulation window is designed to detect BOTH failure modes of a filter:",
        "",
        "**Filter effectiveness** - among FLAT decisions with resolvable counterfactuals, "
        "how often would BOTH candidate sides have lost money? A high share means the "
        "filter is correctly rejecting losers:",
        "",
        f"- FLAT decisions with counterfactuals: {diag['flat_decisions_with_counterfactuals']}",
        f"- Both candidate sides negative (filter saved the loss): {diag['flat_decisions_both_sides_negative']}",
        f"- Filter effectiveness: **{diag['filter_effectiveness'] if diag['filter_effectiveness'] is not None else 'n/a (no observations)'}**",
        "",
        "**Over-restriction evidence** - how many rejected candidates would have cleared "
        f"+{diag['over_restriction_r']}R realized after costs? A material count means the "
        "threshold may be too strict - a claim that can only be evaluated by the "
        "preregistered protocol on the completed window, never by acting on the interim "
        "readout:",
        "",
        f"- Over-restriction count: **{diag['over_restriction_count']}**",
        f"- Detail: {', '.join(diag['over_restriction_detail']) or 'none'}",
        "",
        "**FLAT-decision classification** - the most important Stage-11 measurement: "
        "is NexusQuant actually good at knowing when NOT to trade? Each FLAT decision is "
        "classified by what its counterfactuals subsequently did:",
        "",
        _md_table(
            ["Class", "Meaning", "Count"],
            [
                [
                    "A. Correct FLAT",
                    "both candidate sides subsequently lost",
                    cls.get("correct_flat", 0),
                ],
                [
                    "B. Over-restriction",
                    "a rejected side would have made >= +0.30R after costs",
                    cls.get("over_restriction", 0),
                ],
                [
                    "C. Correct directional rejection",
                    "the decision-time higher-ranked side won; the other lost",
                    cls.get("correct_directional_rejection", 0),
                ],
                [
                    "ambiguous",
                    "positive but below the over-restriction bar",
                    cls.get("ambiguous", 0),
                ],
                ["unresolved", "no forward bars to resolve", cls.get("unresolved", 0)],
            ],
        ),
        "",
        "A preponderance of A means the gates are correctly keeping you out of losers. A "
        "growing B count means the system is too conservative (evaluate ONLY by the "
        "preregistered protocol on the completed window). C means the opportunity-ranking "
        "mechanism is correctly identifying the superior side even when it declines to "
        "trade. **The measurement itself is the deliverable - do not act on interim "
        "counts.**",
        "",
        "## 6. Promotion discipline",
        "",
        "- **Nothing is promoted on interim observations.** A level change requires the "
        "frozen protocol's evaluation of the COMPLETED window (min effective N, independent "
        "window, and the full L3 battery).",
        "- **LONG_REVERSAL stays L1 RESEARCH CANDIDATE (shadow)** until the fresh-window "
        "gate closes with genuinely new observations - it is never silently promoted to "
        "production.",
        "- **SHORT_REVERSAL stays L0/FALSIFIED (hard FLAT)** - Stage-6 falsification is not "
        "revisited by the window; only a new preregistered hypothesis can be.",
        "- **A protocol change is a new campaign**, not an amendment: new hash, old "
        "snapshots excluded, and the new protocol must itself be frozen before it accrues.",
        "",
        "## 7. How to run",
        "",
        "```bash",
        "./venv/bin/python -m src.live.run --format diagnostics --record   # record every decision (incl. FLAT) each pass",
        "./venv/bin/python -m src.live.run --format institutional --record # richer snapshots (mtf)",
        "./venv/bin/python -m src.analysis.stage11                         # accumulate + readout",
        "./venv/bin/python -m src.analysis.stage11 --all                   # + rewrite this doc",
        "```",
        "",
        "Snapshots: `data/validation/prospective_records.jsonl` (immutable, append-only). "
        "Resolutions: `data/validation/prospective_resolutions.jsonl` (separate file - "
        "snapshots never mutate).",
        "",
        "## 8. What a completed window looks like",
        "",
        "When enough untouched bars have accrued (Stage-9 protocol: single-shot evaluation "
        "of the frozen strategy on the fresh window, no tuning), the readout becomes "
        "decidable: either a family clears the L3 battery and moves to tiny controlled "
        "capital, or it fails a gate and stays put / is re-classified. **The correct output "
        "may remain FLAT for a long time - that is a feature, not a bug.** A systematic "
        "trading system has no psychological requirement to trade every day.",
    ]

    p = Path(out_path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return p


def _fmt(x) -> str:
    if x is None:
        return "-"
    if isinstance(x, float):
        return f"{x:+.3f}"
    return str(x)


def main(argv: Optional[List[str]] = None) -> int:
    import argparse

    parser = argparse.ArgumentParser(
        description="Stage-11 prospective accumulation readout"
    )
    parser.add_argument(
        "--all",
        action="store_true",
        help="accumulate, print the summary and rewrite the campaign doc",
    )
    args = parser.parse_args(argv)

    stats = accumulate_prospective()
    print(
        f"protocol {stats['protocol_hash'][:16]}... | records {stats['n_records_total']} "
        f"(eligible {stats['n_eligible']}, mismatch {stats['n_excluded_protocol_mismatch']}) "
        f"| resolved {stats['n_resolved']}"
    )
    fam = stats["families"]
    for fname in sorted(fam):
        f = fam[fname]
        lvl = f["level"]
        print(
            f"  {fname:26s} {FAMILY_STATUS.get(fname):22s} {lvl:24s} "
            f"records={f['records']:3d} resolved={f['resolved']:3d} "
            f"taken={_fmt(f['taken_mean_r'])}"
        )
    d = stats["diagnostics"]
    print(
        f"filter effectiveness: {d['filter_effectiveness'] if d['filter_effectiveness'] is not None else 'n/a'} "
        f"({d['flat_decisions_both_sides_negative']}/{d['flat_decisions_with_counterfactuals']}) | "
        f"over-restriction: {d['over_restriction_count']}"
    )
    if args.all:
        p = build_stage11_doc(stats)
        print(f"Wrote {p}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
