"""
NexusQuant - Frozen Prospective Validation Recorder (Stage-10 spec #14,
Stage-11 counterfactual resolution).

Stage-9 could not promote the LONG reversal because the dataset ended at
2026-08-13: the fresh-window gate is UNRESOLVED and the frozen strategy is
SHADOW-ONLY. To close that gate with genuinely NEW observations - instead
of re-running the same history - every decision must be recorded at
decision time as an immutable snapshot, and every snapshot must later be
resolved against what actually happened, WITHOUT tuning on it.

This module is that recorder:

* ``record_decision(snapshot)`` - appends one immutable JSONL record to
  ``data/validation/prospective_records.jsonl``. Each record is frozen at
  decision time (all fields below) and never modified afterwards.
* ``load_records()`` - read back all snapshots (order-preserving).
* ``resolve_record(record, df)`` - given the NEW bars after the decision
  date, resolve the outcome with the SAME causal first-touch semantics the
  research uses (stop-first over a bounded hold), writing the outcome as a
  separate ``resolution`` file so the snapshot itself stays immutable.
  Every resolution also carries a ``counterfactual`` block resolving BOTH
  candidate sides (Stage-11): what MFE/MAE, TP1/TP2/TP3 hits, SL hit,
  time exit and allocation-weighted realized R each side would have
  produced had it been taken - FLAT-rejected candidates included. That is
  what makes the accumulation window self-diagnosing: if most FLAT
  decisions would have lost money the filter is working; if rejected
  candidates would have been cost-positive the threshold is too strict.
* ``resolve_counterfactual(record, df, side)`` - the per-side full-ladder
  counterfactual resolution (immutable record untouched).
* ``snapshot_from_report(report)`` - build the decision-time snapshot from
  a full institutional report (features, regime, candidate sides, family,
  entry/stop/TPs, probabilities, target probabilities, EV, cost
  assumptions, portfolio state, decision, counterfactual side). Each
  snapshot embeds the Stage-11 frozen protocol hash so evaluation only
  ever pools snapshots recorded under the SAME frozen rules.

Key discipline (spec #14):

  - IMMUTABLE: snapshots are append-only; resolution is stored apart.
  - DECISION-TIME ONLY: every field is available at decision time - no
    future bar is read while recording (the module is a pure function of
    the report + an explicit portfolio-state dict).
  - NO TUNING: nothing in this module adjusts thresholds or retrains; it
    only records and resolves. Evaluating the accruing observations is a
    read-only analysis (the audit does it, never the recorder).
  - FROZEN PROTOCOL: outcome resolution mirrors the Stage-9 frozen exits
    (first-touch, stop-first, bounded horizon) so the new observations
    speak to the SAME hypothesis that was frozen.

The audit wires this into the live pass (``record_live_snapshot`` helper)
so every live decision accrues evidence for the eventual fresh-window test.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional

import pandas as pd

from src.analysis.opportunity import ALLOCATION
from src.analysis.protocol import RESOLVE_HORIZON, protocol_hash

DEFAULT_RECORD_FILE = "data/validation/prospective_records.jsonl"
DEFAULT_RESOLUTION_FILE = "data/validation/prospective_resolutions.jsonl"


def _now_iso() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def snapshot_from_report(report: Dict, portfolio_state: Optional[Dict] = None) -> Dict:
    """Build an immutable decision-time snapshot from a full report.

    Captures everything the Stage-10 audit table requires: the data
    timestamp, the features the decision saw, the regime, BOTH candidate
    sides with their families / probabilities / EV / levels, the decision
    (book verdict) and the counterfactual opposite side.
    """
    ob = report.get("opportunity_book") or {}
    vd = ob.get("verdict") or {}
    sc = report.get("setup_classification") or {}
    tgt = report.get("targets") or {}
    stgt = report.get("short_targets") or {}

    def _ladder(ldr: Dict) -> List[Dict]:
        return [
            {
                "tp": t.get("target"),
                "price": t.get("price"),
                "rr": t.get("rr"),
                "source": t.get("source"),
            }
            for t in (ldr.get("targets") or [])[:3]
        ]

    def _side(side: str) -> Dict:
        opp = ob.get(side) or {}
        return {
            "family": opp.get("setup_family"),
            "score": opp.get("family_score"),
            "probability": opp.get("probability"),
            "expected_r": opp.get("expected_r"),
            "ev_target_level": opp.get("ev_target_level"),
            "expected_r_alloc": opp.get("expected_r_alloc"),
            "cost_r": opp.get("cost_r"),
            "cost_break_even": opp.get("cost_break_even"),
            "validation_status": opp.get("validation_status"),
            "rr": opp.get("rr"),
            "entry": (opp.get("entry_zone") or [None, None])[0],
            "stop": opp.get("invalidation"),
            "target": opp.get("target"),
            "entry_type": opp.get("entry_type"),
            "taken": opp.get("taken"),
            "rejected": opp.get("rejection_reasons") or [],
        }

    return {
        "recorded_at": _now_iso(),
        "protocol": {"version": "stage11.1", "sha256": protocol_hash()},
        "date": report.get("last_date"),
        "symbol": report.get("symbol"),
        "close": report.get("last_close"),
        "regime": (report.get("regime") or {}).get("regime"),
        "features": {
            "rsi_14": (report.get("momentum") or {}).get("rsi_14"),
            "macd_hist": (report.get("momentum") or {}).get("macd_hist"),
            "adx": (report.get("trend_strength") or {}).get("adx"),
            "atr_14": (report.get("volatility") or {}).get("atr_14"),
            "close_vs_sma200": (report.get("moving_averages") or {}).get(
                "price_vs_sma200"
            ),
        },
        "classifier": {
            "direction": sc.get("direction"),
            "family": sc.get("setup_family"),
            "long_score": sc.get("long_score"),
            "short_score": sc.get("short_score"),
            "prob_long": sc.get("prob_long"),
            "prob_short": sc.get("prob_short"),
        },
        "ladders": {"long": _ladder(tgt), "short": _ladder(stgt)},
        "sides": {"long": _side("long"), "short": _side("short")},
        "target_probs": None,  # filled by the caller when the census table is loaded
        "portfolio": portfolio_state or {},
        "decision": {
            "direction": vd.get("direction"),
            "status": vd.get("status"),
            "expected_r": vd.get("expected_r"),
            "validation_status": vd.get("validation_status"),
            "shadow_only": vd.get("shadow_only"),
            "reason": vd.get("reason"),
        },
        "counterfactual_side": ("short" if vd.get("direction") == "long" else "long"),
        "resolution": None,  # resolved separately - snapshots stay immutable
    }


def record_decision(
    snapshot: Dict,
    path: str = DEFAULT_RECORD_FILE,
    overwrite: bool = False,
) -> Path:
    """Append one immutable decision-time snapshot to the JSONL log.

    Append-only by default (``overwrite=False``): existing snapshots are
    never touched, so a re-run of the live pass accumulates evidence rather
    than mutating history. ``overwrite=True`` truncates first (for tests
    and deliberate re-baselines only - the production recorder never uses
    it).
    """
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    if overwrite:
        p.write_text("")
    with open(p, "a", encoding="utf-8") as fh:
        fh.write(json.dumps(snapshot, default=str) + "\n")
    return p


def load_records(path: str = DEFAULT_RECORD_FILE) -> List[Dict]:
    """All snapshots in write order (missing/corrupt tail tolerated)."""
    p = Path(path)
    if not p.exists():
        return []
    out = []
    for line in p.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            out.append(json.loads(line))
        except json.JSONDecodeError:
            continue
    return out


def record_resolution(
    resolution: Dict,
    path: str = DEFAULT_RESOLUTION_FILE,
    overwrite: bool = False,
) -> Path:
    """Append a resolution record (stored separately so snapshots stay
    immutable - a snapshot's ``resolution`` key is always None in the log)."""
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    if overwrite:
        p.write_text("")
    with open(p, "a", encoding="utf-8") as fh:
        fh.write(json.dumps(resolution, default=str) + "\n")
    return p


def load_resolutions(path: str = DEFAULT_RESOLUTION_FILE) -> List[Dict]:
    p = Path(path)
    if not p.exists():
        return []
    out = []
    for line in p.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            out.append(json.loads(line))
        except json.JSONDecodeError:
            continue
    return out


def _ladder_spec(record: Dict, side: str) -> tuple:
    """(entry, stop, [rung prices], [rung rr], cost) for a candidate side.

    Reads the full TP ladder from the snapshot's ladders section; falls
    back to the side's single primary target when the ladder is absent
    (older / minimal snapshots). Returns ``(None,)*6`` when the side has
    no concrete levels.
    """
    opp = (record.get("sides") or {}).get(side) or {}
    entry = opp.get("entry")
    stop = opp.get("stop")
    if entry is None or stop is None:
        return (None,) * 6
    e, s = float(entry), float(stop)
    if abs(e - s) <= 0:
        return (None,) * 6
    ldr = (record.get("ladders") or {}).get(side) or []
    rungs_px = []
    rungs_rr = []
    for t in ldr[:3]:
        px = t.get("price")
        rr = t.get("rr")
        if px is not None and rr is not None:
            rungs_px.append(float(px))
            rungs_rr.append(float(rr))
    if not rungs_px:
        tgt = opp.get("target")
        if tgt is None:
            return (None,) * 6
        rungs_px = [float(tgt)]
        rungs_rr = [abs(float(tgt) - e) / abs(e - s)]
    cost = float(opp.get("cost_r") or 0.0)
    entry_type = opp.get("entry_type") or "limit"
    return e, s, rungs_px, rungs_rr, cost, entry_type


def resolve_counterfactual(record: Dict, df: pd.DataFrame, side: str) -> Optional[Dict]:
    """Resolve ONE candidate side against the bars after the decision date
    as if it had been taken - the Stage-11 counterfactual.

    Frozen semantics (identical to the taken-side resolver): first-touch,
    stop-first within a bar, bounded ``RESOLVE_HORIZON`` hold. The full
    ladder is priced with the frozen 1/3-1/3-1/3 allocation: each rung's
    fraction exits at its own R:R when touched; the fraction still open
    when the stop is touched loses 1R; a time exit costs nothing for the
    remainder. ``realized_r`` is net of the side's cost assumption.

    MFE/MAE are tracked in R units over the whole horizon (including the
    stop bar). Returns None when the side has no concrete levels or no
    forward bars (nothing to resolve). The immutable snapshot is never
    modified.
    """
    spec = _ladder_spec(record, side)
    if spec[0] is None:
        return None
    e, s, rungs_px, rungs_rr, cost, entry_type = spec
    risk = abs(e - s)
    # The frozen 1/3-1/3-1/3 allocation applies to the configured ladder;
    # a fallback single-rung ladder carries the whole position (1.0).
    base = ALLOCATION[: len(rungs_px)] or (1.0,)
    total = sum(base)
    alloc = tuple(f / total for f in base)

    decision_date = str(record.get("date") or "")
    if not decision_date:
        return None
    try:
        idx = df.index.searchsorted(pd.Timestamp(decision_date), side="right")
    except Exception:
        return None
    horizon = df.iloc[idx : idx + RESOLVE_HORIZON]
    if horizon.empty:
        return None

    long = side == "long"
    bars = list(horizon.iterrows())

    # Fill / non-fill semantics: a MARKET entry fills at the decision
    # close (the report re-expresses it in ``entry``) and the position is
    # open from the first bar. A LIMIT entry fills only when the market
    # trades through its zone within the horizon; if the zone is never
    # touched the order never fills - non-fill, 0R, and NO cost (a trade
    # that never happened costs nothing).
    if entry_type == "market":
        start = 0
        filled = True
    else:
        start = None
        for k, (_, bar) in enumerate(bars):
            hi, lo = float(bar["high"]), float(bar["low"])
            zone_touched = lo <= e if long else hi >= e
            if zone_touched:
                start = k
                break
        filled = start is not None
        if not filled:
            return {
                "side": side,
                "family": ((record.get("sides") or {}).get(side) or {}).get("family"),
                "entry": e,
                "stop": s,
                "rungs_rr": rungs_rr,
                "filled": False,
                "outcome": "non_fill",
                "mfe_r": 0.0,
                "mae_r": 0.0,
                "tp1_hit": False,
                "tp2_hit": False,
                "tp3_hit": False,
                "sl_hit": False,
                "time_exit": False,
                "hit_order": [],
                "realized_r": 0.0,
            }

    hit = [False] * len(rungs_px)
    hit_order = []
    mfe_r = 0.0
    mae_r = 0.0
    sl_hit = False
    for _, bar in bars[start:]:
        hi, lo = float(bar["high"]), float(bar["low"])
        fav = (hi - e) / risk if long else (e - lo) / risk
        adv = (e - lo) / risk if long else (hi - e) / risk
        mfe_r = max(mfe_r, fav)
        mae_r = max(mae_r, adv)
        stop_touched = lo <= s if long else hi >= s
        # Stop-first within a bar: a stop touch stops the open remainder
        # and wins against rungs touched in the SAME bar.
        if stop_touched:
            sl_hit = True
            break
        for i, (px, _rr) in enumerate(zip(rungs_px, rungs_rr, strict=True)):
            if hit[i]:
                continue
            touched = hi >= px if long else lo <= px
            if touched:
                hit[i] = True
                hit_order.append(i + 1)

    time_exit = (not sl_hit) and not all(hit)
    exited = sum(a for i, a in enumerate(alloc) if hit[i])
    payoff = sum(
        a * rr for i, (a, rr) in enumerate(zip(alloc, rungs_rr, strict=True)) if hit[i]
    )
    if sl_hit:
        payoff -= 1.0 - exited  # open remainder stops at -1R
    realized = payoff - cost

    return {
        "side": side,
        "family": ((record.get("sides") or {}).get(side) or {}).get("family"),
        "entry": e,
        "stop": s,
        "rungs_rr": rungs_rr,
        "filled": True,
        "outcome": "sl" if sl_hit else ("tp" if any(hit) else "time"),
        "mfe_r": round(mfe_r, 4),
        "mae_r": round(mae_r, 4),
        "tp1_hit": bool(hit[0]) if len(hit) > 0 else False,
        "tp2_hit": bool(hit[1]) if len(hit) > 1 else False,
        "tp3_hit": bool(hit[2]) if len(hit) > 2 else False,
        "sl_hit": sl_hit,
        "time_exit": time_exit,
        "hit_order": hit_order,
        "realized_r": round(realized, 4),
    }


def resolve_record(record: Dict, df: pd.DataFrame) -> Optional[Dict]:
    """Resolve one snapshot against the bars AFTER its decision date.

    Uses the frozen first-touch semantics (stop-first, bounded hold): for
    the TAKEN side, if the stop is touched before the target within
    ``RESOLVE_HORIZON`` bars the outcome is -1R (or the allocation-aware
    partial-exit loss), if the target is touched first it is +R, else the
    trade is a time exit at ~0. A FLAT record resolves to 0 (the correct
    counterfactual: doing nothing costs nothing).

    Stage-11 addition: every resolution carries a ``counterfactual`` block
    with the full-ladder resolution of BOTH candidate sides (``long`` /
    ``short``) - including for FLAT records, whose counterfactuals are the
    whole point of the accumulation window. The immutable snapshot is
    never modified; the resolution echoes ``recorded_at`` and the protocol
    hash so the evaluator can join and eligibility-filter precisely.

    Returns a resolution dict or None when the snapshot's taken side has
    no concrete entry/stop/target or no forward bars.
    """
    decision = record.get("decision") or {}
    direction = decision.get("direction")
    if direction in (None, "flat"):
        return {
            "date": record.get("date"),
            "symbol": record.get("symbol"),
            "recorded_at": record.get("recorded_at"),
            "protocol": (record.get("protocol") or {}).get("sha256"),
            "direction": direction or "flat",
            "outcome": "flat",
            "r": 0.0,
            "counterfactual": {
                sd: resolve_counterfactual(record, df, sd) for sd in ("long", "short")
            },
            "resolved_at": _now_iso(),
        }
    side = record.get("sides") or {}
    opp = side.get(direction) or {}
    entry = opp.get("entry")
    stop = opp.get("stop")
    target = opp.get("target")
    entry_type = opp.get("entry_type")
    if entry is None or stop is None or target is None:
        return None

    decision_date = str(record.get("date") or "")
    if not decision_date:
        return None
    try:
        idx = df.index.searchsorted(pd.Timestamp(decision_date), side="right")
    except Exception:
        return None
    horizon = df.iloc[idx : idx + RESOLVE_HORIZON]
    if horizon.empty:
        return None

    # The taken side is priced with the SAME ladder allocation as the
    # counterfactuals (Stage-11): the realized R is the allocation-weighted
    # payoff of the actual ladder, not the single-target fiction. The
    # outcome label follows the frozen first-touch walk (stop-first).
    cf = resolve_counterfactual(record, df, direction)
    if cf is None:
        return None
    outcome = cf.get("outcome", "time")  # non_fill / sl / tp / time
    return {
        "date": record.get("date"),
        "symbol": record.get("symbol"),
        "recorded_at": record.get("recorded_at"),
        "protocol": (record.get("protocol") or {}).get("sha256"),
        "direction": direction,
        "entry_type": entry_type,
        "outcome": outcome,
        "r": cf["realized_r"],
        "counterfactual": {
            sd: resolve_counterfactual(record, df, sd) for sd in ("long", "short")
        },
        "resolved_at": _now_iso(),
    }


def record_live_snapshot(
    report: Dict,
    portfolio_state: Optional[Dict] = None,
    path: str = DEFAULT_RECORD_FILE,
    overwrite: bool = False,
    tp_probs: Optional[Dict] = None,
) -> Optional[Dict]:
    """Build + persist a live decision snapshot in one call.

    Used by the live pass (Stage-10 spec #14): every decision accrues an
    immutable snapshot so the fresh-window gate can eventually be closed
    with genuinely new observations. Returns the snapshot (None if the
    report has no opportunity book - nothing to record).
    """
    if not (report.get("opportunity_book") or {}).get("verdict"):
        return None
    snap = snapshot_from_report(report, portfolio_state=portfolio_state)
    if tp_probs:
        snap["target_probs"] = tp_probs
    record_decision(snap, path=path, overwrite=overwrite)
    return snap


def prune_records(path: str = DEFAULT_RECORD_FILE) -> int:
    """Remove the record log (destructive - used by tests only)."""
    p = Path(path)
    if p.exists():
        p.unlink()
    return 0
