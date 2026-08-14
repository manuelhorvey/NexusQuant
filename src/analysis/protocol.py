"""
NexusQuant - Frozen Protocol Manifest (Stage-11 prospective accumulation).

Stage-10 proved the correct answer on 2026-08-14 was FLAT x 16: the
architecture now searches the whole opportunity space and lets evidence
select LONG / SHORT / FLAT. Stage-11 freezes that decision engine and turns
every new D1 bar into *prospective evidence* - observations the frozen
system has never seen and must never be tuned to.

This module is the PREREGISTRATION record. It pins every decision
parameter so that:

* a snapshot recorded today can be evaluated against the SAME protocol
  months from now (the snapshot carries this module's ``protocol_hash``);
* any parameter change is visible as a NEW protocol hash - old snapshots
  are excluded from the new protocol's evaluation instead of silently
  being re-interpreted under different rules;
* the frozen protocol is a single object the audit can quote, and the
  researcher cannot accidentally tune "just one threshold" while the
  accumulation window is open.

What is frozen (spec #11 "Freeze"):

  universe, features, labels, opportunity families, thresholds,
  calibration, entry rules, exits, TP allocation, cost model, portfolio
  constraints, EV formula, LONG/SHORT/FLAT arbitration, validation gates,
  and the resolution horizon.

The values are imported from the modules where they actually live (single
source of truth - the manifest pins them, it does not duplicate logic) or
preregistered here with their origin documented. Changing any pinned value
changes the hash, which is exactly the point.
"""

from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from typing import Any, Dict

# --- Decision parameters imported from their single source of truth --------
from src.analysis.opportunity import (  # noqa: E402
    ALLOCATION,
    DEFAULT_COST_R,
    DEFAULT_MIN_EV_R,
    DEFAULT_MIN_ML_PROB,
    DEFAULT_MIN_RR,
    FAMILY_STATUS,
)
from src.live.portfolio import (  # noqa: E402
    CURRENCY_CLUSTERS,
    DEFAULT_MAX_CONCURRENT,
    DEFAULT_MAX_HEAT_PCT,
    DEFAULT_MAX_PER_CLUSTER,
)

PROTOCOL_VERSION = "stage11.1"

# Frozen resolution horizon (bars) for prospective outcomes - mirrors the
# Stage-9/6 bounded hold. Kept in the manifest (not imported from the
# recorder) so the recorder can reference the manifest without a cycle.
RESOLVE_HORIZON = 20

# Frozen exit semantics: first-touch, stop-first, bounded hold.
EXIT_SEMANTICS = {
    "first_touch": True,
    "stop_first": True,  # stop and target in the same bar -> stop wins
    "horizon_bars": RESOLVE_HORIZON,
    "time_exit_r": 0.0,  # remainder exits at ~0 on the horizon
}

# Frozen partial-exit allocation (imported: 1/3-1/3-1/3 at TP1/TP2/TP3).
TP_ALLOCATION = {"fractions": list(ALLOCATION), "rungs_rr": [1.0, 2.0, 3.0]}

# Frozen cost model.
COST_MODEL = {
    "components": ["spread", "slippage_x2"],  # round trip
    "default_r": DEFAULT_COST_R,  # 0.05R when no settings-derived cost
    "never_zero": True,  # a missing cost is never silently 0
    "break_even_reported": True,  # every order reports its break-even cost
}

# Frozen decision variables and arbitration.
DECISION_RULES = {
    "primary_metric": "allocation_weighted_target_level_ev",  # Stage-10 fix
    "ranking_ev_role": "comparison_only",  # demoted, never the decision
    "min_ev_r": DEFAULT_MIN_EV_R,  # +0.20R floor on the primary metric
    "min_rr": DEFAULT_MIN_RR,  # 2.5:1 ladder floor
    "min_ml_prob": DEFAULT_MIN_ML_PROB,  # calibrated-probability floor
    "arbitration": "strict_gt_over_long_short_flat",
    "flat_is_decision": True,  # FLAT = statistical decision, not a code path
}

# Frozen portfolio constraints (imported defaults).
PORTFOLIO_RULES = {
    "max_concurrent": DEFAULT_MAX_CONCURRENT,
    "max_per_currency_cluster": DEFAULT_MAX_PER_CLUSTER,
    "max_heat_pct": DEFAULT_MAX_HEAT_PCT,
    "currency_clusters": {c: sorted(m) for c, m in CURRENCY_CLUSTERS.items()},
}

# Frozen validation gates (Stage-10 spec #13): only a PRODUCTION-VALIDATED
# family may carry a production order; FALSIFIED is hard-rejected; anything
# PROMISING or UNVALIDATED is SHADOW-ONLY at most.
VALIDATION_GATES = {
    "production_requires": "PRODUCTION-VALIDATED",
    "falsified_action": "hard_reject_flat",
    "promising_action": "shadow_only",
    "unvalidated_action": "shadow_only",
    "family_status": dict(sorted(FAMILY_STATUS.items())),
}

# Frozen engine thresholds (imported lazily in ``freeze_manifest`` to keep
# this module import-light; they live in src/features/dip.py / rally.py).
ENGINE_THRESHOLDS = {
    "dip_confirm": None,  # filled by freeze_manifest() -> dip.CONFIRM_THRESHOLD
    "dip_watch": None,
    "rally_confirm": None,
    "rally_watch": None,
}

# Frozen Stage-9 LONG-reversal trigger (research, NOT wired to production).
RESEARCH_TRIGGERS = {
    "long_reversal": {"k_of": 3, "components": ["RSI30", "drop5", "streak5n"]},
    "status": FAMILY_STATUS.get("LONG_REVERSAL"),
}


def _engine_thresholds() -> Dict[str, Any]:
    """Read the engine confirmation/watch thresholds from their modules.

    Both engines are exact mirrors (Stage-10 symmetry audit): dip
    CONFIRM=6 / WATCH=4, rally CONFIRM=6 / WATCH=4.
    """
    import src.features.dip as dip
    import src.features.rally as rally

    return {
        "dip_confirm": dip.CONFIRM_THRESHOLD,
        "dip_watch": dip.WATCH_THRESHOLD,
        "rally_confirm": rally.CONFIRM_THRESHOLD,
        "rally_watch": rally.WATCH_THRESHOLD,
    }


def frozen_parameters() -> Dict[str, Any]:
    """The complete frozen parameter set (everything that pins the
    protocol). This dict is canonicalized for the hash - order-insensitive,
    so the hash is stable across processes and re-imports."""
    eng = _engine_thresholds()
    return {
        "protocol_version": PROTOCOL_VERSION,
        "universe": {"group": "full_fx", "n_symbols": 28, "frozen": True},
        "features": [
            "rsi_14",
            "macd_hist",
            "adx",
            "atr_14",
            "close_vs_sma200",
            "regime",
            "dip_score",
            "rally_score",
        ],
        "labels": {"horizon_bars": RESOLVE_HORIZON, "first_touch": True},
        "engines": eng,
        "tp_allocation": TP_ALLOCATION,
        "cost_model": COST_MODEL,
        "decision_rules": DECISION_RULES,
        "portfolio_rules": PORTFOLIO_RULES,
        "validation_gates": VALIDATION_GATES,
        "exit_semantics": EXIT_SEMANTICS,
        "research_triggers": RESEARCH_TRIGGERS,
    }


def protocol_hash() -> str:
    """sha256 of the frozen parameters - the identifier every snapshot
    carries and every evaluation matches against.

    Changing ANY pinned value (threshold, allocation, cost, cap, gate)
    changes this hash, which is the mechanism that keeps old snapshots out
    of a new protocol's evaluation.
    """
    canonical = json.dumps(frozen_parameters(), sort_keys=True, default=str)
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def freeze_manifest() -> Dict[str, Any]:
    """The full preregistration manifest: version, frozen parameters, and
    the protocol hash. ``frozen_at`` is metadata only - it is NOT part of
    the hash (the hash must be stable across runs)."""
    return {
        "protocol_version": PROTOCOL_VERSION,
        "frozen_at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "parameters": frozen_parameters(),
        "sha256": protocol_hash(),
        "note": (
            "Stage-11 freeze: the decision engine is frozen and every new "
            "D1 bar is prospective evidence. No threshold, allocation, "
            "cost, cap or gate may change while the accumulation window is "
            "open - a change is a NEW protocol (new hash) that must be "
            "preregistered separately, and old snapshots are excluded."
        ),
    }


def matches_protocol(snapshot: Dict) -> bool:
    """True when a snapshot was recorded under the CURRENT frozen protocol.

    A snapshot is eligible for evaluation only when its embedded protocol
    hash equals the live one. Snapshots recorded under any other hash
    (older protocol, or one with a parameter change) are excluded - they
    were generated under different rules and must not be pooled.
    """
    return (snapshot.get("protocol") or {}).get("sha256") == protocol_hash()
