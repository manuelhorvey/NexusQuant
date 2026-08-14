"""
NexusQuant - Strict Promotion Ladder (Stage-11).

Stage-10 established explicit research/production separation; Stage-11
turns that into a hard promotion ladder. A family (or strategy) moves up a
level ONLY when the preregistered gates at the next level pass - never
because the recent live output "looked good". Moving down is automatic.

Levels (spec "strict promotion ladder"):

    L0 UNVALIDATED             no reliable evidence               -> FLAT
    L1 RESEARCH_CANDIDATE      promising historical evidence      -> shadow only
    L2 PROSPECTIVE_VALIDATION  frozen strategy accruing genuinely  -> shadow only
                               unseen observations
    L3 VALIDATED_ALPHA         passes the full gate battery        -> tiny
                                                                     controlled
                                                                     capital
    L4 PRODUCTION_CANDIDATE    survives live/shadow validation     -> controlled
                               with realistic execution              deployment
    L5 PRODUCTION              continuous monitoring with          -> production
                               automatic degradation/freeze rules

The L3 gate battery is the firewall between research and money. Every gate
is a named, checkable predicate; ``promote()`` reports exactly which gates
passed and which failed, so a promotion attempt is auditable and a
rejection is explainable (no silent "not yet").

Discipline: the gates are evaluated ONLY on data the frozen protocol has
not been tuned on (the prospective window), per the Stage-9 frozen
evaluation protocol. None of this module writes anything - it is a pure
function of the evidence summary.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

from src.analysis.opportunity import (
    FALSIFIED,
    PROMISING_SHADOW_ONLY,
    PRODUCTION_VALIDATED,
    UNVALIDATED,
)

# ---------------------------------------------------------------------------
# Level definitions
# ---------------------------------------------------------------------------

L0_UNVALIDATED = "L0_UNVALIDATED"
L1_RESEARCH_CANDIDATE = "L1_RESEARCH_CANDIDATE"
L2_PROSPECTIVE_VALIDATION = "L2_PROSPECTIVE_VALIDATION"
L3_VALIDATED_ALPHA = "L3_VALIDATED_ALPHA"
L4_PRODUCTION_CANDIDATE = "L4_PRODUCTION_CANDIDATE"
L5_PRODUCTION = "L5_PRODUCTION"

ACTIONS = {
    L0_UNVALIDATED: "FLAT",
    L1_RESEARCH_CANDIDATE: "SHADOW_ONLY",
    L2_PROSPECTIVE_VALIDATION: "SHADOW_ONLY",
    L3_VALIDATED_ALPHA: "TINY_CONTROLLED_CAPITAL",
    L4_PRODUCTION_CANDIDATE: "CONTROLLED_DEPLOYMENT",
    L5_PRODUCTION: "CONTINUOUS_MONITORING",
}

LADDER: List[str] = [
    L0_UNVALIDATED,
    L1_RESEARCH_CANDIDATE,
    L2_PROSPECTIVE_VALIDATION,
    L3_VALIDATED_ALPHA,
    L4_PRODUCTION_CANDIDATE,
    L5_PRODUCTION,
]

# ---------------------------------------------------------------------------
# L3 gate battery (all required - the research firewall)
# ---------------------------------------------------------------------------

L3_GATES: List[str] = [
    "min_effective_n",  # enough resolved prospective observations
    "independent_window",  # genuinely unseen data (never the discovery window)
    "walk_forward",  # stable across sequential folds
    "cost_robustness",  # edge survives realistic + stressed costs
    "permutation_test",  # p < threshold vs shuffled labels/timing
    "bootstrap_ci",  # expectancy CI strictly positive
    "loso",  # leave-one-symbol-out stability
    "regime_stability",  # not a single-regime artifact
    "concentration_limits",  # symbol / cluster concentration within caps
    "portfolio_contribution",  # additive to portfolio risk-adjusted return
    "drawdown_limits",  # max drawdown within bounds
]

# ---------------------------------------------------------------------------
# L5 degradation / freeze rules (automatic, not discretionary)
# ---------------------------------------------------------------------------

L5_MONITORING_RULES = {
    "degrade_on": [
        "rolling_expectancy_below_zero",  # trailing window goes negative
        "drawdown_breaker",  # portfolio drawdown limit touched
        "calibration_drift",  # probability calibration ECE degrades
    ],
    "freeze_on": ["protocol_mismatch"],  # parameters changed -> freeze & re-eval
}

# Preregistered default thresholds for the numeric gates (documented; a
# re-preregistration is required to change them - that is a NEW protocol).
DEFAULT_GATE_THRESHOLDS = {
    "min_effective_n": 50,  # resolved observations per family
    "independent_window_days": 90,  # minimum untouched calendar span
    "permutation_p": 0.05,  # one-sided
    "bootstrap_ci_lower_r": 0.05,  # 95% CI lower bound in R units
    "cost_stress_r": 0.15,  # stressed round-trip cost in R
    "max_symbol_concentration": 0.40,
    "max_cluster_concentration": 0.60,
    "max_drawdown_r": 12.0,  # cumulative-R drawdown bound
    "max_ece": 0.12,  # calibration error bound
}


def level_index(level: str) -> int:
    return LADDER.index(level)


def level_from_status(status: Optional[str]) -> str:
    """Map the Stage-10 family status to its ladder level.

    FALSIFIED and UNVALIDATED are both L0 (FLAT). PROMISING-SHADOW-ONLY
    sits at L1: it has promising historical evidence but no prospective
    window - the moment the frozen recorder starts accruing unseen
    observations for it, it moves to L2.
    """
    if status == PRODUCTION_VALIDATED:
        return L3_VALIDATED_ALPHA
    if status == PROMISING_SHADOW_ONLY:
        return L1_RESEARCH_CANDIDATE
    if status in (FALSIFIED, UNVALIDATED):
        return L0_UNVALIDATED
    return L0_UNVALIDATED


def check_l3_gates(stats: Dict[str, Any], thresholds: Optional[Dict] = None) -> Dict:
    """Evaluate the full L3 gate battery against a family evidence summary.

    ``stats`` keys (all optional - a missing key fails its gate, because a
    gate that cannot be checked is a gate that cannot be passed):

      min_effective_n      int     resolved observations
      independent_window_days  int  untouched calendar span
      walk_forward_ok      bool
      net_r_after_costs    float   gross - realistic costs
      net_r_after_stress   float   gross - stressed costs
      permutation_p        float
      bootstrap_ci_lower   float
      loso_ok              bool
      n_regimes            int     regimes with positive net R
      max_symbol_conc      float   0..1
      max_cluster_conc     float   0..1
      max_drawdown_r       float
      ece                  float   calibration error

    Returns ``{level, action, gates: {name: {pass, detail}}, failed}``.
    """
    thr = {**DEFAULT_GATE_THRESHOLDS, **(thresholds or {})}
    s = stats or {}

    def _gate(name: str, ok: bool, detail: str) -> Dict:
        return {"name": name, "pass": bool(ok), "detail": detail}

    gates = [
        _gate(
            "min_effective_n",
            int(s.get("min_effective_n", 0)) >= int(thr["min_effective_n"]),
            f"n={s.get('min_effective_n', 0)} >= {thr['min_effective_n']}",
        ),
        _gate(
            "independent_window",
            int(s.get("independent_window_days", 0))
            >= int(thr["independent_window_days"]),
            f"{s.get('independent_window_days', 0)} days untouched >= "
            f"{thr['independent_window_days']}",
        ),
        _gate(
            "walk_forward",
            bool(s.get("walk_forward_ok", False)),
            "walk-forward folds stable"
            if s.get("walk_forward_ok")
            else "not run / unstable",
        ),
        _gate(
            "cost_robustness",
            float(s.get("net_r_after_costs", float("-inf"))) > 0
            and float(s.get("net_r_after_stress", float("-inf"))) > 0,
            (
                f"net {s.get('net_r_after_costs', 'n/a')}R after costs, "
                f"{s.get('net_r_after_stress', 'n/a')}R after {thr['cost_stress_r']}R stress"
            ),
        ),
        _gate(
            "permutation_test",
            float(s.get("permutation_p", 1.0)) < float(thr["permutation_p"]),
            f"perm p={s.get('permutation_p', 'n/a')} < {thr['permutation_p']}",
        ),
        _gate(
            "bootstrap_ci",
            float(s.get("bootstrap_ci_lower", float("-inf")))
            >= float(thr["bootstrap_ci_lower_r"]),
            f"95% CI lower {s.get('bootstrap_ci_lower', 'n/a')}R >= {thr['bootstrap_ci_lower_r']}R",
        ),
        _gate(
            "loso",
            bool(s.get("loso_ok", False)),
            "leave-one-symbol-out stable" if s.get("loso_ok") else "not run / unstable",
        ),
        _gate(
            "regime_stability",
            int(s.get("n_regimes_positive", 0)) >= 2,
            f"{s.get('n_regimes_positive', 0)} regimes positive (need >= 2)",
        ),
        _gate(
            "concentration_limits",
            float(s.get("max_symbol_conc", 1.0))
            <= float(thr["max_symbol_concentration"])
            and float(s.get("max_cluster_conc", 1.0))
            <= float(thr["max_cluster_concentration"]),
            (
                f"symbol {s.get('max_symbol_conc', 'n/a')} <= "
                f"{thr['max_symbol_concentration']}, cluster "
                f"{s.get('max_cluster_conc', 'n/a')} <= {thr['max_cluster_concentration']}"
            ),
        ),
        _gate(
            "portfolio_contribution",
            bool(s.get("portfolio_contribution_ok", False)),
            "marginal portfolio contribution positive"
            if s.get("portfolio_contribution_ok")
            else "not measured / negative",
        ),
        _gate(
            "drawdown_limits",
            float(s.get("max_drawdown_r", float("inf")))
            <= float(thr["max_drawdown_r"]),
            f"max DD {s.get('max_drawdown_r', 'n/a')}R <= {thr['max_drawdown_r']}R",
        ),
    ]

    failed = [g for g in gates if not g["pass"]]
    if failed:
        return {
            "level": L2_PROSPECTIVE_VALIDATION,  # stays shadow - cannot advance
            "action": ACTIONS[L2_PROSPECTIVE_VALIDATION],
            "gates": gates,
            "failed": [g["name"] for g in failed],
            "reason": f"{len(failed)}/{len(gates)} L3 gates failed: "
            + ", ".join(g["name"] for g in failed),
        }
    return {
        "level": L3_VALIDATED_ALPHA,
        "action": ACTIONS[L3_VALIDATED_ALPHA],
        "gates": gates,
        "failed": [],
        "reason": "all L3 gates passed",
    }


def promote(stats: Dict[str, Any]) -> Dict:
    """Evaluate a family's promotion level from its evidence summary.

    ``stats`` is the accumulated evidence (see ``check_l3_gates``) plus:

      prospective_records   int   recorded decision snapshots (L2 evidence)
      resolved             int   resolved prospective outcomes
      shadow_validated_ok  bool  L4: survived live/shadow with realistic exec
      monitoring_ok        bool  L5: continuous monitoring green

    Pure function: returns the level + action + reasons; never writes.
    """
    s = stats or {}
    n_records = int(s.get("prospective_records", 0))
    n_resolved = int(s.get("resolved", 0))
    status = s.get("status")

    # FALSIFIED is terminal: hard FLAT regardless of any records. The
    # hypothesis was rejected on its own evidence; prospective records of
    # a falsified family cannot resurrect it - only a NEW preregistered
    # hypothesis can, and it enters the ladder from L0/L1 on its own.
    if status == FALSIFIED:
        return {
            "level": L0_UNVALIDATED,
            "action": ACTIONS[L0_UNVALIDATED],
            "reason": "FALSIFIED - hard-rejected, FLAT (Stage-6/10); "
            "a falsified family cannot be promoted by the window",
        }

    # L0/L1 from the research status registry first.
    base = level_from_status(status)

    # L2 requires the prospective window to be genuinely accruing.
    if n_records == 0:
        return {
            "level": base,
            "action": ACTIONS[base],
            "reason": (
                "no prospective records yet - shadow only, "
                "recorder must accrue unseen observations"
            ),
        }
    if n_resolved < int(s.get("min_effective_n", 0)):
        return {
            "level": L2_PROSPECTIVE_VALIDATION,
            "action": ACTIONS[L2_PROSPECTIVE_VALIDATION],
            "reason": f"prospective window open: {n_resolved} resolved < "
            f"{s.get('min_effective_n', 0)} minimum - keep recording, never tune",
        }

    # L3 - the firewall.
    l3 = check_l3_gates(s)
    if l3["failed"]:
        return l3

    # L4 / L5.
    if not s.get("shadow_validated_ok"):
        return {
            "level": L3_VALIDATED_ALPHA,
            "action": ACTIONS[L3_VALIDATED_ALPHA],
            "reason": "L3 passed - tiny controlled capital only; "
            "live/shadow validation with realistic execution not yet green",
        }
    if not s.get("monitoring_ok"):
        return {
            "level": L4_PRODUCTION_CANDIDATE,
            "action": ACTIONS[L4_PRODUCTION_CANDIDATE],
            "reason": "L4 passed - controlled deployment; monitoring not green",
        }
    return {
        "level": L5_PRODUCTION,
        "action": ACTIONS[L5_PRODUCTION],
        "reason": "all levels passed - production with automatic "
        "degradation/freeze rules",
    }
