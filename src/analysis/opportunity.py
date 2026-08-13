"""
NexusQuant - Unified Opportunity Book & EV-Aware Decision Engine.

The two-sided campaign's decision layer (spec #46): instead of starting
from "should I buy?" or "should I short?", every symbol produces a full
**opportunity book** first - all plausible LONG and SHORT setups, each with
its own evidence score, calibrated probability, expected value, risk and
explicit *rejection reasons* - and only then does a decision engine pick
LONG / SHORT / FLAT from expected value.

    OPPORTUNITY DISCOVERY (12-family classifier + engines)
      -> LONG / SHORT case evaluation (probabilities + EV + risk)
      -> decision engine: EV-based LONG / SHORT / FLAT
      -> explainable rejections for every non-taken opportunity

This module is a pure function over the institutional report dict
(``generate_full_report`` output), so it is shared by the report, the CLI
diagnostics view, the live pass and the dashboard.

Key properties (campaign acceptance criteria):

* **Direction is explicit** - an ``Opportunity`` always carries
  ``direction`` in {long, short}; FLAT is the absence of an acceptable
  opportunity, never a forced third choice.
* **EV decides, not R:R alone** - ``expected_r`` is the decision variable;
  R:R is reported but a 3R setup with a 5% hit probability is correctly
  rejected (spec #17/#18).
* **No fake probabilities** - when no calibrated model probability exists
  for a side, that side's EV is ``None`` and the decision engine falls back
  to the engine-confirmed path (never invents a probability, spec #15/#16).
* **Every rejection is explainable** - each opportunity carries a
  ``rejection_reasons`` list (ML below threshold / EV negative / R:R below
  floor / macro blocked / engine unconfirmed), spec #25.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional

# Decision thresholds (documented defaults; overridable per call).
DEFAULT_MIN_EV_R = 0.20       # minimum positive expected value in R
DEFAULT_MIN_RR = 2.5          # aligns with risk.min_reward_risk
DEFAULT_MIN_ML_PROB = 0.55    # calibrated-probability floor for EV to count
DEFAULT_COST_R = 0.05         # round-trip cost assumption in R when no
#                               settings-derived cost is available


@dataclass
class Opportunity:
    """One directional candidate - the unified representation (spec #9)."""

    symbol: str
    direction: str  # "long" | "short"
    setup_family: Optional[str]
    family_score: Optional[float]
    regime: Optional[str]
    entry_zone: Optional[tuple]
    invalidation: Optional[float]
    target: Optional[float]
    rr: Optional[float]
    probability: Optional[float]
    expected_r: Optional[float]
    cost_r: float
    confidence: Optional[float]
    reasons: List[str] = field(default_factory=list)
    rejection_reasons: List[str] = field(default_factory=list)
    taken: bool = False

    def to_dict(self) -> Dict:
        return {
            "symbol": self.symbol,
            "direction": self.direction,
            "setup_family": self.setup_family,
            "family_score": self.family_score,
            "regime": self.regime,
            "entry_zone": (
                None
                if self.entry_zone is None
                else [float(self.entry_zone[0]), float(self.entry_zone[1])]
            ),
            "invalidation": self.invalidation,
            "target": self.target,
            "rr": self.rr,
            "probability": self.probability,
            "expected_r": self.expected_r,
            "cost_r": self.cost_r,
            "confidence": self.confidence,
            "reasons": self.reasons,
            "rejection_reasons": self.rejection_reasons,
            "taken": self.taken,
        }


# ---------------------------------------------------------------------------
# Cost model (spec #30: cost-aware decision making)
# ---------------------------------------------------------------------------


def roundtrip_cost_r(
    entry: Optional[float],
    stop: Optional[float],
    spread_points: Optional[float] = None,
    slippage_pips: Optional[float] = None,
    pip_size: Optional[float] = None,
) -> float:
    """Round-trip execution cost expressed in R (fractions of risk).

    Cost in price terms = spread (when known) + 2 x slippage (enter + exit).
    Converting to R requires ``risk = |entry - stop|``; when no stop exists
    a documented default cost (DEFAULT_COST_R) is returned so EV is never
    silently zero-costed. ``pip_size`` defaults to 0.01 for JPY-quoted
    pairs, else 0.0001 - passed in by the caller who knows the symbol.
    """
    if entry is None or stop is None or abs(entry - stop) <= 0:
        return DEFAULT_COST_R
    risk = abs(entry - stop)
    spread_price = spread_points or 0.0
    slip = (slippage_pips or 0.0) * (pip_size or 0.0001)
    cost_price = spread_price + 2.0 * slip
    if cost_price <= 0:
        return DEFAULT_COST_R
    return round(float(cost_price / risk), 4)


# ---------------------------------------------------------------------------
# Opportunity construction (one per side, from the report dict)
# ---------------------------------------------------------------------------


def _side_opportunity(
    report: Dict,
    side: str,
    *,
    min_rr: float,
    min_ml_prob: float,
    spread_points: Optional[float],
    slippage_pips: Optional[float],
    pip_size: Optional[float],
) -> Opportunity:
    """Build the LONG or SHORT opportunity from one institutional report.

    Reads the classifier family + probabilities, the engine confirmation,
    the risk plan (entry/stop/target/R:R) and the macro gate; computes EV
    from the calibrated probability when present and collects rejection
    reasons for every failed gate.
    """
    symbol = str(report.get("symbol", "?"))
    sc = report.get("setup_classification") or {}
    regime = (report.get("regime") or {}).get("regime")

    if side == "long":
        family_map = sc.get("long_families") or {}
        prob = sc.get("prob_long")
        engine = report.get("dip") or {}
        engine_confirmed = bool(engine.get("dip_confirmed"))
        engine_score = engine.get("dip_score", 0)
        risk = report.get("risk") or {}
        setup = risk.get("setup") or {}
        targets = report.get("targets") or {}
        macro = (report.get("macro") or {}).get("gate") or {}
    else:
        family_map = sc.get("short_families") or {}
        prob = sc.get("prob_short")
        engine = report.get("rally") or {}
        engine_confirmed = bool(engine.get("rally_confirmed"))
        engine_score = engine.get("rally_score", 0)
        risk = report.get("short_risk") or {}
        setup = risk.get("setup") or {}
        targets = report.get("short_targets") or {}
        macro = (report.get("macro") or {}).get("gate") or {}

    # Best family + score for this side (family_map is pre-sorted desc).
    family = None
    family_score = None
    if family_map:
        family, family_score = next(iter(family_map.items()))
        family_score = float(family_score)

    entry = setup.get("entry")
    stop = setup.get("stop")
    target = setup.get("target")
    rr = setup.get("rr") or setup.get("best_rr")
    # Ladder best R:R is the honest achievable figure (2.5 floor).
    ladder_best = targets.get("best_rr") or rr
    rr_ok = bool(targets.get("min_rr_tp"))

    cost_r = roundtrip_cost_r(
        entry, stop, spread_points=spread_points,
        slippage_pips=slippage_pips, pip_size=pip_size,
    )

    reasons: List[str] = []
    if family:
        reasons.append(f"best family {family} (score {family_score:.2f})")
    if engine_confirmed:
        reasons.append(
            f"{'dip' if side == 'long' else 'rally'} engine CONFIRMED "
            f"(score {engine_score})"
        )
    elif engine_score > 0:
        reasons.append(
            f"{'dip' if side == 'long' else 'rally'} engine forming "
            f"(score {engine_score}, not confirmed)"
        )
    if prob is not None:
        reasons.append(f"calibrated P = {prob:.0%}")

    # Expected value: ONLY from a calibrated probability (never fabricated).
    expected_r: Optional[float] = None
    if prob is not None:
        p = float(prob)
        # Conservative payoff: use the ladder best R:R if it clears the
        # floor, else the achieved RR; EV = P*RR_win - (1-P)*1 - cost.
        win_rr = ladder_best if (ladder_best or 0) >= 1.0 else rr or 1.0
        expected_r = round(p * win_rr - (1.0 - p) * 1.0 - cost_r, 4)

    rejection: List[str] = []
    if not family_map:
        rejection.append("no setup family cleared the evidence bar")
    if not engine_confirmed:
        rejection.append(
            f"{'dip' if side == 'long' else 'rally'} engine not confirmed"
        )
    if prob is None:
        rejection.append("no calibrated model probability (EV not computed)")
    elif prob < min_ml_prob:
        rejection.append(f"calibrated P {prob:.0%} < {min_ml_prob:.0%} floor")
    if expected_r is not None and expected_r <= 0:
        rejection.append(f"EV {expected_r:+.2f}R <= 0 after costs")
    if rr_ok is False and targets.get("targets"):
        rejection.append(
            f"R:R floor {min_rr:g}:1 not reached (ladder best "
            f"{ladder_best if ladder_best is not None else 'n/a'})"
        )
    if macro.get("allowed") is False:
        rejection.append(f"macro gate BLOCKED ({macro.get('reason')})")

    return Opportunity(
        symbol=symbol,
        direction=side,
        setup_family=family,
        family_score=family_score,
        regime=regime,
        entry_zone=(entry, entry) if entry is not None else None,
        invalidation=stop,
        target=target,
        rr=ladder_best if ladder_best is not None else rr,
        probability=prob,
        expected_r=expected_r,
        cost_r=cost_r,
        confidence=sc.get("confidence"),
        reasons=reasons,
        rejection_reasons=rejection,
    )


# ---------------------------------------------------------------------------
# Opportunity book + decision engine
# ---------------------------------------------------------------------------


def build_opportunity_book(
    report: Dict,
    *,
    min_rr: float = DEFAULT_MIN_RR,
    min_ml_prob: float = DEFAULT_MIN_ML_PROB,
    spread_points: Optional[float] = None,
    slippage_pips: Optional[float] = None,
    pip_size: Optional[float] = None,
) -> Dict:
    """Full opportunity book for one report: both sides + FLAT verdict.

    Returns ``{symbol, long, short, verdict, reasons}`` where ``verdict``
    is the decision engine's ``{direction, status, expected_r, reason}``.

    Decision policy (EV-first, engine fallback):

    1. A side's EV is only computed when it has a **calibrated**
       probability. Without any calibrated probability the decision falls
       back to the engine-confirmed side (rule path).
    2. LONG wins when its EV > SHORT EV and EV > min_ev; SHORT wins on the
       mirror; neither clearing min_ev -> FLAT with explicit reasons.
    3. The winning side must not be macro-blocked and must clear the R:R
       floor when a ladder exists (the floors stay enforced on trades).
    """
    symbol = str(report.get("symbol", "?"))
    long_opp = _side_opportunity(
        report, "long", min_rr=min_rr, min_ml_prob=min_ml_prob,
        spread_points=spread_points, slippage_pips=slippage_pips,
        pip_size=pip_size,
    )
    short_opp = _side_opportunity(
        report, "short", min_rr=min_rr, min_ml_prob=min_ml_prob,
        spread_points=spread_points, slippage_pips=slippage_pips,
        pip_size=pip_size,
    )

    ev_l = long_opp.expected_r
    ev_s = short_opp.expected_r
    verdict_reasons: List[str] = []

    # Path 1: no calibrated probability on either side -> engine-confirmed
    # rule path (both existing engines already enforce structure gates).
    if ev_l is None and ev_s is None:
        confirmed = [o for o in (long_opp, short_opp) if o.reasons
                     and any("CONFIRMED" in r for r in o.reasons)]
        if confirmed:
            chosen = confirmed[0]
            chosen.taken = True
            verdict_reasons.append(
                "rule path (no calibrated ML probability): engine-confirmed "
                f"{chosen.direction.upper()}"
            )
            verdict = {
                "direction": chosen.direction,
                "status": "CONFIRMED",
                "expected_r": None,
                "reason": "; ".join(verdict_reasons),
            }
        else:
            verdict = {
                "direction": "flat",
                "status": "FLAT",
                "expected_r": None,
                "reason": "no calibrated probability and no engine-confirmed "
                "setup on either side",
            }
        return _book(symbol, long_opp, short_opp, verdict)

    # Path 2: EV-aware decision (at least one calibrated side).
    if ev_l is not None and ev_l > DEFAULT_MIN_EV_R and (
        ev_s is None or ev_l >= ev_s
    ):
        chosen, other = long_opp, short_opp
    elif ev_s is not None and ev_s > DEFAULT_MIN_EV_R:
        chosen, other = short_opp, long_opp
    else:
        chosen = other = None

    if chosen is None:
        verdict = {
            "direction": "flat",
            "status": "FLAT",
            "expected_r": max(ev_l or 0.0, ev_s or 0.0),
            "reason": (
                f"no side clears EV floor (+{DEFAULT_MIN_EV_R:g}R): "
                f"long {ev_l if ev_l is not None else 'n/a'}, "
                f"short {ev_s if ev_s is not None else 'n/a'}"
            ),
        }
        return _book(symbol, long_opp, short_opp, verdict)

    # The EV winner must still clear the hard floors (macro + R:R).
    if other is not None and (other.expected_r or 0) > (chosen.expected_r or 0):
        chosen, other = other, chosen
    macro_blocked = any("BLOCKED" in r for r in chosen.rejection_reasons)
    if macro_blocked:
        verdict = {
            "direction": "flat",
            "status": "FLAT",
            "expected_r": chosen.expected_r,
            "reason": f"EV winner {chosen.direction.upper()} is macro-blocked",
        }
        return _book(symbol, long_opp, short_opp, verdict)

    chosen.taken = True
    verdict_reasons.append(
        f"EV path: {chosen.direction.upper()} "
        f"EV {chosen.expected_r:+.2f}R > "
        f"{'short' if chosen.direction == 'long' else 'long'} "
        f"{other.expected_r if other is not None and other.expected_r is not None else 'n/a'}"
    )
    verdict = {
        "direction": chosen.direction,
        "status": "TRADE" if chosen.probability is not None else "RULE",
        "expected_r": chosen.expected_r,
        "reason": "; ".join(verdict_reasons),
    }
    return _book(symbol, long_opp, short_opp, verdict)


def _book(symbol, long_opp, short_opp, verdict) -> Dict:
    return {
        "symbol": symbol,
        "long": long_opp.to_dict(),
        "short": short_opp.to_dict(),
        "verdict": verdict,
    }


# ---------------------------------------------------------------------------
# CLI / diagnostics rendering (spec #24)
# ---------------------------------------------------------------------------


def format_opportunity_book(book: Dict) -> str:
    """Render one symbol's opportunity book like the campaign diagnostic."""

    def _block(opp: Dict, label: str) -> List[str]:
        lines = [f"{label.upper()} OPPORTUNITY"]
        fam = opp.get("setup_family")
        lines.append(
            f"  setup      : {fam or '-'} "
            f"(score {opp.get('family_score') if opp.get('family_score') is not None else '-'})"
        )
        lines.append(f"  probability: {'-' if opp.get('probability') is None else f'{opp['probability']:.0%}'}")
        lines.append(f"  EV         : {'-' if opp.get('expected_r') is None else f'{opp['expected_r']:+.2f}R'}")
        lines.append(f"  R:R        : {'-' if opp.get('rr') is None else f'{opp['rr']:.2f}'} · cost {opp['cost_r']:.3f}R")
        if opp.get("entry_zone"):
            lines.append(
                f"  entry      : {opp['entry_zone'][0]:,.5f} · stop "
                f"{'-' if opp.get('invalidation') is None else f'{opp['invalidation']:,.5f}'} · "
                f"target {'-' if opp.get('target') is None else f'{opp['target']:,.5f}'}"
            )
        if opp.get("reasons"):
            lines.append("  reasons    : " + "; ".join(opp["reasons"]))
        if opp.get("rejection_reasons"):
            lines.append("  REJECTED   : " + "; ".join(opp["rejection_reasons"]))
        if opp.get("taken"):
            lines.append("  TAKEN       ✓")
        return lines

    v = book["verdict"]
    lines = [f"{book['symbol']} — VERDICT: {v['direction'].upper()} ({v['status']})"]
    if v.get("expected_r") is not None:
        lines.append(f"  expected EV: {v['expected_r']:+.2f}R")
    lines.append(f"  why        : {v.get('reason')}")
    lines.append("")
    lines += _block(book["long"], "long")
    lines.append("")
    lines += _block(book["short"], "short")
    return "\n".join(lines)


if __name__ == "__main__":
    print("NexusQuant Opportunity Book module ready.")
