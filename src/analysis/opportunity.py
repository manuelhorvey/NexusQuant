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
* **Target-level EV decides, not ranking EV** - Stage-2/3/10: ``ranking
  EV`` (P x ladder-best R:R) materially overstates the economics the TP
  ladder actually delivers. When the empirical first-touch TP distribution
  exists, ``ev_target_level`` (cost-adjusted target-level EV, computed
  from the ACTUAL position allocation across TP1/TP2/TP3) is the decision
  variable; ranking EV is reported alongside as the comparison it is.
* **Allocation-aware economics** - ``expected_r_alloc`` prices the real
  partial-exit plan (1/3 at TP1 + 1/3 at TP2 + 1/3 at TP3) instead of
  assuming the full position reaches the ladder best (spec #10/Stage-10).
* **No fake probabilities** - when no calibrated model probability exists
  for a side, that side's EV is ``None`` and the decision engine falls back
  to the engine-confirmed path (never invents a probability, spec #15/#16).
* **Family validation status is enforced** - every family carries a
  research status (PRODUCTION-VALIDATED / PROMISING-SHADOW-ONLY /
  UNVALIDATED / FALSIFIED). A FALSIFIED family (e.g. SHORT_REVERSAL,
  Stage-6) is hard-rejected and can never carry a production trade;
  UNVALIDATED families compete but are flagged SHADOW-ONLY in the verdict
  (Stage-10 live/research separation).
* **Every rejection is explainable** - each opportunity carries a
  ``rejection_reasons`` list (ML below threshold / EV negative / R:R below
  floor / macro blocked / engine unconfirmed / family falsified), spec #25.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional

# Decision thresholds (documented defaults; overridable per call).
DEFAULT_MIN_EV_R = 0.20  # minimum positive expected value in R
DEFAULT_MIN_RR = 2.5  # aligns with risk.min_reward_risk
DEFAULT_MIN_ML_PROB = 0.55  # calibrated-probability floor for EV to count
DEFAULT_COST_R = 0.05  # round-trip cost assumption in R when no
#                               settings-derived cost is available

# Partial-exit allocation for the ladder (spec #11 scaling-out plan):
# 1/3 at TP1, 1/3 at TP2, 1/3 at TP3. The allocation-aware expected R
# prices THIS plan - not the "full position at the ladder best" fiction.
ALLOCATION = (1.0 / 3.0, 1.0 / 3.0, 1.0 / 3.0)

# ---------------------------------------------------------------------------
# Family validation statuses (Stage-10 live/research separation, spec #13)
#
# Status is a RESEARCH verdict on the family's hypothesis, carried into the
# live book so the architecture never silently promotes a family whose
# evidence is absent or falsified. Source of truth:
#   - LONG_REVERSAL        : Stage-9 B. PROMISING BUT INSUFFICIENT EVIDENCE
#                            (frozen 28-symbol universe, fresh window UNRESOLVED)
#   - SHORT_REVERSAL       : Stage-6 FALSIFIED (perm p=0.38, shuffled timing
#                            == signal, untouched -0.77R)
#   - LONG_TREND_CONT / BUY_DIP / BREAKOUT / BREAKOUT_RETEST / MEAN_REVERSION,
#     SHORT_TREND_CONT / SELL_RALLY / BREAKDOWN / BREAKDOWN_RETEST /
#     MEAN_REVERSION      : no independent OOS evidence at target level
#                            (Stage-3 attribution: target-level EV ~ 0 after
#                            costs on both sides) -> UNVALIDATED
#   - The frozen Stage-9 LONG reversal trigger (k=3 of RSI30/drop5/streak5n)
#     is NOT wired into production; when it is, it must enter as its own
#     family with this same status registry, never as a side-channel.
# ---------------------------------------------------------------------------

PRODUCTION_VALIDATED = "PRODUCTION-VALIDATED"
PROMISING_SHADOW_ONLY = "PROMISING-SHADOW-ONLY"
UNVALIDATED = "UNVALIDATED"
FALSIFIED = "FALSIFIED"

FAMILY_STATUS: Dict[str, str] = {
    # LONG side
    "LONG_REVERSAL": PROMISING_SHADOW_ONLY,
    "LONG_TREND_CONTINUATION": UNVALIDATED,
    "LONG_BUY_DIP": UNVALIDATED,
    "LONG_BREAKOUT": UNVALIDATED,
    "LONG_BREAKOUT_RETEST": UNVALIDATED,
    "LONG_MEAN_REVERSION": UNVALIDATED,
    # SHORT side
    "SHORT_REVERSAL": FALSIFIED,
    "SHORT_TREND_CONTINUATION": UNVALIDATED,
    "SHORT_SELL_RALLY": UNVALIDATED,
    "SHORT_BREAKDOWN": UNVALIDATED,
    "SHORT_BREAKDOWN_RETEST": UNVALIDATED,
    "SHORT_MEAN_REVERSION": UNVALIDATED,
}


# Families with NO validated long-horizon evidence at all: a candidate in
# these can be *discovered* (the architecture must see both sides), but the
# final verdict is capped at SHADOW-ONLY until independent evidence accrues.
def family_status(family: Optional[str]) -> str:
    """Validation status for a setup family (default UNVALIDATED)."""
    if not family:
        return UNVALIDATED
    return FAMILY_STATUS.get(str(family), UNVALIDATED)


# An entry within this fraction of a daily ATR of the last close is an
# IMMEDIATE (market) order - the trigger is at price now. Beyond it the
# order waits at the zone (limit). 0.25 means "within a quarter of a
# daily ATR of the close".
MARKET_ENTRY_ATR_FRAC = 0.25


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
    ev_target_level: Optional[float] = None  # honest payoff-distribution EV
    expected_r_alloc: Optional[float] = None  # allocation-aware expected R
    cost_break_even: Optional[float] = None  # cost at which alloc EV = 0
    validation_status: str = UNVALIDATED  # family research status
    entry_type: str = "limit"  # "market" when the entry is at/near the close
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
            "entry_type": self.entry_type,
            "invalidation": self.invalidation,
            "target": self.target,
            "rr": self.rr,
            "probability": self.probability,
            "expected_r": self.expected_r,
            "ev_target_level": self.ev_target_level,
            "expected_r_alloc": self.expected_r_alloc,
            "cost_break_even": self.cost_break_even,
            "validation_status": self.validation_status,
            "cost_r": self.cost_r,
            "confidence": self.confidence,
            "reasons": self.reasons,
            "rejection_reasons": self.rejection_reasons,
            "taken": self.taken,
        }


# ---------------------------------------------------------------------------
# Cost model (spec #30: cost-aware decision making)
# ---------------------------------------------------------------------------


def entry_type_for(
    entry: Optional[float],
    close: Optional[float],
    atr: Optional[float],
    atr_frac: float = MARKET_ENTRY_ATR_FRAC,
) -> str:
    """Classify an entry as an immediate market order or a pending limit.

    When the entry is within ``atr_frac`` of a daily ATR of the last
    close, the trigger is at price NOW -> "market". Otherwise the order
    waits at the zone -> "limit". Missing inputs default to "limit"
    (never claim an immediate fill without the data to justify it).
    """
    if entry is None or close is None or atr is None or atr <= 0:
        return "limit"
    dist_atr = abs(entry - close) / atr
    return "market" if dist_atr <= atr_frac else "limit"


def target_level_ev(
    p_tp1: Optional[float],
    p_tp2: Optional[float],
    p_tp3: Optional[float],
    p_sl: Optional[float],
    rungs: tuple = (1.0, 2.0, 3.0),
    cost_r: float = 0.0,
) -> Optional[float]:
    """Expected value from the target-level payoff distribution (spec #4).

    EV = P(tp1)*r1 + P(tp2)*r2 + P(tp3)*r3 - P(sl)*1 - cost_r, where
    ``rungs`` are the ladder's R:R multiples and the distribution is the
    empirical first-touch table {P(TP_k before SL), P(SL)}. A missing or
    inconsistent distribution (does not sum to a valid probability mass)
    returns None - never a fabricated EV.
    """
    if any(p is None for p in (p_tp1, p_tp2, p_tp3, p_sl)):
        return None
    total = p_tp1 + p_tp2 + p_tp3 + p_sl
    # Tolerance absorbs rounding artifacts (a table summing to 1.0001 is a
    # valid distribution; 1.15 is not).
    if total <= 0 or total > 1.0 + 1e-3:
        return None
    ev = p_tp1 * rungs[0] + p_tp2 * rungs[1] + p_tp3 * rungs[2] - p_sl * 1.0 - cost_r
    return round(float(ev), 4)


def allocation_weighted_ev(
    p_tp1: Optional[float],
    p_tp2: Optional[float],
    p_tp3: Optional[float],
    p_sl: Optional[float],
    rungs: tuple = (1.0, 2.0, 3.0),
    alloc: tuple = ALLOCATION,
    cost_r: float = 0.0,
) -> Optional[float]:
    """Allocation-aware expected R for the real scaling-out plan (Stage-10).

    The ladder's headline 3R is the payoff if the FULL position rides to
    TP3. The actual spec #11 plan scales out 1/3 at TP1, 1/3 at TP2, 1/3
    at TP3 - so the expected portfolio-level R is

        P(tp1) * (a1*r1)
      + P(tp2) * (a1*r1 + a2*r2)
      + P(tp3) * (a1*r1 + a2*r2 + a3*r3)
      - P(sl)  * 1.0
      - cost_r

    i.e. a TP2 first-touch already banked the TP1 third at r1 and the TP2
    third at r2; only the residual third is still riding to r3. This is the
    expected R the trade ACTUALLY pays under the stated allocation - the
    honest number the ladder-best headline hides. P(time exit) contributes
    0 payoff (positions exit at the horizon close, ~breakeven before
    costs); when the residual never reaches TP3 it is conservatively
    counted at 0.

    Returns None when the distribution is missing or invalid (same
    validity rules as ``target_level_ev`` - never a fabricated EV).
    """
    if any(p is None for p in (p_tp1, p_tp2, p_tp3, p_sl)):
        return None
    total = p_tp1 + p_tp2 + p_tp3 + p_sl
    if total <= 0 or total > 1.0 + 1e-3:
        return None
    a1, a2, a3 = alloc
    ev = (
        p_tp1 * (a1 * rungs[0])
        + p_tp2 * (a1 * rungs[0] + a2 * rungs[1])
        + p_tp3 * (a1 * rungs[0] + a2 * rungs[1] + a3 * rungs[2])
        - p_sl * 1.0
        - cost_r
    )
    return round(float(ev), 4)


def cost_break_even_r(
    p_tp1: Optional[float],
    p_tp2: Optional[float],
    p_tp3: Optional[float],
    p_sl: Optional[float],
    rungs: tuple = (1.0, 2.0, 3.0),
    alloc: tuple = ALLOCATION,
) -> Optional[float]:
    """The round-trip cost (in R) at which the allocation-aware expected R
    becomes zero - the economic break-even. None when no valid
    distribution. A trade whose realistic cost exceeds this has no edge
    (Stage-10: reject trades whose edge disappears under realistic costs)."""
    if any(p is None for p in (p_tp1, p_tp2, p_tp3, p_sl)):
        return None
    total = p_tp1 + p_tp2 + p_tp3 + p_sl
    if total <= 0 or total > 1.0 + 1e-3:
        return None
    a1, a2, a3 = alloc
    gross = (
        p_tp1 * (a1 * rungs[0])
        + p_tp2 * (a1 * rungs[0] + a2 * rungs[1])
        + p_tp3 * (a1 * rungs[0] + a2 * rungs[1] + a3 * rungs[2])
        - p_sl * 1.0
    )
    return round(float(gross), 4)


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
    tp_probs: Optional[Dict] = None,
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
        macro = (report.get("macro") or {}).get("gate_short") or {}
        if not macro:
            macro = (report.get("macro") or {}).get("gate") or {}

    # Best family + score for this side. ``max`` by score (not dict order)
    # so the book is robust to unsorted ``long_families``/``short_families``
    # input (classify_setup pre-sorts, but this is a pure function over
    # the report dict and must not depend on it).
    family = None
    family_score = None
    if family_map:
        family, family_score = max(
            family_map.items(), key=lambda kv: kv[1] if kv[1] is not None else 0.0
        )
        family_score = float(family_score)

    entry = setup.get("entry")
    stop = setup.get("stop")
    target = setup.get("target")
    rr = setup.get("rr") or setup.get("best_rr")
    # Ladder best R:R is the honest achievable figure (2.5 floor).
    ladder_best = targets.get("best_rr") or rr
    rr_ok = bool(targets.get("min_rr_tp"))

    # Immediate vs pending: entry within a fraction of a daily ATR of the
    # close is a MARKET order (trigger at price now), else LIMIT at the
    # zone. The engines' pullback/rally zones are almost always away from
    # price, so most opportunities stay LIMIT - but a breakout/breakdown
    # at the trigger can now be an immediate entry.
    close = report.get("last_close")
    atr = (report.get("volatility") or {}).get("atr_14")
    entry_type = entry_type_for(entry, close, atr)

    # Market-fill honesty: a market order fills at ~the close, not at the
    # modeled zone level - so risk, reward and the ladder's R:R must be
    # re-expressed from the actual fill price. Without this a MARKET flag
    # would silently claim the LIMIT-level R:R (spec: no unrealistic
    # execution). When the fill is inside the stop or past the target the
    # recompute is skipped and the LIMIT-level figures are kept.
    if entry_type == "market" and close is not None and stop is not None:
        risk_m = abs(float(close) - float(stop))
        if risk_m > 0:
            entry = float(close)
            if target is not None and abs(float(target) - entry) > 0:
                rr = round(abs(float(target) - entry) / risk_m, 2)
            ladder = targets.get("targets") or []
            rrs = [
                abs(float(t["price"]) - entry) / risk_m
                for t in ladder
                if t.get("price") is not None
            ]
            if rrs:
                ladder_best = round(max(rrs), 2)
            rr_ok = ladder_best is not None and ladder_best >= min_rr

    cost_r = roundtrip_cost_r(
        entry,
        stop,
        spread_points=spread_points,
        slippage_pips=slippage_pips,
        pip_size=pip_size,
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
        if prob < min_ml_prob:
            # Informational only: the live pass applies the ML floor as a
            # FILTER; the book's decision variable is expected value.
            reasons.append(
                f"below live ML floor {min_ml_prob:.0%} (filter applies at pass level)"
            )

    # ------------------------------------------------------------------
    # Expected value - Stage-2/10 correction: the DECISION variable is the
    # cost-adjusted TARGET-LEVEL EV (from the empirical first-touch TP
    # distribution and the ACTUAL scaling-out allocation), not the ranking
    # EV (P x ladder-best) which Stage-2/3 showed materially overstates
    # the economics the ladder delivers.
    #
    # Only a calibrated probability AND a real payoff basis count (never
    # fabricated). When no target ladder and no achieved R:R exist, EV is
    # left None - assuming a 1.0R payoff would be a silent fabrication
    # against the campaign's own "no fake EV" principle, and conservative
    # FLAT is the honest behavior.
    #
    # tp_probs is the per-FAMILY empirical distribution when available
    # (census ``families`` table - the Stage-10 requirement: a candidate's
    # EV uses its OWN family's first-touch table, never the pooled one),
    # falling back to the pooled per-side table. Both are reported; the
    # allocation-aware ``expected_r_alloc`` is the honest economics of the
    # spec #11 scaling-out plan (1/3 at TP1 + 1/3 at TP2 + 1/3 at TP3).
    expected_r: Optional[float] = None  # ranking EV (comparison only)
    ev_target_level: Optional[float] = None  # decision EV (whole-ladder)
    expected_r_alloc: Optional[float] = None  # decision EV (allocation-aware)
    cost_break_even: Optional[float] = None
    payoff_basis: Optional[float] = None
    if prob is not None:
        p = float(prob)
        # Family-specific TP table first (Stage-10), else the pooled side.
        fam_tp = None
        if family and isinstance(tp_probs, dict):
            fam_tp = tp_probs.get("families", {}).get(family)
        side_tp = None
        if isinstance(tp_probs, dict):
            side_tp = tp_probs.get("side", {}).get(side) or tp_probs.get(side)
        dist = fam_tp or side_tp
        if dist is not None:
            # Real ladder rungs (TP1 < TP2 < TP3 R:R multiples); a missing
            # rung contributes 0 payoff. The census table's TP_k maps to
            # the ladder's k-th rung.
            ladder_rrs = sorted(
                t["rr"]
                for t in (targets.get("targets") or [])
                if t.get("rr") is not None
            )
            while len(ladder_rrs) < 3:
                ladder_rrs.append(0.0)
            rungs = tuple(ladder_rrs[:3])
            ev_target_level = target_level_ev(
                dist.get("tp1"),
                dist.get("tp2"),
                dist.get("tp3"),
                dist.get("sl"),
                rungs=rungs,
                cost_r=cost_r,
            )
            expected_r_alloc = allocation_weighted_ev(
                dist.get("tp1"),
                dist.get("tp2"),
                dist.get("tp3"),
                dist.get("sl"),
                rungs=rungs,
                cost_r=cost_r,
            )
            cost_break_even = cost_break_even_r(
                dist.get("tp1"),
                dist.get("tp2"),
                dist.get("tp3"),
                dist.get("sl"),
                rungs=rungs,
            )
        # Ranking EV: conservative payoff (ladder best R:R when it clears
        # 1R, else the achieved RR). Reported for comparison - Stage-2
        # showed it can overstate target-level EV by ~20x (e.g. +2.18R
        # ranking vs -0.11R target-level on the 2026-08-14 GBPUSD long).
        if ladder_best is not None and ladder_best >= 1.0:
            payoff_basis = ladder_best
        elif rr is not None:
            payoff_basis = rr
        if payoff_basis is not None:
            expected_r = round(p * payoff_basis - (1.0 - p) * 1.0 - cost_r, 4)

    # BOOK-LEVEL rejection reasons (decision-relevant gates the EV winner
    # must pass): evidence bar, calibrated probability availability, EV,
    # R:R floor, macro, family validation. Engine confirmation is NOT a
    # book gate - the classifier's breakout/breakdown/retest families are
    # exactly what the engines cannot see - it is reported in ``reasons``
    # for context.
    rejection: List[str] = []
    if not family_map:
        rejection.append("no setup family cleared the evidence bar")
    # Stage-10 (spec #13): a FALSIFIED family must never carry a production
    # trade - the architecture may still *discover* the opportunity (both
    # sides visible), but the verdict caps it at FLAT.
    status = family_status(family)
    if status == FALSIFIED:
        rejection.append(
            f"family {family} is FALSIFIED (research: hypothesis rejected) - "
            "cannot trade this side"
        )
    if prob is None:
        rejection.append("no calibrated model probability (EV not computed)")
    elif payoff_basis is None:
        rejection.append(
            "no target ladder / payoff basis (EV not computed - would be fabricated)"
        )
    elif expected_r is not None and expected_r <= 0:
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
        entry_type=entry_type,
        invalidation=stop,
        target=target,
        rr=ladder_best if ladder_best is not None else rr,
        probability=prob,
        expected_r=expected_r,
        ev_target_level=ev_target_level,
        expected_r_alloc=expected_r_alloc,
        cost_break_even=cost_break_even,
        validation_status=status,
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
    min_ev: float = DEFAULT_MIN_EV_R,
    min_rr: float = DEFAULT_MIN_RR,
    min_ml_prob: float = DEFAULT_MIN_ML_PROB,
    spread_points: Optional[float] = None,
    slippage_pips: Optional[float] = None,
    pip_size: Optional[float] = None,
    tp_probs: Optional[Dict] = None,
) -> Dict:
    """Full opportunity book for one report: both sides + FLAT verdict.

    Returns ``{symbol, long, short, verdict, reasons}`` where ``verdict``
    is the decision engine's ``{direction, status, expected_r, reason}``.

    ``tp_probs`` optionally carries the empirical per-side target-level
    distribution ``{"long": {tp1, tp2, tp3, sl}, "short": {...}}`` from
    the census (``--write-probs``); when supplied, EV is computed from
    the true payoff distribution instead of the ladder-best approximation.

    Decision policy (EV-first, engine fallback):

    1. A side's EV is only computed when it has a **calibrated**
       probability. Without any calibrated probability the decision falls
       back to the engine-confirmed side (rule path).
    2. LONG wins when its EV > SHORT EV and EV > min_ev; SHORT wins on the
       mirror; neither clearing min_ev -> FLAT with explicit reasons.
    3. The winning side must not be macro-blocked and must clear the R:R
       floor when a ladder exists (the floors stay enforced on trades).
    """
    # tp_probs is the FULL census table (``{"long": {...}, "short": {...},
    # "families": {family: {...}}, "rungs_rr": [...]}``) - both sides
    # receive it so the per-family distribution can be selected inside
    # ``_side_opportunity`` (Stage-10: a candidate uses its OWN family's
    # first-touch table, never the pooled side average).
    symbol = str(report.get("symbol", "?"))
    tp_all = tp_probs or {}
    long_opp = _side_opportunity(
        report,
        "long",
        min_rr=min_rr,
        min_ml_prob=min_ml_prob,
        spread_points=spread_points,
        slippage_pips=slippage_pips,
        pip_size=pip_size,
        tp_probs=tp_all,
    )
    short_opp = _side_opportunity(
        report,
        "short",
        min_rr=min_rr,
        min_ml_prob=min_ml_prob,
        spread_points=spread_points,
        slippage_pips=slippage_pips,
        pip_size=pip_size,
        tp_probs=tp_all,
    )

    # Stage-10 decision basis: the allocation-aware target-level EV when it
    # exists (the honest economics of the scaling-out plan), else the
    # whole-ladder target-level EV, else the ranking EV. Ranking EV alone
    # (P x ladder-best) is never the primary decision variable - Stage-2/3
    # showed it overstates the payoff the ladder delivers.
    def _decision_ev(opp: Opportunity) -> Optional[float]:
        if opp.expected_r_alloc is not None:
            return opp.expected_r_alloc
        if opp.ev_target_level is not None:
            return opp.ev_target_level
        return opp.expected_r

    ev_l = _decision_ev(long_opp)
    ev_s = _decision_ev(short_opp)
    verdict_reasons: List[
        str
    ] = []  # Path 1: no calibrated probability on either side -> engine-confirmed
    # rule path (both existing engines already enforce structure gates).
    # When BOTH engines are confirmed, the higher engine score decides
    # (evidence-driven, never an arbitrary long-first list order). The rule
    # path also cannot promote a FALSIFIED family - the hard gate below
    # applies to the chosen side there too.
    if ev_l is None and ev_s is None:
        confirmed = [
            o
            for o in (long_opp, short_opp)
            if o.reasons and any("CONFIRMED" in r for r in o.reasons)
        ]
        if confirmed:
            chosen = max(confirmed, key=lambda o: _engine_score(report, o.direction))
            if chosen.rejection_reasons and any(
                "FALSIFIED" in r for r in chosen.rejection_reasons
            ):
                verdict = {
                    "direction": "flat",
                    "status": "FLAT",
                    "expected_r": None,
                    "reason": (
                        f"rule path: engine-confirmed {chosen.direction.upper()} "
                        "blocked - family FALSIFIED (research: hypothesis "
                        "rejected)"
                    ),
                }
                return _book(symbol, long_opp, short_opp, verdict)
            chosen.taken = True
            verdict_reasons.append(
                "rule path (no calibrated ML probability): engine-confirmed "
                f"{chosen.direction.upper()} "
                f"(score {_engine_score(report, chosen.direction)})"
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
        return _book(
            symbol, long_opp, short_opp, verdict
        )  # Path 2: EV-aware decision (at least one calibrated side).
    if ev_l is not None and ev_l > min_ev and (ev_s is None or ev_l >= ev_s):
        chosen, other = long_opp, short_opp
    elif ev_s is not None and ev_s > min_ev:
        chosen, other = short_opp, long_opp
    else:
        chosen = other = None

    if chosen is None:
        verdict = {
            "direction": "flat",
            "status": "FLAT",
            "expected_r": max(ev_l or 0.0, ev_s or 0.0),
            "reason": (
                f"no side clears EV floor (+{min_ev:g}R): "
                f"long {ev_l if ev_l is not None else 'n/a'}, "
                f"short {ev_s if ev_s is not None else 'n/a'}"
            ),
        }
        return _book(symbol, long_opp, short_opp, verdict)

    # The EV winner must still clear the book's hard floors (R:R + macro +
    # family validation). A FALSIFIED family winner is FLAT - the
    # architecture can discover it but never trade it (Stage-10 spec #13).
    if other is not None and (_decision_ev(other) or 0) > (_decision_ev(chosen) or 0):
        chosen, other = other, chosen
    hard_blocked = [
        r
        for r in chosen.rejection_reasons
        if "floor" in r or "BLOCKED" in r or "FALSIFIED" in r
    ]
    if hard_blocked:
        verdict = {
            "direction": "flat",
            "status": "FLAT",
            "expected_r": chosen.expected_r,
            "reason": (
                f"EV winner {chosen.direction.upper()} fails hard gate: "
                + "; ".join(hard_blocked)
            ),
        }
        return _book(symbol, long_opp, short_opp, verdict)

    # The winner passed every book gate - clear its (now inapplicable)
    # rejection reasons so the output is never self-contradictory.
    chosen.taken = True
    chosen.rejection_reasons = []
    verdict_reasons.append(
        f"EV path: {chosen.direction.upper()} "
        f"EV {_decision_ev(chosen):+.2f}R (target-level allocation) > "
        f"{'short' if chosen.direction == 'long' else 'long'} "
        f"{_decision_ev(other) if other is not None and _decision_ev(other) is not None else 'n/a'}"
    )
    # Stage-10 live/research separation (spec #13): the verdict's
    # ``validation_status`` carries the family's research classification
    # (PRODUCTION-VALIDATED / PROMISING-SHADOW-ONLY / UNVALIDATED /
    # FALSIFIED). A non-PRODUCTION-VALIDATED winner is a SHADOW-ONLY
    # research candidate - the decision mechanism (TRADE/RULE/CONFIRMED)
    # is unchanged, but consumers (plan, alerts, audit) must label it
    # SHADOW-ONLY and never treat it as production-validated alpha. No
    # family is PRODUCTION-VALIDATED yet: the frozen Stage-9 LONG reversal
    # is PROMISING/SHADOW-ONLY until the fresh-window gate closes.
    verdict_reasons.append(
        f"family {chosen.setup_family} is {chosen.validation_status} "
        "(SHADOW-ONLY unless PRODUCTION-VALIDATED)"
    )
    verdict = {
        "direction": chosen.direction,
        "status": "TRADE" if chosen.probability is not None else "RULE",
        "expected_r": chosen.expected_r,
        "validation_status": chosen.validation_status,
        "shadow_only": chosen.validation_status != PRODUCTION_VALIDATED,
        "reason": "; ".join(verdict_reasons),
    }
    return _book(symbol, long_opp, short_opp, verdict)


def _engine_score(report: Dict, direction: str) -> int:
    """Engine confirmation score for a side (dip_score / rally_score)."""
    key = "dip" if direction == "long" else "rally"
    return int(
        (report.get(key) or {}).get(
            "dip_score" if direction == "long" else "rally_score", 0
        )
    )


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
        lines.append(
            f"  probability: {'-' if opp.get('probability') is None else f'{opp['probability']:.0%}'}"
        )
        lines.append(
            f"  EV(ranking): {'-' if opp.get('expected_r') is None else f'{opp['expected_r']:+.2f}R'}  "
            f"(comparison only)"
        )
        if opp.get("expected_r_alloc") is not None:
            lines.append(
                f"  EV(alloc)  : {opp['expected_r_alloc']:+.2f}R "
                f"(target-level, 1/3-1/3-1/3 allocation - DECISION EV)"
            )
        elif opp.get("ev_target_level") is not None:
            lines.append(
                f"  EV(tl)     : {opp['ev_target_level']:+.2f}R (target-level, spec #4)"
            )
        if opp.get("cost_break_even") is not None:
            lines.append(
                f"  cost break-even: {opp['cost_break_even']:.3f}R "
                f"(edge gone above this - realistic {opp['cost_r']:.3f}R)"
            )
        lines.append(
            f"  R:R        : {'-' if opp.get('rr') is None else f'{opp['rr']:.2f}'} "
            f"(ladder best, supplementary) · cost {opp['cost_r']:.3f}R · "
            f"family {opp.get('validation_status') or '-'}"
        )
        if opp.get("entry_zone"):
            kind = (opp.get("entry_type") or "limit").upper()
            lines.append(
                f"  entry      : {kind} {opp['entry_zone'][0]:,.5f} · stop "
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
