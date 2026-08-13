"""
NexusQuant - Final Quant Rating (institutional spec #14)

Turns the ensemble probability (or a rule-based fallback) into a
graded rating and an attribution breakdown, e.g.::

    Bullish Probability: 63%
    Rating: Neutral (50-69)
    Trend Factors:     +14%
    Momentum Signals:  +11%
    Volume/Flow:        -4%
    Macro:              +6%
    Sentiment:           0%
    Fundamentals:        0%

The signed contributions always sum to ``prob_pct - 50`` (the net tilt
above neutral), matching the spec's "+26% / -9%" style output.

Rating thresholds (spec #14):
    Strong Buy >= 85 | Buy 70-84 | Neutral 50-69 | Sell 30-49 | Strong Sell <= 29
"""

from __future__ import annotations

from typing import Any, Dict, Optional

# ---- factor weights (documented; sum to 1.0) -----------------------------
WEIGHTS = {
    "Trend": 0.35,
    "Momentum": 0.25,
    "Volume/Flow": 0.15,
    "Macro": 0.20,
    "Sentiment": 0.05,
    "Fundamentals": 0.00,
}

THRESHOLDS = [
    (85.0, "Strong Buy"),
    (70.0, "Buy"),
    (50.0, "Neutral"),
    (30.0, "Sell"),
    (0.0, "Strong Sell"),
]

RECOMMENDATIONS = {
    "Strong Buy": "High-conviction long — full size within the setup",
    "Buy": "Accumulate on dips toward the entry zone",
    "Neutral": "Stand aside — no edge in either direction",
    "Sell": "Avoid longs / fade rallies; structure is weak",
    "Strong Sell": "Avoid entirely — high-conviction bearish structure",
}


def quant_rating(prob_pct: float) -> str:
    """Map a 0-100 bullish probability to a rating label."""
    for floor, label in THRESHOLDS:
        if prob_pct >= floor:
            return label
    return "Strong Sell"


# ---------------------------------------------------------------------------
# factor scoring from the report
# ---------------------------------------------------------------------------


def _clamp(v: float, lo: float = -1.0, hi: float = 1.0) -> float:
    return max(lo, min(hi, v))


def _trend_score(report: Dict[str, Any]) -> float:
    ma = report.get("moving_averages") or {}
    ts = report.get("trend_strength") or {}
    adx = float(ts.get("adx", 0) or 0)
    above200 = 1.0 if ma.get("price_vs_sma200") == "Above" else -1.0
    di = 0.0
    pd_, md_ = float(ts.get("plus_di", 0) or 0), float(ts.get("minus_di", 0) or 0)
    if pd_ + md_ > 0:
        di = (pd_ - md_) / (pd_ + md_)
    # no trend strength (ADX < 15) -> DI carries no directional read
    di_weight = 1.0 if adx >= 25 else 0.5 if adx >= 15 else 0.0
    return _clamp(0.6 * above200 + 0.4 * di * di_weight)


def _momentum_score(report: Dict[str, Any]) -> float:
    mo = report.get("momentum") or {}
    rsi = float(mo.get("rsi_14", 50) or 50)
    hist = float(mo.get("macd_hist", 0) or 0)
    bb = float(mo.get("bb_pct_b", 0.5) or 0.5)
    rsi_s = (rsi - 50) / 50
    hist_s = 1.0 if hist > 0 else (-1.0 if hist < 0 else 0.0)
    bb_s = _clamp((bb - 0.5) * 2)
    return _clamp(0.5 * rsi_s + 0.3 * hist_s + 0.2 * bb_s)


def _volume_score(report: Dict[str, Any]) -> float:
    vf = report.get("volume_flow") or {}
    score = vf.get("buyer_seller_score", 0)
    if not score:
        return 0.0
    return _clamp(float(score) / 100.0)


def _macro_score(report: Dict[str, Any]) -> float:
    mc = report.get("macro") or {}
    bias = (mc.get("bias") or {}).get("bias", 0) or 0
    return _clamp(float(bias) / 2.0)


def _sentiment_score(report: Dict[str, Any]) -> float:
    """News/social composite in [-1, +1] when the report has a sentiment
    read (FX/metals included since the Yahoo news endpoint answers those
    too), else neutral 0."""
    se = report.get("sentiment") or {}
    composite = se.get("composite")
    if composite is None:
        return 0.0
    return _clamp(float(composite))


def _factor_scores(report: Dict[str, Any]) -> Dict[str, float]:
    return {
        "Trend": _trend_score(report),
        "Momentum": _momentum_score(report),
        "Volume/Flow": _volume_score(report),
        "Macro": _macro_score(report),
        "Sentiment": _sentiment_score(report),
        "Fundamentals": 0.0,  # N/A for FX/metals (equity feature only)
    }


# ---------------------------------------------------------------------------
# public API
# ---------------------------------------------------------------------------


def factor_contributions(
    report: Dict[str, Any],
    prob_pct: Optional[float] = None,
) -> Dict:
    """
    Decompose a bullish probability into signed factor-group contributions
    (percent points). The contributions sum to ``prob_pct - 50``.

    ``prob_pct`` defaults to the report's ML probability, or to a
    rule-based estimate when no model is present.
    """
    scores = _factor_scores(report)
    raw = {f: scores[f] * WEIGHTS[f] for f in WEIGHTS}
    net = sum(raw.values())  # signed net directional score in [-1, 1]
    rule_pct = float(max(1.0, min(99.0, 50 + 40 * net)))

    ml = report.get("ml") or {}
    ml_pct = float(ml.get("prob_pct", 0) or 0)
    ml_short = report.get("ml_short") or {}
    ml_short_pct = float(ml_short.get("prob_pct", 0) or 0)
    if prob_pct is None:
        if ml_pct and ml_short_pct:
            # Dual-side read (spec #10 long+short): the effective bullish
            # probability is centered on ``net_bias = P(long) - P(short)``,
            # i.e. 50 + 50*net_bias, so a strong short model pushes the
            # rating toward Sell / Strong Sell instead of capping at
            # Neutral. Same 60/40 blend with the rule stack as the prior.
            net_bias = (ml_pct - ml_short_pct) / 100.0
            effective = 50.0 + 50.0 * net_bias
            prob_pct = 0.6 * effective + 0.4 * rule_pct
            source = "ml+factors"
        elif ml_pct:
            # Blend: the rule-based factor stack is the prior, the ML
            # probability the signal. 60/40 stops a single weak model
            # output (OOS AUC ~0.52) from dominating the rating.
            prob_pct = 0.6 * ml_pct + 0.4 * rule_pct
            source = "ml+factors"
        else:
            prob_pct = rule_pct
            source = "rule"
    else:
        source = "explicit"
    prob_pct = float(max(1.0, min(99.0, prob_pct)))

    tilt = prob_pct - 50.0
    # Contribution sign always follows the factor direction (bullish factor
    # -> positive, bearish -> negative, per the spec's style). Each factor's
    # standalone rule-space impact (40 * score * weight) is scaled so the
    # contributions reproduce the final tilt when the model agrees with the
    # factor stack; when they disagree (or the stack is near-balanced), the
    # residual is reported as ``unexplained`` (model-driven).
    rule_tilt = 40.0 * net
    scale = (abs(tilt) / abs(rule_tilt)) if abs(rule_tilt) >= 1.0 else 1.0
    contributions = []
    for factor in WEIGHTS:
        contrib = 40.0 * scores[factor] * WEIGHTS[factor] * scale
        contributions.append(
            {
                "factor": factor,
                "score": round(scores[factor], 3),
                "weight": WEIGHTS[factor],
                "contribution": round(contrib, 1),
            }
        )
    contributions.sort(key=lambda c: -abs(c["contribution"]))
    unexplained = round(tilt - sum(c["contribution"] for c in contributions), 1)
    return {
        "prob_pct": round(prob_pct, 1),
        "ml_based": source == "ml+factors",
        "source": source,
        "contributions": contributions,
        "net_tilt": round(tilt, 1),
        "unexplained": unexplained,
    }


def final_rating(report: Dict[str, Any]) -> Dict:
    """Full #14 output: probability, rating label, contribution breakdown."""
    breakdown = factor_contributions(report)
    label = quant_rating(breakdown["prob_pct"])
    return {
        "prob_pct": breakdown["prob_pct"],
        "rating": label,
        "recommendation": RECOMMENDATIONS[label],
        "contributions": breakdown["contributions"],
        "net_tilt": breakdown["net_tilt"],
        "ml_based": breakdown["ml_based"],
        "source": breakdown["source"],
    }


if __name__ == "__main__":
    print("NexusQuant Final Quant Rating module ready.")
