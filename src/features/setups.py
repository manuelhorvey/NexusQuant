"""
NexusQuant - Direction-Neutral Setup Classifier

The two confirmation engines (dip.py / rally.py) are deliberately
*structure-gated*: the dip engine only fires above the 200-SMA, the rally
engine only below it. That is correct for counter-trend pullback trading,
but it means the architecture can only express:

    "find a bullish structure, buy weakness"  /  "find a bearish
    structure, sell strength"

This module is the direction-neutral layer that sits *above* the engines.
It classifies ANY bar into a setup family from an explicit taxonomy and
produces **independent** Long Evidence and Short Evidence scores - not
mirror images of each other - so a breakdown-retest short can fire while
price is still above the 200-SMA, and a breakout-retest long can fire
below it, when the other evidence justifies it.

Pipeline position:

    MARKET DATA -> FEATURES -> REGIME -> SETUP CLASSIFIER (this module)
    -> LONG/SHORT EVIDENCE SCORES -> ML PROBABILITIES -> EV -> RISK
    -> LONG / SHORT / FLAT

Setup taxonomy (each family is independently interpretable and validated):

    LONG:  LONG_TREND_CONTINUATION, LONG_BUY_DIP, LONG_BREAKOUT,
           LONG_BREAKOUT_RETEST, LONG_REVERSAL, LONG_MEAN_REVERSION
    SHORT: SHORT_TREND_CONTINUATION, SHORT_SELL_RALLY, SHORT_BREAKDOWN,
           SHORT_BREAKDOWN_RETEST, SHORT_REVERSAL, SHORT_MEAN_REVERSION

The 200-SMA relationship contributes to the *regime context* (one factor
among many) and never gates a family by itself. The classifier is causal
by construction: it only reads the latest bar's indicator values (all
rolling windows) plus precomputed structural primitives (swings, levels,
divergences) that are themselves causal (swings lag their confirmation
window, divergences are confirmed pivots only).

Usage (library):

    from src.features.setups import classify_setup
    setup = classify_setup(df, dip=dip, rally=rally, ml=ml, ...)
"""

from __future__ import annotations

from typing import Dict, List, Optional

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Taxonomy
# ---------------------------------------------------------------------------

LONG_FAMILIES = [
    "LONG_TREND_CONTINUATION",
    "LONG_BUY_DIP",
    "LONG_BREAKOUT",
    "LONG_BREAKOUT_RETEST",
    "LONG_REVERSAL",
    "LONG_MEAN_REVERSION",
]

SHORT_FAMILIES = [
    "SHORT_TREND_CONTINUATION",
    "SHORT_SELL_RALLY",
    "SHORT_BREAKDOWN",
    "SHORT_BREAKDOWN_RETEST",
    "SHORT_REVERSAL",
    "SHORT_MEAN_REVERSION",
]

ALL_FAMILIES = LONG_FAMILIES + SHORT_FAMILIES

# Number of bars for the breakout/breakdown lookback window.
BREAK_WINDOW = 60
RETEST_WINDOW = 10  # a break within the last N bars is "recent" (retest-able)
VOLUME_LOOKBACK = 20

# ---------------------------------------------------------------------------
# Evidence primitives (all causal: latest bar + rolling windows)
# ---------------------------------------------------------------------------


def _nan0(value, default: float = 0.0) -> float:
    try:
        v = float(value)
        return v if v == v else default
    except (TypeError, ValueError):
        return default


def _trend_context(latest: Dict) -> Dict:
    """Trend evidence from the latest bar: MA stack, ADX, DI, vs-SMA200.

    The SMA200 relationship is contextual (a 0..1 factor), never a gate.
    """
    close = _nan0(latest.get("close"))
    sma20 = latest.get("sma_20")
    sma50 = latest.get("sma_50")
    sma200 = latest.get("sma_200")
    adx = _nan0(latest.get("adx"))
    plus_di = _nan0(latest.get("plus_di"))
    minus_di = _nan0(latest.get("minus_di"))

    above200 = bool(sma200 is not None and close > _nan0(sma200))
    bull_stack = bool(
        sma20 is not None
        and sma50 is not None
        and sma200 is not None
        and _nan0(sma20) > _nan0(sma50) > _nan0(sma200)
    )
    bear_stack = bool(
        sma20 is not None
        and sma50 is not None
        and sma200 is not None
        and _nan0(sma20) < _nan0(sma50) < _nan0(sma200)
    )
    strong_trend = adx >= 25.0
    di_long = plus_di > minus_di
    di_short = minus_di > plus_di

    return {
        "above_sma200": above200,
        "bull_stack": bull_stack,
        "bear_stack": bear_stack,
        "adx": adx,
        "strong_trend": strong_trend,
        "di_long": di_long,
        "di_short": di_short,
    }


def _momentum_context(df: pd.DataFrame) -> Dict:
    """Momentum evidence: RSI level/slope, MACD sign/slope, Bollinger pos."""
    latest = df.iloc[-1]
    rsi14 = _nan0(latest.get("rsi_14"), 50.0)
    macd_hist = _nan0(latest.get("macd_hist"))
    macd_hist_prev = _nan0(latest.get("macd_hist"), macd_hist)
    if len(df) >= 2:
        macd_hist_prev = _nan0(df["macd_hist"].iloc[-2], macd_hist_prev)
    rsi_prev = rsi14
    if "rsi_14" in df.columns and len(df) >= 2:
        rsi_prev = _nan0(df["rsi_14"].iloc[-2], rsi14)

    bb_pos = None
    if "bb_lower" in latest and "bb_upper" in latest:
        lo, up = _nan0(latest.get("bb_lower")), _nan0(latest.get("bb_upper"))
        if up > lo:
            bb_pos = (_nan0(latest.get("close")) - lo) / (up - lo)

    z = None
    close = df["close"]
    if len(close) >= 20:
        z = float((close.iloc[-1] - close.tail(20).mean()) / close.tail(20).std())
        z = z if z == z else None

    return {
        "rsi": rsi14,
        "rsi_oversold": rsi14 <= 30.0,
        "rsi_overbought": rsi14 >= 70.0,
        "rsi_rising": rsi14 > rsi_prev,
        "macd_positive": macd_hist > 0,
        "macd_negative": macd_hist < 0,
        "macd_rising": macd_hist > macd_hist_prev,
        "bb_pos": bb_pos,
        "bb_lower_touch": bb_pos is not None and bb_pos <= 0.05,
        "bb_upper_touch": bb_pos is not None and bb_pos >= 0.95,
        "zscore": z,
    }


def _structure_context(df: pd.DataFrame, levels: Optional[Dict]) -> Dict:
    """Structure evidence: swing H/L breakdowns, retests, support/resistance."""
    close = float(df["close"].iloc[-1])
    if len(df) >= BREAK_WINDOW:
        win = df.iloc[-BREAK_WINDOW:-1]  # exclude the current bar (causal)
        prior_high = float(win["high"].max())
        prior_low = float(win["low"].min())
    else:
        prior_high = float(df["high"].iloc[:-1].max())
        prior_low = float(df["low"].iloc[:-1].min())

    # Breakout / breakdown on the current bar vs the prior window.
    breakout = bool(prior_high > 0 and close > prior_high)
    breakdown = bool(
        prior_low > 0 and close < prior_low
    )  # Recent break (within RETEST_WINDOW bars) -> retest candidate.
    recent_break_high = recent_break_low = False
    if len(df) > RETEST_WINDOW:
        tail = df.iloc[-RETEST_WINDOW - 1 : -1]
        recent_break_high = bool(
            len(tail) > 0 and float(tail["close"].max()) > prior_high
        )
        recent_break_low = bool(
            len(tail) > 0 and float(tail["close"].min()) < prior_low
        )

    # A retest is only a retest when price is BACK at the broken level
    # (within ~1 ATR) - NOT at an arbitrary confluence level. This is the
    # key structural difference between a breakdown-retest short and a
    # generic "at resistance" fade.
    atr = _nan0(df["atr_14"].iloc[-1]) or 0.0
    retest_tol = max(atr, _nan0(df["close"].iloc[-1]) * 0.005)
    retesting_broken_high = bool(
        recent_break_high and prior_high > 0 and abs(close - prior_high) <= retest_tol
    )
    retesting_broken_low = bool(
        recent_break_low and prior_low > 0 and abs(close - prior_low) <= retest_tol
    )

    support = resistance = None
    tol = None
    if levels:
        s = levels.get("nearest_support")
        r = levels.get("nearest_resistance")
        support = float(s["price"]) if s else None
        resistance = float(r["price"]) if r else None
        tol = levels.get("tolerance")

    at_support = at_resistance = False
    if support is not None:
        t = _nan0(tol) or atr or 0.0
        at_support = bool(support <= close <= support + t)
    if resistance is not None:
        t = _nan0(tol) or atr or 0.0
        at_resistance = bool(resistance - t <= close <= resistance)

    return {
        "breakout": breakout,
        "breakdown": breakdown,
        "recent_break_high": recent_break_high,
        "recent_break_low": recent_break_low,
        "retesting_broken_high": retesting_broken_high,
        "retesting_broken_low": retesting_broken_low,
        "at_support": at_support,
        "at_resistance": at_resistance,
        "support": support,
        "resistance": resistance,
    }


def _volume_context(df: pd.DataFrame) -> Dict:
    """Volume evidence: OBV slope and relative volume (causal rolling)."""
    if "volume" not in df.columns or len(df) < VOLUME_LOOKBACK:
        return {"obv_rising": False, "obv_falling": False, "rel_vol": None}
    vol = df["volume"].astype(float)
    rel = float(vol.iloc[-1] / max(vol.tail(VOLUME_LOOKBACK).mean(), 1e-9))
    obv = (np.sign(df["close"].diff().fillna(0.0)) * vol).cumsum()
    if len(obv) >= 10:
        obv_slope = obv.iloc[-1] - obv.iloc[-10]
    else:
        obv_slope = 0.0
    return {
        "obv_rising": obv_slope > 0,
        "obv_falling": obv_slope < 0,
        "rel_vol": float(rel),
    }


def _divergence_context(divergence: Optional[Dict]) -> Dict:
    """Latest confirmed divergences (already causal: confirmed pivots only)."""
    d = divergence or {}
    sigs = d.get("signals", []) if isinstance(d, dict) else []
    latest_date = None
    if isinstance(d, dict) and d.get("dates") is not None:
        try:
            dates = list(d["dates"])
            latest_date = dates[-1] if dates else None
        except (TypeError, IndexError):
            latest_date = None

    bull_reg = bull_hid = bear_reg = bear_hid = False
    for s in sigs:
        if isinstance(s, dict):
            kind = str(s.get("kind", ""))
            side = str(s.get("side", ""))
            if side == "bullish" and "regular" in kind:
                bull_reg = True
            elif side == "bullish" and "hidden" in kind:
                bull_hid = True
            elif side == "bearish" and "regular" in kind:
                bear_reg = True
            elif side == "bearish" and "hidden" in kind:
                bear_hid = True
    return {
        "bull_reg_div": bull_reg,
        "bull_hid_div": bull_hid,
        "bear_reg_div": bear_reg,
        "bear_hid_div": bear_hid,
        "latest_signal_date": latest_date,
    }


def _pattern_context(pattern: Optional[Dict]) -> Dict:
    """Latest structural pattern side (patterns are confirmed, ≥65% rule)."""
    p = pattern or {}
    if isinstance(p, dict):
        return {
            "pattern_side": p.get("side"),
            "pattern_name": p.get("name"),
            "pattern_conf": p.get("confidence"),
        }
    return {"pattern_side": None, "pattern_name": None, "pattern_conf": None}


def _regime_context(regime_label: Optional[str]) -> Dict:
    r = str(regime_label or "")
    return {
        "bull_regime": r == "Bull Trend",
        "bear_regime": r == "Bear Trend",
        "range_regime": r == "Range / Chop",
        "highvol_regime": r == "High Volatility",
    }


# ---------------------------------------------------------------------------
# Scoring
# ---------------------------------------------------------------------------


def _score_setup_family(
    name: str,
    checks: List[bool],
    weights: List[float],
) -> Dict:
    """Weighted evidence score in [0, 1] plus the matched check list."""
    total_w = sum(weights) or 1.0
    score = sum(w for c, w in zip(checks, weights, strict=True) if c) / total_w
    matched = [c for c, _ in zip(checks, weights, strict=True) if c]
    return {"family": name, "score": round(float(score), 3), "matched": len(matched)}


def _evaluate_families(
    t: Dict, m: Dict, s: Dict, v: Dict, d: Dict, p: Dict, rg: Dict
) -> Dict:
    """Score every setup family from the shared evidence context.

    Each family has its own causal logic; families are *not* mirror images.
    E.g. a breakdown-retest short needs a recent break + retest + rejection,
    which has no symmetric long requirement.
    """
    out: Dict[str, Dict] = {}

    # ---- LONG families ---------------------------------------------------
    # Trend-continuation requires MOMENTUM ALIGNMENT (MACD/RSI/DI all
    # agreeing with the direction) - price-vs-SMA alone is context, never a
    # continuation signal. This is what keeps a +2-bias symbol below the
    # 200-SMA out of SHORT_TREND_CONTINUATION and a -2-bias symbol above it
    # out of LONG_TREND_CONTINUATION.
    out["LONG_TREND_CONTINUATION"] = _score_setup_family(
        "LONG_TREND_CONTINUATION",
        [
            t["bull_stack"] or (t["above_sma200"] and m["macd_positive"]),
            t["strong_trend"] and t["di_long"],
            m["macd_positive"] and m["macd_rising"],
            m["rsi"] > 50.0,
            v["obv_rising"],
        ],
        [1.0, 1.0, 0.8, 0.5, 0.5],
    )

    out["LONG_BUY_DIP"] = _score_setup_family(
        "LONG_BUY_DIP",
        [
            t["above_sma200"] or t["bull_stack"],
            m["rsi"] <= 55.0,  # cooled
            s["at_support"],
            m["bb_lower_touch"],
            m["macd_rising"] or m["rsi_rising"],
        ],
        [1.0, 0.8, 1.0, 0.6, 0.8],
    )

    out["LONG_BREAKOUT"] = _score_setup_family(
        "LONG_BREAKOUT",
        [
            s["breakout"],
            m["macd_positive"],
            m["rsi"] > 50.0,
            v["rel_vol"] is not None and v["rel_vol"] > 1.0,
            t["above_sma200"] or rg["bull_regime"] or rg["range_regime"],
        ],
        [2.0, 0.8, 0.6, 0.8, 0.6],
    )

    out["LONG_BREAKOUT_RETEST"] = _score_setup_family(
        "LONG_BREAKOUT_RETEST",
        [
            s["recent_break_high"],  # broke out within the last N bars
            s["retesting_broken_high"],  # back AT the broken level (1 ATR)
            not m["rsi_overbought"],
            m["macd_rising"] or m["rsi_rising"],
            not s["breakdown"],
        ],
        [1.5, 1.5, 0.6, 0.8, 0.6],
    )

    out["LONG_REVERSAL"] = _score_setup_family(
        "LONG_REVERSAL",
        [
            d["bull_reg_div"],
            s["recent_break_low"] and not s["breakdown"],  # failed breakdown
            m["rsi_oversold"] and m["rsi_rising"],
            not t["above_sma200"],  # reversing below the 200-SMA is the point
            p["pattern_side"] == "bullish",
        ],
        [1.5, 1.2, 0.8, 0.4, 0.8],
    )

    out["LONG_MEAN_REVERSION"] = _score_setup_family(
        "LONG_MEAN_REVERSION",
        [
            m["rsi_oversold"],
            m["bb_lower_touch"],
            m["zscore"] is not None and m["zscore"] < -2.0,
            rg["range_regime"],
            not t["strong_trend"],
        ],
        [1.2, 1.0, 0.8, 0.6, 0.6],
    )

    # ---- SHORT families --------------------------------------------------
    out["SHORT_TREND_CONTINUATION"] = _score_setup_family(
        "SHORT_TREND_CONTINUATION",
        [
            t["bear_stack"] or (not t["above_sma200"] and m["macd_negative"]),
            t["strong_trend"] and t["di_short"],
            m["macd_negative"] and not m["macd_rising"],
            m["rsi"] < 50.0,
            v["obv_falling"],
        ],
        [1.0, 1.0, 0.8, 0.5, 0.5],
    )

    out["SHORT_SELL_RALLY"] = _score_setup_family(
        "SHORT_SELL_RALLY",
        [
            not t["above_sma200"] or t["bear_stack"],
            m["rsi"] >= 60.0,  # stretched in a downtrend
            s["at_resistance"],
            m["bb_upper_touch"],
            not m["macd_rising"],
        ],
        [1.0, 0.8, 1.0, 0.6, 0.8],
    )

    out["SHORT_BREAKDOWN"] = _score_setup_family(
        "SHORT_BREAKDOWN",
        [
            s["breakdown"],
            m["macd_negative"],
            m["rsi"] < 50.0,
            v["rel_vol"] is not None and v["rel_vol"] > 1.0,
            not t["above_sma200"] or rg["bear_regime"] or rg["range_regime"],
        ],
        [2.0, 0.8, 0.6, 0.8, 0.6],
    )

    out["SHORT_BREAKDOWN_RETEST"] = _score_setup_family(
        "SHORT_BREAKDOWN_RETEST",
        [
            s["recent_break_low"],  # broke down within the last N bars
            s["retesting_broken_low"],  # back AT the broken level (1 ATR)
            not m["rsi_oversold"],
            not m["macd_rising"],  # bearish confirmation, not a bounce
            not s["breakout"],
        ],
        [1.5, 1.5, 0.6, 0.8, 0.6],
    )

    out["SHORT_REVERSAL"] = _score_setup_family(
        "SHORT_REVERSAL",
        [
            d["bear_reg_div"],
            s["recent_break_high"] and not s["breakout"],  # failed breakout
            m["rsi_overbought"] and not m["rsi_rising"],
            t["above_sma200"],  # can reverse ABOVE the 200-SMA (failed high)
            p["pattern_side"] == "bearish",
        ],
        [1.5, 1.2, 0.8, 0.4, 0.8],
    )

    out["SHORT_MEAN_REVERSION"] = _score_setup_family(
        "SHORT_MEAN_REVERSION",
        [
            m["rsi_overbought"],
            m["bb_upper_touch"],
            m["zscore"] is not None and m["zscore"] > 2.0,
            rg["range_regime"],
            not t["strong_trend"],
        ],
        [1.2, 1.0, 0.8, 0.6, 0.6],
    )

    return out


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def classify_setup(
    df: pd.DataFrame,
    levels: Optional[Dict] = None,
    dip: Optional[Dict] = None,
    rally: Optional[Dict] = None,
    ml: Optional[Dict] = None,
    divergence: Optional[Dict] = None,
    pattern: Optional[Dict] = None,
    macro: Optional[Dict] = None,
    regime_label: Optional[str] = None,
) -> Dict:
    """Direction-neutral setup classification for the latest bar.

    Returns a dict with independent long/short evidence scores, the best
    family per side, a direction verdict (long/short/flat), and an
    explainability trail (``evidence``).

    ``dip`` / ``rally`` (the engine outputs) are folded in as *tie-breakers*
    and confirmation context - they do NOT gate this classifier.
    """
    if len(df) == 0:
        return {"direction": "flat", "long_score": 0.0, "short_score": 0.0}

    latest = df.iloc[-1].to_dict()
    t = _trend_context(latest)
    m = _momentum_context(df)
    s = _structure_context(df, levels)
    v = _volume_context(df)
    d = _divergence_context(divergence)
    p = _pattern_context(pattern)
    rg = _regime_context(regime_label)

    families = _evaluate_families(t, m, s, v, d, p, rg)

    long_scores = {f: families[f]["score"] for f in LONG_FAMILIES}
    short_scores = {
        f: families[f]["score"] for f in SHORT_FAMILIES
    }  # Engine outputs are *confirmation context*, used two ways:
    # 1. A confirmed engine setup BOOSTS its matching pullback family.
    # 2. An engine that explicitly rejects the pullback structure VETOES the
    #    matching family (the engines gate their own pullback families; the
    #    classifier's other families - breakout / breakdown / retest /
    #    reversal / mean-reversion - are exactly what the engines cannot
    #    see, so they stay unconstrained by the structure gate). This
    #    prevents e.g. LONG_BUY_DIP firing on a symbol the dip engine reads
    #    as "No Uptrend".
    if dip:
        if dip.get("dip_confirmed"):
            long_scores["LONG_BUY_DIP"] = max(long_scores["LONG_BUY_DIP"], 1.0)
        elif dip.get("dip_score", 0) >= 4:
            long_scores["LONG_BUY_DIP"] = max(long_scores["LONG_BUY_DIP"], 0.6)
        if dip.get("dip_stage") == "No Uptrend":
            long_scores["LONG_BUY_DIP"] = min(long_scores["LONG_BUY_DIP"], 0.25)
    if rally:
        if rally.get("rally_confirmed"):
            short_scores["SHORT_SELL_RALLY"] = max(
                short_scores["SHORT_SELL_RALLY"], 1.0
            )
        elif rally.get("rally_score", 0) >= 4:
            short_scores["SHORT_SELL_RALLY"] = max(
                short_scores["SHORT_SELL_RALLY"], 0.6
            )
        if rally.get("rally_stage") == "No Downtrend":
            short_scores["SHORT_SELL_RALLY"] = min(
                short_scores["SHORT_SELL_RALLY"], 0.25
            )

    best_long = max(long_scores.items(), key=lambda kv: kv[1])
    best_short = max(short_scores.items(), key=lambda kv: kv[1])

    long_score, short_score = best_long[1], best_short[1]

    # Direction verdict: evidence wins over the engines. FLAT when neither
    # side clears the minimum evidence bar.
    MIN_EVIDENCE = 0.35
    if long_score >= MIN_EVIDENCE and long_score >= short_score:
        direction = "long"
    elif short_score >= MIN_EVIDENCE and short_score > long_score:
        direction = "short"
    else:
        direction = "flat"

    family = (
        best_long[0]
        if direction == "long"
        else (best_short[0] if direction == "short" else None)
    )

    # Blend calibrated ML probabilities when present (they are per-side).
    prob_long = prob_short = None
    if ml:
        prob_long = ml.get("prob_long") or ml.get("prob")
        prob_short = ml.get("prob_short")
    if prob_long is None and prob_short is None:
        # Evidence-normalized fallback (NOT a calibrated probability - the
        # caller must prefer the calibrated ML when available).
        total = long_score + short_score
        if total > 0:
            prob_long = long_score / total
            prob_short = short_score / total

    evidence = _build_evidence(t, m, s, d, p, rg, families, direction)

    return {
        "direction": direction,
        "setup_family": family,
        "long_families": dict(sorted(long_scores.items(), key=lambda kv: -kv[1])),
        "short_families": dict(sorted(short_scores.items(), key=lambda kv: -kv[1])),
        "long_score": round(float(long_score), 3),
        "short_score": round(float(short_score), 3),
        "prob_long": float(prob_long) if prob_long is not None else None,
        "prob_short": float(prob_short) if prob_short is not None else None,
        "confidence": round(float(max(long_score, short_score)), 3),
        "evidence": evidence,
        "macro": macro,
    }


def _build_evidence(t, m, s, d, p, rg, families, direction) -> List[str]:
    ev: List[str] = []
    if t["bull_stack"]:
        ev.append("bullish MA stack (20>50>200)")
    if t["bear_stack"]:
        ev.append("bearish MA stack (20<50<200)")
    if t["strong_trend"]:
        ev.append(f"strong trend ADX {t['adx']:.0f}")
    if m["rsi_oversold"]:
        ev.append(f"RSI oversold ({m['rsi']:.0f})")
    if m["rsi_overbought"]:
        ev.append(f"RSI overbought ({m['rsi']:.0f})")
    if s["breakout"]:
        ev.append("close above prior 60-bar high (breakout)")
    if s["breakdown"]:
        ev.append("close below prior 60-bar low (breakdown)")
    if s["recent_break_high"] and not s["breakout"]:
        ev.append("recent upside break - retest candidate")
    if s["recent_break_low"] and not s["breakdown"]:
        ev.append("recent downside break - retest candidate")
    if s["at_support"]:
        ev.append("price at confluence support")
    if s["at_resistance"]:
        ev.append("price at confluence resistance")
    if d["bull_reg_div"]:
        ev.append("regular bullish divergence")
    if d["bear_reg_div"]:
        ev.append("regular bearish divergence")
    if d["bull_hid_div"]:
        ev.append("hidden bullish divergence")
    if d["bear_hid_div"]:
        ev.append("hidden bearish divergence")
    if p["pattern_side"]:
        ev.append(f"{p['pattern_name']} ({p['pattern_side']})")
    if rg["bull_regime"]:
        ev.append("Bull Trend regime")
    if rg["bear_regime"]:
        ev.append("Bear Trend regime")
    if families and direction:
        top = families.get(direction, {})
        if top:
            ev.append(f"best family {top['family']} ({top['score']:.2f})")
    return ev[:12]


def expected_value(
    prob_win: Optional[float],
    avg_win_r: float,
    avg_loss_r: float = -1.0,
    cost_r: float = 0.0,
) -> Optional[float]:
    """Expected value in R units: P(win)*avg_win + P(loss)*avg_loss - cost.

    ``prob_win`` must be a calibrated probability (see model.predict_*).
    Returns None when no probability is available (EV would be fabricated).
    """
    if prob_win is None:
        return None
    p = float(np.clip(prob_win, 0.0, 1.0))
    return round(p * avg_win_r + (1 - p) * avg_loss_r - cost_r, 4)


def probability_weighted_rr(
    targets: List[Dict], prob_win: Optional[float], cost_r: float = 0.0
) -> Optional[float]:
    """Probability-weighted R:R from a target ladder.

    E[win] = sum over targets of p_reach * rr, using a geometric decay for
    the probability of reaching each further target (the only estimator we
    have without a calibrated per-target model; documented as approximate).
    """
    if not targets or prob_win is None:
        return None
    p = float(np.clip(prob_win, 0.0, 1.0))
    rrs = [t.get("rr", 0.0) for t in targets if t.get("rr") is not None]
    if not rrs:
        return None
    # Geometric decay: P(reach TP_i) = p_win * decay^(i-1), decay=0.6.
    decay = 0.6
    ev = 0.0
    for i, rr in enumerate(rrs):
        ev += p * (decay**i) * rr
    ev -= 1 - p  # stop at -1R
    ev -= cost_r
    return round(float(ev), 4)


if __name__ == "__main__":
    print("NexusQuant Direction-Neutral Setup Classifier ready.")
