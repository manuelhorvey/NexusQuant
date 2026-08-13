"""
NexusQuant - Momentum Divergence Engine (institutional spec #4)

Detects classic price-vs-oscillator divergences between swing points of
price and RSI (and MACD histogram), plus RSI failure swings:

* **Regular divergence**   - a trend-reversal signal:
    - Bearish: price makes a higher high, RSI a lower high.
    - Bullish: price makes a lower low, RSI a higher low.
* **Hidden divergence**    - a trend-continuation signal:
    - Bullish: price makes a higher low, RSI a lower low.
    - Bearish: price makes a lower high, RSI a higher high.
* **Failure swing**        - an early RSI reversal: RSI pushes to a new
  extreme, pulls back, then fails to exceed the prior extreme and turns.

Every signal carries a confidence 0-100 based on how pronounced the
swing-pair is (price move size in ATR, oscillator move size, bar span).
Only signals >= ``MIN_CONFIDENCE`` (65) are reported, mirroring the
pattern engine's 65% threshold.

Usage (library):
    from src.features.divergence import detect_divergences, divergence_summary
    summary = divergence_summary(df)   # df needs indicators (rsi_14 etc.)
"""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import pandas as pd

from src.features.levels import detect_swings

MIN_CONFIDENCE = 65.0
MAX_CONFIDENCE = 95.0


def _atr(df: pd.DataFrame) -> float:
    if "atr_14" in df.columns and not pd.isna(df["atr_14"].iloc[-1]):
        return float(df["atr_14"].iloc[-1])
    return float((df["high"] - df["low"]).tail(14).mean())


def _swing_points(
    df: pd.DataFrame,
    left: int = 2,
    right: int = 2,
) -> List[Tuple[int, float, str]]:
    """Ordered list of (bar_index, price, 'high'|'low') pivots.

    Only CONFIRMED pivots are returned: a pivot at bar ``i`` is reported
    only when the centred detection window (i-left .. i+right) is fully
    inside the data, i.e. ``left <= i < n - right``. The trailing
    ``right`` bars' flags are still provisional (they can repaint when
    the next bar prints), so they are excluded - a reported divergence
    never depends on a swing that could disappear tomorrow. This is the
    standard "confirmation delay": the latest signal is based on
    structure confirmed at least ``right`` bars ago.
    """
    work = df.reset_index(drop=True)
    n = len(work)
    marked = detect_swings(work, left, right)
    pts: List[Tuple[int, float, str]] = []
    for i, row in marked.iterrows():
        if not (left <= i < n - right):
            continue
        if bool(row.get("swing_high", False)):
            pts.append((int(i), float(row["high"]), "high"))
        elif bool(row.get("swing_low", False)):
            pts.append((int(i), float(row["low"]), "low"))
    return pts


def _osc_value(df: pd.DataFrame, i: int, osc: str) -> Optional[float]:
    """Oscillator value at bar index ``i`` (rsi_14 or macd_hist)."""
    if osc not in df.columns:
        return None
    try:
        v = df[osc].iloc[i]
        return None if pd.isna(v) else float(v)
    except (IndexError, KeyError, TypeError):
        return None


def _price_at(df: pd.DataFrame, i: int, kind: str) -> Optional[float]:
    col = "high" if kind == "high" else "low"
    try:
        return float(df[col].iloc[i])
    except (IndexError, KeyError, TypeError):
        return None


def _confidence(
    df: pd.DataFrame,
    i0: int,
    i1: int,
    p0: float,
    p1: float,
    o0: float,
    o1: float,
    atr: float,
) -> float:
    """0-100 confidence: price move relative to ATR + osc move + span."""
    c = 45.0
    price_move = abs(p1 - p0)
    c += 15 if price_move >= 1.5 * atr else (10 if price_move >= atr else 5)
    osc_move = abs(o1 - o0)
    c += 15 if osc_move >= 15.0 else (10 if osc_move >= 8.0 else 5)
    span = abs(i1 - i0)
    c += 10 if span >= 6 else (5 if span >= 3 else 0)
    return float(min(MAX_CONFIDENCE, max(0.0, c)))


def _regular_divergences(
    df: pd.DataFrame,
    swings: List[Tuple[int, float, str]],
    osc: str,
    atr: float,
) -> List[Dict]:
    out: List[Dict] = []
    highs = [s for s in swings if s[2] == "high"]
    lows = [s for s in swings if s[2] == "low"]

    # Bearish regular: higher price high + lower osc high (last two highs)
    if len(highs) >= 2:
        (i2, p2, _), (i1, p1, _) = highs[-1], highs[-2]
        o1, o2 = _osc_value(df, i1, osc), _osc_value(df, i2, osc)
        if o1 is not None and o2 is not None and p2 > p1 and o2 < o1:
            conf = _confidence(df, i1, i2, p1, p2, o1, o2, atr)
            if conf >= MIN_CONFIDENCE:
                out.append(
                    {
                        "name": "Regular Bearish Divergence",
                        "side": "bearish",
                        "type": "regular",
                        "osc": osc,
                        "price_from": round(p1, 5),
                        "price_to": round(p2, 5),
                        "osc_from": round(o1, 2),
                        "osc_to": round(o2, 2),
                        "confidence": round(conf),
                    }
                )

    # Bullish regular: lower price low + higher osc low (last two lows)
    if len(lows) >= 2:
        (i2, p2, _), (i1, p1, _) = lows[-1], lows[-2]
        o1, o2 = _osc_value(df, i1, osc), _osc_value(df, i2, osc)
        if o1 is not None and o2 is not None and p2 < p1 and o2 > o1:
            conf = _confidence(df, i1, i2, p1, p2, o1, o2, atr)
            if conf >= MIN_CONFIDENCE:
                out.append(
                    {
                        "name": "Regular Bullish Divergence",
                        "side": "bullish",
                        "type": "regular",
                        "osc": osc,
                        "price_from": round(p1, 5),
                        "price_to": round(p2, 5),
                        "osc_from": round(o1, 2),
                        "osc_to": round(o2, 2),
                        "confidence": round(conf),
                    }
                )
    return out


def _hidden_divergences(
    df: pd.DataFrame,
    swings: List[Tuple[int, float, str]],
    osc: str,
    atr: float,
) -> List[Dict]:
    out: List[Dict] = []
    highs = [s for s in swings if s[2] == "high"]
    lows = [s for s in swings if s[2] == "low"]

    # Bullish hidden: higher price low + lower osc low (continuation)
    if len(lows) >= 2:
        (i2, p2, _), (i1, p1, _) = lows[-1], lows[-2]
        o1, o2 = _osc_value(df, i1, osc), _osc_value(df, i2, osc)
        if o1 is not None and o2 is not None and p2 > p1 and o2 < o1:
            conf = _confidence(df, i1, i2, p1, p2, o1, o2, atr)
            if conf >= MIN_CONFIDENCE:
                out.append(
                    {
                        "name": "Hidden Bullish Divergence",
                        "side": "bullish",
                        "type": "hidden",
                        "osc": osc,
                        "price_from": round(p1, 5),
                        "price_to": round(p2, 5),
                        "osc_from": round(o1, 2),
                        "osc_to": round(o2, 2),
                        "confidence": round(conf),
                    }
                )

    # Bearish hidden: lower price high + higher osc high (continuation)
    if len(highs) >= 2:
        (i2, p2, _), (i1, p1, _) = highs[-1], highs[-2]
        o1, o2 = _osc_value(df, i1, osc), _osc_value(df, i2, osc)
        if o1 is not None and o2 is not None and p2 < p1 and o2 > o1:
            conf = _confidence(df, i1, i2, p1, p2, o1, o2, atr)
            if conf >= MIN_CONFIDENCE:
                out.append(
                    {
                        "name": "Hidden Bearish Divergence",
                        "side": "bearish",
                        "type": "hidden",
                        "osc": osc,
                        "price_from": round(p1, 5),
                        "price_to": round(p2, 5),
                        "osc_from": round(o1, 2),
                        "osc_to": round(o2, 2),
                        "confidence": round(conf),
                    }
                )
    return out


def _failure_swings(df: pd.DataFrame, swings, atr: float) -> List[Dict]:
    """RSI failure swings: new extreme then a pullback that fails to
    exceed the prior extreme before turning."""
    out: List[Dict] = []
    if "rsi_14" not in df.columns:
        return out
    rsi = df["rsi_14"]

    highs = [s for s in swings if s[2] == "high"]
    lows = [s for s in swings if s[2] == "low"]

    # Bearish failure swing: RSI pushes above 70, drops, retest fails below
    # the previous peak and turns back down.
    if len(highs) >= 3:
        (i3, _, _), (i2, _, _), (i1, _, _) = highs[-1], highs[-2], highs[-3]
        r1, r2, r3 = (rsi.iloc[i1], rsi.iloc[i2], rsi.iloc[i3])
        if (r1 > 70 and r2 > r1 and r3 < r1) or (r1 > 70 and r2 <= r1 and r3 < r2):
            out.append(
                {
                    "name": "RSI Failure Swing (bearish)",
                    "side": "bearish",
                    "type": "failure_swing",
                    "osc": "rsi_14",
                    "peaks": [
                        round(float(r1), 2),
                        round(float(r2), 2),
                        round(float(r3), 2),
                    ],
                    "confidence": round(70.0),
                }
            )

    # Bullish failure swing: RSI drops below 30, bounces, retest holds above
    # the prior trough and turns up.
    if len(lows) >= 3:
        (i3, _, _), (i2, _, _), (i1, _, _) = lows[-1], lows[-2], lows[-3]
        r1, r2, r3 = (rsi.iloc[i1], rsi.iloc[i2], rsi.iloc[i3])
        if (r1 < 30 and r2 < r1 and r3 > r1) or (r1 < 30 and r2 >= r1 and r3 > r2):
            out.append(
                {
                    "name": "RSI Failure Swing (bullish)",
                    "side": "bullish",
                    "type": "failure_swing",
                    "osc": "rsi_14",
                    "peaks": [
                        round(float(r1), 2),
                        round(float(r2), 2),
                        round(float(r3), 2),
                    ],
                    "confidence": round(70.0),
                }
            )
    return out


def detect_divergences(
    df: pd.DataFrame,
    min_confidence: float = MIN_CONFIDENCE,
    oscillators: Optional[List[str]] = None,
) -> List[Dict]:
    """
    Detect regular + hidden divergences and RSI failure swings on the last
    bars of ``df`` (needs indicators). Returns signals sorted by confidence.
    """
    if len(df) < 40 or "close" not in df.columns:
        return []
    if oscillators is None:
        oscillators = [o for o in ("rsi_14", "macd_hist") if o in df.columns]
    if not oscillators:
        return []

    try:
        swings = _swing_points(df)
    except Exception:
        return []
    if len(swings) < 3:
        return []

    atr = _atr(df)
    out: List[Dict] = []
    for osc in oscillators:
        out += _regular_divergences(df, swings, osc, atr)
        out += _hidden_divergences(df, swings, osc, atr)
    out += _failure_swings(df, swings, atr)

    out = [d for d in out if d["confidence"] >= min_confidence]
    out.sort(key=lambda d: -d["confidence"])
    return out


def format_divergence(signal: Dict) -> str:
    """Display detail for one divergence signal dict.

    Regular/hidden divergences carry ``osc_from``/``osc_to``; failure
    swings carry an RSI ``peaks`` list. Rendering both shapes in ONE
    place keeps the report printer and the dashboard in sync (duplicated
    shape logic already drifted once and broke both).
    """
    osc = signal.get("osc", "")
    if signal.get("type") == "failure_swing":
        peaks = " → ".join(str(p) for p in signal.get("peaks", []))
        return f"{osc} peaks {peaks}"
    return f"{osc} {signal.get('osc_from')} → {signal.get('osc_to')}"


def divergence_summary(
    df: pd.DataFrame,
    min_confidence: float = MIN_CONFIDENCE,
) -> Dict:
    """Report-friendly subset: divergences + the strongest one."""
    signals = detect_divergences(df, min_confidence=min_confidence)
    return {
        "signals": signals,
        "best": signals[0] if signals else None,
        "count": len(signals),
        "available": True,
    }


if __name__ == "__main__":
    print("NexusQuant Divergence module ready.")
