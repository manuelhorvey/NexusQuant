"""
NexusQuant - Pattern Recognition Engine

Detects statistically-valid chart structures from swing points using
objective, measurable criteria (institutional spec #6). Every pattern
carries a structural confidence score 0-100; only patterns at/above
``MIN_PROBABILITY`` (65) are reported, matching the spec's
"only patterns with >= 65% statistical probability".

Pivots are CONFIRMED-ONLY (see ``_swing_points``): the trailing bars'
swing flags can repaint, so reported patterns only use structure whose
detection window is complete - no right-edge repainting, no lookahead.

Patterns detected:

* Head & Shoulders / Inverse Head & Shoulders
* Double Top / Double Bottom
* Cup & Handle
* Triangles (Ascending / Descending / Symmetric)
* Flags & Pennants (continuation after a strong pole)

Honesty note: the ``confidence`` score (legacy alias ``prob``) is a
DETERMINISTIC composite of the pattern's *structural quality*
(symmetry, depth, trendline touches, volume confirmation, breakout
proximity). It is a 0-100 structural score - NOT a claimed
statistically-calibrated probability, and it must never be quoted as
"P(pattern works) = X%". To earn a probability label, the pattern
engine would need to be calibrated against the backtester (count how
often each pattern's breakout resolves in the expected direction,
per regime), which is the documented next step, not a current claim.

Usage (library):
    from src.features.patterns import detect_patterns, patterns_summary
    patterns = detect_patterns(df)          # df needs ohlc (+ optional volume)
    summary  = patterns_summary(df)         # report-friendly subset
"""

from __future__ import annotations

import logging
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from src.features.levels import detect_swings

log = logging.getLogger(__name__)

MIN_PROBABILITY = 65.0
MAX_PROBABILITY = 95.0  # confirmed (breakout already happened)
FORMING_CAP = 85.0  # forming patterns cap lower - not yet validated
_BASE = 45.0

# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------


def _atr(df: pd.DataFrame) -> float:
    if "atr_14" in df.columns and not pd.isna(df["atr_14"].iloc[-1]):
        return float(df["atr_14"].iloc[-1])
    return float((df["high"] - df["low"]).tail(14).mean())


def _tol(atr: float, close: float, ratio: float = 0.35) -> float:
    """Price tolerance for level comparisons (scales with volatility)."""
    return max(ratio * atr, 0.001 * close)


def _clamp(p: float, cap: float = MAX_PROBABILITY) -> float:
    return float(min(cap, max(0.0, p)))


def _has_volume(df: pd.DataFrame) -> bool:
    return "volume" in df.columns and df["volume"].notna().sum() > 30


def _swing_points(
    df: pd.DataFrame,
    left: int = 3,
    right: int = 3,
) -> List[Tuple[int, float, str]]:
    """Ordered list of (bar_index, price, 'high'|'low') pivots.

    Only CONFIRMED pivots are returned: a pivot at bar ``i`` is reported
    only when the centred detection window (i-left .. i+right) is fully
    inside the data, i.e. ``left <= i < n - right``. The trailing
    ``right`` bars' flags are still provisional (they can repaint when
    the next bar prints), so they are excluded - a reported pattern
    never depends on a swing that could disappear tomorrow. This is the
    standard "confirmation delay": the latest pattern is based on
    structure confirmed at least ``right`` bars ago.
    """
    df = df.reset_index(drop=True)
    n = len(df)
    marked = detect_swings(df, left, right)
    pts: List[Tuple[int, float, str]] = []
    for i, row in marked.iterrows():
        if not (left <= i < n - right):
            continue
        if bool(row.get("swing_high", False)):
            pts.append((int(i), float(row["high"]), "high"))
        elif bool(row.get("swing_low", False)):
            pts.append((int(i), float(row["low"]), "low"))
    return pts


def _volume_confirm(df: pd.DataFrame, i_early: int, i_late: int, bearish: bool) -> int:
    """+8 when volume *declines* on the second peak (top) or *rises* on the
    second trough (bottom) — classic confirmation of the reversal."""
    if not _has_volume(df):
        return 0
    v_early = float(df["volume"].iloc[i_early]) or 0.0
    v_late = float(df["volume"].iloc[i_late]) or 0.0
    if v_early <= 0:
        return 0
    change = (v_late - v_early) / v_early
    return 8 if (change < 0 if bearish else change > 0) else 0


# ---------------------------------------------------------------------------
# Double Top / Double Bottom
# ---------------------------------------------------------------------------


def _double_top(df: pd.DataFrame, swings, atr: float, close: float) -> Optional[Dict]:
    highs = [s for s in swings if s[2] == "high"]
    if len(highs) < 2:
        return None
    (i2, h2, _), (i1, h1, _) = highs[-1], highs[-2]
    if i2 - i1 < 3:
        return None
    between = [s for s in swings if i1 < s[0] < i2 and s[2] == "low"]
    if not between:
        return None
    neck = min(s[1] for s in between)
    tol = _tol(atr, close)
    sym = abs(h1 - h2)
    if sym > tol:
        return None
    depth = min(h1, h2) - neck
    if depth < 0.75 * atr:
        return None
    confirmed = close < neck

    prob = _BASE
    prob += 10 if sym <= 0.4 * tol else 5
    prob += 10 if depth >= 1.5 * atr else 5
    prob += _volume_confirm(df, i1, i2, bearish=True)
    prob += 10 if confirmed else (5 if (close - neck) <= atr else 0)
    prob = _clamp(prob, MAX_PROBABILITY if confirmed else FORMING_CAP)
    if prob < MIN_PROBABILITY:
        return None
    return {
        "name": "Double Top",
        "side": "bearish",
        "breakout": round(neck, 6),
        "confidence": round(prob),
        "prob": round(prob),  # legacy alias for ``confidence`` (kept so
        # older consumers do not break)
        "status": "Confirmed" if confirmed else "Forming",
        "detail": f"peaks {h1:.5f}/{h2:.5f} · neckline {neck:.5f}",
    }


def _double_bottom(
    df: pd.DataFrame, swings, atr: float, close: float
) -> Optional[Dict]:
    lows = [s for s in swings if s[2] == "low"]
    if len(lows) < 2:
        return None
    (i2, l2, _), (i1, l1, _) = lows[-1], lows[-2]
    if i2 - i1 < 3:
        return None
    between = [s for s in swings if i1 < s[0] < i2 and s[2] == "high"]
    if not between:
        return None
    neck = max(s[1] for s in between)
    tol = _tol(atr, close)
    sym = abs(l1 - l2)
    if sym > tol:
        return None
    depth = neck - max(l1, l2)
    if depth < 0.75 * atr:
        return None
    confirmed = close > neck

    prob = _BASE
    prob += 10 if sym <= 0.4 * tol else 5
    prob += 10 if depth >= 1.5 * atr else 5
    prob += _volume_confirm(df, i1, i2, bearish=False)
    prob += 10 if confirmed else (5 if (neck - close) <= atr else 0)
    prob = _clamp(prob, MAX_PROBABILITY if confirmed else FORMING_CAP)
    if prob < MIN_PROBABILITY:
        return None
    return {
        "name": "Double Bottom",
        "side": "bullish",
        "breakout": round(neck, 6),
        "confidence": round(prob),
        "prob": round(prob),  # legacy alias for ``confidence`` (kept so
        # older consumers do not break)
        "status": "Confirmed" if confirmed else "Forming",
        "detail": f"troughs {l1:.5f}/{l2:.5f} · neckline {neck:.5f}",
    }


# ---------------------------------------------------------------------------
# Head & Shoulders (+ inverse)
# ---------------------------------------------------------------------------


def _head_shoulders(
    df: pd.DataFrame,
    swings,
    atr: float,
    close: float,
    inverse: bool = False,
) -> Optional[Dict]:
    kind = "low" if inverse else "high"
    opp = "high" if inverse else "low"
    exts = [s for s in swings if s[2] == kind]
    if len(exts) < 3:
        return None
    (ir, pr, _), (_, ph, _), (il, pl, _) = exts[-1], exts[-2], exts[-3]
    # centre point must be the extreme of the three
    if inverse:
        if not (ph < pl and ph < pr):
            return None
    elif not (ph > pl and ph > pr):
        return None
    tol = _tol(atr, close)
    if abs(pl - pr) > tol:  # shoulder symmetry
        return None
    prominence = abs(ph - max(pl, pr)) if not inverse else abs(max(pl, pr) - ph)
    if prominence < 0.5 * atr:
        return None
    between = [s for s in swings if il < s[0] < ir and s[2] == opp]
    if len(between) < 2:
        return None
    n1, n2 = between[0][1], between[-1][1]
    neck = max(n1, n2) if inverse else min(n1, n2)
    confirmed = (close > neck) if inverse else (close < neck)

    prob = _BASE
    prob += 10 if abs(pl - pr) <= 0.4 * tol else 5
    prob += 10 if prominence >= 1.5 * atr else 5
    prob += 5 if abs(n1 - n2) <= 0.25 * atr else 0  # flat neckline
    prob += _volume_confirm(df, il, ir, bearish=not inverse)
    prob += 10 if confirmed else (5 if abs(close - neck) <= atr else 0)
    prob = _clamp(prob, MAX_PROBABILITY if confirmed else FORMING_CAP)
    if prob < MIN_PROBABILITY:
        return None
    name = "Inverse Head & Shoulders" if inverse else "Head & Shoulders"
    return {
        "name": name,
        "side": "bullish" if inverse else "bearish",
        "breakout": round(neck, 6),
        "confidence": round(prob),
        "prob": round(prob),  # legacy alias for ``confidence`` (kept so
        # older consumers do not break)
        "status": "Confirmed" if confirmed else "Forming",
        "detail": f"shoulders {pl:.5f}/{pr:.5f} · head {ph:.5f} · neck {neck:.5f}",
    }


def _inverse_hs(df: pd.DataFrame, swings, atr: float, close: float) -> Optional[Dict]:
    return _head_shoulders(df, swings, atr, close, inverse=True)


# ---------------------------------------------------------------------------
# Triangles
# ---------------------------------------------------------------------------


def _triangle(
    df: pd.DataFrame,
    swings,
    atr: float,
    close: float,
    lookback: int = 8,
) -> Optional[Dict]:
    recent = swings[-lookback:]
    highs = [s for s in recent if s[2] == "high"]
    lows = [s for s in recent if s[2] == "low"]
    if len(highs) < 2 or len(lows) < 2:
        return None

    def fit(pts):
        if len(pts) < 2:
            return None
        xs = np.array([p[0] for p in pts], dtype=float)
        ys = np.array([p[1] for p in pts], dtype=float)
        if np.ptp(xs) == 0:
            return None
        return np.polyfit(xs, ys, 1)

    hfit, lfit = fit(highs), fit(lows)
    if hfit is None or lfit is None:
        return None

    last_i = float(swings[-1][0])
    span = max(1.0, last_i - float(recent[0][0]))
    flat = 0.5 * atr / span  # per-bar slope threshold for "flat"
    h_now = float(np.polyval(hfit, last_i))
    l_now = float(np.polyval(lfit, last_i))

    if abs(hfit[0]) <= flat and lfit[0] > flat:
        name, side, breakout = "Ascending Triangle", "bullish", h_now
    elif abs(lfit[0]) <= flat and hfit[0] < -flat:
        name, side, breakout = "Descending Triangle", "bearish", l_now
    elif hfit[0] < -flat and lfit[0] > flat:
        d_h, d_l = abs(close - h_now), abs(close - l_now)
        if d_h <= d_l:
            name, side, breakout = "Symmetric Triangle", "bearish", h_now
        else:
            name, side, breakout = "Symmetric Triangle", "bullish", l_now
    else:
        return None

    width0 = float(np.polyval(hfit, recent[0][0])) - float(
        np.polyval(lfit, recent[0][0])
    )
    width_now = h_now - l_now
    if width0 <= 0 or width_now <= 0 or width_now >= width0:
        return None  # not converging
    confirmed = (close > breakout) if side == "bullish" else (close < breakout)

    prob = _BASE
    prob += 10 if (len(highs) + len(lows)) >= 5 else 5  # trendline touches
    prob += 10 if width_now <= 0.6 * width0 else 5  # convergence
    prob += 5 if abs(close - breakout) <= atr else 0  # trigger proximity
    prob += 10 if confirmed else 0
    prob = _clamp(prob, MAX_PROBABILITY if confirmed else FORMING_CAP)
    if prob < MIN_PROBABILITY:
        return None
    return {
        "name": name,
        "side": side,
        "breakout": round(breakout, 6),
        "confidence": round(prob),
        "prob": round(prob),  # legacy alias for ``confidence`` (kept so
        # older consumers do not break)
        "status": "Confirmed" if confirmed else "Forming",
        "detail": f"highs {len(highs)} touches · lows {len(lows)} touches · "
        f"width {width_now / width0:.0%} of start",
    }


# ---------------------------------------------------------------------------
# Cup & Handle
# ---------------------------------------------------------------------------


def _cup_handle(df: pd.DataFrame, swings, atr: float, close: float) -> Optional[Dict]:
    highs = [s for s in swings if s[2] == "high"]
    lows = [s for s in swings if s[2] == "low"]
    if len(highs) < 2 or len(lows) < 1:
        return None
    tol = _tol(atr, close)
    bi, bottom, _ = lows[-1]  # cup bottom = most recent swing low
    left_rims = [s for s in highs if s[0] < bi]
    right_rims = [s for s in highs if s[0] > bi]
    if not left_rims or not right_rims:
        return None
    li, lrim, _ = left_rims[-1]
    ri, rrim, _ = right_rims[0]
    if ri - li < 5:
        return None
    if abs(lrim - rrim) > tol:
        return None  # rim symmetry
    rim = max(lrim, rrim)
    rise = rim - bottom
    if rise < 1.5 * atr:
        return None
    # handle: a shallow pullback after the right rim, staying above the bottom
    post = [s for s in lows if ri < s[0] and s[1] > bottom]
    if not post:
        return None
    hi, hlow, _ = post[0]
    handle_depth = rrim - hlow
    if handle_depth > 0.382 * rise:
        return None
    confirmed = close > rim

    prob = _BASE
    prob += 10 if abs(lrim - rrim) <= 0.4 * tol else 5
    prob += 10 if rise >= 3.0 * atr else 5
    prob += 10 if handle_depth <= 0.25 * rise else 5
    prob += 10 if confirmed else (5 if (rim - close) <= atr else 0)
    prob = _clamp(prob, MAX_PROBABILITY if confirmed else FORMING_CAP)
    if prob < MIN_PROBABILITY:
        return None
    return {
        "name": "Cup & Handle",
        "side": "bullish",
        "breakout": round(rim, 6),
        "confidence": round(prob),
        "prob": round(prob),  # legacy alias for ``confidence`` (kept so
        # older consumers do not break)
        "status": "Confirmed" if confirmed else "Forming",
        "detail": f"rim {rim:.5f} · bottom {bottom:.5f} · handle depth "
        f"{handle_depth / rise:.0%} of rise",
    }


# ---------------------------------------------------------------------------
# Flags & Pennants (continuation)
# ---------------------------------------------------------------------------


def _flag_pennant(df: pd.DataFrame, swings, atr: float, close: float) -> Optional[Dict]:
    if len(swings) < 2:
        return None
    (i0, p0, k0), (i1, p1, k1) = swings[-1], swings[-2]
    if k0 == k1:
        return None  # need alternating extreme
    if k0 == "high":
        pole, direction = p0 - p1, "bullish"
    else:
        pole, direction = p1 - p0, "bearish"
    if pole < 3.0 * atr or i0 - i1 <= 2:
        return None
    cons = df.iloc[i0 + 1 : i0 + 11]
    if len(cons) < 3:
        return None
    cons_range = float(cons["high"].max() - cons["low"].min())
    tight = cons_range / pole
    # tight in BOTH relative (vs pole) and absolute (vs ATR) terms, so a
    # generic 3-ATR move + any pause does not qualify as a flag
    if tight > 0.6 or cons_range > 0.75 * atr:
        return None
    if direction == "bullish":
        breakout = float(cons["high"].max()) + 0.5 * atr
    else:
        breakout = float(cons["low"].min()) - 0.5 * atr
    confirmed = (close > breakout) if direction == "bullish" else (close < breakout)
    name = "Flag" if len(cons) >= 5 else "Pennant"

    prob = _BASE
    prob += 10 if pole >= 4.0 * atr else 5
    prob += 10 if tight <= 0.4 else 5
    prob += 5 if len(cons) >= 5 else 0
    prob += 10 if confirmed else 0
    prob = _clamp(prob, MAX_PROBABILITY if confirmed else FORMING_CAP)
    if prob < MIN_PROBABILITY:
        return None
    return {
        "name": name,
        "side": direction,
        "breakout": round(breakout, 6),
        "confidence": round(prob),
        "prob": round(prob),  # legacy alias for ``confidence`` (kept so
        # older consumers do not break)
        "status": "Confirmed" if confirmed else "Forming",
        "detail": f"pole {pole / atr:.1f} ATR · consolidation {tight:.0%} of pole",
    }


# ---------------------------------------------------------------------------
# public API
# ---------------------------------------------------------------------------

_DETECTORS = [
    _double_top,
    _double_bottom,
    _head_shoulders,
    _inverse_hs,
    _triangle,
    _cup_handle,
    _flag_pennant,
]


def detect_patterns(
    df: pd.DataFrame,
    min_probability: float = MIN_PROBABILITY,
) -> List[Dict]:
    """
    Detect chart patterns on the last bars of ``df`` (needs OHLC; volume
    optional). Returns patterns with confidence >= ``min_probability``,
    sorted by confidence descending.
    """
    df = df.reset_index(drop=True)
    if len(df) < 25 or "close" not in df.columns:
        return []
    try:
        swings = _swing_points(df)
    except Exception:
        return []
    if len(swings) < 3:
        return []
    atr = _atr(df)
    close = float(df["close"].iloc[-1])

    out: List[Dict] = []
    for detector in _DETECTORS:
        try:
            p = detector(df, swings, atr, close)
        except Exception as exc:  # keep the scan alive, but never silently
            log.warning(
                "pattern detector %s failed on this frame: %s", detector.__name__, exc
            )
            continue
        if p and p["prob"] >= min_probability:
            out.append(p)
    out.sort(key=lambda p: -p["prob"])
    return out


def patterns_summary(
    df: pd.DataFrame,
    min_probability: float = MIN_PROBABILITY,
) -> Dict:
    """Report-friendly subset: detected patterns + the best one."""
    patterns = detect_patterns(df, min_probability=min_probability)
    return {
        "patterns": patterns,
        "best": patterns[0] if patterns else None,
        "count": len(patterns),
    }


if __name__ == "__main__":
    print("NexusQuant Pattern Recognition module ready.")
