"""
NexusQuant - Support/Resistance + Fibonacci Confluence Engine

Detects key structural levels and scores confluence:

1. Swing highs / lows (fractal window)          - structural pivots
2. Classical pivot points (P, R1-R3, S1-S3)     - daily period levels
3. Fibonacci retracements + extensions          - from the last major legs
4. Confluence clustering                        - levels within tolerance merge
5. Nearest strong support / resistance          - for trade setup context

Usage (library):
    from src.features.levels import compute_levels, levels_summary
    info = compute_levels(df)      # df needs open/high/low/close (+atr_14)
    summary = levels_summary(df)   # report-friendly subset
"""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

# Fibonacci ratios (retracement from a leg; extension beyond its extreme).
FIB_RETRACEMENT = [0.236, 0.382, 0.5, 0.618, 0.786]
FIB_EXTENSION = [1.272, 1.618, 2.0, 2.618]

# How many recent swings to keep as candidates.
DEFAULT_MAX_SWINGS = 12


# ---------------------------------------------------------------------------
# 1. Swing highs / lows (fractals)
# ---------------------------------------------------------------------------


def detect_swings(
    df: pd.DataFrame,
    left: int = 2,
    right: int = 2,
) -> pd.DataFrame:
    """
    Add boolean `swing_high` / `swing_low` columns.

    A swing high at bar i is the max of the surrounding window
    (i-left .. i+right); a swing low is the window min. Consecutive
    duplicates (plateaus) are collapsed to the single extreme bar.

    ``left + right`` must be even so the centred window is odd.
    """
    if (left + right) % 2 != 0:
        raise ValueError("left + right must be even (odd centred window)")
    df = df.copy()
    high, low = df["high"], df["low"]
    window = left + right + 1

    swing_high = (high == high.rolling(window, center=True).max()).fillna(False)
    swing_low = (low == low.rolling(window, center=True).min()).fillna(False)

    df["swing_high"] = _collapse_plateaus(swing_high, high, take_max=True)
    df["swing_low"] = _collapse_plateaus(swing_low, low, take_max=False)
    return df


def _collapse_plateaus(
    flag: pd.Series,
    values: pd.Series,
    take_max: bool,
) -> pd.Series:
    """Collapse consecutive True flags to a single True at the extreme value."""
    result = pd.Series(False, index=flag.index)
    if not flag.any():
        return result
    runs = (flag != flag.shift()).cumsum().where(flag)
    positions = (
        values.groupby(runs).idxmax() if take_max else values.groupby(runs).idxmin()
    )
    result.loc[positions.dropna()] = True
    return result


def swing_levels(
    df: pd.DataFrame,
    max_swings: int = DEFAULT_MAX_SWINGS,
) -> List[Tuple[float, str]]:
    """Recent swing levels as [(price, 'swing_high'|'swing_low')]."""
    if "swing_high" not in df.columns:
        df = detect_swings(df)
    highs = df.loc[df["swing_high"], "high"].tolist()
    lows = df.loc[df["swing_low"], "low"].tolist()
    return [(h, "swing_high") for h in highs[-max_swings:]] + [
        (lo, "swing_low") for lo in lows[-max_swings:]
    ]


# ---------------------------------------------------------------------------
# 2. Classical pivot points
# ---------------------------------------------------------------------------


def pivot_levels(df: pd.DataFrame) -> List[Tuple[float, str]]:
    """Classical pivots from the last *closed* bar: P, R1-R3, S1-S3."""
    if len(df) < 2:
        return []
    prev = df.iloc[-2]
    h, lo, c = prev["high"], prev["low"], prev["close"]

    p = (h + lo + c) / 3
    r1, s1 = 2 * p - lo, 2 * p - h
    r2, s2 = p + (h - lo), p - (h - lo)
    r3, s3 = h + 2 * (p - lo), lo - 2 * (h - p)

    return [
        (p, "pivot"),
        (r1, "pivot_r1"),
        (r2, "pivot_r2"),
        (r3, "pivot_r3"),
        (s1, "pivot_s1"),
        (s2, "pivot_s2"),
        (s3, "pivot_s3"),
    ]


# ---------------------------------------------------------------------------
# 3. Fibonacci retracements & extensions
# ---------------------------------------------------------------------------


def _last_legs(df: pd.DataFrame) -> Tuple[Optional[Tuple], Optional[Tuple]]:
    """
    Return (up_leg, down_leg) tuples of the most recent completed legs:
      up_leg   = (low_time, low_price, high_time, high_price)
      down_leg = (high_time, high_price, low_time, low_price)
    """
    if "swing_high" not in df.columns:
        df = detect_swings(df)
    highs = df.loc[df["swing_high"]]
    lows = df.loc[df["swing_low"]]
    if highs.empty or lows.empty:
        return None, None

    last_high_t, last_low_t = highs.index[-1], lows.index[-1]

    up_leg = None
    lows_before = lows[lows.index < last_high_t]
    if not lows_before.empty:
        t0 = lows_before.index[-1]
        up_leg = (t0, lows.loc[t0, "low"], last_high_t, highs.loc[last_high_t, "high"])

    down_leg = None
    highs_before = highs[highs.index < last_low_t]
    if not highs_before.empty:
        t0 = highs_before.index[-1]
        down_leg = (t0, highs.loc[t0, "high"], last_low_t, lows.loc[last_low_t, "low"])

    return up_leg, down_leg


def fibonacci_levels(df: pd.DataFrame) -> List[Tuple[float, str]]:
    """
    Fibonacci retracements + extensions from the last up- and down-legs,
    as [(price, 'fib_up_0.382'), ('fib_down_ext_1.618'), ...].
    """
    up, down = _last_legs(df)
    levels: List[Tuple[float, str]] = []

    if up:
        _, lo, _, hi = up
        rng = hi - lo
        if rng > 0:
            for r in FIB_RETRACEMENT:
                levels.append((hi - r * rng, f"fib_up_{r:.3f}"))
            for e in FIB_EXTENSION:
                levels.append((hi + e * rng, f"fib_up_ext_{e:.3f}"))

    if down:
        _, hi2, _, lo2 = down
        rng = hi2 - lo2
        if rng > 0:
            for r in FIB_RETRACEMENT:
                levels.append((lo2 + r * rng, f"fib_down_{r:.3f}"))
            for e in FIB_EXTENSION:
                levels.append((lo2 - e * rng, f"fib_down_ext_{e:.3f}"))

    return levels


# ---------------------------------------------------------------------------
# 4. Confluence clustering
# ---------------------------------------------------------------------------


def cluster_levels(
    levels: List[Tuple[float, str]],
    tolerance: float,
) -> List[Dict]:
    """
    Greedily cluster levels within `tolerance` of a cluster anchor price.

    Each cluster: {price, score (number of merged levels), strength, tags}.
    """
    if not levels:
        return []
    ordered = sorted(levels, key=lambda x: x[0])

    clusters: List[Dict] = []
    cur_prices = [ordered[0][0]]
    cur_tags = [ordered[0][1]]

    for price, tag in ordered[1:]:
        # Compare against the running mean so chains of near levels merge.
        mean = float(np.mean(cur_prices))
        if abs(price - mean) <= tolerance:
            cur_prices.append(price)
            cur_tags.append(tag)
        else:
            clusters.append((cur_prices, cur_tags))
            cur_prices, cur_tags = [price], [tag]
    clusters.append((cur_prices, cur_tags))

    out = []
    for prices, tags in clusters:
        score = len(tags)
        out.append(
            {
                "price": round(float(np.mean(prices)), 5),
                "score": score,
                "strength": (
                    "Strong" if score >= 3 else "Medium" if score == 2 else "Weak"
                ),
                "tags": sorted(set(tags)),
            }
        )
    return out


# ---------------------------------------------------------------------------
# 5. Full pipeline + nearest support / resistance
# ---------------------------------------------------------------------------


def compute_levels(
    df: pd.DataFrame,
    left: int = 2,
    right: int = 2,
    tolerance_ratio: float = 0.35,
    min_atr_ratio: float = 0.001,
    max_swings: int = DEFAULT_MAX_SWINGS,
) -> Dict:
    """
    Run the full confluence pipeline and return every zone plus the nearest
    strong support / resistance relative to the last close.
    """
    df = detect_swings(df, left, right)
    close = float(df["close"].iloc[-1])

    if "atr_14" in df.columns and not pd.isna(df["atr_14"].iloc[-1]):
        atr = float(df["atr_14"].iloc[-1])
    else:
        atr = float((df["high"] - df["low"]).tail(14).mean())

    tolerance = max(tolerance_ratio * atr, min_atr_ratio * close)

    raw = swing_levels(df, max_swings) + pivot_levels(df) + fibonacci_levels(df)
    clusters = cluster_levels(raw, tolerance)

    supports = [c for c in clusters if c["price"] < close]
    resistances = [c for c in clusters if c["price"] > close]

    nearest_support = max(supports, key=lambda c: c["price"]) if supports else None
    nearest_resistance = (
        min(resistances, key=lambda c: c["price"]) if resistances else None
    )

    up, down = _last_legs(df)
    return {
        "close": close,
        "tolerance": round(tolerance, 6),
        "clusters": clusters,
        "nearest_support": nearest_support,
        "nearest_resistance": nearest_resistance,
        "pivots": {tag: round(price, 5) for price, tag in pivot_levels(df)},
        "last_up_leg": (float(up[1]), float(up[3])) if up else None,
        "last_down_leg": (float(down[1]), float(down[3])) if down else None,
    }


def levels_summary(
    df: pd.DataFrame,
    max_distance_ratio: float = 0.15,
    **kwargs,
) -> Dict:
    """
    Report-friendly subset: nearest S/R, top confluence zones and pivots.

    ``max_distance_ratio`` filters out zones further than that fraction of
    price away from the last close (keeps the report actionable).
    """
    info = compute_levels(df, **kwargs)
    close = info["close"]
    max_dist = max_distance_ratio * close
    nearby = [c for c in info["clusters"] if abs(c["price"] - close) <= max_dist]
    top = sorted(nearby, key=lambda c: (-c["score"], c["price"]))[:6]
    return {
        "close": info["close"],
        "tolerance": info["tolerance"],
        "nearest_support": info["nearest_support"],
        "nearest_resistance": info["nearest_resistance"],
        "top_confluence": top,
        "pivots": info["pivots"],
        "last_up_leg": info["last_up_leg"],
        "last_down_leg": info["last_down_leg"],
    }


# ---------------------------------------------------------------------------
# 6. Standalone Fibonacci confluence-strength map (institutional spec #7)
# ---------------------------------------------------------------------------

# The exact ratio set the spec's map asks for (38.2 ... 161.8), with the
# 0.5 retracement folded in (it is both a retracement and near a 0.618 zone).
_FIB_MAP = [
    ("38.2%", 0.382, "retracement"),
    ("50.0%", 0.500, "retracement"),
    ("61.8%", 0.618, "retracement"),
    ("78.6%", 0.786, "retracement"),
    ("127.2%", 1.272, "extension"),
    ("161.8%", 1.618, "extension"),
]


def fibonacci_map(
    df: pd.DataFrame, tolerance_ratio: float = 0.35, min_atr_ratio: float = 0.001
) -> List[Dict]:
    """
    Standalone Fibonacci confluence map (institutional spec #7).

    Computes each spec ratio from the last up- and down-legs, merges near
    levels into clusters (the same tolerance logic as ``compute_levels``)
    and reports each ratio's price plus a 1-10 confluence strength (how
    many independent structural levels - swings, pivots, other fibs -
    cluster at that price).

    Returns rows like::

        {"level": "38.2%", "side": "up", "price": 1.35,
         "confluence": 7, "tags": ["swing_high", "pivot_r1", ...]}
    """
    up, down = _last_legs(df)
    if "atr_14" in df.columns and not pd.isna(df["atr_14"].iloc[-1]):
        atr = float(df["atr_14"].iloc[-1])
    else:
        atr = float((df["high"] - df["low"]).tail(14).mean())
    close = float(df["close"].iloc[-1])
    tolerance = max(tolerance_ratio * atr, min_atr_ratio * close)

    # Independent structural anchors for the confluence score.
    anchors = set(
        swing_levels(df, max_swings=12) + pivot_levels(df) + fibonacci_levels(df)
    )

    rows: List[Dict] = []
    legs = [("up", up), ("down", down)]
    for side, leg in legs:
        if not leg:
            continue
        if side == "up":
            _, lo, _, hi = leg
            rng = hi - lo
        else:
            _, hi2, _, lo2 = leg
            rng = hi2 - lo2
        if rng <= 0:
            continue
        for label, ratio, kind in _FIB_MAP:
            if side == "up":
                price = hi - ratio * rng if kind == "retracement" else hi + ratio * rng
            else:
                price = (
                    lo2 + ratio * rng if kind == "retracement" else lo2 - ratio * rng
                )
            # confluence: anchors within tolerance of this fib price
            nearby = [p for p, _ in anchors if abs(p - price) <= tolerance]
            tags = [tag for p, tag in anchors if abs(p - price) <= tolerance]
            rows.append(
                {
                    "level": label,
                    "side": side,
                    "kind": kind,
                    "price": round(float(price), 5),
                    "distance_from_close_pct": round(
                        (abs(price - close) / close) * 100.0, 2
                    ),
                    "confluence": round(len(nearby) / 3.0 + 1.0, 1),  # 1-10 scale
                    "tags": sorted(set(tags))[:6],
                }
            )
    return rows


# ---------------------------------------------------------------------------
# 7. Anchored VWAP + volume-profile high-volume nodes (institutional spec #2)
# ---------------------------------------------------------------------------


def anchored_vwap(df: pd.DataFrame, anchor_idx: int = 0) -> Optional[float]:
    """
    Volume-weighted average price anchored at ``anchor_idx`` (0 = from the
    first bar). Returns None when no volume data exists.
    """
    if "volume" not in df.columns or df["volume"].notna().sum() < 10:
        return None
    seg = df.iloc[anchor_idx:]
    seg = seg[seg["volume"] > 0]
    if seg.empty:
        return None
    typical = (seg["high"] + seg["low"] + seg["close"]) / 3.0
    vp = (typical * seg["volume"]).sum()
    tv = seg["volume"].sum()
    if tv <= 0:
        return None
    return round(float(vp / tv), 5)


def volume_profile_nodes(
    df: pd.DataFrame,
    n_bins: int = 20,
    lookback: Optional[int] = None,
    vol_mult: float = 1.5,
) -> List[Dict]:
    """
    Volume-profile high-volume nodes (institutional spec #2).

    Buckets the last ``lookback`` bars (default: all) into ``n_bins``
    price bins, sums volume per bin, and returns bins whose volume is at
    least ``vol_mult`` x the average bin volume - i.e. the price levels
    where institutional participation is concentrated (support/resistance).

    Returns None-friendly rows::

        {"price": 1.35, "volume_pct": 18.2, "is_high_volume": True}
    """
    if "volume" not in df.columns or df["volume"].notna().sum() < 30:
        return []
    work = df[df["volume"] > 0].copy()
    if work.empty:
        return []
    if lookback:
        work = work.tail(lookback)
    lo, hi = float(work["low"].min()), float(work["high"].max())
    if hi <= lo:
        return []

    edges = np.linspace(lo, hi, n_bins + 1)
    bins = pd.cut(work["close"], bins=edges, include_lowest=True)
    vol_by_bin = work["volume"].groupby(bins, observed=True).sum()
    if vol_by_bin.sum() <= 0:
        return []

    avg = vol_by_bin.mean()
    rows: List[Dict] = []
    for interval, vol in vol_by_bin.items():
        center = (interval.left + interval.right) / 2.0
        rows.append(
            {
                "price": round(float(center), 5),
                "volume_pct": round(float(vol / vol_by_bin.sum() * 100.0), 2),
                "is_high_volume": bool(vol >= vol_mult * avg),
            }
        )
    rows.sort(key=lambda r: r["price"])
    return rows


if __name__ == "__main__":
    print("NexusQuant Levels module ready.")
