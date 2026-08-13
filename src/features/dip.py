"""
NexusQuant - Buy-the-Dip Confirmation Engine

Rule-based dip-buy confirmation built on the levels + regime + momentum stack:

  1. Uptrend structure (D1)    price above SMA200, bullish MA stack, ADX
  2. Pullback into support     real dip depth, cooled RSI, price at support
  3. Fibonacci confluence      close inside the 0.382-0.786 retracement zone
  4. Momentum trigger (H4/H1)  MACD/RSI turning up + bullish bar

Output: dip_score (0-8), dip_confirmed, dip_stage, entry_zone, invalidation.

Usage (library):
    from src.features.dip import detect_dip
    dip = detect_dip(df, trigger_df=h4_df, levels=report["levels"])
"""

from __future__ import annotations

from typing import Dict, Optional

import pandas as pd

from src.features.indicators import macd, rsi

# Score thresholds and pullback bounds (configurable).
CONFIRM_THRESHOLD = 6
WATCH_THRESHOLD = 4
MIN_DIP_DEPTH = 0.005  # 0.5% - a real pullback, not just noise
MAX_DIP_DEPTH = 0.15  # 15% - beyond this it is a break, not a dip
COOLED_RSI_MIN = 30
COOLED_RSI_MAX = 55
TREND_ADX = 20.0


def _nan0(value) -> float:
    """Coerce NaN to 0.0 (NaN is truthy, so ``x or 0.0`` does NOT work)."""
    v = float(value)
    return v if v == v else 0.0


def _momentum_trigger(trigger_df: pd.DataFrame) -> Dict:
    """
    Bullish momentum trigger from an intraday (H4/H1) frame:
    MACD histogram rising, RSI turning up, and/or a bullish last bar.
    At least 2 of 3 must hold for the trigger to fire.
    """
    if trigger_df is None or len(trigger_df) < 30:
        return {"triggered": False, "details": "insufficient trigger data"}

    close = trigger_df["close"]
    # Reuse pre-computed indicator columns when present (e.g. the D1 frame
    # already has rsi_14/macd_hist from add_all_indicators).
    if "macd_hist" in trigger_df.columns and "rsi_14" in trigger_df.columns:
        rsi_series = trigger_df["rsi_14"]
        macd_hist = trigger_df["macd_hist"]
    else:
        rsi_series = rsi(close, 14)
        macd_hist = macd(close)["macd_hist"]

    hist_rising = bool(macd_hist.iloc[-1] > macd_hist.iloc[-2])
    rsi_up = bool(rsi_series.iloc[-1] > rsi_series.iloc[-4])
    bullish_bar = bool(trigger_df["close"].iloc[-1] > trigger_df["open"].iloc[-1])

    fired = sum([hist_rising, rsi_up, bullish_bar]) >= 2

    return {
        "triggered": fired,
        "macd_hist_rising": hist_rising,
        "rsi_turning_up": rsi_up,
        "bullish_bar": bullish_bar,
        "rsi": round(float(rsi_series.iloc[-1]), 1),
        "details": ("MACD+RSI+bar" if fired else "not yet confirmed"),
    }


def _last_swing_high(df: pd.DataFrame, levels: Dict) -> Optional[float]:
    """High of the last up leg from the confluence engine, else recent max."""
    leg = levels.get("last_up_leg")
    if leg:
        return float(leg[1])
    highs = df["high"].tail(60)
    if highs.empty:
        return None
    return float(highs.max())


def detect_dip(
    df: pd.DataFrame,
    trigger_df: Optional[pd.DataFrame] = None,
    levels: Optional[Dict] = None,
) -> Dict:
    """
    Evaluate the Buy-the-Dip setup for the latest bar of ``df``.

    Parameters
    ----------
    df : OHLCV frame with indicators (add_all_indicators + detect_regime).
    trigger_df : optional intraday frame (H4/H1) for the momentum trigger.
    levels : optional dict from src.features.levels.levels_summary; computed
        here if not provided.

    Returns
    -------
    dict with dip_score, dip_confirmed, dip_stage, entry_zone,
    invalidation_level, target, dip_depth_pct and a components breakdown.
    """
    if levels is None:
        from src.features.levels import levels_summary

        levels = levels_summary(df)

    latest = df.iloc[-1]
    close = float(latest.get("close", 0.0))
    atr = _nan0(latest.get("atr_14", 0.0))
    sma20 = latest.get("sma_20")
    sma50 = latest.get("sma_50")
    sma200 = latest.get("sma_200")
    adx = _nan0(latest.get("adx", 0.0))
    rsi14 = _nan0(latest.get("rsi_14", 50.0))

    comp: Dict = {}

    # 1. Uptrend structure
    comp["above_sma200"] = bool(sma200 is not None and close > sma200)
    comp["ma_stack"] = bool(
        sma20 is not None
        and sma50 is not None
        and sma200 is not None
        and sma20 > sma50 > sma200
    )
    # 2. Trend strength
    comp["trend"] = bool(adx >= TREND_ADX)

    # Bullish-structure gate: above SMA200 AND net-bullish directional bias.
    # The bias uses the same 4 factors as the scanner: price vs SMA200,
    # RSI vs 50, MACD histogram sign, and ADX/DI alignment.
    plus_di = _nan0(latest.get("plus_di", 0.0))
    minus_di = _nan0(latest.get("minus_di", 0.0))
    macd_hist = _nan0(latest.get("macd_hist", 0.0))
    bull = int(close > sma200) + int(rsi14 > 50) + int(macd_hist > 0)
    bear = int(close < sma200) + int(rsi14 < 50) + int(macd_hist < 0)
    if adx > 25:
        bull += int(plus_di > minus_di)
        bear += int(minus_di > plus_di)
    bias_score = bull - bear  # -4 .. +4
    comp["bias_score"] = bias_score
    comp["bullish_structure"] = bool(comp["above_sma200"] and bias_score >= 0)

    # 3. Pullback depth vs the last swing high
    last_high = _last_swing_high(df, levels)
    if last_high and last_high > 0:
        dip_depth = (last_high - close) / last_high
    else:
        dip_depth = 0.0
    comp["dip_depth_pct"] = round(dip_depth * 100, 2)
    comp["pullback"] = bool(MIN_DIP_DEPTH <= dip_depth <= MAX_DIP_DEPTH)

    # 4. Momentum cooled off (RSI in the 30-55 band)
    comp["cooled"] = bool(COOLED_RSI_MIN <= rsi14 <= COOLED_RSI_MAX)

    # 5. Price at the nearest confluence support
    support = levels.get("nearest_support")
    support_price = float(support["price"]) if support else None
    at_support = False
    if support_price is not None:
        tolerance = float(levels.get("tolerance", 0.0) or atr)
        at_support = bool(support_price <= close <= support_price + max(atr, tolerance))
    comp["at_support"] = at_support

    # 6. Fibonacci confluence (close inside the 0.382-0.786 retracement zone)
    fib_ok = False
    leg = levels.get("last_up_leg")
    if leg:
        lo, hi = float(leg[0]), float(leg[1])
        rng = hi - lo
        if rng > 0:
            fib_ok = bool(hi - 0.786 * rng <= close <= hi - 0.382 * rng)
    comp["fib_zone"] = fib_ok

    # 7. Momentum trigger (intraday frame if given, else the same frame)
    if trigger_df is not None:
        trigger = _momentum_trigger(trigger_df)
    else:
        trigger = _momentum_trigger(df)
    comp["trigger"] = trigger

    score = sum(
        [
            comp["above_sma200"],
            comp["ma_stack"],
            comp["trend"],
            comp["pullback"],
            comp["cooled"],
            comp["at_support"],
            comp["fib_zone"],
            trigger["triggered"],
        ]
    )

    # Trade levels
    entry_zone = None
    invalidation = None
    if support_price is not None:
        tolerance = _nan0(levels.get("tolerance", 0.0)) or atr
        zone_low = round(support_price - 0.5 * tolerance, 5)
        zone_high = round(min(close, support_price + 0.5 * tolerance), 5)
        # Clamp so the zone is never inverted (close may sit below support).
        entry_zone = (zone_low, max(zone_low, zone_high))
        invalidation = round(support_price - atr, 5) if atr else None

    resistance = levels.get("nearest_resistance")
    target = round(resistance["price"], 5) if resistance else None

    # Stage classification (bullish-structure gate takes precedence).
    # "Confirmed" additionally requires an actual pullback - buying a dip that
    # is not a dip is a semantic error.
    if not comp["bullish_structure"]:
        stage = "No Uptrend"
    elif invalidation is not None and close < invalidation:
        stage = "Support Broken"
    elif dip_depth > MAX_DIP_DEPTH:
        stage = "Deep Pullback"
    elif score >= CONFIRM_THRESHOLD and comp["pullback"]:
        stage = "Confirmed"
    elif score >= WATCH_THRESHOLD:
        stage = "In Pullback"
    else:
        stage = "Not a Dip"

    return {
        "dip_score": score,
        "dip_confirmed": bool(stage == "Confirmed"),
        "dip_stage": stage,
        "entry_zone": entry_zone,
        "invalidation_level": invalidation,
        "target": target,
        "dip_depth_pct": comp["dip_depth_pct"],
        "trigger": trigger["details"],
        "components": {k: v for k, v in comp.items() if k != "trigger"},
    }


if __name__ == "__main__":
    print("NexusQuant Buy-the-Dip module ready.")
