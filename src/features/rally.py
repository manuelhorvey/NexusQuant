"""
NexusQuant - Sell-the-Rally Confirmation Engine

Mirror of the Buy-the-Dip engine (src/features/dip.py) for counter-trend
shorts in downtrends:

  1. Downtrend structure (D1)   price below SMA200, bearish MA stack, ADX
  2. Rally into resistance      real rally off the swing low, RSI stretched
                                (>60 in a downtrend), price at resistance
  3. Fibonacci confluence       close inside the 0.382-0.786 retracement
                                of the last down leg
  4. Momentum trigger (H4/H1)   MACD/RSI turning down + bearish bar

Output: rally_score (0-8), rally_confirmed, rally_stage, entry_zone,
invalidation_level (above the swing high), target (below).

Usage (library):
    from src.features.rally import detect_rally
    rally = detect_rally(df, trigger_df=h4_df, levels=report["levels"])
"""

from __future__ import annotations

from typing import Dict, Optional

import pandas as pd

from src.features.indicators import macd, rsi

# Score thresholds and rally bounds (configurable).
CONFIRM_THRESHOLD = 6
WATCH_THRESHOLD = 4
MIN_RALLY_DEPTH = 0.005  # 0.5% - a real bounce, not noise
MAX_RALLY_DEPTH = 0.15  # 15% - beyond this it is a reversal, not a rally
STRETCHED_RSI_MIN = 60.0
TREND_ADX = 20.0


def _nan0(value) -> float:
    """Coerce NaN to 0.0 (NaN is truthy, so ``x or 0.0`` does NOT work)."""
    v = float(value)
    return v if v == v else 0.0


def _bearish_momentum_trigger(trigger_df: pd.DataFrame) -> Dict:
    """
    Bearish momentum trigger from an intraday (H4/H1) frame: MACD
    histogram falling, RSI turning down, and/or a bearish last bar.
    At least 2 of 3 must hold for the trigger to fire.
    """
    if trigger_df is None or len(trigger_df) < 30:
        return {"triggered": False, "details": "insufficient trigger data"}

    close = trigger_df["close"]
    if "macd_hist" in trigger_df.columns and "rsi_14" in trigger_df.columns:
        rsi_series = trigger_df["rsi_14"]
        macd_hist = trigger_df["macd_hist"]
    else:
        rsi_series = rsi(close, 14)
        macd_hist = macd(close)["macd_hist"]

    hist_falling = bool(macd_hist.iloc[-1] < macd_hist.iloc[-2])
    rsi_down = bool(rsi_series.iloc[-1] < rsi_series.iloc[-4])
    bearish_bar = bool(trigger_df["close"].iloc[-1] < trigger_df["open"].iloc[-1])

    fired = sum([hist_falling, rsi_down, bearish_bar]) >= 2

    return {
        "triggered": fired,
        "macd_hist_falling": hist_falling,
        "rsi_turning_down": rsi_down,
        "bearish_bar": bearish_bar,
        "rsi": round(float(rsi_series.iloc[-1]), 1),
        "details": ("MACD+RSI+bar" if fired else "not yet confirmed"),
    }


def _last_swing_low(df: pd.DataFrame, levels: Dict) -> Optional[float]:
    """Low of the last down leg from the confluence engine, else recent min."""
    leg = levels.get("last_down_leg")
    if leg:
        return float(leg[0])
    lows = df["low"].tail(60)
    if lows.empty:
        return None
    return float(lows.min())


def detect_rally(
    df: pd.DataFrame,
    trigger_df: Optional[pd.DataFrame] = None,
    levels: Optional[Dict] = None,
) -> Dict:
    """
    Evaluate the Sell-the-Rally setup for the latest bar of ``df``.

    Parameters
    ----------
    df : OHLCV frame with indicators (add_all_indicators + detect_regime).
    trigger_df : optional intraday frame (H4/H1) for the momentum trigger.
    levels : optional dict from src.features.levels.levels_summary; computed
        here if not provided.

    Returns
    -------
    dict with rally_score, rally_confirmed, rally_stage, entry_zone,
    invalidation_level, target, rally_depth_pct and a components breakdown.
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

    # 1. Downtrend structure
    comp["below_sma200"] = bool(sma200 is not None and close < sma200)
    comp["ma_stack"] = bool(
        sma20 is not None
        and sma50 is not None
        and sma200 is not None
        and sma20 < sma50 < sma200
    )
    # 2. Trend strength
    comp["trend"] = bool(adx >= TREND_ADX)

    # Bearish-structure gate: below SMA200 AND net-bearish directional
    # bias (same 4-factor bias as the scanner / dip engine, mirrored).
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
    comp["bearish_structure"] = bool(comp["below_sma200"] and bias_score <= 0)

    # 3. Rally depth vs the last swing low
    last_low = _last_swing_low(df, levels)
    if last_low and last_low > 0:
        rally_depth = (close - last_low) / last_low
    else:
        rally_depth = 0.0
    comp["rally_depth_pct"] = round(rally_depth * 100, 2)
    comp["rally"] = bool(MIN_RALLY_DEPTH <= rally_depth <= MAX_RALLY_DEPTH)

    # 4. RSI stretched (overbought in a downtrend)
    comp["stretched"] = bool(rsi14 >= STRETCHED_RSI_MIN)

    # 5. Price at the nearest confluence resistance
    resistance = levels.get("nearest_resistance")
    resistance_price = float(resistance["price"]) if resistance else None
    at_resistance = False
    if resistance_price is not None:
        tolerance = float(levels.get("tolerance", 0.0) or atr)
        at_resistance = bool(
            resistance_price - max(atr, tolerance) <= close <= resistance_price
        )
    comp["at_resistance"] = at_resistance

    # 6. Fibonacci confluence (close inside the 0.382-0.786 retracement
    # of the last DOWN leg, i.e. the bounce zone)
    fib_ok = False
    leg = levels.get("last_down_leg")
    if leg:
        lo, hi = float(leg[0]), float(leg[1])
        rng = hi - lo
        if rng > 0:
            fib_ok = bool(hi - 0.786 * rng <= close <= hi - 0.382 * rng)
    comp["fib_zone"] = fib_ok

    # 7. Momentum trigger (intraday frame if given, else the same frame)
    if trigger_df is not None:
        trigger = _bearish_momentum_trigger(trigger_df)
    else:
        trigger = _bearish_momentum_trigger(df)
    comp["trigger"] = trigger

    score = sum(
        [
            comp["below_sma200"],
            comp["ma_stack"],
            comp["trend"],
            comp["rally"],
            comp["stretched"],
            comp["at_resistance"],
            comp["fib_zone"],
            trigger["triggered"],
        ]
    )

    # Trade levels
    entry_zone = None
    invalidation = None
    if resistance_price is not None:
        tolerance = _nan0(levels.get("tolerance", 0.0)) or atr
        zone_low = round(max(close, resistance_price - 0.5 * tolerance), 5)
        zone_high = round(resistance_price + 0.5 * tolerance, 5)
        # Clamp so the zone is never inverted (close may sit above resistance).
        entry_zone = (min(zone_low, zone_high), max(zone_low, zone_high))
        invalidation = round(resistance_price + atr, 5) if atr else None

    support = levels.get("nearest_support")
    target = round(support["price"], 5) if support else None

    # Stage classification (bearish-structure gate takes precedence).
    # "Confirmed" additionally requires an actual rally off the swing low.
    if not comp["bearish_structure"]:
        stage = "No Downtrend"
    elif invalidation is not None and close > invalidation:
        stage = "Resistance Broken"
    elif rally_depth > MAX_RALLY_DEPTH:
        stage = "Deep Rally"
    elif score >= CONFIRM_THRESHOLD and comp["rally"]:
        stage = "Confirmed"
    elif score >= WATCH_THRESHOLD:
        stage = "In Rally"
    else:
        stage = "Not a Rally"

    return {
        "rally_score": score,
        "rally_confirmed": bool(stage == "Confirmed"),
        "rally_stage": stage,
        "entry_zone": entry_zone,
        "invalidation_level": invalidation,
        "target": target,
        "rally_depth_pct": comp["rally_depth_pct"],
        "trigger": trigger["details"],
        "components": {k: v for k, v in comp.items() if k != "trigger"},
    }


if __name__ == "__main__":
    print("NexusQuant Sell-the-Rally module ready.")
