"""
NexusQuant - Causal Buy-the-Dip signal series for backtesting.

Evaluates the same rule set as ``src.features.dip.detect_dip`` on **every
bar**, with strict no-lookahead semantics:

* indicators (SMA/EMA/RSI/MACD/ADX/ATR) use only data up to the current bar
  (they are causal by construction);
* swing highs/lows are usable only after their confirmation window closes
  (a fractal at bar ``i`` becomes actionable at bar ``i + right``);
* the last up/down legs and the entry zone / invalidation / target are built
  from those confirmed swings and the previous bar's classical pivots.

One deliberate simplification versus the live engine: confluence clustering
is replaced by the last *confirmed* swing high/low (the same structural
levels the live pullback / leg logic is built on). This keeps the series
fully vectorized (O(n), fast over whole universes) and still honours the
signal semantics. See README for the full trade-off discussion.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.features.dip import (
    CONFIRM_THRESHOLD,
    WATCH_THRESHOLD,
    MIN_DIP_DEPTH,
    MAX_DIP_DEPTH,
    COOLED_RSI_MIN,
    COOLED_RSI_MAX,
    TREND_ADX,
)
from src.features.levels import _collapse_plateaus
from src.features.rally import (
    CONFIRM_THRESHOLD as RALLY_CONFIRM_THRESHOLD,
    WATCH_THRESHOLD as RALLY_WATCH_THRESHOLD,
    MIN_RALLY_DEPTH,
    MAX_RALLY_DEPTH,
    STRETCHED_RSI_MIN,
    TREND_ADX as RALLY_TREND_ADX,
)


def _causal_swings(
    df: pd.DataFrame,
    left: int = 2,
    right: int = 2,
) -> tuple[pd.Series, pd.Series]:
    """
    Causal swing-high / swing-low flags.

    A swing at bar ``i`` (fractal window ``i-left .. i+right``) is only
    actionable once bar ``i+right`` has closed, so the flags are shifted by
    ``right``. Plateau runs are collapsed to their extreme bar (same rule as
    the live engine, minus the shift).
    """
    high, low = df["high"], df["low"]
    window = left + right + 1

    sh = (high == high.rolling(window, center=True).max()).fillna(False)
    sl = (low == low.rolling(window, center=True).min()).fillna(False)
    sh = _collapse_plateaus(sh, high, take_max=True)
    sl = _collapse_plateaus(sl, low, take_max=False)
    # No lookahead: usable from the bar after the confirmation window.
    return sh.shift(right).fillna(False), sl.shift(right).fillna(False)


def _last_swing_levels(df: pd.DataFrame) -> dict:
    """
    Vectorized last confirmed swing high/low prices AND the last up/down legs
    (index-based) for every bar. Returns dict of arrays indexed like df.
    """
    sh, sl = _causal_swings(df)
    high, low = df["high"], df["low"]
    n = len(df)
    idx = np.arange(n)

    sh_i = pd.Series(np.where(sh.values, idx, -1), index=df.index)
    sl_i = pd.Series(np.where(sl.values, idx, -1), index=df.index)
    sh_ff = sh_i.where(sh_i >= 0).ffill()  # last confirmed swing-high index
    sl_ff = sl_i.where(sl_i >= 0).ffill()  # last confirmed swing-low index

    last_high_i = sh_ff.fillna(-1).astype(int).values
    last_low_i = sl_ff.fillna(-1).astype(int).values

    def _level_price(series: pd.Series, swing_idx: np.ndarray) -> np.ndarray:
        out = np.full(n, np.nan)
        valid = swing_idx >= 0
        out[valid] = series.values[swing_idx[valid]]
        return out

    last_sh = _level_price(high, last_high_i)
    last_sl = _level_price(low, last_low_i)

    # Up leg  = last confirmed low *before* the last confirmed high.
    up_lo_i = np.full(n, -1)
    v = last_high_i >= 0
    pos = np.clip(last_high_i[v] - 1, 0, n - 1)
    leg = np.nan_to_num(sl_ff.values[pos], nan=-1.0).astype(int)
    up_lo_i[v] = leg
    up_lo = _level_price(low, up_lo_i)
    up_hi = _level_price(high, last_high_i)

    # Down leg = last confirmed high *before* the last confirmed low.
    dn_hi_i = np.full(n, -1)
    v = last_low_i >= 0
    pos = np.clip(last_low_i[v] - 1, 0, n - 1)
    leg = np.nan_to_num(sh_ff.values[pos], nan=-1.0).astype(int)
    dn_hi_i[v] = leg
    dn_hi = _level_price(high, dn_hi_i)
    dn_lo = _level_price(low, last_low_i)

    return {
        "last_swing_high": last_sh,
        "last_swing_low": last_sl,
        "up_leg_lo": up_lo,
        "up_leg_hi": up_hi,
        "down_leg_hi": dn_hi,
        "down_leg_lo": dn_lo,
    }


def _directional_bias(df: pd.DataFrame) -> pd.Series:
    """Vectorized {-4..+4} directional bias (same 4 factors as the scanner)."""
    close = df["close"]
    sma200 = df.get("sma_200")
    rsi14 = df.get("rsi_14")
    macd_hist = df.get("macd_hist")
    adx = df.get("adx")
    plus_di = df.get("plus_di")
    minus_di = df.get("minus_di")

    bull = (
        (close > sma200).astype(int)
        + (rsi14 > 50).astype(int)
        + (macd_hist > 0).astype(int)
    )
    bear = (
        (close < sma200).astype(int)
        + (rsi14 < 50).astype(int)
        + (macd_hist < 0).astype(int)
    )
    strong = (adx > 25).fillna(False)
    bull = bull + (strong & (plus_di > minus_di)).astype(int)
    bear = bear + (strong & (minus_di > plus_di)).astype(int)
    return (bull - bear).fillna(0).astype(int)


def dip_signal_series(
    df: pd.DataFrame,
    confirm_threshold: int = CONFIRM_THRESHOLD,
    watch_threshold: int = WATCH_THRESHOLD,
) -> pd.DataFrame:
    """
    Causal per-bar Buy-the-Dip signal.

    ``df`` must contain OHLCV plus the indicator columns produced by
    ``add_all_indicators`` (sma_20/50/200, rsi_14, macd_hist, adx, plus_di,
    minus_di, atr_14).

    Returns a DataFrame (same index) with:
      score, confirmed, stage, bias_score, bullish_structure,
      above_sma200, ma_stack, trend, pullback, cooled, at_support, fib_zone,
      trigger, dip_depth_pct, entry_lo, entry_hi, invalidation, target.
    """
    close = df["close"].astype(float)
    atr = df["atr_14"].astype(float)
    sma20 = df["sma_20"].astype(float)
    sma50 = df["sma_50"].astype(float)
    sma200 = df["sma_200"].astype(float)
    rsi14 = df["rsi_14"].astype(float)
    macd_hist = df["macd_hist"].astype(float)
    adx = df["adx"].astype(float)

    lv = _last_swing_levels(df)
    last_sh = lv["last_swing_high"]
    last_sl = lv["last_swing_low"]

    # ---- 8 score components (all causal) --------------------------------
    above_sma200 = close > sma200
    ma_stack = (sma20 > sma50) & (sma50 > sma200)
    trend = adx >= TREND_ADX

    with np.errstate(divide="ignore", invalid="ignore"):
        dip_depth = np.where(last_sh > 0, (last_sh - close) / last_sh, np.nan)
    pullback = (dip_depth >= MIN_DIP_DEPTH) & (dip_depth <= MAX_DIP_DEPTH)

    cooled = (rsi14 >= COOLED_RSI_MIN) & (rsi14 <= COOLED_RSI_MAX)

    tol = np.maximum(0.35 * atr, 0.001 * close)
    at_support = (last_sl <= close) & (close <= last_sl + np.maximum(atr, tol))

    rng = lv["up_leg_hi"] - lv["up_leg_lo"]
    with np.errstate(divide="ignore", invalid="ignore"):
        fib_zone = (
            (close >= lv["up_leg_hi"] - 0.786 * rng)
            & (close <= lv["up_leg_hi"] - 0.382 * rng)
            & (rng > 0)
        )

    hist_rising = macd_hist > macd_hist.shift(1)
    rsi_up = rsi14 > rsi14.shift(3)
    bull_bar = close > df["open"].astype(float)
    trigger = (hist_rising.astype(int) + rsi_up.astype(int) + bull_bar.astype(int)) >= 2

    score = (
        above_sma200.astype(int)
        + ma_stack.astype(int)
        + trend.astype(int)
        + pullback.astype(int)
        + cooled.astype(int)
        + at_support.astype(int)
        + fib_zone.astype(int)
        + trigger.astype(int)
    )

    # ---- Bullish-structure gate + stage --------------------------------
    bias = _directional_bias(df)
    bull_structure = above_sma200 & (bias >= 0)

    invalidation = np.where(~np.isnan(last_sl), last_sl - atr, np.nan)
    entry_lo = np.where(~np.isnan(last_sl), last_sl - 0.5 * tol, np.nan)
    entry_hi = np.where(
        ~np.isnan(last_sl), np.minimum(close, last_sl + 0.5 * tol), np.nan
    )
    # clamp so the zone is never inverted (close may sit below support)
    entry_hi = np.maximum(entry_lo, entry_hi)

    # Nearest resistance: last confirmed swing high above price, else none
    # (the engine falls back to an R:R multiple).
    resistance = np.where(last_sh > close, last_sh, np.nan)

    stage = np.where(
        ~bull_structure,
        "No Uptrend",
        np.where(
            (~np.isnan(invalidation)) & (close < invalidation),
            "Support Broken",
            np.where(
                pullback & (dip_depth > MAX_DIP_DEPTH),
                "Deep Pullback",
                np.where(
                    (score >= confirm_threshold) & pullback,
                    "Confirmed",
                    np.where(score >= watch_threshold, "In Pullback", "Not a Dip"),
                ),
            ),
        ),
    )
    confirmed = stage == "Confirmed"

    return pd.DataFrame(
        {
            "score": score,
            "confirmed": confirmed,
            "stage": stage,
            "bias_score": bias,
            "bullish_structure": bull_structure,
            "above_sma200": above_sma200,
            "ma_stack": ma_stack,
            "trend": trend,
            "pullback": pullback,
            "cooled": cooled,
            "at_support": at_support,
            "fib_zone": fib_zone,
            "trigger": trigger,
            "dip_depth_pct": np.round(dip_depth * 100, 2),
            "entry_lo": entry_lo,
            "entry_hi": entry_hi,
            "invalidation": invalidation,
            "resistance": resistance,
        },
        index=df.index,
    )


def rally_signal_series(
    df: pd.DataFrame,
    confirm_threshold: int = RALLY_CONFIRM_THRESHOLD,
    watch_threshold: int = RALLY_WATCH_THRESHOLD,
) -> pd.DataFrame:
    """
    Causal per-bar Sell-the-Rally signal (short-side mirror of
    ``dip_signal_series``). Same no-lookahead semantics; ``df`` needs the
    indicator columns from ``add_all_indicators``.

    Returns a DataFrame (same index) with:
      score, confirmed, stage, bias_score, bearish_structure,
      below_sma200, ma_stack, trend, rally, stretched, at_resistance,
      fib_zone, trigger, rally_depth_pct, entry_lo, entry_hi,
      invalidation, target.
    """
    close = df["close"].astype(float)
    atr = df["atr_14"].astype(float)
    sma20 = df["sma_20"].astype(float)
    sma50 = df["sma_50"].astype(float)
    sma200 = df["sma_200"].astype(float)
    rsi14 = df["rsi_14"].astype(float)
    macd_hist = df["macd_hist"].astype(float)
    adx = df["adx"].astype(float)

    lv = _last_swing_levels(df)
    last_sh = lv["last_swing_high"]
    last_sl = lv["last_swing_low"]

    # ---- 8 score components (all causal) --------------------------------
    below_sma200 = close < sma200
    ma_stack = (sma20 < sma50) & (sma50 < sma200)
    trend = adx >= RALLY_TREND_ADX

    with np.errstate(divide="ignore", invalid="ignore"):
        rally_depth = np.where(last_sl > 0, (close - last_sl) / last_sl, np.nan)
    rally = (rally_depth >= MIN_RALLY_DEPTH) & (rally_depth <= MAX_RALLY_DEPTH)

    stretched = rsi14 >= STRETCHED_RSI_MIN

    tol = np.maximum(0.35 * atr, 0.001 * close)
    at_resistance = (last_sh - np.maximum(atr, tol) <= close) & (close <= last_sh)

    rng = lv["down_leg_hi"] - lv["down_leg_lo"]
    with np.errstate(divide="ignore", invalid="ignore"):
        fib_zone = (
            (close >= lv["down_leg_hi"] - 0.786 * rng)
            & (close <= lv["down_leg_hi"] - 0.382 * rng)
            & (rng > 0)
        )

    hist_falling = macd_hist < macd_hist.shift(1)
    rsi_down = rsi14 < rsi14.shift(3)
    bear_bar = close < df["open"].astype(float)
    trigger = (
        hist_falling.astype(int) + rsi_down.astype(int) + bear_bar.astype(int)
    ) >= 2

    score = (
        below_sma200.astype(int)
        + ma_stack.astype(int)
        + trend.astype(int)
        + rally.astype(int)
        + stretched.astype(int)
        + at_resistance.astype(int)
        + fib_zone.astype(int)
        + trigger.astype(int)
    )

    # ---- Bearish-structure gate + stage --------------------------------
    bias = _directional_bias(df)
    bear_structure = below_sma200 & (bias <= 0)

    invalidation = np.where(~np.isnan(last_sh), last_sh + atr, np.nan)
    entry_hi = np.where(~np.isnan(last_sh), last_sh + 0.5 * tol, np.nan)
    entry_lo = np.where(
        ~np.isnan(last_sh), np.maximum(close, last_sh - 0.5 * tol), np.nan
    )
    # clamp so the zone is never inverted (close may sit above resistance)
    entry_lo = np.minimum(entry_lo, entry_hi)

    # Nearest support: last confirmed swing low below price, else none.
    target = np.where(last_sl < close, last_sl, np.nan)

    stage = np.where(
        ~bear_structure,
        "No Downtrend",
        np.where(
            (~np.isnan(invalidation)) & (close > invalidation),
            "Resistance Broken",
            np.where(
                rally & (rally_depth > MAX_RALLY_DEPTH),
                "Deep Rally",
                np.where(
                    (score >= confirm_threshold) & rally,
                    "Confirmed",
                    np.where(score >= watch_threshold, "In Rally", "Not a Rally"),
                ),
            ),
        ),
    )
    confirmed = stage == "Confirmed"

    return pd.DataFrame(
        {
            "score": score,
            "confirmed": confirmed,
            "stage": stage,
            "bias_score": bias,
            "bearish_structure": bear_structure,
            "below_sma200": below_sma200,
            "ma_stack": ma_stack,
            "trend": trend,
            "rally": rally,
            "stretched": stretched,
            "at_resistance": at_resistance,
            "fib_zone": fib_zone,
            "trigger": trigger,
            "rally_depth_pct": np.round(rally_depth * 100, 2),
            "entry_lo": entry_lo,
            "entry_hi": entry_hi,
            "invalidation": invalidation,
            "target": target,
        },
        index=df.index,
    )


if __name__ == "__main__":
    print("NexusQuant backtest signals module ready.")
