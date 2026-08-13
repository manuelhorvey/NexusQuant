"""
NexusQuant - Technical Indicators Module
"""

import pandas as pd
import numpy as np
from typing import Dict


def sma(series: pd.Series, period: int) -> pd.Series:
    return series.rolling(window=period).mean()


def ema(series: pd.Series, period: int) -> pd.Series:
    return series.ewm(span=period, adjust=False).mean()


def rsi(series: pd.Series, period: int = 14) -> pd.Series:
    delta = series.diff()
    gain = delta.where(delta > 0, 0.0)
    loss = -delta.where(delta < 0, 0.0)

    avg_gain = gain.ewm(alpha=1 / period, min_periods=period, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1 / period, min_periods=period, adjust=False).mean()

    rs = avg_gain / avg_loss
    return 100 - (100 / (1 + rs))


def macd(
    series: pd.Series,
    fast: int = 12,
    slow: int = 26,
    signal: int = 9,
) -> pd.DataFrame:
    ema_fast = ema(series, fast)
    ema_slow = ema(series, slow)
    macd_line = ema_fast - ema_slow
    signal_line = ema(macd_line, signal)
    histogram = macd_line - signal_line

    return pd.DataFrame(
        {
            "macd": macd_line,
            "macd_signal": signal_line,
            "macd_hist": histogram,
        }
    )


def bollinger_bands(
    series: pd.Series,
    period: int = 20,
    std_dev: float = 2.0,
) -> pd.DataFrame:
    mid = sma(series, period)
    std = series.rolling(window=period).std()
    upper = mid + (std * std_dev)
    lower = mid - (std * std_dev)
    width = (upper - lower) / mid
    pct_b = (series - lower) / (upper - lower)

    return pd.DataFrame(
        {
            "bb_upper": upper,
            "bb_mid": mid,
            "bb_lower": lower,
            "bb_width": width,
            "bb_pct_b": pct_b,
        }
    )


def atr(
    high: pd.Series,
    low: pd.Series,
    close: pd.Series,
    period: int = 14,
) -> pd.Series:
    prev_close = close.shift(1)
    tr1 = high - low
    tr2 = (high - prev_close).abs()
    tr3 = (low - prev_close).abs()
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    return tr.rolling(window=period).mean()


def adx(
    high: pd.Series,
    low: pd.Series,
    close: pd.Series,
    period: int = 14,
) -> pd.DataFrame:
    # Wilder's directional movement. IMPORTANT (P0 fix): -DM is the *drop*
    # in the low (prev_low - low), i.e. ``-low.diff()``. The previous
    # implementation used ``low.diff()`` (positive when the low RISES),
    # which inverted -DI, flipped the +DI/-DI comparison in every trend
    # read, and produced NaN/0 ADX in downtrends (falling lows make
    # ``low.diff()`` negative -> both DMs zero -> the detector was blind to
    # bear trends).
    up = high.diff()  # +DM candidate: how much the high rose
    down = -low.diff()  # -DM candidate: how much the low fell
    plus_dm = up.where((up > down) & (up > 0), 0.0)
    minus_dm = down.where((down > up) & (down > 0), 0.0)

    tr = atr(high, low, close, period=1)  # True range raw

    atr_n = tr.rolling(window=period).mean()
    plus_di = 100 * (plus_dm.rolling(window=period).mean() / atr_n)
    minus_di = 100 * (minus_dm.rolling(window=period).mean() / atr_n)

    dx = (abs(plus_di - minus_di) / (plus_di + minus_di)) * 100
    adx_val = dx.rolling(window=period).mean()

    return pd.DataFrame(
        {
            "adx": adx_val,
            "plus_di": plus_di,
            "minus_di": minus_di,
        }
    )


def ad_line(
    high: pd.Series,
    low: pd.Series,
    close: pd.Series,
    volume: pd.Series,
) -> pd.Series:
    """
    Accumulation/Distribution line.

    CLV = ((close - low) - (high - close)) / (high - low), 0 when flat;
    A/D = cumsum(CLV * volume). Rising = accumulation, falling =
    distribution.
    """
    rng = (high - low).replace(0, np.nan)
    clv = ((close - low) - (high - close)) / rng
    clv = clv.fillna(0.0)
    return (clv * volume).fillna(0).cumsum()


def add_all_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add a complete set of institutional indicators to the OHLCV dataframe.
    """
    df = df.copy()
    close = df["close"]
    high = df["high"]
    low = df["low"]

    # Moving Averages
    for period in [20, 50, 100, 200]:
        df[f"sma_{period}"] = sma(close, period)
        df[f"ema_{period}"] = ema(close, period)

    # RSI
    df["rsi_14"] = rsi(close, 14)

    # MACD
    macd_df = macd(close)
    df = pd.concat([df, macd_df], axis=1)

    # Bollinger Bands
    bb_df = bollinger_bands(close)
    df = pd.concat([df, bb_df], axis=1)

    # ATR
    df["atr_14"] = atr(high, low, close, 14)

    # ADX
    adx_df = adx(high, low, close, 14)
    df = pd.concat([df, adx_df], axis=1)

    # Returns & Volatility
    df["returns"] = close.pct_change()
    df["log_returns"] = np.log(close / close.shift(1))
    df["volatility_20"] = df["returns"].rolling(20).std() * np.sqrt(252)

    # Volume & flow features (graceful when no volume column exists)
    if "volume" in df.columns:
        df["volume_sma_20"] = sma(df["volume"], 20)
        df["relative_volume"] = df["volume"] / df["volume_sma_20"]
        df["obv"] = (np.sign(df["close"].diff()) * df["volume"]).fillna(0).cumsum()
        df["ad_line"] = ad_line(high, low, close, df["volume"])
        df["volume_delta"] = np.sign(close.diff()).fillna(0) * df["volume"]

    return df


# ---------------------------------------------------------------------------
# Moving-average ribbon structure (institutional spec #3)
# ---------------------------------------------------------------------------

# Golden/death cross probability via Monte-Carlo-free historical proxy:
# how many bars in the past had a (SMA50-SMA200)/ATR gap at least as close
# as the current one AND crossed within the next `horizon` bars.
CROSS_HORIZON = 20


def ma_ribbon_summary(df: pd.DataFrame, horizon: int = CROSS_HORIZON) -> Dict:
    """
    Moving-average structure (institutional spec #3): ribbon slope, ribbon
    width (trend momentum proxy) and a data-driven golden/death cross
    probability for the next ``horizon`` bars.

    The cross probability is empirical: over the history of ``df`` we count
    how often a (SMA50-SMA200)/ATR gap at least as tight as today's was
    followed by a cross within ``horizon`` bars. Bars where history is too
    short to know the outcome are excluded.
    """
    if not {"sma_50", "sma_200", "atr_14"}.issubset(df.columns):
        return {"available": False}

    d = df.copy()
    d = d.dropna(subset=["sma_50", "sma_200", "atr_14"])
    if len(d) < 60:
        return {"available": False}

    gap = (d["sma_50"] - d["sma_200"]) / d["atr_14"].replace(0, np.nan)
    gap = gap.fillna(0.0)

    # cross events: sign change of the raw SMA50-SMA200 gap
    raw = d["sma_50"] - d["sma_200"]
    crossed = np.sign(raw).diff().ne(0)
    crossed.iloc[:1] = False

    current = float(gap.iloc[-1])
    cur_dist = abs(current)

    # Empirical cross probability within `horizon` bars for gaps at least
    # this tight (both signs, symmetric).
    n_similar = 0
    n_crossed = 0
    total = len(gap)
    for i in range(total - horizon):
        dist = abs(float(gap.iloc[i]))
        if dist <= cur_dist or (cur_dist == 0 and dist <= 0.5):
            n_similar += 1
            window = crossed.iloc[i + 1 : i + horizon + 1]
            if window.any():
                n_crossed += 1
    cross_prob = round(n_crossed / n_similar, 3) if n_similar > 0 else 0.0

    # ribbon = the 20/50/100/200 family; slope + width from the last bar
    cols = [c for c in ("sma_20", "sma_50", "sma_100", "sma_200") if c in d.columns]
    if len(cols) < 2:
        return {"available": False}
    ribbon_vals = d[cols].iloc[-1].astype(float).sort_values()
    width = float(ribbon_vals.max() - ribbon_vals.min())
    width_pct = width / float(d["close"].iloc[-1]) * 100.0
    # ribbon slope: linear regression over the mean-ribbon series (20 bars)
    mean_ribbon = d[cols].mean(axis=1)
    tail = mean_ribbon.dropna().tail(20)
    if len(tail) < 5:
        slope = 0.0
    else:
        x = np.arange(len(tail), dtype=float)
        slope = float(np.polyfit(x, tail.values, 1)[0]) / (
            float(np.mean(np.abs(tail.values))) or 1.0
        )

    # alignment: 1 when fully stacked bullish (20>50>100>200), -1 bearish
    vals = [float(v) for _, v in ribbon_vals.items()]
    if len(vals) == len(cols):
        stacked_bull = all(vals[i] >= vals[i - 1] for i in range(1, len(vals)))
        stacked_bear = all(vals[i] <= vals[i - 1] for i in range(1, len(vals)))
        alignment = 1.0 if stacked_bull else (-1.0 if stacked_bear else 0.0)
    else:
        alignment = 0.0

    signal = (
        "Bullish"
        if current > 0 and cross_prob < 0.5
        else "Bearish"
        if current < 0 and cross_prob < 0.5
        else "Neutral"
    )
    return {
        "available": True,
        "gap_atr": round(current, 3),
        "cross_direction": "golden" if current > 0 else "death",
        "cross_prob": round(cross_prob, 3),
        "cross_horizon": horizon,
        "ribbon_width_pct": round(width_pct, 3),
        "ribbon_slope": round(slope, 5),
        "ribbon_alignment": round(alignment, 2),
        "signal": signal,
    }


def _linear_slope(series: pd.Series, lookback: int = 20) -> float:
    """Slope of the last ``lookback`` values (per-bar, scaled by mean abs)."""
    vals = series.dropna().tail(lookback)
    if len(vals) < 5:
        return 0.0
    x = np.arange(len(vals), dtype=float)
    y = vals.to_numpy(dtype=float)
    scale = float(np.mean(np.abs(y))) or 1.0
    return float(np.polyfit(x, y, 1)[0]) / scale


def volume_flow_summary(df: pd.DataFrame, lookback: int = 20) -> Dict:
    """
    Report-friendly Volume & Flow section (institutional spec #5): OBV
    slope, A/D slope, relative volume, volume delta and a buyer-vs-seller
    score in [-100, +100].
    """
    if (
        "volume" not in df.columns
        or len(df) <= lookback
        or df["volume"].notna().sum() < lookback
    ):
        return {"available": False}
    # self-contained: compute the flow columns if the caller did not run
    # add_all_indicators (the report/scanner always do, but stay robust)
    if "obv" not in df.columns:
        df = df.copy()
        df["obv"] = (np.sign(df["close"].diff()) * df["volume"]).fillna(0).cumsum()
        df["ad_line"] = ad_line(df["high"], df["low"], df["close"], df["volume"])
        df["volume_delta"] = np.sign(df["close"].diff()).fillna(0) * df["volume"]
        df["relative_volume"] = df["volume"] / df["volume"].rolling(20).mean()
    latest = df.iloc[-1]
    obv_s = _linear_slope(df["obv"], lookback)
    ad_s = _linear_slope(df["ad_line"], lookback)

    # net flow over the window, scaled by total traded volume
    vol20 = float(df["volume"].tail(lookback).sum()) or 1.0
    obv_delta = float(df["obv"].iloc[-1] - df["obv"].iloc[-lookback - 1])
    ad_delta = float(df["ad_line"].iloc[-1] - df["ad_line"].iloc[-lookback - 1])
    score = (obv_delta + ad_delta) / vol20 * 50.0
    score = float(max(-100.0, min(100.0, score)))

    label = (
        "Strong Accumulation"
        if score >= 30
        else "Accumulation"
        if score >= 10
        else "Strong Distribution"
        if score <= -30
        else "Distribution"
        if score <= -10
        else "Neutral"
    )

    def trend(s):
        return "Up" if s > 0.005 else "Down" if s < -0.005 else "Flat"

    return {
        "available": True,
        "obv_slope_20": round(obv_s, 4),
        "obv_trend": trend(obv_s),
        "ad_line_slope_20": round(ad_s, 4),
        "ad_trend": trend(ad_s),
        "relative_volume": round(float(latest.get("relative_volume", 1) or 1), 2),
        "volume_delta_20": round(float(df["volume_delta"].tail(lookback).sum()), 0),
        "buyer_seller_score": round(score, 1),
        "buyer_seller_label": label,
    }


if __name__ == "__main__":
    print("NexusQuant Indicators module loaded successfully.")
