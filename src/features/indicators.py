"""
NexusQuant - Technical Indicators Module
"""

import pandas as pd
import numpy as np
from typing import List


def sma(series: pd.Series, period: int) -> pd.Series:
    return series.rolling(window=period).mean()


def ema(series: pd.Series, period: int) -> pd.Series:
    return series.ewm(span=period, adjust=False).mean()


def rsi(series: pd.Series, period: int = 14) -> pd.Series:
    delta = series.diff()
    gain = delta.where(delta > 0, 0.0)
    loss = -delta.where(delta < 0, 0.0)

    avg_gain = gain.ewm(alpha=1/period, min_periods=period, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1/period, min_periods=period, adjust=False).mean()

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

    return pd.DataFrame({
        "macd": macd_line,
        "macd_signal": signal_line,
        "macd_hist": histogram,
    })


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

    return pd.DataFrame({
        "bb_upper": upper,
        "bb_mid": mid,
        "bb_lower": lower,
        "bb_width": width,
        "bb_pct_b": pct_b,
    })


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
    plus_dm = high.diff()
    minus_dm = low.diff()
    plus_dm = plus_dm.where((plus_dm > minus_dm) & (plus_dm > 0), 0.0)
    minus_dm = minus_dm.where((minus_dm > plus_dm) & (minus_dm > 0), 0.0)

    tr = atr(high, low, close, period=1)  # True range raw

    atr_n = tr.rolling(window=period).mean()
    plus_di = 100 * (plus_dm.rolling(window=period).mean() / atr_n)
    minus_di = 100 * (minus_dm.rolling(window=period).mean() / atr_n)

    dx = (abs(plus_di - minus_di) / (plus_di + minus_di)) * 100
    adx_val = dx.rolling(window=period).mean()

    return pd.DataFrame({
        "adx": adx_val,
        "plus_di": plus_di,
        "minus_di": minus_di,
    })


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

    # Volume features (if available)
    if "volume" in df.columns:
        df["volume_sma_20"] = sma(df["volume"], 20)
        df["relative_volume"] = df["volume"] / df["volume_sma_20"]
        df["obv"] = (np.sign(df["close"].diff()) * df["volume"]).fillna(0).cumsum()

    return df


if __name__ == "__main__":
    print("NexusQuant Indicators module loaded successfully.")