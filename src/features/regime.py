"""
NexusQuant - Market Regime Detection
"""

import pandas as pd
import numpy as np
from typing import Dict


def linear_regression_slope(series: pd.Series, lookback: int = 20) -> pd.Series:
    """Calculate rolling linear regression slope of price."""
    def slope(x):
        if len(x) < 2:
            return np.nan
        y = np.arange(len(x))
        return np.polyfit(y, x, 1)[0]
    return series.rolling(lookback).apply(slope, raw=True)


def detect_regime(df: pd.DataFrame, adx_threshold: float = 25.0) -> pd.DataFrame:
    """
    Classify market regime for each bar.
    Possible regimes: Bull Trend | Bear Trend | Range / Chop | High Volatility
    """
    df = df.copy()

    # Required columns check
    required = ["close", "adx", "sma_200", "atr_14"]
    for col in required:
        if col not in df.columns:
            raise ValueError(f"Missing column for regime detection: {col}. Run add_all_indicators first.")

    # Trend strength & direction
    df["slope_20"] = linear_regression_slope(df["close"], 20)
    df["price_vs_200sma"] = df["close"] / df["sma_200"] - 1

    # Volatility regime (simple z-score of ATR)
    atr_median = df["atr_14"].rolling(100).median()
    df["atr_ratio"] = df["atr_14"] / atr_median

    conditions = [
        (df["adx"] >= adx_threshold) & (df["close"] > df["sma_200"]) & (df["slope_20"] > 0),
        (df["adx"] >= adx_threshold) & (df["close"] < df["sma_200"]) & (df["slope_20"] < 0),
        (df["atr_ratio"] > 1.8),
    ]
    choices = ["Bull Trend", "Bear Trend", "High Volatility"]

    df["regime"] = np.select(conditions, choices, default="Range / Chop")

    # Confidence score (simple)
    df["regime_confidence"] = np.where(
        df["adx"] > 40, 0.85,
        np.where(df["adx"] > 25, 0.70, 0.55)
    )

    return df


def get_current_regime_summary(df: pd.DataFrame) -> Dict:
    """Return a clean summary of the latest regime."""
    latest = df.iloc[-1]

    return {
        "regime": latest.get("regime", "Unknown"),
        "adx": round(latest.get("adx", 0), 2),
        "price_vs_200sma_pct": round(latest.get("price_vs_200sma", 0) * 100, 2),
        "slope_20": round(latest.get("slope_20", 0), 5),
        "confidence": round(latest.get("regime_confidence", 0), 2),
        "atr_14": round(latest.get("atr_14", 0), 2),
    }


if __name__ == "__main__":
    print("NexusQuant Regime Detection module ready.")