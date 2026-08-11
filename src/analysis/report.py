"""
NexusQuant - Institutional Analysis Report Generator
"""

import pandas as pd
from typing import Dict, Any
from pathlib import Path
import sys

# Allow running from project root
sys.path.append(str(Path(__file__).parent.parent.parent))

from src.features.indicators import add_all_indicators
from src.features.regime import detect_regime, get_current_regime_summary


def generate_full_report(df: pd.DataFrame, symbol: str = "XAUUSD") -> Dict[str, Any]:
    """
    Generate a structured institutional-style analysis report.
    """
    # Ensure indicators and regime exist
    if "rsi_14" not in df.columns:
        df = add_all_indicators(df)
    if "regime" not in df.columns:
        df = detect_regime(df)

    latest = df.iloc[-1]
    prev = df.iloc[-2]

    report = {
        "symbol": symbol,
        "last_date": str(df.index[-1].date()),
        "last_close": round(latest["close"], 3),
        "regime": get_current_regime_summary(df),
        "moving_averages": {
            "sma_50": round(latest.get("sma_50", 0), 3),
            "sma_100": round(latest.get("sma_100", 0), 3),
            "sma_200": round(latest.get("sma_200", 0), 3),
            "ema_50": round(latest.get("ema_50", 0), 3),
            "ema_100": round(latest.get("ema_100", 0), 3),
            "ema_200": round(latest.get("ema_200", 0), 3),
            "price_vs_sma200": "Above" if latest["close"] > latest.get("sma_200", 0) else "Below",
        },
        "momentum": {
            "rsi_14": round(latest.get("rsi_14", 0), 2),
            "macd": round(latest.get("macd", 0), 4),
            "macd_signal": round(latest.get("macd_signal", 0), 4),
            "macd_hist": round(latest.get("macd_hist", 0), 4),
            "bb_pct_b": round(latest.get("bb_pct_b", 0), 3),
        },
        "volatility": {
            "atr_14": round(latest.get("atr_14", 0), 3),
            "bb_width": round(latest.get("bb_width", 0), 4),
            "volatility_20": round(latest.get("volatility_20", 0), 4),
        },
        "trend_strength": {
            "adx": round(latest.get("adx", 0), 2),
            "plus_di": round(latest.get("plus_di", 0), 2),
            "minus_di": round(latest.get("minus_di", 0), 2),
        },
    }

    # Simple signal summary
    bullish_score = 0
    if latest.get("close", 0) > latest.get("sma_200", 0):
        bullish_score += 1
    if latest.get("rsi_14", 50) > 50:
        bullish_score += 1
    if latest.get("macd_hist", 0) > 0:
        bullish_score += 1
    if latest.get("adx", 0) > 25 and latest.get("plus_di", 0) > latest.get("minus_di", 0):
        bullish_score += 1

    report["simple_bias"] = {
        "score": bullish_score,
        "max_score": 4,
        "interpretation": (
            "Bullish" if bullish_score >= 3 else
            "Bearish" if bullish_score <= 1 else
            "Neutral"
        )
    }

    return report


def print_report(report: Dict[str, Any]) -> None:
    """Pretty print the report to console."""
    print("\n" + "="*60)
    print(f"NEXUSQUANT INSTITUTIONAL ANALYSIS — {report['symbol']}")
    print(f"Date: {report['last_date']}  |  Close: {report['last_close']}")
    print("="*60)

    print("\n1. MARKET REGIME")
    for k, v in report["regime"].items():
        print(f"   {k:20}: {v}")

    print("\n2. MOVING AVERAGES")
    for k, v in report["moving_averages"].items():
        print(f"   {k:20}: {v}")

    print("\n3. MOMENTUM")
    for k, v in report["momentum"].items():
        print(f"   {k:20}: {v}")

    print("\n4. TREND STRENGTH (ADX)")
    for k, v in report["trend_strength"].items():
        print(f"   {k:20}: {v}")

    print("\n5. VOLATILITY")
    for k, v in report["volatility"].items():
        print(f"   {k:20}: {v}")

    print("\n6. SIMPLE BIAS")
    print(f"   Score: {report['simple_bias']['score']}/{report['simple_bias']['max_score']}")
    print(f"   Interpretation: {report['simple_bias']['interpretation']}")
    print("="*60 + "\n")


if __name__ == "__main__":
    print("NexusQuant Analysis Report module ready.")
    print("Usage: from src.analysis.report import generate_full_report, print_report")