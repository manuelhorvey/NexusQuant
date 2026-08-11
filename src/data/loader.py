"""
NexusQuant - Data Loading & Cleaning Module
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Optional, Union
import yaml


def load_settings(config_path: str = "config/settings.yaml") -> dict:
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def load_data(
    filepath: Union[str, Path],
    symbol: Optional[str] = None,
) -> pd.DataFrame:
    """
    Load historical data from CSV or Parquet.
    Supports the format you exported (date, symbol, open, high, low, close, volume, spread_points).
    """
    filepath = Path(filepath)

    if filepath.suffix.lower() == ".parquet":
        df = pd.read_parquet(filepath)
    else:
        df = pd.read_csv(filepath)

    # Standardize column names
    df.columns = [c.lower().strip() for c in df.columns]

    # Parse date
    if "date" in df.columns:
        df["date"] = pd.to_datetime(df["date"])
        df = df.set_index("date")
    elif "time" in df.columns:
        df["time"] = pd.to_datetime(df["time"])
        df = df.set_index("time")

    df = df.sort_index()

    # Filter symbol if provided
    if symbol and "symbol" in df.columns:
        df = df[df["symbol"] == symbol].copy()

    # Ensure required columns exist
    required = ["open", "high", "low", "close"]
    for col in required:
        if col not in df.columns:
            raise ValueError(f"Missing required column: {col}")

    # Volume handling
    if "volume" not in df.columns:
        df["volume"] = 0

    return df


def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Clean the raw data:
    - Remove duplicates
    - Handle missing values
    - Remove zero/negative prices
    - Optional: filter very low volume bars
    """
    df = df.copy()

    # Remove exact duplicates
    df = df[~df.index.duplicated(keep="first")]

    # Remove rows with invalid prices
    price_cols = ["open", "high", "low", "close"]
    df = df[(df[price_cols] > 0).all(axis=1)]

    # Basic OHLC integrity
    df = df[df["high"] >= df["low"]]
    df = df[df["high"] >= df["open"]]
    df = df[df["high"] >= df["close"]]
    df = df[df["low"] <= df["open"]]
    df = df[df["low"] <= df["close"]]

    # Forward fill very small gaps (optional)
    df = df.ffill(limit=2)

    # Drop remaining NaNs
    df = df.dropna(subset=price_cols)

    return df


def save_processed(
    df: pd.DataFrame,
    symbol: str = "XAUUSD",
    timeframe: str = "D1",
    output_dir: str = "data/processed",
) -> Path:
    """Save cleaned data as Parquet."""
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    filename = f"{symbol}_{timeframe}.parquet"
    full_path = output_path / filename

    df.to_parquet(full_path, compression="zstd")
    print(f"Saved processed data → {full_path}")
    return full_path


def load_processed(
    symbol: str = "XAUUSD",
    timeframe: str = "D1",
    data_dir: str = "data/processed",
) -> pd.DataFrame:
    """Load previously cleaned Parquet file."""
    path = Path(data_dir) / f"{symbol}_{timeframe}.parquet"
    if not path.exists():
        raise FileNotFoundError(f"Processed file not found: {path}")
    return pd.read_parquet(path)


if __name__ == "__main__":
    # Example usage
    print("NexusQuant Data Loader ready.")
    print("Place your raw CSV/JSON files in data/raw/ then run cleaning.")