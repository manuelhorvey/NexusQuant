"""
NexusQuant - Risk metrics.

Per-trade and portfolio VaR, correlation matrices, portfolio heat, and
correlation-aware position limits. Pure functions on top of OHLCV/returns
frames - unit-testable headless.
"""

from __future__ import annotations

import math
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

# 95% / 99% normal quantiles
Z_95 = 1.645
Z_99 = 2.326


# ---------------------------------------------------------------------------
# Per-trade VaR
# ---------------------------------------------------------------------------


def trade_var(qty: float, atr: float, z: float = Z_95, hold_bars: int = 1) -> float:
    """
    Parametric VaR for one long position over ``hold_bars``:
    VaR = z * ATR * qty * sqrt(hold_bars)  (normal / iid assumption).
    """
    return z * max(atr, 0.0) * qty * math.sqrt(max(hold_bars, 1))


def trade_var_pct(
    qty: float,
    entry: float,
    atr: float,
    equity: float,
    z: float = Z_95,
    hold_bars: int = 1,
) -> float:
    """Per-trade VaR as a % of equity."""
    if equity <= 0:
        return 0.0
    return trade_var(qty, atr, z, hold_bars) / equity


# ---------------------------------------------------------------------------
# Portfolio VaR (correlation-aware)
# ---------------------------------------------------------------------------


def portfolio_var(
    notionals: List[float],
    vol_pcts: List[float],
    corr: np.ndarray,
    z: float = Z_95,
    horizon: int = 1,
) -> float:
    """
    Portfolio VaR from per-symbol notionals and daily vol (as fractions),
    using the correlation matrix:

        sigma_p = sqrt( (w*sigma)' C (w*sigma) );  VaR = z * sigma_p * N.
    """
    n = len(notionals)
    if n == 0:
        return 0.0
    total = float(sum(notionals))
    if total <= 0:
        return 0.0
    w = np.asarray(notionals, dtype=float) / total
    sigma = np.asarray(vol_pcts, dtype=float)
    weighted = w * sigma
    port_sigma = float(
        np.sqrt(max(weighted @ np.asarray(corr, dtype=float) @ weighted, 0.0))
    )
    return z * port_sigma * total * math.sqrt(max(horizon, 1))


def returns_correlation(returns: pd.DataFrame) -> pd.DataFrame:
    """Pairwise correlation of daily returns (symbols in columns)."""
    return returns.corr()


def asset_vol_pct(returns: pd.DataFrame) -> pd.Series:
    """Per-symbol daily volatility as a fraction of price (returns std)."""
    return returns.std()


# ---------------------------------------------------------------------------
# Portfolio heat & correlation-aware limits
# ---------------------------------------------------------------------------


def portfolio_heat(positions: List[Dict], equity: float) -> float:
    """
    Sum of dollars at risk across positions / equity.
    ``positions``: list of {"entry", "stop", "qty"}.
    """
    if equity <= 0:
        return 0.0
    total_risk = sum(max(p["entry"] - p["stop"], 0.0) * p["qty"] for p in positions)
    return total_risk / equity


def top_correlated_pairs(corr: pd.DataFrame, top: int = 6) -> List[Dict]:
    """
    The ``top`` most-correlated distinct pairs, as
    ``[{"pair": "A / B", "corr": 0.7}, ...]``.

    Uses the strict upper triangle so each pair appears once and the
    diagonal is excluded. NaN values (pandas >= 3 keeps them on stack())
    are dropped explicitly.
    """
    if len(corr) < 2:
        return []
    c = corr.where(np.triu(np.ones(corr.shape), k=1).astype(bool))
    pairs = c.stack().dropna().sort_values(ascending=False).head(top)
    return [
        {"pair": f"{i} / {j}", "corr": round(float(v), 3)}
        for (i, j), v in pairs.items()
    ]


def average_correlation(
    new_symbol: str, holdings: List[str], corr: pd.DataFrame
) -> Optional[float]:
    """Mean correlation of ``new_symbol`` with the current holdings."""
    if not holdings:
        return None
    vals = [
        corr.loc[new_symbol, h] for h in holdings if h in corr.index and h != new_symbol
    ]
    if not vals:
        return None
    return float(np.mean(vals))


def check_correlation_limit(
    new_symbol: str,
    holdings: List[str],
    corr: pd.DataFrame,
    max_corr: float = 0.6,
) -> Dict:
    """
    Correlation-aware gate: refuse (or flag) a new position that is too
    correlated with the existing book.

    Returns {allowed, avg_corr, reason}.
    """
    avg = average_correlation(new_symbol, holdings, corr)
    if avg is None:
        return {"allowed": True, "avg_corr": None, "reason": "no holdings"}
    if avg > max_corr:
        return {
            "allowed": False,
            "avg_corr": round(avg, 3),
            "reason": f"avg correlation {avg:.2f} > {max_corr}",
        }
    return {
        "allowed": True,
        "avg_corr": round(avg, 3),
        "reason": f"avg correlation {avg:.2f} <= {max_corr}",
    }
