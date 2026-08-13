"""
NexusQuant - Position sizing.

Three institutional sizing methods, all returning a quantity (units) for a
long position defined by (entry, stop):

1. ``fractional_qty``     - risk a fixed % of equity per trade.
2. ``vol_target_qty``     - target a per-trade volatility contribution
   (notional = equity * vol_target / (asset vol * sqrt(hold))), capped at a
   max risk %.
3. ``kelly_qty``          - fractional Kelly from a win probability and a
   payoff ratio (avg win per unit risked):  f* = p - (1-p)/b.
"""

from __future__ import annotations

import math
from typing import Optional


def _risk_per_unit(entry: float, stop: float, direction: str) -> float:
    """Distance from entry to stop, positive for both sides."""
    return (stop - entry) if direction == "short" else (entry - stop)


def fractional_qty(
    equity: float,
    entry: float,
    stop: float,
    risk_pct: float,
    direction: str = "long",
) -> float:
    """Quantity so that risk-per-unit * qty == equity * risk_pct."""
    risk_per_unit = _risk_per_unit(entry, stop, direction)
    if risk_per_unit <= 0 or equity <= 0:
        return 0.0
    return (equity * risk_pct) / risk_per_unit


def vol_target_qty(
    equity: float,
    entry: float,
    stop: float,
    atr_pct: float,
    vol_target: float,
    hold_bars: int = 1,
    cap_risk_pct: Optional[float] = None,
    direction: str = "long",
) -> float:
    """
    Volatility-targeted size.

    ``atr_pct`` is the asset's per-bar volatility as a fraction of price
    (e.g. 0.01 = 1%). The trade's volatility over ``hold_bars`` is
    approximated as ``atr_pct * sqrt(hold_bars)`` (iid assumption), so:

        notional = equity * vol_target / (atr_pct * sqrt(hold_bars))

    ``vol_target`` is the desired per-trade volatility contribution (e.g.
    0.02 = 2%). Result is capped at ``cap_risk_pct`` of equity risked.
    """
    if atr_pct <= 0 or equity <= 0 or entry <= 0:
        return 0.0
    sigma_trade = atr_pct * math.sqrt(max(hold_bars, 1))
    notional = equity * vol_target / sigma_trade
    qty = notional / entry
    if cap_risk_pct is not None:
        qty = min(
            qty, fractional_qty(equity, entry, stop, cap_risk_pct, direction=direction)
        )
    return max(qty, 0.0)


def kelly_fraction(
    p: float,
    payoff: float,
    fraction: float = 0.5,
) -> float:
    """
    (Fractional) Kelly fraction of equity to risk: f = fraction * (p - (1-p)/b).

    ``payoff`` (b) is average win per unit risked (e.g. 1.5 = 1.5R).
    Returns 0 when the edge is negative or undefined.
    """
    if not (0 < p < 1) or payoff <= 0:
        return 0.0
    edge = p - (1.0 - p) / payoff
    return max(0.0, edge) * fraction


def kelly_qty(
    equity: float,
    entry: float,
    stop: float,
    p: float,
    payoff: float,
    fraction: float = 0.5,
    cap_risk_pct: Optional[float] = None,
    direction: str = "long",
) -> float:
    """Quantity implied by (fractional) Kelly, capped at a max risk %."""
    f = kelly_fraction(p, payoff, fraction)
    qty = fractional_qty(equity, entry, stop, f, direction=direction) if f > 0 else 0.0
    if cap_risk_pct is not None:
        qty = min(
            qty, fractional_qty(equity, entry, stop, cap_risk_pct, direction=direction)
        )
    return max(qty, 0.0)


def size_position(
    equity: float,
    entry: float,
    stop: float,
    mode: str = "fractional",
    risk_pct: float = 0.01,
    atr_pct: Optional[float] = None,
    vol_target: float = 0.02,
    hold_bars: int = 1,
    p: float = 0.5,
    payoff: float = 1.5,
    kelly_fraction_ratio: float = 0.5,
    cap_risk_pct: Optional[float] = None,
    direction: str = "long",
) -> float:
    """Dispatch wrapper used by the backtest engine and the risk CLI.

    ``direction`` threads into the underlying methods so a short (entry
    below stop) still sizes on ``abs(entry - stop)`` risk per unit."""
    if mode == "voltarget" and atr_pct:
        return vol_target_qty(
            equity,
            entry,
            stop,
            atr_pct,
            vol_target,
            hold_bars,
            cap_risk_pct=cap_risk_pct,
            direction=direction,
        )
    if mode == "kelly":
        return kelly_qty(
            equity,
            entry,
            stop,
            p,
            payoff,
            fraction=kelly_fraction_ratio,
            cap_risk_pct=cap_risk_pct,
            direction=direction,
        )
    return fractional_qty(equity, entry, stop, risk_pct, direction=direction)


def risk_dollars(
    qty: float, entry: float, stop: float, direction: str = "long"
) -> float:
    """Dollars at risk for a position (stop distance x quantity)."""
    return max(_risk_per_unit(entry, stop, direction), 0.0) * qty


def risk_pct_of_equity(
    qty: float, entry: float, stop: float, equity: float, direction: str = "long"
) -> float:
    """Risk as a fraction of equity."""
    if equity <= 0:
        return 0.0
    return risk_dollars(qty, entry, stop, direction=direction) / equity
