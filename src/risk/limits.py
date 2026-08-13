"""
NexusQuant - RiskManager: trading limits & session state.

Tracks a paper/live session and gates new trades:

* max daily loss  - halt for the day once realized PnL breaches the floor.
* max weekly loss - halt for the week.
* max concurrent  - maximum number of open positions.
* max heat        - maximum portfolio risk (sum of risk $ / equity).

``record_pnl`` accumulates realized PnL and updates the halted flags; the
``roll_day``/``roll_week`` methods reset the counters at the boundaries.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict


@dataclass
class RiskManager:
    equity: float
    max_daily_loss_pct: float = 0.02  # -2% of equity / day
    max_weekly_loss_pct: float = 0.04  # -4% of equity / week
    max_concurrent: int = 5
    max_heat_pct: float = 0.04  # max portfolio risk / equity

    daily_pnl: float = 0.0
    weekly_pnl: float = 0.0
    n_open: int = 0
    heat: float = 0.0  # current portfolio risk / equity
    daily_halted: bool = False
    weekly_halted: bool = False

    # ------------------------------------------------------------------
    def record_pnl(self, pnl: float) -> Dict:
        """Accumulate realized PnL and re-evaluate the loss limits."""
        self.daily_pnl += pnl
        self.weekly_pnl += pnl
        if self.daily_pnl <= -self.max_daily_loss_pct * self.equity:
            self.daily_halted = True
        if self.weekly_pnl <= -self.max_weekly_loss_pct * self.equity:
            self.weekly_halted = True
        return self.status()

    def roll_day(self) -> None:
        """Reset the daily counter (call at the start of a new day)."""
        self.daily_pnl = 0.0
        self.daily_halted = False

    def roll_week(self) -> None:
        """Reset the weekly counter (call at the start of a new week)."""
        self.weekly_pnl = 0.0
        self.weekly_halted = False

    def can_open(self, trade_risk_pct: float) -> Dict:
        """
        Gate for a new position risking ``trade_risk_pct`` of equity.

        Returns {allowed, reason}.
        """
        reasons = []
        if self.daily_halted:
            reasons.append("daily loss limit hit")
        if self.weekly_halted:
            reasons.append("weekly loss limit hit")
        if self.n_open >= self.max_concurrent:
            reasons.append(f"max concurrent positions ({self.max_concurrent})")
        if self.heat + trade_risk_pct > self.max_heat_pct:
            reasons.append(
                f"portfolio heat {self.heat + trade_risk_pct:.1%} "
                f"> {self.max_heat_pct:.1%}"
            )
        if reasons:
            return {"allowed": False, "reason": "; ".join(reasons)}
        return {"allowed": True, "reason": "ok"}

    def open_position(self, trade_risk_pct: float) -> Dict:
        """Register a new position (after can_open passed)."""
        self.n_open += 1
        self.heat += trade_risk_pct
        return {"n_open": self.n_open, "heat": self.heat}

    def close_position(self, trade_risk_pct: float) -> None:
        self.n_open = max(0, self.n_open - 1)
        self.heat = max(0.0, self.heat - trade_risk_pct)

    def status(self) -> Dict:
        def _pct(v: float) -> float:
            return round(v / self.equity * 100, 2) if self.equity > 0 else 0.0

        return {
            "equity": self.equity,
            "daily_pnl": round(self.daily_pnl, 2),
            "daily_pnl_pct": _pct(self.daily_pnl),
            "weekly_pnl_pct": _pct(self.weekly_pnl),
            "n_open": self.n_open,
            "heat_pct": round(self.heat * 100, 2),
            "daily_halted": self.daily_halted,
            "weekly_halted": self.weekly_halted,
            "trading_enabled": not (self.daily_halted or self.weekly_halted),
        }
