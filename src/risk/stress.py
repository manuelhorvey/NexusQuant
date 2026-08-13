"""
NexusQuant - Stress Testing (institutional spec #12)

Shocks current positions / the live setup against the three canonical
crashes:

* **2008 GFC** - the deepest post-war drawdown (-54% US equities, VIX to
  ~80, USD strength, ~480 trading days to recover). Not in our 2018+ FX
  history, so canonical parameters are used.
* **COVID-19 (2020)** - the fastest crash (-34% in ~23 sessions, VIX to
  ~82) followed by a V-shaped recovery. Fully inside our data: where a
  symbol has history, the *realized* crash is extracted instead of the
  canonical parameters.
* **2022 drawdown** - a slow grind (-25% over ~250 sessions, rising
  rates). Also inside our data; realized stats are used when available.

Every scenario reports: shocked loss (drawdown-adjusted), shocked VaR95
(volatility-multiplied), the loss as % of equity, and whether the
daily-loss limit would be breached - so "how does this position behave in
a 2008?" gets a concrete answer, not a vibe.

Usage (library):
    from src.risk.stress import stress_position, stress_portfolio, \
        stress_table_from_report, historical_crash_stats

    loss = stress_position({"qty": 1000, "entry": 1.10, "atr": 0.004,
                            "direction": "long"}, SCENARIOS["COVID-19 2020"],
                           equity=100_000)
"""

from __future__ import annotations

from typing import Dict, List, Optional

import numpy as np
import pandas as pd

# Canonical crash parameters (documented, conservative).
SCENARIOS: Dict[str, Dict] = {
    "2008 GFC": {
        "name": "2008 GFC",
        "drawdown_pct": 54.0,  # US equities peak-to-trough
        "vol_mult": 3.0,  # realized vol multiple vs normal
        "horizon_days": 200,  # trough reached over ~200 sessions
        "recovery_days": 480,
        "note": "canonical GFC parameters (VIX ~80, -54% equities, USD strength)",
    },
    "COVID-19 2020": {
        "name": "COVID-19 2020",
        "drawdown_pct": 34.0,
        "vol_mult": 2.5,
        "horizon_days": 23,  # fastest modern crash
        "recovery_days": 120,
        "note": "fast crash + V-shaped recovery (realized from local data when available)",
    },
    "2022 drawdown": {
        "name": "2022 drawdown",
        "drawdown_pct": 25.0,
        "vol_mult": 1.8,
        "horizon_days": 250,  # slow grind
        "recovery_days": 300,
        "note": "sustained bear market + rising rates",
    },
}

DAILY_LIMIT_PCT = 2.0  # default daily loss limit used by breach checks


# ---------------------------------------------------------------------------
# historical realization (data-grounded COVID / 2022)
# ---------------------------------------------------------------------------


def _rolling_max_drawdown(closes: pd.Series, window: int) -> pd.Series:
    """Worst peak-to-trough drawdown % over each trailing ``window``."""
    roll_max = closes.rolling(window, min_periods=1).max()
    return (closes / roll_max - 1.0) * 100.0


def historical_crash_stats(df: pd.DataFrame) -> Dict:
    """
    Realized crash stats from the symbol's own history (2018+):

    * ``covid_dd_pct`` / ``covid_vol_mult`` - the worst 23-session
      drawdown and the realized-vol multiple *inside* the Mar-Jun 2020
      window (not a global worst-ever number dressed up as COVID).
    * ``dd_2022_pct`` - worst 250-day drawdown since 2022-01-01.

    Returns ``None`` values when the history is too short, the window
    predates the data, or no dates are available (graceful, canonical
    parameters remain the fallback).
    """
    if "close" not in df.columns or len(df) < 40:
        return {"covid_dd_pct": None, "covid_vol_mult": None, "dd_2022_pct": None}
    closes = df["close"].astype(float)
    out: Dict = {"covid_dd_pct": None, "covid_vol_mult": None, "dd_2022_pct": None}

    dd23 = _rolling_max_drawdown(closes, 23)  # worst 23d dd at each bar

    # data may carry a 'date' column or a datetime index - the loader
    # drops the column for some groups
    if "date" in df.columns and df["date"].notna().any():
        dates = pd.to_datetime(df["date"])
    elif isinstance(df.index, pd.DatetimeIndex):
        dates = pd.Series(pd.to_datetime(df.index), index=df.index)
    else:
        dates = None
    if dates is not None:
        # ``dates`` carries the full frame index, so a mask built on it
        # aligns with both ``dd23`` (full index) and ``closes``.
        mask = (dates >= pd.Timestamp("2020-03-01")) & (
            dates <= pd.Timestamp("2020-06-30")
        )
        # windowed: worst 23d drawdown and vol multiple during the crisis
        if mask.any() and int(mask.sum()) >= 15:
            covid_dd = float(dd23.loc[mask].min())
            if not np.isnan(covid_dd):
                out["covid_dd_pct"] = round(covid_dd, 1)
            rets = closes.pct_change().dropna()
            rm = mask.reindex(rets.index, fill_value=False)  # rets has 1 less row
            crisis_vol = float(rets[rm].std())
            normal_vol = float(rets.loc[~rm].tail(120).std()) or np.nan
            if normal_vol and normal_vol > 0:
                out["covid_vol_mult"] = round(crisis_vol / normal_vol, 2)

        # worst 250-day drawdown from 2022 onward
        recent = closes[dates >= pd.Timestamp("2022-01-01")]
        if len(recent) >= 40:
            dd250 = _rolling_max_drawdown(recent, min(250, len(recent))).min()
            if not np.isnan(dd250):
                out["dd_2022_pct"] = round(float(dd250), 1)
    return out


# ---------------------------------------------------------------------------
# shocks
# ---------------------------------------------------------------------------


def scenario_var(
    qty: float, atr: float, z: float = 1.645, hold_bars: int = 10, vol_mult: float = 1.0
) -> float:
    """Shocked parametric VaR: ``z * (atr * vol_mult) * qty * sqrt(hold)``."""
    return float(z * atr * vol_mult * qty * np.sqrt(hold_bars))


SCENARIO_CAP_PCT = 20.0  # default equity-loss cap for scenario screening


def stress_position(
    position: Dict,
    scenario: Dict,
    equity: float,
    daily_limit_pct: float = DAILY_LIMIT_PCT,
    scenario_cap_pct: float = SCENARIO_CAP_PCT,
) -> Dict:
    """
    Stress one position under ``scenario``.

    ``position``: {qty, entry, atr, direction: 'long'|'short'}.

    Two like-for-like measures:
    *    ``var95_stress`` - shocked parametric VaR over the scenario horizon
      (``horizon_days`` bars, i.e. sessions; correct for daily data),
      comparable to the gap-through loss.
      Note: ``horizon_days`` is used directly as bars - for H4/H1 runs
      pass an explicit ``horizon_days`` in sessions (defaults to D1-style
      day counts) or rely on the D1 path.
    * ``var95_1d_pct_equity`` - the **1-day** shocked VaR, which is the
      right measure to check against the *daily* loss limit.

    ``loss_usd`` is the worst-case gap-through loss (position held the
    whole drawdown, stops assumed not to fill): qty * entry * drawdown.
    ``daily_limit_breach`` = 1-day shocked VaR > daily limit;
    ``scenario_cap_breach`` = gap-through loss > ``scenario_cap_pct`` of
    equity; ``days_of_daily_limit`` = how many normal daily-limit losses
    the gap-through loss equals (readable scale, no tautology).
    """
    qty = float(position.get("qty", 0) or 0)
    entry = float(position.get("entry", 0) or 0)
    atr = float(position.get("atr", 0) or 0)
    direction = position.get("direction", "long")
    vol_mult = float(scenario.get("vol_mult", 1.0))
    horizon = int(scenario.get("horizon_days", 10))

    drawdown = float(scenario["drawdown_pct"]) / 100.0
    # Direction-aware: the canonical scenarios are market *crashes*, so a
    # long loses the drawdown while a short book *earns* it (negative loss
    # = gain). The shocked VaR below still captures the short's real risk -
    # a vol blow-up (short squeeze) gaps AGAINST the position even though
    # the headline move is favorable.
    sign = -1.0 if direction == "short" else 1.0
    loss = sign * qty * entry * drawdown
    loss_pct = (loss / equity * 100.0) if equity else 0.0
    var_horizon = scenario_var(qty, atr, vol_mult=vol_mult, hold_bars=max(horizon, 1))
    var_1d = scenario_var(qty, atr, vol_mult=vol_mult, hold_bars=1)
    var_1d_pct = (var_1d / equity * 100.0) if equity else 0.0

    # ``days_of_daily_limit`` is a loss scale; a favorable scenario (short
    # book in a crash) shows 0 days, never a nonsense negative count.
    days = round(loss_pct / daily_limit_pct, 3) if daily_limit_pct else 0.0

    return {
        "loss_usd": round(loss, 2),
        "loss_pct_equity": round(loss_pct, 2),
        "var95_stress": round(var_horizon, 2),
        "var95_1d_pct_equity": round(var_1d_pct, 2),
        "daily_limit_breach": var_1d_pct > daily_limit_pct,
        "scenario_cap_breach": loss_pct > scenario_cap_pct,
        "days_of_daily_limit": max(days, 0.0),
        "direction": direction,
        "scenario": scenario.get("name", ""),
    }


def stress_portfolio(
    positions: List[Dict],
    equity: float,
    scenario: Dict,
    daily_limit_pct: float = DAILY_LIMIT_PCT,
    scenario_cap_pct: float = SCENARIO_CAP_PCT,
) -> Dict:
    """Aggregate stress across positions (assumes perfectly correlated
    adverse shock - conservative; ignores diversification benefit).
    Direction-aware: long positions lose the drawdown, short positions gain
    it (signed), so a long+short book nets - by construction there is no
    diversification *credit* for uncorrelated assets, but opposite
    directions DO offset (that is the strategy, not a modelling
    assumption)."""
    rows = [
        stress_position(p, scenario, equity, daily_limit_pct, scenario_cap_pct)
        for p in positions
    ]
    total_loss = sum(r["loss_usd"] for r in rows)
    total_var = sum(r["var95_stress"] for r in rows)
    return {
        "scenario": scenario.get("name", ""),
        "positions": rows,
        "total_loss_usd": round(total_loss, 2),
        "total_loss_pct_equity": round(total_loss / equity * 100, 2) if equity else 0.0,
        "portfolio_var95_stress": round(total_var, 2),
        "daily_limit_breach": any(r["daily_limit_breach"] for r in rows),
        "scenario_cap_breach": any(r["scenario_cap_breach"] for r in rows),
    }


# ---------------------------------------------------------------------------
# report integration
# ---------------------------------------------------------------------------


def stress_table_from_report(
    report: Dict,
    symbol: str,
    equity: Optional[float] = None,
    df: Optional[pd.DataFrame] = None,
    direction: str = "long",
    risk_key: str = "risk",
) -> Dict:
    """
    Per-scenario stress table for the report's risk plan (the fractional
    size of the current setup, if actionable). ``historical`` carries the
    data-grounded COVID/2022 numbers extracted from ``df`` when the
    symbol's history has them (pass the prepared OHLCV frame; skipped
    when None).

    ``direction`` propagates to the position so a short book reads the
    crash scenarios as favorable (negative loss) with the vol-multiplied
    VaR still flagging squeeze risk. ``risk_key`` selects which risk plan
    to stress (``"risk"`` = long dip setup, ``"short_risk"`` = short
    rally setup).
    """
    risk = report.get(risk_key) or {}
    setup = risk.get("setup")
    if not setup:
        return {"available": False, "reason": risk.get("reason", "no setup")}

    eq = equity or float((risk.get("inputs") or {}).get("equity", 100_000))
    sizes = risk.get("sizes") or []
    frac = next((s for s in sizes if s.get("method") == "fractional"), None)
    qty = float((frac or {}).get("qty", 0) or 0)
    entry = float(setup.get("entry", 0) or 0)
    atr = float((report.get("volatility") or {}).get("atr_14", 0) or 0)

    if qty <= 0 or entry <= 0:
        return {"available": False, "reason": "no fractional size"}

    position = {
        "qty": qty,
        "entry": entry,
        "atr": atr,
        "direction": direction,
        "symbol": symbol,
    }
    rows = []
    for name, sc in SCENARIOS.items():
        s = stress_position(position, {**sc, "name": name}, eq)
        rows.append(
            {
                "scenario": name,
                "drawdown_pct": sc["drawdown_pct"],
                "vol_mult": sc["vol_mult"],
                "loss_usd": s["loss_usd"],
                "loss_pct_equity": s["loss_pct_equity"],
                "var95_stress": s["var95_stress"],
                "var95_1d_pct_equity": s["var95_1d_pct_equity"],
                "daily_limit_breach": s["daily_limit_breach"],
                "scenario_cap_breach": s["scenario_cap_breach"],
                "days_of_daily_limit": s["days_of_daily_limit"],
            }
        )

    hist = historical_crash_stats(df) if df is not None and len(df) else {}
    return {
        "available": True,
        "equity": eq,
        "qty": round(qty, 2),
        "direction": direction,
        "historical": hist,
        "scenarios": rows,
        "any_breach": any(
            r["daily_limit_breach"] or r["scenario_cap_breach"] for r in rows
        ),
    }


if __name__ == "__main__":
    print("NexusQuant Stress Testing module ready.")
