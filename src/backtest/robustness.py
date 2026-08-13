"""
NexusQuant - Backtest robustness analysis (spec audit #26 / #27).

Two cheap but decisive checks that separate a robust edge from a
curve-fit one:

* **Threshold ablation** - ``ablate_threshold`` re-runs the backtest at
  increasing signal-score floors (score >= t). If raising the bar does
  not improve win rate / expectancy, the "conviction score" carries no
  information and should not be gating entries.
* **Parameter sensitivity** - ``param_sensitivity`` sweeps one
  ``BacktestParams`` field (slippage, risk_pct, rr_fallback, max_hold,
  entry_valid_bars, ...) over a grid. A strategy whose edge vanishes at
  a 1-2 bp cost or a slightly wider stop is not robust enough to trade.

Both return plain ``pandas.DataFrame`` rows so the CLI prints them as
tables and they stay machine-readable.

Usage (library):
    from src.backtest.robustness import ablate_threshold, param_sensitivity
    rows = ablate_threshold(signal, df, params, side="long")
    rows = param_sensitivity(signal, df, params, param="slippage",
                             values=(0.0, 0.001, 0.002))
"""

from __future__ import annotations

from dataclasses import replace
from typing import Iterable, Optional, Sequence

import pandas as pd

from src.backtest.engine import BacktestParams, run_backtest


def _summary_row(stats: dict, label, key: str) -> dict:
    return {
        key: label,
        "n_trades": stats["n_trades"],
        "win_rate": round(stats["win_rate"], 4),
        "expectancy_r": round(stats["expectancy_r"], 4),
        "sharpe": round(stats["sharpe"], 3),
        "return_pct": round(stats["total_return_pct"], 2),
        "max_dd_pct": round(stats["max_drawdown_pct"], 2),
    }


def ablate_threshold(
    signal: pd.DataFrame,
    ohlcv: pd.DataFrame,
    params: Optional[BacktestParams] = None,
    side: str = "long",
    score_col: str = "score",
    thresholds: Sequence[float] = (0, 4, 5, 6, 7),
) -> pd.DataFrame:
    """Backtest at score floors ``score >= t`` and compare the edge.

    Threshold ``t`` is applied by *disabling* signals whose score is
    below it (the signal frame keeps its length and index, so the trade
    simulation is otherwise identical). A monotonically improving
    expectancy/win-rate as ``t`` rises means the score is informative.
    """
    rows = []
    for t in thresholds:
        s = signal.copy()
        if score_col in s.columns and t > 0:
            below = s[score_col] < t
            s.loc[below, "confirmed"] = False
        res = run_backtest(s, ohlcv, params, side=side)
        rows.append(_summary_row(res.stats, label=t, key="threshold"))
    return pd.DataFrame(rows)


def param_sensitivity(
    signal: pd.DataFrame,
    ohlcv: pd.DataFrame,
    params: Optional[BacktestParams] = None,
    side: str = "long",
    param: str = "slippage",
    values: Optional[Iterable[float]] = None,
) -> pd.DataFrame:
    """Sweep one ``BacktestParams`` field over ``values``.

    ``param`` must be a float field of ``BacktestParams`` (slippage,
    risk_pct, rr_fallback, max_hold, entry_valid_bars, ...). Default
    grid for ``slippage``: 0 / 0.5 / 1 / 2 bps.
    """
    if values is None:
        values = (0.0, 0.00005, 0.0001, 0.0002)
    if not hasattr(BacktestParams(), param):
        raise ValueError(f"{param} is not a BacktestParams field")
    rows = []
    for v in values:
        p = params or BacktestParams()
        p = replace(p, **{param: v})
        res = run_backtest(signal, ohlcv, p, side=side)
        rows.append(_summary_row(res.stats, label=v, key="param_value"))
    return pd.DataFrame(rows)


if __name__ == "__main__":
    print("NexusQuant Backtest Robustness module ready.")
