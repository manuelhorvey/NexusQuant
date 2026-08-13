"""
NexusQuant - Backtest engine.

Bar-by-bar trade simulation on top of a signal series (e.g.
``src.backtest.signals.dip_signal_series`` for longs or
``rally_signal_series`` for shorts):

* **Side** - ``run_backtest(..., side="long")`` or ``side="short"``. The
  short side is the exact mirror: limit entry at the signal's entry-zone
  high (fade the rally), stop ABOVE the invalidation, target BELOW at the
  support level; PnL / mark-to-market / slippage are all inverted.
* **Entry** - limit order at the signal level, valid for
  ``entry_valid_bars`` (filled when a bar's low reaches a long limit or a
  bar's high reaches a short limit; a gap through the level fills at the
  limit, which is conservative). Optional market entry at the NEXT bar's
  open (never the signal bar itself - no lookahead).
* **Exit** - stop loss at the signal invalidation, take profit at the signal
  target (falls back to an R:R multiple), or a time stop after
  ``max_hold`` bars. Intrabar, the stop is checked before the target
  (conservative).
* **Sizing** - ``sizing_mode``: ``fractional`` (risk a fixed % of equity),
  ``voltarget`` (target a per-trade volatility contribution, capped at
  ``risk_pct``), or ``kelly`` (fractional Kelly, capped at ``risk_pct``).
  Risk per unit is always ``abs(entry - stop)`` so sizing is side-agnostic.
* **Costs** - proportional slippage applied on both fill and exit, applied
  in the correct direction per side (long buys at ask / sells at bid;
  short sells at bid / buys back at ask).

Returns a ``BacktestResult`` with the trade list, equity curve and stats.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

DEFAULT_BARS_PER_YEAR = 252  # D1; pass ~1512 / ~6048 for H4 / H1


@dataclass
class BacktestParams:
    initial_capital: float = 100_000.0
    risk_pct: float = 0.01  # fraction of equity risked per trade
    entry_valid_bars: int = 3  # limit order validity
    max_hold: int = 20  # time-stop in bars
    rr_fallback: float = 2.0  # target multiple when no level exists
    slippage: float = 0.0  # proportional cost per side (0.0002 = 2 bps)
    cooldown_bars: int = 0  # bars to wait after an exit
    bars_per_year: int = DEFAULT_BARS_PER_YEAR
    entry_type: str = "limit"  # "limit" or "market"
    min_trade_risk_pct: float = 0.001  # skip trades with risk < this of price
    sizing_mode: str = "fractional"  # fractional | voltarget | kelly
    vol_target: float = 0.02  # per-trade vol contribution (voltarget)
    kelly_p: float = 0.55  # win probability (kelly sizing)
    payoff: float = 1.5  # avg win per unit risked (kelly sizing)
    kelly_fraction: float = 0.5  # fractional Kelly multiplier
    n_trials: int = 20  # research-search-space size for the
    # deflated Sharpe ratio (the number
    # of independently tested strategies)


@dataclass
class Trade:
    symbol: str
    entry_time: pd.Timestamp
    exit_time: pd.Timestamp
    entry_price: float
    exit_price: float
    qty: float
    pnl: float
    pnl_pct: float
    r_multiple: float
    reason: str  # stop | target | time | market_stop | market_target
    bars_held: int
    entry_zone_lo: Optional[float] = None
    score: Optional[int] = None


@dataclass
class BacktestResult:
    symbol: str
    params: BacktestParams
    trades: List[Trade] = field(default_factory=list)
    equity: pd.Series = field(default_factory=pd.Series)
    stats: Dict = field(default_factory=dict)

    def trades_frame(self) -> pd.DataFrame:
        cols = [
            "symbol",
            "entry_time",
            "exit_time",
            "entry_price",
            "exit_price",
            "qty",
            "pnl",
            "pnl_pct",
            "r_multiple",
            "reason",
            "bars_held",
            "score",
        ]
        rows = [{c: getattr(t, c) for c in cols} for t in self.trades]
        return pd.DataFrame(rows, columns=cols)


def _size_position(
    equity: float,
    entry: float,
    stop: float,
    params: BacktestParams,
    atr_pct: Optional[float] = None,
    side: str = "long",
) -> float:
    """Dispatch to src.risk.sizing for the configured sizing mode.
    ``side`` is forwarded so the risk-per-unit direction matches
    (``abs(entry - stop)`` internally, but keep the API honest)."""
    from src.risk.sizing import size_position

    return size_position(
        equity,
        entry,
        stop,
        mode=params.sizing_mode,
        risk_pct=params.risk_pct,
        atr_pct=atr_pct,
        vol_target=params.vol_target,
        hold_bars=1,
        p=params.kelly_p,
        payoff=params.payoff,
        kelly_fraction_ratio=params.kelly_fraction,
        cap_risk_pct=params.risk_pct,
        direction=side,
    )


def run_backtest(
    signal: pd.DataFrame,
    ohlcv: pd.DataFrame,
    params: Optional[BacktestParams] = None,
    symbol: str = "",
    side: str = "long",
) -> BacktestResult:
    """
    Simulate trades on a signal series.

    Parameters
    ----------
    signal : df with confirmed/entry_lo/entry_hi/invalidation/{resistance,target}
        - for ``side="long"`` as produced by
          ``src.backtest.signals.dip_signal_series`` (entry_lo below market,
          stop below, resistance target above);
        - for ``side="short"`` as produced by ``rally_signal_series``
          (entry_hi above market, stop ABOVE, ``target`` support BELOW).
    ohlcv : OHLCV frame with the same index as signal.
    side  : "long" or "short" - mirrors every mechanical detail.
    """
    params = params or BacktestParams()
    short = side == "short"

    if not (len(signal) == len(ohlcv)):
        raise ValueError("signal and ohlcv must have the same length")

    open_ = ohlcv["open"].values
    high = ohlcv["high"].values
    low = ohlcv["low"].values
    close = ohlcv["close"].values
    times = ohlcv.index
    atr_values = ohlcv["atr_14"].values if "atr_14" in ohlcv else None

    def _atr_pct(bar: int) -> Optional[float]:
        if atr_values is None or open_[bar] <= 0:
            return None
        v = atr_values[bar] / open_[bar]
        # NaN (ATR warm-up) is truthy in `if atr_pct:` and would poison the
        # voltarget sizing with NaN qty; fall back to fractional instead.
        return v if math.isfinite(v) else None

    confirmed = (
        signal["confirmed"].values
        if "confirmed" in signal
        else np.zeros(len(signal), dtype=bool)
    )
    entry_lo = signal["entry_lo"].values
    entry_hi = signal["entry_hi"].values
    invalidation = signal["invalidation"].values
    # Long target = resistance column; short target = the support column
    # (rally_signal_series calls it ``target``).
    target_col = "resistance" if not short else "target"
    signal_target = signal[target_col].values
    score = signal["score"].values if "score" in signal else np.zeros(len(signal))

    equity = params.initial_capital
    equity_curve = np.empty(len(ohlcv))
    trades: List[Trade] = []

    position = None  # dict: entry, qty, stop, target, entry_bar, entry_lo
    pending = None  # dict: limit, stop, target, expiry_bar, score
    cooldown_until = -1
    n = len(ohlcv)

    def _exit_position(bar: int, price: float, reason: str) -> None:
        nonlocal equity
        # Long exits by selling (fill at bid); short exits by buying back
        # (fill at ask). Slippage always works against us.
        fill = (
            price * (1.0 - params.slippage)
            if not short
            else price * (1.0 + params.slippage)
        )
        pnl = (
            (fill - position["entry"]) * position["qty"]
            if not short
            else (position["entry"] - fill) * position["qty"]
        )
        equity += pnl
        risk_amount = abs(position["entry"] - position["stop"]) * position["qty"]
        r_mult = pnl / risk_amount if risk_amount > 0 else 0.0
        trades.append(
            Trade(
                symbol=symbol,
                entry_time=times[position["entry_bar"]],
                exit_time=times[bar],
                entry_price=position["entry"],
                exit_price=fill,
                qty=position["qty"],
                pnl=pnl,
                pnl_pct=pnl / (position["entry"] * position["qty"])
                if position["qty"]
                else 0.0,
                r_multiple=r_mult,
                reason=reason,
                bars_held=bar - position["entry_bar"],
                entry_zone_lo=position.get("entry_lo"),
                score=position.get("score"),
            )
        )

    for bar in range(n):
        # 1) Exit an open position (stop checked before target: conservative)
        if position is not None:
            reason = None
            # Long: stop below (low touches it); short: stop above (high
            # touches it). Target: the opposite boundary.
            hit_stop = (
                low[bar] <= position["stop"]
                if not short
                else high[bar] >= position["stop"]
            )
            hit_target = position["target"] is not None and (
                high[bar] >= position["target"]
                if not short
                else low[bar] <= position["target"]
            )
            if hit_stop:
                reason, price = "stop", position["stop"]
            elif hit_target:
                reason, price = "target", position["target"]
            elif bar - position["entry_bar"] >= params.max_hold:
                reason, price = "time", close[bar]
            if reason:
                _exit_position(bar, price, reason)
                position = None
                cooldown_until = bar + params.cooldown_bars

        # 2) Fill a pending order (market: at this bar's open - i.e. the bar
        #    AFTER the signal bar, never the signal bar itself - or limit)
        if pending is not None and position is None:
            if pending.get("market"):
                entry = open_[bar]
                qty = _size_position(
                    equity, entry, pending["stop"], params, _atr_pct(bar), side
                )
                if qty > 0:
                    # Long buys at ask; short sells at bid.
                    entry_fill = (
                        entry * (1.0 + params.slippage)
                        if not short
                        else entry * (1.0 - params.slippage)
                    )
                    position = {
                        "entry": entry_fill,
                        "qty": qty,
                        "stop": pending["stop"],
                        "target": pending["target"],
                        "entry_bar": bar,
                        "entry_lo": None,
                        "score": pending["score"],
                    }
                pending = None
            elif (
                (low[bar] <= pending["limit"])
                if not short
                else (high[bar] >= pending["limit"])
            ):
                # Gap-through fills at the limit (conservative).
                entry = pending["limit"]
                qty = _size_position(
                    equity, entry, pending["stop"], params, _atr_pct(bar), side
                )
                if qty > 0:
                    entry_fill = (
                        entry * (1.0 + params.slippage)
                        if not short
                        else entry * (1.0 - params.slippage)
                    )
                    position = {
                        "entry": entry_fill,
                        "qty": qty,
                        "stop": pending["stop"],
                        "target": pending["target"],
                        "entry_bar": bar,
                        "entry_lo": pending["limit"],
                        "score": pending["score"],
                    }
                pending = None
            elif bar >= pending["expiry_bar"]:
                pending = None

        # 3) Open a new trade on a fresh signal
        if position is None and pending is None and bar >= cooldown_until:
            entry_level = entry_hi[bar] if short else entry_lo[bar]
            if confirmed[bar] and not math.isnan(entry_level):
                stop = invalidation[bar]
                if params.entry_type == "market":
                    # Delay to the next bar's open (no same-bar lookahead).
                    # Long: stop below close; short: stop above close.
                    risk_ok = (
                        (close[bar] - stop) > params.min_trade_risk_pct * close[bar]
                        if not short
                        else (stop - close[bar])
                        > params.min_trade_risk_pct * close[bar]
                    )
                    if risk_ok:
                        tgt = _target(
                            signal_target[bar],
                            close[bar],
                            stop,
                            params.rr_fallback,
                            short,
                        )
                        pending = {
                            "market": True,
                            "stop": stop,
                            "target": tgt,
                            "expiry_bar": bar,
                            "score": int(score[bar])
                            if not math.isnan(score[bar])
                            else None,
                        }
                else:
                    # Limit order only when the zone is on the right side of
                    # the market (long: below; short: above).
                    zone_ok = (
                        (stop < entry_level < close[bar])
                        if not short
                        else (stop > entry_level > close[bar])
                    )
                    if zone_ok:
                        pending = {
                            "limit": entry_level,
                            "stop": stop,
                            "target": _target(
                                signal_target[bar],
                                entry_level,
                                stop,
                                params.rr_fallback,
                                short,
                            ),
                            "expiry_bar": bar + params.entry_valid_bars,
                            "score": int(score[bar])
                            if not math.isnan(score[bar])
                            else None,
                        }

        # 4) Mark-to-market equity
        mtm = equity
        if position is not None:
            mtm += (
                (close[bar] - position["entry"])
                if not short
                else (position["entry"] - close[bar])
            ) * position["qty"]
        equity_curve[bar] = mtm

    result = BacktestResult(symbol=symbol, params=params, trades=trades)
    result.equity = pd.Series(equity_curve, index=ohlcv.index, name="equity")
    result.stats = compute_stats(result)
    return result


def run_backtest_both(
    signal_long: pd.DataFrame,
    signal_short: pd.DataFrame,
    ohlcv: pd.DataFrame,
    params: Optional[BacktestParams] = None,
    symbol: str = "",
) -> BacktestResult:
    """
    Run the long and short books on the same frame and merge them into one
    result: both books start with ``initial_capital`` (each independently
    deployed), trades are concatenated chronologically and the combined
    equity curve is ``long + short - initial_capital``.

    Documented simplification: the two books hold separate capital, so this
    is a *research* view of the combined edge, not a shared-margin
    portfolio simulation.
    """
    params = params or BacktestParams()
    long_res = run_backtest(signal_long, ohlcv, params, symbol, side="long")
    short_res = run_backtest(signal_short, ohlcv, params, symbol, side="short")
    merged = BacktestResult(symbol=f"{symbol} (long+short)", params=params)
    merged.equity = long_res.equity + short_res.equity - params.initial_capital
    merged.trades = sorted(
        long_res.trades + short_res.trades, key=lambda t: t.entry_time
    )
    merged.stats = compute_stats(merged)
    return merged


def _target(
    target_level: float,
    entry: float,
    stop: float,
    rr_fallback: float,
    short: bool = False,
) -> Optional[float]:
    # Long: a resistance level above entry, else an R:R multiple above.
    # Short: a support level below entry, else an R:R multiple below.
    if not math.isnan(target_level):
        if (not short and target_level > entry) or (short and target_level < entry):
            return target_level
    risk = abs(entry - stop)
    if risk <= 0:
        return None
    return entry + rr_fallback * risk if not short else entry - rr_fallback * risk


def _norm_cdf(x: float) -> float:
    """Standard normal CDF (math-only, no scipy dependency)."""
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))


def _norm_ppf(p: float) -> float:
    """Inverse standard normal CDF - Acklam's rational approximation.

    Accurate to ~1e-9 for the range used by the deflated-Sharpe
    calculation (P > 1e-12). Returns +/-inf at the endpoints.
    """
    if p <= 0.0:
        return float("-inf")
    if p >= 1.0:
        return float("inf")
    a = [
        -3.969683028665376e01,
        2.209460984245205e02,
        -2.759285104469687e02,
        1.383577518672690e02,
        -3.066479806614716e01,
        2.506628277459239e00,
    ]
    b = [
        -5.447609879822406e01,
        1.615858368580409e02,
        -1.556989798598866e02,
        6.680131188771972e01,
        -1.328068155288572e01,
    ]
    c = [
        -7.784894002430293e-03,
        -3.223964580411365e-01,
        -2.400758277161838e00,
        -2.549732539343734e00,
        4.374664141464968e00,
        2.938163982698783e00,
    ]
    d = [
        7.784695709041462e-03,
        3.224671290700398e-01,
        2.445134137142996e00,
        3.754408661907416e00,
    ]
    plow, phigh = 0.02425, 1.0 - 0.02425
    if p < plow:
        q = math.sqrt(-2.0 * math.log(p))
        return (((((c[0] * q + c[1]) * q + c[2]) * q + c[3]) * q + c[4]) * q + c[5]) / (
            (((d[0] * q + d[1]) * q + d[2]) * q + d[3]) * q + 1.0
        )
    if p <= phigh:
        q = p - 0.5
        r = q * q
        return (
            (((((a[0] * r + a[1]) * r + a[2]) * r + a[3]) * r + a[4]) * r + a[5])
            * q
            / (((((b[0] * r + b[1]) * r + b[2]) * r + b[3]) * r + b[4]) * r + 1.0)
        )
    q = math.sqrt(-2.0 * math.log(1.0 - p))
    return -(((((c[0] * q + c[1]) * q + c[2]) * q + c[3]) * q + c[4]) * q + c[5]) / (
        (((d[0] * q + d[1]) * q + d[2]) * q + d[3]) * q + 1.0
    )


def _bootstrap_ci(
    values: np.ndarray, stat, n_boot: int = 2000, alpha: float = 0.05, seed: int = 42
):
    """Percentile bootstrap CI for ``stat`` (array -> float) on ``values``.

    Returns ``(lo, hi)`` or ``(None, None)`` when there is nothing to
    resample. Fixed seed => reproducible across runs (same data + same
    code => identical output, spec reproducibility requirement).
    """
    values = np.asarray(values, dtype=float)
    if len(values) == 0:
        return (None, None)
    rng = np.random.default_rng(seed)
    stats_ = []
    for _ in range(n_boot):
        sample = rng.choice(values, size=len(values), replace=True)
        try:
            stats_.append(float(stat(sample)))
        except Exception:
            continue
    if not stats_:
        return (None, None)
    return (
        float(np.percentile(stats_, 100.0 * alpha / 2.0)),
        float(np.percentile(stats_, 100.0 * (1.0 - alpha / 2.0))),
    )


def compute_stats(result: BacktestResult) -> Dict:
    """Performance statistics from a finished backtest."""
    trades = result.trades
    equity = result.equity
    n_bars = max(len(equity), 1)
    bars_per_year = result.params.bars_per_year

    pnls = np.array([t.pnl for t in trades], dtype=float)
    r_mults = np.array([t.r_multiple for t in trades], dtype=float)
    wins = pnls[pnls > 0]
    losses = pnls[pnls < 0]
    gross_win = float(wins.sum())
    gross_loss = float(-losses.sum())
    n = len(trades)

    total_return = (
        (float(equity.iloc[-1]) / result.params.initial_capital - 1)
        if len(equity) and result.params.initial_capital
        else 0.0
    )
    years = n_bars / bars_per_year
    cagr = (
        ((float(equity.iloc[-1]) / result.params.initial_capital) ** (1 / years) - 1)
        if years > 0 and float(equity.iloc[-1]) > 0
        else 0.0
    )

    curve = equity.values
    peak = np.maximum.accumulate(curve)
    drawdown = np.where(peak > 0, (curve - peak) / peak, 0.0)
    max_dd = float(drawdown.min())

    in_position = np.zeros(n_bars)
    for t in trades:
        s = equity.index.get_loc(t.entry_time)
        e = equity.index.get_loc(t.exit_time)
        in_position[s : e + 1] = 1.0

    # Sharpe is computed on strategy returns only (bars with exposure) so
    # long flat stretches do not dilute it.
    rets = pd.Series(curve).pct_change().dropna()
    active = in_position[1:] > 0
    rets = rets[active]
    sharpe = (
        (float(rets.mean()) / float(rets.std()) * math.sqrt(bars_per_year))
        if len(rets) > 1 and float(rets.std()) > 0
        else 0.0
    )

    # Downside-only Sharpe (Sortino) and Calmar on the same active returns.
    downside = rets[rets < 0]
    sortino = (
        (float(rets.mean()) / float(downside.std()) * math.sqrt(bars_per_year))
        if len(downside) > 1 and float(downside.std()) > 0
        else 0.0
    )
    calmar = (cagr / abs(max_dd)) if max_dd < 0 else 0.0

    # --- Statistical significance (spec audit #23) ---------------------
    # 1) Percentile-bootstrap CIs on the trade R-multiples and win rate -
    #    gives the 'is this edge real or noise?' band around the point
    #    estimates. Only meaningful with enough trades (>= 10).
    exp_ci = wr_ci = (None, None)
    if n >= 10:
        exp_ci = _bootstrap_ci(r_mults, np.mean)
        wr_ci = _bootstrap_ci((pnls > 0).astype(float), np.mean)

    # 2) Probabilistic Sharpe ratio vs zero and vs the expected max Sharpe
    #    under ``n_trials`` independently tested strategies (deflated
    #    Sharpe, Bailey & Lopez de Prado 2014). Both are probabilities in
    #    [0, 1] that the true per-period Sharpe exceeds the benchmark.
    psr_pos = dsr = None
    n_obs = int(len(rets))
    if n_obs > 3 and float(rets.std()) > 0:
        sr_per = float(rets.mean() / rets.std())
        skew3 = float(pd.Series(rets).skew())
        kurt4 = float(pd.Series(rets).kurt())
        var_sr = (1.0 - skew3 * sr_per + (kurt4 - 1.0) / 4.0 * sr_per**2) / (
            n_obs - 1.0
        )
        if var_sr > 0:
            psr_pos = _norm_cdf(sr_per / math.sqrt(var_sr))
            n_trials = max(1, int(result.params.n_trials))
            gamma = 0.5772156649015329  # Euler-Mascheroni
            e_max_sr = math.sqrt(var_sr) * (
                (1.0 - gamma) * _norm_ppf(1.0 - 1.0 / n_trials)
                + gamma * _norm_ppf(1.0 - 1.0 / (n_trials * math.e))
            )
            dsr = _norm_cdf((sr_per - e_max_sr) / math.sqrt(var_sr))

    tail_loss_pct = (
        float(np.percentile(pnls, 5) / result.params.initial_capital * 100)
        if n
        else 0.0
    )

    return {
        "n_trades": n,
        "wins": int((pnls > 0).sum()),
        "losses": int((pnls < 0).sum()),
        "win_rate": float((pnls > 0).mean()) if n else 0.0,
        "avg_win": float(wins.mean()) if len(wins) else 0.0,
        "avg_loss": float(losses.mean()) if len(losses) else 0.0,
        "gross_profit": gross_win,
        "gross_loss": gross_loss,
        "profit_factor": (gross_win / gross_loss)
        if gross_loss > 0
        else (float("inf") if gross_win > 0 else 0.0),
        "expectancy_pct": float(pnls.mean() / result.params.initial_capital * 100)
        if n
        else 0.0,
        "expectancy_r": float(r_mults.mean()) if n else 0.0,
        "total_return_pct": total_return * 100,
        "cagr_pct": cagr * 100,
        "max_drawdown_pct": max_dd * 100,
        "sharpe": sharpe,
        "sortino": sortino,
        "calmar": calmar,
        "tail_loss_pct": tail_loss_pct,
        "expectancy_ci95": exp_ci,
        "win_rate_ci95": wr_ci,
        "psr_positive": psr_pos,  # P(true Sharpe > 0)
        "deflated_sharpe": dsr,  # P(beat N-trial expected max Sharpe)
        "n_trials": result.params.n_trials,
        "avg_hold_bars": float(np.mean([t.bars_held for t in trades])) if n else 0.0,
        "exposure_pct": float(in_position.mean() * 100),
        "best_trade_pct": float(pnls.max() / result.params.initial_capital * 100)
        if n
        else 0.0,
        "worst_trade_pct": float(pnls.min() / result.params.initial_capital * 100)
        if n
        else 0.0,
    }


def regime_breakdown(result: BacktestResult, regime: Optional[pd.Series]) -> Dict:
    """Per-regime performance: n trades, win rate, avg R and total R.

    ``regime`` is the df's ``regime`` column (Bull/Bear/Range/High-Vol),
    aligned with the equity index and looked up at each trade's
    ``entry_time`` - so the breakdown answers *where the edge actually
    makes and loses money* (spec audit #25), using only the regime known
    at entry (causal: the rule-based regime column is trailing-only).
    Returns ``{}`` when no regime column is provided.
    """
    if regime is None or len(regime) == 0:
        return {}
    by_regime: Dict[str, List[float]] = {}
    for t in result.trades:
        try:
            lab = regime.loc[t.entry_time]
        except (KeyError, TypeError):
            continue
        if lab is None or (isinstance(lab, float) and pd.isna(lab)):
            continue
        by_regime.setdefault(str(lab), []).append(float(t.r_multiple))
    rows: Dict[str, Dict] = {}
    for lab, rs in sorted(by_regime.items()):
        arr = np.asarray(rs, dtype=float)
        rows[lab] = {
            "n": int(len(arr)),
            "win_rate": round(float((arr > 0).mean()), 4),
            "avg_r": round(float(arr.mean()), 4),
            "total_r": round(float(arr.sum()), 2),
        }
    return rows


def _fmt_ci(ci) -> str:
    if ci is None or ci[0] is None or ci[1] is None:
        return "n/a (fewer than 10 trades)"
    return f"[{ci[0]:+.3f}, {ci[1]:+.3f}]"


def _fmt_prob(p, suffix="%"):
    return "n/a (insufficient data)" if p is None else f"{p * 100:.1f}{suffix}"


def print_stats(result: BacktestResult) -> str:
    """Human-readable stats block (incl. statistical significance)."""
    s = result.stats
    lines = [
        f"  Trades          : {s['n_trades']}  ({s['wins']}W / {s['losses']}L)",
        f"  Win rate        : {s['win_rate'] * 100:.1f}%  "
        f"(95% CI {_fmt_ci(s['win_rate_ci95'])})",
        f"  Profit factor   : {s['profit_factor']:.2f}",
        f"  Expectancy      : {s['expectancy_pct']:+.3f}%  "
        f"({s['expectancy_r']:+.2f}R, 95% CI {_fmt_ci(s['expectancy_ci95'])})",
        f"  Total return    : {s['total_return_pct']:+.2f}%",
        f"  CAGR            : {s['cagr_pct']:+.2f}%",
        f"  Max drawdown    : {s['max_drawdown_pct']:.2f}%",
        f"  Sharpe          : {s['sharpe']:.2f}   "
        f"Sortino {s['sortino']:.2f}   Calmar {s['calmar']:.2f}",
        f"  Tail loss (5%)  : {s['tail_loss_pct']:+.2f}% of equity",
        f"  PSR (>0)        : {_fmt_prob(s['psr_positive'])}  "
        f"Deflated Sharpe  : {_fmt_prob(s['deflated_sharpe'])} "
        f"(vs {s['n_trials']} trials)",
        f"  Avg hold        : {s['avg_hold_bars']:.1f} bars",
        f"  Exposure        : {s['exposure_pct']:.1f}%",
        f"  Best / worst    : {s['best_trade_pct']:+.2f}% / {s['worst_trade_pct']:+.2f}%",
    ]
    rb = result.stats.get("regime_breakdown")
    if rb:
        lines.append("  By regime:")
        for lab, row in rb.items():
            lines.append(
                f"    {lab:<15} n={row['n']:<5} wr={row['win_rate'] * 100:>5.1f}% "
                f"avgR={row['avg_r']:+.3f}  totalR={row['total_r']:+.1f}"
            )
    lines.append(
        "  Note: bootstrap CI / PSR / deflated Sharpe guard against "
        "backtest overfitting - treat point estimates without them as noise."
    )
    return "\n".join(lines)
