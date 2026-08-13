"""
NexusQuant - Risk & position sizing CLI.

Per-symbol risk plan (all three sizing methods on the current dip setup) and
a portfolio-level report (correlations, heat, VaR):

    python -m src.risk.run --symbol XAUUSD --group full_fx
    python -m src.risk.run --symbol XAUUSD --group full_fx --equity 250000
    python -m src.risk.run --group full_fx --portfolio          # universe risk
    python -m src.risk.run --symbols XAUUSD,EURUSD,GBPUSD --portfolio --equity 100000
    python -m src.risk.run --symbol XAUUSD --group full_fx --json

When the ensemble model exists, Kelly sizing uses the model's live
probability; otherwise it falls back to the strategy win rate (--win-rate).
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List, Optional

import pandas as pd

sys.path.append(str(Path(__file__).parent.parent.parent))

from src.data.loader import clean_data, load_data
from src.features.indicators import add_all_indicators
from src.features.regime import detect_regime
from src.analysis.scanner import _data_path, discover_symbols
from src.risk.metrics import (
    asset_vol_pct,
    check_correlation_limit,
    portfolio_heat,
    portfolio_var,
    returns_correlation,
    top_correlated_pairs,
    trade_var,
)
from src.risk.sizing import (
    fractional_qty,
    kelly_qty,
    risk_dollars,
    risk_pct_of_equity,
    vol_target_qty,
)

DEFAULT_EQUITY = 100_000.0
DEFAULT_RISK_PCT = 0.01
DEFAULT_VOL_TARGET = 0.02
DEFAULT_KELLY_P = 0.55
DEFAULT_PAYOFF = 1.5

# Institutional minimum reward-to-risk floor (institutional spec #11). The
# live filter enforces it on the *ladder's* best achievable R:R (scaling
# out), not the nearest-resistance R:R that the single-target dip setup
# implies (~0.9-1.2).
DEFAULT_MIN_RR = 2.5


def prepare_frame(
    symbol: str, group: Optional[str], timeframe: str, data_dir: str = "data/raw"
) -> pd.DataFrame:
    path = _data_path(symbol, data_dir, group, timeframe)
    df = load_data(path, symbol=symbol)
    df = clean_data(df)
    df = add_all_indicators(df)
    df = detect_regime(df)
    return df


def risk_plan_from_report(
    report: dict,
    symbol: str,
    equity: float = DEFAULT_EQUITY,
    risk_pct: float = DEFAULT_RISK_PCT,
    vol_target: float = DEFAULT_VOL_TARGET,
    kelly_p: float = DEFAULT_KELLY_P,
    payoff: float = DEFAULT_PAYOFF,
    kelly_fraction: float = 0.5,
    hold_bars: int = 10,
    min_rr: float = DEFAULT_MIN_RR,
) -> dict:
    """
    Risk plan computed from an already-generated full report.

    Pure over ``report`` (dip setup, volatility, ML section) so both the
    risk CLI and the report generator can reuse it without recomputing
    the whole pipeline. Returns ``{symbol, setup, inputs, sizes}`` or
    ``{symbol, setup: None, reason}`` when there is no actionable setup.

    The setup's R:R is the **achievable** one: the target ladder
    (report["targets"]) holds the scaling-out plan (TP1..TP3), so
    ``rr_ok`` reports whether the ladder reaches ``min_rr`` (spec #11's
    2.5:1 floor) - not the nearest-resistance single-target R:R, which is
    often only ~0.9-1.2.
    """
    dip = report["dip"]
    entry = None
    stop = None
    if dip.get("entry_zone"):
        entry = dip["entry_zone"][0]
    if dip.get("invalidation_level"):
        stop = dip["invalidation_level"]
    if not entry or not stop or entry <= stop:
        return {
            "symbol": symbol,
            "setup": None,
            "reason": "no actionable dip setup (entry zone/invalidation)",
        }

    close = report["last_close"]
    atr = report["volatility"]["atr_14"]
    atr_pct = atr / close if close else 0.0
    risk = entry - stop
    target = dip.get("target") or entry + payoff * risk
    rr_nearest = (target - entry) / risk if risk > 0 else 0.0

    # Target ladder (scaling-out plan, spec #11): the first target that
    # clears the min R:R floor is the setup's achievable R:R. The ladder is
    # absent (or empty) on reports without a setup - fall back to the dip
    # target's R:R.
    ladder = report.get("targets") or {}
    best_rr = ladder.get("best_rr") or 0.0
    min_rr_tp = ladder.get("min_rr_tp")
    rr_ok = bool(min_rr_tp is not None and best_rr >= min_rr)
    # Prefer the floor-clearing target for the headline R:R (honest
    # achievable figure), keep the nearest one for reference. Price and
    # R:R always come from the SAME ladder rung so the setup never shows
    # a target whose actual R:R differs from the headline.
    if rr_ok:
        tp = next(
            (t for t in ladder.get("targets", []) if t["target"] == min_rr_tp), None
        )
        if tp is not None:
            target, rr = tp["price"], tp["rr"]
        else:
            rr = best_rr
    else:
        rr = rr_nearest

    # Kelly probability: prefer the ensemble model (from report["ml"]),
    # else the given p.
    prob = None
    ml = report.get("ml")
    if ml and ml.get("prob_pct") is not None:
        prob = ml["prob_pct"] / 100.0
    p = prob if prob is not None else kelly_p

    methods = {
        "fractional": fractional_qty(equity, entry, stop, risk_pct),
        "voltarget": vol_target_qty(
            equity, entry, stop, atr_pct, vol_target, hold_bars, cap_risk_pct=risk_pct
        ),
        "kelly": kelly_qty(
            equity,
            entry,
            stop,
            p,
            payoff,
            fraction=kelly_fraction,
            cap_risk_pct=risk_pct,
        ),
    }
    rows = []
    for name, qty in methods.items():
        risk_d = risk_dollars(qty, entry, stop)
        rows.append(
            {
                "method": name,
                "qty": round(qty, 2),
                "notional": round(qty * entry, 2),
                "risk_usd": round(risk_d, 2),
                "risk_pct_equity": round(
                    risk_pct_of_equity(qty, entry, stop, equity) * 100, 2
                ),
                "var_95_1bar": round(trade_var(qty, atr, hold_bars=1), 2),
                "var_95_hold": round(trade_var(qty, atr, hold_bars=hold_bars), 2),
            }
        )

    return {
        "symbol": symbol,
        "date": report["last_date"],
        "close": close,
        "setup": {
            "entry": round(entry, 5),
            "stop": round(stop, 5),
            "target": round(target, 5),
            "rr": round(rr, 2),
            "rr_nearest": round(rr_nearest, 2),
            "rr_ok": rr_ok,
            "min_rr": min_rr,
            "best_rr": round(best_rr, 2) if best_rr else None,
            "min_rr_tp": min_rr_tp,
            "atr_pct": round(atr_pct * 100, 3),
            "dip_stage": dip["dip_stage"],
            "dip_score": dip["dip_score"],
            "ml_prob": round(p * 100, 1) if prob is not None else None,
        },
        "inputs": {
            "equity": equity,
            "risk_pct": risk_pct,
            "vol_target": vol_target,
            "kelly_p": round(p, 3),
            "kelly_fraction": kelly_fraction,
            "payoff": payoff,
            "hold_bars": hold_bars,
        },
        "sizes": rows,
    }


def risk_plan_from_report_short(
    report: dict,
    symbol: str,
    equity: float = DEFAULT_EQUITY,
    risk_pct: float = DEFAULT_RISK_PCT,
    vol_target: float = DEFAULT_VOL_TARGET,
    kelly_p: float = DEFAULT_KELLY_P,
    payoff: float = DEFAULT_PAYOFF,
    kelly_fraction: float = 0.5,
    hold_bars: int = 10,
    min_rr: float = DEFAULT_MIN_RR,
) -> dict:
    """
    Short-side mirror of ``risk_plan_from_report`` for the Sell-the-Rally
    engine: entry at the rally entry zone, stop above the swing high
    (invalidation), target ladder BELOW entry. ``rr_ok`` reports whether
    the ladder reaches ``min_rr`` (spec #11's 2.5:1 floor).
    """
    rally = report.get("rally") or {}
    entry = None
    stop = None
    if rally.get("entry_zone"):
        entry = rally["entry_zone"][1]  # fade the top of the zone
    if rally.get("invalidation_level"):
        stop = rally["invalidation_level"]
    if not entry or not stop or stop <= entry:
        return {
            "symbol": symbol,
            "setup": None,
            "reason": "no actionable rally setup (entry zone/invalidation)",
        }

    close = report["last_close"]
    atr = report["volatility"]["atr_14"]
    atr_pct = atr / close if close else 0.0
    risk = stop - entry
    target = rally.get("target") or entry - payoff * risk
    rr_nearest = (entry - target) / risk if risk > 0 else 0.0

    ladder = report.get("short_targets") or {}
    best_rr = ladder.get("best_rr") or 0.0
    min_rr_tp = ladder.get("min_rr_tp")
    rr_ok = bool(min_rr_tp is not None and best_rr >= min_rr)
    if rr_ok:
        tp = next(
            (t for t in ladder.get("targets", []) if t["target"] == min_rr_tp), None
        )
        if tp is not None:
            target, rr = tp["price"], tp["rr"]
        else:
            rr = best_rr
    else:
        rr = rr_nearest

    # Short-side Kelly must use the SHORT model probability (P(short win)),
    # not the long one - mixing sides would size a bearish setup with a
    # bullish signal. Falls back to ``kelly_p`` when no short model exists.
    prob = None
    ml_s = report.get("ml_short") or {}
    if ml_s.get("prob_pct") is not None:
        prob = ml_s["prob_pct"] / 100.0
    p = prob if prob is not None else kelly_p

    methods = {
        "fractional": fractional_qty(equity, entry, stop, risk_pct, direction="short"),
        "voltarget": vol_target_qty(
            equity,
            entry,
            stop,
            atr_pct,
            vol_target,
            hold_bars,
            cap_risk_pct=risk_pct,
            direction="short",
        ),
        "kelly": kelly_qty(
            equity,
            entry,
            stop,
            p,
            payoff,
            fraction=kelly_fraction,
            cap_risk_pct=risk_pct,
            direction="short",
        ),
    }
    rows = []
    for name, qty in methods.items():
        risk_d = risk_dollars(qty, entry, stop, direction="short")
        rows.append(
            {
                "method": name,
                "qty": round(qty, 2),
                "notional": round(qty * entry, 2),
                "risk_usd": round(risk_d, 2),
                "risk_pct_equity": round(
                    risk_pct_of_equity(qty, entry, stop, equity, direction="short")
                    * 100,
                    2,
                ),
                "var_95_1bar": round(trade_var(qty, atr, hold_bars=1), 2),
                "var_95_hold": round(trade_var(qty, atr, hold_bars=hold_bars), 2),
            }
        )

    return {
        "symbol": symbol,
        "date": report["last_date"],
        "close": close,
        "direction": "short",
        "setup": {
            "entry": round(entry, 5),
            "stop": round(stop, 5),
            "target": round(target, 5),
            "rr": round(rr, 2),
            "rr_nearest": round(rr_nearest, 2),
            "rr_ok": rr_ok,
            "min_rr": min_rr,
            "best_rr": round(best_rr, 2) if best_rr else None,
            "min_rr_tp": min_rr_tp,
            "atr_pct": round(float(atr_pct) * 100, 3),
            "rally_stage": rally["rally_stage"],
            "rally_score": rally["rally_score"],
            "ml_prob": round(p * 100, 1) if prob is not None else None,
        },
        "inputs": {
            "equity": equity,
            "risk_pct": risk_pct,
            "vol_target": vol_target,
            "kelly_p": round(p, 3),
            "kelly_fraction": kelly_fraction,
            "payoff": payoff,
            "hold_bars": hold_bars,
        },
        "sizes": rows,
    }


def symbol_risk_plan(
    symbol: str,
    group: Optional[str],
    timeframe: str,
    data_dir: str = "data/raw",
    equity: float = DEFAULT_EQUITY,
    risk_pct: float = DEFAULT_RISK_PCT,
    vol_target: float = DEFAULT_VOL_TARGET,
    kelly_p: float = DEFAULT_KELLY_P,
    payoff: float = DEFAULT_PAYOFF,
    kelly_fraction: float = 0.5,
    hold_bars: int = 10,
    side: str = "long",
) -> dict:
    """Compute the current risk plan for a symbol's live dip setup
    (``side="short"`` uses the Sell-the-Rally setup instead)."""
    from src.analysis.report import generate_full_report

    df = prepare_frame(symbol, group, timeframe, data_dir)
    report = generate_full_report(df, symbol=symbol)
    if side == "short":
        return risk_plan_from_report_short(
            report,
            symbol,
            equity=equity,
            risk_pct=risk_pct,
            vol_target=vol_target,
            kelly_p=kelly_p,
            payoff=payoff,
            kelly_fraction=kelly_fraction,
            hold_bars=hold_bars,
        )
    return risk_plan_from_report(
        report,
        symbol,
        equity=equity,
        risk_pct=risk_pct,
        vol_target=vol_target,
        kelly_p=kelly_p,
        payoff=payoff,
        kelly_fraction=kelly_fraction,
        hold_bars=hold_bars,
    )


def stress_plan(
    symbol: str,
    group: Optional[str],
    timeframe: str,
    data_dir: str = "data/raw",
    equity: float = DEFAULT_EQUITY,
) -> dict:
    """Per-scenario stress table (2008 GFC / COVID / 2022) for the
    symbol's current dip setup, with data-grounded realizations."""
    from src.analysis.report import generate_full_report
    from src.risk.stress import stress_table_from_report

    df = prepare_frame(symbol, group, timeframe, data_dir)
    report = generate_full_report(df, symbol=symbol)
    return stress_table_from_report(report, symbol, equity=equity, df=df)


def _print_stress(plan: dict) -> None:
    if not plan.get("available"):
        print(f"   {plan.get('reason', 'no setup to stress')}")
        return
    hist = plan.get("historical") or {}
    print("\n" + "=" * 74)
    print(
        f"STRESS TEST — {plan.get('qty', 0):,.0f} units · equity "
        f"{plan.get('equity', 0):,.0f}"
    )
    print("=" * 74)
    if hist.get("covid_dd_pct") is not None:
        print(
            f"realized: COVID-19 worst 23d drawdown {hist['covid_dd_pct']}% · "
            f"vol mult {hist.get('covid_vol_mult') or '-':<5} · "
            f"2022+ worst drawdown {hist.get('dd_2022_pct')}%"
        )
    for row in plan.get("scenarios", []):
        marks = []
        if row["daily_limit_breach"]:
            marks.append("1d VaR > daily limit")
        elif row["scenario_cap_breach"]:
            marks.append("loss > scenario cap")
        mark = " · ".join(marks) if marks else "ok"
        print(
            f"  {row['scenario']:<16} dd {row['drawdown_pct']:>5.1f}% · "
            f"loss {row['loss_usd']:>14,.0f} "
            f"({row['loss_pct_equity']:>5.2f}% eq ≈ "
            f"{row['days_of_daily_limit']}d limit) · "
            f"VaR95 {row['var95_stress']:>12,.0f} · 1d "
            f"{row['var95_1d_pct_equity']:>5.2f}% · {mark}"
        )
    print(
        "note: loss = gap-through (stops assumed not to fill); VaR95 = "
        "shocked vol over the scenario horizon; 1d = shocked 1-day VaR"
    )
    print("=" * 74 + "\n")


def currency_exposure(positions: List[dict]) -> dict:
    """
    Aggregate portfolio exposure per currency leg (campaign spec #28/#29).

    A long EURUSD position is a long-EUR / short-USD exposure; three JPY
    crosses (EURJPY + GBPJPY + CADJPY) are a concentrated JPY-SHORT
    position even though the symbols look independent. This function maps
    every position to its base/quote legs via ``_symbol_class`` (the same
    classifier the macro overlay uses) and sums signed notionals.

    Returns ``{exposure: {CCY: notional}, gross, net, largest, warnings}``.
    Sign convention: long = +base/-quote; short = -base/+quote. Non-FX
    instruments (equity/index/crypto single assets) contribute a single leg
    on their own symbol - a documented simplification: a metal like
    XAUUSD is economically USD-denominated but is bucketed as its own
    "XAUUSD" leg, so USD concentration from metals is understated (the
    correlation matrix still captures the overlap; a future refinement
    could map metals/commodities into their quote currency).
    """
    from src.macro.overlay import _symbol_class

    exposure: Dict[str, float] = {}
    details = []
    for pos in positions:
        sym = str(pos.get("symbol", "?"))
        notional = float(pos.get("notional") or 0.0)
        direction = pos.get("direction", "long")
        sign = 1.0 if direction == "long" else -1.0
        try:
            kind, detail = _symbol_class(sym)
        except Exception:
            kind, detail = ("equity", sym)
        if kind == "fx" and isinstance(detail, tuple) and len(detail) == 2:
            base, quote = detail
            legs = [(base, sign), (quote, -sign)]
            leg_desc = f"{base} {sign:+.0f} / {quote} {-sign:+.0f}"
        else:
            legs = [(sym, sign)]
            leg_desc = f"{sym} {sign:+.0f}"
        for ccy, s in legs:
            exposure[ccy] = exposure.get(ccy, 0.0) + s * notional
        details.append(
            {
                "symbol": sym,
                "direction": direction,
                "notional": round(notional, 2),
                "legs": leg_desc,
            }
        )

    gross = sum(abs(v) for v in exposure.values())
    net = sum(exposure.values())
    if not exposure:
        largest = None
    else:
        largest = max(exposure.items(), key=lambda kv: abs(kv[1]))

    warnings = []
    if gross > 0:
        # Same-side concentration: a single currency carrying >= 40% of
        # gross exposure is a directional concentration the correlation
        # matrix alone can hide - e.g. three JPY crosses LONG are one big
        # JPY-SHORT (the campaign's canonical example).
        for ccy, v in sorted(exposure.items(), key=lambda kv: -abs(kv[1]))[:2]:
            share = abs(v) / gross
            if share >= 0.4:
                warnings.append(
                    f"{ccy} carries {share:.0%} of gross exposure "
                    f"({v:+,.0f}) - directional concentration "
                    f"(e.g. N JPY crosses can hide one shared leg)"
                )

    return {
        "exposure": {k: round(v, 2) for k, v in exposure.items()},
        "gross": round(gross, 2),
        "net": round(net, 2),
        "largest": largest,
        "warnings": warnings,
        "details": details,
    }


def portfolio_report(
    symbols: List[str],
    group: Optional[str],
    timeframe: str,
    data_dir: str = "data/raw",
    equity: float = DEFAULT_EQUITY,
    risk_pct: float = DEFAULT_RISK_PCT,
    max_heat: float = 0.04,
    max_corr: float = 0.6,
    include_short: bool = False,
) -> dict:
    """Correlations, heat and portfolio VaR across a symbol list.

    Long-only by default; ``include_short`` also folds in actionable
    Sell-the-Rally setups (marked ``direction: short``)."""
    rets = {}
    plans = []
    for sym in symbols:
        try:
            df = prepare_frame(sym, group, timeframe, data_dir)
        except Exception:
            continue
        rets[sym] = df["returns"]
        if include_short:
            # ONE full report per symbol already carries both risk plans -
            # deriving long + short from it avoids regenerating the whole
            # pipeline twice (a 2x slowdown on universe scans).
            from src.analysis.report import generate_full_report

            report = generate_full_report(df, symbol=sym)
            for plan, direction in (
                (
                    risk_plan_from_report(
                        report, sym, equity=equity, risk_pct=risk_pct
                    ),
                    "long",
                ),
                (
                    risk_plan_from_report_short(
                        report, sym, equity=equity, risk_pct=risk_pct
                    ),
                    "short",
                ),
            ):
                if plan.get("setup"):
                    plan["direction"] = direction
                    plans.append(plan)
        else:
            plan = symbol_risk_plan(
                sym, group, timeframe, data_dir, equity=equity, risk_pct=risk_pct
            )
            if plan.get("setup"):
                plan["direction"] = "long"
                plans.append(plan)

    def _empty(reason: str) -> dict:
        """Complete early-return shape so every caller (CLI, dashboard,
        tests) can read the same keys without KeyErrors. ``symbols`` is
        empty (the previous contract) - the reason explains what happened."""
        return {
            "symbols": [],
            "n_setups": len(plans),
            "equity": equity,
            "reason": reason,
            "heat_pct": 0.0,
            "heat_vs_limit": f"0.00% / {max_heat * 100:.0f}%",
            "portfolio_var_95_1bar": 0.0,
            "portfolio_var_pct_equity": 0.0,
            "positions": [],
            "correlation_checks": [],
            "top_correlated_pairs": [],
            "vols": {},
            "currency_exposure": {
                "exposure": {},
                "gross": 0.0,
                "net": 0.0,
                "largest": None,
                "warnings": [],
                "details": [],
            },
        }

    if not rets:
        return _empty("no usable data")

    returns = pd.DataFrame(rets).dropna()
    if len(returns) < 2:
        return _empty(
            f"insufficient overlapping history ({len(returns)} rows) for correlations"
        )
    corr = returns_correlation(returns)
    vols = asset_vol_pct(returns)

    positions = []
    for p in plans:
        direction = p.get("direction", "long")
        positions.append(
            {
                "symbol": p["symbol"],
                "entry": p["setup"]["entry"],
                "stop": p["setup"]["stop"],
                "qty": fractional_qty(
                    equity,
                    p["setup"]["entry"],
                    p["setup"]["stop"],
                    risk_pct,
                    direction=direction,
                ),
                "direction": direction,
            }
        )
    heat = portfolio_heat(positions, equity)
    notionals = [pos["qty"] * pos["entry"] for pos in positions]
    # Reindex the correlation matrix by the actual position symbols: `rets`
    # can contain symbols without setups, and positional slicing would
    # misalign the wrong rows into the VaR.
    pos_syms = [pos["symbol"] for pos in positions]
    vol_list = [float(vols.get(s, 0.0)) for s in pos_syms]
    pvar = (
        portfolio_var(notionals, vol_list, corr.loc[pos_syms, pos_syms].values)
        if positions
        else 0.0
    )

    top_pairs = top_correlated_pairs(corr)

    limit_checks = []
    holdings = []
    for pos in positions:
        chk = check_correlation_limit(pos["symbol"], holdings, corr, max_corr=max_corr)
        limit_checks.append(
            {
                "symbol": pos["symbol"],
                "avg_corr": chk["avg_corr"],
                "allowed": chk["allowed"],
                "reason": chk["reason"],
            }
        )
        if chk["allowed"]:
            holdings.append(pos["symbol"])

    currency = currency_exposure(
        [
            {
                "symbol": pos["symbol"],
                "direction": pos.get("direction", "long"),
                "notional": pos["qty"] * pos["entry"],
            }
            for pos in positions
        ]
    )

    return {
        "symbols": list(returns.columns),
        "n_setups": len(plans),
        "equity": equity,
        "correlation": corr,
        "top_correlated_pairs": top_pairs,
        "positions": [
            {
                "symbol": pos["symbol"],
                "direction": pos.get("direction", "long"),
                "notional": round(pos["qty"] * pos["entry"], 2),
                "risk_usd": round(
                    risk_dollars(
                        pos["qty"],
                        pos["entry"],
                        pos["stop"],
                        direction=pos.get("direction", "long"),
                    ),
                    2,
                ),
            }
            for pos in positions
        ],
        "heat_pct": round(heat * 100, 2),
        "heat_vs_limit": f"{heat * 100:.2f}% / {max_heat * 100:.0f}%",
        "portfolio_var_95_1bar": round(pvar, 2),
        "portfolio_var_pct_equity": round(pvar / equity * 100, 2)
        if equity > 0
        else 0.0,
        "correlation_checks": limit_checks,
        "vols": {s: round(float(v) * 100, 3) for s, v in vols.items()},
        # Currency-leg exposure (spec #28/#29): JPY-short concentration via
        # EURJPY+GBPJPY+CADJPY is invisible to a symbol-level correlation
        # matrix - the leg view surfaces it.
        "currency_exposure": currency,
    }


def _fmt_row(row: dict) -> str:
    return (
        f"{row['method']:<11} qty {row['qty']:>12,.2f}  "
        f"notional {row['notional']:>14,.0f}  "
        f"risk {row['risk_usd']:>10,.0f} ({row['risk_pct_equity']:>5.2f}%)  "
        f"VaR95 1d {row['var_95_1bar']:>9,.0f}  "
        f"VaR95 {row['var_95_hold']:>10,.0f}"
    )


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        description="NexusQuant risk & position sizing",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--symbol", default=None)
    parser.add_argument(
        "--symbols", default=None, help="comma-separated list (portfolio mode)"
    )
    parser.add_argument("--group", default=None)
    parser.add_argument("--data-dir", default="data/raw")
    parser.add_argument(
        "--timeframe", default=None, help="D1/H4/H1 (default from --group)"
    )
    parser.add_argument(
        "--portfolio", action="store_true", help="portfolio-level risk report"
    )
    parser.add_argument("--equity", type=float, default=DEFAULT_EQUITY)
    parser.add_argument("--risk-pct", type=float, default=DEFAULT_RISK_PCT)
    parser.add_argument("--vol-target", type=float, default=DEFAULT_VOL_TARGET)
    parser.add_argument(
        "--kelly-p",
        type=float,
        default=DEFAULT_KELLY_P,
        help="win probability when no ML model",
    )
    parser.add_argument("--payoff", type=float, default=DEFAULT_PAYOFF)
    parser.add_argument("--kelly-fraction", type=float, default=0.5)
    parser.add_argument("--hold-bars", type=int, default=10)
    parser.add_argument("--max-heat", type=float, default=0.04)
    parser.add_argument("--max-corr", type=float, default=0.6)
    parser.add_argument(
        "--include-short",
        action="store_true",
        help="fold Sell-the-Rally setups into the portfolio report (long+short)",
    )
    parser.add_argument(
        "--stress", action="store_true", help="per-symbol stress test (2008/COVID/2022)"
    )
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args(argv)

    if args.timeframe is None:
        args.timeframe = {"h1": "H1", "h4": "H4"}.get(args.group, "D1")

    if args.symbols:
        symbols = [s.strip() for s in args.symbols.split(",") if s.strip()]
    elif args.symbol:
        symbols = [args.symbol]
    else:
        symbols = discover_symbols(args.data_dir, args.group, args.timeframe)

    if not symbols:
        print("No symbols found.", file=sys.stderr)
        return 1

    if args.portfolio or len(symbols) > 1:
        rep = portfolio_report(
            symbols,
            args.group,
            args.timeframe,
            args.data_dir,
            equity=args.equity,
            risk_pct=args.risk_pct,
            max_heat=args.max_heat,
            max_corr=args.max_corr,
            include_short=args.include_short,
        )
        if args.json:
            printable = {k: v for k, v in rep.items() if k != "correlation"}
            printable["correlation"] = rep["correlation"].round(3).to_dict()
            print(json.dumps(printable, indent=2, default=str))
            return 0
        print("\n" + "=" * 74)
        print("NEXUSQUANT PORTFOLIO RISK REPORT")
        print("=" * 74)
        if rep.get("reason"):
            print(f"({rep['reason']})")
        print(
            f"symbols: {len(rep['symbols'])} · setups: {rep['n_setups']} · "
            f"equity {rep['equity']:,.0f}"
        )
        print(
            f"portfolio heat    : {rep['heat_pct']:.2f}% (limit {rep['heat_vs_limit']})"
        )
        print(
            f"portfolio VaR 95% : {rep['portfolio_var_95_1bar']:,.0f} "
            f"({rep['portfolio_var_pct_equity']:.2f}% of equity, 1 bar)"
        )
        for pos in rep["positions"]:
            d = "SHORT" if pos.get("direction") == "short" else "LONG "
            print(
                f"  {d} {pos['symbol']:<10} notional {pos['notional']:>14,.0f} "
                f"risk {pos['risk_usd']:>10,.0f}"
            )
        ce = rep.get("currency_exposure") or {}
        if ce.get("exposure"):
            print("\nCurrency-leg exposure (spec #28/#29):")
            for ccy, v in sorted(ce["exposure"].items(), key=lambda kv: -abs(kv[1])):
                print(f"  {ccy:<6} {v:+,.0f}")
            print(f"  gross {ce['gross']:,.0f} · net {ce['net']:,.0f}")
            for w in ce.get("warnings", []):
                print(f"  ⚠ {w}")
        print("\nCorrelation-aware limit checks:")
        for chk in rep["correlation_checks"]:
            mark = "OK " if chk["allowed"] else "BLOCK"
            print(
                f"  {mark} {chk['symbol']:<10} avg-corr "
                f"{chk['avg_corr'] if chk['avg_corr'] is not None else '-':<6} "
                f"{chk['reason']}"
            )
        if rep["top_correlated_pairs"]:
            print("\nMost correlated pairs:")
            for p in rep["top_correlated_pairs"]:
                print(f"  {p['pair']:<22} {p['corr']:.3f}")
        print("=" * 74 + "\n")
        return 0

    if args.stress:
        sp = stress_plan(
            symbols[0], args.group, args.timeframe, args.data_dir, equity=args.equity
        )
        if args.json:
            print(json.dumps(sp, indent=2, default=str))
            return 0
        _print_stress(sp)
        return 0

    plan = symbol_risk_plan(
        symbols[0],
        args.group,
        args.timeframe,
        args.data_dir,
        equity=args.equity,
        risk_pct=args.risk_pct,
        vol_target=args.vol_target,
        kelly_p=args.kelly_p,
        payoff=args.payoff,
        kelly_fraction=args.kelly_fraction,
        hold_bars=args.hold_bars,
    )
    if args.json:
        print(json.dumps(plan, indent=2, default=str))
        return 0

    if not plan.get("setup"):
        print(f"{plan['symbol']}: {plan.get('reason', 'no setup')}")
        return 0
    s = plan["setup"]
    print("\n" + "=" * 74)
    print(f"NEXUSQUANT RISK PLAN — {plan['symbol']} ({plan['date']})")
    print("=" * 74)
    print(
        f"setup: entry {s['entry']:,.5f} · stop {s['stop']:,.5f} · "
        f"target {s['target']:,.5f} · R:R {s['rr']}"
    )
    ok = s.get("rr_ok")
    if ok is not None:
        mark = "✓ meets floor" if ok else "✗ below floor"
        print(
            f"       R:R floor {s.get('min_rr')}:1 → {mark} · "
            f"nearest-target R:R {s.get('rr_nearest')} · "
            f"ladder best {s.get('best_rr')} ({s.get('min_rr_tp') or 'none'})"
        )
    print(
        f"       dip {s['dip_stage']} (score {s['dip_score']}) · "
        f"ATR {s['atr_pct']}% · "
        f"ML prob {s['ml_prob']}%"
        if s["ml_prob"] is not None
        else f"       dip {s['dip_stage']} (score {s['dip_score']}) · "
        f"ATR {s['atr_pct']}% · no ML model"
    )
    print("-" * 74)
    for row in plan["sizes"]:
        print(_fmt_row(row))
    print("-" * 74)
    src = "ML model" if plan["setup"]["ml_prob"] is not None else "fallback"
    print(
        f"equity {plan['inputs']['equity']:,.0f} · "
        f"kelly p {plan['inputs']['kelly_p']} ({src}) · "
        f"hold {plan['inputs']['hold_bars']} bars"
    )
    print("=" * 74 + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
