"""
NexusQuant - Buy-the-Dip backtest CLI.

Run the causal dip strategy over historical data and get edge statistics:

    python -m src.backtest.run --symbol XAUUSD --group full_fx
    python -m src.backtest.run --symbols EURUSD,GBPUSD --start 2015
    python -m src.backtest.run --group candidates --top 5
    python -m src.backtest.run --symbol US500 --group candidates --risk 0.02 --rr 2 --json
    python -m src.backtest.run --symbol XAUUSD --group full_fx --plot  # saves equity.png

All data is local-only (no MT5 fetches). Sizing risk is fractional equity
per trade by default (--sizing voltarget/kelly switch the sizing engine);
exits are the signal's invalidation (stop) and nearest resistance
(target, with an R:R fallback), plus a time stop.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import List, Optional

import pandas as pd

sys.path.append(str(Path(__file__).parent.parent.parent))

from src.data.loader import clean_data, load_data
from src.features.indicators import add_all_indicators
from src.features.regime import detect_regime
from src.analysis.scanner import _data_path, discover_symbols
from src.backtest.engine import (
    BacktestParams,
    BacktestResult,
    run_backtest,
    run_backtest_both,
    print_stats,
    regime_breakdown,
)
from src.backtest.signals import dip_signal_series, rally_signal_series

BARS_PER_YEAR = {"D1": 252, "H4": 1512, "H1": 6048}


def prepare_frame(
    symbol: str,
    group: Optional[str],
    timeframe: str,
    data_dir: str = "data/raw",
    start: Optional[str] = None,
) -> pd.DataFrame:
    """Load, clean and add indicators for one symbol (local only)."""
    path = _data_path(symbol, data_dir, group, timeframe)
    df = load_data(path, symbol=symbol)
    df = clean_data(df)
    if start:
        df = df[df.index >= pd.Timestamp(start)]
    df = add_all_indicators(df)
    # Causal regime column (trailing windows only) - feeds the per-regime
    # performance breakdown so we know where the edge is earned/lost.
    df = detect_regime(df)
    return df


def backtest_symbol(
    symbol: str,
    group: Optional[str],
    timeframe: str,
    data_dir: str = "data/raw",
    start: Optional[str] = None,
    params: Optional[BacktestParams] = None,
    side: str = "long",
) -> BacktestResult:
    df = prepare_frame(symbol, group, timeframe, data_dir, start)
    if len(df) < 60:
        raise ValueError(f"{symbol}: only {len(df)} bars after filtering")
    p = params or BacktestParams()
    if p.bars_per_year is None:
        p.bars_per_year = BARS_PER_YEAR.get(timeframe.upper(), 252)
    if side == "both":
        return run_backtest_both(
            dip_signal_series(df), rally_signal_series(df), df, p, symbol=symbol
        )
    signal = dip_signal_series(df) if side == "long" else rally_signal_series(df)
    result = run_backtest(signal, df, p, symbol=symbol, side=side)
    if "regime" in df.columns:
        result.stats["regime_breakdown"] = regime_breakdown(result, df["regime"])
    return result


def _symbol_summary_row(result) -> dict:
    s = result.stats
    return {
        "symbol": result.symbol,
        "trades": s["n_trades"],
        "win_rate": s["win_rate"],
        "profit_factor": s["profit_factor"],
        "expectancy_r": s["expectancy_r"],
        "return_pct": s["total_return_pct"],
        "max_dd_pct": s["max_drawdown_pct"],
        "sharpe": s["sharpe"],
    }


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        description="NexusQuant Buy-the-Dip backtester",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--symbol", default=None, help="single symbol")
    parser.add_argument("--symbols", default=None, help="comma-separated symbol list")
    parser.add_argument(
        "--group", default=None, help="data group (full_fx, candidates, majors, ...)"
    )
    parser.add_argument("--data-dir", default="data/raw")
    parser.add_argument(
        "--timeframe", default=None, help="D1/H4/H1 (default derived from --group)"
    )
    parser.add_argument("--start", default=None, help="start date, e.g. 2015-01-01")
    parser.add_argument(
        "--risk", type=float, default=0.01, help="fraction of equity risked per trade"
    )
    parser.add_argument(
        "--rr", type=float, default=2.0, help="R:R fallback target when no level exists"
    )
    parser.add_argument("--max-hold", type=int, default=20, help="time-stop in bars")
    parser.add_argument(
        "--entry-valid", type=int, default=3, help="limit order validity in bars"
    )
    parser.add_argument(
        "--slippage", type=float, default=0.0, help="proportional cost per side"
    )
    parser.add_argument("--entry-type", default="limit", choices=["limit", "market"])
    parser.add_argument(
        "--sizing",
        default="fractional",
        choices=["fractional", "voltarget", "kelly"],
        help="position sizing method",
    )
    parser.add_argument(
        "--trials",
        type=int,
        default=20,
        help="number of independently tested strategies for "
        "the deflated Sharpe ratio (research search "
        "space estimate)",
    )
    parser.add_argument(
        "--ablate",
        action="store_true",
        help="ablate the signal-score threshold (0/4/5/6/7) "
        "on the first symbol - does higher conviction "
        "actually improve the edge?",
    )
    parser.add_argument(
        "--sensitivity",
        default=None,
        metavar="param=lo:hi:n",
        help="sweep one backtest parameter across n points "
        "(e.g. slippage=0:0.002:5) on the first symbol",
    )
    parser.add_argument(
        "--side",
        default="long",
        choices=["long", "short", "both"],
        help="long (Buy-the-Dip), short (Sell-the-Rally) "
        "or both (combined research view)",
    )
    parser.add_argument(
        "--vol-target",
        type=float,
        default=0.02,
        help="per-trade vol contribution (voltarget sizing)",
    )
    parser.add_argument(
        "--kelly-p", type=float, default=0.55, help="win probability (kelly sizing)"
    )
    parser.add_argument(
        "--payoff",
        type=float,
        default=1.5,
        help="avg win per unit risked (kelly sizing)",
    )
    parser.add_argument(
        "--kelly-fraction", type=float, default=0.5, help="fractional Kelly multiplier"
    )
    parser.add_argument("--capital", type=float, default=100_000.0)
    parser.add_argument(
        "--top", type=int, default=None, help="show only the top N ranked symbols"
    )
    parser.add_argument("--json", action="store_true", help="emit JSON")
    parser.add_argument(
        "--plot", action="store_true", help="save equity.png for the last symbol"
    )
    args = parser.parse_args(argv)

    if args.timeframe is None:
        args.timeframe = {
            "h1": "H1",
            "h4": "H4",
            "mt5/D1": "D1",
            "mt5/H4": "H4",
            "mt5/H1": "H1",
        }.get(args.group, "D1")

    if args.symbol:
        symbols = [args.symbol]
    elif args.symbols:
        symbols = [s.strip() for s in args.symbols.split(",") if s.strip()]
    else:
        symbols = discover_symbols(args.data_dir, args.group, args.timeframe)

    if not symbols:
        print(f"No symbols found for group={args.group} tf={args.timeframe}")
        return 1

    params = BacktestParams(
        initial_capital=args.capital,
        risk_pct=args.risk,
        rr_fallback=args.rr,
        max_hold=args.max_hold,
        entry_valid_bars=args.entry_valid,
        slippage=args.slippage,
        entry_type=args.entry_type,
        sizing_mode=args.sizing,
        vol_target=args.vol_target,
        kelly_p=args.kelly_p,
        payoff=args.payoff,
        kelly_fraction=args.kelly_fraction,
        bars_per_year=BARS_PER_YEAR.get(args.timeframe.upper(), 252),
        n_trials=args.trials,
    )

    results, errors = [], []
    for sym in symbols:
        try:
            results.append(
                backtest_symbol(
                    sym,
                    args.group,
                    args.timeframe,
                    args.data_dir,
                    args.start,
                    params,
                    side=args.side,
                )
            )
        except Exception as exc:
            errors.append((sym, str(exc)))

    if not results:
        print(f"No symbol could be backtested. {errors}", file=sys.stderr)
        return 1

    if args.json:
        out = {"params": params.__dict__, "side": args.side, "symbols": []}
        for r in results:
            out["symbols"].append(
                {
                    "symbol": r.symbol,
                    "stats": {
                        k: (None if isinstance(v, float) and math.isnan(v) else v)
                        for k, v in r.stats.items()
                    },
                    "trades": r.trades_frame().to_dict(orient="records"),
                }
            )
        if errors:
            out["errors"] = [list(e) for e in errors]
        print(json.dumps(out, indent=2, default=str))
        return 0

    # Multi-symbol summary
    if len(results) > 1:
        rows = [_symbol_summary_row(r) for r in results]
        table = pd.DataFrame(rows).sort_values(
            ["return_pct", "sharpe"], ascending=False
        )
        if args.top:
            table = table.head(args.top)
        table["win_rate"] = table["win_rate"].map(lambda v: f"{v * 100:.0f}%")
        table["profit_factor"] = table["profit_factor"].map(
            lambda v: "inf" if math.isinf(v) else f"{v:.2f}"
        )
        table["expectancy_r"] = table["expectancy_r"].map(lambda v: f"{v:+.2f}R")
        table["return_pct"] = table["return_pct"].map(lambda v: f"{v:+.1f}%")
        table["max_dd_pct"] = table["max_dd_pct"].map(lambda v: f"{v:.1f}%")
        print("\n" + "=" * 78)
        side_label = {
            "long": "BUY-THE-DIP",
            "short": "SELL-THE-RALLY",
            "both": "LONG+SHORT COMBINED",
        }[args.side]
        print(f"NEXUSQUANT {side_label} BACKTEST — UNIVERSE SUMMARY")
        print(
            f"params: risk {params.risk_pct:.1%} · sizing {params.sizing_mode} · "
            f"R:R fallback {params.rr_fallback} · "
            f"max-hold {params.max_hold} · entry {params.entry_type} · "
            f"side {args.side} · {args.timeframe}"
        )
        print("=" * 78)
        print(table.to_string(index=False))
        print("=" * 78)
        if errors:
            print(f"[!] {len(errors)} failed: {errors}", file=sys.stderr)
        print(f"Backtested {len(results)} symbols.\n")
        return 0

    # Single symbol detail
    r = results[0]
    print("\n" + "=" * 60)
    side_label = {
        "long": "BUY-THE-DIP",
        "short": "SELL-THE-RALLY",
        "both": "LONG+SHORT COMBINED",
    }[args.side]
    print(f"NEXUSQUANT {side_label} BACKTEST — {r.symbol} ({args.timeframe})")
    print(
        f"period {r.equity.index[0].date()} → {r.equity.index[-1].date()} · "
        f"{len(r.equity)} bars"
    )
    print("=" * 60)
    print(print_stats(r))

    if args.ablate:
        from src.backtest.robustness import ablate_threshold

        df0 = prepare_frame(
            symbols[0], args.group, args.timeframe, args.data_dir, args.start
        )
        sig = (
            dip_signal_series(df0)
            if args.side in ("long", "both")
            else rally_signal_series(df0)
        )
        print("\n" + "-" * 60)
        print("THRESHOLD ABLATION (score >= t, does conviction improve edge?)")
        print(
            ablate_threshold(
                sig, df0, params, side=("short" if args.side == "short" else "long")
            ).to_string(index=False)
        )
    if args.sensitivity:
        from src.backtest.robustness import param_sensitivity

        df0 = prepare_frame(
            symbols[0], args.group, args.timeframe, args.data_dir, args.start
        )
        sig = (
            dip_signal_series(df0)
            if args.side in ("long", "both")
            else rally_signal_series(df0)
        )
        try:
            pname, lo, hi, npts = _parse_sweep(args.sensitivity)
        except ValueError as exc:
            print(f"[!] --sensitivity: {exc}", file=sys.stderr)
            return 2
        values = [lo + (hi - lo) * i / (npts - 1) for i in range(npts)]
        print("\n" + "-" * 60)
        print(f"PARAM SENSITIVITY ({pname} swept over {npts} values)")
        print(
            param_sensitivity(
                sig,
                df0,
                params,
                side=("short" if args.side == "short" else "long"),
                param=pname,
                values=values,
            ).to_string(index=False)
        )
    print("-" * 60)
    if r.trades:
        tf = r.trades_frame()
        show = tf.tail(10)[
            [
                "entry_time",
                "exit_time",
                "entry_price",
                "exit_price",
                "pnl",
                "pnl_pct",
                "reason",
                "bars_held",
            ]
        ].copy()
        show["entry_time"] = show["entry_time"].map(lambda v: str(v.date()))
        show["exit_time"] = show["exit_time"].map(lambda v: str(v.date()))
        show["entry_price"] = show["entry_price"].map(lambda v: f"{v:,.5f}")
        show["exit_price"] = show["exit_price"].map(lambda v: f"{v:,.5f}")
        show["pnl"] = show["pnl"].map(lambda v: f"{v:,.0f}")
        show["pnl_pct"] = show["pnl_pct"].map(lambda v: f"{v:+.2%}")
        print(f"Last {len(show)} trades:")
        print(show.to_string(index=False))
    print("=" * 60 + "\n")

    if args.plot:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots(figsize=(10, 4), dpi=110)
        ax.plot(r.equity.index, r.equity.values, lw=1.2, color="#f0b90b")
        ax.set_title(f"{r.symbol} — {side_label} equity curve")
        ax.set_ylabel("Equity")
        ax.grid(alpha=0.3)
        fig.tight_layout()
        fig.savefig("equity.png")
        print("Saved equity curve → equity.png")

    return 0


def _parse_sweep(spec: str):
    """Parse ``param=lo:hi:n`` -> (param, lo, hi, n_points)."""
    if "=" not in spec or ":" not in spec:
        raise ValueError("expected param=lo:hi:n, e.g. slippage=0:0.002:5")
    pname, rest = spec.split("=", 1)
    parts = rest.split(":")
    if len(parts) != 3:
        raise ValueError("expected param=lo:hi:n, e.g. slippage=0:0.002:5")
    try:
        lo, hi, npts = float(parts[0]), float(parts[1]), int(parts[2])
    except ValueError:
        raise ValueError("lo/hi must be numbers and n an integer") from None
    if npts < 2 or hi <= lo:
        raise ValueError("need n >= 2 and hi > lo")
    return pname.strip(), lo, hi, npts


if __name__ == "__main__":
    raise SystemExit(main())
