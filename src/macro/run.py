"""
NexusQuant - Macro Overlay CLI.

Top-down context on top of the technical engines:

    python -m src.macro.run                              # macro snapshot
    python -m src.macro.run --symbols EURUSD,GBPUSD      # + per-symbol bias/gate
    python -m src.macro.run --scan --group full_fx --top 10   # overlay scan
    python -m src.macro.run --compare --symbol XAUUSD --group full_fx
                                                         # gate vs no-gate backtest
    python -m src.macro.run --json                       # machine-readable

The snapshot prints the current USD / risk / rates regime; the scan ranks
symbols by their macro bias (tailwind/headwind) and flags which dip
signals pass the macro gate; the compare mode runs the Buy-the-Dip
backtest with and without the gate and reports the edge delta.

Local-first: DXY comes from ``data/raw/indices/``; VIX/TNX are a
best-effort Yahoo fetch cached under ``data/raw/macro/`` (the overlay
works fine without them).
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Dict, List, Optional

import pandas as pd

sys.path.append(str(Path(__file__).parent.parent.parent))

from src.data.loader import clean_data, load_data
from src.features.indicators import add_all_indicators
from src.backtest.engine import BacktestParams, run_backtest
from src.backtest.signals import dip_signal_series
from src.macro.overlay import (
    SECTOR_ETF,
    _symbol_class as _overlay_symbol_class,
    align_scores,
    full_macro_scores,
    gate_series,
    latest_macro_scores,
    load_yahoo_daily,
    macro_bias_for_symbol,
    macro_gate,
    macro_regime,
)

BARS_PER_YEAR = {"D1": 252, "H4": 1512, "H1": 6048}


# ---------------------------------------------------------------------------
# Snapshot
# ---------------------------------------------------------------------------


def _regime_row(label: str, scores_row: Dict) -> Dict:
    r = macro_regime(scores_row)
    return {
        "factor": label,
        "usd": r["usd"],
        "risk": r["risk"],
        "rates": r["rates"],
        "composite": r["composite"],
        "dxy": r["dxy_score"],
        "vix": r["vix_score"],
        "tnx": r["tnx_score"],
    }


def snapshot(
    data_dir: str = "data/raw", cache_dir: str = "data/raw/macro", fetch: bool = False
) -> Dict:
    """Current macro regime + per-factor scores (empty-safe)."""
    row = latest_macro_scores(data_dir, cache_dir, fetch=fetch)
    if row is None:
        return {"available": False, "reason": "no macro source available"}
    row_dict = row.iloc[-1].to_dict()
    regime = macro_regime(row_dict)
    return {
        "available": True,
        "date": str(row.index[-1].date()),
        "regime": regime,
        "factors": {
            "dxy_score": regime["dxy_score"],
            "vix_score": regime["vix_score"],
            "tnx_score": regime["tnx_score"],
        },
    }


def symbol_bias(
    symbol: str, data_dir: str = "data/raw", cache_dir: str = "data/raw/macro"
) -> Optional[Dict]:
    """Per-symbol macro bias + gate for the latest macro row."""
    row = latest_macro_scores(data_dir, cache_dir)
    if row is None:
        return None
    row_dict = row.iloc[-1].to_dict()
    bias = macro_bias_for_symbol(symbol, row_dict)
    gate = macro_gate(symbol, row_dict)
    return {
        "symbol": symbol,
        **bias,
        "gate_allowed": gate["allowed"],
        "gate_reason": gate["reason"],
    }


def warm_sector_etfs(
    data_dir: str = "data/raw", cache_dir: str = "data/raw/macro"
) -> Dict:
    """Best-effort Yahoo download of every mapped sector ETF (spec #9).

    Cached as ``{cache_dir}/{ETF}_D1.parquet`` - the same key the
    sensitivity table reads. Offline / unknown tickers are skipped, never
    fatal.
    """
    ok, failed = [], []
    for _, etf in sorted(SECTOR_ETF.items()):
        if (Path(cache_dir) / f"{etf}_D1.parquet").exists():
            continue
        s = load_yahoo_daily(etf, cache_dir)
        (ok if s is not None else failed).append(etf)
    return {"cached": sorted(set(ok)), "failed": sorted(set(failed))}


# ---------------------------------------------------------------------------
# Overlay scan
# ---------------------------------------------------------------------------


def _symbol_class(symbol: str) -> str:
    kind, _ = _overlay_symbol_class(symbol)
    return kind


def overlay_scan(
    symbols: List[str], data_dir: str = "data/raw", cache_dir: str = "data/raw/macro"
) -> pd.DataFrame:
    """Rank a symbol list by macro bias (tailwind first)."""
    rows = []
    for sym in symbols:
        b = symbol_bias(sym, data_dir, cache_dir)
        if b is None:
            rows.append(
                {
                    "symbol": sym,
                    "class": _symbol_class(sym),
                    "bias": None,
                    "label": "n/a",
                    "gate": None,
                    "dxy": None,
                    "vix": None,
                    "tnx": None,
                }
            )
            continue
        rows.append(
            {
                "symbol": sym,
                "class": _symbol_class(sym),
                "bias": b["bias"],
                "label": b["label"],
                "gate": "PASS" if b["gate_allowed"] else "BLOCKED",
                "dxy": b["factors"]["dxy"],
                "vix": b["factors"]["vix"],
                "tnx": b["factors"]["tnx"],
            }
        )
    table = pd.DataFrame(rows)
    if not table.empty and table["bias"].notna().any():
        table = table.sort_values("bias", ascending=False)
    return table


# ---------------------------------------------------------------------------
# With-vs-without gate backtest comparison
# ---------------------------------------------------------------------------


def _prepare_frame(
    symbol: str,
    group: Optional[str],
    timeframe: str,
    data_dir: str,
    start: Optional[str],
) -> pd.DataFrame:
    from src.analysis.scanner import _data_path

    path = _data_path(symbol, data_dir, group, timeframe)
    df = load_data(path, symbol=symbol)
    df = clean_data(df)
    if start:
        df = df[df.index >= pd.Timestamp(start)]
    return add_all_indicators(df)


def compare_gate(
    symbol: str,
    group: Optional[str],
    timeframe: str,
    data_dir: str = "data/raw",
    cache_dir: str = "data/raw/macro",
    start: Optional[str] = None,
    min_bias: float = -0.5,
    params: Optional[BacktestParams] = None,
) -> Dict:
    """
    Run the dip backtest twice - raw vs macro-gated - and compare.

    The gate masks ``signal["confirmed"]`` so a dip signal whose macro
    backdrop is a strong headwind never enters. Everything stays causal
    (macro state as of the prior day).
    """
    df = _prepare_frame(symbol, group, timeframe, data_dir, start)
    if len(df) < 60:
        raise ValueError(f"{symbol}: only {len(df)} bars after filtering")

    signal = dip_signal_series(df)
    p = params or BacktestParams()
    if p.bars_per_year is None:
        p.bars_per_year = BARS_PER_YEAR.get(timeframe.upper(), 252)

    raw = run_backtest(signal, df, p, symbol=symbol)

    scores = full_macro_scores(data_dir, cache_dir)
    if scores is None or scores.empty:
        return {
            "symbol": symbol,
            "gate_available": False,
            "raw": raw.stats,
            "reason": "no macro source — gate not applied",
        }

    gate = gate_series(symbol, scores, df.index, min_bias=min_bias)
    # Honest coverage: bars before the first macro score are ungated by
    # definition, not "blocked" - report that fraction separately.
    aligned = align_scores(scores, df.index, shift_days=1)
    pre_macro = aligned["dxy_score"].isna().mean() * 100
    gated_signal = signal.copy()
    gated_signal["confirmed"] = signal["confirmed"] & gate
    gated = run_backtest(gated_signal, df, p, symbol=symbol)

    raw_s, gat_s = raw.stats, gated.stats

    def _d(k):
        rv, gv = raw_s.get(k), gat_s.get(k)
        if isinstance(rv, float) and isinstance(gv, float) and rv != 0:
            return round((gv - rv) / abs(rv) * 100, 1)
        return None

    return {
        "symbol": symbol,
        "gate_available": True,
        "timeframe": timeframe,
        "bars": len(df),
        "start": str(df.index[0].date()),
        "end": str(df.index[-1].date()),
        "gate": {
            "min_bias": min_bias,
            "blocked_bars": int((~gate).sum()),
            "blocked_pct": round(float((~gate).mean() * 100), 1),
            "pre_macro_pct": round(float(pre_macro), 1),
        },
        "raw": {
            k: raw_s.get(k)
            for k in [
                "n_trades",
                "win_rate",
                "profit_factor",
                "expectancy_r",
                "total_return_pct",
                "max_drawdown_pct",
                "sharpe",
            ]
        },
        "gated": {
            k: gat_s.get(k)
            for k in [
                "n_trades",
                "win_rate",
                "profit_factor",
                "expectancy_r",
                "total_return_pct",
                "max_drawdown_pct",
                "sharpe",
            ]
        },
        "delta_pct": {
            "n_trades": _d("n_trades"),
            "total_return_pct": _d("total_return_pct"),
            "sharpe": _d("sharpe"),
        },
    }


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        description="NexusQuant macro overlay",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--data-dir", default="data/raw")
    parser.add_argument("--cache-dir", default="data/raw/macro")
    parser.add_argument(
        "--fetch",
        action="store_true",
        help="attempt Yahoo fetch for VIX/TNX and the sector "
        "ETFs (spec #9) if not cached",
    )
    parser.add_argument(
        "--symbols", default=None, help="comma-separated symbols for the snapshot"
    )
    parser.add_argument(
        "--scan",
        action="store_true",
        help="overlay scan of a group (ranked by macro bias)",
    )
    parser.add_argument(
        "--group", default=None, help="data group for --scan / --compare"
    )
    parser.add_argument(
        "--timeframe", default=None, help="D1/H4/H1 (default derived from --group)"
    )
    parser.add_argument("--top", type=int, default=None)
    parser.add_argument(
        "--compare",
        default=None,
        metavar="SYMBOL",
        help="with-vs-without gate backtest for a symbol",
    )
    parser.add_argument(
        "--start", default=None, help="start date, e.g. 2015-01-01 (compare mode)"
    )
    parser.add_argument(
        "--min-bias", type=float, default=-0.5, help="gate threshold on the macro bias"
    )
    parser.add_argument("--risk", type=float, default=0.01)
    parser.add_argument("--rr", type=float, default=2.0)
    parser.add_argument("--max-hold", type=int, default=20)
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args(argv)

    if args.timeframe is None:
        args.timeframe = {"h1": "H1", "h4": "H4"}.get(args.group, "D1")

    # ---- mode 1: compare backtest --------------------------------------
    if args.compare:
        params = BacktestParams(
            risk_pct=args.risk,
            rr_fallback=args.rr,
            max_hold=args.max_hold,
            bars_per_year=BARS_PER_YEAR.get(args.timeframe.upper(), 252),
        )
        try:
            res = compare_gate(
                args.compare,
                args.group,
                args.timeframe,
                args.data_dir,
                args.cache_dir,
                args.start,
                min_bias=args.min_bias,
                params=params,
            )
        except Exception as exc:
            print(f"[macro] compare failed: {exc}", file=sys.stderr)
            return 1
        if args.json:
            print(json.dumps(res, indent=2, default=str))
            return 0
        print("\n" + "=" * 78)
        print(
            f"MACRO GATE BACKTEST — {res['symbol']} ({res.get('timeframe', '')})"
            + (
                f" · {res['start']} → {res['end']} · {res['bars']} bars"
                if res.get("start")
                else ""
            )
        )
        print("=" * 78)
        if not res.get("gate_available"):
            print(res["reason"])
            return 0
        g = res["gate"]
        print(
            f"Gate: bias ≥ {g['min_bias']} · blocked {g['blocked_bars']} "
            f"bars ({g['blocked_pct']}%) · pre-macro (ungated) "
            f"{g.get('pre_macro_pct', 0)}%"
        )
        print(f"{'metric':<18}{'raw':>12}{'gated':>12}{'Δ%':>9}")
        labels = {
            "n_trades": "trades",
            "win_rate": "win rate",
            "profit_factor": "profit factor",
            "expectancy_r": "expectancy",
            "total_return_pct": "total return",
            "max_drawdown_pct": "max DD",
            "sharpe": "sharpe",
        }
        for k, lab in labels.items():
            rv = res["raw"].get(k)
            gv = res["gated"].get(k)
            if rv is None or (isinstance(rv, float) and math.isnan(rv)):
                continue
            rf = (
                f"{rv * 100:.1f}%"
                if k in ("win_rate",)
                else f"{rv:+.2f}"
                if k in ("expectancy_r", "sharpe")
                else f"{rv:.2f}"
                if k in ("profit_factor",)
                else f"{rv:+.1f}%"
                if k.endswith("_pct")
                else f"{rv}"
            )
            gf = (
                f"{gv * 100:.1f}%"
                if k in ("win_rate",)
                else f"{gv:+.2f}"
                if k in ("expectancy_r", "sharpe")
                else f"{gv:.2f}"
                if k in ("profit_factor",)
                else f"{gv:+.1f}%"
                if k.endswith("_pct")
                else f"{gv}"
            )
            dp = res["delta_pct"].get(k)
            dstr = f"{dp:+.1f}%" if dp is not None else "-"
            print(f"{lab:<18}{rf:>12}{gf:>12}{dstr:>9}")
        print("=" * 78)
        return 0

    # ---- mode 2: overlay scan ------------------------------------------
    if args.scan:
        from src.analysis.scanner import discover_symbols

        symbols = discover_symbols(args.data_dir, args.group, args.timeframe)
        if not symbols:
            print(
                f"No symbols found for group={args.group} tf={args.timeframe}",
                file=sys.stderr,
            )
            return 1
        table = overlay_scan(symbols, args.data_dir, args.cache_dir)
        if args.top:
            table = table.head(args.top)
        if args.json:
            recs = table.to_dict(orient="records")
            for rec in recs:
                for k, v in rec.items():
                    if isinstance(v, float) and math.isnan(v):
                        rec[k] = None
            print(json.dumps(recs, indent=2))
            return 0
        show = table.copy()
        show["bias"] = show["bias"].map(lambda v: "n/a" if v is None else f"{v:+.2f}")
        show = show.fillna("n/a")
        print("\n" + "=" * 78)
        print(f"MACRO OVERLAY SCAN — {args.group or 'default'} · {args.timeframe}")
        print("=" * 78)
        print(show.to_string(index=False))
        print("=" * 78)
        print(f"{len(table)} symbols ranked by macro bias.\n")
        return 0

    # ---- mode 3: snapshot ----------------------------------------------
    if args.fetch:
        warm_sector_etfs(args.data_dir, args.cache_dir)
    snap = snapshot(args.data_dir, args.cache_dir, fetch=args.fetch)
    if args.json:
        print(json.dumps(snap, indent=2, default=str))
        return 0
    if not snap.get("available"):
        print(f"[macro] {snap['reason']}")
        return 0
    r = snap["regime"]
    print("\n" + "=" * 78)
    print(f"MACRO SNAPSHOT — {snap['date']}")
    print("=" * 78)
    print(f"USD regime : {r['usd']:<14} score {r['dxy_score']}")
    print(f"Risk regime: {r['risk']:<14} score {r['vix_score']}")
    print(f"Rates regime: {r['rates']:<14} score {r['tnx_score']}")
    print(f"Composite  : {r['composite']:+.2f}")
    if args.symbols:
        syms = [s.strip() for s in args.symbols.split(",") if s.strip()]
        print("-" * 78)
        print(f"{'symbol':<12}{'bias':>7}  {'label':<17}{'gate':>8}  note")
        for sym in syms:
            b = symbol_bias(sym, args.data_dir, args.cache_dir)
            if b is None:
                print(f"{sym:<12}  n/a")
                continue
            print(
                f"{sym:<12}{b['bias']:>+7.2f}  {b['label']:<17}"
                f"{'PASS' if b['gate_allowed'] else 'BLOCKED':>8}  "
                f"{b['note']}"
            )
    print("=" * 78 + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
