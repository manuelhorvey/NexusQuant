"""
NexusQuant - Multi-Symbol Universe Scanner & Ranking

Scans every symbol in a data directory through the full pipeline
(load -> clean -> indicators -> regime -> report) and produces a
ranked institutional-style comparison table.

Usage:
    python -m src.analysis.scanner                          # top-level FX majors (D1)
    python -m src.analysis.scanner --group full_fx          # all 29 FX pairs
    python -m src.analysis.scanner --group candidates       # US30, US500, USTEC, BTCUSD
    python -m src.analysis.scanner --group equity_universe  # 500 stocks
    python -m src.analysis.scanner --timeframe H1           # higher resolution
    python -m src.analysis.scanner --top 10                 # show only the top 10
    python -m src.analysis.scanner --json                   # machine-readable output
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
from src.features.dip import detect_dip
from src.features.rally import detect_rally
from src.analysis.report import generate_full_report
from src.analysis.plan import trade_plan
from src.data.resolver import effective_group, find_local, resolve_symbol_data
from src.model.model import DEFAULT_MODEL_PATH, predict_series


# ---------------------------------------------------------------------------
# Discovery
# ---------------------------------------------------------------------------


def discover_symbols(
    data_dir: str = "data/raw",
    group: Optional[str] = None,
    timeframe: str = "D1",
) -> List[str]:
    """Discover symbols from `{SYMBOL}_{TIMEFRAME}.parquet` files."""
    base = Path(data_dir)
    if group:
        base = base / group
    files = sorted(base.glob(f"*_{timeframe}.parquet"))
    return [f.stem.replace(f"_{timeframe}", "") for f in files]


def _data_path(
    symbol: str, data_dir: str, group: Optional[str], timeframe: str
) -> Path:
    base = Path(data_dir)
    if group:
        base = base / group
    return base / f"{symbol}_{timeframe.upper()}.parquet"


# ---------------------------------------------------------------------------
# Directional scoring
# ---------------------------------------------------------------------------


def _directional_score(df: pd.DataFrame) -> Dict:
    """
    Compute a directional bias in [-4, +4] from the latest bar:
    + = bullish evidence, - = bearish evidence.
    Components (4 bullish / 4 bearish):
      price vs SMA200, RSI vs 50, MACD histogram sign, ADX(+DI vs -DI)
    """
    latest = df.iloc[-1]
    close = latest.get("close", 0.0)
    sma200 = latest.get("sma_200", close)
    rsi = latest.get("rsi_14", 50.0)
    macd_hist = latest.get("macd_hist", 0.0)
    adx = latest.get("adx", 0.0)
    plus_di = latest.get("plus_di", 0.0)
    minus_di = latest.get("minus_di", 0.0)

    bull = 0
    bear = 0
    bull += int(close > sma200)
    bear += int(close < sma200)
    bull += int(rsi > 50)
    bear += int(rsi < 50)
    bull += int(macd_hist > 0)
    bear += int(macd_hist < 0)
    if adx > 25:
        bull += int(plus_di > minus_di)
        bear += int(minus_di > plus_di)

    score = bull - bear  # -4 .. +4

    if score >= 3:
        label = "Strong Bullish"
    elif score >= 2:
        label = "Bullish"
    elif score >= 1:
        label = "Mild Bullish"
    elif score <= -3:
        label = "Strong Bearish"
    elif score <= -2:
        label = "Bearish"
    elif score <= -1:
        label = "Mild Bearish"
    else:
        label = "Neutral"

    return {"score": score, "label": label}


# ---------------------------------------------------------------------------
# Multi-timeframe trigger (local files only - never MT5, keeps scans fast)
# ---------------------------------------------------------------------------


def _load_trigger(
    symbol: str,
    data_dir: str,
    group: Optional[str],
    timeframe: str,
) -> Optional[pd.DataFrame]:
    """
    Load a local H4 (or H1) frame for the Buy-the-Dip momentum trigger.
    Only parquet files that already exist are used; returns None otherwise.
    """
    if timeframe.upper() == "D1":
        candidates = []
        if group:
            base = Path(data_dir) / group
            # Flat layout (asset-class folders hold all timeframes flat, e.g.
            # full_fx/EURUSD_H4.parquet).
            candidates += [base / f"{symbol}_H4.parquet", base / f"{symbol}_H1.parquet"]
            # Nested layout (e.g. group "mt5/D1" -> sibling "mt5/H4").
            candidates += [
                base / "H4" / f"{symbol}_H4.parquet",
                base / "H1" / f"{symbol}_H1.parquet",
            ]
            if "/" in group:
                top = Path(data_dir) / group.split("/", 1)[0]
                candidates += [
                    top / "H4" / f"{symbol}_H4.parquet",
                    top / "H1" / f"{symbol}_H1.parquet",
                ]
        candidates += [
            Path(data_dir) / "h4" / f"{symbol}_H4.parquet",
            Path(data_dir) / "h1" / f"{symbol}_H1.parquet",
        ]
        # On-demand symbols land in their classified group folder (e.g.
        # equity/AAPL_H4.parquet), which may differ from the requested
        # group - search every folder as the final fallback.
        local_h4 = find_local(symbol, "H4", data_dir)
        if local_h4 is not None:
            candidates.append(local_h4)
        local_h1 = find_local(symbol, "H1", data_dir)
        if local_h1 is not None:
            candidates.append(local_h1)
        for path in candidates:
            if path.exists():
                try:
                    trig = load_data(path, symbol=symbol)
                    trig = clean_data(trig)
                    return trig if not trig.empty else None
                except Exception:
                    return None
    return None


# ---------------------------------------------------------------------------
# Per-symbol scan
# ---------------------------------------------------------------------------


def scan_symbol(
    symbol: str,
    data_dir: str = "data/raw",
    group: Optional[str] = None,
    timeframe: str = "D1",
    fetch_mt5: bool = False,
) -> Dict:
    """Run the full pipeline on one symbol and return a summary row.

    Missing files are resolved on demand: local (any group folder) first,
    then the running MetaTrader 5 terminal, then Yahoo (when ``fetch_mt5``
    is enabled). Fetched files are cached into their classified group
    folder, never the ``data/raw`` root.
    """
    path = resolve_symbol_data(
        symbol, timeframe, data_dir, group, allow_mt5=fetch_mt5, allow_yahoo=fetch_mt5
    )
    # The symbol may have resolved into a different group folder than the
    # one requested (e.g. equity/AAPL_D1.parquet while group=full_fx) -
    # the effective group drives the H4/H1 trigger lookup and the model's
    # H4/cross-asset context, so follow the path, not the hint.
    eff_group = effective_group(path, data_dir, group)

    df = load_data(path, symbol=symbol)
    df = clean_data(df)
    if df.empty:
        raise ValueError(f"Symbol {symbol}: no valid rows after cleaning")
    # Compute indicators here (report.py only enriches its own local copy).
    df = add_all_indicators(df)
    df = detect_regime(df)
    # mtf=False: the D/W/M multi-timeframe table costs ~125ms of extra
    # resampling per symbol but is only rendered by the detail view - the
    # ranking row keeps the cheap rule label + KMeans cluster label.
    report = generate_full_report(
        df, symbol=symbol, group=eff_group, data_dir=data_dir, mtf=False
    )

    d = _directional_score(df)
    close = report["last_close"]
    atr = report["volatility"]["atr_14"]

    # Nearest strong support / resistance (already computed by the report).
    levels = report["levels"]
    support = levels["nearest_support"]
    resistance = levels["nearest_resistance"]

    # Buy-the-Dip confirmation (uses a local H4/H1 trigger when available).
    trigger_df = _load_trigger(symbol, data_dir, eff_group, timeframe)
    dip = detect_dip(df, trigger_df=trigger_df, levels=levels)

    # Sell-the-Rally confirmation (short-side mirror).
    rally = detect_rally(df, trigger_df=trigger_df, levels=levels)

    # Short best achievable R:R from the short target ladder.
    short_best_rr = None
    short_targets = report.get("short_targets") or {}
    if short_targets.get("best_rr") and short_targets.get("best_rr") > 0:
        short_best_rr = round(float(short_targets["best_rr"]), 2)

    # Ensemble bullish probability (graceful: None when no model is saved).
    # The model is served with the same H4 / cross-asset / COT context it
    # was trained on (see predict_series); anything missing falls back to
    # neutral features, never crashes.
    ml_prob = None
    try:
        prob = predict_series(
            df, DEFAULT_MODEL_PATH, symbol=symbol, group=eff_group, data_dir=data_dir
        )
        if prob is not None:
            last = prob.dropna()
            if len(last):
                ml_prob = round(float(last.iloc[-1]) * 100, 1)
    except Exception:
        ml_prob = None

    # Sell-the-Rally (short) probability - separate model, mirror features.
    ml_short_prob = None
    try:
        from src.model.model import predict_short_series

        prob_s = predict_short_series(
            df, symbol=symbol, group=eff_group, data_dir=data_dir
        )
        if prob_s is not None:
            last_s = prob_s.dropna()
            if len(last_s):
                ml_short_prob = round(float(last_s.iloc[-1]) * 100, 1)
    except Exception:
        ml_short_prob = None

    # Macro overlay (top-down USD / risk / rates backdrop; graceful None
    # when no local DXY or cached VIX/TNX exists). The factor scores are
    # memoized, so the whole universe scan pays the load once. df is NOT
    # passed here: the ranking row only consumes bias/gate, and computing
    # the 90-day sensitivity correlations (plus the SPX find_local lookup)
    # per symbol is wasted work on a 500-stock scan - the detail report
    # path computes sensitivities.
    macro = None
    try:
        from src.macro.overlay import macro_report_for_symbol

        macro = macro_report_for_symbol(symbol)
    except Exception:
        macro = None

    # Final quant rating + best chart pattern (report sections #14/#6).
    rating = (report.get("rating") or {}).get("rating")
    patterns = (report.get("patterns") or {}).get("patterns") or []
    best_pattern = patterns[0] if patterns else None

    # Unified trade plan: one actionable verdict per symbol (BUY/SELL-LIMIT,
    # WAIT-*, or NO-SETUP). Uses the trigger-enriched dip/rally so the action
    # matches the columns this row reports.
    plan_report = dict(report)
    plan_report["dip"] = dip
    plan_report["rally"] = rally
    plan = trade_plan(plan_report)
    action = plan["action"]

    # Direction-neutral setup classification (two-sided audit): the
    # classifier's direction + setup family surface in the ranking table so
    # a symbol can show SHORT_BREAKDOWN / LONG_BREAKOUT_RETEST etc. even
    # when the engines have no confirmed setup - direction is evidence-led,
    # not engine-gated.
    sc = report.get("setup_classification") or {}
    setup_label = sc.get("setup_family") or (
        "FLAT" if sc.get("direction") in (None, "flat") else "-"
    )

    def _level_price(level):
        if not level:
            return None
        return round(level["price"], 5)

    def _zone_text(zone):
        if not zone:
            return None
        return f"{zone[0]:.5f}-{zone[1]:.5f}"

    # Best achievable R:R from the target ladder (TP1..TP3, scaling-out
    # makes the spec's 2.5:1 floor reachable). None when no actionable setup.
    best_rr = None
    targets = report.get("targets") or {}
    if targets.get("best_rr") and targets.get("best_rr") > 0:
        best_rr = round(float(targets["best_rr"]), 2)

    return {
        "symbol": symbol,
        "date": report["last_date"],
        "close": round(close, 3),
        "action": action,
        "regime": report["regime"]["regime"],
        "bias_score": d["score"],
        "bias": d["label"],
        "adx": report["trend_strength"]["adx"],
        "rsi_14": report["momentum"]["rsi_14"],
        "macd_hist": report["momentum"]["macd_hist"],
        "vs_sma200_pct": report["regime"]["price_vs_200sma_pct"],
        "atr_pct": round((atr / close) * 100, 2) if close else 0.0,
        "support": _level_price(support),
        "resistance": _level_price(resistance),
        "dip_score": dip["dip_score"],
        "dip_confirmed": "Yes" if dip["dip_confirmed"] else "No",
        "dip_stage": dip["dip_stage"],
        "entry_zone": _zone_text(dip["entry_zone"]),
        "invalidation": dip["invalidation_level"],
        "best_rr": best_rr,
        "rally_score": rally["rally_score"],
        "rally_confirmed": "Yes" if rally["rally_confirmed"] else "No",
        "rally_stage": rally["rally_stage"],
        "short_entry_zone": _zone_text(rally["entry_zone"]),
        "short_invalidation": rally["invalidation_level"],
        "short_best_rr": short_best_rr,
        "ml_short_prob": ml_short_prob,
        "ml_prob": ml_prob,
        "macro_bias": (round(macro["bias"]["bias"], 2) if macro else None),
        "macro_label": (macro["bias"]["label"] if macro else None),
        "macro_gate": (
            "PASS" if macro["gate"]["allowed"] else "BLOCKED" if macro else None
        ),
        "macro_gate_short": (
            "PASS"
            if macro["gate_short"]["allowed"]
            else "BLOCKED"
            if macro else None
        ),
        "rating": rating,
        "pattern": (
            f"{best_pattern['name']} {best_pattern['prob']}%" if best_pattern else None
        ),
        "setup": setup_label,
        "long_evidence": sc.get("long_score"),
        "short_evidence": sc.get("short_score"),
        "setup_ev": sc.get("ev"),
    }


def scan_universe(
    data_dir: str = "data/raw",
    group: Optional[str] = None,
    timeframe: str = "D1",
    symbols: Optional[List[str]] = None,
    fetch_mt5: bool = False,
) -> pd.DataFrame:
    """Scan multiple symbols and return a ranked DataFrame."""
    if symbols is None:
        symbols = discover_symbols(data_dir, group, timeframe)

    if not symbols:
        hint = (
            ". Did you mean to pass --group (e.g. h1, h4, full_fx)?"
            if symbols is None
            else f": check that {data_dir}"
            + (f"/{group}" if group else "")
            + f" contains {timeframe} data"
        )
        raise RuntimeError(f"No {timeframe} data found. {hint}")

    rows = []
    errors = []
    for sym in symbols:
        try:
            rows.append(
                scan_symbol(sym, data_dir, group, timeframe, fetch_mt5=fetch_mt5)
            )
        except Exception as exc:  # keep the scan alive if one symbol fails
            errors.append((sym, str(exc)))

    if not rows:
        raise RuntimeError(f"No symbols could be scanned. {errors}")

    table = pd.DataFrame(rows)
    # Rank: strongest directional bias first, then trend strength (ADX)
    table = table.sort_values(["bias_score", "adx"], ascending=[False, False])
    table["rank"] = range(1, len(table) + 1)
    table = table[
        [
            "rank",
            "symbol",
            "date",
            "close",
            "action",
            "regime",
            "bias",
            "bias_score",
            "adx",
            "rsi_14",
            "macd_hist",
            "vs_sma200_pct",
            "atr_pct",
            "support",
            "resistance",
            "dip_score",
            "dip_confirmed",
            "dip_stage",
            "entry_zone",
            "invalidation",
            "best_rr",
            "ml_prob",
            "rally_score",
            "rally_confirmed",
            "rally_stage",
            "short_entry_zone",
            "short_invalidation",
            "short_best_rr",
            "ml_short_prob",
            "macro_bias",
            "macro_label",
            "macro_gate",
            "macro_gate_short",
            "rating",
            "pattern",
            "setup",
            "long_evidence",
            "short_evidence",
            "setup_ev",
        ]
    ]

    if errors:
        print(f"\n[!] {len(errors)} symbol(s) failed: {errors}", file=sys.stderr)

    return table


def print_table(table: pd.DataFrame) -> None:
    """Pretty-print the ranked universe table."""
    print("\n" + "=" * 116)
    print("NEXUSQUANT UNIVERSE SCANNER — INSTITUTIONAL RANKING")
    print("=" * 116)

    show = table.copy()
    show["macd_hist"] = show["macd_hist"].map(lambda v: f"{v:+.5f}")
    show["vs_sma200_pct"] = show["vs_sma200_pct"].map(lambda v: f"{v:+.2f}%")
    show["atr_pct"] = show["atr_pct"].map(lambda v: f"{v:.2f}%")
    show["bias_score"] = show["bias_score"].map(lambda v: f"{v:+d}")
    show["support"] = show["support"].map(lambda v: "-" if pd.isna(v) else f"{v:,.5f}")
    show["resistance"] = show["resistance"].map(
        lambda v: "-" if pd.isna(v) else f"{v:,.5f}"
    )
    show["invalidation"] = show["invalidation"].map(
        lambda v: "-" if pd.isna(v) else f"{v:,.5f}"
    )
    show["entry_zone"] = show["entry_zone"].map(lambda v: "-" if pd.isna(v) else v)
    show["ml_prob"] = show["ml_prob"].map(lambda v: "-" if pd.isna(v) else f"{v:.0f}%")
    show["ml_short_prob"] = show["ml_short_prob"].map(
        lambda v: "-" if pd.isna(v) else f"{v:.0f}%"
    )
    show["macro_bias"] = show["macro_bias"].map(
        lambda v: "-" if pd.isna(v) else f"{v:+.2f}"
    )
    show["macro_label"] = show["macro_label"].fillna("-")
    show["macro_gate"] = show["macro_gate"].fillna("-")
    if "setup" in show.columns:
        show["setup"] = show["setup"].fillna("-")
    if "long_evidence" in show.columns:
        show["long_evidence"] = show["long_evidence"].map(
            lambda v: "-" if pd.isna(v) else f"{v:.2f}"
        )
    if "short_evidence" in show.columns:
        show["short_evidence"] = show["short_evidence"].map(
            lambda v: "-" if pd.isna(v) else f"{v:.2f}"
        )
    if "setup_ev" in show.columns:
        show["setup_ev"] = show["setup_ev"].map(
            lambda v: "-" if pd.isna(v) else f"{v:+.2f}R"
        )

    print(show.to_string(index=False, col_space=0))
    print("=" * 116 + "\n")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        description="NexusQuant multi-symbol scanner & ranking",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--data-dir", default="data/raw", help="data directory")
    parser.add_argument(
        "--group",
        default=None,
        help="subdirectory to scan (full_fx, candidates, crosses, "
        "equity, equity_universe, h1, h4)",
    )
    parser.add_argument(
        "--timeframe",
        default=None,
        help="D1, H1 or H4 (default: derived from --group, else D1)",
    )
    parser.add_argument(
        "--top", type=int, default=None, help="show only the top N ranked symbols"
    )
    parser.add_argument(
        "--json", action="store_true", help="emit JSON instead of a formatted table"
    )
    parser.add_argument(
        "--symbols", nargs="*", help="explicit symbol list (overrides discovery)"
    )
    parser.add_argument(
        "--no-mt5",
        action="store_true",
        help="do not fetch missing data on demand "
        "(MT5 terminal or Yahoo); fail instead",
    )
    args = parser.parse_args(argv)

    # Derive the timeframe from the group name unless explicitly given.
    if args.timeframe is None:
        args.timeframe = {"h1": "H1", "h4": "H4"}.get(args.group, "D1")

    # After the data-folder cleanup there are no top-level files left; the
    # sensible default scan group is full_fx.
    if args.group is None and not any(
        Path(args.data_dir).glob(f"*_{args.timeframe}.parquet")
    ):
        args.group = "full_fx"
        print(
            f"[scanner] no top-level {args.timeframe} files — "
            f"defaulting to group 'full_fx'",
            file=sys.stderr,
        )

    table = scan_universe(
        data_dir=args.data_dir,
        group=args.group,
        timeframe=args.timeframe,
        symbols=args.symbols,
        fetch_mt5=not args.no_mt5,
    )

    if args.top:
        table = table.head(args.top)

    if args.json:
        # NaN -> null so the output is strict, valid JSON.
        import math

        records = table.to_dict(orient="records")
        for rec in records:
            for k, v in rec.items():
                if isinstance(v, float) and math.isnan(v):
                    rec[k] = None
        print(json.dumps(records, indent=2))
        return 0

    print_table(table)
    print(
        f"Scanned {len(table)} symbols across "
        f"{'data/raw/' + (args.group + '/' if args.group else '')}{args.timeframe}.\n"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
