"""
NexusQuant - Watchlist full-report runner.

Loads each symbol's D1 frame, runs the complete institutional pipeline
(generate_full_report), prints a consolidated decision table, and writes
the full 18-section reports to ``data/reports/watchlist_<date>.txt``.

Usage:
    python scripts/run_watchlist.py
    python scripts/run_watchlist.py EURUSD XAUUSD
"""

import sys
import io
import contextlib
from datetime import datetime
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent))

from src.data.loader import clean_data, load_data
from src.features.indicators import add_all_indicators
from src.features.regime import detect_regime
from src.analysis.report import generate_full_report, print_report

DEFAULT_WATCHLIST = [
    "EURUSD",
    "GBPUSD",
    "USDJPY",
    "USDCHF",
    "AUDUSD",
    "NZDUSD",
    "USDCAD",
    "EURJPY",
    "GBPJPY",
    "NZDJPY",
    "AUDJPY",
    "CADJPY",
    "XAUUSD",
]


def _regime_label(r: dict) -> str:
    rg = r.get("regime") or {}
    d1 = rg.get("regime") or rg.get("label") or "-"
    cons = rg.get("mtf_consensus") or "-"
    cluster = rg.get("regime_cluster") or "-"
    return f"{d1} | MTF {cons} | {cluster}"


def _dip_label(r: dict) -> str:
    d = r.get("dip") or {}
    stage = d.get("dip_stage") or "-"
    score = d.get("dip_score")
    # Avoid "Confirmed CONFIRMED" when the stage name already says it.
    if d.get("dip_confirmed") and "confirmed" not in str(stage).lower():
        stage = f"{stage} CONFIRMED"
    return f"{stage} ({score}/8)" if score is not None else stage or "-"


def _ml_label(r: dict) -> str:
    ml = r.get("ml")
    if not ml or ml.get("prob_pct") is None:
        return "-"
    return f"{ml['prob_pct']:.0f}% {ml.get('label', '')}".strip()


def _setup_label(r: dict) -> str:
    rk = r.get("risk") or {}
    s = rk.get("setup")
    if not s:
        return f"none ({rk.get('reason', '')[:28]})"
    rr = s.get("rr", "-")
    floor = "OK" if s.get("rr_ok") else "BELOW"
    return (
        f"E {s['entry']:.4f} / S {s['stop']:.4f} / T {s['target']:.4f} "
        f"| R {rr} ({floor} {s.get('min_rr')})"
    )


def _action_label(r: dict) -> str:
    """The unified trade-plan verdict (BUY/SELL-LIMIT · WAIT-* · NO-SETUP)."""
    plan = r.get("plan") or {}
    return plan.get("action") or "-"


def _macro_label(r: dict) -> str:
    mc = r.get("macro") or {}
    rg = mc.get("regime") or {}
    bs = mc.get("bias") or {}
    gt = mc.get("gate") or {}
    comp = rg.get("composite") or "-"
    gate = "PASS" if gt.get("allowed") else "BLOCK"
    return f"{comp} | {bs.get('bias', '-')} | {gate}"


def _pattern_label(r: dict) -> str:
    pt = r.get("patterns") or {}
    pats = pt.get("patterns") or []
    if not pats:
        return "-"
    return "; ".join(f"{p['name']} {p['prob']}%" for p in pats[:2])


def _level_label(r: dict, kind: str) -> str:
    lv = r.get("levels") or {}
    lvl = lv.get(f"nearest_{kind}") or {}
    if not lvl:
        return "-"
    return f"{lvl.get('price', '-')} ({lvl.get('score', '-')})"


def _stress_label(r: dict) -> str:
    st = r.get("stress") or {}
    if not st.get("available"):
        return "-"
    worst = max(
        st.get("scenarios", []), key=lambda x: x.get("drawdown_pct", 0), default=None
    )
    if not worst:
        return "-"
    return f"{worst['scenario']} {worst['drawdown_pct']:.1f}%"


def main(argv=None) -> int:
    import argparse

    parser = argparse.ArgumentParser(
        description="Run the full institutional report for a watchlist "
        "(default: 13-symbol FX/metals watchlist)."
    )
    parser.add_argument(
        "symbols", nargs="*", help="symbols to analyse (default: FX watchlist)"
    )
    parser.add_argument(
        "--group",
        default="full_fx",
        help="data group folder under data/raw (default: full_fx)",
    )
    parser.add_argument(
        "--timeframe", default="D1", help="timeframe suffix (default: D1)"
    )
    args = parser.parse_args(argv)

    symbols = args.symbols or DEFAULT_WATCHLIST
    print("=" * 132)
    print(
        f"NEXUSQUANT FULL INSTITUTIONAL RUN — {datetime.now():%Y-%m-%d %H:%M} "
        f"| {len(symbols)} symbols | {args.group}/{args.timeframe}"
    )
    print("=" * 132)

    hdr = (
        f"{'SYMBOL':<8}{'CLOSE':>10}  {'ACTION':<26}{'REGIME (D1|MTF|CLUSTER)':<28}"
        f"{'DIP':<18}{'ML':>8}  {'MACRO':<22}{'SETUP (E/S/T · R)':<40}"
        f"{'SUP':>14}{'RES':>14}  {'PATTERN':<22}{'RATING':<16}{'STRESS':<12}"
    )
    print(hdr)
    print("-" * 132)

    full = []
    table_rows = []
    for sym in symbols:
        try:
            df = clean_data(
                load_data(
                    f"data/raw/{args.group}/{sym}_{args.timeframe}.parquet", symbol=sym
                )
            )
            df = add_all_indicators(df)
            df = detect_regime(df)
            r = generate_full_report(df, symbol=sym, group=args.group)
        except Exception as exc:
            print(f"{sym:<8}  ERROR: {exc}")
            continue

        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            print_report(r)
        full.append(buf.getvalue())

        rt = r.get("rating") or {}
        rating = f"{rt.get('prob_pct', '-')}% {rt.get('rating', '')}".strip()
        row = (
            f"{sym:<8}{r['last_close']:>10,.3f}  "
            f"{_action_label(r):<26}{_regime_label(r):<28}{_dip_label(r):<18}"
            f"{_ml_label(r):>8}  {_macro_label(r):<22}"
            f"{_setup_label(r):<40}{_level_label(r, 'support'):>14}"
            f"{_level_label(r, 'resistance'):>14}  "
            f"{_pattern_label(r):<22}{rating:<16}{_stress_label(r):<12}"
        )
        table_rows.append(row)
        print(row)
    print("-" * 132)

    out_dir = Path("data/reports")
    out_dir.mkdir(parents=True, exist_ok=True)
    out = out_dir / f"watchlist_{datetime.now():%Y%m%d_%H%M}.txt"
    with open(out, "w") as fh:
        fh.write(
            f"NEXUSQUANT FULL INSTITUTIONAL REPORTS — {datetime.now():%Y-%m-%d %H:%M}\n"
        )
        fh.write(f"Symbols: {', '.join(symbols)}\n\n")
        fh.write("CONSOLIDATED DECISION TABLE\n")
        fh.write(hdr + "\n")
        fh.write("-" * 132 + "\n")
        for row in table_rows:
            fh.write(row + "\n")
        fh.write("-" * 132 + "\n\n")
        for block in full:
            fh.write(block)
    print(f"\nFull 18-section reports saved to: {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
