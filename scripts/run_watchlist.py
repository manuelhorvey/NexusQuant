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
    ml = r.get("ml") or {}
    ms = r.get("ml_short") or {}
    long_p = ml.get("prob_pct")
    short_p = ms.get("prob_pct")
    if long_p is None and short_p is None:
        return "-"
    return (
        f"{long_p:.0f}%/{short_p:.0f}%"
        if long_p is not None and short_p is not None
        else f"{long_p or short_p:.0f}% (one-sided)"
    )


def _short_label(r: dict) -> str:
    """Short-side opportunity: family · P · EV from the opportunity book."""
    ob = r.get("opportunity_book") or {}
    so = ob.get("short") or {}
    fam = so.get("setup_family") or "-"
    p = so.get("probability")
    ev = so.get("expected_r")
    bits = [fam]
    if p is not None:
        bits.append(f"P {p:.0%}")
    if ev is not None:
        bits.append(f"EV {ev:+.2f}R")
    return " ".join(bits)


def _book_label(r: dict) -> str:
    """Opportunity-book verdict: LONG/SHORT/FLAT + EV."""
    ob = r.get("opportunity_book") or {}
    vd = ob.get("verdict") or {}
    d = vd.get("direction")
    if d is None:
        return "-"
    ev = vd.get("expected_r")
    txt = d.upper()
    if ev is not None:
        txt += f" {ev:+.2f}R"
    return txt


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
        f"{'SYMBOL':<8}{'CLOSE':>10}  {'ACTION':<26}{'REGIME (D1|MTF|CLUSTER)':<24}"
        f"{'DIP':<17}{'ML L/S':>9}  {'MACRO':<20}{'LONG (E/S/T · R)':<26}"
        f"{'SHORT (family · P · EV)':<30}{'BOOK':<14}{'SUP':>12}{'RES':>12}  "
        f"{'PATTERN':<20}{'RATING':<14}{'STRESS':<12}"
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
            f"{_action_label(r):<26}{_regime_label(r):<24}{_dip_label(r):<17}"
            f"{_ml_label(r):>9}  {_macro_label(r):<20}"
            f"{_setup_label(r):<26}{_short_label(r):<30}{_book_label(r):<14}"
            f"{_level_label(r, 'support'):>12}{_level_label(r, 'resistance'):>12}  "
            f"{_pattern_label(r):<20}{rating:<14}{_stress_label(r):<12}"
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
