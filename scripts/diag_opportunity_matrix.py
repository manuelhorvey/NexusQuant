"""Forensic diagnostic (Phase 3/4): per-symbol LONG vs SHORT opportunity matrix.

Runs the exact pipeline run_watchlist.py uses (generate_full_report) and
prints, per symbol: plan action, classifier long/short candidates with
probability/EV/R:R, and the opportunity-book verdict. Pure read-only -
no production behavior is changed.
"""

import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent))

from src.data.loader import clean_data, load_data
from src.features.indicators import add_all_indicators
from src.features.regime import detect_regime
from src.analysis.report import generate_full_report

SYMBOLS = [
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


def fmt(v, suffix=""):
    if v is None:
        return "-"
    try:
        return (
            f"{v:+.2f}{suffix}"
            if isinstance(v, float) and suffix == "R"
            else f"{v:.2f}{suffix}"
        )
    except (TypeError, ValueError):
        return str(v)


def main() -> int:
    print("=" * 160)
    print("FORENSIC OPPORTUNITY MATRIX (Phase 3/4) — same pipeline as run_watchlist.py")
    print("=" * 160)
    hdr = (
        f"{'SYM':<8}{'CLOSE':>10}  {'PLAN ACTION':<30}{'CLASSIFIER':<10}"
        f"{'LONG fam':<26}{'L score':>7}{'L P':>6}{'L EV':>7}{'L RR':>6}  "
        f"{'SHORT fam':<26}{'S score':>7}{'S P':>6}{'S EV':>7}{'S RR':>6}  {'BOOK':<30}"
    )
    print(hdr)
    print("-" * 160)
    for sym in SYMBOLS:
        try:
            df = clean_data(load_data(f"data/raw/full_fx/{sym}_D1.parquet", symbol=sym))
            df = add_all_indicators(df)
            df = detect_regime(df)
            r = generate_full_report(df, symbol=sym, group="full_fx")
        except Exception as exc:
            print(f"{sym:<8}  ERROR: {exc}")
            continue

        plan = r.get("plan") or {}
        sc = r.get("setup_classification") or {}
        ob = r.get("opportunity_book") or {}
        lo = ob.get("long") or {}
        so = ob.get("short") or {}
        vd = ob.get("verdict") or {}

        long_fam = lo.get("setup_family") or "-"
        short_fam = so.get("setup_family") or "-"
        book = f"{vd.get('direction', '-').upper()} ({vd.get('status', '-')})"
        if vd.get("expected_r") is not None:
            book += f" EV {vd['expected_r']:+.2f}R"
        book = book[:30]
        print(
            f"{sym:<8}{r.get('last_close', 0):>10,.3f}  {plan.get('action', '-'):<30}"
            f"{sc.get('direction', '-'):<10}"
            f"{long_fam:<26}{fmt(lo.get('family_score')):>7}{fmt(lo.get('probability') * 100):>5}%"
            f"{fmt(lo.get('expected_r'), 'R'):>7}{fmt(lo.get('rr')):>6}  "
            f"{short_fam:<26}{fmt(so.get('family_score')):>7}{fmt(so.get('probability') * 100):>5}%"
            f"{fmt(so.get('expected_r'), 'R'):>7}{fmt(so.get('rr')):>6}  {book}"
        )
    print("-" * 160)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
