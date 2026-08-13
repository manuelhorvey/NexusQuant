"""
NexusQuant - Historical Opportunity Census (two-sided audit)

Measures whether the engine is *equally capable* of discovering long and
short opportunities across history (not equal trade frequency - equal
opportunity detection). For each symbol it walks every bar and:

1. Classifies the bar into a setup family (direction-neutral classifier).
2. Labels the realized outcome with the causal backtest engine (dip /
   rally signal series + first-touch R resolution, stop-first logic).
3. Counts candidates, confirmed signals, and realized recalls per side.

Outputs a per-side table:

    candidates / signals / win rate / expectancy / per-family counts

and the headline metric:

    Opportunity Recall(long) vs Opportunity Recall(short)

so the audit can answer: "when the market offered a good long/short,
did the engine see it?"

Causality: the classifier reads only causal indicators (rolling windows,
confirmed pivots, no future bars); the realized labels come from the
same causal signal series the backtester uses (entry on the NEXT bar,
stop-first, no lookahead). The census is deterministic (fixed seeds).
"""

from __future__ import annotations

import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

sys.path.append(str(Path(__file__).parent.parent.parent))

from src.data.loader import clean_data, load_data
from src.features.indicators import add_all_indicators
from src.features.regime import detect_regime
from src.features.setups import classify_setup, LONG_FAMILIES

MIN_BARS = 260  # warm-up: SMA200 + swings + a working window


def _as_bool(v) -> bool:
    if isinstance(v, bool):
        return v
    if isinstance(v, str):
        return v.strip().lower() in ("yes", "true", "1")
    return bool(v)


def _classify_history(df: pd.DataFrame, step: int = 3) -> pd.DataFrame:
    """Per-bar setup classification (causal by construction).

    Returns a frame with ``setup_family``, ``direction``, ``long_score``,
    ``short_score`` for each classified bar. Every bar is classified on a
    window that ends at that bar, and - critically - the precomputed
    structural enrichments (levels / divergences / patterns of the FULL
    frame) are deliberately NOT passed in: they would leak future
    information into early bars. The family scoring therefore uses only
    the window's own rolling indicators (trend / momentum / structure /
    volume), which are causal. ``step`` subsamples for speed; the
    classification is per-bar deterministic so the subsample is unbiased.
    """
    rows = []
    for i in range(MIN_BARS, len(df), step):
        window = df.iloc[: i + 1]
        sc = classify_setup(window, levels=None, divergence=None, pattern=None)
        rows.append(
            {
                "date": window.index[-1],
                "setup_family": sc.get("setup_family"),
                "direction": sc.get("direction"),
                "long_score": sc.get("long_score", 0.0),
                "short_score": sc.get("short_score", 0.0),
            }
        )
    return pd.DataFrame(rows).set_index("date")


# First-touch resolution window after the signal bar.
MAX_HOLD_BARS = 20


def _realized_r(sig_frame: pd.DataFrame, df: pd.DataFrame, side: str) -> pd.Series:
    """First-touch R for each signal bar (causal, stop-first).

    Uses the signal series' own entry/stop/target columns (already causal)
    and resolves first touch over the next ``MAX_HOLD_BARS`` bars: if the
    stop is touched before the target within the window the trade is -1R,
    if the target is touched first it is +R, otherwise the trade is
    unresolved (NaN - not counted as a win or loss). This mirrors the
    backtest engine's resolution semantics over a bounded hold.
    """
    out = pd.Series(np.nan, index=df.index)
    for i in range(len(df) - 1):
        if not _as_bool(sig_frame["confirmed"].iloc[i]):
            continue
        entry = sig_frame["entry_hi" if side == "short" else "entry_lo"].iloc[i]
        stop = sig_frame["invalidation"].iloc[i]
        # The dip series names the target column "resistance"; the rally
        # series uses "target" - handle both.
        tgt_col = "target" if "target" in sig_frame.columns else "resistance"
        tgt = sig_frame[tgt_col].iloc[i]
        e = float(entry)
        s = float(stop)
        t = float(tgt)
        if e != e or s != s or t != t or e <= 0 or s <= 0:
            continue
        risk = abs(e - s)
        if risk <= 0:
            continue
        win = loss = False
        horizon = df.iloc[i + 1 : i + 1 + MAX_HOLD_BARS]
        for _, bar in horizon.iterrows():
            hi, lo = float(bar["high"]), float(bar["low"])
            if side == "long":
                if lo <= s:
                    loss = True
                    break
                if hi >= t:
                    win = True
                    break
            else:
                if hi >= s:
                    loss = True
                    break
                if lo <= t:
                    win = True
                    break
        if win:
            out.iloc[i] = (t - e) / risk if side == "long" else (e - t) / risk
        elif loss:
            out.iloc[i] = -1.0
        # else: unresolved within the window -> NaN (excluded from stats)
    return out


def _uniform_r(df: pd.DataFrame, side: str) -> pd.Series:
    """Symmetric 1R win/loss outcome for EVERY bar, causal by construction.

    This is the same triple-barrier geometry the ML models train on
    (``build_labels`` / ``build_labels_short`` in ``src/model/features.py``):
    entry = close, stop = close - stop_mult*ATR(14) (long) / + (short),
    target = close + target_mult*ATR(14) (long) / - (short). The stop is
    checked before the target over the next ``horizon`` bars; unresolved
    rows are NaN.

    Unlike ``_realized_r`` (which only resolves bars the pullback engines
    CONFIRM), this validates every classified family - so the census can
    report outcome parity for all 12 families, not just the engine's
    BUY_DIP / SELL_RALLY pair. Causality: barriers use only the bar's own
    ATR; resolution reads only forward bars.
    """
    from src.model.features import DEFAULT_HORIZON, DEFAULT_STOP_MULT, DEFAULT_TARGET_MULT

    close = df["close"].astype(float)
    high = df["high"].astype(float)
    low = df["low"].astype(float)
    atr = df["atr_14"].astype(float)

    stop = close - DEFAULT_STOP_MULT * atr if side == "long" else close + DEFAULT_STOP_MULT * atr
    target = (
        close + DEFAULT_TARGET_MULT * atr
        if side == "long"
        else close - DEFAULT_TARGET_MULT * atr
    )

    out = pd.Series(np.nan, index=df.index)
    for i in range(len(df) - 1):
        s, t = stop.iloc[i], target.iloc[i]
        if s != s or t != t:
            continue
        win = loss = False
        horizon = df.iloc[i + 1 : i + 1 + DEFAULT_HORIZON]
        for _, bar in horizon.iterrows():
            hi, lo = float(bar["high"]), float(bar["low"])
            if side == "long":
                if lo <= s:
                    loss = True
                    break
                if hi >= t:
                    win = True
                    break
            else:
                if hi >= s:
                    loss = True
                    break
                if lo <= t:
                    win = True
                    break
        if win:
            out.iloc[i] = DEFAULT_TARGET_MULT / DEFAULT_STOP_MULT
        elif loss:
            out.iloc[i] = -1.0
    return out


def opportunity_census(
    symbols: List[str],
    data_dir: str = "data/raw",
    group: str = "full_fx",
    timeframe: str = "D1",
    min_family_score: float = 0.45,
    show_symbols: int = 6,
) -> Dict:
    """Run the census across symbols; return per-side + per-family stats."""
    from src.backtest.signals import dip_signal_series, rally_signal_series

    stats = {
        "side": {"long": Counter(), "short": Counter()},
        "family": Counter(),
        "family_uniform": Counter(),
        "by_symbol": defaultdict(lambda: {"long": Counter(), "short": Counter()}),
        "n_symbols": 0,
        "bars": 0,
    }
    failures: List[str] = []

    for sym in symbols:
        try:
            path = (
                Path(data_dir) / group / f"{sym}_{timeframe.upper()}.parquet"
                if group
                else Path(data_dir) / f"{sym}_{timeframe.upper()}.parquet"
            )
            if not path.exists():
                failures.append(f"{sym}: no parquet")
                continue
            df = clean_data(load_data(path, symbol=sym))
            if len(df) < MIN_BARS + 50:
                failures.append(f"{sym}: too short ({len(df)} bars)")
                continue
            df = add_all_indicators(df)
            df = detect_regime(df)
            stats["n_symbols"] += 1
            stats["bars"] += len(df)

            hist = _classify_history(df)
            # Causal signal series from the backtest engine (aligned to df).
            sig_long = dip_signal_series(df)
            sig_short = rally_signal_series(df)
            r_long = _realized_r(sig_long, df, "long")
            r_short = _realized_r(sig_short, df, "short")
            # Uniform 1R outcome for EVERY classified bar (all families).
            u_long = _uniform_r(df, "long")
            u_short = _uniform_r(df, "short")

            for date, row in hist.iterrows():
                fam = row["setup_family"]
                if fam is None:
                    continue
                side = "long" if fam in LONG_FAMILIES else "short"
                score = row["long_score"] if side == "long" else row["short_score"]
                if score < min_family_score:
                    continue
                stats["family"][fam] += 1
                stats["side"][side]["candidates"] += 1
                stats["by_symbol"][sym][side]["candidates"] += 1
                i = df.index.get_loc(date)
                if i is None or i >= len(df):
                    continue
                # Uniform 1R outcome (all families): resolve EVERY classified
                # candidate with the symmetric causal geometry so the 10
                # non-pullback families are validated too, not just
                # engine-confirmed bars (the engine path below is the
                # deployment-objective measurement for pullback families).
                u = u_long.iloc[i] if side == "long" else u_short.iloc[i]
                if u == u:
                    stats["family_uniform"][fam + "_RESOLVED"] += 1
                    if u > 0:
                        stats["family_uniform"][fam + "_WIN"] += 1
                    else:
                        stats["family_uniform"][fam + "_LOSS"] += 1
                # Realized label: a family candidate is "validated" when a
                # confirmed engine signal fires on the same bar AND the
                # causal first-touch label wins.
                rv = r_long.iloc[i] if side == "long" else r_short.iloc[i]
                sig = (
                    sig_long["confirmed"] if side == "long" else sig_short["confirmed"]
                )
                if not _as_bool(sig.iloc[i]) or rv != rv:
                    continue
                stats["side"][side]["signals"] += 1
                stats["by_symbol"][sym][side]["signals"] += 1
                if rv > 0:
                    stats["side"][side]["wins"] += 1
                    stats["side"][side]["sum_r"] += float(rv)
                    stats["by_symbol"][sym][side]["wins"] += 1
                else:
                    stats["side"][side]["losses"] += 1
                    stats["side"][side]["sum_r"] += -1.0
                    stats["by_symbol"][sym][side]["losses"] += 1
        except Exception as exc:
            failures.append(f"{sym}: {exc}")

    stats["failures"] = failures
    stats["recall"] = _recall_stats(stats)
    return stats


def _recall_stats(stats: Dict) -> Dict:
    out: Dict[str, Dict] = {}
    for side in ("long", "short"):
        c = stats["side"][side]
        signals = c["signals"]
        wins = c["wins"]
        losses = c["losses"]
        out[side] = {
            "candidates": c["candidates"],
            "signals": signals,
            "wins": wins,
            "losses": losses,
            "win_rate": round(wins / signals, 3) if signals else None,
            "expectancy_r": (round(c["sum_r"] / signals, 3) if signals else None),
            "signal_rate_pct": (round(100 * signals / max(c["candidates"], 1), 1)),
        }
    return out


def print_census(stats: Dict, symbols_shown: int = 6) -> None:
    print("\n" + "=" * 72)
    print("HISTORICAL OPPORTUNITY CENSUS — LONG vs SHORT")
    print("=" * 72)
    print(f"Symbols: {stats['n_symbols']} · bars: {stats['bars']:,}")

    for side in ("long", "short"):
        r = stats["recall"][side]
        print(
            f"\n[{side.upper()}] candidates {r['candidates']:,} · "
            f"confirmed signals {r['signals']:,} · "
            f"signal rate {r['signal_rate_pct']}% · "
            f"win rate {('%.1f%%' % (100 * r['win_rate'])) if r['win_rate'] is not None else '-'} · "
            f"expectancy {('%.3fR' % r['expectancy_r']) if r['expectancy_r'] is not None else '-'}"
        )

    print("\nPer-family candidate counts:")
    for fam, n in sorted(stats["family"].items(), key=lambda kv: -kv[1]):
        print(f"  {fam:<28} {n:>6,}")

    print("\nPer-family uniform 1R outcome (all families, causal geometry):")
    fams = sorted(stats["family"], key=lambda f: -stats["family"][f])
    print(f"{'FAMILY':<28}{'cand':>8}{'win':>7}{'loss':>7}{'winrate':>9}")
    for fam in fams:
        n = stats["family"][fam]
        wins = stats["family_uniform"].get(fam + "_WIN", 0)
        losses = stats["family_uniform"].get(fam + "_LOSS", 0)
        rate = f"{100 * wins / max(wins + losses, 1):.1f}%" if wins + losses else "-"
        print(f"{fam:<28}{n:>8,}{wins:>7,}{losses:>7,}{rate:>9}")

    print("\nPer-symbol side mix (top symbols by total candidates):")
    rows = []
    for sym, s in stats["by_symbol"].items():
        rows.append(
            (
                sym,
                s["long"]["candidates"],
                s["short"]["candidates"],
                s["long"]["signals"],
                s["short"]["signals"],
                s["long"]["wins"],
                s["short"]["wins"],
            )
        )
    rows.sort(key=lambda r: -(r[1] + r[2]))
    print(
        f"{'SYMBOL':<8}{'L cand':>8}{'S cand':>8}{'L sig':>6}{'S sig':>6}"
        f"{'L win':>6}{'S win':>6}"
    )
    for row in rows[:symbols_shown]:
        print(
            f"{row[0]:<8}{row[1]:>8,}{row[2]:>8,}{row[3]:>6,}{row[4]:>6,}"
            f"{row[5]:>6,}{row[6]:>6,}"
        )

    if stats.get("failures"):
        print(
            f"\n[{len(stats['failures'])} symbol(s) skipped]: "
            f"{', '.join(stats['failures'][:4])}..."
        )

    lr, sr = stats["recall"]["long"], stats["recall"]["short"]
    if lr["signals"] and sr["signals"]:
        ratio = lr["signals"] / sr["signals"]
        print(
            f"\nLong/short signal ratio: {ratio:.2f} (1.0 = equal opportunity "
            f"detection; ≠1 is evidence, not bias, unless one side can't "
            f"enter the pipeline at all)"
        )
    print("=" * 72 + "\n")


def main(argv: Optional[List[str]] = None) -> int:
    import argparse

    parser = argparse.ArgumentParser(
        description="Historical long/short opportunity census (two-sided audit)"
    )
    parser.add_argument("--group", default="full_fx")
    parser.add_argument("--timeframe", default="D1")
    parser.add_argument("--symbols", default=None, help="comma-separated")
    parser.add_argument("--min-score", type=float, default=0.45)
    parser.add_argument("--show", type=int, default=6)
    args = parser.parse_args(argv)

    if args.symbols:
        symbols = [s.strip() for s in args.symbols.split(",") if s.strip()]
    else:
        from src.analysis.scanner import discover_symbols

        symbols = discover_symbols(
            "data/raw", group=args.group, timeframe=args.timeframe
        )
        symbols = symbols[:12]  # keep the census fast; --symbols for full runs

    stats = opportunity_census(
        symbols,
        group=args.group,
        timeframe=args.timeframe,
        min_family_score=args.min_score,
    )
    print_census(stats, symbols_shown=args.show)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
