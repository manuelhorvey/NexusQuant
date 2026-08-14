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

# Multi-barrier rungs in R (risk multiples) for the target-level EV model:
# the ladder's TP1/TP2/TP3 at 1R/2R/3R, stop at -1R.
RUNG_RR = (1.0, 2.0, 3.0)


# ---------------------------------------------------------------------------
# Multi-barrier target-level resolution (Stage-2 validation, spec #4)
# ---------------------------------------------------------------------------


def _multi_barrier_outcome(
    df: pd.DataFrame,
    side: str,
    horizon: int = MAX_HOLD_BARS,
    rungs: tuple = RUNG_RR,
) -> pd.Series:
    """First-touch outcome among {SL, TP1, TP2, TP3} for EVERY bar.

    Uniform causal geometry (same as the ML labels): stop at
    ``DEFAULT_STOP_MULT x ATR``, rungs at ``k x STOP_MULT x ATR`` (so TP_k
    sits at kR). For each bar the forward walk checks the stop first, then
    the highest rung touched that bar. Returns a Series of
    ``"sl"/"tp1"/"tp2"/"tp3"`` or NaN when nothing resolves within the
    horizon. This is the empirical P(TP_k before SL) table that feeds the
    target-level expected value model.
    """
    from src.model.features import DEFAULT_STOP_MULT

    close = df["close"].astype(float)
    atr = df["atr_14"].astype(float)
    risk = DEFAULT_STOP_MULT * atr  # 1R in price
    stop = close - risk if side == "long" else close + risk
    names = ["tp1", "tp2", "tp3"]
    rung_prices = {
        nm: (close + k * risk if side == "long" else close - k * risk)
        for nm, k in zip(names, rungs, strict=True)
    }
    out = pd.Series(np.nan, index=df.index, dtype=object)
    for i in range(len(df) - 1):
        s = stop.iloc[i]
        if s != s:
            continue
        tp = {nm: rung_prices[nm].iloc[i] for nm in names}
        result = None
        for _, bar in df.iloc[i + 1 : i + 1 + horizon].iterrows():
            hi, lo = float(bar["high"]), float(bar["low"])
            if side == "long":
                if lo <= s:
                    result = "sl"
                    break
                if hi >= tp["tp3"]:
                    result = "tp3"
                    break
                if hi >= tp["tp2"]:
                    result = "tp2"
                    break
                if hi >= tp["tp1"]:
                    result = "tp1"
                    break
            else:
                if hi >= s:
                    result = "sl"
                    break
                if lo <= tp["tp3"]:
                    result = "tp3"
                    break
                if lo <= tp["tp2"]:
                    result = "tp2"
                    break
                if lo <= tp["tp1"]:
                    result = "tp1"
                    break
        if result is not None:
            out.iloc[i] = result
    return out


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
    from src.model.features import (
        DEFAULT_HORIZON,
        DEFAULT_STOP_MULT,
        DEFAULT_TARGET_MULT,
    )

    close = df["close"].astype(float)
    atr = df["atr_14"].astype(float)

    stop = (
        close - DEFAULT_STOP_MULT * atr
        if side == "long"
        else close + DEFAULT_STOP_MULT * atr
    )
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
        # Stage-2 validation: multi-barrier first-touch outcomes and regime
        # conditioning over EVERY classified candidate (all 12 families).
        "outcome_side": Counter(),  # (side, outcome)  outcome in sl/tp1/tp2/tp3
        "outcome_regime": Counter(),  # (side, regime, outcome)
        "outcome_family": Counter(),  # (family, outcome)
        "regime_side": defaultdict(Counter),  # (side, regime) -> field counts
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
            # Multi-barrier first-touch outcome for EVERY bar (target-level
            # EV table: P(TP1/TP2/TP3 before SL)).
            mb_long = _multi_barrier_outcome(df, "long")
            mb_short = _multi_barrier_outcome(df, "short")
            regimes = df["regime"] if "regime" in df.columns else None

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
                # Multi-barrier target-level outcome + regime conditioning.
                mb = mb_long.iloc[i] if side == "long" else mb_short.iloc[i]
                if mb == mb:
                    stats["outcome_side"][(side, mb)] += 1
                    stats["outcome_family"][(fam, mb)] += 1
                    if regimes is not None:
                        reg = str(regimes.iloc[i])
                        stats["outcome_regime"][(side, reg, mb)] += 1
                # Regime-conditional signal/win bookkeeping.
                if regimes is not None:
                    reg = str(regimes.iloc[i])
                    stats["regime_side"][(side, reg)]["candidates"] += 1
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
                if regimes is not None:
                    reg = str(regimes.iloc[i])
                    stats["regime_side"][(side, reg)]["signals"] += 1
                if rv > 0:
                    stats["side"][side]["wins"] += 1
                    stats["side"][side]["sum_r"] += float(rv)
                    stats["by_symbol"][sym][side]["wins"] += 1
                    if regimes is not None:
                        reg = str(regimes.iloc[i])
                        stats["regime_side"][(side, reg)]["wins"] += 1
                else:
                    stats["side"][side]["losses"] += 1
                    stats["side"][side]["sum_r"] += -1.0
                    stats["by_symbol"][sym][side]["losses"] += 1
                    if regimes is not None:
                        reg = str(regimes.iloc[i])
                        stats["regime_side"][(side, reg)]["losses"] += 1
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


def _target_level_stats(stats: Dict, costs: tuple = (0.0, 0.05, 0.10, 0.15)) -> Dict:
    """Per-side empirical P(TP_k before SL) + target-level EV per cost.

    The outcome distribution comes from the multi-barrier resolver over
    EVERY classified candidate (all families, uniform causal geometry):

        P(tp1), P(tp2), P(tp3), P(sl), P(none)

    Target-level EV (ladder at 1R/2R/3R, stop at -1R, none = 0):

        EV = P(tp1)*1 + P(tp2)*2 + P(tp3)*3 - P(sl) - cost

    ``costs`` sweeps round-trip cost in R so the report shows whether the
    directional edge survives realistic execution costs (spec #12).
    """
    out: Dict[str, Dict] = {}
    for side in ("long", "short"):
        c = stats["outcome_side"]
        n1 = c.get((side, "tp1"), 0)
        n2 = c.get((side, "tp2"), 0)
        n3 = c.get((side, "tp3"), 0)
        nsl = c.get((side, "sl"), 0)
        total = n1 + n2 + n3 + nsl
        if not total:
            out[side] = {"n": 0}
            continue
        p1, p2, p3, psl = n1 / total, n2 / total, n3 / total, nsl / total
        pnone = 1.0 - p1 - p2 - p3 - psl
        evs = {
            f"ev_{int(c * 100):02d}": round(
                p1 * RUNG_RR[0] + p2 * RUNG_RR[1] + p3 * RUNG_RR[2] - psl - c, 4
            )
            for c in costs
        }
        out[side] = {
            "n": total,
            "p_tp1": round(p1, 4),
            "p_tp2": round(p2, 4),
            "p_tp3": round(p3, 4),
            "p_sl": round(psl, 4),
            "p_none": round(pnone, 4),
            "ev": evs,
        }
    return out


def target_probs_json(stats: Dict) -> Dict:
    """The empirical per-side TP distribution as a JSON-safe table.

    Written by ``--write-probs`` and consumed by the live opportunity book
    (``build_opportunity_book(tp_probs=...)``) so its EV uses the
    target-level distribution instead of the ladder-best approximation.
    """
    tl = _target_level_stats(stats)
    return {
        "long": {
            "tp1": tl["long"].get("p_tp1"),
            "tp2": tl["long"].get("p_tp2"),
            "tp3": tl["long"].get("p_tp3"),
            "sl": tl["long"].get("p_sl"),
        },
        "short": {
            "tp1": tl["short"].get("p_tp1"),
            "tp2": tl["short"].get("p_tp2"),
            "tp3": tl["short"].get("p_tp3"),
            "sl": tl["short"].get("p_sl"),
        },
        "rungs_rr": list(RUNG_RR),
        "n": {s: tl[s].get("n") for s in ("long", "short")},
    }


def model_calibration(
    symbols: List[str],
    data_dir: str = "data/raw",
    group: str = "full_fx",
    timeframe: str = "D1",
    max_symbols: int = 12,
) -> Dict:
    """Independent LONG and SHORT model calibration (Stage-2, spec #5).

    For every bar with a model prediction and a resolved uniform outcome
    (the same triple-barrier geometry the labels were built with), pairs
    (predicted p, realized y in {0,1}) are pooled across symbols and
    scored per side: n, mean predicted, actual hit rate, Brier score and
    Expected Calibration Error over 10 equal-width buckets, plus the
    reliability table. This does NOT assume P(short) = 1 - P(long) - each
    side's model is validated on its own predictions and outcomes.
    """
    from src.model.model import predict_series, predict_short_series

    pooled: Dict[str, List[tuple]] = {"long": [], "short": []}
    used_symbols = 0
    failures: List[str] = []
    for sym in symbols[:max_symbols]:
        try:
            path = (
                Path(data_dir) / group / f"{sym}_{timeframe.upper()}.parquet"
                if group
                else Path(data_dir) / f"{sym}_{timeframe.upper()}.parquet"
            )
            if not path.exists():
                continue
            df = clean_data(load_data(path, symbol=sym))
            if len(df) < MIN_BARS + 50:
                continue
            df = add_all_indicators(df)
            df = detect_regime(df)
            u_long = _uniform_r(df, "long")
            u_short = _uniform_r(df, "short")
            p_long = predict_series(df, symbol=sym, group=group, data_dir=data_dir)
            p_short = predict_short_series(
                df, symbol=sym, group=group, data_dir=data_dir
            )
            for side, p_ser, u_ser in (
                ("long", p_long, u_long),
                ("short", p_short, u_short),
            ):
                if p_ser is None:
                    continue
                p = p_ser.reindex(df.index)
                for i in range(MIN_BARS, len(df)):
                    pi, ui = p.iloc[i], u_ser.iloc[i]
                    if pi != pi or ui != ui:
                        continue
                    pooled[side].append((float(pi), 1.0 if ui > 0 else 0.0))
            used_symbols += 1
        except Exception as exc:
            failures.append(f"{sym}: {exc}")

    out: Dict[str, Dict] = {}
    for side in ("long", "short"):
        pairs = pooled[side]
        if not pairs:
            out[side] = {"n": 0}
            continue
        ps = np.array([p for p, _ in pairs])
        ys = np.array([y for _, y in pairs])
        brier = float(np.mean((ps - ys) ** 2))
        # Equal-width reliability buckets over the observed prediction range.
        lo, hi = float(ps.min()), float(ps.max())
        if hi - lo < 1e-9:
            hi = lo + 1e-9
        edges = np.linspace(lo, hi, 11)
        buckets = []
        ece = 0.0
        for k in range(10):
            m = (ps >= edges[k]) & (ps < edges[k + 1])
            if k == 9:
                m = ps >= edges[9]
            nk = int(m.sum())
            if nk == 0:
                continue
            mean_p = float(ps[m].mean())
            actual = float(ys[m].mean())
            buckets.append(
                {
                    "bucket": f"{edges[k]:.3f}-{edges[k + 1]:.3f}",
                    "n": nk,
                    "mean_p": round(mean_p, 3),
                    "actual": round(actual, 3),
                }
            )
            ece += (nk / len(ps)) * abs(actual - mean_p)
        out[side] = {
            "n": len(pairs),
            "mean_pred": round(float(ps.mean()), 3),
            "actual_rate": round(float(ys.mean()), 3),
            "brier": round(brier, 4),
            "ece": round(float(ece), 4),
            "reliability": buckets,
        }
    out["symbols"] = used_symbols
    out["failures"] = failures
    return out


def _ratio_dispersion(stats: Dict) -> Optional[Dict]:
    """Per-market long/short signal-ratio dispersion across symbols.

    Returns None when fewer than 3 symbols have both-side signals (a
    dispersion claim needs enough per-market ratios to be meaningful).
    """
    sig_ratios = []
    for sym, s in stats["by_symbol"].items():
        ls, ss = s["long"]["signals"], s["short"]["signals"]
        if ls and ss:
            sig_ratios.append((sym, ls / ss))
    if len(sig_ratios) < 3:
        return None
    vals = np.array([r for _, r in sig_ratios], dtype=float)
    p25, p75 = np.percentile(vals, [25, 75])
    most = sorted(sig_ratios, key=lambda kv: abs(np.log(kv[1])))[-1]
    return {
        "n": len(vals),
        "median": float(np.median(vals)),
        "mean": float(vals.mean()),
        "std": float(vals.std()),
        "p25": float(p25),
        "p75": float(p75),
        "min": float(vals.min()),
        "min_sym": min(sig_ratios, key=lambda kv: kv[1])[0],
        "max": float(vals.max()),
        "max_sym": max(sig_ratios, key=lambda kv: kv[1])[0],
        "most_skewed": most[1],
        "most_skewed_sym": most[0],
    }


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
        f"{'L win':>6}{'S win':>6}{'ratio':>8}"
    )
    for row in rows[:symbols_shown]:
        _, lc, sc, ls, ss, lw, sw = row
        sig_ratio = (ls / ss) if ss else (None if ls == 0 else float("inf"))
        ratio_txt = "-" if sig_ratio is None else f"{sig_ratio:.2f}"
        print(
            f"{row[0]:<8}{lc:>8,}{sc:>8,}{ls:>6,}{ss:>6,}{lw:>6,}{sw:>6,}{ratio_txt:>8}"
        )

    # Per-market ratio dispersion: the universe average can hide wide
    # symbol-level skew (a handful of markets driving the headline).
    disp = _ratio_dispersion(stats)
    if disp:
        print(
            "\nPer-market signal-ratio dispersion: "
            f"n={disp['n']} · median {disp['median']:.2f} · "
            f"mean {disp['mean']:.2f} · std {disp['std']:.2f} · "
            f"IQR {disp['p25']:.2f}-{disp['p75']:.2f} · "
            f"min {disp['min']:.2f} ({disp['min_sym']}) · "
            f"max {disp['max']:.2f} ({disp['max_sym']})"
        )
        print(
            f"   Most skewed: {disp['most_skewed_sym']} "
            f"({disp['most_skewed']:.2f}x) - the universe median/mean is "
            f"NOT per-market neutrality"
        )

    if stats.get("failures"):
        print(
            f"\n[{len(stats['failures'])} symbol(s) skipped]: "
            f"{', '.join(stats['failures'][:4])}..."
        )

    print_census_target_level(stats)
    print_census_regimes(stats)

    lr, sr = stats["recall"]["long"], stats["recall"]["short"]
    if lr["signals"] and sr["signals"]:
        ratio = lr["signals"] / sr["signals"]
        print(
            f"\nLong/short signal ratio: {ratio:.2f} (1.0 = equal opportunity "
            f"detection; ≠1 is evidence, not bias, unless one side can't "
            f"enter the pipeline at all)"
        )
    print("=" * 72 + "\n")


def print_census_target_level(stats: Dict) -> None:
    """Target-level EV section: empirical P(TP_k before SL) per side and
    the EV after a cost sweep (spec #4/#12)."""
    tl = _target_level_stats(stats)
    print("\nTarget-level EV (multi-barrier first-touch, all classified candidates):")
    print(
        f"{'SIDE':<6}{'n':>8}{'P(tp1)':>8}{'P(tp2)':>8}{'P(tp3)':>8}{'P(sl)':>8}{'P(none)':>8}"
    )
    for side in ("long", "short"):
        s = tl[side]
        if not s.get("n"):
            continue
        print(
            f"{side:<6}{s['n']:>8,}{s['p_tp1']:>8.3f}{s['p_tp2']:>8.3f}"
            f"{s['p_tp3']:>8.3f}{s['p_sl']:>8.3f}{s['p_none']:>8.3f}"
        )
    print(
        "\nTarget-level EV per side after round-trip cost (rungs 1R/2R/3R, stop -1R):"
    )
    print(f"{'SIDE':<6}{'cost 0':>10}{'0.05R':>10}{'0.10R':>10}{'0.15R':>10}")
    for side in ("long", "short"):
        s = tl[side]
        if not s.get("n"):
            continue
        ev = s["ev"]
        print(
            f"{side:<6}{ev['ev_00']:>+10.3f}{ev['ev_05']:>+10.3f}"
            f"{ev['ev_10']:>+10.3f}{ev['ev_15']:>+10.3f}"
        )


def print_census_regimes(stats: Dict) -> None:
    """Regime-conditional long/short section (spec #3)."""
    regimes = sorted({k[1] for k in stats["regime_side"]})
    if not regimes:
        return
    print("\nRegime-conditional long/short signals (confirmed engine signals):")
    print(
        f"{'REGIME':<16}{'L sig':>8}{'L win%':>8}{'L exp':>8}"
        f"{'S sig':>8}{'S win%':>8}{'S exp':>8}{'L/S cand':>10}"
    )
    for reg in sorted(regimes):

        def _row(side, reg_):
            c = stats["regime_side"][(side, reg_)]
            sigs = c.get("signals", 0)
            wins = c.get("wins", 0)
            wr = wins / sigs if sigs else None
            exp = (wins - (sigs - wins)) / sigs if sigs else None
            return sigs, wr, exp

        ls, lwr, lexp = _row("long", reg)
        ss, swr, sexp = _row("short", reg)
        lc = stats["regime_side"][("long", reg)].get("candidates", 0)
        sc = stats["regime_side"][("short", reg)].get("candidates", 0)
        print(
            f"{reg:<16}{ls:>8,}{('- ' if lwr is None else f'{100 * lwr:.0f}%'):>8}"
            f"{('- ' if lexp is None else f'{lexp:+.2f}R'):>8}"
            f"{ss:>8,}{('- ' if swr is None else f'{100 * swr:.0f}%'):>8}"
            f"{('- ' if sexp is None else f'{sexp:+.2f}R'):>8}"
            f"{lc / sc if sc else '-':>10.2f}"
        )


def print_calibration(cal: Dict) -> None:
    """Long/short model calibration section (spec #5)."""
    print("\n" + "=" * 72)
    print("MODEL CALIBRATION — LONG vs SHORT (independent)")
    print("=" * 72)
    print(f"Symbols: {cal.get('symbols', 0)} · P(short)=1-P(long) is NOT assumed")
    for side in ("long", "short"):
        s = cal[side]
        if not s.get("n"):
            print(f"\n[{side}] no resolved prediction/outcome pairs")
            continue
        print(
            f"\n[{side.upper()}] n={s['n']:,} · mean pred {s['mean_pred']:.3f} · "
            f"actual {s['actual_rate']:.3f} · Brier {s['brier']:.4f} · "
            f"ECE {s['ece']:.4f}"
        )
        print(f"{'BUCKET':<16}{'n':>7}{'mean_p':>8}{'actual':>8}")
        for b in s["reliability"]:
            print(
                f"{b['bucket']:<16}{b['n']:>7,}{b['mean_p']:>8.3f}{b['actual']:>8.3f}"
            )
    if cal.get("failures"):
        print(
            f"\n[{len(cal['failures'])} symbol(s) skipped]: {', '.join(cal['failures'][:3])}..."
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
    parser.add_argument(
        "--calibrate",
        action="store_true",
        help="also run independent LONG/SHORT model calibration (Brier/ECE)",
    )
    parser.add_argument(
        "--write-probs",
        action="store_true",
        help="write the empirical target-level TP distribution to "
        "data/validation/target_probs.json for the live book",
    )
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

    if args.calibrate:
        cal = model_calibration(symbols, group=args.group, timeframe=args.timeframe)
        print_calibration(cal)

    if args.write_probs:
        import json

        out = Path("data/validation")
        out.mkdir(parents=True, exist_ok=True)
        payload = target_probs_json(stats)
        with open(out / "target_probs.json", "w") as fh:
            json.dump(payload, fh, indent=2)
        print(
            f"\nTarget-level TP distribution written to "
            f"{out / 'target_probs.json'} (n={payload['n']})"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
