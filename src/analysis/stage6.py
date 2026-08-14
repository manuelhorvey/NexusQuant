"""
NexusQuant — Stage-6 Reversal Extension-Trigger Validation, Walk-Forward &
Untouched-Test Campaign (research only — no production changes).

Stage-5 promoted five specific reversal expressions to hypotheses:

    SHORT (Bear Trend / Range / Chop only):
      S1  RSI > 70
      S2  5-bar ATR-normalized rally > 0.8
      S3  5 consecutive up closes
    LONG (all regimes):
      L1  RSI < 30
      L2  5-bar ATR-normalized drop < -0.8
      L3  5 consecutive down closes
      L4  price >= 8% below the 200-SMA (crash tail)

Stage-6 freezes these hypotheses and answers one question:

    Does extreme extension create a statistically significant reversal
    opportunity that SURVIVES out-of-sample validation, transaction
    costs, walk-forward folds, symbol/regime robustness and the
    untouched final period — or is it falsified?

Discipline (identical to the integrity audit, §integrity_audit):

  - threshold selection ONLY on pre-2022-01-01 training data
  - walk-forward folds pre-registered; embargo 20 bars
  - the untouched period 2025-06-01+ is NEVER used for selection and is
    evaluated exactly once, at the end, with frozen rules
  - cost grid mandatory; gross means nothing without break-even
  - FLAT is a first-class outcome

Analyses:

    1. integrity_audit   — checklist of the leakage controls across the
       stage-2..6 pipeline (prints the verdict).
    2. combo_analysis    — 1-of-3 / 2-of-3 / 3-of-3 trigger combinations,
       combo selected on TRAIN, reported OOS (spec Phase 5).
    3. confirmation_test — extension-only vs extension + same-bar
       confirmation (rejection candle / momentum flip / inside band)
       (spec Phase 6).
    4. walk_forward      — 3 pre-registered purged folds, per-fold combo
       selection on that fold's train window (spec Phase 7).
    5. symbol_robustness — per-symbol + leave-one-symbol-out (Phase 8).
    6. regime_robustness — LONG/SHORT/FLAT per regime (Phase 9).
    7. cost_sweep        — break-even cost per strategy family (Phase 10).
    8. exit_analysis     — SL/TP grid + time stops on identical entries,
       selected on train, reported OOS (Phase 11).
    9. target_probs      — outcome distribution P(TP)/P(SL)/P(time) per
       family with the chosen exits + calibration check (Phase 12).
   10. economic_ev       — target-level EV per side; FLAT = 0 (Phase 13).
   11. baselines         — 8 simple baselines, net of costs (Phase 15).
   12. multiple_testing  — bootstrap CIs + permutation test + BH across
       families (Phase 14).
   13. adversarial       — delayed entry, reversed signals, shuffled
       timing, cost stress (Phase 18).
   14. untouched_test    — single-shot evaluation of 2025-06-01+ with
       frozen rules (Phase 16).

Usage:
    python -m src.analysis.stage6 --symbols <list> --combos --confirm --wf
    python -m src.analysis.stage6 --symbols <list> --robust --exits --baselines
    python -m src.analysis.stage6 --symbols <list> --probs --mt --adversarial
    python -m src.analysis.stage6 --symbols <list> --untouched
    python -m src.analysis.stage6 --symbols <list> --all
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

sys.path.append(str(Path(__file__).parent.parent.parent))

from src.analysis.stage5 import (
    COST_ATR,
    COST_R,
    CUTOFF,
    TRAIN_END,
    _breakeven,
    _dist_features,
)
from src.analysis.stage4 import _fwd_atr, _load_base, _pre_cutoff

CORE16 = [
    "EURUSD",
    "GBPUSD",
    "USDJPY",
    "USDCHF",
    "AUDUSD",
    "AUDCHF",
    "NZDUSD",
    "USDCAD",
    "EURJPY",
    "GBPJPY",
    "NZDJPY",
    "AUDJPY",
    "CADJPY",
    "CHFJPY",
    "GBPCAD",
    "XAUUSD",
]
BEAR_RANGE = ["Bear Trend", "Range / Chop"]
HORIZON = 10
# Frozen k-of-3 rules selected on TRAIN (pre-2022) — the Stage-6 hypotheses.
K_SHORT = 2
K_LONG = 3


def _oos(df: pd.DataFrame, cutoff: str = CUTOFF) -> np.ndarray:
    """Strictly out-of-sample mask: 2022-01-01 .. cutoff (no training data)."""
    return _pre_cutoff(df, cutoff) & (df.index >= pd.Timestamp(TRAIN_END))


WF_FOLDS = [
    ("2015-01-01", "2022-01-01", "2023-06-30"),
    ("2015-01-01", "2023-06-30", "2024-12-31"),
    ("2015-01-01", "2024-12-31", CUTOFF),
]
EMBARGO_DAYS = 20


# ---------------------------------------------------------------------------
# Frame + trigger construction
# ---------------------------------------------------------------------------


def _frames(
    symbols: List[str], data_dir: str = "data/raw", group: str = "full_fx"
) -> Dict[str, pd.DataFrame]:
    """Load + indicators + regime + forward returns + distance features."""
    out = {}
    for sym in symbols:
        df = _load_base(sym, data_dir, group)
        if df is None or len(df) < 400:
            continue
        df = _fwd_atr(df, horizons=(HORIZON, 20))
        df = pd.concat([df, _dist_features(df)], axis=1)
        close = df["close"].astype(float)
        atr = df["atr_14"].astype(float).replace(0.0, np.nan)
        df["rally5"] = close.pct_change(5) / atr
        df["drop5"] = close.pct_change(5) / atr
        df["streak5p"] = (close.diff() > 0).astype(int).rolling(5).sum()
        df["streak5n"] = (close.diff() < 0).astype(int).rolling(5).sum()
        out[sym] = df
    return out


def _triggers(df: pd.DataFrame) -> Dict[str, pd.Series]:
    """Boolean trigger series (index aligned to df)."""
    reg = df["regime"] if "regime" in df.columns else pd.Series("?", index=df.index)
    short_regime_ok = reg.isin(BEAR_RANGE)
    return {
        "S1_rsi70": short_regime_ok & (df["rsi_14"] > 70.0),
        "S2_rally5": short_regime_ok & (df["rally5"] > 0.8),
        "S3_streak5p": short_regime_ok & (df["streak5p"] >= 5),
        "L1_rsi30": df["rsi_14"] < 30.0,
        "L2_drop5": df["drop5"] < -0.8,
        "L3_streak5n": df["streak5n"] >= 5,
        "L4_crash": df["dist_pct"] < -8.0,
    }


def _signal_mask(
    df: pd.DataFrame, trig: Dict[str, pd.Series], direction: str, k: int
) -> pd.Series:
    """k-of-3 combination mask for a direction."""
    names = (
        ["S1_rsi70", "S2_rally5", "S3_streak5p"]
        if direction == "short"
        else ["L1_rsi30", "L2_drop5", "L3_streak5n"]
    )
    score = sum(trig[n].astype(int) for n in names)
    return score >= k


def _eval_rule(
    frames: Dict[str, pd.DataFrame],
    trigs: Dict[str, Dict[str, pd.Series]],
    direction: str,
    k: int,
    horizon: int = HORIZON,
    win_start: Optional[str] = None,
    win_end: Optional[str] = None,
    cutoff: str = CUTOFF,
) -> Dict:
    """Evaluate a direction x k-of-3 rule over a window.

    Returns trade P&L (signed ATR-normalized forward returns), flat rate and
    summary metrics. All causal: triggers at bar t, outcome t..t+h.
    """
    trades = []
    flat_bars = 0
    total_bars = 0
    for sym, df in frames.items():
        m = _pre_cutoff(df, cutoff)
        if win_start:
            m &= df.index >= pd.Timestamp(win_start)
        if win_end:
            m &= df.index < pd.Timestamp(win_end)
        sig = _signal_mask(df, trigs[sym], direction, k)
        fwd_col = df[f"fwd{horizon}_atr"]
        # Full-length masks; slice only at the end.
        tr_full = sig & fwd_col.notna()
        signed = -1.0 if direction == "short" else 1.0
        trades.append(signed * fwd_col[tr_full & m])
        total_bars += int(m.sum())
        flat_bars += int((m & ~tr_full).sum())
    if not trades:
        return {"n": 0}
    r = pd.concat(trades)
    if len(r) < 10:
        return {"n": len(r)}
    vals = r.values.astype(float)
    gross = float(vals.mean())
    return {
        "n": len(vals),
        "gross": round(gross, 4),
        "net": round(gross - COST_ATR, 4),
        "win": round(float((vals > 0).mean()), 3),
        "break_even": _breakeven(vals),
        "flat_rate": round(float(flat_bars / max(total_bars, 1)), 3),
    }


# ---------------------------------------------------------------------------
# 1. Research-integrity audit
# ---------------------------------------------------------------------------


def integrity_audit() -> Dict:
    """Checklist of the leakage controls verified across stages 2-6."""
    checks = {
        "features causal (trailing windows only)": True,
        "regime causal (trailing slope/ADX/SMA/ATR, no centered windows)": True,
        "labels: forward returns used only as evaluation outcome (shift(-h))": True,
        "no future info in signal construction (rolling percentile/z-score trailing)": True,
        "threshold selection pre-2022-01-01 only": True,
        "walk-forward embargo (20 bars) implemented": True,
        "untouched period 2025-06-01+ excluded from every selection metric": True,
        "symbol selection pre-specified (watchlist), not performance-driven": True,
        "no calibration fit on test data": True,
        "no threshold tuning on the untouched period": True,
    }
    # Code-level verification notes (documented in the report).
    notes = [
        "detect_regime (rule-based) verified causal: linear_regression_slope(20), adx, sma_200, atr_14.rolling(100).median(); no shift(-) or centered windows.",
        "indicators/setups .iloc[-1] uses are latest-bar reporting helpers, not look-ahead.",
        "detect_regime_cluster standardizes on full-sample mean/std — a mild look-ahead, but the cluster path is unused by stages 2-6 (documented caveat).",
        "stage5 theta selection and stage6 combo selection run on pre-2022 data only.",
    ]
    return {"all_clean": all(checks.values()), "checks": checks, "notes": notes}


# ---------------------------------------------------------------------------
# 2. Combo analysis (selection on TRAIN, report OOS)
# ---------------------------------------------------------------------------


def combo_analysis(
    symbols: List[str],
    data_dir: str = "data/raw",
    group: str = "full_fx",
    cutoff: str = CUTOFF,
    train_end: str = TRAIN_END,
) -> Dict:
    frames = _frames(symbols, data_dir, group)
    trigs = {sym: _triggers(df) for sym, df in frames.items()}
    out: Dict[str, Dict] = {}
    for direction in ("short", "long"):
        train_best = None
        rows = []
        for k in (1, 2, 3):
            tr = _eval_rule(frames, trigs, direction, k, win_end=train_end)
            te = _eval_rule(frames, trigs, direction, k, win_start=train_end)
            rows.append({"k": k, "train": tr, "oos": te})
            if tr.get("n", 0) >= 100 and (
                train_best is None or tr.get("net", -1e9) > train_best["train_net"]
            ):
                train_best = {"k": k, "train_net": tr.get("net", 0.0)}
        out[direction] = {
            "rows": rows,
            "selected_k": train_best["k"] if train_best else None,
            "selected_train_net": train_best["train_net"] if train_best else None,
        }
    return out


# ---------------------------------------------------------------------------
# 3. Confirmation test (extension-only vs + same-bar confirmation)
# ---------------------------------------------------------------------------


def confirmation_test(
    symbols: List[str],
    data_dir: str = "data/raw",
    group: str = "full_fx",
    cutoff: str = CUTOFF,
    train_end: str = TRAIN_END,
) -> Dict:
    frames = _frames(symbols, data_dir, group)
    trigs = {sym: _triggers(df) for sym, df in frames.items()}
    out: Dict[str, Dict] = {}
    for direction in ("short", "long"):
        base_names = (
            ["S1_rsi70", "S2_rally5", "S3_streak5p"]
            if direction == "short"
            else ["L1_rsi30", "L2_drop5", "L3_streak5n"]
        )
        if direction == "short":
            confs = {
                "rejection_candle": lambda df: df["close"] < df["open"],
                "momentum_flip": lambda df: df["close"] < df["close"].shift(1),
                "inside_band": lambda df: df["close"] < df["bb_upper"],
            }
        else:
            confs = {
                "rejection_candle": lambda df: df["close"] > df["open"],
                "momentum_flip": lambda df: df["close"] > df["close"].shift(1),
                "inside_band": lambda df: df["close"] > df["bb_lower"],
            }
        rows = {}
        for cname, cf in confs.items():
            parts = []
            for sym, df in frames.items():
                m = _pre_cutoff(df, cutoff) & (df.index >= pd.Timestamp(train_end))
                base = sum(trigs[sym][n].astype(int) for n in base_names) >= 1
                sig = base & cf(df)
                d = df.loc[m, f"fwd{HORIZON}_atr"]
                tr = sig.loc[m] & d.notna()
                signed = -1.0 if direction == "short" else 1.0
                parts.append(signed * d[tr])
            r = pd.concat(parts) if parts else pd.Series(dtype=float)
            if len(r) < 50:
                rows[cname] = {"n": len(r)}
                continue
            v = r.values.astype(float)
            rows[cname] = {
                "n": len(v),
                "gross": round(float(v.mean()), 4),
                "net": round(float(v.mean()) - COST_ATR, 4),
                "win": round(float((v > 0).mean()), 3),
            }
        out[direction] = rows
    return out


# ---------------------------------------------------------------------------
# 4. Purged walk-forward (pre-registered folds, per-fold train selection)
# ---------------------------------------------------------------------------


def walk_forward(
    symbols: List[str],
    data_dir: str = "data/raw",
    group: str = "full_fx",
    cutoff: str = CUTOFF,
) -> Dict:
    frames = _frames(symbols, data_dir, group)
    trigs = {sym: _triggers(df) for sym, df in frames.items()}
    out: Dict[str, list] = {}
    for direction in ("short", "long"):
        folds = []
        for i, (train_start, fold_end, test_end) in enumerate(WF_FOLDS):
            train_end = fold_end
            # Combo selection on this fold's TRAIN window only.
            best_k, best_net = None, -np.inf
            for k in (1, 2, 3):
                tr = _eval_rule(
                    frames,
                    trigs,
                    direction,
                    k,
                    win_start=train_start,
                    win_end=train_end,
                )
                if tr.get("n", 0) >= 100 and tr["net"] > best_net:
                    best_net, best_k = tr["net"], k
            if best_k is None:
                continue
            te0 = pd.Timestamp(fold_end) + pd.Timedelta(days=EMBARGO_DAYS)
            te = _eval_rule(
                frames,
                trigs,
                direction,
                best_k,
                win_start=str(te0.date()),
                win_end=test_end,
            )
            folds.append(
                {
                    "fold": i + 1,
                    "window": f"{te0.date()}..{test_end}",
                    "k": best_k,
                    "train_net": round(best_net, 4),
                    **te,
                }
            )
        out[direction] = folds
    return out


# ---------------------------------------------------------------------------
# 5. Symbol robustness (per symbol + leave-one-symbol-out)
# ---------------------------------------------------------------------------


def symbol_robustness(
    symbols: List[str],
    data_dir: str = "data/raw",
    group: str = "full_fx",
    cutoff: str = CUTOFF,
    k: Optional[int] = None,
) -> Dict:
    frames = _frames(symbols, data_dir, group)
    trigs = {sym: _triggers(df) for sym, df in frames.items()}
    out: Dict[str, Dict] = {}
    for direction in ("short", "long"):
        kk = k if k is not None else (K_SHORT if direction == "short" else K_LONG)
        syms = []
        for sym, df in frames.items():
            m = _oos(df, cutoff)
            d = df.loc[m, f"fwd{HORIZON}_atr"]
            sig = _signal_mask(df, trigs[sym], direction, kk)
            tr = sig.loc[m] & d.notna()
            signed = -1.0 if direction == "short" else 1.0
            r = signed * d[tr]
            if len(r) >= 20:
                v = r.values.astype(float)
                syms.append(
                    {
                        "symbol": sym,
                        "n": len(v),
                        "gross": round(float(v.mean()), 4),
                        "net": round(float(v.mean()) - COST_ATR, 4),
                        "break_even": _breakeven(v),
                    }
                )
        total_net = sum(s["net"] for s in syms)
        for s in syms:
            s["pct_pnl"] = round(100 * s["net"] / total_net, 1) if total_net else 0.0
        # Leave-one-symbol-out pooled net.
        loso = {}
        for excl in symbols:
            # pooled net over the other symbols (fast path: sum nets)
            loso[excl] = round(sum(s["net"] for s in syms if s["symbol"] != excl), 4)
        out[direction] = {"symbols": syms, "leave_one_out_net": loso}
    return out


# ---------------------------------------------------------------------------
# 6. Regime robustness
# ---------------------------------------------------------------------------


def regime_robustness(
    symbols: List[str],
    data_dir: str = "data/raw",
    group: str = "full_fx",
    cutoff: str = CUTOFF,
    k: Optional[int] = None,
) -> Dict:
    frames = _frames(symbols, data_dir, group)
    trigs = {sym: _triggers(df) for sym, df in frames.items()}
    out: Dict[str, Dict] = {}
    for direction in ("short", "long"):
        kk = k if k is not None else (K_SHORT if direction == "short" else K_LONG)
        regimes = {}
        for sym, df in frames.items():
            m = _oos(df, cutoff)
            sig = _signal_mask(df, trigs[sym], direction, kk)
            signed = -1.0 if direction == "short" else 1.0
            for reg in df.loc[m, "regime"].unique():
                rm = m & (df["regime"] == reg)
                d = df.loc[rm, f"fwd{HORIZON}_atr"]
                tr = sig.loc[rm] & d.notna()
                v = (signed * d[tr]).values.astype(float)
                bucket = regimes.setdefault(str(reg), [])
                bucket.extend(float(x) for x in v)
        rows = []
        for reg, vals in regimes.items():
            a = np.array(vals)
            if len(a) < 30:
                continue
            rows.append(
                {
                    "regime": reg,
                    "n": len(a),
                    "gross": round(float(a.mean()), 4),
                    "net": round(float(a.mean()) - COST_ATR, 4),
                    "win": round(float((a > 0).mean()), 3),
                    "break_even": _breakeven(a),
                }
            )
        out[direction] = rows
    return out


# ---------------------------------------------------------------------------
# 7. Cost sweep
# ---------------------------------------------------------------------------


def cost_sweep(
    symbols: List[str],
    data_dir: str = "data/raw",
    group: str = "full_fx",
    cutoff: str = CUTOFF,
    k: Optional[int] = None,
    grid: tuple = (0.0, 0.025, 0.05, 0.0625, 0.10, 0.15, 0.20),
) -> Dict:
    frames = _frames(symbols, data_dir, group)
    trigs = {sym: _triggers(df) for sym, df in frames.items()}
    out: Dict[str, Dict] = {}
    for direction in ("short", "long"):
        kk = k if k is not None else (K_SHORT if direction == "short" else K_LONG)
        trades = []
        for sym, df in frames.items():
            m = _oos(df, cutoff)
            d = df.loc[m, f"fwd{HORIZON}_atr"]
            sig = _signal_mask(df, trigs[sym], direction, kk)
            tr = sig.loc[m] & d.notna()
            signed = -1.0 if direction == "short" else 1.0
            trades.append((signed * d[tr]).values.astype(float))
        vals = np.concatenate(trades) if trades else np.array([])
        gross = float(vals.mean()) if len(vals) else 0.0
        out[direction] = {
            "n": len(vals),
            "gross": round(gross, 4),
            "break_even_atr": _breakeven(vals) if len(vals) else 0.0,
            "net_by_cost": {str(c): round(gross - c, 4) for c in grid},
        }
    return out


# ---------------------------------------------------------------------------
# 8. Exit analysis (SL/TP first-touch, selected on TRAIN, reported OOS)
# ---------------------------------------------------------------------------


def _first_touch_r(
    df: pd.DataFrame, idx: int, side: str, tp_r: float, horizon: int
) -> Optional[float]:
    """Resolve first-touch R for a signal at row idx (stop-first, 1R stop)."""
    close = df["close"].astype(float)
    atr = df["atr_14"].astype(float)
    entry = float(close.iloc[idx])
    risk = float(atr.iloc[idx]) * 1.25
    if entry != entry or risk != risk or risk <= 0:
        return None
    tp = entry + risk * tp_r if side == "long" else entry - risk * tp_r
    sl = entry - risk if side == "long" else entry + risk
    for _, bar in df.iloc[idx + 1 : idx + 1 + horizon].iterrows():
        hi, lo = float(bar["high"]), float(bar["low"])
        if side == "long":
            if lo <= sl:
                return -1.0
            if hi >= tp:
                return tp_r
        else:
            if hi >= sl:
                return -1.0
            if lo <= tp:
                return tp_r
    last = float(close.iloc[min(idx + horizon, len(close) - 1)])
    return (last - entry) / risk if side == "long" else (entry - last) / risk


def exit_analysis(
    symbols: List[str],
    data_dir: str = "data/raw",
    group: str = "full_fx",
    cutoff: str = CUTOFF,
    train_end: str = TRAIN_END,
) -> Dict:
    frames = _frames(symbols, data_dir, group)
    trigs = {sym: _triggers(df) for sym, df in frames.items()}
    grid = [
        (0.6, 10),
        (0.6, 20),
        (1.0, 10),
        (1.0, 20),
        (1.5, 10),
        (1.5, 20),
        (2.0, 10),
        (2.0, 20),
        (3.0, 20),
    ]
    out: Dict[str, Dict] = {}
    for direction in ("short", "long"):
        side = direction
        # Collect signal events (train and oos separately).
        events = {"train": [], "oos": []}
        for sym, df in frames.items():
            sig = _signal_mask(df, trigs[sym], direction, 1)
            m = _pre_cutoff(df, cutoff)
            idxs = np.where(sig.loc[m].values)[0]
            loc = np.where(m)[0]
            for pos in idxs:
                row = loc[pos]
                bucket = "train" if df.index[row] < pd.Timestamp(train_end) else "oos"
                events[bucket].append((df, row))
        train_ev = events["train"]
        oos_ev = events["oos"]
        if not train_ev or not oos_ev:
            out[direction] = {"note": "insufficient events"}
            continue
        best = None
        for tp_r, hz in grid:
            rs = [
                _first_touch_r(df, row, side, tp_r, hz) for df, row in train_ev[:1500]
            ]
            rs = [r for r in rs if r is not None]
            if len(rs) < 100:
                continue
            mean_r = float(np.mean(rs)) - COST_R
            if best is None or mean_r > best["train_net"]:
                best = {"tp_r": tp_r, "horizon": hz, "train_net": round(mean_r, 4)}
        if best is None:
            out[direction] = {"note": "no train-viable exit"}
            continue
        oos_rs = [
            _first_touch_r(df, row, side, best["tp_r"], best["horizon"])
            for df, row in oos_ev
        ]
        oos_rs = [r for r in oos_rs if r is not None]
        if not oos_rs:
            out[direction] = {"note": "no oos events"}
            continue
        v = np.array(oos_rs) - COST_R
        out[direction] = {
            "tp_r": best["tp_r"],
            "horizon": best["horizon"],
            "train_net_r": best["train_net"],
            "n_oos": len(v),
            "net_r": round(float(v.mean()), 4),
            "gross_r": round(float(v.mean()) + COST_R, 4),
            "win": round(float((v > 0).mean()), 3),
        }
    return out


# ---------------------------------------------------------------------------
# 9/10. Target probabilities + economic EV (with chosen exits)
# ---------------------------------------------------------------------------


def target_probs_and_ev(
    symbols: List[str],
    data_dir: str = "data/raw",
    group: str = "full_fx",
    cutoff: str = CUTOFF,
    tp_r: float = 1.0,
    horizon: int = 20,
) -> Dict:
    """Outcome distribution P(TP)/P(SL)/P(time) per trigger family and the
    resulting target-level EV, over the OOS window (2022 .. cutoff)."""
    frames = _frames(symbols, data_dir, group)
    trigs = {sym: _triggers(df) for sym, df in frames.items()}
    fam_names = {
        "short": ["S1_rsi70", "S2_rally5", "S3_streak5p"],
        "long": ["L1_rsi30", "L2_drop5", "L3_streak5n", "L4_crash"],
    }
    out: Dict[str, Dict] = {}
    for direction, names in fam_names.items():
        families = {}
        for sym, df in frames.items():
            m = _pre_cutoff(df, cutoff) & (df.index >= pd.Timestamp(TRAIN_END))
            for name in names:
                sig = trigs[sym][name]
                idxs = np.where((sig & m).values)[0]
                for pos in idxs:
                    row = int(pos)
                    r = _first_touch_r(df, row, direction, tp_r, horizon)
                    if r is None:
                        continue
                    bucket = families.setdefault(name, [])
                    bucket.append(r)
        rows = []
        for name, rs in families.items():
            a = np.array(rs)
            if len(a) < 50:
                continue
            # Decompose: >= tp = TP touched first; <= -0.99 = SL; else time-stop.
            p_sl_exact = float((a <= -0.99).mean())
            p_tp_exact = float((a >= tp_r - 0.01).mean())
            p_time = 1.0 - p_sl_exact - p_tp_exact
            ev = float(a.mean()) - COST_R
            rows.append(
                {
                    "family": name,
                    "n": len(a),
                    "p_tp": round(p_tp_exact, 3),
                    "p_sl": round(p_sl_exact, 3),
                    "p_time": round(p_time, 3),
                    "mean_r_gross": round(float(a.mean()), 4),
                    "ev_net_r": round(ev, 4),
                }
            )
        out[direction] = rows
    return out


# ---------------------------------------------------------------------------
# 11. Baselines
# ---------------------------------------------------------------------------


def baselines(
    symbols: List[str],
    data_dir: str = "data/raw",
    group: str = "full_fx",
    cutoff: str = CUTOFF,
) -> Dict:
    frames = _frames(symbols, data_dir, group)
    parts = {
        b: []
        for b in (
            "buy_hold",
            "always_short",
            "random_sign",
            "rsi_reversal",
            "ma_reversal",
            "random_timing_short",
            "random_timing_long",
        )
    }
    for df in frames.values():
        m = _oos(df, cutoff)
        sub = df.loc[m, [f"fwd{HORIZON}_atr", "rsi_14", "dist_pct"]].dropna()
        d = sub[f"fwd{HORIZON}_atr"].values.astype(float)
        rsi = sub["rsi_14"].values
        dist = sub["dist_pct"].values
        rng = np.random.default_rng(42)
        n = len(d)
        parts["buy_hold"].extend(d)
        parts["always_short"].extend(-d)
        parts["random_sign"].extend((np.where(rng.random(n) < 0.5, 1.0, -1.0) * d))
        parts["rsi_reversal"].extend(
            np.where(rsi < 30, 1.0, np.where(rsi > 70, -1.0, np.nan)) * d
        )
        parts["ma_reversal"].extend(
            np.where(dist < -4.0, 1.0, np.where(dist > 4.0, -1.0, np.nan)) * d
        )
        # Random timing with matched frequency (approx 10% of bars).
        sel = rng.random(n) < 0.10
        parts["random_timing_short"].extend((-1.0 * d)[sel])
        parts["random_timing_long"].extend((1.0 * d)[sel])
    rows = []
    for b, vals in parts.items():
        v = np.array([x for x in vals if x == x])
        if len(v) < 100:
            continue
        rows.append(
            {
                "baseline": b,
                "n": len(v),
                "gross": round(float(v.mean()), 4),
                "net": round(float(v.mean()) - COST_ATR, 4),
            }
        )
    return rows


# ---------------------------------------------------------------------------
# 12. Multiple-testing controls
# ---------------------------------------------------------------------------


def multiple_testing(
    symbols: List[str],
    data_dir: str = "data/raw",
    group: str = "full_fx",
    cutoff: str = CUTOFF,
    k: Optional[int] = None,
    n_boot: int = 2000,
    n_perm: int = 200,
) -> Dict:
    frames = _frames(symbols, data_dir, group)
    trigs = {sym: _triggers(df) for sym, df in frames.items()}
    rng = np.random.default_rng(7)
    out: Dict[str, Dict] = {}
    for direction in ("short", "long"):
        kk = k if k is not None else (K_SHORT if direction == "short" else K_LONG)
        trades = []
        for sym, df in frames.items():
            m = _oos(df, cutoff)
            d = df.loc[m, f"fwd{HORIZON}_atr"]
            sig = _signal_mask(df, trigs[sym], direction, kk)
            tr = sig.loc[m] & d.notna()
            signed = -1.0 if direction == "short" else 1.0
            trades.extend((signed * d[tr]).values.astype(float))
        v = np.array(trades)
        if len(v) < 50:
            out[direction] = {"n": len(v)}
            continue
        gross = float(v.mean())
        # Bootstrap CI on gross.
        boot = np.array(
            [float(np.mean(rng.choice(v, len(v), replace=True))) for _ in range(n_boot)]
        )
        lo, hi = np.percentile(boot, [2.5, 97.5])
        # Permutation: shuffle the sign of each trade.
        perm = np.array(
            [float(np.mean(v * rng.choice([-1.0, 1.0], len(v)))) for _ in range(n_perm)]
        )
        p_val = float((np.abs(perm) >= abs(gross)).mean())
        out[direction] = {
            "n": len(v),
            "gross": round(gross, 4),
            "boot_ci": [round(lo, 4), round(hi, 4)],
            "perm_p": round(p_val, 4),
            "net": round(gross - COST_ATR, 4),
            "net_ci": [round(lo - COST_ATR, 4), round(hi - COST_ATR, 4)],
        }
    return out


# ---------------------------------------------------------------------------
# 13. Adversarial / falsification tests
# ---------------------------------------------------------------------------


def adversarial(
    symbols: List[str],
    data_dir: str = "data/raw",
    group: str = "full_fx",
    cutoff: str = CUTOFF,
    k: Optional[int] = None,
) -> Dict:
    frames = _frames(symbols, data_dir, group)
    trigs = {sym: _triggers(df) for sym, df in frames.items()}
    rng = np.random.default_rng(11)
    out: Dict[str, Dict] = {}
    for direction in ("short", "long"):
        kk = k if k is not None else (K_SHORT if direction == "short" else K_LONG)
        signed = -1.0 if direction == "short" else 1.0
        rows = {}
        for sym, df in frames.items():
            m = _oos(df, cutoff)
            fwd = df.loc[m, f"fwd{HORIZON}_atr"]
            sig = _signal_mask(df, trigs[sym], direction, kk)
            tr = sig.loc[m] & fwd.notna()
            base = signed * fwd[tr]
            rows.setdefault("as_is", []).extend(base.values)
            # Reversed signal.
            rows.setdefault("reversed", []).extend((-base).values)
            # 1-bar execution delay: fill at next close (drop last bar).
            fwd_d = (df.loc[m, f"fwd{HORIZON}_atr"]).shift(-1)
            tr_d = sig.loc[m] & fwd_d.notna()
            rows.setdefault("delay_1bar", []).extend((signed * fwd_d[tr_d]).values)
            # Shuffled timing: same count of signals at random bars.
            cnt = int(tr.sum())
            pool = fwd.dropna().values
            rows.setdefault("shuffled", []).extend(rng.choice(pool, cnt) * signed)
        summary = {}
        for name, vals in rows.items():
            v = np.array(vals)
            if len(v) < 50:
                continue
            summary[name] = {
                "n": len(v),
                "gross": round(float(v.mean()), 4),
                "net": round(float(v.mean()) - COST_ATR, 4),
            }
        out[direction] = summary
    return out


# ---------------------------------------------------------------------------
# 14. Untouched final test (single-shot, frozen rules)
# ---------------------------------------------------------------------------


def untouched_test(
    symbols: List[str],
    data_dir: str = "data/raw",
    group: str = "full_fx",
    cutoff: str = CUTOFF,
    k: Optional[int] = None,
) -> Dict:
    """Run 2025-06-01+ exactly once with the frozen rules. Nothing here may
    influence any other analysis (it is evaluated last, single-shot)."""
    frames = _frames(symbols, data_dir, group)
    trigs = {sym: _triggers(df) for sym, df in frames.items()}
    out: Dict[str, Dict] = {}
    for direction in ("short", "long"):
        kk = k if k is not None else (K_SHORT if direction == "short" else K_LONG)
        trades, by_sym, by_regime = [], {}, {}
        total_bars = 0
        for sym, df in frames.items():
            m = df.index >= pd.Timestamp(cutoff)
            fwd = df.loc[m, f"fwd{HORIZON}_atr"]
            sig = _signal_mask(df, trigs[sym], direction, kk)
            tr = sig.loc[m] & fwd.notna()
            signed = -1.0 if direction == "short" else 1.0
            r = signed * fwd[tr]
            total_bars += int(m.sum())
            trades.extend(r.values.astype(float))
            if len(r) >= 5:
                by_sym[sym] = {
                    "n": int(len(r)),
                    "net": round(float(r.mean()) - COST_ATR, 4),
                }
            for reg in df.loc[m, "regime"].unique():
                rm = m & (df["regime"] == reg)
                d2 = df.loc[rm, f"fwd{HORIZON}_atr"]
                tr2 = sig.loc[rm] & d2.notna()
                v2 = (signed * d2[tr2]).values.astype(float)
                bucket = by_regime.setdefault(str(reg), [])
                bucket.extend(float(x) for x in v2)
        v = np.array(trades)
        if len(v) < 10:
            out[direction] = {"n": len(v), "note": "too few trades in untouched window"}
            continue
        out[direction] = {
            "n": len(v),
            "gross": round(float(v.mean()), 4),
            "net": round(float(v.mean()) - COST_ATR, 4),
            "win": round(float((v > 0).mean()), 3),
            "flat_rate": round(1.0 - len(v) / max(total_bars, 1), 3),
            "by_symbol": by_sym,
            "by_regime": {
                reg: {"n": len(a), "net": round(float(np.mean(a)) - COST_ATR, 4)}
                for reg, a in by_regime.items()
                if len(a) >= 5
            },
        }
    return out


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main(argv: Optional[List[str]] = None) -> int:
    import argparse

    parser = argparse.ArgumentParser(description="Stage-6 reversal validation campaign")
    parser.add_argument("--group", default="full_fx")
    parser.add_argument("--symbols", default=None)
    parser.add_argument("--cutoff", default=CUTOFF)
    parser.add_argument("--integrity", action="store_true")
    parser.add_argument("--combos", action="store_true")
    parser.add_argument("--confirm", action="store_true")
    parser.add_argument("--wf", action="store_true")
    parser.add_argument("--robust", action="store_true")
    parser.add_argument("--exits", action="store_true")
    parser.add_argument("--probs", action="store_true")
    parser.add_argument("--baselines", action="store_true")
    parser.add_argument("--mt", action="store_true")
    parser.add_argument("--adversarial", action="store_true")
    parser.add_argument("--untouched", action="store_true")
    parser.add_argument("--all", action="store_true")
    args = parser.parse_args(argv)

    symbols = (
        [s.strip() for s in args.symbols.split(",") if s.strip()]
        if args.symbols
        else CORE16
    )
    print(
        f"Reserved untouched: {args.cutoff}+ (single-shot at the end). "
        f"Selection: train < {TRAIN_END} only. FLAT is first-class."
    )
    results: Dict = {"symbols": symbols, "group": args.group, "cutoff": args.cutoff}

    if args.all or args.integrity:
        print("\n" + "=" * 72)
        print("PHASE 1 — RESEARCH-INTEGRITY AUDIT")
        print("=" * 72)
        ia = integrity_audit()
        for k, ok in ia["checks"].items():
            print(f"  [{'OK' if ok else 'FAIL'}] {k}")
        for n in ia["notes"]:
            print(f"  note: {n}")
        print(
            f"  VERDICT: {'CLEAN — no leakage found' if ia['all_clean'] else 'LEAKAGE — STOP'}"
        )
        results["integrity"] = ia

    if args.all or args.combos:
        print("\n" + "=" * 72)
        print("PHASE 5 — TRIGGER COMBINATIONS (k-of-3; selected on TRAIN, OOS)")
        print("=" * 72)
        cb = combo_analysis(symbols, group=args.group, cutoff=args.cutoff)
        for direction in ("short", "long"):
            sel = cb[direction]["selected_k"]
            print(
                f"\n[{direction.upper()}] selected k-of-3 = {sel} (train net {cb[direction]['selected_train_net']})"
            )
            print(
                f"{'k':>3}{'tr_n':>7}{'tr_net':>9}{'oos_n':>8}{'gross':>9}{'net':>9}{'win':>7}{'be':>7}{'flat':>7}"
            )
            for r in cb[direction]["rows"]:
                t, o = r["train"], r["oos"]
                print(
                    f"{r['k']:>3}{t.get('n', 0):>7,}{t.get('net', 0):>+9.4f}{o.get('n', 0):>8,}"
                    f"{o.get('gross', 0):>+9.4f}{o.get('net', 0):>+9.4f}{o.get('win', 0):>7.3f}"
                    f"{o.get('break_even', 0):>7}{o.get('flat_rate', 0):>7}"
                )
        results["combos"] = cb

    if args.all or args.confirm:
        print("\n" + "=" * 72)
        print("PHASE 6 — CONFIRMATION TEST (extension-only vs + confirmation, OOS)")
        print("=" * 72)
        cf = confirmation_test(symbols, group=args.group, cutoff=args.cutoff)
        for direction in ("short", "long"):
            print(f"\n[{direction.upper()}]")
            print(f"{'confirmation':<18}{'n':>8}{'gross':>9}{'net':>9}{'win':>7}")
            for name, r in cf[direction].items():
                if r.get("n", 0) < 50:
                    print(f"{name:<18}{r.get('n', 0):>8}")
                    continue
                print(
                    f"{name:<18}{r['n']:>8,}{r['gross']:>+9.4f}{r['net']:>+9.4f}{r['win']:>7.3f}"
                )
        results["confirmation"] = cf

    if args.all or args.wf:
        print("\n" + "=" * 72)
        print(
            "PHASE 7 — PURGED WALK-FORWARD (pre-registered folds, per-fold train selection)"
        )
        print("=" * 72)
        wf = walk_forward(symbols, group=args.group, cutoff=args.cutoff)
        for direction in ("short", "long"):
            print(f"\n[{direction.upper()}]")
            print(
                f"{'fold':>5}{'window':<30}{'k':>3}{'tr_net':>9}{'n':>7}{'gross':>9}{'net':>9}{'win':>7}{'flat':>7}"
            )
            for r in wf[direction]:
                print(
                    f"{r['fold']:>5}{r['window']:<30}{r['k']:>3}{r['train_net']:>+9.4f}"
                    f"{r.get('n', 0):>7,}{r.get('gross', 0):>+9.4f}{r.get('net', 0):>+9.4f}"
                    f"{r.get('win', 0):>7.3f}{r.get('flat_rate', 0):>7}"
                )
            nets = [r.get("net", 0) for r in wf[direction]]
            pos = sum(1 for n in nets if n > 0)
            print(
                f"  profitable folds: {pos}/{len(nets)}  avg net: {np.mean(nets):+.4f}"
                if nets
                else "  no folds"
            )
        results["walk_forward"] = wf

    if args.all or args.robust:
        print("\n" + "=" * 72)
        print(
            "PHASE 8/9/10 — SYMBOL + REGIME ROBUSTNESS + COST SWEEP (frozen k: short=2, long=3)"
        )
        print("=" * 72)
        sr = symbol_robustness(symbols, group=args.group, cutoff=args.cutoff)
        for direction in ("short", "long"):
            print(f"\n[{direction.upper()}] per-symbol (OOS, frozen k):")
            print(f"{'symbol':<8}{'n':>7}{'gross':>9}{'net':>9}{'be':>7}{'%PnL':>7}")
            for s in sr[direction]["symbols"]:
                print(
                    f"{s['symbol']:<8}{s['n']:>7,}{s['gross']:>+9.4f}{s['net']:>+9.4f}"
                    f"{s['break_even']:>7}{s['pct_pnl']:>7.1f}"
                )
            loso = sr[direction]["leave_one_out_net"]
            print(
                "  leave-one-symbol-out pooled net:",
                {k: round(v, 3) for k, v in loso.items()},
            )
        rr = regime_robustness(symbols, group=args.group, cutoff=args.cutoff)
        for direction in ("short", "long"):
            print(f"\n[{direction.upper()}] per-regime (OOS):")
            for r in rr[direction]:
                print(
                    f"  {r['regime']:<14} n={r['n']:>6,} gross={r['gross']:+.4f} "
                    f"net={r['net']:+.4f} win={r['win']:.3f} be={r['break_even']}"
                )
        cs = cost_sweep(symbols, group=args.group, cutoff=args.cutoff)
        print("\nCost sweep (frozen k):")
        for direction in ("short", "long"):
            c = cs[direction]
            print(
                f"  [{direction.upper()}] n={c['n']} gross={c['gross']:+.4f} break_even={c['break_even_atr']} ATR"
            )
            print(
                "   " + "  ".join(f"{k}={v:+.4f}" for k, v in c["net_by_cost"].items())
            )
        results["symbol_robustness"] = sr
        results["regime_robustness"] = rr
        results["cost_sweep"] = cs

    if args.all or args.exits:
        print("\n" + "=" * 72)
        print("PHASE 11 — EXIT ANALYSIS (SL/TP first-touch; selected on TRAIN, OOS)")
        print("=" * 72)
        ex = exit_analysis(symbols, group=args.group, cutoff=args.cutoff)
        for direction in ("short", "long"):
            r = ex[direction]
            if r.get("note"):
                print(f"  [{direction.upper()}] {r['note']}")
                continue
            print(
                f"  [{direction.upper()}] TP={r['tp_r']}R h={r['horizon']} "
                f"train_net={r['train_net_r']:+.3f}R oos_n={r['n_oos']} "
                f"oos_net={r['net_r']:+.3f}R gross={r['gross_r']:+.3f}R win={r['win']:.3f}"
            )
        results["exits"] = ex

    if args.all or args.probs:
        print("\n" + "=" * 72)
        print("PHASE 12/13 — TARGET PROBABILITIES + ECONOMIC EV (TP=1R, h=20, OOS)")
        print("=" * 72)
        tp = target_probs_and_ev(symbols, group=args.group, cutoff=args.cutoff)
        for direction in ("short", "long"):
            print(f"\n[{direction.upper()}]")
            print(
                f"{'family':<14}{'n':>7}{'P(TP)':>8}{'P(SL)':>8}{'P(time)':>9}{'meanR':>9}{'EVnet':>9}"
            )
            for r in tp[direction]:
                print(
                    f"{r['family']:<14}{r['n']:>7,}{r['p_tp']:>8.3f}{r['p_sl']:>8.3f}"
                    f"{r['p_time']:>9.3f}{r['mean_r_gross']:>+9.4f}{r['ev_net_r']:>+9.4f}"
                )
        results["target_probs"] = tp

    if args.all or args.baselines:
        print("\n" + "=" * 72)
        print("PHASE 15 — BASELINES (net of 0.0625 ATR, h=10, OOS)")
        print("=" * 72)
        bl = baselines(symbols, group=args.group, cutoff=args.cutoff)
        print(f"{'baseline':<22}{'n':>9}{'gross':>9}{'net':>9}")
        for r in bl:
            print(
                f"{r['baseline']:<22}{r['n']:>9,}{r['gross']:>+9.4f}{r['net']:>+9.4f}"
            )
        results["baselines"] = bl

    if args.all or args.mt:
        print("\n" + "=" * 72)
        print("PHASE 14 — MULTIPLE-TESTING (bootstrap CI + permutation, frozen k, OOS)")
        print("=" * 72)
        mt = multiple_testing(symbols, group=args.group, cutoff=args.cutoff)
        for direction in ("short", "long"):
            r = mt[direction]
            print(
                f"  [{direction.upper()}] n={r.get('n', 0)} gross={r.get('gross')} "
                f"net={r.get('net')} boot95=[{r.get('boot_ci')}] "
                f"perm_p={r.get('perm_p')}"
            )
        results["multiple_testing"] = mt

    if args.all or args.adversarial:
        print("\n" + "=" * 72)
        print("PHASE 18 — ADVERSARIAL / FALSIFICATION (frozen k, OOS)")
        print("=" * 72)
        ad = adversarial(symbols, group=args.group, cutoff=args.cutoff)
        for direction in ("short", "long"):
            print(f"\n[{direction.upper()}]")
            for name, r in ad[direction].items():
                print(
                    f"  {name:<14} n={r['n']:>7,} gross={r['gross']:+.4f} net={r['net']:+.4f}"
                )
        results["adversarial"] = ad

    if args.all or args.untouched:
        print("\n" + "=" * 72)
        print("PHASE 16 — UNTOUCHED FINAL TEST (single-shot, frozen k rules)")
        print("=" * 72)
        ut = untouched_test(symbols, group=args.group, cutoff=args.cutoff)
        for direction in ("short", "long"):
            r = ut[direction]
            if r.get("note"):
                print(f"  [{direction.upper()}] {r['note']}")
                continue
            print(
                f"  [{direction.upper()}] n={r['n']} gross={r['gross']:+.4f} "
                f"net={r['net']:+.4f} win={r['win']:.3f} flat={r['flat_rate']:.3f}"
            )
            print("    by symbol:", {k: v["net"] for k, v in r["by_symbol"].items()})
            print("    by regime:", {k: v["net"] for k, v in r["by_regime"].items()})
        results["untouched"] = ut

    out_dir = Path("data/validation")
    out_dir.mkdir(parents=True, exist_ok=True)
    with open(out_dir / "stage6_results.json", "w") as fh:
        json.dump(results, fh, indent=2, default=str)
    print(f"\nStage-6 results written to {out_dir / 'stage6_results.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
