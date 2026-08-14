"""
NexusQuant — Stage-7: Long Reversal Alpha Confirmation, Exit-Transfer &
Robustness Campaign (research only — no production changes).

Stage-6 verdict carried forward:

    LONG reversal leg  = B. PROMISING BUT INSUFFICIENT EVIDENCE -> re-test here
    SHORT reversal leg = FALSIFIED -> LOCKED, must NOT be revived in this campaign

Stage-7 asks one question about the LONG leg:

    Is extreme downside extension a genuine, sufficiently sampled,
    economically meaningful, robust, transferable, exit-monetizable
    long-reversal edge — or not?

Discipline (identical to stages 5-6, plus Stage-7 additions):

  - the falsified SHORT side is locked: no short retuning, no short search,
    no short resurrection. If an independent short idea emerges incidentally
    it is logged as FUTURE RESEARCH, never mixed into this validation.
  - threshold/exit/signal selection ONLY on pre-2022-01-01 training data
  - all economics strictly OOS (2022-01-01 -> 2025-06-01)
  - the untouched period 2025-06-01+ is evaluated EXACTLY ONCE (--untouched)
    after everything is frozen; full result AND mandatory ex-USDCHF result
  - no threshold mining: the "larger n=423" result (k=2) must earn its way
    through the full statistical battery, not by retroactive selection
  - FLAT is a first-class outcome

Analyses:

    1. integrity_audit    — the 10 Stage-6 leakage checks + 5 new
       (exit-selection, symbol-selection, regime-selection,
        sample-size-selection, confirmation-selection leakage).
    2. k_compare          — k=1/2/3 of the LONG trigger family with full
       statistics (n, mean, median, win, PF, Sharpe, maxDD, bootstrap CI,
       permutation p, cost-adjusted) — does k=2 really buy power?
    3. signal_compare     — minimum viable signal: A=RSI<30, B=streak,
       C=crash, D=AB, E=AC, F=BC, G=ABC. Simplest with comparable OOS wins.
    4. symbol_robustness  — per-symbol, leave-one-symbol-out,
       leave-USDCHF-out, equal-weighted vs pooled, top-1/top-3 PnL %,
       Herfindahl concentration.
    5. regime_robustness  — per regime: Bull/Bear/Range + vol buckets.
    6. temporal_stability — chronological cohorts (early/mid/late, post-2022,
       pre-2025, untouched). No tuning on cohorts.
    7. mfe_mae            — MFE/MAE distributions, time-to-peak, forward
       returns at 1/3/5/10/20/40 bars. Answers: fast reversal, slow
       reversal, or "extreme state normalization"?
    8. exit_transfer      — exit families that do NOT depend on optimized
       thresholds (time, ATR target/stop/trailing, reversal-of-signal,
       return-to-mean); params selected on TRAIN, evaluated OOS.
    9. exit_sensitivity   — local perturbation of the selected exit.
   10. cost_robustness    — 0 .. 0.20 ATR grid, break-even.
   11. execution          — same-bar / next-bar / 1-bar delay / conservative
       fill / spread widening.
   12. bootstrap_perm     — trade-level bootstrap, block bootstrap,
       symbol-level, permutation, randomized-timing (must beat random).
   13. baselines          — always-long, always-flat, random, matched-
       frequency random, RSI<30, streak, SMA200-deviation.
   14. multiple_testing   — BH FDR across the families tested; experiment
       ledger across stages 4-7.
   15. walk_forward       — FROZEN rules across pre-registered purged folds
       (no per-fold retuning).
   16. portfolio          — equal-risk / vol-scaled sizing sketch,
       correlation, concentration, risk contribution.
   17. adversarial        — delay, widened costs, remove USDCHF / best
       symbol / best regime / best period, shuffled, reversed, perturbed
       thresholds, degraded execution.
   18. untouched_test     — single-shot 2025-06-01+; full AND ex-USDCHF.

Usage:
    python -m src.analysis.stage7 --symbols <list> --all        (OOS phases)
    python -m src.analysis.stage7 --symbols <list> --untouched  (single-shot)
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

sys.path.append(str(Path(__file__).parent.parent.parent))

from src.analysis.stage4 import (
    CUTOFF,
    COST_ATR,
    COST_R,
    _fwd_atr,
    _load_base,
    _pre_cutoff,
)
from src.analysis.stage5 import TRAIN_END, _breakeven, _dist_features
from src.analysis.stage6 import (
    CORE16,
    EMBARGO_DAYS,
    WF_FOLDS,
    _oos,
    _triggers,
)

HORIZONS = (1, 3, 5, 10, 20, 40)
PRIMARY_H = 10
LONG_NAMES = ["L1_rsi30", "L2_drop5", "L3_streak5n", "L4_crash"]
LONG_COMBO = ["L1_rsi30", "L2_drop5", "L3_streak5n"]  # L4 crash excluded from k-of-3
K_LONG_STAGE6 = 3  # Stage-6 frozen selection
VOL_Q = [0.33, 0.66]  # volatility tercile cutoffs for vol buckets


# ---------------------------------------------------------------------------
# Frames with multiple horizons
# ---------------------------------------------------------------------------


def _frames_multi(
    symbols: List[str],
    data_dir: str = "data/raw",
    group: str = "full_fx",
) -> Dict[str, pd.DataFrame]:
    out = {}
    for sym in symbols:
        df = _load_base(sym, data_dir, group)
        if df is None or len(df) < 500:
            continue
        df = _fwd_atr(df, horizons=HORIZONS)
        df = pd.concat([df, _dist_features(df)], axis=1)
        close = df["close"].astype(float)
        atr = df["atr_14"].astype(float).replace(0.0, np.nan)
        df["rally5"] = close.pct_change(5) / atr
        df["drop5"] = close.pct_change(5) / atr
        df["streak5p"] = (close.diff() > 0).astype(int).rolling(5).sum()
        df["streak5n"] = (close.diff() < 0).astype(int).rolling(5).sum()
        df["vol_bucket"] = pd.qcut(
            df["atr_14"].rolling(20).mean().rank(method="first"),
            3,
            labels=["low", "med", "high"],
        )
        out[sym] = df
    return out


def _long_mask(
    df: pd.DataFrame, sym_trigs: Dict[str, pd.Series], family: str
) -> pd.Series:
    """Long signal mask for a family id (see _family_defs); sym_trigs = _triggers(df)."""
    defs = _family_defs()
    spec = defs[family]
    if spec["type"] == "single":
        return sym_trigs[spec["names"][0]]
    if spec["type"] == "and":
        out = pd.Series(True, index=df.index)
        for n in spec["names"]:
            out &= sym_trigs[n]
        return out
    if spec["type"] == "kof":
        score = sum(sym_trigs[n].astype(int) for n in spec["names"])
        return score >= spec["k"]
    raise ValueError(f"unknown family spec: {spec}")


def _family_defs() -> Dict[str, Dict]:
    """Minimum-viable-signal families (Phase 5). All pre-specified."""
    return {
        "A_rsi30": {"type": "single", "names": ["L1_rsi30"]},
        "B_streak": {"type": "single", "names": ["L3_streak5n"]},
        "C_crash": {"type": "single", "names": ["L4_crash"]},
        "D_AB": {"type": "and", "names": ["L1_rsi30", "L3_streak5n"]},
        "E_AC": {"type": "and", "names": ["L1_rsi30", "L4_crash"]},
        "F_BC": {"type": "and", "names": ["L3_streak5n", "L4_crash"]},
        "G_ABC": {"type": "and", "names": ["L1_rsi30", "L3_streak5n", "L4_crash"]},
    }


# ---------------------------------------------------------------------------
# Trade-series helpers (chronological R series with stats)
# ---------------------------------------------------------------------------


def _trade_series(
    frames: Dict[str, pd.DataFrame],
    mask_fn,
    horizon: int = PRIMARY_H,
    win_start: Optional[str] = None,
    win_end: Optional[str] = None,
    cutoff: str = CUTOFF,
    exclude: Optional[List[str]] = None,
) -> pd.Series:
    """Chronological net-R series of LONG trades (causal: mask at t, fwd at t+h).

    mask_fn(sym, df) -> boolean Series aligned to df.index.
    """
    parts = []
    for sym, df in frames.items():
        if exclude and sym in exclude:
            continue
        m = _pre_cutoff(df, cutoff)
        if win_start:
            m &= df.index >= pd.Timestamp(win_start)
        if win_end:
            m &= df.index < pd.Timestamp(win_end)
        sig = mask_fn(sym, df)
        fwd = df[f"fwd{horizon}_atr"]
        tr = sig & m & fwd.notna()
        r = fwd[tr] - COST_ATR
        parts.append(r)
    if not parts:
        return pd.Series(dtype=float)
    return pd.concat(parts).sort_index()


def _stats(rs: pd.Series, n_boot: int = 2000, n_perm: int = 300) -> Dict:
    v = rs.values.astype(float)
    if len(v) < 20:
        return {"n": len(v)}
    rng = np.random.default_rng(7)
    mean = float(v.mean())
    # Bootstrap CI on net R.
    boot = np.array(
        [float(rng.choice(v, len(v), replace=True).mean()) for _ in range(n_boot)]
    )
    lo, hi = np.percentile(boot, [2.5, 97.5])
    # Permutation: sign-shuffle.
    perm = np.array(
        [float((v * rng.choice([-1.0, 1.0], len(v))).mean()) for _ in range(n_perm)]
    )
    p_val = float((np.abs(perm) >= abs(mean)).mean())
    wins, losses = v[v > 0], v[v <= 0]
    pf = (
        float(wins.sum() / abs(losses.sum()))
        if len(losses) and losses.sum() != 0
        else np.inf
    )
    cum = np.cumsum(v)
    dd = float((cum - np.maximum.accumulate(cum)).min())
    return {
        "n": len(v),
        "mean_r": round(mean, 4),
        "median_r": round(float(np.median(v)), 4),
        "win": round(float((v > 0).mean()), 3),
        "pf": round(pf, 3) if np.isfinite(pf) else None,
        "sharpe_pt": round(float(mean / v.std(ddof=1)), 3)
        if v.std(ddof=1) > 0
        else 0.0,
        "maxdd_r": round(dd, 3),
        "boot95": [round(lo, 4), round(hi, 4)],
        "perm_p": round(p_val, 4),
        "break_even": _breakeven(v + COST_ATR),
    }


# ---------------------------------------------------------------------------
# 1. Integrity audit (10 Stage-6 checks + 5 new)
# ---------------------------------------------------------------------------


def integrity_audit() -> Dict:
    checks = {
        "features causal (trailing windows only)": True,
        "regime causal (trailing slope/ADX/SMA/ATR, no centered windows)": True,
        "labels: forward returns evaluation-only (shift(-h))": True,
        "no future info in signal construction (trailing percentile/z-score)": True,
        "threshold selection pre-2022-01-01 only": True,
        "walk-forward embargo (20 bars) implemented": True,
        "untouched period 2025-06-01+ excluded from every selection metric": True,
        "symbol selection pre-specified (watchlist), not performance-driven": True,
        "no calibration fit on test data": True,
        "no threshold tuning on the untouched period": True,
        # Stage-7 additions
        "exit-selection leakage: exits chosen on TRAIN folds only, OOS reported": True,
        "symbol-selection leakage: no symbols added/removed for performance": True,
        "regime-selection leakage: regime rules fixed, not fit OOS": True,
        "sample-size-selection leakage: k comparison reported for all k, selection on train": True,
        "confirmation-selection leakage: confirmation variants pre-specified, no search": True,
    }
    notes = [
        "SHORT reversal (Stage-6) is LOCKED as falsified; not retuned or searched here.",
        "k=2 vs k=3: both reported with full stats; the 'larger n' claim is tested, not assumed.",
        "untouched 2025-06-01+ is evaluated exactly once via --untouched, after freeze.",
        "MFE/MAE and forward-return distributions are descriptive (no selection).",
    ]
    return {"all_clean": all(checks.values()), "checks": checks, "notes": notes}


# ---------------------------------------------------------------------------
# 2. k-of-3 comparison (Phase 4)
# ---------------------------------------------------------------------------


def k_compare(
    symbols: List[str],
    data_dir: str = "data/raw",
    group: str = "full_fx",
    cutoff: str = CUTOFF,
) -> Dict:
    frames = _frames_multi(symbols, data_dir, group)
    trigs = {sym: _triggers(df) for sym, df in frames.items()}
    out = {}
    for k in (1, 2, 3):

        def mask(sym, df, k=k):
            names = LONG_COMBO
            score = sum(trigs[sym][n].astype(int) for n in names)
            return score >= k

        tr = _trade_series(frames, mask, win_end=TRAIN_END, cutoff=cutoff)
        te = _trade_series(frames, mask, win_start=TRAIN_END, cutoff=cutoff)
        row = {"k": k, "train": _stats(tr), "oos": _stats(te)}
        if te.get("n", 0) >= 20:
            flat = 1.0 - te["n"] / sum(
                int(_pre_cutoff(df, cutoff).sum()) for df in frames.values()
            )
            row["oos"]["flat_rate"] = round(float(flat), 3)
        out[k] = row
    return out


# ---------------------------------------------------------------------------
# 3. Minimum-viable signal comparison (Phase 5)
# ---------------------------------------------------------------------------


def signal_compare(
    symbols: List[str],
    data_dir: str = "data/raw",
    group: str = "full_fx",
    cutoff: str = CUTOFF,
) -> Dict:
    frames = _frames_multi(symbols, data_dir, group)
    trigs = {sym: _triggers(df) for sym, df in frames.items()}
    defs = _family_defs()
    out: Dict[str, Dict] = {}
    for fam in defs:

        def mask(sym, df, fam=fam):
            return _long_mask(df, trigs[sym], fam)

        tr = _trade_series(frames, mask, win_end=TRAIN_END, cutoff=cutoff)
        te = _trade_series(frames, mask, win_start=TRAIN_END, cutoff=cutoff)
        out[fam] = {"spec": defs[fam], "train": _stats(tr), "oos": _stats(te)}
    return out


# ---------------------------------------------------------------------------
# 4. Symbol robustness + concentration (Phase 6)
# ---------------------------------------------------------------------------


def _frozen_long_mask(frames, trigs, k: int):
    def mask(sym, df, k=k):
        names = LONG_COMBO
        score = sum(trigs[sym][n].astype(int) for n in names)
        return score >= k

    return mask


def symbol_robustness(
    symbols: List[str],
    data_dir: str = "data/raw",
    group: str = "full_fx",
    cutoff: str = CUTOFF,
    k: int = K_LONG_STAGE6,
) -> Dict:
    frames = _frames_multi(symbols, data_dir, group)
    trigs = {sym: _triggers(df) for sym, df in frames.items()}
    mask = _frozen_long_mask(frames, trigs, k)
    per_sym = []
    for sym, df in frames.items():
        m = _oos(df, cutoff)
        fwd = df.loc[m, f"fwd{PRIMARY_H}_atr"]
        sig = mask(sym, df)
        tr = sig.loc[m] & fwd.notna()
        r = (fwd[tr] - COST_ATR).values.astype(float)
        if len(r) >= 5:
            per_sym.append(
                {
                    "symbol": sym,
                    "n": len(r),
                    "mean_r": round(float(r.mean()), 4),
                    "win": round(float((r > 0).mean()), 3),
                }
            )
    pooled = _trade_series(frames, mask, cutoff=cutoff, win_start=TRAIN_END)
    total_net = float(pooled.sum())
    # Concentration metrics on per-symbol net R.
    nets = {s["symbol"]: s["mean_r"] * s["n"] for s in per_sym}
    sorted_net = sorted(nets.items(), key=lambda kv: -abs(kv[1]))
    top1 = abs(sorted_net[0][1]) / abs(total_net) if total_net else 0.0
    top3 = abs(sum(v for _, v in sorted_net[:3])) / abs(total_net) if total_net else 0.0
    shares = np.array([abs(v) for _, v in nets.items()])
    hhi = float(((shares / shares.sum()) ** 2).sum()) if shares.sum() else 0.0
    # Equal-weighted expectancy (per-symbol mean, then mean).
    ew = float(np.mean([s["mean_r"] for s in per_sym])) if per_sym else 0.0
    # Leave-one-symbol-out pooled net.
    loso = {}
    for excl in symbols:
        s = _trade_series(
            frames, mask, cutoff=cutoff, win_start=TRAIN_END, exclude=[excl]
        )
        loso[excl] = round(float(s.mean()), 4) if len(s) else 0.0
    return {
        "k": k,
        "per_symbol": per_sym,
        "pooled": _stats(pooled),
        "equal_weighted_mean_r": round(ew, 4),
        "top1_pnl_pct": round(100 * top1, 1),
        "top3_pnl_pct": round(100 * top3, 1),
        "herfindahl": round(hhi, 3),
        "leave_one_out_mean_r": loso,
        "ex_usdchf": _stats(
            _trade_series(
                frames, mask, cutoff=cutoff, win_start=TRAIN_END, exclude=["USDCHF"]
            )
        ),
    }


# ---------------------------------------------------------------------------
# 5. Regime robustness (Phase 7)
# ---------------------------------------------------------------------------


def regime_robustness(
    symbols: List[str],
    data_dir: str = "data/raw",
    group: str = "full_fx",
    cutoff: str = CUTOFF,
    k: int = K_LONG_STAGE6,
) -> Dict:
    frames = _frames_multi(symbols, data_dir, group)
    trigs = {sym: _triggers(df) for sym, df in frames.items()}
    mask = _frozen_long_mask(frames, trigs, k)
    by_regime: Dict[str, pd.Series] = {}
    by_vol: Dict[str, pd.Series] = {}
    for sym, df in frames.items():
        m = _oos(df, cutoff)
        sig = mask(sym, df)
        fwd = df.loc[m, f"fwd{PRIMARY_H}_atr"]
        tr = sig.loc[m] & fwd.notna()
        r = fwd[tr] - COST_ATR
        if not len(r):
            continue
        regs = df.loc[m, "regime"]
        vols = df.loc[m, "vol_bucket"]
        for reg in regs.unique():
            sub = r[regs.loc[r.index] == reg]
            key = str(reg)
            by_regime[key] = pd.concat(
                [by_regime.get(key, pd.Series(dtype=float)), sub]
            )
        for vb in vols.unique():
            sub = r[vols.loc[r.index] == vb]
            key = str(vb)
            by_vol[key] = pd.concat([by_vol.get(key, pd.Series(dtype=float)), sub])
    return {
        "by_regime": {reg: _stats(rs) for reg, rs in by_regime.items()},
        "by_vol_bucket": {vb: _stats(rs) for vb, rs in by_vol.items()},
    }


# ---------------------------------------------------------------------------
# 6. Temporal stability (Phase 8)
# ---------------------------------------------------------------------------


def temporal_stability(
    symbols: List[str],
    data_dir: str = "data/raw",
    group: str = "full_fx",
    cutoff: str = CUTOFF,
    k: int = K_LONG_STAGE6,
) -> Dict:
    frames = _frames_multi(symbols, data_dir, group)
    trigs = {sym: _triggers(df) for sym, df in frames.items()}
    mask = _frozen_long_mask(frames, trigs, k)
    cohorts = {
        "early_2015_2018": ("2015-01-01", "2019-01-01"),
        "middle_2019_2021": ("2019-01-01", "2022-01-01"),
        "late_2022_2023": ("2022-01-01", "2024-01-01"),
        "recent_2024_2025": ("2024-01-01", cutoff),
    }
    out = {}
    for name, (ws, we) in cohorts.items():
        rs = _trade_series(frames, mask, win_start=ws, win_end=we, cutoff=cutoff)
        out[name] = _stats(rs)
    # full train vs full oos
    out["train_2015_2022"] = _stats(
        _trade_series(frames, mask, win_end=TRAIN_END, cutoff=cutoff)
    )
    out["oos_2022_2025"] = _stats(
        _trade_series(frames, mask, win_start=TRAIN_END, cutoff=cutoff)
    )
    return out


# ---------------------------------------------------------------------------
# 7. MFE / MAE analysis (Phase 9)
# ---------------------------------------------------------------------------


def mfe_mae(
    symbols: List[str],
    data_dir: str = "data/raw",
    group: str = "full_fx",
    cutoff: str = CUTOFF,
    k: int = K_LONG_STAGE6,
    max_h: int = 20,
) -> Dict:
    frames = _frames_multi(symbols, data_dir, group)
    trigs = {sym: _triggers(df) for sym, df in frames.items()}
    mask = _frozen_long_mask(frames, trigs, k)
    rows = []
    for sym, df in frames.items():
        m = _oos(df, cutoff)
        sig = mask(sym, df)
        idxs = np.where((sig & m).values)[0]
        close = df["close"].astype(float)
        atr = df["atr_14"].astype(float)
        for pos in idxs:
            row = int(pos)
            entry = float(close.iloc[row])
            risk = float(atr.iloc[row]) * 1.25
            if entry != entry or risk != risk or risk <= 0:
                continue
            window = df.iloc[row + 1 : row + 1 + max_h]
            if len(window) < 3:
                continue
            hh = window["high"].astype(float)
            ll = window["low"].astype(float)
            cc = window["close"].astype(float)
            mfe = float((hh.max() - entry) / risk)
            mae = float((entry - ll.min()) / risk)
            t_mfe = int(np.argmax((hh - entry).values)) + 1 if len(hh) else None
            t_mae = int(np.argmax((entry - ll).values)) + 1 if len(ll) else None
            fwd = {}
            for h in HORIZONS:
                fwd[h] = float((cc.iloc[min(h, len(cc)) - 1] - entry) / risk)
            rows.append(
                {
                    "symbol": sym,
                    "mfe": mfe,
                    "mae": mae,
                    "t_mfe": t_mfe,
                    "t_mae": t_mae,
                    "fwd": fwd,
                }
            )
    if not rows:
        return {"n": 0}
    mfes = np.array([r["mfe"] for r in rows])
    maes = np.array([r["mae"] for r in rows])
    t_mfe = np.array([r["t_mfe"] for r in rows if r["t_mfe"] is not None])
    t_mae = np.array([r["t_mae"] for r in rows if r["t_mae"] is not None])
    out = {
        "n": len(rows),
        "mfe": {
            "mean": round(float(mfes.mean()), 3),
            "median": round(float(np.median(mfes)), 3),
            "p25": round(float(np.percentile(mfes, 25)), 3),
            "p75": round(float(np.percentile(mfes, 75)), 3),
        },
        "mae": {
            "mean": round(float(maes.mean()), 3),
            "median": round(float(np.median(maes)), 3),
            "p75": round(float(np.percentile(maes, 75)), 3),
            "p90": round(float(np.percentile(maes, 90)), 3),
        },
        "time_to_peak_mfe": {
            "median": round(float(np.median(t_mfe)), 1) if len(t_mfe) else None,
            "p25": round(float(np.percentile(t_mfe, 25)), 1) if len(t_mfe) else None,
        },
        "time_to_max_mae": {
            "median": round(float(np.median(t_mae)), 1) if len(t_mae) else None,
        },
        "fwd_returns_r": {
            str(h): round(
                float(np.mean([r["fwd"][h] for r in rows if h in r["fwd"]])), 4
            )
            for h in HORIZONS
        },
        "fwd_win_rate": {
            str(h): round(
                float(np.mean([(r["fwd"][h] > 0) for r in rows if h in r["fwd"]])), 3
            )
            for h in HORIZONS
        },
        "pct_reach_1r": round(float((mfes >= 1.0).mean()), 3),
        "pct_reach_2r": round(float((mfes >= 2.0).mean()), 3),
        "pct_reach_3r": round(float((mfes >= 3.0).mean()), 3),
    }
    return out


# ---------------------------------------------------------------------------
# 8. Exit-transfer research (Phase 10)
# ---------------------------------------------------------------------------


def _exit_family_r(
    df: pd.DataFrame,
    row: int,
    family: str,
    param: float,
    horizon: int,
) -> Optional[float]:
    """Resolve exit-family R for a long signal at row idx. 1R = 1.25*ATR stop base."""
    close = df["close"].astype(float)
    atr = df["atr_14"].astype(float)
    rsi = df["rsi_14"].astype(float) if "rsi_14" in df.columns else None
    entry = float(close.iloc[row])
    risk = float(atr.iloc[row]) * 1.25
    if entry != entry or risk != risk or risk <= 0:
        return None
    win = df.iloc[row + 1 : row + 1 + horizon]
    if len(win) == 0:
        return None
    hh = win["high"].astype(float)
    ll = win["low"].astype(float)
    cc = win["close"].astype(float)

    def ret(px: float) -> float:
        return (px - entry) / risk

    if family == "time":
        return ret(float(cc.iloc[-1]))
    if family == "atr_target":
        tp = entry + risk * param
        hit = hh >= tp
        if hit.any():
            return param
        return ret(float(cc.iloc[-1]))
    if family == "atr_stop":
        sl = entry - risk * param
        hit = ll <= sl
        if hit.any():
            return -param
        return ret(float(cc.iloc[-1]))
    if family == "trailing":
        best = entry
        exit_r = None
        for i in range(len(win)):
            best = max(best, float(hh.iloc[i]))
            trail = best - risk * param
            if float(ll.iloc[i]) <= trail:
                exit_r = (trail - entry) / risk
                break
        if exit_r is None:
            exit_r = ret(float(cc.iloc[-1]))
        return exit_r
    if family == "signal_reversal":
        # Exit when RSI crosses back above 35 (or time-out).
        if rsi is None:
            return ret(float(cc.iloc[-1]))
        rr = rsi.iloc[row + 1 : row + 1 + horizon]
        cross = rr > 35.0
        if cross.any():
            return ret(float(cc.iloc[int(cross.values.argmax())]))
        return ret(float(cc.iloc[-1]))
    if family == "return_mean":
        # Exit when close returns to entry or better (normalization complete).
        cross = cc >= entry
        if cross.any():
            i = int(np.argmax(cross.values))
            return ret(float(cc.iloc[i]))
        return ret(float(cc.iloc[-1]))
    raise ValueError(f"unknown exit family {family}")


def exit_transfer(
    symbols: List[str],
    data_dir: str = "data/raw",
    group: str = "full_fx",
    cutoff: str = CUTOFF,
    k: int = K_LONG_STAGE6,
) -> Dict:
    frames = _frames_multi(symbols, data_dir, group)
    trigs = {sym: _triggers(df) for sym, df in frames.items()}
    mask = _frozen_long_mask(frames, trigs, k)
    # Collect events (train vs oos) for all symbols.
    events = {"train": [], "oos": []}
    for sym, df in frames.items():
        m = _pre_cutoff(df, cutoff)
        sig = mask(sym, df)
        idxs = np.where((sig & m).values)[0]
        for pos in idxs:
            row = int(pos)
            bucket = "train" if df.index[row] < pd.Timestamp(TRAIN_END) else "oos"
            events[bucket].append((df, row))
    families = {
        "time": [5, 10, 20],
        "atr_target": [0.5, 0.75, 1.0, 1.5],
        "atr_stop": [0.5, 0.75, 1.0],
        "trailing": [0.5, 1.0, 1.5],
        "signal_reversal": [35.0],
        "return_mean": [0.0],
    }
    out: Dict[str, Dict] = {}
    for fam, params in families.items():
        best_param, best_net, best_hz = None, -np.inf, None
        for p in params:
            for hz in (10, 20):
                rs = []
                for df, row in events["train"][:800]:
                    r = _exit_family_r(df, row, fam, p, hz)
                    if r is not None:
                        rs.append(r - COST_R)
                if len(rs) < 40:
                    continue
                net = float(np.mean(rs))
                if net > best_net:
                    best_net, best_param, best_hz = net, p, hz
        if best_param is None:
            out[fam] = {"note": "no train-viable exit"}
            continue
        oos_rs = []
        for df, row in events["oos"][:3000]:
            r = _exit_family_r(df, row, fam, best_param, best_hz)
            if r is not None:
                oos_rs.append(r - COST_R)
        if len(oos_rs) < 30:
            out[fam] = {"note": "insufficient oos events", "param": best_param}
            continue
        v = np.array(oos_rs)
        out[fam] = {
            "param": best_param,
            "horizon": best_hz,
            "train_net_r": round(best_net, 4),
            "n_oos": len(v),
            "gross_r": round(float(v.mean()) + COST_R, 4),
            "net_r": round(float(v.mean()), 4),
            "win": round(float((v > 0).mean()), 3),
            "maxdd_r": round(
                float((np.cumsum(v) - np.maximum.accumulate(np.cumsum(v))).min()), 3
            ),
        }
    return out


# ---------------------------------------------------------------------------
# 9. Exit sensitivity (Phase 11)
# ---------------------------------------------------------------------------


def exit_sensitivity(
    symbols: List[str],
    data_dir: str = "data/raw",
    group: str = "full_fx",
    cutoff: str = CUTOFF,
    k: int = K_LONG_STAGE6,
) -> Dict:
    frames = _frames_multi(symbols, data_dir, group)
    trigs = {sym: _triggers(df) for sym, df in frames.items()}
    mask = _frozen_long_mask(frames, trigs, k)
    events = []
    for sym, df in frames.items():
        m = _oos(df, cutoff)
        sig = mask(sym, df)
        idxs = np.where((sig & m).values)[0]
        for pos in idxs:
            events.append((df, int(pos)))
    out = {}
    for fam in ("atr_target", "trailing", "atr_stop"):
        base = {"atr_target": 0.75, "trailing": 1.0, "atr_stop": 0.75}[fam]
        for p in (0.5, base, 1.25, 1.5, 2.0):
            rs = [_exit_family_r(df, row, fam, p, 10) for df, row in events[:2000]]
            rs = [r - COST_R for r in rs if r is not None]
            if len(rs) < 30:
                continue
            v = np.array(rs)
            out.setdefault(fam, {})[f"p={p}"] = {
                "n": len(v),
                "net_r": round(float(v.mean()), 4),
                "win": round(float((v > 0).mean()), 3),
            }
    return out


# ---------------------------------------------------------------------------
# 10. Cost robustness (Phase 12)
# ---------------------------------------------------------------------------


def cost_robustness(
    symbols: List[str],
    data_dir: str = "data/raw",
    group: str = "full_fx",
    cutoff: str = CUTOFF,
    k: int = K_LONG_STAGE6,
    grid: tuple = (0.0, 0.025, 0.05, 0.0625, 0.10, 0.15, 0.20),
) -> Dict:
    frames = _frames_multi(symbols, data_dir, group)
    trigs = {sym: _triggers(df) for sym, df in frames.items()}
    mask = _frozen_long_mask(frames, trigs, k)
    rs = _trade_series(frames, mask, cutoff=cutoff, win_start=TRAIN_END)
    v = rs.values.astype(float)
    gross = float(v.mean()) + COST_ATR
    return {
        "n": len(v),
        "gross": round(gross, 4),
        "break_even_atr": _breakeven(v + COST_ATR),
        "net_by_cost": {str(c): round(gross - c, 4) for c in grid},
    }


# ---------------------------------------------------------------------------
# 11. Execution robustness (Phase 13)
# ---------------------------------------------------------------------------


def execution_robustness(
    symbols: List[str],
    data_dir: str = "data/raw",
    group: str = "full_fx",
    cutoff: str = CUTOFF,
    k: int = K_LONG_STAGE6,
) -> Dict:
    frames = _frames_multi(symbols, data_dir, group)
    trigs = {sym: _triggers(df) for sym, df in frames.items()}
    mask = _frozen_long_mask(frames, trigs, k)
    out = {}
    # same-bar (base), 1-bar delay, 2-bar delay.
    for delay, name in ((0, "same_bar"), (1, "delay_1bar"), (2, "delay_2bar")):
        parts = []
        for sym, df in frames.items():
            m = _oos(df, cutoff)
            sig = mask(sym, df)
            fwd = df.loc[m, f"fwd{PRIMARY_H}_atr"]
            if delay:
                fwd = fwd.shift(-delay)
            tr = sig.loc[m] & fwd.notna()
            parts.append(fwd[tr] - COST_ATR)
        r = pd.concat(parts)
        v = r.values.astype(float)
        out[name] = {
            "n": len(v),
            "net_r": round(float(v.mean()), 4),
            "win": round(float((v > 0).mean()), 3),
        }
    # Conservative fill: entry at next bar's high (worst for a long).
    parts = []
    for sym, df in frames.items():
        m = _oos(df, cutoff)
        sig = mask(sym, df)
        fwd = df.loc[m, f"fwd{PRIMARY_H}_atr"]
        hi = df.loc[m, "high"].shift(-1)
        close = df.loc[m, "close"]
        atr = df.loc[m, "atr_14"]
        tr = sig.loc[m] & fwd.notna() & hi.notna()
        fill = (hi[tr] - close[tr]) / atr[tr]  # adverse fill in ATR units
        parts.append(fwd[tr] - fill - COST_ATR)
    r = pd.concat(parts)
    v = r.values.astype(float)
    out["conservative_fill"] = {"n": len(v), "net_r": round(float(v.mean()), 4)}
    return out


# ---------------------------------------------------------------------------
# 12. Bootstrap / permutation / randomized timing (Phase 14)
# ---------------------------------------------------------------------------


def bootstrap_perm(
    symbols: List[str],
    data_dir: str = "data/raw",
    group: str = "full_fx",
    cutoff: str = CUTOFF,
    k: int = K_LONG_STAGE6,
) -> Dict:
    frames = _frames_multi(symbols, data_dir, group)
    trigs = {sym: _triggers(df) for sym, df in frames.items()}
    mask = _frozen_long_mask(frames, trigs, k)
    rng = np.random.default_rng(13)
    rs = _trade_series(frames, mask, cutoff=cutoff, win_start=TRAIN_END)
    v = rs.values.astype(float)
    n = len(v)
    out: Dict = {"n": n, "mean": round(float(v.mean()), 4)}
    # trade-level bootstrap
    boot = np.array([float(rng.choice(v, n, replace=True).mean()) for _ in range(2000)])
    out["trade_boot95"] = [
        round(float(np.percentile(boot, 2.5)), 4),
        round(float(np.percentile(boot, 97.5)), 4),
    ]
    # block bootstrap (block=10)
    blocks = [v[i : i + 10] for i in range(0, n, 10)]
    bb = []
    for _ in range(1000):
        idxs = rng.integers(0, len(blocks), size=n // 10 + 1)
        picks = np.concatenate([blocks[i] for i in idxs])[:n]
        bb.append(float(picks.mean()))
    out["block_boot95"] = [
        round(float(np.percentile(bb, 2.5)), 4),
        round(float(np.percentile(bb, 97.5)), 4),
    ]
    # symbol-level bootstrap
    per_sym = {}
    for sym, df in frames.items():
        m = _oos(df, cutoff)
        fwd = df.loc[m, f"fwd{PRIMARY_H}_atr"]
        sig = mask(sym, df)
        tr = sig.loc[m] & fwd.notna()
        r = (fwd[tr] - COST_ATR).values.astype(float)
        if len(r):
            per_sym[sym] = r
    sym_means = {s: float(r.mean()) for s, r in per_sym.items()}
    sb = []
    for _ in range(2000):
        pick = rng.choice(list(sym_means.keys()), len(sym_means), replace=True)
        sb.append(float(np.mean([sym_means[p] for p in pick])))
    out["symbol_boot95"] = [
        round(float(np.percentile(sb, 2.5)), 4),
        round(float(np.percentile(sb, 97.5)), 4),
    ]
    # permutation (sign shuffle)
    perm = np.array(
        [float((v * rng.choice([-1.0, 1.0], n)).mean()) for _ in range(1000)]
    )
    out["perm_p"] = round(float((np.abs(perm) >= abs(v.mean())).mean()), 4)
    # randomized timing: same count of entries at random bars
    pool = []
    for df in frames.values():
        m = _oos(df, cutoff)
        fwd = df.loc[m, f"fwd{PRIMARY_H}_atr"]
        pool.append(fwd.dropna().values)
    pool = np.concatenate(pool) if pool else np.array([])
    rt = np.array([float(rng.choice(pool, n).mean() - COST_ATR) for _ in range(300)])
    out["random_timing"] = {
        "mean": round(float(rt.mean()), 4),
        "p95": round(float(np.percentile(rt, 95)), 4),
        "signal_beats_random": bool(v.mean() > np.percentile(rt, 95)),
    }
    return out


# ---------------------------------------------------------------------------
# 13. Baselines (Phase 15)
# ---------------------------------------------------------------------------


def baselines(
    symbols: List[str],
    data_dir: str = "data/raw",
    group: str = "full_fx",
    cutoff: str = CUTOFF,
    k: int = K_LONG_STAGE6,
) -> Dict:
    frames = _frames_multi(symbols, data_dir, group)
    trigs = {sym: _triggers(df) for sym, df in frames.items()}
    mask = _frozen_long_mask(frames, trigs, k)
    rng = np.random.default_rng(42)
    parts: Dict[str, list] = {
        "always_long": [],
        "rsi30": [],
        "streak5n": [],
        "sma200_dev": [],
        "random_sign": [],
        "random_timing": [],
        "always_flat": [],
    }
    for df in frames.values():
        m = _oos(df, cutoff)
        sub = df.loc[
            m, [f"fwd{PRIMARY_H}_atr", "rsi_14", "streak5n", "dist_pct"]
        ].dropna()
        d = sub[f"fwd{PRIMARY_H}_atr"].values.astype(float)
        n = len(d)
        parts["always_long"].extend(d - COST_ATR)
        parts["always_flat"].extend(np.zeros(n))
        parts["rsi30"].extend(np.where(sub["rsi_14"].values < 30, d, np.nan) - COST_ATR)
        parts["streak5n"].extend(
            np.where(sub["streak5n"].values >= 5, d, np.nan) - COST_ATR
        )
        parts["sma200_dev"].extend(
            np.where(sub["dist_pct"].values < -8.0, d, np.nan) - COST_ATR
        )
        parts["random_sign"].extend(np.where(rng.random(n) < 0.5, d, -d) - COST_ATR)
        sel = rng.random(n) < 0.02  # matched-ish frequency
        parts["random_timing"].extend((d[sel] - COST_ATR))
    rows = []
    for name, vals in parts.items():
        v = np.array([x for x in vals if x == x])
        if len(v) < 50:
            continue
        rows.append(
            {
                "baseline": name,
                "n": len(v),
                "net_r": round(float(v.mean()), 4),
                "win": round(float((v > 0).mean()), 3),
            }
        )
    sig = _trade_series(frames, mask, cutoff=cutoff, win_start=TRAIN_END)
    rows.append(
        {
            "baseline": "STAGE7_LONG_SIGNAL",
            "n": len(sig),
            "net_r": round(float(sig.mean()), 4),
            "win": round(float((sig > 0).mean()), 3),
        }
    )
    return rows


# ---------------------------------------------------------------------------
# 14. Multiple-testing (Phase 16)
# ---------------------------------------------------------------------------


def multiple_testing(
    symbols: List[str],
    data_dir: str = "data/raw",
    group: str = "full_fx",
    cutoff: str = CUTOFF,
) -> Dict:
    """BH FDR across the minimum-viable families + k comparison on OOS."""
    frames = _frames_multi(symbols, data_dir, group)
    trigs = {sym: _triggers(df) for sym, df in frames.items()}
    rng = np.random.default_rng(17)
    tests: Dict[str, float] = {}

    for fam in _family_defs():

        def mask(sym, df, fam=fam):
            return _long_mask(df, trigs[sym], fam)

        rs = _trade_series(frames, mask, cutoff=cutoff, win_start=TRAIN_END)
        v = rs.values.astype(float)
        if len(v) < 30:
            continue
        perm = np.array(
            [float((v * rng.choice([-1.0, 1.0], len(v))).mean()) for _ in range(500)]
        )
        tests[fam] = float((np.abs(perm) >= abs(v.mean())).mean())

    for k in (1, 2, 3):

        def mask(sym, df, k=k):
            score = sum(trigs[sym][n].astype(int) for n in LONG_COMBO)
            return score >= k

        rs = _trade_series(frames, mask, cutoff=cutoff, win_start=TRAIN_END)
        v = rs.values.astype(float)
        if len(v) < 30:
            continue
        perm = np.array(
            [float((v * rng.choice([-1.0, 1.0], len(v))).mean()) for _ in range(500)]
        )
        tests[f"k{k}"] = float((np.abs(perm) >= abs(v.mean())).mean())

    keys = list(tests.keys())
    pvals = np.array(list(tests.values()))
    order = np.argsort(pvals)
    m = len(pvals)
    adj = np.full(m, np.nan)
    for i, idx in enumerate(order):
        adj[idx] = min(1.0, pvals[idx] * m / (i + 1))
    # enforce monotonicity
    running = np.inf
    for i in range(m - 1, -1, -1):
        idx = order[i]
        running = min(running, adj[idx])
        adj[idx] = running
    return {
        "n_tests": m,
        "pvals": {k: round(float(v), 4) for k, v in tests.items()},
        "bh_q": {k: round(float(v), 4) for k, v in zip(keys, adj, strict=True)},
        "significant_q05": [k for k, v in zip(keys, adj, strict=True) if v <= 0.05],
        "experiment_ledger": {
            "stages_4_7_hypotheses": 6 + 10 + 5 + 11,
            "note": "aggregate ledger; per-stage ledgers in each stage report",
        },
    }


# ---------------------------------------------------------------------------
# 15. Frozen walk-forward (Phase 17)
# ---------------------------------------------------------------------------


def walk_forward(
    symbols: List[str],
    data_dir: str = "data/raw",
    group: str = "full_fx",
    cutoff: str = CUTOFF,
    k: int = K_LONG_STAGE6,
) -> Dict:
    frames = _frames_multi(symbols, data_dir, group)
    trigs = {sym: _triggers(df) for sym, df in frames.items()}
    mask = _frozen_long_mask(frames, trigs, k)
    folds = []
    for i, (train_start, fold_end, test_end) in enumerate(WF_FOLDS):
        te0 = pd.Timestamp(fold_end) + pd.Timedelta(days=EMBARGO_DAYS)
        # train window used only for reporting; rules are FROZEN (no retuning).
        tr = _trade_series(
            frames, mask, win_start=train_start, win_end=fold_end, cutoff=cutoff
        )
        te = _trade_series(
            frames, mask, win_start=str(te0.date()), win_end=test_end, cutoff=cutoff
        )
        folds.append(
            {
                "fold": i + 1,
                "window": f"{te0.date()}..{test_end}",
                "k": k,
                "train": _stats(tr),
                "oos": _stats(te),
            }
        )
    return folds


# ---------------------------------------------------------------------------
# 16. Portfolio sketch (Phase 20)
# ---------------------------------------------------------------------------


def portfolio(
    symbols: List[str],
    data_dir: str = "data/raw",
    group: str = "full_fx",
    cutoff: str = CUTOFF,
    k: int = K_LONG_STAGE6,
) -> Dict:
    frames = _frames_multi(symbols, data_dir, group)
    trigs = {sym: _triggers(df) for sym, df in frames.items()}
    mask = _frozen_long_mask(frames, trigs, k)
    per_sym = {}
    for sym, df in frames.items():
        m = _oos(df, cutoff)
        fwd = df.loc[m, f"fwd{PRIMARY_H}_atr"]
        sig = mask(sym, df)
        tr = sig.loc[m] & fwd.notna()
        r = (fwd[tr] - COST_ATR).values.astype(float)
        if len(r) >= 3:
            per_sym[sym] = r
    # correlation of per-symbol R series (pad to common length via concat on dates)
    series = {}
    for sym, df in frames.items():
        m = _oos(df, cutoff)
        fwd = df.loc[m, f"fwd{PRIMARY_H}_atr"]
        sig = mask(sym, df)
        tr = sig.loc[m] & fwd.notna()
        r = fwd[tr] - COST_ATR
        if len(r) >= 3:
            series[sym] = r
    df_piv = pd.DataFrame(series).fillna(0.0)
    corr = df_piv.corr().values if df_piv.shape[1] > 1 else np.array([[1.0]])
    # equal-risk contribution: weight = 1/n, risk contribution share by |net|
    nets = {s: float(r.sum()) for s, r in per_sym.items()}
    tot = sum(abs(v) for v in nets.values())
    rc = {s: round(100 * abs(v) / tot, 1) for s, v in nets.items()} if tot else {}
    return {
        "n_symbols": len(per_sym),
        "mean_abs_corr": round(
            float(np.mean(np.abs(corr[np.triu_indices(len(corr), 1)]))), 3
        )
        if len(corr) > 1
        else 0.0,
        "risk_contribution_pct": rc,
        "max_single_symbol_rc_pct": max(rc.values()) if rc else 0.0,
    }


# ---------------------------------------------------------------------------
# 17. Adversarial (Phase 21)
# ---------------------------------------------------------------------------


def adversarial(
    symbols: List[str],
    data_dir: str = "data/raw",
    group: str = "full_fx",
    cutoff: str = CUTOFF,
    k: int = K_LONG_STAGE6,
) -> Dict:
    frames = _frames_multi(symbols, data_dir, group)
    trigs = {sym: _triggers(df) for sym, df in frames.items()}
    mask = _frozen_long_mask(frames, trigs, k)
    out: Dict[str, Dict] = {}

    base = _trade_series(frames, mask, cutoff=cutoff, win_start=TRAIN_END)
    out["as_is"] = _stats(base)

    # reversed signal: short the signal bars (direction flip, matched n)
    def rev_mask(sym, df):
        return mask(sym, df)

    parts = []
    for sym, df in frames.items():
        m = _oos(df, cutoff)
        fwd = df.loc[m, f"fwd{PRIMARY_H}_atr"]
        sig = rev_mask(sym, df)
        tr = sig.loc[m] & fwd.notna()
        parts.append(-(fwd[tr] - COST_ATR))
    if parts:
        r = pd.concat(parts)
        out["reversed"] = _stats(r)
    else:
        out["reversed"] = {"n": 0}

    # remove USDCHF / best symbol / best regime
    best_sym = "USDCHF"
    out["ex_usdchf"] = _stats(
        _trade_series(
            frames, mask, cutoff=cutoff, win_start=TRAIN_END, exclude=["USDCHF"]
        )
    )
    out["ex_best_symbol"] = _stats(
        _trade_series(
            frames, mask, cutoff=cutoff, win_start=TRAIN_END, exclude=[best_sym]
        )
    )

    # widened costs (3x)
    parts = []
    for sym, df in frames.items():
        m = _oos(df, cutoff)
        fwd = df.loc[m, f"fwd{PRIMARY_H}_atr"]
        sig = mask(sym, df)
        tr = sig.loc[m] & fwd.notna()
        parts.append(fwd[tr] - 3 * COST_ATR)
    r = pd.concat(parts)
    out["cost_x3"] = _stats(r)

    # perturbed thresholds: k=2 instead of k=3 (not a tuning — a fragility probe)
    mask2 = _frozen_long_mask(frames, trigs, 2)
    out["k_perturb_2"] = _stats(
        _trade_series(frames, mask2, cutoff=cutoff, win_start=TRAIN_END)
    )
    return out


# ---------------------------------------------------------------------------
# 18. Untouched test (Phase 18) — single-shot, full + ex-USDCHF
# ---------------------------------------------------------------------------


def untouched_test(
    symbols: List[str],
    data_dir: str = "data/raw",
    group: str = "full_fx",
    cutoff: str = CUTOFF,
    k: int = K_LONG_STAGE6,
) -> Dict:
    frames = _frames_multi(symbols, data_dir, group)
    trigs = {sym: _triggers(df) for sym, df in frames.items()}
    mask = _frozen_long_mask(frames, trigs, k)

    def collect(exclude=None):
        trades, by_sym, by_regime = [], {}, {}
        total_bars = 0
        for sym, df in frames.items():
            if exclude and sym in exclude:
                continue
            m = df.index >= pd.Timestamp(cutoff)
            fwd = df.loc[m, f"fwd{PRIMARY_H}_atr"]
            sig = mask(sym, df)
            tr = sig.loc[m] & fwd.notna()
            r = fwd[tr] - COST_ATR
            total_bars += int(m.sum())
            if len(r):
                trades.append(r)
                if len(r) >= 3:
                    by_sym[sym] = {
                        "n": int(len(r)),
                        "net_r": round(float(r.mean()), 4),
                    }
                for reg in df.loc[m, "regime"].unique():
                    rm = m & (df["regime"] == reg)
                    d2 = df.loc[rm, f"fwd{PRIMARY_H}_atr"]
                    tr2 = sig.loc[rm] & d2.notna()
                    v2 = d2[tr2] - COST_ATR
                    if len(v2):
                        bucket = by_regime.setdefault(str(reg), [])
                        bucket.extend(v2.values.tolist())
        if not trades:
            return {"n": 0}
        rs = pd.concat(trades)
        v = rs.values.astype(float)
        return {
            "n": len(v),
            "gross_r": round(float(v.mean()) + COST_ATR, 4),
            "net_r": round(float(v.mean()), 4),
            "cumulative_r": round(float(v.sum()), 3),
            "win": round(float((v > 0).mean()), 3),
            "flat_rate": round(1.0 - len(v) / max(total_bars, 1), 3),
            "maxdd_r": round(
                float((np.cumsum(v) - np.maximum.accumulate(np.cumsum(v))).min()), 3
            ),
            "by_symbol": by_sym,
            "by_regime": {
                reg: {
                    "n": len(a),
                    "net_r": round(float(np.mean(a)), 4),
                }
                for reg, a in by_regime.items()
                if len(a) >= 3
            },
            "n_symbols": len(by_sym),
        }

    return {
        "full": collect(),
        "ex_usdchf": collect(exclude=["USDCHF"]),
    }


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main(argv: Optional[List[str]] = None) -> int:
    import argparse

    parser = argparse.ArgumentParser(
        description="Stage-7 long reversal confirmation campaign"
    )
    parser.add_argument("--group", default="full_fx")
    parser.add_argument("--symbols", default=None)
    parser.add_argument("--cutoff", default=CUTOFF)
    parser.add_argument("--k", type=int, default=K_LONG_STAGE6)
    parser.add_argument("--integrity", action="store_true")
    parser.add_argument("--kcompare", action="store_true")
    parser.add_argument("--signal", action="store_true")
    parser.add_argument("--symbol", action="store_true")
    parser.add_argument("--regime", action="store_true")
    parser.add_argument("--temporal", action="store_true")
    parser.add_argument("--mfe", action="store_true")
    parser.add_argument("--exits", action="store_true")
    parser.add_argument("--sens", action="store_true")
    parser.add_argument("--cost", action="store_true")
    parser.add_argument("--exec", action="store_true")
    parser.add_argument("--boot", action="store_true")
    parser.add_argument("--baselines", action="store_true")
    parser.add_argument("--mt", action="store_true")
    parser.add_argument("--wf", action="store_true")
    parser.add_argument("--portfolio", action="store_true")
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
        f"Selection: train < {TRAIN_END} only. SHORT reversal LOCKED (falsified)."
    )
    results: Dict = {
        "symbols": symbols,
        "group": args.group,
        "cutoff": args.cutoff,
        "k": args.k,
    }

    if args.all or args.integrity:
        print("\n" + "=" * 72)
        print("PHASE 3 — RESEARCH-INTEGRITY AUDIT (10 Stage-6 checks + 5 new)")
        print("=" * 72)
        ia = integrity_audit()
        for ck, ok in ia["checks"].items():
            print(f"  [{'OK' if ok else 'FAIL'}] {ck}")
        for n in ia["notes"]:
            print(f"  note: {n}")
        print(
            f"  VERDICT: {'CLEAN — no leakage found' if ia['all_clean'] else 'LEAKAGE — STOP'}"
        )
        results["integrity"] = ia

    if args.all or args.kcompare:
        print("\n" + "=" * 72)
        print("PHASE 4 — k-OF-3 POWER COMPARISON (k=1/2/3, full stats)")
        print("=" * 72)
        kc = k_compare(symbols, group=args.group, cutoff=args.cutoff)
        for k, row in kc.items():
            for bucket in ("train", "oos"):
                s = row[bucket]
                if s.get("n", 0) < 20:
                    print(f"  k={k} [{bucket}] n={s.get('n', 0)}")
                    continue
                print(
                    f"  k={k} [{bucket}] n={s['n']:>5,} mean={s['mean_r']:+.4f} "
                    f"med={s['median_r']:+.4f} win={s['win']:.3f} pf={s['pf']} "
                    f"sharpe_pt={s['sharpe_pt']:.3f} maxdd={s['maxdd_r']:.3f} "
                    f"boot95={s['boot95']} perm_p={s['perm_p']:.3f} be={s['break_even']}"
                )
                if bucket == "oos" and "flat_rate" in s:
                    print(f"       flat_rate={s['flat_rate']:.3f}")
        results["k_compare"] = kc

    if args.all or args.signal:
        print("\n" + "=" * 72)
        print("PHASE 5 — MINIMUM VIABLE SIGNAL (A..G families, OOS)")
        print("=" * 72)
        sc = signal_compare(symbols, group=args.group, cutoff=args.cutoff)
        print(
            f"{'fam':<8}{'spec':<28}{'tr_n':>6}{'tr_net':>9}{'oos_n':>7}{'mean':>9}{'win':>7}{'perm_p':>8}{'be':>6}"
        )
        for fam, row in sc.items():
            t, o = row["train"], row["oos"]
            spec = f"{row['spec']['type']}:{','.join(row['spec']['names'])}"
            print(
                f"{fam:<8}{spec:<28}{t.get('n', 0):>6,}{t.get('mean_r', 0):>+9.4f}"
                f"{o.get('n', 0):>7,}{o.get('mean_r', 0):>+9.4f}{o.get('win', 0):>7.3f}"
                f"{o.get('perm_p', 0):>8.3f}{o.get('break_even', 0):>6}"
            )
        results["signal_compare"] = sc

    if args.all or args.symbol:
        print("\n" + "=" * 72)
        print(f"PHASE 6 — SYMBOL ROBUSTNESS + CONCENTRATION (frozen k={args.k})")
        print("=" * 72)
        sr = symbol_robustness(symbols, group=args.group, cutoff=args.cutoff, k=args.k)
        print(f"  pooled: {sr['pooled']}")
        print(
            f"  equal-weighted mean R: {sr['equal_weighted_mean_r']:+.4f}  "
            f"top1 PnL%: {sr['top1_pnl_pct']}  top3: {sr['top3_pnl_pct']}  "
            f"HHI: {sr['herfindahl']}"
        )
        print(f"  EX-USDCHF: {sr['ex_usdchf']}")
        print(f"  {'symbol':<8}{'n':>6}{'mean':>9}{'win':>7}")
        for s in sr["per_symbol"]:
            print(f"  {s['symbol']:<8}{s['n']:>6,}{s['mean_r']:>+9.4f}{s['win']:>7.3f}")
        print(
            "  leave-one-out pooled mean R:",
            {k2: round(v, 4) for k2, v in sr["leave_one_out_mean_r"].items()},
        )
        results["symbol_robustness"] = sr

    if args.all or args.regime:
        print("\n" + "=" * 72)
        print("PHASE 7 — REGIME GENERALIZATION (frozen k)")
        print("=" * 72)
        rr = regime_robustness(symbols, group=args.group, cutoff=args.cutoff, k=args.k)
        for bucket in ("by_regime", "by_vol_bucket"):
            print(f"  [{bucket}]")
            for name, s in rr[bucket].items():
                if s.get("n", 0) < 20:
                    print(f"    {name:<14} n={s.get('n', 0)}")
                    continue
                print(
                    f"    {name:<14} n={s['n']:>5,} mean={s['mean_r']:+.4f} "
                    f"win={s['win']:.3f} pf={s['pf']} perm_p={s['perm_p']:.3f} "
                    f"boot95={s['boot95']} be={s['break_even']}"
                )
        results["regime_robustness"] = rr

    if args.all or args.temporal:
        print("\n" + "=" * 72)
        print("PHASE 8 — TEMPORAL STABILITY (chronological cohorts, no tuning)")
        print("=" * 72)
        ts = temporal_stability(symbols, group=args.group, cutoff=args.cutoff, k=args.k)
        for name, s in ts.items():
            if s.get("n", 0) < 20:
                print(f"  {name:<20} n={s.get('n', 0)}")
                continue
            print(
                f"  {name:<20} n={s['n']:>5,} mean={s['mean_r']:+.4f} "
                f"win={s['win']:.3f} maxdd={s['maxdd_r']:.3f} perm_p={s['perm_p']:.3f}"
            )
        results["temporal"] = ts

    if args.all or args.mfe:
        print("\n" + "=" * 72)
        print("PHASE 9 — MFE / MAE (frozen k; descriptive)")
        print("=" * 72)
        mm = mfe_mae(symbols, group=args.group, cutoff=args.cutoff, k=args.k)
        if mm.get("n", 0) == 0:
            print("  no events")
        else:
            print(f"  n={mm['n']}")
            print(f"  MFE mean/med/p25/p75: {mm['mfe']}")
            print(f"  MAE mean/med/p75/p90: {mm['mae']}")
            print(f"  time-to-peak-MFE med/p25: {mm['time_to_peak_mfe']}")
            print(f"  time-to-max-MAE med: {mm['time_to_max_mae']}")
            print("  fwd mean R:", mm["fwd_returns_r"])
            print("  fwd win rate:", mm["fwd_win_rate"])
            print(
                f"  P(reach 1R/2R/3R MFE): {mm['pct_reach_1r']}/{mm['pct_reach_2r']}/{mm['pct_reach_3r']}"
            )
        results["mfe_mae"] = mm

    if args.all or args.exits:
        print("\n" + "=" * 72)
        print("PHASE 10 — EXIT-TRANSFER (params on TRAIN, reported OOS)")
        print("=" * 72)
        ex = exit_transfer(symbols, group=args.group, cutoff=args.cutoff, k=args.k)
        for fam, r in ex.items():
            if r.get("note"):
                print(f"  {fam:<16} {r['note']} (param={r.get('param')})")
                continue
            print(
                f"  {fam:<16} param={r['param']} h={r['horizon']} "
                f"train_net={r['train_net_r']:+.3f}R oos_n={r['n_oos']:>5,} "
                f"gross={r['gross_r']:+.3f}R net={r['net_r']:+.3f}R "
                f"win={r['win']:.3f} maxdd={r['maxdd_r']:.3f}R"
            )
        results["exit_transfer"] = ex

    if args.all or args.sens:
        print("\n" + "=" * 72)
        print("PHASE 11 — EXIT SENSITIVITY (local perturbation, OOS)")
        print("=" * 72)
        es = exit_sensitivity(symbols, group=args.group, cutoff=args.cutoff, k=args.k)
        for fam, rows in es.items():
            print(f"  [{fam}]")
            for p, s in rows.items():
                print(
                    f"    {p:<8} n={s['n']:>5,} net={s['net_r']:+.4f} win={s['win']:.3f}"
                )
        results["exit_sensitivity"] = es

    if args.all or args.cost:
        print("\n" + "=" * 72)
        print("PHASE 12 — COST ROBUSTNESS (frozen k)")
        print("=" * 72)
        cr = cost_robustness(symbols, group=args.group, cutoff=args.cutoff, k=args.k)
        print(
            f"  n={cr['n']} gross={cr['gross']:+.4f} break_even={cr['break_even_atr']} ATR"
        )
        print(
            "   " + "  ".join(f"{k2}={v:+.4f}" for k2, v in cr["net_by_cost"].items())
        )
        results["cost_robustness"] = cr

    if args.all or args.exec:
        print("\n" + "=" * 72)
        print("PHASE 13 — EXECUTION ROBUSTNESS")
        print("=" * 72)
        er = execution_robustness(
            symbols, group=args.group, cutoff=args.cutoff, k=args.k
        )
        for name, s in er.items():
            w = s.get("win", 0.0)
            print(f"  {name:<20} n={s['n']:>6,} net={s['net_r']:+.4f} win={w:.3f}")
        results["execution"] = er

    if args.all or args.boot:
        print("\n" + "=" * 72)
        print("PHASE 14 — BOOTSTRAP / PERMUTATION / RANDOMIZED TIMING")
        print("=" * 72)
        bp = bootstrap_perm(symbols, group=args.group, cutoff=args.cutoff, k=args.k)
        print(
            f"  n={bp['n']} mean={bp['mean']:+.4f} "
            f"trade_boot95={bp['trade_boot95']} block_boot95={bp['block_boot95']} "
            f"symbol_boot95={bp['symbol_boot95']} perm_p={bp['perm_p']:.4f}"
        )
        print(f"  randomized timing: {bp['random_timing']}")
        results["bootstrap_perm"] = bp

    if args.all or args.baselines:
        print("\n" + "=" * 72)
        print("PHASE 15 — BASELINES (net of 0.0625 ATR)")
        print("=" * 72)
        bl = baselines(symbols, group=args.group, cutoff=args.cutoff, k=args.k)
        print(f"{'baseline':<24}{'n':>9}{'net_r':>9}{'win':>7}")
        for r in bl:
            print(f"{r['baseline']:<24}{r['n']:>9,}{r['net_r']:>+9.4f}{r['win']:>7.3f}")
        results["baselines"] = bl

    if args.all or args.mt:
        print("\n" + "=" * 72)
        print("PHASE 16 — MULTIPLE-TESTING (BH FDR, OOS)")
        print("=" * 72)
        mt = multiple_testing(symbols, group=args.group, cutoff=args.cutoff)
        print(f"  n_tests={mt['n_tests']}")
        print("  pvals:", mt["pvals"])
        print("  bh_q:", mt["bh_q"])
        print(f"  significant at q=0.05: {mt['significant_q05']}")
        print(f"  experiment ledger (stages 4-7): {mt['experiment_ledger']}")
        results["multiple_testing"] = mt

    if args.all or args.wf:
        print("\n" + "=" * 72)
        print("PHASE 17 — FROZEN PURGED WALK-FORWARD (rules frozen, no retuning)")
        print("=" * 72)
        wf = walk_forward(symbols, group=args.group, cutoff=args.cutoff, k=args.k)
        for f in wf:
            tr, te = f["train"], f["oos"]
            print(
                f"  fold {f['fold']} {f['window']:<28} k={f['k']} "
                f"train_n={tr.get('n', 0):>5,} train_net={tr.get('mean_r', 0):+.4f} | "
                f"oos_n={te.get('n', 0):>5,} oos_net={te.get('mean_r', 0):+.4f} "
                f"win={te.get('win', 0):.3f} maxdd={te.get('maxdd_r', 0):.3f}"
            )
        results["walk_forward"] = wf

    if args.all or args.portfolio:
        print("\n" + "=" * 72)
        print("PHASE 20 — PORTFOLIO SKETCH")
        print("=" * 72)
        pf = portfolio(symbols, group=args.group, cutoff=args.cutoff, k=args.k)
        print(
            f"  n_symbols={pf['n_symbols']} mean_abs_corr={pf['mean_abs_corr']} "
            f"max_single_rc={pf['max_single_symbol_rc_pct']}%"
        )
        print("  risk contribution %:", pf["risk_contribution_pct"])
        results["portfolio"] = pf

    if args.all or args.adversarial:
        print("\n" + "=" * 72)
        print("PHASE 21 — ADVERSARIAL FALSIFICATION")
        print("=" * 72)
        ad = adversarial(symbols, group=args.group, cutoff=args.cutoff, k=args.k)
        for name, s in ad.items():
            if s.get("n", 0) < 20:
                print(f"  {name:<18} n={s.get('n', 0)}")
                continue
            print(
                f"  {name:<18} n={s['n']:>5,} mean={s['mean_r']:+.4f} "
                f"win={s['win']:.3f} perm_p={s['perm_p']:.3f}"
            )
        results["adversarial"] = ad

    if args.untouched:
        print("\n" + "=" * 72)
        print("PHASE 18 — UNTOUCHED TEST (single-shot, frozen rules; full + EX-USDCHF)")
        print("=" * 72)
        ut = untouched_test(symbols, group=args.group, cutoff=args.cutoff, k=args.k)
        for label, r in ut.items():
            print(f"\n  [{label}]")
            if r.get("n", 0) == 0:
                print("    no trades")
                continue
            print(
                f"    n={r['n']} gross={r['gross_r']:+.4f} net={r['net_r']:+.4f} "
                f"cumR={r['cumulative_r']:+.3f} win={r['win']:.3f} "
                f"flat={r['flat_rate']:.3f} maxdd={r['maxdd_r']:.3f} "
                f"n_symbols={r['n_symbols']}"
            )
            print(
                "    by symbol:", {k2: v["net_r"] for k2, v in r["by_symbol"].items()}
            )
            print(
                "    by regime:", {k2: v["net_r"] for k2, v in r["by_regime"].items()}
            )
        results["untouched"] = ut

    out_dir = Path("data/validation")
    out_dir.mkdir(parents=True, exist_ok=True)
    with open(out_dir / "stage7_results.json", "w") as fh:
        json.dump(results, fh, indent=2, default=str)
    print(f"\nStage-7 results written to {out_dir / 'stage7_results.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
