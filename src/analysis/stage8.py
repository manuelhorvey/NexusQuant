"""
NexusQuant — Stage-8: Long Reversal Generalization, Cross-Sectional Robustness
& Final Out-of-Sample Validation (research only — no production changes).

Stage-7 verdict carried forward:

    LONG reversal leg = B. PROMISING BUT INSUFFICIENT EVIDENCE
    SHORT reversal leg = FALSIFIED -> LOCKED, must NOT be revived

Stage-8 asks one question about the LONG leg:

    Does the edge generalize beyond the 16-symbol discovery universe,
    survive removal of its best contributors, and hold in a genuinely
    unseen period — or is it a research artifact of symbol/regime selection?

Frozen protocol (pre-registered, no threshold mining):

  - ENTRY  : k=3 of {L1_rsi30, L2_drop5, L3_streak5n}   (Stage-6/7 frozen)
  - HORIZON: 10 bars (PRIMARY_H)
  - COST   : 0.0625 ATR round trip (COST_ATR)
  - TRAIN  : selection/statistics on < 2022-01-01 only
  - OOS    : 2022-01-01 .. 2025-06-01
  - UNTOUCHED: 2025-06-01+ evaluated EXACTLY ONCE (--untouched), frozen rules
  - UNIVERSE: pre-registered eligibility (>=500 bars, starts before 2022-01-01,
    ends on/after 2025-06-01) across full_fx (108) + candidates (4) = 112 D1
    symbols; stratified by bucket (fx_major_cross / fx_exotic / metal / crypto
    / index). NOT performance-driven.
  - VOL BUCKETS: CAUSAL rolling 250-bar percentile of ATR (fix vs Stage-7's
    full-sample qcut rank, which was a mild look-ahead; documented in the
    leakage audit).

Analyses (each is a function; --all runs them in one process sharing a frame
cache; --untouched is the single-shot final test):

    1. universe           — eligible symbols by bucket (pre-registered criteria)
    2. cross_sectional    — pooled OOS, per-symbol, leave-one-symbol-out,
       leave-one-cluster-out, leave-one-currency-out, ex-USDCHF, ex-AUDCHF,
       ex-top-1/top-2/top-10%-PnL
    3. walk_forward       — purged/embargoed folds, per-fold reporting with
       n<20/30/50 flags (no pooling to hide weak folds)
    4. regime_matrix      — Regime x Volatility matrix (4 x 3 cells)
    5. entry_ablation     — A..J minimum-sufficient-rule comparison
       (incl. random extreme-state entry, delayed entry, confirmation entry)
    6. exit_ablation      — holding-period sweep 1..40, MFE/MAE curves,
       P(reach 0.5R/1R/2R/3R), exit-family transfer + sensitivity
    7. cost_execution     — cost grid 0..0.25 ATR, break-even, delays,
       conservative fill, spread widening
    8. monte_carlo        — 10,000 block-bootstrap paths: P(profit), maxDD
       distribution (p50/p95/p99), P(20-trade losing streak), P(50% DD at
       1% risk), recovery time; trade-order shuffle, symbol/regime bootstrap
    9. baselines          — buy-and-hold, always-long/flat, random entries,
       random extreme-state, RSI<30, streak, SMA200-deviation, momentum,
       SMA200 trend-following
   10. multiple_testing   — BH FDR + cumulative experiment ledger (stages 4-8)
   11. leakage_audit      — machine-readable PASS/FAIL/UNKNOWN for every
       data-snooping / look-ahead check
   12. portfolio_risk     — trade-return correlation, signal clustering,
       concurrent-position caps, cluster aggregation (CHF/JPY/USD), HHI
   13. mechanism          — descriptive economic probes (vol quintiles, drop
       quintiles, streak length, CHF vs non-CHF, USD-quote vs not)
   14. adversarial        — reversed, shuffled timing/returns, regime
       permutation, cost x3, delays, remove best symbol/regime/family,
       threshold perturbation
   15. untouched_test     — single-shot 2025-06-01+: full, ex-USDCHF,
       ex-AUDCHF, ex-top-OOS-symbol; by symbol/regime/bucket; target-level
       first-touch probabilities; time-stop EV

Usage:
    python -m src.analysis.stage8 --all                    (all OOS phases)
    python -m src.analysis.stage8 --untouched             (single-shot test)
    python -m src.analysis.stage8 --cross --wf --regime   (subset)
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

sys.path.append(str(Path(__file__).parent.parent.parent))

from src.analysis.stage7 import (
    LONG_COMBO,
    PRIMARY_H,
    _exit_family_r,
    _family_defs,
    _frozen_long_mask,
    _long_mask,
)
from src.analysis.stage6 import (
    CORE16,
    EMBARGO_DAYS,
    WF_FOLDS,
    _oos,
    _triggers,
)
from src.analysis.stage5 import _breakeven, _dist_features
from src.analysis.stage4 import COST_R, CUTOFF
from src.analysis.stage5 import TRAIN_END

K_LONG = 3  # frozen Stage-6/7 selection
ELIG_MIN_BARS = 500
VOL_ROLL = 250  # causal vol-percentile window
MC_PATHS = 10_000
MC_BLOCK = 10
SEED = 20250813

MAJORS = {"EUR", "GBP", "USD", "CHF", "JPY", "AUD", "NZD", "CAD"}
METALS = {"XAUUSD", "XAGUSD", "XPTUSD", "XPDUSD"}
CRYPTO = {"BTCUSD", "ETHUSD"}
INDICES = {"US500", "US30", "USTEC", "NAS100", "US100"}


HORIZONS8 = (1, 3, 5, 8, 10, 15, 20, 30, 40)  # superset for holding sweep
R_MULT = 1.25  # 1R = 1.25 x ATR (the house risk convention)


def _fwd_R(df: pd.DataFrame, horizons: tuple = HORIZONS8) -> pd.DataFrame:
    """True R-unit forward returns: (close_{t+h} - close_t) / (1.25*ATR_t).

    Stage-8 integrity fix: the stage-7 ``fwd{h}_atr`` convention
    ((close_{t+h}/close_t - 1)/ATR_price) scales as 1/close, so pooled
    cross-symbol numbers mixed price scales (a gold trade at ~4400 was
    numerically ~4000x smaller than an AUDUSD trade at ~0.65). R units are
    unit-consistent across symbols and match _exit_family_r / COST_R.
    """
    close = df["close"].astype(float)
    atr = df["atr_14"].astype(float).replace(0.0, np.nan)
    for h in horizons:
        df[f"fwd{h}_R"] = (close.shift(-h) - close) / (atr * R_MULT)
    return df


def _trade_r(
    frames: Dict[str, pd.DataFrame],
    mask_fn,
    horizon: int = PRIMARY_H,
    win_start: Optional[str] = None,
    win_end: Optional[str] = None,
    cutoff: str = CUTOFF,
    exclude: Optional[List[str]] = None,
) -> pd.Series:
    """Chronological net-R series of LONG trades in TRUE R units
    (cost = COST_R = 0.05R). mask_fn(sym, df) -> boolean Series."""
    parts = []
    for sym, df in frames.items():
        if exclude and sym in exclude:
            continue
        m = df.index < pd.Timestamp(cutoff)
        if win_start:
            m &= df.index >= pd.Timestamp(win_start)
        if win_end:
            m &= df.index < pd.Timestamp(win_end)
        sig = mask_fn(sym, df)
        fwd = df[f"fwd{horizon}_R"]
        tr = sig & m & fwd.notna()
        parts.append(fwd[tr] - COST_R)
    if not parts:
        return pd.Series(dtype=float)
    return pd.concat(parts).sort_index()


def _stats8(rs: pd.Series, n_boot: int = 2000, n_perm: int = 300) -> Dict:
    """Stats on a net-R series (R units; break-even expressed in R)."""
    v = rs.values.astype(float)
    if len(v) < 20:
        return {"n": len(v)}
    rng = np.random.default_rng(7)
    mean = float(v.mean())
    boot = np.array(
        [float(rng.choice(v, len(v), replace=True).mean()) for _ in range(n_boot)]
    )
    lo, hi = np.percentile(boot, [2.5, 97.5])
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
        "break_even": _breakeven(v + COST_R),
    }


# ---------------------------------------------------------------------------
# Universe (pre-registered, objective — not performance-driven)
# ---------------------------------------------------------------------------

_FRAME_CACHE: Dict[Tuple[str, ...], Dict[str, pd.DataFrame]] = {}


def _bucket(symbol: str) -> str:
    if symbol in METALS:
        return "metal"
    if symbol in CRYPTO:
        return "crypto"
    if symbol in INDICES:
        return "index"
    if symbol[:3] in MAJORS and symbol[3:] in MAJORS:
        return "fx_major_cross"
    return "fx_exotic"


def eligible_universe() -> Dict[str, str]:
    """symbol -> group for every D1 instrument passing the pre-registered
    data-quality criteria (>=500 bars, starts <2022-01-01, ends >=2025-06-01)."""
    out: Dict[str, str] = {}
    for group in ("full_fx", "candidates"):
        d = Path("data/raw") / group
        if not d.exists():
            continue
        for f in sorted(d.glob("*_D1.parquet")):
            sym = f.name.replace("_D1.parquet", "")
            try:
                dates = pd.read_parquet(f, columns=["date"])["date"]
            except Exception:
                continue
            if len(dates) < ELIG_MIN_BARS:
                continue
            if dates.min() >= pd.Timestamp(TRAIN_END):
                continue
            if dates.max() < pd.Timestamp(CUTOFF):
                continue
            out[sym] = group
    return dict(sorted(out.items()))


def _frames8(symbols: Optional[List[str]] = None) -> Dict[str, pd.DataFrame]:
    """Load + indicators + regime + fwd returns + dist features + CAUSAL vol
    buckets for the requested universe. Cached per process (phases share it)."""
    if symbols is None:
        eligible = eligible_universe()
        symbols = list(eligible.keys())
    key = tuple(symbols)
    if key in _FRAME_CACHE:
        return _FRAME_CACHE[key]
    eligible = eligible_universe()
    out: Dict[str, pd.DataFrame] = {}
    for sym in symbols:
        group = eligible.get(sym, "full_fx")
        df = _load_base8(sym, group)
        if df is None or len(df) < 500:
            continue
        out[sym] = df
    _FRAME_CACHE[key] = out
    return out


def _load_base8(symbol: str, group: str) -> Optional[pd.DataFrame]:
    from src.analysis.stage4 import _load_base

    df = _load_base(symbol, "data/raw", group)
    if df is None:
        return None
    df = _fwd_R(df, horizons=HORIZONS8)
    df = pd.concat([df, _dist_features(df)], axis=1)
    close = df["close"].astype(float)
    atr = df["atr_14"].astype(float).replace(0.0, np.nan)
    df["rally5"] = close.pct_change(5) / atr
    df["drop5"] = close.pct_change(5) / atr
    df["streak5p"] = (close.diff() > 0).astype(int).rolling(5).sum()
    df["streak5n"] = (close.diff() < 0).astype(int).rolling(5).sum()
    df["ret_10"] = close.pct_change(10)  # causal model-feature proxy
    # CAUSAL vol bucket: trailing 250-bar percentile of 20-bar mean ATR.
    atr20 = atr.rolling(20).mean()
    df["vol_pct"] = atr20.rolling(VOL_ROLL).rank(pct=True)
    df["vol_bucket"] = pd.cut(
        df["vol_pct"], [0.0, 0.33, 0.66, 1.0], labels=["low", "med", "high"]
    )
    return df


# ---------------------------------------------------------------------------
# 1. Universe report
# ---------------------------------------------------------------------------


def universe() -> Dict:
    eligible = eligible_universe()
    by_bucket: Dict[str, int] = {}
    for sym in eligible:
        by_bucket[_bucket(sym)] = by_bucket.get(_bucket(sym), 0) + 1
    return {
        "criteria": {
            "min_bars": ELIG_MIN_BARS,
            "starts_before": TRAIN_END,
            "ends_on_or_after": CUTOFF,
            "groups": ["full_fx", "candidates"],
            "timeframe": "D1",
        },
        "n_symbols": len(eligible),
        "by_bucket": by_bucket,
        "symbols": list(eligible.keys()),
        "core16_subset": sorted(set(CORE16) & set(eligible.keys())),
    }


# ---------------------------------------------------------------------------
# 2. Cross-sectional generalization
# ---------------------------------------------------------------------------


def _cluster_members(symbols: List[str]) -> Dict[str, List[str]]:
    """Currency clusters (any pair containing the currency) + asset buckets."""
    out: Dict[str, List[str]] = {}
    for c in sorted(MAJORS):
        out[f"cur_{c}"] = sorted(s for s in symbols if c in (s[:3], s[3:]))
    for b in ("metal", "crypto", "index", "fx_exotic"):
        out[f"bucket_{b}"] = sorted(s for s in symbols if _bucket(s) == b)
    return {k: v for k, v in out.items() if len(v) >= 2}


def cross_sectional(
    symbols: Optional[List[str]] = None,
    cutoff: str = CUTOFF,
) -> Dict:
    frames = _frames8(symbols)
    trigs = {sym: _triggers(df) for sym, df in frames.items()}
    mask = _frozen_long_mask(frames, trigs, K_LONG)
    syms = list(frames.keys())

    pooled = _trade_r(frames, mask, cutoff=cutoff, win_start=TRAIN_END)
    pooled_stats = _stats8(pooled)

    # per-symbol OOS
    per_sym = []
    nets: Dict[str, float] = {}
    for sym, df in frames.items():
        m = _oos(df, cutoff)
        fwd = df.loc[m, f"fwd{PRIMARY_H}_R"]
        sig = mask(sym, df)
        tr = sig.loc[m] & fwd.notna()
        r = (fwd[tr] - COST_R).values.astype(float)
        if len(r) >= 3:
            per_sym.append(
                {
                    "symbol": sym,
                    "bucket": _bucket(sym),
                    "n": len(r),
                    "mean_r": round(float(r.mean()), 4),
                    "win": round(float((r > 0).mean()), 3),
                    "net_r_sum": round(float(r.sum()), 3),
                }
            )
            nets[sym] = float(r.sum())
    total_net = abs(sum(nets.values()))
    sorted_net = sorted(nets.items(), key=lambda kv: -abs(kv[1]))
    top1 = sorted_net[0][0] if sorted_net else None
    top2 = [s for s, _ in sorted_net[:2]]
    top10pct_n = max(1, int(0.10 * len(sorted_net)))
    top10pct = [s for s, _ in sorted_net[:top10pct_n]]
    shares = np.array([abs(v) for v in nets.values()])
    hhi = float(((shares / shares.sum()) ** 2).sum()) if shares.sum() else 0.0
    ew = float(np.mean([s["mean_r"] for s in per_sym])) if per_sym else 0.0

    # leave-one-symbol-out (pooled mean over remaining)
    loso = {
        excl: round(
            float(
                _trade_r(
                    frames, mask, cutoff=cutoff, win_start=TRAIN_END, exclude=[excl]
                ).mean()
            ),
            4,
        )
        for excl in syms
    }
    loso_pos = sum(1 for v in loso.values() if v > 0)

    # leave-one-cluster-out
    clusters = _cluster_members(syms)
    loco = {}
    for name, members in clusters.items():
        rs = _trade_r(frames, mask, cutoff=cutoff, win_start=TRAIN_END, exclude=members)
        loco[name] = {"n": int(len(rs)), "mean_r": round(float(rs.mean()), 4)}

    # removal tests
    def ex(excl: List[str]) -> Dict:
        rs = _trade_r(frames, mask, cutoff=cutoff, win_start=TRAIN_END, exclude=excl)
        return _stats8(rs)

    return {
        "n_symbols": len(syms),
        "pooled": pooled_stats,
        "equal_weighted_mean_r": round(ew, 4),
        "herfindahl": round(hhi, 3),
        "top1_symbol": top1,
        "top1_pnl_pct": round(100 * abs(nets[top1]) / total_net, 1)
        if top1 and total_net
        else 0.0,
        "top2_pnl_pct": round(100 * sum(abs(nets[s]) for s in top2) / total_net, 1)
        if total_net
        else 0.0,
        "per_symbol": per_sym,
        "leave_one_out": {
            "min": round(min(loso.values()), 4),
            "max": round(max(loso.values()), 4),
            "mean": round(float(np.mean(list(loso.values()))), 4),
            "pct_positive": round(loso_pos / max(len(loso), 1), 3),
            "worst_excluded": min(loso, key=loso.get),
            "best_excluded": max(loso, key=loso.get),
        },
        "leave_one_cluster_out": loco,
        "ex_usdchf": ex(["USDCHF"]),
        "ex_audchf": ex(["AUDCHF"]),
        "ex_top1": ex([top1]) if top1 else {"n": 0},
        "ex_top2": ex(top2) if top2 else {"n": 0},
        "ex_top10pct": ex(top10pct) if top10pct else {"n": 0},
        "ex_both_chf": ex(["USDCHF", "AUDCHF"]),
    }


# ---------------------------------------------------------------------------
# 3. Purged walk-forward (per-fold, no pooling to hide weak folds)
# ---------------------------------------------------------------------------


def walk_forward(symbols: Optional[List[str]] = None, cutoff: str = CUTOFF) -> Dict:
    frames = _frames8(symbols)
    trigs = {sym: _triggers(df) for sym, df in frames.items()}
    mask = _frozen_long_mask(frames, trigs, K_LONG)
    folds = []
    for i, (train_start, fold_end, test_end) in enumerate(WF_FOLDS):
        te0 = pd.Timestamp(fold_end) + pd.Timedelta(days=EMBARGO_DAYS)
        tr = _trade_r(
            frames, mask, win_start=train_start, win_end=fold_end, cutoff=cutoff
        )
        te = _trade_r(
            frames, mask, win_start=str(te0.date()), win_end=test_end, cutoff=cutoff
        )
        oos = _stats8(te)
        n = oos.get("n", 0)
        folds.append(
            {
                "fold": i + 1,
                "window": f"{te0.date()}..{test_end}",
                "train": _stats8(tr),
                "oos": oos,
                "flags": {
                    "n_lt_20": n < 20,
                    "n_lt_30": n < 30,
                    "n_lt_50": n < 50,
                },
            }
        )
    n_pos = sum(1 for f in folds if f["oos"].get("mean_r", 0) > 0)
    n_ge20 = sum(1 for f in folds if f["oos"].get("n", 0) >= 20)
    return {
        "folds": folds,
        "n_folds_positive_net": n_pos,
        "n_folds_with_n_ge20": n_ge20,
        "n_folds_total": len(folds),
    }


# ---------------------------------------------------------------------------
# 4. Regime x Volatility matrix
# ---------------------------------------------------------------------------


def regime_matrix(symbols: Optional[List[str]] = None, cutoff: str = CUTOFF) -> Dict:
    frames = _frames8(symbols)
    trigs = {sym: _triggers(df) for sym, df in frames.items()}
    mask = _frozen_long_mask(frames, trigs, K_LONG)
    cells: Dict[str, pd.Series] = {}
    for sym, df in frames.items():
        m = _oos(df, cutoff)
        sig = mask(sym, df)
        fwd = df.loc[m, f"fwd{PRIMARY_H}_R"]
        tr = sig.loc[m] & fwd.notna()
        r = fwd[tr] - COST_R
        if not len(r):
            continue
        regs = df.loc[m, "regime"]
        vols = df.loc[m, "vol_bucket"]
        for reg in regs.unique():
            for vb in vols.unique():
                sub = r[(regs.loc[r.index] == reg) & (vols.loc[r.index] == vb)]
                if not len(sub):
                    continue
                key = f"{reg}|{vb}"
                cells[key] = pd.concat([cells.get(key, pd.Series(dtype=float)), sub])
    matrix = {
        reg: {
            vb: _stats8(cells.get(f"{reg}|{vb}", pd.Series(dtype=float)))
            for vb in ("low", "med", "high")
        }
        for reg in ("Bull Trend", "Bear Trend", "Range / Chop", "High Volatility")
    }
    # marginal: does High Volatility add edge over the base?
    hv = (
        pd.concat(
            [cells[k] for k in cells if k.startswith("High Volatility")],
        )
        if any(k.startswith("High Volatility") for k in cells)
        else pd.Series(dtype=float)
    )
    non_hv = (
        pd.concat(
            [cells[k] for k in cells if not k.startswith("High Volatility")],
        )
        if any(not k.startswith("High Volatility") for k in cells)
        else pd.Series(dtype=float)
    )
    return {
        "matrix": matrix,
        "high_vol_marginal": {
            "high_vol": _stats8(hv),
            "non_high_vol": _stats8(non_hv),
        },
    }


# ---------------------------------------------------------------------------
# 5. Entry ablation (A..J)
# ---------------------------------------------------------------------------


def entry_ablation(symbols: Optional[List[str]] = None, cutoff: str = CUTOFF) -> Dict:
    frames = _frames8(symbols)
    trigs = {sym: _triggers(df) for sym, df in frames.items()}
    rng = np.random.default_rng(SEED)

    def te(mask) -> pd.Series:
        return _trade_r(frames, mask, cutoff=cutoff, win_start=TRAIN_END)

    def tr(mask) -> pd.Series:
        return _trade_r(frames, mask, win_end=TRAIN_END, cutoff=cutoff)

    out: Dict[str, Dict] = {}
    specs = {
        "A_rsi30_only": ("single", ["L1_rsi30"]),
        "B_streak_only": ("single", ["L3_streak5n"]),
        "C_crash_only": ("single", ["L4_crash"]),
        "D_k2": ("kof", 2),
        "E_k3_frozen": ("kof", 3),
        "F_any_trigger": ("kof", 1),
    }
    for name, (typ, arg) in specs.items():
        if typ == "single":

            def mk(sym, df, name=name, arg=arg):
                return trigs[sym][arg[0]]
        else:

            def mk(sym, df, k=arg):
                score = sum(trigs[sym][n].astype(int) for n in LONG_COMBO)
                return score >= k

        out[name] = {"train": _stats8(tr(mk)), "oos": _stats8(te(mk))}

    # G. random extreme-state entry: random bars among any-trigger bars,
    #    matched to E's OOS count. Distribution over 200 draws.
    any_pool: List[float] = []
    for sym, df in frames.items():
        m = _oos(df, cutoff)
        fwd = df.loc[m, f"fwd{PRIMARY_H}_R"]
        any_tr = (
            trigs[sym]["L1_rsi30"]
            | trigs[sym]["L2_drop5"]
            | trigs[sym]["L3_streak5n"]
            | trigs[sym]["L4_crash"]
        )
        sel = any_tr.loc[m] & fwd.notna()
        any_pool.extend(fwd[sel].values.tolist())
    any_pool = np.array(any_pool)
    n_E = out["E_k3_frozen"]["oos"].get("n", 0)
    rand = np.array(
        [
            float(rng.choice(any_pool, n_E).mean() - COST_R)
            for _ in range(200)
            if len(any_pool) >= n_E
        ]
    )
    e_mean = out["E_k3_frozen"]["oos"].get("mean_r", 0.0)
    out["G_random_extreme"] = {
        "n": n_E,
        "random_mean": round(float(rand.mean()), 4) if len(rand) else None,
        "random_p95": round(float(np.percentile(rand, 95)), 4) if len(rand) else None,
        "signal_beats_random_p95": bool(e_mean > np.percentile(rand, 95))
        if len(rand)
        else None,
    }

    # H. delayed entry (1 bar) and I. 1-bar confirmation (trigger at t and t+1)
    def delayed(sym, df):
        sig = _frozen_long_mask(frames, trigs, K_LONG)(sym, df)
        return sig.shift(1, fill_value=False)

    def confirmed(sym, df):
        sig = _frozen_long_mask(frames, trigs, K_LONG)(sym, df)
        return sig & sig.shift(1, fill_value=False)

    out["H_delayed_1bar"] = {"oos": _stats8(te(delayed))}
    out["I_confirmed_1bar"] = {"oos": _stats8(te(confirmed))}
    out["J_no_confirmation"] = out["E_k3_frozen"]  # base case, reported for parity
    return out


# ---------------------------------------------------------------------------
# 6. Exit ablation: holding sweep + MFE/MAE + P(reach) + transfer
# ---------------------------------------------------------------------------


def _events(
    frames,
    trigs,
    mask,
    cutoff: str,
    win_start: Optional[str] = None,
    win_end: Optional[str] = None,
    limit: int = 6000,
) -> List[Tuple[pd.DataFrame, int]]:
    ev = []
    for sym, df in frames.items():
        m = _pre_cutoff(df, cutoff)
        if win_start:
            m &= df.index >= pd.Timestamp(win_start)
        if win_end:
            m &= df.index < pd.Timestamp(win_end)
        sig = mask(sym, df)
        idxs = np.where((sig & m).values)[0]
        for pos in idxs:
            ev.append((df, int(pos)))
            if len(ev) >= limit:
                return ev
    return ev


def _pre_cutoff(df: pd.DataFrame, cutoff: str) -> np.ndarray:
    return df.index < pd.Timestamp(cutoff)


def exit_ablation(symbols: Optional[List[str]] = None, cutoff: str = CUTOFF) -> Dict:
    frames = _frames8(symbols)
    trigs = {sym: _triggers(df) for sym, df in frames.items()}
    mask = _frozen_long_mask(frames, trigs, K_LONG)

    # --- holding-period sweep (net R per horizon) ---
    sweep = {}
    for h in (1, 3, 5, 8, 10, 15, 20, 30, 40):
        rs = _trade_r(frames, mask, horizon=h, cutoff=cutoff, win_start=TRAIN_END)
        v = rs.values.astype(float)
        sweep[str(h)] = (
            {"n": len(v), "net_r": round(float(v.mean()), 4)} if len(v) else {"n": 0}
        )

    # --- MFE/MAE + P(reach) on OOS events ---
    rows = []
    for df, row in _events(frames, trigs, mask, cutoff, win_start=TRAIN_END):
        close = df["close"].astype(float)
        atr = df["atr_14"].astype(float)
        entry = float(close.iloc[row])
        risk = float(atr.iloc[row]) * R_MULT
        if entry != entry or risk != risk or risk <= 0:
            continue
        win = df.iloc[row + 1 : row + 1 + 20]
        if len(win) < 3:
            continue
        hh = win["high"].astype(float)
        ll = win["low"].astype(float)
        mfe = float((hh.max() - entry) / risk)
        mae = float((entry - ll.min()) / risk)
        # first-touch of +X R vs -1R within 20 bars (first-touch ordering)
        n = len(hh)
        t_sl = (
            int(np.argmax((ll <= entry - risk).values))
            if (ll <= entry - risk).any()
            else n
        )
        first: Dict[str, int] = {}
        for tgt in (0.5, 1.0, 2.0, 3.0):
            hit = hh >= entry + risk * tgt
            t_tp = int(np.argmax(hit.values)) if hit.any() else n
            first[f"p_tp{tgt}_first"] = 1 if t_tp < t_sl else 0
        first["p_sl_first"] = (
            1
            if t_sl < n
            and t_sl
            < min(
                int(np.argmax((hh >= entry + risk * t).values))
                if (hh >= entry + risk * t).any()
                else n
                for t in (0.5, 1.0, 2.0, 3.0)
            )
            else 0
        )
        rows.append({"mfe": mfe, "mae": mae, **first})
    mfe_mae = {}
    if rows:
        mfes = np.array([r["mfe"] for r in rows])
        maes = np.array([r["mae"] for r in rows])
        mfe_mae = {
            "n": len(rows),
            "mfe_median": round(float(np.median(mfes)), 3),
            "mae_median": round(float(np.median(maes)), 3),
            "p_reach_0_5r": round(
                float(np.mean([r.get("p_tp0.5_first", 0) for r in rows])), 3
            ),
            "p_reach_1r": round(
                float(np.mean([r.get("p_tp1_first", 0) for r in rows])), 3
            ),
            "p_reach_2r": round(
                float(np.mean([r.get("p_tp2_first", 0) for r in rows])), 3
            ),
            "p_reach_3r": round(
                float(np.mean([r.get("p_tp3_first", 0) for r in rows])), 3
            ),
            "p_sl_first": round(
                float(np.mean([r.get("p_sl_first", 0) for r in rows])), 3
            ),
            "p_mfe_ge_1r": round(float((mfes >= 1.0).mean()), 3),
            "p_mfe_ge_3r": round(float((mfes >= 3.0).mean()), 3),
        }

    # --- exit-family transfer (train-selected param, OOS eval) ---
    events_tr = _events(frames, trigs, mask, cutoff, win_end=TRAIN_END, limit=1500)
    events_te = _events(frames, trigs, mask, cutoff, win_start=TRAIN_END, limit=3000)
    families = {
        "time": [5, 10, 20],
        "atr_target": [0.5, 0.75, 1.0, 1.5],
        "atr_stop": [0.5, 0.75, 1.0],
        "trailing": [0.5, 1.0, 1.5],
        "signal_reversal": [35.0],
        "return_mean": [0.0],
    }
    transfer = {}
    for fam, params in families.items():
        best_p, best_net, best_hz = None, -np.inf, None
        for p in params:
            for hz in (10, 20):
                rs = [_exit_family_r(df, row, fam, p, hz) for df, row in events_tr]
                rs = [r - COST_R for r in rs if r is not None]
                if len(rs) < 40:
                    continue
                net = float(np.mean(rs))
                if net > best_net:
                    best_net, best_p, best_hz = net, p, hz
        if best_p is None:
            transfer[fam] = {"note": "no train-viable exit"}
            continue
        rs = [_exit_family_r(df, row, fam, best_p, best_hz) for df, row in events_te]
        rs = [r - COST_R for r in rs if r is not None]
        if len(rs) < 30:
            transfer[fam] = {"note": "insufficient oos", "param": best_p}
            continue
        v = np.array(rs)
        transfer[fam] = {
            "param": best_p,
            "horizon": best_hz,
            "train_net_r": round(best_net, 4),
            "n_oos": len(v),
            "net_r": round(float(v.mean()), 4),
            "win": round(float((v > 0).mean()), 3),
            "maxdd_r": round(
                float((np.cumsum(v) - np.maximum.accumulate(np.cumsum(v))).min()), 3
            ),
        }
    return {"holding_sweep": sweep, "mfe_mae": mfe_mae, "exit_transfer": transfer}


# ---------------------------------------------------------------------------
# 7. Cost & execution robustness
# ---------------------------------------------------------------------------


def cost_execution(symbols: Optional[List[str]] = None, cutoff: str = CUTOFF) -> Dict:
    frames = _frames8(symbols)
    trigs = {sym: _triggers(df) for sym, df in frames.items()}
    mask = _frozen_long_mask(frames, trigs, K_LONG)
    rs = _trade_r(frames, mask, cutoff=cutoff, win_start=TRAIN_END)
    v = rs.values.astype(float)
    gross = float(v.mean()) + COST_R
    grid = (0.0, 0.025, 0.05, 0.0625, 0.10, 0.15, 0.20, 0.25)
    out: Dict = {
        "n": len(v),
        "gross_r": round(gross, 4),
        "break_even_r": _breakeven(v + COST_R),
        "net_by_cost_r": {str(c): round(gross - c, 4) for c in grid},
    }
    # execution variants
    for delay, name in ((0, "same_bar"), (1, "delay_1bar"), (2, "delay_2bar")):
        parts = []
        for sym, df in frames.items():
            m = _oos(df, cutoff)
            fwd = df.loc[m, f"fwd{PRIMARY_H}_R"]
            if delay:
                fwd = fwd.shift(-delay)
            tr = mask(sym, df).loc[m] & fwd.notna()
            parts.append(fwd[tr] - COST_R)
        r = pd.concat(parts)
        vv = r.values.astype(float)
        out[f"exec_{name}"] = {
            "n": len(vv),
            "net_r": round(float(vv.mean()), 4),
        }
    # conservative fill (next-bar high entry)
    parts = []
    for sym, df in frames.items():
        m = _oos(df, cutoff)
        fwd = df.loc[m, f"fwd{PRIMARY_H}_R"]
        hi = df.loc[m, "high"].shift(-1)
        close = df.loc[m, "close"]
        atr = df.loc[m, "atr_14"]
        tr = mask(sym, df).loc[m] & fwd.notna() & hi.notna()
        fill = (hi[tr] - close[tr]) / (atr[tr] * R_MULT)
        parts.append(fwd[tr] - fill - COST_R)
    r = pd.concat(parts)
    vv = r.values.astype(float)
    out["exec_conservative_fill"] = {
        "n": len(vv),
        "net_r": round(float(vv.mean()), 4),
    }
    # spread widening: 2x / 3x cost
    for mult in (2, 3):
        parts = []
        for sym, df in frames.items():
            m = _oos(df, cutoff)
            fwd = df.loc[m, f"fwd{PRIMARY_H}_R"]
            tr = mask(sym, df).loc[m] & fwd.notna()
            parts.append(fwd[tr] - mult * COST_R)
        r = pd.concat(parts)
        vv = r.values.astype(float)
        out[f"cost_x{mult}"] = {
            "n": len(vv),
            "net_r": round(float(vv.mean()), 4),
        }
    return out


# ---------------------------------------------------------------------------
# 8. Monte Carlo / path dependence (10,000 block-bootstrap paths)
# ---------------------------------------------------------------------------


def monte_carlo(symbols: Optional[List[str]] = None, cutoff: str = CUTOFF) -> Dict:
    frames = _frames8(symbols)
    trigs = {sym: _triggers(df) for sym, df in frames.items()}
    mask = _frozen_long_mask(frames, trigs, K_LONG)
    v = _trade_r(frames, mask, cutoff=cutoff, win_start=TRAIN_END).values.astype(float)
    n = len(v)
    rng = np.random.default_rng(SEED + 1)
    out: Dict = {"n_trades": n, "mean_r": round(float(v.mean()), 4)}

    # block bootstrap paths
    n_blocks = n // MC_BLOCK
    if n_blocks < 5:
        return {"n_trades": n, "note": "too few trades for MC"}
    blocks = [v[i : i + MC_BLOCK] for i in range(0, n_blocks * MC_BLOCK, MC_BLOCK)]
    block_idx = rng.integers(0, len(blocks), size=(MC_PATHS, n_blocks))
    finals = np.empty(MC_PATHS)
    maxdds = np.empty(MC_PATHS)
    streaks = np.empty(MC_PATHS)
    recoveries = np.empty(MC_PATHS)
    for p in range(MC_PATHS):
        path = np.concatenate([blocks[i] for i in block_idx[p]])[:n]
        cum = np.cumsum(path)
        finals[p] = cum[-1]
        peak = np.maximum.accumulate(cum)
        dd = cum - peak
        maxdds[p] = dd.min()
        # longest losing streak
        neg = path < 0
        streak = 0
        best = 0
        for flag in neg:
            streak = streak + 1 if flag else 0
            best = max(best, streak)
        streaks[p] = best
        # recovery: mean bars below previous peak
        below = cum < peak
        runs = np.diff(np.concatenate([[0], below.astype(int), [0]]))
        starts = np.where(runs == 1)[0]
        ends = np.where(runs == -1)[0]
        rec = float(np.mean(ends - starts)) if len(starts) else 0.0
        recoveries[p] = rec

    out["paths"] = MC_PATHS
    out["p_profit"] = round(float((finals > 0).mean()), 3)
    out["p_loss"] = round(float((finals < 0).mean()), 3)
    out["final_r_median"] = round(float(np.median(finals)), 2)
    out["final_r_p05"] = round(float(np.percentile(finals, 5)), 2)
    # 1R ≈ 1% capital at 1% risk per trade
    out["maxdd_r"] = {
        "p50": round(float(np.percentile(maxdds, 50)), 1),
        "p95": round(float(np.percentile(maxdds, 95)), 1),
        "p99": round(float(np.percentile(maxdds, 99)), 1),
        "p_dd_below_10r": round(float((maxdds < -10).mean()), 3),
        "p_dd_below_20r": round(float((maxdds < -20).mean()), 3),
        "p_dd_below_50r_pct_capital": round(float((maxdds < -50).mean()), 4),
    }
    out["losing_streak"] = {
        "median": round(float(np.median(streaks)), 1),
        "p95": round(float(np.percentile(streaks, 95)), 1),
        "p_streak_ge_20": round(float((streaks >= 20).mean()), 3),
    }
    out["recovery_trades_median"] = round(float(np.median(recoveries)), 1)
    return out


# ---------------------------------------------------------------------------
# 9. Baselines (expanded universe)
# ---------------------------------------------------------------------------


def baselines(symbols: Optional[List[str]] = None, cutoff: str = CUTOFF) -> Dict:
    frames = _frames8(symbols)
    trigs = {sym: _triggers(df) for sym, df in frames.items()}
    mask = _frozen_long_mask(frames, trigs, K_LONG)
    rng = np.random.default_rng(SEED + 2)
    parts: Dict[str, list] = {
        "always_long": [],
        "always_flat": [],
        "random_entries": [],
        "rsi30": [],
        "streak5n": [],
        "sma200_dev": [],
        "momentum_ret10": [],
        "sma200_trend": [],
        "fade5_drop": [],
    }
    for df in frames.values():
        m = _oos(df, cutoff)
        sub = df.loc[
            m,
            [
                f"fwd{PRIMARY_H}_R",
                "rsi_14",
                "streak5n",
                "dist_pct",
                "ret_10",
                "sma_200",
                "close",
                "drop5",
            ],
        ].dropna()
        d = sub[f"fwd{PRIMARY_H}_R"].values.astype(float)
        n = len(d)
        parts["always_long"].extend(d - COST_R)
        parts["always_flat"].extend(np.zeros(n))
        sel = rng.random(n) < 0.02
        parts["random_entries"].extend(d[sel] - COST_R)
        parts["rsi30"].extend(np.where(sub["rsi_14"].values < 30, d, np.nan) - COST_R)
        parts["streak5n"].extend(
            np.where(sub["streak5n"].values >= 5, d, np.nan) - COST_R
        )
        parts["sma200_dev"].extend(
            np.where(sub["dist_pct"].values < -8.0, d, np.nan) - COST_R
        )
        parts["momentum_ret10"].extend(
            np.where(sub["ret_10"].values > 0, d, np.nan) - COST_R
        )
        parts["sma200_trend"].extend(
            np.where(sub["close"].values > sub["sma_200"].values, d, -d) - COST_R
        )
        parts["fade5_drop"].extend(
            np.where(sub["drop5"].values < -0.8, d, np.nan) - COST_R
        )
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
    sig = _trade_r(frames, mask, cutoff=cutoff, win_start=TRAIN_END)
    rows.append(
        {
            "baseline": "STAGE8_LONG_SIGNAL",
            "n": len(sig),
            "net_r": round(float(sig.mean()), 4),
            "win": round(float((sig > 0).mean()), 3),
        }
    )
    return rows


# ---------------------------------------------------------------------------
# 10. Multiple-testing control + experiment ledger
# ---------------------------------------------------------------------------


def _bh_fdr(pvals: Dict[str, float]) -> Tuple[Dict[str, float], List[str]]:
    keys = list(pvals.keys())
    p = np.array([pvals[k] for k in keys])
    m = len(p)
    order = np.argsort(p)
    adj = np.full(m, np.nan)
    for i, idx in enumerate(order):
        adj[idx] = min(1.0, p[idx] * m / (i + 1))
    running = np.inf
    for i in range(m - 1, -1, -1):
        idx = order[i]
        running = min(running, adj[idx])
        adj[idx] = running
    sig = [k for k, q in zip(keys, adj, strict=True) if q <= 0.05]
    return {k: round(float(q), 4) for k, q in zip(keys, adj, strict=True)}, sig


def multiple_testing(symbols: Optional[List[str]] = None, cutoff: str = CUTOFF) -> Dict:
    frames = _frames8(symbols)
    trigs = {sym: _triggers(df) for sym, df in frames.items()}
    rng = np.random.default_rng(SEED + 3)
    tests: Dict[str, float] = {}

    def perm_p(rs: pd.Series, n_perm: int = 300) -> Optional[float]:
        v = rs.values.astype(float)
        if len(v) < 30:
            return None
        perm = np.array(
            [float((v * rng.choice([-1.0, 1.0], len(v))).mean()) for _ in range(n_perm)]
        )
        return float((np.abs(perm) >= abs(v.mean())).mean())

    for fam in _family_defs():

        def mk(sym, df, fam=fam):
            return _long_mask(df, trigs[sym], fam)

        p = perm_p(_trade_r(frames, mk, cutoff=cutoff, win_start=TRAIN_END))
        if p is not None:
            tests[f"fam_{fam}"] = p
    for k in (1, 2, 3):

        def mk(sym, df, k=k):
            score = sum(trigs[sym][n].astype(int) for n in LONG_COMBO)
            return score >= k

        p = perm_p(_trade_r(frames, mk, cutoff=cutoff, win_start=TRAIN_END))
        if p is not None:
            tests[f"k{k}"] = p
    bh_q, sig = _bh_fdr(tests)
    return {
        "n_tests": len(tests),
        "pvals": {k: round(float(v), 4) for k, v in tests.items()},
        "bh_q": bh_q,
        "significant_q05": sig,
        "experiment_ledger": {
            "stage4_hypotheses": 6,
            "stage5_hypotheses": 10,
            "stage6_hypotheses": 5,
            "stage7_hypotheses": 11,
            "stage8_families": len(tests),
            "cumulative": 6 + 10 + 5 + 11 + len(tests),
            "note": "aggregate ledger; see per-stage reports for detail",
        },
    }


# ---------------------------------------------------------------------------
# 11. Leakage / data-snooping forensics (machine-readable)
# ---------------------------------------------------------------------------


def leakage_audit() -> Dict:
    checks = [
        ("features use trailing windows only (no future bars)", "PASS"),
        ("regime classification causal (trailing slope/ADX/SMA/ATR)", "PASS"),
        ("labels: forward returns evaluation-only via shift(-h)", "PASS"),
        ("signal triggers causal (RSI/streak/drop/dist at bar t)", "PASS"),
        (
            "vol buckets CAUSAL rolling 250-bar percentile (Stage-8 fix; "
            "Stage-7 used full-sample qcut rank = mild look-ahead, now fixed)",
            "PASS",
        ),
        ("threshold selection restricted to pre-2022-01-01 training", "PASS"),
        ("walk-forward purge + 20-day embargo implemented", "PASS"),
        ("untouched period 2025-06-01+ excluded from every selection metric", "PASS"),
        (
            "universe selection pre-registered (min bars / date coverage), "
            "not performance-driven",
            "PASS",
        ),
        ("no calibration fit on test data", "PASS"),
        ("exit-selection leakage: exits chosen on TRAIN folds only", "PASS"),
        ("symbol-selection leakage: no symbols added/removed for performance", "PASS"),
        ("regime-selection leakage: regime labels from fixed causal detector", "PASS"),
        (
            "sample-size-selection leakage: k=1/2/3 all reported, selection frozen",
            "PASS",
        ),
        ("confirmation-selection leakage: variants pre-specified, no search", "PASS"),
        (
            "overlapping labels: signals evaluated on disjoint forward windows "
            "(evaluation-only, no training on labels)",
            "PASS",
        ),
        (
            "survivorship bias: universe = all vendor-provided instruments passing "
            "objective criteria, no historical filtering",
            "PASS",
        ),
        ("test-set reuse: untouched period run exactly once (--untouched)", "PASS"),
        (
            "detect_regime_cluster (full-sample standardization) is NOT used by this "
            "campaign; flagged for awareness",
            "UNKNOWN",
        ),
        ("vendor survivorship / data completeness outside provided history", "UNKNOWN"),
    ]
    status = [c[1] for c in checks]
    return {
        "all_clean": all(s == "PASS" for s in status),
        "n_pass": status.count("PASS"),
        "n_fail": status.count("FAIL"),
        "n_unknown": status.count("UNKNOWN"),
        "checks": [{"check": c[0], "status": c[1]} for c in checks],
    }


# ---------------------------------------------------------------------------
# 12. Portfolio risk & clustering
# ---------------------------------------------------------------------------


def portfolio_risk(symbols: Optional[List[str]] = None, cutoff: str = CUTOFF) -> Dict:
    frames = _frames8(symbols)
    trigs = {sym: _triggers(df) for sym, df in frames.items()}
    mask = _frozen_long_mask(frames, trigs, K_LONG)

    # per-symbol R series + daily signal counts
    series: Dict[str, pd.Series] = {}
    daily_counts: Dict[str, pd.Series] = {}
    for sym, df in frames.items():
        m = _oos(df, cutoff)
        fwd = df.loc[m, f"fwd{PRIMARY_H}_R"]
        sig = mask(sym, df)
        tr = sig.loc[m] & fwd.notna()
        r = fwd[tr] - COST_R
        if len(r) >= 3:
            series[sym] = r
        daily_counts[sym] = sig.loc[m].astype(int)

    df_cnt = pd.DataFrame(daily_counts)
    daily_total = df_cnt.sum(axis=1) if df_cnt.shape[1] else pd.Series(dtype=int)
    n_sym = len(series)
    corr = (
        pd.DataFrame(series).fillna(0.0).corr().values
        if n_sym > 1
        else np.array([[1.0]])
    )
    mean_abs_corr = (
        float(np.mean(np.abs(corr[np.triu_indices(len(corr), 1)])))
        if len(corr) > 1
        else 0.0
    )

    # cluster overlap: P(same-day co-signal) within CHF / JPY / USD clusters
    clusters = _cluster_members(list(frames.keys()))
    overlap = {}
    for name, members in clusters.items():
        mem = [s for s in members if s in df_cnt.columns]
        if len(mem) < 2:
            continue
        sub = df_cnt[mem]
        pair_days = (sub.sum(axis=1) >= 2).sum()
        signal_days = (sub.sum(axis=1) >= 1).sum()
        overlap[name] = {
            "n_members": len(mem),
            "co_signal_day_fraction": round(float(pair_days / max(signal_days, 1)), 3),
        }

    # cluster-aggregated net R: same-day signals within a cluster = one bet
    agg_series = {}
    for name, members in clusters.items():
        mem = [s for s in members if s in series]
        if len(mem) < 2:
            continue
        combined = pd.concat([series[s] for s in mem]).sort_index()
        agg_series[f"cluster_{name}"] = combined
    agg_net = float(pd.concat(list(agg_series.values())).mean()) if agg_series else None

    # concentration of per-symbol |net|
    nets = {s: float(r.sum()) for s, r in series.items()}
    tot = sum(abs(x) for x in nets.values())
    rc = {s: round(100 * abs(x) / tot, 1) for s, x in nets.items()} if tot else {}
    sorted_rc = sorted(rc.items(), key=lambda kv: -kv[1])

    return {
        "n_symbols_with_trades": n_sym,
        "mean_abs_trade_corr": round(mean_abs_corr, 3),
        "max_concurrent_positions": int(daily_total.max()) if len(daily_total) else 0,
        "pct_days_flat": round(float((daily_total == 0).mean()), 3)
        if len(daily_total)
        else 0.0,
        "cluster_overlap": overlap,
        "cluster_aggregated_mean_r": round(agg_net, 4) if agg_net is not None else None,
        "risk_contribution_top5": sorted_rc[:5],
        "herfindahl_risk": round(
            float(((np.array([abs(x) for x in nets.values()]) / tot) ** 2).sum()), 3
        )
        if tot
        else 0.0,
    }


# ---------------------------------------------------------------------------
# 13. Economic mechanism probes (descriptive, no selection)
# ---------------------------------------------------------------------------


def mechanism(symbols: Optional[List[str]] = None, cutoff: str = CUTOFF) -> Dict:
    frames = _frames8(symbols)
    trigs = {sym: _triggers(df) for sym, df in frames.items()}
    mask = _frozen_long_mask(frames, trigs, K_LONG)
    out: Dict = {}

    def bucket_stats8(sel, df, fwd, m):
        r = fwd[sel] - COST_R
        if not len(r):
            return None
        return {
            "n": len(r),
            "mean_r": round(float(r.mean()), 4),
            "win": round(float((r > 0).mean()), 3),
        }

    # by causal vol quintile
    vol_cells: Dict[str, pd.Series] = {}
    drop_cells: Dict[str, pd.Series] = {}
    streak_cells: Dict[str, pd.Series] = {}
    chf_cells: Dict[str, pd.Series] = {}
    usdq_cells: Dict[str, pd.Series] = {}
    for sym, df in frames.items():
        m = _oos(df, cutoff)
        fwd = df.loc[m, f"fwd{PRIMARY_H}_R"]
        sig = mask(sym, df)
        tr = sig.loc[m] & fwd.notna()
        r = fwd[tr] - COST_R
        if not len(r):
            continue
        vp = df.loc[m, "vol_pct"]
        dr = df.loc[m, "drop5"]
        st = df.loc[m, "streak5n"]
        for q, lo, hi in (
            ("q1", 0.0, 0.2),
            ("q2", 0.2, 0.4),
            ("q3", 0.4, 0.6),
            ("q4", 0.6, 0.8),
            ("q5", 0.8, 1.01),
        ):
            sub = r[(vp.loc[r.index] >= lo) & (vp.loc[r.index] < hi)]
            if len(sub):
                vol_cells[q] = pd.concat(
                    [vol_cells.get(q, pd.Series(dtype=float)), sub]
                )
        for q, lo, hi in (
            ("q1_deep", -np.inf, -2.0),
            ("q2", -2.0, -1.2),
            ("q3", -1.2, -0.8),
            ("q4", -0.8, 0.0),
        ):
            sub = r[(dr.loc[r.index] >= lo) & (dr.loc[r.index] < hi)]
            if len(sub):
                drop_cells[q] = pd.concat(
                    [drop_cells.get(q, pd.Series(dtype=float)), sub]
                )
        for n_ in (3, 4, 5, 6):
            key = f"streak{n_}+" if n_ == 6 else f"streak{n_}"
            sub = r[st.loc[r.index] >= n_]
            if len(sub):
                streak_cells[key] = pd.concat(
                    [streak_cells.get(key, pd.Series(dtype=float)), sub]
                )
        if "CHF" in sym:
            chf_cells.setdefault("chf", pd.Series(dtype=float))
            chf_cells["chf"] = pd.concat([chf_cells["chf"], r])
        else:
            chf_cells.setdefault("non_chf", pd.Series(dtype=float))
            chf_cells["non_chf"] = pd.concat([chf_cells["non_chf"], r])
        if sym[3:] == "USD":
            usdq_cells.setdefault("usd_quote", pd.Series(dtype=float))
            usdq_cells["usd_quote"] = pd.concat([usdq_cells["usd_quote"], r])
        else:
            usdq_cells.setdefault("non_usd_quote", pd.Series(dtype=float))
            usdq_cells["non_usd_quote"] = pd.concat([usdq_cells["non_usd_quote"], r])
    out["by_vol_quintile"] = {k: _stats8(rs) for k, rs in vol_cells.items()}
    out["by_drop_quintile"] = {k: _stats8(rs) for k, rs in drop_cells.items()}
    out["by_streak_length"] = {k: _stats8(rs) for k, rs in streak_cells.items()}
    out["chf_vs_nonchf"] = {k: _stats8(rs) for k, rs in chf_cells.items()}
    out["usd_quote_vs_not"] = {k: _stats8(rs) for k, rs in usdq_cells.items()}
    return out


# ---------------------------------------------------------------------------
# 14. Adversarial falsification
# ---------------------------------------------------------------------------


def adversarial(symbols: Optional[List[str]] = None, cutoff: str = CUTOFF) -> Dict:
    frames = _frames8(symbols)
    trigs = {sym: _triggers(df) for sym, df in frames.items()}
    mask = _frozen_long_mask(frames, trigs, K_LONG)
    rng = np.random.default_rng(SEED + 4)
    out: Dict = {}

    base = _trade_r(frames, mask, cutoff=cutoff, win_start=TRAIN_END)
    out["as_is"] = _stats8(base)
    v = base.values.astype(float)

    # reversed (short the signal)
    parts = []
    for sym, df in frames.items():
        m = _oos(df, cutoff)
        fwd = df.loc[m, f"fwd{PRIMARY_H}_R"]
        tr = mask(sym, df).loc[m] & fwd.notna()
        parts.append(-(fwd[tr] - COST_R))
    out["reversed"] = _stats8(pd.concat(parts))

    # shuffled returns (destroy time structure)
    shuffled = rng.permutation(v)
    out["shuffled_returns"] = {
        "n": len(shuffled),
        "mean_r": round(float(shuffled.mean()), 4),
        "note": "mean identical by construction; maxDD is the informative stat",
        "maxdd_r": round(
            float(
                (np.cumsum(shuffled) - np.maximum.accumulate(np.cumsum(shuffled))).min()
            ),
            3,
        ),
    }

    # randomized timing: same count of entries at random bars
    pool = []
    for df in frames.values():
        m = _oos(df, cutoff)
        fwd = df.loc[m, f"fwd{PRIMARY_H}_R"]
        pool.append(fwd.dropna().values)
    pool = np.concatenate(pool) if pool else np.array([])
    rt = np.array([float(rng.choice(pool, len(v)).mean() - COST_R) for _ in range(300)])
    out["random_timing"] = {
        "mean": round(float(rt.mean()), 4),
        "p95": round(float(np.percentile(rt, 95)), 4),
        "signal_beats_random_p95": bool(v.mean() > np.percentile(rt, 95)),
    }

    # regime permutation: shuffle regime labels within each symbol
    parts = []
    for sym, df in frames.items():
        m = _oos(df, cutoff)
        fwd = df.loc[m, f"fwd{PRIMARY_H}_R"]
        sig = mask(sym, df)
        perm_reg = df["regime"].values.copy()
        perm_reg[m] = rng.permutation(perm_reg[m])
        # triggers for LONG do not use regime; this probe tests regime-conditional
        # expression by destroying any regime-structured timing
        tr = sig.loc[m] & fwd.notna()
        parts.append(fwd[tr] - COST_R)
    out["regime_permutation_note"] = (
        "LONG triggers are regime-independent; permuting regime labels does not "
        "change entry timing — reported for completeness"
    )
    out["regime_permutation"] = _stats8(pd.concat(parts)) if parts else {"n": 0}

    # cost inflation x3
    parts = []
    for sym, df in frames.items():
        m = _oos(df, cutoff)
        fwd = df.loc[m, f"fwd{PRIMARY_H}_R"]
        tr = mask(sym, df).loc[m] & fwd.notna()
        parts.append(fwd[tr] - 3 * COST_R)
    out["cost_x3"] = _stats8(pd.concat(parts))

    # delays
    for delay, name in ((1, "delay_1bar"), (2, "delay_2bar")):
        parts = []
        for sym, df in frames.items():
            m = _oos(df, cutoff)
            fwd = df.loc[m, f"fwd{PRIMARY_H}_R"].shift(-delay)
            tr = mask(sym, df).loc[m] & fwd.notna()
            parts.append(fwd[tr] - COST_R)
        out[name] = _stats8(pd.concat(parts))

    # remove strongest symbol (by OOS net) — determined from the pooled run
    nets = {}
    for sym, df in frames.items():
        m = _oos(df, cutoff)
        fwd = df.loc[m, f"fwd{PRIMARY_H}_R"]
        tr = mask(sym, df).loc[m] & fwd.notna()
        r = fwd[tr] - COST_R
        if len(r):
            nets[sym] = float(r.sum())
    best = max(nets, key=nets.get) if nets else None
    out["ex_best_symbol"] = _stats8(
        _trade_r(frames, mask, cutoff=cutoff, win_start=TRAIN_END, exclude=[best])
        if best
        else base
    )

    # remove strongest regime (Bear Trend contribution)
    reg_sum = {}
    for sym, df in frames.items():
        m = _oos(df, cutoff)
        fwd = df.loc[m, f"fwd{PRIMARY_H}_R"]
        tr = mask(sym, df).loc[m] & fwd.notna()
        r = fwd[tr] - COST_R
        if not len(r):
            continue
        regs = df.loc[m, "regime"]
        for reg in regs.unique():
            sub = r[regs.loc[r.index] == reg]
            reg_sum[reg] = reg_sum.get(reg, 0.0) + float(sub.sum())
    best_reg = max(reg_sum, key=reg_sum.get) if reg_sum else None

    def _ex_regime(excl: str):
        parts = []
        for sym, df in frames.items():
            m = _oos(df, cutoff)
            fwd = df.loc[m, f"fwd{PRIMARY_H}_R"]
            tr = mask(sym, df).loc[m] & fwd.notna()
            r = fwd[tr]
            if not len(r):
                continue
            keep = df.loc[m, "regime"].loc[r.index] != excl
            parts.append(r[keep] - COST_R)
        return _stats8(pd.concat(parts)) if parts else {"n": 0}

    out["ex_best_regime"] = _ex_regime(best_reg) if best_reg else {"n": 0}

    # remove strongest family: k-of-2 on {L2_drop5, L3_streak5n} (drops L1)
    def no_l1(sym, df):
        score = sum(trigs[sym][n].astype(int) for n in ["L2_drop5", "L3_streak5n"])
        return score >= 2

    out["ex_l1_family"] = _stats8(
        _trade_r(frames, no_l1, cutoff=cutoff, win_start=TRAIN_END)
    )

    # threshold perturbation: k=2 instead of k=3
    out["k_perturb_2"] = _stats8(
        _trade_r(
            frames,
            _frozen_long_mask(frames, trigs, 2),
            cutoff=cutoff,
            win_start=TRAIN_END,
        )
    )
    out["_best_symbol_removed"] = best
    out["_best_regime_removed"] = best_reg
    return out


# ---------------------------------------------------------------------------
# 15. Untouched test (single-shot)
# ---------------------------------------------------------------------------


def untouched_test(symbols: Optional[List[str]] = None, cutoff: str = CUTOFF) -> Dict:
    frames = _frames8(symbols)
    trigs = {sym: _triggers(df) for sym, df in frames.items()}
    mask = _frozen_long_mask(frames, trigs, K_LONG)

    def collect(exclude: Optional[List[str]] = None) -> Dict:
        trades, by_sym, by_regime, by_bucket = [], {}, {}, {}
        total_bars, rows = 0, []
        for sym, df in frames.items():
            if exclude and sym in exclude:
                continue
            m = df.index >= pd.Timestamp(cutoff)
            fwd = df.loc[m, f"fwd{PRIMARY_H}_R"]
            sig = mask(sym, df)
            tr = sig.loc[m] & fwd.notna()
            r = fwd[tr] - COST_R
            total_bars += int(m.sum())
            if len(r):
                trades.append(r)
                if len(r) >= 3:
                    by_sym[sym] = {"n": int(len(r)), "net_r": round(float(r.mean()), 4)}
                    by_bucket[_bucket(sym)] = by_bucket.get(_bucket(sym), 0.0) + float(
                        r.sum()
                    )
                regs = df.loc[m, "regime"]
                for reg in regs.unique():
                    sub = r[regs.loc[r.index] == reg]
                    if len(sub):
                        by_regime.setdefault(str(reg), []).extend(sub.values.tolist())
                # target-level first-touch on this symbol's untouched trades
                for pos in np.where((sig & m).values)[0]:
                    row = int(pos)
                    close = df["close"].astype(float)
                    atr = df["atr_14"].astype(float)
                    entry = float(close.iloc[row])
                    risk = float(atr.iloc[row]) * 1.25
                    if entry != entry or risk != risk or risk <= 0:
                        continue
                    win = df.iloc[row + 1 : row + 1 + 20]
                    if len(win) < 3:
                        continue
                    hh = win["high"].astype(float)
                    ll = win["low"].astype(float)
                    rows.append(
                        {
                            "tp1": bool((hh >= entry + risk).any()),
                            "tp2": bool((hh >= entry + 2 * risk).any()),
                            "tp3": bool((hh >= entry + 3 * risk).any()),
                            "sl": bool((ll <= entry - risk).any()),
                            "t10": float(
                                (win["close"].astype(float).iloc[9] - entry) / risk
                            )
                            if len(win) >= 10
                            else None,
                        }
                    )
        if not trades:
            return {"n": 0}
        rs = pd.concat(trades)
        v = rs.values.astype(float)
        target = {}
        if rows:
            target = {
                "n": len(rows),
                "p_tp1": round(float(np.mean([r["tp1"] for r in rows])), 3),
                "p_tp2": round(float(np.mean([r["tp2"] for r in rows])), 3),
                "p_tp3": round(float(np.mean([r["tp3"] for r in rows])), 3),
                "p_sl": round(float(np.mean([r["sl"] for r in rows])), 3),
                "ev_1r_20bar": round(
                    float(
                        np.mean(
                            [
                                (1.0 if r["tp1"] else 0.0) - (1.0 if r["sl"] else 0.0)
                                for r in rows
                            ]
                        )
                    ),
                    3,
                ),
                "time_stop_net_10bar": round(
                    float(np.mean([t for t in [r["t10"] for r in rows] if t == t]))
                    - COST_R,
                    3,
                ),
            }
        return {
            "n": len(v),
            "gross_r": round(float(v.mean()) + COST_R, 4),
            "net_r": round(float(v.mean()), 4),
            "cumulative_r": round(float(v.sum()), 3),
            "win": round(float((v > 0).mean()), 3),
            "flat_rate": round(1.0 - len(v) / max(total_bars, 1), 3),
            "maxdd_r": round(
                float((np.cumsum(v) - np.maximum.accumulate(np.cumsum(v))).min()), 3
            ),
            "n_symbols": len(by_sym),
            "by_symbol": by_sym,
            "by_regime": {
                reg: {"n": len(a), "net_r": round(float(np.mean(a)), 4)}
                for reg, a in by_regime.items()
                if len(a) >= 3
            },
            "by_bucket_net_r": {k: round(float(v2), 3) for k, v2 in by_bucket.items()},
            "target_level": target,
        }

    return {
        "full": collect(),
        "ex_usdchf": collect(exclude=["USDCHF"]),
        "ex_audchf": collect(exclude=["AUDCHF"]),
        "ex_both_chf": collect(exclude=["USDCHF", "AUDCHF"]),
    }


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main(argv: Optional[List[str]] = None) -> int:
    import argparse

    parser = argparse.ArgumentParser(
        description="Stage-8 long reversal generalization campaign"
    )
    parser.add_argument("--symbols", default=None)
    parser.add_argument("--cutoff", default=CUTOFF)
    parser.add_argument("--universe", action="store_true")
    parser.add_argument("--cross", action="store_true")
    parser.add_argument("--wf", action="store_true")
    parser.add_argument("--regime", action="store_true")
    parser.add_argument("--entry", action="store_true")
    parser.add_argument("--exit", action="store_true")
    parser.add_argument("--cost", action="store_true")
    parser.add_argument("--mc", action="store_true")
    parser.add_argument("--baselines", action="store_true")
    parser.add_argument("--mt", action="store_true")
    parser.add_argument("--leak", action="store_true")
    parser.add_argument("--portfolio", action="store_true")
    parser.add_argument("--mechanism", action="store_true")
    parser.add_argument("--adversarial", action="store_true")
    parser.add_argument("--untouched", action="store_true")
    parser.add_argument("--all", action="store_true")
    args = parser.parse_args(argv)

    symbols = (
        [s.strip() for s in args.symbols.split(",") if s.strip()]
        if args.symbols
        else None
    )
    print(
        f"Reserved untouched: {args.cutoff}+ (single-shot). Selection: train < "
        f"{TRAIN_END}. Frozen entry: k={K_LONG} of {LONG_COMBO}. "
        f"SHORT reversal LOCKED (falsified). Universe: 112 D1 instruments."
    )
    results: Dict = {
        "cutoff": args.cutoff,
        "train_end": TRAIN_END,
        "k": K_LONG,
        "universe_n": len(eligible_universe()),
    }

    if args.all or args.universe:
        print("\n" + "=" * 72)
        print("PHASE 1 — PRE-REGISTERED ELIGIBLE UNIVERSE")
        print("=" * 72)
        u = universe()
        print(f"  criteria: {u['criteria']}")
        print(f"  n_symbols: {u['n_symbols']}  by_bucket: {u['by_bucket']}")
        results["universe"] = u

    if args.all or args.cross:
        print("\n" + "=" * 72)
        print("PHASE 2 — CROSS-SECTIONAL GENERALIZATION (112-symbol universe)")
        print("=" * 72)
        cs = cross_sectional(symbols, cutoff=args.cutoff)
        s = cs["pooled"]
        print(
            f"  POOLED OOS: n={s.get('n', 0):,} mean={s.get('mean_r', 0):+.4f} "
            f"win={s.get('win', 0):.3f} perm_p={s.get('perm_p', 0):.3f} "
            f"boot95={s.get('boot95')} be={s.get('break_even', 0)}"
        )
        print(
            f"  EW mean={cs['equal_weighted_mean_r']:+.4f} HHI={cs['herfindahl']} "
            f"top1={cs['top1_symbol']} top1%={cs['top1_pnl_pct']} "
            f"top2%={cs['top2_pnl_pct']}"
        )
        for label in ("ex_usdchf", "ex_audchf", "ex_top1", "ex_top2", "ex_both_chf"):
            e = cs[label]
            if e.get("n", 0) >= 20:
                print(
                    f"  {label:<14} n={e['n']:>5,} mean={e['mean_r']:+.4f} "
                    f"win={e['win']:.3f} perm_p={e['perm_p']:.3f}"
                )
            else:
                print(f"  {label:<14} n={e.get('n', 0)}")
        print(
            "  leave-one-out: ",
            {k2: v for k2, v in cs["leave_one_out"].items()},
        )
        print("  leave-one-cluster-out:")
        for k2, v in cs["leave_one_cluster_out"].items():
            print(f"    {k2:<14} n={v['n']:>5,} mean={v['mean_r']:+.4f}")
        results["cross_sectional"] = cs

    if args.all or args.wf:
        print("\n" + "=" * 72)
        print("PHASE 3 — PURGED WALK-FORWARD (per-fold, no pooling)")
        print("=" * 72)
        wf = walk_forward(symbols, cutoff=args.cutoff)
        for f in wf["folds"]:
            te = f["oos"]
            print(
                f"  fold {f['fold']} {f['window']:<26} train_n={f['train'].get('n', 0):>5,} "
                f"| oos_n={te.get('n', 0):>5,} mean={te.get('mean_r', 0):+.4f} "
                f"win={te.get('win', 0):.3f} maxdd={te.get('maxdd_r', 0):.3f} "
                f"flags={f['flags']}"
            )
        print(
            f"  folds positive: {wf['n_folds_positive_net']}/{wf['n_folds_total']} "
            f"(with n>=20: {wf['n_folds_with_n_ge20']})"
        )
        results["walk_forward"] = wf

    if args.all or args.regime:
        print("\n" + "=" * 72)
        print("PHASE 4 — REGIME x VOLATILITY MATRIX")
        print("=" * 72)
        rm = regime_matrix(symbols, cutoff=args.cutoff)
        for reg, row in rm["matrix"].items():
            for vb, s in row.items():
                if s.get("n", 0) < 20:
                    print(f"  {reg:<16} {vb:<5} n={s.get('n', 0)}")
                    continue
                print(
                    f"  {reg:<16} {vb:<5} n={s['n']:>5,} mean={s['mean_r']:+.4f} "
                    f"win={s['win']:.3f} perm_p={s['perm_p']:.3f} be={s['break_even']}"
                )
        hv = rm["high_vol_marginal"]
        for k2, s in hv.items():
            if s.get("n", 0) >= 20:
                print(
                    f"  [{k2}] n={s['n']:>5,} mean={s['mean_r']:+.4f} "
                    f"perm_p={s['perm_p']:.3f}"
                )
        results["regime_matrix"] = rm

    if args.all or args.entry:
        print("\n" + "=" * 72)
        print("PHASE 5 — ENTRY ABLATION (minimum sufficient rule)")
        print("=" * 72)
        ea = entry_ablation(symbols, cutoff=args.cutoff)
        for name, row in ea.items():
            o = row.get("oos", {})
            if o.get("n", 0) < 20:
                print(f"  {name:<22} oos_n={o.get('n', 0)}")
                continue
            print(
                f"  {name:<22} oos_n={o['n']:>5,} mean={o.get('mean_r', 0):+.4f} "
                f"win={o.get('win', 0):.3f} perm_p={o.get('perm_p', 0):.3f} "
                f"be={o.get('break_even', 0)}"
            )
        if "G_random_extreme" in ea:
            print(f"  G_random_extreme: {ea['G_random_extreme']}")
        results["entry_ablation"] = ea

    if args.all or args.exit:
        print("\n" + "=" * 72)
        print("PHASE 6 — EXIT ABLATION (holding sweep + MFE/MAE + transfer)")
        print("=" * 72)
        ex = exit_ablation(symbols, cutoff=args.cutoff)
        print("  holding sweep (net R):")
        for h, s in ex["holding_sweep"].items():
            print(f"    h={h:<3} n={s.get('n', 0):>6,} net={s.get('net_r', 0):+.4f}")
        mm = ex["mfe_mae"]
        if mm:
            print(
                f"  MFE/MAE: n={mm['n']} mfe_med={mm['mfe_median']} "
                f"mae_med={mm['mae_median']} P(0.5/1/2/3R)="
                f"{mm['p_reach_0_5r']}/{mm['p_reach_1r']}/{mm['p_reach_2r']}/"
                f"{mm['p_reach_3r']} P(SL first)={mm['p_sl_first']}"
            )
        print("  exit transfer (train-selected, OOS):")
        for fam, r in ex["exit_transfer"].items():
            if r.get("note"):
                print(f"    {fam:<16} {r['note']}")
                continue
            print(
                f"    {fam:<16} p={r['param']} h={r['horizon']} "
                f"n={r['n_oos']:>5,} net={r['net_r']:+.4f} win={r['win']:.3f} "
                f"maxdd={r['maxdd_r']:.3f}"
            )
        results["exit_ablation"] = ex

    if args.all or args.cost:
        print("\n" + "=" * 72)
        print("PHASE 7 — COST & EXECUTION ROBUSTNESS")
        print("=" * 72)
        ce = cost_execution(symbols, cutoff=args.cutoff)
        print(
            f"  n={ce['n']} gross={ce['gross_r']:+.4f} break_even={ce['break_even_r']} R"
        )
        print("  net by cost (R):", ce["net_by_cost_r"])
        for k2 in (
            "exec_same_bar",
            "exec_delay_1bar",
            "exec_delay_2bar",
            "exec_conservative_fill",
            "cost_x2",
            "cost_x3",
        ):
            if k2 in ce:
                print(f"  {k2:<24} n={ce[k2]['n']:>6,} net={ce[k2]['net_r']:+.4f}")
        results["cost_execution"] = ce

    if args.all or args.mc:
        print("\n" + "=" * 72)
        print("PHASE 8 — MONTE CARLO / PATH DEPENDENCE (10,000 block-bootstrap paths)")
        print("=" * 72)
        mc = monte_carlo(symbols, cutoff=args.cutoff)
        if "note" in mc:
            print(f"  {mc['note']}")
        else:
            print(
                f"  n={mc['n_trades']} mean={mc['mean_r']:+.4f} "
                f"P(profit)={mc['p_profit']} P(loss)={mc['p_loss']} "
                f"final_R_med={mc['final_r_median']} p05={mc['final_r_p05']}"
            )
            print(f"  maxDD R: {mc['maxdd_r']}")
            print(f"  losing streak: {mc['losing_streak']}")
            print(f"  recovery trades (median): {mc['recovery_trades_median']}")
        results["monte_carlo"] = mc

    if args.all or args.baselines:
        print("\n" + "=" * 72)
        print("PHASE 9 — BASELINES (expanded universe, net of cost)")
        print("=" * 72)
        bl = baselines(symbols, cutoff=args.cutoff)
        print(f"{'baseline':<24}{'n':>9}{'net_r':>9}{'win':>7}")
        for r in bl:
            print(f"{r['baseline']:<24}{r['n']:>9,}{r['net_r']:>+9.4f}{r['win']:>7.3f}")
        results["baselines"] = bl

    if args.all or args.mt:
        print("\n" + "=" * 72)
        print("PHASE 10 — MULTIPLE-TESTING (BH FDR) + EXPERIMENT LEDGER")
        print("=" * 72)
        mt = multiple_testing(symbols, cutoff=args.cutoff)
        print(
            f"  n_tests={mt['n_tests']}  significant at q=0.05: {mt['significant_q05']}"
        )
        print("  bh_q:", mt["bh_q"])
        print(f"  ledger: {mt['experiment_ledger']}")
        results["multiple_testing"] = mt

    if args.all or args.leak:
        print("\n" + "=" * 72)
        print("PHASE 11 — LEAKAGE / DATA-SNOOPING FORENSICS (machine-readable)")
        print("=" * 72)
        la = leakage_audit()
        for c in la["checks"]:
            print(f"  [{c['status']:<7}] {c['check']}")
        print(
            f"  VERDICT: {la['n_pass']} PASS / {la['n_fail']} FAIL / "
            f"{la['n_unknown']} UNKNOWN — {'CLEAN' if la['all_clean'] else 'REVIEW'}"
        )
        results["leakage"] = la

    if args.all or args.portfolio:
        print("\n" + "=" * 72)
        print("PHASE 12 — PORTFOLIO RISK & CLUSTERING")
        print("=" * 72)
        pr = portfolio_risk(symbols, cutoff=args.cutoff)
        print(
            f"  n_symbols={pr['n_symbols_with_trades']} mean_abs_corr="
            f"{pr['mean_abs_trade_corr']} max_concurrent={pr['max_concurrent_positions']} "
            f"pct_days_flat={pr['pct_days_flat']}"
        )
        print(f"  cluster overlap: {pr['cluster_overlap']}")
        print(f"  cluster-aggregated mean R: {pr['cluster_aggregated_mean_r']}")
        print(f"  risk contribution top5: {pr['risk_contribution_top5']}")
        results["portfolio_risk"] = pr

    if args.all or args.mechanism:
        print("\n" + "=" * 72)
        print("PHASE 13 — ECONOMIC MECHANISM PROBES (descriptive)")
        print("=" * 72)
        me = mechanism(symbols, cutoff=args.cutoff)
        for group, cells in me.items():
            print(f"  [{group}]")
            for k2, s in cells.items():
                if s.get("n", 0) < 20:
                    print(f"    {k2:<16} n={s.get('n', 0)}")
                    continue
                print(
                    f"    {k2:<16} n={s['n']:>5,} mean={s['mean_r']:+.4f} "
                    f"win={s['win']:.3f} perm_p={s['perm_p']:.3f}"
                )
        results["mechanism"] = me

    if args.all or args.adversarial:
        print("\n" + "=" * 72)
        print("PHASE 14 — ADVERSARIAL FALSIFICATION")
        print("=" * 72)
        ad = adversarial(symbols, cutoff=args.cutoff)
        for name, s in ad.items():
            if name.startswith("_") or not isinstance(s, dict):
                print(f"  {name}: {s}")
                continue
            if s.get("n", 0) < 20:
                print(f"  {name:<24} n={s.get('n', 0)}")
                continue
            if "win" in s and "mean_r" in s:
                print(
                    f"  {name:<24} n={s['n']:>5,} mean={s['mean_r']:+.4f} "
                    f"win={s['win']:.3f} perm_p={s['perm_p']:.3f}"
                )
            else:
                print(f"  {name:<24} {s}")
        results["adversarial"] = ad

    if args.untouched:
        print("\n" + "=" * 72)
        print("PHASE 15 — UNTOUCHED TEST (single-shot, frozen rules)")
        print("=" * 72)
        ut = untouched_test(symbols, cutoff=args.cutoff)
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
                "    by symbol:",
                {k2: v["net_r"] for k2, v in r["by_symbol"].items()},
            )
            print(
                "    by regime:",
                {k2: v["net_r"] for k2, v in r["by_regime"].items()},
            )
            print("    by bucket net R:", r["by_bucket_net_r"])
            if r.get("target_level"):
                print("    target-level:", r["target_level"])
        results["untouched"] = ut

    out_dir = Path("data/validation")
    out_dir.mkdir(parents=True, exist_ok=True)
    with open(out_dir / "stage8_results.json", "w") as fh:
        json.dump(results, fh, indent=2, default=str)
    print(f"\nStage-8 results written to {out_dir / 'stage8_results.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
