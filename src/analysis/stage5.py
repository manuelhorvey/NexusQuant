"""
NexusQuant — Stage-5 Reversal Alpha Expression & Economic Viability (research CLI).

Stage-4 established a reproducible mean-reversion effect in the feature set
(78/159 FDR-significant; vs_sma200 rank-IC -0.059 @20 bars, -0.114 in Bear
regimes) but no expression of it survives costs and the unified LGBM destroys
the univariate signal (OOS AUC 0.495). Stage-5 asks the single question:

    Can this reversal effect be expressed as a COST-POSITIVE,
    out-of-sample trading strategy?

Nothing in this module modifies production logic. Discipline:

  - Threshold selection happens ONLY on training data (pre-2022-01-01).
  - All reported economics are 2022-01-01 .. CUTOFF (strictly OOS).
  - CUTOFF (2025-06-01)+ remains reserved untouched for the eventual
    single-shot final test and is excluded from every number here.
  - Cost grid is mandatory: gross means nothing without break-even.

Analyses:

    1. reversal_benchmark — deterministic distance-from-SMA200 signals
       (raw %, ATR-normalized, rolling z-score, rolling percentile),
       LONG when stretched below / SHORT when stretched above; thresholds
       chosen on train, evaluated OOS; per-regime breakdown; FLAT rate.
    2. bear_rally_shorts — eight non-overlapping definitions of a
       stretched rally in Bear/Range regimes; which individual signal
       carries the short effect (spec Phase 4).
    3. high_vol_longs — oversold/excursion definitions in high-vol
       conditions; which carries the bounce (spec Phase 5).
    4. economic_horizon — gross/net expectancy + IC at 1..60 bars for the
       primary signal; the monetizable horizon where signal > cost.
    5. cost_breakeven — net expectancy grid 0..0.20 ATR per signal family;
       the maximum tolerable cost (spec Phase 8, mandatory).
    6. monotonicity — decile table of forward return vs extension; tests
       the monotone structure (spec Phase 11).
    7. model_comparison — MODELS A..E (single reversal feature -> all
       features) + monotone-constrained LGBM; why the tree ensemble
       destroys the signal (spec Phase 10).
    8. symbol_heterogeneity — per-symbol reversal IC, gross/net EV and
       break-even cost (spec Phase 12).
    9. cross_asset_timing — does risk-on/off context change reversal
       timing rather than direction (spec Phase 13).
   10. walk_forward — frozen simple rule, purged 3-fold walk-forward
       (spec Phase 14; preliminary).

Usage:
    python -m src.analysis.stage5 --symbols <list> --reversal --shorts --hv
    python -m src.analysis.stage5 --symbols <list> --models --hetero --wf
    python -m src.analysis.stage5 --symbols <list> --all
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

sys.path.append(str(Path(__file__).parent.parent.parent))

from src.analysis.stage4 import COST_ATR, CUTOFF, _fwd_atr, _load_base, _pre_cutoff

TRAIN_END = "2022-01-01"  # threshold-selection boundary (models' own split)
MAX_HOLD = 20
HORIZONS = (1, 3, 5, 10, 15, 20, 30, 40, 60)
COST_R = 0.05  # round-trip cost in R (1R = 1.25 x ATR)
COST_GRID = (0.0, 0.005, 0.01, 0.025, 0.05, 0.0625, 0.075, 0.10, 0.15, 0.20)

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


# ---------------------------------------------------------------------------
# Signal construction (deterministic, causal)
# ---------------------------------------------------------------------------


def _dist_features(df: pd.DataFrame) -> pd.DataFrame:
    """Distance-from-SMA200 in four normalizations + RSI/BB (all causal)."""
    close = df["close"].astype(float)
    atr = df["atr_14"].astype(float).replace(0.0, np.nan)
    sma200 = df["sma_200"]
    dist = close - sma200
    out = pd.DataFrame(index=df.index)
    out["dist_pct"] = (close / sma200 - 1.0) * 100
    out["dist_atr"] = dist / atr
    roll = dist.rolling(100, min_periods=60).std()
    out["dist_z"] = dist / roll.replace(0.0, np.nan)
    out["dist_pct_rank"] = dist.rolling(250, min_periods=150).apply(
        lambda x: (x <= x[-1]).mean() if len(x) else np.nan, raw=True
    )
    return out


def _risk_proxy(df: pd.DataFrame, window: int = 5) -> pd.Series:
    """Risk-on/off proxy: ATR-normalized window return of the frame."""
    close = df["close"].astype(float)
    atr = df["atr_14"].astype(float).replace(0.0, np.nan)
    return (close.pct_change(window) / atr).rename("risk")


# ---------------------------------------------------------------------------
# Phase 2/3: pure reversal benchmark (thresholds selected on TRAIN only)
# ---------------------------------------------------------------------------


def reversal_benchmark(
    symbols: List[str],
    data_dir: str = "data/raw",
    group: str = "full_fx",
    cutoff: str = CUTOFF,
    train_end: str = TRAIN_END,
) -> Dict:
    """Distance-from-SMA200 reversal rules.

    SHORT when extension above SMA200 exceeds +theta, LONG when below -theta,
    FLAT otherwise. theta grid per normalization; the best theta on TRAIN
    (pre-``train_end``) per family is selected once, then the SAME rule is
    reported strictly OOS (train_end .. cutoff). Symmetric and asymmetric
    thresholds are tested (do NOT assume symmetry).
    """
    frames = {}
    for sym in symbols:
        df = _load_base(sym, data_dir, group)
        if df is None or len(df) < 400:
            continue
        frames[sym] = pd.concat(
            [_fwd_atr(df, horizons=(10,)), _dist_features(df)], axis=1
        )

    families = {
        "dist_atr": {"col": "dist_atr", "grid": (1.0, 1.5, 2.0, 2.5, 3.0)},
        "dist_z": {"col": "dist_z", "grid": (1.0, 1.5, 2.0, 2.5, 3.0)},
        "dist_pct": {"col": "dist_pct", "grid": (2.0, 3.0, 4.0, 6.0, 8.0)},
    }
    train_end_ts = pd.Timestamp(train_end)
    out: Dict[str, Dict] = {}

    def _select(theta, col, side) -> Dict:
        """Evaluate rule on the TRAIN window only (selection)."""
        parts = []
        for _, df in frames.items():
            m = _pre_cutoff(df, cutoff) & (df.index < train_end_ts)
            d = df.loc[m, [col, "fwd10_atr"]].copy()
            d["sig"] = np.where(
                d[col] > theta, -1.0, np.where(d[col] < -theta, 1.0, np.nan)
            )
            d = d.dropna(subset=["sig", "fwd10_atr"])
            parts.append(d)
        pooled = pd.concat(parts, ignore_index=True)
        if len(pooled) < 200:
            return {}
        # side filter for asymmetric candidates
        if side == "short":
            pooled = pooled[pooled["sig"] < 0]
        elif side == "long":
            pooled = pooled[pooled["sig"] > 0]
        if len(pooled) < 100:
            return {}
        return {
            "n": len(pooled),
            "gross": float((pooled["sig"] * pooled["fwd10_atr"]).mean()),
        }

    for fam, cfg in families.items():
        col, grid = cfg["col"], cfg["grid"]
        best = None
        for th in grid:
            # Asymmetric long-only / short-only / symmetric candidates.
            for side in ("both", "short", "long"):
                s = _select(th, col, side)
                if s and (best is None or s["gross"] > best["gross"]):
                    best = {**s, "theta": th, "side": side}
        if best is None:
            continue
        th, side = best["theta"], best["side"]
        # OOS evaluation of the frozen rule.
        parts = []
        for _, df in frames.items():
            m = _pre_cutoff(df, cutoff) & (df.index >= train_end_ts)
            d = df.loc[m, [col, "fwd10_atr"]].copy()
            d["regime"] = (
                df.loc[m, "regime"].astype(str) if "regime" in df.columns else "?"
            )
            d["sig"] = np.where(d[col] > th, -1.0, np.where(d[col] < -th, 1.0, np.nan))
            d = d.dropna(subset=["fwd10_atr"])
            parts.append(d)
        pooled = pd.concat(parts, ignore_index=True)
        if side == "short":
            pooled = pooled[pooled["sig"] < 0]
        elif side == "long":
            pooled = pooled[pooled["sig"] > 0]
        else:
            pooled = pooled[pooled["sig"].notna()]
        if len(pooled) < 100:
            continue
        r = (pooled["sig"] * pooled["fwd10_atr"]).astype(float)
        gross = float(r.mean())
        # Flat rate = fraction of ALL OOS bars with |dist| < theta.
        all_parts = []
        for _, df in frames.items():
            m = _pre_cutoff(df, cutoff) & (df.index >= train_end_ts)
            all_parts.append(df.loc[m, col])
        all_dist = pd.concat(all_parts) if all_parts else pd.Series(dtype=float)
        flat = float((all_dist.abs() < th).mean()) if len(all_dist) else None
        out[fam] = {
            "theta": th,
            "side": side,
            "n_train": best["n"],
            "train_gross": round(best["gross"], 4),
            "n_oos": len(pooled),
            "gross_oos": round(gross, 4),
            "net_oos": round(gross - COST_ATR, 4),
            "break_even_atr": round(_breakeven(r.values), 3),
            "flat_rate": round(flat, 3) if flat is not None else None,
            "per_regime": {},
        }
        for reg, d in pooled.groupby("regime"):
            if len(d) < 50:
                continue
            r = (d["sig"] * d["fwd10_atr"]).astype(float)
            out[fam]["per_regime"][str(reg)] = {
                "n": len(d),
                "gross": round(float(r.mean()), 4),
                "net": round(float(r.mean()) - COST_ATR, 4),
            }
    return out


def _breakeven(vals: np.ndarray) -> float:
    """Cost level (ATR) where mean signed return - cost crosses zero."""
    gross = float(np.mean(vals))
    if gross <= 0:
        return 0.0
    for c in COST_GRID:
        if gross - c <= 0:
            return round(float(c), 3)
    return round(float(COST_GRID[-1]), 3)


# ---------------------------------------------------------------------------
# Phase 4: bear-rally short hypothesis (individual definitions)
# ---------------------------------------------------------------------------


def bear_rally_shorts(
    symbols: List[str],
    data_dir: str = "data/raw",
    group: str = "full_fx",
    cutoff: str = CUTOFF,
) -> Dict:
    """Eight non-overlapping 'stretched rally' definitions, evaluated as
    SHORT signals ONLY in Bear Trend / Range regimes at h=10."""
    frames = {}
    for sym in symbols:
        df = _load_base(sym, data_dir, group)
        if df is None or len(df) < 400:
            continue
        frames[sym] = _fwd_atr(df, horizons=(10,))
        close = df["close"].astype(float)
        atr = df["atr_14"].astype(float).replace(0.0, np.nan)
        d = frames[sym]
        d["A_dist_sma200_atr"] = (close - df["sma_200"]) / atr
        d["B_dist_sma50_atr"] = (close - df["sma_50"]) / atr
        d["C_rsi"] = df["rsi_14"]
        d["D_rally_atr"] = close.pct_change(5) / atr
        d["E_pos_streak"] = (close.diff() > 0).astype(int).rolling(5).sum()
        d["F_from_low"] = (close - df["low"].rolling(10).min()) / atr
        d["G_res_prox"] = (df["high"].rolling(60).max().shift(1) - close) / atr
        d["H_vol_adj"] = close.pct_change(5) / (
            df["volatility_20"] if "volatility_20" in df.columns else atr
        )
    defs = {
        "A_dist_sma200": ("A_dist_sma200_atr", 1.5),
        "B_dist_sma50": ("B_dist_sma50_atr", 1.0),
        "C_rsi_extreme": ("C_rsi", 70.0),
        "D_5bar_rally": ("D_rally_atr", 0.8),
        "E_pos_streak5": ("E_pos_streak", 4),
        "F_off_low": ("F_from_low", 2.0),
        "G_resistance": ("G_res_prox", 0.5),
        "H_vol_adj_rally": ("H_vol_adj", 1.0),
    }
    out = {}
    for name, (col, th) in defs.items():
        parts = []
        for _, df in frames.items():
            m = _pre_cutoff(df, cutoff)
            d = df.loc[m, [col, "fwd10_atr"]].copy()
            d["regime"] = (
                df.loc[m, "regime"].astype(str) if "regime" in df.columns else "?"
            )
            d = d.dropna(subset=[col, "fwd10_atr"])
            d = d[d["regime"].isin(["Bear Trend", "Range / Chop"])]
            d = d[d[col] > th]  # stretched rally
            parts.append(d)
        if not parts:
            continue
        pooled = pd.concat(parts, ignore_index=True)
        # Resistance-proximity is a NEARNESS signal (close within th ATR of
        # the recent high), unlike the stretched-rally signals.
        if name == "G_resistance":
            pooled = pooled[pooled[col] < th]
        if len(pooled) < 100:
            continue
        r = -pooled["fwd10_atr"].astype(float)  # short P&L
        out[name] = {
            "col": col,
            "threshold": th,
            "n": len(pooled),
            "gross": round(float(r.mean()), 4),
            "net": round(float(r.mean()) - COST_ATR, 4),
            "break_even": round(_breakeven(r.values), 3),
            "win_rate": round(float((r > 0).mean()), 3),
        }
    return out


# ---------------------------------------------------------------------------
# Phase 5: high-volatility long hypothesis
# ---------------------------------------------------------------------------


def high_vol_longs(
    symbols: List[str],
    data_dir: str = "data/raw",
    group: str = "full_fx",
    cutoff: str = CUTOFF,
) -> Dict:
    """Oversold/excursion definitions under high-volatility conditions,
    evaluated as LONG signals at h=10."""
    frames = {}
    for sym in symbols:
        df = _load_base(sym, data_dir, group)
        if df is None or len(df) < 400:
            continue
        frames[sym] = _fwd_atr(df, horizons=(10,))
        close = df["close"].astype(float)
        atr = df["atr_14"].astype(float).replace(0.0, np.nan)
        d = frames[sym]
        d["A_down_exc_atr"] = (close - close.shift(5)) / atr
        band = (df["bb_upper"] - df["bb_lower"]).replace(0.0, np.nan)
        d["B_bb_pos"] = (close - df["bb_lower"]) / band
        d["C_rsi"] = df["rsi_14"]
        d["D_vol_pctile"] = (
            df["volatility_20"]
            .rolling(250, min_periods=150)
            .apply(lambda x: (x <= x[-1]).mean() if len(x) else np.nan, raw=True)
            if "volatility_20" in df.columns
            else np.nan
        )
        d["E_below_sma"] = (df["sma_50"] - close) / atr
    defs = {
        "A_5bar_drop": ("A_down_exc_atr", -0.8),
        "B_bb_below": ("B_bb_pos", -0.5),
        "C_rsi_oversold": ("C_rsi", 30.0),
        "D_high_vol": ("D_vol_pctile", 0.8),
        "E_below_sma50": ("E_below_sma", 0.8),
    }
    out = {}
    for name, (col, th) in defs.items():
        parts = []
        for _, df in frames.items():
            m = _pre_cutoff(df, cutoff)
            d = df.loc[m, [col, "fwd10_atr"]].copy()
            d = d.dropna(subset=[col, "fwd10_atr"])
            d = d[d[col] < th] if name != "D_high_vol" else d[d[col] > th]
            parts.append(d)
        if not parts:
            continue
        pooled = pd.concat(parts, ignore_index=True)
        if len(pooled) < 100:
            continue
        r = pooled["fwd10_atr"].astype(float)  # long P&L
        out[name] = {
            "col": col,
            "threshold": th,
            "n": len(pooled),
            "gross": round(float(r.mean()), 4),
            "net": round(float(r.mean()) - COST_ATR, 4),
            "break_even": round(_breakeven(r.values), 3),
            "win_rate": round(float((r > 0).mean()), 3),
        }
    return out


# ---------------------------------------------------------------------------
# Phase 6: economic horizon
# ---------------------------------------------------------------------------


def economic_horizon(
    symbols: List[str],
    data_dir: str = "data/raw",
    group: str = "full_fx",
    cutoff: str = CUTOFF,
) -> Dict:
    """Primary reversal signal (SHORT when dist_atr > 2.0, LONG < -2.0)
    gross/net at every horizon — the monetizable horizon where signal
    strength clears the cost hurdle."""
    frames = {}
    for sym in symbols:
        df = _load_base(sym, data_dir, group)
        if df is None or len(df) < 400:
            continue
        frames[sym] = pd.concat(
            [_fwd_atr(df, horizons=HORIZONS), _dist_features(df)], axis=1
        )
    rows = []
    for h in HORIZONS:
        parts = []
        flats = []
        for _, df in frames.items():
            m = _pre_cutoff(df, cutoff)
            d = df.loc[m, ["dist_atr", f"fwd{h}_atr"]].copy()
            d["sig"] = np.where(
                d["dist_atr"] > 2.0, -1.0, np.where(d["dist_atr"] < -2.0, 1.0, np.nan)
            )
            flats.append(d["sig"].isna())
            d = d.dropna(subset=["sig", f"fwd{h}_atr"])
            parts.append(d)
        pooled = pd.concat(parts, ignore_index=True)
        if len(pooled) < 100:
            continue
        flat_all = pd.concat(flats) if flats else pd.Series(dtype=bool)
        r = (pooled["sig"] * pooled[f"fwd{h}_atr"]).astype(float)
        rows.append(
            {
                "horizon": h,
                "n": len(pooled),
                "gross": round(float(r.mean()), 4),
                "net": round(float(r.mean()) - COST_ATR, 4),
                "win_rate": round(float((r > 0).mean()), 3),
                "flat_rate": round(float(flat_all.mean()), 3)
                if len(flat_all)
                else None,
            }
        )
    return rows


# ---------------------------------------------------------------------------
# Phase 8: cost break-even curve
# ---------------------------------------------------------------------------


def cost_breakeven(
    symbols: List[str],
    data_dir: str = "data/raw",
    group: str = "full_fx",
    cutoff: str = CUTOFF,
) -> Dict:
    """Net expectancy of the primary reversal rule across the cost grid."""
    frames = {}
    for sym in symbols:
        df = _load_base(sym, data_dir, group)
        if df is None or len(df) < 400:
            continue
        frames[sym] = pd.concat(
            [_fwd_atr(df, horizons=(10,)), _dist_features(df)], axis=1
        )
    parts = []
    for _, df in frames.items():
        m = _pre_cutoff(df, cutoff)
        d = df.loc[m, ["dist_atr", "fwd10_atr"]].copy()
        d["sig"] = np.where(
            d["dist_atr"] > 2.0, -1.0, np.where(d["dist_atr"] < -2.0, 1.0, np.nan)
        )
        d = d.dropna(subset=["sig", "fwd10_atr"])
        parts.append(d)
    pooled = pd.concat(parts, ignore_index=True)
    r = (pooled["sig"] * pooled["fwd10_atr"]).astype(float)
    gross = float(r.mean())
    curve = []
    for c in COST_GRID:
        curve.append({"cost_atr": c, "net_atr": round(gross - c, 4)})
    return {
        "n": len(pooled),
        "gross_atr": round(gross, 4),
        "break_even_atr": _breakeven(r.values),
        "realistic_cost_atr": COST_ATR,
        "net_at_realistic": round(gross - COST_ATR, 4),
        "curve": curve,
    }


# ---------------------------------------------------------------------------
# Phase 11: monotonicity
# ---------------------------------------------------------------------------


def monotonicity(
    symbols: List[str],
    data_dir: str = "data/raw",
    group: str = "full_fx",
    cutoff: str = CUTOFF,
) -> Dict:
    """Decile table: forward 10-bar return vs distance-from-SMA200 (ATR).
    A monotonically decreasing table supports a monotone expression."""
    frames = {}
    for sym in symbols:
        df = _load_base(sym, data_dir, group)
        if df is None or len(df) < 400:
            continue
        frames[sym] = pd.concat(
            [_fwd_atr(df, horizons=(10,)), _dist_features(df)], axis=1
        )
    parts = []
    for _, df in frames.items():
        m = _pre_cutoff(df, cutoff)
        d = df.loc[m, ["dist_atr", "fwd10_atr"]].dropna()
        parts.append(d)
    pooled = pd.concat(parts, ignore_index=True)
    if pooled["dist_atr"].std() == 0:
        return {}
    pooled["decile"] = pd.qcut(pooled["dist_atr"], 10, labels=False)
    rows = []
    for k in range(10):
        d = pooled[pooled["decile"] == k]
        rows.append(
            {
                "decile": k,
                "dist_atr_mid": round(float(d["dist_atr"].median()), 2),
                "n": len(d),
                "mean_fwd10": round(float(d["fwd10_atr"].mean()), 4),
            }
        )
    return {"rows": rows, "n": len(pooled)}


# ---------------------------------------------------------------------------
# Phase 10: why does the LGBM destroy the signal (MODELS A..E + monotone)
# ---------------------------------------------------------------------------


REVERSAL_FEATURES = [
    "vs_sma200_pct",
    "rsi_14",
    "ma_stack",
    "sma50_gap",
    "sma20_gap",
    "bb_pct_b",
    "slope_20",
    "ret_5",
    "ret_10",
]
TECH_GROUPS = [
    "momentum",
    "trend",
    "volatility",
    "volume",
    "regime",
    "time",
    "interactions",
    "mtf",
    "dip",
]
MONOTONE = ["vs_sma200_pct", "rsi_14", "ma_stack", "sma50_gap", "sma20_gap"]


def _build_pool(symbols: List[str], data_dir: str, group: str, cutoff: str):
    from src.analysis.stage4 import ALL_FEATURES, GROUPS, _ml_frames

    frames = _ml_frames(symbols, data_dir, group)
    parts = []
    for _, df in frames.items():
        df = _fwd_atr(df, horizons=(10,))
        m = _pre_cutoff(df, cutoff)
        d = df.loc[m, ALL_FEATURES + ["fwd10_atr", "symbol"]].copy()
        d["date"] = df.index[m]
        d["y"] = (d["fwd10_atr"] > 0).astype(int)
        parts.append(d)
    data = pd.concat(parts, ignore_index=True)
    data = data.dropna(subset=ALL_FEATURES + ["y"])
    return data, ALL_FEATURES, GROUPS


def model_comparison(
    symbols: List[str],
    data_dir: str = "data/raw",
    group: str = "full_fx",
    cutoff: str = CUTOFF,
    split: str = TRAIN_END,
) -> Dict:
    """MODELS A..E + monotone-constrained LGBM; OOS (split..cutoff)."""
    from lightgbm import LGBMClassifier
    from sklearn.metrics import roc_auc_score

    data, ALL_FEATURES, GROUPS = _build_pool(symbols, data_dir, group, cutoff)
    if len(data) < 5000:
        return {"note": "insufficient data"}
    dates = pd.to_datetime(data["date"])
    tr = dates < pd.Timestamp(split)
    te = ~tr
    y_tr, y_te = data.loc[tr, "y"].values, data.loc[te, "y"].values
    X_tr, X_te = data.loc[tr, ALL_FEATURES], data.loc[te, ALL_FEATURES]

    tech_cols = [f for g in TECH_GROUPS for f in GROUPS.get(g, []) if f in ALL_FEATURES]
    models = {
        "A_single_reversal": REVERSAL_FEATURES[:1],  # vs_sma200_pct
        "B_reversal_only": REVERSAL_FEATURES,
        "C_reversal_regime": REVERSAL_FEATURES
        + ["regime_confidence", "slope_20", "adx"],
        "D_all_technical": tech_cols,
        "E_all_features": ALL_FEATURES,
    }
    out: Dict = {"n_train": int(tr.sum()), "n_test": int(te.sum()), "models": {}}
    for name, cols in models.items():
        present = list(dict.fromkeys(c for c in cols if c in X_tr.columns))
        res = _train_eval(
            X_tr[present],
            y_tr,
            X_te[present],
            y_te,
            fwd_te=data.loc[te, "fwd10_atr"].values,
        )
        out["models"][name] = res
    # Monotone-constrained LGBM on the reversal features.
    mono = [c for c in MONOTONE if c in X_tr.columns]
    other = [c for c in REVERSAL_FEATURES if c not in mono]
    m = LGBMClassifier(
        n_estimators=250,
        learning_rate=0.05,
        num_leaves=31,
        max_depth=7,
        colsample_bytree=0.8,
        subsample=0.8,
        subsample_freq=1,
        min_child_samples=20,
        reg_alpha=0.1,
        reg_lambda=1.0,
        monotone_constraints=[-1] * len(mono) + [0] * len(other),
        random_state=42,
        n_jobs=-1,
        verbose=-1,
    )
    m.fit(X_tr[mono + other], y_tr)
    p = m.predict_proba(X_te[mono + other])[:, 1]
    out["models"]["F_monotone_reversal"] = {
        "auc": round(float(roc_auc_score(y_te, p)), 4),
        "brier": round(float(np.mean((p - y_te) ** 2)), 4),
        "rank_ic": round(float(pd.Series(p).rank().corr(pd.Series(y_te).rank())), 4),
    }
    return out


def _train_eval(X_tr, y_tr, X_te, y_te, fwd_te=None) -> Dict:
    from lightgbm import LGBMClassifier
    from sklearn.metrics import roc_auc_score

    m = LGBMClassifier(
        n_estimators=250,
        learning_rate=0.05,
        num_leaves=31,
        max_depth=7,
        colsample_bytree=0.8,
        subsample=0.8,
        subsample_freq=1,
        min_child_samples=20,
        reg_alpha=0.1,
        reg_lambda=1.0,
        random_state=42,
        n_jobs=-1,
        verbose=-1,
    )
    m.fit(X_tr, y_tr)
    p = m.predict_proba(X_te)[:, 1]
    res = {
        "auc": round(float(roc_auc_score(y_te, p)), 4),
        "brier": round(float(np.mean((p - y_te) ** 2)), 4),
        "rank_ic": round(float(pd.Series(p).rank().corr(pd.Series(y_te).rank())), 4),
    }
    if fwd_te is not None:
        # Long/short expectancy at +/-0.55/0.45 probability thresholds,
        # in R units: fwd ATR / 1.25 minus the 0.05R cost.
        fwd = np.asarray(fwd_te)
        lmask, smask = p >= 0.55, p <= 0.45
        lr = fwd[lmask] / 1.25 - COST_R if lmask.sum() else np.array([0.0])
        sr = -fwd[smask] / 1.25 - COST_R if smask.sum() else np.array([0.0])
        res["n_long"], res["n_short"] = int(lmask.sum()), int(smask.sum())
        res["long_exp_r"] = round(float(lr.mean()), 4)
        res["short_exp_r"] = round(float(sr.mean()), 4)
    return res


# ---------------------------------------------------------------------------
# Phase 12: symbol heterogeneity
# ---------------------------------------------------------------------------


def symbol_heterogeneity(
    symbols: List[str],
    data_dir: str = "data/raw",
    group: str = "full_fx",
    cutoff: str = CUTOFF,
) -> Dict:
    """Per symbol: reversal rank-IC (dist_atr vs fwd10), gross/net of the
    theta=2.0 rule, break-even cost, and n."""
    frames = {}
    for sym in symbols:
        df = _load_base(sym, data_dir, group)
        if df is None or len(df) < 400:
            continue
        frames[sym] = pd.concat(
            [_fwd_atr(df, horizons=(10,)), _dist_features(df)], axis=1
        )
    rows = []
    for sym, df in frames.items():
        m = _pre_cutoff(df, cutoff)
        d = df.loc[m, ["dist_atr", "fwd10_atr"]].dropna()
        if len(d) < 200 or d["dist_atr"].std() == 0 or d["fwd10_atr"].std() == 0:
            continue
        ic = float(d["dist_atr"].rank().corr(d["fwd10_atr"].rank()))
        d["sig"] = np.where(
            d["dist_atr"] > 2.0, -1.0, np.where(d["dist_atr"] < -2.0, 1.0, np.nan)
        )
        d = d.dropna(subset=["sig"])
        if len(d) < 50:
            continue
        r = (d["sig"] * d["fwd10_atr"]).astype(float)
        rows.append(
            {
                "symbol": sym,
                "rank_ic": round(ic, 4),
                "n_trades": len(d),
                "gross": round(float(r.mean()), 4),
                "net": round(float(r.mean()) - COST_ATR, 4),
                "break_even": round(_breakeven(r.values), 3),
            }
        )
    return {"rows": rows}


# ---------------------------------------------------------------------------
# Phase 13: cross-asset timing
# ---------------------------------------------------------------------------


def cross_asset_timing(
    symbols: List[str],
    data_dir: str = "data/raw",
    group: str = "full_fx",
    cutoff: str = CUTOFF,
) -> Dict:
    """Does the risk-on/off context change reversal TIMING? Reversal rank-IC
    of dist_atr split by the sign of the cross-asset risk proxy
    (mean ATR-normalized 5-bar return of AUDJPY/NZDJPY, 1-day lag)."""
    frames = {}
    risk = None
    for sym in symbols:
        df = _load_base(sym, data_dir, group)
        if df is None or len(df) < 400:
            continue
        frames[sym] = pd.concat(
            [_fwd_atr(df, horizons=(10,)), _dist_features(df)], axis=1
        )
        if sym in ("AUDJPY", "NZDJPY") and risk is None:
            r = _risk_proxy(df, 5).shift(1)  # 1-day lag, causal
            risk = r if risk is None else risk.add(r.fillna(0.0), fill_value=0.0)
    if risk is None:
        return {}
    risk = risk / 2.0
    rows = []
    for sym in ["EURUSD", "USDJPY", "EURJPY", "AUDJPY", "XAUUSD", "USDCAD"]:
        if sym not in frames:
            continue
        df = frames[sym]
        m = _pre_cutoff(df, cutoff)
        d = df.loc[m, ["dist_atr", "fwd10_atr"]].copy()
        d["risk"] = risk.reindex(d.index)
        d = d.dropna()
        if len(d) < 300:
            continue
        for tag, mask in (("risk_off", d["risk"] < 0), ("risk_on", d["risk"] >= 0)):
            sub = d[mask]
            if (
                len(sub) < 100
                or sub["dist_atr"].std() == 0
                or sub["fwd10_atr"].std() == 0
            ):
                continue
            rows.append(
                {
                    "symbol": sym,
                    "state": tag,
                    "n": len(sub),
                    "reversal_ic": round(
                        float(sub["dist_atr"].rank().corr(sub["fwd10_atr"].rank())), 4
                    ),
                }
            )
    return {"rows": rows}


# ---------------------------------------------------------------------------
# Phase 14: frozen-rule walk-forward (preliminary)
# ---------------------------------------------------------------------------


def walk_forward(
    symbols: List[str],
    data_dir: str = "data/raw",
    group: str = "full_fx",
    cutoff: str = CUTOFF,
) -> Dict:
    """Frozen simple rule (dist_atr > 2 short / < -2 long, h=10), purged
    3-fold walk-forward with a 20-bar embargo; gross/net per fold."""
    frames = {}
    for sym in symbols:
        df = _load_base(sym, data_dir, group)
        if df is None or len(df) < 400:
            continue
        frames[sym] = pd.concat(
            [_fwd_atr(df, horizons=(10,)), _dist_features(df)], axis=1
        )
    folds = [
        ("2015-01-01", "2022-01-01", "2023-06-30"),
        ("2015-01-01", "2023-06-30", "2024-12-31"),
        ("2015-01-01", "2024-12-31", cutoff),
    ]
    out = []
    for i, (_, te_start, te_end) in enumerate(folds):
        te0 = pd.Timestamp(te_start) + pd.Timedelta(days=20)
        te1 = pd.Timestamp(te_end)
        parts = []
        flats = []
        for _, df in frames.items():
            m = _pre_cutoff(df, cutoff) & (df.index >= te0) & (df.index < te1)
            d = df.loc[m, ["dist_atr", "fwd10_atr"]].copy()
            d["sig"] = np.where(
                d["dist_atr"] > 2.0, -1.0, np.where(d["dist_atr"] < -2.0, 1.0, np.nan)
            )
            flats.append(d["sig"].isna())
            d = d.dropna(subset=["sig", "fwd10_atr"])
            parts.append(d)
        pooled = pd.concat(parts, ignore_index=True) if parts else pd.DataFrame()
        if len(pooled) < 100:
            continue
        flat_all = pd.concat(flats) if flats else pd.Series(dtype=bool)
        r = (pooled["sig"] * pooled["fwd10_atr"]).astype(float)
        out.append(
            {
                "fold": i + 1,
                "window": f"{te0.date()}..{te1.date()}",
                "n": len(pooled),
                "gross": round(float(r.mean()), 4),
                "net": round(float(r.mean()) - COST_ATR, 4),
                "n_long": int((pooled["sig"] > 0).sum()),
                "n_short": int((pooled["sig"] < 0).sum()),
                "flat_rate": round(float(flat_all.mean()), 3)
                if len(flat_all)
                else None,
            }
        )
    return {"folds": out}


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main(argv: Optional[List[str]] = None) -> int:
    import argparse

    parser = argparse.ArgumentParser(description="Stage-5 reversal economic viability")
    parser.add_argument("--group", default="full_fx")
    parser.add_argument("--symbols", default=None)
    parser.add_argument("--cutoff", default=CUTOFF)
    parser.add_argument("--reversal", action="store_true")
    parser.add_argument("--shorts", action="store_true")
    parser.add_argument("--hv", action="store_true")
    parser.add_argument("--horizon", action="store_true")
    parser.add_argument("--cost", action="store_true")
    parser.add_argument("--mono", action="store_true")
    parser.add_argument("--models", action="store_true")
    parser.add_argument("--hetero", action="store_true")
    parser.add_argument("--cross", action="store_true")
    parser.add_argument("--wf", action="store_true")
    parser.add_argument("--all", action="store_true")
    args = parser.parse_args(argv)

    if args.symbols:
        symbols = [s.strip() for s in args.symbols.split(",") if s.strip()]
    else:
        symbols = CORE16
    print(
        f"Reserved untouched test: {args.cutoff}+ (excluded everywhere). "
        f"Threshold selection: train < {TRAIN_END} only."
    )
    results: Dict = {"symbols": symbols, "group": args.group, "cutoff": args.cutoff}

    if args.all or args.reversal:
        print("\n" + "=" * 72)
        print("PHASE 2/3 — PURE REVERSAL BENCHMARK (theta on TRAIN, eval OOS)")
        print("=" * 72)
        rb = reversal_benchmark(symbols, group=args.group, cutoff=args.cutoff)
        print(
            f"{'family':<10}{'theta':>7}{'side':>6}{'n_tr':>7}{'tr_gross':>10}"
            f"{'n_oos':>8}{'gross':>9}{'net':>9}{'be':>7}"
        )
        for fam, v in rb.items():
            print(
                f"{fam:<10}{v['theta']:>7}{v['side']:>6}{v['n_train']:>7,}"
                f"{v['train_gross']:>+10.4f}{v['n_oos']:>8,}{v['gross_oos']:>+9.4f}"
                f"{v['net_oos']:>+9.4f}{v['break_even_atr']:>7}"
            )
        print("  per-regime OOS gross (net):")
        for fam, v in rb.items():
            for reg, r in v.get("per_regime", {}).items():
                print(
                    f"    {fam:<10} {reg:<14} n={r['n']:>6,} gross={r['gross']:+.4f} net={r['net']:+.4f}"
                )
        results["reversal_benchmark"] = rb

    if args.all or args.shorts:
        print("\n" + "=" * 72)
        print("PHASE 4 — BEAR-RALLY SHORT HYPOTHESIS (8 definitions, Bear/Range)")
        print("=" * 72)
        bs = bear_rally_shorts(symbols, group=args.group, cutoff=args.cutoff)
        print(f"{'def':<18}{'n':>7}{'gross':>9}{'net':>9}{'win%':>7}{'be':>7}")
        for k, v in bs.items():
            print(
                f"{k:<18}{v['n']:>7,}{v['gross']:>+9.4f}{v['net']:>+9.4f}"
                f"{100 * v['win_rate']:>6.0f}%{v['break_even']:>7}"
            )
        results["bear_rally_shorts"] = bs

    if args.all or args.hv:
        print("\n" + "=" * 72)
        print("PHASE 5 — HIGH-VOLATILITY LONG HYPOTHESIS (oversold definitions)")
        print("=" * 72)
        hv = high_vol_longs(symbols, group=args.group, cutoff=args.cutoff)
        print(f"{'def':<16}{'n':>7}{'gross':>9}{'net':>9}{'win%':>7}{'be':>7}")
        for k, v in hv.items():
            print(
                f"{k:<16}{v['n']:>7,}{v['gross']:>+9.4f}{v['net']:>+9.4f}"
                f"{100 * v['win_rate']:>6.0f}%{v['break_even']:>7}"
            )
        results["high_vol_longs"] = hv

    if args.all or args.horizon:
        print("\n" + "=" * 72)
        print("PHASE 6 — ECONOMIC HORIZON (dist_atr +/-2 rule)")
        print("=" * 72)
        eh = economic_horizon(symbols, group=args.group, cutoff=args.cutoff)
        print(f"{'h':>4}{'n':>8}{'gross':>9}{'net':>9}{'win%':>7}{'flat%':>8}")
        for r in eh:
            print(
                f"{r['horizon']:>4}{r['n']:>8,}{r['gross']:>+9.4f}{r['net']:>+9.4f}"
                f"{100 * r['win_rate']:>6.0f}%{100 * r['flat_rate']:>7.0f}%"
            )
        results["economic_horizon"] = eh

    if args.all or args.cost:
        print("\n" + "=" * 72)
        print("PHASE 8 — COST BREAK-EVEN (dist_atr +/-2 rule, h=10)")
        print("=" * 72)
        cb = cost_breakeven(symbols, group=args.group, cutoff=args.cutoff)
        print(
            f"gross={cb['gross_atr']:+.4f} n={cb['n']:,} break_even={cb['break_even_atr']} ATR"
            f" net@0.0625={cb['net_at_realistic']:+.4f}"
        )
        for r in cb["curve"]:
            print(f"  cost {r['cost_atr']:.3f}: net {r['net_atr']:+.4f}")
        results["cost_breakeven"] = cb

    if args.all or args.mono:
        print("\n" + "=" * 72)
        print("PHASE 11 — MONOTONICITY (fwd10 vs dist_atr deciles)")
        print("=" * 72)
        mo = monotonicity(symbols, group=args.group, cutoff=args.cutoff)
        print(f"{'decile':>7}{'dist_mid':>10}{'n':>8}{'mean_fwd10':>12}")
        for r in mo.get("rows", []):
            print(
                f"{r['decile']:>7}{r['dist_atr_mid']:>10.2f}{r['n']:>8,}{r['mean_fwd10']:>+12.4f}"
            )
        results["monotonicity"] = mo

    if args.all or args.models:
        print("\n" + "=" * 72)
        print("PHASE 10 — MODEL COMPARISON (why LGBM destroys the signal)")
        print("=" * 72)
        mc = model_comparison(symbols, group=args.group, cutoff=args.cutoff)
        print(f"n_train={mc.get('n_train')} n_test={mc.get('n_test')}")
        print(
            f"{'MODEL':<24}{'AUC':>8}{'Brier':>8}{'rankIC':>8}{'nL':>6}{'nS':>6}{'Lexp':>8}{'Sexp':>8}"
        )
        for name, r in mc.get("models", {}).items():
            print(
                f"{name:<24}{r.get('auc'):>8}{r.get('brier'):>8}{r.get('rank_ic'):>8}"
                f"{r.get('n_long', '-'):>6}{r.get('n_short', '-'):>6}"
                f"{r.get('long_exp_r', '-'):>8}{r.get('short_exp_r', '-'):>8}"
            )
        results["models"] = mc

    if args.all or args.hetero:
        print("\n" + "=" * 72)
        print("PHASE 12 — SYMBOL HETEROGENEITY (reversal per symbol)")
        print("=" * 72)
        sh = symbol_heterogeneity(symbols, group=args.group, cutoff=args.cutoff)
        print(f"{'symbol':<8}{'rankIC':>8}{'n':>7}{'gross':>9}{'net':>9}{'be':>7}")
        for r in sh.get("rows", []):
            print(
                f"{r['symbol']:<8}{r['rank_ic']:>+8.4f}{r['n_trades']:>7,}"
                f"{r['gross']:>+9.4f}{r['net']:>+9.4f}{r['break_even']:>7}"
            )
        results["symbol_heterogeneity"] = sh

    if args.all or args.cross:
        print("\n" + "=" * 72)
        print("PHASE 13 — CROSS-ASSET TIMING (reversal IC by risk-on/off)")
        print("=" * 72)
        ct = cross_asset_timing(symbols, group=args.group, cutoff=args.cutoff)
        print(f"{'symbol':<8}{'state':<10}{'n':>7}{'reversal_ic':>13}")
        for r in ct.get("rows", []):
            print(
                f"{r['symbol']:<8}{r['state']:<10}{r['n']:>7,}{r['reversal_ic']:>+13.4f}"
            )
        results["cross_asset"] = ct

    if args.all or args.wf:
        print("\n" + "=" * 72)
        print("PHASE 14 (preliminary) — FROZEN-RULE WALK-FORWARD (purged, h=10)")
        print("=" * 72)
        wf = walk_forward(symbols, group=args.group, cutoff=args.cutoff)
        print(
            f"{'fold':>5}{'window':<30}{'n':>7}{'gross':>9}{'net':>9}{'nL':>6}{'nS':>6}{'flat%':>7}"
        )
        for r in wf.get("folds", []):
            print(
                f"{r['fold']:>5}{r['window']:<30}{r['n']:>7,}{r['gross']:>+9.4f}"
                f"{r['net']:>+9.4f}{r['n_long']:>6}{r['n_short']:>6}{100 * r['flat_rate']:>6.0f}%"
            )
        results["walk_forward"] = wf

    out_dir = Path("data/validation")
    out_dir.mkdir(parents=True, exist_ok=True)
    with open(out_dir / "stage5_results.json", "w") as fh:
        json.dump(results, fh, indent=2, default=str)
    print(f"\nStage-5 results written to {out_dir / 'stage5_results.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
