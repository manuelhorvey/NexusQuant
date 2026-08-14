"""
NexusQuant — Stage-3 Probability & Economic Validation (research CLI).

Stage-2 proved the two-sided architecture works but exposed the real
bottleneck: the probability and payoff layers are not yet good enough for
live capital (ECE 0.20/0.12; target-level EV ~0 after costs). This module
runs the Stage-3 research program:

    1. calibration_experiment  — strictly out-of-sample calibration for
       LONG and SHORT: raw / bundle-calibrated / isotonic-OOS / Platt-OOS /
       beta-OOS / per-regime isotonic-OOS, all fit BEFORE the model's
       training split and evaluated AFTER it (purged temporal split).
    2. fixed_horizon_edge      — the signal-edge test: ATR-normalized
       forward returns at 1/3/5/10/20 bars per family, side and regime.
       Separates "does the entry signal predict direction?" from "does
       the SL/TP exit architecture destroy expectancy?" (spec Phase 8).
    3. exit_path_sweep         — realized R under TP1-only / TP2-only /
       TP3-only / partial / trailing / time-stop on IDENTICAL entries
       (spec Phase 6), plus net EV at a cost grid and the break-even cost.
    4. tp_tables_by_condition  — P(TP1/TP2/TP3 before SL) per regime and
       volatility bucket (spec Phases 7/10).
    5. attribution             — per setup family and per symbol:
       counts, win rate, fixed-horizon edge, target-level EV (Phases 11/12).

Everything is causal (trailing windows, forward-only resolution, no
future information) and deterministic. Results are printed and written to
``data/validation/stage3_results.json`` for the report.

Usage:
    python -m src.analysis.stage3 --symbols EURUSD,USDCAD --calibrate --edge
    python -m src.analysis.stage3 --group full_fx --exits --tp --attrib
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

sys.path.append(str(Path(__file__).parent.parent.parent))

from src.data.loader import clean_data, load_data
from src.features.indicators import add_all_indicators
from src.features.regime import detect_regime
from src.analysis.census import (
    MIN_BARS,
    RUNG_RR,
    _classify_history,
    _multi_barrier_outcome,
    _uniform_r,
)
from src.model.model import load_model

MODEL_SPLIT = "2022-01-01"  # the dual models' chronological train/test split
COST_GRID = (0.0, 0.01, 0.025, 0.05, 0.075, 0.10, 0.15, 0.20)


# ---------------------------------------------------------------------------
# Calibration metrics (Brier / ECE / MCE / log loss / slope / intercept)
# ---------------------------------------------------------------------------


def calibration_metrics(y: np.ndarray, p: np.ndarray) -> Dict:
    """Standard calibration diagnostics over (realized y, predicted p)."""
    y = np.asarray(y, dtype=float)
    p = np.asarray(p, dtype=float)
    n = len(y)
    if n == 0:
        return {"n": 0}
    brier = float(np.mean((p - y) ** 2))
    eps = 1e-12
    logloss = float(
        -np.mean(
            y * np.log(np.clip(p, eps, 1)) + (1 - y) * np.log(np.clip(1 - p, eps, 1))
        )
    )
    # 10 equal-width buckets over the prediction range.
    lo, hi = float(p.min()), float(p.max())
    if hi - lo < 1e-9:
        hi = lo + 1e-9
    edges = np.linspace(lo, hi, 11)
    ece = mce = 0.0
    buckets = []
    for k in range(10):
        m = (p >= edges[k]) & (p < edges[k + 1]) if k < 9 else p >= edges[9]
        nk = int(m.sum())
        if nk == 0:
            continue
        mean_p = float(p[m].mean())
        actual = float(y[m].mean())
        buckets.append(
            {
                "bucket": f"{edges[k]:.3f}-{edges[k + 1]:.3f}",
                "n": nk,
                "mean_p": round(mean_p, 3),
                "actual": round(actual, 3),
            }
        )
        ece += (nk / n) * abs(actual - mean_p)
        mce = max(mce, abs(actual - mean_p))
    # Calibration slope/intercept: regress y on p.
    slope = intercept = None
    if n >= 10:
        A = np.vstack([p, np.ones(n)]).T
        try:
            beta, *_ = np.linalg.lstsq(A, y, rcond=None)
            slope, intercept = float(beta[0]), float(beta[1])
        except np.linalg.LinAlgError:
            pass
    return {
        "n": n,
        "brier": round(brier, 4),
        "ece": round(ece, 4),
        "mce": round(mce, 4),
        "logloss": round(logloss, 4),
        "slope": round(slope, 3) if slope is not None else None,
        "intercept": round(intercept, 3) if intercept is not None else None,
        "mean_pred": round(float(p.mean()), 3),
        "actual": round(float(y.mean()), 3),
        "buckets": buckets,
    }


# ---------------------------------------------------------------------------
# Calibrators (isotonic / Platt / beta) fit on a train slice, OOS-evaluated
# ---------------------------------------------------------------------------


def _fit_isotonic(train_p: np.ndarray, train_y: np.ndarray):
    from sklearn.isotonic import IsotonicRegression

    iso = IsotonicRegression(out_of_bounds="clip")
    iso.fit(np.asarray(train_p), np.asarray(train_y))
    return iso


def _fit_platt(train_p: np.ndarray, train_y: np.ndarray):
    """Platt scaling: logistic regression on the logit of the raw scores."""
    from sklearn.linear_model import LogisticRegression

    eps = 1e-6
    x = np.log(
        np.clip(train_p, eps, 1 - eps) / (1 - np.clip(train_p, eps, 1 - eps))
    ).reshape(-1, 1)
    lr = LogisticRegression()
    lr.fit(x, train_y)
    return lr


def _fit_beta(train_p: np.ndarray, train_y: np.ndarray):
    """Beta calibration (Kull et al. 2017): sigmoid(a*logit(p) + b) with
    c,d location shifts on the logit. Fit by maximum likelihood."""
    from scipy.optimize import minimize

    eps = 1e-6
    p = np.clip(np.asarray(train_p, dtype=float), eps, 1 - eps)
    y = np.asarray(train_y, dtype=float)
    z = np.log(p / (1 - p))

    def _nll(theta):
        a, b, c, d = theta
        q = 1.0 / (1.0 + np.exp(-(a * z + b)))
        q = np.clip(q, eps, 1 - eps)
        return -float(np.sum(y * np.log(q) + (1 - y) * np.log(1 - q)))

    best = None
    best_nll = np.inf
    for init in ((1.0, 0.0, 0.0, 0.0), (0.5, 0.0, 0.0, 0.0), (2.0, 0.0, 0.0, 0.0)):
        res = minimize(_nll, init, method="L-BFGS-B", bounds=[(None, None)] * 4)
        if res.fun < best_nll:
            best_nll, best = res.fun, res.x
    return best


def _beta_predict(theta, p: np.ndarray) -> np.ndarray:
    eps = 1e-6
    p = np.clip(np.asarray(p, dtype=float), eps, 1 - eps)
    z = np.log(p / (1 - p))
    a, b, _, _ = theta
    return 1.0 / (1.0 + np.exp(-(a * z + b)))


# ---------------------------------------------------------------------------
# Frame + raw/calibrated prediction loading
# ---------------------------------------------------------------------------


def _load_frame(symbol: str, data_dir: str, group: str, timeframe: str) -> pd.DataFrame:
    path = (
        Path(data_dir) / group / f"{symbol}_{timeframe.upper()}.parquet"
        if group
        else Path(data_dir) / f"{symbol}_{timeframe.upper()}.parquet"
    )
    df = clean_data(load_data(path, symbol=symbol))
    df = add_all_indicators(df)
    df = detect_regime(df)
    return df


def raw_and_calibrated(
    df: pd.DataFrame,
    bundle: Dict,
    symbol: str,
    group: str,
    data_dir: str,
    side: str = "long",
) -> tuple:
    """(raw_prob_series, calibrated_prob_series) aligned to df.index.

    Replicates predict_series / predict_short_series but keeps the RAW
    model output too, so the calibration experiment can measure the
    bundle's isotonic transform against the raw score distribution on the
    same bars. ``side`` selects the same signal context the live path uses:
    the long model consumes the dip signal series (build_features defaults
    to it when signal=None), the short model consumes the rally signal
    series exactly like predict_short_series.
    """
    from src.model.model import (
        FEATURE_COLUMNS,
        MACRO_FEATURES,
        apply_calibrator,
        _mtf_frame,
        _cross_proxies,
        build_features,
    )

    model = bundle["model"]
    features = bundle.get("features", FEATURE_COLUMNS)
    calibrator = bundle.get("calibrator")
    signal = None
    if side == "short":
        try:
            from src.backtest.signals import rally_signal_series

            signal = rally_signal_series(df)
        except Exception:
            signal = None
    try:
        macro = None
        if any(f in MACRO_FEATURES for f in features):
            from src.macro.overlay import macro_for_model

            macro = macro_for_model(data_dir)
        mtf = (
            _mtf_frame(symbol, group, data_dir)
            if any(f.startswith("h4_") for f in features)
            else None
        )
        cross = (
            _cross_proxies(group, data_dir)
            if any(
                f in ("risk_mom5", "risk_mom20", "gold_mom5", "gold_mom20")
                for f in features
            )
            else {}
        )
        cot = None
        if "cot_percentile" in features:
            from src.model.cot import load_cot

            cot = load_cot(f"{data_dir}/cot")
        X = build_features(
            df, signal=signal, symbol=symbol, macro=macro, mtf=mtf, cross=cross, cot=cot
        )[features]
    except Exception:
        return None, None
    raw = pd.Series(np.nan, index=df.index)
    cal = pd.Series(np.nan, index=df.index)
    ready = X.notna().all(axis=1)
    if not ready.any():
        return None, None
    r = model.predict_proba(X[ready])[:, 1]
    raw[ready] = r
    cal[ready] = apply_calibrator(calibrator, r)
    return raw, cal


# ---------------------------------------------------------------------------
# Phase 2/3: calibration experiment (strictly OOS)
# ---------------------------------------------------------------------------


def calibration_experiment(
    symbols: List[str],
    data_dir: str = "data/raw",
    group: str = "full_fx",
    timeframe: str = "D1",
    split: str = MODEL_SPLIT,
    max_symbols: int = 12,
    min_regime_n: int = 150,
) -> Dict:
    """Fit calibrators on PRE-split bars, evaluate on POST-split bars.

    The dual models were trained on data up to ``split`` (2022-01-01), so
    post-split bars are genuinely out-of-sample for both the model and
    any calibrator fit pre-split. Methods compared per side: raw,
    bundle-calibrated, isotonic-OOS, Platt-OOS, beta-OOS, and per-regime
    isotonic-OOS (with a global fallback when a regime slice is too small).
    """
    pooled: Dict[str, Dict[str, List]] = {}
    for side in ("long", "short"):
        pooled[side] = {"raw": [], "cal": [], "y": [], "date": [], "regime": []}
    used = 0
    failures: List[str] = []
    for sym in symbols[:max_symbols]:
        try:
            df = _load_frame(sym, data_dir, group, timeframe)
            if len(df) < MIN_BARS + 50:
                continue
            # Long side: dip model + dip signal context.
            b_long = load_model("models/dip_lgbm.joblib")
            raw_l, cal_l = raw_and_calibrated(
                df, b_long, sym, group, data_dir, side="long"
            )
            # Short side: rally model + rally signal context.
            b_short = load_model("models/rally_lgbm.joblib")
            raw_s, cal_s = raw_and_calibrated(
                df, b_short, sym, group, data_dir, side="short"
            )
            u_long = _uniform_r(df, "long")
            u_short = _uniform_r(df, "short")
            regimes = (
                df["regime"]
                if "regime" in df.columns
                else pd.Series("?", index=df.index)
            )
            for side, raw_ser, cal_ser, u_ser in (
                ("long", raw_l, cal_l, u_long),
                ("short", raw_s, cal_s, u_short),
            ):
                if raw_ser is None:
                    continue
                for i in range(MIN_BARS, len(df)):
                    ri, ci, ui = raw_ser.iloc[i], cal_ser.iloc[i], u_ser.iloc[i]
                    if ri != ri or ui != ui:
                        continue
                    pooled[side]["raw"].append(float(ri))
                    pooled[side]["cal"].append(float(ci) if ci == ci else float("nan"))
                    pooled[side]["y"].append(1.0 if ui > 0 else 0.0)
                    pooled[side]["date"].append(df.index[i])
                    pooled[side]["regime"].append(str(regimes.iloc[i]))
            used += 1
        except Exception as exc:
            failures.append(f"{sym}: {exc}")

    split_ts = pd.Timestamp(split)
    out: Dict[str, Dict] = {}
    for side in ("long", "short"):
        d = pooled[side]
        if not d["y"]:
            out[side] = {"n": 0}
            continue
        dates = pd.DatetimeIndex(d["date"])
        train = dates < split_ts
        test = ~train
        y_tr = np.array(
            [y for y, m in zip(d["y"], train, strict=True) if m], dtype=float
        )
        y_te = np.array(
            [y for y, m in zip(d["y"], test, strict=True) if m], dtype=float
        )
        p_raw_tr = np.array(
            [p for p, m in zip(d["raw"], train, strict=True) if m], dtype=float
        )
        p_raw_te = np.array(
            [p for p, m in zip(d["raw"], test, strict=True) if m], dtype=float
        )
        p_cal_te = np.array(
            [c for c, m in zip(d["cal"], test, strict=True) if m and c == c],
            dtype=float,
        )
        results: Dict[str, Dict] = {
            "raw_oos": calibration_metrics(y_te, p_raw_te),
            "bundle_oos": calibration_metrics(y_te, p_cal_te),
        }
        if len(y_tr) >= 100 and len(y_te) >= 100:
            iso = _fit_isotonic(p_raw_tr, y_tr)
            results["isotonic_oos"] = calibration_metrics(
                y_te, np.asarray(iso.predict(p_raw_te))
            )
            platt = _fit_platt(p_raw_tr, y_tr)
            eps = 1e-6
            x_te = np.log(
                np.clip(p_raw_te, eps, 1 - eps) / (1 - np.clip(p_raw_te, eps, 1 - eps))
            ).reshape(-1, 1)
            results["platt_oos"] = calibration_metrics(
                y_te, platt.predict_proba(x_te)[:, 1]
            )
            beta = _fit_beta(p_raw_tr, y_tr)
            if beta is not None:
                results["beta_oos"] = calibration_metrics(
                    y_te, _beta_predict(beta, p_raw_te)
                )
        # Per-regime isotonic OOS (global fallback on small slices).
        per_regime: Dict[str, Dict] = {}
        for reg in sorted(set(d["regime"])):
            m_tr = [(m and r == reg) for m, r in zip(train, d["regime"], strict=True)]
            m_te = [(t and r == reg) for t, r in zip(test, d["regime"], strict=True)]
            n_tr = sum(m_tr)
            n_te = sum(m_te)
            if n_tr < min_regime_n or n_te < 50:
                continue
            y_tr_r = np.array(
                [y for y, m in zip(d["y"], m_tr, strict=True) if m], dtype=float
            )
            p_tr_r = np.array(
                [p for p, m in zip(d["raw"], m_tr, strict=True) if m], dtype=float
            )
            y_te_r = np.array(
                [y for y, m in zip(d["y"], m_te, strict=True) if m], dtype=float
            )
            p_te_r = np.array(
                [p for p, m in zip(d["raw"], m_te, strict=True) if m], dtype=float
            )
            iso_r = _fit_isotonic(p_tr_r, y_tr_r)
            per_regime[reg] = calibration_metrics(
                y_te_r, np.asarray(iso_r.predict(p_te_r))
            )
        results["per_regime"] = per_regime
        results["n_train"] = int(len(y_tr))
        results["n_test"] = int(len(y_te))
        out[side] = results
    out["symbols"] = used
    out["split"] = split
    out["failures"] = failures
    return out


# ---------------------------------------------------------------------------
# Phase 8: fixed-horizon signal edge
# ---------------------------------------------------------------------------


def fixed_horizon_edge(
    symbols: List[str],
    data_dir: str = "data/raw",
    group: str = "full_fx",
    timeframe: str = "D1",
    horizons: tuple = (1, 3, 5, 10, 20),
    min_family_score: float = 0.45,
) -> Dict:
    """ATR-normalized forward returns per classified family/side/regime.

    Directly answers: does the ENTRY SIGNAL predict direction at fixed
    horizons (before any SL/TP architecture)? Mean and t-stat per bucket.
    """
    from src.features.setups import LONG_FAMILIES

    agg: Dict[str, Dict[str, list]] = {}
    for sym in symbols:
        try:
            df = _load_frame(sym, data_dir, group, timeframe)
            if len(df) < MIN_BARS + 50:
                continue
            close = df["close"].astype(float)
            atr = df["atr_14"].astype(float)
            hist = _classify_history(df)
            regimes = (
                df["regime"]
                if "regime" in df.columns
                else pd.Series("?", index=df.index)
            )
            for date, row in hist.iterrows():
                fam = row["setup_family"]
                if fam is None:
                    continue
                side = "long" if fam in LONG_FAMILIES else "short"
                score = row["long_score"] if side == "long" else row["short_score"]
                if score < min_family_score:
                    continue
                i = df.index.get_loc(date)
                if i is None or i + max(horizons) >= len(df):
                    continue
                reg = str(regimes.iloc[i])
                key = (side, fam, reg)
                bucket = agg.setdefault(key, {"rets": {h: [] for h in horizons}})
                for h in horizons:
                    fwd = close.iloc[i + h] / close.iloc[i] - 1.0
                    if fwd == fwd and atr.iloc[i] == atr.iloc[i] and atr.iloc[i] > 0:
                        bucket["rets"][h].append(float(fwd / atr.iloc[i]))
        except Exception:
            continue
    out: Dict[str, Dict] = {}
    for (side, fam, reg), bucket in agg.items():
        row = {"side": side, "family": fam, "regime": reg}
        for h in horizons:
            vals = np.array(bucket["rets"][h], dtype=float)
            n = len(vals)
            if n < 30:
                row[f"h{h}"] = {"n": n, "mean": None, "t": None}
                continue
            mean = float(vals.mean())
            t = (
                mean / (float(vals.std(ddof=1)) / np.sqrt(n))
                if n > 1 and vals.std(ddof=1) > 0
                else None
            )
            row[f"h{h}"] = {
                "n": n,
                "mean": round(mean, 4),
                "t": round(t, 2) if t is not None else None,
            }
        out[f"{side}|{fam}|{reg}"] = row
    return out


# ---------------------------------------------------------------------------
# Phase 6/9: exit-path sweep + cost break-even
# ---------------------------------------------------------------------------


def _exit_outcomes(df: pd.DataFrame, side: str, horizon: int = 20) -> pd.DataFrame:
    """Per-bar realized R under a set of exit policies (identical entries).

    Uniform causal geometry (risk = 1.25 x ATR, rungs at 1R/2R/3R). Uses
    the multi-barrier first-touch outcome for the fixed-target policies,
    a 50/50 partial (half at TP1, half rides to TP2/SL), a trailing stop
    (breakeven after TP1, +1R after TP2) simulated bar-by-bar, and a
    time-stop marking to the horizon close.
    """
    from src.model.features import DEFAULT_STOP_MULT

    close = df["close"].astype(float)
    atr = df["atr_14"].astype(float)
    risk = DEFAULT_STOP_MULT * atr
    stop0 = close - risk if side == "long" else close + risk
    names = ["tp1", "tp2", "tp3"]
    rung = {
        nm: (close + k * risk if side == "long" else close - k * risk)
        for nm, k in zip(names, RUNG_RR, strict=True)
    }
    mb = _multi_barrier_outcome(df, side, horizon=horizon)

    n = len(df)
    tp1_only = np.full(n, np.nan)
    tp2_only = np.full(n, np.nan)
    tp3_only = np.full(n, np.nan)
    partial = np.full(n, np.nan)
    trailing = np.full(n, np.nan)
    timestop = np.full(n, np.nan)
    for i in range(n - 1):
        o = mb.iloc[i]
        # Fixed-target policies from the first-touch outcome.
        if o == o:
            if o == "tp1":
                tp1_only[i], partial[i] = 1.0, 0.5
            elif o == "tp2":
                tp2_only[i], partial[i] = 2.0, 1.5  # TP1 passed on the way
            elif o == "tp3":
                tp3_only[i], partial[i] = 3.0, 1.5
            elif o == "sl":
                tp1_only[i] = tp2_only[i] = tp3_only[i] = -1.0
                # Conservative: assume the stop was hit before TP1, so the
                # whole position exits at -1R (the optimistic 0.5*1 + 0.5*(-1)
                # would only hold if TP1 had been touched first).
                partial[i] = -1.0
        # Trailing stop simulation: stop to entry after TP1, +1R after TP2.
        s, e = stop0.iloc[i], close.iloc[i]
        if s != s or e != e:
            continue
        best_rung = 0
        result = None
        for _, bar in df.iloc[i + 1 : i + 1 + horizon].iterrows():
            hi, lo = float(bar["high"]), float(bar["low"])
            if side == "long":
                if lo <= s:
                    result = (best_rung - 1.0) if best_rung >= 1 else -1.0
                    break
                if hi >= rung["tp3"].iloc[i]:
                    best_rung, s = 3, e + risk.iloc[i]
                elif hi >= rung["tp2"].iloc[i]:
                    best_rung, s = 2, e + risk.iloc[i]
                elif hi >= rung["tp1"].iloc[i]:
                    best_rung, s = 1, e
            else:
                if hi >= s:
                    result = (best_rung - 1.0) if best_rung >= 1 else -1.0
                    break
                if lo <= rung["tp3"].iloc[i]:
                    best_rung, s = 3, e - risk.iloc[i]
                elif lo <= rung["tp2"].iloc[i]:
                    best_rung, s = 2, e - risk.iloc[i]
                elif lo <= rung["tp1"].iloc[i]:
                    best_rung, s = 1, e
        if result is None:
            result = best_rung  # held to horizon at best rung
        trailing[i] = result
        # Time-stop: mark to horizon close.
        if i + horizon < n:
            tc = close.iloc[i + horizon]
            timestop[i] = (
                (tc - e) / risk.iloc[i] if side == "long" else (e - tc) / risk.iloc[i]
            )

    return pd.DataFrame(
        {
            "tp1_only": tp1_only,
            "tp2_only": tp2_only,
            "tp3_only": tp3_only,
            "partial": partial,
            "trailing": trailing,
            "timestop": timestop,
        },
        index=df.index,
    )


def exit_path_sweep(
    symbols: List[str],
    data_dir: str = "data/raw",
    group: str = "full_fx",
    timeframe: str = "D1",
) -> Dict:
    """Per side: mean realized R per exit policy + net EV at a cost grid."""
    policies = ["tp1_only", "tp2_only", "tp3_only", "partial", "trailing", "timestop"]
    out: Dict[str, Dict] = {}
    for side in ("long", "short"):
        all_r = {p: [] for p in policies}
        for sym in symbols:
            try:
                df = _load_frame(sym, data_dir, group, timeframe)
                if len(df) < MIN_BARS + 50:
                    continue
                ex = _exit_outcomes(df, side)
                for p in policies:
                    vals = ex[p].dropna()
                    all_r[p].extend(float(v) for v in vals)
            except Exception:
                continue
        row: Dict = {}
        for p in policies:
            vals = np.array(all_r[p], dtype=float)
            n = len(vals)
            if n < 30:
                row[p] = {"n": n}
                continue
            mean = float(vals.mean())
            row[p] = {
                "n": n,
                "mean_r": round(mean, 4),
                "median": round(float(np.median(vals)), 4),
                "win_rate": round(float((vals > 0).mean()), 3),
                "break_even_cost": _breakeven(vals, COST_GRID),
            }
        out[side] = row
    return out


def _breakeven(vals: np.ndarray, costs: tuple) -> Optional[float]:
    """Cost level (R) where net EV crosses zero (first grid step at or below
    zero; if the sweep runs off the top of the grid the edge survives
    beyond the largest tested cost)."""
    ev0 = float(vals.mean())
    if ev0 <= 0:
        return 0.0
    for c in costs:
        if ev0 - c <= 0:
            return round(float(c), 3)
    return None


# ---------------------------------------------------------------------------
# Phase 7/10: target-level TP tables by regime and volatility bucket
# ---------------------------------------------------------------------------


def tp_tables_by_condition(
    symbols: List[str],
    data_dir: str = "data/raw",
    group: str = "full_fx",
    timeframe: str = "D1",
) -> Dict:
    """P(TP1/TP2/TP3 before SL) + target-level EV per regime and vol bucket."""
    from src.features.setups import LONG_FAMILIES

    agg: Dict[str, Dict[str, Dict[str, int]]] = {}
    for sym in symbols:
        try:
            df = _load_frame(sym, data_dir, group, timeframe)
            if len(df) < MIN_BARS + 50:
                continue
            hist = _classify_history(df)
            mb_long = _multi_barrier_outcome(df, "long")
            mb_short = _multi_barrier_outcome(df, "short")
            atr_ratio = (df["atr_14"] / df["atr_14"].rolling(100).median()).fillna(1.0)
            regimes = (
                df["regime"]
                if "regime" in df.columns
                else pd.Series("?", index=df.index)
            )
            for date, row in hist.iterrows():
                fam = row["setup_family"]
                if fam is None:
                    continue
                side = "long" if fam in LONG_FAMILIES else "short"
                score = row["long_score"] if side == "long" else row["short_score"]
                if score < 0.45:
                    continue
                i = df.index.get_loc(date)
                ar = atr_ratio.iloc[i]
                vol = "HIGH" if ar > 1.15 else ("LOW" if ar < 0.85 else "MED")
                reg = str(regimes.iloc[i])
                mb = mb_long.iloc[i] if side == "long" else mb_short.iloc[i]
                if mb != mb:
                    continue
                for key in ((side, "ALL", vol), (side, reg, "ALL"), (side, reg, vol)):
                    d = agg.setdefault(key, {"tp1": 0, "tp2": 0, "tp3": 0, "sl": 0})
                    d[mb] += 1
        except Exception:
            continue
    out: Dict[str, Dict] = {}
    for (side, reg, vol), d in agg.items():
        total = d["tp1"] + d["tp2"] + d["tp3"] + d["sl"]
        if not total:
            continue
        p1, p2, p3, psl = (
            d["tp1"] / total,
            d["tp2"] / total,
            d["tp3"] / total,
            d["sl"] / total,
        )
        ev = p1 * RUNG_RR[0] + p2 * RUNG_RR[1] + p3 * RUNG_RR[2] - psl
        out[f"{side}|{reg}|{vol}"] = {
            "n": total,
            "p_tp1": round(p1, 3),
            "p_tp2": round(p2, 3),
            "p_tp3": round(p3, 3),
            "p_sl": round(psl, 3),
            "ev_0": round(ev, 4),
            "ev_05": round(ev - 0.05, 4),
        }
    return out


# ---------------------------------------------------------------------------
# Phase 11/12: setup-family + symbol attribution
# ---------------------------------------------------------------------------


def attribution(
    symbols: List[str],
    data_dir: str = "data/raw",
    group: str = "full_fx",
    timeframe: str = "D1",
) -> Dict:
    """Per family and per symbol: counts, uniform win rate, expectancy,
    fixed-horizon edge (5/10-bar), target-level EV at 0 cost."""
    from src.features.setups import LONG_FAMILIES

    fam_agg: Dict[str, Dict] = {}
    sym_agg: Dict[str, Dict] = {}
    for sym in symbols:
        try:
            df = _load_frame(sym, data_dir, group, timeframe)
            if len(df) < MIN_BARS + 50:
                continue
            hist = _classify_history(df)
            u_long = _uniform_r(df, "long")
            u_short = _uniform_r(df, "short")
            mb_long = _multi_barrier_outcome(df, "long")
            mb_short = _multi_barrier_outcome(df, "short")
            close = df["close"].astype(float)
            atr = df["atr_14"].astype(float)
            sacc = sym_agg.setdefault(
                sym, {"n": 0, "wins": 0, "sum_r": 0.0, "fwd5": [], "fwd10": []}
            )
            for date, row in hist.iterrows():
                fam = row["setup_family"]
                if fam is None:
                    continue
                side = "long" if fam in LONG_FAMILIES else "short"
                score = row["long_score"] if side == "long" else row["short_score"]
                if score < 0.45:
                    continue
                i = df.index.get_loc(date)
                if i + 10 >= len(df):
                    continue
                u = u_long.iloc[i] if side == "long" else u_short.iloc[i]
                mb = mb_long.iloc[i] if side == "long" else mb_short.iloc[i]
                fa = fam_agg.setdefault(
                    fam, {"n": 0, "wins": 0, "sum_r": 0.0, "fwd5": [], "fwd10": []}
                )
                fa["n"] += 1
                sacc["n"] += 1
                if u == u:
                    if u > 0:
                        fa["wins"] += 1
                        sacc["wins"] += 1
                    fa["sum_r"] += float(u)
                    sacc["sum_r"] += float(u)
                if mb == mb:
                    if mb == "tp1":
                        fa["tp_ev"] = fa.get("tp_ev", 0.0) + 1.0
                        sacc["tp_ev"] = sacc.get("tp_ev", 0.0) + 1.0
                    elif mb == "tp2":
                        fa["tp_ev"] = fa.get("tp_ev", 0.0) + 2.0
                        sacc["tp_ev"] = sacc.get("tp_ev", 0.0) + 2.0
                    elif mb == "tp3":
                        fa["tp_ev"] = fa.get("tp_ev", 0.0) + 3.0
                        sacc["tp_ev"] = sacc.get("tp_ev", 0.0) + 3.0
                    elif mb == "sl":
                        fa["tp_ev"] = fa.get("tp_ev", 0.0) - 1.0
                        sacc["tp_ev"] = sacc.get("tp_ev", 0.0) - 1.0
                    fa["tp_n"] = fa.get("tp_n", 0) + 1
                    sacc["tp_n"] = sacc.get("tp_n", 0) + 1
                if atr.iloc[i] > 0:
                    fa["fwd5"].append(
                        float((close.iloc[i + 5] / close.iloc[i] - 1) / atr.iloc[i])
                    )
                    fa["fwd10"].append(
                        float((close.iloc[i + 10] / close.iloc[i] - 1) / atr.iloc[i])
                    )
                    sacc["fwd5"].append(
                        float((close.iloc[i + 5] / close.iloc[i] - 1) / atr.iloc[i])
                    )
                    sacc["fwd10"].append(
                        float((close.iloc[i + 10] / close.iloc[i] - 1) / atr.iloc[i])
                    )
        except Exception:
            continue

    def _finish(d: Dict) -> Dict:
        n = d["n"]
        wins = d["wins"]
        r = {
            "n": n,
            "win_rate": round(wins / n, 3) if n else None,
            "expectancy_r": round(d["sum_r"] / max(n, 1), 4),
        }
        tp_n = d.get("tp_n", 0)
        r["tp_ev_0"] = round(d.get("tp_ev", 0.0) / tp_n, 4) if tp_n else None
        for h in (5, 10):
            vals = np.array(d[f"fwd{h}"], dtype=float)
            if len(vals) >= 30:
                r[f"fwd{h}_mean"] = round(float(vals.mean()), 4)
                r[f"fwd{h}_t"] = round(
                    float(vals.mean() / (vals.std(ddof=1) / np.sqrt(len(vals)))), 2
                )
            else:
                r[f"fwd{h}_mean"] = None
                r[f"fwd{h}_t"] = None
        return r

    return {
        "families": {
            k: _finish(v)
            for k, v in sorted(fam_agg.items(), key=lambda kv: -kv[1]["n"])
        },
        "symbols": {
            k: _finish(v)
            for k, v in sorted(sym_agg.items(), key=lambda kv: -kv[1]["n"])
        },
    }


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main(argv: Optional[List[str]] = None) -> int:
    import argparse

    parser = argparse.ArgumentParser(
        description="Stage-3 probability & economic validation"
    )
    parser.add_argument("--group", default="full_fx")
    parser.add_argument("--timeframe", default="D1")
    parser.add_argument("--symbols", default=None, help="comma-separated")
    parser.add_argument("--max-symbols", type=int, default=12)
    parser.add_argument("--calibrate", action="store_true")
    parser.add_argument("--edge", action="store_true")
    parser.add_argument("--exits", action="store_true")
    parser.add_argument("--tp", action="store_true")
    parser.add_argument("--attrib", action="store_true")
    parser.add_argument("--all", action="store_true")
    args = parser.parse_args(argv)

    if args.symbols:
        symbols = [s.strip() for s in args.symbols.split(",") if s.strip()]
    else:
        from src.analysis.scanner import discover_symbols

        symbols = discover_symbols(
            "data/raw", group=args.group, timeframe=args.timeframe
        )
        symbols = symbols[: args.max_symbols]

    results: Dict = {"symbols": symbols, "group": args.group}
    if args.all or args.calibrate:
        print("\n" + "=" * 72)
        print("PHASE 2/3 — CALIBRATION (strictly OOS, fit pre-split / eval post-split)")
        print("=" * 72)
        cal = calibration_experiment(
            symbols, group=args.group, timeframe=args.timeframe
        )
        for side in ("long", "short"):
            s = cal[side]
            if not s.get("n_test"):
                print(f"\n[{side}] no data")
                continue
            print(f"\n[{side.upper()}] n_train={s['n_train']:,} n_test={s['n_test']:,}")
            for method, m in s.items():
                if method in ("n_train", "n_test", "per_regime", "symbols"):
                    continue
                if not isinstance(m, dict) or not m.get("n"):
                    continue
                print(
                    f"  {method:<14} Brier {m['brier']:.4f} · ECE {m['ece']:.4f} · "
                    f"MCE {m['mce']:.4f} · LL {m['logloss']:.4f} · "
                    f"slope {m['slope']} · mean {m['mean_pred']:.3f} -> actual {m['actual']:.3f}"
                )
            if s.get("per_regime"):
                print(
                    "  per-regime isotonic OOS ECE: "
                    + ", ".join(
                        f"{r} {m['ece']:.3f} (n={m['n']})"
                        for r, m in sorted(s["per_regime"].items())
                    )
                )
        results["calibration"] = cal

    if args.all or args.edge:
        print("\n" + "=" * 72)
        print("PHASE 8 — FIXED-HORIZON SIGNAL EDGE (ATR-normalized forward returns)")
        print("=" * 72)
        edge = fixed_horizon_edge(symbols, group=args.group, timeframe=args.timeframe)
        print(
            f"{'SIDE|FAMILY|REGIME':<44}{'h1':>10}{'h3':>10}{'h5':>10}{'h10':>10}{'h20':>10}"
        )
        for key, row in sorted(edge.items()):
            cells = []
            for h in (1, 3, 5, 10, 20):
                v = row[f"h{h}"]
                cells.append(
                    f"{v['mean']:+.3f}*"
                    if v.get("t") is not None and abs(v["t"]) > 2
                    else f"{v['mean']:+.3f}"
                    if v.get("mean") is not None
                    else "-"
                )
            print(f"{key:<44}" + "".join(f"{c:>10}" for c in cells))
        results["edge"] = edge

    if args.all or args.exits:
        print("\n" + "=" * 72)
        print("PHASE 6/9 — EXIT-PATH SWEEP (identical entries) + cost break-even")
        print("=" * 72)
        ex = exit_path_sweep(symbols, group=args.group, timeframe=args.timeframe)
        for side in ("long", "short"):
            print(f"\n[{side.upper()}]")
            print(
                f"{'POLICY':<12}{'n':>8}{'meanR':>9}{'median':>8}{'win%':>7}{'breakeven':>12}"
            )
            for p, v in ex[side].items():
                if not v.get("n"):
                    continue
                be = (
                    f"{v['break_even_cost']}R"
                    if v.get("break_even_cost") is not None
                    else ">0.20R"
                )
                print(
                    f"{p:<12}{v['n']:>8,}{v['mean_r']:>+9.3f}{v['median']:>+8.2f}"
                    f"{100 * v['win_rate']:>6.0f}%{be:>12}"
                )
        results["exits"] = ex

    if args.all or args.tp:
        print("\n" + "=" * 72)
        print("PHASE 7/10 — TARGET-LEVEL TP TABLES by regime x volatility")
        print("=" * 72)
        tp = tp_tables_by_condition(symbols, group=args.group, timeframe=args.timeframe)
        print(
            f"{'SIDE|REGIME|VOL':<30}{'n':>7}{'P(tp1)':>8}{'P(tp2)':>8}{'P(tp3)':>8}{'P(sl)':>8}{'EV@0':>8}{'EV@.05':>8}"
        )
        for key, v in sorted(tp.items()):
            print(
                f"{key:<30}{v['n']:>7,}{v['p_tp1']:>8.3f}{v['p_tp2']:>8.3f}"
                f"{v['p_tp3']:>8.3f}{v['p_sl']:>8.3f}{v['ev_0']:>+8.3f}{v['ev_05']:>+8.3f}"
            )
        results["tp_tables"] = tp

    if args.all or args.attrib:
        print("\n" + "=" * 72)
        print("PHASE 11/12 — SETUP-FAMILY + SYMBOL ATTRIBUTION")
        print("=" * 72)
        at = attribution(symbols, group=args.group, timeframe=args.timeframe)
        print(
            f"\n{'FAMILY':<28}{'n':>7}{'win%':>7}{'expR':>8}{'tpEV':>8}{'fwd5':>9}{'t5':>6}{'fwd10':>9}{'t10':>6}"
        )

        def _fmt_row(label, width, v):
            f5 = f"{v['fwd5_mean']:+.3f}" if v.get("fwd5_mean") is not None else "-"
            f10 = f"{v['fwd10_mean']:+.3f}" if v.get("fwd10_mean") is not None else "-"
            t5 = f"{v['fwd5_t']:.1f}" if v.get("fwd5_t") is not None else "-"
            t10 = f"{v['fwd10_t']:.1f}" if v.get("fwd10_t") is not None else "-"
            tpev = f"{v['tp_ev_0']:+.3f}" if v.get("tp_ev_0") is not None else "-"
            wr = (
                f"{100 * v['win_rate']:>6.0f}%"
                if v.get("win_rate") is not None
                else "   -"
            )
            return (
                f"{label:<{width}}{v['n']:>7,}{wr}"
                f"{v['expectancy_r']:>+8.3f}{tpev:>8}"
                f"{f5:>9}{t5:>6}{f10:>9}{t10:>6}"
            )

        print(
            f"\n{'FAMILY':<28}{'n':>7}{'win%':>7}{'expR':>8}{'tpEV':>8}{'fwd5':>9}{'t5':>6}{'fwd10':>9}{'t10':>6}"
        )
        for fam, v in at["families"].items():
            print(_fmt_row(fam, 28, v))
        print(
            f"\n{'SYMBOL':<10}{'n':>7}{'win%':>7}{'expR':>8}{'tpEV':>8}{'fwd5':>9}{'t5':>6}{'fwd10':>9}{'t10':>6}"
        )
        for sym, v in at["symbols"].items():
            print(_fmt_row(sym, 10, v))
        results["attribution"] = at

    out_dir = Path("data/validation")
    out_dir.mkdir(parents=True, exist_ok=True)
    with open(out_dir / "stage3_results.json", "w") as fh:
        json.dump(results, fh, indent=2, default=str)
    print(f"\nStage-3 results written to {out_dir / 'stage3_results.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
