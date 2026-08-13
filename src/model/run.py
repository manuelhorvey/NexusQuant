"""
NexusQuant - Ensemble signal model CLI.

Train / evaluate / apply the LightGBM bullish-probability model:

    python -m src.model.run --group full_fx                      # train + OOS eval (meta label, CV)
    python -m src.model.run --group full_fx --label 1r           # 1R triple-barrier label instead
    python -m src.model.run --group full_fx --label 1r --drop-censored
    python -m src.model.run --group full_fx --cv 5               # purged walk-forward CV
    python -m src.model.run --group full_fx --search             # hyperparameter search first
    python -m src.model.run --group full_fx --stack              # rule-score stacking comparison
    python -m src.model.run --group full_fx --per-group          # separate gold/jpy/majors models
    python -m src.model.run --group full_fx --weight-vol         # vol-aware sample weighting
    python -m src.model.run --group full_fx --fetch-cot          # refresh CFTC COT cache first
    python -m src.model.run --predict XAUUSD --group full_fx     # live probability

Validation is chronological with a label-horizon embargo around every
boundary (single split or purged walk-forward) and **early stopping on a
chronological validation tail** - the single biggest guard against the
fixed-iteration overfit a raw 400-tree model had. Features and labels are
strictly causal; macro context is aligned with a 1-day lag; H4 multi-
timeframe features use only same-day H4 closes; cross-asset and COT features
use a 1-day lag. Probabilities are calibrated (isotonic) before they leave
the model, so they are usable for fractional Kelly sizing.

The default label is now **meta** - the *actual engine outcome* on confirmed
dips only (entry-zone-low fill, swing-low stop, resistance target), which is
the question the model answers in production. Use ``--label 1r`` for the
generic asymmetric-triple-barrier 1R label (all bars).
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

sys.path.append(str(Path(__file__).parent.parent.parent))

from src.data.loader import clean_data, load_data
from src.features.indicators import add_all_indicators
from src.features.regime import detect_regime
from src.analysis.scanner import _data_path, discover_symbols
from src.model.cot import load_cot
from src.model.features import (
    DEFAULT_HORIZON,
    DEFAULT_META_HORIZON,
    build_dataset,
)
from src.model.model import (
    DEFAULT_MODEL_PATH,
    DEFAULT_SHORT_MODEL_PATH,
    _cross_proxies,
    _mtf_frame,
    decile_report,
    dip_filter_gate,
    evaluate,
    feature_importance,
    fit_calibrator,
    load_model,
    predict_series,
    save_model,
    top_decile_lift,
    train_model,
)

# ---------------------------------------------------------------------------
# Chronological validation-slice helper (early stopping)
# ---------------------------------------------------------------------------


def _chrono_val_split(X, y, w, val_frac: float = 0.2, min_val: int = 200):
    """Split a training set into train + chronological tail validation slice."""
    n = len(X)
    if n < min_val * 3:
        return X, y, w, None, None, None
    n_val = min(int(n * val_frac), 8000)
    return (
        X.iloc[:-n_val],
        y.iloc[:-n_val],
        w.iloc[:-n_val],
        X.iloc[-n_val:],
        y.iloc[-n_val:],
        w.iloc[-n_val:],
    )


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------


def prepare_frame(
    symbol: str,
    group: Optional[str],
    timeframe: str,
    data_dir: str = "data/raw",
    start: Optional[str] = None,
) -> pd.DataFrame:
    """Load, clean, indicators + regime (local files only)."""
    path = _data_path(symbol, data_dir, group, timeframe)
    df = load_data(path, symbol=symbol)
    df = clean_data(df)
    if start:
        df = df[df.index >= pd.Timestamp(start)]
    df = add_all_indicators(df)
    df = detect_regime(df)
    return df


def collect_dataset(
    symbols: List[str],
    group: Optional[str],
    timeframe: str,
    data_dir: str,
    start: Optional[str],
    horizon: int,
    macro: Optional[pd.DataFrame],
    meta: bool,
    drop_censored: bool,
    weight_dip: float,
    label_kwargs: Optional[dict] = None,
    use_mtf: bool = True,
    use_cross: bool = True,
    use_cot: bool = True,
    weight_vol: bool = False,
    side: str = "long",
) -> dict:
    """Build the pooled training matrix across symbols."""
    # Context loaders are shared with the serve path (predict_series), so
    # training and live inference read the *same* MTF / cross-asset code.
    cross = _cross_proxies(group, data_dir) if use_cross else None
    cot = load_cot(f"{data_dir}/cot") if use_cot else None
    X_parts, y_parts = [], []
    times, syms, confirmed, censored = [], [], [], []
    weights = []
    for sym in symbols:
        try:
            df = prepare_frame(sym, group, timeframe, data_dir, start)
        except Exception as exc:
            print(f"  [!] {sym}: {exc}", file=sys.stderr)
            continue
        if len(df) < 250:
            continue
        mtf = _mtf_frame(sym, group, data_dir) if use_mtf else None
        ds = build_dataset(
            df,
            horizon=horizon,
            symbol=sym,
            macro=macro,
            meta=meta,
            drop_censored=drop_censored,
            label_kwargs=label_kwargs,
            mtf=mtf,
            cross=cross,
            cot=cot,
            side=side,
        )
        if ds["X"].empty:
            continue
        w = ds["weight"]
        if weight_dip != 1.0:
            w = w * np.where(ds["confirmed"], weight_dip, 1.0)
        if weight_vol and "volatility_20" in ds["X"].columns:
            vol = pd.to_numeric(ds["X"]["volatility_20"], errors="coerce").replace(
                0.0, np.nan
            )
            med = vol.median()
            if med and med > 0:
                # Down-weight high-vol bars (noise-heavy), up-weight calm ones.
                w = w * np.clip(med / vol, 0.25, 4.0).fillna(1.0)
        X_parts.append(ds["X"])
        y_parts.append(ds["y"])
        weights.append(w)
        times.append(pd.Series(ds["time"], index=ds["time"]))
        syms.append(pd.Series(sym, index=ds["time"]))
        confirmed.append(pd.Series(ds["confirmed"], index=ds["time"]))
        censored.append(pd.Series(ds["censored"], index=ds["time"]))

    if not X_parts:
        raise RuntimeError("No usable data collected.")

    X = pd.concat(X_parts)
    # Pooled concat promotes the categorical symbol column back to object;
    # keep it categorical so LightGBM can treat it as such at predict time.
    if "symbol" in X.columns:
        X["symbol"] = X["symbol"].astype("category")
    return {
        "X": X,
        "y": pd.concat(y_parts),
        "weight": pd.concat(weights),
        "time": pd.concat(times),
        "symbol": pd.concat(syms),
        "confirmed": pd.concat(confirmed),
        "censored": pd.concat(censored),
    }


def split_chronological(ds: dict, split: str, horizon: int):
    """Train/test split by date with a one-horizon embargo at the boundary."""
    cut = pd.Timestamp(split)
    train_idx = ds["time"] < cut
    # Drop the horizon bars before the split from training (embargo).
    embargo_idx = ds["time"] < (cut - pd.Timedelta(days=horizon))
    train_keep = train_idx & embargo_idx
    test_keep = ~train_idx
    return {
        "X_train": ds["X"][train_keep],
        "y_train": ds["y"][train_keep],
        "weight_train": ds["weight"][train_keep],
        "X_test": ds["X"][test_keep],
        "y_test": ds["y"][test_keep],
        "weight_test": ds["weight"][test_keep],
        "time_train": ds["time"][train_keep],
        "time_test": ds["time"][test_keep],
        "symbol_test": ds["symbol"][test_keep],
        "confirmed_test": ds["confirmed"][test_keep],
        "censored_test": ds["censored"][test_keep],
    }


def walk_forward(
    ds: dict,
    horizon: int,
    n_folds: int,
    min_train: int = 500,
    min_test: int = 100,
    model_kwargs: Optional[dict] = None,
) -> dict:
    """
    Purged walk-forward CV: n_folds chronological folds, each with a
    label-horizon embargo between train and test and **early stopping on a
    chronological tail of the training fold**. Returns pooled OOS
    predictions (the honest estimate) plus per-fold metrics and the OOS row
    positions (``oos_idx``, for stacking against the dataset rows).
    """
    time = ds["time"]
    n = len(time)
    rank = time.rank(method="first").astype(int)
    edges = [int(i * n / n_folds) for i in range(n_folds + 1)]

    oos_y, oos_p, oos_conf, oos_idx = [], [], [], []
    fold_metrics = []
    for f in range(n_folds):
        test_keep = (rank > edges[f]) & (rank <= edges[f + 1])
        test_start = time[test_keep].min()
        train_keep = (rank <= edges[f]) & (
            time < (test_start - pd.Timedelta(days=horizon))
        )
        n_tr = int(train_keep.sum())
        n_te = int(test_keep.sum())
        if n_tr < min_train or n_te < min_test:
            continue
        X_tr, y_tr, w_tr, X_val, y_val, w_val = _chrono_val_split(
            ds["X"][train_keep], ds["y"][train_keep], ds["weight"][train_keep]
        )
        model = train_model(
            X_tr,
            y_tr,
            sample_weight=w_tr,
            X_val=X_val,
            y_val=y_val,
            sample_weight_val=w_val,
            model_kwargs=model_kwargs,
        )
        p = model.predict_proba(ds["X"][test_keep])[:, 1]
        oos_y.append(ds["y"][test_keep].values)
        oos_p.append(p)
        oos_conf.append(ds["confirmed"][test_keep])
        oos_idx.append(np.where(test_keep.values)[0])
        fold_metrics.append(
            {
                "fold": f + 1,
                "train_start": str(ds["time"][train_keep].min().date()),
                "test_start": str(test_start.date()),
                "test_end": str(ds["time"][test_keep].max().date()),
                **evaluate(ds["y"][test_keep].values, p),
            }
        )
        print(
            f"  fold {f + 1}/{n_folds}: train {n_tr:,} -> test {n_te:,} "
            f"(AUC {fold_metrics[-1]['auc']:.3f})"
        )

    if not oos_y:
        raise RuntimeError("Walk-forward produced no usable folds.")

    y_all = np.concatenate(oos_y)
    p_all = np.concatenate(oos_p)
    conf_all = np.concatenate([np.asarray(c, dtype=bool) for c in oos_conf])
    idx_all = np.concatenate(oos_idx) if oos_idx else np.array([], dtype=int)
    return {
        "metrics": evaluate(y_all, p_all),
        "fold_metrics": fold_metrics,
        "oos_y": y_all,
        "oos_p": p_all,
        "oos_confirmed": conf_all,
        "oos_idx": idx_all,
    }


# ---------------------------------------------------------------------------
# Hyperparameter search + stacking
# ---------------------------------------------------------------------------

# Conservative base config (mirrors make_model) with a small grid over the
# dimensions that matter most for FX/gold daily bars.
SEARCH_GRID = [
    {
        "num_leaves": 15,
        "learning_rate": 0.04,
        "min_child_samples": 20,
        "colsample_bytree": 0.8,
    },
    {
        "num_leaves": 31,
        "learning_rate": 0.04,
        "min_child_samples": 20,
        "colsample_bytree": 0.8,
    },
    {
        "num_leaves": 31,
        "learning_rate": 0.02,
        "min_child_samples": 40,
        "colsample_bytree": 0.7,
    },
    {
        "num_leaves": 63,
        "learning_rate": 0.03,
        "min_child_samples": 30,
        "colsample_bytree": 0.6,
    },
    {
        "num_leaves": 127,
        "learning_rate": 0.02,
        "min_child_samples": 50,
        "colsample_bytree": 0.5,
    },
    {
        "num_leaves": 15,
        "learning_rate": 0.05,
        "min_child_samples": 10,
        "colsample_bytree": 0.9,
    },
]


def search_hyperparams(
    ds: dict, horizon: int, max_rows: int = 60000, seed: int = 42
) -> dict:
    """
    Lightweight chronological hyperparameter search.

    Each config is evaluated on a single chronological split (75/25, embargo
    applied) of the most recent ``max_rows`` rows - honest, fast, and
    overfit-resistant because the split is never shuffled. Returns the best
    config dict (used to build the production model).
    """
    X, y, w = ds["X"], ds["y"], ds["weight"]
    if len(X) > max_rows:
        X = X.iloc[-max_rows:]
        y = y.iloc[-max_rows:]
        w = w.iloc[-max_rows:]
    n = len(X)
    cut = int(0.75 * n)
    tr, te = slice(None, cut), slice(cut, None)
    X_tr, y_tr, w_tr, X_te, y_te = (
        X.iloc[tr],
        y.iloc[tr],
        w.iloc[tr],
        X.iloc[te],
        y.iloc[te],
    )
    if len(X_tr) < 1000 or len(X_te) < 300:
        return {}

    from lightgbm import LGBMClassifier

    categorical = [c for c in X.columns if c == "symbol"]

    def _auc(cfg: dict) -> Tuple[float, object]:
        model = LGBMClassifier(
            n_estimators=400,
            max_depth=7,
            subsample=0.8,
            subsample_freq=1,
            reg_alpha=0.1,
            reg_lambda=1.0,
            random_state=seed,
            n_jobs=-1,
            verbose=-1,
            **cfg,
        )
        fit_kwargs = {"sample_weight": w_tr}
        if categorical:
            fit_kwargs["categorical_feature"] = categorical
        model.fit(X_tr, y_tr, **fit_kwargs)
        p = model.predict_proba(X_te)[:, 1]
        return evaluate(y_te.values, p)["auc"], model

    best_auc, best_cfg = -1.0, {}
    for cfg in SEARCH_GRID:
        try:
            auc, _ = _auc(cfg)
        except Exception as exc:
            print(f"  [!] config {cfg} failed: {exc}")
            continue
        print(f"  {cfg} -> OOS AUC {auc:.3f}")
        if auc > best_auc:
            best_auc, best_cfg = auc, cfg
    print(f"  best config {best_cfg} (AUC {best_auc:.3f})")
    return best_cfg


def stack_oos(
    ds: dict,
    oos_idx: np.ndarray,
    oos_p: np.ndarray,
    stack_cols: Tuple[str, ...] = ("dip_score", "macro_bias", "adx"),
) -> Optional[dict]:
    """
    Rule-score stacking comparison on the OOS rows.

    Fits a logistic regression on [LGBM prob, dip_score, macro_bias, adx]
    over the first 60% of the OOS rows (chronological) and compares stacked
    vs raw-LGBM AUC on the last 40%. Returns None when there are too few
    OOS rows to split honestly.
    """
    from sklearn.linear_model import LogisticRegression

    if oos_idx is None or len(oos_idx) < 600:
        return None
    X = ds["X"].iloc[oos_idx]
    frame = pd.DataFrame({"lgb": oos_p}, index=X.index)
    for c in stack_cols:
        if c in X.columns:
            frame[c] = X[c].values
    Xs = frame.fillna(0.0).values
    ys = ds["y"].iloc[oos_idx].values
    n = len(ys)
    cut = int(0.6 * n)
    lr = LogisticRegression(max_iter=2000)
    lr.fit(Xs[:cut], ys[:cut])
    p_stack = lr.predict_proba(Xs[cut:])[:, 1]
    m_base = evaluate(ys[cut:], oos_p[cut:])
    m_stack = evaluate(ys[cut:], p_stack)
    return {
        "n_eval": n - cut,
        "lgb_auc": m_base["auc"],
        "stacked_auc": m_stack["auc"],
        "delta": round(m_stack["auc"] - m_base["auc"], 4),
    }


# ---------------------------------------------------------------------------
# Training / evaluation driver
# ---------------------------------------------------------------------------


def train_and_evaluate(
    symbols,
    group,
    timeframe,
    data_dir,
    start,
    split,
    horizon,
    meta,
    drop_censored,
    weight_dip,
    n_cv,
    stop_mult=1.25,
    target_mult=0.75,
    save_path: str = DEFAULT_MODEL_PATH,
    use_mtf: bool = True,
    use_cross: bool = True,
    use_cot: bool = True,
    weight_vol: bool = False,
    search: bool = False,
    stack: bool = False,
    side: str = "long",
) -> dict:
    from src.macro.overlay import macro_for_model

    label_horizon = DEFAULT_META_HORIZON if meta else horizon
    label_kwargs = (
        {"stop_mult": stop_mult, "target_mult": target_mult}
        if not meta
        else {"horizon": label_horizon}
    )

    macro = macro_for_model(data_dir)
    ds = collect_dataset(
        symbols,
        group,
        timeframe,
        data_dir,
        start,
        label_horizon,
        macro,
        meta,
        drop_censored,
        weight_dip,
        label_kwargs,
        use_mtf,
        use_cross,
        use_cot,
        weight_vol,
        side=side,
    )
    X, y = ds["X"], ds["y"]
    base_rate = y.mean() * 100
    print(
        f"Dataset: {len(X):,} bars across "
        f"{ds['symbol'].nunique()} symbols · "
        f"{int((y == 1).sum())} wins / {int((y == 0).sum())} losses "
        f"({base_rate:.1f}% base rate)"
        f"{' · confirmed dips only' if meta else ''}"
    )

    best_params = None
    if search:
        print("\nHyperparameter search (chronological split, recent rows):")
        best_params = search_hyperparams(ds, label_horizon)

    calibrator = None
    metrics = None
    model = None
    stack_result = None

    if n_cv > 1:
        try:
            wf = walk_forward(ds, label_horizon, n_cv, model_kwargs=best_params or None)
        except RuntimeError:
            print(
                f"  [!] Walk-forward found no usable folds "
                f"({len(ds['X']):,} rows) — falling back to single split"
            )
            wf = None
        if wf is not None:
            metrics = wf["metrics"]
            print(f"\nWalk-forward OOS ({n_cv} purged folds, early stopping):")
            _print_metrics(metrics)
            _print_deciles(wf["oos_y"], wf["oos_p"])
            _print_gate(wf["oos_y"], wf["oos_p"], wf["oos_confirmed"])
            _print_dip_metrics(wf, ds)
            if stack:
                stack_result = stack_oos(ds, wf["oos_idx"], wf["oos_p"])
                if stack_result:
                    print(
                        f"  Stack (LR on lgb+dip+macro+adx, {stack_result['n_eval']} rows): "
                        f"lgb {stack_result['lgb_auc']:.3f} -> stacked "
                        f"{stack_result['stacked_auc']:.3f} "
                        f"(Δ{stack_result['delta']:+.3f})"
                    )
            # Fit the isotonic calibrator on the pooled OOS predictions
            # (genuinely out-of-fold) so probabilities are true
            # probabilities at serve time.
            calibrator = fit_calibrator(wf["oos_y"], wf["oos_p"])
            print(f"  Calibrated on {len(wf['oos_y']):,} pooled OOS rows")
            model = _train_production(ds, split, label_horizon, best_params)

    if model is None:
        sp = split_chronological(ds, split, label_horizon)
        n_tr, n_te = len(sp["X_train"]), len(sp["X_test"])
        print(
            f"Split {split}: train {n_tr:,} bars (embargo {label_horizon}d) "
            f"· test {n_te:,} bars"
        )
        if n_tr < 2000 or n_te < 500:
            print(
                "  [!] Too few samples for a meaningful split — training on "
                "all data for the prediction model instead."
            )
            X_tr, y_tr, w_tr = X, y, ds["weight"]
            X_val, y_val, w_val = _chrono_val_split(X_tr, y_tr, w_tr)[3:]
            model = train_model(
                X_tr,
                y_tr,
                sample_weight=w_tr,
                X_val=X_val,
                y_val=y_val,
                sample_weight_val=w_val,
                model_kwargs=best_params or None,
            )
            metrics = {}
        else:
            X_tr, y_tr, w_tr = sp["X_train"], sp["y_train"], sp["weight_train"]
            X_val, y_val, w_val = _chrono_val_split(X_tr, y_tr, w_tr)[3:]
            model = train_model(
                X_tr,
                y_tr,
                sample_weight=w_tr,
                X_val=X_val,
                y_val=y_val,
                sample_weight_val=w_val,
                model_kwargs=best_params or None,
            )
            prob_test = model.predict_proba(sp["X_test"])[:, 1]
            metrics = evaluate(sp["y_test"].values, prob_test)
            print(f"\nOut-of-sample ({split} → now, early stopping):")
            _print_metrics(metrics)
            _print_deciles(sp["y_test"].values, prob_test)
            _print_gate(sp["y_test"].values, prob_test, sp["confirmed_test"].values)
            _print_dip_metrics_manual(sp, prob_test, metrics)
            calibrator = _fit_tail_calibrator(sp, best_params)

    print("\nTop 12 features by gain:")
    imp = feature_importance(model, X.columns.tolist())
    print(imp.head(12).to_string(index=False))

    meta_info = {
        "model": "lgbm-dip-prob-v3",
        "trained_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "label": "meta" if meta else "1r",
        "drop_censored": drop_censored,
        "weight_dip": weight_dip,
        "weight_vol": weight_vol,
        "use_mtf": use_mtf,
        "use_cross": use_cross,
        "use_cot": use_cot,
        "cv_folds": n_cv,
        "horizon": horizon,
        "split": split,
        "n_symbols": int(ds["symbol"].nunique()),
        "n_train": int((ds["time"] < pd.Timestamp(split)).sum()),
        "n_test": int((ds["time"] >= pd.Timestamp(split)).sum()),
        "symbols": sorted(ds["symbol"].unique().tolist()),
    }
    if metrics:
        meta_info["auc_oos"] = metrics["auc"]
        meta_info["dip_auc"] = metrics.get("dip_auc")
        meta_info["dip_n"] = metrics.get("dip_n")
    if stack_result:
        meta_info["stacked_auc"] = stack_result["stacked_auc"]
    if best_params:
        meta_info["best_params"] = best_params
    path = save_model(
        model, X.columns.tolist(), meta_info, save_path, calibrator=calibrator
    )
    print(
        f"\nSaved model → {path} (OOS AUC {meta_info.get('auc_oos', 'n/a')} · "
        f"calibrated {calibrator is not None})"
    )
    return {"metrics": metrics, "model_path": path, "importance": imp}


def _train_production(
    ds: dict, split: str, horizon: int, model_kwargs: Optional[dict] = None
):
    """Final production model on all history up to the split (with embargo)."""
    sp = split_chronological(ds, split, horizon)
    X_tr, y_tr, w_tr = sp["X_train"], sp["y_train"], sp["weight_train"]
    X_val, y_val, w_val = _chrono_val_split(X_tr, y_tr, w_tr)[3:]
    return train_model(
        X_tr,
        y_tr,
        sample_weight=w_tr,
        X_val=X_val,
        y_val=y_val,
        sample_weight_val=w_val,
        model_kwargs=model_kwargs,
    )


def _fit_tail_calibrator(sp: dict, model_kwargs: Optional[dict] = None):
    """Isotonic calibrator on a chronological tail of the training set."""
    n_tr = len(sp["X_train"])
    if n_tr < 2000:
        return None
    n_cal = max(200, int(0.15 * n_tr))
    X_cal, y_cal = sp["X_train"].tail(n_cal), sp["y_train"].tail(n_cal)
    X_fit, y_fit = sp["X_train"].head(n_tr - n_cal), sp["y_train"].head(n_tr - n_cal)
    w_fit = sp["weight_train"].head(n_tr - n_cal)
    X_val, y_val, w_val = _chrono_val_split(X_fit, y_fit, w_fit)[3:]
    model = train_model(
        X_fit,
        y_fit,
        sample_weight=w_fit,
        X_val=X_val,
        y_val=y_val,
        sample_weight_val=w_val,
        model_kwargs=model_kwargs,
    )
    p_cal = model.predict_proba(X_cal)[:, 1]
    calib = fit_calibrator(y_cal.values, p_cal)
    print(f"  Calibrated on last {n_cal:,} training bars")
    return calib


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------


def _print_metrics(metrics: dict) -> None:
    print(f"  AUC        : {metrics['auc']:.3f}")
    print(f"  Log-loss   : {metrics['logloss']:.3f}")
    print(
        f"  Accuracy   : {metrics['accuracy']:.3f} "
        f"(base {metrics['positive_rate']:.2f})"
    )
    print(f"  Precision  : {metrics['precision']:.3f} · Recall {metrics['recall']:.3f}")


def _print_deciles(y: np.ndarray, p: np.ndarray) -> None:
    rows = decile_report(y, p)
    if not rows:
        return
    lift = top_decile_lift(y, p)
    rates = " ".join(f"{r['win_rate'] * 100:4.0f}" for r in rows)
    print(f"  Deciles 0→9 win% : {rates}")
    print(
        f"  Base {lift['base'] * 100:.0f}% · top decile "
        f"{lift['top_decile'] * 100:.0f}% · lift {lift['lift']:.2f}x"
    )


def _print_gate(y: np.ndarray, p: np.ndarray, confirmed: np.ndarray) -> None:
    g = dip_filter_gate(y, p, confirmed)
    if g is None:
        print("  Dip filter gate: too few confirmed-dip OOS rows to evaluate")
        return
    print(
        f"  Dip filter gate ({g['n_dips']} dips): base "
        f"{g['base_win_rate'] * 100:.0f}% → top-half "
        f"{g['top_half_win_rate'] * 100:.0f}% · bottom-half "
        f"{g['bottom_half_win_rate'] * 100:.0f}% (lift {g['lift']:.2f}x)"
    )


def _print_dip_metrics(wf: dict, ds: dict) -> None:
    """Dip-specific validation for walk-forward: OOS rows that were dips."""
    idx = wf["oos_confirmed"]
    if idx.any():
        dm = evaluate(wf["oos_y"][idx], wf["oos_p"][idx])
        print(
            f"  On confirmed dips ({int(idx.sum())} bars): "
            f"AUC {dm['auc']:.3f} · win rate "
            f"{dm['positive_rate'] * 100:.0f}% "
            f"(hit-rate {dm['recall']:.2f})"
        )


def _print_dip_metrics_manual(sp: dict, prob_test: np.ndarray, metrics: dict) -> None:
    """Dip-specific validation: only bars with a confirmed dip setup."""
    dip_idx = sp["confirmed_test"]
    if dip_idx.any():
        dm = evaluate(sp["y_test"].values[dip_idx], prob_test[dip_idx])
        print(
            f"  On confirmed dips ({int(dip_idx.sum())} bars): "
            f"AUC {dm['auc']:.3f} · win rate "
            f"{dm['positive_rate'] * 100:.0f}% "
            f"(hit-rate {dm['recall']:.2f})"
        )
        metrics["dip_auc"] = dm["auc"]
        metrics["dip_n"] = int(dip_idx.sum())
        metrics["dip_win_rate"] = dm["positive_rate"]


# ---------------------------------------------------------------------------
# Group dispatch
# ---------------------------------------------------------------------------


def _symbol_groups(symbols: List[str]) -> Dict[str, List[str]]:
    """Split symbols into gold / JPY-cross / majors groups."""
    groups: Dict[str, List[str]] = {"gold": [], "jpy": [], "majors": []}
    for s in symbols:
        if s == "XAUUSD":
            groups["gold"].append(s)
        elif "JPY" in s:
            groups["jpy"].append(s)
        else:
            groups["majors"].append(s)
    return {k: v for k, v in groups.items() if v}


def _group_model_path(group: Optional[str]) -> str:
    return f"models/dip_lgbm_{group}.joblib" if group else DEFAULT_MODEL_PATH


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        description="NexusQuant ensemble signal model",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--group", default=None, help="data group")
    parser.add_argument("--symbols", default=None, help="comma-separated symbol list")
    parser.add_argument("--data-dir", default="data/raw")
    parser.add_argument(
        "--timeframe", default=None, help="D1/H4/H1 (default from --group)"
    )
    parser.add_argument("--start", default=None, help="start date")
    parser.add_argument(
        "--split", default="2022-01-01", help="chronological train/test split date"
    )
    parser.add_argument(
        "--horizon",
        type=int,
        default=DEFAULT_HORIZON,
        help="1R label horizon in bars (1r label only)",
    )
    parser.add_argument(
        "--label",
        choices=["1r", "meta"],
        default="meta",
        help="meta = real engine outcome on confirmed dips "
        "only (the deployment objective, default); "
        "1r = asymmetric triple-barrier on all bars",
    )
    parser.add_argument(
        "--side",
        choices=["long", "short"],
        default="long",
        help="long = Buy-the-Dip model (default); short = "
        "Sell-the-Rally model (mirror labels + rally "
        "signal features, saved to "
        "models/rally_lgbm.joblib)",
    )
    parser.add_argument(
        "--drop-censored",
        action="store_true",
        help="drop censored 1R labels (no forward-close label)",
    )
    parser.add_argument(
        "--stop-mult",
        type=float,
        default=1.25,
        help="stop distance as a multiple of ATR",
    )
    parser.add_argument(
        "--target-mult",
        type=float,
        default=0.75,
        help="target distance as a multiple of ATR",
    )
    parser.add_argument(
        "--weight-dip",
        type=float,
        default=5.0,
        help="row-weight multiplier for confirmed-dip bars "
        "(aligns the training distribution with where "
        "the model is deployed; 1.0 = equal weights)",
    )
    parser.add_argument(
        "--cv", type=int, default=5, help="purged walk-forward folds (1 = single split)"
    )
    parser.add_argument(
        "--weight-vol",
        action="store_true",
        help="vol-aware sample weighting: down-weight "
        "high-volatility bars, up-weight calm ones",
    )
    parser.add_argument(
        "--no-mtf", action="store_true", help="disable H4 multi-timeframe features"
    )
    parser.add_argument(
        "--no-cross", action="store_true", help="disable cross-asset risk/gold features"
    )
    parser.add_argument(
        "--no-cot", action="store_true", help="disable COT positioning features"
    )
    parser.add_argument(
        "--fetch-cot",
        action="store_true",
        help="refresh the cached CFTC COT positioning files (weekly) before training",
    )
    parser.add_argument(
        "--search",
        action="store_true",
        help="chronological hyperparameter search first",
    )
    parser.add_argument(
        "--stack",
        action="store_true",
        help="report rule-score stacking comparison on OOS",
    )
    parser.add_argument(
        "--per-group",
        action="store_true",
        help="train separate gold / JPY-cross / majors models",
    )
    parser.add_argument(
        "--predict", default=None, help="symbol to score with the saved model"
    )
    parser.add_argument("--json", action="store_true", help="emit JSON")
    args = parser.parse_args(argv)

    if args.timeframe is None:
        args.timeframe = {"h1": "H1", "h4": "H4"}.get(args.group, "D1")

    if args.predict:
        df = prepare_frame(
            args.predict, args.group, args.timeframe, args.data_dir, args.start
        )
        path = _group_model_path(args.group)
        bundle = load_model(path)
        if bundle is None and path != DEFAULT_MODEL_PATH:
            bundle = load_model(DEFAULT_MODEL_PATH)
        if bundle is None:
            print(
                f"No model found ({path}) — train first with "
                f"`python -m src.model.run --group full_fx`",
                file=sys.stderr,
            )
            return 1
        prob = predict_series(
            df,
            bundle=bundle,
            symbol=args.predict,
            group=args.group,
            data_dir=args.data_dir,
        )
        if prob is None or prob.notna().sum() == 0:
            print("Prediction failed.", file=sys.stderr)
            return 1
        last = prob.dropna().iloc[-1]
        out = {
            "symbol": args.predict,
            "date": str(prob.dropna().index[-1].date()),
            "prob_up_1r": round(float(last), 4),
            "prob_pct": round(float(last) * 100, 1),
            "label": (
                "High conviction"
                if last >= 0.6
                else "Moderate"
                if last >= 0.5
                else "Low"
            ),
            "auc_oos": bundle.get("meta", {}).get("auc_oos"),
            "calibrated": bundle.get("calibrator") is not None,
        }
        if args.json:
            print(json.dumps(out, indent=2))
        else:
            print(
                f"\n{args.predict} {out['date']} — Bullish probability: "
                f"{out['prob_pct']:.1f}% ({out['label']}) · "
                f"model OOS AUC {out['auc_oos']} · "
                f"calibrated {out['calibrated']}\n"
            )
        return 0

    if args.symbols:
        symbols = [s.strip() for s in args.symbols.split(",") if s.strip()]
    else:
        symbols = discover_symbols(args.data_dir, args.group, args.timeframe)
    if not symbols:
        print("No symbols found.")
        return 1

    if args.fetch_cot and not args.no_cot:
        from src.model.cot import cot_status, update_cot

        before = cot_status(args.data_dir)
        res = update_cot(args.data_dir)
        if res["fetched"]:
            print(
                f"COT: fetched {len(res['currencies'])} currencies "
                f"({len(before['currencies'])} cached before)"
            )
        elif res["error"]:
            print(
                f"COT: {res['error']} - using cached files "
                f"({len(res['currencies'])} currencies)"
            )
        else:
            print(f"COT: {res['reason']} ({len(res['currencies'])} currencies)")

    common = dict(
        group=args.group,
        timeframe=args.timeframe,
        data_dir=args.data_dir,
        start=args.start,
        split=args.split,
        horizon=args.horizon,
        meta=args.label == "meta",
        drop_censored=args.drop_censored,
        weight_dip=args.weight_dip,
        n_cv=args.cv,
        stop_mult=args.stop_mult,
        target_mult=args.target_mult,
        use_mtf=not args.no_mtf,
        use_cross=not args.no_cross,
        use_cot=not args.no_cot,
        weight_vol=args.weight_vol,
        search=args.search,
        stack=args.stack,
        side=args.side,
    )
    # Short-side models save to their own path so the two heads stay
    # independent (dip_lgbm.joblib vs rally_lgbm.joblib).
    if args.side == "short":
        common["save_path"] = DEFAULT_SHORT_MODEL_PATH

    results = {}
    if args.per_group:
        groups = _symbol_groups(symbols)
        print(
            f"Training per-group models: "
            f"{', '.join(f'{g} ({len(s)})' for g, s in groups.items())}"
        )
        for g, syms_g in groups.items():
            print(f"\n=== Group: {g} ===")
            res = train_and_evaluate(syms_g, save_path=_group_model_path(g), **common)
            results[g] = res
            print(
                f"[{g}] OOS AUC {res['metrics'].get('auc', 'n/a')} → "
                f"{res['model_path']}"
            )
    else:
        print(
            f"Training on {len(symbols)} symbols "
            f"({args.group or 'majors'} · {args.timeframe} · "
            f"label={args.label} · cv={args.cv}"
            f"{' · search' if args.search else ''})"
        )
        results["pooled"] = train_and_evaluate(symbols, **common)

    if args.json:
        if args.per_group:
            out = {}
            for g, res in results.items():
                imp = res["importance"].to_dict(orient="records")
                for row in imp:
                    if isinstance(row.get("gain"), float) and math.isnan(row["gain"]):
                        row["gain"] = None
                out[g] = {
                    "metrics": res["metrics"],
                    "model_path": res["model_path"],
                    "top_features": imp[:15],
                }
            print(json.dumps(out, indent=2))
        else:
            res = results["pooled"]
            imp = res["importance"].to_dict(orient="records")
            for row in imp:
                if isinstance(row.get("gain"), float) and math.isnan(row["gain"]):
                    row["gain"] = None
            print(
                json.dumps(
                    {
                        "metrics": res["metrics"],
                        "model_path": res["model_path"],
                        "top_features": imp[:15],
                    },
                    indent=2,
                )
            )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
