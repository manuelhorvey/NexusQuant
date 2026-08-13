"""
NexusQuant - Ensemble signal model (LightGBM).

Trains a gradient-boosted classifier that turns the causal indicator /
regime / dip-feature stack into a **Bullish Probability**: P(the next 1R
move is up) for a bar. Used to filter and rank Buy-the-Dip setups.

Validation is chronological (train < split <= test, with an embargo of one
label horizon around the split) - never shuffled across time.
"""

from __future__ import annotations

import functools
from pathlib import Path
from typing import Dict, List, Optional

import joblib
import numpy as np
import pandas as pd

from src.model.features import (
    CATEGORICAL_FEATURES,
    FEATURE_COLUMNS,
    MACRO_FEATURES,
    build_features,
)

DEFAULT_MODEL_PATH = "models/dip_lgbm.joblib"
# Sell-the-Rally (short-side) model - same feature columns, mirror labels.
DEFAULT_SHORT_MODEL_PATH = "models/rally_lgbm.joblib"
DEFAULT_SEED = 42


# ---------------------------------------------------------------------------
# Model factory
# ---------------------------------------------------------------------------


def make_model(seed: int = DEFAULT_SEED):
    """LightGBM classifier with conservative, overfit-resistant defaults."""
    from lightgbm import LGBMClassifier

    return LGBMClassifier(
        n_estimators=400,
        learning_rate=0.04,
        num_leaves=31,
        max_depth=7,
        colsample_bytree=0.8,
        subsample=0.8,
        subsample_freq=1,
        min_child_samples=20,
        reg_alpha=0.1,
        reg_lambda=1.0,
        random_state=seed,
        n_jobs=-1,
        verbose=-1,
    )


# ---------------------------------------------------------------------------
# Training / evaluation
# ---------------------------------------------------------------------------


def train_model(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    sample_weight: Optional[pd.Series] = None,
    scale_pos_weight: Optional[float] = None,
    X_val: Optional[pd.DataFrame] = None,
    y_val: Optional[pd.Series] = None,
    sample_weight_val: Optional[pd.Series] = None,
    early_stopping_rounds: int = 50,
    model_kwargs: Optional[Dict] = None,
):
    """
    Train on the given matrix; returns the fitted LightGBM model.

    ``sample_weight`` is passed straight to LightGBM (row weighting).
    ``model_kwargs`` (from the hyperparameter search) override the default
    LightGBM hyperparameters. Categorical features (``symbol``) are declared
    to LightGBM automatically.

    When a chronological validation slice is supplied (``X_val``/``y_val``),
    training uses **early stopping on validation AUC** - the single most
    effective guard against the fixed-iteration overfit that a raw 400-tree
    model was subject to. The slice must be the chronologically last rows of
    the training window (callers reserve it from the tail); it is never
    touched by the final model weights.

    On very small datasets ``min_child_samples`` can prevent any tree after
    the first from splitting (LightGBM stops at 1 iteration) - detect that
    and refit with relaxed constraints.
    """
    categorical = [c for c in CATEGORICAL_FEATURES if c in X_train.columns]
    # Pooled concat can promote a categorical column back to object dtype;
    # LightGBM only accepts category/int categorical columns, so coerce.
    if categorical:
        X_train = X_train.copy()
        for c in categorical:
            X_train[c] = X_train[c].astype("category")
        if X_val is not None and len(X_val):
            X_val = X_val.copy()
            for c in categorical:
                X_val[c] = X_val[c].astype("category")

    use_es = (
        X_val is not None
        and y_val is not None
        and len(X_val) >= 200
        and len(X_val) <= len(X_train) * 0.5
    )

    def _fit(model, X, y, w):
        fit_kwargs = {"sample_weight": w}
        if categorical:
            fit_kwargs["categorical_feature"] = categorical
        if use_es:
            fit_kwargs.update(
                _early_stopping_kwargs(
                    X_val, y_val, sample_weight_val, early_stopping_rounds
                )
            )
        return model.fit(X, y, **fit_kwargs)

    if scale_pos_weight is None:
        n_pos = float((y_train == 1).sum())
        n_neg = float((y_train == 0).sum())
        scale_pos_weight = n_neg / n_pos if n_pos > 0 else 1.0

    model = make_model().set_params(**model_kwargs) if model_kwargs else make_model()
    model = model.set_params(scale_pos_weight=scale_pos_weight)
    _fit(model, X_train, y_train, sample_weight)

    n_iters = int(getattr(model, "n_iter_", 0) or 0)
    if n_iters <= 1:
        # Truly degenerate fit (no tree after the first could split, e.g.
        # min_child_samples too large for a tiny dataset). This must NOT
        # fire for legitimately-short early-stopped fits, so it is gated on
        # a single iteration only. The relaxed refit keeps the same
        # validation slice + model_kwargs so early stopping and the search
        # config still apply.
        from lightgbm import LGBMClassifier

        relaxed = LGBMClassifier(
            n_estimators=300,
            learning_rate=0.04,
            num_leaves=15,
            max_depth=6,
            min_child_samples=5,
            colsample_bytree=0.8,
            subsample=1.0,
            reg_alpha=0.0,
            reg_lambda=0.1,
            scale_pos_weight=scale_pos_weight,
            random_state=DEFAULT_SEED,
            n_jobs=-1,
            verbose=-1,
            **(model_kwargs or {}),
        )
        _fit(relaxed, X_train, y_train, sample_weight)
        model = relaxed
    return model


def _early_stopping_kwargs(
    X_val: pd.DataFrame,
    y_val: pd.Series,
    sample_weight_val: Optional[pd.Series],
    rounds: int,
) -> Dict:
    """Version-aware early-stopping fit kwargs (LightGBM >= 4.5 uses the
    new ``eval_X``/``eval_y`` API; older versions use ``eval_set``)."""
    kwargs: Dict = {"eval_metric": "auc"}
    try:
        import lightgbm
        from packaging.version import Version

        if Version(lightgbm.__version__) >= Version("4.5"):
            kwargs["eval_X"] = X_val
            kwargs["eval_y"] = y_val
            kwargs["callbacks"] = [
                lightgbm.early_stopping(rounds, verbose=False),
                lightgbm.log_evaluation(0),
            ]
        else:
            kwargs["eval_set"] = [(X_val, y_val)]
            kwargs["early_stopping_rounds"] = rounds
    except Exception:
        kwargs["eval_set"] = [(X_val, y_val)]
        kwargs["early_stopping_rounds"] = rounds
    if sample_weight_val is not None:
        kwargs["eval_sample_weight"] = [sample_weight_val]
    return kwargs


def fit_calibrator(y_true: np.ndarray, prob: np.ndarray):
    """
    Fit an isotonic probability calibrator on a held-out validation slice.

    Probabilities from LightGBM are only rank-order informative; the risk
    layer feeds them into fractional Kelly, which needs them to be *true*
    probabilities. Isotonic regression (clipped) maps raw scores to
    calibrated ones on a chronological validation slice.
    """
    from sklearn.isotonic import IsotonicRegression

    iso = IsotonicRegression(out_of_bounds="clip")
    iso.fit(np.asarray(prob), np.asarray(y_true))
    return iso


def apply_calibrator(calibrator, prob: np.ndarray) -> np.ndarray:
    """Transform raw model probabilities through the calibrator."""
    if calibrator is None:
        return prob
    return np.asarray(calibrator.predict(np.asarray(prob)))


def evaluate(y_true: np.ndarray, prob: np.ndarray) -> Dict:
    """OOS metrics: AUC, log-loss, accuracy, precision/recall @ 50%."""
    from sklearn.metrics import (
        accuracy_score,
        log_loss,
        precision_score,
        recall_score,
        roc_auc_score,
    )

    y = np.asarray(y_true)
    p = np.asarray(prob)
    pred = (p >= 0.5).astype(int)
    n = len(y)
    pos_rate = float((y == 1).mean()) if n else 0.0
    return {
        "n": n,
        "positive_rate": round(pos_rate, 4),
        "auc": round(float(roc_auc_score(y, p)), 4)
        if n > 1 and len(np.unique(y)) > 1
        else 0.0,
        "logloss": round(float(log_loss(y, p)), 4) if n else 0.0,
        "accuracy": round(float(accuracy_score(y, pred)), 4) if n else 0.0,
        "precision": round(float(precision_score(y, pred)), 4)
        if n and pred.sum() > 0
        else 0.0,
        "recall": round(float(recall_score(y, pred)), 4) if n else 0.0,
    }


def decile_report(y_true: np.ndarray, prob: np.ndarray, n_bins: int = 10) -> List[Dict]:
    """
    Win rate per probability decile (decile 9 = highest predicted prob).

    Equal-count buckets ordered by model probability. A model with real
    edge shows win rates that rise across deciles; a coin-flip model shows a
    flat line. This is the honest diagnostic for a *filter* model - global
    AUC hides exactly this monotonicity (or lack of it).
    """
    y = np.asarray(y_true)
    p = np.asarray(prob)
    if len(y) < n_bins * 2:
        return []
    order = np.argsort(p)
    y_s, p_s = y[order], p[order]
    n = len(y_s)
    rows = []
    for d in range(n_bins):
        lo = int(d * n / n_bins)
        hi = int((d + 1) * n / n_bins)
        if hi - lo < 1:
            continue
        seg = y_s[lo:hi]
        rows.append(
            {
                "decile": d,
                "n": int(hi - lo),
                "win_rate": round(float(seg.mean()), 4),
                "prob_mid": round(float(p_s[lo:hi].mean()), 4),
            }
        )
    return rows


def top_decile_lift(y_true: np.ndarray, prob: np.ndarray, n_bins: int = 10) -> Dict:
    """Lift of the top probability decile vs the base rate."""
    rows = decile_report(y_true, prob, n_bins)
    base = float(np.asarray(y_true).mean())
    top = rows[-1]["win_rate"] if rows else base
    return {
        "base": round(base, 4),
        "top_decile": round(top, 4),
        "lift": round(top / base, 3) if base > 0 else 0.0,
    }


def dip_filter_gate(
    y_true: np.ndarray, prob: np.ndarray, confirmed: np.ndarray
) -> Optional[Dict]:
    """
    Backtest-style gate: does filtering dips by model prob help?

    Splits the *confirmed-dip* OOS rows at the median predicted probability
    and compares win rates. This is the deployment question - "on the
    setups we actually trade, does the model separate winners?" - which
    global pooled AUC does not answer. Returns None when too few dips.
    """
    y = np.asarray(y_true)
    p = np.asarray(prob)
    conf = np.asarray(confirmed, dtype=bool)
    idx = conf
    if int(idx.sum()) < 40:
        return None
    y_c, p_c = y[idx], p[idx]
    med = float(np.median(p_c))
    hi = p_c >= med
    lo = p_c < med
    # A tied median can leave one side empty (calibrated probabilities are
    # heavily tied); treat an empty side as "no separation" rather than NaN.
    if not hi.any() or not lo.any():
        return {
            "n_dips": int(idx.sum()),
            "base_win_rate": round(float(y_c.mean()), 4),
            "top_half_win_rate": round(float(y_c.mean()), 4),
            "bottom_half_win_rate": round(float(y_c.mean()), 4),
            "lift": 1.0,
        }
    hi_wr = float(y_c[hi].mean())
    lo_wr = float(y_c[lo].mean())
    return {
        "n_dips": int(idx.sum()),
        "base_win_rate": round(float(y_c.mean()), 4),
        "top_half_win_rate": round(hi_wr, 4),
        "bottom_half_win_rate": round(lo_wr, 4),
        "lift": round(hi_wr / y_c.mean(), 3) if y_c.mean() > 0 else 0.0,
    }


def feature_importance(model, features: List[str]) -> pd.DataFrame:
    """Gain-based feature importance, sorted descending."""
    try:
        imp = model.feature_importances_
        gain = getattr(model, "booster_", None) and model.booster_.feature_importance(
            importance_type="gain"
        )
    except Exception:
        gain = None
    df = pd.DataFrame({"feature": features, "gain": gain if gain is not None else imp})
    df = df.sort_values("gain", ascending=False).reset_index(drop=True)
    df["gain_pct"] = (df["gain"] / df["gain"].sum() * 100).round(2)
    return df


# Feature-name prefix/group map for the spec #10 factor-contribution table:
# the model's raw features are bucketed into the institutional factor groups
# (Trend / Momentum / Volatility / Volume / Positioning / Macro / Time /
# Cross-asset / Dip setup / Identity) and each group's total gain share is
# reported - the same style as "Trend Factors: +26%" in the final output.
FEATURE_GROUPS = {
    "Trend": {
        "adx",
        "plus_di",
        "minus_di",
        "vs_sma200_pct",
        "sma20_gap",
        "sma50_gap",
        "slope_20",
        "regime_confidence",
        "ma_stack",
        "trend",
        "above_sma200",
        "adx_x_slope",
        "vol_x_trend",
    },
    "Momentum": {
        "rsi_14",
        "macd_hist",
        "macd",
        "bb_pct_b",
        "ret_1",
        "ret_5",
        "ret_10",
        "vol_x_momentum",
    },
    "Volatility": {"atr_pct", "volatility_20", "bb_width"},
    "Multi-timeframe": {"h4_mom5", "h4_mom20", "h4_vol_ratio", "h4_vs_sma200"},
    "Volume": {"relative_volume"},
    "Positioning": {"cot_percentile"},
    "Macro": {"dxy_score", "vix_score", "tnx_score", "macro_bias", "dxy_mom20"},
    "Time": {"day_of_week", "month", "month_end", "mid_month"},
    "Cross-asset": {"risk_mom5", "risk_mom20", "gold_mom5", "gold_mom20"},
    "Dip setup": {
        "dip_score",
        "bias_score",
        "pullback",
        "cooled",
        "at_support",
        "fib_zone",
        "trigger",
        "dip_depth_pct",
        "entry_lo_gap",
        "invalidation_gap",
    },
    "Identity": {"symbol"},
}


def importance_summary(
    model_path: str = DEFAULT_MODEL_PATH, top_n: int = 10
) -> Optional[Dict]:
    """
    Feature importance for the saved model: ``{top, by_group, n}``.

    * ``top`` - the ``top_n`` features by gain share (``{feature, gain_pct}``).
    * ``by_group`` - gain share aggregated into the spec #10 factor groups
      (``{group, gain_pct}``), sorted descending.
    * ``n`` - total feature count.

    Returns None when no model is saved (the report degrades gracefully,
    exactly like the probability itself).
    """
    bundle = load_model(model_path)
    if bundle is None:
        return None
    features = bundle.get("features", FEATURE_COLUMNS)
    df = feature_importance(bundle["model"], features)
    top = df.head(top_n)[["feature", "gain_pct"]].to_dict("records")
    groups: Dict[str, float] = {}
    for _, row in df.iterrows():
        f = row["feature"]
        g = next((name for name, cols in FEATURE_GROUPS.items() if f in cols), "Other")
        groups[g] = groups.get(g, 0.0) + float(row["gain_pct"])
    by_group = sorted(
        [{"group": g, "gain_pct": round(v, 2)} for g, v in groups.items()],
        key=lambda r: -r["gain_pct"],
    )
    return {"top": top, "by_group": by_group, "n": len(features)}


# ---------------------------------------------------------------------------
# Persistence
# ---------------------------------------------------------------------------


def save_model(
    model, features: List[str], meta: Dict, path: str, calibrator=None
) -> str:
    """Save model + feature list + metadata + calibrator as one bundle.

    Also appends a record to the model-governance registry (see
    ``src.model.registry``) so every saved probability can be traced back
    to its training run. The registry write is best-effort - a ledger
    failure must never break a model save."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    bundle = {
        "kind": "nexusquant-dip-prob",
        "model": model,
        "features": features,
        "meta": meta,
        "calibrator": calibrator,
    }
    joblib.dump(bundle, path)
    try:
        from src.model.registry import record

        record(str(path), meta=meta)
    except Exception:
        pass
    return str(path)


@functools.lru_cache(maxsize=8)
def _load_model_cached(path: str, mtime: float) -> Optional[Dict]:
    """Uncached loader body; ``load_model`` keys the cache on ``(path,
    mtime)`` so a retrained model is picked up by long-lived processes
    (API server / dashboard / watch loops) without a restart."""
    p = Path(path)
    if not p.exists():
        return None
    try:
        bundle = joblib.load(p)
    except Exception:
        return None
    if not isinstance(bundle, dict) or "model" not in bundle:
        return None
    return bundle


def load_model(path: str = DEFAULT_MODEL_PATH) -> Optional[Dict]:
    """
    Load a saved bundle; returns None if missing or incompatible.

    Cached so universe scans (scanner + report + meta per symbol) do not
    re-read the model from disk on every symbol. The cache key includes the
    file's mtime, so re-training (which rewrites the file) is picked up on
    the next call even in a long-lived process. Callers must not mutate
    the returned bundle.
    """
    p = Path(path)
    try:
        mtime = p.stat().st_mtime if p.exists() else 0.0
    except OSError:
        mtime = 0.0
    return _load_model_cached(str(p), mtime)


def model_meta_summary(model_path: str = DEFAULT_MODEL_PATH) -> Optional[Dict]:
    """Human-readable summary of the saved model (for reports/dashboard)."""
    bundle = load_model(model_path)
    if bundle is None:
        return None
    meta = bundle.get("meta", {})
    return {
        "path": model_path,
        "auc_oos": meta.get("auc_oos"),
        "features": len(bundle.get("features", [])),
        "calibrated": bundle.get("calibrator") is not None,
        "label": meta.get("label"),
        "trained_at": meta.get("trained_at"),
        "symbols": len(meta.get("symbols", [])),
    }


# ---------------------------------------------------------------------------
# Live prediction
# ---------------------------------------------------------------------------


def _proxies_cache_key(group: Optional[str], data_dir: str) -> tuple:
    """``(group, data_dir, latest mtime)`` so a data refresh busts the cache.
    A long-lived process (e.g. the API server) picks up updated parquet
    files on the next call without a restart."""
    latest = 0.0
    base = Path(data_dir) / group if group else Path(data_dir)
    for s in ("AUDJPY", "NZDJPY", "XAUUSD"):
        p = base / f"{s}_D1.parquet"
        try:
            latest = max(latest, p.stat().st_mtime)
        except OSError:
            pass
    return (group, data_dir, latest)


@functools.lru_cache(maxsize=8)
def _cross_proxies_cached(key: tuple) -> Dict:
    """
    Daily close frames for the risk (AUDJPY/NZDJPY) and gold (XAUUSD) legs
    from the same group the model was trained on. Empty dict when the files
    are absent - the feature builder treats that as neutral. Cached (keyed
    on the files' mtimes) so a universe scan pays the load once.
    """
    group, data_dir, _ = key
    from src.data.loader import clean_data, load_data

    proxies: Dict[str, pd.DataFrame] = {}
    for key_, syms in (("risk", ["AUDJPY", "NZDJPY"]), ("gold", ["XAUUSD"])):
        parts: Dict[str, pd.Series] = {}
        for s in syms:
            try:
                base = Path(data_dir)
                if group:
                    base = base / group
                p = base / f"{s}_D1.parquet"
                if not p.exists():
                    continue
                df = clean_data(load_data(p, symbol=s))
                parts[s] = df["close"].astype(float)
            except Exception:
                continue
        if parts:
            proxies[key_] = pd.DataFrame(parts)
    return proxies


def _cross_proxies(group: Optional[str], data_dir: str = "data/raw") -> Dict:
    """mtime-aware wrapper over ``_cross_proxies_cached``."""
    return _cross_proxies_cached(_proxies_cache_key(group, data_dir))


def _mtf_frame(
    symbol: Optional[str], group: Optional[str], data_dir: str
) -> Optional[pd.DataFrame]:
    """
    The symbol's H4 OHLCV frame for multi-timeframe features, or None.

    Mirrors the scanner's trigger loader layouts (flat group dir, nested
    ``H4/`` inside the group, sibling ``top/H4`` for ``mt5/D1``-style
    groups, data root, ``h4/`` subdir). The H4 resample inside
    ``mtf_features`` only uses same-day closes, so the features stay causal.
    """
    if not symbol:
        return None
    from src.data.loader import clean_data, load_data

    candidates = []
    if group:
        base = Path(data_dir) / group
        candidates += [
            base / f"{symbol}_H4.parquet",
            base / "H4" / f"{symbol}_H4.parquet",
        ]
        if "/" in group:
            top = Path(data_dir) / group.split("/", 1)[0]
            candidates += [top / "H4" / f"{symbol}_H4.parquet"]
    candidates += [
        Path(data_dir) / f"{symbol}_H4.parquet",
        Path(data_dir) / "h4" / f"{symbol}_H4.parquet",
    ]
    for p in candidates:
        if p.exists():
            try:
                return clean_data(load_data(p, symbol=symbol))
            except Exception:
                return None
    return None


def predict_series(
    df: pd.DataFrame,
    model_path: str = DEFAULT_MODEL_PATH,
    bundle: Optional[Dict] = None,
    symbol: Optional[str] = None,
    group: Optional[str] = None,
    data_dir: str = "data/raw",
    signal: Optional[pd.DataFrame] = None,
) -> Optional[pd.Series]:
    """
    Bullish probability series for a prepared OHLCV frame.

    Returns None if no model is available or the features cannot be built
    (callers fall back to rule-based output). All features are causal.

    The **same context the model was trained on is loaded at serve time** -
    top-down macro (1-day lag), the symbol's H4 frame, the cross-asset
    risk/gold proxies and the cached COT positioning - so live probabilities
    match training instead of silently using neutral zeros. Everything is
    memoized or graceful: when any context file is absent the feature
    builder falls back to neutral values, never crashes.

    Probabilities are passed through the saved isotonic calibrator when one
    exists, so the output is usable for Kelly sizing.
    """
    if bundle is None:
        bundle = load_model(model_path)
    if bundle is None:
        return None
    model = bundle["model"]
    features = bundle.get("features", FEATURE_COLUMNS)
    calibrator = bundle.get("calibrator")
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
        return None
    # Only rows with complete features get a probability.
    prob = pd.Series(np.nan, index=df.index)
    ready = X.notna().all(axis=1)
    if not ready.any():
        return None
    raw = model.predict_proba(X[ready])[:, 1]
    prob[ready] = apply_calibrator(calibrator, raw)
    return prob


def predict_short_series(
    df: pd.DataFrame,
    model_path: str = DEFAULT_SHORT_MODEL_PATH,
    bundle: Optional[Dict] = None,
    symbol: Optional[str] = None,
    group: Optional[str] = None,
    data_dir: str = "data/raw",
) -> Optional[pd.Series]:
    """
    Short-side mirror of ``predict_series``: P(short 1R win) from the
    Sell-the-Rally model. Builds the rally signal series (the same setup
    components, mirrored) so the features match the short model's training
    distribution. Returns None when no short model is saved - callers fall
    back to the rule-based rally engine.
    """
    try:
        from src.backtest.signals import rally_signal_series

        signal = rally_signal_series(df)
    except Exception:
        signal = None
    return predict_series(
        df,
        model_path,
        bundle,
        symbol=symbol,
        group=group,
        data_dir=data_dir,
        signal=signal,
    )


def predict_long_short(
    df: pd.DataFrame,
    symbol: Optional[str] = None,
    group: Optional[str] = None,
    data_dir: str = "data/raw",
    long_model: str = DEFAULT_MODEL_PATH,
    short_model: str = DEFAULT_SHORT_MODEL_PATH,
) -> Optional[Dict]:
    """
    Dual-side probability read: ``{prob_long, prob_short, net_bias}`` for
    the latest bar, using the two calibrated models.

    ``net_bias = prob_long - prob_short`` in [−1, +1] - the direction-aware
    headline that lets the rating print Strong Sell / Sell as well as Buy.
    Each side is graceful: when its model is missing the probability is
    ``None`` (callers fall back to the rule-based engine for that side).
    ``net_bias`` is only computed when BOTH sides are available - a missing
    short model must not read as P(short)=0 (that would fabricate a
    bearish tilt from an absent model). Returns None only when *neither*
    model is usable.
    """
    p_long = predict_series(
        df, long_model, symbol=symbol, group=group, data_dir=data_dir
    )
    p_short = predict_short_series(
        df, short_model, symbol=symbol, group=group, data_dir=data_dir
    )
    out: Dict = {
        "prob_long": None,
        "prob_short": None,
        "net_bias": None,
        "long_available": p_long is not None,
        "short_available": p_short is not None,
    }
    if p_long is not None:
        last_l = p_long.dropna()
        if len(last_l):
            out["prob_long"] = round(float(last_l.iloc[-1]), 4)
    if p_short is not None:
        last_s = p_short.dropna()
        if len(last_s):
            out["prob_short"] = round(float(last_s.iloc[-1]), 4)
    if out["prob_long"] is not None and out["prob_short"] is not None:
        out["net_bias"] = round(float(out["prob_long"] - out["prob_short"]), 4)
        return out
    if out["prob_long"] is not None or out["prob_short"] is not None:
        return out  # one side only: probabilities, no net_bias
    return None


if __name__ == "__main__":
    print("NexusQuant model module ready.")
