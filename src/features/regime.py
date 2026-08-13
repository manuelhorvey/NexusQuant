"""
NexusQuant - Market Regime Detection
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional


def linear_regression_slope(series: pd.Series, lookback: int = 20) -> pd.Series:
    """Calculate rolling linear regression slope of price."""

    def slope(x):
        if len(x) < 2:
            return np.nan
        y = np.arange(len(x))
        return np.polyfit(y, x, 1)[0]

    return series.rolling(lookback).apply(slope, raw=True)


def detect_regime(df: pd.DataFrame, adx_threshold: float = 25.0) -> pd.DataFrame:
    """
    Classify market regime for each bar.
    Possible regimes: Bull Trend | Bear Trend | Range / Chop | High Volatility
    """
    df = df.copy()

    # Required columns check
    required = ["close", "adx", "sma_200", "atr_14"]
    for col in required:
        if col not in df.columns:
            raise ValueError(
                f"Missing column for regime detection: {col}. Run add_all_indicators first."
            )

    # Trend strength & direction
    df["slope_20"] = linear_regression_slope(df["close"], 20)
    # Normalized slope (per-bar % of price): the raw slope is not
    # comparable across assets (0.01 pt/bar is huge for EURUSD at ~1.15
    # and tiny for XAUUSD at ~4400). The % form is cross-asset
    # comparable for reporting/ranking; the sign check below uses the
    # raw slope (scale-invariant for a sign test).
    df["slope_pct_20"] = df["slope_20"] / df["close"] * 100.0
    df["price_vs_200sma"] = df["close"] / df["sma_200"] - 1

    # Volatility regime (simple z-score of ATR)
    atr_median = df["atr_14"].rolling(100).median()
    df["atr_ratio"] = df["atr_14"] / atr_median

    conditions = [
        (df["adx"] >= adx_threshold)
        & (df["close"] > df["sma_200"])
        & (df["slope_20"] > 0),
        (df["adx"] >= adx_threshold)
        & (df["close"] < df["sma_200"])
        & (df["slope_20"] < 0),
        (df["atr_ratio"] > 1.8),
    ]
    choices = ["Bull Trend", "Bear Trend", "High Volatility"]

    df["regime"] = np.select(conditions, choices, default="Range / Chop")

    # Confidence score (simple)
    df["regime_confidence"] = np.where(
        df["adx"] > 40, 0.85, np.where(df["adx"] > 25, 0.70, 0.55)
    )

    return df


def get_current_regime_summary(df: pd.DataFrame) -> Dict:
    """Return a clean summary of the latest regime."""
    latest = df.iloc[-1]

    return {
        "regime": latest.get("regime", "Unknown"),
        "adx": round(latest.get("adx", 0), 2),
        "price_vs_200sma_pct": round(latest.get("price_vs_200sma", 0) * 100, 2),
        "slope_20": round(latest.get("slope_20", 0), 5),
        # normalized per-bar slope (% of price) - cross-asset comparable
        "slope_pct_20": round(latest.get("slope_pct_20", 0), 4),
        "confidence": round(latest.get("regime_confidence", 0), 2),
        "atr_14": round(latest.get("atr_14", 0), 2),
    }


# ---------------------------------------------------------------------------
# Clustering-based regime (institutional spec #1: "HMM or clustering")
# ---------------------------------------------------------------------------

REGIME_LEVELS = ["Bull Trend", "Bear Trend", "Range / Chop", "High Volatility"]


def _regime_features(df: pd.DataFrame) -> pd.DataFrame:
    """Normalized feature matrix for regime clustering.

    Uses the same causal building blocks as the rule-based detector but
    standardized: 20-bar linear slope, ADX, distance from 200SMA and the
    ATR vol ratio. Bars with missing values (warm-up) are dropped.
    """
    if "slope_20" not in df.columns:
        df = df.copy()
        df["slope_20"] = linear_regression_slope(df["close"], 20)
    if "price_vs_200sma" not in df.columns:
        df = df.copy()
        df["price_vs_200sma"] = df["close"] / df["sma_200"] - 1
    if "atr_ratio" not in df.columns:
        med = df["atr_14"].rolling(100).median()
        df = df.copy()
        df["atr_ratio"] = df["atr_14"] / med
    if "adx" not in df.columns:
        raise ValueError(
            "Missing column for regime detection: adx. Run add_all_indicators first."
        )

    feats = pd.DataFrame(
        {
            "slope_20": df["slope_20"],
            "adx": df["adx"],
            "vs200": df["price_vs_200sma"] * 100.0,
            "atr_ratio": df["atr_ratio"],
        }
    )
    return (feats - feats.mean()) / feats.std().replace(0, 1.0)


def _fallback_labels(df: pd.DataFrame) -> List[str]:
    """Deterministic labels when clustering/HMM cannot run. Uses the
    rule-based ``regime`` column when present; otherwise a neutral
    ``Range / Chop`` per bar (the detector must work standalone, even
    before ``detect_regime`` has added the column)."""
    if "regime" in df.columns:
        return df["regime"].tolist()
    return ["Range / Chop"] * len(df)


def cluster_regime_labels(df: pd.DataFrame, n_clusters: int = 4) -> List[str]:
    """Cluster the last ``n_clusters`` regimes with KMeans and label each
    cluster Bull/Bear/Range/High-Vol by its centroid.

    Returns one label per bar (numpy array). Uses scikit-learn when
    available; otherwise falls back to the deterministic rule-based
    labels (or a neutral label when no ``regime`` column exists yet).
    """
    feats = _regime_features(df).dropna()
    if len(feats) < max(60, n_clusters * 10):
        # too little history -> reuse the deterministic rule-based labels
        return _fallback_labels(df)
    try:
        from sklearn.cluster import KMeans

        km = KMeans(n_clusters=n_clusters, n_init=10, random_state=42)
        labels = km.fit_predict(feats.values)
        cents = km.cluster_centers_
        # centroid columns: 0=slope, 1=adx, 2=vs200, 3=atr_ratio
        mapping = {}
        for c in range(n_clusters):
            sl, adx, vs200, atr = cents[c]
            if atr > 1.0:
                mapping[c] = "High Volatility"
            elif sl > 0 and vs200 > 0:
                mapping[c] = "Bull Trend"
            elif sl < 0 and vs200 < 0:
                mapping[c] = "Bear Trend"
            else:
                mapping[c] = "Range / Chop"
        labeled = [mapping[c] for c in labels]
    except Exception:
        # the fallback is full-length (row order) - return it directly;
        # running it through the alignment below would mislabel warm-up
        # rows (same trap the HMM path already avoids).
        return _fallback_labels(df)

    # align the fitted labels (only non-warm-up rows) back onto the full
    # input index
    idx = feats.index
    out = ["Range / Chop"] * len(df)
    for i, lab in zip(idx, labeled, strict=True):
        out[df.index.get_loc(i)] = lab
    return out


def detect_regime_cluster(df: pd.DataFrame, n_clusters: int = 4) -> pd.DataFrame:
    """Add a ``regime_cluster`` column alongside the deterministic one."""
    df = df.copy()
    df["regime_cluster"] = cluster_regime_labels(df, n_clusters)
    return df


# ---------------------------------------------------------------------------
# HMM-based regime (institutional spec #1: "HMM or clustering")
# ---------------------------------------------------------------------------


def hmm_regime_labels(df: pd.DataFrame, n_states: int = 4, seed: int = 42) -> List[str]:
    """
    Four-state Gaussian HMM over the standardized regime features
    (slope / ADX / vs200 / ATR-ratio), labeled Bull / Bear / Range /
    High-Vol by the emission means of each state.

    Gracefully falls back to the deterministic rule-based labels when
    ``hmmlearn`` is not installed or the fit fails (same contract as the
    KMeans clustering variant).
    """
    feats = _regime_features(df).dropna()
    if len(feats) < max(120, n_states * 30):
        return _fallback_labels(df)
    try:
        from hmmlearn.hmm import GaussianHMM

        model = GaussianHMM(
            n_components=n_states,
            covariance_type="full",
            n_iter=200,
            tol=1e-3,
            random_state=seed,
        )
        model.fit(feats.values)
        hidden = model.predict(feats.values)
        means = model.means_  # columns: slope, adx, vs200, atr_ratio
        mapping = {}
        atr_col = means[:, 3]
        atr_bar = float(atr_col.mean() + 1.0 * atr_col.std())
        for s in range(n_states):
            sl, _adx, vs200, atr = means[s]
            if atr > atr_bar:
                mapping[s] = "High Volatility"
            elif sl > 0 and vs200 > 0:
                mapping[s] = "Bull Trend"
            elif sl < 0 and vs200 < 0:
                mapping[s] = "Bear Trend"
            else:
                mapping[s] = "Range / Chop"
        labeled = [mapping[s] for s in hidden]
    except Exception:
        # hmmlearn can fail numerically (singular covariance on
        # degenerate synthetic frames, etc.) - degrade, never crash.
        return _fallback_labels(df)

    # align the fitted labels (only non-warm-up rows) back onto the full
    # input index; the fallback above already returns full-length labels
    # in row order, so it must NOT go through this alignment.
    idx = feats.index
    out = ["Range / Chop"] * len(df)
    for i, lab in zip(idx, labeled, strict=True):
        out[df.index.get_loc(i)] = lab
    return out


def detect_regime_hmm(df: pd.DataFrame, n_states: int = 4) -> pd.DataFrame:
    """Add a ``regime_hmm`` column alongside the deterministic / cluster
    regime labels (opt-in: the fit costs seconds, so universe scans keep
    it off)."""
    df = df.copy()
    df["regime_hmm"] = hmm_regime_labels(df, n_states)
    return df


# ---------------------------------------------------------------------------
# Multi-timeframe regime table (institutional spec #1: M/W/D)
# ---------------------------------------------------------------------------


def multi_timeframe_regime(
    df: pd.DataFrame,
    include: Optional[List[str]] = None,
    use_cluster: bool = True,
    use_hmm: bool = False,
) -> Dict:
    """Regime summary on Daily / Weekly / Monthly resamples.

    Resamples OHLC to weekly (W) and monthly (M), recomputes indicators
    and regime on each, and returns a ``rows`` table of
    {timeframe, regime, adx, vs200_pct, slope, confidence} plus the
    ``consensus`` across timeframes. Bars shorter than the indicator
    warm-up are skipped gracefully.

    ``use_cluster`` uses the KMeans label when present; ``use_hmm`` uses
    the 4-state HMM label (fit on each timeframe's own features, so the
    weekly/monthly views get their own hidden states). HMM is opt-in
    because the fit costs seconds; it degrades to the deterministic label
    when ``hmmlearn`` is absent or the fit fails.
    """
    from src.features.indicators import add_all_indicators

    if not include:
        include = ["D", "W", "M"]
    if "regime" not in df.columns:
        df = detect_regime(df)
    if use_hmm and "regime_hmm" not in df.columns:
        df = detect_regime_hmm(df)
    # pandas >= 2.2 renamed the monthly offset alias M -> ME
    _FREQ = {"D": "D", "W": "W", "M": "ME", "W-FRI": "W-FRI"}

    rows: List[Dict] = []
    current = df.copy()

    def _resample(freq: str) -> pd.DataFrame:
        if len(current) < 30:
            return current
        r = (
            current.resample(_FREQ.get(freq, freq), label="right")
            .agg(
                {
                    "open": "first",
                    "high": "max",
                    "low": "min",
                    "close": "last",
                    **({"volume": "sum"} if "volume" in current.columns else {}),
                }
            )
            .dropna(subset=["close"])
        )
        return r

    def _pick_regime(frame: pd.DataFrame) -> str:
        if use_hmm and "regime_hmm" in frame.columns:
            v = frame["regime_hmm"].iloc[-1]
            if isinstance(v, str) and v:
                return v
        if use_cluster and "regime_cluster" in frame.columns:
            return frame["regime_cluster"].iloc[-1]
        return frame["regime"].iloc[-1]

    for tf in include:
        frame = _resample(tf) if tf != "D" else current
        if frame.empty:
            continue
        try:
            if "rsi_14" not in frame.columns:
                frame = add_all_indicators(frame)
            if "regime" not in frame.columns:
                frame = detect_regime(frame)
            if use_hmm and "regime_hmm" not in frame.columns:
                frame = detect_regime_hmm(frame)
        except Exception:
            continue
        last = frame.iloc[-1]
        rows.append(
            {
                "timeframe": tf,
                "regime": _pick_regime(frame),
                "adx": round(float(last.get("adx", 0) or 0), 2),
                "vs_200_pct": round(
                    float(last.get("price_vs_200sma", 0) or 0) * 100, 2
                ),
                "slope_20": round(float(last.get("slope_20", 0) or 0), 5),
                # normalized per-bar slope (% of price) - cross-asset comparable
                "slope_pct_20": round(float(last.get("slope_pct_20", 0) or 0), 4),
                "confidence": round(float(last.get("regime_confidence", 0) or 0), 2),
            }
        )

    # consensus: most common regime across timeframes
    consensus = (
        "Range / Chop"
        if not rows
        else max(
            {r["regime"] for r in rows},
            key=lambda rg: sum(r["regime"] == rg for r in rows),
        )
    )
    return {"rows": rows, "consensus": consensus}


def get_current_regime_summary_mtf(
    df: pd.DataFrame, use_cluster: bool = True, use_hmm: bool = False
) -> Dict:
    """The current-regime summary extended with a clustering label, the HMM
    label (opt-in) and the multi-timeframe (D/W/M) view - used by the
    institutional report."""
    base = get_current_regime_summary(df)
    if "regime_cluster" not in df.columns and use_cluster:
        df = detect_regime_cluster(df)
    base["regime_cluster"] = (
        df["regime_cluster"].iloc[-1]
        if "regime_cluster" in df.columns
        else base["regime"]
    )
    if use_hmm:
        if "regime_hmm" not in df.columns:
            df = detect_regime_hmm(df)
        base["regime_hmm"] = (
            df["regime_hmm"].iloc[-1] if "regime_hmm" in df.columns else base["regime"]
        )
    mtf = multi_timeframe_regime(df, use_cluster=use_cluster, use_hmm=use_hmm)
    base["mtf"] = mtf["rows"]
    base["mtf_consensus"] = mtf["consensus"]
    return base


if __name__ == "__main__":
    print("NexusQuant Regime Detection module ready.")
