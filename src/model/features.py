"""
NexusQuant - Ensemble model features & labels.

Turns the causal indicator / regime / dip-signal stack into a training
matrix for the Buy-the-Dip probability model.

* **Features** - all causal (indicators use past/current data, the dip
  components come from ``src.backtest.signals.dip_signal_series`` which is
  strictly no-lookahead). Missing "setup" features (entry-zone / invalidation
  gaps) are filled with 0 = "no setup".
* **Label (default)** - asymmetric triple-barrier: entry = close, risk =
  ATR(14), stop = close - ``stop_mult`` * risk, target = close +
  ``target_mult`` * risk. Looking at the next ``horizon`` bars: stop touched
  first -> 0, target touched first -> 1, neither -> censored (labeled by the
  forward close, or dropped with ``drop_censored``).
* **Label (meta)** - for confirmed dips only, the *actual* trade outcome from
  the engine geometry (limit entry, swing-low stop, resistance target), which
  matches how the model is deployed. See ``build_meta_labels``.
"""

from __future__ import annotations

from typing import Dict, Optional

import numpy as np
import pandas as pd

from src.backtest.signals import dip_signal_series

# Ordered feature list used for both training and live prediction.
FEATURE_COLUMNS = [
    # Momentum
    "rsi_14",
    "macd_hist",
    "macd",
    "bb_pct_b",
    # Trend
    "adx",
    "plus_di",
    "minus_di",
    "vs_sma200_pct",
    "sma20_gap",
    "sma50_gap",
    # Volatility
    "atr_pct",
    "volatility_20",
    "bb_width",
    # Returns
    "ret_1",
    "ret_5",
    "ret_10",
    # Volume
    "relative_volume",
    # Regime
    "slope_20",
    "regime_confidence",
    # Time structure + interactions (calendar effects on FX/gold)
    "day_of_week",
    "month",
    "month_end",
    "mid_month",
    "vol_x_momentum",
    "vol_x_trend",
    "adx_x_slope",
    # Multi-timeframe (H4, causal same-day alignment; 0 when absent)
    "h4_mom5",
    "h4_mom20",
    "h4_vol_ratio",
    "h4_vs_sma200",
    # Cross-asset risk/gold proxies (1-day lag; 0 when absent)
    "risk_mom5",
    "risk_mom20",
    "gold_mom5",
    "gold_mom20",
    # Positioning (COT; 0 when absent)
    "cot_percentile",
    # Dip components (causal series)
    "dip_score",
    "bias_score",
    "above_sma200",
    "ma_stack",
    "trend",
    "pullback",
    "cooled",
    "at_support",
    "fib_zone",
    "trigger",
    "dip_depth_pct",
    "entry_lo_gap",
    "invalidation_gap",
    # Macro (top-down USD / risk / rates context, aligned with a 1-day lag)
    "dxy_score",
    "vix_score",
    "tnx_score",
    "macro_bias",
    "dxy_mom20",
    # Instrument identity (categorical, lets the model separate vol regimes)
    "symbol",
]

# Columns that must be treated as categorical by the model (not numeric).
CATEGORICAL_FEATURES = ["symbol"]

# Macro columns (used to decide when macro context must be loaded).
MACRO_FEATURES = ["dxy_score", "vix_score", "tnx_score", "macro_bias", "dxy_mom20"]

# Multi-timeframe (H4) feature columns - default to 0 when H4 data is absent.
MTF_FEATURES = ["h4_mom5", "h4_mom20", "h4_vol_ratio", "h4_vs_sma200"]

# Cross-asset risk / gold proxy feature columns - default to 0 when absent.
CROSS_FEATURES = ["risk_mom5", "risk_mom20", "gold_mom5", "gold_mom20"]

# Calendar-structure feature columns.
TIME_FEATURES = ["day_of_week", "month", "month_end", "mid_month"]

DEFAULT_HORIZON = 10
DEFAULT_META_HORIZON = 20
# Asymmetric barriers mirror the real trade: stop further away than the
# target, so wins/losses are not a symmetric coin-flip race.
DEFAULT_STOP_MULT = 1.25
DEFAULT_TARGET_MULT = 0.75
# Down-weight labels that only resolved on the forward close (censored by
# the time barrier): they are noise-heavy relative to clean barrier touches.
CENSORED_WEIGHT = 0.3


def build_features(
    df: pd.DataFrame,
    signal: Optional[pd.DataFrame] = None,
    symbol: Optional[str] = None,
    macro: Optional[pd.DataFrame] = None,
    mtf: Optional[pd.DataFrame] = None,
    cross: Optional[Dict[str, pd.DataFrame]] = None,
    cot: Optional[Dict[str, pd.DataFrame]] = None,
) -> pd.DataFrame:
    """
    Build the causal feature matrix from a prepared OHLCV frame.

    ``df`` must contain the indicator columns from ``add_all_indicators`` and
    the regime columns from ``detect_regime``. The dip components are derived
    from ``dip_signal_series`` (computed here if not provided).

    ``macro`` is the daily macro frame (raw DXY/VIX/TNX closes, or precomputed
    factor scores with a ``dxy`` column appended) - see ``src.macro.overlay``.
    Macro features are aligned with a 1-day lag (strictly causal) and default
    to 0 (neutral) when no macro source exists.

    ``mtf`` is the H4 OHLCV frame for the same symbol (indicators optional) -
    resampled to daily with the *last H4 close on each day*, so all H4
    information for a day is known by that day's close (causal for next-day
    labels). Defaults to neutral 0 when absent.

    ``cross`` is a dict of daily proxy frames (``risk`` = AUDJPY/NZDJPY
    average, ``gold`` = XAUUSD) used for risk-on/off context; aligned with a
    1-day lag (strictly causal). Defaults to 0 when absent.

    ``cot`` is a dict of COT percentile Series keyed by currency (see
    ``src.model.cot``); aligned with a 1-day lag. Defaults to 0 when absent.

    ``symbol`` becomes the categorical instrument-identity feature.
    """
    close = df["close"].astype(float)
    atr = df["atr_14"].astype(float)

    if signal is None:
        signal = dip_signal_series(df)
    signal = _normalize_signal(signal)

    feats = pd.DataFrame(index=df.index)
    feats["rsi_14"] = df["rsi_14"]
    feats["macd_hist"] = df["macd_hist"]
    feats["macd"] = df["macd"]
    feats["bb_pct_b"] = df["bb_pct_b"]
    feats["adx"] = df["adx"]
    feats["plus_di"] = df["plus_di"]
    feats["minus_di"] = df["minus_di"]
    feats["vs_sma200_pct"] = (close / df["sma_200"] - 1.0) * 100
    feats["sma20_gap"] = (close / df["sma_20"] - 1.0) * 100
    feats["sma50_gap"] = (close / df["sma_50"] - 1.0) * 100
    feats["atr_pct"] = atr / close * 100
    feats["volatility_20"] = df["volatility_20"]
    feats["bb_width"] = df["bb_width"]
    feats["ret_1"] = df["returns"] * 100
    feats["ret_5"] = close.pct_change(5) * 100
    feats["ret_10"] = close.pct_change(10) * 100
    # Volume degrades to neutral 1.0 (average) when missing OR all-NaN -
    # MT5 FX tick volume is often 0, which makes relative_volume 0/0 = NaN
    # on every bar. An all-NaN column must never poison the feature matrix
    # (same graceful-degradation contract as macro/MTF/cross/COT context).
    feats["relative_volume"] = _neutral_fill(df.get("relative_volume"), df.index, 1.0)
    feats["slope_20"] = df.get("slope_20", 0.0)
    feats["regime_confidence"] = df.get("regime_confidence", 0.0)

    # Calendar structure + interactions (all causal, purely time-derived).
    idx = df.index
    feats["day_of_week"] = idx.dayofweek.astype(float)
    feats["month"] = idx.month.astype(float)
    feats["month_end"] = (idx.day >= 28).astype(float)
    feats["mid_month"] = ((idx.day >= 13) & (idx.day <= 21)).astype(float)
    feats["vol_x_momentum"] = feats["volatility_20"] * feats["ret_5"]
    feats["vol_x_trend"] = feats["volatility_20"] * feats["vs_sma200_pct"]
    feats["adx_x_slope"] = feats["adx"] * feats["slope_20"]

    # Multi-timeframe / cross-asset / positioning context (0 when absent).
    for col, s in mtf_features(df, mtf).items():
        feats[col] = s
    for col, s in cross_asset_features(df, cross).items():
        feats[col] = s
    for col, s in cot_features(df, cot, symbol).items():
        feats[col] = s
    feats["rsi_14"] = df["rsi_14"]
    feats["macd_hist"] = df["macd_hist"]
    feats["macd"] = df["macd"]
    feats["bb_pct_b"] = df["bb_pct_b"]
    feats["adx"] = df["adx"]
    feats["plus_di"] = df["plus_di"]
    feats["minus_di"] = df["minus_di"]
    feats["vs_sma200_pct"] = (close / df["sma_200"] - 1.0) * 100
    feats["sma20_gap"] = (close / df["sma_20"] - 1.0) * 100
    feats["sma50_gap"] = (close / df["sma_50"] - 1.0) * 100
    feats["atr_pct"] = atr / close * 100
    feats["volatility_20"] = df["volatility_20"]
    feats["bb_width"] = df["bb_width"]
    feats["ret_1"] = df["returns"] * 100
    feats["ret_5"] = close.pct_change(5) * 100
    feats["ret_10"] = close.pct_change(10) * 100
    feats["relative_volume"] = _neutral_fill(df.get("relative_volume"), df.index, 1.0)
    feats["slope_20"] = df.get("slope_20", 0.0)
    feats["regime_confidence"] = df.get("regime_confidence", 0.0)

    # Dip components (0 = no setup for the gap features)
    feats["dip_score"] = signal["score"]
    feats["bias_score"] = signal["bias_score"]
    for col in [
        "above_sma200",
        "ma_stack",
        "trend",
        "pullback",
        "cooled",
        "at_support",
        "fib_zone",
        "trigger",
    ]:
        feats[col] = signal[col].astype(float)
    feats["dip_depth_pct"] = signal["dip_depth_pct"]
    feats["entry_lo_gap"] = (close / signal["entry_lo"] - 1.0).fillna(0.0)
    feats["invalidation_gap"] = (close / signal["invalidation"] - 1.0).fillna(0.0)

    # Macro + instrument identity (macro defaults to neutral zeros).
    m = _macro_features(macro, df.index, symbol)
    for col in MACRO_FEATURES:
        feats[col] = m[col]
    # Categorical dtype: LightGBM rejects raw strings, but treats pandas
    # `category` columns listed in categorical_feature as true categoricals.
    # Unseen symbols at predict time become NaN (handled as missing).
    feats["symbol"] = pd.Series(
        symbol if symbol else "", index=df.index, dtype="category"
    )

    return feats[FEATURE_COLUMNS]


def _normalize_signal(signal: pd.DataFrame) -> pd.DataFrame:
    """
    Map a rally-side causal signal onto the dip feature slots.

    ``build_features`` reads dip-named columns (above_sma200, pullback,
    cooled, at_support, dip_depth_pct) as generic "setup" features. The
    rally signal (``src.backtest.signals.rally_signal_series``) carries the
    same semantic slots under mirror names - rename them so one feature
    builder serves both directions (long and short models share the same
    FEATURE_COLUMNS).
    """
    if signal is None or "below_sma200" not in signal.columns:
        return signal
    s = signal.rename(
        columns={
            "below_sma200": "above_sma200",
            "rally": "pullback",
            "stretched": "cooled",
            "at_resistance": "at_support",
            "rally_depth_pct": "dip_depth_pct",
        }
    ).copy()
    # The direction flag flips sign: "below" is the bearish analogue of
    # "above". The feature then reads correctly for a short-side model.
    s["above_sma200"] = ~s["above_sma200"]
    return s


def _neutral_fill(
    series: Optional[pd.Series], index: pd.DatetimeIndex, default: float
) -> pd.Series:
    """Column -> neutral ``default`` when missing or all-NaN (graceful
    degradation for volume/context features, e.g. FX tick volume of 0)."""
    if series is None:
        return pd.Series(default, index=index)
    out = pd.to_numeric(series, errors="coerce")
    if out.notna().any():
        return out.fillna(default)
    return pd.Series(default, index=index)


def _macro_features(
    macro: Optional[pd.DataFrame], index: pd.DatetimeIndex, symbol: Optional[str]
) -> pd.DataFrame:
    """
    Causal macro features for ``index`` (aligned with a 1-day lag).

    Returns zeros (neutral) when no macro source is available so the rest of
    the pipeline never has to branch on macro availability.
    """
    n = len(index)
    out = pd.DataFrame(
        {
            "dxy_score": np.zeros(n),
            "vix_score": np.zeros(n),
            "tnx_score": np.zeros(n),
            "macro_bias": np.zeros(n),
            "dxy_mom20": np.zeros(n),
        },
        index=index,
    )
    if macro is None or macro.empty:
        return out

    from src.macro.overlay import align_scores, macro_bias_series

    aligned = align_scores(macro, index, shift_days=1)
    for col in ("dxy_score", "vix_score", "tnx_score"):
        if col in macro.columns and col in aligned:
            out[col] = pd.to_numeric(aligned[col], errors="coerce").fillna(0.0)

    if "dxy" in macro.columns:
        dxy = pd.to_numeric(macro["dxy"], errors="coerce")
        mom = (dxy / dxy.shift(20) - 1.0) * 100
        mom.index = mom.index + pd.Timedelta(days=1)
        out["dxy_mom20"] = mom.reindex(index, method="ffill").fillna(0.0)

    if symbol:
        out["macro_bias"] = (
            macro_bias_series(symbol, aligned).reindex(index).fillna(0.0)
        )
    return out


def mtf_features(df: pd.DataFrame, h4: Optional[pd.DataFrame] = None) -> pd.DataFrame:
    """
    Causal H4 -> D1 features (neutral 0 when no H4 data is provided).

    H4 bars are resampled to daily by taking the *last H4 close of each day*,
    so everything known about day D from H4 is available at D's close - which
    is causal for a label defined on bars after D. Features: 5/20-day H4
    momentum, H4 volatility relative to D1 volatility, and distance from the
    H4 200-day average.
    """
    out = pd.DataFrame({c: np.zeros(len(df)) for c in MTF_FEATURES}, index=df.index)
    if h4 is None or h4.empty or len(h4) < 30:
        return out
    try:
        c = pd.to_numeric(h4["close"], errors="coerce").dropna()
        daily = c.resample("D").last().dropna()
        if len(daily) < 30:
            return out
        mom5 = (daily / daily.shift(5) - 1.0) * 100
        mom20 = (daily / daily.shift(20) - 1.0) * 100
        h4_vol = daily.pct_change().rolling(20).std() * 100
        sma200 = daily.rolling(200).mean()
        trend = (daily / sma200 - 1.0) * 100
        merged = pd.concat([mom5, mom20, h4_vol, trend], axis=1)
        merged.columns = MTF_FEATURES[:4]
        # Values known as of each D1 bar's close (same-day H4 info only).
        aligned = merged.reindex(df.index).ffill()
        for col in MTF_FEATURES[:4]:
            out[col] = aligned[col].fillna(0.0)
        d1_vol = (
            pd.to_numeric(df.get("returns"), errors="coerce").rolling(20).std() * 100
        )
        ratio = aligned["h4_vol_ratio"].rename("x") / d1_vol.reindex(
            df.index
        ).ffill().rename("y")
        out["h4_vol_ratio"] = (
            ratio.replace([np.inf, -np.inf], np.nan).fillna(0.0).clip(-5.0, 5.0)
        )
    except Exception:
        return out  # graceful degradation to neutral zeros
    return out


def cross_asset_features(
    df: pd.DataFrame,
    cross: Optional[Dict[str, pd.DataFrame]] = None,
) -> pd.DataFrame:
    """
    Cross-asset risk/gold context features (neutral 0 when absent).

    ``cross`` is a dict of daily frames: ``risk`` (AUDJPY and NZDJPY average
    momentum - the classic risk-on/off pair) and ``gold`` (XAUUSD). Each
    series is aligned with a 1-day lag so only *yesterday's* cross-asset state
    reaches a bar - strictly causal.
    """
    out = pd.DataFrame({c: np.zeros(len(df)) for c in CROSS_FEATURES}, index=df.index)
    if not cross:
        return out
    idx = df.index
    try:
        # Risk proxy: average 5/20-day momentum across the AUDJPY/NZDJPY legs.
        risk = cross.get("risk")
        if isinstance(risk, pd.DataFrame) and len(risk.columns):
            closes = [
                pd.to_numeric(risk[c], errors="coerce").dropna()
                for c in risk.columns
                if c != "symbol"
            ]
            if closes:
                m5 = sum(
                    ((c / c.shift(5) - 1.0) * 100).reindex(idx).shift(1) for c in closes
                ) / len(closes)
                m20 = sum(
                    ((c / c.shift(20) - 1.0) * 100).reindex(idx).shift(1)
                    for c in closes
                ) / len(closes)
                out["risk_mom5"] = m5.fillna(0.0)
                out["risk_mom20"] = m20.fillna(0.0)

        # Gold proxy: XAUUSD momentum (single daily close Series or frame).
        gold = cross.get("gold")
        if gold is not None:
            if isinstance(gold, pd.DataFrame):
                if len(gold.columns) == 0:
                    gold = None
                elif "close" in gold.columns:
                    gold = gold["close"]
                else:
                    gold = gold[gold.columns[0]]
            if gold is not None:
                c = pd.to_numeric(gold, errors="coerce").dropna()
                out["gold_mom5"] = (
                    ((c / c.shift(5) - 1.0) * 100).reindex(idx).shift(1).fillna(0.0)
                )
                out["gold_mom20"] = (
                    ((c / c.shift(20) - 1.0) * 100).reindex(idx).shift(1).fillna(0.0)
                )
    except Exception:
        return out
    return out


def cot_features(
    df: pd.DataFrame,
    cot: Optional[Dict[str, pd.DataFrame]] = None,
    symbol: Optional[str] = None,
) -> pd.DataFrame:
    """
    COT positioning percentile feature (neutral 50 when unavailable).

    ``cot`` maps a market key to a daily Series of net-position percentiles
    (0-100); the symbol is resolved through ``SYMBOL_MARKET`` first (metals /
    energy / indices, e.g. US500 -> SP500), then through ``CCY_MAP`` for FX
    symbols. Aligned with a 1-day lag (strictly causal). Missing data reads
    the neutral 50 baseline consistently (both "no COT source at all" and
    "no row yet"). See ``src.model.cot`` for the loader.
    """
    out = pd.DataFrame({"cot_percentile": np.full(len(df), 50.0)}, index=df.index)
    if not cot or not symbol:
        return out
    from src.model.cot import CCY_MAP, SYMBOL_MARKET

    # Instrument symbols (metals / energy / indices) resolve through
    # SYMBOL_MARKET first, FX symbols through CCY_MAP.
    key = SYMBOL_MARKET.get(symbol) or CCY_MAP.get(symbol)
    if key is None or key not in cot:
        return out
    s = pd.to_numeric(cot[key], errors="coerce")
    out["cot_percentile"] = (
        s.reindex(df.index).shift(1).ffill().fillna(50.0).clip(0.0, 100.0)
    )
    return out


def build_labels(
    df: pd.DataFrame,
    horizon: int = DEFAULT_HORIZON,
    stop_mult: float = DEFAULT_STOP_MULT,
    target_mult: float = DEFAULT_TARGET_MULT,
    drop_censored: bool = False,
) -> pd.DataFrame:
    """
    Triple-barrier 1R win/loss label for every bar.

    Entry = close, risk = ATR(14); stop = close - ``stop_mult`` * ATR,
    target = close + ``target_mult`` * ATR. Looking at the next ``horizon``
    bars: stop touched first -> 0, target touched first -> 1, neither ->
    censored (labeled by the forward close unless ``drop_censored``, in which
    case those rows get no label and are excluded from training).

    The asymmetric barriers (``stop_mult`` > ``target_mult``) mirror the real
    trade geometry (stop further away than the target) so the outcome is not
    a symmetric first-touch coin flip. Rows in the last ``horizon`` bars (no
    future info) get no label.
    """
    close = df["close"].astype(float)
    high = df["high"].astype(float)
    low = df["low"].astype(float)
    atr = df["atr_14"].astype(float)

    # Forward extreme of the NEXT `horizon` bars (excludes the current bar).
    fwd_lo = low.shift(-1).iloc[::-1].rolling(horizon, min_periods=1).min().iloc[::-1]
    fwd_hi = high.shift(-1).iloc[::-1].rolling(horizon, min_periods=1).max().iloc[::-1]

    stop = close - stop_mult * atr
    target = close + target_mult * atr

    stop_hit = fwd_lo <= stop
    target_hit = (~stop_hit) & (fwd_hi >= target)  # stop checked first

    close_fwd = close.shift(-horizon)
    label = pd.Series(np.nan, index=close.index)
    label[stop_hit] = 0.0
    label[target_hit] = 1.0
    censored = (~stop_hit) & (~target_hit) & close_fwd.notna()
    label[censored] = (close_fwd[censored] > close[censored]).astype(float)
    if drop_censored:
        label[censored] = np.nan
    label[close_fwd.isna() & label.isna()] = np.nan

    return pd.DataFrame({"label": label, "censored": censored}, index=df.index)


def build_meta_labels(
    df: pd.DataFrame,
    signal: pd.DataFrame,
    horizon: int = DEFAULT_META_HORIZON,
    entry_valid_bars: int = 3,
    rr_fallback: float = 2.0,
) -> pd.Series:
    """
    Meta-label: the *actual* trade outcome for each confirmed-dip bar.

    Mirrors the backtest engine (``src.backtest.engine.run_backtest``):
    limit entry at ``entry_lo`` (valid ``entry_valid_bars``), stop at
    ``invalidation``, target at the nearest resistance above entry or an
    R:R fallback, stop checked before target, time-stop after ``horizon``
    bars. Only confirmed dips that actually fill get a label; a bar that
    never fills or is censored by the time barrier gets NaN (dropped).

    This is the deployment objective - "does THIS setup work?" - instead of
    the generic symmetric 1R move.
    """
    close = df["close"].astype(float)
    high = df["high"].astype(float)
    low = df["low"].astype(float)
    confirmed = signal["confirmed"].values
    entry_lo = signal["entry_lo"].values
    invalidation = signal["invalidation"].values
    resistance = signal["resistance"].values
    close_v = close.values
    high_v = high.values
    low_v = low.values
    n = len(df)

    label = pd.Series(np.nan, index=df.index)
    for i in range(n - 1):
        if not confirmed[i]:
            continue
        e = entry_lo[i]
        s = invalidation[i]
        if np.isnan(e) or np.isnan(s):
            continue
        # The engine only places a limit when the zone is below the market.
        if not (s < e < close_v[i]):
            continue
        r = resistance[i]
        tgt = r if (not np.isnan(r)) and r > e else e + rr_fallback * (e - s)

        fill = -1
        for j in range(i + 1, min(i + 1 + entry_valid_bars, n)):
            if low_v[j] <= e:
                fill = j
                break
        if fill < 0:
            continue

        outcome = np.nan
        for k in range(fill, min(fill + horizon, n)):
            if low_v[k] <= s:  # stop first (conservative)
                outcome = 0.0
                break
            if high_v[k] >= tgt:
                outcome = 1.0
                break
        label.iloc[i] = outcome
    return label


def build_dataset(
    df: pd.DataFrame,
    signal: Optional[pd.DataFrame] = None,
    horizon: int = DEFAULT_HORIZON,
    symbol: Optional[str] = None,
    macro: Optional[pd.DataFrame] = None,
    mtf: Optional[pd.DataFrame] = None,
    cross: Optional[Dict[str, pd.DataFrame]] = None,
    cot: Optional[Dict[str, pd.DataFrame]] = None,
    meta: bool = False,
    drop_censored: bool = False,
    label_kwargs: Optional[Dict] = None,
    side: str = "long",
) -> Dict:
    """
    Full dataset for one symbol: features + labels + metadata.

    ``meta=True`` labels confirmed setups with the real engine outcome
    (``build_meta_labels`` for long, ``make_meta_labels_short`` for
    ``side="short"``) instead of the symmetric 1R label. ``side="short"``
    uses the rally signal series so the model trains on the Sell-the-Rally
    objective (P(short 1R win)). Returns a dict with ``X``, ``y``,
    ``weight``, ``symbol``, ``time``, ``confirmed``, ``censored``.
    """
    if signal is None:
        if side == "short":
            from src.backtest.signals import rally_signal_series

            signal = rally_signal_series(df)
        else:
            signal = dip_signal_series(df)
    X = build_features(
        df, signal, symbol=symbol, macro=macro, mtf=mtf, cross=cross, cot=cot
    )

    if meta:
        if side == "short":
            labels = pd.DataFrame(
                {"label": make_meta_labels_short(df, signal, **(label_kwargs or {}))},
                index=df.index,
            )
        else:
            labels = pd.DataFrame(
                {"label": build_meta_labels(df, signal, **(label_kwargs or {}))},
                index=df.index,
            )
        censored = pd.Series(False, index=df.index)
    else:
        if side == "short":
            labels = build_labels_short(
                df, horizon, drop_censored=drop_censored, **(label_kwargs or {})
            )
        else:
            labels = build_labels(
                df, horizon, drop_censored=drop_censored, **(label_kwargs or {})
            )
        censored = labels["censored"]

    keep = X.notna().all(axis=1) & labels["label"].notna()
    keep_idx = X.index[keep]

    weight = pd.Series(1.0, index=keep_idx)
    if not meta and not drop_censored:
        weight = weight * np.where(censored.loc[keep_idx], CENSORED_WEIGHT, 1.0)

    return {
        "X": X[keep],
        "y": labels.loc[keep, "label"].astype(float),
        "weight": weight,
        "symbol": pd.Series(symbol, index=keep_idx),
        "time": keep_idx,
        "confirmed": signal.loc[keep, "confirmed"].values,
        "censored": censored.loc[keep].values,
    }


def build_labels_short(
    df: pd.DataFrame,
    horizon: int = DEFAULT_HORIZON,
    stop_mult: float = DEFAULT_STOP_MULT,
    target_mult: float = DEFAULT_TARGET_MULT,
    drop_censored: bool = False,
) -> pd.DataFrame:
    """
    Triple-barrier 1R win/loss label for a SHORT trade on every bar.

    Entry = close, risk = ATR(14); stop = close + ``stop_mult`` * ATR
    (ABOVE), target = close - ``target_mult`` * ATR (below). Stop touched
    first -> 0, target touched first -> 1; censored rows use the forward
    close (close_fwd < close is a win) unless dropped.
    """
    close = df["close"].astype(float)
    high = df["high"].astype(float)
    low = df["low"].astype(float)
    atr = df["atr_14"].astype(float)

    fwd_lo = low.shift(-1).iloc[::-1].rolling(horizon, min_periods=1).min().iloc[::-1]
    fwd_hi = high.shift(-1).iloc[::-1].rolling(horizon, min_periods=1).max().iloc[::-1]

    stop = close + stop_mult * atr
    target = close - target_mult * atr

    stop_hit = fwd_hi >= stop
    target_hit = (~stop_hit) & (fwd_lo <= target)  # stop checked first

    close_fwd = close.shift(-horizon)
    label = pd.Series(np.nan, index=close.index)
    label[stop_hit] = 0.0
    label[target_hit] = 1.0
    censored = (~stop_hit) & (~target_hit) & close_fwd.notna()
    label[censored] = (close_fwd[censored] < close[censored]).astype(float)
    if drop_censored:
        label[censored] = np.nan
    label[close_fwd.isna() & label.isna()] = np.nan

    return pd.DataFrame({"label": label, "censored": censored}, index=df.index)


def make_meta_labels_short(
    df: pd.DataFrame,
    signal: pd.DataFrame,
    horizon: int = DEFAULT_META_HORIZON,
    entry_valid_bars: int = 3,
    rr_fallback: float = 2.0,
) -> pd.Series:
    """
    Short meta-label: the *actual* trade outcome for each confirmed-rally
    bar (mirror of ``build_meta_labels``).

    Limit SHORT at ``entry_hi`` (price rises into the resistance zone),
    stop at ``invalidation`` (above the swing high), target below at the
    rally signal's ``target`` or an R:R fallback. Stop checked before
    target; time-stop after ``horizon`` bars; censored/unfilled -> NaN.
    """
    close = df["close"].astype(float)
    high = df["high"].astype(float)
    low = df["low"].astype(float)
    confirmed = signal["confirmed"].values
    entry_hi = signal["entry_hi"].values
    invalidation = signal["invalidation"].values
    target_s = signal["target"].values
    close_v = close.values
    high_v = high.values
    low_v = low.values
    n = len(df)

    label = pd.Series(np.nan, index=df.index)
    for i in range(n - 1):
        if not confirmed[i]:
            continue
        e = entry_hi[i]
        s = invalidation[i]
        if np.isnan(e) or np.isnan(s):
            continue
        # The engine only places the short when the zone is above the market.
        if not (close_v[i] < e < s):
            continue
        t = target_s[i]
        tgt = t if (not np.isnan(t)) and t < e else e - rr_fallback * (s - e)

        fill = -1
        for j in range(i + 1, min(i + 1 + entry_valid_bars, n)):
            if high_v[j] >= e:
                fill = j
                break
        if fill < 0:
            continue

        outcome = np.nan
        for k in range(fill, min(fill + horizon, n)):
            if high_v[k] >= s:  # stop first (conservative)
                outcome = 0.0
                break
            if low_v[k] <= tgt:
                outcome = 1.0
                break
        label.iloc[i] = outcome
    return label


def build_dataset_dual(
    df: pd.DataFrame,
    signal_long: Optional[pd.DataFrame] = None,
    signal_short: Optional[pd.DataFrame] = None,
    horizon: int = DEFAULT_HORIZON,
    symbol: Optional[str] = None,
    macro: Optional[pd.DataFrame] = None,
    mtf: Optional[pd.DataFrame] = None,
    cross: Optional[Dict[str, pd.DataFrame]] = None,
    cot: Optional[Dict[str, pd.DataFrame]] = None,
    meta: bool = True,
    drop_censored: bool = False,
    label_kwargs: Optional[Dict] = None,
) -> Dict:
    """
    Dual-side dataset: the same features built twice (long- and short-side
    setup components) with BOTH label sets, for the dual-head model
    (Gap 2: P(long) + P(short)).

    Returns ``{X, X_short, y_long, y_short, confirmed_long,
    confirmed_short, weight_long, weight_short, symbol, time}``.
    """
    from src.backtest.signals import dip_signal_series, rally_signal_series

    if signal_long is None:
        signal_long = dip_signal_series(df)
    if signal_short is None:
        signal_short = rally_signal_series(df)

    X_long = build_features(
        df, signal_long, symbol=symbol, macro=macro, mtf=mtf, cross=cross, cot=cot
    )
    X_short = build_features(
        df, signal_short, symbol=symbol, macro=macro, mtf=mtf, cross=cross, cot=cot
    )

    if meta:
        y_long = build_meta_labels(df, signal_long, **(label_kwargs or {}))
        y_short = make_meta_labels_short(df, signal_short, **(label_kwargs or {}))
        censored_l = pd.Series(False, index=df.index)
        censored_s = pd.Series(False, index=df.index)
    else:
        lb = build_labels(
            df, horizon, drop_censored=drop_censored, **(label_kwargs or {})
        )
        sb = build_labels_short(
            df, horizon, drop_censored=drop_censored, **(label_kwargs or {})
        )
        y_long, censored_l = lb["label"], lb["censored"]
        y_short, censored_s = sb["label"], sb["censored"]

    keep_l = X_long.notna().all(axis=1) & y_long.notna()
    keep_s = X_short.notna().all(axis=1) & y_short.notna()
    idx_l, idx_s = X_long.index[keep_l], X_short.index[keep_s]

    w_l = pd.Series(1.0, index=idx_l)
    w_s = pd.Series(1.0, index=idx_s)
    if not meta and not drop_censored:
        w_l = w_l * np.where(censored_l.loc[idx_l], CENSORED_WEIGHT, 1.0)
        w_s = w_s * np.where(censored_s.loc[idx_s], CENSORED_WEIGHT, 1.0)

    return {
        "X": X_long[keep_l],
        "X_short": X_short[keep_s],
        "y_long": y_long.loc[keep_l].astype(float),
        "y_short": y_short.loc[keep_s].astype(float),
        "weight_long": w_l,
        "weight_short": w_s,
        "symbol": pd.Series(symbol, index=idx_l),
        "time": idx_l,
        "confirmed_long": signal_long.loc[keep_l, "confirmed"].values,
        "confirmed_short": signal_short.loc[keep_s, "confirmed"].values,
        "censored_long": censored_l.loc[keep_l].values,
        "censored_short": censored_s.loc[keep_s].values,
    }
