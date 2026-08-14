"""
NexusQuant — Stage-4 Alpha Discovery & Signal-Architecture Forensics (research CLI).

Stage-3 proved calibration is fixable but that the trading architecture does
not yet demonstrate economic alpha: LONG_BUY_DIP is negative at 10-20 bars,
short families are ~zero/negative at fixed horizons, and the raw models are
near-flat rankers (slope 0.01-0.025). Stage-4 therefore strips the trading
architecture away and asks the prior question:

    Does the existing feature / information set contain any statistically
    defensible directional alpha?

Analyses (each strictly pre-``CUTOFF`` — the final chronological period is
reserved untouched for the eventual single-shot test):

    1. unconditional_edge  — forward returns (1/3/5/10/20/40 bars,
       ATR-normalized) over EVERY bar with no gates at all, per symbol, per
       regime, plus the FLAT base rates P(+/- k ATR) that frame the
       LONG/SHORT/FLAT decision (spec Phases 2, 5, 8).
    2. baselines           — always-long / always-short / random / 200-SMA /
       momentum / mean-reversion / Donchian-breakout net of 0.05R costs
       (spec Phase 17).
    3. label_audit         — class balance, entropy and threshold base rates
       per label scheme (spec Phase 11).
    4. cross_sectional     — daily tercile long-short momentum spread,
       turnover and cost-adjusted spread (spec Phase 9).
    5. basket_alpha        — currency-basket-relative momentum (JPY/USD/EUR/
       GBP/CHF) vs forward returns (spec Phase 10).
    6. horizon_curve       — IC decay/reversal of momentum, mean-reversion
       and trend proxies at 1..40 bars (spec Phases 7, 12).
    7. feature_alpha_audit — univariate rank-IC / hit-rate / stability across
       time halves and regimes for every model feature, with Benjamini-
       Hochberg FDR across all feature x horizon tests (spec Phase 3, 14).
    8. grouped_ablation    — unified LightGBM (chronological split) with
       group-removal ablations (spec Phase 4).
    9. walk_forward        — compact 3-fold purged walk-forward of the
       unified signal (preliminary; the frozen study follows discovery).

Everything is causal (features at t vs realized returns t..t+h) and
deterministic. Results are printed and written to
``data/validation/stage4_results.json``.

Usage:
    python -m src.analysis.stage4 --symbols EURUSD,USDCAD --uncond --baselines
    python -m src.analysis.stage4 --symbols <list> --all
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

# The final chronological period is reserved untouched for the eventual
# single-shot final test; NO Stage-4 number uses it.
CUTOFF = "2025-06-01"
MODEL_SPLIT = "2022-01-01"
HORIZONS = (1, 3, 5, 10, 20, 40)
COST_R = 0.05  # round-trip cost in R
STOP_MULT = 1.25  # 1R in price = 1.25 x ATR
COST_ATR = COST_R * STOP_MULT  # 0.0625 ATR per round trip
FDR_Q = 0.05

# Feature groups covering FEATURE_COLUMNS (minus the "symbol" identity column).
GROUPS: Dict[str, List[str]] = {
    "momentum": ["rsi_14", "macd_hist", "macd", "bb_pct_b", "ret_1", "ret_5", "ret_10"],
    "trend": ["adx", "plus_di", "minus_di", "vs_sma200_pct", "sma20_gap", "sma50_gap"],
    "volatility": ["atr_pct", "volatility_20", "bb_width"],
    "volume": ["relative_volume"],
    "regime": ["slope_20", "regime_confidence"],
    "time": ["day_of_week", "month", "month_end", "mid_month"],
    "interactions": ["vol_x_momentum", "vol_x_trend", "adx_x_slope"],
    "mtf": ["h4_mom5", "h4_mom20", "h4_vol_ratio", "h4_vs_sma200"],
    "cross": ["risk_mom5", "risk_mom20", "gold_mom5", "gold_mom20"],
    "cot": ["cot_percentile"],
    "dip": [
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
    ],
    "macro": ["dxy_score", "vix_score", "tnx_score", "macro_bias", "dxy_mom20"],
}
ALL_FEATURES = [f for g in GROUPS.values() for f in g]


# ---------------------------------------------------------------------------
# Frame helpers
# ---------------------------------------------------------------------------


def _load_base(
    symbol: str,
    data_dir: str = "data/raw",
    group: str = "full_fx",
    timeframe: str = "D1",
) -> Optional[pd.DataFrame]:
    path = (
        Path(data_dir) / group / f"{symbol}_{timeframe.upper()}.parquet"
        if group
        else Path(data_dir) / f"{symbol}_{timeframe.upper()}.parquet"
    )
    if not path.exists():
        return None
    try:
        df = clean_data(load_data(path, symbol=symbol))
        df = add_all_indicators(df)
        df = detect_regime(df)
        return df
    except Exception:
        return None


def _fwd_atr(df: pd.DataFrame, horizons: tuple = HORIZONS) -> pd.DataFrame:
    """Add fwd{h}_atr = (close_{t+h}/close_t - 1) / atr_14 (causal eval)."""
    close = df["close"].astype(float)
    atr = df["atr_14"].astype(float).replace(0.0, np.nan)
    for h in horizons:
        df[f"fwd{h}_atr"] = (close.shift(-h) / close - 1.0) / atr
    return df


def _pre_cutoff(df: pd.DataFrame, cutoff: str = CUTOFF) -> np.ndarray:
    return df.index < pd.Timestamp(cutoff)


def _ensure_proxies(df: pd.DataFrame) -> pd.DataFrame:
    """Add the return / gap / slope proxies that are model features rather
    than indicator columns (ret_5, ret_10, vs_sma200_pct, sma20_gap,
    slope20), computed causally from OHLC when absent."""
    close = df["close"].astype(float)
    if "ret_5" not in df.columns:
        df["ret_5"] = close.pct_change(5)
    if "ret_10" not in df.columns:
        df["ret_10"] = close.pct_change(10)
    if "vs_sma200_pct" not in df.columns and "sma_200" in df.columns:
        df["vs_sma200_pct"] = (close / df["sma_200"] - 1.0) * 100
    if "sma20_gap" not in df.columns and "sma_20" in df.columns:
        df["sma20_gap"] = (close / df["sma_20"] - 1.0) * 100
    if "slope20" not in df.columns:
        df["slope20"] = close.pct_change(20)
    return df


def _pooled(
    frames: Dict[str, pd.DataFrame],
    horizons: tuple = HORIZONS,
    cutoff: str = CUTOFF,
) -> Dict:
    """Stack forward returns across symbols (pre-cutoff only)."""
    parts = []
    for sym, df in frames.items():
        m = _pre_cutoff(df, cutoff)
        keep = [f"fwd{h}_atr" for h in horizons]
        keep += [c for c in df.columns if c.startswith("s_")]
        d = df.loc[m, keep].copy()
        d["symbol"] = sym
        if "regime" in df.columns:
            d["regime"] = df.loc[m, "regime"].astype(str)
        parts.append(d)
    return pd.concat(parts, ignore_index=True)


# ---------------------------------------------------------------------------
# Phase 2/5/8: unconditional forward-return edge + FLAT base rates
# ---------------------------------------------------------------------------


def unconditional_edge(
    symbols: List[str],
    data_dir: str = "data/raw",
    group: str = "full_fx",
    timeframe: str = "D1",
    cutoff: str = CUTOFF,
) -> Dict:
    frames = {}
    for sym in symbols:
        df = _load_base(sym, data_dir, group, timeframe)
        if df is None or len(df) < 300:
            continue
        frames[sym] = _fwd_atr(df)

    out: Dict = {}
    # Pooled per-horizon stats.
    pooled = _pooled(frames, cutoff=cutoff)
    rows = []
    for h in HORIZONS:
        col = f"fwd{h}_atr"
        v = pooled[col].dropna()
        n = len(v)
        mean = float(v.mean())
        t = (
            mean / (float(v.std(ddof=1)) / np.sqrt(n))
            if n > 1 and v.std(ddof=1) > 0
            else None
        )
        rows.append(
            {
                "horizon": h,
                "n": n,
                "mean_atr": round(mean, 4),
                "median_atr": round(float(v.median()), 4),
                "hit_rate_up": round(float((v > 0).mean()), 4),
                "t": round(t, 2) if t is not None else None,
            }
        )
    out["pooled_horizons"] = rows

    # FLAT base rates: P(fwd10 > k ATR), P(fwd10 < -k ATR), P(both).
    thresholds = (0.25, 0.5, 1.0, 1.5)
    base = []
    for k in thresholds:
        v = pooled["fwd10_atr"].dropna()
        up = float((v > k).mean())
        dn = float((v < -k).mean())
        base.append(
            {
                "threshold_atr": k,
                "p_up": round(up, 4),
                "p_down": round(dn, 4),
                "p_flat": round(1.0 - up - dn, 4),
            }
        )
    out["flat_base_rates_h10"] = base

    # Per-regime pooled stats (h=10).
    regime_rows = []
    for reg, d in pooled.groupby("regime"):
        v = d["fwd10_atr"].dropna()
        if len(v) < 50:
            continue
        mean = float(v.mean())
        t = (
            mean / (float(v.std(ddof=1)) / np.sqrt(len(v)))
            if v.std(ddof=1) > 0
            else None
        )
        regime_rows.append(
            {
                "regime": str(reg),
                "n": len(v),
                "mean_atr": round(mean, 4),
                "hit_rate_up": round(float((v > 0).mean()), 4),
                "t": round(t, 2) if t is not None else None,
            }
        )
    out["per_regime_h10"] = regime_rows

    # Per-symbol h=10/h=20.
    sym_rows = []
    for sym, df in frames.items():
        m = _pre_cutoff(df, cutoff)
        row = {"symbol": sym}
        for h in (10, 20):
            v = df.loc[m, f"fwd{h}_atr"].dropna()
            mean = float(v.mean()) if len(v) else float("nan")
            t = (
                mean / (float(v.std(ddof=1)) / np.sqrt(len(v)))
                if len(v) > 1 and v.std(ddof=1) > 0
                else None
            )
            row[f"h{h}_mean"] = round(mean, 4)
            row[f"h{h}_hit"] = round(float((v > 0).mean()), 4) if len(v) else None
            row[f"h{h}_t"] = round(t, 2) if t is not None else None
        sym_rows.append(row)
    out["per_symbol"] = sym_rows
    out["n_symbols"] = len(frames)
    return out


# ---------------------------------------------------------------------------
# Phase 17: simple baselines, net of costs
# ---------------------------------------------------------------------------


def baselines(
    symbols: List[str],
    data_dir: str = "data/raw",
    group: str = "full_fx",
    timeframe: str = "D1",
    cutoff: str = CUTOFF,
    horizon: int = 10,
) -> Dict:
    frames = {}
    for sym in symbols:
        df = _load_base(sym, data_dir, group, timeframe)
        if df is None or len(df) < 300:
            continue
        df = _fwd_atr(df, horizons=(horizon,))
        df = _ensure_proxies(df)
        close = df["close"].astype(float)
        ret5 = df["ret_5"]
        sma200 = df["sma_200"] if "sma_200" in df.columns else close.rolling(200).mean()
        hi20 = df["high"].rolling(20).max().shift(1)
        lo20 = df["low"].rolling(20).min().shift(1)
        df["s_long"] = 1.0
        df["s_short"] = -1.0
        df["s_rand"] = np.where(np.arange(len(df)) % 2 == 0, 1.0, -1.0)
        df["s_sma200"] = np.where(close > sma200, 1.0, -1.0)
        df["s_mom5"] = np.where(ret5 > 0, 1.0, -1.0)
        df["s_mr5"] = np.where(ret5 < 0, 1.0, -1.0)  # fade 5-bar moves
        df["s_donch"] = np.where(
            close > hi20, 1.0, np.where(close < lo20, -1.0, np.nan)
        )
        frames[sym] = df

    pooled = _pooled(frames, horizons=(horizon,), cutoff=cutoff)
    strategies = [
        "s_long",
        "s_short",
        "s_rand",
        "s_sma200",
        "s_mom5",
        "s_mr5",
        "s_donch",
    ]
    col = f"fwd{horizon}_atr"
    out = []
    for s in strategies:
        d = pooled[[col, s]].dropna()
        if len(d) < 100:
            continue
        # Strategy P&L = sign(s) x forward return (long +, short -, signal
        # strategies flip sign per bar). Gross is the mean signed return.
        signed = d[s].astype(float) * d[col].astype(float)
        gross = float(signed.mean())
        net = gross - COST_ATR
        out.append(
            {
                "strategy": s,
                "n": len(d),
                "gross_atr": round(gross, 4),
                "net_atr": round(net, 4),
                "hit_rate": round(float((signed > 0).mean()), 4),
            }
        )
    return {"horizon": horizon, "cost_atr": COST_ATR, "strategies": out}


# ---------------------------------------------------------------------------
# Phase 11: label audit (class balance / entropy / threshold base rates)
# ---------------------------------------------------------------------------


def label_audit(
    symbols: List[str],
    data_dir: str = "data/raw",
    group: str = "full_fx",
    timeframe: str = "D1",
    cutoff: str = CUTOFF,
) -> Dict:
    frames = {}
    for sym in symbols:
        df = _load_base(sym, data_dir, group, timeframe)
        if df is None or len(df) < 300:
            continue
        frames[sym] = _fwd_atr(df)
    pooled = _pooled(frames, cutoff=cutoff)
    out = []
    for h in HORIZONS:
        v = pooled[f"fwd{h}_atr"].dropna()
        up = float((v > 0).mean())
        ent = -(up * np.log2(up) + (1 - up) * np.log2(1 - up)) if 0 < up < 1 else 0.0
        out.append(
            {
                "horizon": h,
                "n": len(v),
                "p_up": round(up, 4),
                "entropy_bits": round(float(ent), 4),
                "max_accuracy_always_long": round(max(up, 1 - up), 4),
            }
        )
    return out


# ---------------------------------------------------------------------------
# Phase 9: cross-sectional (daily tercile long-short momentum)
# ---------------------------------------------------------------------------


def cross_sectional(
    symbols: List[str],
    data_dir: str = "data/raw",
    group: str = "full_fx",
    timeframe: str = "D1",
    cutoff: str = CUTOFF,
) -> Dict:
    frames = {}
    for sym in symbols:
        df = _load_base(sym, data_dir, group, timeframe)
        if df is None or len(df) < 300:
            continue
        frames[sym] = _ensure_proxies(_fwd_atr(df, horizons=(10,)))
    # Build daily panel: date x symbol {ret5, ret10, fwd10_atr}
    panel = {}
    for sym, df in frames.items():
        m = _pre_cutoff(df, cutoff)
        for col in ("ret_5", "ret_10", "fwd10_atr"):
            if col not in df.columns:
                continue
            s = df.loc[m, col]
            for date, val in s.items():
                panel.setdefault(date, {})[sym] = panel.setdefault(date, {}).get(
                    sym, {}
                )
                panel[date][sym][col] = val
    ic_rows = []
    spread_rows = []
    turnover_rows = []
    for date in sorted(panel):
        d = panel[date]
        syms = [
            s
            for s in d
            if d[s].get("ret_10") == d[s].get("ret_10")
            and d[s].get("fwd10_atr") == d[s].get("fwd10_atr")
        ]
        if len(syms) < 8:
            continue
        # Cross-sectional rank IC of 10-bar momentum vs fwd10 ATR return.
        pred = np.array([d[s]["ret_10"] for s in syms])
        fwd = np.array([d[s]["fwd10_atr"] for s in syms])
        if pred.std() == 0 or fwd.std() == 0:
            continue
        ic = float(
            np.corrcoef(pd.Series(pred).rank().values, pd.Series(fwd).rank().values)[
                0, 1
            ]
        )
        ic_rows.append(ic)
        # Terciles by ret_10.
        r = pd.Series({s: d[s]["ret_10"] for s in syms}).rank(pct=True)
        top = [s for s in syms if r[s] >= 2.0 / 3.0]
        bot = [s for s in syms if r[s] <= 1.0 / 3.0]
        if top and bot:
            spread_rows.append(
                float(np.mean([d[s]["fwd10_atr"] for s in top]))
                - float(np.mean([d[s]["fwd10_atr"] for s in bot]))
            )
        turnover_rows.append(len(syms))
    ic = np.array(ic_rows)
    spr = np.array(spread_rows)
    if len(ic) < 30:
        return {"n_days": len(ic), "note": "insufficient cross-section"}
    mean_ic = float(ic.mean())
    t_ic = (
        mean_ic / (float(ic.std(ddof=1)) / np.sqrt(len(ic)))
        if ic.std(ddof=1) > 0
        else None
    )
    mean_spr = float(spr.mean())
    # Turnover: proxy = fraction of symbols changing tercile membership day to day.
    return {
        "n_days": len(ic),
        "cross_sectional_ic": round(mean_ic, 4),
        "ic_t": round(t_ic, 2) if t_ic is not None else None,
        "tercile_spread_gross_atr": round(mean_spr, 4),
        "tercile_spread_net_atr": round(mean_spr - 2 * COST_ATR, 4),
        "avg_symbols_per_day": round(float(np.mean(turnover_rows)), 1),
    }


# ---------------------------------------------------------------------------
# Phase 10: currency-basket relative momentum
# ---------------------------------------------------------------------------


BASKETS = {
    "JPY": ["EURJPY", "GBPJPY", "AUDJPY", "NZDJPY", "CADJPY", "CHFJPY"],
    "USD": ["EURUSD", "GBPUSD", "AUDUSD", "NZDUSD", "USDCAD", "USDCHF"],
    "EUR": ["EURUSD", "EURJPY", "EURGBP"],
    "GBP": ["GBPUSD", "GBPJPY", "GBPEUR"],
    "CHF": ["USDCHF", "CHFJPY", "EURCHF"],
}


def basket_alpha(
    symbols: List[str],
    data_dir: str = "data/raw",
    group: str = "full_fx",
    timeframe: str = "D1",
    cutoff: str = CUTOFF,
) -> Dict:
    frames = {}
    for sym in symbols:
        df = _load_base(sym, data_dir, group, timeframe)
        if df is None or len(df) < 300:
            continue
        frames[sym] = _ensure_proxies(_fwd_atr(df, horizons=(10,)))
    # Date x symbol matrices of 5-bar momentum (ATR units) and fwd10 ATR.
    P, F = {}, {}
    for sym, df in frames.items():
        m = _pre_cutoff(df, cutoff)
        atr = df["atr_14"].astype(float).replace(0.0, np.nan)
        P[sym] = (df.loc[m, "ret_5"] / atr.loc[m]).rename(sym)
        F[sym] = df.loc[m, "fwd10_atr"].rename(sym)
    P = pd.DataFrame(P)
    F = pd.DataFrame(F)
    out_rows = []
    for basket, members in BASKETS.items():
        present = [s for s in members if s in P.columns]
        if len(present) < 3:
            continue
        basket_mean = P[present].mean(axis=1, skipna=True)
        ics = []
        for sym in present:
            rel = (P[sym] - basket_mean).dropna()
            fwd = F[sym].reindex(rel.index)
            d = pd.DataFrame({"rel": rel, "fwd": fwd}).dropna()
            if len(d) < 200 or d["rel"].std() == 0 or d["fwd"].std() == 0:
                continue
            ic = float(d["rel"].rank().corr(d["fwd"].rank()))
            ics.append(ic)
        if len(ics) >= 3:
            arr = np.array(ics)
            out_rows.append(
                {
                    "basket": basket,
                    "members": present,
                    "mean_rank_ic": round(float(arr.mean()), 4),
                    "n_assets": len(ics),
                }
            )
    return {"baskets": out_rows}


# ---------------------------------------------------------------------------
# Phases 7/12: horizon curve / signal decay for representative features
# ---------------------------------------------------------------------------


def horizon_curve(
    symbols: List[str],
    data_dir: str = "data/raw",
    group: str = "full_fx",
    timeframe: str = "D1",
    cutoff: str = CUTOFF,
) -> Dict:
    frames = {}
    for sym in symbols:
        df = _load_base(sym, data_dir, group, timeframe)
        if df is None or len(df) < 300:
            continue
        frames[sym] = _ensure_proxies(_fwd_atr(df))
    proxies = ["ret_5", "rsi_14", "bb_pct_b", "vs_sma200_pct", "sma20_gap", "slope20"]
    rows = []
    for feat in proxies:
        parts = []
        for sym, df in frames.items():
            m = _pre_cutoff(df, cutoff)
            if feat not in df.columns:
                continue
            d = df.loc[m, [feat] + [f"fwd{h}_atr" for h in HORIZONS]].copy()
            d["symbol"] = sym
            parts.append(d)
        if not parts:
            continue
        all_d = pd.concat(parts, ignore_index=True)
        row: Dict = {"feature": feat}
        for h in HORIZONS:
            d = all_d[[feat, f"fwd{h}_atr"]].dropna()
            if len(d) < 500 or d[feat].std() == 0 or d[f"fwd{h}_atr"].std() == 0:
                row[f"h{h}"] = None
                continue
            row[f"h{h}"] = round(float(d[feat].rank().corr(d[f"fwd{h}_atr"].rank())), 4)
        rows.append(row)
    return rows


# ---------------------------------------------------------------------------
# Multiple-testing: Benjamini-Hochberg FDR (no statsmodels dependency)
# ---------------------------------------------------------------------------


def _bh_fdr(pvals: np.ndarray, q: float = FDR_Q) -> np.ndarray:
    p = np.asarray(pvals, dtype=float)
    n = len(p)
    order = np.argsort(p)
    ranked = p[order]
    thresh = np.arange(1, n + 1) * q / n
    below = ranked <= thresh
    if not below.any():
        return np.zeros(n, dtype=bool)
    k = int(np.max(np.where(below)[0]))
    sig = np.zeros(n, dtype=bool)
    sig[order[: k + 1]] = True
    return sig


def _spearman_p(r: float, n: int) -> float:
    from scipy.stats import t as tdist

    if n <= 2 or abs(r) >= 1.0:
        return 0.0
    t = r * np.sqrt((n - 2) / (1 - r * r))
    return float(2 * tdist.sf(abs(t), n - 2))


# ---------------------------------------------------------------------------
# Phase 3: feature-level alpha audit (needs the ML feature matrix)
# ---------------------------------------------------------------------------


def _ml_frames(
    symbols: List[str], data_dir: str = "data/raw", group: str = "full_fx"
) -> Dict[str, pd.DataFrame]:
    """Feature matrix per symbol (dip-context shared feature set, causal)."""
    from src.model.features import build_features

    out = {}
    try:
        from src.macro.overlay import macro_for_model

        macro = macro_for_model(data_dir)
    except Exception:
        macro = None
    try:
        from src.model.model import _cross_proxies, _mtf_frame

        cross = _cross_proxies(group, data_dir)
        mtf_map = {s: _mtf_frame(s, group, data_dir) for s in symbols}
    except Exception:
        cross, mtf_map = {}, {}
    cot = None
    try:
        from src.model.cot import load_cot

        cot = load_cot(f"{data_dir}/cot")
    except Exception:
        pass
    for sym in symbols:
        df = _load_base(sym, data_dir, group)
        if df is None or len(df) < 300:
            continue
        try:
            X = build_features(
                df,
                signal=None,
                symbol=sym,
                macro=macro,
                mtf=mtf_map.get(sym),
                cross=cross,
                cot=cot,
            )
            # Always take build_features' version of the ML columns: the
            # indicator frame's `relative_volume` etc. can be broken (NaN)
            # while the feature builder computes them correctly.
            drop = [c for c in ALL_FEATURES if c in df.columns]
            df = pd.concat([df.drop(columns=drop), X[ALL_FEATURES]], axis=1)
        except Exception:
            continue
        out[sym] = df
    return out


def feature_alpha_audit(
    symbols: List[str],
    data_dir: str = "data/raw",
    group: str = "full_fx",
    cutoff: str = CUTOFF,
) -> Dict:
    frames = _ml_frames(symbols, data_dir, group)
    for sym in frames:
        frames[sym] = _fwd_atr(frames[sym], horizons=(5, 10, 20))
    # Pooled frame carrying the feature columns (unlike _pooled, which is
    # restricted to forward-return columns for the market-wide analyses).
    parts = []
    for sym, df in frames.items():
        m = _pre_cutoff(df, cutoff)
        keep = ALL_FEATURES + [f"fwd{h}_atr" for h in (5, 10, 20)]
        d = df.loc[m, keep].copy()
        d["symbol"] = sym
        if "regime" in df.columns:
            d["regime"] = df.loc[m, "regime"].astype(str)
        parts.append(d)
    pooled = pd.concat(parts, ignore_index=True)

    rows = []
    for feat in ALL_FEATURES:
        if feat not in pooled.columns:
            continue
        for h in (5, 10, 20):
            d = pooled[[feat, f"fwd{h}_atr"]].dropna()
            if len(d) < 1000 or d[feat].std() == 0 or d[f"fwd{h}_atr"].std() == 0:
                continue
            rank_ic = float(d[feat].rank().corr(d[f"fwd{h}_atr"].rank()))
            n = len(d)
            p = _spearman_p(rank_ic, n)
            # Directional hit-rate spread: P(up | feature > median) - P(up).
            med = d[feat].median()
            hi = d[d[feat] > med]
            base_up = float((d[f"fwd{h}_atr"] > 0).mean())
            cond_up = float((hi[f"fwd{h}_atr"] > 0).mean()) if len(hi) > 50 else base_up
            # Stability: rank-IC in first vs second chronological half.
            half = len(d) // 2
            ic1 = float(
                d.iloc[:half][feat].rank().corr(d.iloc[:half][f"fwd{h}_atr"].rank())
            )
            ic2 = float(
                d.iloc[half:][feat].rank().corr(d.iloc[half:][f"fwd{h}_atr"].rank())
            )
            rows.append(
                {
                    "feature": feat,
                    "horizon": h,
                    "n": n,
                    "rank_ic": round(rank_ic, 4),
                    "p": p,
                    "hit_spread": round(cond_up - base_up, 4),
                    "ic_first_half": round(ic1, 4) if ic1 == ic1 else None,
                    "ic_second_half": round(ic2, 4) if ic2 == ic2 else None,
                }
            )

    # FDR across all feature x horizon tests.
    pvals = np.array([r["p"] for r in rows])
    sig = _bh_fdr(pvals, FDR_Q)
    for r, s in zip(rows, sig, strict=True):
        r["fdr_sig"] = bool(s)
    n_sig = int(sig.sum())
    # Regime-conditional rank-IC at h=10 for the strongest features.
    regime_ic = []
    top_feats = sorted(
        [r for r in rows if r["horizon"] == 10], key=lambda r: -abs(r["rank_ic"])
    )[:8]
    for r in top_feats:
        feat = r["feature"]
        for reg in ("Bull Trend", "Bear Trend", "Range / Chop"):
            d = pooled[pooled["regime"] == reg][[feat, "fwd10_atr"]].dropna()
            if len(d) < 300 or d[feat].std() == 0 or d["fwd10_atr"].std() == 0:
                continue
            regime_ic.append(
                {
                    "feature": feat,
                    "regime": reg,
                    "rank_ic": round(
                        float(d[feat].rank().corr(d["fwd10_atr"].rank())), 4
                    ),
                }
            )
    return {
        "tests": len(rows),
        "fdr_q": FDR_Q,
        "n_fdr_significant": n_sig,
        "significant": sorted(
            [r for r in rows if r["fdr_sig"]],
            key=lambda r: -abs(r["rank_ic"]),
        ),
        "regime_ic_top": regime_ic,
    }


# ---------------------------------------------------------------------------
# Phases 4/15: unified-model grouped ablation + compact walk-forward
# ---------------------------------------------------------------------------


def _train_eval(
    X_tr: pd.DataFrame,
    y_tr: np.ndarray,
    X_te: pd.DataFrame,
    y_te: np.ndarray,
    n_estimators: int = 250,
) -> Dict:
    from lightgbm import LGBMClassifier
    from sklearn.metrics import roc_auc_score

    model = LGBMClassifier(
        n_estimators=n_estimators,
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
    model.fit(X_tr, y_tr)
    p = model.predict_proba(X_te)[:, 1]
    auc = roc_auc_score(y_te, p)
    brier = float(np.mean((p - y_te) ** 2))
    # rank IC of prob vs realized fwd10 ATR return.
    rank_ic = float(pd.Series(p).rank().corr(pd.Series(y_te).rank()))
    return {
        "auc": round(auc, 4),
        "brier": round(brier, 4),
        "rank_ic": round(rank_ic, 4),
    }


def grouped_ablation(
    symbols: List[str],
    data_dir: str = "data/raw",
    group: str = "full_fx",
    cutoff: str = CUTOFF,
    split: str = MODEL_SPLIT,
) -> Dict:
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
    if len(data) < 5000:
        return {"note": "insufficient pooled data"}
    X = data[ALL_FEATURES].copy()
    y = data["y"].values
    dates = pd.to_datetime(data["date"])
    split_ts = pd.Timestamp(split)
    tr_mask = dates < split_ts
    if tr_mask.sum() < 3000 or (~tr_mask).sum() < 2000:
        return {"note": "split too unbalanced"}
    X_tr, X_te = X[tr_mask], X[~tr_mask]
    y_tr, y_te = y[tr_mask], y[~tr_mask]

    out: Dict = {
        "n_train": int(tr_mask.sum()),
        "n_test": int((~tr_mask).sum()),
        "ablations": {},
    }
    out["baseline"] = _train_eval(X_tr, y_tr, X_te, y_te)
    for gname, feats in GROUPS.items():
        drop = [f for f in feats if f in X.columns]
        keep = [f for f in X.columns if f not in drop]
        Xa_tr, Xa_te = X_tr[keep], X_te[keep]
        res = _train_eval(Xa_tr, y_tr, Xa_te, y_te)
        res["dropped"] = drop
        out["ablations"][gname] = res
    return out


def walk_forward(
    symbols: List[str],
    data_dir: str = "data/raw",
    group: str = "full_fx",
    cutoff: str = CUTOFF,
) -> Dict:
    """Compact 3-fold purged walk-forward of the unified signal (preliminary).

    Folds over the pre-cutoff sample; embargo = 20 bars after each train end.
    Preliminary only: the frozen study (Phase 15) follows discovery and uses
    the untouched final period for the single-shot test.
    """
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
    if len(data) < 5000:
        return {"note": "insufficient data"}
    X = data[ALL_FEATURES].copy()
    y = data["y"].values
    dates = pd.to_datetime(data["date"])
    folds = [
        ("2015-01-01", "2022-01-01", "2023-06-30"),
        ("2015-01-01", "2023-06-30", "2024-12-31"),
        ("2015-01-01", "2024-12-31", cutoff),
    ]
    results = []
    for i, (train_start, train_end, test_end) in enumerate(folds):
        te_start = pd.Timestamp(train_end) + pd.Timedelta(days=20)  # embargo
        te_end = pd.Timestamp(test_end)
        tr = (dates >= pd.Timestamp(train_start)) & (dates < pd.Timestamp(train_end))
        te = (dates >= te_start) & (dates < te_end)
        if tr.sum() < 3000 or te.sum() < 800:
            continue
        from lightgbm import LGBMClassifier
        from sklearn.metrics import roc_auc_score

        model = LGBMClassifier(
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
        model.fit(X[tr], y[tr])
        X_te = X[te]
        p = model.predict_proba(X_te)[:, 1]
        y_te, p_te = y[te], p
        fwd = data.loc[te, "fwd10_atr"].values
        long_mask = p_te >= 0.55
        short_mask = p_te <= 0.45
        long_r = (
            fwd[long_mask] / STOP_MULT - COST_R if long_mask.sum() else np.array([0.0])
        )
        short_r = (
            (-fwd[short_mask]) / STOP_MULT - COST_R
            if short_mask.sum()
            else np.array([0.0])
        )
        results.append(
            {
                "fold": i + 1,
                "test": f"{te_start.date()}..{te_end.date()}",
                "n_test": int(te.sum()),
                "auc": round(float(roc_auc_score(y_te, p_te)), 4),
                "brier": round(float(np.mean((p_te - y_te) ** 2)), 4),
                "n_long": int(long_mask.sum()),
                "n_short": int(short_mask.sum()),
                "long_exp_r": round(float(long_r.mean()), 4),
                "short_exp_r": round(float(short_r.mean()), 4),
            }
        )
    return {"folds": results}


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main(argv: Optional[List[str]] = None) -> int:
    import argparse

    parser = argparse.ArgumentParser(description="Stage-4 alpha discovery & forensics")
    parser.add_argument("--group", default="full_fx")
    parser.add_argument("--timeframe", default="D1")
    parser.add_argument("--symbols", default=None)
    parser.add_argument("--cutoff", default=CUTOFF)
    parser.add_argument("--uncond", action="store_true")
    parser.add_argument("--baselines", action="store_true")
    parser.add_argument("--labels", action="store_true")
    parser.add_argument("--cross", action="store_true")
    parser.add_argument("--basket", action="store_true")
    parser.add_argument("--curve", action="store_true")
    parser.add_argument("--features", action="store_true")
    parser.add_argument("--ablate", action="store_true")
    parser.add_argument("--wf", action="store_true")
    parser.add_argument("--all", action="store_true")
    args = parser.parse_args(argv)

    if args.symbols:
        symbols = [s.strip() for s in args.symbols.split(",") if s.strip()]
    else:
        from src.analysis.scanner import discover_symbols

        symbols = discover_symbols(
            "data/raw", group=args.group, timeframe=args.timeframe
        )[:16]

    print(
        f"Reserved untouched test period: {args.cutoff}+ (excluded from every number below)"
    )
    results: Dict = {"symbols": symbols, "group": args.group, "cutoff": args.cutoff}

    if args.all or args.uncond:
        print("\n" + "=" * 72)
        print("PHASE 2/5/8 — UNCONDITIONAL FORWARD-RETURN EDGE (no gates at all)")
        print("=" * 72)
        u = unconditional_edge(
            symbols, group=args.group, timeframe=args.timeframe, cutoff=args.cutoff
        )
        print(f"\nPooled horizons ({u['n_symbols']} symbols):")
        print(f"{'h':>4}{'n':>9}{'meanATR':>10}{'medATR':>9}{'hitUp':>8}{'t':>7}")
        for r in u["pooled_horizons"]:
            print(
                f"{r['horizon']:>4}{r['n']:>9,}{r['mean_atr']:>+10.4f}{r['median_atr']:>+9.3f}"
                f"{r['hit_rate_up']:>8.3f}{r['t']:>7.2f}"
            )
        print("\nFLAT base rates (10-bar, ATR thresholds):")
        print(f"{'kATR':>6}{'P(up>k)':>10}{'P(dn<-k)':>10}{'P(flat)':>10}")
        for r in u["flat_base_rates_h10"]:
            print(
                f"{r['threshold_atr']:>6}{r['p_up']:>10.3f}{r['p_down']:>10.3f}{r['p_flat']:>10.3f}"
            )
        print("\nPer regime (h=10):")
        for r in u["per_regime_h10"]:
            print(
                f"  {r['regime']:<16} n={r['n']:>7,} meanATR={r['mean_atr']:+.4f} "
                f"hit={r['hit_rate_up']:.3f} t={r['t']}"
            )
        print("\nPer symbol (h=10):")
        for r in u["per_symbol"]:
            print(
                f"  {r['symbol']:<8} mean={r['h10_mean']:+.4f} hit={r['h10_hit']:.3f} t={r['h10_t']}"
            )
        results["unconditional"] = u

    if args.all or args.baselines:
        print("\n" + "=" * 72)
        print("PHASE 17 — SIMPLE BASELINES net of costs (h=10)")
        print("=" * 72)
        b = baselines(
            symbols, group=args.group, timeframe=args.timeframe, cutoff=args.cutoff
        )
        print(f"{'STRATEGY':<14}{'n':>9}{'grossATR':>10}{'netATR':>10}{'hit':>7}")
        for r in b["strategies"]:
            print(
                f"{r['strategy']:<14}{r['n']:>9,}{r['gross_atr']:>+10.4f}{r['net_atr']:>+10.4f}{r['hit_rate']:>7.3f}"
            )
        results["baselines"] = b

    if args.all or args.labels:
        print("\n" + "=" * 72)
        print("PHASE 11 — LABEL AUDIT (class balance / entropy)")
        print("=" * 72)
        la = label_audit(
            symbols, group=args.group, timeframe=args.timeframe, cutoff=args.cutoff
        )
        print(f"{'h':>4}{'n':>9}{'P(up)':>8}{'entropy':>9}{'alwaysLongAcc':>14}")
        for r in la:
            print(
                f"{r['horizon']:>4}{r['n']:>9,}{r['p_up']:>8.3f}{r['entropy_bits']:>9.3f}{r['max_accuracy_always_long']:>14.3f}"
            )
        results["labels"] = la

    if args.all or args.cross:
        print("\n" + "=" * 72)
        print("PHASE 9 — CROSS-SECTIONAL (daily tercile long-short, 10-bar momentum)")
        print("=" * 72)
        cs = cross_sectional(
            symbols, group=args.group, timeframe=args.timeframe, cutoff=args.cutoff
        )
        for k, v in cs.items():
            print(f"  {k}: {v}")
        results["cross_sectional"] = cs

    if args.all or args.basket:
        print("\n" + "=" * 72)
        print(
            "PHASE 10 — CURRENCY-BASKET RELATIVE MOMENTUM (5-bar rel. momentum vs fwd10)"
        )
        print("=" * 72)
        bk = basket_alpha(
            symbols, group=args.group, timeframe=args.timeframe, cutoff=args.cutoff
        )
        for r in bk.get("baskets", []):
            print(
                f"  {r['basket']:<6} members={','.join(r['members']):<28} meanRankIC={r['mean_rank_ic']:+.4f} (n={r['n_assets']})"
            )
        results["basket"] = bk

    if args.all or args.curve:
        print("\n" + "=" * 72)
        print("PHASE 7/12 — HORIZON CURVE / SIGNAL DECAY (rank IC by horizon)")
        print("=" * 72)
        hc = horizon_curve(
            symbols, group=args.group, timeframe=args.timeframe, cutoff=args.cutoff
        )
        print(f"{'FEATURE':<16}" + "".join(f"{h:>8}" for h in HORIZONS))
        for r in hc:
            cells = "".join(
                f"{r[f'h{h}']:>8.4f}" if r.get(f"h{h}") is not None else f"{'-':>8}"
                for h in HORIZONS
            )
            print(f"{r['feature']:<16}{cells}")
        results["horizon_curve"] = hc

    if args.all or args.features:
        print("\n" + "=" * 72)
        print("PHASE 3 — FEATURE-LEVEL ALPHA AUDIT (univariate, FDR-controlled)")
        print("=" * 72)
        fa = feature_alpha_audit(symbols, group=args.group, cutoff=args.cutoff)
        print(
            f"tests={fa['tests']} fdr_q={fa['fdr_q']} significant={fa['n_fdr_significant']}"
        )
        print(f"{'FEATURE':<20}{'h':>4}{'rankIC':>9}{'hitSpr':>8}{'ic1':>8}{'ic2':>8}")
        for r in fa["significant"]:
            print(
                f"{r['feature']:<20}{r['horizon']:>4}{r['rank_ic']:>+9.4f}{r['hit_spread']:>+8.3f}"
                f"{r['ic_first_half']:>8.3f}{r['ic_second_half']:>8.3f}"
            )
        print("\nRegime-conditional rank IC (h=10, top features):")
        for r in fa["regime_ic_top"]:
            print(f"  {r['feature']:<20} {r['regime']:<14} {r['rank_ic']:+.4f}")
        results["features"] = fa

    if args.all or args.ablate:
        print("\n" + "=" * 72)
        print("PHASE 4 — GROUPED ABLATION (unified LGBM, chronological split)")
        print("=" * 72)
        ab = grouped_ablation(symbols, group=args.group, cutoff=args.cutoff)
        print(f"n_train={ab.get('n_train')} n_test={ab.get('n_test')}")
        base = ab.get("baseline", {})
        print(
            f"baseline: AUC={base.get('auc')} Brier={base.get('brier')} rankIC={base.get('rank_ic')}"
        )
        print(f"{'GROUP':<14}{'AUC':>8}{'Brier':>9}{'rankIC':>9}{'dAUC':>8}")
        for g, r in ab.get("ablations", {}).items():
            d_auc = (r.get("auc", 0) - base.get("auc", 0)) if base else None
            print(
                f"{g:<14}{r.get('auc'):>8}{r.get('brier'):>9}{r.get('rank_ic'):>9}"
                f"{d_auc:>+8.4f}"
            )
        results["ablation"] = ab

    if args.all or args.wf:
        print("\n" + "=" * 72)
        print("PHASE 15 (preliminary) — 3-FOLD PURGED WALK-FORWARD")
        print("=" * 72)
        wf = walk_forward(symbols, group=args.group, cutoff=args.cutoff)
        if wf.get("folds"):
            print(
                f"{'fold':>5}{'test window':<32}{'n':>7}{'AUC':>8}{'Brier':>8}{'nL':>6}{'nS':>6}{'L expR':>9}{'S expR':>9}"
            )
            for r in wf["folds"]:
                print(
                    f"{r['fold']:>5}{r['test']:<32}{r['n_test']:>7,}{r['auc']:>8.4f}{r['brier']:>8.4f}"
                    f"{r['n_long']:>6}{r['n_short']:>6}{r['long_exp_r']:>+9.4f}{r['short_exp_r']:>+9.4f}"
                )
        results["walk_forward"] = wf

    out_dir = Path("data/validation")
    out_dir.mkdir(parents=True, exist_ok=True)
    with open(out_dir / "stage4_results.json", "w") as fh:
        json.dump(results, fh, indent=2, default=str)
    print(f"\nStage-4 results written to {out_dir / 'stage4_results.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
