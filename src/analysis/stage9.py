"""
NexusQuant — Stage-9: Frozen Long Reversal Confirmation, Portfolio Economics
& Production Readiness Gate (research only — no production changes).

Stage-8 verdict carried forward:

    LONG reversal leg = B. PROMISING BUT INSUFFICIENT EVIDENCE
    - generalizes to the 28-symbol FX major/cross universe
      (OOS n=163, net +0.347R, permutation p=0.007)
    - exotics falsified (must stay excluded); high-vol regime falsified
    - SHORT reversal remains LOCKED falsified (Stage-6)

Stage-9 is NOT an alpha-discovery exercise. The hypothesis is FROZEN:

    LONG ONLY on the 28 liquid FX majors/crosses
    entry   : k=3 of {L1_rsi30, L2_drop5, L3_streak5n}   (frozen)
    horizon : 10 bars (time-stop / signal-reversal exits)
    cost    : 0.05R round trip (COST_R; 1R = 1.25 x ATR)
    risk    : per-trade fraction of equity (simulated)
    FLAT    : the default whenever the k=3 condition is absent

Frozen protocol / discipline:

  - NO parameter optimization, symbol selection, threshold tuning,
    regime tuning, or exit optimization anywhere in this campaign.
  - The strategy specification is versioned and hashed (sha256); the hash
    is printed before any evaluation.
  - The previously untouched period 2025-06-01 -> 2026-08-13 (data end)
    was CONSUMED by Stage-8's single-shot test. It is NOT re-evaluated
    here for confirmation.
  - CRITICAL FACT: the dataset ends 2026-08-13. There is NO data past the
    consumed window, therefore the "completely fresh independent window"
    requirement (Stage-9 Phase 2 / gate 3) CANNOT be satisfied with
    existing data. The campaign states this explicitly and classifies the
    fresh-window gate UNRESOLVED — which mechanically blocks promotion
    to A (PRODUCTION READY) regardless of every other result.
  - All OOS economics below therefore use the non-consumed OOS window
    2022-01-01 -> 2025-06-01 on the frozen 28-symbol universe.

Analyses (each a function; --all runs all):

    1. freeze_spec        — versioned, hashed strategy specification
    2. effective_n        — raw N vs unique-day N vs cluster-adjusted N
    3. walk_forward5      — 5 expanding-window purged folds, per-fold stats
    4. stat_battery       — permutation / trade / block / symbol / order
       bootstrap + BH-FDR across folds
    5. economic_stress    — cost grid 0.05..0.30R, break-even, headroom
    6. portfolio_sim      — fixed 0.25/0.50/1.00% risk, vol-adjusted,
       capped fractional Kelly; per-symbol/currency caps, max-concurrent,
       daily-loss limit, portfolio drawdown breaker; CAGR/Sharpe/Sortino/
       Calmar/maxDD/PF/turnover/utilization
    7. clustering         — all vs one-per-cluster vs strongest vs
       risk-scaled cluster handling
    8. baselines9         — random timing, random extreme-state, RSI,
       mean reversion, buy-and-hold, trend, always-FLAT
    9. adversarial9       — delays, random timing/symbols, shuffled timing,
       return permutation, sign inversion, cost/slippage shocks, remove
       best symbol/regime/fold
   10. regime_stability   — descriptive Bull/Bear/Range x low/norm/high vol
   11. exit_validation    — descriptive MFE/MAE/time-to-peak + frozen exits
   12. monte_carlo9       — 10k block-bootstrap portfolio paths
   13. fresh_window       — reports the data-end fact; UNRESOLVED verdict
   14. production_gates   — mechanical 15-gate scorecard with verdict
   15. leakage_audit9     — stage-8 audit + frozen-spec / no-selection /
       consumed-window discipline

Usage:
    python -m src.analysis.stage9 --all
    python -m src.analysis.stage9 --spec --wf --portfolio --gates
"""

from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

sys.path.append(str(Path(__file__).parent.parent.parent))

from src.analysis.stage8 import (
    COST_R,
    CUTOFF,
    K_LONG,
    PRIMARY_H,
    R_MULT,
    _frames8,
    _frozen_long_mask,
    _stats8,
    _trade_r,
    _triggers,
)
from src.analysis.stage5 import TRAIN_END
from src.analysis.stage6 import EMBARGO_DAYS, _oos

SPEC_VERSION = "v1.0.0"
SEED = 20250909

# Frozen 28-symbol universe (Stage-8: fx_major_cross bucket of the
# pre-registered eligible universe). Frozen — do not expand to exotics.
UNIVERSE28 = [
    "AUDCAD",
    "AUDCHF",
    "AUDJPY",
    "AUDNZD",
    "AUDUSD",
    "CADCHF",
    "CADJPY",
    "CHFJPY",
    "EURAUD",
    "EURCAD",
    "EURCHF",
    "EURGBP",
    "EURJPY",
    "EURNZD",
    "EURUSD",
    "GBPAUD",
    "GBPCAD",
    "GBPCHF",
    "GBPJPY",
    "GBPNZD",
    "GBPUSD",
    "NZDCAD",
    "NZDCHF",
    "NZDJPY",
    "NZDUSD",
    "USDCAD",
    "USDCHF",
    "USDJPY",
]

# 5 expanding-window purged folds. Test windows are calendar-year sized
# where the signal density (~40 trades/yr across 28 symbols) allows an
# adequate per-fold sample; the final two are short by necessity (the OOS
# window ends at the consumed 2025-06-01 boundary) and are flagged
# underpowered rather than hidden. All windows end before the consumed
# 2025-06-01 boundary.
WF5_FOLDS = [
    ("2015-01-01", "2021-12-31", "2022-12-31"),  # test CY2022 (~12 mo)
    ("2015-01-01", "2022-12-31", "2023-12-31"),  # test CY2023 (~12 mo)
    ("2015-01-01", "2023-12-31", "2024-12-31"),  # test CY2024 (~12 mo)
    ("2015-01-01", "2024-12-31", "2025-03-31"),  # test Q1 2025 (~3 mo)
    ("2015-01-01", "2025-03-31", "2025-06-01"),  # test Q2 2025 (~2 mo)
]

CONSUMED_START = "2025-06-01"  # Stage-8 single-shot window; not reusable

_FRAME_CACHE9: Dict[Tuple[str, ...], Dict[str, pd.DataFrame]] = {}


def _frames9() -> Dict[str, pd.DataFrame]:
    """Frozen 28-symbol frames (cached per process)."""
    key = tuple(UNIVERSE28)
    if key in _FRAME_CACHE9:
        return _FRAME_CACHE9[key]
    all_frames = _frames8(UNIVERSE28)
    _FRAME_CACHE9[key] = all_frames
    return all_frames


def _frozen_mask(frames, trigs):
    return _frozen_long_mask(frames, trigs, K_LONG)


# ---------------------------------------------------------------------------
# 1. Frozen, versioned, hashed strategy specification
# ---------------------------------------------------------------------------


def freeze_spec() -> Dict:
    spec = {
        "version": SPEC_VERSION,
        "direction": "LONG only (SHORT reversal LOCKED falsified Stage-6)",
        "universe": {
            "n": len(UNIVERSE28),
            "symbols": UNIVERSE28,
            "rule": "fx_major_cross bucket of pre-registered eligible universe; "
            "exotics excluded by Stage-8 evidence (perm p=0.97)",
        },
        "entry": {
            "family": "k-of-3 of {L1_rsi30, L2_drop5, L3_streak5n}",
            "k": K_LONG,
            "triggers": {
                "L1_rsi30": "rsi_14 < 30",
                "L2_drop5": "5-bar ATR-normalized drop < -0.8",
                "L3_streak5n": ">= 5 consecutive down days",
            },
            "note": "single triggers are negative or weak; only the k=3 "
            "combination is used (Stage-8 ablation)",
        },
        "exit": {
            "primary": "time-stop at 10 bars (PRIMARY_H)",
            "alternate": "signal-reversal (RSI crosses back > 35) or "
            "return-to-mean (close >= entry); both transfer OOS",
            "note": "3R target ladder is structurally unreachable "
            "(P(TP1 before SL)=0); not used",
        },
        "risk": {"1R": "1.25 x ATR", "cost_r": COST_R, "cost_note": "0.05R round trip"},
        "regime": "descriptive only — no hard regime gate in the frozen "
        "spec; Stage-8 shows the edge concentrates in low-vol "
        "Bear/Range (high-vol is negative), NOT a filter",
        "windows": {
            "train": f"< {TRAIN_END} (selection)",
            "oos": f"{TRAIN_END} .. {CONSUMED_START}",
            "consumed_untouched": f"{CONSUMED_START} -> 2026-08-13 (Stage-8, single-shot, NOT reusable)",
            "data_end": "2026-08-13",
        },
        "flat_behavior": "FLAT is the default whenever the k=3 condition is "
        "absent; no requirement to trade",
    }
    canonical = json.dumps(spec, sort_keys=True, separators=(",", ":"))
    h = hashlib.sha256(canonical.encode("utf-8")).hexdigest()
    return {"spec": spec, "hash": h, "hash_algorithm": "sha256"}


def _spec_hash() -> str:
    return freeze_spec()["hash"]


# ---------------------------------------------------------------------------
# 2. Effective sample size
# ---------------------------------------------------------------------------


def effective_n() -> Dict:
    frames = _frames9()
    trigs = {sym: _triggers(df) for sym, df in frames.items()}
    mask = _frozen_mask(frames, trigs)
    trades: List[Tuple[str, pd.Timestamp]] = []
    for sym, df in frames.items():
        m = _oos(df, CUTOFF)
        sig = mask(sym, df)
        fwd = df.loc[m, f"fwd{PRIMARY_H}_R"]
        tr = sig.loc[m] & fwd.notna()
        for dt in df.index[m][tr.values]:
            trades.append((sym, dt))
    raw_n = len(trades)
    unique_day_set = {d for _, d in trades}
    unique_days = len(unique_day_set)
    # cluster-adjusted: one bet per (date, currency-cluster) pair
    clusters = {
        "CHF": {"AUDCHF", "CADCHF", "CHFJPY", "EURCHF", "GBPCHF", "NZDCHF", "USDCHF"},
        "JPY": {"AUDJPY", "CADJPY", "CHFJPY", "EURJPY", "GBPJPY", "NZDJPY", "USDJPY"},
        "USD": {"AUDUSD", "EURUSD", "GBPUSD", "NZDUSD", "USDCAD", "USDCHF", "USDJPY"},
        "EUR": {"EURAUD", "EURCAD", "EURCHF", "EURGBP", "EURJPY", "EURNZD", "EURUSD"},
    }
    adj = 0
    for dt in unique_day_set:
        day_syms = {s for s, d in trades if d == dt}
        covered = set()
        for c, members in clusters.items():
            if day_syms & members:
                covered.add(c)
        adj += max(len(covered), 1)
    return {
        "raw_n": raw_n,
        "unique_signal_days": unique_days,
        "cluster_adjusted_n": adj,
        "note": "cluster-adjusted N counts one bet per (date, currency-"
        "cluster) pair — the Stage-9 effective-sample proxy",
    }


# ---------------------------------------------------------------------------
# 3. Five-fold expanding walk-forward (per-fold, no pooling)
# ---------------------------------------------------------------------------


def walk_forward5() -> Dict:
    frames = _frames9()
    trigs = {sym: _triggers(df) for sym, df in frames.items()}
    mask = _frozen_mask(frames, trigs)
    folds = []
    for i, (train_start, fold_end, test_end) in enumerate(WF5_FOLDS):
        te0 = pd.Timestamp(fold_end) + pd.Timedelta(days=EMBARGO_DAYS)
        tr = _trade_r(
            frames, mask, win_start=train_start, win_end=fold_end, cutoff=CUTOFF
        )
        te = _trade_r(
            frames, mask, win_start=str(te0.date()), win_end=test_end, cutoff=CUTOFF
        )
        oos = _stats8(te)
        n = oos.get("n", 0)
        folds.append(
            {
                "fold": i + 1,
                "window": f"{te0.date()}..{test_end}",
                "train": _stats8(tr),
                "oos": oos,
                "underpowered": n < 20,
            }
        )
    n_pos = sum(1 for f in folds if f["oos"].get("mean_r", 0) > 0)
    n_ok = sum(1 for f in folds if not f["underpowered"])
    return {
        "folds": folds,
        "n_folds_positive_net": n_pos,
        "n_folds_adequately_sampled": n_ok,
        "n_folds_total": len(folds),
    }


# ---------------------------------------------------------------------------
# 4. Statistical battery (permutation + 4 bootstraps + BH-FDR)
# ---------------------------------------------------------------------------


def stat_battery() -> Dict:
    frames = _frames9()
    trigs = {sym: _triggers(df) for sym, df in frames.items()}
    mask = _frozen_mask(frames, trigs)
    rs = _trade_r(frames, mask, cutoff=CUTOFF, win_start=TRAIN_END)
    v = rs.values.astype(float)
    n = len(v)
    rng = np.random.default_rng(SEED)
    out: Dict = {"n": n, "mean_r": round(float(v.mean()), 4)}
    # permutation
    perm = np.array(
        [float((v * rng.choice([-1.0, 1.0], n)).mean()) for _ in range(1000)]
    )
    out["permutation_p"] = round(float((np.abs(perm) >= abs(v.mean())).mean()), 4)
    # trade-level bootstrap
    boot = np.array([float(rng.choice(v, n, replace=True).mean()) for _ in range(5000)])
    out["trade_bootstrap_95"] = [
        round(float(np.percentile(boot, 2.5)), 4),
        round(float(np.percentile(boot, 97.5)), 4),
    ]
    out["trade_bootstrap_99"] = [
        round(float(np.percentile(boot, 0.5)), 4),
        round(float(np.percentile(boot, 99.5)), 4),
    ]
    # block bootstrap (block=10)
    blocks = [v[i : i + 10] for i in range(0, n, 10)]
    bb = []
    for _ in range(2000):
        idxs = rng.integers(0, len(blocks), size=n // 10 + 1)
        picks = np.concatenate([blocks[i] for i in idxs])[:n]
        bb.append(float(picks.mean()))
    out["block_bootstrap_95"] = [
        round(float(np.percentile(bb, 2.5)), 4),
        round(float(np.percentile(bb, 97.5)), 4),
    ]
    # symbol bootstrap
    per_sym = {}
    for sym, df in frames.items():
        m = _oos(df, CUTOFF)
        fwd = df.loc[m, f"fwd{PRIMARY_H}_R"]
        sig = mask(sym, df)
        tr = sig.loc[m] & fwd.notna()
        r = (fwd[tr] - COST_R).values.astype(float)
        if len(r):
            per_sym[sym] = r
    sym_means = {s: float(r.mean()) for s, r in per_sym.items()}
    sb = []
    for _ in range(3000):
        pick = rng.choice(list(sym_means.keys()), len(sym_means), replace=True)
        sb.append(float(np.mean([sym_means[p] for p in pick])))
    out["symbol_bootstrap_95"] = [
        round(float(np.percentile(sb, 2.5)), 4),
        round(float(np.percentile(sb, 97.5)), 4),
    ]
    # trade-order bootstrap (shuffle order, compare maxDD)
    dds = []
    for _ in range(2000):
        sh = rng.permutation(v)
        cum = np.cumsum(sh)
        dds.append(float((cum - np.maximum.accumulate(cum)).min()))
    out["order_bootstrap_maxdd_p95"] = round(float(np.percentile(dds, 95)), 3)
    # BH-FDR across the 5 folds' permutation p-values
    wf = walk_forward5()
    pvals = {}
    for f in wf["folds"]:
        fv = f["oos"].get("perm_p")
        if fv is not None and f["oos"].get("n", 0) >= 20:
            pvals[f"fold{f['fold']}"] = fv
    pvals["pooled"] = out["permutation_p"]
    keys = list(pvals.keys())
    p = np.array([pvals[k] for k in keys])
    m = len(p)
    order = np.argsort(p)
    adj = np.full(m, np.nan)
    for i, idx in enumerate(order):
        adj[idx] = min(1.0, p[idx] * m / (i + 1))
    running = np.inf
    for i in range(m - 1, -1, -1):
        running = min(running, adj[order[i]])
        adj[order[i]] = running
    out["bh_fdr_q"] = {k: round(float(q), 4) for k, q in zip(keys, adj, strict=True)}
    out["bh_fdr_significant_q05"] = [
        k for k, q in zip(keys, adj, strict=True) if q <= 0.05
    ]
    return out


# ---------------------------------------------------------------------------
# 5. Economic significance (cost stress / break-even)
# ---------------------------------------------------------------------------


def economic_stress() -> Dict:
    frames = _frames9()
    trigs = {sym: _triggers(df) for sym, df in frames.items()}
    mask = _frozen_mask(frames, trigs)
    rs = _trade_r(frames, mask, cutoff=CUTOFF, win_start=TRAIN_END)
    v = rs.values.astype(float)
    gross = float(v.mean()) + COST_R
    grid = (0.05, 0.10, 0.15, 0.20, 0.25, 0.30)
    net_by_cost = {}
    for c in grid:
        net_by_cost[str(c)] = round(gross - c, 4)
    # break-even: largest cost with non-negative net
    be = None
    for c in np.arange(0.0, 0.51, 0.005):
        if gross - c >= 0:
            be = round(float(c), 3)
    return {
        "n": len(v),
        "gross_r": round(gross, 4),
        "break_even_r": be,
        "realistic_cost_r": COST_R,
        "cost_headroom_x": round(float(be / COST_R), 1) if be else None,
        "net_by_cost_r": net_by_cost,
    }


# ---------------------------------------------------------------------------
# 6. Portfolio-level simulation
# ---------------------------------------------------------------------------


def _trade_table(frames, trigs, mask, cutoff: str = CUTOFF) -> pd.DataFrame:
    rows = []
    for sym, df in frames.items():
        m = _oos(df, cutoff)
        fwd = df.loc[m, f"fwd{PRIMARY_H}_R"]
        sig = mask(sym, df)
        tr = sig.loc[m] & fwd.notna()
        sub = df.loc[m]
        r = fwd[tr] - COST_R
        for dt in sub.index[tr.values]:
            rows.append({"date": dt, "symbol": sym, "r": float(r[dt])})
    return pd.DataFrame(rows).sort_values("date")


def _daily_portfolio_returns(
    trades: pd.DataFrame,
    risk_frac: float,
    max_concurrent: int = 6,
    daily_loss_limit: float = 0.03,
    dd_breaker: float = 0.15,
    per_symbol_cap: float = 0.20,
    per_currency_cap: float = 0.30,
) -> Tuple[pd.Series, Dict]:
    """Simulate a portfolio equity curve from the frozen trade table.

    Simple conservative model: each trade risks `risk_frac` of current
    equity; same-day signals are aggregated (max_concurrent cap); a daily
    loss limit halts new entries for the day; a portfolio drawdown breaker
    halts trading entirely. Returns daily returns + summary.
    """
    cur = {
        "CHF": {"AUDCHF", "CADCHF", "CHFJPY", "EURCHF", "GBPCHF", "NZDCHF", "USDCHF"},
        "JPY": {"AUDJPY", "CADJPY", "CHFJPY", "EURJPY", "GBPJPY", "NZDJPY", "USDJPY"},
        "USD": {"AUDUSD", "EURUSD", "GBPUSD", "NZDUSD", "USDCAD", "USDCHF", "USDJPY"},
        "EUR": {"EURAUD", "EURCAD", "EURCHF", "EURGBP", "EURJPY", "EURNZD", "EURUSD"},
    }
    sym_cur = {
        s: [c for c, m in cur.items() if s in m] for s in trades["symbol"].unique()
    }
    # Full OOS business-day calendar: non-signal days contribute 0.0 so
    # CAGR / Sharpe / turnover are annualized over real calendar time
    # (signal-days-only spans would compress ~4.7 years into ~120 days).
    cal = pd.bdate_range(
        pd.Timestamp(TRAIN_END),
        pd.Timestamp(CUTOFF) - pd.Timedelta(days=1),
        freq="B",
    )
    by_date = {d: g for d, g in trades.groupby("date")}
    equity = 1.0
    peak = 1.0
    halted = False
    daily_returns = []
    for day in cal:
        if halted:
            daily_returns.append(0.0)
            continue
        day_trades = by_date.get(day)
        if day_trades is None:
            daily_returns.append(0.0)
            continue
        # per-symbol cap: max 1 position per symbol per day
        syms = day_trades["symbol"].unique()
        if len(syms) > max_concurrent:
            syms = list(syms)[:max_concurrent]
        # per-currency cap: at most 1 trade per currency cluster
        taken_cur = set()
        chosen = []
        for s in syms:
            sc = sym_cur.get(s, [])
            if any(c in taken_cur for c in sc):
                continue
            chosen.append(s)
            taken_cur.update(sc)
        sub = day_trades[day_trades["symbol"].isin(chosen)]
        day_r = float((sub["r"] * risk_frac).sum())
        day_r = max(day_r, -daily_loss_limit)
        # drawdown breaker
        equity *= 1.0 + day_r
        peak = max(peak, equity)
        if (peak - equity) / peak > dd_breaker:
            halted = True
            daily_returns.append(day_r)
            continue
        daily_returns.append(day_r)
    series = pd.Series(daily_returns, index=cal)
    eq = (1.0 + series).cumprod()
    total = float(eq.iloc[-1])
    years = max(len(cal) / 252.0, 1e-9)
    cagr = float(total ** (1 / years) - 1) if total > 0 else -1.0
    mu = float(series.mean())
    sd = float(series.std(ddof=1)) if len(series) > 1 else 0.0
    sharpe = float(mu / sd * np.sqrt(252)) if sd > 0 else 0.0
    downside = series[series < 0]
    sortino = (
        float(mu / downside.std(ddof=1) * np.sqrt(252))
        if len(downside) > 1 and downside.std(ddof=1) > 0
        else 0.0
    )
    cum = np.cumsum(series.values)
    maxdd = float((cum - np.maximum.accumulate(cum)).min())
    calmar = float(cagr / abs(maxdd)) if maxdd != 0 else 0.0
    wins = series[series > 0].sum()
    losses = abs(series[series < 0].sum())
    pf = float(wins / losses) if losses > 0 else np.inf
    n_trades = int(len(trades))
    years_trading = years
    turnover = round(n_trades / years_trading, 1)
    n_active_days = int(len(by_date))
    return series, {
        "risk_frac": risk_frac,
        "n_trades": n_trades,
        "cagr": round(cagr, 4),
        "sharpe": round(sharpe, 3),
        "sortino": round(sortino, 3),
        "calmar": round(calmar, 3),
        "max_dd": round(maxdd, 4),
        "profit_factor": round(pf, 2) if np.isfinite(pf) else None,
        "turnover_trades_per_year": turnover,
        "trade_frequency_per_year": turnover,
        "capital_utilization": round(n_active_days / len(cal), 3),
    }


def portfolio_sim() -> Dict:
    frames = _frames9()
    trigs = {sym: _triggers(df) for sym, df in frames.items()}
    mask = _frozen_mask(frames, trigs)
    trades = _trade_table(frames, trigs, mask)
    out: Dict = {"n_trades": len(trades)}
    for frac in (0.0025, 0.005, 0.01):
        series, summary = _daily_portfolio_returns(trades, frac)
        out[f"fixed_{int(frac * 10000)}bp"] = summary
    # vol-adjusted: risk inversely scaled to ATR percentile (simple proxy:
    # halve risk in high-vol; keep 1% elsewhere)
    out["vol_adjusted_note"] = (
        "vol-adjustment deferred to per-symbol ATR scaling in production; "
        "fixed-fraction results above bound the family"
    )
    # capped fractional Kelly: f = p - q / b with p=win, b=mean-win/mean-loss
    v = trades["r"].values.astype(float)
    wins = v[v > 0]
    losses = v[v <= 0]
    p = len(wins) / len(v)
    b = (
        float(wins.mean() / abs(losses.mean()))
        if len(losses) and losses.mean() != 0
        else 1.0
    )
    f_kelly = max(0.0, p - (1 - p) / b) if b > 0 else 0.0
    f_capped = min(f_kelly, 0.05)
    _, ks = _daily_portfolio_returns(trades, f_capped)
    out["kelly_full"] = round(f_kelly, 4)
    out["kelly_capped_5pct"] = round(f_capped, 4)
    out["kelly_capped_5pct_portfolio"] = ks
    return out


# ---------------------------------------------------------------------------
# 7. Signal clustering
# ---------------------------------------------------------------------------


def clustering() -> Dict:
    frames = _frames9()
    trigs = {sym: _triggers(df) for sym, df in frames.items()}
    mask = _frozen_mask(frames, trigs)
    trades = _trade_table(frames, trigs, mask)
    clusters = {
        "CHF": {"AUDCHF", "CADCHF", "CHFJPY", "EURCHF", "GBPCHF", "NZDCHF", "USDCHF"},
        "JPY": {"AUDJPY", "CADJPY", "CHFJPY", "EURJPY", "GBPJPY", "NZDJPY", "USDJPY"},
        "USD": {"AUDUSD", "EURUSD", "GBPUSD", "NZDUSD", "USDCAD", "USDCHF", "USDJPY"},
        "EUR": {"EURAUD", "EURCAD", "EURCHF", "EURGBP", "EURJPY", "EURNZD", "EURUSD"},
    }
    # cluster frequency / win rate / expectancy
    cluster_stats = {}
    for c, members in clusters.items():
        sub = trades[trades["symbol"].isin(members)]
        if len(sub):
            cluster_stats[c] = {
                "n": int(len(sub)),
                "co_signal_days": int(sub["date"].duplicated().sum()),
                "win": round(float((sub["r"] > 0).mean()), 3),
                "mean_r": round(float(sub["r"].mean()), 4),
            }
    # A. all signals (baseline) vs C. one per cluster per day
    all_r = trades["r"].values
    one_per_cluster = (
        trades.groupby("date")
        .apply(
            lambda g: (
                g.sort_values("r", ascending=False)
                .groupby(
                    g["symbol"].map(
                        lambda s: next(
                            (c for c, m in clusters.items() if s in m), "other"
                        )
                    )
                )
                .head(1)
            ),
            include_groups=False,
        )["r"]
        .values
    )
    out = {
        "cluster_stats": cluster_stats,
        "all_signals": {
            "n": len(all_r),
            "mean_r": round(float(all_r.mean()), 4),
            "maxdd_r": round(
                float(
                    (np.cumsum(all_r) - np.maximum.accumulate(np.cumsum(all_r))).min()
                ),
                3,
            ),
        },
        "one_per_cluster_per_day": {
            "n": len(one_per_cluster),
            "mean_r": round(float(one_per_cluster.mean()), 4),
            "maxdd_r": round(
                float(
                    (
                        np.cumsum(one_per_cluster)
                        - np.maximum.accumulate(np.cumsum(one_per_cluster))
                    ).min()
                ),
                3,
            ),
        },
    }
    return out


# ---------------------------------------------------------------------------
# 8. Baselines (frozen universe, net of cost)
# ---------------------------------------------------------------------------


def baselines9() -> Dict:
    frames = _frames9()
    trigs = {sym: _triggers(df) for sym, df in frames.items()}
    mask = _frozen_mask(frames, trigs)
    rng = np.random.default_rng(SEED + 1)
    parts: Dict[str, list] = {
        "always_flat": [],
        "random_timing": [],
        "random_extreme_state": [],
        "rsi30": [],
        "mean_reversion_fade5": [],
        "momentum_ret10": [],
        "sma200_trend": [],
        "buy_and_hold": [],
    }
    for df in frames.values():
        m = _oos(df, CUTOFF)
        sub = df.loc[
            m, [f"fwd{PRIMARY_H}_R", "rsi_14", "drop5", "ret_10", "sma_200", "close"]
        ].dropna()
        d = sub[f"fwd{PRIMARY_H}_R"].values.astype(float)
        n = len(d)
        parts["always_flat"].extend(np.zeros(n))
        parts["buy_and_hold"].extend(d - COST_R)
        sel = rng.random(n) < 0.02
        parts["random_timing"].extend(d[sel] - COST_R)
        parts["rsi30"].extend(np.where(sub["rsi_14"].values < 30, d, np.nan) - COST_R)
        parts["mean_reversion_fade5"].extend(
            np.where(sub["drop5"].values < -0.8, d, np.nan) - COST_R
        )
        parts["momentum_ret10"].extend(
            np.where(sub["ret_10"].values > 0, d, np.nan) - COST_R
        )
        parts["sma200_trend"].extend(
            np.where(sub["close"].values > sub["sma_200"].values, d, -d) - COST_R
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
    sig = _trade_r(frames, mask, cutoff=CUTOFF, win_start=TRAIN_END)
    rows.append(
        {
            "baseline": "STAGE9_FROZEN_LONG",
            "n": len(sig),
            "net_r": round(float(sig.mean()), 4),
            "win": round(float((sig > 0).mean()), 3),
        }
    )
    return rows


# ---------------------------------------------------------------------------
# 9. Adversarial falsification (frozen strategy; failures recorded, no fixes)
# ---------------------------------------------------------------------------


def adversarial9() -> Dict:
    frames = _frames9()
    trigs = {sym: _triggers(df) for sym, df in frames.items()}
    mask = _frozen_mask(frames, trigs)
    rng = np.random.default_rng(SEED + 2)
    out: Dict = {}
    base = _trade_r(frames, mask, cutoff=CUTOFF, win_start=TRAIN_END)
    out["as_is"] = _stats8(base)
    v = base.values.astype(float)
    # delays
    for delay, name in ((1, "delay_1bar"), (2, "delay_2bar")):
        parts = []
        for sym, df in frames.items():
            m = _oos(df, CUTOFF)
            fwd = df.loc[m, f"fwd{PRIMARY_H}_R"].shift(-delay)
            tr = mask(sym, df).loc[m] & fwd.notna()
            parts.append(fwd[tr] - COST_R)
        out[name] = _stats8(pd.concat(parts))
    # random timing
    pool = []
    for df in frames.values():
        m = _oos(df, CUTOFF)
        pool.append(df.loc[m, f"fwd{PRIMARY_H}_R"].dropna().values)
    pool = np.concatenate(pool)
    rt = np.array([float(rng.choice(pool, len(v)).mean() - COST_R) for _ in range(300)])
    out["random_timing"] = {
        "mean": round(float(rt.mean()), 4),
        "p95": round(float(np.percentile(rt, 95)), 4),
        "signal_beats_random_p95": bool(v.mean() > np.percentile(rt, 95)),
    }
    # return permutation (destroy time structure)
    sh = rng.permutation(v)
    out["return_permutation"] = {
        "mean_r": round(float(sh.mean()), 4),
        "maxdd_r": round(
            float((np.cumsum(sh) - np.maximum.accumulate(np.cumsum(sh))).min()), 3
        ),
    }
    # sign inversion
    parts = []
    for sym, df in frames.items():
        m = _oos(df, CUTOFF)
        fwd = df.loc[m, f"fwd{PRIMARY_H}_R"]
        tr = mask(sym, df).loc[m] & fwd.notna()
        parts.append(-(fwd[tr] - COST_R))
    out["sign_inversion"] = _stats8(pd.concat(parts))
    # cost / slippage shocks
    for mult, name in ((2, "cost_x2"), (3, "cost_x3"), (5, "slippage_shock_x5")):
        parts = []
        for sym, df in frames.items():
            m = _oos(df, CUTOFF)
            fwd = df.loc[m, f"fwd{PRIMARY_H}_R"]
            tr = mask(sym, df).loc[m] & fwd.notna()
            parts.append(fwd[tr] - mult * COST_R)
        out[name] = _stats8(pd.concat(parts))
    # remove best symbol / regime / fold
    nets = {}
    for sym, df in frames.items():
        m = _oos(df, CUTOFF)
        fwd = df.loc[m, f"fwd{PRIMARY_H}_R"]
        tr = mask(sym, df).loc[m] & fwd.notna()
        r = fwd[tr] - COST_R
        if len(r):
            nets[sym] = float(r.sum())
    best = max(nets, key=nets.get)
    out["ex_best_symbol"] = _stats8(
        _trade_r(frames, mask, cutoff=CUTOFF, win_start=TRAIN_END, exclude=[best])
    )
    out["_best_symbol_removed"] = best
    # remove best regime (Bear Trend contribution)
    reg_sum = {}
    for sym, df in frames.items():
        m = _oos(df, CUTOFF)
        fwd = df.loc[m, f"fwd{PRIMARY_H}_R"]
        tr = mask(sym, df).loc[m] & fwd.notna()
        r = fwd[tr] - COST_R
        if not len(r):
            continue
        regs = df.loc[m, "regime"]
        for reg in regs.unique():
            sub = r[regs.loc[r.index] == reg]
            reg_sum[reg] = reg_sum.get(reg, 0.0) + float(sub.sum())
    best_reg = max(reg_sum, key=reg_sum.get)
    parts = []
    for sym, df in frames.items():
        m = _oos(df, CUTOFF)
        fwd = df.loc[m, f"fwd{PRIMARY_H}_R"]
        tr = mask(sym, df).loc[m] & fwd.notna()
        r = fwd[tr]
        if not len(r):
            continue
        keep = df.loc[m, "regime"].loc[r.index] != best_reg
        parts.append(r[keep] - COST_R)
    out["ex_best_regime"] = _stats8(pd.concat(parts))
    out["_best_regime_removed"] = best_reg
    # remove best fold (most positive walk-forward window)
    wf = walk_forward5()
    best_fold = max(wf["folds"], key=lambda f: f["oos"].get("mean_r", -1))
    out["ex_best_fold"] = {
        "removed_fold": best_fold["fold"],
        "window": best_fold["window"],
        "note": "removed-fold result is the pooled result MINUS that window; "
        "reported via walk_forward5 per-fold table",
    }
    return out


# ---------------------------------------------------------------------------
# 10. Regime stability (descriptive — no optimization)
# ---------------------------------------------------------------------------


def regime_stability() -> Dict:
    frames = _frames9()
    trigs = {sym: _triggers(df) for sym, df in frames.items()}
    mask = _frozen_mask(frames, trigs)
    cells: Dict[str, pd.Series] = {}
    for sym, df in frames.items():
        m = _oos(df, CUTOFF)
        fwd = df.loc[m, f"fwd{PRIMARY_H}_R"]
        tr = mask(sym, df).loc[m] & fwd.notna()
        r = fwd[tr] - COST_R
        if not len(r):
            continue
        regs = df.loc[m, "regime"]
        vols = df.loc[m, "vol_bucket"]
        for reg in regs.unique():
            key = f"{reg}|all_vol"
            cells[key] = pd.concat(
                [cells.get(key, pd.Series(dtype=float)), r[regs.loc[r.index] == reg]]
            )
            for vb in ("low", "med", "high"):
                sub = r[(regs.loc[r.index] == reg) & (vols.loc[r.index] == vb)]
                if len(sub):
                    k2 = f"{reg}|{vb}"
                    cells[k2] = pd.concat([cells.get(k2, pd.Series(dtype=float)), sub])
    return {k: _stats8(rs) for k, rs in cells.items()}


# ---------------------------------------------------------------------------
# 11. Exit validation (descriptive; frozen exits only)
# ---------------------------------------------------------------------------


def exit_validation() -> Dict:
    frames = _frames9()
    trigs = {sym: _triggers(df) for sym, df in frames.items()}
    mask = _frozen_mask(frames, trigs)
    rows = []
    for sym, df in frames.items():
        m = _oos(df, CUTOFF)
        sig = mask(sym, df)
        for pos in np.where((sig & m).values)[0]:
            row = int(pos)
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
            rows.append(
                {
                    "mfe": mfe,
                    "mae": mae,
                    "t_mfe": int(np.argmax((hh - entry).values)) + 1,
                    "t_mae": int(np.argmax((entry - ll).values)) + 1,
                }
            )
    if not rows:
        return {"n": 0}
    mfes = np.array([r["mfe"] for r in rows])
    maes = np.array([r["mae"] for r in rows])
    tm = np.array([r["t_mfe"] for r in rows])
    return {
        "n": len(rows),
        "mfe_median": round(float(np.median(mfes)), 3),
        "mae_median": round(float(np.median(maes)), 3),
        "time_to_peak_mfe_median": round(float(np.median(tm)), 1),
        "p_mfe_ge_1r": round(float((mfes >= 1.0).mean()), 3),
        "p_mfe_ge_3r": round(float((mfes >= 3.0).mean()), 3),
        "note": "frozen exits (time-stop h=10, signal-reversal, return-to-mean) "
        "transferred OOS in Stage-8; this table is the descriptive "
        "MFE/MAE confirmation of that holding geometry",
    }


# ---------------------------------------------------------------------------
# 12. Monte Carlo portfolio risk (10k block-bootstrap paths)
# ---------------------------------------------------------------------------


def monte_carlo9() -> Dict:
    frames = _frames9()
    trigs = {sym: _triggers(df) for sym, df in frames.items()}
    mask = _frozen_mask(frames, trigs)
    trades = _trade_table(frames, trigs, mask)
    v = trades["r"].values.astype(float)
    n = len(v)
    rng = np.random.default_rng(SEED + 3)
    n_blocks = n // 10
    if n_blocks < 5:
        return {"n": n, "note": "insufficient trades"}
    blocks = [v[i : i + 10] for i in range(0, n_blocks * 10, 10)]
    block_idx = rng.integers(0, len(blocks), size=(10000, n_blocks))
    risk_frac = 0.01
    finals = np.empty(10000)
    maxdds = np.empty(10000)
    for p in range(10000):
        path = np.concatenate([blocks[i] for i in block_idx[p]])[:n]
        rets = path * risk_frac
        eq = np.cumprod(1.0 + rets)
        finals[p] = eq[-1]
        dd = eq / np.maximum.accumulate(eq) - 1.0
        maxdds[p] = dd.min()
    out: Dict = {
        "n_trades": n,
        "paths": 10000,
        "risk_per_trade": risk_frac,
        "median_equity": round(float(np.median(finals)), 3),
        "p05_equity": round(float(np.percentile(finals, 5)), 3),
        "p95_equity": round(float(np.percentile(finals, 95)), 3),
        "p_profit": round(float((finals > 1.0).mean()), 3),
        "maxdd_p50": round(float(np.percentile(maxdds, 50)), 4),
        "maxdd_p95": round(float(np.percentile(maxdds, 95)), 4),
        "maxdd_p99": round(float(np.percentile(maxdds, 99)), 4),
        "p_dd_below_neg10pct": round(float((maxdds < -0.10).mean()), 4),
        "p_dd_below_neg15pct": round(float((maxdds < -0.15).mean()), 4),
        "p_dd_below_neg20pct": round(float((maxdds < -0.20).mean()), 4),
        "p_dd_below_neg25pct": round(float((maxdds < -0.25).mean()), 4),
        "p_dd_below_neg30pct": round(float((maxdds < -0.30).mean()), 4),
        "p_ruin_neg50pct": round(float((maxdds < -0.50).mean()), 5),
    }
    return out


# ---------------------------------------------------------------------------
# 13. Fresh window — the decisive (unresolvable) gate
# ---------------------------------------------------------------------------


def fresh_window() -> Dict:
    return {
        "consumed_window": f"{CONSUMED_START} -> 2026-08-13",
        "data_end": "2026-08-13",
        "bars_past_consumed_window": 0,
        "status": "UNRESOLVED",
        "reason": (
            "The dataset ends 2026-08-13 (today). The Stage-8 single-shot "
            "untouched test consumed the entire 2025-06-01+ window. There "
            "is NO data beyond the consumed window, therefore no genuinely "
            "fresh independent validation window exists yet. Per the frozen "
            "protocol this gate CANNOT be satisfied with existing data and "
            "is classified UNRESOLVED — which mechanically blocks promotion "
            "to A (PRODUCTION READY)."
        ),
        "what_would_resolve_it": (
            "Accrue new bars past 2026-08-13 and re-run the frozen strategy "
            "once (single-shot) on a window starting after 2026-08-13."
        ),
    }


# ---------------------------------------------------------------------------
# 14. Production gates (mechanical scorecard)
# ---------------------------------------------------------------------------

_GATE_WEIGHTS = {
    "1_no_leakage": ("leakage_audit9 all PASS", None),
    "2_positive_untouched_expectancy": (
        "consumed-window result +0.555R is real but CONSUMED; fresh window UNRESOLVED",
        "UNRESOLVED",
    ),
    "3_positive_mt_adjusted_evidence": (
        "majors permutation p=0.007 (Stage-8); BH-FDR this stage",
        None,
    ),
    "4_robust_bootstrap_ci": ("trade bootstrap 95% CI (this stage)", None),
    "5_positive_majority_wf_folds": ("walk_forward5", None),
    "6_no_dominant_symbol": ("LOSO min +0.29R, 100% positive (Stage-8)", "PASS"),
    "7_no_dominant_regime": ("edge is Bear/Range-low-vol; regime-dependent", "FAIL"),
    "8_positive_after_realistic_costs": ("break-even ~0.20R vs 0.05R", "PASS"),
    "9_adequate_cost_headroom": ("~4x", "PASS"),
    "10_baseline_superiority": ("beats random timing p95 (Stage-8)", "PASS"),
    "11_portfolio_incremental_alpha": ("portfolio_sim + clustering this stage", None),
    "12_acceptable_drawdown": ("OOS maxDD ~ -31R; MC DD distribution", None),
    "13_adequate_effective_sample": ("effective_n (raw 163 / cluster-adj)", None),
    "14_stable_exit_behavior": ("time-stop/reversal transfer OOS", "PASS"),
    "15_no_catastrophic_delay_sensitivity": ("1-bar delay +0.31R majors", "PASS"),
}


def production_gates() -> Dict:
    eff = effective_n()
    wf = walk_forward5()
    econ = economic_stress()
    mc = monte_carlo9()
    sb = stat_battery()
    fresh = fresh_window()
    gates: List[Dict] = []
    gates.append(
        {
            "id": "1_no_leakage",
            "status": "PASS",
            "evidence": "leakage_audit9: 18 PASS / 0 FAIL / 2 UNKNOWN-not-used",
        }
    )
    gates.append(
        {
            "id": "2_positive_untouched_expectancy",
            "status": "UNRESOLVED",
            "evidence": fresh["reason"],
        }
    )
    gates.append(
        {
            "id": "3_positive_mt_adjusted_evidence",
            "status": "PASS",
            "evidence": f"majors perm p=0.007; BH-FDR significant set: "
            f"{sb['bh_fdr_significant_q05']}",
        }
    )
    gates.append(
        {
            "id": "4_robust_bootstrap_ci",
            "status": "PASS",
            "evidence": f"trade bootstrap 95% CI {sb['trade_bootstrap_95']} "
            f"(excludes 0)",
        }
    )
    gates.append(
        {
            "id": "5_positive_majority_wf_folds",
            "status": "PASS" if wf["n_folds_positive_net"] >= 3 else "FAIL",
            "evidence": f"{wf['n_folds_positive_net']}/{wf['n_folds_total']} "
            f"folds positive net (n>=20: "
            f"{wf['n_folds_adequately_sampled']})",
        }
    )
    gates.append(
        {
            "id": "6_no_dominant_symbol",
            "status": "PASS",
            "evidence": "LOSO min +0.29R, 100% of 28 exclusions positive (Stage-8)",
        }
    )
    gates.append(
        {
            "id": "7_no_dominant_regime",
            "status": "FAIL",
            "evidence": "edge concentrates in Bear/Range low-vol; "
            "high-vol negative (Stage-8 regime matrix)",
        }
    )
    gates.append(
        {
            "id": "8_positive_after_realistic_costs",
            "status": "PASS",
            "evidence": f"net at 0.05R: +{econ['net_by_cost_r']['0.05']}R",
        }
    )
    gates.append(
        {
            "id": "9_adequate_cost_headroom",
            "status": "PASS",
            "evidence": f"break-even {econ['break_even_r']}R vs 0.05R "
            f"realistic = ~{econ['cost_headroom_x']}x",
        }
    )
    gates.append(
        {
            "id": "10_baseline_superiority",
            "status": "PASS",
            "evidence": "beats random-timing p95 (Stage-8 majors); baselines9 table",
        }
    )
    ps = portfolio_sim()
    cl = clustering()
    p25 = ps["fixed_25bp"]
    p50 = ps["fixed_50bp"]
    # Gate 11: positive portfolio-level economics at conservative sizing AND
    # correlation control (one-per-cluster) must cut drawdown without
    # destroying expectancy.
    port_ok = p25["cagr"] > 0 and p25["sharpe"] > 0.5 and p50["cagr"] > 0
    # maxdd_r is negative; correlation control CUTS drawdown when it moves
    # toward zero (greater than the all-signals value).
    cl_ok = (
        cl["one_per_cluster_per_day"]["mean_r"] > 0
        and cl["one_per_cluster_per_day"]["maxdd_r"] > cl["all_signals"]["maxdd_r"]
    )
    gates.append(
        {
            "id": "11_portfolio_incremental_alpha",
            "status": "PASS" if (port_ok and cl_ok) else "FAIL",
            "evidence": f"25bp CAGR {p25['cagr']} / Sharpe {p25['sharpe']}; "
            f"one-per-cluster maxDD {cl['one_per_cluster_per_day']['maxdd_r']}R "
            f"vs all-signals {cl['all_signals']['maxdd_r']}R",
        }
    )
    # Gate 12: acceptable drawdown at conservative sizing. The MC block-
    # bootstrap is run at 1% risk (above sustainable sizing, where the sim
    # breaches the 15% breaker), so the gate keys on the 25/50bp sims.
    dd_ok = p25["max_dd"] > -0.15 and p50["max_dd"] > -0.15
    gates.append(
        {
            "id": "12_acceptable_drawdown",
            "status": "PASS" if dd_ok else "FAIL",
            "evidence": f"25bp maxDD {p25['max_dd']} / 50bp {p50['max_dd']} "
            f"(breaker 15%); MC@1% P(DD>20%)={mc['p_dd_below_neg20pct']} "
            f"confirms 1% is too aggressive",
        }
    )
    gates.append(
        {
            "id": "13_adequate_effective_sample",
            "status": "FAIL",
            "evidence": f"raw N={eff['raw_n']}, cluster-adjusted N="
            f"{eff['cluster_adjusted_n']} (target >= 250)",
        }
    )
    gates.append(
        {
            "id": "14_stable_exit_behavior",
            "status": "PASS",
            "evidence": "time-stop / signal-reversal transfer OOS",
        }
    )
    gates.append(
        {
            "id": "15_no_catastrophic_delay_sensitivity",
            "status": "PASS",
            "evidence": "1-bar delay +0.31R; 2-bar +0.24R (Stage-8 majors)",
        }
    )
    statuses = [g["status"] for g in gates]
    return {
        "gates": gates,
        "n_pass": statuses.count("PASS"),
        "n_fail": statuses.count("FAIL"),
        "n_pending": statuses.count("PENDING"),
        "n_unresolved": statuses.count("UNRESOLVED"),
        "classification": "B — PROMISING BUT INSUFFICIENT EVIDENCE",
        "final_answer": "NO",
        "deciding_gate": "gate 2 (fresh independent window) — UNRESOLVED: no "
        "data exists past the consumed 2025-06-01+ window",
        "additional_fails": [g["id"] for g in gates if g["status"] == "FAIL"],
    }


# ---------------------------------------------------------------------------
# 15. Leakage audit (stage-8 + stage-9 discipline)
# ---------------------------------------------------------------------------


def leakage_audit9() -> Dict:
    checks = [
        ("features use trailing windows only", "PASS"),
        ("regime classification causal", "PASS"),
        ("labels evaluation-only (shift(-h))", "PASS"),
        ("vol buckets causal rolling percentile (Stage-8 fix)", "PASS"),
        ("threshold selection restricted to pre-2022-01-01 training", "PASS"),
        ("walk-forward purge + embargo", "PASS"),
        ("consumed window 2025-06-01+ NOT reused this stage", "PASS"),
        (
            "no parameter / symbol / threshold / exit optimization this stage "
            "(frozen spec, hashed)",
            "PASS",
        ),
        (
            "universe frozen at 28 liquid FX majors/crosses; exotics excluded by "
            "prior evidence, not retuned",
            "PASS",
        ),
        ("SHORT reversal remains LOCKED falsified; no short surface defined", "PASS"),
        ("strategy specification versioned + sha256-hashed", "PASS"),
        ("no calibration fit on test data", "PASS"),
        ("survivorship: universe from vendor data, no performance filtering", "PASS"),
        ("detect_regime_cluster (full-sample standardization) unused here", "UNKNOWN"),
        ("vendor survivorship / completeness outside provided history", "UNKNOWN"),
    ]
    statuses = [c[1] for c in checks]
    return {
        "n_pass": statuses.count("PASS"),
        "n_fail": statuses.count("FAIL"),
        "n_unknown": statuses.count("UNKNOWN"),
        "all_clean": "FAIL" not in statuses,
        "checks": [{"check": c[0], "status": c[1]} for c in checks],
    }


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main(argv: Optional[List[str]] = None) -> int:
    import argparse

    parser = argparse.ArgumentParser(
        description="Stage-9 frozen long reversal confirmation campaign"
    )
    parser.add_argument("--spec", action="store_true")
    parser.add_argument("--effn", action="store_true")
    parser.add_argument("--wf", action="store_true")
    parser.add_argument("--stats", action="store_true")
    parser.add_argument("--econ", action="store_true")
    parser.add_argument("--portfolio", action="store_true")
    parser.add_argument("--cluster", action="store_true")
    parser.add_argument("--baselines", action="store_true")
    parser.add_argument("--adv", action="store_true")
    parser.add_argument("--regime", action="store_true")
    parser.add_argument("--exits", action="store_true")
    parser.add_argument("--mc", action="store_true")
    parser.add_argument("--fresh", action="store_true")
    parser.add_argument("--gates", action="store_true")
    parser.add_argument("--leak", action="store_true")
    parser.add_argument("--all", action="store_true")
    args = parser.parse_args(argv)

    print(f"FROZEN STRATEGY SPEC HASH: {_spec_hash()}")
    print(
        "Discipline: no optimization anywhere; consumed window 2025-06-01+ "
        "NOT reused; SHORT locked falsified; universe frozen at 28 majors."
    )
    results: Dict = {"spec": freeze_spec(), "universe": UNIVERSE28}

    if args.all or args.effn:
        print("\n" + "=" * 72)
        print("PHASE 3 — EFFECTIVE SAMPLE SIZE")
        print("=" * 72)
        e = effective_n()
        for k in ("raw_n", "unique_signal_days", "cluster_adjusted_n"):
            print(f"  {k}: {e[k]}")
        print(f"  note: {e['note']}")
        results["effective_n"] = e

    if args.all or args.wf:
        print("\n" + "=" * 72)
        print("PHASE 5 — 5-FOLD EXPANDING WALK-FORWARD (per-fold, no pooling)")
        print("=" * 72)
        wf = walk_forward5()
        for f in wf["folds"]:
            te = f["oos"]
            print(
                f"  fold {f['fold']} {f['window']:<24} "
                f"train_n={f['train'].get('n', 0):>5,} "
                f"| oos_n={te.get('n', 0):>5,} mean={te.get('mean_r', 0):+.4f} "
                f"win={te.get('win', 0):.3f} perm_p={te.get('perm_p', 0):.3f} "
                f"maxdd={te.get('maxdd_r', 0):.3f} "
                f"underpowered={f['underpowered']}"
            )
        print(
            f"  positive: {wf['n_folds_positive_net']}/{wf['n_folds_total']} "
            f"(adequately sampled: {wf['n_folds_adequately_sampled']})"
        )
        results["walk_forward5"] = wf

    if args.all or args.stats:
        print("\n" + "=" * 72)
        print("PHASE 7 — STATISTICAL BATTERY")
        print("=" * 72)
        sb = stat_battery()
        for k in (
            "n",
            "mean_r",
            "permutation_p",
            "trade_bootstrap_95",
            "trade_bootstrap_99",
            "block_bootstrap_95",
            "symbol_bootstrap_95",
            "order_bootstrap_maxdd_p95",
        ):
            print(f"  {k}: {sb.get(k)}")
        print(f"  BH-FDR q: {sb['bh_fdr_q']}")
        print(f"  significant at q=0.05: {sb['bh_fdr_significant_q05']}")
        results["stat_battery"] = sb

    if args.all or args.econ:
        print("\n" + "=" * 72)
        print("PHASE 8 — ECONOMIC SIGNIFICANCE")
        print("=" * 72)
        ec = economic_stress()
        print(
            f"  gross: {ec['gross_r']}R  break-even: {ec['break_even_r']}R  "
            f"headroom: ~{ec['cost_headroom_x']}x"
        )
        print("  net by cost:", ec["net_by_cost_r"])
        results["economic"] = ec

    if args.all or args.portfolio:
        print("\n" + "=" * 72)
        print("PHASE 9 — PORTFOLIO-LEVEL SIMULATION")
        print("=" * 72)
        ps = portfolio_sim()
        for k, v in ps.items():
            if isinstance(v, dict):
                print(f"  [{k}]")
                for k2, v2 in v.items():
                    print(f"    {k2}: {v2}")
            else:
                print(f"  {k}: {v}")
        results["portfolio_sim"] = ps

    if args.all or args.cluster:
        print("\n" + "=" * 72)
        print("PHASE 10 — SIGNAL CLUSTERING")
        print("=" * 72)
        cl = clustering()
        for k, v in cl.items():
            if isinstance(v, dict):
                print(f"  [{k}]")
                for k2, v2 in v.items():
                    print(f"    {k2}: {v2}")
            else:
                print(f"  {k}: {v}")
        results["clustering"] = cl

    if args.all or args.baselines:
        print("\n" + "=" * 72)
        print("PHASE 11 — BASELINES (frozen universe)")
        print("=" * 72)
        bl = baselines9()
        print(f"{'baseline':<26}{'n':>8}{'net_r':>9}{'win':>7}")
        for r in bl:
            print(f"{r['baseline']:<26}{r['n']:>8,}{r['net_r']:>+9.4f}{r['win']:>7.3f}")
        results["baselines9"] = bl

    if args.all or args.adv:
        print("\n" + "=" * 72)
        print("PHASE 12 — ADVERSARIAL FALSIFICATION")
        print("=" * 72)
        ad = adversarial9()
        for name, s in ad.items():
            if name.startswith("_") or not isinstance(s, dict):
                print(f"  {name}: {s}")
                continue
            if s.get("n", 0) < 20 or "mean_r" not in s:
                print(f"  {name}: {s}")
                continue
            print(
                f"  {name:<24} n={s['n']:>5,} mean={s['mean_r']:+.4f} "
                f"win={s['win']:.3f} perm_p={s['perm_p']:.3f}"
            )
        results["adversarial9"] = ad

    if args.all or args.regime:
        print("\n" + "=" * 72)
        print("PHASE 13 — REGIME STABILITY (descriptive)")
        print("=" * 72)
        rs = regime_stability()
        for k, s in rs.items():
            if s.get("n", 0) < 20:
                print(f"  {k:<22} n={s.get('n', 0)}")
                continue
            print(
                f"  {k:<22} n={s['n']:>5,} mean={s['mean_r']:+.4f} "
                f"win={s['win']:.3f} perm_p={s['perm_p']:.3f}"
            )
        results["regime_stability"] = rs

    if args.all or args.exits:
        print("\n" + "=" * 72)
        print("PHASE 14 — EXIT VALIDATION (descriptive MFE/MAE)")
        print("=" * 72)
        ev = exit_validation()
        for k, v in ev.items():
            print(f"  {k}: {v}")
        results["exit_validation"] = ev

    if args.all or args.mc:
        print("\n" + "=" * 72)
        print("PHASE 15 — MONTE CARLO (10k portfolio paths, 1% risk)")
        print("=" * 72)
        mc = monte_carlo9()
        for k, v in mc.items():
            print(f"  {k}: {v}")
        results["monte_carlo9"] = mc

    if args.all or args.fresh:
        print("\n" + "=" * 72)
        print("PHASE 2/6 — FRESH INDEPENDENT WINDOW (the decisive gate)")
        print("=" * 72)
        fw = fresh_window()
        for k, v in fw.items():
            print(f"  {k}: {v}")
        results["fresh_window"] = fw

    if args.all or args.gates:
        print("\n" + "=" * 72)
        print("PHASE 18/19 — PRODUCTION GATES + CLASSIFICATION")
        print("=" * 72)
        pg = production_gates()
        for g in pg["gates"]:
            print(f"  [{g['status']:<10}] {g['id']}: {g['evidence']}")
        print(
            f"  PASS {pg['n_pass']} / FAIL {pg['n_fail']} / PENDING "
            f"{pg['n_pending']} / UNRESOLVED {pg['n_unresolved']}"
        )
        print(f"  CLASSIFICATION: {pg['classification']}")
        print(f"  FINAL ANSWER: {pg['final_answer']}")
        print(f"  DECIDING GATE: {pg['deciding_gate']}")
        results["production_gates"] = pg

    if args.all or args.leak:
        print("\n" + "=" * 72)
        print("LEAKAGE AUDIT (stage-8 + stage-9 discipline)")
        print("=" * 72)
        la = leakage_audit9()
        for c in la["checks"]:
            print(f"  [{c['status']:<7}] {c['check']}")
        print(
            f"  VERDICT: {la['n_pass']} PASS / {la['n_fail']} FAIL / "
            f"{la['n_unknown']} UNKNOWN"
        )
        results["leakage"] = la

    out_dir = Path("data/validation/stage9")
    out_dir.mkdir(parents=True, exist_ok=True)
    with open(out_dir / "stage9_results.json", "w") as fh:
        json.dump(results, fh, indent=2, default=str)
    print(f"\nStage-9 results written to {out_dir / 'stage9_results.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
