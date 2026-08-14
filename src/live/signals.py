"""
NexusQuant - Live Signal Pass.

Turns the full research stack into actionable alerts:

1. **Scan** the watchlist through the whole pipeline (regime -> dip ->
   ML probability -> macro overlay).
2. **Filter** to high-conviction setups: confirmed dip, strong bias,
   macro gate PASS, ML probability above a threshold (when the model
   exists), minimum dip score.
3. **Size** each setup with the risk module (fractional sizing + R:R +
   VaR on the live setup).
4. **Dedup** against a state file so a daily scheduler only alerts on a
   setup once (keyed by symbol + entry zone, not by date - the same zone
   staying valid across days is one alert, not N).

Missing watchlist symbols are resolved on demand (local -> MT5 terminal
-> Yahoo) so the pass works on any symbol, not just the ones already
backfilled; the CLI in ``run.py`` wires in the channels.
"""

from __future__ import annotations

import json
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

import pandas as pd

sys.path.append(str(Path(__file__).parent.parent.parent))

from src.analysis.scanner import _data_path, scan_universe
from src.analysis.dashboard_data import load_symbol_report
from src.data.freshness import staleness_days
from src.macro.overlay import latest_macro_scores, macro_regime

DEFAULT_STATE_FILE = "data/live/alerts.json"
DEFAULT_GROUP = "full_fx"
DEFAULT_TIMEFRAME = "D1"

# Warn in the briefing when a watchlist file is older than this many
# calendar days - decisions on stale bars are decisions on the past.
DEFAULT_MAX_STALE_DAYS = 4


# ---------------------------------------------------------------------------
# Filtering
# ---------------------------------------------------------------------------


def filter_signals(
    table: pd.DataFrame,
    min_dip_score: int = 5,
    require_confirmed: bool = True,
    min_bias: float = -4.0,
    min_ml_prob: Optional[float] = None,
    require_macro_pass: bool = True,
    min_rr: Optional[float] = None,
) -> pd.DataFrame:
    """
    Apply the high-conviction filters to a scanner table.

    ``min_ml_prob`` only filters rows where ``ml_prob`` is available; rows
    with no model are kept (graceful, same as the rest of the system).
    ``require_macro_pass`` keeps only rows whose macro gate is PASS; rows
    where no macro data exists are kept (no macro = no gate). ``min_rr``
    drops setups whose reward:risk (entry->target vs entry->stop) is too
    low - the dip target is the nearest resistance, which can be close.
    """
    f = table.copy()
    f = f[f["dip_score"] >= min_dip_score]
    if require_confirmed and "dip_confirmed" in f:
        f = f[f["dip_confirmed"] == "Yes"]
    if "bias_score" in f:
        f = f[f["bias_score"] >= min_bias]
    if min_ml_prob is not None and "ml_prob" in f:
        has_ml = f["ml_prob"].notna()
        f = f[~has_ml | (f["ml_prob"] >= min_ml_prob)]
    if require_macro_pass and "macro_gate" in f:
        no_macro = f["macro_gate"].isna()
        f = f[no_macro | (f["macro_gate"] == "PASS")]
    if min_rr is not None and {"entry_zone", "invalidation", "resistance"}.issubset(
        f.columns
    ):
        rr = _rr_series(f)
        f = f[rr.isna() | (rr >= min_rr)]
    return f.sort_values(["dip_score", "ml_prob"], ascending=False)


def _rr_series(table: pd.DataFrame) -> pd.Series:
    """Reward:risk per row for the filter.

    Prefers the scanner's ``best_rr`` column when present - that is the
    target ladder's best achievable R:R (TP1..TP3, scaling out), which is
    what the spec's 2.5:1 floor is meant to gate on. Falls back to the
    nearest-resistance approximation when the ladder is unavailable.
    NaN when no actionable entry/stop exists.
    """
    if "best_rr" in table.columns and table["best_rr"].notna().any():
        series = table["best_rr"].astype(float)
        return series.where(series > 0, float("nan"))

    rows = []
    for _, r in table.iterrows():
        ez = r.get("entry_zone")
        inv = r.get("invalidation")
        tgt = r.get("resistance")
        if ez is None or inv is None or tgt is None:
            rows.append(float("nan"))
            continue
        try:
            if isinstance(ez, str) and "-" in ez:
                lo, hi = ez.split("-", 1)
                entry = (float(lo) + float(hi)) / 2
            elif isinstance(ez, (list, tuple)) and len(ez) == 2:
                entry = (float(ez[0]) + float(ez[1])) / 2
            else:
                entry = float(ez)
            risk = abs(entry - float(inv))
            reward = abs(float(tgt) - entry)
            rows.append(reward / risk if risk > 0 else float("nan"))
        except (TypeError, ValueError):
            rows.append(float("nan"))
    return pd.Series(rows, index=table.index)


# ---------------------------------------------------------------------------
# Dedup state
# ---------------------------------------------------------------------------


def load_state(path: str = DEFAULT_STATE_FILE) -> set:
    """The deduped setup keys already alerted (set view - legacy format
    compatible). See ``load_state_with_meta`` for timestamps."""
    seen, _ = load_state_with_meta(path)
    return seen


def load_state_with_meta(path: str = DEFAULT_STATE_FILE) -> tuple:
    """``(seen, sent_at)`` from the state file.

    ``sent_at`` maps each key to the ISO timestamp it was first alerted
    (older state files have no timestamps - those keys are treated as
    never-expiring). Corrupt/missing files yield ``(set(), {})``.
    """
    if not Path(path).exists():
        return set(), {}
    try:
        data = json.loads(Path(path).read_text())
        return set(data.get("seen", [])), dict(data.get("sent_at", {}) or {})
    except Exception:
        return set(), {}


def save_state(
    seen: set, path: str = DEFAULT_STATE_FILE, sent_at: Optional[Dict] = None
) -> None:
    """Persist the dedup state. ``sent_at`` (key -> ISO timestamp of the
    first alert) is stored alongside so expired setups can be re-alerted
    later instead of being deduped forever."""
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(
        json.dumps({"seen": sorted(seen), "sent_at": dict(sent_at or {})}, indent=2)
    )


def purge_expired(seen: set, sent_at: Optional[Dict], max_age_days: float) -> tuple:
    """Drop keys alerted more than ``max_age_days`` ago.

    A setup key older than the expiry window is re-eligible for alerting
    (the signal may have expired and a fresh setup is a new decision).
    Keys without a recorded timestamp are kept (unknown age is not
    expired). Returns ``(purged_seen, purged_sent_at)``.
    """
    if max_age_days is None or max_age_days <= 0:
        return seen, dict(sent_at or {})
    sent_at = dict(sent_at or {})
    from datetime import datetime, timezone

    cutoff = datetime.now(timezone.utc).timestamp() - max_age_days * 86400.0
    keep = set()
    keep_ts = {}
    for key in seen:
        ts = sent_at.get(key)
        if ts is None:
            keep.add(key)  # unknown age -> never expires
            keep_ts[key] = ts
            continue
        try:
            t = datetime.fromisoformat(ts.replace("Z", "+00:00"))
            if t.timestamp() >= cutoff:
                keep.add(key)
                keep_ts[key] = ts
        except (TypeError, ValueError):
            keep.add(key)
            keep_ts[key] = ts
    return keep, keep_ts


def _setup_key(symbol: str, entry_zone) -> str:
    """Canonical dedup key: ``SYMBOL:lo-hi`` with 6-decimal prices.

    Normalizes both representations of the zone - the scanner's
    ``"1.10000-1.20000"`` string and a ``(lo, hi)`` tuple - to the same
    key so the same setup never re-alerts just because the format differs.
    """
    if entry_zone is None:
        return f"{symbol}:none"
    lo = hi = None
    if isinstance(entry_zone, str) and "-" in entry_zone:
        parts = entry_zone.split("-", 1)
        try:
            lo, hi = float(parts[0]), float(parts[1])
        except ValueError:
            return f"{symbol}:{entry_zone}"
    elif isinstance(entry_zone, (list, tuple)) and len(entry_zone) == 2:
        try:
            lo, hi = float(entry_zone[0]), float(entry_zone[1])
        except (TypeError, ValueError):
            return f"{symbol}:{entry_zone}"
    if lo is not None and hi is not None:
        return f"{symbol}:{lo:.6f}-{hi:.6f}"
    return f"{symbol}:{entry_zone}"


# ---------------------------------------------------------------------------
# Sizing + formatting
# ---------------------------------------------------------------------------


def size_setup(
    symbol: str,
    group: Optional[str],
    timeframe: str,
    equity: float = 250_000.0,
    risk_pct: float = 0.01,
    hold_bars: int = 10,
) -> Optional[Dict]:
    """Full detail + risk plan for one symbol (None if not actionable)."""
    try:
        df, report = load_symbol_report(symbol, group, timeframe)
    except Exception:
        return None
    if len(df) < 60:
        return None
    from src.risk.run import risk_plan_from_report

    plan = risk_plan_from_report(
        report, symbol, equity=equity, risk_pct=risk_pct, hold_bars=hold_bars
    )
    if not plan.get("setup"):
        return None
    return {"symbol": symbol, "report": report, "plan": plan}


def format_alert(setup: Dict) -> str:
    """One human-readable alert block for a sized setup."""
    s = setup["symbol"]
    report = setup["report"]
    plan = setup["plan"]
    st = plan["setup"]
    dip = report["dip"]
    macro = report.get("macro") or {}

    lines = [
        f"🟢 LONG {s} — BUY-THE-DIP CONFIRMED",
        f"   Close {report['last_close']:,.5f} · {report['last_date']}",
    ]
    lines.append(
        f"   Regime {report['regime']['regime']} · "
        f"bias {report['simple_bias']['interpretation']}"
    )

    ez = dip.get("entry_zone")
    if ez:
        lines.append(f"   Entry zone {ez[0]:,.5f} → {ez[1]:,.5f}")
    if st.get("stop"):
        lines.append(f"   Stop (invalidation) {st['stop']:,.5f}")
    if st.get("target"):
        lines.append(f"   Target {st['target']:,.5f} · R:R {st['rr']:.2f}")
    # spec #11 floor: show the achievable (ladder) R:R vs the floor and the
    # nearest single-target R:R for honesty.
    if st.get("rr_ok") is not None:
        mark = "✓" if st["rr_ok"] else "✗ below floor"
        lines.append(
            f"   R:R floor {st.get('min_rr', 2.5)}:1 → {mark} "
            f"(nearest target {st.get('rr_nearest', st['rr'])})"
        )

    ml = report.get("ml")
    if ml:
        lines.append(f"   ML bullish prob {ml['prob_pct']:.0f}% ({ml['label']})")
    if macro.get("bias"):
        b = macro["bias"]
        lines.append(f"   Macro {b['label']} (bias {b['bias']:+.2f})")
    if st.get("atr_pct"):
        lines.append(f"   ATR {st['atr_pct']:.2f}%")

    frac = next((r for r in plan.get("sizes", []) if r["method"] == "fractional"), None)
    if frac:
        lines.append(
            f"   Size {frac['qty']:,.0f} units · "
            f"risk ${frac['risk_usd']:,.0f} "
            f"({frac['risk_pct_equity']:.2f}% of equity) · "
            f"VaR95 1d ${frac['var_95_1bar']:,.0f}"
        )
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Short-side (Sell-the-Rally) filtering + alerts
# ---------------------------------------------------------------------------


def filter_short_signals(
    table: pd.DataFrame,
    min_rally_score: int = 5,
    require_confirmed: bool = True,
    max_bias: float = 4.0,
    min_ml_prob: Optional[float] = None,
    require_macro_pass: bool = True,
    min_rr: Optional[float] = None,
) -> pd.DataFrame:
    """
    Mirror of ``filter_signals`` for the Sell-the-Rally engine: high-
    conviction SHORT candidates (confirmed rally, bearish bias, macro
    gate PASS, ML probability above a threshold when available, minimum
    rally score). ``max_bias`` keeps rows whose bias is at most this
    (default +4 = anything) - pass 0 to require a bearish bias.

    The ML filter uses the DEDICATED short model's ``ml_short_prob``
    column (the scanner's ``predict_short_series`` output) when present,
    so a strong short-model read keeps the setup. Legacy rows with only
    ``ml_prob`` (the long model) fall back to the inverted long
    probability so pre-short-model tables still filter gracefully.
    """
    f = table.copy()
    f = f[f["rally_score"] >= min_rally_score]
    if require_confirmed and "rally_confirmed" in f:
        f = f[f["rally_confirmed"] == "Yes"]
    if "bias_score" in f:
        f = f[f["bias_score"] <= max_bias]
    if min_ml_prob is not None:
        if "ml_short_prob" in f and f["ml_short_prob"].notna().any():
            has_ml = f["ml_short_prob"].notna()
            f = f[~has_ml | (f["ml_short_prob"] >= min_ml_prob)]
        elif "ml_prob" in f:
            has_ml = f["ml_prob"].notna()
            f = f[~has_ml | (f["ml_prob"] <= (100.0 - min_ml_prob))]
    if require_macro_pass:
        gate_col = "macro_gate_short" if "macro_gate_short" in f else "macro_gate"
        if gate_col in f:
            no_macro = f[gate_col].isna()
            f = f[no_macro | (f[gate_col] == "PASS")]
    if min_rr is not None:
        rr = _short_rr_series(f)
        f = f[rr.isna() | (rr >= min_rr)]
    return f.sort_values(["rally_score", "bias_score"], ascending=False)


def _short_rr_series(table: pd.DataFrame) -> pd.Series:
    """Reward:risk per short row (entry -> support target vs entry -> stop).
    Prefers ``short_best_rr`` (ladder); falls back to the nearest-support
    approximation. NaN when no actionable short setup exists."""
    if "short_best_rr" in table.columns and table["short_best_rr"].notna().any():
        series = table["short_best_rr"].astype(float)
        return series.where(series > 0, float("nan"))
    rows = []
    for _, r in table.iterrows():
        ez = r.get("short_entry_zone")
        inv = r.get("short_invalidation")
        tgt = r.get("support")
        if ez is None or inv is None or tgt is None:
            rows.append(float("nan"))
            continue
        try:
            if isinstance(ez, str) and "-" in ez:
                lo, hi = ez.split("-", 1)
                entry = (float(lo) + float(hi)) / 2
            elif isinstance(ez, (list, tuple)) and len(ez) == 2:
                entry = (float(ez[0]) + float(ez[1])) / 2
            else:
                entry = float(ez)
            risk = abs(float(inv) - entry)
            reward = abs(entry - float(tgt))
            rows.append(reward / risk if risk > 0 else float("nan"))
        except (TypeError, ValueError):
            rows.append(float("nan"))
    return pd.Series(rows, index=table.index)


def size_short_setup(
    symbol: str,
    group: Optional[str],
    timeframe: str,
    equity: float = 250_000.0,
    risk_pct: float = 0.01,
    hold_bars: int = 10,
) -> Optional[Dict]:
    """Full detail + SHORT risk plan for one symbol (None if not actionable)."""
    try:
        df, report = load_symbol_report(symbol, group, timeframe)
    except Exception:
        return None
    if len(df) < 60:
        return None
    from src.risk.run import risk_plan_from_report_short

    plan = risk_plan_from_report_short(
        report, symbol, equity=equity, risk_pct=risk_pct, hold_bars=hold_bars
    )
    if not plan.get("setup"):
        return None
    return {"symbol": symbol, "report": report, "plan": plan}


def format_short_alert(setup: Dict) -> str:
    """One human-readable SHORT alert block for a sized rally setup."""
    s = setup["symbol"]
    report = setup["report"]
    plan = setup["plan"]
    st = plan["setup"]
    rally = report.get("rally") or {}
    macro = report.get("macro") or {}

    lines = [
        f"🔴 SHORT {s} — SELL-THE-RALLY CONFIRMED",
        f"   Close {report['last_close']:,.5f} · {report['last_date']}",
    ]
    lines.append(
        f"   Regime {report['regime']['regime']} · "
        f"bias {report['simple_bias']['interpretation']}"
    )

    ez = rally.get("entry_zone")
    if ez:
        lines.append(f"   Entry zone {ez[0]:,.5f} → {ez[1]:,.5f}")
    if st.get("stop"):
        lines.append(f"   Stop (above swing high) {st['stop']:,.5f}")
    if st.get("target"):
        lines.append(f"   Target {st['target']:,.5f} · R:R {st['rr']:.2f}")
    if st.get("rr_ok") is not None:
        mark = "✓" if st["rr_ok"] else "✗ below floor"
        lines.append(
            f"   R:R floor {st.get('min_rr', 2.5)}:1 → {mark} "
            f"(nearest target {st.get('rr_nearest', st['rr'])})"
        )

    ml = report.get("ml")
    if ml:
        lines.append(
            f"   ML bullish prob {ml['prob_pct']:.0f}% "
            f"({ml['label']}) → short edge when low"
        )
    if macro.get("bias"):
        b = macro["bias"]
        lines.append(f"   Macro {b['label']} (bias {b['bias']:+.2f})")
    if st.get("atr_pct"):
        lines.append(f"   ATR {st['atr_pct']:.2f}%")

    frac = next((r for r in plan.get("sizes", []) if r["method"] == "fractional"), None)
    if frac:
        lines.append(
            f"   Size {frac['qty']:,.0f} units (short) · "
            f"risk ${frac['risk_usd']:,.0f} "
            f"({frac['risk_pct_equity']:.2f}% of equity) · "
            f"VaR95 1d ${frac['var_95_1bar']:,.0f}"
        )
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# The full pass
# ---------------------------------------------------------------------------


def _data_age(
    symbols: List[str], group: Optional[str], timeframe: str, data_dir: str
) -> float:
    """Max calendar-day age of the watchlist's parquet files (0 if none)."""
    from src.data.resolver import find_local

    ages = []
    for sym in symbols:
        path = find_local(sym, timeframe, data_dir, group) or _data_path(
            sym, data_dir, group, timeframe
        )
        age = staleness_days(path)
        if age is not None:
            ages.append(age)
    return max(ages) if ages else 0.0


def live_signal_pass(
    group: Optional[str] = DEFAULT_GROUP,
    timeframe: str = DEFAULT_TIMEFRAME,
    symbols: Optional[List[str]] = None,
    data_dir: str = "data/raw",
    equity: float = 250_000.0,
    risk_pct: float = 0.01,
    min_dip_score: int = 5,
    min_ml_prob: Optional[float] = None,
    require_confirmed: bool = True,
    require_macro_pass: bool = True,
    min_rr: Optional[float] = None,
    state_file: str = DEFAULT_STATE_FILE,
    max_stale_days: float = DEFAULT_MAX_STALE_DAYS,
    dry_run: bool = False,
    signal_expiry_days: float = 30.0,
) -> Dict:
    """
    Run one full signal pass: scan -> filter -> size -> dedup.

    ``dry_run=True`` never advances the dedup state (nothing is marked as
    already-alerted), so repeated calls behave like an idempotent read-only
    pass - used by the API's GET endpoint. ``signal_expiry_days``
    re-eligibilizes setups first alerted more than N days ago (a stale
    setup is a stale decision; a fresh setup is a new alert).

    Returns ``{date, group, timeframe, scanned, candidates, new_alerts,
    skipped_dup, macro}``. ``new_alerts`` is a list of ``{symbol, text,
    key, report}`` - ready to send.
    """
    # On-demand: missing watchlist symbols are fetched (MT5 -> Yahoo) and
    # cached into their classified group folder before analysis.
    table = scan_universe(
        data_dir=data_dir,
        group=group,
        timeframe=timeframe,
        symbols=symbols,
        fetch_mt5=True,
    )

    cands = filter_signals(
        table,
        min_dip_score=min_dip_score,
        require_confirmed=require_confirmed,
        min_ml_prob=min_ml_prob,
        require_macro_pass=require_macro_pass,
        min_rr=min_rr,
    )

    seen, sent_at = load_state_with_meta(state_file)
    seen, sent_at = purge_expired(seen, sent_at, signal_expiry_days)
    new_alerts = []
    skipped = 0
    sizing_failed = 0
    for _, row in cands.head(20).iterrows():
        key = _setup_key(row["symbol"], row.get("entry_zone"))
        if key in seen:
            skipped += 1
            continue
        setup = size_setup(
            row["symbol"], group, timeframe, equity=equity, risk_pct=risk_pct
        )
        if setup is None:
            sizing_failed += 1
            continue
        text = format_alert(setup)
        new_alerts.append(
            {
                "symbol": row["symbol"],
                "key": key,
                "text": text,
                "report": setup["report"],
            }
        )

    if new_alerts and not dry_run:
        from datetime import datetime, timezone

        now = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
        for a in new_alerts:
            seen.add(a["key"])
            sent_at[a["key"]] = now
        save_state(seen, state_file, sent_at)

    # Macro snapshot for the message header (graceful None).
    macro = None
    try:
        row = latest_macro_scores(data_dir)
        if row is not None:
            macro = macro_regime(row.iloc[-1].to_dict())
    except Exception:
        macro = None

    # Data freshness: warn when the watchlist bars are older than the
    # threshold - the pass still runs, but the briefing flags it.
    scanned_syms = symbols if symbols else table["symbol"].tolist()
    data_age = _data_age(scanned_syms, group, timeframe, data_dir)
    stale = data_age > max_stale_days

    return {
        "date": str(pd.Timestamp.today().date()),
        "group": group or "default",
        "timeframe": timeframe,
        "scanned": len(table),
        "candidates": len(cands),
        "new_alerts": new_alerts,
        "skipped_dup": skipped,
        "sizing_failed": sizing_failed,
        "macro": macro,
        "data_age_days": round(data_age, 1),
        "data_stale": stale,
    }


def live_short_pass(
    group: Optional[str] = DEFAULT_GROUP,
    timeframe: str = DEFAULT_TIMEFRAME,
    symbols: Optional[List[str]] = None,
    data_dir: str = "data/raw",
    equity: float = 250_000.0,
    risk_pct: float = 0.01,
    min_rally_score: int = 5,
    min_ml_prob: Optional[float] = None,
    require_confirmed: bool = True,
    require_macro_pass: bool = True,
    min_rr: Optional[float] = None,
    state_file: str = DEFAULT_STATE_FILE,
    max_stale_days: float = DEFAULT_MAX_STALE_DAYS,
    dry_run: bool = False,
    signal_expiry_days: float = 30.0,
) -> Dict:
    """
    Short-side mirror of ``live_signal_pass``: scan -> filter_short ->
    size -> dedup, emitting SELL-THE-RALLY alerts.

    Returns the same shape as the long pass; ``new_alerts`` items carry
    ``direction: "short"`` and a ``rally`` report.
    """
    table = scan_universe(
        data_dir=data_dir,
        group=group,
        timeframe=timeframe,
        symbols=symbols,
        fetch_mt5=True,
    )

    cands = filter_short_signals(
        table,
        min_rally_score=min_rally_score,
        require_confirmed=require_confirmed,
        min_ml_prob=min_ml_prob,
        require_macro_pass=require_macro_pass,
        min_rr=min_rr,
    )

    seen, sent_at = load_state_with_meta(state_file)
    seen, sent_at = purge_expired(seen, sent_at, signal_expiry_days)
    new_alerts = []
    skipped = 0
    sizing_failed = 0
    for _, row in cands.head(20).iterrows():
        key = _setup_key(row["symbol"], row.get("short_entry_zone"))
        if key in seen:
            skipped += 1
            continue
        setup = size_short_setup(
            row["symbol"], group, timeframe, equity=equity, risk_pct=risk_pct
        )
        if setup is None:
            sizing_failed += 1
            continue
        text = format_short_alert(setup)
        new_alerts.append(
            {
                "symbol": row["symbol"],
                "key": key,
                "text": text,
                "report": setup["report"],
                "direction": "short",
            }
        )

    if new_alerts and not dry_run:
        from datetime import datetime, timezone

        now = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
        for a in new_alerts:
            seen.add(a["key"])
            sent_at[a["key"]] = now
        save_state(seen, state_file, sent_at)

    macro = None
    try:
        row = latest_macro_scores(data_dir)
        if row is not None:
            macro = macro_regime(row.iloc[-1].to_dict())
    except Exception:
        macro = None

    scanned_syms = symbols if symbols else table["symbol"].tolist()
    data_age = _data_age(scanned_syms, group, timeframe, data_dir)
    stale = data_age > max_stale_days

    return {
        "date": str(pd.Timestamp.today().date()),
        "group": group or "default",
        "timeframe": timeframe,
        "scanned": len(table),
        "candidates": len(cands),
        "new_alerts": new_alerts,
        "skipped_dup": skipped,
        "sizing_failed": sizing_failed,
        "macro": macro,
        "data_age_days": round(data_age, 1),
        "data_stale": stale,
        "direction": "short",
    }


def format_briefing(result: Dict) -> str:
    """A full pass rendered as one alert message (header + each setup)."""
    now = datetime.now().strftime("%Y-%m-%d %H:%M")
    lines = [
        f"📡 NEXUSQUANT LIVE SIGNALS — {now}",
        f"   {result['group']} · {result['timeframe']} · "
        f"{result['scanned']} symbols scanned",
    ]

    if result.get("data_stale"):
        lines.append(
            f"   ⚠️ DATA STALE: bars {result.get('data_age_days')}d old "
            f"- run `python -m src.data.update --group "
            f"{result['group']}` first"
        )
    else:
        lines.append(f"   Data as of {result.get('data_age_days', 0)}d ago")

    m = result.get("macro")
    if m:
        lines.append(
            f"   Macro: USD {m['usd']} · Risk {m['risk']} · "
            f"Rates {m['rates']} · composite {m['composite']:+.2f}"
        )

    alerts = result["new_alerts"]
    if not alerts:
        lines.append(
            "\nNo new high-conviction setups. "
            f"({result['candidates']} candidates, "
            f"{result['skipped_dup']} already alerted, "
            f"{result.get('sizing_failed', 0)} no risk setup"
            + (
                f", {result['conflicts_resolved']} dual-side conflict "
                "arbitrated to FLAT)"
                if result.get("conflicts_resolved")
                else ")"
            )
        )
        return "\n".join(lines)

    lines.append(f"\n🎯 {len(alerts)} new setup(s)")
    for a in alerts:
        lines.append("\n" + a["text"])
    foot = "— filters: dip confirmed + macro PASS + sized setup —"
    if result.get("conflicts_resolved"):
        foot = (
            f"— filters: dip/rally confirmed + macro PASS + sized setup; "
            f"{result['conflicts_resolved']} dual-side conflict(s) "
            f"arbitrated by EV —"
        )
    lines.append("\n" + foot)
    # Stage-10 portfolio selection footer: show how many proposed orders
    # survived the cluster/concurrent/heat caps (and why the rest dropped).
    pf = result.get("portfolio") or {}
    pf_summary = (pf.get("summary") or {}).get("n_rejected", 0)
    if pf_summary:
        from src.live.portfolio import format_portfolio_summary

        txt = format_portfolio_summary(pf)
        if txt:
            lines.append(txt)
    return "\n".join(lines)
