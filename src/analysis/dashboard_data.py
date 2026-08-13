"""
NexusQuant - Dashboard data layer & chart builders (Streamlit-free).

Pure functions used by `dashboard.py`: group discovery, universe / detail
loaders, and Plotly chart builders. Kept free of Streamlit so the whole
layer is unit-testable headless (``tests/test_dashboard_data.py``).

Loaders are **local-only by default** (``allow_fetch=False``) so the
dashboard never contends with the MT5 bridge / backfill; the API passes
``allow_fetch=True`` to resolve missing symbols on demand (local -> MT5
-> Yahoo).
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

sys.path.append(str(Path(__file__).parent.parent.parent))

from src.data.loader import clean_data, load_data
from src.features.indicators import add_all_indicators
from src.features.regime import detect_regime
from src.features.dip import detect_dip
from src.analysis.report import generate_full_report
from src.analysis.scanner import (
    _directional_score,
    _load_trigger,
    discover_symbols,
    scan_universe,
)
from src.backtest.engine import BacktestParams, run_backtest
from src.backtest.signals import dip_signal_series

DEFAULT_DATA_DIR = "data/raw"

# ---------------------------------------------------------------------------
# Colours (kept in sync with the dark theme)
# ---------------------------------------------------------------------------

REGIME_COLORS = {
    "Bull Trend": "#26a69a",
    "Bear Trend": "#ef5350",
    "Range / Chop": "#f0b90b",
    "High Volatility": "#ab47bc",
}

DIP_COLORS = {
    "Confirmed": "#22c55e",
    "In Pullback": "#f59e0b",
    "No Uptrend": "#64748b",
    "Not a Dip": "#94a3b8",
    "Support Broken": "#ef4444",
    "Deep Pullback": "#f97316",
}

PALETTE = [
    "#f0b90b",
    "#42a5f5",
    "#26a69a",
    "#ef5350",
    "#ab47bc",
    "#ffa726",
    "#26c6da",
    "#d4e157",
]


def regime_color(regime: str) -> str:
    return REGIME_COLORS.get(regime, "#78909c")


def dip_color(stage: str) -> str:
    return DIP_COLORS.get(stage, "#94a3b8")


# ---------------------------------------------------------------------------
# Group discovery
# ---------------------------------------------------------------------------


def discover_groups(data_dir: str = DEFAULT_DATA_DIR) -> List[Dict]:
    """
    List every scannable dataset in ``data_dir`` as
    ``{label, group, timeframe, n}``.

    Handles both layouts:
      * flat groups   - ``data/raw/full_fx/*_D1.parquet``
      * nested groups - ``data/raw/mt5/D1/*.parquet``
    plus the top-level FX majors (files directly in ``data/raw``).
    """
    base = Path(data_dir)
    groups: List[Dict] = []

    # The data folder is organised purely as asset-class group folders with
    # all timeframes flat inside (no top-level files) since the cleanup.
    for sub in sorted(p for p in base.iterdir() if p.is_dir()):
        for tf in ("D1", "H4", "H1"):
            files = sorted(sub.glob(f"*_{tf}.parquet"))
            if files:
                groups.append(
                    {
                        "label": f"{sub.name} · {tf}",
                        "group": sub.name,
                        "timeframe": tf,
                        "n": len(files),
                    }
                )
        for nested in sorted(p for p in sub.iterdir() if p.is_dir()):
            for tf in ("D1", "H4", "H1"):
                files = sorted(nested.glob(f"*_{tf}.parquet"))
                if files:
                    groups.append(
                        {
                            "label": f"{sub.name}/{nested.name}",
                            "group": f"{sub.name}/{nested.name}",
                            "timeframe": tf,
                            "n": len(files),
                        }
                    )

    groups.sort(key=lambda g: g["label"])
    return groups


def get_symbols(
    group: Optional[str], timeframe: str, data_dir: str = DEFAULT_DATA_DIR
) -> List[str]:
    """Symbols available in a (group, timeframe) dataset."""
    return discover_symbols(data_dir, group, timeframe)


# ---------------------------------------------------------------------------
# Loaders (local files only)
# ---------------------------------------------------------------------------


def load_universe(
    group: Optional[str], timeframe: str, data_dir: str = DEFAULT_DATA_DIR
) -> pd.DataFrame:
    """Ranked universe table for a (group, timeframe) dataset."""
    return scan_universe(
        data_dir=data_dir, group=group, timeframe=timeframe, fetch_mt5=False
    )


def load_symbol_report(
    symbol: str,
    group: Optional[str],
    timeframe: str,
    data_dir: str = DEFAULT_DATA_DIR,
    allow_fetch: bool = False,
) -> Tuple[pd.DataFrame, Dict]:
    """
    Full pipeline for one symbol: returns ``(df_with_indicators, report)``.

    ``allow_fetch`` enables the on-demand resolver (local -> MT5 -> Yahoo)
    for missing files; the dashboard keeps it False so it stays offline and
    fast, the API turns it on so any ticker works.

    The report's Buy-the-Dip section is enriched with the local H4/H1
    momentum trigger when one is available (same behaviour as the scanner).
    """
    from src.data.resolver import effective_group, resolve_symbol_data

    path = resolve_symbol_data(
        symbol,
        timeframe,
        data_dir,
        group,
        allow_mt5=allow_fetch,
        allow_yahoo=allow_fetch,
    )
    eff_group = effective_group(path, data_dir, group)
    df = load_data(path, symbol=symbol)
    df = clean_data(df)
    if df.empty:
        raise ValueError(f"Symbol {symbol}: no valid rows after cleaning")
    df = add_all_indicators(df)
    df = detect_regime(df)
    report = generate_full_report(df, symbol=symbol, group=eff_group, data_dir=data_dir)

    trigger = _load_trigger(symbol, data_dir, eff_group, timeframe)
    if trigger is not None:
        report["dip"] = detect_dip(df, trigger_df=trigger, levels=report["levels"])
    return df, report


def directional_bias(df: pd.DataFrame) -> Dict:
    """Directional bias {-4..+4, label} for the latest bar (scanner logic)."""
    return _directional_score(df)


# ---------------------------------------------------------------------------
# Chart builders (Plotly, dark theme)
# ---------------------------------------------------------------------------


def build_price_chart(df: pd.DataFrame, report: Dict, lookback: int = 180) -> go.Figure:
    """Candlestick + MA overlay + nearest S/R + entry zone + invalidation."""
    d = df.tail(lookback)
    fig = make_subplots(
        rows=2,
        cols=1,
        shared_xaxes=True,
        row_heights=[0.75, 0.25],
        vertical_spacing=0.03,
    )

    fig.add_trace(
        go.Candlestick(
            x=d.index,
            open=d["open"],
            high=d["high"],
            low=d["low"],
            close=d["close"],
            name="OHLC",
            increasing_line_color="#26a69a",
            decreasing_line_color="#ef5350",
        ),
        row=1,
        col=1,
    )

    for ma, color, width in (
        ("sma_50", "#f0b90b", 1.2),
        ("sma_100", "#42a5f5", 1.2),
        ("sma_200", "#ab47bc", 1.4),
    ):
        if ma in d.columns:
            fig.add_trace(
                go.Scatter(
                    x=d.index,
                    y=d[ma],
                    name=ma.upper(),
                    line=dict(color=color, width=width),
                ),
                row=1,
                col=1,
            )

    lv = report.get("levels", {})
    ns = lv.get("nearest_support") or {}
    nr = lv.get("nearest_resistance") or {}
    if ns.get("price"):
        fig.add_hline(
            y=ns["price"],
            line_dash="dot",
            line_color="#26a69a",
            annotation_text=f"S {ns['price']:,.5f}",
            row=1,
            col=1,
        )
    if nr.get("price"):
        fig.add_hline(
            y=nr["price"],
            line_dash="dot",
            line_color="#ef5350",
            annotation_text=f"R {nr['price']:,.5f}",
            row=1,
            col=1,
        )

    dip = report.get("dip", {})
    ez = dip.get("entry_zone")
    if ez:
        fig.add_hrect(
            y0=ez[0],
            y1=ez[1],
            fillcolor="#26a69a",
            opacity=0.10,
            line_width=0,
            row=1,
            col=1,
            annotation_text=f"Entry {ez[0]:,.5f}–{ez[1]:,.5f}",
        )
    inv = dip.get("invalidation_level")
    if inv:
        fig.add_hline(
            y=inv,
            line_dash="dash",
            line_color="#ef5350",
            annotation_text=f"Invalidation {inv:,.5f}",
            row=1,
            col=1,
        )

    vol_colors = np.where(d["close"] >= d["open"], "#26a69a", "#ef5350")
    fig.add_trace(
        go.Bar(
            x=d.index,
            y=d["volume"],
            name="Volume",
            marker_color=vol_colors,
            opacity=0.55,
        ),
        row=2,
        col=1,
    )

    fig.update_layout(
        template="plotly_dark",
        height=620,
        legend=dict(orientation="h", yanchor="bottom", y=1.02),
        margin=dict(t=40, b=10),
        xaxis_rangeslider_visible=False,
    )
    return fig


def build_momentum_chart(df: pd.DataFrame, lookback: int = 240) -> go.Figure:
    """Price + RSI(30/70 bands) + MACD/signal/histogram subplots."""
    d = df.tail(lookback)
    fig = make_subplots(
        rows=3,
        cols=1,
        shared_xaxes=True,
        row_heights=[0.4, 0.3, 0.3],
        vertical_spacing=0.05,
    )

    fig.add_trace(
        go.Scatter(
            x=d.index, y=d["close"], name="Close", line=dict(color="#f0b90b", width=1.4)
        ),
        row=1,
        col=1,
    )

    fig.add_trace(
        go.Scatter(x=d.index, y=d["rsi_14"], name="RSI 14", line=dict(color="#42a5f5")),
        row=2,
        col=1,
    )
    fig.add_hline(y=70, line_dash="dot", line_color="#ef5350", row=2, col=1)
    fig.add_hline(y=30, line_dash="dot", line_color="#26a69a", row=2, col=1)
    fig.add_hline(y=50, line_dash="dot", line_color="#4b5563", row=2, col=1)

    fig.add_trace(
        go.Scatter(x=d.index, y=d["macd"], name="MACD", line=dict(color="#f0b90b")),
        row=3,
        col=1,
    )
    fig.add_trace(
        go.Scatter(
            x=d.index, y=d["macd_signal"], name="Signal", line=dict(color="#ef5350")
        ),
        row=3,
        col=1,
    )
    hist_colors = np.where(d["macd_hist"] >= 0, "#26a69a", "#ef5350")
    fig.add_trace(
        go.Bar(
            x=d.index,
            y=d["macd_hist"],
            name="Histogram",
            marker_color=hist_colors,
            opacity=0.6,
        ),
        row=3,
        col=1,
    )

    fig.update_layout(
        template="plotly_dark",
        height=700,
        legend=dict(orientation="h", yanchor="bottom", y=1.02),
        margin=dict(t=40, b=10),
    )
    return fig


def build_compare_chart(
    frames: List[pd.DataFrame], symbols: List[str], lookback: int = 250
) -> go.Figure:
    """Normalized close (base = 100) for several symbols, one line each."""
    fig = go.Figure()
    for i, (sym, df) in enumerate(zip(symbols, frames, strict=True)):
        d = df.tail(lookback)
        if d.empty:
            continue
        base = float(d["close"].iloc[0])
        fig.add_trace(
            go.Scatter(
                x=d.index,
                y=d["close"] / base * 100,
                name=sym,
                line=dict(color=PALETTE[i % len(PALETTE)], width=1.6),
            )
        )
    fig.update_layout(
        template="plotly_dark",
        height=460,
        yaxis_title="Normalized (base = 100)",
        legend=dict(orientation="h", yanchor="bottom", y=1.02),
        margin=dict(t=40, b=10),
    )
    return fig


def build_bias_chart(records: List[Dict]) -> go.Figure:
    """Bar chart of directional bias per symbol, coloured by sign."""
    syms = [r["symbol"] for r in records]
    scores = [float(r["bias_score"]) for r in records]
    colors = ["#26a69a" if s > 0 else "#ef5350" if s < 0 else "#f0b90b" for s in scores]
    fig = go.Figure(
        go.Bar(
            x=syms, y=scores, marker_color=colors, text=scores, textposition="outside"
        )
    )
    fig.update_layout(
        template="plotly_dark",
        height=360,
        yaxis_title="Bias score (−4…+4)",
        margin=dict(t=30, b=10),
    )
    return fig


def build_dip_chart(records: List[Dict]) -> go.Figure:
    """Bar chart of dip score per symbol, coloured by stage."""
    syms = [r["symbol"] for r in records]
    scores = [float(r["dip_score"]) for r in records]
    colors = [dip_color(r["dip_stage"]) for r in records]
    fig = go.Figure(
        go.Bar(
            x=syms, y=scores, marker_color=colors, text=scores, textposition="outside"
        )
    )
    fig.update_layout(
        template="plotly_dark",
        height=360,
        yaxis_title="Dip score (0–8)",
        margin=dict(t=30, b=10),
    )
    return fig


def build_heatmap(table: pd.DataFrame, max_rows: int = 60) -> go.Figure:
    """Min-max normalised feature heatmap (bias, ADX, RSI, dip, ATR%)."""
    t = table.head(max_rows)
    cols = ["bias_score", "adx", "rsi_14", "dip_score", "atr_pct"]
    z = t[cols].astype(float)
    norm = z.apply(
        lambda c: (
            (c - c.min()) / (c.max() - c.min())
            if c.max() > c.min()
            else pd.Series(0.5, index=c.index)
        )
    )
    text = z.round(2).astype(str).values
    fig = go.Figure(
        go.Heatmap(
            z=norm.values,
            x=cols,
            y=t["symbol"].tolist(),
            text=text,
            texttemplate="%{text}",
            colorscale="balance",
            zmid=0.5,
            hovertemplate="%{y}<br>%{x}: %{text}<extra></extra>",
        )
    )
    fig.update_layout(
        template="plotly_dark",
        height=max(300, 24 * len(t)),
        margin=dict(t=20, b=10),
        yaxis=dict(autorange="reversed"),
    )
    return fig


def build_regime_chart(table: pd.DataFrame) -> go.Figure:
    """Regime distribution bar chart."""
    counts = table["regime"].value_counts()
    colors = [regime_color(r) for r in counts.index]
    fig = go.Figure(
        go.Bar(
            x=counts.index,
            y=counts.values,
            marker_color=colors,
            text=counts.values,
            textposition="outside",
        )
    )
    fig.update_layout(
        template="plotly_dark",
        height=360,
        xaxis_title="Regime",
        yaxis_title="Symbols",
        margin=dict(t=30, b=10),
    )
    return fig


# ---------------------------------------------------------------------------
# Backtesting
# ---------------------------------------------------------------------------


def run_symbol_backtest(
    symbol: str,
    group: Optional[str],
    timeframe: str,
    data_dir: str = DEFAULT_DATA_DIR,
    risk_pct: float = 0.01,
    rr_fallback: float = 2.0,
    max_hold: int = 20,
    entry_type: str = "limit",
):
    """Run the causal Buy-the-Dip backtest on a single symbol."""
    df, report = load_symbol_report(symbol, group, timeframe, data_dir)
    if len(df) < 60:
        raise ValueError(f"{symbol}: only {len(df)} bars")
    signal = dip_signal_series(df)
    bars_per_year = {"D1": 252, "H4": 1512, "H1": 6048}.get(timeframe.upper(), 252)
    params = BacktestParams(
        risk_pct=risk_pct,
        rr_fallback=rr_fallback,
        max_hold=max_hold,
        entry_type=entry_type,
        bars_per_year=bars_per_year,
    )
    return run_backtest(signal, df, params, symbol=symbol)


def build_equity_chart(result) -> go.Figure:
    """Equity curve with drawdown shading."""
    eq = result.equity
    curve = eq.values
    peak = np.maximum.accumulate(curve)
    dd = np.where(peak > 0, (curve - peak) / peak * 100, 0.0)

    fig = make_subplots(
        rows=2,
        cols=1,
        shared_xaxes=True,
        row_heights=[0.72, 0.28],
        vertical_spacing=0.05,
    )
    fig.add_trace(
        go.Scatter(
            x=eq.index,
            y=eq.values,
            name="Equity",
            line=dict(color="#f0b90b", width=1.6),
        ),
        row=1,
        col=1,
    )
    fig.add_trace(
        go.Scatter(
            x=eq.index,
            y=dd,
            name="Drawdown %",
            fill="tozeroy",
            line=dict(color="#ef5350"),
            fillcolor="rgba(239,83,80,0.25)",
        ),
        row=2,
        col=1,
    )

    # Trade markers on the equity curve
    x_win, y_win, x_loss, y_loss = [], [], [], []
    for t in result.trades:
        loc = eq.index.get_loc(t.exit_time)
        (x_win if t.pnl > 0 else x_loss).append(t.exit_time)
        (y_win if t.pnl > 0 else y_loss).append(eq.values[loc])
    if x_win:
        fig.add_trace(
            go.Scatter(
                x=x_win,
                y=y_win,
                mode="markers",
                name="Wins",
                marker=dict(color="#22c55e", size=7, symbol="triangle-up"),
            ),
            row=1,
            col=1,
        )
    if x_loss:
        fig.add_trace(
            go.Scatter(
                x=x_loss,
                y=y_loss,
                mode="markers",
                name="Losses",
                marker=dict(color="#ef4444", size=7, symbol="triangle-down"),
            ),
            row=1,
            col=1,
        )

    fig.update_layout(
        template="plotly_dark",
        height=520,
        legend=dict(orientation="h", yanchor="bottom", y=1.02),
        margin=dict(t=40, b=10),
    )
    fig.update_yaxes(title_text="Equity", row=1, col=1)
    fig.update_yaxes(title_text="Drawdown %", row=2, col=1)
    return fig
