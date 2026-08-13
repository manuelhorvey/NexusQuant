"""
NexusQuant - Streamlit Dashboard
================================

Institutional multi-asset dashboard on top of the regime / confluence /
buy-the-dip engines.

Run:
    streamlit run dashboard.py
    # or
    python -m streamlit run dashboard.py

Data layer & charts live in src/analysis/dashboard_data.py (testable).
All analysis is done on local files only (no MT5 fetches).
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import streamlit as st

sys.path.append(str(Path(__file__).parent))

from src.analysis.dashboard_data import (
    build_bias_chart,
    build_compare_chart,
    build_dip_chart,
    build_equity_chart,
    build_heatmap,
    build_momentum_chart,
    build_price_chart,
    build_regime_chart,
    dip_color,
    directional_bias,
    discover_groups,
    get_symbols,
    load_symbol_report,
    load_universe,
    regime_color,
    run_symbol_backtest,
)
from src.features.divergence import format_divergence

st.set_page_config(
    page_title="NexusQuant — Institutional Dashboard", page_icon="📊", layout="wide"
)

# ---------------------------------------------------------------------------
# Theme
# ---------------------------------------------------------------------------

st.markdown(
    """
<style>
    .stApp { background: #0D1117; }
    [data-testid="stSidebar"] {
        background: #0A0E14; border-right: 1px solid #1B2330;
    }
    div[data-testid="stMetric"] {
        background: #161B22; border: 1px solid #232B3A;
        border-radius: 10px; padding: 10px 14px;
    }
    div[data-testid="stMetricLabel"] { color: #8B949E; }
    h1, h2, h3 { letter-spacing: 0.3px; }
    .nx-banner {
        border-radius: 10px; padding: 14px 20px; margin: 8px 0 18px 0;
        font-size: 1.35rem; font-weight: 700;
    }
    .nx-banner small {
        font-weight: 400; font-size: 0.95rem; color: #8B949E;
    }
    .nx-chip {
        display: inline-block; padding: 2px 10px; border-radius: 999px;
        font-size: 0.8rem; font-weight: 600;
    }
</style>
""",
    unsafe_allow_html=True,
)

# ---------------------------------------------------------------------------
# Caching (local files only, 15 min TTL)
# ---------------------------------------------------------------------------


@st.cache_data(ttl=900, show_spinner=False)
def cached_universe(group, timeframe, data_dir="data/raw"):
    return load_universe(group, timeframe, data_dir)


@st.cache_data(ttl=900, show_spinner=False)
def cached_detail(symbol, group, timeframe, data_dir="data/raw"):
    return load_symbol_report(symbol, group, timeframe, data_dir)


# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------

with st.sidebar:
    st.markdown("# 📊 NexusQuant")
    st.caption("Institutional multi-asset analysis")

    groups = discover_groups()
    if not groups:
        st.error("No data found under `data/raw/`.")
        st.stop()

    labels = [f"{g['label']} ({g['n']})" for g in groups]
    default_idx = next(
        (i for i, g in enumerate(groups) if g["label"] == "full_fx · D1"), 0
    )
    choice = st.selectbox("Data group", labels, index=default_idx)
    g = groups[labels.index(choice)]

    page = st.radio(
        "Page",
        [
            "🌐 Universe Ranking",
            "🔍 Symbol Detail",
            "⚖️ Compare Symbols",
            "🧪 Backtest",
            "🛡️ Risk",
        ],
    )

    st.divider()
    if st.button("🔄 Refresh data", use_container_width=True):
        cached_universe.clear()
        cached_detail.clear()
        st.rerun()

    st.caption(
        f"Group: `{g['group'] or 'majors'}` · `{g['timeframe']}` · {g['n']} symbols"
    )
    st.caption("Local files only — no MT5 fetches.")


def _dip_banner_html(report) -> str:
    dip = report["dip"]
    stage = dip["dip_stage"]
    color = dip_color(stage)
    return (
        f'<div class="nx-banner" style="background:{color}1A;'
        f"border:1px solid {color};border-left:6px solid {color};"
        f'color:{color};">{stage.upper()}'
        f" <small>· Buy-the-Dip score {dip['dip_score']}/8 · "
        f"{report['symbol']} @ {report['last_close']:,.5f} · "
        f"{report['last_date']}</small></div>"
    )


def _dict_table(data: dict, cols: tuple = ("Metric", "Value")) -> pd.DataFrame:
    return pd.DataFrame({cols[0]: list(data.keys()), cols[1]: list(data.values())})


def _rally_banner_html(report) -> str:
    rally = report.get("rally") or {}
    stage = rally.get("rally_stage") or "—"
    color = "#ef5350" if stage == "Confirmed" else "#7c8794"
    return (
        f'<div class="nx-banner" style="background:{color}1A;'
        f"border:1px solid {color};border-left:6px solid {color};"
        f'color:{color};">🔻 SELL-THE-RALLY {stage.upper()}'
        f" <small>· score {rally.get('rally_score', '-')}/8 · "
        f"{report['symbol']} · short setup when confirmed</small></div>"
    )


# ---------------------------------------------------------------------------
# Page 1 — Universe ranking
# ---------------------------------------------------------------------------


def render_universe(g: dict) -> None:
    st.title("🌐 Universe Ranking")

    with st.spinner(
        f"Running the full pipeline on {g['n']} symbols (first load only)…"
    ):
        table = cached_universe(g["group"], g["timeframe"])

    st.caption(
        f"Data as of **{table['date'].max()}** · ranked by bias then "
        f"trend strength · {len(table)} symbols analysed"
    )

    c1, c2, c3, c4 = st.columns(4)
    with c1:
        search = st.text_input(
            "🔎 Symbol filter", "", placeholder="e.g. USD, XAU, AAPL"
        )
    with c2:
        regimes = st.multiselect("Regime", sorted(table["regime"].unique()), default=[])
    with c3:
        stages = st.multiselect(
            "Dip status", sorted(table["dip_stage"].unique()), default=[]
        )
    with c4:
        bias_dir = st.selectbox("Bias", ["All", "Bullish", "Neutral", "Bearish"])
    c5, c6, c7, c8 = st.columns(4)
    with c5:
        macro_g = st.selectbox("Macro gate", ["All", "PASS", "BLOCKED"])
    with c6:
        min_score = st.slider("Min dip score", 0, 8, 0)
    with c7:
        top = st.slider("Top N", min(10, len(table)), len(table), min(60, len(table)))
    with c8:
        sort_by = st.selectbox(
            "Sort by", ["Rank", "Dip score", "ATR %", "Close", "Macro bias"]
        )
    c9, c10, c11 = st.columns(3)
    with c9:
        rally_stages = st.multiselect(
            "Short setups (rally stage)",
            sorted(table["rally_stage"].dropna().unique()),
            default=[],
        )
    with c10:
        min_rally = st.slider("Min rally score", 0, 8, 0)
    with c11:
        st.caption("")
        st.caption(
            "🔻 Short columns (rally_score / stage / short ML prob) "
            "live in the table below."
        )
    st.caption(
        "🌍 Macro overlay: top-down USD / risk / rates backdrop. "
        "Gate **BLOCKED** = strong macro headwind filters the dip signal."
    )

    f = table.copy()
    if search:
        f = f[f["symbol"].str.contains(search.upper(), na=False)]
    if regimes:
        f = f[f["regime"].isin(regimes)]
    if stages:
        f = f[f["dip_stage"].isin(stages)]
    if rally_stages:
        f = f[f["rally_stage"].isin(rally_stages)]
    f = f[f["rally_score"] >= min_rally]
    if bias_dir == "Bullish":
        f = f[f["bias_score"] > 0]
    elif bias_dir == "Bearish":
        f = f[f["bias_score"] < 0]
    else:
        f = f[f["bias_score"] == 0]
    if macro_g != "All":
        f = f[f["macro_gate"] == macro_g]
    f = f[f["dip_score"] >= min_score]

    # Sort first, then take the top N, so "Sort by" is honoured.
    if sort_by == "Dip score":
        f = f.sort_values("dip_score", ascending=False)
    elif sort_by == "ATR %":
        f = f.sort_values("atr_pct", ascending=False)
    elif sort_by == "Close":
        f = f.sort_values("close", ascending=False)
    elif sort_by == "Macro bias":
        f = f.sort_values("macro_bias", ascending=False)
    else:
        f = f.sort_values(["rank", "bias_score", "adx"], ascending=[True, False, False])
    f = f.head(top)

    m1, m2, m3, m4, m5 = st.columns(5)
    m1.metric("Symbols", len(f))
    m2.metric("Bull Trend", int((f["regime"] == "Bull Trend").sum()))
    m3.metric("Dips confirmed", int((f["dip_stage"] == "Confirmed").sum()))
    m4.metric("Avg bias", f"{f['bias_score'].mean():+.2f}")
    m5.metric("Avg ATR %", f"{f['atr_pct'].mean():.2f}%")

    if f.empty:
        st.info("No symbols match the current filters.")
        return

    styled = f.style
    styled = styled.background_gradient(
        subset=["bias_score"], cmap="RdYlGn", vmin=-4, vmax=4
    )
    styled = styled.background_gradient(
        subset=["dip_score"], cmap="RdYlGn", vmin=0, vmax=8
    )
    styled = styled.map(
        lambda v: f"color: {dip_color(v)}; font-weight:600", subset=["dip_stage"]
    )
    styled = styled.map(lambda v: f"color: {regime_color(v)}", subset=["regime"])

    def _action_color(v):
        if not isinstance(v, str):
            return ""
        if v.startswith("BUY"):
            return "color: #22c55e; font-weight:700"
        if v.startswith("SELL"):
            return "color: #ef5350; font-weight:700"
        if v.startswith("WAIT"):
            return "color: #f0b90b; font-weight:600"
        return "color: #64748b"

    styled = styled.map(_action_color, subset=["action"])
    styled = styled.format(
        {
            "close": "{:,.2f}",
            "bias_score": "{:+d}",
            "vs_sma200_pct": "{:+.2f}%",
            "atr_pct": "{:.2f}%",
            "ml_prob": "{:.0f}%",
            "ml_short_prob": "{:.0f}%",
            "rally_score": "{:.0f}",
            "long_evidence": "{:.2f}",
            "short_evidence": "{:.2f}",
            "setup_ev": "{:+.2f}R",
        },
        na_rep="-",
    )

    def _setup_color(v):
        if not isinstance(v, str):
            return ""
        if v.startswith("LONG"):
            return "color: #22c55e; font-weight:600"
        if v.startswith("SHORT"):
            return "color: #ef5350; font-weight:600"
        return "color: #64748b"

    if "setup" in styled.columns:
        styled = styled.map(_setup_color, subset=["setup"])
    styled = styled.format(
        {
            "support": "{:,.5f}",
            "resistance": "{:,.5f}",
            "invalidation": "{:,.5f}",
        },
        na_rep="-",
    )

    st.dataframe(styled, use_container_width=True, height=520)

    col_a, col_b = st.columns([1.4, 1])
    with col_a:
        st.subheader("Feature heatmap")
        st.plotly_chart(build_heatmap(f), use_container_width=True)
    with col_b:
        st.subheader("Regime distribution")
        st.plotly_chart(build_regime_chart(f), use_container_width=True)


# ---------------------------------------------------------------------------
# Page 2 — Symbol detail
# ---------------------------------------------------------------------------


def render_detail(g: dict) -> None:
    st.title("🔍 Symbol Detail")

    symbols = get_symbols(g["group"], g["timeframe"])
    if not symbols:
        st.error("No symbols in this dataset.")
        return

    sym_filter = st.text_input("🔎 Search symbol", "", placeholder="e.g. EUR")
    cands = [s for s in symbols if sym_filter.upper() in s] or symbols
    symbol = st.selectbox("Symbol", cands)

    with st.spinner(f"Analysing {symbol}…"):
        df, report = cached_detail(symbol, g["group"], g["timeframe"])

    st.markdown(_dip_banner_html(report), unsafe_allow_html=True)
    st.markdown(_rally_banner_html(report), unsafe_allow_html=True)

    m1, m2, m3, m4, m5, m6 = st.columns(6)
    bias = directional_bias(df)
    m1.metric("Close", f"{report['last_close']:,.5f}")
    m2.metric("Bias", f"{bias['label']} ({bias['score']:+d})")
    m3.metric("Regime", report["regime"]["regime"])
    m4.metric("ADX", f"{report['trend_strength']['adx']:.1f}")
    m5.metric("RSI 14", f"{report['momentum']['rsi_14']:.1f}")
    m6.metric(
        "ATR %", f"{report['volatility']['atr_14'] / report['last_close'] * 100:.2f}"
    )

    ml = report.get("ml")
    if ml:
        m = ml.get("model") or {}
        auc = m.get("auc_oos")
        calibrated = m.get("calibrated")
        st.markdown(
            f'<div class="nx-banner" style="background:#ab47bc1A;'
            f"border:1px solid #ab47bc;border-left:6px solid #ab47bc;"
            f'color:#ce93d8;">🤖 ML Bullish Probability '
            f"<b>{ml['prob_pct']}%</b> ({ml['label']})"
            f" <small>· P(1R move up) · model OOS AUC "
            f"{auc if auc is not None else 'n/a'}"
            f"{' · calibrated' if calibrated else ''}</small></div>",
            unsafe_allow_html=True,
        )
        imp = ml.get("importance")
        if imp:
            with st.expander("🧬 Feature importance (spec #10)"):
                c1, c2 = st.columns(2)
                with c1:
                    st.caption("By factor group (gain share)")
                    grp = pd.DataFrame(imp["by_group"])
                    if not grp.empty:
                        st.dataframe(
                            grp.style.background_gradient(
                                subset=["gain_pct"], cmap="viridis"
                            ),
                            hide_index=True,
                            use_container_width=True,
                        )
                with c2:
                    st.caption("Top features")
                    top = pd.DataFrame(imp["top"])
                    if not top.empty:
                        st.dataframe(top, hide_index=True, use_container_width=True)

    macro = report.get("macro")
    if macro:
        rg = macro.get("regime") or {}
        bs = macro.get("bias") or {}
        gt = macro.get("gate") or {}
        gate_color = "#22c55e" if gt.get("allowed") else "#ef4444"
        gate_txt = "PASS" if gt.get("allowed") else "BLOCKED"
        st.markdown(
            f'<div class="nx-banner" style="background:#42a5f51A;'
            f"border:1px solid #42a5f5;border-left:6px solid #42a5f5;"
            f'color:#90caf9;">🌍 Macro Overlay '
            f"<b>{bs.get('label', 'n/a')}</b> "
            f"<small>· bias {bs.get('bias', 0):+.2f} · USD "
            f"{rg.get('usd', 'n/a')} · Risk {rg.get('risk', 'n/a')} · "
            f"Rates {rg.get('rates', 'n/a')}</small>"
            f'<span class="nx-chip" style="background:{gate_color}22;'
            f"border:1px solid {gate_color};color:{gate_color};"
            f'margin-left:10px;">Gate {gate_txt}</span></div>',
            unsafe_allow_html=True,
        )

    overview, chart, momentum, levels, dip_tab = st.tabs(
        [
            "📋 Overview",
            "📈 Price & Levels",
            "🔄 Momentum",
            "🎯 Key Levels",
            "💧 Buy-the-Dip",
        ]
    )

    with overview:
        c1, c2 = st.columns(2)
        with c1:
            st.subheader("Market Regime")
            rg = report["regime"]
            mtf = rg.get("mtf", [])
            if mtf:
                mtf_rows = pd.DataFrame(
                    [
                        {
                            "TF": r["timeframe"],
                            "Regime": r["regime"],
                            "ADX": r["adx"],
                            "vs200 %": r["vs_200_pct"],
                            "Conf": r["confidence"],
                        }
                        for r in mtf
                    ]
                )
                st.dataframe(mtf_rows, hide_index=True, use_container_width=True)
                st.caption(
                    f"Consensus: **{rg.get('mtf_consensus', '-')}** · "
                    f"cluster: {rg.get('regime_cluster', '-')}"
                )
            else:
                st.dataframe(
                    _dict_table({k: v for k, v in rg.items() if k not in ("mtf",)}),
                    hide_index=True,
                    use_container_width=True,
                )
            st.subheader("Moving Averages")
            st.dataframe(
                _dict_table(report["moving_averages"]),
                hide_index=True,
                use_container_width=True,
            )
        with c2:
            st.subheader("Momentum")
            st.dataframe(
                _dict_table(report["momentum"]),
                hide_index=True,
                use_container_width=True,
            )
            st.subheader("Trend Strength")
            st.dataframe(
                _dict_table(report["trend_strength"]),
                hide_index=True,
                use_container_width=True,
            )
            st.subheader("Volatility")
            st.dataframe(
                _dict_table(report["volatility"]),
                hide_index=True,
                use_container_width=True,
            )
        st.subheader("Simple Bias")
        sb = report["simple_bias"]
        st.progress(sb["score"] / sb["max_score"])
        st.caption(
            f"Score {sb['score']}/{sb['max_score']} — **{sb['interpretation']}**"
        )

        rb = report.get("ma_ribbon") or {}
        if rb.get("available"):
            st.subheader("MA Ribbon Structure")
            m1, m2, m3 = st.columns(3)
            m1.metric(
                "Cross prob (20d)",
                f"{rb['cross_prob'] * 100:.0f}%",
                rb["cross_direction"].upper(),
            )
            m2.metric("Ribbon slope", f"{rb['ribbon_slope']:.4f}")
            m3.metric("Ribbon width", f"{rb['ribbon_width_pct']:.2f}%")
            st.caption(
                f"Alignment {rb['ribbon_alignment']:+.2f} · signal **{rb['signal']}**"
            )

        se_ = report.get("sentiment")
        if se_ and se_.get("available"):
            st.subheader("News & Social Sentiment")
            n = se_.get("news") or {}
            c1, c2 = st.columns(2)
            c1.metric(
                "Composite", f"{se_.get('composite', '—')}", f"{n.get('source', '—')}"
            )
            c2.metric(
                "Articles",
                f"{n.get('n_articles', 0)} ({n.get('relevant', 0)} relevant)",
            )
            sc = se_.get("social") or {}
            if sc.get("score") is not None:
                st.caption(f"Social: {sc['score']} · {sc.get('note', '')}")

        mc = report.get("macro") or {}
        ss = mc.get("sensitivities") or {}
        if ss:
            st.subheader(f"Macro Sensitivities ({ss.get('lookback', 90)}d, 1d-lagged)")
            m1, m2, m3, m4 = st.columns(4)
            m1.metric(
                "Market β (S&P500)",
                f"{ss.get('market_beta', '—')}",
                f"corr {ss.get('spx_corr', '—')}",
            )
            m2.metric("Dollar", f"{ss.get('dollar_sens', '—')}")
            m3.metric("Yields (10Y)", f"{ss.get('yield_sens', '—')}")
            m4.metric("Volatility (VIX)", f"{ss.get('vol_sens', '—')}")
            se = ss.get("sector_etf")
            if se:
                st.caption(
                    f"Sector ETF **{se['ticker']}** correlation: {se.get('corr', '—')}"
                )

    with chart:
        lookback = st.slider(
            "Lookback (bars)",
            min(60, len(df)),
            min(len(df), 1000),
            min(180, len(df)),
            20,
        )
        st.plotly_chart(
            build_price_chart(df, report, lookback), use_container_width=True
        )

    with momentum:
        st.plotly_chart(build_momentum_chart(df), use_container_width=True)
        dv = report.get("divergences") or {}
        if dv.get("signals"):
            st.subheader("Divergences (≥65% confidence)")
            _rows = []
            for d in dv["signals"]:
                # Shape-agnostic detail (regular/hidden vs failure swings)
                # from divergence.format_divergence - same source of truth
                # as the report printer.
                _rows.append(
                    {
                        "Signal": d["name"],
                        "Side": d["side"],
                        "Osc": d["osc"],
                        "From→To": format_divergence(d),
                        "Conf": d["confidence"],
                    }
                )
            dv_rows = pd.DataFrame(_rows)
            st.dataframe(dv_rows, hide_index=True, use_container_width=True)
        else:
            st.caption("No divergences above the 65% confidence threshold.")

    with levels:
        lv = report["levels"]
        ns = lv.get("nearest_support") or {}
        nr = lv.get("nearest_resistance") or {}
        c1, c2, c3 = st.columns(3)
        c1.metric(
            "Nearest Support",
            f"{ns.get('price', 0):,.5f}",
            f"confluence {ns.get('score', 0)} · {ns.get('strength', '-')}",
        )
        c2.metric(
            "Nearest Resistance",
            f"{nr.get('price', 0):,.5f}",
            f"confluence {nr.get('score', 0)} · {nr.get('strength', '-')}",
        )
        c3.metric("Cluster tolerance", f"{lv.get('tolerance', 0):.6f}")

        legs = []
        for name, leg in (
            ("Up", lv.get("last_up_leg")),
            ("Down", lv.get("last_down_leg")),
        ):
            if leg:
                legs.append(
                    {
                        "leg": f"Last {name} leg",
                        "from": f"{leg[0]:,.5f}",
                        "to": f"{leg[1]:,.5f}",
                    }
                )
        if legs:
            st.subheader("Fibonacci legs")
            st.dataframe(pd.DataFrame(legs), hide_index=True, use_container_width=True)

        st.subheader("Top confluence zones")
        zones = pd.DataFrame(
            [
                {
                    "price": f"{z['price']:,.5f}",
                    "score": z["score"],
                    "strength": z["strength"],
                    "tags": ", ".join(z["tags"]),
                }
                for z in lv.get("top_confluence", [])
            ]
        )
        if zones.empty:
            st.caption("No confluence zones within range.")
        else:
            st.dataframe(zones, hide_index=True, use_container_width=True)

        st.subheader("Classical pivots (last closed bar)")
        st.dataframe(
            _dict_table(lv.get("pivots", {})), hide_index=True, use_container_width=True
        )

        av = lv.get("anchored_vwap")
        if av:
            st.metric("Anchored VWAP (from first bar)", f"{av:,.5f}")

        vp = lv.get("volume_profile") or []
        if vp:
            st.subheader("Volume-profile high-volume nodes")
            vp_df = pd.DataFrame(
                [
                    {
                        "Price": f"{n['price']:,.5f}",
                        "Vol %": n["volume_pct"],
                        "High-volume": "●" if n["is_high_volume"] else "○",
                    }
                    for n in vp
                ]
            )
            st.dataframe(vp_df, hide_index=True, use_container_width=True, height=260)

        fm = report.get("fib_map") or []
        if fm:
            st.subheader("Fibonacci confluence map")
            fm_df = pd.DataFrame(
                [
                    {
                        "Level": row["level"],
                        "Side": row["side"],
                        "Price": f"{row['price']:,.5f}",
                        "Confluence": row["confluence"],
                        "Dist %": row["distance_from_close_pct"],
                    }
                    for row in fm
                ]
            )
            st.dataframe(fm_df, hide_index=True, use_container_width=True, height=280)

    with dip_tab:
        dip = report["dip"]
        comp = dip.get("components", {})
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Dip score", f"{dip['dip_score']}/8")
        c2.metric(
            "Entry zone",
            (f"{dip['entry_zone'][0]:,.5f} – {dip['entry_zone'][1]:,.5f}")
            if dip.get("entry_zone")
            else "—",
        )
        c3.metric(
            "Invalidation",
            f"{dip['invalidation_level']:,.5f}"
            if dip.get("invalidation_level")
            else "—",
        )
        c4.metric("Target", f"{dip['target']:,.5f}" if dip.get("target") else "—")

        st.subheader("Confirmation components")
        st.progress(dip["dip_score"] / 8)
        factors = [
            ("above_sma200", "Price above SMA200", None),
            ("ma_stack", "Bullish MA stack (20>50>200)", None),
            ("trend", "Trend strength (ADX ≥ 20)", None),
            (
                "pullback",
                "Real pullback (0.5%–15% off the swing high)",
                f"{dip.get('dip_depth_pct', 0):.2f}%",
            ),
            ("cooled", "RSI cooled into the 30–55 band", None),
            ("at_support", "At confluence support", None),
            ("fib_zone", "Inside 0.382–0.786 fib retracement", None),
            ("trigger", "Momentum trigger (MACD/RSI/bar)", None),
        ]
        rows = []
        for key, label, extra in factors:
            val = comp.get(key)
            if isinstance(val, dict):
                ok = bool(val.get("triggered"))
                detail = val.get("details", "")
            else:
                ok = bool(val)
                detail = ""
            rows.append(
                {
                    "Factor": label,
                    "Status": "✓" if ok else "✗",
                    "Detail": detail or (extra or ""),
                }
            )
        st.dataframe(pd.DataFrame(rows), hide_index=True, use_container_width=True)

        st.subheader("Trigger")
        st.caption(
            f"**{dip.get('trigger', '—')}** — momentum from the "
            f"intraday H4/H1 frame when available locally, else D1."
        )


# ---------------------------------------------------------------------------
# Page 3 — Compare symbols
# ---------------------------------------------------------------------------


def render_compare(g: dict) -> None:
    st.title("⚖️ Compare Symbols")

    symbols = get_symbols(g["group"], g["timeframe"])
    if not symbols:
        st.error("No symbols in this dataset.")
        return

    sym_filter = st.text_input("🔎 Filter symbols", "", placeholder="e.g. USD")
    cands = [s for s in symbols if sym_filter.upper() in s] or symbols
    sel = st.multiselect(
        "Symbols to compare", cands, default=cands[: min(3, len(cands))]
    )
    if not sel:
        st.info("Select at least one symbol.")
        return

    records, frames = [], []
    for s in sel:
        with st.spinner(f"Analysing {s}…"):
            df, report = cached_detail(s, g["group"], g["timeframe"])
        frames.append(df)
        bias = directional_bias(df)
        dip = report["dip"]
        levels = report["levels"]
        records.append(
            {
                "symbol": s,
                "close": report["last_close"],
                "regime": report["regime"]["regime"],
                "bias": bias["label"],
                "bias_score": bias["score"],
                "adx": report["trend_strength"]["adx"],
                "rsi_14": report["momentum"]["rsi_14"],
                "dip_score": dip["dip_score"],
                "dip_stage": dip["dip_stage"],
                "support": ((levels.get("nearest_support") or {}).get("price")),
                "resistance": ((levels.get("nearest_resistance") or {}).get("price")),
                "invalidation": dip.get("invalidation_level"),
                "macro_bias": (report.get("macro") or {}).get("bias", {}).get("bias"),
                "macro_label": (report.get("macro") or {}).get("bias", {}).get("label"),
                "macro_gate": (
                    "PASS"
                    if (report.get("macro") or {}).get("gate", {}).get("allowed")
                    else "BLOCKED"
                    if report.get("macro")
                    else None
                ),
            }
        )

    cmp = pd.DataFrame(records).set_index("symbol")
    st.subheader("Metrics")
    st.dataframe(cmp.T, use_container_width=True)

    c1, c2 = st.columns(2)
    with c1:
        st.subheader("Directional bias")
        st.plotly_chart(build_bias_chart(records), use_container_width=True)
    with c2:
        st.subheader("Buy-the-Dip score")
        st.plotly_chart(build_dip_chart(records), use_container_width=True)

    st.subheader("Normalized close (base = 100)")
    st.plotly_chart(
        build_compare_chart(frames, [r["symbol"] for r in records]),
        use_container_width=True,
    )


# ---------------------------------------------------------------------------
# Page 4 — Backtest
# ---------------------------------------------------------------------------


def render_backtest(g: dict) -> None:
    st.title("🧪 Buy-the-Dip Backtest")

    symbols = get_symbols(g["group"], g["timeframe"])
    if not symbols:
        st.error("No symbols in this dataset.")
        return

    c1, c2, c3 = st.columns([1.4, 1, 1])
    with c1:
        sym_filter = st.text_input("🔎 Search symbol", "", placeholder="e.g. USD, XAU")
        cands = [s for s in symbols if sym_filter.upper() in s] or symbols
        symbol = st.selectbox("Symbol", cands)
    with c2:
        risk = st.slider("Risk per trade", 0.5, 5.0, 1.0, 0.5)
    with c3:
        rr = st.select_slider("R:R fallback", [1.0, 1.5, 2.0, 2.5, 3.0], 2.0)
    c4, c5, c6 = st.columns(3)
    with c4:
        max_hold = st.slider("Max hold (bars)", 5, 60, 20, 5)
    with c5:
        entry_type = st.radio("Entry", ["limit", "market"], horizontal=True)
    with c6:
        start_year = st.selectbox("Start year", [2010, 2012, 2015, 2018, 2020], 1)

    if st.button("▶ Run backtest", type="primary"):
        with st.spinner(f"Backtesting {symbol}…"):
            result = run_symbol_backtest(
                symbol,
                g["group"],
                g["timeframe"],
                risk_pct=risk / 100,
                rr_fallback=rr,
                max_hold=max_hold,
                entry_type=entry_type,
            )
            # apply start-year filter on the equity/trades afterwards
            cutoff = pd.Timestamp(f"{start_year}-01-01")
            eq = result.equity[result.equity.index >= cutoff]
            if eq.empty:
                st.warning("No data after the selected start year.")
                return
            # Filter by *entry* so every trade's entry_time exists in the
            # sliced equity index (compute_stats uses index.get_loc).
            trades = [t for t in result.trades if t.entry_time >= cutoff]
            result.equity = eq
            result.trades = trades
            result.stats = _recompute_stats(result)

        s = result.stats
        st.markdown(_bt_banner_html(result), unsafe_allow_html=True)
        m1, m2, m3, m4, m5, m6 = st.columns(6)
        m1.metric("Trades", s["n_trades"], f"{s['wins']}W / {s['losses']}L")
        m2.metric("Win rate", f"{s['win_rate'] * 100:.0f}%")
        m3.metric(
            "Profit factor",
            "∞" if s["profit_factor"] == float("inf") else f"{s['profit_factor']:.2f}",
        )
        m4.metric("Expectancy", f"{s['expectancy_r']:+.2f}R")
        m5.metric("Total return", f"{s['total_return_pct']:+.1f}%")
        m6.metric("Max drawdown", f"{s['max_drawdown_pct']:.1f}%")

        c1, c2 = st.columns([1.4, 1])
        with c1:
            st.subheader("Equity curve")
            st.plotly_chart(build_equity_chart(result), use_container_width=True)
        with c2:
            st.subheader("Details")
            detail_rows = [
                ("CAGR", f"{s['cagr_pct']:+.2f}%"),
                ("Sharpe", f"{s['sharpe']:.2f}"),
                ("Avg hold", f"{s['avg_hold_bars']:.1f} bars"),
                ("Exposure", f"{s['exposure_pct']:.1f}%"),
                ("Best trade", f"{s['best_trade_pct']:+.2f}%"),
                ("Worst trade", f"{s['worst_trade_pct']:+.2f}%"),
            ]
            st.dataframe(
                pd.DataFrame(detail_rows, columns=["Metric", "Value"]),
                hide_index=True,
                use_container_width=True,
            )

        if result.trades:
            st.subheader(f"Trades ({len(result.trades)})")
            tf = result.trades_frame()
            tf = tf.sort_values("exit_time", ascending=False)
            tf["entry_time"] = tf["entry_time"].map(lambda v: str(v.date()))
            tf["exit_time"] = tf["exit_time"].map(lambda v: str(v.date()))
            tf["entry_price"] = tf["entry_price"].map(lambda v: f"{v:,.5f}")
            tf["exit_price"] = tf["exit_price"].map(lambda v: f"{v:,.5f}")
            tf["pnl"] = tf["pnl"].map(lambda v: f"{v:,.0f}")
            tf["pnl_pct"] = tf["pnl_pct"].map(lambda v: f"{v:+.2%}")
            tf["r_multiple"] = tf["r_multiple"].map(lambda v: f"{v:+.2f}R")
            st.dataframe(tf, hide_index=True, use_container_width=True, height=420)
        else:
            st.caption(
                "No closed trades in this period — try looser "
                "params (longer max hold, lower threshold) or a "
                "different symbol."
            )
    else:
        st.info(
            "Configure the strategy parameters and press **▶ Run "
            "backtest**.\n\nRules: limit entry at the dip's entry zone "
            "(0.382–0.618 fib band), stop at the invalidation below the "
            "swing low, target at the nearest resistance (or R:R "
            "fallback), time-stop after max-hold bars. Fully causal — no "
            "lookahead."
        )


def _recompute_stats(result):
    from src.backtest.engine import compute_stats

    return compute_stats(result)


# ---------------------------------------------------------------------------
# Page 5 — Risk & position sizing
# ---------------------------------------------------------------------------


def render_risk(g: dict) -> None:
    st.title("🛡️ Risk & Position Sizing")
    st.caption(
        "Per-trade sizing (fractional / volatility-targeted / Kelly) "
        "+ per-trade VaR on the live dip setup, and a portfolio-level "
        "heat / correlation report."
    )

    symbols = get_symbols(g["group"], g["timeframe"])
    if not symbols:
        st.error("No symbols in this dataset.")
        return

    c1, c2, c3 = st.columns([1.4, 1, 1])
    with c1:
        sym_filter = st.text_input("🔎 Search symbol", "", placeholder="e.g. USD, XAU")
        cands = [s for s in symbols if sym_filter.upper() in s] or symbols
        symbol = st.selectbox("Symbol", cands)
    with c2:
        equity = st.number_input("Equity ($)", 10_000, 10_000_000, 250_000, 10_000)
    with c3:
        risk = st.slider("Risk per trade", 0.25, 5.0, 1.0, 0.25)

    with st.expander("⚙️ Sizing parameters"):
        c4, c5, c6 = st.columns(3)
        with c4:
            vol_target = st.slider("Vol target (per trade)", 0.1, 5.0, 2.0, 0.1)
        with c5:
            hold_bars = st.slider("Holding period (bars)", 1, 40, 10)
        with c6:
            payoff = st.slider("Kelly payoff (b)", 1.0, 3.0, 1.5, 0.1)
        c7, c8 = st.columns(2)
        with c7:
            kelly_frac = st.slider("Kelly fraction", 0.1, 1.0, 0.5, 0.1)
        with c8:
            st.caption(
                "Kelly win probability comes from the ML model when present, else 55%."
            )

    with st.spinner(f"Analysing {symbol}…"):
        df, report = cached_detail(symbol, g["group"], g["timeframe"])

    from src.risk.run import risk_plan_from_report

    plan = risk_plan_from_report(
        report,
        symbol,
        equity=equity,
        risk_pct=risk / 100,
        vol_target=vol_target / 100,
        payoff=payoff,
        kelly_fraction=kelly_frac,
        hold_bars=hold_bars,
    )

    if not plan.get("setup"):
        st.info(
            f"No actionable dip setup for **{symbol}** — "
            f"{plan.get('reason', 'no entry zone / invalidation')}. "
            "The risk plan needs a valid entry zone and invalidation "
            "from the Buy-the-Dip engine."
        )
    else:
        s = plan["setup"]
        m1, m2, m3, m4, m5, m6 = st.columns(6)
        m1.metric("Entry", f"{s['entry']:,.5f}")
        m2.metric("Stop (invalidation)", f"{s['stop']:,.5f}")
        m3.metric("Target", f"{s['target']:,.5f}")
        m4.metric(
            "R:R",
            f"{s['rr']:.2f}",
            f"floor {s.get('min_rr', 2.5)}:1 " + ("✓" if s.get("rr_ok") else "✗")
            if s.get("rr_ok") is not None
            else None,
        )
        m5.metric("ATR %", f"{s['atr_pct']:.2f}%")
        m6.metric(
            "ML prob", f"{s['ml_prob']:.0f}%" if s["ml_prob"] is not None else "n/a"
        )

        st.subheader("Position sizes (1 trade, per method)")
        sizes = pd.DataFrame(plan["sizes"])
        sizes.columns = [
            "Method",
            "Qty",
            "Notional ($)",
            "Risk ($)",
            "Risk %",
            "VaR95 1d ($)",
            f"VaR95 {hold_bars}d ($)",
        ]
        st.dataframe(sizes, hide_index=True, use_container_width=True)

        st.caption(
            f"Kelly p = **{plan['inputs']['kelly_p']}** "
            f"({'ML model' if s['ml_prob'] is not None else 'fallback'}) · "
            f"fractional Kelly {kelly_frac} · payoff {payoff}R · "
            f"vol-target capped at the {risk:.2f}% risk budget · "
            "VaR95 = size × ATR × 1.645 × √hold."
        )
        if s.get("min_rr_tp"):
            st.caption(
                f"🎯 Target ladder: best achievable R:R "
                f"**{s.get('best_rr')}** at **{s['min_rr_tp']}** "
                f"(scaling out) · nearest single target "
                f"{s.get('rr_nearest')} · floor "
                f"{s.get('min_rr', 2.5)}:1 "
                f"{'met' if s.get('rr_ok') else 'NOT met'}."
            )

    st.divider()
    st.subheader("🏦 Portfolio heat & correlation")
    st.caption(
        "Portfolio VaR and correlation-aware limits across the "
        "strongest dip setups (top 12 by dip score in this dataset)."
    )
    if st.button("⚡ Compute portfolio risk", type="primary"):
        from src.risk.run import portfolio_report

        table = cached_universe(g["group"], g["timeframe"])
        top = table.sort_values("dip_score", ascending=False).head(12)
        with st.spinner("Computing correlations, heat and VaR…"):
            rep = portfolio_report(
                top["symbol"].tolist(),
                g["group"],
                g["timeframe"],
                equity=equity,
                risk_pct=risk / 100,
                max_heat=0.04,
                max_corr=0.6,
            )
        if not rep.get("symbols"):
            st.warning(rep.get("reason", "No usable data for the portfolio."))
            return
        h1, h2, h3, h4 = st.columns(4)
        h1.metric(
            "Portfolio heat", f"{rep['heat_pct']:.2f}%", f"limit {rep['heat_vs_limit']}"
        )
        h2.metric(
            "VaR95 (1d)",
            f"${rep['portfolio_var_95_1bar']:,.0f}",
            f"{rep['portfolio_var_pct_equity']:.2f}% of equity",
        )
        h3.metric("Setups", rep["n_setups"])
        h4.metric("Symbols", len(rep["symbols"]))

        c1, c2 = st.columns([1.2, 1])
        with c1:
            st.subheader("Correlation-aware gates")
            gates = pd.DataFrame(rep["correlation_checks"])
            gates["avg_corr"] = gates["avg_corr"].fillna("-")
            gates.columns = ["Symbol", "Avg corr", "Allowed", "Reason"]
            st.dataframe(gates, hide_index=True, use_container_width=True)
        with c2:
            st.subheader("Most correlated pairs")
            if rep["top_correlated_pairs"]:
                pairs = pd.DataFrame(rep["top_correlated_pairs"])
                pairs.columns = ["Pair", "Corr"]
                st.dataframe(pairs, hide_index=True, use_container_width=True)
            else:
                st.caption("Need ≥ 2 symbols for pairs.")

        st.subheader("Per-symbol risk (fractional 1% sizing)")
        pos = pd.DataFrame(rep["positions"])
        pos.columns = ["Symbol", "Notional ($)", "Risk ($)"]
        st.dataframe(pos, hide_index=True, use_container_width=True)
    else:
        st.info(
            "Press **⚡ Compute portfolio risk** to run the "
            "correlation-aware portfolio report."
        )


def _bt_banner_html(result) -> str:
    s = result.stats
    if s["n_trades"] == 0:
        return (
            f'<div class="nx-banner" style="background:#94a3b81A;'
            f"border:1px solid #94a3b8;border-left:6px solid #94a3b8;"
            f'color:#94a3b8;">NO TRADES'
            f" <small>· {result.symbol} · adjust the parameters</small></div>"
        )
    good = s["total_return_pct"] > 0 and s["profit_factor"] >= 1
    color = "#22c55e" if good else "#ef4444"
    label = "POSITIVE EDGE" if good else "NO EDGE"
    return (
        f'<div class="nx-banner" style="background:{color}1A;'
        f"border:1px solid {color};border-left:6px solid {color};"
        f'color:{color};">{label}'
        f" <small>· {result.symbol} · {s['n_trades']} trades · "
        f"{s['win_rate'] * 100:.0f}% win rate · "
        f"PF {s['profit_factor']:.2f}</small></div>"
    )


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if page.startswith("🌐"):
    render_universe(g)
elif page.startswith("🔍"):
    render_detail(g)
elif page.startswith("🧪"):
    render_backtest(g)
elif page.startswith("🛡"):
    render_risk(g)
else:
    render_compare(g)

st.markdown("---")
st.caption("NexusQuant · regime + confluence + buy-the-dip engines · local data only")
