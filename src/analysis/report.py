"""
NexusQuant - Institutional Analysis Report Generator
"""

import pandas as pd
from typing import Dict, Any, Optional
from pathlib import Path
import sys

# Allow running from project root
sys.path.append(str(Path(__file__).parent.parent.parent))

from src.features.indicators import (
    add_all_indicators,
    ma_ribbon_summary,
    volume_flow_summary,
)
from src.features.regime import (
    detect_regime,
    get_current_regime_summary,
    get_current_regime_summary_mtf,
)
from src.features.levels import (
    anchored_vwap,
    fibonacci_map,
    levels_summary,
    volume_profile_nodes,
)
from src.features.dip import detect_dip
from src.features.rally import detect_rally
from src.analysis.plan import format_plan, trade_plan
from src.features.patterns import patterns_summary
from src.features.divergence import divergence_summary, format_divergence
from src.model.model import (
    DEFAULT_MODEL_PATH,
    importance_summary,
    model_meta_summary,
    predict_series,
)


def _direction_win_prob(sc: Dict) -> Optional[float]:
    """
    Direction-winning calibrated probability for a setup classification.

    Picks the probability matching the classifier's direction so a short
    setup's EV reads from the SHORT model (never the inverted or unrelated
    long read), falling back to the other side only when the matching model
    is absent. Explicit None checks - a 0.0 calibrated probability is valid
    evidence, not "missing". Returns None only when neither side has one.
    """
    direction = sc.get("direction")
    if direction == "short":
        p = sc.get("prob_short")
        if p is not None:
            return p
        return sc.get("prob_long")
    p = sc.get("prob_long")
    if p is not None:
        return p
    return sc.get("prob_short")


_SETTINGS_CACHE: Optional[dict] = None
_TP_PROBS_CACHE: Optional[dict] = None
_TP_PROBS_LOADED = False


def _load_target_probs() -> Optional[dict]:
    """The empirical target-level TP distribution, memoized at module
    level (universe scans generate one report per symbol). Returns None
    when data/validation/target_probs.json is absent - the book then
    falls back to the documented ladder-best EV approximation."""
    global _TP_PROBS_CACHE, _TP_PROBS_LOADED
    if not _TP_PROBS_LOADED:
        _TP_PROBS_LOADED = True
        try:
            import json

            from pathlib import Path as _P

            p = _P("data/validation/target_probs.json")
            if p.exists():
                with open(p) as _f:
                    _TP_PROBS_CACHE = json.load(_f)
        except Exception:
            _TP_PROBS_CACHE = None
    return _TP_PROBS_CACHE


def _settings_slippage_pips() -> Optional[float]:
    """Settings slippage (pips) for the cost model, memoized at module
    level - universe scans generate one report per symbol, so the yaml
    read must not happen per symbol (mirrors the macro frame cache)."""
    global _SETTINGS_CACHE
    try:
        if _SETTINGS_CACHE is None:
            import yaml

            with open("config/settings.yaml") as _f:
                _SETTINGS_CACHE = yaml.safe_load(_f) or {}
        return (_SETTINGS_CACHE.get("backtest") or {}).get("slippage_pips")
    except Exception:
        return None


def generate_full_report(
    df: pd.DataFrame,
    symbol: str = "XAUUSD",
    group: Optional[str] = None,
    data_dir: str = "data/raw",
    mtf: bool = True,
    use_hmm: bool = False,
) -> Dict[str, Any]:
    """
    Generate a structured institutional-style analysis report.

    ``group`` / ``data_dir`` are forwarded to the ML probability so the
    model is served with the same H4 / cross-asset / COT context it was
    trained on (graceful when files are missing). ``mtf`` gates the
    D/W/M multi-timeframe regime table - the cheapest scans (universe
    ranking rows) skip it since only the detail view renders it.
    ``use_hmm`` additionally fits the 4-state HMM and surfaces its label
    in the regime summary + the D/W/M rows (opt-in: the fit costs a few
    seconds per symbol; degrades gracefully to the deterministic label
    when ``hmmlearn`` is missing).
    """
    # Ensure indicators and regime exist
    if "rsi_14" not in df.columns:
        df = add_all_indicators(df)
    if "regime" not in df.columns:
        df = detect_regime(df)

    latest = df.iloc[-1]

    report = {
        "symbol": symbol,
        "last_date": str(df.index[-1].date()),
        "last_close": round(latest["close"], 3),
        # Regime incl. the KMeans cluster label and the D/W/M multi-timeframe
        # table (institutional spec #1: regression slope + ADX + 200SMA +
        # vol clustering, Monthly/Weekly/Daily rows). mtf=False skips the
        # clustering label AND the resample-heavy D/W/M table (both live in
        # get_current_regime_summary_mtf) - the cheap path used by universe
        # ranking rows.
        "regime": get_current_regime_summary_mtf(df, use_hmm=use_hmm)
        if mtf
        else get_current_regime_summary(df),
        "moving_averages": {
            "sma_50": round(latest.get("sma_50", 0), 3),
            "sma_100": round(latest.get("sma_100", 0), 3),
            "sma_200": round(latest.get("sma_200", 0), 3),
            "ema_50": round(latest.get("ema_50", 0), 3),
            "ema_100": round(latest.get("ema_100", 0), 3),
            "ema_200": round(latest.get("ema_200", 0), 3),
            "price_vs_sma200": "Above"
            if latest["close"] > latest.get("sma_200", 0)
            else "Below",
        },
        # MA ribbon structure (spec #3): golden/death cross probability,
        # ribbon slope + width (trend-momentum proxy).
        "ma_ribbon": ma_ribbon_summary(df),
        "momentum": {
            "rsi_14": round(latest.get("rsi_14", 0), 2),
            "macd": round(latest.get("macd", 0), 4),
            "macd_signal": round(latest.get("macd_signal", 0), 4),
            "macd_hist": round(latest.get("macd_hist", 0), 4),
            "bb_pct_b": round(latest.get("bb_pct_b", 0), 3),
        },
        "volatility": {
            "atr_14": round(latest.get("atr_14", 0), 3),
            "bb_width": round(latest.get("bb_width", 0), 4),
            "volatility_20": round(latest.get("volatility_20", 0), 4),
        },
        "trend_strength": {
            "adx": round(latest.get("adx", 0), 2),
            "plus_di": round(latest.get("plus_di", 0), 2),
            "minus_di": round(latest.get("minus_di", 0), 2),
        },
    }

    # Key levels (support/resistance + Fibonacci confluence)
    lv = levels_summary(df)
    # Spec #2 additions: anchored VWAP + volume-profile high-volume nodes.
    lv["anchored_vwap"] = anchored_vwap(df)
    lv["volume_profile"] = volume_profile_nodes(df)
    report["levels"] = lv

    # Fibonacci confluence map (spec #7): the exact 38.2 ... 161.8 ratio
    # table with 1-10 confluence strength per level.
    report["fib_map"] = fibonacci_map(df)

    # Divergence engine (spec #4): regular + hidden divergences and RSI
    # failure swings, confidence >= 65.
    report["divergences"] = divergence_summary(df)

    # Buy-the-Dip confirmation
    report["dip"] = detect_dip(df, levels=report["levels"])

    # Sell-the-Rally confirmation (short-side mirror)
    report["rally"] = detect_rally(df, levels=report["levels"])

    # Short target ladder (mirror of the long ladder below).
    try:
        from src.risk.targets import build_short_target_ladder

        rl = report.get("rally") or {}
        ez = rl.get("entry_zone")
        inv = rl.get("invalidation_level")
        if ez and inv:
            entry = (float(ez[0]) + float(ez[1])) / 2.0
            report["short_targets"] = build_short_target_ladder(
                entry, float(inv), float(latest["close"]), report["levels"]
            )
    except Exception:
        pass

    # Ensemble bullish probability (graceful: absent when no model is saved)
    try:
        prob = predict_series(
            df, DEFAULT_MODEL_PATH, symbol=symbol, group=group, data_dir=data_dir
        )
        if prob is not None:
            last = prob.dropna()
            if len(last):
                p = float(last.iloc[-1])
                report["ml"] = {
                    "prob_pct": round(p * 100, 1),
                    "label": (
                        "High conviction"
                        if p >= 0.6
                        else "Moderate"
                        if p >= 0.5
                        else "Low"
                    ),
                    "model": model_meta_summary(DEFAULT_MODEL_PATH),
                    # Spec #10: feature importance - top features by gain
                    # share + the factor-group breakdown (Trend / Momentum /
                    # Macro / ...). Graceful None when no model is saved.
                    "importance": importance_summary(DEFAULT_MODEL_PATH),
                }
    except Exception:
        pass

    # Sell-the-Rally ensemble probability (short-side mirror; graceful
    # None when no short model is saved - the rule engine still runs).
    try:
        from src.model.model import predict_short_series

        prob_s = predict_short_series(df, symbol=symbol, group=group, data_dir=data_dir)
        if prob_s is not None:
            last_s = prob_s.dropna()
            if len(last_s):
                ps = float(last_s.iloc[-1])
                report["ml_short"] = {
                    "prob_pct": round(ps * 100, 1),
                    "label": (
                        "High conviction"
                        if ps >= 0.6
                        else "Moderate"
                        if ps >= 0.5
                        else "Low"
                    ),
                    "model": model_meta_summary("models/rally_lgbm.joblib"),
                }
    except Exception:
        pass

    # Multi-target ladder (institutional spec #11) - from the dip setup
    # when actionable; makes a 2.5:1 minimum reachable via scaling out.
    # Built BEFORE the risk plan: risk_plan_from_report reads
    # report["targets"] for the honest ladder-best R:R (rr_ok on the
    # 2.5:1 floor). Building it after risk left the setup stuck on the
    # nearest-target R:R (~0.9) and mis-reported every long as "BELOW 2.5".
    try:
        from src.risk.targets import build_target_ladder

        dip = report.get("dip") or {}
        ez = dip.get("entry_zone")
        inv = dip.get("invalidation_level")
        if ez and inv:
            entry = (float(ez[0]) + float(ez[1])) / 2.0
            report["targets"] = build_target_ladder(
                entry, float(inv), float(latest["close"]), report["levels"]
            )
    except Exception:
        pass

    # Risk & position sizing plan - after the ML section so Kelly can use
    # the model probability (graceful: absent when no actionable setup).
    try:
        from src.risk.run import risk_plan_from_report

        report["risk"] = risk_plan_from_report(report, symbol)
    except Exception:
        pass

    # Short-side risk plan (mirror for the Sell-the-Rally engine).
    try:
        from src.risk.run import risk_plan_from_report_short

        report["short_risk"] = risk_plan_from_report_short(report, symbol)
    except Exception:
        pass

    # Macro overlay - top-down USD / risk / rates backdrop for this symbol
    # (graceful: absent when no local DXY or cached VIX/TNX exists). The
    # factor scores are memoized, so a universe scan pays the load once.
    try:
        from src.macro.overlay import macro_report_for_symbol

        report["macro"] = macro_report_for_symbol(symbol, df=df)
    except Exception:
        pass

    # Volume & flow (institutional spec #5)
    report["volume_flow"] = volume_flow_summary(df)

    # Pattern recognition (institutional spec #6)
    report["patterns"] = patterns_summary(df)

    # Final quant rating (institutional spec #14) - needs ml + macro +
    # volume sections, so it runs last.
    try:
        from src.analysis.rating import final_rating

        report["rating"] = final_rating(report)
    except Exception:
        pass

    # Stress testing (institutional spec #12) - how the current setup
    # behaves under the 2008 GFC / COVID / 2022 crashes, with the
    # data-grounded COVID/2022 realizations where history exists.
    try:
        from src.risk.stress import stress_table_from_report

        report["stress"] = stress_table_from_report(report, symbol, df=df)
    except Exception:
        pass

    # Short-side stress (spec #12, short book): the same scenarios applied
    # to the Sell-the-Rally setup. Crashes are *favorable* to shorts
    # (negative loss = gain); the vol-multiplied VaR still flags squeeze
    # risk. Absent when no actionable short setup exists.
    try:
        from src.risk.stress import stress_table_from_report

        report["stress_short"] = stress_table_from_report(
            report, symbol, df=df, direction="short", risk_key="short_risk"
        )
    except Exception:
        pass

    # Institutional spec #8: fundamental factor model - equity-class only
    # (FX/metals have no P/E, ROE etc.); graceful None otherwise.
    try:
        from src.macro.overlay import _symbol_class

        if _symbol_class(symbol)[0] == "equity":
            from src.equity.fundamentals import factor_scores, load_fundamentals

            report["fundamentals"] = factor_scores(
                symbol, df, load_fundamentals(symbol)
            )
    except Exception:
        pass

    # Institutional spec #9 (sentiment half): news/social sentiment for
    # EVERY symbol class - the Yahoo news search endpoint answers FX and
    # metals headlines too, not just equities, so the macro overlay's
    # sentiment factor is no longer hard-zero for the FX/metals universe.
    # fetch_news=False: the report runs inside the scanner/dashboard over
    # the whole universe - never fire per-symbol network calls from there.
    # Sentiment uses the 1-day cache; run
    # `python -m src.equity.run --fetch-news` to warm it first.
    try:
        from src.equity.sentiment import sentiment_report

        report["sentiment"] = sentiment_report(symbol, fetch_news=False)
    except Exception:
        pass

    # Simple signal summary
    bullish_score = 0
    if latest.get("close", 0) > latest.get("sma_200", 0):
        bullish_score += 1
    if latest.get("rsi_14", 50) > 50:
        bullish_score += 1
    if latest.get("macd_hist", 0) > 0:
        bullish_score += 1
    if latest.get("adx", 0) > 25 and latest.get("plus_di", 0) > latest.get(
        "minus_di", 0
    ):
        bullish_score += 1

    report["simple_bias"] = {
        "score": bullish_score,
        "max_score": 4,
        "interpretation": (
            "Bullish"
            if bullish_score >= 3
            else "Bearish"
            if bullish_score <= 1
            else "Neutral"
        ),
    }

    # Direction-neutral setup classification (two-sided audit): independent
    # long/short evidence scores + setup-family taxonomy where the 200-SMA
    # is contextual, not a gate. Runs after dip/rally/ml/divergence/pattern
    # so it can fold their outputs in; the unified plan (below) surfaces the
    # classifier direction + family + EV when the engines have no confirmed
    # setup.
    try:
        from src.features.setups import classify_setup, expected_value

        levels_for_setup = report.get("levels") or levels_summary(df)
        sc = classify_setup(
            df,
            levels=levels_for_setup,
            dip=report.get("dip"),
            rally=report.get("rally"),
            ml={
                "prob": (
                    report["ml"]["prob_pct"] / 100.0
                    if report.get("ml", {}).get("prob_pct") is not None
                    else None
                ),
                "prob_long": (
                    report["ml"]["prob_pct"] / 100.0
                    if report.get("ml", {}).get("prob_pct") is not None
                    else None
                ),
                "prob_short": (
                    report.get("ml_short", {}).get("prob_pct", 0) / 100.0
                    if report.get("ml_short", {}).get("prob_pct") is not None
                    else None
                ),
            },
            divergence=report.get("divergences"),
            pattern=(
                (report.get("patterns") or {}).get("patterns")[-1]
                if (report.get("patterns") or {}).get("patterns")
                else None
            ),
            macro=report.get("macro"),
            regime_label=(
                (report.get("regime") or {}).get("regime")
                or (report.get("regime") or {}).get("label")
            ),
        )
        # Probability-weighted R:R from the ladders (approximate; prefers
        # calibrated ML when present).
        # Direction-winning probability (explicit None checks - a 0.0
        # calibrated probability is valid evidence, not "missing"). Picks
        # the probability matching the classifier's direction so a short
        # setup's EV reads from the SHORT model (never the inverted or
        # unrelated long read), falling back to the other side only when
        # the matching model is absent.
        p_win = _direction_win_prob(sc)
        # Probability-weighted R:R from the ladders (approximate; prefers
        # calibrated ML when present).
        pw_rr = None
        try:
            from src.features.setups import probability_weighted_rr

            ladder = (report.get("targets") or {}).get("targets") or []
            if not ladder:
                ladder = (report.get("short_targets") or {}).get("targets") or []
            pw_rr = probability_weighted_rr(ladder, p_win)
        except Exception:
            pw_rr = None
        sc["pw_rr"] = pw_rr
        # EV in R units (None when no calibrated probability - never fabricated).
        sc["ev"] = expected_value(p_win, avg_win_r=2.5)
        report["setup_classification"] = sc

        # Unified Opportunity Book + EV-aware LONG/SHORT/FLAT decision
        # (campaign spec #9/#18/#46): per-side opportunities with explicit
        # rejection reasons and a verdict driven by expected value when
        # calibrated probabilities exist. Cost model: settings slippage
        # (pips, memoized) converted to R via the stop distance; JPY pairs
        # use 0.01 pips, everything else 0.0001.
        #
        # Stage-2 (spec #4): when the empirical target-level distribution
        # exists (data/validation/target_probs.json, written by the census
        # --write-probs), EV is computed from the true payoff distribution
        # (P(TP_k before SL)) instead of the ladder-best approximation.
        # Missing file -> None -> the documented approximation is used.
        try:
            from src.analysis.opportunity import build_opportunity_book

            pip = 0.01 if symbol.upper().endswith("JPY") else 0.0001
            report["opportunity_book"] = build_opportunity_book(
                report,
                slippage_pips=_settings_slippage_pips(),
                pip_size=pip,
                tp_probs=_load_target_probs(),
            )
        except Exception:
            pass
    except Exception:
        pass

    # Unified trade plan (spec #11/#14): one actionable verdict per symbol -
    # BUY-LIMIT / SELL-LIMIT (confirmed), WAIT-* (pending limit at a zone),
    # or NO-SETUP with the levels that would change the read.
    report["plan"] = trade_plan(report)

    return report


def print_report(report: Dict[str, Any]) -> None:
    """Pretty print the report to console."""
    print("\n" + "=" * 60)
    print(f"NEXUSQUANT INSTITUTIONAL ANALYSIS — {report['symbol']}")
    print(f"Date: {report['last_date']}  |  Close: {report['last_close']}")
    print("=" * 60)

    print("\n1. MARKET REGIME")
    for k, v in report["regime"].items():
        if k == "mtf":
            continue
        print(f"   {k:20}: {v}")
    mtf = report["regime"].get("mtf", [])
    if mtf:
        print("   Multi-timeframe:")
        for row in mtf:
            print(
                f"     {row['timeframe']:<3} {row['regime']:<14} "
                f"ADX {row['adx']:>5.1f} · vs200 {row['vs_200_pct']:>+6.2f}% · "
                f"slope/bar {row.get('slope_pct_20', float('nan')):>+7.4f}% · "
                f"conf {row['confidence']}"
            )
        print(f"     consensus: {report['regime'].get('mtf_consensus', '-')}")

    print("\n1b. MA RIBBON STRUCTURE")
    rb = report.get("ma_ribbon") or {}
    if rb.get("available"):
        print(
            f"   {rb['cross_direction'].upper()} cross prob (next "
            f"{rb['cross_horizon']} bars): {rb['cross_prob']:.1%}"
        )
        print(
            f"   Ribbon slope: {rb['ribbon_slope']} · "
            f"width {rb['ribbon_width_pct']}% of price · "
            f"alignment {rb['ribbon_alignment']:+.2f} · signal {rb['signal']}"
        )
    else:
        print("   (not enough history)")

    print("\n2. MOVING AVERAGES")
    for k, v in report["moving_averages"].items():
        print(f"   {k:20}: {v}")

    print("\n3. MOMENTUM")
    for k, v in report["momentum"].items():
        print(f"   {k:20}: {v}")

    print("\n4. TREND STRENGTH (ADX)")
    for k, v in report["trend_strength"].items():
        print(f"   {k:20}: {v}")

    print("\n5. VOLATILITY")
    for k, v in report["volatility"].items():
        print(f"   {k:20}: {v}")

    print("\n6. SIMPLE BIAS")
    print(
        f"   Score: {report['simple_bias']['score']}/{report['simple_bias']['max_score']}"
    )
    print(f"   Interpretation: {report['simple_bias']['interpretation']}")

    print("\n7. KEY LEVELS (confluence)")
    lv = report.get("levels", {})
    print(f"   Tolerance: {lv.get('tolerance', 0)}")
    for name, leg in (("Up", lv.get("last_up_leg")), ("Down", lv.get("last_down_leg"))):
        if leg:
            print(f"   Last {name} leg: {leg[0]:.5f} -> {leg[1]:.5f}")

    ns = lv.get("nearest_support") or {}
    nr = lv.get("nearest_resistance") or {}
    print(
        f"   Nearest Support   : {ns.get('price', '-')}  "
        f"(score {ns.get('score', '-')}, {ns.get('strength', '-')})"
    )
    print(
        f"   Nearest Resistance: {nr.get('price', '-')}  "
        f"(score {nr.get('score', '-')}, {nr.get('strength', '-')})"
    )

    print("   Top confluence zones:")
    for zone in lv.get("top_confluence", [])[:5]:
        print(
            f"     - {zone['price']:>10.5f}  score {zone['score']}  "
            f"{zone['strength']:<6}  {','.join(zone['tags'])}"
        )

    av = lv.get("anchored_vwap")
    if av:
        print(f"   Anchored VWAP: {av:,.5f}")
    vp = lv.get("volume_profile") or []
    high_vol = [n for n in vp if n.get("is_high_volume")]
    if high_vol:
        print(
            f"   Volume-profile high-volume nodes ({len(high_vol)}): "
            + ", ".join(
                f"{n['price']:,.5f} ({n['volume_pct']:.1f}%)" for n in high_vol[:5]
            )
        )

    ml = report.get("ml")
    if ml:
        print("\n8. ENSEMBLE MODEL (bullish probability)")
        print(f"   P(1R move up): {ml['prob_pct']}%  ({ml['label']})")
        m = ml.get("model") or {}
        if m.get("auc_oos") is not None:
            print(
                f"   Model OOS AUC: {m['auc_oos']}  · trained on "
                f"{m.get('symbols', '-')} symbols"
            )
        imp = ml.get("importance")
        if imp:
            print("   Feature importance (gain, spec #10):")
            for g in imp["by_group"][:6]:
                print(f"     {g['group']:<11} {g['gain_pct']:>6.2f}%")
            top3 = ", ".join(
                f"{t['feature']} {t['gain_pct']:.0f}%" for t in imp["top"][:3]
            )
            print(f"     top: {top3}")

    rk = report.get("risk")
    if rk:
        print("\n9. RISK & POSITION SIZING")
        setup = rk.get("setup")
        if not setup:
            print(f"   {rk.get('reason', 'no actionable setup')}")
        else:
            print(
                f"   Setup: entry {setup['entry']:,.5f} · "
                f"stop {setup['stop']:,.5f} · target {setup['target']:,.5f} · "
                f"R:R {setup['rr']}"
            )
            ok = setup.get("rr_ok")
            if ok is not None:
                mark = "✓ meets floor" if ok else "✗ below floor"
                print(
                    f"   R:R floor {setup.get('min_rr', 2.5)}:1 → {mark} · "
                    f"nearest target {setup.get('rr_nearest', '-')} · "
                    f"ladder best {setup.get('best_rr', '-')} "
                    f"({setup.get('min_rr_tp') or 'none'})"
                )
            print(
                f"   Kelly p: {rk['inputs']['kelly_p']} "
                f"({'ML model' if setup['ml_prob'] is not None else 'fallback'}) · "
                f"hold {rk['inputs']['hold_bars']} bars · "
                f"equity {rk['inputs']['equity']:,.0f}"
            )
            for row in rk["sizes"]:
                print(
                    f"   {row['method']:<11} qty {row['qty']:>10,.2f} · "
                    f"risk {row['risk_usd']:>8,.0f} "
                    f"({row['risk_pct_equity']:>5.2f}%) · "
                    f"VaR95 1d {row['var_95_1bar']:>9,.0f}"
                )

    mc = report.get("macro")
    if mc:
        print("\n10. MACRO OVERLAY")
        rg = mc.get("regime") or {}
        print(f"   USD regime : {rg.get('usd')} (score {rg.get('dxy_score')})")
        print(f"   Risk regime: {rg.get('risk')} (score {rg.get('vix_score')})")
        print(f"   Rates regime: {rg.get('rates')} (score {rg.get('tnx_score')})")
        print(f"   Composite  : {rg.get('composite')}")
        bs = mc.get("bias") or {}
        gt = mc.get("gate") or {}
        print(
            f"   Macro bias : {bs.get('bias')} ({bs.get('label')}) — {bs.get('note')}"
        )
        print(
            f"   Gate       : {'PASS' if gt.get('allowed') else 'BLOCKED'} "
            f"({gt.get('reason')})"
        )
        ss = mc.get("sensitivities") or {}
        if ss:
            print(f"   Sensitivities (trailing {ss.get('lookback', 90)}d, 1d-lagged):")
            print(
                f"     Market beta vs S&P500 : {ss.get('market_beta', '-')}  "
                f"(corr {ss.get('spx_corr', '-')})"
            )
            print(f"     Dollar sensitivity   : {ss.get('dollar_sens', '-')}")
            print(f"     Yield sensitivity    : {ss.get('yield_sens', '-')}")
            print(f"     Volatility sensitivity: {ss.get('vol_sens', '-')}")
            se = ss.get("sector_etf")
            if se and se.get("ticker"):
                print(
                    f"     Sector ETF {se['ticker']:<4} corr      : "
                    f"{se.get('corr', '-')}"
                )

    print("\n11. BUY-THE-DIP CONFIRMATION")
    dip = report.get("dip", {})
    print(f"   Score: {dip.get('dip_score', '-')}/8")
    print(
        f"   Status: {dip.get('dip_stage', '-')}  "
        f"({('CONFIRMED' if dip.get('dip_confirmed') else 'not confirmed')})"
    )
    ez = dip.get("entry_zone")
    if ez:
        print(f"   Entry zone: {ez[0]} -> {ez[1]}")
    else:
        print("   Entry zone: -")
    print(f"   Invalidation: {dip.get('invalidation_level', '-')}")
    print(f"   Target (nearest resistance): {dip.get('target', '-')}")
    print(f"   Dip depth: {dip.get('dip_depth_pct', '-')}%")
    print(f"   Trigger: {dip.get('trigger', '-')}")
    print(f"   Components: {dip.get('components', {})}")

    print("\n11b. SELL-THE-RALLY CONFIRMATION (short)")
    rl = report.get("rally", {})
    print(f"   Score: {rl.get('rally_score', '-')}/8")
    print(
        f"   Status: {rl.get('rally_stage', '-')}  "
        f"({('CONFIRMED' if rl.get('rally_confirmed') else 'not confirmed')})"
    )
    rez = rl.get("entry_zone")
    if rez:
        print(f"   Entry zone: {rez[0]} -> {rez[1]}")
    else:
        print("   Entry zone: -")
    print(f"   Invalidation: {rl.get('invalidation_level', '-')}")
    print(f"   Target (nearest support): {rl.get('target', '-')}")
    print(f"   Rally depth: {rl.get('rally_depth_pct', '-')}%")
    print(f"   Trigger: {rl.get('trigger', '-')}")
    srs = report.get("short_risk") or {}
    sst = srs.get("setup")
    if sst:
        sok = sst.get("rr_ok")
        mark = "✓ meets floor" if sok else "✗ below floor"
        print(
            f"   Short setup: entry {sst['entry']:,.5f} · "
            f"stop {sst['stop']:,.5f} · target {sst['target']:,.5f} · "
            f"R:R {sst['rr']}"
        )
        if sok is not None:
            print(
                f"   R:R floor {sst.get('min_rr', 2.5)}:1 → {mark} · "
                f"nearest target {sst.get('rr_nearest', '-')} · "
                f"ladder best {sst.get('best_rr', '-')} "
                f"({sst.get('min_rr_tp') or 'none'})"
            )
        stg = srs.get("targets") or report.get("short_targets") or {}
        for t in (stg.get("targets") or [])[:3]:
            print(
                f"   {t['target']:<4} {t['price']:>12,.5f} · R:R {t['rr']:>4.2f} · "
                f"{t['source']}"
            )

    plan = report.get("plan")
    if plan:
        print("\n11c. TRADE PLAN / ACTION")
        print(format_plan(plan))

    sc = report.get("setup_classification")
    if sc:
        print("\n11d. DIRECTION-NEUTRAL SETUP CLASSIFICATION")
        fam = sc.get("setup_family") or "-"
        print(
            f"   Direction: {sc.get('direction', '-').upper():<6} · best family {fam}"
        )
        print(
            f"   Long evidence {sc.get('long_score', 0):.2f} · "
            f"Short evidence {sc.get('short_score', 0):.2f}"
        )
        p_l = sc.get("prob_long")
        p_s = sc.get("prob_short")
        if p_l is not None or p_s is not None:
            print(
                f"   P(long) {('%.0f%%' % (p_l * 100)) if p_l is not None else '-'} · "
                f"P(short) {('%.0f%%' % (p_s * 100)) if p_s is not None else '-'}"
            )
        ev = sc.get("ev")
        pw = sc.get("pw_rr")
        if ev is not None:
            print(f"   Expected value: {ev:+.3f}R")
        if pw is not None:
            print(f"   Probability-weighted R:R: {pw:+.3f}")
        top_l = next(iter(sc.get("long_families", {}).items()), None)
        top_s = next(iter(sc.get("short_families", {}).items()), None)
        print(
            f"   Best long family: {top_l[0] if top_l else '-'} ({top_l[1]:.2f})"
            if top_l
            else "   Best long family: -"
        )
        print(
            f"   Best short family: {top_s[0] if top_s else '-'} ({top_s[1]:.2f})"
            if top_s
            else "   Best short family: -"
        )
        evd = sc.get("evidence") or []
        if evd:
            print("   Evidence:")
            for e in evd[:8]:
                print(f"     - {e}")

    ob = report.get("opportunity_book")
    if ob:
        print("\n11e. OPPORTUNITY BOOK (unified LONG/SHORT/FLAT decision)")
        v = ob.get("verdict") or {}
        print(
            f"   VERDICT: {str(v.get('direction', '-')).upper()} ({v.get('status', '-')})"
        )
        if v.get("expected_r") is not None:
            print(f"   Expected EV: {v['expected_r']:+.3f}R")
        if v.get("reason"):
            print(f"   Why: {v['reason']}")
        for side in ("long", "short"):
            opp = ob.get(side) or {}
            fam = opp.get("setup_family") or "-"
            p = opp.get("probability")
            ev = opp.get("expected_r")
            print(
                f"   {side.upper():<6} {fam:<26} P "
                f"{'-' if p is None else f'{p:.0%}':>4} · "
                f"EV {'-' if ev is None else f'{ev:+.2f}R':>8} · "
                f"RR {opp.get('rr') if opp.get('rr') is not None else '-':<5} · "
                f"{'TAKEN' if opp.get('taken') else 'rejected'}"
            )
            rej = opp.get("rejection_reasons") or []
            for r in rej[:3]:
                print(f"       ✗ {r}")

    vf = report.get("volume_flow")
    if vf and vf.get("available"):
        print("\n12. VOLUME & FLOW")
        print(f"   OBV slope (20)     : {vf['obv_slope_20']} ({vf['obv_trend']})")
        print(f"   A/D line slope (20): {vf['ad_line_slope_20']} ({vf['ad_trend']})")
        print(f"   Relative volume    : {vf['relative_volume']}x 20d avg")
        print(f"   Volume delta (20)  : {vf['volume_delta_20']:,.0f}")
        print(
            f"   Buyer vs Seller    : {vf['buyer_seller_score']:+.0f} "
            f"({vf['buyer_seller_label']})"
        )

    dv = report.get("divergences") or {}
    if dv.get("signals"):
        print("\n12b. DIVERGENCES (>=65% confidence)")
        for d in dv["signals"]:
            # Shape-agnostic detail (regular/hidden vs failure swings) -
            # single source of truth in divergence.format_divergence.
            print(
                f"   - {d['name']:<32} {d['side']:>8} · "
                f"{format_divergence(d)} · conf {d['confidence']}"
            )
    elif dv:
        # The engine always runs; on quiet days it finds nothing above the
        # 65% confidence bar - say so explicitly (same empty-state as the
        # pattern section) instead of making the section look unwired.
        print("\n12b. DIVERGENCES (>=65% confidence)")
        print("   No divergence above the 65% confidence threshold.")

    fm = report.get("fib_map") or []
    if fm:
        print("\n12c. FIBONACCI CONFLUENCE MAP")
        print(
            f"   {'Level':<8} {'Side':<6} {'Price':>12} {'Confluence':>6} {'Dist%':>8}"
        )
        for row in fm[:8]:
            print(
                f"   {row['level']:<8} {row['side']:<6} {row['price']:>12,.5f} "
                f"{row['confluence']:>6.1f} {row['distance_from_close_pct']:>7.2f}%"
            )

    pt = report.get("patterns") or {}
    if pt.get("patterns"):
        print("\n13. PATTERN RECOGNITION (>=65% structural confidence)")
        for p in pt["patterns"]:
            print(
                f"   - {p['name']:<28} {p['side']:>8} · breakout "
                f"{p['breakout']:>10.5f} · conf "
                f"{p.get('confidence', p['prob']):>3} · {p['status']}"
            )
            print(f"       {p['detail']}")
        print(
            "   (confidence = structural quality score 0-100, NOT a "
            "calibrated win probability)"
        )
    else:
        print("\n13. PATTERN RECOGNITION (>=65% structural confidence)")
        print("   No pattern above the 65% confidence threshold.")

    tg = report.get("targets")
    if tg and tg.get("targets"):
        print("\n14. TARGET LADDER (scaling-out plan)")
        for t in tg["targets"]:
            print(
                f"   {t['target']:<4} {t['price']:>12,.5f} · R:R {t['rr']:>4.2f} · "
                f"{t['source']}"
            )
        print(
            f"   Best R:R: {tg['best_rr']} · first TP >= {tg['min_rr']}:1 "
            f"→ {tg['min_rr_tp'] or 'none'}"
        )

    rt = report.get("rating")
    if rt:
        print("\n15. FINAL QUANT RATING")
        src = {
            "ml+factors": "ML + factor stack",
            "rule": "rule-based",
            "explicit": "explicit",
        }.get(rt.get("source", ""), "")
        print(f"   Bullish probability: {rt['prob_pct']}%  ({src})")
        print(f"   Rating: {rt['rating']} — {rt['recommendation']}")
        for c in rt["contributions"]:
            if abs(c["contribution"]) >= 0.05:
                print(
                    f"   {c['factor']:<14}: {c['contribution']:+.1f}%  "
                    f"(score {c['score']:+.2f})"
                )
        if abs(rt.get("unexplained", 0)) >= 0.5:
            print(
                f"   {'Model / other':<14}: {rt['unexplained']:+.1f}%  "
                f"(not explained by factors)"
            )

    fu = report.get("fundamentals")
    if fu:
        print("\n16. FUNDAMENTAL FACTORS (equity)")
        print(
            f"   Value {fu.get('value')} · Quality {fu.get('quality')} · "
            f"Momentum {fu.get('momentum')} · Composite {fu.get('composite')}"
        )
        print(
            f"   Sources: {', '.join(fu.get('sources', []))} · "
            f"fundamentals: {fu.get('fundamentals_source', 'none')}"
        )

    se = report.get("sentiment")
    if se and se.get("available"):
        print("\n17. NEWS & SOCIAL SENTIMENT")
        n = se.get("news") or {}
        print(
            f"   News: {n.get('score')} "
            f"({n.get('n_articles', 0)} articles, "
            f"{n.get('relevant', 0)} relevant, {n.get('source')})"
        )
        print(f"   Composite sentiment: {se.get('composite')}")

    st = report.get("stress")
    if st and st.get("available"):
        print("\n18. STRESS TEST (2008 / COVID-2020 / 2022)")
        hist = st.get("historical") or {}
        if hist.get("covid_dd_pct") is not None:
            print(
                f"   realized: COVID worst 23d dd {hist['covid_dd_pct']}% · "
                f"vol mult {hist.get('covid_vol_mult') or '-'} · "
                f"2022+ worst dd {hist.get('dd_2022_pct')}%"
            )
        for row in st.get("scenarios", []):
            mark = (
                "⚠ 1dVaR>limit"
                if row["daily_limit_breach"]
                else "loss>cap"
                if row["scenario_cap_breach"]
                else "ok"
            )
            print(
                f"   {row['scenario']:<16} dd {row['drawdown_pct']:>5.1f}% · "
                f"loss {row['loss_usd']:>12,.0f} "
                f"({row['loss_pct_equity']:>5.2f}% eq ≈ "
                f"{row['days_of_daily_limit']}d limit) · "
                f"VaR95 {row['var95_stress']:>11,.0f} · 1d "
                f"{row['var95_1d_pct_equity']:>4.2f}% · {mark}"
            )
        print(
            "   (loss=gap-through; VaR95 over scenario horizon; 1d=shocked "
            "1-day VaR vs daily loss limit)"
        )

    sts = report.get("stress_short")
    if sts and sts.get("available"):
        print("\n18b. SHORT BOOK STRESS (Sell-the-Rally setup)")
        print(
            "   (crash scenarios are favorable to shorts: negative loss = "
            "gain; VaR95 still flags squeeze risk)"
        )
        for row in sts.get("scenarios", []):
            print(
                f"   {row['scenario']:<16} loss {row['loss_usd']:>12,.0f} "
                f"({row['loss_pct_equity']:>+5.2f}% eq) · "
                f"VaR95 1d {row['var95_1d_pct_equity']:>4.2f}%"
            )

    print("=" * 60 + "\n")


if __name__ == "__main__":
    print("NexusQuant Analysis Report module ready.")
    print("Usage: from src.analysis.report import generate_full_report, print_report")
