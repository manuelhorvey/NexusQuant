"""
NexusQuant - Unified Trade Plan / Decision Layer (institutional spec #11/#14).

For ANY symbol this synthesizes the two confirmation engines (Buy-the-Dip
and Sell-the-Rally), the ensemble probabilities, the final rating and the
macro gate into ONE actionable verdict:

    CONFIRMED LONG   -> "BUY-LIMIT  <zone>"    place a limit buy in the zone
    CONFIRMED SHORT  -> "SELL-LIMIT <zone>"    place a limit sell in the zone
    WATCH LONG       -> "WAIT-LONG  <zone>"    pending limit buy - not confirmed yet
    WATCH SHORT      -> "WAIT-SHORT <zone>"    pending limit sell - not confirmed yet
    NO SETUP         -> "NO-SETUP"             stand aside (+ what would change the read)

So running the system on ANY symbol always answers: which side, at which
price, and what to wait for when nothing is confirmed yet.

Two adapters share ONE decision function:

* ``trade_plan(report)``     - the full institutional report dict
* ``scanner_action(row)``    - a flat scanner row (used by the ranking table)

The verdict rules mirror the live filter (settings ``live.filters``):
a setup is CONFIRMED only when the engine confirms it (score >= ``min_score``,
default 5, aligned with ``min_dip_score``); a WATCH requires the engine's
watch stage (``In Pullback`` / ``In Rally`` or deeper) with an actionable
zone; everything else is NO-SETUP with the levels that would flip the read.
"""

from __future__ import annotations

import sys
from typing import Dict, Optional

# Aligned with settings: live.filters.min_dip_score (the long side) and the
# short mirror. A confirmed engine stage implies score >= 6, so >= 5 always
# holds for confirmed setups; the floor protects against low-score edge cases.
MIN_SETUP_SCORE = 5
WATCH_SCORE = 4

WATCH_LONG_STAGES = {"In Pullback", "Deep Pullback"}
WATCH_SHORT_STAGES = {"In Rally", "Deep Rally"}

# Ratings that argue against the opposite side (rating-driven "avoid" flag).
AVOID_LONG_RATINGS = {"Sell", "Strong Sell"}
AVOID_SHORT_RATINGS = {"Buy", "Strong Buy"}


def _as_bool(v) -> bool:
    if isinstance(v, bool):
        return v
    if isinstance(v, str):
        return v.strip().lower() in ("yes", "true", "1")
    return bool(v)


def _zone_text(zone) -> Optional[str]:
    """Normalize a (lo, hi) tuple or a 'lo-hi' string to 'lo-hi' text."""
    if zone is None:
        return None
    if isinstance(zone, (tuple, list)) and len(zone) == 2:
        try:
            return f"{float(zone[0]):.5f}-{float(zone[1]):.5f}"
        except (TypeError, ValueError):
            return None
    text = str(zone)
    return text if "-" in text else None


def _num(v) -> Optional[float]:
    try:
        return None if v is None else float(v)
    except (TypeError, ValueError):
        return None


def decide_plan(
    *,
    symbol: str,
    close: Optional[float] = None,
    date: Optional[str] = None,
    dip_stage: Optional[str] = None,
    dip_score=0,
    dip_confirmed=False,
    dip_zone=None,
    dip_inv: Optional[float] = None,
    dip_target: Optional[float] = None,
    long_best_rr: Optional[float] = None,
    long_rr_ok: Optional[bool] = None,
    rally_stage: Optional[str] = None,
    rally_score=0,
    rally_confirmed=False,
    rally_zone=None,
    rally_inv: Optional[float] = None,
    rally_target: Optional[float] = None,
    short_best_rr: Optional[float] = None,
    short_rr_ok: Optional[bool] = None,
    ml_pct: Optional[float] = None,
    ml_short_pct: Optional[float] = None,
    rating_pct: Optional[float] = None,
    rating_label: Optional[str] = None,
    gate_allowed: Optional[bool] = None,
    support: Optional[float] = None,
    resistance: Optional[float] = None,
    min_score: int = MIN_SETUP_SCORE,
    watch_score: int = WATCH_SCORE,
) -> Dict:
    """
    The single decision function: one verdict per symbol.

    Returns a plan dict with ``direction`` (long/short/neutral), ``status``
    (CONFIRMED/WATCH/NO_SETUP), a compact ``action``, a ``summary``, the
    per-side levels and ``what_changes`` (the price levels/conditions that
    would flip the read).
    """
    dip_score = int(dip_score or 0)
    rally_score = int(rally_score or 0)

    long_c = _as_bool(dip_confirmed) and dip_score >= min_score
    short_c = _as_bool(rally_confirmed) and rally_score >= min_score

    long_watch = (
        not long_c
        and dip_score >= watch_score
        and (dip_stage or "") in WATCH_LONG_STAGES
        and _zone_text(dip_zone) is not None
    )
    short_watch = (
        not short_c
        and rally_score >= watch_score
        and (rally_stage or "") in WATCH_SHORT_STAGES
        and _zone_text(rally_zone) is not None
    )

    if long_c:
        direction, status = "long", "CONFIRMED"
    elif short_c:
        direction, status = "short", "CONFIRMED"
    elif long_watch and short_watch:
        # both sides are forming - lean to the stronger score (tie -> rating)
        if dip_score > rally_score:
            direction, status = "long", "WATCH"
        elif rally_score > dip_score:
            direction, status = "short", "WATCH"
        else:
            direction = "short" if (rating_pct or 50) < 50 else "long"
            status = "WATCH"
    elif long_watch:
        direction, status = "long", "WATCH"
    elif short_watch:
        direction, status = "short", "WATCH"
    else:
        direction, status = "neutral", "NO_SETUP"

    long_zone = _zone_text(dip_zone)
    short_zone = _zone_text(rally_zone)

    gate = (
        "PASS"
        if gate_allowed is not False
        else ("BLOCKED" if gate_allowed is False else None)
    )

    if status == "CONFIRMED" and direction == "long":
        action = f"BUY-LIMIT {long_zone or '-'}"
    elif status == "CONFIRMED" and direction == "short":
        action = f"SELL-LIMIT {short_zone or '-'}"
    elif status == "WATCH" and direction == "long":
        action = f"WAIT-LONG {long_zone or '-'}"
    elif status == "WATCH" and direction == "short":
        action = f"WAIT-SHORT {short_zone or '-'}"
    else:
        action = "NO-SETUP"
    if status != "NO_SETUP" and gate == "BLOCKED":
        action += " · macro BLOCKED"

    # ---- what would change the read --------------------------------------
    if status == "NO_SETUP":
        sup = "-" if support is None else f"{support:,.5f}"
        res = "-" if resistance is None else f"{resistance:,.5f}"
        what_changes = (
            f"No structural setup. Long would need a fresh pullback into the "
            f"{sup} support area (or a 200-SMA reclaim with bullish bias); "
            f"short would need a close below the 200-SMA with bearish bias "
            f"and a rally into the {res} resistance area to fade."
        )
    elif direction == "long":
        inv = "-" if dip_inv is None else f"{dip_inv:,.5f}"
        what_changes = (
            f"Long activates when price pulls back into {long_zone} and the "
            f"dip trigger fires (MACD/RSI/bar) while holding above {inv}. "
            f"Short side would need a confirmed rally into {short_zone or '-'} "
            f"with a rejection."
        )
    else:
        inv = "-" if rally_inv is None else f"{rally_inv:,.5f}"
        what_changes = (
            f"Short activates when price rallies into {short_zone} and the "
            f"rally trigger fires while holding below {inv}. Long side would "
            f"need a confirmed pullback into {long_zone or '-'}."
        )

    # ---- one-line summary ------------------------------------------------
    if status == "CONFIRMED":
        summary = f"{symbol}: {action} — confirmed {'Buy-the-Dip' if direction == 'long' else 'Sell-the-Rally'}"
    elif status == "WATCH":
        stage = dip_stage if direction == "long" else rally_stage
        summary = (
            f"{symbol}: {action} — not confirmed yet ({stage} stage), pending limit"
        )
    else:
        summary = f"{symbol}: no confirmed setup on either side — stand aside"

    return {
        "symbol": symbol,
        "date": date,
        "close": _num(close),
        "direction": direction,
        "status": status,
        "action": action,
        "summary": summary,
        "long": {
            "active": direction == "long" and status == "CONFIRMED",
            "watch": long_watch,
            "stage": dip_stage,
            "score": dip_score,
            "entry_zone": long_zone,
            "stop": _num(dip_inv),
            "target": _num(dip_target),
            "best_rr": _num(long_best_rr),
            "rr_ok": long_rr_ok,
        },
        "short": {
            "active": direction == "short" and status == "CONFIRMED",
            "watch": short_watch,
            "stage": rally_stage,
            "score": rally_score,
            "entry_zone": short_zone,
            "stop": _num(rally_inv),
            "target": _num(rally_target),
            "best_rr": _num(short_best_rr),
            "rr_ok": short_rr_ok,
        },
        "ml_long_pct": _num(ml_pct),
        "ml_short_pct": _num(ml_short_pct),
        "rating_pct": _num(rating_pct),
        "rating_label": rating_label,
        "gate": gate,
        "what_changes": what_changes,
    }


def trade_plan(report: Dict) -> Dict:
    """Adapt the full institutional report dict (generate_full_report)."""
    d = report.get("dip") or {}
    r = report.get("rally") or {}
    ml = report.get("ml") or {}
    ml_s = report.get("ml_short") or {}
    rt = report.get("rating") or {}
    macro = report.get("macro") or {}
    lv = report.get("levels") or {}
    risk = report.get("risk") or {}
    srisk = report.get("short_risk") or {}
    setup = risk.get("setup") or {}
    ssetup = srisk.get("setup") or {}
    tgt = report.get("targets") or {}
    stgt = report.get("short_targets") or {}
    plan = decide_plan(
        symbol=str(report.get("symbol", "?")),
        close=report.get("last_close"),
        date=report.get("last_date"),
        dip_stage=d.get("dip_stage"),
        dip_score=d.get("dip_score", 0),
        dip_confirmed=d.get("dip_confirmed", False),
        dip_zone=d.get("entry_zone"),
        dip_inv=d.get("invalidation_level"),
        dip_target=d.get("target"),
        long_best_rr=tgt.get("best_rr") or setup.get("best_rr"),
        long_rr_ok=setup.get("rr_ok"),
        rally_stage=r.get("rally_stage"),
        rally_score=r.get("rally_score", 0),
        rally_confirmed=r.get("rally_confirmed", False),
        rally_zone=r.get("entry_zone"),
        rally_inv=r.get("invalidation_level"),
        rally_target=r.get("target"),
        short_best_rr=stgt.get("best_rr") or ssetup.get("best_rr"),
        short_rr_ok=ssetup.get("rr_ok"),
        ml_pct=ml.get("prob_pct"),
        ml_short_pct=ml_s.get("prob_pct"),
        rating_pct=rt.get("prob_pct"),
        rating_label=rt.get("rating"),
        gate_allowed=(macro.get("gate") or {}).get("allowed"),
        support=(lv.get("nearest_support") or {}).get("price"),
        resistance=(lv.get("nearest_resistance") or {}).get("price"),
    )
    # Direction-neutral setup classification (spec: direction first, setup
    # second, 200-SMA contextual not a gate). Prefer the classifier's
    # direction when the engines have no confirmed setup - it can see
    # breakout/breakdown/retest/reversal families the engines cannot.
    setup_cls = report.get("setup_classification") or {}
    if setup_cls:
        plan["setup_family"] = setup_cls.get("setup_family")
        plan["classifier_direction"] = setup_cls.get("direction")
        plan["long_evidence"] = setup_cls.get("long_score")
        plan["short_evidence"] = setup_cls.get("short_score")
        plan["prob_long"] = setup_cls.get("prob_long")
        plan["prob_short"] = setup_cls.get("prob_short")
        plan["ev"] = setup_cls.get("ev")
        plan["pw_rr"] = setup_cls.get("pw_rr")
        plan["setup_evidence"] = setup_cls.get("evidence")

    # Opportunity-book verdict merge (two-sided campaign, forensic fix):
    # the action column must reflect the FULL opportunity space, not just
    # the 200-SMA-gated engine confirmations. When the EV-driven book
    # produces a TRADE verdict with concrete levels, that side wins the
    # action - a SELL-LIMIT can now fire above the 200-SMA when the short
    # hypothesis has higher expected value, and vice versa. The engine
    # path stays as the fallback for reports without a book (or a book
    # without a TRADE verdict).
    book = report.get("opportunity_book") or {}
    vd = book.get("verdict") or {}
    sym = str(report.get("symbol", "?"))
    if vd.get("direction") in ("long", "short") and vd.get("status") == "TRADE":
        opp = book.get(vd["direction"]) or {}
        zone = opp.get("entry_zone")
        if zone:
            side = vd["direction"]
            price = f"{float(zone[0]):.5f}" if len(zone) == 2 else str(zone[0])
            plan["direction"] = side
            plan["status"] = "CONFIRMED"
            order_kind = (opp.get("entry_type") or "limit").upper()
            plan["entry_type"] = order_kind
            plan["action"] = (
                f"{'BUY' if side == 'long' else 'SELL'}-{order_kind} {price}"
            )
            plan["decision_source"] = "opportunity_book"
            plan["expected_r"] = vd.get("expected_r")
            side_dict = plan["long"] if side == "long" else plan["short"]
            side_dict.update(
                {
                    "active": True,
                    "watch": False,
                    "entry_zone": f"{float(zone[0]):.5f}-{float(zone[1]):.5f}"
                    if len(zone) == 2
                    else str(zone[0]),
                    "stop": opp.get("invalidation"),
                    "target": opp.get("target"),
                    "best_rr": opp.get("rr"),
                    "rr_ok": bool((opp.get("rr") or 0) >= 2.5),
                }
            )
            plan["summary"] = (
                f"{sym}: {plan['action']} - EV verdict "
                f"({vd.get('expected_r') if vd.get('expected_r') is not None else 'n/a'}R) "
                f"from opportunity book"
            )
    return plan


def scanner_action(row: Dict) -> str:
    """Compact action label for a flat scanner row (ranking table)."""

    def _conf(v) -> bool:
        return (
            str(v).strip().lower() in ("yes", "true", "1")
            if not isinstance(v, bool)
            else v
        )

    gate_allowed = row.get("macro_gate")
    if gate_allowed is not None:
        gate_allowed = str(gate_allowed).upper() == "PASS"
    return decide_plan(
        symbol=str(row.get("symbol", "?")),
        dip_stage=row.get("dip_stage"),
        dip_score=row.get("dip_score", 0),
        dip_confirmed=_conf(row.get("dip_confirmed", False)),
        dip_zone=row.get("entry_zone"),
        dip_inv=row.get("invalidation"),
        dip_target=row.get("resistance"),
        long_best_rr=row.get("best_rr"),
        rally_stage=row.get("rally_stage"),
        rally_score=row.get("rally_score", 0),
        rally_confirmed=_conf(row.get("rally_confirmed", False)),
        rally_zone=row.get("short_entry_zone"),
        rally_inv=row.get("short_invalidation"),
        rally_target=row.get("support"),
        short_best_rr=row.get("short_best_rr"),
        ml_pct=row.get("ml_prob"),
        ml_short_pct=row.get("ml_short_prob"),
        rating_label=row.get("rating"),
        gate_allowed=gate_allowed,
        support=row.get("support"),
        resistance=row.get("resistance"),
    )["action"]


def format_plan(plan: Dict) -> str:
    """Multi-line rendering for the institutional report (section 11c)."""
    lines = [f"   → {plan['action']}"]
    side = plan["long"] if plan["direction"] == "long" else plan["short"]
    other = plan["short"] if plan["direction"] == "long" else plan["long"]
    if side.get("entry_zone"):
        lines.append(
            f"   Entry {side['entry_zone']} · stop "
            f"{'-' if side.get('stop') is None else f'{side['stop']:,.5f}'} · "
            f"target {'-' if side.get('target') is None else f'{side['target']:,.5f}'}"
        )
    if plan.get("ml_long_pct") is not None or plan.get("ml_short_pct") is not None:
        lines.append(
            f"   ML long {plan.get('ml_long_pct', 0) or 0:.0f}% / "
            f"short {plan.get('ml_short_pct', 0) or 0:.0f}% · "
            f"rating {plan.get('rating_pct', 0) or 0:.1f}% "
            f"({plan.get('rating_label', '-')})"
        )
    # Direction-neutral setup family + expected value (two-sided audit):
    # the classifier's verdict surfaces even when the engines have no
    # confirmed setup, so the plan always states WHICH setup family the
    # evidence favors (LONG_BREAKOUT_RETEST / SHORT_BREAKDOWN / ...).
    if plan.get("setup_family"):
        ev_txt = "" if plan.get("ev") is None else f" · EV {plan['ev']:+.2f}R"
        lines.append(
            f"   Setup family: {plan['setup_family']} "
            f"(long evid {plan.get('long_evidence', 0):.2f} / "
            f"short evid {plan.get('short_evidence', 0):.2f}){ev_txt}"
        )
    if other.get("entry_zone") and plan["status"] == "WATCH":
        lines.append(
            f"   Opposite watch: pending "
            f"{'SELL-LIMIT' if plan['direction'] == 'long' else 'BUY-LIMIT'} "
            f"{other['entry_zone']}"
        )
    if plan.get("gate"):
        lines.append(f"   Macro gate: {plan['gate']}")
    lines.append(f"   What changes: {plan['what_changes']}")
    return "\n".join(lines)


def print_plan_table(
    symbols,
    group: str = "full_fx",
    timeframe: str = "D1",
    data_dir: str = "data/raw",
) -> int:
    """Build the full report per symbol and print a compact DECISION table.

    Used by ``python -m src.live.run --format plan`` - every symbol gets a
    row with its action (BUY-LIMIT / SELL-LIMIT / WAIT-* / NO-SETUP) plus
    the key levels and probabilities, so a run on ANY symbol list always
    answers the trade question.
    """
    from src.analysis.report import generate_full_report
    from src.analysis.scanner import _data_path, discover_symbols
    from src.data.loader import clean_data, load_data
    from src.features.indicators import add_all_indicators
    from src.features.regime import detect_regime

    if symbols is None:
        symbols = discover_symbols(data_dir, group, timeframe)
    if not symbols:
        print("No symbols found.", file=sys.stderr)
        return 1

    header = (
        f"{'SYMBOL':<8}{'CLOSE':>11}  {'ACTION':<30}{'SETUP':<24}"
        f"{'LONG (stage · zone · stop · target)':<40}"
        f"{'SHORT (stage · zone · stop · target)':<40}{'ML':>10}  "
        f"{'RATING':<16}{'GATE':<8}"
    )
    print("\n" + "=" * 152)
    print("NEXUSQUANT TRADE PLAN — DECISION TABLE")
    print("=" * 152)
    print(header)
    print("-" * 152)

    n = 0
    for sym in symbols:
        try:
            path = _data_path(sym, data_dir, group, timeframe)
            df = clean_data(load_data(path, symbol=sym))
            df = add_all_indicators(df)
            df = detect_regime(df)
            report = generate_full_report(
                df, symbol=sym, group=group, data_dir=data_dir
            )
        except Exception as exc:
            print(f"{sym:<8}  ERROR: {exc}")
            continue
        plan = report["plan"]

        def _side(side_dict: Dict) -> str:
            if not side_dict.get("entry_zone"):
                return f"{side_dict.get('stage') or '-':<14} -"
            return (
                f"{(side_dict.get('stage') or ''):<14} {side_dict['entry_zone']:<14} "
                f"{'-' if side_dict.get('stop') is None else f'{side_dict['stop']:,.4f}':<11} "
                f"{'-' if side_dict.get('target') is None else f'{side_dict['target']:,.4f}'}"
            )

        ml = f"{plan.get('ml_long_pct', 0) or 0:.0f}%/{plan.get('ml_short_pct', 0) or 0:.0f}%"
        rating = (
            f"{plan.get('rating_pct', 0) or 0:.0f}% {plan.get('rating_label') or '-'}"
        )
        gate = plan.get("gate") or "-"
        setup = plan.get("setup_family") or (
            "-" if plan["direction"] == "neutral" else "?"
        )
        print(
            f"{sym:<8}{report['last_close']:>11,.3f}  {plan['action']:<30}{setup:<24}"
            f"{_side(plan['long']):<40}{_side(plan['short']):<40}"
            f"{ml:>10}  {rating:<16}{gate:<8}"
        )
        n += 1
    print("-" * 152)
    print(
        f"{n} symbol(s) · BUY/SELL-LIMIT = place the order at that zone now; "
        f"WAIT-* = pending limit, not confirmed yet; NO-SETUP = stand aside. "
        f"See the institutional report per symbol for full detail."
    )
    print("=" * 152 + "\n")
    return 0


if __name__ == "__main__":
    print("NexusQuant Trade Plan module ready.")
