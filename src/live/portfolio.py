"""
NexusQuant - Portfolio-Level Order Selection (Stage-10).

The live pass is alert-based: every symbol that clears its own filters is
emitted. Stage-9A flagged this as the open correlation gap - the research
already showed one-per-currency-cluster caps cut cumulative-R drawdown
(31.3R -> 24.2R), but nothing enforced that at the order layer. This
module closes it: after the LONG/SHORT passes are merged, a portfolio
selection pass caps the proposed orders by currency-cluster concentration,
max concurrent positions and portfolio heat - rejecting individually
attractive trades whose *marginal* contribution to portfolio risk is poor.

Selection policy (conservative, documented defaults; overridable):

1. Sort candidates by their book's decision EV (target-level allocation EV
   when present, else ranking EV) descending - evidence first.
2. Max-concurrent cap (default 6) - a hard bound on simultaneous orders.
3. One-per-currency-cluster cap (default 1) - EURJPY + GBPJPY + CADJPY +
   AUDJPY + NZDJPY are ONE JPY-short bet, not five independent ones; the
   cluster view surfaces the shared leg (spec #28/#29 currency exposure).
   Same for USD / CHF / EUR / GBP legs.
4. Portfolio heat cap (default 4% of equity, aligned with RiskManager) -
   sum of per-trade risk dollars / equity must stay under the bound.

Every rejection is explainable: a dropped order carries
``portfolio_rejected`` with the exact reason (cluster cap / max concurrent /
heat), so the output never silently disappears a trade.

This is a pure function over the merged alerts (each carrying its report ->
opportunity_book verdict) - no data access, no side effects.
"""

from __future__ import annotations

from typing import Dict, List, Optional

# Currency clusters: members of the same cluster share a quote/base leg
# (EURJPY + GBPJPY + CADJPY are all JPY-SHORT / their-base-LONG; EURUSD +
# GBPUSD + AUDUSD all share the USD leg). One bet per cluster per pass.
CURRENCY_CLUSTERS: Dict[str, set] = {
    "JPY": {
        "AUDJPY",
        "CADJPY",
        "CHFJPY",
        "EURJPY",
        "GBPJPY",
        "HKDJPY",
        "HUFJPY",
        "MXNJPY",
        "NOKJPY",
        "NZDJPY",
        "PLNJPY",
        "SEKJPY",
        "SGDJPY",
        "TRYJPY",
        "USDJPY",
        "ZARJPY",
    },
    "USD": {
        "AUDUSD",
        "EURUSD",
        "GBPUSD",
        "NZDUSD",
        "USDCAD",
        "USDCHF",
        "USDJPY",
        "XAUUSD",
    },
    "CHF": {
        "AUDCHF",
        "CADCHF",
        "CHFJPY",
        "EURCHF",
        "GBPCHF",
        "NZDCHF",
        "USDCHF",
    },
    "EUR": {
        "EURAUD",
        "EURCAD",
        "EURCHF",
        "EURGBP",
        "EURJPY",
        "EURNZD",
        "EURUSD",
    },
    "GBP": {
        "GBPAUD",
        "GBPCAD",
        "GBPCHF",
        "GBPJPY",
        "GBPNZD",
        "GBPUSD",
    },
    "AUD": {"AUDCAD", "AUDCHF", "AUDJPY", "AUDNZD", "AUDUSD", "EURAUD", "GBPAUD"},
    "CAD": {"AUDCAD", "CADCHF", "CADJPY", "EURCAD", "GBPCAD", "NZDCAD", "USDCAD"},
    "NZD": {"AUDNZD", "EURNZD", "GBPNZD", "NZDCAD", "NZDCHF", "NZDJPY", "NZDUSD"},
}

DEFAULT_MAX_CONCURRENT = 6
DEFAULT_MAX_PER_CLUSTER = 1
DEFAULT_MAX_HEAT_PCT = 0.04  # 4% of equity in total risk

# Per-trade risk as a fraction of equity when the alert carries no explicit
# risk figure (the live alert blocks embed a sized risk plan when present).
DEFAULT_RISK_FRAC = 0.01


def _alert_ev(alert: Dict) -> Optional[float]:
    """The alert's decision EV from its book verdict.

    Prefers the taken side's allocation-aware target-level EV (the
    Stage-10 decision variable), falling back to the verdict's ranking EV.
    None when the alert carries no book (older/degraded path).
    """
    book = (alert.get("report") or {}).get("opportunity_book") or {}
    vd = book.get("verdict") or {}
    direction = vd.get("direction")
    if direction not in ("long", "short"):
        return None
    opp = book.get(direction) or {}
    ev_alloc = opp.get("expected_r_alloc")
    if ev_alloc is not None:
        return float(ev_alloc)
    return vd.get("expected_r")


def _alert_risk_frac(alert: Dict) -> float:
    """Per-trade risk as a fraction of equity (from the sized plan when
    present, else the documented default)."""
    report = alert.get("report") or {}
    direction = alert.get("direction") or "long"
    key = "short_risk" if direction == "short" else "risk"
    plan = report.get(key) or {}
    sizes = plan.get("sizes") or []
    frac = next(
        (
            r.get("risk_pct_equity", 0) / 100.0
            for r in sizes
            if r.get("method") == "fractional"
        ),
        None,
    )
    if frac is not None and frac > 0:
        return float(frac)
    return DEFAULT_RISK_FRAC


def _clusters_of(symbol: str) -> List[str]:
    sym = str(symbol).upper()
    return [c for c, members in CURRENCY_CLUSTERS.items() if sym in members]


def select_portfolio_orders(
    alerts: List[Dict],
    *,
    max_concurrent: int = DEFAULT_MAX_CONCURRENT,
    max_per_cluster: int = DEFAULT_MAX_PER_CLUSTER,
    max_heat_pct: float = DEFAULT_MAX_HEAT_PCT,
) -> Dict:
    """Apply portfolio constraints to the merged alerts.

    Returns ``{kept, rejected, summary}`` where ``kept`` is the
    portfolio-capped order list (sorted by EV descending) and ``rejected``
    lists ``{symbol, direction, ev, reason}`` for every dropped order.
    ``summary`` carries the final cluster/heat/concurrent state.

    Greedy, evidence-first: candidates are considered in EV order; the
    first time a cluster hits its cap, subsequent same-cluster candidates
    are rejected (the highest-EV representative of each shared leg wins).
    """
    if not alerts:
        return {"kept": [], "rejected": [], "summary": _empty_summary()}

    scored = []
    for a in alerts:
        ev = _alert_ev(a)
        if ev is None:
            ev = float("-inf")  # no book -> lowest priority (degraded path)
        scored.append((ev, a))
    scored.sort(key=lambda kv: kv[0], reverse=True)

    kept: List[dict] = []
    rejected: List[dict] = []
    cluster_counts: Dict[str, int] = {}
    heat = 0.0

    for ev, alert in scored:
        symbol = str(alert.get("symbol", "?"))
        direction = alert.get("direction") or "long"
        sym_clusters = _clusters_of(symbol)
        risk_frac = _alert_risk_frac(alert)

        # 1. Max concurrent (hard bound).
        if len(kept) >= max_concurrent:
            rejected.append(
                {
                    "symbol": symbol,
                    "direction": direction,
                    "ev": ev if ev != float("-inf") else None,
                    "reason": f"max concurrent positions ({max_concurrent}) reached",
                }
            )
            continue
        # 2. One-per-cluster cap: an additional order sharing a currency
        # leg with an already-kept order is the SAME macro bet re-expressed.
        blocked_cluster = next(
            (c for c in sym_clusters if cluster_counts.get(c, 0) >= max_per_cluster),
            None,
        )
        if blocked_cluster is not None:
            rejected.append(
                {
                    "symbol": symbol,
                    "direction": direction,
                    "ev": ev if ev != float("-inf") else None,
                    "reason": (
                        f"currency-cluster cap ({max_per_cluster} per {blocked_cluster} "
                        f"leg) - same shared leg as a higher-EV order"
                    ),
                }
            )
            continue
        # 3. Portfolio heat: total risk / equity under the bound.
        if heat + risk_frac > max_heat_pct:
            rejected.append(
                {
                    "symbol": symbol,
                    "direction": direction,
                    "ev": ev if ev != float("-inf") else None,
                    "reason": (
                        f"portfolio heat {heat + risk_frac:.2%} would exceed "
                        f"{max_heat_pct:.0%} of equity"
                    ),
                }
            )
            continue

        kept.append(alert)
        heat += risk_frac
        for c in sym_clusters:
            cluster_counts[c] = cluster_counts.get(c, 0) + 1

    summary = {
        "n_candidates": len(alerts),
        "n_kept": len(kept),
        "n_rejected": len(rejected),
        "heat_pct": round(heat * 100, 2),
        "heat_limit_pct": round(max_heat_pct * 100, 2),
        "cluster_counts": {c: n for c, n in sorted(cluster_counts.items())},
        "max_concurrent": max_concurrent,
    }
    return {"kept": kept, "rejected": rejected, "summary": summary}


def _empty_summary() -> Dict:
    return {
        "n_candidates": 0,
        "n_kept": 0,
        "n_rejected": 0,
        "heat_pct": 0.0,
        "heat_limit_pct": DEFAULT_MAX_HEAT_PCT * 100,
        "cluster_counts": {},
        "max_concurrent": DEFAULT_MAX_CONCURRENT,
    }


def format_portfolio_summary(result: Dict) -> str:
    """One-line portfolio selection summary for the briefing footer."""
    s = result.get("summary") or {}
    rej = result.get("rejected") or []
    if not rej:
        return ""
    return (
        f"portfolio selection: {s.get('n_kept', 0)}/{s.get('n_candidates', 0)} orders "
        f"kept · heat {s.get('heat_pct', 0):.1f}%/{s.get('heat_limit_pct', 0):.0f}% · "
        f"clusters {len(s.get('cluster_counts') or {})} · "
        f"{len(rej)} rejected (cluster/concurrent/heat caps)"
    )
