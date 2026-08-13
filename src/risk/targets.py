"""
NexusQuant - Multi-Target Trade Construction (institutional spec #11)

The live engine currently targets the nearest resistance, which is why
confirmed dips frequently carry a single-target R:R of only 0.6-1.0.
This module builds a proper **TP1 / TP2 / TP3 ladder** so a sensible
minimum reward-to-risk (the spec asks for 2.5:1) is actually reachable:

* TP1 - nearest confluence resistance above entry (or +1R fallback)
* TP2 - the next confluence level / swing high (or +2R fallback)
* TP3 - a 2.618-fib extension or swing extreme (or +3R fallback)

Every target reports its own R:R. ``best_rr()`` returns the first target
that clears a required floor, so "min R:R 2.5" becomes a *scaling-out*
instruction (e.g. TP1 + move stop to breakeven, hold remainder for TP3)
rather than an impossible single-target demand.

Usage (library):
    from src.risk.targets import build_target_ladder, best_rr
    ladder = build_target_ladder(entry, stop, close, levels, atr)
    tp = best_rr(ladder, min_rr=2.5)
"""

from __future__ import annotations

from typing import Dict, List, Optional

MIN_RR = 2.5  # institutional default floor (spec #11)


def _risk(entry: float, stop: float) -> Optional[float]:
    if entry is None or stop is None:
        return None
    r = entry - stop
    if not r or r <= 0:
        return None
    return float(r)


def _cluster_prices_above(levels: Optional[Dict], above: float) -> List[Dict]:
    """Confluence clusters strictly above ``above``, ascending, with tags."""
    if not levels:
        return []
    clusters = levels.get("clusters") or []
    out = []
    for c in clusters:
        price = c.get("price")
        if price is None or price <= above:
            continue
        out.append(
            {
                "price": float(price),
                "score": c.get("score", 0),
                "tags": c.get("tags", []),
            }
        )
    out.sort(key=lambda c: c["price"])
    return out


def build_target_ladder(
    entry: float,
    stop: float,
    close: Optional[float] = None,
    levels: Optional[Dict] = None,
    atr: Optional[float] = None,
    max_targets: int = 3,
) -> Dict:
    """
    Build a TP1/TP2/TP3 ladder above ``entry``.

    Returns::

        {
          "targets": [{"target": "TP1", "price": .., "rr": 1.0,
                       "source": "fib_up_0.618"}, ...],
          "best_rr": 3.0,          # best achievable R:R in the ladder
          "min_rr_tp": "TP3",      # first target >= MIN_RR (or None)
        }
    """
    risk = _risk(entry, stop)
    if risk is None:
        return {"targets": [], "best_rr": 0.0, "min_rr_tp": None}

    candidates = _cluster_prices_above(levels, entry)
    # add a swing-high candidate from the last completed up-leg
    if levels:
        leg = levels.get("last_up_leg")
        if leg and leg[1] and leg[1] > entry:
            candidates.append(
                {"price": float(leg[1]), "score": 1, "tags": ["swing_high"]}
            )
    candidates.sort(key=lambda c: c["price"])

    targets: List[Dict] = []
    used = set()

    def add(target: str, price: float, source: str) -> None:
        price = round(float(price), 6)
        if price <= entry or price in used:
            return
        used.add(price)
        targets.append(
            {
                "target": target,
                "price": price,
                "rr": round((price - entry) / risk, 2),
                "source": source,
            }
        )

    # TP1 - nearest confluence (or +1R fallback)
    if candidates:
        add("TP1", candidates[0]["price"], ",".join(candidates[0]["tags"][:3]))
    if not targets:
        add("TP1", entry + 1.0 * risk, "1R fallback")

    # TP2 - next distinct confluence / swing high (or +2R fallback)
    for c in candidates[1:]:
        add("TP2", c["price"], ",".join(c["tags"][:3]))
        if any(t["target"] == "TP2" for t in targets):
            break
    if not any(t["target"] == "TP2" for t in targets):
        add("TP2", entry + 2.0 * risk, "2R fallback")

    # TP3 - a 2.618 extension if present, else +3R
    ext = next((c for c in candidates if any("2.618" in t for t in c["tags"])), None)
    if ext:
        add("TP3", ext["price"], ",".join(ext["tags"][:3]))
    if not any(t["target"] == "TP3" for t in targets):
        add("TP3", entry + 3.0 * risk, "3R fallback")

    # fill any gaps if a target was dropped (e.g. TP2 fell below TP1)
    order = ["TP1", "TP2", "TP3"]
    for i, label in enumerate(order):
        if label not in {t["target"] for t in targets}:
            if i == 0:
                add("TP1", entry + 1.0 * risk, "1R fallback")
            elif i == 1:
                add("TP2", entry + 2.0 * risk, "2R fallback")
            else:
                add("TP3", entry + 3.0 * risk, "3R fallback")

    targets = sorted(targets, key=lambda t: t["price"])
    # If a fallback collided with an already-used price and was dropped, the
    # ladder relabels by price order so TP1/TP2/TP3 stay monotonic (a
    # cluster-sourced level may thus move up a label - intentional).
    if len(targets) > 1:
        for i, t in enumerate(targets[:max_targets]):
            t["target"] = order[i]
    targets = targets[:max_targets]

    best_rr = max((t["rr"] for t in targets), default=0.0)
    min_rr_tp = next((t["target"] for t in targets if t["rr"] >= MIN_RR), None)
    return {
        "targets": targets,
        "best_rr": round(best_rr, 2),
        "min_rr_tp": min_rr_tp,
        "min_rr": MIN_RR,
    }


def build_short_target_ladder(
    entry: float,
    stop: float,
    close: Optional[float] = None,
    levels: Optional[Dict] = None,
    atr: Optional[float] = None,
    max_targets: int = 3,
) -> Dict:
    """
    Short-side mirror of ``build_target_ladder``: TP1/TP2/TP3 BELOW entry
    (risk = stop - entry, above the short). Levels come from confluence
    clusters below the entry plus the last down-leg swing low; falls back
    to -1R / -2R / -3R.
    """
    if entry is None or stop is None or stop <= entry:
        return {"targets": [], "best_rr": 0.0, "min_rr_tp": None}
    risk = float(stop - entry)

    def _below(above: float) -> List[Dict]:
        out = []
        for c in (levels or {}).get("clusters") or []:
            price = c.get("price")
            if price is None or price >= above:
                continue
            out.append(
                {
                    "price": float(price),
                    "score": c.get("score", 0),
                    "tags": c.get("tags", []),
                }
            )
        if levels:
            leg = levels.get("last_down_leg")
            if leg and leg[0] and leg[0] < above:
                out.append({"price": float(leg[0]), "score": 1, "tags": ["swing_low"]})
        out.sort(key=lambda c: -c["price"])
        return out

    candidates = _below(entry)
    targets: List[Dict] = []
    used = set()

    def add(target: str, price: float, source: str) -> None:
        price = round(float(price), 6)
        if price >= entry or price in used:
            return
        used.add(price)
        targets.append(
            {
                "target": target,
                "price": price,
                "rr": round((entry - price) / risk, 2),
                "source": source,
            }
        )

    if candidates:
        add("TP1", candidates[0]["price"], ",".join(candidates[0]["tags"][:3]))
    if not targets:
        add("TP1", entry - 1.0 * risk, "1R fallback")

    for c in candidates[1:]:
        add("TP2", c["price"], ",".join(c["tags"][:3]))
        if any(t["target"] == "TP2" for t in targets):
            break
    if not any(t["target"] == "TP2" for t in targets):
        add("TP2", entry - 2.0 * risk, "2R fallback")

    ext = next((c for c in candidates if any("1.618" in t for t in c["tags"])), None)
    if ext:
        add("TP3", ext["price"], ",".join(ext["tags"][:3]))
    if not any(t["target"] == "TP3" for t in targets):
        add("TP3", entry - 3.0 * risk, "3R fallback")

    order = ["TP1", "TP2", "TP3"]
    for i, label in enumerate(order):
        if label not in {t["target"] for t in targets}:
            add(label, entry - float(i + 1) * risk, f"{i + 1}R fallback")

    targets = sorted(targets, key=lambda t: -t["price"])
    if len(targets) > 1:
        for i, t in enumerate(targets[:max_targets]):
            t["target"] = order[i]
    targets = targets[:max_targets]

    best_rr = max((t["rr"] for t in targets), default=0.0)
    min_rr_tp = next((t["target"] for t in targets if t["rr"] >= MIN_RR), None)
    return {
        "targets": targets,
        "best_rr": round(best_rr, 2),
        "min_rr_tp": min_rr_tp,
        "min_rr": MIN_RR,
    }


def best_rr(ladder: Dict, min_rr: float = MIN_RR) -> Optional[Dict]:
    """First target in the ladder with R:R >= ``min_rr`` (scaling-out tip)."""
    for t in ladder.get("targets", []):
        if t["rr"] >= min_rr:
            return t
    return None


if __name__ == "__main__":
    print("NexusQuant Multi-Target module ready.")
