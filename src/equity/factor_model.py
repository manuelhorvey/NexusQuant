"""
NexusQuant - Institutional Factor Model (institutional spec #8).

Cross-sectional Value / Quality / Momentum factor scores. The plan
proposed this module with a DataFrame-based ``compute_factor_scores`` API;
the per-symbol scorer lives in ``src.equity.fundamentals`` (band-mapped,
works with a single row) and is re-exported here so both APIs are real.

Factor definitions (0-100, higher = better):

* **Value**    - cheaper is better: P/E, EV/EBITDA, P/B, FCF yield.
* **Quality**  - strong profitability, disciplined balance sheet: ROE,
  ROIC, Debt/Equity (lower is better), Current ratio.
* **Momentum** - earnings surprise, analyst revisions, 12M price momentum.

``compute_factor_scores`` takes a fundamentals table (one row per symbol)
and returns cross-sectional z-scores winsorized at the 1/99 percentile,
equal-weighted within each category, composite = 33/33/34.
"""

from __future__ import annotations

from typing import Dict, Optional

import numpy as np
import pandas as pd

from src.equity.fundamentals import (  # noqa: F401  (re-exported API)
    factor_scores,
    load_fundamentals,
    momentum_score,
    quality_score,
    value_score,
)

# category -> (field, direction) where direction +1 = higher is better,
# -1 = lower is better.
FUNDAMENTAL_FIELDS: Dict[str, list] = {
    "value": [("pe_ratio", -1), ("ev_ebitda", -1), ("pb_ratio", -1), ("fcf_yield", +1)],
    "quality": [("roe", +1), ("roic", +1), ("debt_equity", -1), ("current_ratio", +1)],
    "momentum": [("earnings_surprise", +1), ("analyst_revisions", +1), ("mom_12m", +1)],
}

# Source-key aliases so the module consumes BOTH the plan's canonical
# schema AND the keys emitted by ``src.equity.fundamentals.load_fundamentals``
# / ``src.equity.data_provider`` (pe, pb, roe_pct, debt_to_equity,
# earnings_surprise_pct, ...).
FIELD_ALIASES: Dict[str, str] = {
    "pe_ratio": "pe",
    "pb_ratio": "pb",
    "roe": "roe_pct",
    "debt_equity": "debt_to_equity",
    "earnings_surprise": "earnings_surprise_pct",
}

COMPOSITE_WEIGHTS = {"value": 0.33, "quality": 0.33, "momentum": 0.34}


def _zscore(series: pd.Series) -> pd.Series:
    """Winsorized cross-sectional z-score (1/99 percentile clip)."""
    s = series.astype(float)
    lo, hi = s.quantile(0.01), s.quantile(0.99)
    s = s.clip(lo, hi)
    sd = s.std(ddof=0)
    return (s - s.mean()) / sd if sd and sd > 0 else s * 0.0


def compute_factor_scores(fundamentals: pd.DataFrame) -> pd.DataFrame:
    """
    Cross-sectional factor scores from a fundamentals table.

    Input columns: ``symbol`` plus any of the ``FUNDAMENTAL_FIELDS``
    (all optional; missing fields are simply skipped). Output has one row
    per symbol with ``value_score / quality_score / momentum_score``
    (0-100, ``None`` when no field in the category was present) and a
    ``composite`` (0-100, renormalised over the available categories).

    A category z-score maps to 0-100 via ``50 + 25 * z`` (clipped), so a
    median symbol reads 50 and a strong one reads 75-90 - comparable to
    the band-mapped per-symbol scorer.
    """
    if (
        fundamentals is None
        or fundamentals.empty
        or "symbol" not in fundamentals.columns
    ):
        return pd.DataFrame(
            columns=[
                "symbol",
                "value_score",
                "quality_score",
                "momentum_score",
                "composite",
            ]
        )

    df = fundamentals.copy()
    # Normalize provider/CSV keys to the canonical schema (alias -> canon).
    for canon, alias in FIELD_ALIASES.items():
        if canon not in df.columns and alias in df.columns:
            df[canon] = df[alias]
    rows = []
    for _, rec in df.iterrows():
        symbol = rec.get("symbol")
        out: Dict = {"symbol": symbol}
        avail = []
        for cat, fields in FUNDAMENTAL_FIELDS.items():
            parts = []
            for field, direction in fields:
                if field in df.columns:
                    col = df[field].astype(float)
                    if col.notna().sum() >= 2:
                        z = _zscore(col).loc[rec.name]
                        if not np.isnan(z):
                            parts.append(float(direction) * float(z))
            if parts:
                zcat = float(np.mean(parts))
                out[f"{cat}_score"] = round(
                    float(np.clip(50.0 + 25.0 * zcat, 0.0, 100.0)), 1
                )
                avail.append((cat, out[f"{cat}_score"], COMPOSITE_WEIGHTS[cat]))
            else:
                out[f"{cat}_score"] = None
        if avail:
            wsum = sum(w for _, _, w in avail)
            out["composite"] = round(sum(s * w for _, s, w in avail) / wsum, 1)
        else:
            out["composite"] = None
        rows.append(out)

    return pd.DataFrame(
        rows,
        columns=[
            "symbol",
            "value_score",
            "quality_score",
            "momentum_score",
            "composite",
        ],
    )


def get_factor_interpretation(score: Optional[float]) -> str:
    """Human label for a 0-100 factor score (None -> 'N/A')."""
    if score is None:
        return "N/A"
    if score >= 70:
        return "Strong"
    if score >= 50:
        return "Neutral"
    return "Weak"


if __name__ == "__main__":
    demo = pd.DataFrame(
        {
            "symbol": ["ALPHA", "BETA", "GAMMA"],
            "pe_ratio": [8.0, 18.0, 42.0],
            "ev_ebitda": [5.0, 11.0, 26.0],
            "pb_ratio": [0.9, 2.5, 9.0],
            "roe": [30.0, 15.0, 4.0],
            "debt_equity": [0.2, 0.9, 2.2],
            "mom_12m": [0.35, 0.05, -0.30],
        }
    )
    print(compute_factor_scores(demo).to_string(index=False))
