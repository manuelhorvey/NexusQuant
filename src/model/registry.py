"""
NexusQuant - Model governance registry (spec audit #38).

A minimal, append-only JSON ledger of trained models so any deployed
probability can be traced back to *what* was trained, *when*, on *which
data*, and with *what* out-of-sample quality. This is the audit trail
behind "which model produced this 62% probability?":

    models/registry.json
    [ { "model": "models/dip_lgbm.joblib",
        "ts": "2026-08-12T09:00:00Z",
        "auc_oos": 0.542, "n_samples": 21800,
        "symbols": 13, "n_features": 38,
        "params": {...}, "note": "" }, ... ]

It is deliberately append-only with a size cap (keep the last
``MAX_ENTRIES`` records per path) - old records document history, they
are never rewritten. Writing is atomic (temp file + rename) and every
write is wrapped so a registry failure can never break a model save.
"""

from __future__ import annotations

import json
import os
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional

REGISTRY_PATH = "models/registry.json"
MAX_ENTRIES = 200


def _ts() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def all_entries(registry_path: str = REGISTRY_PATH) -> List[Dict]:
    """All registry records (oldest first). [] when absent/corrupt."""
    p = Path(registry_path)
    if not p.exists():
        return []
    try:
        data = json.loads(p.read_text())
        return data if isinstance(data, list) else []
    except Exception:
        return []


def entries_for(model_path: str, registry_path: str = REGISTRY_PATH) -> List[Dict]:
    """Records for one model path (oldest first)."""
    return [e for e in all_entries(registry_path) if e.get("model") == str(model_path)]


def latest(model_path: str, registry_path: str = REGISTRY_PATH) -> Optional[Dict]:
    """The most recent registry record for a model path, or None."""
    recs = entries_for(model_path, registry_path)
    return recs[-1] if recs else None


def record(
    model_path: str,
    meta: Optional[Dict] = None,
    note: str = "",
    registry_path: str = REGISTRY_PATH,
) -> Optional[Dict]:
    """Append one registry record (atomic write, never raises).

    ``meta`` is the model's own training metadata (auc_oos, n_samples,
    symbols, best_params, ...) - it is copied verbatim so the registry
    stays a faithful mirror of what was actually saved.
    """
    try:
        meta = meta or {}
        entry = {
            "model": str(model_path),
            "ts": _ts(),
            "auc_oos": meta.get("auc_oos"),
            "n_samples": meta.get("n_samples"),
            "symbols": meta.get("symbols"),
            "n_features": meta.get("n_features"),
            "params": meta.get("best_params"),
            "note": note,
        }
        entries = all_entries(registry_path)
        entries.append(entry)
        # size cap: keep the newest MAX_ENTRIES globally (the ledger stays
        # bounded on disk; history per path survives until the cap bites).
        entries = entries[-MAX_ENTRIES:]
        p = Path(registry_path)
        p.parent.mkdir(parents=True, exist_ok=True)
        fd, tmp = tempfile.mkstemp(dir=str(p.parent), suffix=".tmp")
        try:
            with os.fdopen(fd, "w") as fh:
                json.dump(entries, fh, indent=2)
            os.replace(tmp, p)  # atomic: no partial/corrupt ledger
        finally:
            if os.path.exists(tmp):
                os.unlink(tmp)
        return entry
    except Exception:
        return None


if __name__ == "__main__":
    print("NexusQuant Model Registry module ready.")
