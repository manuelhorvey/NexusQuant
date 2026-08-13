"""
NexusQuant - Backfill regroup tool.

The bulk backfill (`python -m src.data.mt5 --backfill`) stages the whole MT5
universe under ``data/raw/mt5/{D1,H4,H1}`` - organised by timeframe, mixing
equities, FX, crypto, metals, indices and commodities in one place.

This module regroups those files into the asset-class group folders the rest
of the system expects (``full_fx``, ``candidates``, ``equity_universe``,
``crosses``, ``equity`` + new ``crypto``, ``metals``, ``commodities``,
``indices``). Each group folder holds ``*_{D1,H4,H1}.parquet`` files flat -
which is exactly how ``discover_symbols`` / the dashboard read groups, so no
downstream code changes are needed.

Duplicate policy: when a symbol/timeframe already exists in the destination
group, the file with the **longest history** wins (the other copy is
dropped) - a fresh MT5 pull never silently truncates a longer Yahoo / older
dataset.

Usage:
    python -m src.data.regroup --dry-run          # show the plan, move nothing
    python -m src.data.regroup                     # regroup + prune empty sources
    python -m src.data.regroup --source data/raw/mt5 data/raw/h4 data/raw/h1
"""

from __future__ import annotations

import argparse
import shutil
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import pandas as pd

# Currency codes used to recognise FX pairs (exotic codes included).
CURRENCY_CODES = {
    "AED",
    "AMD",
    "AOA",
    "ARS",
    "AUD",
    "AZN",
    "BDT",
    "BGN",
    "BHD",
    "BND",
    "BRL",
    "CAD",
    "CHF",
    "CLP",
    "CNH",
    "CNY",
    "COP",
    "CZK",
    "DKK",
    "DZD",
    "EGP",
    "EUR",
    "GBP",
    "GEL",
    "GHS",
    "GTQ",
    "HKD",
    "HNL",
    "HRK",
    "HUF",
    "IDR",
    "ILS",
    "INR",
    "IQD",
    "ISK",
    "JMD",
    "JOD",
    "JPY",
    "KES",
    "KGS",
    "KRW",
    "KWD",
    "KZT",
    "LBP",
    "LKR",
    "MAD",
    "MXN",
    "MYR",
    "NGN",
    "NOK",
    "NPR",
    "NZD",
    "OMR",
    "PEN",
    "PHP",
    "PKR",
    "PLN",
    "QAR",
    "RON",
    "RUB",
    "RUR",
    "SAR",
    "SEK",
    "SGD",
    "SYP",
    "THB",
    "TJS",
    "TMT",
    "TND",
    "TRY",
    "TTD",
    "TWD",
    "UAH",
    "UGX",
    "USD",
    "UZS",
    "VND",
    "VUV",
    "XCD",
    "XOF",
    "ZAR",
    "ZMW",
}

# Crypto bases (matches *USD pairs like ADAUSD and crypto crosses like BTCAUD).
CRYPTO_BASES = {
    "1INCH",
    "AAVE",
    "ADA",
    "BAT",
    "BCH",
    "BNB",
    "BTC",
    "BWP",
    "CAKE",
    "COMP",
    "DOGE",
    "DOT",
    "ETH",
    "FIL",
    "HBAR",
    "HT",
    "IOST",
    "LINK",
    "LTC",
    "MBT",
    "SOL",
    "TET",
    "UNI",
    "XEM",
    "XLM",
    "XRP",
    "XTZ",
}

METAL_BASES = {"XAU", "XAG", "XPT", "XPD"}

# Base names that are commodity contracts (X-prefixed metals/energy/base metals).
COMMODITY_BASES = {
    "XAL",
    "XBR",
    "XCK",
    "XCO",
    "XCU",
    "XNG",
    "XNI",
    "XPB",
    "XSN",
    "XTI",
    "XZN",
    "USOIL",
    "WTICO",
    "BRENT",
}

INDEX_SYMBOLS = {
    "AUS200",
    "CHN50",
    "ESP35",
    "FRA40",
    "GER30",
    "HKG50",
    "IN50",
    "IND50",
    "DXY",
    "JPN225",
    "SPA35",
    "SWI20",
    "UK100",
    "US30",
    "US500",
    "USTEC",
    # MT5 lot-multiplier variants
    "US30_x10",
    "US500_x100",
    "USTEC_x100",
}

# Existing curated groups take precedence (their folder contents define them).
CURATED_GROUPS = ["full_fx", "candidates", "crosses", "equity", "equity_universe"]


def classify_symbol(symbol: str, equity_membership: set) -> str:
    """Map a symbol to its asset-class group folder name."""
    s = symbol.upper()
    if not s:
        raise ValueError("empty symbol")

    # 1) Curated equity universe (SP500 constituents).
    if s in equity_membership:
        return "equity_universe"
    # 2) Curated group membership is resolved by the caller (folder contents).
    # 3) Metals.
    if s[:3] in METAL_BASES or s.startswith("XAU") or s.startswith("XAG"):
        return "metals"
    # 4) Commodities (explicit bases only - the generic X*USD pattern is
    #    ambiguous and would swallow crypto pairs like XRPUSD / XTZUSD).
    if s[:3] in COMMODITY_BASES or s in COMMODITY_BASES:
        return "commodities"
    # 5) Indices (explicit names or names containing digits).
    if s in INDEX_SYMBOLS or (any(ch.isdigit() for ch in s) and not s.endswith("USD")):
        return "indices"
    # 6) Crypto: base in the crypto set, or *USD/*USDT with a non-currency base.
    if s[:4] in CRYPTO_BASES or s[:3] in CRYPTO_BASES:
        return "crypto"
    for suffix in ("USDT", "USD"):
        if s.endswith(suffix) and s[: -len(suffix)] not in CURRENCY_CODES:
            return "crypto"
    # 7) FX pairs: 6-char currency/currency.
    if len(s) == 6 and s[:3] in CURRENCY_CODES and s[3:] in CURRENCY_CODES:
        return "full_fx"
    # 8) Everything else is an equity / ETF / unknown CFD.
    return "equity"


def group_for_symbol(
    symbol: str,
    curated: Dict[str, set],
    equity_membership: set,
) -> str:
    """Curated folder membership wins; otherwise classify."""
    s = symbol.upper()
    for group in CURATED_GROUPS:
        if group in curated and s in curated[group]:
            return group
    return classify_symbol(s, equity_membership)


def _row_count(path: Path) -> Optional[int]:
    """Bar count, or None when the file cannot be read."""
    try:
        return len(pd.read_parquet(path, columns=["date"]))
    except Exception:
        return None


def plan_merge(
    source_dirs: Sequence[str],
    data_dir: str = "data/raw",
    curated: Optional[Dict[str, set]] = None,
    equity_membership: Optional[set] = None,
) -> Tuple[List[Tuple[str, str, str]], int, int]:
    """
    Compute the move plan: list of ``(src_path, dst_path, action, group)``
    where action is ``move`` | ``skip-shorter`` | ``keep-both``. Returns
    ``(plan, n_move, n_skip)``.
    """
    if curated is None:
        curated = {
            g: {p.stem.split("_")[0] for p in (Path(data_dir) / g).glob("*.parquet")}
            for g in CURATED_GROUPS
            if (Path(data_dir) / g).is_dir()
        }
    if equity_membership is None:
        mem_path = Path(data_dir) / "equity_universe" / "_membership.csv"
        equity_membership = set()
        if mem_path.exists():
            try:
                equity_membership = set(pd.read_csv(mem_path)["symbol"].str.upper())
            except Exception:
                pass

    plan: List[Tuple[str, str, str, str]] = []
    n_skip = 0
    for src_dir in source_dirs:
        base = Path(src_dir)
        for src in sorted(base.glob("*.parquet")):
            stem = src.stem  # e.g. EURUSD_H4
            tf = stem.rsplit("_", 1)[-1].upper()
            symbol = stem[: -len(tf) - 1]
            group = group_for_symbol(symbol, curated, equity_membership)
            dst = Path(data_dir) / group / src.name
            if dst.exists():
                n_src = _row_count(src)
                n_dst = _row_count(dst)
                if n_src is None or n_dst is None:
                    # Unreadable on either side: never delete the source on
                    # an unconfirmed "shorter" - keep both for manual review.
                    plan.append((str(src), str(dst), "keep-both", group))
                elif n_src > n_dst:
                    plan.append((str(src), str(dst), "move", group))
                else:
                    plan.append((str(src), "", "skip-shorter", group))
                    n_skip += 1
            else:
                plan.append((str(src), str(dst), "move", group))
    return plan, sum(1 for p in plan if p[2] == "move"), n_skip


def archive_short(
    data_dir: str = "data/raw",
    archive_dir: str = "data/archive",
    min_bars: int = 250,
) -> List[Tuple[str, str, int]]:
    """
    Move instruments with fewer than ``min_bars`` bars to ``archive_dir``
    (out of discovery) with a ``_manifest.txt``.

    The default threshold (250 bars) is the SMA-200 warm-up requirement -
    anything below it can never produce a signal in the regime/dip engines.
    Files are never deleted, just moved; re-backfilling restores them.
    """
    archive = Path(archive_dir)
    archive.mkdir(parents=True, exist_ok=True)
    moved: List[Tuple[str, str, int]] = []
    for d in sorted(Path(data_dir).iterdir()):
        if not d.is_dir():
            continue
        for f in sorted(d.glob("*.parquet")):
            n = _row_count(f)
            if n is not None and n < min_bars:
                dst = archive / f.name
                shutil.move(str(f), str(dst))
                moved.append((str(dst), f.name, n))
    if moved:
        with open(archive / "_manifest.txt", "w") as fh:
            fh.write(
                f"NexusQuant short-history quarantine (<{min_bars} bars - "
                f"cannot feed the SMA-200 pipeline).\n"
            )
            for _, name, n in sorted(moved, key=lambda m: m[1]):
                fh.write(f"  {name} ({n} bars)\n")
    return moved


def clean_legacy(data_dir: str = "data/raw") -> Tuple[List[Tuple], int, int]:
    """
    Consolidate legacy duplicates into full_fx (keep-longest policy):

    * top-level ``data/raw/{SYMBOL}_D1.parquet`` majors (historic default
      scan group, now duplicated by full_fx/), and
    * the redundant ``crosses/`` folder (a strict subset of full_fx/).

    Returns ``(plan, n_move, n_skip)`` from ``plan_merge`` so the caller can
    dry-run / execute it like any other regroup.
    """
    root = Path(data_dir)
    sources = [str(root)]
    if (root / "crosses").is_dir():
        sources.append(str(root / "crosses"))
    return plan_merge(sources, data_dir)


def execute_merge(plan: List[Tuple[str, str, str, str]], prune: bool = True) -> Dict:
    """Apply the plan. Returns per-group move counts."""
    per_group: Dict[str, int] = {}
    for src, dst, action, group in plan:
        if action == "skip-shorter":
            Path(src).unlink(missing_ok=True)
            continue
        if action == "keep-both":
            print(
                f"[regroup] kept {Path(src).name} in staging (unreadable "
                f"duplicate, review manually)"
            )
            continue
        dst_path = Path(dst)
        dst_path.parent.mkdir(parents=True, exist_ok=True)
        shutil.move(src, dst_path)
        per_group[group] = per_group.get(group, 0) + 1
    if prune:
        # Prune empty dirs up the chain (mt5/D1 -> mt5 -> ...) until a
        # non-empty ancestor is reached.
        for src_dir in sorted({Path(p[0]).parent for p in plan}):
            cur = src_dir
            while cur != cur.parent and cur.is_dir() and not any(cur.iterdir()):
                cur.rmdir()
                print(f"[regroup] removed empty dir {cur}")
                cur = cur.parent
    return per_group


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        description="Regroup the bulk MT5 backfill into asset-class folders",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--source",
        nargs="*",
        default=["data/raw/mt5", "data/raw/h4", "data/raw/h1"],
        help="source dirs to regroup (timeframe subdirs)",
    )
    parser.add_argument("--data-dir", default="data/raw")
    parser.add_argument(
        "--dry-run", action="store_true", help="print the plan without moving anything"
    )
    parser.add_argument(
        "--no-prune", action="store_true", help="keep empty source directories"
    )
    parser.add_argument(
        "--clean",
        action="store_true",
        help="consolidate legacy duplicates (top-level majors + "
        "crosses/) into full_fx and prune",
    )
    parser.add_argument(
        "--archive-short",
        nargs="?",
        type=int,
        const=250,
        default=None,
        metavar="MIN_BARS",
        help="move instruments with < MIN_BARS bars to "
        "data/archive/ (default 250 = SMA-200 warm-up)",
    )
    args = parser.parse_args(argv)

    if args.archive_short is not None:
        moved = archive_short(args.data_dir, min_bars=args.archive_short)
        if moved:
            print(
                f"Archived {len(moved)} short-history files (<"
                f"{args.archive_short} bars) -> data/archive/ "
                f"(manifest in _manifest.txt):"
            )
            for _, name, n in sorted(moved, key=lambda m: m[1]):
                print(f"  - {name} ({n} bars)")
        else:
            print(f"No files below {args.archive_short} bars to archive.")
        return 0

    if args.clean:
        plan, n_move, n_skip = clean_legacy(args.data_dir)
        n_keep = sum(1 for p in plan if p[2] == "keep-both")
        print("=" * 64)
        print(
            f"CLEAN LEGACY — {len(plan)} files · {n_move} move into "
            f"full_fx · {n_skip} skip (shorter dup) · {n_keep} keep-both"
        )
        print("=" * 64)
        if args.dry_run:
            dropped = sorted(Path(p[0]).name for p in plan if p[2] == "skip-shorter")
            if dropped:
                print("\nWould drop (shorter duplicates):")
                for name in dropped:
                    print(f"  - {name}")
            print("\n(dry run — nothing moved; re-run with --clean to apply)")
            return 0
        if plan:
            execute_merge(plan, prune=not args.no_prune)
            print("Done.")
        else:
            print("Nothing to do.")
        return 0

    # Expand dirs that contain timeframe subdirs (e.g. mt5/ -> D1, H4, H1)
    # and also keep the dir itself when it has stray flat parquet files;
    # flat source dirs (h4/, h1/) are used directly.
    source_dirs: List[str] = []
    for s in args.source:
        p = Path(s)
        if not p.is_dir():
            continue
        subs = [c for c in p.iterdir() if c.is_dir() and any(c.glob("*.parquet"))]
        if subs:
            source_dirs += [str(c) for c in subs]
            if any(p.glob("*.parquet")):
                source_dirs.append(s)  # stray files directly in the root
        else:
            source_dirs.append(s)

    plan, n_move, n_skip = plan_merge(source_dirs, args.data_dir)
    n_keep = sum(1 for p in plan if p[2] == "keep-both")

    # Summarise per group: moves and (when in dry-run) the rest.
    from collections import Counter

    moves_by_group = Counter(p[3] for p in plan if p[2] == "move")
    print("=" * 64)
    print(
        f"REGROUP PLAN — {len(plan)} files · {n_move} move · {n_skip} "
        f"skip (shorter dup) · {n_keep} keep-both (unreadable)"
    )
    print("=" * 64)
    for g in sorted(moves_by_group):
        print(f"  -> {g:<16} {moves_by_group[g]:>4} files")
    if args.dry_run:
        # Itemise the destructive actions so they can be audited first.
        dropped = sorted(Path(p[0]).name for p in plan if p[2] == "skip-shorter")
        if dropped:
            print("\nWould drop (shorter duplicates):")
            for name in dropped[:20]:
                print(f"  - {name}")
            if len(dropped) > 20:
                print(f"  ... and {len(dropped) - 20} more")
        print("\n(dry run — nothing moved; re-run without --dry-run to apply)")
        return 0
    if plan:
        print("\nApplying plan...")
        per_group = execute_merge(plan, prune=not args.no_prune)
        print(
            f"Done: moved {sum(per_group.values())} files; "
            f"{n_skip} shorter duplicates dropped; {n_keep} kept in "
            f"staging for review."
        )
    else:
        print("Nothing to do.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
