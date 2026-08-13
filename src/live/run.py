"""
NexusQuant - Live Signal & Alert CLI.

Run the full-stack signal pass and push high-conviction setups to your
alert channels (Discord webhook / console / file):

    python -m src.live.run                          # one pass, full_fx D1
    python -m src.live.run --group candidates --top N
    python -m src.live.run --symbols EURUSD,GBPUSD,XAUUSD
    python -m src.live.run --min-dip-score 6 --min-ml-prob 55
    python -m src.live.run --dry-run                # print, don't send
    python -m src.live.run --watch --interval-min 60  # loop for a scheduler
    python -m src.live.run --json                   # machine-readable

Discord: set ``NEXUS_DISCORD_WEBHOOK`` (or ``live.discord_webhook`` in
config/settings.yaml) and enable ``live.discord: true``. The console and
file channels are on by default, so it works out of the box with zero
configuration.

The pass is local-only (no MT5 fetches) and safe for cron.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from pathlib import Path
from typing import List, Optional

sys.path.append(str(Path(__file__).parent.parent.parent))

from src.live.alerts import build_channels, send_all
from src.live.signals import (
    DEFAULT_GROUP,
    DEFAULT_TIMEFRAME,
    live_signal_pass,
    live_short_pass,
    format_briefing,
)

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")


def _load_settings() -> dict:
    try:
        import yaml

        with open("config/settings.yaml") as fh:
            return yaml.safe_load(fh) or {}
    except Exception:
        return {}


def diagnostics_reports(
    group: Optional[str],
    timeframe: str,
    symbols: Optional[List[str]] = None,
    data_dir: str = "data/raw",
) -> int:
    """
    Directional diagnostics (campaign spec #24): per symbol, the full
    opportunity book - LONG/SHORT/FLAT verdict with EV, per-side setup /
    probability / EV / R:R, and every rejection reason. Answers "what are
    the opportunities here, and why was each one taken or rejected?".
    ``data_dir`` mirrors ``institutional_reports`` (default ``data/raw``).
    """
    from src.data.loader import clean_data, load_data
    from src.features.indicators import add_all_indicators
    from src.features.regime import detect_regime
    from src.analysis.report import generate_full_report
    from src.analysis.opportunity import format_opportunity_book
    from src.analysis.scanner import _data_path, discover_symbols

    if symbols is None:
        symbols = discover_symbols(data_dir, group, timeframe)
    if not symbols:
        print("No symbols found.", file=sys.stderr)
        return 0

    n = 0
    for sym in symbols:
        try:
            path = _data_path(sym, data_dir, group, timeframe)
            df = clean_data(load_data(path, symbol=sym))
            df = add_all_indicators(df)
            df = detect_regime(df)
            report = generate_full_report(
                df, symbol=sym, group=group, data_dir=data_dir, mtf=False
            )
            ob = report.get("opportunity_book")
            if ob:
                print(format_opportunity_book(ob))
            else:
                v = (report.get("plan") or {}).get("action", "NO-SETUP")
                print(f"{sym} — no opportunity book (plan: {v})")
            print("=" * 60)
            n += 1
        except Exception as exc:
            print(f"[{sym}] diagnostics failed: {exc}", file=sys.stderr)
    return n


def institutional_reports(
    group: Optional[str],
    timeframe: str,
    symbols: Optional[List[str]] = None,
    data_dir: str = "data/raw",
    use_hmm: bool = False,
) -> int:
    """
    Full institutional report (all 18 sections) per symbol - the plan's
    end-state ``--format institutional``. Local-only (like the live pass);
    symbols without local data are skipped with a warning. Returns the
    number of reports printed.
    """
    from src.data.loader import clean_data, load_data
    from src.features.indicators import add_all_indicators
    from src.features.regime import detect_regime
    from src.analysis.report import generate_full_report, print_report
    from src.analysis.scanner import _data_path, discover_symbols

    if symbols is None:
        symbols = discover_symbols(data_dir, group, timeframe)
    if not symbols:
        print("No symbols found.", file=sys.stderr)
        return 0

    n = 0
    for sym in symbols:
        try:
            path = _data_path(sym, data_dir, group, timeframe)
            df = clean_data(load_data(path, symbol=sym))
            df = add_all_indicators(df)
            df = detect_regime(df)
            report = generate_full_report(
                df,
                symbol=sym,
                group=group,
                data_dir=data_dir,
                mtf=True,
                use_hmm=use_hmm,
            )
            print_report(report)
            n += 1
        except Exception as exc:
            print(f"[{sym}] report failed: {exc}", file=sys.stderr)
    return n


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        description="NexusQuant live signal & alert runner",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--group", default=None, help="data group (default from settings/full_fx)"
    )
    parser.add_argument(
        "--timeframe", default=None, help="D1/H4/H1 (default derived from group)"
    )
    parser.add_argument(
        "--symbols", default=None, help="comma-separated watchlist override"
    )
    parser.add_argument(
        "--min-dip-score",
        type=int,
        default=None,
        help="minimum dip score (default from settings)",
    )
    parser.add_argument(
        "--min-ml-prob",
        type=float,
        default=None,
        help="only alert setups with ML prob >= this (pct)",
    )
    parser.add_argument(
        "--min-rr",
        type=float,
        default=None,
        help="drop setups with reward:risk below this",
    )
    parser.add_argument(
        "--mode",
        choices=["long", "short", "both"],
        default="both",
        help="which engine to run: Buy-the-Dip (long), Sell-the-Rally "
        "(short), or both (default; the system is dual-sided)",
    )
    parser.add_argument(
        "--format",
        choices=["briefing", "institutional", "plan", "diagnostics"],
        default="briefing",
        help="briefing = compact alert pass; institutional = "
        "full 18-section report per symbol; plan = one-action-per-symbol "
        "decision table (BUY/SELL-LIMIT · WAIT-* · NO-SETUP); "
        "diagnostics = per-symbol opportunity book (LONG/SHORT/FLAT + "
        "EV + rejection reasons)",
    )
    parser.add_argument(
        "--hmm",
        action="store_true",
        help="with --format institutional: fit the 4-state "
        "HMM and surface its label in the regime table",
    )
    parser.add_argument(
        "--min-rally-score",
        type=int,
        default=None,
        help="minimum rally score for short setups (default from settings)",
    )
    parser.add_argument(
        "--min-short-rr",
        type=float,
        default=None,
        help="minimum reward:risk for short setups",
    )
    parser.add_argument(
        "--no-macro-gate",
        action="store_true",
        help="do not require the macro gate to PASS",
    )
    parser.add_argument(
        "--equity",
        type=float,
        default=None,
        help="account equity for sizing (default from settings)",
    )
    parser.add_argument(
        "--risk",
        type=float,
        default=None,
        help="fraction of equity risked per trade (default from settings)",
    )
    parser.add_argument("--state-file", default="data/live/alerts.json")
    parser.add_argument(
        "--expiry",
        type=float,
        default=None,
        help="re-alert setups first sent more than N days "
        "ago (signal expiry; default from settings)",
    )
    parser.add_argument(
        "--dry-run", action="store_true", help="print the briefing without sending"
    )
    parser.add_argument("--json", action="store_true")
    parser.add_argument(
        "--watch", action="store_true", help="loop forever (for a scheduler/daemon)"
    )
    parser.add_argument(
        "--interval-min",
        type=float,
        default=60.0,
        help="minutes between passes in --watch mode",
    )
    parser.add_argument(
        "--top", type=int, default=None, help="cap the number of setups alerted"
    )
    args = parser.parse_args(argv)

    settings = _load_settings()
    live_cfg = settings.get("live", {})
    filters_cfg = live_cfg.get("filters", {}) or {}
    risk_cfg = live_cfg.get("risk", {}) or {}
    # spec #11's 2.5:1 minimum reward:risk - the default floor the live
    # filter enforces on the *ladder's* achievable R:R (scaling out), so a
    # ~0.9R nearest-resistance target never passes as a valid setup.
    project_risk_cfg = settings.get("risk", {}) or {}

    group = args.group or live_cfg.get("watchlist", {}).get("group") or DEFAULT_GROUP
    timeframe = (
        args.timeframe
        or live_cfg.get("watchlist", {}).get("timeframe")
        or DEFAULT_TIMEFRAME
    )
    if args.timeframe is None and timeframe == "D1" and group:
        timeframe = {"h1": "H1", "h4": "H4"}.get(group, "D1")

    symbols = None
    if args.symbols:
        symbols = [s.strip() for s in args.symbols.split(",") if s.strip()]

    # CLI flags override settings.yaml (settings are the defaults).
    equity = args.equity or risk_cfg.get("equity", 250_000.0)
    risk_pct = args.risk or risk_cfg.get("risk_per_trade", 0.01)
    min_dip = args.min_dip_score or filters_cfg.get("min_dip_score", 5)
    min_ml = args.min_ml_prob
    if min_ml is None and filters_cfg.get("min_ml_prob") is not None:
        min_ml = float(filters_cfg["min_ml_prob"])
    from src.risk.run import DEFAULT_MIN_RR

    min_rr = (
        args.min_rr
        or filters_cfg.get("min_rr")
        or project_risk_cfg.get("min_reward_risk")
        or DEFAULT_MIN_RR
    )
    min_rally = args.min_rally_score or filters_cfg.get("min_rally_score", 5)
    min_short_rr = args.min_short_rr or filters_cfg.get("min_short_rr")
    macro_gate = not args.no_macro_gate
    if filters_cfg.get("require_macro_pass") is False and not args.no_macro_gate:
        macro_gate = False
    signal_expiry = (
        args.expiry
        if args.expiry is not None
        else filters_cfg.get("signal_expiry_days", 30.0)
    )

    channels = build_channels(live_cfg)

    if args.format == "institutional":
        n = institutional_reports(group, timeframe, symbols, use_hmm=args.hmm)
        return 0 if n else 1

    if args.format == "diagnostics":
        n = diagnostics_reports(group, timeframe, symbols)
        return 0 if n else 1

    if args.format == "plan":
        from src.analysis.plan import print_plan_table

        return print_plan_table(symbols, group, timeframe)

    def _one_pass_long() -> dict:
        res = live_signal_pass(
            group=group,
            timeframe=timeframe,
            symbols=symbols,
            equity=equity,
            risk_pct=risk_pct,
            min_dip_score=min_dip,
            min_ml_prob=min_ml,
            min_rr=min_rr,
            require_macro_pass=macro_gate,
            state_file=args.state_file,
            signal_expiry_days=signal_expiry,
        )
        if args.top:
            res["new_alerts"] = res["new_alerts"][: args.top]
        return res

    def _one_pass_short() -> dict:
        res = live_short_pass(
            group=group,
            timeframe=timeframe,
            symbols=symbols,
            equity=equity,
            risk_pct=risk_pct,
            min_rally_score=min_rally,
            min_ml_prob=min_ml,
            min_rr=min_short_rr,
            require_macro_pass=macro_gate,
            state_file=args.state_file,
            signal_expiry_days=signal_expiry,
        )
        if args.top:
            res["new_alerts"] = res["new_alerts"][: args.top]
        return res

    def _merge_long_short(long: dict, short: dict) -> dict:
        merged = dict(long)
        merged["new_alerts"] = long["new_alerts"] + short["new_alerts"]
        merged["candidates"] = long["candidates"] + short["candidates"]
        merged["skipped_dup"] = long["skipped_dup"] + short["skipped_dup"]
        merged["sizing_failed"] = long.get("sizing_failed", 0) + short.get(
            "sizing_failed", 0
        )
        merged["data_stale"] = long["data_stale"] or short["data_stale"]
        merged["data_age_days"] = max(long["data_age_days"], short["data_age_days"])
        return merged

    def _one_pass() -> dict:
        if args.mode == "short":
            return _one_pass_short()
        if args.mode == "both":
            return _merge_long_short(_one_pass_long(), _one_pass_short())
        return _one_pass_long()

    def _emit(res: dict) -> None:
        briefing = format_briefing(res)
        if args.json:
            out = {
                "date": res["date"],
                "group": res["group"],
                "timeframe": res["timeframe"],
                "scanned": res["scanned"],
                "candidates": res["candidates"],
                "skipped_dup": res["skipped_dup"],
                "sizing_failed": res.get("sizing_failed", 0),
                "macro": res.get("macro"),
                "alerts": [
                    {
                        "symbol": a["symbol"],
                        "key": a["key"],
                        "direction": a.get("direction", "long"),
                        "text": a["text"],
                    }
                    for a in res["new_alerts"]
                ],
            }
            print(json.dumps(out, indent=2, default=str))
            return
        if args.dry_run:
            print("[dry-run] " + briefing)
            return
        n_ok = send_all(channels, briefing)
        logging.info(
            "pass done: %d new alerts, %d channels delivered",
            len(res["new_alerts"]),
            n_ok,
        )

    if args.watch:
        logging.info(
            "watch mode: %s %s every %.0f min", group, timeframe, args.interval_min
        )
        while True:
            try:
                _emit(_one_pass())
            except Exception as exc:
                logging.error("pass failed: %s", exc)
            time.sleep(args.interval_min * 60)
        return 0

    try:
        _emit(_one_pass())
    except Exception as exc:
        print(f"[live] pass failed: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
