"""
NexusQuant - Equity Factor Scanner CLI.

Combines the fundamental factor model, news/social sentiment and the
existing technical pipeline into one ranked equity read:

    python -m src.equity.run                          # all equity_universe D1
    python -m src.equity.run --symbols AAPL,MSFT,TSLA
    python -m src.equity.run --top 10
    python -m src.equity.run --fetch-news             # pull news (cached 1d)
    python -m src.equity.run --json
    python -m src.equity.run --symbols AAPL --detail  # full factor+news read

Everything is graceful: missing fundamentals -> value/quality report None
(price momentum still scores), offline news -> sentiment None. Put a
``data/fundamentals/universe.csv`` (see src/equity/fundamentals.py) to
activate the fundamental factors.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Dict, List, Optional

import pandas as pd

sys.path.append(str(Path(__file__).parent.parent.parent))

from src.data.loader import clean_data, load_data
from src.features.indicators import add_all_indicators
from src.equity.fundamentals import factor_scores, load_fundamentals
from src.equity.sentiment import sentiment_report

DEFAULT_GROUP = "equity_universe"
DEFAULT_TIMEFRAME = "D1"
DEFAULT_DATA_DIR = "data/raw"


def equity_scan(
    symbols: Optional[List[str]] = None,
    data_dir: str = DEFAULT_DATA_DIR,
    group: str = DEFAULT_GROUP,
    timeframe: str = DEFAULT_TIMEFRAME,
    fetch_news: bool = False,
) -> pd.DataFrame:
    """Factor + sentiment + technical summary for a list of equities."""
    if symbols is None:
        base = Path(data_dir) / group
        files = sorted(base.glob(f"*_{timeframe}.parquet"))
        symbols = [f.stem.replace(f"_{timeframe}", "") for f in files]
    if not symbols:
        raise RuntimeError(f"No {timeframe} data found in {data_dir}/{group}")

    rows = []
    for sym in symbols:
        try:
            rows.append(
                equity_read(sym, data_dir, group, timeframe, fetch_news=fetch_news)
            )
        except Exception as exc:
            print(f"[!] {sym}: {exc}", file=sys.stderr)

    table = pd.DataFrame(rows)
    if not table.empty:
        # Rank by composite factor score, then technical bias.
        table["composite"] = pd.to_numeric(table["composite"], errors="coerce")
        table["bias_score"] = pd.to_numeric(table["bias_score"], errors="coerce")
        table = table.sort_values(["composite", "bias_score"], ascending=[False, False])
        table["rank"] = range(1, len(table) + 1)
    return table


def equity_read(
    symbol: str,
    data_dir: str = DEFAULT_DATA_DIR,
    group: str = DEFAULT_GROUP,
    timeframe: str = DEFAULT_TIMEFRAME,
    fetch_news: bool = False,
) -> Dict:
    """One equity: fundamentals factors + sentiment + technical snapshot."""
    path = Path(data_dir) / group / f"{symbol}_{timeframe.upper()}.parquet"
    df = clean_data(load_data(path, symbol=symbol))
    df = add_all_indicators(df)

    fund = load_fundamentals(symbol)
    factors = factor_scores(symbol, df, fund)
    sent = sentiment_report(symbol, fetch_news=fetch_news)
    from src.analysis.scanner import _directional_score

    bias = _directional_score(df)["score"]

    latest = df.iloc[-1]
    close = float(latest["close"])
    sma200 = latest.get("sma_200")
    rsi = latest.get("rsi_14")
    adx = latest.get("adx")

    def num(v):
        try:
            f = float(v)
            return None if math.isnan(f) else round(f, 2)
        except (TypeError, ValueError):
            return None

    return {
        "symbol": symbol,
        "close": round(close, 2),
        "date": str(df.index[-1].date()),
        "value": factors["value"],
        "quality": factors["quality"],
        "momentum": factors["momentum"],
        "composite": factors["composite"],
        "news_score": num(sent["news"].get("score")),
        "news_n": sent["news"].get("n_articles", 0),
        "news_relevant": sent["news"].get("relevant", 0),
        "sentiment": num(sent.get("composite")),
        "rsi_14": num(rsi),
        "adx": num(adx),
        "vs_sma200_pct": num((close / sma200 - 1) * 100) if sma200 else None,
        "fundamentals_source": factors["fundamentals_source"],
        "bias_score": bias,
    }


def print_table(table: pd.DataFrame) -> None:
    print("\n" + "=" * 108)
    print("NEXUSQUANT EQUITY FACTOR SCANNER")
    print("=" * 108)
    show = table.copy()
    for col in (
        "value",
        "quality",
        "momentum",
        "composite",
        "news_score",
        "sentiment",
        "rsi_14",
        "adx",
        "vs_sma200_pct",
    ):
        if col in show:
            show[col] = show[col].map(lambda v: "-" if pd.isna(v) else f"{v:.1f}")
    print(show.to_string(index=False))
    print("=" * 108 + "\n")


def print_detail(record: Dict) -> None:
    print("\n" + "=" * 60)
    print(f"NEXUSQUANT EQUITY READ — {record['symbol']}")
    print(f"Date: {record['date']}  |  Close: {record['close']}")
    print("=" * 60)
    print(
        f"  Value      : {record['value']}   "
        f"(fundamentals source: {record['fundamentals_source']})"
    )
    print(f"  Quality    : {record['quality']}")
    print(f"  Momentum   : {record['momentum']}")
    print(f"  Composite  : {record['composite']}")
    print(
        f"  News score : {record['news_score']} "
        f"({record['news_n']} articles, "
        f"{record['news_relevant']} relevant)"
    )
    print(f"  Sentiment  : {record['sentiment']}")
    print(
        f"  RSI14 {record['rsi_14']} · ADX {record['adx']} · "
        f"vs SMA200 {record['vs_sma200_pct']}%"
    )
    print("=" * 60 + "\n")


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        description="NexusQuant equity factor scanner",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--symbols", default=None, help="comma-separated symbol list")
    parser.add_argument("--group", default=DEFAULT_GROUP)
    parser.add_argument("--timeframe", default=DEFAULT_TIMEFRAME)
    parser.add_argument("--data-dir", default=DEFAULT_DATA_DIR)
    parser.add_argument("--top", type=int, default=None)
    parser.add_argument(
        "--fetch-news",
        action="store_true",
        help="fetch fresh news headlines (cached 1 day)",
    )
    parser.add_argument(
        "--detail", action="store_true", help="full read for the first symbol"
    )
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args(argv)

    syms = (
        [s.strip() for s in args.symbols.split(",") if s.strip()]
        if args.symbols
        else None
    )

    if args.detail:
        sym = syms[0] if syms else None
        if not sym:
            base = Path(args.data_dir) / args.group
            files = sorted(base.glob(f"*_{args.timeframe}.parquet"))
            sym = files[0].stem.replace(f"_{args.timeframe}", "") if files else None
        if not sym:
            print("No symbol found.", file=sys.stderr)
            return 1
        print_detail(
            equity_read(
                sym,
                args.data_dir,
                args.group,
                args.timeframe,
                fetch_news=args.fetch_news,
            )
        )
        return 0

    table = equity_scan(
        syms, args.data_dir, args.group, args.timeframe, fetch_news=args.fetch_news
    )
    if args.top:
        table = table.head(args.top)
    if args.json:
        records = table.to_dict(orient="records")
        for rec in records:
            for k, v in rec.items():
                if isinstance(v, float) and math.isnan(v):
                    rec[k] = None
        print(json.dumps(records, indent=2))
        return 0
    print_table(table)
    print(
        f"Scanned {len(table)} equities across "
        f"{args.data_dir}/{args.group} ({args.timeframe})."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
