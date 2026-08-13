# NexusQuant

**Institutional-Grade Multi-Asset Quantitative Trading System**  
Focused on FX, Metals, and Commodities.

NexusQuant is a systematic research & trading platform that produces structured institutional-style analysis (regime detection, multi-timeframe technicals, confluence levels, ensemble signals, risk management, and backtesting).

---

## Features

- Multi-timeframe Market Regime Detection
- Support / Resistance + Fibonacci Confluence (swing/fractals, classical pivots, Fib retracements & extensions, confluence scoring)
- Buy-the-Dip **and** Sell-the-Rally Engines (long + short rule-based setup
  scoring with multi-timeframe triggers — full symmetric stack)
- HMM + KMeans regime detection (institutional spec #1's "HMM or clustering")
- Dual long/short ensemble model (P(long) and P(short) with net bias)
- Anchored VWAP + volume-profile HVN/LVN nodes in the S/R engine
- Equity fundamentals provider (yfinance) + S&P 500 universe management
- Composite news/social sentiment package (`src/sentiment/`)
- Interactive Streamlit Dashboard (universe ranking, symbol detail, comparison, backtest)
- Causal Backtesting Engine for the Buy-the-Dip strategy (no-lookahead signals)
- Ensemble Signal Model (LightGBM): Bullish Probability % + feature
  importance (top features + factor-group gain share in every report)
- Risk & Position Sizing Module (fractional / volatility-targeted / Kelly,
  per-trade & portfolio VaR, portfolio heat, correlation-aware limits,
  daily/weekly loss limits)
- Macro Overlay (USD strength from DXY, risk sentiment from VIX, rates
  pressure from US10Y → per-symbol macro bias + a causal signal gate)
- Live Signal & Alert System (scheduled full-stack signal pass → Discord /
  console / file alerts with dedup)
- Full Moving Average Structure & Ribbon Analysis (golden/death cross
  probability, ribbon slope/width)
- Momentum & Oscillator Suite (RSI, MACD, Bollinger, regular/hidden
  divergences + RSI failure swings)
- Volume & Flow Analysis (OBV, A/D line, relative volume, anchored VWAP,
  volume-profile high-volume nodes)
- Fibonacci Confluence Map (38.2–161.8 ratio table with 1–10 strength)
- Macro Sensitivity Table (market β vs S&P 500, dollar / yield / vol
  correlations, sector ETF)
- Trade Setup Construction (Entry / ATR Stop / Targets)
- Clean modular architecture

## Documentation

- [`docs/architecture.md`](docs/architecture.md) — system map, data flow,
  module responsibilities, deployment
- [`docs/api_reference.md`](docs/api_reference.md) — every REST endpoint
  with params and examples
- [`docs/trading_models.md`](docs/trading_models.md) — methodology of every
  engine (regime, levels, dip, ML, macro, risk, backtest, rating)
- [`docs/TWO_SIDED_ENGINE_AUDIT.md`](docs/TWO_SIDED_ENGINE_AUDIT.md) — the
  two-sided forensic audit: long/short symmetry table, root cause,
  remediation, census evidence and remaining limitations

---

## Project Structure

```
NexusQuant/
├── data/
│   ├── raw/                  # Original MT5 / CSV exports
│   └── processed/            # Clean Parquet files
├── notebooks/                # Research & analysis notebooks
├── api/                      # FastAPI service + SQLAlchemy persistence
├── src/
│   ├── data/                 # Data loading & cleaning + MT5 bridge,
│   │                         #   freshness checks & incremental updates
│   ├── features/             # Indicators, regime (incl. HMM/cluster), levels,
│   │                         #   dip + rally engines, patterns, VWAP, vol profile
│   ├── equity/               # Factor model, fundamentals, sentiment, universe,
│   │                         #   yfinance data provider
│   ├── sentiment/            # News / social composite sentiment aggregator
│   ├── analysis/             # Report generation, scanner, dashboard data,
│   │                         #   final quant rating
│   ├── backtest/             # Causal signals + trade simulation engine
│   ├── model/                # LightGBM ensemble (long + short heads)
│   ├── risk/                 # Sizing, targets, stress, VaR, RiskManager
│   ├── macro/                # Macro overlay (DXY/VIX/TNX scores + gate)
│   └── live/                 # Signal pass + alert channels (Discord)
├── config/                   # Settings
├── Dockerfile                # API service image (Python 3.14)
├── docker-compose.yml        # Postgres + API stack
├── Makefile                  # Dev convenience targets
├── tests/
└── docs/
```

---

## Quick Start

1. Create virtual environment:
```bash
python -m venv venv
source venv/bin/activate          # Linux/Mac
# or
venv\Scripts\activate             # Windows
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Place your historical data (CSV/JSON) into `data/raw/`

4. Run the data cleaning notebook or script.

5. Open `notebooks/03_full_analysis.ipynb` to generate institutional reports.

---

## Recommended Workflow

1. Clean & process data → `src/data/loader.py`
2. Generate features → `src/features/`
3. Produce full analysis report → `src/analysis/report.py`
4. Scan & rank the whole universe → `src/analysis/scanner.py`
5. Backtest strategies → `src/backtest/`
6. Train the ensemble model → `src/model/`
7. Size positions & check portfolio risk → `src/risk/`
8. Add top-down macro context → `src/macro/`
9. Alert on live signals → `src/live/`

---

## Universe Scanner

Run the institutional ranking across every symbol in a data group in one command:

```bash
python -m src.analysis.scanner                          # defaults to full_fx (D1)
python -m src.analysis.scanner --group full_fx          # all FX pairs
python -m src.analysis.scanner --group candidates       # US30, US500, USTEC, BTCUSD
python -m src.analysis.scanner --group equity_universe  # 500 US stocks
python -m src.analysis.scanner --group full_fx --timeframe H4  # intraday
python -m src.analysis.scanner --group crypto          # crypto universe
python -m src.analysis.scanner --group metals          # gold & silver crosses
python -m src.analysis.scanner --top 10                 # show only the top 10
python -m src.analysis.scanner --symbols EURUSD GBPUSD --json  # JSON output
```

Each row shows the current regime, a directional bias score (−4…+4) with label,
ADX, RSI, MACD histogram, distance from the 200-SMA, and ATR % — sorted by
bias strength then trend strength.

---

## MetaTrader 5 Integration

NexusQuant can pull live market data from the running MetaTrader 5 terminal
(under Wine, via the `rpyc` bridge on port 8001) whenever a symbol/timeframe
is missing locally. The scanner does this automatically — no flags needed:

```bash
python -m src.analysis.scanner --symbols XAUUSD --group metals --timeframe H4
python -m src.data.mt5 --symbol EURUSD --timeframe H1 --group full_fx --bars 20000
python -m src.data.mt5 --symbols EURUSD,GBPUSD --timeframe D1 --group full_fx
python -m src.data.mt5 --list-symbols --group-filter '*XAU*'
python -m src.analysis.scanner --no-mt5   # disable the MT5 fallback
```

### Bulk backfill of the full universe

Download every symbol in MT5 across several timeframes in one command, then
**regroup** the staged files into the asset-class group folders the rest of
the system consumes:

```bash
python -m src.data.mt5 --backfill                    # full universe D1/H4/H1
python -m src.data.regroup                           # classify into group folders
python -m src.data.regroup --dry-run                 # preview before moving
python -m src.data.regroup --clean --dry-run         # preview legacy cleanup
python -m src.data.regroup --clean                   # merge root majors + crosses/
```

The backfill stages everything under `data/raw/mt5/{D1,H4,H1}/` (timeframe-
first, resumable: existing files are skipped, so re-running picks up new
symbols or continues after an interruption). The regroup step moves those
files into the **asset-class group folders**, keeping all timeframes flat in
each folder exactly as the curated datasets are stored:

```
data/raw/
├── full_fx/           # FX pairs incl. exotics (EURUSD, AUDTRY, GBPCZK, ...)
├── candidates/        # US30, US500, USTEC, BTCUSD
├── equity_universe/   # S&P 500 constituents + Yahoo D1 + MT5 H4/H1
├── equity/            # other stocks / ETFs
├── crypto/            # BTCUSD, ADAUSD, ETHBTC, BTCAUD, ...
├── metals/            # XAUUSD, XAGJPY, XAUUSD247, ...
├── commodities/       # XCUUSD, XNGUSD, XALUSD, USOIL, ...
└── indices/           # US30, AUS200, DE30, DXY, US30_x10, ...
```

Each folder holds `{SYMBOL}_{D1,H4,H1}.parquet` files flat, so
`--group full_fx --timeframe H4` works without any special handling. There
are **no files at the `data/raw/` root** (the historic top-level majors and
the redundant `crosses/` subset were merged into `full_fx/`; the scanner
defaults to `full_fx` when no group is given).

**Duplicate policy** — when a symbol/timeframe already exists in a group
(e.g. the curated Yahoo `equity_universe` D1 files), the file with the
longest history wins; a fresh MT5 pull never silently truncates a longer
dataset. Empty staging directories are pruned automatically. Symbols without
data for a timeframe (e.g. some crypto pairs on D1) are reported as failures
at the end without stopping the run.

**Data hygiene** — instruments with fewer than 100 bars (unable to feed the
SMA-200 pipeline, e.g. newly listed crypto or 6-bar `XAUUSD247`) are moved
to `data/archive/` (with a `_manifest.txt`) so they stay out of discovery
without being deleted.

If the Wine bridge is not running, the scanner reports a clear error and
continues with the symbols that are available locally.

### Keeping data fresh (incremental updates)

The local parquet files are the basis for every decision, so freshness is
a first-class concern. Two commands cover it:

```bash
python -m src.data.update --check                  # freshness report (offline)
python -m src.data.update --group full_fx          # append missing bars from MT5
python -m src.data.update --group full_fx --timeframes D1,H4
python -m src.data.update --symbols EURUSD,GBPUSD --group full_fx
python -m src.data.update --group full_fx --dry-run  # preview, no fetch
```

* **`--check`** reads only the local files and reports every symbol's last
  bar date, age in days, bar count and a **FRESH / STALE / MISSING**
  status. Thresholds live in `config/settings.yaml`
  (`data.max_stale_days`: D1=4, H4=2, H1=2) and absorb weekends — a Friday
  D1 bar is still fresh on Monday.
* **the update itself** is **incremental**: for each file it reads the
  last bar date and fetches only the bars since then from MT5
  (`copy_rates_range`), merges with dedup on date and writes back — a full
  universe refresh is seconds, not a re-download. Missing files are
  full-downloaded. Per-symbol failures never stop the run.

The **live signal pass also gates on freshness**: the briefing shows
"Data as of Nd ago", and when the watchlist bars are older than the
threshold it prints a ⚠️ DATA STALE warning with the exact update command
(decisions on stale bars are decisions on the past).

Suggested cron (daily before the London open) — **installed** in your
crontab as `0 6 * * 1-5` (Mon–Fri 06:00 GMT, append-only — existing
entries are preserved, prior crontab is backed up under `logs/`). It runs
`scripts/daily_morning.sh`, which freshens data → confirms freshness →
runs the live signal pass, logging everything to `logs/daily.log`:

```bash
scripts/daily_morning.sh    # the installed routine (run it manually anytime)
# or the equivalent raw cron line, if you'd rather manage it yourself:
0 6 * * 1-5  cd /home/manuelhorveydaniel/Projects/NexusQuant && \
              ./venv/bin/python -m src.data.update --group full_fx && \
              ./venv/bin/python -m src.live.run --group full_fx
```

The live pass is **dual-sided by default** (`--mode both`): it evaluates
Buy-the-Dip longs *and* Sell-the-Rally shorts every run and alerts on
whichever side qualifies (`--mode long` / `--mode short` to restrict).

Cron has none of your shell exports, so push-channel secrets (Discord
webhook / Telegram token+chat) are sourced by the script from a
**gitignored** `.env.live` at the project root when present
(`cp .env.example .env.live`, or create it with the `NEXUS_*` vars).
Without it, the scheduled run still logs the briefing locally.

---

## Buy-the-Dip Confirmation Engine

Rule-based trade-setup logic that turns the regime + levels + momentum work
into an actionable **buy-the-dip** signal. A dip is only "confirmed" when the
broader structure is still bullish — bear trends are never confirmed, no
matter how oversold.

```bash
python -m src.analysis.scanner --group full_fx   # dip_score/dip_confirmed columns
```

Each symbol gets:

| Column | Meaning |
|--------|---------|
| `dip_score` | 0–8 confirmation strength |
| `dip_confirmed` | yes / no (requires bullish structure + ≥5 score) |
| `dip_stage` | `Uptrend` → `In Pullback` → `Dip Confirmed` |
| `entry_zone` | pullback zone between the 0.382–0.618 fib retracement |
| `invalidation` | level that negates the setup (below the dip's swing low) |

### Scoring rules (0–8)

- **Trend** (ADX ≥ 20) and **MA stack** (SMA20 > SMA50 > SMA200) — 2 pts
- **Above SMA200** — 1 pt
- **Pullback**: price ≥ 2 ATR below the last swing high, RSI < 55, MACD
  histogram negative or declining — 2 pts
- **Support**: price near a confluence level or the 0.618 fib retracement
  (within 0.25 × ATR) — 1 pt
- **Trigger** (from H4/H1 if available locally, else D1): MACD histogram
  rising, RSI turning up, bullish bar — ≥ 2 of 3 — 2 pts

A score of **≥ 5 with bullish structure** confirms the setup. `dip_confirmed`
comes with an `entry_zone` (0.382–0.618 of the last up-leg) and an
`invalidation` level. Multi-timeframe triggers are loaded from local
`data/raw/h4/` and `data/raw/h1/` files when present — no extra MT5 fetches.

---

## Sell-the-Rally Short Engine

The exact mirror of the Buy-the-Dip engine for **short setups in downtrends**
(`src/features/rally.py`): a rally into resistance is only "confirmed" when
the broader structure is still bearish — bull trends are never confirmed, no
matter how overbought. The full long/short stack is symmetric, and `both`
is the live pass default (a plain `src.live.run` scans both sides):

```bash
python -m src.analysis.scanner --group full_fx   # rally_score/rally_confirmed columns
python -m src.live.run                            # default: long + short pass
python -m src.live.run --mode short --symbols USDJPY,EURJPY,GBPJPY   # short-only pass
python -m src.risk.run --symbol USDJPY --group full_fx --short       # short risk plan
python -m src.model.run --group full_fx --side short                 # train the short model
python -m src.model.run --group full_fx --side long                  # retrain long with side param
```

Short alerts surface `ml_short_prob` (from `models/rally_lgbm.joblib` when
trained — rule-based fallback until then) and the short book's own stress
view in the institutional report.

Each symbol gets the mirror-image columns:

| Column | Meaning |
|--------|---------|
| `rally_score` | 0–8 confirmation strength |
| `rally_confirmed` | yes / no (requires bearish structure + ≥5 score) |
| `rally_stage` | `Downtrend` → `In Rally` → `Rally Confirmed` |
| `entry_zone` | resistance zone (0.382–0.618 of the last down-leg) |
| `invalidation` | level that negates the setup (above the rally's swing high) |

### Scoring rules (0–8, mirror of dip)

- **Trend** (ADX ≥ 20) and **MA stack** (SMA20 < SMA50 < SMA200) — 2 pts
- **Below SMA200** — 1 pt
- **Rally**: price ≥ 2 ATR above the last swing low, RSI > 60, MACD
  histogram positive or rising — 2 pts
- **Resistance**: price near a confluence level or the 0.618 fib retracement
  of the last down-leg — 1 pt
- **Trigger** (H4/H1 if local, else D1): MACD histogram falling, RSI turning
  down, bearish bar — ≥ 2 of 3 — 2 pts

A score of **≥ 5 with bearish structure** confirms the short setup. The
short target ladder, risk plan (`risk_plan_from_report_short`), live filter
(`filter_short_signals` / `format_short_alert`), backtest signal series
(`rally_signal_series`) and scanner columns all mirror the long side, and
the **minimum R:R floor is enforced on the short ladder too**.

---

## Direction-Neutral Setup Classifier (two-sided audit)

The two confirmation engines above are *structure-gated pullback* engines
(a dip only fires above the 200-SMA, a rally only below it) — correct for
counter-trend trading, but it means the architecture could only express
"find a bullish structure, buy weakness" / "find a bearish structure, sell
strength". The setup classifier (`src/features/setups.py`) is the
**direction-neutral layer that sits above the engines**:

```text
MARKET DATA -> FEATURES -> REGIME -> SETUP CLASSIFIER
  -> LONG EVIDENCE SCORE / SHORT EVIDENCE SCORE (independent)
  -> best family per side -> direction verdict (long/short/flat)
  -> calibrated ML probabilities -> EV -> risk
```

- **12-family taxonomy** — LONG_TREND_CONTINUATION / LONG_BUY_DIP /
  LONG_BREAKOUT / LONG_BREAKOUT_RETEST / LONG_REVERSAL /
  LONG_MEAN_REVERSION and the six short mirrors (SHORT_BREAKDOWN,
  SHORT_BREAKDOWN_RETEST, SHORT_SELL_RALLY, ...).
- **Direction first, setup second** — the 200-SMA is *context, not a gate*:
  a breakdown-retest short can fire above the 200-SMA and a breakout-retest
  long below it when the other evidence justifies it.
- **Independent long/short evidence scores** — not `short = -long`; each
  family has its own causal logic (retests are level-relative within ~1 ATR,
  trend-continuation requires momentum alignment, reversals fire against the
  SMA relationship).
- **Engine veto/boost** — the engines gate their own pullback families
  (`No Uptrend` vetoes LONG_BUY_DIP) but the classifier's new families stay
  unconstrained, which is exactly what the engines cannot see.
- **Expected value** — `expected_value()` and `probability_weighted_rr()`
  return `None` (never fabricate) without a *calibrated* ML probability.

Surfaced everywhere: report **section 11d**, plan table **SETUP** column,
scanner `setup / long_evidence / short_evidence / setup_ev` columns, and
`--format plan` output.

**Historical opportunity census** (`src/analysis/census.py`) replays every
bar causally (rolling-window classification, first-touch realized R) to
answer the spec's central question: *does the engine see both sides?*

```bash
python -m src.analysis.census --group full_fx          # long vs short census
python -m src.analysis.census --group full_fx --show 5 # per-family counts
```

Across 30k+ bars of full_fx history the signal ratio is **0.99 ≈ 1.0**
(4,469 long vs 3,793 short candidates) — direction-neutral in opportunity
detection; the market decides relative trade frequency. See
[`docs/TWO_SIDED_ENGINE_AUDIT.md`](docs/TWO_SIDED_ENGINE_AUDIT.md) for the
full forensic asymmetry audit, root-cause analysis and validation.

---

## HMM Regime Detection

`src/features/regime.py` now offers the institutional spec's "HMM *or*
clustering" choice in full: the rule-based label, the **KMeans clustering
variant** (`regime_cluster`), and a **4-state Gaussian HMM**
(`detect_regime_hmm`, via `hmmlearn`):

```python
from src.features.regime import detect_regime_hmm, multi_timeframe_regime

frame = detect_regime_hmm(df)                  # adds regime_hmm column (4 states)
mtf   = multi_timeframe_regime(df, use_hmm=True)  # D/W/M table with HMM label
```

The four HMM states are auto-labeled by feature ordering (volatility ×
return centroids) as Bull Trend / Bear Trend / Range / High Volatility. The
MTF table and reports can consume the HMM label interchangeably with the
cluster label; when `hmmlearn` is missing the module degrades gracefully to
the deterministic label. Also new: **anchored VWAP** + **volume-profile
HVN/LVN nodes** in the levels engine — both are wired into the report's
S/R section.

---

## Dual Long/Short Ensemble Model

The ML layer now trains **separate calibrated models per side** — a
long-side model on confirmed dips (existing) and a **short-side model on
confirmed rallies** — so the system outputs `P(long)` and `P(short)`
independently:

```bash
python -m src.model.run --group full_fx --side short   # train models/dip_lgbm_short.joblib
python -m src.model.run --group full_fx --side long    # retrain long with --side
python -m src.live.run --mode both                     # uses both probabilities
```

* `src/model/features.py` — `make_meta_labels_short` / `build_labels_short`
  mirror the long labels on confirmed-rally bars; `build_dataset_dual`
  returns both `y_long` and `y_short`; `build_dataset(..., side=...)`
  parameterises the whole pipeline for either side.
* `src/model/model.py` — `DEFAULT_SHORT_MODEL_PATH`
  (`models/dip_lgbm_short.joblib`), `predict_short_series` (same
  causal-context serving as the long path), and
  `predict_long_short` → `{prob_long, prob_short, net_bias}`.
* Reports carry a **short ML probability** and the rating consumes
  `net_bias = P(long) − P(short)` so a symbol can rate **Strong Sell / Sell**
  as well as Buy — the full 5-tier scale (85/70/50/30) works in both
  directions.

---

## Equity Fundamentals Provider + Sentiment Package

Two additions close the data-side gaps behind specs #8/#9:

* `src/equity/data_provider.py` — **yfinance fundamentals provider**: pulls
  trailing P/E, EV/EBITDA, P/B, ROE, Debt/Equity, ROIC, current ratio, FCF
  yield, earnings surprise and analyst revisions per symbol, cached to
  `data/fundamentals/{SYMBOL}.csv` (gitignored). Handles the unauthenticated
  Yahoo case gracefully (returns `None` fields, keeps momentum working).
* `src/equity/universe.py` — **S&P 500 universe management**: membership
  sync (`_membership.csv`), current constituents via yfinance with cached
  fallback, and per-ticker Yahoo symbol mapping.
* `src/sentiment/` — the planned package structure
  (`news.py`, `social.py`, `aggregator.py`): the **aggregator** combines a
  news score (60%) and a social score (40%) into a composite −1…+1 with a
  relevance-weighted confidence, defaulting to the existing Yahoo-headline
  lexicon scorer when keyed news/social providers are unavailable. The
  equity scanner reads the composite when present.

```bash
python -m src.equity.data_provider --symbols AAPL,MSFT,TSLA   # warm fundamentals cache
python -m src.equity.universe --sync                          # refresh S&P 500 membership
```

---

## Streamlit Dashboard

Interactive dashboard on top of all three engines (regime, confluence,
buy-the-dip). Dark institutional theme, **local files only** — no MT5
fetches, so it never contends with the backfill:

```bash
streamlit run dashboard.py
# or
python -m streamlit run dashboard.py
```

Five pages:

- **🌐 Universe Ranking** — ranked table with colour-coded bias / regime /
  dip status, filters (regime, dip status, bias direction, macro gate,
  min dip score, top N, symbol search), a feature heatmap and regime
  distribution chart.
- **🔍 Symbol Detail** — Buy-the-Dip status banner, a 🌍 Macro Overlay
  banner (regime + bias + gate chip), metric cards, and tabs:
  overview tables, candlestick chart with nearest S/R + entry zone +
  invalidation, RSI/MACD momentum subplots, confluence zones + classical
  pivots, and the 8-factor dip confirmation checklist.
- **⚖️ Compare Symbols** — side-by-side metrics, normalized close chart,
  bias and dip-score bar charts for any selection of symbols.
- **🧪 Backtest** — interactive dip backtest (risk, R:R, max-hold, entry
  type, sizing method, start year) with edge banner, metric cards,
  equity/drawdown chart and trade table.
- **🛡️ Risk** — per-trade risk plan (all three sizing methods + VaR on the
  live dip setup) with adjustable equity / risk / vol-target / Kelly
  inputs, and a button-triggered portfolio report (heat, portfolio VaR,
  correlation-aware gates, most-correlated pairs).

Every dataset discovered under `data/raw/` appears in the sidebar group
selector — one entry per `group · timeframe` combination (majors, full_fx,
candidates, equity_universe, equity, crosses, crypto, metals, commodities,
indices × D1/H4/H1). After a backfill + regroup, the new timeframes appear
automatically. Results are cached for 15 minutes; use **🔄 Refresh data** in
the sidebar to force a re-scan.

The dashboard data layer & chart builders live in
`src/analysis/dashboard_data.py` — pure functions, unit-tested headless
(no Streamlit import needed).

---

## Buy-the-Dip Backtester

Validates the edge of the dip strategy on historical data with a **fully
causal signal** (no lookahead) and a bar-by-bar trade simulation:

```bash
python -m src.backtest.run --symbol XAUUSD --group full_fx
python -m src.backtest.run --group full_fx --start 2012      # universe scan
python -m src.backtest.run --symbol US500 --group candidates --risk 0.02 --rr 2
python -m src.backtest.run --symbol XAUUSD --group full_fx --json  # machine output
python -m src.backtest.run --symbol XAUUSD --group full_fx --plot   # saves equity.png
python -m src.backtest.run --symbol USDJPY --group full_fx --side short   # Sell-the-Rally book
python -m src.backtest.run --symbol USDJPY --group full_fx --side both    # long+short combined view
```

The engine is **side-aware** (`--side long|short|both`): the short side is
the exact mirror (limit entry at the rally entry-zone high, stop ABOVE,
target BELOW at support; inverted PnL / mark-to-market / slippage
direction). ``both`` runs both books (each deploying independent capital)
and merges the equity curves - a documented research view of the combined
edge, not a shared-margin simulation.

**Signal** (`src/backtest/signals.py`) — the same 8-factor dip score evaluated
on every bar with strict causality: indicators use only past data, swing
highs/lows become usable only after their confirmation window closes, and
trade levels come from confirmed swings + the previous bar's pivots. One
documented simplification vs the live engine: confluence clustering is
replaced by the last confirmed swing high/low (keeps it O(n) so a 29-pair
universe backtests in ~3 s).

**Engine** (`src/backtest/engine.py`) — limit entry at the entry-zone low
(valid 3 bars) or market entry at the next open; stop at the invalidation;
target at the nearest resistance (R:R fallback); time-stop after max-hold
bars; fractional-risk sizing; slippage; equity curve. Intrabar, stops are
checked before targets (conservative).

**Stats** — trades, win rate, profit factor, expectancy (in R), total
return, CAGR, max drawdown, Sharpe, exposure, best/worst trade.

Example output (29 FX pairs, D1, 2012→now, 1% risk): the strategy shows
positive edge on trending pairs — XAUUSD **+15.9%** (56% win rate,
PF 2.47), CADJPY +12.9%, EURCAD +1.02R — and loses on structurally weak
ones (GBPNZD −11.1%, EURAUD −9.8%), i.e. it faithfully trades the
institutional dip logic.

Sizing is pluggable: `--sizing voltarget` targets a per-trade volatility
contribution and `--sizing kelly` uses fractional Kelly (win probability
passed with `--kelly-p`, or the ML probability for live plans) — both
capped at the per-trade risk budget:

```bash
python -m src.backtest.run --symbol XAUUSD --group full_fx --sizing voltarget
python -m src.backtest.run --symbol XAUUSD --group full_fx --sizing kelly --kelly-p 0.6
```

A **🧪 Backtest** tab in the dashboard runs the same engine interactively
(risk, R:R, max-hold, entry type, start year) with an equity/drawdown
chart and trade table.

---

## Risk & Position Sizing Module

Institutional risk layer on top of the signals & probabilities — the gap
between "trade this" and "how much":

```bash
python -m src.risk.run --symbol XAUUSD --group full_fx           # per-symbol plan
python -m src.risk.run --symbol XAUUSD --group full_fx --equity 250000
python -m src.risk.run --symbols XAUUSD,CADJPY,EURUSD --portfolio --equity 250000
python -m src.risk.run --symbol XAUUSD --group full_fx --json
python -m src.risk.run --symbols EURUSD,USDJPY,XAUUSD --portfolio \
                       --include-short          # long + short positions
```

``--include-short`` folds actionable Sell-the-Rally setups into the
portfolio report (each position tagged LONG/SHORT with direction-aware
risk). Short-book stress is included in every full report (section 18b) -
crash scenarios read as *favorable* to shorts (negative loss = gain) while
the vol-multiplied VaR still flags squeeze risk.

**Sizing** (`src/risk/sizing.py`) — three methods on the live dip setup
(entry zone / invalidation): fractional risk (`qty = equity·risk/(entry−stop)`),
volatility-targeted (notional sized so the trade contributes a target vol,
`vol·√hold` iid assumption, capped at the risk budget), and fractional
Kelly (`f·(p − (1−p)/b)`, probability from the ML model when present, also
capped). Kelly correctly goes to zero when the ML probability implies a
negative edge.

**Metrics** (`src/risk/metrics.py`) — per-trade parametric VaR (95/99,
`z·ATR·qty·√hold`), correlation-aware portfolio VaR, portfolio heat
(sum of risk $ / equity), and correlation-aware position gates.

**Limits** (`src/risk/limits.py`) — `RiskManager` tracks a paper/live
session: max daily / weekly loss halts, max concurrent positions, and max
portfolio heat, with `can_open` gating and day/week rollovers.

Every full report (CLI or dashboard) carries a **risk section**: the three
sizing rows with notional, risk $, risk % of equity and VaR — computed on
the current setup, using the ML probability for Kelly when the model exists.
The dashboard's **🛡️ Risk** page adds a button-triggered portfolio report
(heat vs 4% limit, portfolio VaR95, correlation-aware gate table, most
correlated pairs).

---

## Ensemble Signal Model (LightGBM)

Upgrades the rule stack from binary signals to a **Bullish Probability**
for the setups the system actually trades. Trained on the causal feature
matrix with strict chronological validation (single split or **purged
walk-forward CV**, one-horizon embargo at every boundary, and **early
stopping on a chronological validation tail** — the single biggest guard
against fixed-iteration overfit):

```bash
python -m src.model.run --group full_fx                  # train + walk-forward OOS (meta label)
python -m src.model.run --group full_fx --label 1r       # asymmetric 1R triple-barrier instead
python -m src.model.run --group full_fx --drop-censored  # 1r: drop censored labels
python -m src.model.run --group full_fx --search         # chronological hyperparameter search first
python -m src.model.run --group full_fx --stack          # report rule-score stacking comparison
python -m src.model.run --group full_fx --per-group      # separate gold / JPY-cross / majors models
python -m src.model.run --group full_fx --weight-vol     # vol-aware sample weighting
python -m src.model.run --group full_fx --cv 5           # purged walk-forward folds
python -m src.model.run --predict XAUUSD --group full_fx # live probability
python -m src.model.run --json                           # machine output
```

**Labels** (configurable, `--label`, **default `meta`**):

* `meta` (default) — the *actual* trade outcome for each **confirmed dip**
  only (limit entry at the fib-zone low, stop at the swing-low
  invalidation, target at the nearest resistance with an R:R fallback),
  exactly the question the model is deployed on. Non-filling /
  time-censored bars are dropped. This is the deployment objective —
  "does THIS setup work?" — instead of a generic 1R move, and it is what
  the live filter uses.
* `1r` — **asymmetric triple-barrier**: entry = close, stop = close −
  1.25·ATR, target = close + 0.75·ATR (`--stop-mult` / `--target-mult`),
  scanned stop-first over the next `horizon` bars. Rows that only resolve
  on the forward close are **censored** — down-weighted by default, dropped
  entirely with `--drop-censored`.

**Features** (54 total) — on top of momentum / trend / volatility /
returns / volume / regime / the 8 dip components / macro context
(`dxy_score`, `vix_score`, `tnx_score`, `macro_bias`, `dxy_mom20`) and the
`symbol` categorical, the v3 stack adds:

* **Calendar structure** — `day_of_week`, `month`, `month_end` (≥ 28th),
  `mid_month` (13th–21st, an FOMC-week proxy).
* **Interactions** — `vol × momentum`, `vol × trend`, `adx × slope`.
* **Multi-timeframe (H4)** — `h4_mom5/20`, `h4_vol_ratio`, `h4_vs_sma200`
  resampled to daily using only same-day H4 closes (strictly causal; 0
  when no H4 file exists).
* **Cross-asset risk context** — `risk_mom5/20` (AUDJPY/NZDJPY average
  momentum, the classic risk-on/off proxy) and `gold_mom5/20` (XAUUSD),
  aligned with a 1-day lag. Already among the top features by gain.
* **Positioning (COT)** — `cot_percentile`: the causal percentile of
  CFTC non-commercial net positioning for the symbol's positioning market
  (expanding window - each week ranked only against its own history),
  1-day lagged; neutral 50 when absent. Disable with `--no-cot`.

**COT data** is downloaded automatically from the CFTC public Socrata API
(Disaggregated Futures-Only report, no auth) and cached as
`data/raw/cot/{KEY}_cot.csv` (`date,percentile`) per market:

```bash
python -m src.model.cot --status              # offline cache report
python -m src.model.cot --fetch               # refresh when stale (weekly)
python -m src.model.cot --fetch --force       # always refetch
python -m src.model.run --group full_fx --fetch-cot   # refresh + train
```

**15 markets** — the 8 FX/metals futures (EURO FX, BRITISH POUND, JAPANESE
YEN, SWISS FRANC, AUSTRALIAN DOLLAR, NZ DOLLAR, CANADIAN DOLLAR, GOLD) plus
silver (XAG), WTI & Brent crude, and the equity-index futures
(S&P 500, NASDAQ, DJIA, Russell). Instrument symbols resolve through
`SYMBOL_MARKET` first (US500→SP500, USTEC→NASDAQ, US30→DOW, XAGUSD→XAG,
USOIL/XTIUSD→WTI, XBRUSD→BRENT), then `CCY_MAP` for FX symbols (JPY crosses
use the dollar/risk leg). The ICE Dollar Index has no disaggregated futures
report, so there is no DXY series — USD positioning is already covered by
the seven currency futures.

The cache counts as fresh while every market's last report is ≤ 16 days
old (two weekly cycles: some contracts like `WTI CRUDE OIL 1ST LINE`
consistently report one week behind the rest); a failed fetch never touches
the existing cache. On first pull it grabs 20+ years of weekly reports per
market (~1,150–1,930 rows), e.g. GOLD 82 (stretched long), EUR 18 / GBP 7
(short in the strong-dollar regime), DOW 73 (stretched long), NASDAQ 14
(squeezed short). With live COT the meta-label walk-forward OOS AUC moved
**0.562 → 0.578** and the dip filter gate to 47% top-half vs 38%
bottom-half.

**Sample weighting** — confirmed-dip bars are up-weighted
(`--weight-dip`, default 5×), censored labels down-weighted, and
`--weight-vol` additionally down-weights high-volatility bars (noise-heavy)
while up-weighting calm ones. Macro/MTF/cross/COT features all default to
neutral zeros when their data source is absent, so the pipeline never
branches on availability.

**Validation & calibration** — `--cv N` runs a **purged walk-forward** with
**early stopping on a chronological tail of each training fold** (AUC) and
reports pooled OOS metrics, a **decile table** (win rate per probability
decile — the honest test of filter monotonicity), and a **dip filter
gate** (does splitting confirmed dips at the median model probability
separate winners? — the deployment question global AUC can't answer).
Probabilities pass through an **isotonic calibrator** (fit on the
out-of-fold predictions) before saving, so they are true probabilities,
safe for the fractional-Kelly sizing in `src/risk/`.

**Hyperparameter search (`--search`)** — a 6-config grid (depth,
learning rate, min-child-samples, colsample) evaluated on an honest
chronological 75/25 split of the recent rows; the winning config is used
for the production model and recorded in its metadata. **`--per-group`**
trains separate `models/dip_lgbm_{gold,jpy,majors}.joblib` models (gold ≠
JPY crosses ≠ majors vol regimes) with per-group dispatch at predict time.
**`--stack`** reports a logistic-regression stack over
[LGBM prob, dip_score, macro_bias, adx] — on the current dataset it does
*not* add value (Δ −0.025), which is itself a useful result: the rule
scores are already inside the trees.

**Honest OOS numbers** (111-symbol FX universe, D1):

| Setup | OOS AUC | Deciles 0→9 win% | Dip gate (top-half vs base) |
|-------|---------|------------------|------------------------------|
| legacy (symmetric 1R, fixed 400 trees) | 0.52 | flat | — |
| v3 `1r` (early stop + new features) | **0.547** | 38 → 57 | 50% → 53% |
| v3 `meta` (default) + search | **0.561** | 31 → 57 | 42% → 46% (vs 38% bottom) |

AUC on daily FX/metals with pure technicals has a hard ceiling (~0.55–0.6
is already a good filter); the *lift* that matters is the dip gate — the
model separates the top half of confirmed dips from the bottom half, which
is exactly what the live `--min-ml-prob` filter exploits. Model saved to
`models/dip_lgbm.joblib` with metadata (features, label, OOS AUC, best
params, calibration flag, trained-at, symbols).

Integration is graceful — the scanner gains an `ml_prob` column, the report
an *Ensemble Model* section, and the dashboard a 🤖 probability banner — all
auto-enable when `models/dip_lgbm.joblib` exists and fall back to pure
rule-based output otherwise.

---

## Macro Overlay

Adds a **top-down context layer** on top of the technical engines. Three
macro factors are scored causally in [−2, +2] on daily bars:

- **USD strength** — from DXY (`data/raw/indices/DXY_H4.parquet`, resampled
  to daily when no D1 file exists) — price vs SMA200 + RSI vs 50.
- **Risk sentiment** — VIX level bands + 20-day change; positive = risk-on.
- **Rates pressure** — US10Y (`^TNX`) trend + slope; positive = easing.

VIX/TNX are a best-effort Yahoo fetch (no auth), cached under
`data/raw/macro/`; the overlay works fine with only DXY locally. Each factor
combines into a regime (USD Bullish/Bearish, Risk-On/Off, Easing/Tightening)
and a per-symbol **macro bias**: a strong dollar is a headwind for EURUSD but
a tailwind for USDJPY; gold likes a weak dollar; crypto is strongly
risk-sensitive; indices/equities like risk-on.

Every report also carries a **macro sensitivity table** (spec #9):
trailing 90-day (1-day lagged, causal) correlations of the symbol's daily
returns vs the S&P 500 (`market_beta` + `spx_corr`), DXY (`dollar_sens`),
US10Y (`yield_sens`), VIX (`vol_sens`) and the symbol's **sector ETF**
(`sector_etf`, from the `SECTOR_ETF` mega-cap map, cache-only). Warm the
sector-ETF cache once with `--fetch`; the table is omitted in cheap scans
that don't pass the symbol frame.

```bash
python -m src.macro.run                                  # macro snapshot
python -m src.macro.run --fetch                          # populate VIX/TNX + sector ETFs
python -m src.macro.run --symbols EURUSD,GBPUSD,XAUUSD   # per-symbol bias/gate
python -m src.macro.run --scan --group full_fx --top 10  # universe ranked by bias
python -m src.macro.run --compare XAUUSD --group full_fx # gate vs no-gate backtest
python -m src.macro.run --scan --group candidates --json
```

The bias powers a **gate**: a Buy-the-Dip signal is filtered out when the
macro backdrop for that symbol is a strong headwind. `--compare` runs the
same dip backtest with and without the gate (fully causal — the macro state
is only known from the prior day) and reports the edge delta. Example
(XAUUSD, D1, 2020→now): the gate blocked 60% of bars, cut trades from 22 to
8, and lifted win rate 54.5% → 75%, profit factor 2.42 → 11.99 and Sharpe
2.14 → 4.91.

Integration is graceful and automatic — the scanner gains
`macro_bias`/`macro_label`/`macro_gate` columns (and a **Macro gate** filter
on the Universe page), the report a *Macro Overlay* section, and the
Symbol Detail page a 🌍 banner with the regime + bias + gate chip. When no
DXY is available the overlay is simply absent.

---

## Live Signal & Alert System

Turns the whole research stack into a daily signal feed you can actually
act on: scan → filter → size → alert, with dedup so a setup that stays
valid for several days is alerted once, not every run.

```bash
python -m src.live.run                        # one pass, full_fx D1
python -m src.live.run --group candidates     # different watchlist
python -m src.live.run --symbols EURUSD,GBPUSD,XAUUSD
python -m src.live.run --min-ml-prob 55       # only high-conviction ML setups
python -m src.live.run --min-rr 1.5           # reward:risk floor
python -m src.live.run --dry-run              # print without sending
python -m src.live.run --json                 # machine-readable
python -m src.live.run --watch --interval-min 60   # loop for a scheduler
python -m src.live.run --format institutional --symbols EURUSD,SPY,XAUUSD \
                       --hmm                   # full 18-section report per symbol
```

(`--format institutional` prints reports directly - it is not an alert
push, so `--dry-run`/channels do not apply to it.)

``--format institutional`` prints the complete Citadel-style report per
symbol (all 18 sections: M/W/D regime incl. optional HMM, S/R + anchored
VWAP + volume profile, MA ribbon, momentum + divergences, patterns, fib
confluence map, fundamentals, macro + sentiment, ensemble long+short +
feature importance, long AND short setups with the 2.5 R:R floor, risk,
long+short stress, final quant rating).

**The pass** (`src/live/signals.py`) scans the watchlist through the whole
pipeline, keeps only high-conviction setups (confirmed dip, macro gate
PASS, optional ML-probability and R:R floors), sizes each one with the
risk module (fractional qty + R:R + VaR95), and formats a compact alert
with entry zone, invalidation, target, ML probability, macro bias and the
position size. Setups are deduped by `symbol + entry zone` in
`data/live/alerts.json`, so a daily cron only alerts when something is
actually new.

**Minimum reward:risk is enforced by default** — spec #11's 2.5:1 floor.
The nearest-resistance single target often offers only ~1R, so the floor is
evaluated on the **target ladder** (TP1→TP3 scaling-out plan from
`src/risk/targets.py`): `live.filters.min_rr` (default: `risk.min_reward_risk`
= 2.5) drops setups whose best achievable R:R is below the floor, and the
alert prints the floor verdict, the nearest-target R:R and the ladder's
best R:R. `python -m src.risk.run` shows the same verdict on every plan.

### On-demand run for any symbols

One command runs the **whole stack on any watchlist — any asset class**:
freshness check, weekly COT positioning refresh, universe scan through
regime / levels / dip / macro overlay, the trained ensemble model (served
with the *same* H4 / cross-asset / COT context it was trained on), risk
sizing — and prints the ranking plus every actionable trade setup:

```bash
python -m src.live.run --symbols EURUSD,GBPUSD,USDJPY,XAUUSD --group full_fx
python -m src.live.run --symbols GLD,AAPL,NVDA,US30 --group equity --dry-run  # stocks/ETFs too
python -m src.analysis.scanner --symbols EURUSD GBPUSD XAUUSD --group full_fx  # ranking table view
python -m src.data.update --check --symbols EURUSD,GBPUSD --group full_fx      # freshness only
python -m src.data.resolver GLD --timeframe H4        # resolve/fetch one symbol (debug)
python -m src.data.yahoo --symbols GLD,AAPL --timeframe D1   # Yahoo fetch only
python -m src.data.yahoo --ticker                     # symbol -> Yahoo ticker map
```

**On-demand data resolution** — missing symbols are fetched automatically
through a cascade (`src/data/resolver.py`), so *any* ticker works whether
or not it has been backfilled:

1. **Local, any group folder** — `data/raw/*/{SYMBOL}_{TF}.parquet` is
   searched across every group (not just the one requested), longest
   history wins when a duplicate exists.
2. **MetaTrader 5 terminal** — pulled from the running Wine bridge (covers
   its 357-symbol universe: FX, metals, indices, stocks, crypto,
   commodities) and cached into the **classified** asset-class folder.
3. **Yahoo Finance** — no-auth public chart API (US stocks / ETFs / exotic
   indices the terminal does not list, or when the bridge is down);
   `src/data/yahoo.py` maps symbols (US500→^GSPC, XAUUSD→GC=F, EURUSD→
   EURUSD=X, BTCUSD→BTC-USD, stocks as-is), supports D1 (10y) and H1/H4
   (1h bars, 730-day intraday cap), and caches into the classified folder.

Fetched files land in their classified group folder (equity/, indices/,
metals/, commodities/, full_fx/, crypto/) — never the `data/raw` root —
so a stock fetched on demand appears in the dashboard and future scans
without re-fetching. The scanner CLI, live pass and API all resolve on
demand by default; the dashboard stays local-only (fast, offline, no
contention with the bridge). `--no-mt5` disables both network fallbacks.

Symbols from any group work the same way (`--group candidates`, `metals`,
`equity_universe`, ...); the ML probability is `None` (rule-based fallback)
if the model files are missing, and the COT/MTF context degrades to
neutral features if those files are absent — the pass never fails on
missing data.

**Channels** (`src/live/alerts.py`) — Discord webhook, Telegram bot,
console and file, with per-channel failure isolation (one broken webhook
never kills the run). Console + file are on by default; to get push
alerts on your phone pick either chat channel:

**Discord**

1. In Discord, open **Server Settings → Integrations → Webhooks → New
   Webhook** and copy the URL.
2. Export it (preferred, keeps it out of the repo):
   ```bash
   export NEXUS_DISCORD_WEBHOOK="https://discord.com/api/webhooks/..."
   ```
3. Flip Discord on in `config/settings.yaml`:
   ```yaml
   live:
     channels:
       discord: true
   ```

**Telegram**

1. Create a bot with [@BotFather](https://t.me/BotFather) and copy the
   token.
2. Get your chat id: message your bot, then open
   `https://api.telegram.org/bot<TOKEN>/getUpdates` and read the
   `chat.id` from the JSON (or use @userinfobot).
3. Export both (preferred, keeps them out of the repo):
   ```bash
   export NEXUS_TELEGRAM_BOT_TOKEN="123456:ABC-DEF..."
   export NEXUS_TELEGRAM_CHAT_ID="123456789"
   ```
4. Flip Telegram on in `config/settings.yaml`:
   ```yaml
   live:
     channels:
       telegram: true
   ```

No new dependencies — both channels use only the stdlib (`urllib`).
Everything is local-file-based and safe for cron; the MT5 bridge can be
added later for intraday live data.

---

## REST API (FastAPI)

The whole research stack is exposed as a read-only REST service with
optional Postgres persistence — a natural front-end for the dashboard,
mobile clients, or scheduled jobs:

```bash
# Local dev (SQLite fallback persistence — no Docker needed):
make api                                  # uvicorn on :8000
# Full stack (Postgres + API in Docker):
make up                                   # docker compose up --build
# Just Postgres, API locally:
make db-up && NEXUS_DATABASE_URL=postgresql+psycopg://nexus:nexus@localhost:5432/nexus uvicorn api.app:app --port 8000
```

Interactive docs at `http://localhost:8000/docs` (OpenAPI). Endpoints:

| Method | Path | Description |
|--------|------|-------------|
| GET | `/health` | service + active DB backend (postgres / sqlite) |
| GET | `/api/v1/groups` | every discoverable `group · timeframe` dataset |
| GET | `/api/v1/scan?group=&timeframe=&symbols=&top=` | ranked universe table (all scanner columns incl. rating + pattern) |
| GET | `/api/v1/symbol/{s}` / `.../report` | full institutional report (regime, levels, dip, macro, ML, targets, rating, stress) |
| GET | `/api/v1/symbol/{s}/risk?equity=` | risk plan (fractional / voltarget / Kelly sizing + VaR) |
| GET | `/api/v1/symbol/{s}/stress` | 2008 / COVID / 2022 stress table |
| GET | `/api/v1/symbol/{s}/backtest` | causal dip backtest stats |
| GET | `/api/v1/live/pass?group=&symbols=` | one signal pass (dry-run by default — no dedup state written) |
| POST | `/api/v1/snapshots?symbol=&group=` | generate + persist today's report snapshot |
| GET | `/api/v1/snapshots?symbol=` | snapshot history |
| GET | `/api/v1/signals?symbol=` | persisted signal-event history |

**Persistence** — report snapshots and signal events are stored via
SQLAlchemy. Set `NEXUS_DATABASE_URL` to a Postgres DSN (the compose stack
does this automatically); without it, a local SQLite file under
`data/db/` (gitignored) is used and `/health` reports which backend is
live. Snapshots are upserted per `(symbol, group, timeframe, as_of)` and
signal events per dedup key, so repeated calls never duplicate rows. The
schema is reviewable as SQL in `database/migrations/` (`001_initial.sql`
for SQLite, `001_initial.postgres.sql` for PostgreSQL) and a demo
snapshot + signal row can be inserted idempotently with `make db-seed`
(or `python database/seed_data/seed.py`).

**Data freshness** — the report cache is keyed on the parquet file's mtime,
so running `python -m src.data.update` is picked up on the next request
(no server restart needed). API reads are local-file only (no MT5
fetches), so it never contends with a backfill.

---

## Equity Expansion (Fundamental Factors + Sentiment)

Institutional specs #8 (factor model) and #9 (news/social sentiment) for the
**equity universe** (S&P 500 D1/H4/H1). Runs as its own scanner on top of
the technical stack:

```bash
python -m src.equity.run                          # rank all equity_universe D1
python -m src.equity.run --symbols AAPL,MSFT,TSLA
python -m src.equity.run --top 10
python -m src.equity.run --fetch-news             # warm the 1-day news cache
python -m src.equity.run --symbols AAPL --detail  # full factor + news read
python -m src.equity.run --json
```

**Fundamental factor model** (`src/equity/fundamentals.py`) — three style
factors, each 0–100, composite weighted 25/25/50 (renormalised when a factor
is missing):

* **Value** — low P/E, EV/EBITDA, P/B are cheaper = better (band-mapped).
* **Quality** — high ROE, low Debt/Equity.
* **Momentum** — 1/3/6/12-month price returns + RSI (always available),
  blended 70/30 with earnings surprise + analyst revisions when provided.

Fundamentals come from a **local CSV** — `data/fundamentals/universe.csv`
(one row per symbol) or `data/fundamentals/{SYMBOL}.csv`. A template is
committed as `data/fundamentals/universe.csv.example`; copy it to
`universe.csv` and fill in vendor data. Yahoo's fundamentals endpoints now
require an authenticated crumb (auth-blocked), so the built-in Yahoo
attempt degrades gracefully and the module keeps working on price momentum
alone — Value/Quality simply read `None` (shown as `-` in the table) until
a CSV is provided.

**News & social sentiment** (`src/equity/sentiment.py`) — a financial
lexicon over Yahoo Finance headlines (public search endpoint, no auth),
cached 1 day under `data/raw/sentiment/`. Two honesty guards keep it
calibrated: a **neutral band** (ties read 0, not a coin-flip) and
**relevance weighting** — generic market stories Yahoo lists under every
ticker contribute little; confidence scales with the number of *relevant*
(symbol-specific) articles, so 10 headlines of which 2 are about the stock
read as a 0.4-confidence signal, not a saturated 1.0. Social sentiment
(StockTwits etc.) now requires API keys, so it reports gracefully
unavailable; the scoring layer is ready for a keyed provider.

The report generator adds **section 16 (Fundamental Factors)** for
**equity-class symbols only** (FX/metals/indices have no P/E, ROE etc.) and
**section 17 (News & Social Sentiment) for every symbol class** — the Yahoo
news endpoint answers FX and metals headlines too, so the rating's
Sentiment factor is no longer hard-zero on the FX/metals universe. The
report path never fires network calls — run `--fetch-news` (or any cached
pass) to warm the sentiment cache first.

---

## License

Private / Research use — see [LICENSE](LICENSE) (all rights reserved;
not for commercial redistribution). Trading involves substantial risk of
loss; this is a research system, not financial advice.