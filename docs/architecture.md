# NexusQuant — System Architecture

NexusQuant is an institutional-grade **research + signal system** for FX,
metals, commodities, indices and equities. It is a pure-Python stack (no
C++ execution engine — daily-bar research does not need one; the Python
layer IS the value). This document maps the components, the data flow and
how to run/deploy it.

---

## 1. High-level pipeline

```
                    ┌────────────────────────────────────────────┐
                    │            DATA INGESTION                  │
                    │  local parquet · MT5 bridge · Yahoo chart  │
                    └──────────────┬─────────────────────────────┘
                                   ▼
                    ┌────────────────────────────────────────────┐
                    │            FEATURE ENGINE                  │
                    │  indicators · regime · levels · dip        │
                    └──────────────┬─────────────────────────────┘
                                   ▼
        ┌──────────────────────────┼───────────────────────────┐
        ▼                          ▼                           ▼
┌───────────────┐        ┌──────────────────┐        ┌──────────────────┐
│  ANALYSIS     │        │   MODEL (ML)     │        │   MACRO (top-down)│
│  report       │        │  LightGBM prob   │        │  DXY/VIX/TNX      │
│  scanner      │        │  importance      │        │  bias + gate      │
│  rating       │        └────────┬─────────┘        └────────┬─────────┘
└───────────────┘                 │                           │
        └─────────────────────────┴──────────────┬────────────┘
                                                 ▼
                    ┌────────────────────────────────────────────┐
                    │         RISK & EXECUTION LAYER             │
                    │  sizing · targets · VaR · stress · alerts  │
                    └──────────────┬─────────────────────────────┘
                                   ▼
        ┌──────────────────────────┼───────────────────────────┐
        ▼                          ▼                           ▼
┌───────────────┐        ┌──────────────────┐        ┌──────────────────┐
│  DASHBOARD    │        │  REST API        │        │  LIVE SIGNALS    │
│  Streamlit    │        │  FastAPI + DB    │        │  Discord/Telegram│
└───────────────┘        └──────────────────┘        └──────────────────┘
```

**Everything is causal**: every indicator, regime label, dip component,
macro score and model feature uses only data available at that bar. No
lookahead anywhere in the research, backtest or live paths.

---

## 2. Module map

| Directory | Responsibility | Key entry points |
|---|---|---|
| `src/data/` | Ingestion + hygiene | `loader.py` (clean/load), `mt5.py` (Wine bridge), `yahoo.py` (no-auth OHLCV), `resolver.py` (local→MT5→Yahoo cascade), `regroup.py` (asset-class folders), `update.py` (incremental), `freshness.py` (staleness) |
| `src/features/` | Technical engine | `indicators.py` (MA/RSI/MACD/BB/ADX/ATR/volume flow/ribbon), `regime.py` (rule + KMeans cluster + D/W/M table), `levels.py` (swings, pivots, confluence, fib map, anchored VWAP, volume profile), `dip.py` (Buy-the-Dip), `patterns.py` (H&S, double top/bottom, cup&handle, triangles, flags), `divergence.py` (regular/hidden + failure swings) |
| `src/analysis/` | Output layer | `report.py` (18-section institutional report), `scanner.py` (universe ranking), `dashboard_data.py` (pure functions for the UI), `rating.py` (spec #14 quant rating + factor contributions) |
| `src/model/` | ML | `model.py` (LightGBM + isotonic calibrator + feature importance), `features.py` (causal feature matrix, labels), `run.py` (train/search/CV), `cot.py` (CFTC positioning cache) |
| `src/risk/` | Money management | `sizing.py`, `targets.py` (TP1–3 ladder, min 2.5R), `metrics.py` (VaR, heat, correlation), `limits.py` (RiskManager), `stress.py` (2008/COVID/2022), `run.py` (CLI) |
| `src/macro/` | Top-down overlay | `overlay.py` (factor scores, per-symbol bias, gate, sensitivities), `run.py` (CLI) |
| `src/live/` | Signals & alerts | `signals.py` (scan→filter→size→dedup), `alerts.py` (Discord/Telegram/file/console), `run.py` (CLI + scheduler) |
| `src/backtest/` | Validation | `signals.py` (causal dip signal), `engine.py` (trade simulation, stats), `run.py` (CLI) |
| `src/equity/` | Equity-only factors | `fundamentals.py` (Value/Quality/Momentum 0–100), `sentiment.py` (Yahoo news lexicon — now serves FX/metals too), `run.py` |
| `api/` | REST service | `app.py` (FastAPI routes), `db.py` (SQLAlchemy persistence, Postgres→SQLite fallback) |
| `dashboard.py` | Streamlit UI | five pages: Universe / Detail / Compare / Backtest / Risk |
| `config/settings.yaml` | Central configuration | data paths, feature periods, risk params, live filters, channels |

---

## 3. Data flow & freshness

**On-demand resolution cascade** (`src/data/resolver.py`) — when a symbol /
timeframe is missing, in strict order:

1. **Local** — search *every* group folder under `data/raw/` for
   `{SYMBOL}_{TF}.parquet` (any group, any layout); longest history wins.
2. **MT5 terminal** — pull from the running Wine bridge (357 symbols),
   cached into the correctly classified asset-class folder.
3. **Yahoo chart API** — no-auth fetch (stocks, ETFs, exotic indices,
   FX `=X`, crypto `-USD`), D1 (10y) / H1-H4 (1h bars, 730d cap).

Fetched files land in their classified group folder — never the
`data/raw` root — so the dashboard and future scans see them immediately.

**Freshness** — every decision runs on parquet bars. `src/data/update.py`
does incremental MT5 appends (reads last bar date, fetches only the delta,
dedups). `src/data/freshness.py` flags STALE files against
`data.max_stale_days` (D1=4, H4=2, H1=2 — absorbs weekends). The live
briefing prints "Data as of Nd ago" and warns when stale.

**Storage layout**:

```
data/raw/
├── full_fx/        # FX pairs incl. crosses & exotics
├── candidates/     # US30, US500, USTEC, BTCUSD
├── equity_universe/# S&P 500 constituents
├── equity/         # other stocks / ETFs (incl. on-demand fetches)
├── metals/  commodities/  indices/  crypto/
├── macro/          # cached VIX/TNX/sector-ETF closes
├── sentiment/      # 1-day news cache (all symbol classes)
├── cot/            # CFTC positioning percentiles
└── mt5/            # staging for bulk backfill (regrouped away)
```

---

## 4. Analysis layers

**Report** (`src/analysis/report.py`) — `generate_full_report(df, symbol,
group, data_dir, mtf=...)` returns a dict with 18 sections:

1. Market Regime (rule + KMeans cluster + D/W/M multi-timeframe table)
1b. MA Ribbon (cross probability, slope, width)
2. Moving Averages (50/100/200 SMA+EMA)
3. Momentum (RSI, MACD, Bollinger)
4. Trend Strength (ADX ±DI)
5. Volatility (ATR, BB width)
6. Simple Bias
7. Key Levels (confluence zones, nearest S/R, anchored VWAP, volume profile)
8. Ensemble Model (bullish probability + feature importance)
9. Risk & Position Sizing (3 methods + VaR + R:R floor verdict)
10. Macro Overlay (regime, per-symbol bias, gate, sensitivities)
11. Buy-the-Dip Confirmation (8-factor score, entry zone, invalidation)
12. Volume & Flow (OBV, A/D, rel vol, B/S score)
12b. Divergences (≥65% confidence)
12c. Fibonacci Confluence Map (38.2→161.8 with 1–10 strength)
13. Pattern Recognition (≥65% confidence)
14. Target Ladder (TP1–3, scaling out to 2.5R+)
15. Final Quant Rating (85/70/50/30 thresholds + factor contributions)
16. Fundamental Factors (equity only)
17. News & Social Sentiment (all symbol classes)
18. Stress Test (2008 GFC / COVID / 2022)

**Scanner** (`src/analysis/scanner.py`) — `scan_universe()` runs the full
pipeline per symbol and ranks by directional bias then ADX. The ranking
path skips the resample-heavy D/W/M table (`mtf=False`) and per-symbol
macro sensitivities to stay fast on 500-stock scans; the detail path
includes both.

---

## 5. ML model

`src/model/model.py` trains a LightGBM classifier on ~54 causal features
(momentum, trend, volatility, returns, volume, regime, 8 dip components,
calendar structure, H4 multi-timeframe, cross-asset risk/gold proxies,
COT positioning, macro scores, symbol categorical) with:

- **Chronological validation** — train/test split with a one-horizon
  embargo, or purged walk-forward CV (`--cv N`), early stopping on a
  chronological validation tail.
- **Isotonic calibration** — probabilities are true probabilities, safe
  for fractional-Kelly sizing.
- **Meta labels** — the default label is the *actual* dip trade outcome
  (limit entry, swing-low stop, resistance target) rather than a generic
  1R move.
- **Feature importance** (`importance_summary`) — top-N features by gain
  plus a factor-group breakdown surfaced in the report and dashboard.

The model is used as a **filter** on top of the rule stack (dip gate
win-rate lift 42% → 46% top-half vs 38% bottom-half), never as a black-box
signal generator.

---

## 6. Risk management

`src/risk/` — three sizing methods (fractional, volatility-targeted,
fractional Kelly with the ML probability) on the live dip setup; per-trade
and portfolio VaR95; portfolio heat + correlation-aware gates; a
`RiskManager` with daily/weekly loss halts and max-concurrent/heat limits;
2008/COVID/2022 stress tables with data-grounded realizations.

**Min R:R 2.5 is enforced by default** — `risk_plan_from_report` derives
the achievable R:R from the target ladder (TP1→TP3 scaling out), reports
`rr_ok` / `rr_nearest` / `best_rr` / `min_rr_tp`, and the live filter drops
setups below the floor.

---

## 7. Deployment

| Mode | How |
|---|---|
| Research / CLI | `./venv/bin/python -m src.analysis.scanner --group full_fx` |
| Dashboard | `streamlit run dashboard.py` (local files only, 15-min cache) |
| API (local) | `make api` → uvicorn :8000 (SQLite fallback persistence) |
| API (full stack) | `make up` → Postgres + API in docker compose |
| Scheduled signals | cron `0 6 * * 1-5` → `scripts/daily_morning.sh` (update → check → live pass, logs to `logs/daily.log`) |
| Database | Postgres via docker-compose, or SQLite under `data/db/` (gitignored); schema in `api/db.py` + `database/migrations/` |

**Secrets** — never in the repo: Discord webhook / Telegram bot+chat via
env vars or the gitignored `.env.live`; the API reads `NEXUS_DATABASE_URL`.
