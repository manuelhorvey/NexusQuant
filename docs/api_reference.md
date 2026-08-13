# NexusQuant — API Reference

The NexusQuant REST API serves the whole research stack (regime, levels,
dip, ML probability, macro overlay, risk, stress, backtest) over HTTP.
Built with FastAPI; OpenAPI docs auto-generated at `http://localhost:8000/docs`.

**Run locally (SQLite fallback persistence — no Docker):**

```bash
make api        # uvicorn api.app:app --reload --port 8000
```

**Full stack (Postgres via docker-compose):**

```bash
make up         # docker compose up --build
```

---

## Common behavior

- **Base URL**: `http://localhost:8000`
- **JSON only** — all responses are strict JSON (`NaN`/`Inf` → `null`,
  numpy/pandas scalars normalized).
- **On-demand data**: the scan and symbol endpoints resolve missing
  symbols automatically (local → MT5 terminal → Yahoo), so any ticker
  works whether or not it has been backfilled. `group` hints the
  asset-class folder; when omitted the resolver searches every group.
- **Caching**: symbol reports are cached server-side, keyed on the
  parquet file's mtime — a `python -m src.data.update` is picked up on
  the next request with no server restart.
- **CORS**: open (dashboard runs on a different origin in dev).

### Error format

| Status | Meaning |
|---|---|
| `404` | symbol data not found in any group/timeframe |
| `422` | validation failure (e.g. no valid rows, too-short history, bad scan) |

Both return `{"detail": "<message>"}` (FastAPI convention).

---

## Endpoints

### GET `/health`

Service + active database backend + data directory.

```json
{ "status": "ok", "service": "NexusQuant API",
  "db_backend": "postgres", "data_dir": "data/raw" }
```

### GET `/api/v1/groups`

Every discoverable `group · timeframe` dataset under `data/raw/`.

```json
[ { "group": "full_fx", "timeframe": "D1", "n": 29, "label": "full_fx · D1" },
  ... ]
```

### GET `/api/v1/scan`

Ranked universe table (all scanner columns incl. rating + pattern).

| Param | Type | Default | Notes |
|---|---|---|---|
| `group` | str | — | asset-class folder |
| `timeframe` | str | `D1` | `D1` / `H4` / `H1` |
| `symbols` | str | — | comma-separated override of discovery |
| `top` | int | — | keep only the top N rows |

```json
{ "group": "full_fx", "timeframe": "D1", "n": 2,
  "rows": [ { "rank": 1, "symbol": "EURUSD", "regime": "Range / Chop",
              "bias": "Mild Bullish", "bias_score": 1, "adx": 19.4,
              "ml_prob": 54.0, "macro_gate": "PASS", "rating": "Neutral",
              "pattern": "Pennant 65%", ... } ] }
```

### GET `/api/v1/symbol/{symbol}` — alias of `/report`

### GET `/api/v1/symbol/{symbol}/report`

Full 18-section institutional report.

| Param | Type | Default |
|---|---|---|
| `group` | str | — |
| `timeframe` | str | `D1` |
| `persist` | bool | `false` — also upsert a report snapshot |

Returns the complete report dict — top-level keys include `regime`
(rule + cluster + `mtf` D/W/M table), `ma_ribbon`, `moving_averages`,
`momentum`, `trend_strength`, `volatility`, `levels` (confluence, nearest
S/R, `anchored_vwap`, `volume_profile`), `fib_map`, `divergences`, `dip`,
`ml` (probability + `importance`), `risk` (sizing + R:R floor), `macro`
(regime/bias/gate/sensitivities), `volume_flow`, `patterns`, `targets`
(ladder), `rating`, `fundamentals` (equity), `sentiment`, `stress`,
`simple_bias`.

### GET `/api/v1/symbol/{symbol}/risk`

Risk & position sizing plan (fractional / vol-target / Kelly + VaR).

| Param | Type | Default |
|---|---|---|
| `group` / `timeframe` | str | — / `D1` |
| `equity` | float | `100000` |

```json
{ "symbol": "EURUSD", "date": "2026-08-12", "close": 1.154,
  "setup": { "entry": 1.15226, "stop": 1.14828, "target": 1.16732,
             "rr": 3.0, "rr_nearest": 1.01, "rr_ok": true,
             "min_rr": 2.5, "best_rr": 3.0, "min_rr_tp": "TP3",
             "atr_pct": 0.433, "dip_stage": "No Uptrend",
             "dip_score": 1, "ml_prob": 53.6 },
  "sizes": [ { "method": "fractional", "qty": 251256.28, "risk_usd": 1000.0,
               "risk_pct_equity": 1.0, "var_95_1bar": 2067.0, ... }, ... ],
  "inputs": { "equity": 100000, "risk_pct": 0.01, "kelly_p": 0.536, ... } }
```

### GET `/api/v1/symbol/{symbol}/stress`

2008 GFC / COVID-2020 / 2022 stress table for the current setup.

### GET `/api/v1/symbol/{symbol}/backtest`

Causal Buy-the-Dip backtest statistics.

| Param | Type | Default |
|---|---|---|
| `risk_pct` | float | `0.01` |
| `rr_fallback` | float | `2.0` |
| `max_hold` | int | `20` |
| `entry_type` | str | `limit` (`limit` / `market`) |

```json
{ "symbol": "XAUUSD", "timeframe": "D1", "params": { ... },
  "stats": { "n_trades": 22, "win_rate": 0.545, "profit_factor": 2.42,
             "sharpe": 2.14, "max_drawdown_pct": 18.3, ... },
  "n_trades": 22 }
```

### GET `/api/v1/live/pass`

Run one live signal pass (scan → filter → size → dedup).

| Param | Type | Default | Notes |
|---|---|---|---|
| `group` | str | `full_fx` | |
| `timeframe` | str | `D1` | |
| `symbols` | str | — | comma-separated watchlist |
| `dry_run` | bool | `true` | **true = read-only** (does not write the dedup state file). Set `false` to actually consume the pass. |
| `persist` | bool | `false` | also store new setups as `signal_events` |

```json
{ "date": "2026-08-12", "group": "full_fx", "timeframe": "D1",
  "scanned": 13, "candidates": 3, "new_alerts": [ { "symbol": "XAUUSD",
  "key": "XAUUSD:4360.123456-4388.654321", "text": "🎯 XAUUSD — BUY-THE-DIP ..." } ],
  "skipped_dup": 0, "macro": { "usd": "USD Neutral", "risk": "Risk-On", ... } }
```

---

## Persistence endpoints

Reports and signals are stored via SQLAlchemy (`api/db.py`). Backend:
`NEXUS_DATABASE_URL` (Postgres DSN from the compose stack) or a local
SQLite file under `data/db/` (gitignored). Schema is created automatically
(`create_all`) and available as SQL migrations in `database/migrations/`
(`001_initial.sql` — SQLite, `001_initial.postgres.sql` — PostgreSQL).

### GET `/api/v1/snapshots?symbol=&group=&limit=`

List persisted report snapshots for a symbol (newest first).

### POST `/api/v1/snapshots?symbol=&group=&timeframe=`

Generate + persist today's report snapshot.

```json
{ "status": "stored", "symbol": "XAUUSD", "group": "full_fx",
  "timeframe": "D1", "as_of": "2026-08-12" }
```

### GET `/api/v1/signals?symbol=&limit=`

Signal-event history (deduped by key, newest first).

---

## Quick start examples

```bash
# health + discovery
curl -s localhost:8000/health
curl -s localhost:8000/api/v1/groups

# scan a watchlist
curl -s "localhost:8000/api/v1/scan?symbols=EURUSD,GBPUSD,XAUUSD&group=full_fx"

# full institutional report + risk plan + stress + backtest
curl -s "localhost:8000/api/v1/symbol/XAUUSD/report?group=full_fx"
curl -s "localhost:8000/api/v1/symbol/XAUUSD/risk?equity=250000"
curl -s "localhost:8000/api/v1/symbol/XAUUSD/stress"
curl -s "localhost:8000/api/v1/symbol/XAUUSD/backtest"

# one live signal pass (read-only by default)
curl -s "localhost:8000/api/v1/live/pass?symbols=EURUSD,GBPUSD,XAUUSD"

# persist a snapshot, then read it back
curl -s -X POST "localhost:8000/api/v1/snapshots?symbol=XAUUSD&group=full_fx"
curl -s "localhost:8000/api/v1/snapshots?symbol=XAUUSD"
```

---

## Testing

```bash
make test        # ./venv/bin/python -m unittest tests.test_api -v
```

See `tests/test_api.py` for endpoint-level coverage.
