# NexusQuant — Trading Models & Methodology

This document describes *how* each analytical engine works — the rules,
parameters and honest limitations — and how the layers combine into a
decision. Everything is causal (no lookahead).

---

## 1. Market Regime Detection

**Files**: `src/features/regime.py`

Three complementary views on the latest bar:

| View | What it does |
|---|---|
| **Rule-based** | `ADX ≥ 25` + price vs SMA200 + 20-bar linear regression slope → `Bull Trend` / `Bear Trend` / `High Volatility` (ATR ratio > 1.8) / `Range / Chop`. Confidence from ADX magnitude. |
| **KMeans cluster** | Spec #1 allows "HMM *or* clustering". Standardized features (slope, ADX, vs-200 distance, ATR ratio) are clustered into 4 regimes, each labeled by its centroid (High-Vol first, then trend direction). Falls back to the rule labels when history is short or sklearn is absent. |
| **Multi-timeframe (D/W/M)** | The report's regime section resamples to Weekly and Monthly, recomputes indicators + regime per timeframe and shows a `D/W/M` table with a consensus row. Gated by `mtf=True` (detail views); the universe-ranking path skips it for speed. |

---

## 2. Key Levels — Support / Resistance + Confluence

**Files**: `src/features/levels.py`

- **Swings / fractals** — pivot highs/lows with a confirmation window;
  become usable only after confirmation (causal).
- **Classical pivots** — floor/cardinal pivots from the last closed bar.
- **Confluence clustering** — swing highs/lows + fib retracements that
  land near each other (within tolerance) merge into one zone with a
  1–10 **strength** score from the number of contributing factors.
- **Fibonacci** — retracements (38.2/50/61.8/78.6) and extensions
  (127.2/161.8) of the last up/down legs, each tagged and scored.
- **Anchored VWAP** — VWAP anchored at the first bar of the series.
- **Volume profile** — price-node volume buckets; high-volume nodes
  flagged (`is_high_volume`) as institutional interest zones.

The report surfaces `nearest_support` / `nearest_resistance`, the top
confluence zones, the `fib_map` table (each ratio with 1–10 confluence and
distance from close) and the high-volume nodes.

---

## 3. Moving Average Structure & Ribbon

**Files**: `src/features/indicators.py`

50/100/200 SMA **and** EMA in the report, plus `ma_ribbon_summary`:

- **Cross probability** — over the next `cross_horizon` (20) bars, the
  probability the 50/200 MA ribbons cross, estimated from the current
  gap and per-bar convergence (golden vs death).
- **Ribbon slope** — direction of the MA stack (trend momentum).
- **Ribbon width** — spread of the MAs as % of price (width + slope =
  trend-strength proxy).
- **Alignment** — signed count of ordered MA pairs.

---

## 4. Momentum & Oscillators

**Files**: `src/features/indicators.py`, `src/features/divergence.py`

RSI(14), MACD(12,26,9) histogram/signal, Bollinger %B and width, ADX ±DI
in the report. The divergence engine (`divergence_summary`) detects:

- **Regular divergences** (bullish/bearish) — price makes a new extreme,
  oscillator doesn't.
- **Hidden divergences** — higher-low/higher-low-in-momentum continuation
  signals.
- **RSI failure swings** — a classic swing-failure structure.

Only signals with **≥ 65% confidence** are reported (same threshold
discipline as pattern recognition).

---

## 5. Volume & Flow

**Files**: `src/features/indicators.py` (`volume_flow_summary`)

OBV 20-bar slope + trend label, Accumulation/Distribution line slope,
relative volume vs 20-day average, volume delta, and a normalized
**buyer-vs-seller score** (positive = accumulation). Gracefully reports
`available=False` when a series has no volume column (e.g. some index
feeds).

---

## 6. Pattern Recognition

**Files**: `src/features/patterns.py`

Peak/trough detection with a confirmation window (causal), then pattern
geometry checks: Head & Shoulders (+inverse), Double Top/Bottom, Cup &
Handle, Triangles, Flags/Pennants. Each pattern computes a breakout level
and a **statistical probability**; only patterns with **≥ 65%** confidence
are reported, with status (e.g. "forming" vs "confirmed").

---

## 7. Buy-the-Dip Confirmation

**Files**: `src/features/dip.py`

The core trade setup. An 8-factor score (0–8):

| Factor | Rule |
|---|---|
| Trend | ADX ≥ 20 |
| MA stack | SMA20 > SMA50 > SMA200 |
| Above SMA200 | close > 200-SMA |
| Pullback | price ≥ 2 ATR off the last swing high, RSI < 55, MACD hist negative/declining |
| Cooled | RSI in the 30–55 band |
| At support | near a confluence level or 0.618 fib (within 0.25 × ATR) |
| Fib zone | inside the 0.382–0.786 retracement of the last up-leg |
| Trigger | momentum turn (MACD/RSI/bar) from H4/H1 when available |

**Confirmed** requires bullish structure + score ≥ 5, and produces an
`entry_zone` (fib band), `invalidation` (below the swing low) and a
target (nearest resistance). Bear trends are never confirmed — no matter
how oversold.

---

## 8. Ensemble Signal Model (LightGBM)

**Files**: `src/model/`

- **Features** (~54): momentum, trend, volatility, returns, volume,
  regime, calendar structure (day-of-week, month-end FOMC proxy),
  interactions (vol×momentum, vol×trend, adx×slope), H4 multi-timeframe,
  cross-asset risk/gold proxies, COT percentile, macro scores, symbol
  categorical. All causal; missing sources degrade to neutral values.
- **Labels** — default `meta`: the *actual* dip trade outcome (limit
  entry at zone low, swing-low stop, resistance target) so the model
  learns the deployment objective. `1r` alternative: asymmetric
  triple-barrier (stop 1.25×ATR, target 0.75×ATR) with censored-row
  down-weighting.
- **Validation** — chronological train/test with one-horizon embargo, or
  purged walk-forward CV with early stopping on a chronological tail.
  Isotonic calibration so probabilities are true (safe for Kelly).
- **Honest numbers** — OOS AUC ~0.55–0.56 on daily FX/metals is the
  realistic ceiling; the *usable* signal is the **dip gate lift**
  (top-half of confirmed dips ≈ 46% win rate vs 38% bottom-half), which
  the live `--min-ml-prob` filter exploits.
- **Feature importance** — `importance_summary()` gives the top-N
  features by gain plus a factor-group breakdown (Trend / Momentum /
  Volatility / Multi-timeframe / Macro / …) in the report and dashboard.

The model is a **filter on the rule stack**, not a standalone signal.

---

## 9. Macro Overlay

**Files**: `src/macro/`

Three daily factors scored causally in [−2, +2]:

- **USD strength** — DXY vs SMA200 + RSI(14).
- **Risk sentiment** — VIX level bands + 20-day change.
- **Rates pressure** — US10Y (^TNX) trend + slope.

Combined into a regime (USD Bullish/Bearish, Risk-On/Off,
Easing/Tightening), then translated into a **per-symbol macro bias**
(e.g. strong dollar = headwind for EURUSD, tailwind for USDJPY; gold is
anti-dollar/anti-real-yield; crypto strongly risk-sensitive). The bias
powers a **gate**: setups are filtered when the backdrop is a strong
headwind. A **sensitivity table** (trailing 90d, 1-day lagged) reports
market beta vs S&P 500, dollar/yield/vol correlations and the symbol's
sector-ETF correlation.

---

## 10. Risk & Position Sizing

**Files**: `src/risk/`

- **Sizing** — fractional (`equity·risk/(entry−stop)`), volatility-
  targeted (size so the trade contributes a target vol, `vol·√hold`,
  capped at the risk budget), fractional Kelly (`f·(p−(1−p)/b)` with the
  ML probability).
- **Targets** — TP1/TP2/TP3 ladder (nearest confluence → next level /
  swing high → 2.618 extension), each with R:R. **Min 2.5R is enforced
  by default** via `rr_ok` / `min_rr_tp`; the live filter drops setups
  below the floor.
- **VaR** — per-trade parametric `z·ATR·qty·√hold`; portfolio VaR with
  a correlation matrix; portfolio heat = Σ risk$ / equity.
- **Limits** — `RiskManager`: max daily/weekly loss halts, max
  concurrent positions, max heat.
- **Stress** — 2008 GFC / COVID-2020 / 2022 scenarios with
  data-grounded realizations (gap-through assumption; stops assumed not
  to fill).

---

## 11. Backtester

**Files**: `src/backtest/`

Causal dip backtest: the same 8-factor signal evaluated bar-by-bar with
strict no-lookahead (swing levels usable only after their confirmation
window; pivots from the previous bar). Trade simulation: limit entry at
the zone low (3-bar validity) or market, stop at the invalidation,
target at nearest resistance or R:R fallback, time-stop after max-hold.
Metrics: win rate, profit factor, expectancy (R), Sharpe, max drawdown,
CAGR, exposure. Sizing pluggable (fractional / vol-target / Kelly).

---

## 12. Final Quant Rating

**Files**: `src/analysis/rating.py`

Blends the ML probability (when present) with a rule-based factor stack
(60/40) into a 0–100 bullish probability, then maps to the spec #14
thresholds: **Strong Buy ≥ 85 | Buy 70–84 | Neutral 50–69 | Sell 30–49 |
Strong Sell ≤ 29**. Signed factor contributions (Trend / Momentum /
Volume / Macro / Sentiment / Fundamentals) always sum to
`prob − 50`, with a residual marked `unexplained` when the model disagrees
with the factor stack.

---

## How the layers combine (a decision)

1. **Regime + levels + dip** establish *what the setup is* (8-factor
   confirmation, entry zone, invalidation).
2. **ML probability** filters/ranks confirmed setups (model trained on
   the actual trade outcome).
3. **Macro gate** blocks setups fighting a strong top-down headwind.
4. **Rating** produces the final label + factor attribution.
5. **Risk** sizes the position, builds the TP ladder (≥ 2.5R floor),
   checks VaR and stress, and the live pass alerts.
6. **Backtest** validates the edge of the identical rules historically.
