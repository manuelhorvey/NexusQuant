# NexusQuant

**Institutional-Grade Multi-Asset Quantitative Trading System**  
Focused on FX, Metals, and Commodities.

NexusQuant is a systematic research & trading platform that produces structured institutional-style analysis (regime detection, multi-timeframe technicals, confluence levels, ensemble signals, risk management, and backtesting).

---

## Features

- Multi-timeframe Market Regime Detection
- Support / Resistance + Fibonacci Confluence
- Full Moving Average Structure & Ribbon Analysis
- Momentum & Oscillator Suite (RSI, MACD, Bollinger, Divergences)
- Volume & Flow Analysis
- Pattern Recognition Engine
- Ensemble Signal Model (LightGBM / XGBoost ready)
- Trade Setup Construction (Entry / ATR Stop / Targets)
- Risk Management (Kelly, VaR, Stress Testing)
- Historical Backtesting Engine
- Clean modular architecture

---

## Project Structure

```
NexusQuant/
├── data/
│   ├── raw/                  # Original MT5 / CSV exports
│   └── processed/            # Clean Parquet files
├── notebooks/                # Research & analysis notebooks
├── src/
│   ├── data/                 # Data loading & cleaning
│   ├── features/             # Indicators, regime, levels, patterns
│   ├── analysis/             # Report generation (tables)
│   ├── backtest/             # Backtesting engine
│   └── utils/                # Helpers
├── config/                   # Settings
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
4. Backtest strategies → `src/backtest/`
5. Move to live signals + MT5 execution later

---

## License

Private / Research use.