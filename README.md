# 📈 Kronos Trading System

Multi-strategy algorithmic trading system built with Python and Alpaca Markets integration.

## Overview

Kronos is a self-hosted algorithmic trading platform that runs multiple concurrent strategies with machine learning-driven regime detection. Designed around the **Kronos Philosophy** — fully local, zero paid API dependencies, running 24/7 on a VPS.

## Strategies

| Strategy | Type | Description |
|----------|------|-------------|
| **Gap & Go** | Momentum | Catches opening gap plays with volume confirmation |
| **Mean Reversion** | Statistical | Fades extended moves using z-score thresholds |
| **Bollinger Band** | Volatility | Trades band squeezes and expansions |
| **OBV Divergence** | Volume | Detects price/volume divergence for reversals |
| **Parabolic Short** | Momentum | Shorts parabolic extensions with trailing stops |

## Architecture

```
┌─────────────────────────────────────────────┐
│              Kronos Core Engine              │
├──────────┬──────────┬──────────┬────────────┤
│ Strategy │  Regime  │ Risk     │ Execution  │
│ Manager  │ Detector │ Manager  │ Engine     │
├──────────┴──────────┴──────────┴────────────┤
│           Alpaca Markets API                 │
└─────────────────────────────────────────────┘
```

## Key Features

- **Random Forest Regime Detection** — ML model classifies current market regime (trending, mean-reverting, volatile, quiet) to select optimal strategies
- **Automated Backtesting** — Historical strategy validation with walk-forward optimization
- **Web Dashboard** — Real-time P&L, position monitoring, and strategy performance metrics
- **Risk Management** — Per-strategy position sizing, portfolio-level exposure limits, and drawdown circuit breakers
- **Self-Hosted** — Runs entirely on a Hetzner VPS with no cloud dependencies

## Tech Stack

- Python (pandas, scikit-learn, numpy)
- Alpaca Markets API
- Random Forest / ML models
- Flask web dashboard
- SQLite for trade history
- Linux / Hetzner VPS

## Status

🟢 Live — Running in production with real capital

---

*Built by [Josiah Garcia](https://www.linkedin.com/in/josiah-garcia-470890390/)*
