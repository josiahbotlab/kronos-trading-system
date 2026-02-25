# Kronos Trading System

Dual trading system: **stock bots** (Alpaca API) and **crypto engine** (Coinbase Advanced Trade). Built on Moon Dev methodology — liquidation cascade detection, robustness-tested strategies, and a self-improving loop that tunes itself from trade history.

## Project Structure

```
trading-bot/
├── bots/                          # Stock trading bots (Alpaca)
│   ├── base_bot.py                # Abstract base — market hours, bracket orders, cooldowns, regime gating
│   ├── momentum_bot.py            # SMA(20)/EMA(9) pullback entries in uptrends
│   ├── breakout_bot.py            # 24h resistance breakout, TP 4%, SL 12%
│   ├── mean_reversion_bot.py      # RSI<30 pullbacks with SMA filter
│   ├── bb_bounce_bot.py           # Bollinger Band bounce entries
│   └── macd_bot.py                # MACD crossover signals
│
├── kronos-trading/                # Crypto trading system
│   ├── execution/
│   │   ├── live_engine.py         # Main loop — loads strategies, feeds candles, executes signals
│   │   ├── coinbase_connector.py  # Coinbase Advanced Trade JWT auth wrapper
│   │   ├── coinbase_executor.py   # Market/limit orders, paper simulation (5 bps slippage)
│   │   ├── exchange_connector.py  # Unified interface over coinbase_connector
│   │   ├── position_manager.py    # Kelly sizing, drawdown breaker, exposure limits, crash recovery
│   │   ├── risk_manager.py        # Portfolio-level risk, multi-strategy coordination, kill switch
│   │   ├── telegram_notifier.py   # Trade alerts to @jmurkedbot
│   │   └── portfolio.py           # Portfolio state tracking
│   │
│   ├── strategies/
│   │   ├── templates/
│   │   │   └── base_strategy.py   # BaseStrategy ABC — on_candle(CandleData) -> Signal
│   │   ├── momentum/              # Trend-following: cascade_p99, cascade_ride, liq_bb_combo, sma_crossover
│   │   ├── reversal/              # Mean reversion: double_decay, exhaustion_fade
│   │   └── generated/             # Auto-generated & hand-improved: parabolic_short, obv_divergence, adx_macd_momentum, etc.
│   │
│   ├── core/
│   │   ├── backtester.py          # OHLCV + liquidation replay, fee modeling, equity curves
│   │   ├── metrics.py             # Sharpe, Sortino, Calmar, PF, win rate, drawdown, CAGR
│   │   └── robustness.py          # 5-test Moon Dev suite (OOS, walk-forward, param sweep, Monte Carlo, rolling)
│   │
│   ├── collectors/
│   │   ├── price_collector.py     # Binance Futures OHLCV via CCXT -> data/prices.db
│   │   ├── liquidation_collector.py # Binance WebSocket forced liquidations -> data/liquidations.db
│   │   └── position_collector.py  # Legacy Hyperliquid (deprecated)
│   │
│   ├── monitoring/
│   │   ├── dashboard.py           # Terminal dashboard + Telegram status reports
│   │   └── incubation_tracker.py  # Strategy budget allocation tracking
│   │
│   ├── config/
│   │   ├── kronos.json            # Live config (Coinbase keys, risk params, symbols)
│   │   └── kronos.json.example    # Template — copy and fill in
│   │
│   ├── data/                      # SQLite databases (gitignored, stays on VPS)
│   │   ├── prices.db              # OHLCV candles (16 symbols, 5m/15m/1h/1d)
│   │   └── liquidations.db        # Forced liquidation events
│   │
│   └── research/                  # Strategy research notebooks and data
│
├── src/
│   ├── utils/                     # Shared utilities for stock bots
│   │   ├── order_utils.py         # Alpaca API wrapper — bracket orders, daily loss limit, position checks
│   │   ├── stock_journal.py       # SQLite trade journal for stocks
│   │   ├── regime_detector.py     # SPY-based regime: trending_up/down, ranging, volatile
│   │   ├── telegram_notifier.py   # Telegram alerts
│   │   ├── trade_journal.py       # Legacy trade logging
│   │   └── preflight_check.py    # Pre-trade safety checks
│   ├── dashboard/
│   │   └── app.py                 # Flask dashboard at localhost:5001 (account, positions, risk, scanner)
│   ├── agents/                    # Experimental agent bots (gap_and_go, regime_agent, risk_agent, scanner)
│   ├── backtesting/               # Stock backtesting scripts (breakout, mean reversion, gap & go, optimizer)
│   ├── data/                      # Market indicators, liquidation tracker
│   └── scanner/                   # Gap scanner for pre-market
│
├── scripts/
│   ├── skill_updater.py           # Analyze trades -> discover patterns -> write skills/strategy_performance.md
│   ├── daily_summary.py           # 24h trade report -> Telegram
│   ├── strategy_tournament.py     # Scan/evaluate/promote/demote strategies (7d backtest)
│   ├── migrate_csv.py             # One-time CSV -> SQLite migration
│   └── bots.sh                    # Shell launcher for stock bots
│
├── skills/
│   └── strategy_performance.md    # Auto-generated: per-strategy stats, regime matrix, learned rules
│
├── systemd/                       # Service files for VPS deployment
│   ├── stock-skill-update.*       # Skill updater timer (Mon-Fri 21:30 UTC)
│   └── stock-daily-summary.*      # Daily summary timer (Mon-Fri 21:00 UTC)
│
├── data/
│   └── trade_journal.db           # SQLite — open_positions, closed_trades, skill_updates, parameter_recommendations
│
├── models/                        # Saved ML models
├── csvs/                          # Historical data exports
├── reports/                       # Daily markdown reports
├── rbi/                           # Robustness backtesting framework
├── polymarket_mr_bot/             # Mean reversion bot for Polymarket (standalone)
├── config.py                      # Legacy HyperLiquid config (symbols, leverage, per-symbol SL/TP)
├── requirements.txt               # Root Python dependencies
└── .env                           # API keys (NEVER commit)
```

## Environment Variables

Copy `.env.example` to `.env` and fill in real values:

```bash
# Stock Trading (Alpaca) — get from https://app.alpaca.markets/
ALPACA_API_KEY=your_key
ALPACA_SECRET_KEY=your_secret
ALPACA_PAPER=true              # true = paper trading, false = live (be careful)

# Crypto Liquidation Data
MOONDEV_API_KEY=your_key       # Optional — Moon Dev liquidation feed

# Legacy HyperLiquid (unused, kept for reference)
HL_SECRET_KEY=your_private_key

# AI Strategy Validation
OPENROUTER_API_KEY=your_key    # For LLM-based strategy generation
```

For the Kronos crypto engine, copy `kronos-trading/config/kronos.json.example` to `kronos.json`:

```json
{
  "coinbase": {
    "api_key": "YOUR_COINBASE_API_KEY",
    "api_secret": "YOUR_COINBASE_API_SECRET",
    "paper": true
  },
  "telegram": {
    "bot_token": "YOUR_BOT_TOKEN",
    "chat_id": "YOUR_CHAT_ID"
  }
}
```

Telegram alerts go to `@jmurkedbot`. Both stock and crypto systems share the same bot.

## Alpaca API (Stock Trading)

Stock bots use Alpaca Trade API through `src/utils/order_utils.py`:

- **Bracket orders** with atomic TP/SL (filled or canceled together)
- **Position dedup** — won't open if already holding a symbol
- **Daily loss limit** — 3% circuit breaker stops all trading
- **Trading halts** — checks before every order
- Set `ALPACA_PAPER=true` for paper trading (default), `false` for live

Symbols: AMD, NVDA, GOOGL, META, AAPL, MSFT, AMZN, TSLA, QQQ. Configurable per bot via `--symbols`.

Trading hours: entries only 10:30 ET - 16:00 ET (skips first hour of market).

## Coinbase Advanced Trade (Crypto)

The Kronos engine connects via `kronos-trading/execution/coinbase_connector.py`:

- **JWT auth** (not API key+secret like Alpaca)
- **CCXT** library for order execution
- **Paper mode** simulates fills with 5 bps slippage
- Symbols: BTC-USD, ETH-USD, SOL-USD on 5m timeframe

Symbol format gotcha:
- Config/display: `BTC-USD`
- CCXT fetch: `BTC/USD`
- Liquidation mapping: `BTCUSDT` (Binance)

## Moon Dev Methodology

Core philosophy: **follow the money** (liquidation cascades), **simple rules**, **tight risk**, **robustness testing**.

### Strategy Types

**Momentum (trend-following):**
- `cascade_p99` / `cascade_ride` — ride liquidation cascade moves
- `liq_bb_combo` — combined liquidation + Bollinger Band breakout
- `adx_macd_momentum` — ADX threshold + MACD crossover (Grade A in tournament)

**Reversal (mean reversion):**
- `parabolic_short` — parabolic spikes above 2.5 sigma, fade the move
- `obv_divergence` — OBV trend vs price divergence
- `exhaustion_fade` — over-extended wicks on declining volume

**Breakout:**
- `consolidation_breakout` — Bollinger Band squeeze then expansion
- `vwap_mean_reversion` — VWAP breakout with volume confirmation

### Signal Convention
Strategies return `Signal` from `on_candle(CandleData)`:
- `1` = go long
- `-1` = go short
- `0` = close position
- `None` = hold / no action

### Writing a New Strategy
Subclass `BaseStrategy` in `kronos-trading/strategies/templates/base_strategy.py`:
- Implement `on_candle(candle: CandleData) -> Signal`
- Implement `default_params() -> dict`
- Place in `strategies/momentum/`, `strategies/reversal/`, or `strategies/generated/`
- The engine auto-discovers strategies from those directories

For stock bots, subclass `BaseTradingBot` in `bots/base_bot.py`:
- Set `BOT_NAME`, `STRATEGY`, `DEFAULT_SYMBOLS`, `TP_PCT`, `SL_PCT`
- Implement `check_signal(symbol, price, prices, candles) -> (bool, dict)`

## ML Regime Detection

Two regime detectors, same methodology:

- **Stocks**: `src/utils/regime_detector.py` — uses SPY hourly bars
- **Crypto**: `kronos-trading/execution/regime_detector.py` — uses BTC-USD hourly bars

Detection logic:
- SMA(20) > SMA(50) → `trending_up`
- SMA(20) < SMA(50) → `trending_down`
- ATR(14) / ATR(50) > 1.2 → `volatile`
- Otherwise → `ranging`

Results cached for 5 minutes. The live engine and stock bots gate signals based on per-strategy regime performance tracked in `skills/strategy_performance.md`. If a strategy has WR < 45% in a regime with 5+ trades, signals are blocked.

## Backtesting

### Quick Run
```bash
cd kronos-trading
python -c "
from core.backtester import Backtester
from strategies.generated.parabolic_short import ParabolicShort

bt = Backtester(symbol='BTC/USDT', timeframe='1h', start_date='2025-01-01', end_date='2025-12-31')
report = bt.run(ParabolicShort())
print(report.summary())
"
```

### Robustness Suite (Moon Dev 5-test)
A strategy must pass ALL 5 to be considered robust:

1. **Out-of-Sample (70/30 split)** — edge must survive unseen data
2. **Walk-Forward (rolling windows)** — consistent across time periods
3. **Parameter Sweep (500+ combos)** — 80%+ of param combos must be profitable
4. **Monte Carlo (100 sims)** — remove 20% of trades randomly, 100% survival rate
5. **Rolling Window (quarterly)** — all quarters must be profitable

```python
from core.robustness import RobustnessTestSuite
suite = RobustnessTestSuite(backtester, StrategyClass, params)
results = suite.run_all()
print(results.summary())
```

### Strategy Tournament
```bash
python scripts/strategy_tournament.py --scan --evaluate --review
```
Lifecycle: CANDIDATE -> TESTING -> PROMOTED / DEMOTED / STANDBY. Promotion requires WR >= 40% and PF >= 1.0 over 7-day backtest. Demotion at WR < 30%.

## Paper Trading vs Live

### Stock Bots (Alpaca)
```bash
# Paper (default)
python bots/momentum_bot.py --symbols AMD,NVDA

# Dry run (signals only, no orders)
python bots/momentum_bot.py --dry-run --symbols AMD,NVDA

# Live (real money — double check)
ALPACA_PAPER=false python bots/momentum_bot.py --symbols AMD,NVDA
```

### Kronos Crypto Engine
```bash
cd kronos-trading

# Paper trading (default when kronos.json has "paper": true)
python execution/live_engine.py --strategy liq_bb_combo --symbol BTC-USD --timeframe 5m

# Multi-strategy with capital allocation
python execution/live_engine.py --strategy liq_bb_combo,obv_divergence,parabolic_short --capital 1000

# Live trading — set "paper": false in kronos.json (use with extreme caution)
```

## Dashboard

Two dashboards:

**Stock Dashboard** (`src/dashboard/app.py`) — Flask web app at `http://localhost:5001`:
- Account overview (equity, cash, buying power, daily P&L)
- Open positions with entry/current price and unrealized P&L
- Risk status (daily loss, exposure, trading halted indicator)
- Scanner results (stocks near breakout levels)
- Recent entry signals and exit decisions
- Auto-refreshes every 30 seconds

```bash
pip install flask
python src/dashboard/app.py    # Opens at http://localhost:5001
```

**Kronos Monitor** (`kronos-trading/monitoring/dashboard.py`) — terminal-based + Telegram:
- Portfolio status, risk metrics, incubation progress
- Sends periodic Telegram alerts when configured

```bash
cd kronos-trading
python monitoring/dashboard.py
```

## Self-Improving Loop

The system learns from its own trades:

1. **Trade Journal** (`data/trade_journal.db`) — logs every entry/exit with 37 columns (regime, signal strength, candle context, slippage)
2. **Skill Updater** (`scripts/skill_updater.py`) — analyzes closed trades, discovers patterns (bad regimes, bad hours, bad days, consecutive losses, edge decay), writes `skills/strategy_performance.md`
3. **Regime Gating** — engine reads skill file, blocks signals in regimes where strategy underperforms
4. **Parameter Self-Tuning** — analyzes SL hit rate, TP capture, hold timeouts; applies adjustments via `set_param()` with safety limits (max 20% change, auto-revert on 5 consecutive losses)
5. **Daily Summary** — sends 24h trade report to Telegram

```bash
# Manual skill update
python scripts/skill_updater.py --dry-run    # Preview
python scripts/skill_updater.py              # Apply
python scripts/skill_updater.py --force      # Override 10-trade minimum
```

## Deploy / Run Commands

### Local Development
```bash
pip install -r requirements.txt
pip install -r kronos-trading/requirements.txt
cp .env.example .env  # Fill in API keys
```

### Stock Bots
```bash
python bots/momentum_bot.py --symbols AMD,NVDA
python bots/breakout_bot.py --symbols TSLA --size 500
python bots/mean_reversion_bot.py --dry-run --symbols AAPL,MSFT
```

### Kronos Crypto Engine
```bash
cd kronos-trading

# Start data collectors (run in background on VPS)
python collectors/price_collector.py          # OHLCV from Binance
python collectors/liquidation_collector.py    # WebSocket liquidations

# Run engine
python execution/live_engine.py --strategy liq_bb_combo,obv_divergence,parabolic_short --capital 1000

# Check status
python execution/live_engine.py --status
```

### VPS Deployment
```bash
ssh agent@100.113.94.124
cd ~/kronos-trading/

# Systemd services
systemctl --user start kronos-engine
systemctl --user start kronos-prices
systemctl --user start kronos-liquidations
systemctl --user status kronos-engine

# Timers (auto-configured)
systemctl --user list-timers  # Shows skill-update (6h) and daily-summary (24h)
```

### Stock Bot Timers (local or VPS)
```bash
# Install systemd timers
cp systemd/*.service systemd/*.timer ~/.config/systemd/user/
systemctl --user daemon-reload
systemctl --user enable --now stock-skill-update.timer
systemctl --user enable --now stock-daily-summary.timer
```

## Known Issues

1. **Price collector crashes on 4h timeframe** — Coinbase doesn't support 4h granularity. Remove `4h` from `DEFAULT_TIMEFRAMES` in `collectors/price_collector.py`.

2. **Systemd restart rate limit** — price collector can exhaust restart budget and stay dead. Fix: add `StartLimitIntervalSec=0` to service unit.

3. **12+ strategies produce no signals** — many auto-generated v0.1 strategies weren't properly customized from the template. Hand-improved v2.0 versions (parabolic_short, obv_divergence) perform much better.

4. **Symbol format inconsistency** — storage uses `BTC-USD`, CCXT uses `BTC/USD`, Binance liquidations use `BTCUSDT`. Conversion logic exists but can be confusing.

5. **Position collector is deprecated** — `collectors/position_collector.py` targets Hyperliquid, not Coinbase. Safe to ignore.

6. **Low stock bot win rates** — momentum (16%), mean reversion (18%), gap & go (0%). The self-improving loop is actively tuning parameters and gating bad regimes.

7. **Dashboard is terminal-only** — despite references to `localhost:5001`, there is no HTTP dashboard server. Monitoring happens via terminal output and Telegram.

## Dependencies

Root (`requirements.txt`): hyperliquid-python-sdk, pandas, numpy, ccxt, python-dotenv, schedule, termcolor, requests

Kronos (`kronos-trading/requirements.txt`): websockets, ccxt, aiosqlite, pandas, numpy
