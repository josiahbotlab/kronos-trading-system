"""
Smart Trading Bot

Master bot that automatically selects the best strategy based on:
1. Per-symbol backtested performance (SYMBOL_STRATEGY_OVERRIDE)
2. Market regime detection (fallback for unknown symbols)

Strategy Selection:
- TSLA → Always use 24h Breakout (+15.11% backtested)
- AMD → Prefer Gap and Go (+8.89% backtested)
- NVDA, SPY → Skip/Hold if trending (can't beat buy & hold)
- Others → Use regime detector (RANGING→Breakout, TREND_DOWN→Mean Reversion)

Integrates all agents:
- Regime Detector: Determines market condition per symbol
- Risk Agent: Position sizing and daily loss limits
- Entry Agent: Volume/trend/RSI confirmation
- Exit Agent: Trailing stops and partial profits

Usage:
    python src/agents/smart_bot.py              # Live trading
    python src/agents/smart_bot.py --dry-run    # Test mode
    python src/agents/smart_bot.py --mock       # Mock data
    python src/agents/smart_bot.py --use-scanner --duration 60
"""

import argparse
import os
import random
import sys
import time
from datetime import datetime, timedelta
from collections import defaultdict

import numpy as np
import schedule
from termcolor import cprint
from dotenv import load_dotenv

sys.path.append('/Users/josiahgarcia/trading-bot')
load_dotenv()

# Import all agents
from src.agents.risk_agent import check_risk_verbose, update_position, close_position, get_risk_status
from src.agents.exit_agent import manage_exit_verbose, reset_position, get_position_status
from src.agents.entry_agent import check_entry
from src.agents.scanner import get_breakout_candidates

# Trade journal
from src.utils.trade_journal import log_trade

# Order execution utilities
from src.utils.order_utils import (
    check_existing_position,
    cancel_symbol_orders,
    place_bracket_order,
    round_to_valid_qty,
    check_daily_loss_limit,
    reset_daily_tracker,
    update_daily_pnl,
    get_daily_pnl,
    is_trading_halted,
    execute_entry as execute_order_entry,
    execute_exit as execute_order_exit,
    get_account_info,
    get_all_positions,
)

# Regime detection - prefer ML regime agent when available
try:
    from src.agents.regime_agent import RegimeAgent, get_ml_regime
    ML_REGIME_AVAILABLE = True
except ImportError:
    ML_REGIME_AVAILABLE = False
    cprint("Warning: ML regime_agent not available", "yellow")

try:
    from src.agents.regime_detector import detect_regime_verbose, RegimeType, STRATEGY_MAP
    REGIME_DETECTOR_AVAILABLE = True
except ImportError:
    REGIME_DETECTOR_AVAILABLE = False
    cprint("Warning: regime_detector not available", "yellow")

# ta library for indicators
try:
    from ta.trend import SMAIndicator, EMAIndicator, MACD as TALibMACD
    from ta.momentum import RSIIndicator
    from ta.volatility import BollingerBands
    import pandas as pd
    TA_LIB_AVAILABLE = True
except ImportError:
    TA_LIB_AVAILABLE = False
    import pandas as pd
    cprint("Warning: ta library not available, using manual indicators", "yellow")

# Alpaca imports
try:
    import alpaca_trade_api as tradeapi
    ALPACA_AVAILABLE = True
except ImportError:
    ALPACA_AVAILABLE = False
    cprint("Warning: alpaca-trade-api not installed", "yellow")


# Configuration
ALPACA_API_KEY = os.getenv("ALPACA_API_KEY")
ALPACA_SECRET_KEY = os.getenv("ALPACA_SECRET_KEY")
ALPACA_PAPER = os.getenv("ALPACA_PAPER", "true").lower() == "true"

# Trading parameters
SYMBOLS = ['TSLA', 'AMD', 'NVDA', 'QQQ', 'AAPL']
ORDER_USD_SIZE = 500

# Breakout strategy params (optimized)
BREAKOUT_TP_PCT = 6.0
BREAKOUT_SL_PCT = 2.0
LOOKBACK_HOURS = 24

# Mean reversion params
RSI_PERIOD = 14
RSI_OVERSOLD = 30
RSI_EXIT = 50
SMA_PERIOD = 20
SMA_DEVIATION = 0.02
MR_SL_PCT = 3.0

# Entry confirmation
MIN_ENTRY_CONFIDENCE = 60

# Regime refresh interval (minutes)
REGIME_REFRESH_MINUTES = 60

# Scanner config
SCANNER_REFRESH_MINUTES = 30
SCANNER_TOP_N = 10

# ─────────────────────────────────────────────────────────────
# PER-SYMBOL STRATEGY OVERRIDES (based on 1-year backtests)
# ─────────────────────────────────────────────────────────────
# These override the regime detector when a symbol has a proven
# backtested strategy that works regardless of current regime.
#
# Backtest Results Summary:
# - TSLA: 24h Breakout +15.11% (best), Gap&Go -9.89%
# - AMD: Gap and Go +8.89% (best), Breakout N/A
# - NVDA: Buy & Hold +53.79% (best), all strategies lost
# - SPY: Buy & Hold +17.87% (best), all strategies lost
# ─────────────────────────────────────────────────────────────

SYMBOL_STRATEGY_OVERRIDE = {
    'TSLA': 'BREAKOUT',      # +15.11% backtested - always use breakout
    'AMD': 'MACD',           # +58% backtested from strategy generator - MACD bullish cross
    'NVDA': 'MOMENTUM',      # Changed from HOLD - use momentum pullbacks in uptrends
    'QQQ': 'MOMENTUM',       # Changed from HOLD - use momentum pullbacks in uptrends
    'META': 'BB_BOUNCE',     # BB Lower Bounce strategy
    'AAPL': 'MOMENTUM',      # Big cap that trends well - momentum pullbacks
    'MSFT': 'MOMENTUM',      # Big cap that trends well - momentum pullbacks
    'GOOGL': 'MOMENTUM',     # Big cap that trends well - momentum pullbacks
    'AMZN': 'MOMENTUM',      # Big cap that trends well - momentum pullbacks
}

# Gap and Go strategy params
GAP_MIN_PCT = 1.5           # Minimum gap up percentage
GAP_VOLUME_MULT = 1.5       # Volume must be 1.5x average
GAP_TP_PCT = 3.0            # 3% take profit
GAP_SL_PCT = 2.0            # 2% stop loss

# MACD Bullish strategy params (from strategy generator - +58% on AMD)
MACD_TP_PCT = 10.0          # 10% take profit
MACD_SL_PCT = 7.0           # 7% stop loss
MACD_FAST = 12
MACD_SLOW = 26
MACD_SIGNAL = 9

# BB Lower Bounce strategy params (from strategy generator - +42% on AMD)
BB_BOUNCE_TP_PCT = 8.0      # 8% take profit
BB_BOUNCE_SL_PCT = 5.0      # 5% stop loss
BB_PERIOD = 20
BB_STD = 2.0

# MOMENTUM strategy params (for TREND_UP / bull markets)
# Trades WITH the trend using pullback entries
MOMENTUM_TP_PCT = 6.0       # 6% take profit
MOMENTUM_SL_PCT = 3.0       # 3% stop loss (2:1 R/R)
MOMENTUM_SMA_PERIOD = 20    # Price must be above 20 SMA
MOMENTUM_EMA_PERIOD = 9     # Entry on pullback to 9 EMA
MOMENTUM_RSI_MIN = 40       # RSI must be above 40 (not oversold) - was 50
MOMENTUM_RSI_MAX = 80       # RSI must be below 80 (not extremely overbought) - was 70
MOMENTUM_VOLUME_MULT = 0.0  # Volume check DISABLED - was blocking signals on IEX feed
MOMENTUM_EMA_TOLERANCE = 0.03  # 3% tolerance for "near" EMA - was 1.5%

# ML Regime confidence threshold
ML_REGIME_CONFIDENCE_THRESHOLD = 60.0  # Use ML regime when confidence > 60%

# Entry time restriction (avoid morning volatility — data shows 12% WR before 10:30)
# Format: "HH:MM" in ET. Set to None to disable.
EARLIEST_ENTRY_ET = "10:30"

# Strategy → (TP%, SL%) lookup for synced positions and validation
STRATEGY_TP_SL = {
    'BREAKOUT':   (BREAKOUT_TP_PCT, BREAKOUT_SL_PCT),
    'MEAN_REV':   (5.0, MR_SL_PCT),
    'GAP_GO':     (GAP_TP_PCT, GAP_SL_PCT),
    'MACD':       (MACD_TP_PCT, MACD_SL_PCT),
    'BB_BOUNCE':  (BB_BOUNCE_TP_PCT, BB_BOUNCE_SL_PCT),
    'MOMENTUM':   (MOMENTUM_TP_PCT, MOMENTUM_SL_PCT),
}
SYNCED_DEFAULT_TP_PCT = 6.0
SYNCED_DEFAULT_SL_PCT = 3.0


def calculate_rsi(prices, period=RSI_PERIOD):
    """Calculate RSI indicator."""
    if len(prices) < period + 1:
        return 50

    deltas = np.diff(prices)
    gains = np.where(deltas > 0, deltas, 0)
    losses = np.where(deltas < 0, -deltas, 0)

    avg_gain = np.mean(gains[-period:])
    avg_loss = np.mean(losses[-period:])

    if avg_loss == 0:
        return 100

    rs = avg_gain / avg_loss
    return 100 - (100 / (1 + rs))


def calculate_sma(prices, period=SMA_PERIOD):
    """Calculate Simple Moving Average."""
    if len(prices) < period:
        return np.mean(prices) if len(prices) > 0 else 0
    return np.mean(prices[-period:])


class SmartBot:
    """
    Intelligent trading bot that adapts strategy based on market regime.

    Moon Dev's philosophy: "Trade what the market gives you."
    """

    def __init__(self, dry_run=False, mock_mode=False, use_scanner=False, verbose=False):
        self.dry_run = dry_run
        self.mock_mode = mock_mode
        self.use_scanner = use_scanner
        self.verbose = verbose
        self.api = None

        # Initialize Alpaca API
        if not mock_mode:
            if not ALPACA_AVAILABLE:
                raise RuntimeError("alpaca-trade-api package not installed")
            if not ALPACA_API_KEY or not ALPACA_SECRET_KEY:
                raise RuntimeError("ALPACA_API_KEY and ALPACA_SECRET_KEY required in .env")

            base_url = "https://paper-api.alpaca.markets" if ALPACA_PAPER else "https://api.alpaca.markets"
            self.api = tradeapi.REST(ALPACA_API_KEY, ALPACA_SECRET_KEY, base_url, api_version='v2')

        # Track positions and state
        self.active_trades = {}
        self.cooldowns = {}
        self.cooldown_minutes = 60
        self.cycle_count = 0

        # Regime cache
        self.symbol_regimes = {}
        self.last_regime_check = {}

        # Scanner state
        self.current_symbols = SYMBOLS.copy()
        self.last_scan_time = None

        # Mock data
        self.mock_prices = {}
        self.mock_candles = {}
        self.mock_price_history = {}

        # Stats tracking
        self.strategy_stats = {
            'breakout': {'trades': 0, 'wins': 0},
            'mean_reversion': {'trades': 0, 'wins': 0},
            'gap_and_go': {'trades': 0, 'wins': 0},
            'macd': {'trades': 0, 'wins': 0},
            'bb_bounce': {'trades': 0, 'wins': 0},
            'momentum': {'trades': 0, 'wins': 0},
            'skipped': 0
        }

        # ML Regime Agent (lazy init)
        self._regime_agent = None

        # Gap and Go tracking
        self.prev_closes = {}  # Track previous day closes for gap calculation
        self.todays_gaps = {}  # Symbols that gapped today

        if self.dry_run:
            cprint("*** DRY RUN MODE - No real trades will be executed ***", "yellow")
        if self.mock_mode:
            cprint("*** MOCK MODE - Using simulated price data ***", "yellow")
            self._init_mock_data()
        else:
            account = self.api.get_account()
            cprint(f"Connected to Alpaca ({'PAPER' if ALPACA_PAPER else 'LIVE'})", "green")
            cprint(f"Equity: ${float(account.equity):,.2f}", "green")

            # Initialize daily loss tracker
            reset_daily_tracker()

            # Sync with existing positions
            self._sync_existing_positions()

        self._print_banner()

        if self.use_scanner:
            cprint("*** SCANNER MODE - Dynamic symbol selection enabled ***", "yellow")
            self._refresh_symbols()
        else:
            cprint(f"Trading symbols: {self.current_symbols}", "cyan")

    def _sync_existing_positions(self):
        """Sync bot state with existing Alpaca positions and attach protective orders."""
        try:
            positions = get_all_positions()

            if positions:
                cprint(f"\n  Syncing {len(positions)} existing positions:", "yellow")

                for pos in positions:
                    symbol = pos['symbol']

                    if symbol not in self.active_trades:
                        entry_price = pos['avg_entry_price']
                        qty = pos['qty']
                        direction = 'LONG' if pos['side'] == 'long' else 'SHORT'

                        # Look up strategy from overrides (instead of generic 'SYNCED')
                        strategy = SYMBOL_STRATEGY_OVERRIDE.get(symbol, None)
                        tp_pct, sl_pct = STRATEGY_TP_SL.get(
                            strategy, (SYNCED_DEFAULT_TP_PCT, SYNCED_DEFAULT_SL_PCT)
                        )
                        if strategy is None:
                            strategy = 'SYNCED'

                        stop_loss = entry_price * (1 - sl_pct / 100)
                        take_profit = entry_price * (1 + tp_pct / 100)

                        self.active_trades[symbol] = {
                            'entry': entry_price,
                            'entry_time': datetime.now(),
                            'stop_loss': stop_loss,
                            'take_profit': take_profit,
                            'direction': direction,
                            'shares': qty,
                            'strategy': strategy,
                            'is_synced': True,
                        }

                        color = "green" if pos['unrealized_pnl'] >= 0 else "red"
                        cprint(
                            f"    {symbol}: {qty:.2f} shares @ ${entry_price:.2f} "
                            f"(P&L: ${pos['unrealized_pnl']:+.2f}) → {strategy} "
                            f"SL: ${stop_loss:.2f} / TP: ${take_profit:.2f}",
                            color
                        )

                        # Place protective stop order via Alpaca (broker-side protection)
                        if not self.dry_run:
                            try:
                                cancel_symbol_orders(symbol)
                                side = 'sell' if direction == 'LONG' else 'buy'
                                stop_qty = int(qty) if int(qty) > 0 else qty
                                self.api.submit_order(
                                    symbol=symbol,
                                    qty=stop_qty,
                                    side=side,
                                    type='stop',
                                    stop_price=round(stop_loss, 2),
                                    time_in_force='gtc',
                                )
                                cprint(f"      Protective stop placed @ ${stop_loss:.2f}", "green")
                            except Exception as e:
                                cprint(f"      Could not place protective stop: {e}", "red")
            else:
                cprint("  No existing positions to sync", "white")

        except Exception as e:
            cprint(f"  Error syncing positions: {e}", "red")

    def _print_banner(self):
        """Print startup banner."""
        cprint("\n" + "=" * 60, "cyan")
        cprint("  SMART TRADING BOT v2.0", "cyan", attrs=['bold'])
        cprint("  Backtested Strategy Selection", "cyan")
        cprint("=" * 60, "cyan")

        cprint("\n  PER-SYMBOL OVERRIDES (backtested):", "yellow")
        cprint("  ─────────────────────────────────────", "white")
        for symbol, strategy in SYMBOL_STRATEGY_OVERRIDE.items():
            if strategy == 'BREAKOUT':
                cprint(f"  {symbol:<6} → 24h Breakout (TP:6%, SL:2%)", "magenta")
            elif strategy == 'GAP_AND_GO':
                cprint(f"  {symbol:<6} → Gap & Go (TP:3%, SL:2%)", "yellow")
            elif strategy == 'MACD':
                cprint(f"  {symbol:<6} → MACD Bullish (TP:10%, SL:7%) +58% backtested", "cyan")
            elif strategy == 'BB_BOUNCE':
                cprint(f"  {symbol:<6} → BB Lower Bounce (TP:8%, SL:5%) +42% backtested", "blue")
            elif strategy == 'MOMENTUM':
                cprint(f"  {symbol:<6} → Momentum Pullback (TP:6%, SL:3%) trend-following", "green")
            elif strategy == 'HOLD':
                cprint(f"  {symbol:<6} → Skip/Hold (buy & hold wins)", "white")

        cprint("\n  REGIME FALLBACK (other symbols):", "white")
        cprint("  ─────────────────────────────────────", "white")
        cprint("  RANGING/VOLATILE → Breakout", "white")
        cprint("  TREND_DOWN       → Mean Reversion", "white")
        cprint("  TREND_UP         → Momentum Pullback (NEW!)", "green")

        cprint("\n  RISK CONTROLS:", "white")
        cprint(f"  Order Size: ${ORDER_USD_SIZE} | Min Confidence: {MIN_ENTRY_CONFIDENCE}%", "white")
        risk_status = get_risk_status()
        cprint(f"  Daily Loss Limit: ${-risk_status['daily_loss_limit']:,.0f}", "white")
        cprint("=" * 60 + "\n", "cyan")

    def _init_mock_data(self):
        """Initialize mock price data."""
        base_prices = {
            'SPY': 590.0, 'QQQ': 520.0, 'TSLA': 410.0,
            'AMD': 125.0, 'NVDA': 140.0, 'AAPL': 240.0,
            'INTC': 25.0, 'IWM': 220.0, 'DIA': 430.0,
            'MSFT': 450.0, 'GOOGL': 200.0, 'AMZN': 230.0,
            'META': 630.0,
        }

        # Check which symbols have MOMENTUM strategy
        momentum_symbols = [s for s, strat in SYMBOL_STRATEGY_OVERRIDE.items() if strat == 'MOMENTUM']

        for symbol in self.current_symbols:
            base = base_prices.get(symbol, 100 + random.uniform(0, 200))

            # For MOMENTUM symbols, create uptrend data
            is_momentum = symbol in momentum_symbols
            self.mock_candles[symbol] = self._generate_mock_candles(base, 50, uptrend=is_momentum)
            self.mock_price_history[symbol] = [c['close'] for c in self.mock_candles[symbol]]

            # Set current price to last candle close for consistency
            self.mock_prices[symbol] = self.mock_candles[symbol][-1]['close']

            # Assign regime based on strategy type for better testing
            if is_momentum:
                self.symbol_regimes[symbol] = 'TREND_UP'
            else:
                self.symbol_regimes[symbol] = random.choice(['RANGING', 'TREND_DOWN', 'VOLATILE', 'TREND_UP'])
            self.last_regime_check[symbol] = datetime.now()

    def _generate_mock_candles(self, base_price, num_candles, uptrend=False):
        """
        Generate mock OHLCV candles with realistic trend + pullback patterns.

        Args:
            base_price: The target price level
            num_candles: Number of candles to generate
            uptrend: If True, creates clear uptrend for MOMENTUM strategy testing
        """
        candles = []

        if uptrend:
            # UPTREND mode: Start 10% below base, trend up to base, then pullback to EMA
            price = base_price * 0.90
            trend_strength = 0.004  # 0.4% avg upward bias per candle (stronger uptrend)
            pullback_probability = 0.20  # 20% chance of pullback
        else:
            # Normal mode: random walk with slight downtrend for mean reversion testing
            price = base_price * random.uniform(0.95, 1.05)
            trend_strength = -0.001  # Slight downward bias
            pullback_probability = 0.30

        for i in range(num_candles):
            volatility = base_price * 0.008  # 0.8% volatility

            # Decide if this is a pullback candle or trend continuation
            if random.random() < pullback_probability:
                # Pullback candle
                change = random.uniform(-volatility * 1.5, -volatility * 0.3)
            else:
                # Trend continuation
                change = random.uniform(-volatility * 0.5, volatility * 1.5) + (base_price * trend_strength)

            open_price = price
            close = open_price + change
            high = max(open_price, close) + random.uniform(0, volatility * 0.5)
            low = min(open_price, close) - random.uniform(0, volatility * 0.5)

            candles.append({
                'open': open_price,
                'high': high,
                'low': low,
                'close': close,
                'volume': random.uniform(500000, 1500000)  # Higher volume
            })
            price = close

        if uptrend and len(candles) > 10:
            # For MOMENTUM: Ensure last candle is a pullback to 9 EMA
            # Calculate 9 EMA
            closes = [c['close'] for c in candles]
            ema = closes[0]
            for p in closes:
                ema = p * 0.2 + ema * 0.8  # 9 EMA multiplier = 2/(9+1) = 0.2

            # Calculate 20 SMA
            sma = np.mean(closes[-20:]) if len(closes) >= 20 else np.mean(closes)

            # Ensure price is above SMA (uptrend) but near EMA (pullback entry)
            # Set last close to be: above SMA, within 1% of EMA
            target_price = max(sma * 1.02, ema * random.uniform(0.995, 1.010))
            candles[-1]['close'] = target_price
            candles[-1]['low'] = min(candles[-1]['low'], target_price * 0.998)
            candles[-1]['high'] = max(candles[-1]['high'], target_price * 1.002)

        return candles

    def _refresh_symbols(self):
        """Refresh symbol list from scanner."""
        if not self.use_scanner:
            return

        now = datetime.now()
        if self.last_scan_time:
            elapsed = (now - self.last_scan_time).total_seconds() / 60
            if elapsed < SCANNER_REFRESH_MINUTES:
                return

        cprint(f"\n--- Refreshing symbols from scanner ---", "yellow")

        try:
            candidates = get_breakout_candidates(
                top_n=SCANNER_TOP_N,
                threshold_pct=2.0,
                mock_mode=self.mock_mode
            )

            if candidates:
                new_symbols = [c['symbol'] for c in candidates]

                # ALWAYS keep override symbols (these have backtested strategies)
                for symbol in SYMBOL_STRATEGY_OVERRIDE:
                    if symbol not in new_symbols:
                        new_symbols.append(symbol)

                # Keep symbols with active positions
                for symbol in self.active_trades.keys():
                    if symbol not in new_symbols:
                        new_symbols.append(symbol)

                self.current_symbols = new_symbols
                self.last_scan_time = now

                if self.mock_mode:
                    for symbol in new_symbols:
                        if symbol not in self.mock_prices:
                            base = 100 + random.uniform(0, 300)
                            self.mock_prices[symbol] = base
                            self.mock_candles[symbol] = self._generate_mock_candles(base, 50)
                            self.mock_price_history[symbol] = [c['close'] for c in self.mock_candles[symbol]]
                            self.symbol_regimes[symbol] = random.choice(['RANGING', 'TREND_DOWN', 'VOLATILE'])

                cprint(f"Scanner found {len(candidates)} candidates", "green")
        except Exception as e:
            cprint(f"Scanner error: {e}", "red")

    def get_regime(self, symbol):
        """
        Get current regime for symbol (with caching).

        Priority:
        1. ML Regime Agent if available and confidence > 60%
        2. Rule-based regime detector (fallback)
        3. Random (mock mode) or RANGING (default)
        """
        now = datetime.now()

        # Check cache
        if symbol in self.last_regime_check:
            elapsed = (now - self.last_regime_check[symbol]).total_seconds() / 60
            if elapsed < REGIME_REFRESH_MINUTES and symbol in self.symbol_regimes:
                return self.symbol_regimes[symbol]

        # Mock mode - return cached or random regime
        if self.mock_mode:
            if symbol not in self.symbol_regimes:
                # Use more varied regimes for testing all strategies
                self.symbol_regimes[symbol] = random.choice(['RANGING', 'TREND_DOWN', 'VOLATILE', 'TREND_UP', 'BULL', 'BEAR'])
            self.last_regime_check[symbol] = now
            return self.symbol_regimes[symbol]

        # Try ML Regime Agent first (better accuracy)
        try:
            if ML_REGIME_AVAILABLE:
                # Lazy init regime agent
                if self._regime_agent is None:
                    self._regime_agent = RegimeAgent()

                ml_regime, ml_confidence = self._regime_agent.get_ml_regime(symbol)

                if ml_confidence >= ML_REGIME_CONFIDENCE_THRESHOLD:
                    # Map ML regimes to existing regime types
                    regime_map = {
                        'BULL': 'TREND_UP',
                        'BEAR': 'TREND_DOWN',
                        'RANGE': 'RANGING',
                        'HIGH_VOL': 'VOLATILE'
                    }
                    regime = regime_map.get(ml_regime, ml_regime)
                    self.symbol_regimes[symbol] = regime
                    self.last_regime_check[symbol] = now
                    cprint(f"  [ML Regime] {symbol}: {ml_regime} ({ml_confidence:.1f}% confidence)", "cyan")
                    return regime
        except Exception as e:
            cprint(f"  ML Regime detection failed for {symbol}: {e}", "yellow")

        # Fallback to rule-based regime detection
        try:
            if REGIME_DETECTOR_AVAILABLE:
                analysis = detect_regime_verbose(symbol)
                regime = analysis['regime']
                self.symbol_regimes[symbol] = regime
                self.last_regime_check[symbol] = now
                return regime
        except Exception as e:
            cprint(f"  Rule-based regime detection failed for {symbol}: {e}", "red")

        # Default to RANGING if all detection fails
        return self.symbol_regimes.get(symbol, 'RANGING')

    def fetch_candles(self, symbol, hours=50):
        """Fetch hourly candles."""
        if self.mock_mode:
            return self.mock_candles.get(symbol, [])

        try:
            end = datetime.now()
            start = end - timedelta(hours=hours + 8)

            bars = self.api.get_bars(
                symbol,
                tradeapi.TimeFrame.Hour,
                start=start.strftime('%Y-%m-%d'),
                end=end.strftime('%Y-%m-%d'),
                limit=hours,
                feed='iex'
            ).df

            if bars.empty:
                return []

            return [
                {
                    'open': float(row['open']),
                    'high': float(row['high']),
                    'low': float(row['low']),
                    'close': float(row['close']),
                    'volume': float(row['volume'])
                }
                for _, row in bars.iterrows()
            ]
        except Exception as e:
            cprint(f"Error fetching candles for {symbol}: {e}", "red")
            return []

    def get_current_price(self, symbol):
        """Get current price."""
        if self.mock_mode:
            if symbol in self.mock_prices:
                base = self.mock_prices[symbol]
                movement = base * random.uniform(-0.003, 0.003)
                self.mock_prices[symbol] = base + movement
                return self.mock_prices[symbol]
            return None

        try:
            trade = self.api.get_latest_trade(symbol)
            return float(trade.price)
        except Exception as e:
            return None

    def get_price_history(self, symbol):
        """Get price history for indicators."""
        if self.mock_mode:
            return self.mock_price_history.get(symbol, [])

        candles = self.fetch_candles(symbol, hours=50)
        return [c['close'] for c in candles]

    def is_on_cooldown(self, symbol):
        """Check if symbol is on cooldown."""
        if symbol not in self.cooldowns:
            return False
        if datetime.now() > self.cooldowns[symbol]:
            del self.cooldowns[symbol]
            return False
        return True

    def set_cooldown(self, symbol):
        """Set cooldown after trade."""
        self.cooldowns[symbol] = datetime.now() + timedelta(minutes=self.cooldown_minutes)

    # ─────────────────────────────────────────────────────────────
    # BREAKOUT STRATEGY
    # ─────────────────────────────────────────────────────────────

    def check_breakout_signal(self, symbol, current_price, candles):
        """Check for breakout entry signal."""
        if len(candles) < LOOKBACK_HOURS:
            return None, None, None

        recent = candles[-LOOKBACK_HOURS:]
        high_24h = max(c['high'] for c in recent)
        low_24h = min(c['low'] for c in recent)

        buffer = high_24h * 0.001

        if current_price > high_24h + buffer:
            return 'LONG', high_24h, low_24h
        elif current_price < low_24h - buffer:
            return 'SHORT', high_24h, low_24h

        return None, high_24h, low_24h

    def enter_breakout(self, symbol, direction, entry_price, candles):
        """Enter a breakout trade."""
        # Calculate levels
        stop_distance = entry_price * (BREAKOUT_SL_PCT / 100)
        profit_distance = entry_price * (BREAKOUT_TP_PCT / 100)

        if direction == 'LONG':
            stop_loss = entry_price - stop_distance
            take_profit = entry_price + profit_distance
        else:
            stop_loss = entry_price + stop_distance
            take_profit = entry_price - profit_distance

        return self._execute_entry(
            symbol, direction, entry_price, stop_loss, take_profit, 'BREAKOUT'
        )

    # ─────────────────────────────────────────────────────────────
    # MEAN REVERSION STRATEGY
    # ─────────────────────────────────────────────────────────────

    def check_mean_reversion_signal(self, symbol, current_price, prices):
        """Check for mean reversion entry signal."""
        if len(prices) < max(RSI_PERIOD + 1, SMA_PERIOD):
            return False, None, None

        rsi = calculate_rsi(prices)
        sma = calculate_sma(prices)
        sma_dev = (sma - current_price) / sma if sma > 0 else 0

        # Entry: RSI < 30 AND price > 2% below SMA
        if rsi < RSI_OVERSOLD and sma_dev >= SMA_DEVIATION:
            return True, rsi, sma

        return False, rsi, sma

    def check_mean_reversion_exit(self, symbol, current_price, prices, entry_price):
        """Check for mean reversion exit signal."""
        if len(prices) < max(RSI_PERIOD + 1, SMA_PERIOD):
            return False, None

        rsi = calculate_rsi(prices)
        sma = calculate_sma(prices)
        stop_price = entry_price * (1 - MR_SL_PCT / 100)

        if current_price <= stop_price:
            return True, f"Stop loss hit (${current_price:.2f} <= ${stop_price:.2f})"
        if rsi > RSI_EXIT:
            return True, f"RSI exit ({rsi:.1f} > {RSI_EXIT})"
        if current_price >= sma:
            return True, f"Mean reverted to SMA (${sma:.2f})"

        return False, None

    def enter_mean_reversion(self, symbol, entry_price, rsi, sma):
        """Enter a mean reversion trade."""
        stop_loss = entry_price * (1 - MR_SL_PCT / 100)
        take_profit = sma  # Target is mean reversion to SMA

        return self._execute_entry(
            symbol, 'LONG', entry_price, stop_loss, take_profit, 'MEAN_REV'
        )

    # ─────────────────────────────────────────────────────────────
    # GAP AND GO STRATEGY
    # ─────────────────────────────────────────────────────────────

    def get_previous_close(self, symbol):
        """Get previous day's close for gap calculation."""
        if self.mock_mode:
            # In mock mode, simulate a previous close
            if symbol not in self.prev_closes:
                current = self.mock_prices.get(symbol, 100)
                # Randomly set prev close to create gap opportunities
                gap = random.uniform(-0.03, 0.05)  # -3% to +5%
                self.prev_closes[symbol] = current / (1 + gap)
            return self.prev_closes[symbol]

        try:
            bars = self.api.get_bars(
                symbol,
                tradeapi.TimeFrame.Day,
                limit=2,
                feed='iex'
            ).df

            if len(bars) >= 2:
                return float(bars['close'].iloc[-2])
            return None
        except:
            return None

    def get_volume_ratio(self, symbol, candles):
        """Get current volume vs average volume ratio."""
        if not candles or len(candles) < 20:
            return 1.0

        volumes = [c['volume'] for c in candles]
        current_vol = volumes[-1] if volumes else 0
        avg_vol = np.mean(volumes[-20:]) if len(volumes) >= 20 else np.mean(volumes)

        return current_vol / avg_vol if avg_vol > 0 else 1.0

    def check_gap_and_go_signal(self, symbol, current_price, candles):
        """Check for Gap and Go entry signal."""
        prev_close = self.get_previous_close(symbol)
        if prev_close is None:
            return False, None, None

        # Calculate gap percentage
        gap_pct = ((current_price - prev_close) / prev_close) * 100

        # Check gap threshold
        if gap_pct < GAP_MIN_PCT:
            return False, gap_pct, None

        # Check volume
        vol_ratio = self.get_volume_ratio(symbol, candles)
        if vol_ratio < GAP_VOLUME_MULT:
            return False, gap_pct, vol_ratio

        # Gap holds (price still above prev close)
        if current_price <= prev_close:
            return False, gap_pct, vol_ratio

        return True, gap_pct, vol_ratio

    def enter_gap_and_go(self, symbol, entry_price, gap_pct, vol_ratio):
        """Enter a Gap and Go trade."""
        stop_loss = entry_price * (1 - GAP_SL_PCT / 100)
        take_profit = entry_price * (1 + GAP_TP_PCT / 100)

        return self._execute_entry(
            symbol, 'LONG', entry_price, stop_loss, take_profit, 'GAP_GO'
        )

    # ─────────────────────────────────────────────────────────────
    # MACD BULLISH STRATEGY (from strategy generator - +58% on AMD)
    # ─────────────────────────────────────────────────────────────

    def calculate_macd(self, prices):
        """Calculate MACD indicator."""
        if len(prices) < MACD_SLOW + MACD_SIGNAL:
            return None, None, None

        if TA_LIB_AVAILABLE:
            close = pd.Series(prices)
            macd_indicator = TALibMACD(close, window_fast=MACD_FAST, window_slow=MACD_SLOW, window_sign=MACD_SIGNAL)
            macd_line = macd_indicator.macd().iloc[-1]
            signal_line = macd_indicator.macd_signal().iloc[-1]
            histogram = macd_indicator.macd_diff().iloc[-1]
            return macd_line, signal_line, histogram
        else:
            # Manual calculation
            prices_arr = np.array(prices)
            ema_fast = self._ema(prices_arr, MACD_FAST)
            ema_slow = self._ema(prices_arr, MACD_SLOW)
            macd_line = ema_fast - ema_slow
            signal_line = self._ema([macd_line], MACD_SIGNAL) if len([macd_line]) >= MACD_SIGNAL else macd_line
            return macd_line, signal_line, macd_line - signal_line

    def _ema(self, prices, period):
        """Calculate EMA manually."""
        if len(prices) < period:
            return np.mean(prices) if len(prices) > 0 else 0
        prices = np.array(prices)
        weights = np.exp(np.linspace(-1., 0., period))
        weights /= weights.sum()
        return np.convolve(prices, weights, mode='valid')[-1]

    def check_macd_signal(self, symbol, prices):
        """Check for MACD bullish crossover signal."""
        if len(prices) < MACD_SLOW + MACD_SIGNAL + 2:
            return False, None, None

        # Get current and previous MACD values
        macd_curr, signal_curr, hist_curr = self.calculate_macd(prices)
        macd_prev, signal_prev, hist_prev = self.calculate_macd(prices[:-1])

        if macd_curr is None or macd_prev is None:
            return False, None, None

        # Bullish crossover: MACD crosses above signal line
        bullish_cross = (macd_prev <= signal_prev) and (macd_curr > signal_curr)

        return bullish_cross, macd_curr, signal_curr

    def enter_macd(self, symbol, entry_price, macd_val, signal_val):
        """Enter a MACD Bullish trade."""
        stop_loss = entry_price * (1 - MACD_SL_PCT / 100)
        take_profit = entry_price * (1 + MACD_TP_PCT / 100)

        return self._execute_entry(
            symbol, 'LONG', entry_price, stop_loss, take_profit, 'MACD'
        )

    # ─────────────────────────────────────────────────────────────
    # BB LOWER BOUNCE STRATEGY (from strategy generator - +42% on AMD)
    # ─────────────────────────────────────────────────────────────

    def calculate_bollinger_bands(self, prices):
        """Calculate Bollinger Bands."""
        if len(prices) < BB_PERIOD:
            return None, None, None

        if TA_LIB_AVAILABLE:
            close = pd.Series(prices)
            bb = BollingerBands(close, window=BB_PERIOD, window_dev=BB_STD)
            upper = bb.bollinger_hband().iloc[-1]
            middle = bb.bollinger_mavg().iloc[-1]
            lower = bb.bollinger_lband().iloc[-1]
            return upper, middle, lower
        else:
            # Manual calculation
            prices_arr = np.array(prices[-BB_PERIOD:])
            middle = np.mean(prices_arr)
            std = np.std(prices_arr)
            upper = middle + BB_STD * std
            lower = middle - BB_STD * std
            return upper, middle, lower

    def check_bb_bounce_signal(self, symbol, current_price, prices):
        """Check for BB Lower Bounce signal (price touches lower band)."""
        if len(prices) < BB_PERIOD:
            return False, None, None, None

        upper, middle, lower = self.calculate_bollinger_bands(prices)

        if lower is None:
            return False, None, None, None

        # Also check RSI for oversold confirmation
        rsi = calculate_rsi(prices)

        # Signal: price at or below lower BB AND RSI < 30 (oversold)
        at_lower_band = current_price <= lower
        rsi_oversold = rsi < RSI_OVERSOLD

        return (at_lower_band and rsi_oversold), lower, middle, rsi

    def enter_bb_bounce(self, symbol, entry_price, lower_band, middle_band, rsi):
        """Enter a BB Lower Bounce trade."""
        stop_loss = entry_price * (1 - BB_BOUNCE_SL_PCT / 100)
        take_profit = entry_price * (1 + BB_BOUNCE_TP_PCT / 100)

        return self._execute_entry(
            symbol, 'LONG', entry_price, stop_loss, take_profit, 'BB_BOUNCE'
        )

    # ─────────────────────────────────────────────────────────────
    # MOMENTUM STRATEGY (for TREND_UP / bull markets)
    # Trades WITH the trend using pullback entries to 9 EMA
    # ─────────────────────────────────────────────────────────────

    def calculate_ema(self, prices, period):
        """Calculate Exponential Moving Average."""
        if len(prices) < period:
            return np.mean(prices) if len(prices) > 0 else 0

        if TA_LIB_AVAILABLE:
            close = pd.Series(prices)
            ema = EMAIndicator(close, window=period)
            return ema.ema_indicator().iloc[-1]
        else:
            # Manual EMA calculation
            prices_arr = np.array(prices)
            multiplier = 2 / (period + 1)
            ema = prices_arr[0]
            for price in prices_arr[1:]:
                ema = (price * multiplier) + (ema * (1 - multiplier))
            return ema

    def get_volume_average(self, candles, period=20):
        """Get average volume over period."""
        if not candles or len(candles) < period:
            volumes = [c['volume'] for c in candles] if candles else [0]
            return np.mean(volumes) if volumes else 0
        return np.mean([c['volume'] for c in candles[-period:]])

    def check_momentum_signal(self, symbol, current_price, prices, candles, verbose=False):
        """
        Check for MOMENTUM pullback entry signal.

        Conditions for entry:
        1. Price > 20 SMA (uptrend confirmed)
        2. RSI between 50-70 (strong but not overbought)
        3. Price pulls back to 9 EMA (within tolerance)
        4. Current volume >= average volume

        Returns:
            (should_enter, sma_20, ema_9, rsi, volume_ratio)
        """
        if len(prices) < max(MOMENTUM_SMA_PERIOD, MOMENTUM_EMA_PERIOD, RSI_PERIOD + 1):
            if verbose:
                cprint(f"    [MOMENTUM DEBUG] {symbol}: Insufficient data ({len(prices)} prices, need {max(MOMENTUM_SMA_PERIOD, MOMENTUM_EMA_PERIOD, RSI_PERIOD + 1)})", "yellow")
            return False, None, None, None, None

        # Calculate indicators
        sma_20 = calculate_sma(prices, MOMENTUM_SMA_PERIOD)
        ema_9 = self.calculate_ema(prices, MOMENTUM_EMA_PERIOD)
        rsi = calculate_rsi(prices, RSI_PERIOD)

        # Calculate volume ratio
        avg_volume = self.get_volume_average(candles, 20)
        current_volume = candles[-1]['volume'] if candles else 0
        volume_ratio = current_volume / avg_volume if avg_volume > 0 else 0

        # Check conditions
        # 1. Price above 20 SMA (uptrend)
        above_sma = current_price > sma_20

        # 2. RSI in momentum zone (50-70)
        rsi_ok = MOMENTUM_RSI_MIN <= rsi <= MOMENTUM_RSI_MAX

        # 3. Price at or near 9 EMA (pullback entry)
        ema_distance = abs(current_price - ema_9) / current_price
        at_ema = ema_distance <= MOMENTUM_EMA_TOLERANCE

        # Also accept if price just bounced off EMA (within 1%)
        bouncing_off_ema = current_price > ema_9 and ema_distance <= 0.01

        # 4. Volume confirmation
        volume_ok = volume_ratio >= MOMENTUM_VOLUME_MULT

        # Count passing conditions
        conditions = [above_sma, rsi_ok, (at_ema or bouncing_off_ema), volume_ok or MOMENTUM_VOLUME_MULT == 0]
        passing = sum(conditions)

        # Full signal: all conditions met
        # Near-miss: 3 of 4 conditions met AND price above SMA (must be in uptrend)
        should_enter = passing >= 4 or (passing >= 3 and above_sma)

        # Verbose logging for debugging
        if verbose:
            sma_mark = "✓" if above_sma else "✗"
            rsi_mark = "✓" if rsi_ok else "✗"
            ema_mark = "✓" if (at_ema or bouncing_off_ema) else "✗"
            vol_mark = "✓" if (volume_ok or MOMENTUM_VOLUME_MULT == 0) else "✗"

            cprint(f"    [MOMENTUM DEBUG] {symbol} @ ${current_price:.2f}", "cyan")
            cprint(f"      SMA20: ${sma_20:.2f} | Price {'>' if above_sma else '<'} SMA [{sma_mark}]", "green" if above_sma else "red")
            cprint(f"      RSI: {rsi:.1f} | Range {MOMENTUM_RSI_MIN}-{MOMENTUM_RSI_MAX} [{rsi_mark}]", "green" if rsi_ok else "red")
            cprint(f"      EMA9: ${ema_9:.2f} | Distance: {ema_distance*100:.2f}% (need <{MOMENTUM_EMA_TOLERANCE*100:.1f}%) [{ema_mark}]", "green" if (at_ema or bouncing_off_ema) else "red")
            cprint(f"      Volume: {volume_ratio:.2f}x avg [{vol_mark}]", "green" if (volume_ok or MOMENTUM_VOLUME_MULT == 0) else "red")
            cprint(f"      → Conditions: {passing}/4 | SIGNAL: {'YES' if should_enter else 'NO'}{' (3/4 near-miss)' if passing == 3 and should_enter else ''}", "green" if should_enter else "yellow")

        return should_enter, sma_20, ema_9, rsi, volume_ratio, passing

    def enter_momentum(self, symbol, entry_price, sma_20, ema_9, rsi, volume_ratio, signal_strength=None):
        """Enter a MOMENTUM pullback trade."""
        stop_loss = entry_price * (1 - MOMENTUM_SL_PCT / 100)
        take_profit = entry_price * (1 + MOMENTUM_TP_PCT / 100)

        return self._execute_entry(
            symbol, 'LONG', entry_price, stop_loss, take_profit, 'MOMENTUM',
            signal_strength=signal_strength,
        )

    # ─────────────────────────────────────────────────────────────
    # COMMON EXECUTION (with robust order management)
    # ─────────────────────────────────────────────────────────────

    def _execute_entry(self, symbol, direction, entry_price, stop_loss, take_profit, strategy, signal_strength=None):
        """
        Execute trade entry with full safety checks.

        Flow:
        1. Check cooldown
        2. Check daily loss limit
        3. Check existing position (prevent duplicates)
        4. Cancel any open orders for symbol
        5. Validate quantity with proper rounding
        6. Risk agent check
        7. Entry confirmation (for breakouts)
        8. Place bracket order with atomic TP/SL
        9. Log to trade journal
        """
        # Step 1: Check cooldown
        if self.is_on_cooldown(symbol):
            cprint(f"    {symbol} on cooldown, skipping", "yellow")
            return False

        # Step 1b: Check entry time restriction (avoid morning volatility)
        if EARLIEST_ENTRY_ET and not self.mock_mode:
            try:
                clock = self.api.get_clock()
                et_now = clock.timestamp
                hour, minute = map(int, EARLIEST_ENTRY_ET.split(':'))
                if et_now.hour < hour or (et_now.hour == hour and et_now.minute < minute):
                    cprint(f"    [TIME] Skipping {symbol} entry — before {EARLIEST_ENTRY_ET} ET ({et_now.strftime('%H:%M')} ET)", "yellow")
                    return False
            except Exception:
                pass  # Don't block entries if clock check fails

        # Step 2: Check daily loss limit (unless mock mode)
        if not self.mock_mode and not check_daily_loss_limit():
            cprint(f"    {symbol} BLOCKED: Daily loss limit reached", "red")
            return False

        # Step 3: Check existing position (prevent duplicates)
        if not self.mock_mode:
            has_position, position_info = check_existing_position(symbol)
            if has_position:
                cprint(f"    {symbol} SKIPPED: Already in position ({position_info['qty']} shares {position_info['side']})", "yellow")
                return False

        # Step 4: Cancel any open orders for this symbol (prevent orphans)
        if not self.mock_mode:
            cancelled = cancel_symbol_orders(symbol)
            if cancelled > 0:
                cprint(f"    Cancelled {cancelled} existing orders for {symbol}", "yellow")

        # Step 5: Calculate and validate shares with proper rounding
        # IMPORTANT: Alpaca bracket orders require WHOLE shares (no fractional)
        raw_shares = ORDER_USD_SIZE / entry_price
        if self.mock_mode:
            shares = round(raw_shares, 4)
        else:
            # Bracket orders don't support fractional - round DOWN to whole shares
            shares = int(raw_shares)  # Floor to whole number

            # If position too small for whole shares, try 1 share minimum
            if shares == 0:
                one_share_value = entry_price
                if one_share_value <= ORDER_USD_SIZE * 2:  # Allow up to 2x normal size
                    shares = 1
                    cprint(f"    Position size too small for whole shares, using 1 share (${one_share_value:.2f})", "yellow")
                else:
                    cprint(f"    SKIPPED: Price ${entry_price:.2f} too high for ${ORDER_USD_SIZE} position (need whole shares for bracket orders)", "yellow")
                    return False

        if shares < 0.0001:
            cprint(f"    Order size too small for {symbol}", "yellow")
            return False

        # Step 6: Risk check
        position_value = shares * entry_price
        risk_allowed, risk_reason = check_risk_verbose(symbol, position_value)
        if not risk_allowed:
            cprint(f"    RISK BLOCKED: {risk_reason}", "red")
            return False

        # Step 7: Entry confirmation (for breakouts)
        confidence = None
        entry_reasons = []
        if strategy == 'BREAKOUT':
            candles = self.fetch_candles(symbol, hours=50)
            should_enter, confidence, entry_reasons = check_entry(symbol, direction, entry_price, candles, None)
            if not should_enter or confidence < MIN_ENTRY_CONFIDENCE:
                cprint(f"    Entry rejected: confidence {confidence}% < {MIN_ENTRY_CONFIDENCE}%", "red")
                return False

        # Get current regime for logging
        regime = self.symbol_regimes.get(symbol, 'UNKNOWN')

        # Build reasoning string
        if strategy == 'BREAKOUT':
            reasoning = f"24h breakout {direction}. {'; '.join(entry_reasons) if entry_reasons else ''}"
        elif strategy == 'MEAN_REV':
            reasoning = "RSI oversold + price below SMA"
        elif strategy == 'GAP_GO':
            reasoning = "Gap up with volume confirmation"
        elif strategy == 'MACD':
            reasoning = "MACD bullish crossover (10% TP / 7% SL)"
        elif strategy == 'BB_BOUNCE':
            reasoning = "BB lower band touch + RSI oversold (8% TP / 5% SL)"
        elif strategy == 'MOMENTUM':
            reasoning = "Uptrend pullback to 9 EMA with RSI 50-70 (6% TP / 3% SL)"
        else:
            reasoning = f"{strategy} signal"

        # Strategy labels and stats keys
        strategy_labels = {
            'BREAKOUT': 'BREAKOUT',
            'MEAN_REV': 'MEAN_REV',
            'GAP_GO': 'GAP&GO',
            'MACD': 'MACD',
            'BB_BOUNCE': 'BB_BOUNCE',
            'MOMENTUM': 'MOMENTUM'
        }
        strategy_label = strategy_labels.get(strategy, strategy)

        stats_keys = {
            'BREAKOUT': 'breakout',
            'MEAN_REV': 'mean_reversion',
            'GAP_GO': 'gap_and_go',
            'MACD': 'macd',
            'BB_BOUNCE': 'bb_bounce',
            'MOMENTUM': 'momentum'
        }
        stats_key = stats_keys.get(strategy, 'breakout')

        # Step 8: Execute order
        if self.mock_mode or self.dry_run:
            # Mock/Dry run mode - simulate execution
            mode = 'MOCK' if self.mock_mode else 'DRY RUN'
            cprint(f"[{mode}] {strategy_label} {direction}: {symbol} | {shares} @ ${entry_price:.2f}", "magenta")
            cprint(f"    SL: ${stop_loss:.2f} | TP: ${take_profit:.2f}", "cyan")

            self.active_trades[symbol] = {
                'entry': entry_price,
                'entry_time': datetime.now(),
                'stop_loss': stop_loss,
                'take_profit': take_profit,
                'direction': direction,
                'shares': shares,
                'strategy': strategy,
                'order_id': f'MOCK_{symbol}_{datetime.now().timestamp()}'
            }
            update_position(symbol, position_value)
            self.set_cooldown(symbol)
            self.strategy_stats[stats_key]['trades'] += 1

            # Log to trade journal
            log_trade(
                action='ENTRY',
                symbol=symbol,
                strategy=strategy,
                direction=direction,
                price=entry_price,
                shares=shares,
                stop_loss=stop_loss,
                take_profit=take_profit,
                confidence=confidence,
                regime=regime,
                reasoning=reasoning,
                entry_signal_strength=signal_strength,
            )
            return True

        # Live trading with bracket order
        try:
            side = 'buy' if direction == 'LONG' else 'sell'

            # Place bracket order with atomic TP/SL
            order_id = place_bracket_order(
                symbol=symbol,
                side=side,
                qty=shares,
                entry_price=entry_price,
                tp_price=take_profit,
                sl_price=stop_loss,
                order_type='market'
            )

            if not order_id:
                cprint(f"    Failed to place bracket order for {symbol}", "red")
                return False

            cprint(f"{strategy_label} {direction}: {symbol} | {shares} @ ${entry_price:.2f}", "magenta")
            cprint(f"    Bracket Order ID: {order_id}", "green")
            cprint(f"    TP: ${take_profit:.2f} | SL: ${stop_loss:.2f}", "cyan")

            self.active_trades[symbol] = {
                'entry': entry_price,
                'entry_time': datetime.now(),
                'stop_loss': stop_loss,
                'take_profit': take_profit,
                'direction': direction,
                'shares': shares,
                'strategy': strategy,
                'order_id': order_id,
                'is_bracket': True  # Flag to indicate bracket order
            }
            update_position(symbol, position_value)
            self.set_cooldown(symbol)
            self.strategy_stats[stats_key]['trades'] += 1

            # Log to trade journal
            log_trade(
                action='ENTRY',
                symbol=symbol,
                strategy=strategy,
                direction=direction,
                price=entry_price,
                shares=shares,
                stop_loss=stop_loss,
                take_profit=take_profit,
                confidence=confidence,
                regime=regime,
                reasoning=f"{reasoning} [Bracket order]",
                entry_signal_strength=signal_strength,
            )
            return True

        except Exception as e:
            cprint(f"    Error entering {symbol}: {e}", "red")
            return False

    def close_trade(self, symbol, current_price, reason):
        """
        Close a trade with proper cleanup.

        For bracket orders, the TP/SL are already set with Alpaca,
        so this is mainly for manual exits or tracking.
        """
        if symbol not in self.active_trades:
            return

        trade = self.active_trades[symbol]
        entry = trade['entry']
        shares = trade['shares']
        direction = trade['direction']
        strategy = trade['strategy']
        is_bracket = trade.get('is_bracket', False)

        if direction == 'LONG':
            pnl_pct = ((current_price - entry) / entry) * 100
            pnl_dollars = (current_price - entry) * shares
        else:
            pnl_pct = ((entry - current_price) / entry) * 100
            pnl_dollars = (entry - current_price) * shares

        # Compute enhanced exit metrics
        entry_time = trade.get('entry_time')
        if entry_time:
            duration = datetime.now() - entry_time
            total_min = int(duration.total_seconds() / 60)
            hours, mins = divmod(total_min, 60)
            time_in_trade = f"{hours}h {mins}m" if hours > 0 else f"{mins}m"
        else:
            time_in_trade = None

        # Slippage: difference between expected exit (TP or SL) and actual fill
        stop_loss = trade.get('stop_loss', 0)
        take_profit = trade.get('take_profit', 0)
        slippage = None  # Computed as actual_exit - nearest_target

        # R-multiple: actual return / risk per share
        if stop_loss and entry:
            risk_per_share = abs(entry - stop_loss)
            if risk_per_share > 0:
                if direction == 'LONG':
                    reward_per_share = current_price - entry
                else:
                    reward_per_share = entry - current_price
                r_multiple = reward_per_share / risk_per_share
            else:
                r_multiple = None
        else:
            r_multiple = None

        # Map strategy to stats key
        stats_key_map = {
            'BREAKOUT': 'breakout',
            'MEAN_REV': 'mean_reversion',
            'GAP_GO': 'gap_and_go',
            'MACD': 'macd',
            'BB_BOUNCE': 'bb_bounce',
            'MOMENTUM': 'momentum'
        }
        strat_key = stats_key_map.get(strategy, 'breakout')

        if self.mock_mode or self.dry_run:
            mode = 'MOCK' if self.mock_mode else 'DRY RUN'
            color = "green" if pnl_pct > 0 else "red"
            cprint(f"[{mode}] CLOSE {symbol}: {reason} | PnL: {pnl_pct:+.2f}% (${pnl_dollars:+.2f})", color)

            # Track stats
            if pnl_pct > 0:
                self.strategy_stats[strat_key]['wins'] += 1

            # Log to trade journal
            log_trade(
                action='EXIT',
                symbol=symbol,
                strategy=strategy,
                direction=direction,
                price=current_price,
                shares=shares,
                entry_price=entry,
                exit_price=current_price,
                pnl=pnl_dollars,
                pnl_pct=pnl_pct,
                reasoning=reason,
                time_in_trade=time_in_trade,
                slippage=slippage,
                r_multiple=r_multiple,
            )

            close_position(symbol, pnl_dollars)
            reset_position(symbol)
            del self.active_trades[symbol]
            return

        # Live trading exit
        try:
            # Cancel any remaining open orders for this symbol
            cancelled = cancel_symbol_orders(symbol)
            if cancelled > 0:
                cprint(f"    Cancelled {cancelled} remaining orders for {symbol}", "yellow")

            # Check if we still have a position to close
            has_position, position_info = check_existing_position(symbol)

            if has_position:
                # Close the position
                side = 'sell' if direction == 'LONG' else 'buy'
                from src.utils.order_utils import place_simple_order
                order_id = place_simple_order(
                    symbol=symbol,
                    side=side,
                    qty=shares,
                    order_type='market'
                )

                if order_id:
                    cprint(f"CLOSE {symbol}: {reason} | Order ID: {order_id}", "green")
                else:
                    cprint(f"CLOSE {symbol}: {reason} (order may have failed)", "yellow")
            else:
                # Position already closed (likely by bracket TP/SL)
                cprint(f"CLOSE {symbol}: {reason} (position already closed by bracket order)", "green")

            color = "green" if pnl_pct > 0 else "red"
            cprint(f"    PnL: {pnl_pct:+.2f}% (${pnl_dollars:+.2f})", color)

            if pnl_pct > 0:
                self.strategy_stats[strat_key]['wins'] += 1

            # Update daily P&L tracker
            update_daily_pnl(pnl_dollars)

            # Log to trade journal
            log_trade(
                action='EXIT',
                symbol=symbol,
                strategy=strategy,
                direction=direction,
                price=current_price,
                shares=shares,
                entry_price=entry,
                exit_price=current_price,
                pnl=pnl_dollars,
                pnl_pct=pnl_pct,
                reasoning=reason,
                time_in_trade=time_in_trade,
                slippage=slippage,
                r_multiple=r_multiple,
            )

            close_position(symbol, pnl_dollars)
            reset_position(symbol)
            del self.active_trades[symbol]

        except Exception as e:
            cprint(f"Error closing {symbol}: {e}", "red")

    def manage_position(self, symbol, current_price):
        """Manage an active position."""
        if symbol not in self.active_trades:
            return

        trade = self.active_trades[symbol]
        direction = trade['direction']
        entry = trade['entry']
        stop_loss = trade['stop_loss']
        take_profit = trade['take_profit']
        strategy = trade['strategy']

        # Check hard SL/TP
        if direction == 'LONG':
            if current_price <= stop_loss:
                self.close_trade(symbol, current_price, f"SL hit (${current_price:.2f})")
                return
            if current_price >= take_profit:
                self.close_trade(symbol, current_price, f"TP hit (${current_price:.2f})")
                return
        else:
            if current_price >= stop_loss:
                self.close_trade(symbol, current_price, f"SL hit (${current_price:.2f})")
                return
            if current_price <= take_profit:
                self.close_trade(symbol, current_price, f"TP hit (${current_price:.2f})")
                return

        # Mean reversion specific exits
        if strategy == 'MEAN_REV':
            prices = self.get_price_history(symbol)
            should_exit, exit_reason = self.check_mean_reversion_exit(
                symbol, current_price, prices, entry
            )
            if should_exit:
                self.close_trade(symbol, current_price, exit_reason)
                return

        # Check exit agent for trailing stops
        entry_time = trade.get('entry_time', datetime.now())
        exit_decision, exit_reason, new_stop = manage_exit_verbose(
            symbol, entry, entry_time, current_price,
            stop_loss, take_profit, direction
        )

        if exit_decision == 'CLOSE_FULL':
            self.close_trade(symbol, current_price, exit_reason)
        elif exit_decision == 'MOVE_STOP':
            self.active_trades[symbol]['stop_loss'] = new_stop
            cprint(f"    {symbol}: {exit_reason}", "green")

    def update_mock_data(self):
        """Update mock prices with realistic trend + pullback behavior."""
        for symbol in self.current_symbols:
            if symbol in self.mock_prices:
                base = self.mock_prices[symbol]

                # Calculate 9 EMA from price history
                prices = self.mock_price_history.get(symbol, [base])
                if len(prices) >= 9:
                    ema = prices[-9]
                    for p in prices[-8:]:
                        ema = p * 0.2 + ema * 0.8
                else:
                    ema = base

                # Simulate pullback/bounce behavior around EMA
                dist_to_ema = (base - ema) / base if base > 0 else 0

                # If price is far above EMA, bias toward pullback
                # If price is near EMA, allow bounce with uptrend bias
                if dist_to_ema > 0.02:  # >2% above EMA, likely to pull back
                    noise = base * random.uniform(-0.008, 0.002)
                elif dist_to_ema < -0.01:  # Below EMA, likely to bounce
                    noise = base * random.uniform(-0.002, 0.008)
                else:  # Near EMA - could go either way with slight upward bias
                    noise = base * random.uniform(-0.004, 0.006)

                self.mock_prices[symbol] = base + noise

                if symbol in self.mock_price_history:
                    self.mock_price_history[symbol].append(self.mock_prices[symbol])
                    self.mock_price_history[symbol] = self.mock_price_history[symbol][-50:]

                if symbol in self.mock_candles:
                    last = self.mock_candles[symbol][-1]
                    new_candle = {
                        'open': last['close'],
                        'high': self.mock_prices[symbol] * 1.005,
                        'low': self.mock_prices[symbol] * 0.995,
                        'close': self.mock_prices[symbol],
                        'volume': random.uniform(700000, 1200000)  # Higher volume for consistency
                    }
                    self.mock_candles[symbol] = self.mock_candles[symbol][-49:] + [new_candle]

    def is_market_open(self):
        """Check if market is open."""
        if self.mock_mode:
            return True
        try:
            clock = self.api.get_clock()
            return clock.is_open
        except:
            return False

    def check_symbols(self):
        """Main loop - check all symbols with adaptive strategy."""
        if self.use_scanner:
            self._refresh_symbols()

        # Get P&L status from both risk agent and order utils
        risk_status = get_risk_status()
        daily_pnl = get_daily_pnl() if not self.mock_mode else risk_status['daily_pnl']

        cprint(f"\n{'─'*60}", "cyan")
        cprint(f"  Cycle {self.cycle_count} | {datetime.now().strftime('%H:%M:%S')} | Daily P&L: ${daily_pnl:+,.2f}", "cyan")
        cprint(f"  Active positions: {len(self.active_trades)}", "cyan")
        cprint(f"{'─'*60}", "cyan")

        # Check both daily loss limits
        if not risk_status['trading_allowed']:
            cprint("TRADING HALTED: Risk agent daily loss limit reached!", "red")
            self.cycle_count += 1
            return

        if not self.mock_mode and is_trading_halted():
            cprint("TRADING HALTED: Order utils daily loss limit reached!", "red")
            self.cycle_count += 1
            return

        if not self.mock_mode and not self.is_market_open():
            cprint("Market is closed", "yellow")
            self.cycle_count += 1
            return

        if self.mock_mode:
            self.update_mock_data()

        for symbol in self.current_symbols:
            current_price = self.get_current_price(symbol)
            if current_price is None:
                continue

            # Get regime
            regime = self.get_regime(symbol)
            regime_color = STRATEGY_MAP.get(regime, {}).get('color', 'white') if REGIME_DETECTOR_AVAILABLE else 'white'

            # Status line
            cprint(f"\n{symbol}: ${current_price:.2f} | Regime: {regime}", regime_color)

            # Manage existing position
            if symbol in self.active_trades:
                trade = self.active_trades[symbol]
                pnl_pct = ((current_price - trade['entry']) / trade['entry']) * 100
                strat = trade['strategy']
                cprint(f"  Active {strat} {trade['direction']}: Entry ${trade['entry']:.2f} | PnL: {pnl_pct:+.2f}%", "magenta")
                self.manage_position(symbol, current_price)
                continue

            # Skip if on cooldown
            if self.is_on_cooldown(symbol):
                continue

            # ─────────────────────────────────────────────────────────
            # STRATEGY SELECTION: 1) Check override, 2) Fallback to regime
            # ─────────────────────────────────────────────────────────

            # Check for per-symbol strategy override first
            override_strategy = SYMBOL_STRATEGY_OVERRIDE.get(symbol)

            if override_strategy:
                # Use backtested strategy override
                if override_strategy == 'HOLD':
                    cprint(f"  → HOLD (backtested: buy & hold beats active trading)", "green")
                    self.strategy_stats['skipped'] += 1
                    continue

                elif override_strategy == 'BREAKOUT':
                    # Always use breakout for this symbol
                    candles = self.fetch_candles(symbol, hours=50)
                    signal, high_24h, low_24h = self.check_breakout_signal(symbol, current_price, candles)

                    if signal:
                        if signal == 'SHORT':
                            cprint(f"  → SHORT signal skipped (long-only mode)", "yellow")
                            continue
                        cprint(f"  → [OVERRIDE] BREAKOUT {signal}: ${current_price:.2f} > 24h high ${high_24h:.2f}", "magenta")
                        self.enter_breakout(symbol, signal, current_price, candles)
                    else:
                        dist_high = ((high_24h - current_price) / current_price * 100) if high_24h else 0
                        cprint(f"  → [OVERRIDE] Waiting for breakout ({dist_high:.1f}% to high)", "white")
                    continue

                elif override_strategy == 'GAP_AND_GO':
                    # Check for gap and go signal
                    candles = self.fetch_candles(symbol, hours=50)
                    should_enter, gap_pct, vol_ratio = self.check_gap_and_go_signal(symbol, current_price, candles)

                    if should_enter:
                        cprint(f"  → [OVERRIDE] GAP&GO: Gap {gap_pct:.1f}%, Vol {vol_ratio:.1f}x", "yellow")
                        self.enter_gap_and_go(symbol, current_price, gap_pct, vol_ratio)
                    else:
                        gap_str = f"Gap {gap_pct:.1f}%" if gap_pct else "No gap"
                        vol_str = f"Vol {vol_ratio:.1f}x" if vol_ratio else ""
                        cprint(f"  → [OVERRIDE] Waiting for gap ({gap_str} {vol_str})", "white")
                    continue

                elif override_strategy == 'MACD':
                    # MACD Bullish strategy (+58% backtested on AMD)
                    prices = self.get_price_history(symbol)
                    should_enter, macd_val, signal_val = self.check_macd_signal(symbol, prices)

                    if should_enter:
                        cprint(f"  → [OVERRIDE] MACD BULLISH: MACD {macd_val:.3f} > Signal {signal_val:.3f}", "cyan")
                        self.enter_macd(symbol, current_price, macd_val, signal_val)
                    else:
                        if macd_val is not None:
                            cprint(f"  → [OVERRIDE] Waiting for MACD cross (MACD:{macd_val:.3f} Signal:{signal_val:.3f})", "white")
                        else:
                            cprint(f"  → [OVERRIDE] Waiting for MACD data", "white")
                    continue

                elif override_strategy == 'BB_BOUNCE':
                    # BB Lower Bounce strategy (+42% backtested)
                    prices = self.get_price_history(symbol)
                    should_enter, lower_band, middle_band, rsi = self.check_bb_bounce_signal(symbol, current_price, prices)

                    if should_enter:
                        cprint(f"  → [OVERRIDE] BB BOUNCE: Price ${current_price:.2f} at lower band ${lower_band:.2f}, RSI {rsi:.1f}", "blue")
                        self.enter_bb_bounce(symbol, current_price, lower_band, middle_band, rsi)
                    else:
                        if lower_band is not None:
                            dist_to_lower = ((current_price - lower_band) / current_price) * 100
                            rsi_str = f"RSI:{rsi:.1f}" if rsi else ""
                            cprint(f"  → [OVERRIDE] Waiting for BB touch ({dist_to_lower:.1f}% above lower band, {rsi_str})", "white")
                        else:
                            cprint(f"  → [OVERRIDE] Waiting for BB data", "white")
                    continue

                elif override_strategy == 'MOMENTUM':
                    # MOMENTUM pullback strategy (for trend-following in uptrends)
                    prices = self.get_price_history(symbol)
                    candles = self.fetch_candles(symbol, hours=50)
                    should_enter, sma_20, ema_9, rsi, vol_ratio, passing = self.check_momentum_signal(
                        symbol, current_price, prices, candles, verbose=self.verbose
                    )

                    if should_enter:
                        cprint(f"  → [OVERRIDE] MOMENTUM: Pullback to 9EMA ${ema_9:.2f}, RSI {rsi:.1f}, Vol {vol_ratio:.1f}x", "green")
                        self.enter_momentum(symbol, current_price, sma_20, ema_9, rsi, vol_ratio, signal_strength=f"{passing}/4")
                    else:
                        if ema_9 is not None and sma_20 is not None:
                            above_sma = "✓" if current_price > sma_20 else "✗"
                            rsi_ok = "✓" if rsi and MOMENTUM_RSI_MIN <= rsi <= MOMENTUM_RSI_MAX else "✗"
                            ema_dist = ((current_price - ema_9) / current_price * 100) if ema_9 else 0
                            cprint(f"  → [OVERRIDE] Waiting for pullback (SMA:{above_sma} RSI:{rsi_ok} EMA dist:{ema_dist:.1f}%)", "white")
                        else:
                            cprint(f"  → [OVERRIDE] Waiting for MOMENTUM data", "white")
                    continue

            # No override - use regime-based strategy selection
            if regime == 'TREND_UP':
                # Use MOMENTUM pullback strategy for uptrends (NEW!)
                prices = self.get_price_history(symbol)
                candles = self.fetch_candles(symbol, hours=50)
                should_enter, sma_20, ema_9, rsi, vol_ratio, passing = self.check_momentum_signal(
                    symbol, current_price, prices, candles, verbose=self.verbose
                )

                if should_enter:
                    cprint(f"  → MOMENTUM: Pullback to 9EMA ${ema_9:.2f}, RSI {rsi:.1f}, Vol {vol_ratio:.1f}x", "green")
                    self.enter_momentum(symbol, current_price, sma_20, ema_9, rsi, vol_ratio, signal_strength=f"{passing}/4")
                else:
                    if ema_9 is not None and sma_20 is not None:
                        above_sma = "✓" if current_price > sma_20 else "✗"
                        rsi_str = f"{rsi:.1f}" if rsi else "N/A"
                        ema_dist = ((current_price - ema_9) / current_price * 100) if ema_9 else 0
                        cprint(f"  → Waiting for pullback (SMA:{above_sma} RSI:{rsi_str} EMA dist:{ema_dist:.1f}%)", "white")
                    else:
                        cprint(f"  → Waiting for MOMENTUM data", "white")

            elif regime in ('RANGING', 'VOLATILE'):
                # Use BREAKOUT strategy
                candles = self.fetch_candles(symbol, hours=50)
                signal, high_24h, low_24h = self.check_breakout_signal(symbol, current_price, candles)

                if signal:
                    # Skip shorts for now
                    if signal == 'SHORT':
                        cprint(f"  → SHORT signal skipped (long-only mode)", "yellow")
                        continue

                    cprint(f"  → BREAKOUT {signal}: Price ${current_price:.2f} > 24h high ${high_24h:.2f}", "yellow")
                    self.enter_breakout(symbol, signal, current_price, candles)
                else:
                    dist_high = ((high_24h - current_price) / current_price * 100) if high_24h else 0
                    dist_low = ((current_price - low_24h) / current_price * 100) if low_24h else 0
                    cprint(f"  → Waiting for breakout ({dist_high:.1f}% to high, {dist_low:.1f}% to low)", "white")

            elif regime == 'TREND_DOWN':
                # Use MEAN REVERSION strategy
                prices = self.get_price_history(symbol)
                should_enter, rsi, sma = self.check_mean_reversion_signal(symbol, current_price, prices)

                if should_enter:
                    sma_dev = ((sma - current_price) / sma * 100) if sma else 0
                    cprint(f"  → MEAN REV: RSI={rsi:.1f}, {sma_dev:.1f}% below SMA", "yellow")
                    self.enter_mean_reversion(symbol, current_price, rsi, sma)
                else:
                    rsi_val = rsi if rsi else 50
                    cprint(f"  → Waiting for oversold (RSI={rsi_val:.1f}, need <{RSI_OVERSOLD})", "white")

        self.cycle_count += 1

    def print_stats(self):
        """Print session statistics."""
        cprint("\n" + "=" * 60, "cyan")
        cprint("  SESSION STATISTICS", "cyan")
        cprint("=" * 60, "cyan")

        bo = self.strategy_stats['breakout']
        mr = self.strategy_stats['mean_reversion']
        gg = self.strategy_stats['gap_and_go']
        macd = self.strategy_stats['macd']
        bb = self.strategy_stats['bb_bounce']
        mom = self.strategy_stats['momentum']

        cprint(f"\n  Momentum Pullback Strategy (6% TP / 3% SL):", "green")
        cprint(f"    Trades: {mom['trades']} | Wins: {mom['wins']} | Win Rate: {(mom['wins']/mom['trades']*100) if mom['trades'] > 0 else 0:.1f}%", "white")

        cprint(f"\n  Breakout Strategy:", "magenta")
        cprint(f"    Trades: {bo['trades']} | Wins: {bo['wins']} | Win Rate: {(bo['wins']/bo['trades']*100) if bo['trades'] > 0 else 0:.1f}%", "white")

        cprint(f"\n  MACD Bullish Strategy (10% TP / 7% SL):", "cyan")
        cprint(f"    Trades: {macd['trades']} | Wins: {macd['wins']} | Win Rate: {(macd['wins']/macd['trades']*100) if macd['trades'] > 0 else 0:.1f}%", "white")

        cprint(f"\n  BB Lower Bounce Strategy (8% TP / 5% SL):", "blue")
        cprint(f"    Trades: {bb['trades']} | Wins: {bb['wins']} | Win Rate: {(bb['wins']/bb['trades']*100) if bb['trades'] > 0 else 0:.1f}%", "white")

        cprint(f"\n  Gap and Go Strategy:", "yellow")
        cprint(f"    Trades: {gg['trades']} | Wins: {gg['wins']} | Win Rate: {(gg['wins']/gg['trades']*100) if gg['trades'] > 0 else 0:.1f}%", "white")

        cprint(f"\n  Mean Reversion Strategy:", "red")
        cprint(f"    Trades: {mr['trades']} | Wins: {mr['wins']} | Win Rate: {(mr['wins']/mr['trades']*100) if mr['trades'] > 0 else 0:.1f}%", "white")

        cprint(f"\n  Skipped (HOLD only): {self.strategy_stats['skipped']}", "white")
        cprint("=" * 60 + "\n", "cyan")

    def run(self, interval_seconds=60, duration_minutes=None):
        """Run the bot."""
        cprint("\nStarting Smart Bot...", "green")
        cprint(f"Check interval: {interval_seconds}s", "cyan")
        if duration_minutes:
            cprint(f"Duration: {duration_minutes} minutes", "cyan")

        start_time = datetime.now()
        self.check_symbols()

        schedule.every(interval_seconds).seconds.do(self.check_symbols)

        try:
            while True:
                if duration_minutes:
                    elapsed = (datetime.now() - start_time).total_seconds() / 60
                    if elapsed >= duration_minutes:
                        cprint(f"\nDuration of {duration_minutes} minutes reached.", "yellow")
                        break
                schedule.run_pending()
                time.sleep(1)
        except KeyboardInterrupt:
            cprint("\nBot stopped by user", "yellow")

        self.print_stats()


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Smart Trading Bot")
    parser.add_argument("--dry-run", action="store_true", help="Test mode without real trades")
    parser.add_argument("--mock", action="store_true", help="Use mock price data")
    parser.add_argument("--interval", type=int, default=60, help="Check interval in seconds")
    parser.add_argument("--duration", type=int, default=None, help="Stop after X minutes")
    parser.add_argument("--use-scanner", action="store_true", default=True, help="Dynamic symbol selection (default: enabled)")
    parser.add_argument("--no-scanner", action="store_true", help="Disable scanner, use static SYMBOLS list")
    parser.add_argument("--verbose", "-v", action="store_true", help="Verbose debug output for signal analysis")
    return parser.parse_args()


def main():
    """Entry point."""
    args = parse_args()

    # Pre-flight checklist (Moondev methodology)
    from src.utils.preflight_check import run_preflight
    if not run_preflight(mock_mode=args.mock, verbose=args.verbose):
        cprint("Pre-flight checks failed. Exiting.", "red")
        sys.exit(1)

    # Scanner is enabled by default unless --no-scanner is passed
    use_scanner = args.use_scanner and not args.no_scanner
    bot = SmartBot(
        dry_run=args.dry_run,
        mock_mode=args.mock,
        use_scanner=use_scanner,
        verbose=args.verbose
    )
    bot.run(interval_seconds=args.interval, duration_minutes=args.duration)


if __name__ == "__main__":
    main()
