"""
Alpaca Breakout Trading Bot

Momentum strategy for stocks that enters positions on 24-hour high/low breakouts.
- LONG: Price breaks above 24-hour high
- SHORT: Price breaks below 24-hour low (if shortable)
- Stop Loss: 1.5x ATR
- Take Profit: 2:1 risk/reward ratio

Usage:
    python src/agents/alpaca_breakout_bot.py              # Live trading
    python src/agents/alpaca_breakout_bot.py --dry-run   # Test mode (no real trades)
    python src/agents/alpaca_breakout_bot.py --mock      # Mock data (no API needed)
"""

import argparse
import os
import random
import sys
import time
from datetime import datetime, timedelta
from collections import defaultdict

import schedule
from termcolor import cprint
from dotenv import load_dotenv

sys.path.append('/Users/josiahgarcia/trading-bot')
load_dotenv()

# Risk management
from src.agents.risk_agent import check_risk_verbose, update_position, close_position, get_risk_status

# Exit management
from src.agents.exit_agent import manage_exit_verbose, reset_position, get_position_status

# Entry confirmation
from src.agents.entry_agent import check_entry

# Scanner for dynamic symbol selection
from src.agents.scanner import get_breakout_candidates

# Entry confidence threshold
MIN_ENTRY_CONFIDENCE = 60

# Scanner configuration
SCANNER_REFRESH_MINUTES = 30  # Refresh symbols every 30 minutes
SCANNER_TOP_N = 10            # Get top 10 candidates from scanner

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
# TSLA prioritized - performs best in backtests
SYMBOLS = ['TSLA', 'NVDA', 'AAPL', 'SPY']
ORDER_USD_SIZE = 500  # Per-trade size in USD
ATR_PERIOD = 14
ATR_MULTIPLIER = 1.5
RISK_REWARD_RATIO = 2.0

# Optimized via backtest (see src/backtesting/optimize_strategy.py)
# TSLA with TP=6%, SL=2% yielded +15.11% return
TAKE_PROFIT_PCT = 6.0  # 6% take profit
STOP_LOSS_PCT = 2.0    # 2% stop loss


class AlpacaBreakoutBot:
    """Bot that trades 24-hour high/low breakouts on stocks."""

    def __init__(self, dry_run=False, mock_mode=False, use_scanner=False):
        self.dry_run = dry_run
        self.mock_mode = mock_mode
        self.use_scanner = use_scanner
        self.api = None

        # Initialize Alpaca API
        if not mock_mode:
            if not ALPACA_AVAILABLE:
                raise RuntimeError("alpaca-trade-api package not installed")
            if not ALPACA_API_KEY or not ALPACA_SECRET_KEY:
                raise RuntimeError("ALPACA_API_KEY and ALPACA_SECRET_KEY required in .env")

            base_url = "https://paper-api.alpaca.markets" if ALPACA_PAPER else "https://api.alpaca.markets"
            self.api = tradeapi.REST(ALPACA_API_KEY, ALPACA_SECRET_KEY, base_url, api_version='v2')

        # Track positions and levels
        self.active_trades = {}
        self.cooldowns = {}
        self.cooldown_minutes = 60
        self.mock_cycle = 0

        # Scanner state
        self.current_symbols = SYMBOLS.copy()
        self.last_scan_time = None

        # Mock data state
        self.mock_prices = {}
        self.mock_candles = {}

        if self.dry_run:
            cprint("*** DRY RUN MODE - No real trades will be executed ***", "yellow")
        if self.mock_mode:
            cprint("*** MOCK MODE - Using simulated price data ***", "yellow")
            self._init_mock_data()
        else:
            account = self.api.get_account()
            cprint(f"Connected to Alpaca ({'PAPER' if ALPACA_PAPER else 'LIVE'})", "green")
            cprint(f"Account: {account.account_number}", "green")
            cprint(f"Equity: ${float(account.equity):,.2f}", "green")
            cprint(f"Buying Power: ${float(account.buying_power):,.2f}", "cyan")

        if self.use_scanner:
            cprint("*** SCANNER MODE - Dynamic symbol selection enabled ***", "yellow")
            cprint(f"Scanner: Top {SCANNER_TOP_N} candidates, refresh every {SCANNER_REFRESH_MINUTES} mins", "cyan")
            self._refresh_symbols()
        else:
            cprint(f"Trading symbols: {self.current_symbols}", "cyan")

        cprint(f"Order size: ${ORDER_USD_SIZE}", "cyan")
        cprint(f"TP: {TAKE_PROFIT_PCT}% | SL: {STOP_LOSS_PCT}% (optimized via backtest)", "cyan")

        # Display risk limits
        risk_status = get_risk_status()
        cprint(f"Risk limits: Daily loss ${-risk_status['daily_loss_limit']:,.0f} | Max position $5,000 | Max exposure $20,000", "cyan")
        cprint(f"Entry confirmation: Min confidence {MIN_ENTRY_CONFIDENCE}% (Volume + Trend + RSI)", "cyan")

    def _init_mock_data(self):
        """Initialize mock price data."""
        base_prices = {
            'SPY': 590.0,
            'AAPL': 240.0,
            'TSLA': 410.0,
            'NVDA': 140.0,
            'AMD': 125.0,
            'MSFT': 420.0,
            'GOOGL': 175.0,
            'META': 590.0,
            'AMZN': 220.0,
            'NFLX': 920.0,
        }

        for symbol in self.current_symbols:
            base = base_prices.get(symbol, 100 + random.uniform(0, 200))
            self.mock_prices[symbol] = base
            self.mock_candles[symbol] = self._generate_mock_candles(base, 24)

    def _refresh_symbols(self):
        """Refresh symbol list from scanner."""
        if not self.use_scanner:
            return

        # Check if we need to refresh (every 30 minutes)
        now = datetime.now()
        if self.last_scan_time:
            elapsed_mins = (now - self.last_scan_time).total_seconds() / 60
            if elapsed_mins < SCANNER_REFRESH_MINUTES:
                return

        cprint(f"\n--- Refreshing symbols from scanner ---", "yellow")

        try:
            # Get candidates from scanner
            candidates = get_breakout_candidates(
                top_n=SCANNER_TOP_N,
                threshold_pct=2.0,  # Within 2% of breakout
                mock_mode=self.mock_mode
            )

            if candidates:
                new_symbols = [c['symbol'] for c in candidates]

                # Keep any symbols we have active positions in
                for symbol in self.active_trades.keys():
                    if symbol not in new_symbols:
                        new_symbols.append(symbol)

                self.current_symbols = new_symbols
                self.last_scan_time = now

                # Initialize mock data for new symbols
                if self.mock_mode:
                    for symbol in new_symbols:
                        if symbol not in self.mock_prices:
                            base = 100 + random.uniform(0, 300)
                            self.mock_prices[symbol] = base
                            self.mock_candles[symbol] = self._generate_mock_candles(base, 24)

                cprint(f"Scanner found {len(candidates)} candidates:", "green")
                for c in candidates[:5]:
                    cprint(f"  {c['symbol']}: {c['nearest_distance_pct']:.2f}% to {c['nearest_level']} ({c['direction']})", "cyan")
                if len(candidates) > 5:
                    cprint(f"  ... and {len(candidates) - 5} more", "cyan")
            else:
                cprint("Scanner returned no candidates, keeping current symbols", "yellow")

        except Exception as e:
            cprint(f"Scanner error: {e}, keeping current symbols", "red")

    def _generate_mock_candles(self, base_price, num_candles):
        """Generate mock OHLCV candles."""
        candles = []
        price = base_price * random.uniform(0.97, 1.0)

        for i in range(num_candles):
            volatility = base_price * 0.008  # 0.8% volatility per candle
            open_price = price
            high = open_price + random.uniform(0, volatility * 2)
            low = open_price - random.uniform(0, volatility * 2)
            close = open_price + random.uniform(-volatility, volatility)

            candles.append({
                'open': open_price,
                'high': high,
                'low': low,
                'close': close,
                'volume': random.uniform(100000, 1000000)
            })
            price = close

        return candles

    def fetch_candles(self, symbol, hours=24):
        """Fetch hourly candles from Alpaca."""
        if self.mock_mode:
            return self.mock_candles.get(symbol, [])

        try:
            end = datetime.now()
            start = end - timedelta(hours=hours + 8)  # Extra buffer for market hours

            bars = self.api.get_bars(
                symbol,
                tradeapi.TimeFrame.Hour,
                start=start.isoformat(),
                end=end.isoformat(),
                limit=hours
            ).df

            if bars.empty:
                return []

            candles = []
            for idx, row in bars.iterrows():
                candles.append({
                    'open': float(row['open']),
                    'high': float(row['high']),
                    'low': float(row['low']),
                    'close': float(row['close']),
                    'volume': float(row['volume'])
                })

            return candles

        except Exception as e:
            cprint(f"Error fetching candles for {symbol}: {e}", "red")
            return []

    def get_current_price(self, symbol):
        """Get current price."""
        if self.mock_mode:
            base = self.mock_prices.get(symbol, 100)
            movement = base * random.uniform(-0.003, 0.003)
            self.mock_prices[symbol] = base + movement
            return self.mock_prices[symbol]

        try:
            trade = self.api.get_latest_trade(symbol)
            return float(trade.price)
        except Exception as e:
            cprint(f"Error getting price for {symbol}: {e}", "red")
            return None

    def calculate_24h_range(self, candles):
        """Calculate 24-hour high and low from candles."""
        if not candles:
            return None, None

        high_24h = max(c['high'] for c in candles)
        low_24h = min(c['low'] for c in candles)

        return high_24h, low_24h

    def calculate_atr(self, candles, period=ATR_PERIOD):
        """Calculate Average True Range."""
        if len(candles) < period + 1:
            if candles:
                ranges = [c['high'] - c['low'] for c in candles]
                return sum(ranges) / len(ranges) if ranges else 0
            return 0

        true_ranges = []
        for i in range(1, len(candles)):
            high = candles[i]['high']
            low = candles[i]['low']
            prev_close = candles[i-1]['close']

            tr = max(
                high - low,
                abs(high - prev_close),
                abs(low - prev_close)
            )
            true_ranges.append(tr)

        recent_tr = true_ranges[-period:]
        return sum(recent_tr) / len(recent_tr)

    def is_on_cooldown(self, symbol):
        """Check if symbol is on entry cooldown."""
        if symbol not in self.cooldowns:
            return False

        if datetime.now() > self.cooldowns[symbol]:
            del self.cooldowns[symbol]
            return False

        return True

    def set_cooldown(self, symbol):
        """Set cooldown after trade."""
        self.cooldowns[symbol] = datetime.now() + timedelta(minutes=self.cooldown_minutes)

    def check_breakout(self, symbol, current_price, high_24h, low_24h):
        """Check for breakout signals."""
        if current_price is None or high_24h is None or low_24h is None:
            return None

        buffer = high_24h * 0.001  # 0.1% buffer

        if current_price > high_24h + buffer:
            return 'LONG'
        elif current_price < low_24h - buffer:
            return 'SHORT'

        return None

    def calculate_levels(self, entry_price, atr, direction):
        """Calculate stop loss and take profit levels using optimized percentages."""
        # Use percentage-based levels (optimized via backtest)
        stop_distance = entry_price * (STOP_LOSS_PCT / 100)
        profit_distance = entry_price * (TAKE_PROFIT_PCT / 100)

        if direction == 'LONG':
            stop_loss = entry_price - stop_distance
            take_profit = entry_price + profit_distance
        else:
            stop_loss = entry_price + stop_distance
            take_profit = entry_price - profit_distance

        return stop_loss, take_profit

    def get_position(self, symbol):
        """Get current position for symbol."""
        if self.mock_mode:
            return None

        try:
            position = self.api.get_position(symbol)
            return {
                'qty': int(position.qty),
                'side': 'long' if int(position.qty) > 0 else 'short',
                'entry': float(position.avg_entry_price),
                'market_value': float(position.market_value),
                'unrealized_pl': float(position.unrealized_pl),
                'unrealized_plpc': float(position.unrealized_plpc) * 100
            }
        except:
            return None

    def enter_position(self, symbol, direction, entry_price, stop_loss, take_profit):
        """Enter a position."""
        try:
            if self.is_on_cooldown(symbol):
                cprint(f"{symbol} on cooldown, skipping entry", "yellow")
                return False

            # Calculate shares (fractional shares supported)
            shares = round(ORDER_USD_SIZE / entry_price, 4)
            if shares < 0.0001:
                cprint(f"Order size too small for {symbol}", "yellow")
                return False

            # Risk check before entry
            position_value = shares * entry_price
            risk_allowed, risk_reason = check_risk_verbose(symbol, position_value)
            if not risk_allowed:
                cprint(f"  RISK BLOCKED: {risk_reason}", "red")
                return False

            # Mock mode
            if self.mock_mode:
                cprint(f"[MOCK] Entering {direction}: {symbol} | {shares} shares @ ${entry_price:.2f}", "magenta")
                cprint(f"[MOCK] SL: ${stop_loss:.2f} | TP: ${take_profit:.2f}", "cyan")
                self.active_trades[symbol] = {
                    'entry': entry_price,
                    'entry_time': datetime.now(),
                    'stop_loss': stop_loss,
                    'take_profit': take_profit,
                    'direction': direction,
                    'shares': shares,
                    'original_shares': shares
                }
                update_position(symbol, position_value)  # Track for risk management
                self.set_cooldown(symbol)
                return True

            # Check existing position
            existing = self.get_position(symbol)
            if existing:
                cprint(f"Already in position for {symbol}, skipping entry", "yellow")
                return False

            cprint(f"Entering {direction}: {symbol} | {shares} shares @ ${entry_price:.2f}", "magenta")
            cprint(f"SL: ${stop_loss:.2f} | TP: ${take_profit:.2f}", "cyan")

            if self.dry_run:
                cprint(f"[DRY RUN] Would enter {direction} {symbol} with {shares} shares", "cyan")
                self.active_trades[symbol] = {
                    'entry': entry_price,
                    'entry_time': datetime.now(),
                    'stop_loss': stop_loss,
                    'take_profit': take_profit,
                    'direction': direction,
                    'shares': shares,
                    'original_shares': shares
                }
                update_position(symbol, position_value)  # Track for risk management
                self.set_cooldown(symbol)
                return True

            # Execute trade
            side = 'buy' if direction == 'LONG' else 'sell'
            order = self.api.submit_order(
                symbol=symbol,
                qty=shares,
                side=side,
                type='market',
                time_in_force='day'
            )

            cprint(f"Order submitted: {order.id}", "green")
            self.active_trades[symbol] = {
                'entry': entry_price,
                'entry_time': datetime.now(),
                'stop_loss': stop_loss,
                'take_profit': take_profit,
                'direction': direction,
                'shares': shares,
                'original_shares': shares,
                'order_id': order.id
            }
            update_position(symbol, position_value)  # Track for risk management
            self.set_cooldown(symbol)
            return True

        except Exception as e:
            cprint(f"Error entering position for {symbol}: {e}", "red")
            return False

    def manage_position(self, symbol, current_price):
        """Check and manage active position using exit agent."""
        if symbol not in self.active_trades:
            return

        trade = self.active_trades[symbol]
        direction = trade['direction']
        stop_loss = trade['stop_loss']
        take_profit = trade['take_profit']
        entry = trade['entry']
        entry_time = trade.get('entry_time', datetime.now())
        shares = trade['shares']

        # First check original SL/TP (these are hard stops)
        should_close_full = False
        close_reason = None

        if direction == 'LONG':
            if current_price <= stop_loss:
                should_close_full = True
                close_reason = f"SL hit (${current_price:.2f} <= ${stop_loss:.2f})"
            elif current_price >= take_profit:
                should_close_full = True
                close_reason = f"TP hit (${current_price:.2f} >= ${take_profit:.2f})"
        else:
            if current_price >= stop_loss:
                should_close_full = True
                close_reason = f"SL hit (${current_price:.2f} >= ${stop_loss:.2f})"
            elif current_price <= take_profit:
                should_close_full = True
                close_reason = f"TP hit (${current_price:.2f} <= ${take_profit:.2f})"

        # If hard SL/TP not hit, consult exit agent for smart exits
        if not should_close_full:
            exit_decision, exit_reason, new_stop = manage_exit_verbose(
                symbol, entry, entry_time, current_price,
                stop_loss, take_profit, direction
            )

            if exit_decision == 'CLOSE_FULL':
                should_close_full = True
                close_reason = exit_reason
            elif exit_decision == 'CLOSE_HALF':
                self._close_partial(symbol, current_price, exit_reason)
                return
            elif exit_decision == 'MOVE_STOP':
                self.active_trades[symbol]['stop_loss'] = new_stop
                cprint(f"  {symbol}: {exit_reason}", "green")
                return
            else:
                # HOLD - do nothing
                return

        # Execute full close
        if should_close_full:
            self._close_full(symbol, current_price, close_reason)

    def _close_partial(self, symbol, current_price, reason):
        """Close 50% of position for partial profit."""
        trade = self.active_trades[symbol]
        direction = trade['direction']
        entry = trade['entry']
        shares = trade['shares']

        # Close half
        close_shares = round(shares / 2, 4)
        remaining_shares = round(shares - close_shares, 4)

        pnl_pct = ((current_price - entry) / entry * 100) if direction == 'LONG' else ((entry - current_price) / entry * 100)
        pnl_dollars = (current_price - entry) * close_shares if direction == 'LONG' else (entry - current_price) * close_shares

        if self.mock_mode or self.dry_run:
            mode = 'MOCK' if self.mock_mode else 'DRY RUN'
            cprint(f"[{mode}] PARTIAL CLOSE {symbol}: {close_shares} shares @ ${current_price:.2f} | PnL: {pnl_pct:+.2f}% (${pnl_dollars:+.2f})", "magenta")
            cprint(f"  Reason: {reason}", "cyan")
            cprint(f"  Remaining: {remaining_shares} shares", "white")

            # Update trade with remaining shares
            self.active_trades[symbol]['shares'] = remaining_shares
            # Update risk tracking for closed portion
            close_position(symbol, pnl_dollars)
            update_position(symbol, remaining_shares * current_price)
            return

        try:
            side = 'sell' if direction == 'LONG' else 'buy'
            order = self.api.submit_order(
                symbol=symbol,
                qty=close_shares,
                side=side,
                type='market',
                time_in_force='day'
            )

            cprint(f"PARTIAL CLOSE {symbol}: {close_shares} shares @ ${current_price:.2f} | PnL: {pnl_pct:+.2f}% (${pnl_dollars:+.2f})", "magenta")
            cprint(f"  Reason: {reason}", "cyan")
            cprint(f"  Remaining: {remaining_shares} shares", "white")

            self.active_trades[symbol]['shares'] = remaining_shares
            close_position(symbol, pnl_dollars)
            update_position(symbol, remaining_shares * current_price)

        except Exception as e:
            cprint(f"Error partial closing {symbol}: {e}", "red")

    def _close_full(self, symbol, current_price, reason):
        """Close entire position."""
        trade = self.active_trades[symbol]
        direction = trade['direction']
        entry = trade['entry']
        shares = trade['shares']

        pnl_pct = ((current_price - entry) / entry * 100) if direction == 'LONG' else ((entry - current_price) / entry * 100)
        pnl_dollars = (current_price - entry) * shares if direction == 'LONG' else (entry - current_price) * shares

        if self.mock_mode or self.dry_run:
            mode = 'MOCK' if self.mock_mode else 'DRY RUN'
            color = "green" if pnl_pct > 0 else "red"
            cprint(f"[{mode}] FULL CLOSE {symbol} {direction}: {reason} | PnL: {pnl_pct:+.2f}% (${pnl_dollars:+.2f})", color)
            close_position(symbol, pnl_dollars)
            reset_position(symbol)  # Clear exit agent state
            del self.active_trades[symbol]
            return

        try:
            side = 'sell' if direction == 'LONG' else 'buy'
            order = self.api.submit_order(
                symbol=symbol,
                qty=shares,
                side=side,
                type='market',
                time_in_force='day'
            )

            color = "green" if pnl_pct > 0 else "red"
            cprint(f"FULL CLOSE {symbol} {direction}: {reason} | PnL: {pnl_pct:+.2f}% (${pnl_dollars:+.2f})", color)
            close_position(symbol, pnl_dollars)
            reset_position(symbol)  # Clear exit agent state
            del self.active_trades[symbol]

        except Exception as e:
            cprint(f"Error closing position for {symbol}: {e}", "red")

    def update_mock_candles(self):
        """Update mock candles with new data."""
        for symbol in self.current_symbols:
            if symbol in self.mock_candles and self.mock_candles[symbol]:
                last_close = self.mock_candles[symbol][-1]['close']
                volatility = last_close * 0.008

                if random.random() < 0.15:
                    volatility *= 3

                new_candle = {
                    'open': last_close,
                    'high': last_close + random.uniform(0, volatility * 2),
                    'low': last_close - random.uniform(0, volatility * 2),
                    'close': last_close + random.uniform(-volatility, volatility),
                    'volume': random.uniform(100000, 1000000)
                }

                self.mock_candles[symbol] = self.mock_candles[symbol][-23:] + [new_candle]
                self.mock_prices[symbol] = new_candle['close']

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
        """Main loop - check all symbols for breakouts."""
        # Refresh symbols from scanner if enabled
        if self.use_scanner:
            self._refresh_symbols()

        risk_status = get_risk_status()
        symbols_str = f" | Symbols: {len(self.current_symbols)}" if self.use_scanner else ""
        cprint(f"\n--- Check cycle {self.mock_cycle} at {datetime.now().strftime('%H:%M:%S')} | Daily P&L: ${risk_status['daily_pnl']:+,.2f} | Exposure: ${risk_status['total_exposure']:,.0f}{symbols_str} ---", "cyan")

        if not risk_status['trading_allowed']:
            cprint("TRADING HALTED: Daily loss limit reached!", "red")
            self.mock_cycle += 1
            return

        if not self.mock_mode and not self.is_market_open():
            cprint("Market is closed", "yellow")
            self.mock_cycle += 1
            return

        if self.mock_mode:
            self.update_mock_candles()

        for symbol in self.current_symbols:
            candles = self.fetch_candles(symbol, hours=24)

            if not candles:
                cprint(f"{symbol}: No candle data available", "yellow")
                continue

            high_24h, low_24h = self.calculate_24h_range(candles)
            atr = self.calculate_atr(candles)
            current_price = self.get_current_price(symbol)

            if current_price is None:
                continue

            range_pct = ((high_24h - low_24h) / low_24h * 100) if low_24h else 0
            cprint(
                f"{symbol}: ${current_price:.2f} | 24h: ${low_24h:.2f} - ${high_24h:.2f} ({range_pct:.1f}%) | ATR: ${atr:.2f}",
                "white"
            )

            # Check active position management
            if symbol in self.active_trades:
                trade = self.active_trades[symbol]
                pnl_pct = ((current_price - trade['entry']) / trade['entry'] * 100) if trade['direction'] == 'LONG' else ((trade['entry'] - current_price) / trade['entry'] * 100)
                exit_status = get_position_status(symbol)
                status_str = ""
                if exit_status['partial_taken']:
                    status_str += " [PARTIAL]"
                if exit_status['trailing_active']:
                    status_str += f" [TRAIL@${exit_status['trailing_stop']:.2f}]" if exit_status['trailing_stop'] else " [TRAIL]"
                elif exit_status['breakeven_triggered']:
                    status_str += " [BE]"
                cprint(f"  Active {trade['direction']}: Entry ${trade['entry']:.2f} | {trade['shares']} shares | PnL: {pnl_pct:+.2f}%{status_str}", "magenta")
                self.manage_position(symbol, current_price)
                continue

            # Check for breakout
            signal = self.check_breakout(symbol, current_price, high_24h, low_24h)

            if signal:
                # Only short if shorting is enabled (skip for now, stocks harder to short)
                if signal == 'SHORT':
                    cprint(f"  SHORT signal for {symbol} - skipping (long-only mode)", "yellow")
                    continue

                cprint(f"  BREAKOUT {signal}: {symbol} @ ${current_price:.2f}", "yellow")

                # Entry confirmation check
                should_enter, confidence, reasons = check_entry(
                    symbol, signal, current_price, candles, None
                )

                if not should_enter or confidence < MIN_ENTRY_CONFIDENCE:
                    cprint(f"  ENTRY REJECTED: Confidence {confidence}% < {MIN_ENTRY_CONFIDENCE}%", "red")
                    for reason in reasons:
                        cprint(f"    {reason}", "white")
                    continue

                cprint(f"  ENTRY CONFIRMED: Confidence {confidence}%", "green")
                for reason in reasons:
                    cprint(f"    {reason}", "cyan")

                stop_loss, take_profit = self.calculate_levels(current_price, atr, signal)
                self.enter_position(symbol, signal, current_price, stop_loss, take_profit)

        self.mock_cycle += 1

    def run(self, interval_seconds=60, duration_minutes=None):
        """Run the bot with scheduled checks."""
        cprint("\nStarting Alpaca Breakout Bot...", "green")
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
                        cprint(f"\nDuration of {duration_minutes} minutes reached. Stopping bot.", "yellow")
                        break
                schedule.run_pending()
                time.sleep(1)
        except KeyboardInterrupt:
            cprint("\nBot stopped by user", "yellow")


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Alpaca Breakout Trading Bot"
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Run in test mode without executing real trades"
    )
    parser.add_argument(
        "--mock",
        action="store_true",
        help="Use mock price data (no API calls needed)"
    )
    parser.add_argument(
        "--interval",
        type=int,
        default=60,
        help="Check interval in seconds (default: 60)"
    )
    parser.add_argument(
        "--duration",
        type=int,
        default=None,
        help="Stop the bot after X minutes (default: run indefinitely)"
    )
    parser.add_argument(
        "--use-scanner",
        action="store_true",
        help="Use scanner to dynamically select symbols near breakout (refreshes every 30 mins)"
    )
    return parser.parse_args()


def main():
    """Entry point."""
    args = parse_args()
    bot = AlpacaBreakoutBot(
        dry_run=args.dry_run,
        mock_mode=args.mock,
        use_scanner=args.use_scanner
    )
    bot.run(interval_seconds=args.interval, duration_minutes=args.duration)


if __name__ == "__main__":
    main()
