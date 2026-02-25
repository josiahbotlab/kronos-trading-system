"""
Base Trading Bot — MoonDev Methodology

Shared foundation for all strategy bots. Handles:
- Alpaca API connection (via order_utils.py)
- Market hours check
- Position check (prevent duplicates)
- Account check (buying power)
- Per-bot logging to logs/{bot_name}.log
- Trade journal integration with bot_name column
- Cooldown tracking
- Data fetching (candles, current price, price history)
- Entry execution flow (safety checks + bracket orders)
- Main run loop with configurable interval

Subclasses override:
    BOT_NAME, STRATEGY, DEFAULT_SYMBOLS, TP_PCT, SL_PCT
    check_signal(symbol, price, prices, candles) -> (bool, dict)

Usage:
    # Subclass implements check_signal(), then:
    python bots/momentum_bot.py --dry-run --symbols AMD,NVDA
"""

import argparse
import logging
import os
import sys
import time
from abc import ABC, abstractmethod
from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np
from termcolor import cprint
from dotenv import load_dotenv

# Project root setup
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
load_dotenv(PROJECT_ROOT / '.env')

# Order execution utilities
from src.utils.order_utils import (
    get_api,
    check_existing_position,
    cancel_symbol_orders,
    place_bracket_order,
    round_to_valid_qty,
    check_daily_loss_limit,
    reset_daily_tracker,
    get_daily_pnl,
    is_trading_halted,
    get_account_info,
    get_all_positions,
    ensure_exit_protection,
)

# Trade journal
from src.utils.trade_journal import log_trade, reconcile_exits

# SQLite trade journal
from src.utils.stock_journal import StockJournal

# Market regime detector
from src.utils.regime_detector import detect_market_regime

# Alpaca API
try:
    import alpaca_trade_api as tradeapi
    ALPACA_AVAILABLE = True
except ImportError:
    ALPACA_AVAILABLE = False

# ta library for indicators
try:
    from ta.trend import SMAIndicator, EMAIndicator
    from ta.momentum import RSIIndicator
    from ta.volatility import BollingerBands as TABollingerBands
    from ta.trend import MACD as TALibMACD
    import pandas as pd
    TA_AVAILABLE = True
except ImportError:
    TA_AVAILABLE = False
    import pandas as pd


# ─────────────────────────────────────────────────────────────
# GLOBAL DEFAULTS
# ─────────────────────────────────────────────────────────────

DEFAULT_ORDER_USD = 500
DEFAULT_COOLDOWN_MINUTES = 60
EARLIEST_ENTRY_ET = "10:30"
SCAN_INTERVAL_SECONDS = 300  # 5 minutes between full scans


# ─────────────────────────────────────────────────────────────
# INDICATOR HELPERS (used by subclasses)
# ─────────────────────────────────────────────────────────────

def calc_rsi(prices, period=14):
    """Calculate RSI from a list of close prices."""
    if len(prices) < period + 1:
        return 50.0
    deltas = np.diff(prices)
    gains = np.where(deltas > 0, deltas, 0)
    losses = np.where(deltas < 0, -deltas, 0)
    avg_gain = np.mean(gains[-period:])
    avg_loss = np.mean(losses[-period:])
    if avg_loss == 0:
        return 100.0
    rs = avg_gain / avg_loss
    return 100.0 - (100.0 / (1.0 + rs))


def calc_sma(prices, period=20):
    """Calculate Simple Moving Average."""
    if len(prices) < period:
        return np.mean(prices) if len(prices) > 0 else 0.0
    return float(np.mean(prices[-period:]))


def calc_ema(prices, period=9):
    """Calculate Exponential Moving Average."""
    if len(prices) < period:
        return np.mean(prices) if len(prices) > 0 else 0.0
    if TA_AVAILABLE:
        close = pd.Series(prices)
        ema = EMAIndicator(close, window=period)
        return float(ema.ema_indicator().iloc[-1])
    # Manual EMA
    multiplier = 2.0 / (period + 1)
    ema_val = float(prices[0])
    for p in prices[1:]:
        ema_val = (float(p) * multiplier) + (ema_val * (1 - multiplier))
    return ema_val


def calc_bollinger_bands(prices, period=20, std_dev=2.0):
    """Calculate Bollinger Bands. Returns (upper, middle, lower)."""
    if len(prices) < period:
        return None, None, None
    if TA_AVAILABLE:
        close = pd.Series(prices)
        bb = TABollingerBands(close, window=period, window_dev=std_dev)
        return (
            float(bb.bollinger_hband().iloc[-1]),
            float(bb.bollinger_mavg().iloc[-1]),
            float(bb.bollinger_lband().iloc[-1]),
        )
    arr = np.array(prices[-period:])
    middle = float(np.mean(arr))
    std = float(np.std(arr))
    return middle + std_dev * std, middle, middle - std_dev * std


def calc_macd(prices, fast=12, slow=26, signal=9):
    """Calculate MACD. Returns (macd_line, signal_line, histogram)."""
    if len(prices) < slow + signal:
        return None, None, None
    if TA_AVAILABLE:
        close = pd.Series(prices)
        macd = TALibMACD(close, window_fast=fast, window_slow=slow, window_sign=signal)
        return (
            float(macd.macd().iloc[-1]),
            float(macd.macd_signal().iloc[-1]),
            float(macd.macd_diff().iloc[-1]),
        )
    # Manual fallback
    def _manual_ema(arr, p):
        mult = 2.0 / (p + 1)
        e = arr[0]
        for v in arr[1:]:
            e = v * mult + e * (1 - mult)
        return e

    arr = np.array(prices, dtype=float)
    ema_fast = _manual_ema(arr, fast)
    ema_slow = _manual_ema(arr, slow)
    macd_val = ema_fast - ema_slow
    return macd_val, macd_val, 0.0  # Simplified fallback


# ─────────────────────────────────────────────────────────────
# BASE BOT CLASS
# ─────────────────────────────────────────────────────────────

class BaseBot(ABC):
    """
    Abstract base class for all strategy bots.

    Subclasses MUST define:
        BOT_NAME:   str  — e.g. "momentum_bot"
        STRATEGY:   str  — e.g. "MOMENTUM"
        DEFAULT_SYMBOLS: list
        TP_PCT:     float
        SL_PCT:     float

    Subclasses MUST implement:
        check_signal(symbol, price, prices, candles) -> (bool, dict)
            Returns (should_enter, metadata_dict).
            metadata_dict is logged and can contain any debug info.
    """

    # Subclasses override these
    BOT_NAME = "base_bot"
    STRATEGY = "UNKNOWN"
    DEFAULT_SYMBOLS = ['TSLA', 'AMD', 'NVDA', 'AAPL', 'QQQ']
    TP_PCT = 6.0
    SL_PCT = 3.0

    def __init__(self, dry_run=False, symbols=None, order_size=None):
        self.dry_run = dry_run
        self.symbols = symbols or self.DEFAULT_SYMBOLS
        self.order_size = order_size or DEFAULT_ORDER_USD
        self.api = None
        self.active_trades = {}
        self.cooldowns = {}
        self.cooldown_minutes = DEFAULT_COOLDOWN_MINUTES
        self.cycle_count = 0
        self.trades_entered_today = 0
        self.market_was_open = False

        # Tunable parameters (wraps class-level constants)
        self._params = {
            'tp_pct': self.TP_PCT,
            'sl_pct': self.SL_PCT,
            'order_size': self.order_size,
            'cooldown_minutes': DEFAULT_COOLDOWN_MINUTES,
        }

        # SQLite trade journal
        try:
            self.stock_journal = StockJournal()
        except Exception:
            self.stock_journal = None

        # Current market regime (set each scan_once)
        self._current_regime = "unknown"

        # Skill file regime gating cache
        self._regime_matrix_cache = None
        self._regime_matrix_mtime = 0

        # Per-bot logger
        self.logger = self._setup_logger()

        # Connect to Alpaca
        if not ALPACA_AVAILABLE:
            raise RuntimeError("alpaca-trade-api package not installed")
        self.api = get_api()

        # Verify connection
        try:
            account = self.api.get_account()
            equity = float(account.equity)
            paper = "PAPER" if os.getenv("ALPACA_PAPER", "true").lower() == "true" else "LIVE"
            self._log(f"Connected to Alpaca ({paper}) | Equity: ${equity:,.2f}")
        except Exception as e:
            raise RuntimeError(f"Alpaca connection failed: {e}")

        # Initialize daily loss tracker
        reset_daily_tracker()

        # Sync existing positions for this bot's symbols
        self._sync_positions()

        # Reconcile exits from Alpaca fills (log TP/SL fills to journal)
        try:
            reconciled = reconcile_exits(hours=24, stock_journal=self.stock_journal)
            if reconciled > 0:
                self._log(f"Reconciled {reconciled} exit(s) from Alpaca fills")
        except Exception as e:
            self._log(f"Exit reconciliation failed: {e}", "yellow", "warning")

        if self.dry_run:
            self._log("*** DRY RUN MODE — no real trades ***", "yellow")

    # ─────────────────────────────────────────────────────────
    # LOGGING
    # ─────────────────────────────────────────────────────────

    def _setup_logger(self):
        """Create per-bot log file at logs/{bot_name}.log."""
        log_dir = PROJECT_ROOT / 'logs'
        log_dir.mkdir(exist_ok=True)
        log_path = log_dir / f'{self.BOT_NAME}.log'

        logger = logging.getLogger(self.BOT_NAME)
        logger.setLevel(logging.DEBUG)

        # Prevent duplicate handlers on re-init
        if not logger.handlers:
            fh = logging.FileHandler(log_path)
            fh.setLevel(logging.DEBUG)
            fh.setFormatter(logging.Formatter(
                '%(asctime)s [%(levelname)s] %(message)s',
                datefmt='%Y-%m-%d %H:%M:%S'
            ))
            logger.addHandler(fh)

        return logger

    def _log(self, msg, color="white", level="info"):
        """Print to console with color AND write to log file."""
        cprint(f"  [{self.BOT_NAME}] {msg}", color)
        getattr(self.logger, level)(msg)

    # ─────────────────────────────────────────────────────────
    # MARKET HOURS
    # ─────────────────────────────────────────────────────────

    def is_market_open(self):
        """Check if market is currently open."""
        try:
            clock = self.api.get_clock()
            return clock.is_open
        except Exception as e:
            self._log(f"Clock check failed: {e}", "red", "error")
            return False

    def get_market_time(self):
        """Get current market (ET) time from Alpaca clock."""
        try:
            clock = self.api.get_clock()
            return clock.timestamp
        except Exception:
            return None

    def is_before_entry_time(self):
        """Check if current time is before EARLIEST_ENTRY_ET."""
        if not EARLIEST_ENTRY_ET:
            return False
        try:
            et_now = self.get_market_time()
            if et_now is None:
                return False
            hour, minute = map(int, EARLIEST_ENTRY_ET.split(':'))
            return et_now.hour < hour or (et_now.hour == hour and et_now.minute < minute)
        except Exception:
            return False

    # ─────────────────────────────────────────────────────────
    # DATA FETCHING
    # ─────────────────────────────────────────────────────────

    def fetch_candles(self, symbol, hours=50):
        """Fetch hourly OHLCV candles from Alpaca.

        Uses calendar-day lookback (not hours) so weekends/holidays are
        covered.  Returns the most recent *hours* candles.
        """
        try:
            # Convert desired hourly bars to calendar days.
            # ~7 trading hours/day → divide by 7, add padding for
            # weekends (2 days per 5) and holidays.
            cal_days = max((hours // 7) * 2 + 4, 10)

            end = datetime.now(timezone.utc)
            start = end - timedelta(days=cal_days)

            bars = self.api.get_bars(
                symbol,
                tradeapi.TimeFrame.Hour,
                start=start.isoformat(),
                end=end.isoformat(),
                feed='iex'
            ).df

            if bars.empty:
                return []

            candles = [
                {
                    'open': float(row['open']),
                    'high': float(row['high']),
                    'low': float(row['low']),
                    'close': float(row['close']),
                    'volume': float(row['volume']),
                }
                for _, row in bars.iterrows()
            ]

            # Return only the most recent N candles
            return candles[-hours:] if len(candles) > hours else candles
        except Exception as e:
            self._log(f"Error fetching candles for {symbol}: {e}", "red", "error")
            return []

    def get_current_price(self, symbol):
        """Get latest trade price from Alpaca."""
        try:
            trade = self.api.get_latest_trade(symbol)
            return float(trade.price)
        except Exception as e:
            self._log(f"Error getting price for {symbol}: {e}", "red", "error")
            return None

    def get_price_history(self, symbol, hours=50):
        """Get close-price list for indicator calculations."""
        candles = self.fetch_candles(symbol, hours=hours)
        return [c['close'] for c in candles]

    # ─────────────────────────────────────────────────────────
    # COOLDOWN
    # ─────────────────────────────────────────────────────────

    def is_on_cooldown(self, symbol):
        """Check if symbol is on cooldown after a recent trade."""
        if symbol not in self.cooldowns:
            return False
        if datetime.now() > self.cooldowns[symbol]:
            del self.cooldowns[symbol]
            return False
        remaining = (self.cooldowns[symbol] - datetime.now()).total_seconds() / 60
        self._log(f"{symbol} on cooldown ({remaining:.0f} min remaining)", "yellow")
        return True

    def set_cooldown(self, symbol):
        """Set cooldown after placing a trade."""
        self.cooldowns[symbol] = datetime.now() + timedelta(minutes=self.cooldown_minutes)

    # ─────────────────────────────────────────────────────────
    # TUNABLE PARAMETERS
    # ─────────────────────────────────────────────────────────

    def get_param(self, key):
        """Get a tunable parameter value."""
        return self._params.get(key)

    def set_param(self, key, value):
        """Set a tunable parameter value (syncs back to instance attrs)."""
        self._params[key] = value
        if key == 'tp_pct':
            self.TP_PCT = value
        elif key == 'sl_pct':
            self.SL_PCT = value
        elif key == 'order_size':
            self.order_size = value
        elif key == 'cooldown_minutes':
            self.cooldown_minutes = value

    # ─────────────────────────────────────────────────────────
    # SKILL FILE REGIME GATING
    # ─────────────────────────────────────────────────────────

    def _load_regime_matrix(self):
        """Parse regime matrix from skill file, cached by file mtime."""
        skill_path = PROJECT_ROOT / 'skills' / 'strategy_performance.md'
        if not skill_path.exists():
            return {}

        try:
            mtime = skill_path.stat().st_mtime
            if mtime == self._regime_matrix_mtime and self._regime_matrix_cache is not None:
                return self._regime_matrix_cache

            with open(skill_path, 'r') as f:
                content = f.read()

            # Find Regime Matrix section
            matrix = {}
            in_matrix = False
            headers = []

            for line in content.split('\n'):
                if '## Regime Matrix' in line:
                    in_matrix = True
                    continue
                if in_matrix and line.startswith('##'):
                    break
                if not in_matrix:
                    continue
                if not line.strip() or line.startswith('|--'):
                    continue

                cells = [c.strip() for c in line.split('|') if c.strip()]
                if not cells:
                    continue

                if not headers:
                    headers = cells  # First row: "Strategy", regime1, regime2, ...
                    continue

                strategy = cells[0]
                matrix[strategy] = {}
                for i, regime in enumerate(headers[1:], 1):
                    if i < len(cells):
                        cell = cells[i]
                        if cell == '—' or not cell:
                            continue
                        # Parse "65% (n=12)" format
                        try:
                            wr_str = cell.split('%')[0]
                            wr = float(wr_str)
                            n = int(cell.split('n=')[1].split(')')[0])
                            matrix[strategy][regime] = {'wr': wr, 'n': n}
                        except (ValueError, IndexError):
                            pass

            self._regime_matrix_cache = matrix
            self._regime_matrix_mtime = mtime
            return matrix
        except Exception:
            return {}

    def _is_qualified_for_regime(self, regime):
        """Check if this bot's strategy is qualified to trade in the current regime."""
        try:
            matrix = self._load_regime_matrix()
            if not matrix:
                return True  # No data = allow

            strategy = self.STRATEGY
            if strategy not in matrix:
                return True  # No data for this strategy = allow

            regime_data = matrix[strategy].get(regime)
            if regime_data is None:
                return True  # No data for this regime = allow

            if regime_data['n'] < 5:
                return True  # Insufficient sample = allow

            return regime_data['wr'] >= 45.0
        except Exception:
            return True  # Parse error = allow

    # ─────────────────────────────────────────────────────────
    # POSITION SYNC
    # ─────────────────────────────────────────────────────────

    def _sync_positions(self):
        """Sync bot state with existing Alpaca positions for this bot's symbols."""
        try:
            positions = get_all_positions()
            synced = 0
            for pos in positions:
                symbol = pos['symbol']
                if symbol in self.symbols and symbol not in self.active_trades:
                    entry_price = pos['avg_entry_price']
                    qty = pos['qty']
                    direction = 'LONG' if pos['side'] == 'long' else 'SHORT'
                    stop_loss = entry_price * (1 - self.SL_PCT / 100)
                    take_profit = entry_price * (1 + self.TP_PCT / 100)

                    self.active_trades[symbol] = {
                        'entry': entry_price,
                        'entry_time': datetime.now(),
                        'stop_loss': stop_loss,
                        'take_profit': take_profit,
                        'direction': direction,
                        'shares': qty,
                        'strategy': self.STRATEGY,
                        'is_synced': True,
                    }
                    color = "green" if pos['unrealized_pnl'] >= 0 else "red"
                    self._log(
                        f"Synced {symbol}: {qty:.0f} shares @ ${entry_price:.2f} "
                        f"(P&L: ${pos['unrealized_pnl']:+.2f}) "
                        f"SL: ${stop_loss:.2f} / TP: ${take_profit:.2f}",
                        color
                    )
                    synced += 1

            if synced == 0:
                self._log("No existing positions to sync for this bot's symbols")
            else:
                self._log(f"Synced {synced} existing position(s)")

            # SAFETY CHECK: Ensure all positions have exit protection (TP or SL)
            bot_positions = [p for p in positions if p['symbol'] in self.symbols]
            if bot_positions:
                self._log("Checking exit protection for positions...", "cyan")
                result = ensure_exit_protection(
                    bot_positions,
                    sl_pct=self.SL_PCT,
                    tp_pct=self.TP_PCT
                )
                if result['added_sl'] > 0:
                    self._log(f"Added {result['added_sl']} stop-loss order(s)", "yellow")
                if result['added_tp'] > 0:
                    self._log(f"Added {result['added_tp']} take-profit order(s)", "cyan")
                if result['failed'] > 0:
                    self._log(f"FAILED to add {result['failed']} exit order(s)!", "red", "error")

        except Exception as e:
            self._log(f"Error syncing positions: {e}", "red", "error")

    # ─────────────────────────────────────────────────────────
    # ENTRY EXECUTION
    # ─────────────────────────────────────────────────────────

    def execute_entry(self, symbol, entry_price, direction='LONG', signal_strength=None, reasoning=None):
        """
        Execute trade entry with full safety checks.

        Flow:
        1. Cooldown check
        2. Entry time restriction (before 10:30 AM ET)
        3. Daily loss limit
        4. Existing position check (prevent duplicates)
        5. Cancel stale orders
        6. Calculate & validate qty (whole shares for brackets)
        7. Place bracket order with TP/SL
        8. Log to trade journal with bot_name
        """
        # 1. Cooldown
        if self.is_on_cooldown(symbol):
            return False

        # 2. Entry time
        if self.is_before_entry_time():
            et_now = self.get_market_time()
            time_str = et_now.strftime('%H:%M') if et_now else '??:??'
            self._log(f"Skipping {symbol} — before {EARLIEST_ENTRY_ET} ET ({time_str} ET)", "yellow")
            return False

        # 3. Daily loss limit
        if not check_daily_loss_limit():
            self._log(f"{symbol} BLOCKED: Daily loss limit reached", "red")
            return False

        # 4. Existing position
        has_position, position_info = check_existing_position(symbol)
        if has_position:
            self._log(
                f"{symbol} SKIPPED: Already in position "
                f"({position_info['qty']} shares {position_info['side']})",
                "yellow"
            )
            return False

        # 5. Cancel stale orders
        cancelled = cancel_symbol_orders(symbol)
        if cancelled > 0:
            self._log(f"Cancelled {cancelled} stale orders for {symbol}", "yellow")

        # 6. Calculate quantity — bracket orders need whole shares
        raw_shares = self.order_size / entry_price
        shares = int(raw_shares)  # Floor to whole number
        if shares == 0:
            if entry_price <= self.order_size * 2:
                shares = 1
                self._log(f"{symbol}: Using 1 share (${entry_price:.2f}/share)", "yellow")
            else:
                self._log(
                    f"{symbol} SKIPPED: ${entry_price:.2f}/share too high "
                    f"for ${self.order_size} position",
                    "yellow"
                )
                return False

        # Compute TP/SL prices (use tunable params)
        tp_pct = self.get_param('tp_pct') or self.TP_PCT
        sl_pct = self.get_param('sl_pct') or self.SL_PCT
        if direction == 'LONG':
            stop_loss = round(entry_price * (1 - sl_pct / 100), 2)
            take_profit = round(entry_price * (1 + tp_pct / 100), 2)
        else:
            stop_loss = round(entry_price * (1 + sl_pct / 100), 2)
            take_profit = round(entry_price * (1 - tp_pct / 100), 2)

        # Build reasoning
        if reasoning is None:
            reasoning = f"{self.STRATEGY} signal"

        # 7. Execute
        if self.dry_run:
            self._log(
                f"[DRY RUN] {self.STRATEGY} {direction}: {symbol} | "
                f"{shares} shares @ ${entry_price:.2f} | "
                f"TP: ${take_profit:.2f} | SL: ${stop_loss:.2f}",
                "magenta"
            )
        else:
            side = 'buy' if direction == 'LONG' else 'sell'
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
                self._log(f"Failed to place bracket order for {symbol}", "red", "error")
                return False
            self._log(
                f"{self.STRATEGY} {direction}: {symbol} | {shares} @ ${entry_price:.2f} | "
                f"Order: {order_id}",
                "green"
            )

        # Track locally
        self.active_trades[symbol] = {
            'entry': entry_price,
            'entry_time': datetime.now(),
            'stop_loss': stop_loss,
            'take_profit': take_profit,
            'direction': direction,
            'shares': shares,
            'strategy': self.STRATEGY,
            'order_id': 'DRY_RUN' if self.dry_run else order_id,
        }
        self.trades_entered_today += 1
        self.set_cooldown(symbol)

        # 8. Journal (CSV)
        log_trade(
            action='ENTRY',
            symbol=symbol,
            strategy=self.STRATEGY,
            direction=direction,
            price=entry_price,
            shares=shares,
            stop_loss=stop_loss,
            take_profit=take_profit,
            reasoning=f"[{self.BOT_NAME}] {reasoning}",
            entry_signal_strength=signal_strength,
            bot_name=self.BOT_NAME,
        )

        # 9. SQLite journal
        try:
            if self.stock_journal:
                self.stock_journal.log_entry(
                    bot_name=self.BOT_NAME,
                    strategy=self.STRATEGY,
                    symbol=symbol,
                    direction=direction,
                    entry_price=entry_price,
                    shares=shares,
                    stop_loss=stop_loss,
                    take_profit=take_profit,
                    signal_strength=signal_strength,
                    regime=self._current_regime,
                    reasoning=reasoning,
                )
        except Exception:
            pass

        return True

    # ─────────────────────────────────────────────────────────
    # SIGNAL INTERFACE (subclasses implement)
    # ─────────────────────────────────────────────────────────

    @abstractmethod
    def check_signal(self, symbol, price, prices, candles):
        """
        Check for entry signal.

        Args:
            symbol: Stock ticker
            price: Current price
            prices: List of close prices (50+ bars)
            candles: List of OHLCV dicts

        Returns:
            (should_enter: bool, metadata: dict)
            metadata should contain indicator values for logging.
        """
        pass

    # ─────────────────────────────────────────────────────────
    # MAIN LOOP
    # ─────────────────────────────────────────────────────────

    def scan_once(self):
        """Run one full scan across all symbols."""
        self.cycle_count += 1
        self._log(f"--- Scan #{self.cycle_count} | {datetime.now().strftime('%H:%M:%S')} ---", "cyan")

        # Detect market regime (SPY-based, cached 5 min)
        try:
            self._current_regime = detect_market_regime(self.api)
        except Exception:
            self._current_regime = "unknown"

        # Regime-aware skill gating
        if not self._is_qualified_for_regime(self._current_regime):
            self._log(
                f"BLOCKED: {self.BOT_NAME} not qualified in {self._current_regime} regime",
                "yellow"
            )
            return

        for symbol in self.symbols:
            # Skip if already in a trade
            if symbol in self.active_trades:
                trade = self.active_trades[symbol]
                self._log(
                    f"{symbol}: In trade ({trade['strategy']} @ ${trade['entry']:.2f})",
                    "white"
                )
                continue

            # Get data
            price = self.get_current_price(symbol)
            if price is None:
                self._log(f"{symbol}: Could not get price, skipping", "red")
                continue

            candles = self.fetch_candles(symbol, hours=50)
            if len(candles) < 20:
                self._log(f"{symbol}: Insufficient data ({len(candles)} candles), skipping", "yellow")
                continue

            prices = [c['close'] for c in candles]

            # Check signal
            should_enter, metadata = self.check_signal(symbol, price, prices, candles)

            if should_enter:
                signal_strength = metadata.get('signal_strength') if metadata else None
                reasoning = metadata.get('reasoning') if metadata else None
                direction = metadata.get('direction', 'LONG') if metadata else 'LONG'
                self.execute_entry(
                    symbol, price,
                    direction=direction,
                    signal_strength=signal_strength,
                    reasoning=reasoning,
                )

    def run(self, duration_minutes=None, interval_seconds=None, continuous=False):
        """
        Main run loop.

        Args:
            duration_minutes: Run for N minutes then stop. None = run forever.
            interval_seconds: Seconds between scans. None = use SCAN_INTERVAL_SECONDS.
            continuous: If True, sleep until next market open after close
                        instead of exiting.
        """
        interval = interval_seconds or SCAN_INTERVAL_SECONDS
        self._print_banner()

        start_time = datetime.now()

        try:
            while True:
                # Check market hours
                if not self.is_market_open():
                    if self.market_was_open:
                        # Market just closed — print summary
                        self._log("Market closed.", "yellow")
                        self._print_summary()

                        if not continuous:
                            self._log("Exiting (use --continuous to keep running).", "yellow")
                            self._summary_printed = True
                            return

                        # Continuous mode: sleep until next open
                        self.market_was_open = False
                        self._sleep_until_open()
                        continue

                    # Market hasn't opened yet (pre-market wait)
                    self._wait_for_open()
                    continue

                # Market is open — mark it and scan
                self.market_was_open = True
                self.scan_once()

                # Duration check
                if duration_minutes:
                    elapsed = (datetime.now() - start_time).total_seconds() / 60
                    if elapsed >= duration_minutes:
                        self._log(f"Duration limit ({duration_minutes} min) reached.", "yellow")
                        break

                self._log(f"Sleeping {interval}s until next scan...", "white")
                time.sleep(interval)

        except KeyboardInterrupt:
            self._log("Shutting down (KeyboardInterrupt)...", "yellow")
        finally:
            if not getattr(self, '_summary_printed', False):
                self._print_summary()

    def _wait_for_open(self):
        """Wait for market to open, logging once per minute."""
        try:
            clock = self.api.get_clock()
            next_open = clock.next_open
            now = clock.timestamp
            delta = next_open - now
            minutes = int(delta.total_seconds() / 60)
            self._log(
                f"Market closed. Opens at {next_open.strftime('%H:%M ET')} "
                f"(~{minutes} min). Waiting...",
                "yellow"
            )
        except Exception:
            self._log("Market closed. Waiting for open...", "yellow")
        time.sleep(60)

    def _sleep_until_open(self):
        """Sleep until next market open (continuous mode)."""
        try:
            clock = self.api.get_clock()
            next_open = clock.next_open
            now = clock.timestamp
            delta = next_open - now
            wait_seconds = max(delta.total_seconds(), 60)
            hours = int(wait_seconds // 3600)
            mins = int((wait_seconds % 3600) // 60)
            self._log(
                f"[CONTINUOUS] Next market open: {next_open.strftime('%Y-%m-%d %H:%M ET')} "
                f"(~{hours}h {mins}m). Sleeping...",
                "cyan"
            )
            # Sleep in 5-min chunks so KeyboardInterrupt is responsive
            while wait_seconds > 0:
                chunk = min(wait_seconds, 300)
                time.sleep(chunk)
                wait_seconds -= chunk
        except Exception:
            # Fallback: sleep 60s and let the main loop re-check
            self._log("[CONTINUOUS] Sleeping 60s before re-check...", "yellow")
            time.sleep(60)
        # Reset session counters for next trading day
        self.cycle_count = 0
        self.trades_entered_today = 0
        reset_daily_tracker()

    def _print_banner(self):
        """Print startup banner."""
        cprint("\n" + "=" * 60, "cyan")
        cprint(f"  {self.BOT_NAME.upper()} | {self.STRATEGY} Strategy", "cyan", attrs=['bold'])
        cprint(f"  TP: {self.TP_PCT}% | SL: {self.SL_PCT}% | Size: ${self.order_size}", "cyan")
        cprint("=" * 60, "cyan")
        cprint(f"  Symbols: {', '.join(self.symbols)}", "white")
        cprint(f"  Dry Run: {self.dry_run}", "white")
        cprint(f"  Entry Window: After {EARLIEST_ENTRY_ET} ET", "white")
        cprint(f"  Log: logs/{self.BOT_NAME}.log", "white")
        cprint("=" * 60 + "\n", "cyan")

    def _print_summary(self):
        """Print session summary on exit."""
        cprint("\n" + "=" * 60, "cyan")
        cprint(f"  {self.BOT_NAME.upper()} — SESSION SUMMARY", "cyan", attrs=['bold'])
        cprint("=" * 60, "cyan")
        cprint(f"  Scans completed:   {self.cycle_count}", "white")
        cprint(f"  Trades entered:    {self.trades_entered_today}", "white")
        cprint(f"  Positions held:    {len(self.active_trades)}", "white")

        # Show live P&L per position
        unrealized_total = 0.0
        if self.active_trades:
            cprint("  ─────────────────────────────────────────", "cyan")
            for symbol, trade in self.active_trades.items():
                cur_price = self.get_current_price(symbol)
                if cur_price is not None:
                    entry = trade['entry']
                    shares = trade.get('shares', 0)
                    if trade['direction'] == 'LONG':
                        pnl = (cur_price - entry) * shares
                        pnl_pct = (cur_price - entry) / entry * 100
                    else:
                        pnl = (entry - cur_price) * shares
                        pnl_pct = (entry - cur_price) / entry * 100
                    unrealized_total += pnl
                    color = "green" if pnl >= 0 else "red"
                    cprint(
                        f"    {symbol:6s} {trade['direction']:5s} "
                        f"{shares} @ ${entry:.2f} → ${cur_price:.2f}  "
                        f"P&L: ${pnl:+.2f} ({pnl_pct:+.1f}%)",
                        color
                    )
                else:
                    cprint(
                        f"    {symbol:6s} {trade['direction']:5s} "
                        f"@ ${trade['entry']:.2f} (price unavailable)",
                        "yellow"
                    )
            cprint("  ─────────────────────────────────────────", "cyan")

        realized_pnl = get_daily_pnl()
        total_pnl = realized_pnl + unrealized_total
        if realized_pnl != 0:
            cprint(f"  Realized P&L:      ${realized_pnl:+,.2f}", "green" if realized_pnl >= 0 else "red")
        if unrealized_total != 0:
            cprint(f"  Unrealized P&L:    ${unrealized_total:+,.2f}", "green" if unrealized_total >= 0 else "red")
        color = "green" if total_pnl >= 0 else "red"
        cprint(f"  Daily P&L:         ${total_pnl:+,.2f}", color)
        cprint("=" * 60 + "\n", "cyan")

    # ─────────────────────────────────────────────────────────
    # CLI PARSER (shared by all bots)
    # ─────────────────────────────────────────────────────────

    @classmethod
    def parse_args(cls):
        """Parse common CLI arguments."""
        parser = argparse.ArgumentParser(
            description=f"{cls.BOT_NAME} — {cls.STRATEGY} strategy bot"
        )
        parser.add_argument(
            '--dry-run', action='store_true',
            help='Log signals but do not place real orders'
        )
        parser.add_argument(
            '--symbols', type=str, default=None,
            help='Comma-separated symbols (e.g. AMD,NVDA,TSLA)'
        )
        parser.add_argument(
            '--size', type=int, default=None,
            help=f'Order size in USD (default: {DEFAULT_ORDER_USD})'
        )
        parser.add_argument(
            '--duration', type=int, default=None,
            help='Run for N minutes then stop (default: run forever)'
        )
        parser.add_argument(
            '--interval', type=int, default=None,
            help=f'Seconds between scans (default: {SCAN_INTERVAL_SECONDS})'
        )
        parser.add_argument(
            '--continuous', action='store_true',
            help='Keep running after market close — sleep until next open'
        )
        return parser.parse_args()

    @classmethod
    def main(cls):
        """Standard entry point for all bots."""
        args = cls.parse_args()
        symbols = args.symbols.split(',') if args.symbols else None

        bot = cls(
            dry_run=args.dry_run,
            symbols=symbols,
            order_size=args.size,
        )
        bot.run(
            duration_minutes=args.duration,
            interval_seconds=args.interval,
            continuous=args.continuous,
        )
