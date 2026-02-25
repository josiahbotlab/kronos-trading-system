"""
Gap and Go Trading Bot

Dedicated bot for Gap and Go strategy with optimized per-symbol parameters.
Combines pre-market scanning, order execution, and position management.

Morning Workflow:
    Phase A (4:00 AM - 9:25 AM): Pre-market monitoring every 5 minutes
    Phase B (9:25 AM): Final scan, lock candidates
    Phase C (9:32 AM): Execute trades 2 minutes after open
    Phase D (9:32 AM - 11:00 AM): Monitor positions
    Phase E (11:00 AM): Close remaining, generate summary

Usage:
    python3 src/agents/gap_and_go_bot.py                    # Paper trading
    python3 src/agents/gap_and_go_bot.py --live             # Real trading
    python3 src/agents/gap_and_go_bot.py --scan-only        # Scanner only
    python3 src/agents/gap_and_go_bot.py --dry-run          # Full workflow, no orders
    python3 src/agents/gap_and_go_bot.py --position-size 500
"""

import argparse
import csv
import json
import os
import signal
import sys
import time
from datetime import datetime, date, time as dt_time, timedelta
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import pytz


# ============================================================================
# EXCEPTIONS
# ============================================================================

class PositionLockViolation(Exception):
    """Raised when attempting to modify a locked parameter while in position."""
    pass
from termcolor import cprint
from dotenv import load_dotenv

sys.path.append('/Users/josiahgarcia/trading-bot')
load_dotenv()

# Import gap scanner
from src.scanner.gap_scanner import (
    scan_premarket,
    get_gap_candidates,
    get_best_candidate,
    get_current_gaps,
    get_symbol_params,
    get_tradeable_symbols,
    is_premarket,
    is_market_open,
    is_weekend,
    is_market_holiday,
    get_market_status,
    get_et_now,
    ET,
)

# Import order utilities
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
    execute_entry,
    execute_exit,
    get_account_info,
    get_all_positions,
    get_open_orders,
    close_position,
)

# Import trade journal
from src.utils.trade_journal import log_trade

# Import regime detector
try:
    from src.agents.regime_detector import detect_regime_verbose, RegimeType, STRATEGY_MAP
    REGIME_DETECTOR_AVAILABLE = True
except ImportError:
    REGIME_DETECTOR_AVAILABLE = False
    cprint("Warning: regime_detector not available", "yellow")

# Import crypto market data trackers
try:
    from src.data.liquidation_tracker import start_tracker as start_liquidation_tracker, get_liquidation_totals
    from src.data.market_indicators import start_market_indicators as start_market_trackers, get_market_snapshot
    from src.agents.regime_agent import get_crypto_risk_level, get_full_regime_status
    CRYPTO_TRACKERS_AVAILABLE = True
except ImportError as e:
    CRYPTO_TRACKERS_AVAILABLE = False
    cprint(f"Warning: crypto trackers not available: {e}", "yellow")

# Import AI swarm validation
try:
    from src.agents.ai_swarm import get_swarm_validation
    AI_SWARM_AVAILABLE = True
except ImportError as e:
    AI_SWARM_AVAILABLE = False
    cprint(f"Warning: AI swarm not available: {e}", "yellow")

# ============================================================================
# CONFIGURATION
# ============================================================================

# Paths
PROJECT_ROOT = Path(__file__).parent.parent.parent
STATE_FILE = PROJECT_ROOT / 'csvs' / 'gap_go_state.json'
DAILY_SUMMARY_DIR = PROJECT_ROOT / 'csvs' / 'gap_go_daily'

# Trading parameters
DEFAULT_POSITION_SIZE = 1000  # USD per trade
MAX_POSITIONS = 2             # Maximum concurrent positions

# Regime-based trading rules
# TREND_UP (BULL): Trade gap UPs only
# TREND_DOWN (BEAR): Trade gap DOWNs only
# RANGING: Trade both directions
# VOLATILE: Reduce position size 50%, widen stops 25%
REGIME_RULES = {
    'TREND_UP': {
        'allowed_directions': ['UP'],
        'position_size_mult': 1.0,
        'stop_loss_mult': 1.0,
        'description': 'BULL market - only trading gap UPs'
    },
    'TREND_DOWN': {
        'allowed_directions': ['DOWN'],
        'position_size_mult': 1.0,
        'stop_loss_mult': 1.0,
        'description': 'BEAR market - only trading gap DOWNs'
    },
    'RANGING': {
        'allowed_directions': ['UP', 'DOWN'],
        'position_size_mult': 1.0,
        'stop_loss_mult': 1.0,
        'description': 'RANGE market - trading both directions'
    },
    'VOLATILE': {
        'allowed_directions': ['UP', 'DOWN'],
        'position_size_mult': 0.5,
        'stop_loss_mult': 1.25,
        'description': 'HIGH VOLATILITY - reduced size, wider stops'
    }
}

# Timing (Eastern Time)
PREMARKET_START = dt_time(4, 0)     # 4:00 AM ET
FINAL_SCAN_TIME = dt_time(9, 25)    # 9:25 AM ET
EXECUTION_TIME = dt_time(9, 32)     # 9:32 AM ET (2 min after open)
SESSION_END_TIME = dt_time(11, 0)   # 11:00 AM ET
SCAN_INTERVAL_MINUTES = 5           # Scan every 5 minutes during pre-market
PNL_LOG_INTERVAL_MINUTES = 5        # Log P&L every 5 minutes during monitoring

# ============================================================================
# BOT STATE
# ============================================================================

class BotState(Enum):
    INITIALIZING = "INITIALIZING"
    WAITING_PREMARKET = "WAITING_PREMARKET"
    MONITORING = "MONITORING"
    FINAL_SCAN = "FINAL_SCAN"
    READY_TO_TRADE = "READY_TO_TRADE"
    TRADING = "TRADING"
    MONITORING_POSITIONS = "MONITORING_POSITIONS"
    CLOSING_POSITIONS = "CLOSING_POSITIONS"
    IDLE = "IDLE"
    SHUTDOWN = "SHUTDOWN"


class GapAndGoBot:
    """
    Gap and Go Trading Bot.

    Executes a daily workflow for Gap and Go strategy with
    per-symbol optimized parameters.
    """

    def __init__(
        self,
        position_size: float = DEFAULT_POSITION_SIZE,
        max_positions: int = MAX_POSITIONS,
        dry_run: bool = False,
        scan_only: bool = False,
        live_trading: bool = False,
    ):
        self.position_size = position_size
        self.max_positions = max_positions
        self.dry_run = dry_run
        self.scan_only = scan_only
        self.live_trading = live_trading

        # State
        self.state = BotState.INITIALIZING
        self.running = False
        self.locked_candidates: List[Dict] = []
        self.active_trades: Dict[str, Dict] = {}
        self.daily_summary: Dict = {}
        self.last_scan_time: Optional[datetime] = None
        self.last_pnl_log_time: Optional[datetime] = None

        # Regime tracking
        self.current_regime: Optional[str] = None
        self.regime_info: Optional[Dict] = None
        self.trades_skipped_regime: List[Dict] = []

        # Crypto tracking
        self.trades_skipped_crypto: List[Dict] = []
        self.total_crypto_liquidations_session: float = 0.0
        self.crypto_position_multiplier: float = 1.0

        # AI swarm tracking
        self.trades_skipped_swarm: List[Dict] = []
        self.swarm_validations: List[Dict] = []

        # Daily loss limit (percentage of account equity)
        self.daily_loss_limit_pct: float = 3.0  # Default 3%

        # Position locks - prevents dangerous parameter changes while in position
        # When position opens, current values are locked. When position closes, locks clear.
        self.position_locks: Dict[str, Any] = {
            'position_size_locked': None,      # float - cannot increase
            'stop_loss_pct_locked': None,      # float - cannot widen (can only tighten)
            'daily_loss_limit_locked': None,   # float - cannot change
            'max_positions_locked': None,      # int - cannot increase
        }

        # Start crypto trackers if available
        if CRYPTO_TRACKERS_AVAILABLE:
            cprint("\n[CRYPTO] Starting crypto market data trackers...", "cyan")
            try:
                start_liquidation_tracker()
                start_market_trackers()
                cprint("[CRYPTO] Liquidation tracker started", "green")
                cprint("[CRYPTO] Market indicators tracker started", "green")
            except Exception as e:
                cprint(f"[CRYPTO] Error starting trackers: {e}", "yellow")
        else:
            cprint("[CRYPTO] Crypto trackers not available - trading without crypto filter", "yellow")

        # Symbol parameters
        self.symbol_params = get_symbol_params()
        self.tradeable_symbols = get_tradeable_symbols()

        # Ensure directories exist
        DAILY_SUMMARY_DIR.mkdir(parents=True, exist_ok=True)

        # Set up signal handlers for graceful shutdown
        signal.signal(signal.SIGINT, self._handle_shutdown)
        signal.signal(signal.SIGTERM, self._handle_shutdown)

    # =========================================================================
    # STATE MANAGEMENT
    # =========================================================================

    def save_state(self):
        """Save current bot state to file for recovery."""
        state_data = {
            'state': self.state.value,
            'date': date.today().isoformat(),
            'timestamp': datetime.now().isoformat(),
            'locked_candidates': self.locked_candidates,
            'active_trades': self.active_trades,
            'position_size': self.position_size,
            'dry_run': self.dry_run,
        }

        with open(STATE_FILE, 'w') as f:
            json.dump(state_data, f, indent=2)

    def load_state(self) -> bool:
        """
        Load state from file if from today's session.

        Returns:
            True if valid state was loaded
        """
        if not STATE_FILE.exists():
            return False

        try:
            with open(STATE_FILE, 'r') as f:
                state_data = json.load(f)

            # Check if state is from today
            if state_data.get('date') != date.today().isoformat():
                cprint("[STATE] Previous state is from different day, starting fresh", "yellow")
                return False

            self.locked_candidates = state_data.get('locked_candidates', [])
            self.active_trades = state_data.get('active_trades', {})

            cprint(f"[STATE] Loaded state from: {state_data.get('timestamp')}", "cyan")
            cprint(f"[STATE] Locked candidates: {len(self.locked_candidates)}", "cyan")
            cprint(f"[STATE] Active trades: {len(self.active_trades)}", "cyan")

            return True

        except Exception as e:
            cprint(f"[STATE] Error loading state: {e}", "red")
            return False

    def set_state(self, new_state: BotState):
        """Update bot state with logging."""
        old_state = self.state
        self.state = new_state
        cprint(f"[STATE] {old_state.value} -> {new_state.value}", "magenta")
        self.save_state()

    # =========================================================================
    # POSITION LOCK CONTROLS
    # =========================================================================

    def _has_open_positions(self) -> bool:
        """Check if bot has any open positions."""
        return len(self.active_trades) > 0

    def _is_locked(self) -> bool:
        """Check if any position locks are active."""
        return any(v is not None for v in self.position_locks.values())

    def _apply_position_locks(self, symbol: str):
        """
        Apply position locks when a position is opened.
        Stores current parameter values that become immutable until position closes.
        """
        if self._is_locked():
            cprint(f"[LOCK] Locks already active, maintaining existing lock values", "cyan")
            return

        self.position_locks = {
            'position_size_locked': self.position_size,
            'stop_loss_pct_locked': self._get_current_stop_loss_pct(symbol),
            'daily_loss_limit_locked': self.daily_loss_limit_pct,
            'max_positions_locked': self.max_positions,
        }

        cprint(f"[LOCK] Position locks APPLIED for {symbol}:", "yellow")
        cprint(f"[LOCK]   position_size: ${self.position_locks['position_size_locked']:,.0f} (cannot increase)", "yellow")
        cprint(f"[LOCK]   stop_loss_pct: {self.position_locks['stop_loss_pct_locked']:.1f}% (cannot widen)", "yellow")
        cprint(f"[LOCK]   daily_loss_limit: {self.position_locks['daily_loss_limit_locked']:.1f}% (cannot change)", "yellow")
        cprint(f"[LOCK]   max_positions: {self.position_locks['max_positions_locked']} (cannot increase)", "yellow")

    def _release_position_locks(self, symbol: str):
        """
        Release position locks when position closes.
        Only releases if no other positions remain open.
        """
        # Check if other positions still open
        remaining_positions = {s: t for s, t in self.active_trades.items() if s != symbol}

        if remaining_positions:
            cprint(f"[LOCK] Position closed for {symbol}, but {len(remaining_positions)} position(s) still open - locks maintained", "cyan")
            return

        # Clear all locks
        self.position_locks = {
            'position_size_locked': None,
            'stop_loss_pct_locked': None,
            'daily_loss_limit_locked': None,
            'max_positions_locked': None,
        }

        cprint(f"[LOCK] All positions closed - Position locks RELEASED", "green")

    def _get_current_stop_loss_pct(self, symbol: str) -> float:
        """Get the current stop loss percentage for a symbol from its parameters."""
        params = self.symbol_params.get(symbol, {})
        return params.get('stop_loss', 10.0)  # Default 10% if not found

    def set_position_size(self, value: float):
        """
        Set position size with lock validation.
        Cannot increase while in position.
        """
        locked_value = self.position_locks.get('position_size_locked')

        if locked_value is not None and value > locked_value:
            raise PositionLockViolation(
                f"Cannot increase position_size from ${locked_value:,.0f} to ${value:,.0f} while in position. "
                f"Close all positions first or reduce to ${locked_value:,.0f} or less."
            )

        old_value = self.position_size
        self.position_size = value
        cprint(f"[CONFIG] position_size: ${old_value:,.0f} -> ${value:,.0f}", "cyan")

    def set_stop_loss_pct(self, symbol: str, value: float):
        """
        Set stop loss percentage with lock validation.
        Cannot widen (increase) while in position - can only tighten (decrease).
        """
        locked_value = self.position_locks.get('stop_loss_pct_locked')

        if locked_value is not None and value > locked_value:
            raise PositionLockViolation(
                f"Cannot widen stop_loss from {locked_value:.1f}% to {value:.1f}% while in position. "
                f"Close all positions first or tighten to {locked_value:.1f}% or less."
            )

        params = self.symbol_params.get(symbol, {})
        old_value = params.get('stop_loss', 10.0)
        params['stop_loss'] = value
        self.symbol_params[symbol] = params
        cprint(f"[CONFIG] {symbol} stop_loss: {old_value:.1f}% -> {value:.1f}%", "cyan")

    def set_daily_loss_limit(self, value: float):
        """
        Set daily loss limit percentage with lock validation.
        Cannot change while in position.
        """
        locked_value = self.position_locks.get('daily_loss_limit_locked')

        if locked_value is not None and value != locked_value:
            raise PositionLockViolation(
                f"Cannot change daily_loss_limit from {locked_value:.1f}% to {value:.1f}% while in position. "
                f"Close all positions first."
            )

        old_value = self.daily_loss_limit_pct
        self.daily_loss_limit_pct = value
        cprint(f"[CONFIG] daily_loss_limit: {old_value:.1f}% -> {value:.1f}%", "cyan")

    def set_max_positions(self, value: int):
        """
        Set max positions with lock validation.
        Cannot increase while in position.
        """
        locked_value = self.position_locks.get('max_positions_locked')

        if locked_value is not None and value > locked_value:
            raise PositionLockViolation(
                f"Cannot increase max_positions from {locked_value} to {value} while in position. "
                f"Close all positions first or reduce to {locked_value} or less."
            )

        old_value = self.max_positions
        self.max_positions = value
        cprint(f"[CONFIG] max_positions: {old_value} -> {value}", "cyan")

    def get_lock_status(self) -> Dict[str, Any]:
        """Get current lock status for monitoring."""
        return {
            'is_locked': self._is_locked(),
            'open_positions': len(self.active_trades),
            'locks': self.position_locks.copy(),
        }

    # =========================================================================
    # SHUTDOWN HANDLING
    # =========================================================================

    def _handle_shutdown(self, signum, frame):
        """Handle graceful shutdown on Ctrl+C or SIGTERM."""
        cprint("\n" + "=" * 60, "red")
        cprint("  SHUTDOWN SIGNAL RECEIVED", "red", attrs=['bold'])
        cprint("=" * 60, "red")

        self.set_state(BotState.SHUTDOWN)
        self.running = False

        # Close any open positions if not dry run
        if not self.dry_run and not self.scan_only:
            self._emergency_close_positions()

        # Save final state
        self.save_state()
        self._generate_daily_summary()

        cprint("\n[SHUTDOWN] Bot shut down gracefully", "green")
        sys.exit(0)

    def _emergency_close_positions(self):
        """Close all positions on emergency shutdown."""
        cprint("\n[SHUTDOWN] Closing all positions...", "yellow")

        positions = get_all_positions()
        closed_symbols = []
        for pos in positions:
            if pos['symbol'] in self.tradeable_symbols:
                symbol = pos['symbol']
                cprint(f"[SHUTDOWN] Closing {symbol}: {pos['qty']} shares", "yellow")
                cancel_symbol_orders(symbol)
                close_position(symbol)
                closed_symbols.append(symbol)

        # Release locks for all closed positions
        for symbol in closed_symbols:
            if symbol in self.active_trades:
                del self.active_trades[symbol]
            self._release_position_locks(symbol)

    def _check_positions_closed_by_orders(self):
        """
        Check if any positions were closed by TP/SL orders.
        Updates active_trades and releases locks if positions no longer exist.
        """
        if not self.active_trades:
            return

        positions = get_all_positions()
        position_symbols = {p['symbol'] for p in positions}

        # Find symbols that were in active_trades but no longer have positions
        closed_symbols = []
        for symbol in list(self.active_trades.keys()):
            if symbol not in position_symbols:
                closed_symbols.append(symbol)
                cprint(f"[MONITOR] Position closed by TP/SL for {symbol}", "green")

        # Release locks for closed positions
        for symbol in closed_symbols:
            del self.active_trades[symbol]
            self._release_position_locks(symbol)

    # =========================================================================
    # PRE-MARKET MONITORING (Phase A)
    # =========================================================================

    def run_premarket_scan(self) -> List[Dict]:
        """Run pre-market scan and display results."""
        now = get_et_now()

        cprint(f"\n[SCAN] Pre-market scan at {now.strftime('%H:%M:%S')} ET", "cyan")
        cprint("-" * 50, "cyan")

        gaps = get_current_gaps()

        if not gaps:
            cprint("[SCAN] No gap data available", "yellow")
            return []

        # Display each symbol's gap status
        for symbol in self.tradeable_symbols:
            gap_info = gaps.get(symbol)
            if gap_info:
                gap_pct = gap_info['gap_pct']
                threshold = gap_info['gap_threshold']
                direction = gap_info['gap_direction']
                meets = gap_info['meets_threshold']

                status = "TRIGGER" if meets else "below"
                color = "green" if meets else "white"

                cprint(
                    f"  {symbol}: {gap_pct:+.2f}% {direction} (needs {threshold}%) [{status}]",
                    color
                )
            else:
                cprint(f"  {symbol}: No data", "yellow")

        self.last_scan_time = now
        return list(gaps.values())

    def wait_for_premarket(self):
        """Wait until pre-market hours begin."""
        now = get_et_now()
        current_time = now.time()

        if current_time < PREMARKET_START:
            # Calculate time until pre-market
            premarket_dt = now.replace(
                hour=PREMARKET_START.hour,
                minute=PREMARKET_START.minute,
                second=0
            )
            wait_seconds = (premarket_dt - now).total_seconds()

            cprint(f"\n[BOT] Pre-market starts at {PREMARKET_START.strftime('%H:%M')} ET", "cyan")
            cprint(f"[BOT] Waiting {wait_seconds / 60:.1f} minutes...", "cyan")

            while get_et_now().time() < PREMARKET_START and self.running:
                time.sleep(60)  # Check every minute

    # =========================================================================
    # FINAL SCAN (Phase B)
    # =========================================================================

    def run_final_scan(self) -> List[Dict]:
        """
        Run final scan at 9:25 AM and lock in candidates.

        Returns:
            List of locked candidates
        """
        cprint("\n" + "=" * 60, "green")
        cprint("  FINAL PRE-MARKET SCAN - LOCKING CANDIDATES", "green", attrs=['bold'])
        cprint("=" * 60, "green")

        # Get candidates meeting thresholds
        candidates = get_gap_candidates()

        if not candidates:
            cprint("\n[FINAL] No symbols meeting gap thresholds", "yellow")
            self.locked_candidates = []
            return []

        # Lock candidates
        self.locked_candidates = candidates[:self.max_positions]

        cprint(f"\n[FINAL] LOCKED {len(self.locked_candidates)} CANDIDATES:", "green")
        for c in self.locked_candidates:
            cprint(f"\n  {c['symbol']} - Gap {c['gap_direction']} {c['abs_gap_pct']:.2f}%", "green")
            cprint(f"    Entry: ${c['entry_price']:.2f}", "white")
            cprint(f"    Take Profit: ${c['tp_price']:.2f} ({c['tp_pct']:.1f}%)", "green")
            cprint(f"    Stop Loss: ${c['sl_price']:.2f} ({c['sl_pct']:.1f}%)", "red")

        self.save_state()
        return self.locked_candidates

    # =========================================================================
    # REGIME DETECTION
    # =========================================================================

    def detect_market_regime(self, reference_symbol: str = 'SPY') -> Optional[str]:
        """
        Detect current market regime using SPY as reference.

        Returns:
            Regime type: 'TREND_UP', 'TREND_DOWN', 'RANGING', 'VOLATILE'
        """
        if not REGIME_DETECTOR_AVAILABLE:
            cprint("[REGIME] Regime detector not available - using default (RANGING)", "yellow")
            self.current_regime = 'RANGING'
            self.regime_info = {'regime': 'RANGING', 'confidence': 0}
            return 'RANGING'

        try:
            cprint(f"\n[REGIME] Detecting market regime from {reference_symbol}...", "cyan")

            analysis = detect_regime_verbose(reference_symbol)
            regime = analysis['regime']
            confidence = analysis.get('confidence', 0)
            reasons = analysis.get('reasons', [])

            self.current_regime = regime
            self.regime_info = analysis

            # Get regime rules
            rules = REGIME_RULES.get(regime, REGIME_RULES['RANGING'])

            # Display regime info
            color = 'green' if regime == 'TREND_UP' else 'red' if regime == 'TREND_DOWN' else 'yellow' if regime == 'VOLATILE' else 'cyan'
            cprint(f"\n[REGIME] Market Regime: {regime} (confidence: {confidence}%)", color, attrs=['bold'])
            cprint(f"[REGIME] {rules['description']}", color)

            for reason in reasons[:3]:
                cprint(f"[REGIME]   - {reason}", "white")

            if regime == 'VOLATILE':
                cprint(f"[REGIME] Position size: 50% | Stop loss: +25%", "yellow")
            elif regime == 'TREND_UP':
                cprint(f"[REGIME] Will only trade gap UPs (skip gap DOWNs)", "green")
            elif regime == 'TREND_DOWN':
                cprint(f"[REGIME] Will only trade gap DOWNs (skip gap UPs)", "red")

            return regime

        except Exception as e:
            cprint(f"[REGIME] Error detecting regime: {e}", "red")
            self.current_regime = 'RANGING'
            self.regime_info = {'regime': 'RANGING', 'confidence': 0, 'error': str(e)}
            return 'RANGING'

    def apply_regime_filter(self, candidate: Dict) -> Tuple[bool, str]:
        """
        Apply regime filter to a candidate.

        Returns:
            (should_trade, reason)
        """
        if not self.current_regime:
            return True, "No regime detected"

        rules = REGIME_RULES.get(self.current_regime, REGIME_RULES['RANGING'])
        direction = candidate['gap_direction']

        if direction not in rules['allowed_directions']:
            return False, f"Regime {self.current_regime} doesn't allow gap {direction}"

        return True, f"Regime {self.current_regime} allows gap {direction}"

    def adjust_for_regime(self, position_size: float, sl_price: float, entry_price: float, direction: str) -> Tuple[float, float]:
        """
        Adjust position size and stop loss based on regime.

        Returns:
            (adjusted_position_size, adjusted_sl_price)
        """
        if not self.current_regime:
            return position_size, sl_price

        rules = REGIME_RULES.get(self.current_regime, REGIME_RULES['RANGING'])

        # Adjust position size
        adjusted_size = position_size * rules['position_size_mult']

        # Adjust stop loss (widen it)
        if rules['stop_loss_mult'] != 1.0:
            sl_distance = abs(entry_price - sl_price)
            adjusted_distance = sl_distance * rules['stop_loss_mult']

            if direction == 'UP':
                adjusted_sl = entry_price - adjusted_distance
            else:
                adjusted_sl = entry_price + adjusted_distance
        else:
            adjusted_sl = sl_price

        return adjusted_size, round(adjusted_sl, 2)

    # =========================================================================
    # CRYPTO CONDITIONS CHECK
    # =========================================================================

    def check_crypto_conditions(self, symbol: str) -> Tuple[bool, float, str]:
        """
        Check crypto market conditions before entering a trade.

        Uses liquidation data and funding rates to assess market risk:
        - If liquidations in last 15 min > $10M: HIGH_RISK, skip trade
        - If liquidations > $5M: ELEVATED, reduce position 50%
        - If funding extreme (>50% yearly): CAUTION, reduce position 25%

        Args:
            symbol: The stock symbol being traded (for logging)

        Returns:
            Tuple of (should_trade, position_multiplier, reason)
        """
        if not CRYPTO_TRACKERS_AVAILABLE:
            return True, 1.0, "Crypto trackers not available"

        try:
            # Get crypto risk assessment
            risk_data = get_crypto_risk_level()
            risk_level = risk_data.get('risk_level', 'NORMAL')
            position_mult = risk_data.get('position_multiplier', 1.0)
            risk_factors = risk_data.get('risk_factors', [])
            liquidations_15m = risk_data.get('liquidations_15m_usd', 0)
            funding_sentiment = risk_data.get('funding_sentiment', 'NEUTRAL')

            # Track session liquidations
            self.total_crypto_liquidations_session = max(
                self.total_crypto_liquidations_session,
                liquidations_15m
            )
            self.crypto_position_multiplier = position_mult

            # Display crypto conditions
            risk_color = 'red' if risk_level == 'HIGH' else 'yellow' if risk_level == 'ELEVATED' else 'green'
            cprint(f"\n[CRYPTO] Market Risk: {risk_level}", risk_color)
            cprint(f"[CRYPTO] 15m Liquidations: ${liquidations_15m:,.0f}", "white")
            cprint(f"[CRYPTO] Funding Sentiment: {funding_sentiment}", "white")
            cprint(f"[CRYPTO] Position Multiplier: {position_mult:.0%}", "white")

            if risk_factors:
                for factor in risk_factors:
                    cprint(f"[CRYPTO]   - {factor}", "yellow")

            # Decision logic
            if risk_level == 'HIGH':
                reason = f"HIGH crypto risk: {', '.join(risk_factors)}"
                cprint(f"[CRYPTO] BLOCKING trade for {symbol}: {reason}", "red")
                return False, 0.0, reason

            if risk_level == 'ELEVATED':
                reason = f"ELEVATED crypto risk - reducing position: {', '.join(risk_factors)}"
                cprint(f"[CRYPTO] CAUTION for {symbol}: {reason}", "yellow")
                return True, position_mult, reason

            return True, position_mult, f"NORMAL crypto conditions"

        except Exception as e:
            cprint(f"[CRYPTO] Error checking conditions: {e}", "yellow")
            return True, 1.0, f"Error: {e}"

    # =========================================================================
    # TRADE EXECUTION (Phase C)
    # =========================================================================

    def execute_trades(self):
        """Execute trades for locked candidates at market open."""
        if not self.locked_candidates:
            cprint("\n[EXECUTE] No candidates to trade", "yellow")
            return

        cprint("\n" + "=" * 60, "cyan")
        cprint("  EXECUTING TRADES", "cyan", attrs=['bold'])
        cprint("=" * 60, "cyan")

        # Detect market regime before trading
        self.detect_market_regime('SPY')

        # Check daily loss limit
        if not check_daily_loss_limit():
            cprint("[EXECUTE] Daily loss limit reached - no trades", "red")
            return

        # Get account info for position sizing
        account = get_account_info()
        if not account:
            cprint("[EXECUTE] Cannot get account info - aborting", "red")
            return

        buying_power = account.get('buying_power', 0)
        cprint(f"[EXECUTE] Buying power: ${buying_power:,.2f}", "white")

        for candidate in self.locked_candidates:
            symbol = candidate['symbol']
            entry_price = candidate['entry_price']
            tp_price = candidate['tp_price']
            sl_price = candidate['sl_price']
            direction = candidate['gap_direction']

            # Apply regime filter
            should_trade, regime_reason = self.apply_regime_filter(candidate)
            if not should_trade:
                cprint(f"\n[EXECUTE] SKIPPING {symbol}: {regime_reason}", "yellow")
                self.trades_skipped_regime.append({
                    'symbol': symbol,
                    'direction': direction,
                    'reason': regime_reason,
                    'regime': self.current_regime
                })
                continue

            # Apply crypto conditions filter
            crypto_ok, crypto_mult, crypto_reason = self.check_crypto_conditions(symbol)
            if not crypto_ok:
                cprint(f"\n[EXECUTE] SKIPPING {symbol}: {crypto_reason}", "red")
                self.trades_skipped_crypto.append({
                    'symbol': symbol,
                    'direction': direction,
                    'reason': crypto_reason,
                    'crypto_mult': crypto_mult
                })
                continue

            # Get crypto conditions for swarm validation
            crypto_conditions = {
                'risk_level': 'NORMAL',
                'liquidations_15m_usd': self.total_crypto_liquidations_session,
                'funding_sentiment': 'NEUTRAL',
                'position_multiplier': crypto_mult
            }
            if CRYPTO_TRACKERS_AVAILABLE:
                try:
                    risk_data = get_crypto_risk_level()
                    crypto_conditions.update({
                        'risk_level': risk_data.get('risk_level', 'NORMAL'),
                        'funding_sentiment': risk_data.get('funding_sentiment', 'NEUTRAL')
                    })
                except:
                    pass

            # Apply AI swarm validation
            swarm_mult = 1.0
            swarm_result = None
            if AI_SWARM_AVAILABLE:
                swarm_result = get_swarm_validation(
                    symbol=symbol,
                    gap_pct=candidate['abs_gap_pct'],
                    direction=direction,
                    crypto_conditions=crypto_conditions,
                    regime=self.current_regime or 'RANGING',
                    verbose=True
                )

                # Store validation result
                self.swarm_validations.append({
                    'symbol': symbol,
                    'result': swarm_result
                })

                if not swarm_result['should_trade']:
                    cprint(f"\n[EXECUTE] SKIPPING {symbol}: AI swarm consensus is {swarm_result['consensus']}", "red")
                    self.trades_skipped_swarm.append({
                        'symbol': symbol,
                        'direction': direction,
                        'consensus': swarm_result['consensus'],
                        'enter_count': swarm_result['enter_count'],
                        'responses': swarm_result['responses']
                    })
                    continue

                swarm_mult = swarm_result['position_multiplier']

            # Calculate position size (with regime, crypto, and swarm adjustment)
            base_size = self.position_size / entry_price
            adjusted_size, adjusted_sl = self.adjust_for_regime(
                base_size, sl_price, entry_price, direction
            )
            # Apply crypto multiplier
            adjusted_size = adjusted_size * crypto_mult
            # Apply swarm multiplier
            adjusted_size = adjusted_size * swarm_mult
            shares = round_to_valid_qty(symbol, adjusted_size)

            # Determine side
            side = 'buy' if direction == 'UP' else 'sell'

            cprint(f"\n[EXECUTE] Trading {symbol}:", "cyan")
            cprint(f"  Side: {side.upper()}", "white")
            cprint(f"  Shares: {shares:.4f}", "white")
            cprint(f"  Entry: ${entry_price:.2f}", "white")
            cprint(f"  TP: ${tp_price:.2f} | SL: ${adjusted_sl:.2f}", "white")

            if adjusted_sl != sl_price:
                cprint(f"  (SL adjusted from ${sl_price:.2f} due to {self.current_regime} regime)", "yellow")
            if crypto_mult < 1.0:
                cprint(f"  (Size reduced to {crypto_mult:.0%} due to crypto market conditions)", "yellow")
            if swarm_mult < 1.0:
                consensus = swarm_result['consensus'] if swarm_result else 'N/A'
                cprint(f"  (Size reduced to {swarm_mult:.0%} due to AI swarm consensus: {consensus})", "yellow")

            # Execute entry
            success, order_id, msg = execute_entry(
                symbol=symbol,
                side=side,
                qty=shares,
                entry_price=entry_price,
                tp_price=tp_price,
                sl_price=adjusted_sl,
                strategy='GAP_GO',
                dry_run=self.dry_run
            )

            if success:
                # Build swarm info for logging
                swarm_consensus = swarm_result['consensus'] if swarm_result else 'N/A'
                swarm_enter_count = swarm_result['enter_count'] if swarm_result else 0

                # Log to trade journal
                log_trade(
                    action='ENTRY',
                    symbol=symbol,
                    strategy='GAP_GO',
                    direction='LONG' if side == 'buy' else 'SHORT',
                    price=entry_price,
                    shares=shares,
                    stop_loss=adjusted_sl,
                    take_profit=tp_price,
                    regime=self.current_regime,
                    reasoning=f"Gap {direction} {candidate['abs_gap_pct']:.2f}% | Regime: {self.current_regime} | Swarm: {swarm_consensus} ({swarm_enter_count}/3)"
                )

                # Track active trade
                self.active_trades[symbol] = {
                    'order_id': order_id,
                    'entry_price': entry_price,
                    'tp_price': tp_price,
                    'sl_price': sl_price,
                    'shares': shares,
                    'side': side,
                    'direction': 'LONG' if side == 'buy' else 'SHORT',
                    'entry_time': datetime.now().isoformat(),
                    'swarm_consensus': swarm_consensus,
                    'swarm_multiplier': swarm_mult,
                }

                # Apply position locks (risk control)
                self._apply_position_locks(symbol)

            else:
                cprint(f"[EXECUTE] Failed to enter {symbol}: {msg}", "red")

        self.save_state()

    # =========================================================================
    # POSITION MONITORING (Phase D)
    # =========================================================================

    def monitor_positions(self):
        """Monitor open positions and log P&L updates."""
        # Check if any positions were closed by TP/SL orders
        self._check_positions_closed_by_orders()

        positions = get_all_positions()

        # Filter to our tradeable symbols
        our_positions = [p for p in positions if p['symbol'] in self.tradeable_symbols]

        if not our_positions:
            return False  # No positions to monitor

        now = get_et_now()

        # Log P&L periodically
        should_log = (
            self.last_pnl_log_time is None or
            (now - self.last_pnl_log_time).total_seconds() >= PNL_LOG_INTERVAL_MINUTES * 60
        )

        if should_log:
            cprint(f"\n[MONITOR] Position Update - {now.strftime('%H:%M:%S')} ET", "cyan")
            cprint("-" * 50, "cyan")

            total_pnl = 0
            for pos in our_positions:
                symbol = pos['symbol']
                pnl = pos['unrealized_pnl']
                pnl_pct = pos['unrealized_pnl_pct']
                total_pnl += pnl

                color = "green" if pnl > 0 else "red"
                cprint(
                    f"  {symbol}: {pos['qty']:.4f} shares | "
                    f"P&L: ${pnl:+.2f} ({pnl_pct:+.2f}%)",
                    color
                )

            cprint(f"  Total Unrealized P&L: ${total_pnl:+.2f}", "cyan")
            self.last_pnl_log_time = now

        return True  # Still have positions

    def check_positions_closed(self) -> bool:
        """Check if all positions have been closed by TP/SL."""
        positions = get_all_positions()
        our_positions = [p for p in positions if p['symbol'] in self.tradeable_symbols]
        return len(our_positions) == 0

    # =========================================================================
    # SESSION END (Phase E)
    # =========================================================================

    def close_remaining_positions(self):
        """Close any remaining positions at session end."""
        cprint("\n" + "=" * 60, "yellow")
        cprint("  SESSION END - CLOSING REMAINING POSITIONS", "yellow", attrs=['bold'])
        cprint("=" * 60, "yellow")

        positions = get_all_positions()
        our_positions = [p for p in positions if p['symbol'] in self.tradeable_symbols]

        if not our_positions:
            cprint("[END] No positions to close", "green")
            return

        for pos in our_positions:
            symbol = pos['symbol']
            qty = pos['qty']
            pnl = pos['unrealized_pnl']
            entry_price = pos['avg_entry_price']
            current_price = pos['current_price']

            cprint(f"\n[END] Closing {symbol}: {qty:.4f} shares", "yellow")

            if self.dry_run:
                cprint(f"[END] DRY RUN - Would close {symbol} with P&L ${pnl:+.2f}", "magenta")
            else:
                # Cancel any open orders first
                cancel_symbol_orders(symbol)

                # Close position
                success = close_position(symbol)

                if success:
                    # Log to journal
                    log_trade(
                        action='EXIT',
                        symbol=symbol,
                        strategy='GAP_GO',
                        direction=pos['side'].upper(),
                        price=current_price,
                        shares=abs(qty),
                        entry_price=entry_price,
                        exit_price=current_price,
                        pnl=pnl,
                        pnl_pct=pos['unrealized_pnl_pct'],
                        reasoning='Session end - time exit'
                    )

                    # Update daily P&L
                    update_daily_pnl(pnl)

                    # Remove from active trades and check locks
                    if symbol in self.active_trades:
                        del self.active_trades[symbol]
                    self._release_position_locks(symbol)

    def _generate_daily_summary(self):
        """Generate and save daily summary with regime and crypto information."""
        today = date.today().strftime('%Y%m%d')
        summary_file = DAILY_SUMMARY_DIR / f'summary_{today}.csv'

        # Regime info
        regime = self.current_regime or 'UNKNOWN'
        regime_confidence = self.regime_info.get('confidence', 0) if self.regime_info else 0
        trades_skipped = len(self.trades_skipped_regime)
        skipped_symbols = ','.join([t['symbol'] for t in self.trades_skipped_regime]) if self.trades_skipped_regime else ''

        # Crypto info
        trades_skipped_crypto = len(self.trades_skipped_crypto)
        crypto_skipped_symbols = ','.join([t['symbol'] for t in self.trades_skipped_crypto]) if self.trades_skipped_crypto else ''

        # Swarm info
        trades_skipped_swarm = len(self.trades_skipped_swarm)
        swarm_skipped_symbols = ','.join([t['symbol'] for t in self.trades_skipped_swarm]) if self.trades_skipped_swarm else ''
        swarm_validations_count = len(self.swarm_validations)

        summary = {
            'date': date.today().isoformat(),
            'symbols_scanned': len(self.tradeable_symbols),
            'candidates_locked': len(self.locked_candidates),
            'trades_executed': len(self.active_trades),
            'trades_skipped_regime': trades_skipped,
            'skipped_symbols': skipped_symbols,
            'trades_skipped_crypto': trades_skipped_crypto,
            'crypto_skipped_symbols': crypto_skipped_symbols,
            'max_crypto_liquidations_usd': self.total_crypto_liquidations_session,
            'crypto_position_multiplier': self.crypto_position_multiplier,
            'daily_pnl': get_daily_pnl(),
            'regime': regime,
            'regime_confidence': regime_confidence,
            'trades_skipped_swarm': trades_skipped_swarm,
            'swarm_skipped_symbols': swarm_skipped_symbols,
            'swarm_validations_count': swarm_validations_count,
            'dry_run': self.dry_run,
            'position_size': self.position_size,
        }

        # Write summary
        with open(summary_file, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=summary.keys())
            writer.writeheader()
            writer.writerow(summary)

        # Display summary
        cprint(f"\n[SUMMARY] Daily summary saved to: {summary_file}", "green")
        cprint(f"[SUMMARY] Daily P&L: ${summary['daily_pnl']:+,.2f}",
               "green" if summary['daily_pnl'] >= 0 else "red")
        cprint(f"[SUMMARY] Regime: {regime} ({regime_confidence}% confidence)", "cyan")

        if trades_skipped > 0:
            cprint(f"[SUMMARY] Trades skipped due to regime: {trades_skipped}", "yellow")
            for skip in self.trades_skipped_regime:
                cprint(f"  - {skip['symbol']} gap {skip['direction']}: {skip['reason']}", "yellow")

        # Crypto stats
        if trades_skipped_crypto > 0:
            cprint(f"[SUMMARY] Trades skipped due to crypto conditions: {trades_skipped_crypto}", "red")
            for skip in self.trades_skipped_crypto:
                cprint(f"  - {skip['symbol']} gap {skip['direction']}: {skip['reason']}", "red")

        if self.total_crypto_liquidations_session > 0:
            cprint(f"[SUMMARY] Max 15m crypto liquidations: ${self.total_crypto_liquidations_session:,.0f}", "cyan")

        if self.crypto_position_multiplier < 1.0:
            cprint(f"[SUMMARY] Crypto position multiplier applied: {self.crypto_position_multiplier:.0%}", "yellow")

        # Swarm stats
        if swarm_validations_count > 0:
            cprint(f"[SUMMARY] AI Swarm validations: {swarm_validations_count}", "magenta")

        if trades_skipped_swarm > 0:
            cprint(f"[SUMMARY] Trades skipped due to AI swarm: {trades_skipped_swarm}", "magenta")
            for skip in self.trades_skipped_swarm:
                cprint(f"  - {skip['symbol']}: {skip['reason']}", "magenta")

        return summary

    # =========================================================================
    # MAIN WORKFLOW
    # =========================================================================

    def run(self):
        """Main bot workflow."""
        self.running = True
        self.set_state(BotState.INITIALIZING)

        cprint("\n" + "=" * 60, "cyan")
        cprint("  GAP AND GO TRADING BOT", "cyan", attrs=['bold'])
        cprint("=" * 60, "cyan")

        # Print configuration
        mode = "LIVE TRADING" if self.live_trading else "PAPER TRADING"
        if self.dry_run:
            mode = "DRY RUN (no orders)"
        if self.scan_only:
            mode = "SCAN ONLY"

        cprint(f"\n[CONFIG] Mode: {mode}", "white")
        cprint(f"[CONFIG] Position Size: ${self.position_size:,.0f}", "white")
        cprint(f"[CONFIG] Max Positions: {self.max_positions}", "white")
        cprint(f"[CONFIG] Tradeable Symbols: {', '.join(self.tradeable_symbols)}", "white")

        # Display symbol parameters
        cprint("\n[CONFIG] Symbol Parameters:", "white")
        for symbol, params in self.symbol_params.items():
            cprint(
                f"  {symbol}: Gap {params.get('gap_threshold')}% | "
                f"TP {params.get('profit_target')}% | SL {params.get('stop_loss')}%",
                "white"
            )

        # Check for recovery state
        if self.load_state():
            cprint("\n[INIT] Recovered state from previous session", "cyan")

        # Check for existing positions
        self._sync_existing_positions()

        # Reset daily tracker
        reset_daily_tracker()

        # Market status check
        status = get_market_status()
        cprint(f"\n[MARKET] Current Status: {status}", "cyan")

        if is_weekend() or is_market_holiday():
            cprint("[MARKET] Market is closed today", "yellow")
            if not self.scan_only:
                self.set_state(BotState.IDLE)
                return

        # Main workflow loop
        try:
            self._run_workflow_loop()
        except KeyboardInterrupt:
            self._handle_shutdown(None, None)
        except Exception as e:
            cprint(f"\n[ERROR] Unexpected error: {e}", "red")
            import traceback
            traceback.print_exc()
            self._handle_shutdown(None, None)

    def _sync_existing_positions(self):
        """Check for and sync any existing positions."""
        positions = get_all_positions()
        our_positions = [p for p in positions if p['symbol'] in self.tradeable_symbols]

        if our_positions:
            cprint("\n[INIT] Found existing positions:", "yellow")
            for pos in our_positions:
                cprint(
                    f"  {pos['symbol']}: {pos['qty']:.4f} shares @ ${pos['avg_entry_price']:.2f} "
                    f"(P&L: ${pos['unrealized_pnl']:+.2f})",
                    "yellow"
                )
                # Add to active trades if not already tracked
                if pos['symbol'] not in self.active_trades:
                    self.active_trades[pos['symbol']] = {
                        'entry_price': pos['avg_entry_price'],
                        'shares': pos['qty'],
                        'side': 'buy' if pos['qty'] > 0 else 'sell',
                        'direction': pos['side'].upper(),
                        'entry_time': 'RECOVERED',
                    }

    def _run_workflow_loop(self):
        """Main workflow loop handling all phases."""
        while self.running:
            now = get_et_now()
            current_time = now.time()

            # ─────────────────────────────────────────────────────────
            # SCAN ONLY MODE
            # ─────────────────────────────────────────────────────────
            if self.scan_only:
                self.run_premarket_scan()
                cprint("\n[BOT] Scan complete. Exiting (--scan-only mode)", "cyan")
                break

            # ─────────────────────────────────────────────────────────
            # LATE START HANDLER: Started during market hours (9:32 AM - 11:00 AM)
            # If we missed the normal workflow, do an immediate scan and trade
            # ─────────────────────────────────────────────────────────
            if (self.state == BotState.INITIALIZING and
                current_time >= EXECUTION_TIME and
                current_time < SESSION_END_TIME):

                cprint("\n" + "=" * 60, "yellow")
                cprint("  LATE START - MARKET IS OPEN", "yellow", attrs=['bold'])
                cprint("  Running immediate scan and trade check...", "yellow")
                cprint("=" * 60, "yellow")

                # Run immediate scan
                self.set_state(BotState.FINAL_SCAN)
                self.run_final_scan()

                # If we have candidates, execute trades
                if self.locked_candidates:
                    self.set_state(BotState.TRADING)
                    self.execute_trades()
                    self.set_state(BotState.MONITORING_POSITIONS)
                else:
                    cprint("\n[LATE START] No candidates meeting thresholds", "yellow")
                    self.set_state(BotState.MONITORING_POSITIONS)
                continue

            # ─────────────────────────────────────────────────────────
            # SESSION OVER: Started after 11:00 AM
            # ─────────────────────────────────────────────────────────
            if self.state == BotState.INITIALIZING and current_time >= SESSION_END_TIME:
                cprint("\n" + "=" * 60, "yellow")
                cprint("  SESSION ENDED", "yellow", attrs=['bold'])
                cprint(f"  Trading session ends at {SESSION_END_TIME.strftime('%H:%M')} ET", "yellow")
                cprint("  Start the bot before 9:32 AM ET tomorrow.", "yellow")
                cprint("=" * 60, "yellow")
                self.set_state(BotState.IDLE)
                break

            # ─────────────────────────────────────────────────────────
            # PHASE A: Pre-Market Monitoring (4:00 AM - 9:25 AM)
            # ─────────────────────────────────────────────────────────
            if current_time < FINAL_SCAN_TIME:
                if current_time < PREMARKET_START:
                    self.set_state(BotState.WAITING_PREMARKET)
                    self.wait_for_premarket()
                    continue

                self.set_state(BotState.MONITORING)

                # Run scan if interval has passed
                should_scan = (
                    self.last_scan_time is None or
                    (now - self.last_scan_time).total_seconds() >= SCAN_INTERVAL_MINUTES * 60
                )

                if should_scan:
                    self.run_premarket_scan()

                # Wait until next scan or final scan time
                sleep_seconds = min(60, self._seconds_until(FINAL_SCAN_TIME))
                if sleep_seconds > 0:
                    time.sleep(sleep_seconds)
                continue

            # ─────────────────────────────────────────────────────────
            # PHASE B: Final Scan (9:25 AM)
            # ─────────────────────────────────────────────────────────
            if current_time >= FINAL_SCAN_TIME and current_time < EXECUTION_TIME:
                if self.state != BotState.READY_TO_TRADE:
                    self.set_state(BotState.FINAL_SCAN)
                    self.run_final_scan()
                    self.set_state(BotState.READY_TO_TRADE)

                # Wait for execution time
                sleep_seconds = self._seconds_until(EXECUTION_TIME)
                if sleep_seconds > 0:
                    cprint(f"\n[BOT] Waiting {sleep_seconds / 60:.1f} min for execution at {EXECUTION_TIME.strftime('%H:%M')} ET", "cyan")
                    time.sleep(min(sleep_seconds, 60))
                continue

            # ─────────────────────────────────────────────────────────
            # PHASE C: Trade Execution (9:32 AM)
            # ─────────────────────────────────────────────────────────
            if current_time >= EXECUTION_TIME and self.state == BotState.READY_TO_TRADE:
                self.set_state(BotState.TRADING)
                self.execute_trades()
                self.set_state(BotState.MONITORING_POSITIONS)

            # ─────────────────────────────────────────────────────────
            # PHASE D: Position Monitoring (9:32 AM - 11:00 AM)
            # ─────────────────────────────────────────────────────────
            if self.state == BotState.MONITORING_POSITIONS and current_time < SESSION_END_TIME:
                has_positions = self.monitor_positions()

                if not has_positions or self.check_positions_closed():
                    cprint("\n[MONITOR] All positions closed (TP/SL hit)", "green")
                    self.set_state(BotState.IDLE)
                else:
                    time.sleep(60)  # Check every minute
                continue

            # ─────────────────────────────────────────────────────────
            # PHASE E: Session End (11:00 AM)
            # ─────────────────────────────────────────────────────────
            if current_time >= SESSION_END_TIME and self.state == BotState.MONITORING_POSITIONS:
                self.set_state(BotState.CLOSING_POSITIONS)
                self.close_remaining_positions()
                self._generate_daily_summary()
                self.set_state(BotState.IDLE)

            # ─────────────────────────────────────────────────────────
            # IDLE STATE
            # ─────────────────────────────────────────────────────────
            if self.state == BotState.IDLE:
                cprint("\n[BOT] Session complete. Bot is idle.", "green")
                cprint("[BOT] Press Ctrl+C to exit or wait for next trading day.", "green")
                break

    def _seconds_until(self, target_time: dt_time) -> float:
        """Calculate seconds until target time today."""
        now = get_et_now()
        target_dt = now.replace(
            hour=target_time.hour,
            minute=target_time.minute,
            second=0,
            microsecond=0
        )
        delta = (target_dt - now).total_seconds()
        return max(0, delta)


# ============================================================================
# WORKFLOW SIMULATION (for dry-run testing)
# ============================================================================

def simulate_workflow():
    """Simulate the full workflow for testing."""
    cprint("\n" + "=" * 60, "magenta")
    cprint("  WORKFLOW SIMULATION (DRY RUN)", "magenta", attrs=['bold'])
    cprint("=" * 60, "magenta")

    symbol_params = get_symbol_params()

    # Detect regime first
    cprint("\n[SIMULATE] REGIME DETECTION:", "cyan")
    regime = None
    regime_rules = REGIME_RULES['RANGING']

    if REGIME_DETECTOR_AVAILABLE:
        try:
            analysis = detect_regime_verbose('SPY')
            regime = analysis['regime']
            confidence = analysis.get('confidence', 0)
            reasons = analysis.get('reasons', [])
            regime_rules = REGIME_RULES.get(regime, REGIME_RULES['RANGING'])

            color = 'green' if regime == 'TREND_UP' else 'red' if regime == 'TREND_DOWN' else 'yellow' if regime == 'VOLATILE' else 'cyan'
            cprint(f"[SIMULATE] Market Regime: {regime} ({confidence}% confidence)", color, attrs=['bold'])
            cprint(f"[SIMULATE] {regime_rules['description']}", color)

            for reason in reasons[:3]:
                cprint(f"[SIMULATE]   - {reason}", "white")

        except Exception as e:
            cprint(f"[SIMULATE] Regime detection error: {e}", "red")
            regime = 'RANGING'
    else:
        cprint("[SIMULATE] Regime detector not available - using RANGING", "yellow")
        regime = 'RANGING'

    # Phase A: Pre-market
    cprint("\n[SIMULATE] PHASE A: Pre-Market Monitoring", "cyan")
    cprint("[SIMULATE] Would scan every 5 minutes from 4:00 AM - 9:25 AM ET", "white")

    # Run current scan
    gaps = get_current_gaps()
    if gaps:
        cprint("\n[SIMULATE] Current gap status:", "white")
        for symbol, info in gaps.items():
            gap_pct = info['gap_pct']
            threshold = info['gap_threshold']
            status = "TRIGGER" if info['meets_threshold'] else "below threshold"
            cprint(f"  {symbol}: {gap_pct:+.2f}% (needs {threshold}%) - {status}", "white")

    # Phase B: Final scan
    cprint("\n[SIMULATE] PHASE B: Final Scan at 9:25 AM", "cyan")
    candidates = get_gap_candidates()
    if candidates:
        cprint(f"[SIMULATE] Would lock {len(candidates)} candidate(s):", "green")
        for c in candidates:
            cprint(f"  {c['symbol']}: Gap {c['gap_direction']} {c['abs_gap_pct']:.2f}%", "green")
    else:
        cprint("[SIMULATE] No candidates meeting thresholds", "yellow")

    # Phase C: Execution with regime filtering
    cprint("\n[SIMULATE] PHASE C: Trade Execution at 9:32 AM", "cyan")
    if candidates:
        executed = 0
        skipped = 0
        for c in candidates:
            direction = c['gap_direction']

            # Apply regime filter
            if direction not in regime_rules['allowed_directions']:
                cprint(f"[SIMULATE] SKIP {c['symbol']}: Gap {direction} not allowed in {regime} regime", "yellow")
                skipped += 1
                continue

            side = 'BUY' if direction == 'UP' else 'SELL'
            shares = DEFAULT_POSITION_SIZE / c['entry_price']

            # Apply regime adjustments
            shares = shares * regime_rules['position_size_mult']
            sl_price = c['sl_price']

            if regime_rules['stop_loss_mult'] != 1.0:
                sl_distance = abs(c['entry_price'] - sl_price) * regime_rules['stop_loss_mult']
                if direction == 'UP':
                    sl_price = c['entry_price'] - sl_distance
                else:
                    sl_price = c['entry_price'] + sl_distance

            cprint(f"[SIMULATE] Would execute: {side} {shares:.4f} {c['symbol']} @ ~${c['entry_price']:.2f}", "white")
            cprint(f"            TP: ${c['tp_price']:.2f} | SL: ${sl_price:.2f}", "white")

            if regime_rules['position_size_mult'] != 1.0:
                cprint(f"            (Size reduced to {regime_rules['position_size_mult']*100:.0f}% due to {regime} regime)", "yellow")
            if regime_rules['stop_loss_mult'] != 1.0:
                cprint(f"            (SL widened {regime_rules['stop_loss_mult']*100:.0f}% due to {regime} regime)", "yellow")

            executed += 1

        cprint(f"\n[SIMULATE] Would execute {executed} trades, skip {skipped} due to regime filter", "cyan")
    else:
        cprint("[SIMULATE] No trades to execute", "yellow")

    # Phase D: Monitoring
    cprint("\n[SIMULATE] PHASE D: Position Monitoring 9:32 AM - 11:00 AM", "cyan")
    cprint("[SIMULATE] Would monitor positions every minute, log P&L every 5 minutes", "white")
    cprint("[SIMULATE] Bracket orders would auto-exit at TP or SL", "white")

    # Phase E: Session end
    cprint("\n[SIMULATE] PHASE E: Session End at 11:00 AM", "cyan")
    cprint("[SIMULATE] Would close any remaining positions at market", "white")
    cprint("[SIMULATE] Would generate daily summary with regime info", "white")

    cprint("\n" + "=" * 60, "magenta")
    cprint("  SIMULATION COMPLETE", "magenta")
    cprint("=" * 60 + "\n", "magenta")


# ============================================================================
# POSITION LOCK TESTS
# ============================================================================

def test_position_locks():
    """
    Test that position locks prevent dangerous parameter changes while in position.

    Demonstrates:
    1. Parameters can be changed freely with no positions
    2. Locks are applied when position opens
    3. Attempting to widen stop loss raises PositionLockViolation
    4. Attempting to increase position size raises PositionLockViolation
    5. Attempting to change daily loss limit raises PositionLockViolation
    6. Attempting to increase max positions raises PositionLockViolation
    7. Tightening stop loss (decreasing %) is allowed
    8. Decreasing position size is allowed
    9. Locks are released when position closes
    """
    cprint("\n" + "=" * 60, "cyan")
    cprint("  POSITION LOCK TEST", "cyan", attrs=['bold'])
    cprint("=" * 60, "cyan")

    # Create bot in dry-run mode for testing
    bot = GapAndGoBot(
        position_size=1000,
        max_positions=2,
        dry_run=True,
        scan_only=True,
    )

    # Ensure AAPL is in symbol_params for testing
    if 'AAPL' not in bot.symbol_params:
        bot.symbol_params['AAPL'] = {'stop_loss': 10.0, 'profit_target': 8.0}

    cprint("\n[TEST 1] No positions - all changes should work", "white")
    cprint("-" * 50, "white")

    try:
        bot.set_position_size(1500)
        cprint("  PASS: position_size increased to $1,500", "green")
    except PositionLockViolation as e:
        cprint(f"  FAIL: {e}", "red")

    try:
        bot.set_stop_loss_pct('AAPL', 15.0)  # Widen stop loss
        cprint("  PASS: stop_loss widened to 15%", "green")
    except PositionLockViolation as e:
        cprint(f"  FAIL: {e}", "red")

    try:
        bot.set_daily_loss_limit(5.0)
        cprint("  PASS: daily_loss_limit changed to 5%", "green")
    except PositionLockViolation as e:
        cprint(f"  FAIL: {e}", "red")

    try:
        bot.set_max_positions(3)
        cprint("  PASS: max_positions increased to 3", "green")
    except PositionLockViolation as e:
        cprint(f"  FAIL: {e}", "red")

    cprint("\n[TEST 2] Simulating position open - locks should be applied", "white")
    cprint("-" * 50, "white")

    # Simulate opening a position
    bot.active_trades['AAPL'] = {
        'order_id': 'test-123',
        'entry_price': 150.0,
        'tp_price': 162.0,
        'sl_price': 135.0,
        'shares': 10,
        'side': 'buy',
        'direction': 'LONG',
        'entry_time': datetime.now().isoformat(),
    }
    bot._apply_position_locks('AAPL')

    cprint(f"\n  Lock status: {bot.get_lock_status()}", "cyan")

    cprint("\n[TEST 3] With position - DANGEROUS changes should be BLOCKED", "white")
    cprint("-" * 50, "white")

    # Test: Widen stop loss (should fail)
    cprint("\n  Attempting to WIDEN stop loss from 15% to 20%...", "yellow")
    try:
        bot.set_stop_loss_pct('AAPL', 20.0)
        cprint("  FAIL: Stop loss widening was NOT blocked!", "red")
    except PositionLockViolation as e:
        cprint(f"  PASS: Correctly blocked - {e}", "green")

    # Test: Increase position size (should fail)
    cprint("\n  Attempting to INCREASE position size from $1,500 to $2,000...", "yellow")
    try:
        bot.set_position_size(2000)
        cprint("  FAIL: Position size increase was NOT blocked!", "red")
    except PositionLockViolation as e:
        cprint(f"  PASS: Correctly blocked - {e}", "green")

    # Test: Change daily loss limit (should fail)
    cprint("\n  Attempting to CHANGE daily loss limit from 5% to 10%...", "yellow")
    try:
        bot.set_daily_loss_limit(10.0)
        cprint("  FAIL: Daily loss limit change was NOT blocked!", "red")
    except PositionLockViolation as e:
        cprint(f"  PASS: Correctly blocked - {e}", "green")

    # Test: Increase max positions (should fail)
    cprint("\n  Attempting to INCREASE max positions from 3 to 5...", "yellow")
    try:
        bot.set_max_positions(5)
        cprint("  FAIL: Max positions increase was NOT blocked!", "red")
    except PositionLockViolation as e:
        cprint(f"  PASS: Correctly blocked - {e}", "green")

    cprint("\n[TEST 4] With position - SAFE changes should be ALLOWED", "white")
    cprint("-" * 50, "white")

    # Test: Tighten stop loss (should work)
    cprint("\n  Attempting to TIGHTEN stop loss from 15% to 8%...", "yellow")
    try:
        bot.set_stop_loss_pct('AAPL', 8.0)
        cprint("  PASS: Stop loss tightening allowed", "green")
    except PositionLockViolation as e:
        cprint(f"  FAIL: Should have allowed tightening - {e}", "red")

    # Test: Decrease position size (should work)
    cprint("\n  Attempting to DECREASE position size from $1,500 to $800...", "yellow")
    try:
        bot.set_position_size(800)
        cprint("  PASS: Position size decrease allowed", "green")
    except PositionLockViolation as e:
        cprint(f"  FAIL: Should have allowed decrease - {e}", "red")

    # Test: Decrease max positions (should work)
    cprint("\n  Attempting to DECREASE max positions from 3 to 1...", "yellow")
    try:
        bot.set_max_positions(1)
        cprint("  PASS: Max positions decrease allowed", "green")
    except PositionLockViolation as e:
        cprint(f"  FAIL: Should have allowed decrease - {e}", "red")

    cprint("\n[TEST 5] Position close - locks should be released", "white")
    cprint("-" * 50, "white")

    # Simulate closing the position
    del bot.active_trades['AAPL']
    bot._release_position_locks('AAPL')

    cprint(f"\n  Lock status: {bot.get_lock_status()}", "cyan")

    # Now all changes should work again
    cprint("\n  Attempting to WIDEN stop loss after position closed...", "yellow")
    try:
        bot.set_stop_loss_pct('AAPL', 25.0)
        cprint("  PASS: Stop loss widening now allowed (no position)", "green")
    except PositionLockViolation as e:
        cprint(f"  FAIL: Should work with no position - {e}", "red")

    cprint("\n" + "=" * 60, "cyan")
    cprint("  POSITION LOCK TEST COMPLETE", "cyan", attrs=['bold'])
    cprint("=" * 60 + "\n", "cyan")


# ============================================================================
# MAIN
# ============================================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Gap and Go Trading Bot")
    parser.add_argument("--live", action="store_true", help="Enable live trading (requires confirmation)")
    parser.add_argument("--scan-only", action="store_true", help="Only run scanner, no trading")
    parser.add_argument("--dry-run", action="store_true", help="Full workflow but no order execution")
    parser.add_argument("--position-size", type=float, default=DEFAULT_POSITION_SIZE, help="Position size in USD")
    parser.add_argument("--max-positions", type=int, default=MAX_POSITIONS, help="Maximum concurrent positions")
    parser.add_argument("--simulate", action="store_true", help="Run workflow simulation")
    parser.add_argument("--test-locks", action="store_true", help="Run position lock tests")

    args = parser.parse_args()

    # Live trading confirmation
    if args.live:
        cprint("\n" + "=" * 60, "red")
        cprint("  WARNING: LIVE TRADING MODE", "red", attrs=['bold'])
        cprint("  This will execute REAL trades with REAL money!", "red")
        cprint("=" * 60, "red")

        confirm = input("\nType 'CONFIRM' to proceed with live trading: ")
        if confirm != 'CONFIRM':
            cprint("Live trading cancelled.", "yellow")
            sys.exit(0)

    # Run simulation if requested
    if args.simulate:
        simulate_workflow()
        sys.exit(0)

    # Run position lock tests if requested
    if args.test_locks:
        test_position_locks()
        sys.exit(0)

    # Create and run bot
    bot = GapAndGoBot(
        position_size=args.position_size,
        max_positions=args.max_positions,
        dry_run=args.dry_run,
        scan_only=args.scan_only,
        live_trading=args.live,
    )

    bot.run()
