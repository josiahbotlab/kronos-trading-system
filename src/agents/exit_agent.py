"""
Exit Management Agent

Moon Dev's philosophy: Exits are harder than entries.
- Protect profits with trailing stops
- Don't let winners turn into losers
- Take partial profits to lock in gains
- Time-based exits prevent capital lockup

Exit Rules (optimized via backtest - TP=6%, SL=2%):
1. Trailing Stop:
   - Up 1%: Move stop to breakeven
   - Up 2%+: Trail stop at 1% below current price

2. Time-Based Exit:
   - Close position after max hold time (default 2 hours)

3. Partial Profit Taking:
   - Close 50% at 3% gain (half of 6% TP target)
   - Let remaining 50% run to full 6% target

Usage:
    from src.agents.exit_agent import manage_exit, ExitAgent

    # Quick check for exit decision
    decision = manage_exit('AAPL', entry_price=150, entry_time=datetime(...),
                          current_price=153, stop_loss=147, take_profit=156)
    # Returns: 'HOLD', 'CLOSE_FULL', 'CLOSE_HALF', or 'MOVE_STOP'
"""

import csv
from datetime import datetime, timedelta
from pathlib import Path
from typing import Literal, Optional, Dict, Any

# Exit configuration - optimized via backtest (TP=6%, SL=2%)
BREAKEVEN_THRESHOLD = 0.01  # 1% gain to move stop to breakeven
TRAILING_THRESHOLD = 0.02   # 2% gain to start trailing
TRAIL_DISTANCE = 0.01       # Trail 1% below current price
PARTIAL_PROFIT_PCT = 0.03   # Take 50% profit at 3% gain (half of 6% TP)
MAX_HOLD_HOURS = 2          # Default max hold time in hours

# Exit decision types
ExitDecision = Literal['HOLD', 'CLOSE_FULL', 'CLOSE_HALF', 'MOVE_STOP']

# Paths
CSV_DIR = Path(__file__).parent.parent.parent / 'csvs'
EXIT_LOG_PATH = CSV_DIR / 'exit_log.csv'


class ExitAgent:
    """
    Exit management agent that handles trailing stops, partial profits, and time exits.

    Moon Dev says: "Your entry doesn't matter if your exit is trash."
    """

    def __init__(self, max_hold_hours=MAX_HOLD_HOURS):
        self.max_hold_hours = max_hold_hours

        # Track position states
        self.position_states = {}  # symbol -> state dict

        # Ensure CSV directory exists
        CSV_DIR.mkdir(parents=True, exist_ok=True)
        self._init_csv()

    def _init_csv(self):
        """Initialize exit log CSV with headers if needed."""
        if not EXIT_LOG_PATH.exists():
            with open(EXIT_LOG_PATH, 'w', newline='') as f:
                writer = csv.writer(f)
                writer.writerow([
                    'timestamp',
                    'symbol',
                    'entry_price',
                    'current_price',
                    'pnl_pct',
                    'hold_time_mins',
                    'decision',
                    'reason',
                    'new_stop',
                    'partial_taken'
                ])

    def _log_decision(self, symbol, entry_price, current_price, pnl_pct,
                      hold_time_mins, decision, reason, new_stop=None, partial_taken=False):
        """Log an exit decision to CSV."""
        with open(EXIT_LOG_PATH, 'a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([
                datetime.now().isoformat(),
                symbol,
                f"{entry_price:.2f}",
                f"{current_price:.2f}",
                f"{pnl_pct:.2f}",
                f"{hold_time_mins:.1f}",
                decision,
                reason,
                f"{new_stop:.2f}" if new_stop else "",
                "YES" if partial_taken else "NO"
            ])

    def get_position_state(self, symbol):
        """Get or create position state tracking."""
        if symbol not in self.position_states:
            self.position_states[symbol] = {
                'breakeven_triggered': False,
                'trailing_active': False,
                'trailing_stop': None,
                'partial_taken': False,
                'highest_price': None
            }
        return self.position_states[symbol]

    def reset_position_state(self, symbol):
        """Reset position state when position is closed."""
        if symbol in self.position_states:
            del self.position_states[symbol]

    def calculate_pnl_pct(self, entry_price, current_price, direction='LONG'):
        """Calculate P&L percentage."""
        if direction == 'LONG':
            return ((current_price - entry_price) / entry_price) * 100
        else:
            return ((entry_price - current_price) / entry_price) * 100

    def calculate_rr_ratio(self, entry_price, current_price, stop_loss, direction='LONG'):
        """Calculate current R:R ratio achieved."""
        if direction == 'LONG':
            risk = entry_price - stop_loss
            reward = current_price - entry_price
        else:
            risk = stop_loss - entry_price
            reward = entry_price - current_price

        if risk <= 0:
            return 0
        return reward / risk

    def check_time_exit(self, entry_time, symbol, entry_price, current_price, pnl_pct):
        """Check if position has exceeded max hold time."""
        hold_duration = datetime.now() - entry_time
        hold_minutes = hold_duration.total_seconds() / 60
        max_minutes = self.max_hold_hours * 60

        if hold_minutes >= max_minutes:
            reason = f"Max hold time ({self.max_hold_hours}h) exceeded - {hold_minutes:.0f} mins"
            self._log_decision(symbol, entry_price, current_price, pnl_pct,
                             hold_minutes, 'CLOSE_FULL', reason)
            return True, reason
        return False, None

    def check_trailing_stop(self, symbol, entry_price, current_price, direction='LONG'):
        """Check and update trailing stop logic."""
        state = self.get_position_state(symbol)
        pnl_pct = self.calculate_pnl_pct(entry_price, current_price, direction)

        # Track highest price for trailing
        if state['highest_price'] is None:
            state['highest_price'] = current_price
        elif direction == 'LONG' and current_price > state['highest_price']:
            state['highest_price'] = current_price
        elif direction == 'SHORT' and current_price < state['highest_price']:
            state['highest_price'] = current_price

        new_stop = None
        reason = None

        # Check if we should move to breakeven (1% gain)
        if pnl_pct >= BREAKEVEN_THRESHOLD * 100 and not state['breakeven_triggered']:
            state['breakeven_triggered'] = True
            new_stop = entry_price
            reason = f"Moving stop to breakeven (up {pnl_pct:.1f}%)"

        # Check if we should start trailing (2% gain)
        if pnl_pct >= TRAILING_THRESHOLD * 100:
            state['trailing_active'] = True

            if direction == 'LONG':
                trail_stop = state['highest_price'] * (1 - TRAIL_DISTANCE)
                # Only update if new trail stop is higher
                if state['trailing_stop'] is None or trail_stop > state['trailing_stop']:
                    state['trailing_stop'] = trail_stop
                    new_stop = trail_stop
                    reason = f"Trailing stop updated (high: ${state['highest_price']:.2f}, stop: ${trail_stop:.2f})"
            else:
                trail_stop = state['highest_price'] * (1 + TRAIL_DISTANCE)
                if state['trailing_stop'] is None or trail_stop < state['trailing_stop']:
                    state['trailing_stop'] = trail_stop
                    new_stop = trail_stop
                    reason = f"Trailing stop updated (low: ${state['highest_price']:.2f}, stop: ${trail_stop:.2f})"

        return new_stop, reason

    def check_partial_profit(self, symbol, entry_price, current_price, stop_loss, direction='LONG'):
        """Check if we should take partial profits at 3% gain (half of 6% TP)."""
        state = self.get_position_state(symbol)

        if state['partial_taken']:
            return False, None

        pnl_pct = self.calculate_pnl_pct(entry_price, current_price, direction) / 100

        if pnl_pct >= PARTIAL_PROFIT_PCT:
            state['partial_taken'] = True
            reason = f"Partial profit target hit ({pnl_pct*100:.1f}% >= {PARTIAL_PROFIT_PCT*100:.0f}%)"
            return True, reason

        return False, None

    def check_trailing_stop_hit(self, symbol, current_price, direction='LONG'):
        """Check if trailing stop has been hit."""
        state = self.get_position_state(symbol)

        if not state['trailing_active'] or state['trailing_stop'] is None:
            return False, None

        if direction == 'LONG' and current_price <= state['trailing_stop']:
            reason = f"Trailing stop hit (${current_price:.2f} <= ${state['trailing_stop']:.2f})"
            return True, reason
        elif direction == 'SHORT' and current_price >= state['trailing_stop']:
            reason = f"Trailing stop hit (${current_price:.2f} >= ${state['trailing_stop']:.2f})"
            return True, reason

        return False, None

    def manage_exit(self, symbol, entry_price, entry_time, current_price,
                    stop_loss, take_profit, direction='LONG') -> tuple[ExitDecision, str, Optional[float]]:
        """
        Main exit management function.

        Args:
            symbol: Trading symbol
            entry_price: Position entry price
            entry_time: Position entry datetime
            current_price: Current market price
            stop_loss: Current stop loss price
            take_profit: Take profit target
            direction: 'LONG' or 'SHORT'

        Returns:
            tuple: (decision, reason, new_stop_price)
            - decision: 'HOLD', 'CLOSE_FULL', 'CLOSE_HALF', or 'MOVE_STOP'
            - reason: Explanation string
            - new_stop_price: New stop price if MOVE_STOP, else None
        """
        state = self.get_position_state(symbol)
        pnl_pct = self.calculate_pnl_pct(entry_price, current_price, direction)
        hold_duration = datetime.now() - entry_time
        hold_minutes = hold_duration.total_seconds() / 60

        # Check 1: Time-based exit
        time_exit, time_reason = self.check_time_exit(
            entry_time, symbol, entry_price, current_price, pnl_pct
        )
        if time_exit:
            return 'CLOSE_FULL', time_reason, None

        # Check 2: Trailing stop hit
        trail_hit, trail_reason = self.check_trailing_stop_hit(symbol, current_price, direction)
        if trail_hit:
            self._log_decision(symbol, entry_price, current_price, pnl_pct,
                             hold_minutes, 'CLOSE_FULL', trail_reason,
                             partial_taken=state['partial_taken'])
            return 'CLOSE_FULL', trail_reason, None

        # Check 3: Partial profit taking (if not already taken)
        take_partial, partial_reason = self.check_partial_profit(
            symbol, entry_price, current_price, stop_loss, direction
        )
        if take_partial:
            self._log_decision(symbol, entry_price, current_price, pnl_pct,
                             hold_minutes, 'CLOSE_HALF', partial_reason, partial_taken=True)
            return 'CLOSE_HALF', partial_reason, None

        # Check 4: Update trailing stop
        new_stop, stop_reason = self.check_trailing_stop(
            symbol, entry_price, current_price, direction
        )
        if new_stop is not None:
            self._log_decision(symbol, entry_price, current_price, pnl_pct,
                             hold_minutes, 'MOVE_STOP', stop_reason,
                             new_stop=new_stop, partial_taken=state['partial_taken'])
            return 'MOVE_STOP', stop_reason, new_stop

        # Default: Hold position
        return 'HOLD', f"Holding (P&L: {pnl_pct:+.2f}%, {hold_minutes:.0f} mins)", None

    def get_position_status(self, symbol):
        """Get current exit management status for a position."""
        state = self.get_position_state(symbol)
        return {
            'breakeven_triggered': state['breakeven_triggered'],
            'trailing_active': state['trailing_active'],
            'trailing_stop': state['trailing_stop'],
            'partial_taken': state['partial_taken'],
            'highest_price': state['highest_price']
        }


# Global instance for simple function interface
_exit_agent = None


def get_exit_agent(max_hold_hours=MAX_HOLD_HOURS):
    """Get or create the global exit agent instance."""
    global _exit_agent
    if _exit_agent is None:
        _exit_agent = ExitAgent(max_hold_hours=max_hold_hours)
    return _exit_agent


def manage_exit(symbol, entry_price, entry_time, current_price,
                stop_loss, take_profit, direction='LONG') -> ExitDecision:
    """
    Quick exit check for a position.

    Args:
        symbol: Trading symbol
        entry_price: Position entry price
        entry_time: Position entry datetime
        current_price: Current market price
        stop_loss: Current stop loss price
        take_profit: Take profit target
        direction: 'LONG' or 'SHORT'

    Returns:
        str: 'HOLD', 'CLOSE_FULL', 'CLOSE_HALF', or 'MOVE_STOP'
    """
    agent = get_exit_agent()
    decision, reason, new_stop = agent.manage_exit(
        symbol, entry_price, entry_time, current_price,
        stop_loss, take_profit, direction
    )
    return decision


def manage_exit_verbose(symbol, entry_price, entry_time, current_price,
                        stop_loss, take_profit, direction='LONG'):
    """
    Exit check with detailed reason and new stop price.

    Returns:
        tuple: (decision, reason, new_stop_price)
    """
    agent = get_exit_agent()
    return agent.manage_exit(
        symbol, entry_price, entry_time, current_price,
        stop_loss, take_profit, direction
    )


def reset_position(symbol):
    """Reset position state when position is fully closed."""
    agent = get_exit_agent()
    agent.reset_position_state(symbol)


def get_position_status(symbol):
    """Get exit management status for a position."""
    agent = get_exit_agent()
    return agent.get_position_status(symbol)


if __name__ == "__main__":
    # Demo/test the exit agent
    from termcolor import cprint
    import time

    agent = ExitAgent(max_hold_hours=2)

    cprint("\n=== Exit Agent Demo ===\n", "cyan")
    cprint(f"Breakeven threshold: {BREAKEVEN_THRESHOLD*100}% gain", "white")
    cprint(f"Trailing threshold: {TRAILING_THRESHOLD*100}% gain", "white")
    cprint(f"Trail distance: {TRAIL_DISTANCE*100}% below high", "white")
    cprint(f"Partial profit: {PARTIAL_PROFIT_PCT*100}% gain (50%)", "white")
    cprint(f"Max hold time: {MAX_HOLD_HOURS} hours", "white")

    # Simulate a winning trade
    symbol = 'AAPL'
    entry_price = 150.00
    stop_loss = 147.00  # $3 risk
    take_profit = 156.00  # $6 reward (2:1)
    entry_time = datetime.now()

    cprint(f"\n--- Simulated Trade: {symbol} ---", "yellow")
    cprint(f"Entry: ${entry_price} | SL: ${stop_loss} | TP: ${take_profit}", "white")
    cprint(f"Risk: ${entry_price - stop_loss:.2f} | Reward: ${take_profit - entry_price:.2f}", "white")

    # Simulate price movements
    price_scenarios = [
        (150.50, "Small gain"),
        (151.50, "Up 1% - should trigger breakeven"),
        (153.00, "Up 2% - should start trailing"),
        (154.50, "Up 3% - should hit 1.5:1 R:R partial"),
        (155.00, "Continued rally"),
        (153.50, "Pullback - trailing stop should protect"),
    ]

    for price, description in price_scenarios:
        cprint(f"\n{description}: ${price:.2f}", "yellow")
        decision, reason, new_stop = agent.manage_exit(
            symbol, entry_price, entry_time, price,
            stop_loss, take_profit, 'LONG'
        )

        color = {
            'HOLD': 'white',
            'CLOSE_FULL': 'red',
            'CLOSE_HALF': 'magenta',
            'MOVE_STOP': 'green'
        }.get(decision, 'white')

        cprint(f"  Decision: {decision}", color)
        cprint(f"  Reason: {reason}", "cyan")
        if new_stop:
            cprint(f"  New Stop: ${new_stop:.2f}", "green")

        status = agent.get_position_status(symbol)
        cprint(f"  State: BE={status['breakeven_triggered']} | Trail={status['trailing_active']} | Partial={status['partial_taken']}", "white")

    # Test time-based exit
    cprint("\n--- Time-Based Exit Test ---", "yellow")
    old_entry = datetime.now() - timedelta(hours=3)
    decision, reason, _ = agent.manage_exit(
        'TSLA', 400.00, old_entry, 402.00, 395.00, 410.00, 'LONG'
    )
    cprint(f"3-hour old position: {decision}", "red" if decision == 'CLOSE_FULL' else "white")
    cprint(f"Reason: {reason}", "cyan")

    cprint(f"\nExit log written to: {EXIT_LOG_PATH}", "cyan")
