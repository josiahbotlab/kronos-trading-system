"""
Risk Management Agent

Moon Dev's risk philosophy: Protect capital first, profits second.
- Daily drawdown limit: 2% ($2,000)
- Max single position: 5% ($5,000)
- Max total exposure: 20% ($20,000)

Usage:
    from src.agents.risk_agent import check_risk, RiskAgent

    # Quick check before trade
    if check_risk('AAPL', 500):
        execute_trade()

    # Or use the agent directly
    agent = RiskAgent()
    allowed, reason = agent.check_trade('AAPL', 500)
"""

import csv
import os
from datetime import datetime, date
from pathlib import Path

# Portfolio configuration
PORTFOLIO_VALUE = 100_000  # Base portfolio value
DAILY_LOSS_LIMIT_PCT = 0.02  # 2% max daily loss
MAX_POSITION_PCT = 0.05  # 5% max single position
MAX_EXPOSURE_PCT = 0.20  # 20% max total exposure

# Calculate dollar limits
DAILY_LOSS_LIMIT = PORTFOLIO_VALUE * DAILY_LOSS_LIMIT_PCT  # $2,000
MAX_POSITION_SIZE = PORTFOLIO_VALUE * MAX_POSITION_PCT  # $5,000
MAX_TOTAL_EXPOSURE = PORTFOLIO_VALUE * MAX_EXPOSURE_PCT  # $20,000

# Paths
CSV_DIR = Path(__file__).parent.parent.parent / 'csvs'
RISK_LOG_PATH = CSV_DIR / 'risk_log.csv'


class RiskAgent:
    """
    Risk management agent that enforces trading limits.

    Moon Dev says: "The best trade is sometimes no trade at all."
    """

    def __init__(self, portfolio_value=PORTFOLIO_VALUE):
        self.portfolio_value = portfolio_value
        self.daily_loss_limit = portfolio_value * DAILY_LOSS_LIMIT_PCT
        self.max_position_size = portfolio_value * MAX_POSITION_PCT
        self.max_total_exposure = portfolio_value * MAX_EXPOSURE_PCT

        # Track daily P&L and positions
        self.daily_pnl = 0.0
        self.current_date = date.today()
        self.open_positions = {}  # symbol -> position_value

        # Ensure CSV directory exists
        CSV_DIR.mkdir(parents=True, exist_ok=True)
        self._init_csv()

    def _init_csv(self):
        """Initialize risk log CSV with headers if needed."""
        if not RISK_LOG_PATH.exists():
            with open(RISK_LOG_PATH, 'w', newline='') as f:
                writer = csv.writer(f)
                writer.writerow([
                    'timestamp',
                    'symbol',
                    'proposed_size',
                    'check_type',
                    'current_value',
                    'limit_value',
                    'result',
                    'reason'
                ])

    def _log_check(self, symbol, proposed_size, check_type, current_value, limit_value, result, reason):
        """Log a risk check to CSV."""
        with open(RISK_LOG_PATH, 'a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([
                datetime.now().isoformat(),
                symbol,
                f"{proposed_size:.2f}",
                check_type,
                f"{current_value:.2f}",
                f"{limit_value:.2f}",
                'PASS' if result else 'FAIL',
                reason
            ])

    def _reset_daily_if_needed(self):
        """Reset daily P&L if it's a new day."""
        today = date.today()
        if today != self.current_date:
            self.daily_pnl = 0.0
            self.current_date = today

    def update_pnl(self, pnl_change):
        """Update daily P&L with a realized gain/loss."""
        self._reset_daily_if_needed()
        self.daily_pnl += pnl_change

    def set_position(self, symbol, value):
        """Set or update an open position's value."""
        if value <= 0:
            self.open_positions.pop(symbol, None)
        else:
            self.open_positions[symbol] = value

    def close_position(self, symbol, pnl=0.0):
        """Close a position and record P&L."""
        self.open_positions.pop(symbol, None)
        self.update_pnl(pnl)

    def get_total_exposure(self):
        """Get total value of all open positions."""
        return sum(self.open_positions.values())

    def check_daily_loss(self, symbol, proposed_size):
        """Check if daily loss limit has been hit."""
        self._reset_daily_if_needed()

        # If we're down more than the limit, block all trading
        if self.daily_pnl <= -self.daily_loss_limit:
            reason = f"Daily loss limit hit: ${self.daily_pnl:.2f} <= -${self.daily_loss_limit:.2f}"
            self._log_check(symbol, proposed_size, 'DAILY_LOSS',
                          self.daily_pnl, -self.daily_loss_limit, False, reason)
            return False, reason

        reason = f"Daily P&L OK: ${self.daily_pnl:.2f} > -${self.daily_loss_limit:.2f}"
        self._log_check(symbol, proposed_size, 'DAILY_LOSS',
                       self.daily_pnl, -self.daily_loss_limit, True, reason)
        return True, reason

    def check_position_size(self, symbol, proposed_size):
        """Check if proposed position size is within limits."""
        if proposed_size > self.max_position_size:
            reason = f"Position too large: ${proposed_size:.2f} > ${self.max_position_size:.2f} (5% limit)"
            self._log_check(symbol, proposed_size, 'POSITION_SIZE',
                          proposed_size, self.max_position_size, False, reason)
            return False, reason

        reason = f"Position size OK: ${proposed_size:.2f} <= ${self.max_position_size:.2f}"
        self._log_check(symbol, proposed_size, 'POSITION_SIZE',
                       proposed_size, self.max_position_size, True, reason)
        return True, reason

    def check_total_exposure(self, symbol, proposed_size):
        """Check if adding this position would exceed max exposure."""
        current_exposure = self.get_total_exposure()
        new_exposure = current_exposure + proposed_size

        if new_exposure > self.max_total_exposure:
            reason = f"Exposure too high: ${new_exposure:.2f} > ${self.max_total_exposure:.2f} (20% limit)"
            self._log_check(symbol, proposed_size, 'TOTAL_EXPOSURE',
                          new_exposure, self.max_total_exposure, False, reason)
            return False, reason

        reason = f"Exposure OK: ${new_exposure:.2f} <= ${self.max_total_exposure:.2f}"
        self._log_check(symbol, proposed_size, 'TOTAL_EXPOSURE',
                       new_exposure, self.max_total_exposure, True, reason)
        return True, reason

    def check_trade(self, symbol, proposed_size):
        """
        Run all risk checks for a proposed trade.

        Returns:
            tuple: (allowed: bool, reason: str)
        """
        # Check 1: Daily loss limit
        passed, reason = self.check_daily_loss(symbol, proposed_size)
        if not passed:
            return False, reason

        # Check 2: Single position size
        passed, reason = self.check_position_size(symbol, proposed_size)
        if not passed:
            return False, reason

        # Check 3: Total exposure
        passed, reason = self.check_total_exposure(symbol, proposed_size)
        if not passed:
            return False, reason

        return True, "All risk checks passed"

    def get_status(self):
        """Get current risk status summary."""
        self._reset_daily_if_needed()
        total_exposure = self.get_total_exposure()

        return {
            'daily_pnl': self.daily_pnl,
            'daily_loss_limit': -self.daily_loss_limit,
            'daily_pnl_pct': (self.daily_pnl / self.portfolio_value) * 100,
            'total_exposure': total_exposure,
            'max_exposure': self.max_total_exposure,
            'exposure_pct': (total_exposure / self.portfolio_value) * 100,
            'positions': len(self.open_positions),
            'trading_allowed': self.daily_pnl > -self.daily_loss_limit
        }


# Global instance for simple function interface
_risk_agent = None


def get_risk_agent():
    """Get or create the global risk agent instance."""
    global _risk_agent
    if _risk_agent is None:
        _risk_agent = RiskAgent()
    return _risk_agent


def check_risk(symbol, proposed_size):
    """
    Quick risk check for a proposed trade.

    Args:
        symbol: Trading symbol (e.g., 'AAPL')
        proposed_size: Dollar value of proposed position

    Returns:
        bool: True if trade is allowed, False otherwise
    """
    agent = get_risk_agent()
    allowed, reason = agent.check_trade(symbol, proposed_size)
    return allowed


def check_risk_verbose(symbol, proposed_size):
    """
    Risk check with detailed reason.

    Returns:
        tuple: (allowed: bool, reason: str)
    """
    agent = get_risk_agent()
    return agent.check_trade(symbol, proposed_size)


def update_position(symbol, value):
    """Update an open position's value."""
    agent = get_risk_agent()
    agent.set_position(symbol, value)


def close_position(symbol, pnl=0.0):
    """Close a position and record realized P&L."""
    agent = get_risk_agent()
    agent.close_position(symbol, pnl)


def get_risk_status():
    """Get current risk status."""
    agent = get_risk_agent()
    return agent.get_status()


if __name__ == "__main__":
    # Demo/test the risk agent
    from termcolor import cprint

    agent = RiskAgent()

    cprint("\n=== Risk Agent Demo ===\n", "cyan")
    cprint(f"Portfolio Value: ${agent.portfolio_value:,.0f}", "white")
    cprint(f"Daily Loss Limit: ${agent.daily_loss_limit:,.0f} ({DAILY_LOSS_LIMIT_PCT*100}%)", "white")
    cprint(f"Max Position Size: ${agent.max_position_size:,.0f} ({MAX_POSITION_PCT*100}%)", "white")
    cprint(f"Max Total Exposure: ${agent.max_total_exposure:,.0f} ({MAX_EXPOSURE_PCT*100}%)", "white")

    # Test 1: Normal trade
    cprint("\n--- Test 1: Normal $500 trade ---", "yellow")
    allowed, reason = agent.check_trade('AAPL', 500)
    cprint(f"Result: {'ALLOWED' if allowed else 'BLOCKED'} - {reason}", "green" if allowed else "red")

    # Test 2: Position too large
    cprint("\n--- Test 2: $6000 trade (exceeds 5% limit) ---", "yellow")
    allowed, reason = agent.check_trade('TSLA', 6000)
    cprint(f"Result: {'ALLOWED' if allowed else 'BLOCKED'} - {reason}", "green" if allowed else "red")

    # Test 3: Exposure limit
    cprint("\n--- Test 3: Building up exposure ---", "yellow")
    agent.set_position('AAPL', 5000)
    agent.set_position('TSLA', 5000)
    agent.set_position('NVDA', 5000)
    agent.set_position('SPY', 4000)
    cprint(f"Current exposure: ${agent.get_total_exposure():,.0f}", "white")

    allowed, reason = agent.check_trade('GOOGL', 2000)
    cprint(f"Result: {'ALLOWED' if allowed else 'BLOCKED'} - {reason}", "green" if allowed else "red")

    # Test 4: Daily loss limit
    cprint("\n--- Test 4: Daily loss limit ---", "yellow")
    agent.update_pnl(-2500)  # Simulate $2500 loss
    cprint(f"Daily P&L: ${agent.daily_pnl:,.0f}", "red")

    allowed, reason = agent.check_trade('META', 500)
    cprint(f"Result: {'ALLOWED' if allowed else 'BLOCKED'} - {reason}", "green" if allowed else "red")

    # Show final status
    cprint("\n--- Risk Status ---", "cyan")
    status = agent.get_status()
    for key, value in status.items():
        cprint(f"  {key}: {value}", "white")

    cprint(f"\nRisk log written to: {RISK_LOG_PATH}", "cyan")
