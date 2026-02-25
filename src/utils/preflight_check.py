"""
Pre-Flight Checklist — Moondev Methodology

Verifies 8 safety checks before trading begins.
If ANY check fails, the bot should not proceed.

Usage:
    from src.utils.preflight_check import run_preflight
    if not run_preflight():
        sys.exit(1)
"""

import os
import subprocess
import sys
from datetime import datetime

from termcolor import cprint
from dotenv import load_dotenv

load_dotenv()

sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))

# Strategy constants from smart_bot
from src.agents.smart_bot import (
    ORDER_USD_SIZE,
    BREAKOUT_TP_PCT, BREAKOUT_SL_PCT,
    MR_SL_PCT,
    GAP_TP_PCT, GAP_SL_PCT,
    MACD_TP_PCT, MACD_SL_PCT,
    BB_BOUNCE_TP_PCT, BB_BOUNCE_SL_PCT,
    MOMENTUM_TP_PCT, MOMENTUM_SL_PCT,
    SYMBOL_STRATEGY_OVERRIDE,
)

# Order utilities for API checks
from src.utils.order_utils import (
    get_api,
    get_account_info,
    get_all_positions,
    get_open_orders,
)

# ─────────────────────────────────────────────────────────────
# Constants
# ─────────────────────────────────────────────────────────────

LAUNCHD_LABELS = [
    'com.tradingbot.smartbot',
    'com.tradingbot.gapgo',
    'com.tradingbot.momentum',
    'com.tradingbot.breakout',
    'com.tradingbot.meanrev',
    'com.tradingbot.macd',
    'com.tradingbot.bbbounce',
    'com.tradingbot.analyzer',
]

# Maps strategy name -> (TP%, SL%). MEAN_REV TP is None (exits on SMA/RSI signal).
STRATEGY_RISK_MAP = {
    'BREAKOUT':    (BREAKOUT_TP_PCT, BREAKOUT_SL_PCT),
    'MEAN_REV':    (None, MR_SL_PCT),
    'GAP_AND_GO':  (GAP_TP_PCT, GAP_SL_PCT),
    'MACD':        (MACD_TP_PCT, MACD_SL_PCT),
    'BB_BOUNCE':   (BB_BOUNCE_TP_PCT, BB_BOUNCE_SL_PCT),
    'MOMENTUM':    (MOMENTUM_TP_PCT, MOMENTUM_SL_PCT),
}

MAX_POSITION_PCT = 10.0  # Max % of equity per position


# ─────────────────────────────────────────────────────────────
# Check 1: Risk Controls
# ─────────────────────────────────────────────────────────────

def check_risk_controls():
    """Verify TP/SL is defined for every strategy used by symbol overrides."""
    issues = []

    for strategy, (tp, sl) in STRATEGY_RISK_MAP.items():
        if sl is None or sl <= 0:
            issues.append(f"{strategy} has no stop loss defined")
        if tp is None and strategy != 'MEAN_REV':
            issues.append(f"{strategy} has no take profit defined")

    for symbol, strategy in SYMBOL_STRATEGY_OVERRIDE.items():
        if strategy == 'HOLD':
            continue
        if strategy not in STRATEGY_RISK_MAP:
            issues.append(f"{symbol} override '{strategy}' not in STRATEGY_RISK_MAP")

    if issues:
        return False, "Risk control gaps: " + "; ".join(issues)

    return True, f"All {len(STRATEGY_RISK_MAP)} strategies have TP/SL defined"


# ─────────────────────────────────────────────────────────────
# Check 2: Position Sizing
# ─────────────────────────────────────────────────────────────

def check_position_sizing():
    """Verify ORDER_USD_SIZE is < 10% of account equity."""
    try:
        account = get_account_info()
        equity = account.get('equity', 0)
    except Exception:
        equity = 0

    if equity <= 0:
        try:
            from src.agents.risk_agent import PORTFOLIO_VALUE
            equity = PORTFOLIO_VALUE
        except ImportError:
            equity = 100_000

    pct = (ORDER_USD_SIZE / equity) * 100

    if pct >= MAX_POSITION_PCT:
        return False, (
            f"Position size ${ORDER_USD_SIZE} is {pct:.1f}% of equity "
            f"${equity:,.0f} (max {MAX_POSITION_PCT:.0f}%)"
        )

    return True, (
        f"Position size ${ORDER_USD_SIZE} = {pct:.1f}% of "
        f"${equity:,.0f} equity (limit: {MAX_POSITION_PCT:.0f}%)"
    )


# ─────────────────────────────────────────────────────────────
# Check 3: No Duplicate Orders
# ─────────────────────────────────────────────────────────────

def check_no_duplicate_orders():
    """Verify no symbol has both an open position AND a same-direction pending order."""
    try:
        positions = get_all_positions()
        open_orders = get_open_orders()
    except Exception as e:
        return False, f"Could not check for duplicates: {e}"

    position_symbols = {p['symbol']: p['side'] for p in positions}
    duplicates = []

    for order in open_orders:
        symbol = order['symbol']
        if symbol in position_symbols:
            pos_side = position_symbols[symbol]
            order_side = order['side']
            if (pos_side == 'long' and order_side == 'buy') or \
               (pos_side == 'short' and order_side == 'sell'):
                duplicates.append(
                    f"{symbol} ({pos_side} position + {order_side} order)"
                )

    if duplicates:
        return False, "Duplicate orders detected: " + "; ".join(duplicates)

    return True, f"No duplicates ({len(positions)} positions, {len(open_orders)} open orders)"


# ─────────────────────────────────────────────────────────────
# Check 4: Market Hours
# ─────────────────────────────────────────────────────────────

def check_market_hours():
    """Verify market is open or will open within 5 minutes."""
    try:
        api = get_api()
        clock = api.get_clock()

        if clock.is_open:
            return True, "Market is currently OPEN"

        now = clock.timestamp
        next_open = clock.next_open

        try:
            diff_seconds = next_open.timestamp() - now.timestamp()
            minutes_until = diff_seconds / 60
        except Exception:
            minutes_until = 999

        if minutes_until <= 5:
            return True, f"Market opens in {minutes_until:.0f} minutes"

        if hasattr(next_open, 'strftime'):
            next_str = next_open.strftime('%Y-%m-%d %H:%M %Z')
        else:
            next_str = str(next_open)

        return False, f"Market closed. Next open: {next_str} ({minutes_until:.0f} min away)"

    except Exception as e:
        return False, f"Could not check market hours: {e}"


# ─────────────────────────────────────────────────────────────
# Check 5: API Connection
# ─────────────────────────────────────────────────────────────

def check_api_connection():
    """Verify Alpaca API connection works and account is active."""
    try:
        api = get_api()
        account = api.get_account()

        status = account.status
        paper_or_live = "PAPER" if os.getenv("ALPACA_PAPER", "true").lower() == "true" else "LIVE"

        if status != 'ACTIVE':
            return False, f"Account status is '{status}' (expected ACTIVE)"

        equity = float(account.equity)
        return True, f"API connected ({paper_or_live}) | Equity: ${equity:,.2f} | Status: {status}"

    except Exception as e:
        return False, f"API connection failed: {e}"


# ─────────────────────────────────────────────────────────────
# Check 6: Buying Power
# ─────────────────────────────────────────────────────────────

def check_buying_power():
    """Verify account has enough buying power for at least one position."""
    try:
        account = get_account_info()
        buying_power = account.get('buying_power', 0)

        if buying_power < ORDER_USD_SIZE:
            return False, (
                f"Insufficient buying power: ${buying_power:,.2f} "
                f"< ${ORDER_USD_SIZE} (min order size)"
            )

        max_positions = int(buying_power / ORDER_USD_SIZE)
        return True, (
            f"Buying power: ${buying_power:,.2f} "
            f"({max_positions} positions @ ${ORDER_USD_SIZE})"
        )

    except Exception as e:
        return False, f"Could not check buying power: {e}"


# ─────────────────────────────────────────────────────────────
# Check 7: No Conflicting Positions
# ─────────────────────────────────────────────────────────────

def check_no_conflicting_positions():
    """Verify no positions have losses beyond the widest SL (orphaned bracket orders)."""
    try:
        positions = get_all_positions()

        if not positions:
            return True, "No existing positions"

        max_sl_pct = max(sl for _, sl in STRATEGY_RISK_MAP.values() if sl is not None)

        critical = []
        for pos in positions:
            symbol = pos['symbol']
            pnl_pct = pos.get('unrealized_pnl_pct', 0)
            if pnl_pct < -max_sl_pct:
                critical.append(f"{symbol}: {pnl_pct:+.1f}% (beyond max SL of -{max_sl_pct}%)")

        if critical:
            return False, "Positions beyond max SL: " + "; ".join(critical)

        pos_summary = ", ".join(
            f"{p['symbol']} {p['side']} {p['qty']:.0f}sh ({p.get('unrealized_pnl_pct', 0):+.1f}%)"
            for p in positions
        )

        return True, f"{len(positions)} positions OK: {pos_summary}"

    except Exception as e:
        return False, f"Could not check positions: {e}"


# ─────────────────────────────────────────────────────────────
# Check 8: Scheduler Running
# ─────────────────────────────────────────────────────────────

def check_scheduler_running():
    """Verify launchd jobs are loaded."""
    try:
        result = subprocess.run(
            ['launchctl', 'list'],
            capture_output=True,
            text=True,
            timeout=10,
        )

        output = result.stdout
        missing = []
        found = []

        for label in LAUNCHD_LABELS:
            if label in output:
                found.append(label)
            else:
                missing.append(label)

        if missing:
            return False, (
                f"LaunchAgent(s) not loaded: {', '.join(missing)}. "
                f"Run: launchctl load ~/Library/LaunchAgents/<plist>"
            )

        return True, f"All {len(found)} LaunchAgents loaded"

    except subprocess.TimeoutExpired:
        return False, "launchctl timed out"
    except FileNotFoundError:
        return False, "launchctl not found (not macOS?)"
    except Exception as e:
        return False, f"Could not check scheduler: {e}"


# ─────────────────────────────────────────────────────────────
# Main orchestrator
# ─────────────────────────────────────────────────────────────

def run_preflight(mock_mode=False, verbose=True) -> bool:
    """
    Run all pre-flight checks before trading.

    Args:
        mock_mode: Skip API-dependent checks (4, 5, 6, 7 + duplicate orders)
        verbose: Print detailed output

    Returns:
        True if all checks pass, False if any fail
    """
    # (name, function, api_dependent)
    checks = [
        ("Risk Controls",            check_risk_controls,            False),
        ("Position Sizing",          check_position_sizing,          False),
        ("No Duplicate Orders",      check_no_duplicate_orders,      True),
        ("Market Hours",             check_market_hours,             True),
        ("API Connection",           check_api_connection,           True),
        ("Buying Power",             check_buying_power,             True),
        ("No Conflicting Positions", check_no_conflicting_positions, True),
        ("Scheduler Running",        check_scheduler_running,        False),
    ]

    mode_str = "MOCK MODE" if mock_mode else "LIVE"

    cprint("\n" + "=" * 60, "cyan")
    cprint("  PRE-FLIGHT CHECKLIST", "cyan", attrs=["bold"])
    cprint(f"  {mode_str} | {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}", "cyan")
    cprint("=" * 60, "cyan")

    all_passed = True
    passed_count = 0
    failed_count = 0
    skipped_count = 0

    for i, (name, check_fn, api_dependent) in enumerate(checks, 1):
        tag = f"[{i}/8]"

        if mock_mode and api_dependent:
            cprint(f"  {tag} {name:<26} [ SKIP ] (mock mode)", "yellow")
            skipped_count += 1
            continue

        try:
            passed, message = check_fn()
        except Exception as e:
            passed = False
            message = f"Unexpected error: {e}"

        if passed:
            cprint(f"  {tag} {name:<26} [ PASS ] {message}", "green")
            passed_count += 1
        else:
            cprint(f"  {tag} {name:<26} [ FAIL ] {message}", "red")
            failed_count += 1
            all_passed = False

    cprint("=" * 60, "cyan")

    if all_passed:
        cprint(
            f"  PREFLIGHT PASSED: {passed_count} passed, {skipped_count} skipped",
            "green", attrs=["bold"],
        )
    else:
        cprint(
            f"  PREFLIGHT FAILED: {failed_count} failed, {passed_count} passed, {skipped_count} skipped",
            "red", attrs=["bold"],
        )

    cprint("=" * 60 + "\n", "cyan")

    return all_passed
