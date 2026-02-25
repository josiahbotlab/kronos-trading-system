"""
Trading Bot Auto-Starter

Automatically starts trading bots at scheduled times:
- Gap and Go Bot: 3:55 AM ET (pre-market monitoring)
- Smart Bot: 9:30 AM ET (market open)

Runs Monday-Friday and logs all start/stop times.

Usage:
    python src/scheduler/auto_start.py           # Run scheduler
    python src/scheduler/auto_start.py --test    # Run immediately (test mode)
    python src/scheduler/auto_start.py --gap-go  # Start Gap and Go bot only

Background:
    nohup python src/scheduler/auto_start.py > logs/scheduler.log 2>&1 &
"""

import argparse
import csv
import os
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

import schedule
from termcolor import cprint

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.append(str(PROJECT_ROOT))

# Configuration
SMART_BOT_SCRIPT = PROJECT_ROOT / 'src' / 'agents' / 'smart_bot.py'
GAP_GO_BOT_SCRIPT = PROJECT_ROOT / 'src' / 'agents' / 'gap_and_go_bot.py'
LOG_CSV = PROJECT_ROOT / 'csvs' / 'scheduler_log.csv'
LOGS_DIR = PROJECT_ROOT / 'logs'
PID_FILE = PROJECT_ROOT / 'csvs' / '.gap_go_bot.pid'

# Smart Bot parameters
SMART_BOT_ARGS = [
    '--use-scanner',
    '--interval', '30',
    '--duration', '390'  # 6.5 hours = full market day
]

# Gap and Go Bot parameters (paper trading by default)
GAP_GO_BOT_ARGS = []  # No extra args = paper trading

# Schedule times (Eastern Time)
MARKET_OPEN = "09:30"
GAP_GO_START = "03:55"  # 5 minutes before pre-market monitoring


def get_eastern_time():
    """Get current time in Eastern timezone."""
    try:
        import pytz
        eastern = pytz.timezone('US/Eastern')
        return datetime.now(eastern)
    except ImportError:
        # Fallback: assume local time is close enough or adjust manually
        # This is a rough approximation
        cprint("Warning: pytz not installed, using local time", "yellow")
        return datetime.now()


def ensure_directories():
    """Ensure required directories exist."""
    LOG_CSV.parent.mkdir(parents=True, exist_ok=True)
    LOGS_DIR.mkdir(parents=True, exist_ok=True)


def log_event(event_type, details=""):
    """Log an event to the CSV file."""
    ensure_directories()

    file_exists = LOG_CSV.exists()

    with open(LOG_CSV, 'a', newline='') as f:
        writer = csv.writer(f)
        if not file_exists:
            writer.writerow(['timestamp', 'event', 'details'])

        timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        writer.writerow([timestamp, event_type, details])


def is_weekday():
    """Check if today is a weekday (Mon-Fri)."""
    return datetime.now().weekday() < 5


def is_bot_running(pid_file: Path) -> bool:
    """Check if a bot is already running via PID file."""
    if not pid_file.exists():
        return False

    try:
        with open(pid_file, 'r') as f:
            pid = int(f.read().strip())

        # Check if process is running
        os.kill(pid, 0)  # Signal 0 = check if alive
        return True
    except (ValueError, ProcessLookupError, PermissionError):
        # PID file invalid or process not running
        pid_file.unlink(missing_ok=True)
        return False


def write_pid_file(pid_file: Path, pid: int):
    """Write PID to file."""
    with open(pid_file, 'w') as f:
        f.write(str(pid))


def run_gap_go_bot():
    """Start the Gap and Go trading bot."""
    if not is_weekday():
        cprint("Weekend - skipping Gap and Go bot start", "yellow")
        log_event("GAP_GO_SKIPPED", "Weekend")
        return

    # Check if already running
    if is_bot_running(PID_FILE):
        cprint("Gap and Go bot already running - skipping", "yellow")
        log_event("GAP_GO_SKIPPED", "Already running")
        return

    cprint("\n" + "=" * 60, "green")
    cprint("  STARTING GAP AND GO BOT", "green", attrs=['bold'])
    cprint(f"  Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}", "green")
    cprint("=" * 60, "green")

    log_event("GAP_GO_START", f"Args: {' '.join(GAP_GO_BOT_ARGS) or 'default (paper)'}")

    try:
        # Build command
        cmd = [sys.executable, str(GAP_GO_BOT_SCRIPT)] + GAP_GO_BOT_ARGS

        cprint(f"\nRunning: {' '.join(cmd)}", "cyan")
        cprint(f"Mode: Paper Trading\n", "cyan")

        # Run the bot (blocking)
        process = subprocess.Popen(
            cmd,
            cwd=str(PROJECT_ROOT),
            stdout=sys.stdout,
            stderr=sys.stderr
        )

        # Write PID file
        write_pid_file(PID_FILE, process.pid)

        # Wait for completion
        exit_code = process.wait()

        # Clean up PID file
        PID_FILE.unlink(missing_ok=True)

        log_event("GAP_GO_STOP", f"Exit code: {exit_code}")

        cprint("\n" + "=" * 60, "yellow")
        cprint(f"  GAP AND GO BOT STOPPED (exit code: {exit_code})", "yellow")
        cprint(f"  Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}", "yellow")
        cprint("=" * 60 + "\n", "yellow")

    except Exception as e:
        error_msg = str(e)
        log_event("GAP_GO_ERROR", error_msg)
        cprint(f"Error running Gap and Go bot: {error_msg}", "red")
        PID_FILE.unlink(missing_ok=True)


def run_trading_bot():
    """Start the smart trading bot."""
    if not is_weekday():
        cprint("Weekend - skipping bot start", "yellow")
        log_event("SKIPPED", "Weekend")
        return

    cprint("\n" + "=" * 60, "green")
    cprint("  STARTING TRADING BOT", "green", attrs=['bold'])
    cprint(f"  Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}", "green")
    cprint("=" * 60, "green")

    log_event("START", f"Args: {' '.join(BOT_ARGS)}")

    try:
        # Build command
        cmd = [sys.executable, str(BOT_SCRIPT)] + BOT_ARGS

        cprint(f"\nRunning: {' '.join(cmd)}", "cyan")
        cprint(f"Duration: 390 minutes (6.5 hours)\n", "cyan")

        # Run the bot (blocking)
        process = subprocess.run(
            cmd,
            cwd=str(PROJECT_ROOT),
            stdout=sys.stdout,
            stderr=sys.stderr
        )

        exit_code = process.returncode
        log_event("STOP", f"Exit code: {exit_code}")

        cprint("\n" + "=" * 60, "yellow")
        cprint(f"  BOT STOPPED (exit code: {exit_code})", "yellow")
        cprint(f"  Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}", "yellow")
        cprint("=" * 60 + "\n", "yellow")

    except Exception as e:
        error_msg = str(e)
        log_event("ERROR", error_msg)
        cprint(f"Error running bot: {error_msg}", "red")


def print_banner():
    """Print startup banner."""
    cprint("\n" + "=" * 60, "cyan")
    cprint("  TRADING BOT AUTO-STARTER", "cyan", attrs=['bold'])
    cprint("=" * 60, "cyan")
    cprint(f"  Gap and Go Bot:  {GAP_GO_START} ET (pre-market)", "white")
    cprint(f"  Smart Bot:       {MARKET_OPEN} ET (market open)", "white")
    cprint(f"  Schedule:        Mon-Fri only", "white")
    cprint(f"  Log File:        {LOG_CSV}", "white")
    cprint("=" * 60 + "\n", "cyan")


def run_scheduler():
    """Run the scheduler loop."""
    print_banner()

    log_event("SCHEDULER_START", f"Gap Go @ {GAP_GO_START}, Smart @ {MARKET_OPEN} ET")

    # Schedule Gap and Go bot at 3:55 AM ET (Mon-Fri)
    schedule.every().monday.at(GAP_GO_START).do(run_gap_go_bot)
    schedule.every().tuesday.at(GAP_GO_START).do(run_gap_go_bot)
    schedule.every().wednesday.at(GAP_GO_START).do(run_gap_go_bot)
    schedule.every().thursday.at(GAP_GO_START).do(run_gap_go_bot)
    schedule.every().friday.at(GAP_GO_START).do(run_gap_go_bot)

    # Schedule Smart bot at 9:30 AM ET (Mon-Fri)
    schedule.every().monday.at(MARKET_OPEN).do(run_trading_bot)
    schedule.every().tuesday.at(MARKET_OPEN).do(run_trading_bot)
    schedule.every().wednesday.at(MARKET_OPEN).do(run_trading_bot)
    schedule.every().thursday.at(MARKET_OPEN).do(run_trading_bot)
    schedule.every().friday.at(MARKET_OPEN).do(run_trading_bot)

    cprint(f"Scheduler running.", "green")
    cprint(f"  Gap and Go: {GAP_GO_START} ET", "green")
    cprint(f"  Smart Bot:  {MARKET_OPEN} ET", "green")
    cprint("Press Ctrl+C to stop.\n", "yellow")

    try:
        while True:
            schedule.run_pending()
            time.sleep(60)  # Check every minute
    except KeyboardInterrupt:
        cprint("\nScheduler stopped by user", "yellow")
        log_event("SCHEDULER_STOP", "User interrupt")


def run_test():
    """Run immediately in test mode."""
    cprint("\n" + "=" * 60, "magenta")
    cprint("  TEST MODE - Running immediately", "magenta", attrs=['bold'])
    cprint("=" * 60 + "\n", "magenta")

    log_event("TEST_START", "Manual test run")
    run_trading_bot()


def run_gap_go_test():
    """Run Gap and Go bot immediately in test mode."""
    cprint("\n" + "=" * 60, "magenta")
    cprint("  TEST MODE - Running Gap and Go bot immediately", "magenta", attrs=['bold'])
    cprint("=" * 60 + "\n", "magenta")

    log_event("GAP_GO_TEST_START", "Manual test run")
    run_gap_go_bot()


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Auto-start trading bots at scheduled times"
    )
    parser.add_argument(
        '--test',
        action='store_true',
        help='Run Smart bot immediately (test mode)'
    )
    parser.add_argument(
        '--gap-go',
        action='store_true',
        help='Run Gap and Go bot immediately'
    )
    args = parser.parse_args()

    ensure_directories()

    if args.gap_go:
        run_gap_go_test()
    elif args.test:
        run_test()
    else:
        run_scheduler()


if __name__ == "__main__":
    main()
