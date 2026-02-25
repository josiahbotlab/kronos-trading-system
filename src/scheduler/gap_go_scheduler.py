"""
Gap and Go Dedicated Scheduler

Lightweight scheduler specifically for the Gap and Go trading bot.
Calculates next market open and starts the bot 5 minutes before
pre-market monitoring should begin.

Features:
- Calculates next market open
- Handles weekends and holidays (skips those days)
- Can run as a daemon/service
- Prevents duplicate bot instances

Usage:
    python3 src/scheduler/gap_go_scheduler.py             # Run scheduler
    python3 src/scheduler/gap_go_scheduler.py --daemon    # Run as daemon
    python3 src/scheduler/gap_go_scheduler.py --status    # Check status
    python3 src/scheduler/gap_go_scheduler.py --next      # Show next run time

Background:
    nohup python3 src/scheduler/gap_go_scheduler.py --daemon > logs/gap_go_scheduler.log 2>&1 &
"""

import argparse
import csv
import os
import signal
import subprocess
import sys
import time
from datetime import datetime, date, time as dt_time, timedelta
from pathlib import Path

import pytz
from termcolor import cprint

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.append(str(PROJECT_ROOT))

# Configuration
GAP_GO_BOT_SCRIPT = PROJECT_ROOT / 'src' / 'agents' / 'gap_and_go_bot.py'
LOG_CSV = PROJECT_ROOT / 'csvs' / 'scheduler_log.csv'
PID_FILE = PROJECT_ROOT / 'csvs' / '.gap_go_scheduler.pid'
BOT_PID_FILE = PROJECT_ROOT / 'csvs' / '.gap_go_bot.pid'
LOGS_DIR = PROJECT_ROOT / 'logs'

# Timezone
ET = pytz.timezone('America/New_York')

# Schedule times (Eastern Time)
PREMARKET_START = dt_time(4, 0)   # 4:00 AM ET - pre-market monitoring starts
BOT_START_TIME = dt_time(3, 55)   # 3:55 AM ET - start bot 5 min early

# US Market holidays (2024-2026) - approximate dates
MARKET_HOLIDAYS = [
    # 2024
    (2024, 1, 1), (2024, 1, 15), (2024, 2, 19), (2024, 3, 29),
    (2024, 5, 27), (2024, 6, 19), (2024, 7, 4), (2024, 9, 2),
    (2024, 11, 28), (2024, 12, 25),
    # 2025
    (2025, 1, 1), (2025, 1, 20), (2025, 2, 17), (2025, 4, 18),
    (2025, 5, 26), (2025, 6, 19), (2025, 7, 4), (2025, 9, 1),
    (2025, 11, 27), (2025, 12, 25),
    # 2026
    (2026, 1, 1), (2026, 1, 19), (2026, 2, 16), (2026, 4, 3),
    (2026, 5, 25), (2026, 6, 19), (2026, 7, 3), (2026, 9, 7),
    (2026, 11, 26), (2026, 12, 25),
]


def get_et_now() -> datetime:
    """Get current time in Eastern timezone."""
    return datetime.now(ET)


def is_weekend(dt: datetime) -> bool:
    """Check if date is weekend."""
    return dt.weekday() >= 5


def is_market_holiday(dt: datetime) -> bool:
    """Check if date is a US market holiday."""
    return (dt.year, dt.month, dt.day) in MARKET_HOLIDAYS


def is_trading_day(dt: datetime) -> bool:
    """Check if date is a valid trading day."""
    return not is_weekend(dt) and not is_market_holiday(dt)


def get_next_trading_day() -> datetime:
    """Get the next trading day (including today if before bot start time)."""
    now = get_et_now()

    # Create target time for today
    today_target = now.replace(
        hour=BOT_START_TIME.hour,
        minute=BOT_START_TIME.minute,
        second=0,
        microsecond=0
    )

    # If we're before the start time today (with 5 minute buffer after),
    # and today is a trading day, return today's target
    # The buffer handles the case where we're exactly at or just past the start time
    buffer_end = today_target + timedelta(minutes=5)

    if now < buffer_end and is_trading_day(now):
        return today_target

    # Otherwise, find the next trading day
    check_date = now + timedelta(days=1)
    check_date = check_date.replace(
        hour=BOT_START_TIME.hour,
        minute=BOT_START_TIME.minute,
        second=0,
        microsecond=0
    )

    # Skip weekends and holidays
    max_days = 10  # Safety limit
    for _ in range(max_days):
        if is_trading_day(check_date):
            return check_date
        check_date += timedelta(days=1)

    # Fallback (shouldn't happen)
    return check_date


def seconds_until_next_run() -> float:
    """Calculate seconds until the next bot start time."""
    now = get_et_now()
    next_run = get_next_trading_day()
    delta = (next_run - now).total_seconds()
    return max(0, delta)


def ensure_directories():
    """Ensure required directories exist."""
    LOG_CSV.parent.mkdir(parents=True, exist_ok=True)
    LOGS_DIR.mkdir(parents=True, exist_ok=True)


def log_event(event_type: str, details: str = ""):
    """Log an event to the CSV file."""
    ensure_directories()

    file_exists = LOG_CSV.exists()

    with open(LOG_CSV, 'a', newline='') as f:
        writer = csv.writer(f)
        if not file_exists:
            writer.writerow(['timestamp', 'event', 'details'])

        timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        writer.writerow([timestamp, event_type, details])


def is_process_running(pid_file: Path) -> bool:
    """Check if a process is running via PID file."""
    if not pid_file.exists():
        return False

    try:
        with open(pid_file, 'r') as f:
            pid = int(f.read().strip())

        os.kill(pid, 0)  # Signal 0 = check if alive
        return True
    except (ValueError, ProcessLookupError, PermissionError):
        pid_file.unlink(missing_ok=True)
        return False


def write_pid_file(pid_file: Path, pid: int):
    """Write PID to file."""
    with open(pid_file, 'w') as f:
        f.write(str(pid))


def cleanup_pid_file(pid_file: Path):
    """Remove PID file."""
    pid_file.unlink(missing_ok=True)


def start_gap_go_bot() -> bool:
    """
    Start the Gap and Go trading bot.

    Returns:
        True if bot started successfully
    """
    # Check if bot is already running
    if is_process_running(BOT_PID_FILE):
        cprint("[SCHEDULER] Gap and Go bot already running - skipping", "yellow")
        log_event("GAP_GO_SKIP", "Bot already running")
        return False

    cprint("\n" + "=" * 60, "green")
    cprint("  STARTING GAP AND GO BOT", "green", attrs=['bold'])
    cprint(f"  Time: {get_et_now().strftime('%Y-%m-%d %H:%M:%S %Z')}", "green")
    cprint("=" * 60, "green")

    log_event("GAP_GO_START", "Scheduler initiated start")

    try:
        cmd = [sys.executable, str(GAP_GO_BOT_SCRIPT)]

        cprint(f"\n[SCHEDULER] Running: {' '.join(cmd)}", "cyan")

        # Start bot as subprocess
        process = subprocess.Popen(
            cmd,
            cwd=str(PROJECT_ROOT),
            stdout=sys.stdout,
            stderr=sys.stderr
        )

        # Write bot PID file
        write_pid_file(BOT_PID_FILE, process.pid)

        cprint(f"[SCHEDULER] Bot started with PID {process.pid}", "green")

        # Wait for completion
        exit_code = process.wait()

        # Cleanup
        cleanup_pid_file(BOT_PID_FILE)

        log_event("GAP_GO_STOP", f"Exit code: {exit_code}")

        cprint("\n" + "=" * 60, "yellow")
        cprint(f"  GAP AND GO BOT FINISHED (exit code: {exit_code})", "yellow")
        cprint(f"  Time: {get_et_now().strftime('%Y-%m-%d %H:%M:%S %Z')}", "yellow")
        cprint("=" * 60 + "\n", "yellow")

        return exit_code == 0

    except Exception as e:
        log_event("GAP_GO_ERROR", str(e))
        cprint(f"[SCHEDULER] Error starting bot: {e}", "red")
        cleanup_pid_file(BOT_PID_FILE)
        return False


def run_scheduler_loop():
    """Main scheduler loop."""
    cprint("\n" + "=" * 60, "cyan")
    cprint("  GAP AND GO SCHEDULER", "cyan", attrs=['bold'])
    cprint("=" * 60, "cyan")
    cprint(f"  Bot Start Time: {BOT_START_TIME.strftime('%H:%M')} ET", "white")
    cprint(f"  Current Time:   {get_et_now().strftime('%H:%M:%S %Z')}", "white")
    cprint(f"  Log File:       {LOG_CSV}", "white")
    cprint("=" * 60 + "\n", "cyan")

    log_event("SCHEDULER_START", f"Gap Go Scheduler @ {BOT_START_TIME.strftime('%H:%M')} ET")

    running = True
    last_hourly_log = None

    def handle_shutdown(signum, frame):
        nonlocal running
        cprint("\n[SCHEDULER] Shutdown signal received", "yellow")
        log_event("SCHEDULER_STOP", "Signal received")
        running = False

    signal.signal(signal.SIGINT, handle_shutdown)
    signal.signal(signal.SIGTERM, handle_shutdown)

    while running:
        now = get_et_now()
        next_run = get_next_trading_day()
        wait_seconds = (next_run - now).total_seconds()

        cprint(f"[SCHEDULER] Next run: {next_run.strftime('%Y-%m-%d %H:%M:%S %Z')}", "cyan")
        cprint(f"[SCHEDULER] Waiting {wait_seconds / 3600:.1f} hours...", "cyan")
        log_event("SCHEDULER_WAITING", f"Next: {next_run.strftime('%Y-%m-%d %H:%M')} ET, Wait: {wait_seconds / 3600:.1f}h")

        # Wait until next run time, checking every minute
        while wait_seconds > 60 and running:
            time.sleep(60)
            now = get_et_now()
            wait_seconds = (next_run - now).total_seconds()

            # Hourly status log
            current_hour = now.hour
            if last_hourly_log != current_hour:
                last_hourly_log = current_hour
                hours_left = wait_seconds / 3600
                cprint(f"[SCHEDULER] {now.strftime('%H:%M')} ET - {hours_left:.1f} hours until bot start", "white")
                log_event("SCHEDULER_HEARTBEAT", f"{now.strftime('%H:%M')} ET, {hours_left:.1f}h remaining")

        if not running:
            break

        # Final countdown - sleep remaining seconds
        if wait_seconds > 0:
            cprint(f"[SCHEDULER] Final wait: {wait_seconds:.0f} seconds...", "yellow")
            time.sleep(wait_seconds)

        # === TRIGGER TIME ===
        now = get_et_now()
        cprint(f"\n[SCHEDULER] === TRIGGER TIME: {now.strftime('%H:%M:%S %Z')} ===", "green")
        log_event("SCHEDULER_TRIGGER", f"Trigger at {now.strftime('%H:%M:%S %Z')}")

        if is_trading_day(now):
            cprint(f"[SCHEDULER] Starting Gap and Go bot...", "green")
            start_gap_go_bot()
        else:
            cprint(f"[SCHEDULER] {now.strftime('%Y-%m-%d')} is not a trading day - skipping", "yellow")
            log_event("SCHEDULER_SKIP", f"{now.strftime('%Y-%m-%d')} not a trading day")

        # After bot finishes, wait a bit before calculating next run
        # This prevents immediately re-triggering
        cprint(f"[SCHEDULER] Bot session complete. Waiting 1 minute before next cycle...", "cyan")
        time.sleep(60)

    cprint("\n[SCHEDULER] Scheduler stopped", "yellow")
    log_event("SCHEDULER_STOP", "Normal shutdown")


def is_running_under_service_manager() -> bool:
    """
    Check if we're running under a service manager (launchd, systemd, etc).

    In these cases, we should NOT fork - the service manager handles process lifecycle.
    """
    # Check for launchd (macOS) - stdin won't be a tty when run by launchd
    # Also check for common service manager environment variables
    if not sys.stdin.isatty():
        return True
    if os.environ.get('LAUNCHED_BY_LAUNCHD'):
        return True
    if os.environ.get('INVOCATION_ID'):  # systemd sets this
        return True
    return False


def run_as_daemon():
    """
    Run scheduler as a daemon process.

    When run under a service manager (launchd, systemd), runs in foreground.
    When run manually, forks to background.
    """
    # Check if running under service manager
    under_service_manager = is_running_under_service_manager()

    if under_service_manager:
        # Running under launchd/systemd - don't fork, run directly
        cprint("[DAEMON] Running under service manager - no fork needed", "cyan")

        # Write PID file
        write_pid_file(PID_FILE, os.getpid())
        log_event("DAEMON_START", f"PID: {os.getpid()} (service managed)")

        try:
            run_scheduler_loop()
        finally:
            cleanup_pid_file(PID_FILE)
        return

    # Manual run - check if already running
    if is_process_running(PID_FILE):
        cprint("[DAEMON] Scheduler already running", "yellow")
        try:
            with open(PID_FILE, 'r') as f:
                existing_pid = f.read().strip()
            cprint(f"[DAEMON] Existing PID: {existing_pid}", "white")
        except:
            pass
        return

    cprint("[DAEMON] Starting Gap and Go scheduler as daemon...", "cyan")

    # Fork process (Unix only) - for manual background execution
    try:
        pid = os.fork()
        if pid > 0:
            # Parent process
            cprint(f"[DAEMON] Scheduler started with PID {pid}", "green")
            cprint(f"[DAEMON] Log: {LOGS_DIR / 'gap_go_scheduler.log'}", "white")
            return
    except AttributeError:
        # Windows doesn't have fork
        cprint("[DAEMON] Daemon mode not supported on Windows", "red")
        cprint("[DAEMON] Use: start /B python gap_go_scheduler.py", "yellow")
        return

    # Child process continues
    os.setsid()

    # Write PID file
    write_pid_file(PID_FILE, os.getpid())
    log_event("DAEMON_START", f"PID: {os.getpid()}")

    try:
        run_scheduler_loop()
    finally:
        cleanup_pid_file(PID_FILE)


def show_status():
    """Show current scheduler status."""
    cprint("\n" + "=" * 60, "cyan")
    cprint("  GAP AND GO SCHEDULER STATUS", "cyan", attrs=['bold'])
    cprint("=" * 60, "cyan")

    # Scheduler status
    scheduler_running = is_process_running(PID_FILE)
    if scheduler_running:
        with open(PID_FILE, 'r') as f:
            pid = f.read().strip()
        cprint(f"  Scheduler: RUNNING (PID {pid})", "green")
    else:
        cprint("  Scheduler: NOT RUNNING", "yellow")

    # Bot status
    bot_running = is_process_running(BOT_PID_FILE)
    if bot_running:
        with open(BOT_PID_FILE, 'r') as f:
            pid = f.read().strip()
        cprint(f"  Bot:       RUNNING (PID {pid})", "green")
    else:
        cprint("  Bot:       NOT RUNNING", "white")

    # Next run info
    now = get_et_now()
    next_run = get_next_trading_day()
    wait_seconds = seconds_until_next_run()

    cprint(f"\n  Current Time: {now.strftime('%Y-%m-%d %H:%M:%S %Z')}", "white")
    cprint(f"  Next Run:     {next_run.strftime('%Y-%m-%d %H:%M:%S %Z')}", "white")
    cprint(f"  Wait Time:    {wait_seconds / 3600:.1f} hours", "white")

    # Today's status
    if is_weekend(now):
        cprint(f"\n  Today: WEEKEND (no trading)", "yellow")
    elif is_market_holiday(now):
        cprint(f"\n  Today: HOLIDAY (no trading)", "yellow")
    else:
        if now.time() < BOT_START_TIME:
            cprint(f"\n  Today: Trading day - bot starts at {BOT_START_TIME.strftime('%H:%M')} ET", "green")
        elif now.time() < dt_time(11, 0):
            cprint(f"\n  Today: Trading day - session in progress", "green")
        else:
            cprint(f"\n  Today: Trading day - session ended", "yellow")

    cprint("=" * 60 + "\n", "cyan")


def show_next_run():
    """Show information about the next scheduled run."""
    now = get_et_now()
    next_run = get_next_trading_day()
    wait_seconds = seconds_until_next_run()

    cprint(f"\nNext Gap and Go Bot Run:", "cyan")
    cprint(f"  Date:     {next_run.strftime('%A, %B %d, %Y')}", "white")
    cprint(f"  Time:     {next_run.strftime('%H:%M:%S %Z')}", "white")
    cprint(f"  In:       {wait_seconds / 3600:.1f} hours ({wait_seconds / 60:.0f} minutes)", "white")

    # Show schedule for next 5 trading days
    cprint(f"\nUpcoming Schedule:", "cyan")
    check_date = now
    count = 0
    while count < 5:
        check_date += timedelta(days=1)
        if is_trading_day(check_date):
            run_time = check_date.replace(
                hour=BOT_START_TIME.hour,
                minute=BOT_START_TIME.minute,
                second=0
            )
            cprint(f"  {run_time.strftime('%a %m/%d')}: {run_time.strftime('%H:%M')} ET", "white")
            count += 1


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Gap and Go Dedicated Scheduler"
    )
    parser.add_argument(
        '--daemon',
        action='store_true',
        help='Run as background daemon'
    )
    parser.add_argument(
        '--status',
        action='store_true',
        help='Show scheduler status'
    )
    parser.add_argument(
        '--next',
        action='store_true',
        help='Show next run time'
    )
    parser.add_argument(
        '--now',
        action='store_true',
        help='Start bot immediately (bypass schedule)'
    )

    args = parser.parse_args()

    ensure_directories()

    if args.status:
        show_status()
    elif args.next:
        show_next_run()
    elif args.now:
        cprint("\n[SCHEDULER] Starting bot immediately...", "cyan")
        start_gap_go_bot()
    elif args.daemon:
        run_as_daemon()
    else:
        run_scheduler_loop()


if __name__ == "__main__":
    main()
