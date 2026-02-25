"""
Gap and Go Monitoring System

Comprehensive monitoring, alerting, and performance validation for the Gap and Go trading bot.

Features:
- Performance tracking with rolling statistics
- Daily and weekly reports
- Alert conditions with configurable thresholds
- Backtest validation
- Dashboard integration
- Health checks

Usage:
    python3 src/monitoring/gap_go_monitor.py --status      # Current status + health check
    python3 src/monitoring/gap_go_monitor.py --daily       # Generate daily report
    python3 src/monitoring/gap_go_monitor.py --weekly      # Generate weekly report
    python3 src/monitoring/gap_go_monitor.py --alerts      # Show active alerts
    python3 src/monitoring/gap_go_monitor.py --validate    # Run backtest validation
    python3 src/monitoring/gap_go_monitor.py --performance # Show rolling performance stats
"""

import argparse
import csv
import json
import os
import sys
from datetime import datetime, date, timedelta, time as dt_time
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from enum import Enum

import pytz
from termcolor import cprint
from dotenv import load_dotenv

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.append(str(PROJECT_ROOT))
load_dotenv()

# ============================================================================
# CONFIGURATION
# ============================================================================

# Paths
TRADE_JOURNAL_PATH = PROJECT_ROOT / 'csvs' / 'trade_journal.csv'
REPORTS_DIR = PROJECT_ROOT / 'csvs' / 'gap_go_reports'
ALERTS_DIR = PROJECT_ROOT / 'csvs' / 'alerts'
DASHBOARD_DATA_FILE = PROJECT_ROOT / 'csvs' / 'gap_go_dashboard_data.json'
STATE_FILE = PROJECT_ROOT / 'csvs' / 'gap_go_state.json'

# Ensure directories exist
REPORTS_DIR.mkdir(parents=True, exist_ok=True)
ALERTS_DIR.mkdir(parents=True, exist_ok=True)

# Timezone
ET = pytz.timezone('America/New_York')

# Backtest expectations (from optimization results)
BACKTEST_EXPECTATIONS = {
    'AMD': {
        'win_rate': 100.0,  # 100% win rate from backtest
        'avg_trade_pct': 11.63,  # +11.63% average trade
        'num_trades': 8,
        'expected_return': 140.79,
    },
    'META': {
        'win_rate': 80.0,  # 80% win rate from backtest
        'avg_trade_pct': 3.12,  # ~3% average trade
        'num_trades': 5,
        'expected_return': 15.61,
    }
}

# Alert thresholds
ALERT_THRESHOLDS = {
    'min_win_rate_10_trades': 50.0,  # Alert if win rate < 50% over last 10 trades
    'max_loss_to_win_ratio': 2.0,    # Alert if avg loss > 2x avg win
    'max_days_no_trades': 5,         # Alert if no trades for 5 trading days
    'max_consecutive_losses': 3,     # Alert after 3 consecutive losses
    'min_backtest_performance': 0.4, # Alert if live < 40% of backtest
}

# ============================================================================
# ENUMS AND TYPES
# ============================================================================

class AlertSeverity(Enum):
    INFO = "INFO"
    WARNING = "WARNING"
    CRITICAL = "CRITICAL"


class HealthStatus(Enum):
    OK = "OK"
    WARN = "WARN"
    ERROR = "ERROR"


# ============================================================================
# TRADE DATA LOADING
# ============================================================================

def load_trade_journal() -> List[Dict]:
    """Load all trades from the trade journal."""
    if not TRADE_JOURNAL_PATH.exists():
        return []

    trades = []
    with open(TRADE_JOURNAL_PATH, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            trades.append(row)

    return trades


def get_gap_go_trades(trades: List[Dict] = None) -> List[Dict]:
    """Filter trades to only Gap and Go strategy."""
    if trades is None:
        trades = load_trade_journal()

    return [t for t in trades if t.get('strategy') == 'GAP_GO']


def get_exit_trades(trades: List[Dict]) -> List[Dict]:
    """Get only exit trades (with P&L data)."""
    return [t for t in trades if t.get('action') == 'EXIT' and t.get('pnl')]


def parse_trade_pnl(trade: Dict) -> Optional[float]:
    """Parse P&L from trade record."""
    try:
        return float(trade.get('pnl', 0))
    except (ValueError, TypeError):
        return None


def parse_trade_date(trade: Dict) -> Optional[date]:
    """Parse date from trade timestamp."""
    try:
        timestamp = trade.get('timestamp', '')
        return datetime.strptime(timestamp[:10], '%Y-%m-%d').date()
    except (ValueError, TypeError):
        return None


# ============================================================================
# PERFORMANCE TRACKING
# ============================================================================

class PerformanceTracker:
    """Track and calculate rolling performance statistics."""

    def __init__(self):
        self.trades = get_gap_go_trades()
        self.exit_trades = get_exit_trades(self.trades)

    def refresh(self):
        """Refresh trade data."""
        self.trades = get_gap_go_trades()
        self.exit_trades = get_exit_trades(self.trades)

    def get_rolling_stats(self, n_trades: int = 10) -> Dict:
        """Calculate statistics over last N trades."""
        recent_exits = self.exit_trades[-n_trades:] if len(self.exit_trades) >= n_trades else self.exit_trades

        if not recent_exits:
            return {
                'n_trades': 0,
                'win_rate': None,
                'avg_pnl': None,
                'total_pnl': 0,
                'avg_win': None,
                'avg_loss': None,
                'winners': 0,
                'losers': 0,
            }

        pnls = [parse_trade_pnl(t) for t in recent_exits if parse_trade_pnl(t) is not None]
        winners = [p for p in pnls if p > 0]
        losers = [p for p in pnls if p <= 0]

        return {
            'n_trades': len(pnls),
            'win_rate': (len(winners) / len(pnls) * 100) if pnls else None,
            'avg_pnl': sum(pnls) / len(pnls) if pnls else None,
            'total_pnl': sum(pnls),
            'avg_win': sum(winners) / len(winners) if winners else None,
            'avg_loss': sum(losers) / len(losers) if losers else None,
            'winners': len(winners),
            'losers': len(losers),
        }

    def get_all_time_stats(self) -> Dict:
        """Get all-time statistics."""
        return self.get_rolling_stats(len(self.exit_trades))

    def get_today_stats(self) -> Dict:
        """Get today's statistics."""
        today = date.today()
        today_trades = [t for t in self.exit_trades if parse_trade_date(t) == today]

        if not today_trades:
            return {
                'n_trades': 0,
                'win_rate': None,
                'total_pnl': 0,
                'winners': 0,
                'losers': 0,
            }

        pnls = [parse_trade_pnl(t) for t in today_trades if parse_trade_pnl(t) is not None]
        winners = [p for p in pnls if p > 0]
        losers = [p for p in pnls if p <= 0]

        return {
            'n_trades': len(pnls),
            'win_rate': (len(winners) / len(pnls) * 100) if pnls else None,
            'total_pnl': sum(pnls),
            'winners': len(winners),
            'losers': len(losers),
        }

    def get_weekly_stats(self) -> Dict:
        """Get this week's statistics (Mon-Sun)."""
        today = date.today()
        week_start = today - timedelta(days=today.weekday())

        week_trades = [
            t for t in self.exit_trades
            if parse_trade_date(t) and parse_trade_date(t) >= week_start
        ]

        if not week_trades:
            return {
                'n_trades': 0,
                'win_rate': None,
                'total_pnl': 0,
                'winners': 0,
                'losers': 0,
            }

        pnls = [parse_trade_pnl(t) for t in week_trades if parse_trade_pnl(t) is not None]
        winners = [p for p in pnls if p > 0]
        losers = [p for p in pnls if p <= 0]

        return {
            'n_trades': len(pnls),
            'win_rate': (len(winners) / len(pnls) * 100) if pnls else None,
            'total_pnl': sum(pnls),
            'winners': len(winners),
            'losers': len(losers),
        }

    def compare_to_backtest(self) -> Dict:
        """Compare live results to backtest expectations."""
        all_stats = self.get_all_time_stats()

        comparison = {
            'live_win_rate': all_stats['win_rate'],
            'live_avg_pnl': all_stats['avg_pnl'],
            'live_total_trades': all_stats['n_trades'],
        }

        # Calculate weighted expected performance
        expected_win_rate = (
            BACKTEST_EXPECTATIONS['AMD']['win_rate'] * 0.5 +
            BACKTEST_EXPECTATIONS['META']['win_rate'] * 0.5
        )
        expected_avg_trade = (
            BACKTEST_EXPECTATIONS['AMD']['avg_trade_pct'] * 0.5 +
            BACKTEST_EXPECTATIONS['META']['avg_trade_pct'] * 0.5
        )

        comparison['expected_win_rate'] = expected_win_rate
        comparison['expected_avg_trade_pct'] = expected_avg_trade

        if all_stats['win_rate'] is not None:
            comparison['win_rate_vs_expected'] = all_stats['win_rate'] / expected_win_rate

        return comparison

    def get_consecutive_losses(self) -> int:
        """Count current consecutive losing streak."""
        if not self.exit_trades:
            return 0

        count = 0
        for trade in reversed(self.exit_trades):
            pnl = parse_trade_pnl(trade)
            if pnl is not None and pnl <= 0:
                count += 1
            else:
                break

        return count

    def get_last_trade_date(self) -> Optional[date]:
        """Get date of last trade."""
        if not self.exit_trades:
            return None

        return parse_trade_date(self.exit_trades[-1])

    def get_days_since_last_trade(self) -> Optional[int]:
        """Get number of trading days since last trade."""
        last_date = self.get_last_trade_date()
        if not last_date:
            return None

        today = date.today()
        # Count weekdays between last trade and today
        days = 0
        current = last_date + timedelta(days=1)
        while current <= today:
            if current.weekday() < 5:  # Monday = 0, Friday = 4
                days += 1
            current += timedelta(days=1)

        return days


# ============================================================================
# ALERT SYSTEM
# ============================================================================

class AlertManager:
    """Manage alerts and alert history."""

    ALERTS_LOG = ALERTS_DIR / 'gap_go_alerts.csv'
    ACTIVE_ALERTS_FILE = ALERTS_DIR / 'active_alerts.json'

    def __init__(self):
        self.active_alerts: List[Dict] = []
        self._load_active_alerts()

    def _load_active_alerts(self):
        """Load active alerts from file."""
        if self.ACTIVE_ALERTS_FILE.exists():
            try:
                with open(self.ACTIVE_ALERTS_FILE, 'r') as f:
                    self.active_alerts = json.load(f)
            except:
                self.active_alerts = []

    def _save_active_alerts(self):
        """Save active alerts to file."""
        with open(self.ACTIVE_ALERTS_FILE, 'w') as f:
            json.dump(self.active_alerts, f, indent=2)

    def log_alert(self, severity: AlertSeverity, code: str, message: str, details: Dict = None):
        """Log an alert to CSV and add to active alerts."""
        timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')

        # Log to CSV
        file_exists = self.ALERTS_LOG.exists()
        with open(self.ALERTS_LOG, 'a', newline='') as f:
            writer = csv.writer(f)
            if not file_exists:
                writer.writerow(['timestamp', 'severity', 'code', 'message', 'details'])
            writer.writerow([
                timestamp,
                severity.value,
                code,
                message,
                json.dumps(details) if details else ''
            ])

        # Add to active alerts
        alert = {
            'timestamp': timestamp,
            'severity': severity.value,
            'code': code,
            'message': message,
            'details': details or {},
        }
        self.active_alerts.append(alert)
        self._save_active_alerts()

        # Print alert
        color = 'yellow' if severity == AlertSeverity.WARNING else 'red' if severity == AlertSeverity.CRITICAL else 'cyan'
        attrs = ['bold'] if severity == AlertSeverity.CRITICAL else []
        cprint(f"\n[ALERT] [{severity.value}] {code}: {message}", color, attrs=attrs)

    def clear_alerts(self, code: str = None):
        """Clear active alerts, optionally filtered by code."""
        if code:
            self.active_alerts = [a for a in self.active_alerts if a['code'] != code]
        else:
            self.active_alerts = []
        self._save_active_alerts()

    def get_active_alerts(self) -> List[Dict]:
        """Get list of active alerts."""
        return self.active_alerts

    def check_alerts(self, tracker: PerformanceTracker) -> List[Dict]:
        """Check all alert conditions and generate alerts."""
        alerts_generated = []

        # Clear stale alerts before checking
        self.active_alerts = []

        # 1. Win rate below threshold (last 10 trades)
        stats_10 = tracker.get_rolling_stats(10)
        if stats_10['n_trades'] >= 10 and stats_10['win_rate'] is not None:
            if stats_10['win_rate'] < ALERT_THRESHOLDS['min_win_rate_10_trades']:
                self.log_alert(
                    AlertSeverity.WARNING,
                    'LOW_WIN_RATE',
                    f"Win rate {stats_10['win_rate']:.1f}% over last 10 trades (threshold: {ALERT_THRESHOLDS['min_win_rate_10_trades']}%)",
                    {'win_rate': stats_10['win_rate'], 'trades': 10}
                )
                alerts_generated.append('LOW_WIN_RATE')

        # 2. Average loss exceeds threshold
        if stats_10['avg_win'] and stats_10['avg_loss']:
            loss_to_win_ratio = abs(stats_10['avg_loss']) / stats_10['avg_win']
            if loss_to_win_ratio > ALERT_THRESHOLDS['max_loss_to_win_ratio']:
                self.log_alert(
                    AlertSeverity.WARNING,
                    'HIGH_LOSS_RATIO',
                    f"Avg loss ({abs(stats_10['avg_loss']):.2f}) is {loss_to_win_ratio:.1f}x avg win ({stats_10['avg_win']:.2f})",
                    {'ratio': loss_to_win_ratio}
                )
                alerts_generated.append('HIGH_LOSS_RATIO')

        # 3. No trades for extended period
        days_since = tracker.get_days_since_last_trade()
        if days_since is not None and days_since >= ALERT_THRESHOLDS['max_days_no_trades']:
            self.log_alert(
                AlertSeverity.WARNING,
                'NO_TRADES',
                f"No trades for {days_since} trading days (threshold: {ALERT_THRESHOLDS['max_days_no_trades']})",
                {'days': days_since}
            )
            alerts_generated.append('NO_TRADES')

        # 4. Consecutive losses
        consecutive_losses = tracker.get_consecutive_losses()
        if consecutive_losses >= ALERT_THRESHOLDS['max_consecutive_losses']:
            self.log_alert(
                AlertSeverity.WARNING,
                'CONSECUTIVE_LOSSES',
                f"{consecutive_losses} consecutive losing trades (threshold: {ALERT_THRESHOLDS['max_consecutive_losses']})",
                {'count': consecutive_losses}
            )
            alerts_generated.append('CONSECUTIVE_LOSSES')

        # 5. Performance vs backtest
        comparison = tracker.compare_to_backtest()
        if comparison.get('win_rate_vs_expected') is not None:
            if comparison['win_rate_vs_expected'] < ALERT_THRESHOLDS['min_backtest_performance']:
                self.log_alert(
                    AlertSeverity.CRITICAL,
                    'BELOW_BACKTEST',
                    f"Live win rate ({comparison['live_win_rate']:.1f}%) is {comparison['win_rate_vs_expected']*100:.0f}% of backtest expectation",
                    {'live': comparison['live_win_rate'], 'expected': comparison['expected_win_rate']}
                )
                alerts_generated.append('BELOW_BACKTEST')

        self._save_active_alerts()
        return alerts_generated


# ============================================================================
# HEALTH CHECK
# ============================================================================

def health_check() -> Tuple[HealthStatus, Dict]:
    """
    Perform comprehensive health check.

    Verifies:
    - Alpaca API connection
    - Account buying power
    - No unexpected positions
    - Scanner returning valid data
    - Trade journal accessible

    Returns:
        (status, details)
    """
    checks = {}
    overall_status = HealthStatus.OK

    # 1. Alpaca API connection
    try:
        from src.utils.order_utils import get_account_info, get_all_positions
        account = get_account_info()
        if account:
            checks['alpaca_api'] = {
                'status': 'OK',
                'equity': account.get('equity'),
                'buying_power': account.get('buying_power'),
            }
        else:
            checks['alpaca_api'] = {'status': 'ERROR', 'message': 'No account data'}
            overall_status = HealthStatus.ERROR
    except Exception as e:
        checks['alpaca_api'] = {'status': 'ERROR', 'message': str(e)}
        overall_status = HealthStatus.ERROR

    # 2. Buying power check
    if checks.get('alpaca_api', {}).get('status') == 'OK':
        buying_power = checks['alpaca_api'].get('buying_power', 0)
        if buying_power >= 1000:
            checks['buying_power'] = {'status': 'OK', 'amount': buying_power}
        elif buying_power >= 500:
            checks['buying_power'] = {'status': 'WARN', 'amount': buying_power, 'message': 'Low buying power'}
            if overall_status == HealthStatus.OK:
                overall_status = HealthStatus.WARN
        else:
            checks['buying_power'] = {'status': 'ERROR', 'amount': buying_power, 'message': 'Insufficient buying power'}
            overall_status = HealthStatus.ERROR

    # 3. Position check
    try:
        positions = get_all_positions()
        gap_go_symbols = ['AMD', 'META']
        unexpected = [p for p in positions if p['symbol'] not in gap_go_symbols]

        if not unexpected:
            checks['positions'] = {'status': 'OK', 'count': len(positions)}
        else:
            checks['positions'] = {
                'status': 'WARN',
                'message': f'Unexpected positions: {[p["symbol"] for p in unexpected]}',
                'count': len(positions)
            }
            if overall_status == HealthStatus.OK:
                overall_status = HealthStatus.WARN
    except Exception as e:
        checks['positions'] = {'status': 'ERROR', 'message': str(e)}
        overall_status = HealthStatus.ERROR

    # 4. Scanner check
    try:
        from src.scanner.gap_scanner import get_current_gaps, get_tradeable_symbols
        symbols = get_tradeable_symbols()
        gaps = get_current_gaps()

        if gaps and len(gaps) > 0:
            checks['scanner'] = {'status': 'OK', 'symbols': list(gaps.keys())}
        else:
            checks['scanner'] = {'status': 'WARN', 'message': 'No gap data returned'}
            if overall_status == HealthStatus.OK:
                overall_status = HealthStatus.WARN
    except Exception as e:
        checks['scanner'] = {'status': 'ERROR', 'message': str(e)}
        overall_status = HealthStatus.ERROR

    # 5. Trade journal check
    if TRADE_JOURNAL_PATH.exists():
        checks['trade_journal'] = {
            'status': 'OK',
            'path': str(TRADE_JOURNAL_PATH),
            'exists': True
        }
    else:
        checks['trade_journal'] = {
            'status': 'WARN',
            'message': 'Trade journal not found (may be first run)',
            'path': str(TRADE_JOURNAL_PATH)
        }
        if overall_status == HealthStatus.OK:
            overall_status = HealthStatus.WARN

    # 6. State file check
    if STATE_FILE.exists():
        try:
            with open(STATE_FILE, 'r') as f:
                state = json.load(f)
            checks['state_file'] = {
                'status': 'OK',
                'state': state.get('state'),
                'date': state.get('date')
            }
        except:
            checks['state_file'] = {'status': 'WARN', 'message': 'State file unreadable'}
    else:
        checks['state_file'] = {'status': 'OK', 'message': 'No state file (first run)'}

    return overall_status, checks


# ============================================================================
# REPORT GENERATION
# ============================================================================

def generate_daily_report(save: bool = True) -> str:
    """Generate daily performance report."""
    today = date.today()
    report_lines = []

    report_lines.append("=" * 60)
    report_lines.append(f"  GAP AND GO DAILY REPORT - {today.strftime('%Y-%m-%d')}")
    report_lines.append("=" * 60)
    report_lines.append("")

    tracker = PerformanceTracker()

    # Today's stats
    today_stats = tracker.get_today_stats()
    report_lines.append("TODAY'S PERFORMANCE:")
    report_lines.append(f"  Trades Executed: {today_stats['n_trades']}")
    report_lines.append(f"  Winners: {today_stats['winners']}")
    report_lines.append(f"  Losers: {today_stats['losers']}")
    if today_stats['win_rate'] is not None:
        report_lines.append(f"  Win Rate: {today_stats['win_rate']:.1f}%")
    report_lines.append(f"  Total P&L: ${today_stats['total_pnl']:+,.2f}")
    report_lines.append("")

    # Regime info
    if STATE_FILE.exists():
        try:
            with open(STATE_FILE, 'r') as f:
                state = json.load(f)
            report_lines.append("REGIME:")
            # Try to get regime from daily summary
            summary_file = PROJECT_ROOT / 'csvs' / 'gap_go_daily' / f"summary_{today.strftime('%Y%m%d')}.csv"
            if summary_file.exists():
                with open(summary_file, 'r') as f:
                    reader = csv.DictReader(f)
                    for row in reader:
                        report_lines.append(f"  Detected: {row.get('regime', 'N/A')}")
                        report_lines.append(f"  Trades Skipped: {row.get('trades_skipped_regime', 0)}")
            report_lines.append("")
        except:
            pass

    # Rolling stats
    report_lines.append("ROLLING PERFORMANCE:")
    for n in [10, 20, 50]:
        stats = tracker.get_rolling_stats(n)
        if stats['n_trades'] >= n:
            report_lines.append(f"  Last {n} trades: {stats['win_rate']:.1f}% win rate, ${stats['avg_pnl']:.2f} avg")

    report_lines.append("")

    # Active alerts
    alert_mgr = AlertManager()
    alerts = alert_mgr.get_active_alerts()
    if alerts:
        report_lines.append("ACTIVE ALERTS:")
        for alert in alerts:
            report_lines.append(f"  [{alert['severity']}] {alert['code']}: {alert['message']}")
    else:
        report_lines.append("ACTIVE ALERTS: None")

    report_lines.append("")
    report_lines.append("=" * 60)
    report_lines.append(f"  Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    report_lines.append("=" * 60)

    report = "\n".join(report_lines)

    if save:
        report_file = REPORTS_DIR / f"daily_{today.strftime('%Y%m%d')}.txt"
        with open(report_file, 'w') as f:
            f.write(report)
        cprint(f"[REPORT] Daily report saved to: {report_file}", "green")

    return report


def generate_weekly_report(save: bool = True) -> str:
    """Generate weekly performance report."""
    today = date.today()
    week_start = today - timedelta(days=today.weekday())
    week_end = week_start + timedelta(days=6)

    report_lines = []

    report_lines.append("=" * 60)
    report_lines.append(f"  GAP AND GO WEEKLY REPORT")
    report_lines.append(f"  Week: {week_start.strftime('%Y-%m-%d')} to {week_end.strftime('%Y-%m-%d')}")
    report_lines.append("=" * 60)
    report_lines.append("")

    tracker = PerformanceTracker()

    # Weekly stats
    week_stats = tracker.get_weekly_stats()
    report_lines.append("THIS WEEK'S PERFORMANCE:")
    report_lines.append(f"  Trades Executed: {week_stats['n_trades']}")
    report_lines.append(f"  Winners: {week_stats['winners']}")
    report_lines.append(f"  Losers: {week_stats['losers']}")
    if week_stats['win_rate'] is not None:
        report_lines.append(f"  Win Rate: {week_stats['win_rate']:.1f}%")
    report_lines.append(f"  Total P&L: ${week_stats['total_pnl']:+,.2f}")
    report_lines.append("")

    # All-time stats
    all_stats = tracker.get_all_time_stats()
    report_lines.append("ALL-TIME PERFORMANCE:")
    report_lines.append(f"  Total Trades: {all_stats['n_trades']}")
    if all_stats['win_rate'] is not None:
        report_lines.append(f"  Win Rate: {all_stats['win_rate']:.1f}%")
    if all_stats['avg_pnl'] is not None:
        report_lines.append(f"  Avg Trade: ${all_stats['avg_pnl']:.2f}")
    report_lines.append(f"  Cumulative P&L: ${all_stats['total_pnl']:+,.2f}")
    report_lines.append("")

    # Comparison to backtest
    comparison = tracker.compare_to_backtest()
    report_lines.append("VS BACKTEST EXPECTATIONS:")
    report_lines.append(f"  Expected Win Rate: {comparison['expected_win_rate']:.1f}%")
    report_lines.append(f"  Live Win Rate: {comparison['live_win_rate']:.1f}%" if comparison['live_win_rate'] else "  Live Win Rate: N/A")
    if comparison.get('win_rate_vs_expected'):
        pct = comparison['win_rate_vs_expected'] * 100
        status = "OK" if pct >= 70 else "CONCERNING" if pct >= 40 else "CRITICAL"
        report_lines.append(f"  Performance Ratio: {pct:.0f}% ({status})")
    report_lines.append("")

    # Red flags
    report_lines.append("RED FLAGS:")
    red_flags = []

    if comparison.get('win_rate_vs_expected') and comparison['win_rate_vs_expected'] < 0.4:
        red_flags.append("Live performance significantly below backtest")

    consecutive_losses = tracker.get_consecutive_losses()
    if consecutive_losses >= 3:
        red_flags.append(f"{consecutive_losses} consecutive losing trades")

    days_since = tracker.get_days_since_last_trade()
    if days_since and days_since >= 5:
        red_flags.append(f"No trades for {days_since} trading days")

    if red_flags:
        for flag in red_flags:
            report_lines.append(f"  - {flag}")
    else:
        report_lines.append("  None")

    report_lines.append("")
    report_lines.append("=" * 60)
    report_lines.append(f"  Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    report_lines.append("=" * 60)

    report = "\n".join(report_lines)

    if save:
        report_file = REPORTS_DIR / f"weekly_{today.strftime('%Y%m%d')}.txt"
        with open(report_file, 'w') as f:
            f.write(report)
        cprint(f"[REPORT] Weekly report saved to: {report_file}", "green")

    return report


# ============================================================================
# DASHBOARD INTEGRATION
# ============================================================================

def update_dashboard_data():
    """Update dashboard data file with current status."""
    tracker = PerformanceTracker()
    alert_mgr = AlertManager()

    # Get bot state
    bot_state = "UNKNOWN"
    if STATE_FILE.exists():
        try:
            with open(STATE_FILE, 'r') as f:
                state = json.load(f)
            bot_state = state.get('state', 'UNKNOWN')
        except:
            pass

    # Get next scheduled run
    try:
        from src.scheduler.gap_go_scheduler import get_next_trading_day
        next_run = get_next_trading_day()
        next_run_str = next_run.strftime('%Y-%m-%d %H:%M:%S %Z')
    except:
        next_run_str = "Unknown"

    # Get positions
    try:
        from src.utils.order_utils import get_all_positions
        positions = get_all_positions()
        gap_go_positions = [p for p in positions if p['symbol'] in ['AMD', 'META']]
    except:
        gap_go_positions = []

    # Build dashboard data
    today_stats = tracker.get_today_stats()

    # Get last 5 trades
    last_trades = []
    for trade in tracker.exit_trades[-5:]:
        last_trades.append({
            'timestamp': trade.get('timestamp'),
            'symbol': trade.get('symbol'),
            'direction': trade.get('direction'),
            'pnl': parse_trade_pnl(trade),
        })

    dashboard_data = {
        'updated': datetime.now().isoformat(),
        'bot_status': bot_state,
        'next_run': next_run_str,
        'today': {
            'trades': today_stats['n_trades'],
            'pnl': today_stats['total_pnl'],
            'winners': today_stats['winners'],
            'losers': today_stats['losers'],
        },
        'positions': gap_go_positions,
        'last_trades': last_trades,
        'active_alerts': alert_mgr.get_active_alerts(),
        'health': health_check()[0].value,
    }

    with open(DASHBOARD_DATA_FILE, 'w') as f:
        json.dump(dashboard_data, f, indent=2, default=str)

    return dashboard_data


# ============================================================================
# BACKTEST VALIDATION
# ============================================================================

def run_backtest_validation() -> Dict:
    """
    Run validation comparing recent performance to backtest expectations.

    Note: Full re-backtesting would require significant computation.
    This is a simplified validation comparing live metrics to expectations.
    """
    tracker = PerformanceTracker()
    validation = {
        'timestamp': datetime.now().isoformat(),
        'status': 'OK',
        'findings': [],
    }

    # Compare live to backtest for each symbol
    for symbol, expected in BACKTEST_EXPECTATIONS.items():
        # Filter trades by symbol
        symbol_trades = [t for t in tracker.exit_trades if t.get('symbol') == symbol]

        if not symbol_trades:
            validation['findings'].append({
                'symbol': symbol,
                'status': 'NO_DATA',
                'message': f'No completed trades for {symbol}'
            })
            continue

        pnls = [parse_trade_pnl(t) for t in symbol_trades if parse_trade_pnl(t) is not None]
        if not pnls:
            continue

        winners = [p for p in pnls if p > 0]
        live_win_rate = len(winners) / len(pnls) * 100

        finding = {
            'symbol': symbol,
            'live_trades': len(pnls),
            'live_win_rate': live_win_rate,
            'expected_win_rate': expected['win_rate'],
            'win_rate_ratio': live_win_rate / expected['win_rate'],
        }

        if finding['win_rate_ratio'] < 0.4:
            finding['status'] = 'CRITICAL'
            validation['status'] = 'CRITICAL'
        elif finding['win_rate_ratio'] < 0.7:
            finding['status'] = 'WARNING'
            if validation['status'] == 'OK':
                validation['status'] = 'WARNING'
        else:
            finding['status'] = 'OK'

        validation['findings'].append(finding)

    # Save validation report
    report_file = REPORTS_DIR / f"validation_{date.today().strftime('%Y%m%d')}.csv"
    with open(report_file, 'w', newline='') as f:
        if validation['findings']:
            writer = csv.DictWriter(f, fieldnames=validation['findings'][0].keys())
            writer.writeheader()
            writer.writerows(validation['findings'])

    cprint(f"[VALIDATE] Validation report saved to: {report_file}", "green")

    return validation


# ============================================================================
# CLI COMMANDS
# ============================================================================

def cmd_status():
    """Show current status and health check."""
    cprint("\n" + "=" * 60, "cyan")
    cprint("  GAP AND GO MONITORING STATUS", "cyan", attrs=['bold'])
    cprint("=" * 60, "cyan")

    # Health check
    status, checks = health_check()
    color = 'green' if status == HealthStatus.OK else 'yellow' if status == HealthStatus.WARN else 'red'
    cprint(f"\n[HEALTH] Overall Status: {status.value}", color, attrs=['bold'])

    for check_name, check_result in checks.items():
        check_status = check_result.get('status', 'UNKNOWN')
        color = 'green' if check_status == 'OK' else 'yellow' if check_status == 'WARN' else 'red'
        cprint(f"  {check_name}: {check_status}", color)
        if check_result.get('message'):
            cprint(f"    {check_result['message']}", "white")

    # Performance summary
    tracker = PerformanceTracker()
    today_stats = tracker.get_today_stats()
    all_stats = tracker.get_all_time_stats()

    cprint(f"\n[PERFORMANCE]", "cyan")
    cprint(f"  Today: {today_stats['n_trades']} trades, ${today_stats['total_pnl']:+,.2f} P&L", "white")
    cprint(f"  All-time: {all_stats['n_trades']} trades, ${all_stats['total_pnl']:+,.2f} P&L", "white")
    if all_stats['win_rate'] is not None:
        cprint(f"  Win Rate: {all_stats['win_rate']:.1f}%", "white")

    # Active alerts
    alert_mgr = AlertManager()
    alerts = alert_mgr.get_active_alerts()
    cprint(f"\n[ALERTS] Active: {len(alerts)}", "cyan")
    for alert in alerts:
        color = 'yellow' if alert['severity'] == 'WARNING' else 'red'
        cprint(f"  [{alert['severity']}] {alert['code']}: {alert['message']}", color)

    # Update dashboard
    update_dashboard_data()
    cprint(f"\n[DASHBOARD] Data updated: {DASHBOARD_DATA_FILE}", "green")

    cprint("\n" + "=" * 60 + "\n", "cyan")


def cmd_daily():
    """Generate and display daily report."""
    report = generate_daily_report(save=True)
    print(report)


def cmd_weekly():
    """Generate and display weekly report."""
    report = generate_weekly_report(save=True)
    print(report)


def cmd_alerts():
    """Check and display alerts."""
    cprint("\n" + "=" * 60, "cyan")
    cprint("  ALERT CHECK", "cyan", attrs=['bold'])
    cprint("=" * 60, "cyan")

    tracker = PerformanceTracker()
    alert_mgr = AlertManager()

    cprint("\n[CHECKING] Running alert checks...", "white")
    alerts_generated = alert_mgr.check_alerts(tracker)

    if not alerts_generated:
        cprint("\n[RESULT] No new alerts generated", "green")

    active = alert_mgr.get_active_alerts()
    cprint(f"\n[ACTIVE ALERTS] Total: {len(active)}", "cyan")

    for alert in active:
        color = 'yellow' if alert['severity'] == 'WARNING' else 'red'
        cprint(f"\n  [{alert['severity']}] {alert['code']}", color, attrs=['bold'])
        cprint(f"  {alert['message']}", color)
        cprint(f"  Time: {alert['timestamp']}", "white")

    cprint("\n" + "=" * 60 + "\n", "cyan")


def cmd_validate():
    """Run backtest validation."""
    cprint("\n" + "=" * 60, "cyan")
    cprint("  BACKTEST VALIDATION", "cyan", attrs=['bold'])
    cprint("=" * 60, "cyan")

    validation = run_backtest_validation()

    status_color = 'green' if validation['status'] == 'OK' else 'yellow' if validation['status'] == 'WARNING' else 'red'
    cprint(f"\n[VALIDATION] Status: {validation['status']}", status_color, attrs=['bold'])

    for finding in validation['findings']:
        color = 'green' if finding.get('status') == 'OK' else 'yellow' if finding.get('status') == 'WARNING' else 'red'
        cprint(f"\n  {finding.get('symbol', 'UNKNOWN')}:", color)

        if finding.get('status') == 'NO_DATA':
            cprint(f"    {finding.get('message')}", "yellow")
        else:
            cprint(f"    Live Trades: {finding.get('live_trades', 0)}", "white")
            cprint(f"    Live Win Rate: {finding.get('live_win_rate', 0):.1f}%", "white")
            cprint(f"    Expected Win Rate: {finding.get('expected_win_rate', 0):.1f}%", "white")
            ratio = finding.get('win_rate_ratio', 0)
            cprint(f"    Ratio: {ratio*100:.0f}% of expected", color)

    cprint("\n" + "=" * 60 + "\n", "cyan")


def cmd_performance():
    """Show rolling performance statistics."""
    cprint("\n" + "=" * 60, "cyan")
    cprint("  PERFORMANCE STATISTICS", "cyan", attrs=['bold'])
    cprint("=" * 60, "cyan")

    tracker = PerformanceTracker()

    # Today
    today_stats = tracker.get_today_stats()
    cprint(f"\n[TODAY]", "cyan")
    cprint(f"  Trades: {today_stats['n_trades']}", "white")
    cprint(f"  P&L: ${today_stats['total_pnl']:+,.2f}", "green" if today_stats['total_pnl'] >= 0 else "red")

    # Rolling stats
    for n in [10, 20, 50]:
        stats = tracker.get_rolling_stats(n)
        cprint(f"\n[LAST {n} TRADES]", "cyan")
        cprint(f"  Trades: {stats['n_trades']}", "white")
        if stats['win_rate'] is not None:
            cprint(f"  Win Rate: {stats['win_rate']:.1f}%", "white")
        if stats['avg_pnl'] is not None:
            cprint(f"  Avg Trade: ${stats['avg_pnl']:.2f}", "white")
        cprint(f"  Total P&L: ${stats['total_pnl']:+,.2f}", "green" if stats['total_pnl'] >= 0 else "red")
        if stats['avg_win'] and stats['avg_loss']:
            cprint(f"  Avg Win: ${stats['avg_win']:.2f} | Avg Loss: ${stats['avg_loss']:.2f}", "white")

    # All-time
    all_stats = tracker.get_all_time_stats()
    cprint(f"\n[ALL-TIME]", "cyan")
    cprint(f"  Trades: {all_stats['n_trades']}", "white")
    if all_stats['win_rate'] is not None:
        cprint(f"  Win Rate: {all_stats['win_rate']:.1f}%", "white")
    if all_stats['avg_pnl'] is not None:
        cprint(f"  Avg Trade: ${all_stats['avg_pnl']:.2f}", "white")
    cprint(f"  Total P&L: ${all_stats['total_pnl']:+,.2f}", "green" if all_stats['total_pnl'] >= 0 else "red")

    # Comparison
    comparison = tracker.compare_to_backtest()
    cprint(f"\n[VS BACKTEST]", "cyan")
    cprint(f"  Expected Win Rate: {comparison['expected_win_rate']:.1f}%", "white")
    if comparison['live_win_rate'] is not None:
        cprint(f"  Live Win Rate: {comparison['live_win_rate']:.1f}%", "white")
    if comparison.get('win_rate_vs_expected'):
        pct = comparison['win_rate_vs_expected'] * 100
        color = 'green' if pct >= 70 else 'yellow' if pct >= 40 else 'red'
        cprint(f"  Performance: {pct:.0f}% of expected", color)

    cprint("\n" + "=" * 60 + "\n", "cyan")


# ============================================================================
# MAIN
# ============================================================================

def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description="Gap and Go Monitoring System")
    parser.add_argument('--status', action='store_true', help='Show current status and health check')
    parser.add_argument('--daily', action='store_true', help='Generate daily report')
    parser.add_argument('--weekly', action='store_true', help='Generate weekly report')
    parser.add_argument('--alerts', action='store_true', help='Check and show alerts')
    parser.add_argument('--validate', action='store_true', help='Run backtest validation')
    parser.add_argument('--performance', action='store_true', help='Show performance statistics')
    parser.add_argument('--update-dashboard', action='store_true', help='Update dashboard data file')

    args = parser.parse_args()

    if args.status:
        cmd_status()
    elif args.daily:
        cmd_daily()
    elif args.weekly:
        cmd_weekly()
    elif args.alerts:
        cmd_alerts()
    elif args.validate:
        cmd_validate()
    elif args.performance:
        cmd_performance()
    elif args.update_dashboard:
        data = update_dashboard_data()
        cprint(f"Dashboard data updated: {DASHBOARD_DATA_FILE}", "green")
        print(json.dumps(data, indent=2, default=str))
    else:
        # Default: show status
        cmd_status()


if __name__ == "__main__":
    main()


# ============================================================================
# FUTURE ENHANCEMENT HOOKS
# ============================================================================

# TODO: Email alerts
# def send_email_alert(subject: str, body: str, recipients: List[str]):
#     """Send email alert via SMTP."""
#     # import smtplib
#     # from email.mime.text import MIMEText
#     # Configure SMTP_HOST, SMTP_PORT, SMTP_USER, SMTP_PASS in .env
#     pass

# TODO: SMS alerts via Twilio
# def send_sms_alert(message: str, phone_numbers: List[str]):
#     """Send SMS alert via Twilio."""
#     # from twilio.rest import Client
#     # Configure TWILIO_ACCOUNT_SID, TWILIO_AUTH_TOKEN in .env
#     pass

# TODO: Discord/Slack webhook
# def send_webhook_alert(message: str, webhook_url: str):
#     """Send alert via Discord or Slack webhook."""
#     # import requests
#     # requests.post(webhook_url, json={'content': message})
#     pass
