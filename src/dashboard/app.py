"""
Trading Bot Dashboard

Simple Flask dashboard showing:
- Account balance and P&L from Alpaca
- Open positions with entry price and P&L
- Recent trades from entry/exit logs
- Scanner results (stocks near breakout)
- Risk status (daily P&L, exposure)

Usage:
    python3 src/dashboard/app.py

Dashboard runs at http://localhost:5000
Auto-refreshes every 30 seconds
"""

import csv
import os
import sys
from datetime import datetime
from pathlib import Path

from flask import Flask, render_template_string
from dotenv import load_dotenv

sys.path.append('/Users/josiahgarcia/trading-bot')
load_dotenv()

# Alpaca imports
try:
    import alpaca_trade_api as tradeapi
    ALPACA_AVAILABLE = True
except ImportError:
    ALPACA_AVAILABLE = False

# Configuration
ALPACA_API_KEY = os.getenv("ALPACA_API_KEY")
ALPACA_SECRET_KEY = os.getenv("ALPACA_SECRET_KEY")
ALPACA_PAPER = os.getenv("ALPACA_PAPER", "true").lower() == "true"

# Paths
CSV_DIR = Path(__file__).parent.parent.parent / 'csvs'
ENTRY_LOG_PATH = CSV_DIR / 'entry_log.csv'
EXIT_LOG_PATH = CSV_DIR / 'exit_log.csv'
SCANNER_PATH = CSV_DIR / 'scanner_results.csv'
RISK_LOG_PATH = CSV_DIR / 'risk_log.csv'

app = Flask(__name__)


def get_alpaca_api():
    """Get Alpaca API instance."""
    if not ALPACA_AVAILABLE:
        return None
    if not ALPACA_API_KEY or not ALPACA_SECRET_KEY:
        return None

    base_url = "https://paper-api.alpaca.markets" if ALPACA_PAPER else "https://api.alpaca.markets"
    return tradeapi.REST(ALPACA_API_KEY, ALPACA_SECRET_KEY, base_url, api_version='v2')


def get_account_info():
    """Get account balance and P&L from Alpaca."""
    api = get_alpaca_api()
    if not api:
        return {
            'equity': 0,
            'cash': 0,
            'buying_power': 0,
            'daily_pnl': 0,
            'daily_pnl_pct': 0,
            'status': 'API Not Connected'
        }

    try:
        account = api.get_account()
        equity = float(account.equity)
        last_equity = float(account.last_equity)
        daily_pnl = equity - last_equity
        daily_pnl_pct = (daily_pnl / last_equity * 100) if last_equity > 0 else 0

        return {
            'equity': equity,
            'cash': float(account.cash),
            'buying_power': float(account.buying_power),
            'daily_pnl': daily_pnl,
            'daily_pnl_pct': daily_pnl_pct,
            'status': 'PAPER' if ALPACA_PAPER else 'LIVE'
        }
    except Exception as e:
        return {
            'equity': 0,
            'cash': 0,
            'buying_power': 0,
            'daily_pnl': 0,
            'daily_pnl_pct': 0,
            'status': f'Error: {str(e)}'
        }


def get_positions():
    """Get open positions from Alpaca."""
    api = get_alpaca_api()
    if not api:
        return []

    try:
        positions = api.list_positions()
        result = []
        for pos in positions:
            result.append({
                'symbol': pos.symbol,
                'qty': float(pos.qty),
                'entry_price': float(pos.avg_entry_price),
                'current_price': float(pos.current_price),
                'market_value': float(pos.market_value),
                'unrealized_pnl': float(pos.unrealized_pl),
                'unrealized_pnl_pct': float(pos.unrealized_plpc) * 100,
                'side': 'LONG' if float(pos.qty) > 0 else 'SHORT'
            })
        return result
    except Exception as e:
        return []


def read_csv_file(path, limit=20):
    """Read a CSV file and return rows as dicts."""
    if not path.exists():
        return []

    try:
        with open(path, 'r') as f:
            reader = csv.DictReader(f)
            rows = list(reader)
            # Return most recent first
            return rows[-limit:][::-1]
    except Exception as e:
        return []


def get_entry_logs():
    """Get recent entry logs."""
    return read_csv_file(ENTRY_LOG_PATH, limit=15)


def get_exit_logs():
    """Get recent exit logs."""
    return read_csv_file(EXIT_LOG_PATH, limit=15)


def get_scanner_results():
    """Get scanner results."""
    return read_csv_file(SCANNER_PATH, limit=20)


def get_risk_status():
    """Get latest risk status from log."""
    rows = read_csv_file(RISK_LOG_PATH, limit=1)
    if rows:
        row = rows[0]
        return {
            'daily_pnl': float(row.get('daily_pnl', 0)),
            'total_exposure': float(row.get('total_exposure', 0)),
            'trading_allowed': row.get('trading_allowed', 'YES') == 'YES',
            'open_positions': int(row.get('open_positions', 0)),
            'timestamp': row.get('timestamp', '')
        }
    return {
        'daily_pnl': 0,
        'total_exposure': 0,
        'trading_allowed': True,
        'open_positions': 0,
        'timestamp': 'No data'
    }


# HTML Template
DASHBOARD_HTML = """
<!DOCTYPE html>
<html>
<head>
    <title>Trading Bot Dashboard</title>
    <meta http-equiv="refresh" content="30">
    <style>
        body {
            font-family: monospace;
            background: #1a1a2e;
            color: #eee;
            margin: 20px;
            font-size: 14px;
        }
        h1 { color: #00d4ff; margin-bottom: 5px; }
        h2 { color: #00d4ff; margin-top: 30px; border-bottom: 1px solid #333; padding-bottom: 5px; }
        .timestamp { color: #888; font-size: 12px; margin-bottom: 20px; }
        table {
            border-collapse: collapse;
            width: 100%;
            margin-top: 10px;
            background: #16213e;
        }
        th, td {
            border: 1px solid #333;
            padding: 8px 12px;
            text-align: left;
        }
        th { background: #0f3460; color: #00d4ff; }
        tr:hover { background: #1f4068; }
        .positive { color: #00ff88; }
        .negative { color: #ff4757; }
        .neutral { color: #ffd93d; }
        .status-box {
            display: inline-block;
            padding: 15px 25px;
            margin: 5px;
            background: #16213e;
            border: 1px solid #333;
            border-radius: 5px;
        }
        .status-label { color: #888; font-size: 12px; }
        .status-value { font-size: 24px; font-weight: bold; }
        .grid { display: flex; flex-wrap: wrap; gap: 10px; }
        .warning { background: #ff4757; color: white; padding: 10px; border-radius: 5px; }
        .info { color: #888; font-style: italic; }
    </style>
</head>
<body>
    <h1>Trading Bot Dashboard</h1>
    <div class="timestamp">Last updated: {{ now }} | Auto-refresh: 30s | Mode: {{ account.status }}</div>

    <h2>Account Overview</h2>
    <div class="grid">
        <div class="status-box">
            <div class="status-label">Equity</div>
            <div class="status-value">${{ "{:,.2f}".format(account.equity) }}</div>
        </div>
        <div class="status-box">
            <div class="status-label">Cash</div>
            <div class="status-value">${{ "{:,.2f}".format(account.cash) }}</div>
        </div>
        <div class="status-box">
            <div class="status-label">Buying Power</div>
            <div class="status-value">${{ "{:,.2f}".format(account.buying_power) }}</div>
        </div>
        <div class="status-box">
            <div class="status-label">Daily P&L</div>
            <div class="status-value {{ 'positive' if account.daily_pnl >= 0 else 'negative' }}">
                ${{ "{:+,.2f}".format(account.daily_pnl) }} ({{ "{:+.2f}".format(account.daily_pnl_pct) }}%)
            </div>
        </div>
    </div>

    <h2>Risk Status</h2>
    {% if not risk.trading_allowed %}
    <div class="warning">TRADING HALTED - Daily loss limit reached!</div>
    {% endif %}
    <div class="grid">
        <div class="status-box">
            <div class="status-label">Bot Daily P&L</div>
            <div class="status-value {{ 'positive' if risk.daily_pnl >= 0 else 'negative' }}">
                ${{ "{:+,.2f}".format(risk.daily_pnl) }}
            </div>
        </div>
        <div class="status-box">
            <div class="status-label">Exposure</div>
            <div class="status-value">${{ "{:,.0f}".format(risk.total_exposure) }}</div>
        </div>
        <div class="status-box">
            <div class="status-label">Open Positions</div>
            <div class="status-value">{{ risk.open_positions }}</div>
        </div>
        <div class="status-box">
            <div class="status-label">Trading Status</div>
            <div class="status-value {{ 'positive' if risk.trading_allowed else 'negative' }}">
                {{ 'ACTIVE' if risk.trading_allowed else 'HALTED' }}
            </div>
        </div>
    </div>

    <h2>Open Positions ({{ positions|length }})</h2>
    {% if positions %}
    <table>
        <tr>
            <th>Symbol</th>
            <th>Side</th>
            <th>Qty</th>
            <th>Entry</th>
            <th>Current</th>
            <th>Value</th>
            <th>P&L</th>
            <th>P&L %</th>
        </tr>
        {% for pos in positions %}
        <tr>
            <td><strong>{{ pos.symbol }}</strong></td>
            <td>{{ pos.side }}</td>
            <td>{{ "{:.4f}".format(pos.qty) }}</td>
            <td>${{ "{:.2f}".format(pos.entry_price) }}</td>
            <td>${{ "{:.2f}".format(pos.current_price) }}</td>
            <td>${{ "{:,.2f}".format(pos.market_value) }}</td>
            <td class="{{ 'positive' if pos.unrealized_pnl >= 0 else 'negative' }}">
                ${{ "{:+,.2f}".format(pos.unrealized_pnl) }}
            </td>
            <td class="{{ 'positive' if pos.unrealized_pnl_pct >= 0 else 'negative' }}">
                {{ "{:+.2f}".format(pos.unrealized_pnl_pct) }}%
            </td>
        </tr>
        {% endfor %}
    </table>
    {% else %}
    <p class="info">No open positions</p>
    {% endif %}

    <h2>Scanner Results - Near Breakout</h2>
    {% if scanner %}
    <table>
        <tr>
            <th>Symbol</th>
            <th>Price</th>
            <th>24h High</th>
            <th>24h Low</th>
            <th>ATR</th>
            <th>Distance %</th>
            <th>Direction</th>
        </tr>
        {% for s in scanner %}
        <tr>
            <td><strong>{{ s.get('symbol', 'N/A') }}</strong></td>
            <td>${{ s.get('price', 'N/A') }}</td>
            <td>${{ s.get('high_24h', 'N/A') }}</td>
            <td>${{ s.get('low_24h', 'N/A') }}</td>
            <td>${{ s.get('atr', 'N/A') }}</td>
            <td class="neutral">{{ s.get('nearest_distance_pct', 'N/A') }}%</td>
            <td>{{ s.get('direction', 'N/A') }}</td>
        </tr>
        {% endfor %}
    </table>
    {% else %}
    <p class="info">No scanner results - run scanner to populate</p>
    {% endif %}

    <h2>Recent Entry Signals</h2>
    {% if entries %}
    <table>
        <tr>
            <th>Time</th>
            <th>Symbol</th>
            <th>Direction</th>
            <th>Price</th>
            <th>Confidence</th>
            <th>Decision</th>
            <th>Reasons</th>
        </tr>
        {% for e in entries %}
        <tr>
            <td>{{ e.get('timestamp', '')[:19] }}</td>
            <td><strong>{{ e.get('symbol', 'N/A') }}</strong></td>
            <td>{{ e.get('direction', 'N/A') }}</td>
            <td>${{ e.get('price', 'N/A') }}</td>
            <td>{{ e.get('confidence', 'N/A') }}%</td>
            <td class="{{ 'positive' if e.get('should_enter') == 'YES' else 'negative' }}">
                {{ 'ENTER' if e.get('should_enter') == 'YES' else 'SKIP' }}
            </td>
            <td style="font-size: 11px;">{{ e.get('reasons', '')[:60] }}...</td>
        </tr>
        {% endfor %}
    </table>
    {% else %}
    <p class="info">No entry signals logged yet</p>
    {% endif %}

    <h2>Recent Exit Decisions</h2>
    {% if exits %}
    <table>
        <tr>
            <th>Time</th>
            <th>Symbol</th>
            <th>Entry</th>
            <th>Current</th>
            <th>P&L %</th>
            <th>Hold (mins)</th>
            <th>Decision</th>
            <th>Reason</th>
        </tr>
        {% for e in exits %}
        <tr>
            <td>{{ e.get('timestamp', '')[:19] }}</td>
            <td><strong>{{ e.get('symbol', 'N/A') }}</strong></td>
            <td>${{ e.get('entry_price', 'N/A') }}</td>
            <td>${{ e.get('current_price', 'N/A') }}</td>
            <td class="{{ 'positive' if (e.get('pnl_pct', '0')|float) >= 0 else 'negative' }}">
                {{ e.get('pnl_pct', 'N/A') }}%
            </td>
            <td>{{ e.get('hold_time_mins', 'N/A') }}</td>
            <td class="{{ 'positive' if 'CLOSE' in e.get('decision', '') else 'neutral' }}">
                {{ e.get('decision', 'N/A') }}
            </td>
            <td style="font-size: 11px;">{{ e.get('reason', '')[:50] }}...</td>
        </tr>
        {% endfor %}
    </table>
    {% else %}
    <p class="info">No exit decisions logged yet</p>
    {% endif %}

    <div style="margin-top: 40px; color: #555; font-size: 12px;">
        Trading Bot Dashboard | Data from Alpaca API and local CSV logs
    </div>
</body>
</html>
"""


@app.route('/')
def dashboard():
    """Main dashboard route."""
    return render_template_string(
        DASHBOARD_HTML,
        now=datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        account=get_account_info(),
        positions=get_positions(),
        risk=get_risk_status(),
        scanner=get_scanner_results(),
        entries=get_entry_logs(),
        exits=get_exit_logs()
    )


@app.route('/api/status')
def api_status():
    """Simple API endpoint for status check."""
    account = get_account_info()
    risk = get_risk_status()
    return {
        'status': 'ok',
        'timestamp': datetime.now().isoformat(),
        'equity': account['equity'],
        'daily_pnl': account['daily_pnl'],
        'trading_allowed': risk['trading_allowed'],
        'open_positions': len(get_positions())
    }


if __name__ == '__main__':
    print("\n" + "=" * 50)
    print("  TRADING BOT DASHBOARD")
    print("=" * 50)
    print(f"  URL: http://localhost:5001")
    print(f"  Mode: {'PAPER' if ALPACA_PAPER else 'LIVE'}")
    print(f"  Auto-refresh: 30 seconds")
    print("=" * 50 + "\n")

    app.run(host='0.0.0.0', port=5001, debug=True)
