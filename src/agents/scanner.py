"""
Breakout Scanner Agent

Scans the top 50 most liquid stocks for potential breakout setups.
- Fetches most active stocks from Alpaca API
- Calculates 24h high/low and ATR for each
- Ranks by proximity to breakout level (within 1% of high/low)
- Outputs results to csvs/scanner_results.csv

Usage:
    from src.agents.scanner import get_breakout_candidates, run_scan

    # Get candidates for the main bot
    candidates = get_breakout_candidates(top_n=10)

    # Run full scan with output
    results = run_scan()
"""

import csv
import os
import sys
from datetime import datetime, timedelta
from pathlib import Path
from typing import List, Dict, Any, Optional

from dotenv import load_dotenv
from termcolor import cprint

sys.path.append('/Users/josiahgarcia/trading-bot')
load_dotenv()

# Alpaca imports
try:
    import alpaca_trade_api as tradeapi
    from alpaca_trade_api.rest import REST
    ALPACA_AVAILABLE = True
except ImportError:
    ALPACA_AVAILABLE = False

# Configuration
ALPACA_API_KEY = os.getenv("ALPACA_API_KEY")
ALPACA_SECRET_KEY = os.getenv("ALPACA_SECRET_KEY")
ALPACA_PAPER = os.getenv("ALPACA_PAPER", "true").lower() == "true"

# Scanner parameters
TOP_N_STOCKS = 50          # Number of most active stocks to scan
BREAKOUT_THRESHOLD = 0.01  # 1% from high/low to consider "near breakout"
ATR_PERIOD = 14            # ATR calculation period
CANDLE_HOURS = 24          # Hours of data for high/low calculation

# Paths
CSV_DIR = Path(__file__).parent.parent.parent / 'csvs'
SCANNER_RESULTS_PATH = CSV_DIR / 'scanner_results.csv'


class BreakoutScanner:
    """
    Scanner that identifies stocks near breakout levels.

    Moon Dev says: "Find the coiled springs before they pop."
    """

    def __init__(self, mock_mode=False):
        self.api = None
        self.mock_mode = mock_mode

        if not mock_mode:
            if not ALPACA_AVAILABLE:
                raise RuntimeError("alpaca-trade-api package not installed")

            if not ALPACA_API_KEY or not ALPACA_SECRET_KEY:
                raise RuntimeError("ALPACA_API_KEY and ALPACA_SECRET_KEY required in .env")

            base_url = "https://paper-api.alpaca.markets" if ALPACA_PAPER else "https://api.alpaca.markets"
            self.api = tradeapi.REST(ALPACA_API_KEY, ALPACA_SECRET_KEY, base_url, api_version='v2')

        # Ensure CSV directory exists
        CSV_DIR.mkdir(parents=True, exist_ok=True)

        # Mock data for testing
        self.mock_data = self._generate_mock_data() if mock_mode else {}

    def _generate_mock_data(self) -> Dict:
        """Generate realistic mock data for testing."""
        import random

        symbols_data = {
            'SPY': {'base': 590, 'vol': 50000000},
            'QQQ': {'base': 510, 'vol': 40000000},
            'AAPL': {'base': 240, 'vol': 60000000},
            'MSFT': {'base': 420, 'vol': 25000000},
            'GOOGL': {'base': 175, 'vol': 20000000},
            'AMZN': {'base': 220, 'vol': 35000000},
            'NVDA': {'base': 140, 'vol': 45000000},
            'META': {'base': 590, 'vol': 15000000},
            'TSLA': {'base': 410, 'vol': 80000000},
            'AMD': {'base': 125, 'vol': 50000000},
            'NFLX': {'base': 920, 'vol': 5000000},
            'INTC': {'base': 22, 'vol': 40000000},
            'COST': {'base': 920, 'vol': 2000000},
            'PEP': {'base': 155, 'vol': 5000000},
            'AVGO': {'base': 235, 'vol': 3000000},
            'CSCO': {'base': 58, 'vol': 20000000},
            'ADBE': {'base': 450, 'vol': 3000000},
            'CRM': {'base': 330, 'vol': 5000000},
            'ORCL': {'base': 165, 'vol': 8000000},
            'ACN': {'base': 350, 'vol': 2000000},
            'TXN': {'base': 195, 'vol': 4000000},
            'QCOM': {'base': 165, 'vol': 6000000},
            'INTU': {'base': 630, 'vol': 1500000},
            'ISRG': {'base': 520, 'vol': 1000000},
            'AMAT': {'base': 180, 'vol': 6000000},
            'BKNG': {'base': 4800, 'vol': 300000},
            'SBUX': {'base': 95, 'vol': 8000000},
            'GILD': {'base': 95, 'vol': 6000000},
            'MDLZ': {'base': 68, 'vol': 5000000},
            'ADI': {'base': 220, 'vol': 2500000},
            'LRCX': {'base': 75, 'vol': 2000000},
            'REGN': {'base': 760, 'vol': 600000},
            'PANW': {'base': 390, 'vol': 2000000},
            'KLAC': {'base': 720, 'vol': 1000000},
            'MELI': {'base': 2100, 'vol': 400000},
            'PYPL': {'base': 88, 'vol': 10000000},
            'CRWD': {'base': 370, 'vol': 3000000},
            'ABNB': {'base': 135, 'vol': 5000000},
            'COIN': {'base': 280, 'vol': 8000000},
            'PLTR': {'base': 75, 'vol': 50000000},
            'SOFI': {'base': 15, 'vol': 30000000},
            'HOOD': {'base': 45, 'vol': 25000000},
            'RBLX': {'base': 55, 'vol': 10000000},
            'SNAP': {'base': 12, 'vol': 20000000},
            'ROKU': {'base': 85, 'vol': 3000000},
            'SQ': {'base': 95, 'vol': 8000000},
            'SHOP': {'base': 110, 'vol': 5000000},
            'DDOG': {'base': 135, 'vol': 3000000},
            'ZS': {'base': 230, 'vol': 1500000},
            'NET': {'base': 115, 'vol': 5000000},
        }

        mock_data = {}
        for symbol, info in symbols_data.items():
            base = info['base']
            vol = info['vol']

            # Generate 24 hourly candles
            candles = []
            price = base * random.uniform(0.96, 0.99)

            for _ in range(24):
                volatility = base * random.uniform(0.005, 0.015)
                open_p = price
                close_p = price + random.uniform(-volatility, volatility)
                high_p = max(open_p, close_p) + random.uniform(0, volatility * 0.5)
                low_p = min(open_p, close_p) - random.uniform(0, volatility * 0.5)

                candles.append({
                    'open': open_p,
                    'high': high_p,
                    'low': low_p,
                    'close': close_p,
                    'volume': vol * random.uniform(0.03, 0.08)
                })
                price = close_p

            # Current price - some near highs, some near lows
            high_24h = max(c['high'] for c in candles)
            low_24h = min(c['low'] for c in candles)

            # Randomly position current price - bias some toward breakout levels
            if random.random() < 0.2:  # 20% near high
                current_price = high_24h * random.uniform(0.995, 1.005)
            elif random.random() < 0.2:  # 20% near low
                current_price = low_24h * random.uniform(0.995, 1.005)
            else:  # 60% somewhere in range
                current_price = low_24h + (high_24h - low_24h) * random.uniform(0.2, 0.8)

            mock_data[symbol] = {
                'candles': candles,
                'current_price': current_price
            }

        return mock_data

    def get_most_active_stocks(self, top_n=TOP_N_STOCKS) -> List[str]:
        """
        Get the most active stocks by volume from Alpaca.
        Falls back to a curated list if API fails.
        """
        if self.mock_mode:
            return list(self.mock_data.keys())[:top_n]

        try:
            # Get most active stocks using screener
            # Alpaca's asset API with filters
            assets = self.api.list_assets(status='active', asset_class='us_equity')

            # Filter for tradeable, easy-to-borrow stocks
            tradeable = [
                a for a in assets
                if a.tradable and a.easy_to_borrow and a.fractionable
            ]

            # Get recent snapshots for volume data
            # Use a curated list of known liquid symbols to start
            liquid_symbols = [
                'SPY', 'QQQ', 'AAPL', 'MSFT', 'GOOGL', 'AMZN', 'NVDA', 'META',
                'TSLA', 'AMD', 'NFLX', 'INTC', 'COST', 'PEP', 'AVGO', 'CSCO',
                'ADBE', 'TXN', 'QCOM', 'TMUS', 'AMGN', 'INTU', 'ISRG', 'CMCSA',
                'HON', 'AMAT', 'BKNG', 'SBUX', 'VRTX', 'ADP', 'GILD', 'MDLZ',
                'ADI', 'LRCX', 'REGN', 'PANW', 'SNPS', 'KLAC', 'ASML', 'CDNS',
                'MELI', 'PYPL', 'CRWD', 'MAR', 'ABNB', 'ORLY', 'MNST', 'FTNT',
                'NXPI', 'MRVL', 'CTAS', 'WDAY', 'DXCM', 'PCAR', 'ROST', 'PAYX',
                'CPRT', 'ODFL', 'KDP', 'MRNA', 'AEP', 'FAST', 'EA', 'IDXX',
                'BKR', 'VRSK', 'XEL', 'EXC', 'GEHC', 'CCEP', 'KHC', 'CTSH',
                'CHTR', 'CSGP', 'FANG', 'DDOG', 'ZS', 'TTD', 'TEAM', 'ANSS',
                'WBD', 'CDW', 'ILMN', 'GFS', 'DLTR', 'MDB', 'BIIB', 'ENPH',
                'WBA', 'JD', 'SIRI', 'LCID', 'RIVN', 'PLTR', 'SOFI', 'HOOD',
                'COIN', 'RBLX', 'SNAP', 'U', 'ROKU', 'SQ', 'SHOP', 'SE'
            ]

            # Try to get snapshots to rank by volume
            try:
                snapshots = self.api.get_snapshots(liquid_symbols[:top_n * 2])
                volume_data = []

                for symbol, snapshot in snapshots.items():
                    if snapshot and snapshot.daily_bar:
                        volume_data.append({
                            'symbol': symbol,
                            'volume': snapshot.daily_bar.volume,
                            'price': snapshot.daily_bar.close
                        })

                # Sort by volume and return top N
                volume_data.sort(key=lambda x: x['volume'], reverse=True)
                return [d['symbol'] for d in volume_data[:top_n]]

            except Exception as e:
                cprint(f"Snapshot fetch failed, using default list: {e}", "yellow")
                return liquid_symbols[:top_n]

        except Exception as e:
            cprint(f"Error fetching active stocks: {e}", "red")
            # Fallback to default list
            return [
                'SPY', 'QQQ', 'AAPL', 'MSFT', 'GOOGL', 'AMZN', 'NVDA', 'META',
                'TSLA', 'AMD', 'NFLX', 'INTC', 'COST', 'PEP', 'AVGO', 'CSCO',
                'ADBE', 'TXN', 'QCOM', 'TMUS', 'AMGN', 'INTU', 'ISRG', 'CMCSA',
                'HON', 'AMAT', 'BKNG', 'SBUX', 'VRTX', 'ADP', 'GILD', 'MDLZ',
                'ADI', 'LRCX', 'REGN', 'PANW', 'SNPS', 'KLAC', 'ASML', 'CDNS',
                'MELI', 'PYPL', 'CRWD', 'MAR', 'ABNB', 'ORLY', 'MNST', 'FTNT'
            ][:top_n]

    def fetch_candles(self, symbol, hours=CANDLE_HOURS) -> List[Dict]:
        """Fetch hourly candles for a symbol."""
        if self.mock_mode:
            if symbol in self.mock_data:
                return self.mock_data[symbol]['candles']
            return []

        try:
            end = datetime.now()
            start = end - timedelta(hours=hours + 8)  # Buffer for market hours

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
            return []

    def get_current_price(self, symbol) -> Optional[float]:
        """Get current price for a symbol."""
        if self.mock_mode:
            if symbol in self.mock_data:
                return self.mock_data[symbol]['current_price']
            return None

        try:
            trade = self.api.get_latest_trade(symbol)
            return float(trade.price)
        except:
            return None

    def calculate_24h_range(self, candles) -> tuple:
        """Calculate 24-hour high and low."""
        if not candles:
            return None, None

        high_24h = max(c['high'] for c in candles)
        low_24h = min(c['low'] for c in candles)

        return high_24h, low_24h

    def calculate_atr(self, candles, period=ATR_PERIOD) -> float:
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

    def calculate_breakout_proximity(self, current_price, high_24h, low_24h) -> Dict:
        """
        Calculate how close price is to breakout levels.
        Returns distance to high and low as percentages.
        """
        if current_price is None or high_24h is None or low_24h is None:
            return {
                'distance_to_high_pct': None,
                'distance_to_low_pct': None,
                'nearest_level': None,
                'nearest_distance_pct': None,
                'direction': None
            }

        distance_to_high = ((high_24h - current_price) / current_price) * 100
        distance_to_low = ((current_price - low_24h) / current_price) * 100

        if distance_to_high <= distance_to_low:
            nearest_level = 'HIGH'
            nearest_distance = distance_to_high
            direction = 'LONG'
        else:
            nearest_level = 'LOW'
            nearest_distance = distance_to_low
            direction = 'SHORT'

        return {
            'distance_to_high_pct': distance_to_high,
            'distance_to_low_pct': distance_to_low,
            'nearest_level': nearest_level,
            'nearest_distance_pct': nearest_distance,
            'direction': direction
        }

    def scan_symbol(self, symbol) -> Optional[Dict]:
        """Scan a single symbol for breakout setup."""
        candles = self.fetch_candles(symbol, CANDLE_HOURS)

        if not candles:
            return None

        high_24h, low_24h = self.calculate_24h_range(candles)
        atr = self.calculate_atr(candles)
        current_price = self.get_current_price(symbol)

        if current_price is None or high_24h is None:
            return None

        proximity = self.calculate_breakout_proximity(current_price, high_24h, low_24h)

        # Calculate range and volatility metrics
        range_pct = ((high_24h - low_24h) / low_24h) * 100 if low_24h else 0
        atr_pct = (atr / current_price) * 100 if current_price else 0

        # Calculate average volume
        avg_volume = sum(c['volume'] for c in candles) / len(candles) if candles else 0

        return {
            'symbol': symbol,
            'current_price': current_price,
            'high_24h': high_24h,
            'low_24h': low_24h,
            'range_pct': range_pct,
            'atr': atr,
            'atr_pct': atr_pct,
            'avg_volume': avg_volume,
            **proximity
        }

    def scan_all(self, symbols: List[str] = None, progress=True) -> List[Dict]:
        """
        Scan all symbols and return results sorted by breakout proximity.
        """
        if symbols is None:
            symbols = self.get_most_active_stocks(TOP_N_STOCKS)

        results = []
        total = len(symbols)

        if progress:
            cprint(f"\nScanning {total} symbols for breakout setups...\n", "cyan")

        for i, symbol in enumerate(symbols):
            if progress and (i + 1) % 10 == 0:
                cprint(f"  Progress: {i + 1}/{total} symbols scanned", "white")

            result = self.scan_symbol(symbol)
            if result and result['nearest_distance_pct'] is not None:
                results.append(result)

        # Sort by nearest distance to breakout level
        results.sort(key=lambda x: x['nearest_distance_pct'])

        return results

    def get_candidates(self, top_n=10, threshold=BREAKOUT_THRESHOLD * 100) -> List[Dict]:
        """
        Get top breakout candidates within threshold percentage of breakout level.
        """
        all_results = self.scan_all(progress=False)

        # Filter to those within threshold
        candidates = [
            r for r in all_results
            if r['nearest_distance_pct'] <= threshold
        ]

        return candidates[:top_n]

    def save_results(self, results: List[Dict], filepath=SCANNER_RESULTS_PATH):
        """Save scan results to CSV."""
        if not results:
            cprint("No results to save", "yellow")
            return

        with open(filepath, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([
                'timestamp',
                'rank',
                'symbol',
                'current_price',
                'high_24h',
                'low_24h',
                'range_pct',
                'atr',
                'atr_pct',
                'avg_volume',
                'distance_to_high_pct',
                'distance_to_low_pct',
                'nearest_level',
                'nearest_distance_pct',
                'direction',
                'within_1pct'
            ])

            timestamp = datetime.now().isoformat()

            for rank, result in enumerate(results, 1):
                within_1pct = result['nearest_distance_pct'] <= 1.0

                writer.writerow([
                    timestamp,
                    rank,
                    result['symbol'],
                    f"{result['current_price']:.2f}",
                    f"{result['high_24h']:.2f}",
                    f"{result['low_24h']:.2f}",
                    f"{result['range_pct']:.2f}",
                    f"{result['atr']:.2f}",
                    f"{result['atr_pct']:.2f}",
                    f"{result['avg_volume']:.0f}",
                    f"{result['distance_to_high_pct']:.2f}",
                    f"{result['distance_to_low_pct']:.2f}",
                    result['nearest_level'],
                    f"{result['nearest_distance_pct']:.2f}",
                    result['direction'],
                    "YES" if within_1pct else "NO"
                ])

        cprint(f"\nResults saved to: {filepath}", "green")

    def print_results(self, results: List[Dict], top_n=20):
        """Print formatted scan results."""
        cprint(f"\n{'='*80}", "cyan")
        cprint(f"  BREAKOUT SCANNER RESULTS - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}", "cyan")
        cprint(f"{'='*80}\n", "cyan")

        if not results:
            cprint("No results found", "yellow")
            return

        # Print header
        cprint(f"{'Rank':<5} {'Symbol':<8} {'Price':<10} {'24h Range':<20} {'Dist':<8} {'Dir':<6} {'ATR%':<8}", "white")
        cprint("-" * 80, "white")

        for rank, result in enumerate(results[:top_n], 1):
            symbol = result['symbol']
            price = result['current_price']
            high = result['high_24h']
            low = result['low_24h']
            distance = result['nearest_distance_pct']
            direction = result['direction']
            atr_pct = result['atr_pct']

            range_str = f"${low:.2f} - ${high:.2f}"

            # Color based on proximity
            if distance <= 0.5:
                color = "green"
                status = "***"
            elif distance <= 1.0:
                color = "yellow"
                status = "**"
            elif distance <= 2.0:
                color = "cyan"
                status = "*"
            else:
                color = "white"
                status = ""

            cprint(
                f"{rank:<5} {symbol:<8} ${price:<9.2f} {range_str:<20} {distance:<6.2f}% {direction:<6} {atr_pct:<6.2f}% {status}",
                color
            )

        cprint("\n" + "-" * 80, "white")
        cprint("Legend: *** Within 0.5% | ** Within 1% | * Within 2%", "white")

        # Summary
        within_1pct = len([r for r in results if r['nearest_distance_pct'] <= 1.0])
        within_2pct = len([r for r in results if r['nearest_distance_pct'] <= 2.0])

        cprint(f"\nSummary: {within_1pct} symbols within 1% of breakout, {within_2pct} within 2%", "cyan")


# Global scanner instance
_scanner = None


def get_scanner(mock_mode=False):
    """Get or create the global scanner instance."""
    global _scanner
    if _scanner is None or (_scanner.mock_mode != mock_mode):
        _scanner = BreakoutScanner(mock_mode=mock_mode)
    return _scanner


def get_breakout_candidates(top_n=10, threshold_pct=1.0, mock_mode=False) -> List[Dict]:
    """
    Get top breakout candidates for the main bot.

    Args:
        top_n: Maximum number of candidates to return
        threshold_pct: Maximum distance from breakout level (default 1%)
        mock_mode: Use mock data instead of live API

    Returns:
        List of dicts with symbol, direction, distance, and other metrics
    """
    scanner = get_scanner(mock_mode=mock_mode)
    return scanner.get_candidates(top_n=top_n, threshold=threshold_pct)


def run_scan(save=True, print_results=True, mock_mode=False) -> List[Dict]:
    """
    Run a full scan and optionally save/print results.

    Args:
        save: Save results to CSV
        print_results: Print results to console
        mock_mode: Use mock data instead of live API

    Returns:
        List of all scan results sorted by breakout proximity
    """
    scanner = get_scanner(mock_mode=mock_mode)
    results = scanner.scan_all()

    if save:
        scanner.save_results(results)

    if print_results:
        scanner.print_results(results)

    return results


def get_symbols_near_breakout(threshold_pct=1.0, mock_mode=False) -> List[str]:
    """
    Simple function to get just the symbol names near breakout.

    Returns:
        List of symbol strings within threshold of breakout
    """
    candidates = get_breakout_candidates(top_n=50, threshold_pct=threshold_pct, mock_mode=mock_mode)
    return [c['symbol'] for c in candidates]


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Breakout Scanner")
    parser.add_argument("--mock", action="store_true", help="Use mock data instead of live API")
    args = parser.parse_args()

    mode_str = "MOCK MODE" if args.mock else "LIVE MODE"

    cprint("\n" + "=" * 60, "cyan")
    cprint("  BREAKOUT SCANNER", "cyan")
    cprint(f"  Scanning top 50 liquid stocks for breakout setups ({mode_str})", "cyan")
    cprint("=" * 60, "cyan")

    try:
        results = run_scan(save=True, print_results=True, mock_mode=args.mock)

        # Show candidates that would be picked
        cprint("\n" + "=" * 60, "yellow")
        cprint("  TOP CANDIDATES (within 1% of breakout)", "yellow")
        cprint("=" * 60 + "\n", "yellow")

        candidates = [r for r in results if r['nearest_distance_pct'] <= 1.0]

        if candidates:
            for c in candidates[:10]:
                direction_emoji = "LONG" if c['direction'] == 'LONG' else "SHORT"
                cprint(
                    f"  {c['symbol']:<6} | ${c['current_price']:.2f} | "
                    f"{c['nearest_distance_pct']:.2f}% to {c['nearest_level']} | {direction_emoji}",
                    "green"
                )
        else:
            cprint("  No symbols currently within 1% of breakout level", "yellow")

        cprint(f"\nTotal symbols scanned: {len(results)}", "cyan")

    except Exception as e:
        cprint(f"\nError running scanner: {e}", "red")
        import traceback
        traceback.print_exc()
