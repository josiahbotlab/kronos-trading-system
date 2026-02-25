"""
Gap and Go Pre-Market Scanner

Focused scanner for tradeable Gap and Go symbols based on optimization results.
Each symbol has its own parameters from backtesting optimization.

Usage:
    from src.scanner.gap_scanner import scan_premarket, get_gap_candidates, get_best_candidate

    # Run a scan
    candidates = scan_premarket()

    # Get best opportunity
    best = get_best_candidate()

    # CLI usage
    python src/scanner/gap_scanner.py          # Run live scan
    python src/scanner/gap_scanner.py --mock   # Use mock data for testing
"""

import json
import os
import sys
from datetime import datetime, time, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import pytz

from dotenv import load_dotenv
from termcolor import cprint

sys.path.append('/Users/josiahgarcia/trading-bot')
load_dotenv()

# Alpaca imports
try:
    import alpaca_trade_api as tradeapi
    ALPACA_AVAILABLE = True
except ImportError:
    ALPACA_AVAILABLE = False

# ============================================================================
# CONFIGURATION
# ============================================================================

# Paths
PROJECT_ROOT = Path(__file__).parent.parent.parent
CONFIG_PATH = PROJECT_ROOT / 'csvs' / 'tradeable_symbols.json'
SCAN_OUTPUT_DIR = PROJECT_ROOT / 'csvs' / 'gap_scans'

# Alpaca credentials
ALPACA_API_KEY = os.getenv("ALPACA_API_KEY")
ALPACA_SECRET_KEY = os.getenv("ALPACA_SECRET_KEY")
ALPACA_PAPER = os.getenv("ALPACA_PAPER", "true").lower() == "true"

# Timezone
ET = pytz.timezone('America/New_York')

# Fallback hardcoded config (if JSON not found)
FALLBACK_SYMBOL_PARAMS = {
    'AMD': {
        'gap_threshold': 2.5,
        'volume_multiplier': 1.0,
        'profit_target': 8.0,
        'stop_loss': 15.0
    },
    'META': {
        'gap_threshold': 1.0,
        'volume_multiplier': 1.0,
        'profit_target': 5.0,
        'stop_loss': 7.0
    }
}

# ============================================================================
# SYMBOL PARAMETERS LOADER
# ============================================================================

def load_symbol_params() -> Dict[str, Dict]:
    """
    Load tradeable symbol parameters from JSON config.
    Falls back to hardcoded config if file not found.
    """
    if CONFIG_PATH.exists():
        try:
            with open(CONFIG_PATH, 'r') as f:
                data = json.load(f)

            # Extract symbol parameters
            params = {}
            for symbol, info in data.get('symbols', {}).items():
                params[symbol] = info.get('parameters', {})

            # Add AMD from separate optimization if not in multi-symbol results
            if 'AMD' not in params:
                amd_config = PROJECT_ROOT / 'csvs' / 'gap_and_go_best_params.json'
                if amd_config.exists():
                    with open(amd_config, 'r') as f:
                        amd_data = json.load(f)
                    if amd_data.get('symbol') == 'AMD':
                        params['AMD'] = amd_data.get('parameters', FALLBACK_SYMBOL_PARAMS['AMD'])

            if params:
                return params

        except Exception as e:
            cprint(f"Error loading config: {e}, using fallback", "yellow")

    return FALLBACK_SYMBOL_PARAMS.copy()


# Global cached parameters
_SYMBOL_PARAMS = None

def get_symbol_params() -> Dict[str, Dict]:
    """Get cached symbol parameters."""
    global _SYMBOL_PARAMS
    if _SYMBOL_PARAMS is None:
        _SYMBOL_PARAMS = load_symbol_params()
    return _SYMBOL_PARAMS


def get_tradeable_symbols() -> List[str]:
    """Get list of tradeable symbols."""
    return list(get_symbol_params().keys())


# ============================================================================
# MARKET HOURS UTILITIES
# ============================================================================

def get_et_now() -> datetime:
    """Get current time in Eastern timezone."""
    return datetime.now(ET)


def is_weekend() -> bool:
    """Check if today is weekend."""
    now = get_et_now()
    return now.weekday() >= 5  # Saturday = 5, Sunday = 6


def is_market_holiday() -> bool:
    """
    Check if today is a US market holiday.
    Basic check - for production, use a proper calendar API.
    """
    now = get_et_now()

    # 2024/2025/2026 US market holidays (simplified)
    holidays = [
        (1, 1),   # New Year's Day
        (1, 20),  # MLK Day (3rd Monday January) - approximate
        (2, 17),  # Presidents Day (3rd Monday February) - approximate
        (4, 18),  # Good Friday - varies
        (5, 26),  # Memorial Day (last Monday May) - approximate
        (6, 19),  # Juneteenth
        (7, 4),   # Independence Day
        (9, 1),   # Labor Day (1st Monday September) - approximate
        (11, 27), # Thanksgiving (4th Thursday November) - approximate
        (12, 25), # Christmas
    ]

    return (now.month, now.day) in holidays


def is_premarket() -> bool:
    """
    Check if currently in pre-market hours (4:00 AM - 9:30 AM ET).
    """
    now = get_et_now()
    current_time = now.time()

    premarket_start = time(4, 0)   # 4:00 AM ET
    premarket_end = time(9, 30)    # 9:30 AM ET

    return premarket_start <= current_time < premarket_end


def is_market_open() -> bool:
    """
    Check if regular market hours (9:30 AM - 4:00 PM ET).
    """
    now = get_et_now()
    current_time = now.time()

    market_open = time(9, 30)   # 9:30 AM ET
    market_close = time(16, 0)  # 4:00 PM ET

    return market_open <= current_time < market_close


def is_after_hours() -> bool:
    """
    Check if in after-hours (4:00 PM - 8:00 PM ET).
    """
    now = get_et_now()
    current_time = now.time()

    ah_start = time(16, 0)  # 4:00 PM ET
    ah_end = time(20, 0)    # 8:00 PM ET

    return ah_start <= current_time < ah_end


def get_market_status() -> str:
    """Get current market status string."""
    if is_weekend():
        return "WEEKEND"
    if is_market_holiday():
        return "HOLIDAY"
    if is_premarket():
        return "PRE-MARKET"
    if is_market_open():
        return "MARKET OPEN"
    if is_after_hours():
        return "AFTER-HOURS"
    return "MARKET CLOSED"


def can_scan() -> Tuple[bool, str]:
    """
    Check if scanning is meaningful right now.
    Returns (can_scan, reason).
    """
    if is_weekend():
        return False, "Weekend - market closed"
    if is_market_holiday():
        return False, "Holiday - market closed"

    # We can scan during pre-market, market hours, or after-hours
    # Gap data will be available once market opens
    return True, get_market_status()


# ============================================================================
# GAP SCANNER CLASS
# ============================================================================

class GapScanner:
    """
    Gap and Go Pre-Market Scanner.

    Scans tradeable symbols for gap opportunities based on
    per-symbol optimized parameters.
    """

    def __init__(self, mock_mode: bool = False):
        self.mock_mode = mock_mode
        self.api = None
        self.symbol_params = get_symbol_params()

        if not mock_mode:
            if not ALPACA_AVAILABLE:
                raise RuntimeError("alpaca-trade-api package not installed")

            if not ALPACA_API_KEY or not ALPACA_SECRET_KEY:
                raise RuntimeError("ALPACA_API_KEY and ALPACA_SECRET_KEY required in .env")

            base_url = "https://paper-api.alpaca.markets" if ALPACA_PAPER else "https://api.alpaca.markets"
            self.api = tradeapi.REST(ALPACA_API_KEY, ALPACA_SECRET_KEY, base_url, api_version='v2')

        # Ensure output directory exists
        SCAN_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

        # Cache for scan results
        self._last_scan_results: List[Dict] = []
        self._last_scan_time: Optional[datetime] = None

    def get_previous_close(self, symbol: str) -> Optional[float]:
        """
        Get previous day's closing price for a symbol.
        """
        if self.mock_mode:
            return self._get_mock_prev_close(symbol)

        try:
            # Get daily bars - last 2 trading days
            # Use date-only format for Alpaca API
            # Use IEX feed for paper accounts (SIP requires paid subscription)
            end = datetime.now()
            start = end - timedelta(days=5)  # Buffer for weekends

            bars = self.api.get_bars(
                symbol,
                tradeapi.TimeFrame.Day,
                start=start.strftime('%Y-%m-%d'),
                end=end.strftime('%Y-%m-%d'),
                limit=5,
                feed='iex'  # Use IEX for paper accounts
            ).df

            if len(bars) < 1:
                return None

            # Get the previous day's close (second to last bar if today has data)
            # If only one bar, use that
            if len(bars) >= 2:
                return float(bars.iloc[-2]['close'])
            else:
                return float(bars.iloc[-1]['close'])

        except Exception as e:
            cprint(f"Error getting prev close for {symbol}: {e}", "red")
            return None

    def get_current_price(self, symbol: str) -> Optional[float]:
        """
        Get current/latest price for a symbol.
        Works during pre-market, regular hours, and after-hours.
        """
        if self.mock_mode:
            return self._get_mock_current_price(symbol)

        try:
            # Try to get latest trade using IEX feed
            trade = self.api.get_latest_trade(symbol, feed='iex')
            return float(trade.price)
        except Exception as e:
            # Fall back to latest quote
            try:
                quote = self.api.get_latest_quote(symbol, feed='iex')
                # Use midpoint of bid/ask
                if quote.bid_price and quote.ask_price:
                    return (float(quote.bid_price) + float(quote.ask_price)) / 2
                return float(quote.ask_price) if quote.ask_price else None
            except Exception as e2:
                cprint(f"Error getting price for {symbol}: {e2}", "red")
                return None

    def get_premarket_volume(self, symbol: str) -> Optional[float]:
        """
        Get pre-market volume if available.
        Note: Volume filter is relaxed (1.0x) based on optimization.
        """
        if self.mock_mode:
            return 1000000  # Mock volume

        try:
            # Get snapshot which includes pre-market data (use IEX feed)
            snapshot = self.api.get_snapshot(symbol, feed='iex')
            if snapshot and snapshot.minute_bar:
                return float(snapshot.minute_bar.volume)
            return None
        except:
            return None

    def calculate_gap(self, current_price: float, prev_close: float) -> Tuple[float, str]:
        """
        Calculate gap percentage and direction.

        Returns:
            (gap_pct, direction) where direction is 'UP' or 'DOWN'
        """
        gap_pct = ((current_price - prev_close) / prev_close) * 100
        direction = 'UP' if gap_pct > 0 else 'DOWN'
        return gap_pct, direction

    def calculate_targets(self, entry_price: float, direction: str,
                         tp_pct: float, sl_pct: float) -> Tuple[float, float]:
        """
        Calculate take profit and stop loss prices.

        For Gap and Go:
        - Long (gap UP): TP above entry, SL below entry
        - Short (gap DOWN): TP below entry, SL above entry
        """
        if direction == 'UP':
            # Long position
            tp_price = entry_price * (1 + tp_pct / 100)
            sl_price = entry_price * (1 - sl_pct / 100)
        else:
            # Short position
            tp_price = entry_price * (1 - tp_pct / 100)
            sl_price = entry_price * (1 + sl_pct / 100)

        return round(tp_price, 2), round(sl_price, 2)

    def scan_symbol(self, symbol: str) -> Optional[Dict]:
        """
        Scan a single symbol for gap opportunity.

        Returns candidate dict if gap meets symbol's threshold, None otherwise.
        """
        params = self.symbol_params.get(symbol)
        if not params:
            return None

        gap_threshold = params.get('gap_threshold', 2.0)
        tp_pct = params.get('profit_target', 5.0)
        sl_pct = params.get('stop_loss', 10.0)

        # Get prices
        prev_close = self.get_previous_close(symbol)
        current_price = self.get_current_price(symbol)

        if prev_close is None or current_price is None:
            return None

        # Calculate gap
        gap_pct, direction = self.calculate_gap(current_price, prev_close)
        abs_gap = abs(gap_pct)

        # Check if gap meets this symbol's threshold
        meets_threshold = abs_gap >= gap_threshold

        # Calculate entry, TP, SL
        entry_price = current_price
        tp_price, sl_price = self.calculate_targets(entry_price, direction, tp_pct, sl_pct)

        # Get pre-market volume (informational)
        pm_volume = self.get_premarket_volume(symbol)

        return {
            'symbol': symbol,
            'prev_close': round(prev_close, 2),
            'current_price': round(current_price, 2),
            'gap_pct': round(gap_pct, 2),
            'abs_gap_pct': round(abs_gap, 2),
            'gap_direction': direction,
            'gap_threshold': gap_threshold,
            'meets_threshold': meets_threshold,
            'entry_price': round(entry_price, 2),
            'tp_price': tp_price,
            'sl_price': sl_price,
            'tp_pct': tp_pct,
            'sl_pct': sl_pct,
            'pm_volume': pm_volume,
            'scan_time': datetime.now().isoformat()
        }

    def scan_all(self) -> List[Dict]:
        """
        Scan all tradeable symbols.

        Returns list of all results (both meeting and not meeting threshold).
        """
        results = []
        symbols = list(self.symbol_params.keys())

        for symbol in symbols:
            result = self.scan_symbol(symbol)
            if result:
                results.append(result)

        # Sort by absolute gap size (largest first)
        results.sort(key=lambda x: x['abs_gap_pct'], reverse=True)

        # Cache results
        self._last_scan_results = results
        self._last_scan_time = datetime.now()

        return results

    def get_candidates(self) -> List[Dict]:
        """
        Get candidates that meet their respective thresholds.
        """
        results = self.scan_all()
        return [r for r in results if r['meets_threshold']]

    def get_best_candidate(self) -> Optional[Dict]:
        """
        Get single best opportunity (largest gap that meets threshold).
        """
        candidates = self.get_candidates()
        return candidates[0] if candidates else None

    def save_scan_results(self, results: List[Dict], filepath: Optional[Path] = None):
        """
        Save scan results to CSV.
        """
        if not results:
            return None

        if filepath is None:
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            filepath = SCAN_OUTPUT_DIR / f'scan_{timestamp}.csv'

        import csv

        with open(filepath, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([
                'symbol', 'prev_close', 'current_price', 'gap_pct',
                'gap_direction', 'gap_threshold', 'meets_threshold',
                'entry_price', 'tp_price', 'sl_price', 'tp_pct', 'sl_pct',
                'pm_volume', 'scan_time'
            ])

            for r in results:
                writer.writerow([
                    r['symbol'],
                    r['prev_close'],
                    r['current_price'],
                    r['gap_pct'],
                    r['gap_direction'],
                    r['gap_threshold'],
                    r['meets_threshold'],
                    r['entry_price'],
                    r['tp_price'],
                    r['sl_price'],
                    r['tp_pct'],
                    r['sl_pct'],
                    r['pm_volume'] or '',
                    r['scan_time']
                ])

        return filepath

    # Mock data methods for testing
    def _get_mock_prev_close(self, symbol: str) -> float:
        """Get mock previous close price."""
        mock_prices = {
            'AMD': 125.00,
            'META': 590.00,
            'NVDA': 140.00,
            'TSLA': 410.00,
        }
        return mock_prices.get(symbol, 100.00)

    def _get_mock_current_price(self, symbol: str) -> float:
        """
        Get mock current price with simulated gap.
        Creates realistic gap scenarios for testing.
        """
        import random
        prev = self._get_mock_prev_close(symbol)

        # Simulate different gap scenarios
        if symbol == 'AMD':
            # AMD needs 2.5% gap - simulate a 3% gap
            gap_pct = random.uniform(2.8, 3.5)
        elif symbol == 'META':
            # META needs 1.0% gap - simulate a 1.5% gap
            gap_pct = random.uniform(1.2, 2.0)
        else:
            # Random small gap for other symbols
            gap_pct = random.uniform(-1.0, 1.0)

        # Randomly make some gaps negative (down)
        if random.random() < 0.3:
            gap_pct = -gap_pct

        return round(prev * (1 + gap_pct / 100), 2)


# ============================================================================
# MODULE-LEVEL FUNCTIONS
# ============================================================================

# Global scanner instance
_scanner: Optional[GapScanner] = None


def get_scanner(mock_mode: bool = False) -> GapScanner:
    """Get or create global scanner instance."""
    global _scanner
    if _scanner is None or _scanner.mock_mode != mock_mode:
        _scanner = GapScanner(mock_mode=mock_mode)
    return _scanner


def scan_premarket(save: bool = True, mock_mode: bool = False) -> List[Dict]:
    """
    Run a pre-market gap scan.

    Args:
        save: Save results to CSV
        mock_mode: Use mock data for testing

    Returns:
        List of all scan results
    """
    scanner = get_scanner(mock_mode=mock_mode)

    # Check if we can scan
    can, reason = can_scan()
    if not can and not mock_mode:
        cprint(f"Cannot scan: {reason}", "yellow")
        return []

    results = scanner.scan_all()

    if save and results:
        filepath = scanner.save_scan_results(results)
        if filepath:
            cprint(f"Results saved to: {filepath}", "green")

    return results


def get_gap_candidates(mock_mode: bool = False) -> List[Dict]:
    """
    Get gap candidates that meet their thresholds.

    Returns:
        List of candidates with symbol, gap_pct, entry/tp/sl prices
    """
    scanner = get_scanner(mock_mode=mock_mode)
    return scanner.get_candidates()


def get_best_candidate(mock_mode: bool = False) -> Optional[Dict]:
    """
    Get the single best gap opportunity.

    Returns:
        Best candidate dict or None if no candidates
    """
    scanner = get_scanner(mock_mode=mock_mode)
    return scanner.get_best_candidate()


def get_current_gaps(mock_mode: bool = False) -> Dict[str, Dict]:
    """
    Get current gap status for all tradeable symbols.

    Returns:
        Dict mapping symbol to gap info
    """
    scanner = get_scanner(mock_mode=mock_mode)
    results = scanner.scan_all()
    return {r['symbol']: r for r in results}


# ============================================================================
# DISPLAY FUNCTIONS
# ============================================================================

def print_scan_results(results: List[Dict]):
    """Print formatted scan results."""
    cprint(f"\n{'='*70}", "cyan")
    cprint(f"  GAP AND GO SCANNER - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}", "cyan")
    cprint(f"  Market Status: {get_market_status()}", "cyan")
    cprint(f"{'='*70}\n", "cyan")

    if not results:
        cprint("No symbols scanned", "yellow")
        return

    # Header
    cprint(f"{'Symbol':<8} {'Prev':<10} {'Current':<10} {'Gap%':<10} {'Threshold':<10} {'Signal':<10}", "white")
    cprint("-" * 70, "white")

    for r in results:
        symbol = r['symbol']
        prev = f"${r['prev_close']:.2f}"
        current = f"${r['current_price']:.2f}"
        gap_pct = f"{r['gap_pct']:+.2f}%"
        threshold = f"{r['gap_threshold']:.1f}%"

        if r['meets_threshold']:
            signal = f"TRIGGER ({r['gap_direction']})"
            color = "green"
        else:
            signal = f"Below ({r['gap_direction']})"
            color = "yellow"

        cprint(f"{symbol:<8} {prev:<10} {current:<10} {gap_pct:<10} {threshold:<10} {signal:<10}", color)

    # Candidates section
    candidates = [r for r in results if r['meets_threshold']]

    cprint(f"\n{'-'*70}", "white")
    cprint(f"CANDIDATES ({len(candidates)} of {len(results)} symbols meeting threshold):", "cyan")
    cprint("-" * 70, "white")

    if candidates:
        for c in candidates:
            cprint(f"\n  {c['symbol']} - Gap {c['gap_direction']} {c['abs_gap_pct']:.2f}% (threshold: {c['gap_threshold']:.1f}%)", "green")
            cprint(f"    Entry: ${c['entry_price']:.2f}", "white")
            cprint(f"    Take Profit: ${c['tp_price']:.2f} ({c['tp_pct']:.1f}%)", "green")
            cprint(f"    Stop Loss: ${c['sl_price']:.2f} ({c['sl_pct']:.1f}%)", "red")
    else:
        cprint("  No symbols currently meeting their gap thresholds", "yellow")


def print_symbol_config():
    """Print current symbol configuration."""
    params = get_symbol_params()

    cprint(f"\n{'='*60}", "cyan")
    cprint("  TRADEABLE SYMBOLS CONFIGURATION", "cyan")
    cprint(f"{'='*60}\n", "cyan")

    for symbol, p in params.items():
        cprint(f"{symbol}:", "white")
        cprint(f"  Gap Threshold: {p.get('gap_threshold', 'N/A')}%", "white")
        cprint(f"  Profit Target: {p.get('profit_target', 'N/A')}%", "green")
        cprint(f"  Stop Loss: {p.get('stop_loss', 'N/A')}%", "red")
        cprint("", "white")


# ============================================================================
# MAIN
# ============================================================================

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Gap and Go Pre-Market Scanner")
    parser.add_argument("--mock", action="store_true", help="Use mock data for testing")
    parser.add_argument("--config", action="store_true", help="Show symbol configuration")
    args = parser.parse_args()

    if args.config:
        print_symbol_config()
        sys.exit(0)

    mode_str = "MOCK MODE" if args.mock else "LIVE MODE"

    cprint("\n" + "=" * 60, "cyan")
    cprint("  GAP AND GO PRE-MARKET SCANNER", "cyan")
    cprint(f"  Mode: {mode_str}", "cyan")
    cprint("=" * 60, "cyan")

    # Show configuration
    print_symbol_config()

    # Run scan
    try:
        results = scan_premarket(save=True, mock_mode=args.mock)
        print_scan_results(results)

        # Show best candidate
        best = get_best_candidate(mock_mode=args.mock)
        if best:
            cprint(f"\n{'='*60}", "green")
            cprint(f"  BEST OPPORTUNITY: {best['symbol']}", "green")
            cprint(f"  Gap: {best['gap_pct']:+.2f}% {best['gap_direction']}", "green")
            cprint(f"  Entry: ${best['entry_price']:.2f} | TP: ${best['tp_price']:.2f} | SL: ${best['sl_price']:.2f}", "green")
            cprint(f"{'='*60}", "green")

    except Exception as e:
        cprint(f"\nError running scanner: {e}", "red")
        import traceback
        traceback.print_exc()
