"""
Crypto Integration Test

Complete integration test for crypto market data tracking:
- Liquidation tracker
- Market indicators (funding rates)
- Regime detection with crypto risk
- Gap & Go filter status
"""

import sys
import time
import io
import contextlib
from datetime import datetime

sys.path.append('/Users/josiahgarcia/trading-bot')

from termcolor import cprint, colored


@contextlib.contextmanager
def suppress_output():
    """Suppress stdout temporarily."""
    old_stdout = sys.stdout
    sys.stdout = io.StringIO()
    try:
        yield
    finally:
        sys.stdout = old_stdout


def print_header():
    """Print the dashboard header."""
    print()
    print(colored("══════════════════════════════════════════════════════════", "cyan"))
    print(colored("   TRADING BOT - CRYPTO INTEGRATION STATUS", "cyan", attrs=['bold']))
    print(colored("══════════════════════════════════════════════════════════", "cyan"))
    print()


def print_section(title: str):
    """Print a section header."""
    print(colored(f"   {title}", "white", attrs=['bold']))


def format_usd(value: float) -> str:
    """Format USD value with commas."""
    if value >= 1_000_000:
        return f"${value/1_000_000:,.2f}M"
    elif value >= 1_000:
        return f"${value/1_000:,.1f}K"
    else:
        return f"${value:,.0f}"


def main():
    print_header()

    # =========================================================================
    # Step 1: Start all trackers
    # =========================================================================
    print_section("STARTING TRACKERS")
    print("   ├─ Initializing liquidation tracker...")

    try:
        from src.data.liquidation_tracker import (
            start_tracker as start_liquidation_tracker,
            stop_tracker as stop_liquidation_tracker,
            get_liquidation_totals
        )
        start_liquidation_tracker()
        print(colored("   │  ✓ Liquidation tracker started", "green"))
    except Exception as e:
        print(colored(f"   │  ✗ Liquidation tracker failed: {e}", "red"))
        return

    print("   ├─ Initializing market indicators...")

    try:
        from src.data.market_indicators import (
            start_market_indicators,
            stop_market_indicators,
            get_market_snapshot,
            get_funding_rates
        )
        start_market_indicators()
        print(colored("   │  ✓ Market indicators started", "green"))
    except Exception as e:
        print(colored(f"   │  ✗ Market indicators failed: {e}", "red"))
        return

    print("   └─ Importing regime agent...")

    try:
        from src.agents.regime_agent import (
            get_crypto_risk_level,
            get_full_regime_status
        )
        print(colored("      ✓ Regime agent imported", "green"))
    except Exception as e:
        print(colored(f"      ✗ Regime agent failed: {e}", "red"))
        return

    # =========================================================================
    # Step 2: Wait 30 seconds to collect data
    # =========================================================================
    print()
    print_section("COLLECTING DATA")
    print("   └─ Waiting 30 seconds for data collection...")
    print()

    collection_time = 30
    time.sleep(collection_time)

    print(colored("   └─ Data collection complete!", "green"))
    print()

    # =========================================================================
    # Step 3: Print formatted dashboard
    # =========================================================================
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(colored("══════════════════════════════════════════════════════════", "cyan"))
    print(colored(f"   DASHBOARD - {timestamp}", "cyan", attrs=['bold']))
    print(colored("══════════════════════════════════════════════════════════", "cyan"))
    print()

    # -------------------------------------------------------------------------
    # Liquidation Data
    # -------------------------------------------------------------------------
    print_section("LIQUIDATION DATA (Last 30 seconds of collection)")

    try:
        liq_5m = get_liquidation_totals('5 minutes')
        liq_15m = get_liquidation_totals('15 minutes')
        liq_30m = get_liquidation_totals('30 minutes')

        print(f"   ├─ 5 min:  {format_usd(liq_5m['total_usd']):>10} total "
              f"({format_usd(liq_5m['long_usd']):>8} long / {format_usd(liq_5m['short_usd']):>8} short)")
        print(f"   ├─ 15 min: {format_usd(liq_15m['total_usd']):>10} total "
              f"({format_usd(liq_15m['long_usd']):>8} long / {format_usd(liq_15m['short_usd']):>8} short)")
        print(f"   └─ 30 min: {format_usd(liq_30m['total_usd']):>10} total "
              f"({format_usd(liq_30m['long_usd']):>8} long / {format_usd(liq_30m['short_usd']):>8} short)")
    except Exception as e:
        print(colored(f"   └─ Error getting liquidation data: {e}", "red"))

    print()

    # -------------------------------------------------------------------------
    # Funding Rates
    # -------------------------------------------------------------------------
    print_section("FUNDING RATES")

    try:
        # Get funding rates directly - returns flat structure
        funding = get_funding_rates()
        btc_yearly = funding.get('btc_yearly', 0)
        eth_yearly = funding.get('eth_yearly', 0)

        # Get market snapshot for sentiment
        snapshot = get_market_snapshot()
        sentiment = snapshot.get('overall_sentiment', 'NEUTRAL')

        # Color for sentiment
        if 'EXTREME' in sentiment and 'GREED' in sentiment:
            sent_color = "red"
        elif 'GREED' in sentiment:
            sent_color = "yellow"
        elif 'EXTREME' in sentiment and 'FEAR' in sentiment:
            sent_color = "blue"
        elif 'FEAR' in sentiment:
            sent_color = "cyan"
        else:
            sent_color = "white"

        btc_color = "green" if btc_yearly > 0 else "red"
        eth_color = "green" if eth_yearly > 0 else "red"

        print(f"   ├─ BTC: {colored(f'{btc_yearly:+.2f}% yearly', btc_color)}")
        print(f"   ├─ ETH: {colored(f'{eth_yearly:+.2f}% yearly', eth_color)}")
        print(f"   └─ Sentiment: {colored(sentiment, sent_color, attrs=['bold'])}")
    except Exception as e:
        print(colored(f"   └─ Error getting funding rates: {e}", "red"))

    print()

    # -------------------------------------------------------------------------
    # Regime Status
    # -------------------------------------------------------------------------
    print_section("REGIME STATUS")

    try:
        full_status = get_full_regime_status('SPY')

        stock_regime = full_status.get('stock_regime', 'UNKNOWN')
        regime_confidence = full_status.get('regime_confidence', 0)

        # Get crypto risk directly from crypto risk level function
        crypto_risk_data = get_crypto_risk_level()
        crypto_risk = crypto_risk_data.get('risk_level', 'UNKNOWN')
        position_mult = crypto_risk_data.get('position_multiplier', 1.0)

        # Use combined multiplier if available
        combined_mult = full_status.get('position_multiplier', position_mult)
        if isinstance(combined_mult, (int, float)):
            position_mult = combined_mult

        # Color coding for regime
        if stock_regime == 'BULL':
            regime_color = "green"
        elif stock_regime == 'BEAR':
            regime_color = "red"
        elif stock_regime == 'HIGH_VOL':
            regime_color = "yellow"
        else:
            regime_color = "cyan"

        # Color coding for crypto risk
        if crypto_risk == 'HIGH':
            risk_color = "red"
        elif crypto_risk == 'ELEVATED':
            risk_color = "yellow"
        else:
            risk_color = "green"

        # Color coding for position multiplier
        if isinstance(position_mult, (int, float)):
            mult_pct = f"{position_mult:.0%}"
            if position_mult >= 1.0:
                mult_color = "green"
            elif position_mult >= 0.5:
                mult_color = "yellow"
            else:
                mult_color = "red"
        else:
            mult_pct = str(position_mult)
            mult_color = "white"

        print(f"   ├─ Stock Regime: {colored(stock_regime, regime_color, attrs=['bold'])} "
              f"({regime_confidence}% confidence)")
        print(f"   ├─ Crypto Risk: {colored(crypto_risk, risk_color, attrs=['bold'])}")
        print(f"   └─ Position Multiplier: {colored(mult_pct, mult_color, attrs=['bold'])}")
    except Exception as e:
        print(colored(f"   └─ Error getting regime status: {e}", "red"))

    print()

    # -------------------------------------------------------------------------
    # Gap & Go Filter Status
    # -------------------------------------------------------------------------
    print_section("GAP & GO FILTER STATUS")

    try:
        # Get regime rules
        stock_regime = full_status.get('stock_regime', 'RANGE')
        crypto_risk = full_status.get('crypto_risk', 'NORMAL')

        # Determine allowed trade directions based on regime
        REGIME_RULES = {
            'BULL': {'allowed': ['UP'], 'desc': 'BULL market - only gap UPs'},
            'BEAR': {'allowed': ['DOWN'], 'desc': 'BEAR market - only gap DOWNs'},
            'RANGE': {'allowed': ['UP', 'DOWN'], 'desc': 'RANGE market - both directions'},
            'HIGH_VOL': {'allowed': ['UP', 'DOWN'], 'desc': 'HIGH_VOL - both with reduced size'},
        }

        rules = REGIME_RULES.get(stock_regime, REGIME_RULES['RANGE'])

        allow_bullish = 'UP' in rules['allowed'] and crypto_risk != 'HIGH'
        allow_bearish = 'DOWN' in rules['allowed'] and crypto_risk != 'HIGH'

        # Build reason
        reasons = []
        if stock_regime == 'BULL':
            reasons.append("BULL regime favors gap UPs only")
        elif stock_regime == 'BEAR':
            reasons.append("BEAR regime favors gap DOWNs only")
        elif stock_regime == 'HIGH_VOL':
            reasons.append("HIGH_VOL reduces position 50%")
        else:
            reasons.append("RANGE allows both directions")

        if crypto_risk == 'HIGH':
            reasons.append("HIGH crypto risk blocks all trades")
            allow_bullish = False
            allow_bearish = False
        elif crypto_risk == 'ELEVATED':
            reasons.append("ELEVATED crypto risk reduces position")

        reason = " | ".join(reasons)

        bull_status = colored("YES", "green", attrs=['bold']) if allow_bullish else colored("NO", "red", attrs=['bold'])
        bear_status = colored("YES", "green", attrs=['bold']) if allow_bearish else colored("NO", "red", attrs=['bold'])

        print(f"   ├─ Allow Bullish Trades: {bull_status}")
        print(f"   ├─ Allow Bearish Trades: {bear_status}")
        print(f"   └─ Reason: {reason}")
    except Exception as e:
        print(colored(f"   └─ Error getting filter status: {e}", "red"))

    print()
    print(colored("══════════════════════════════════════════════════════════", "cyan"))
    print()

    # =========================================================================
    # Step 4: Stop all trackers cleanly
    # =========================================================================
    print_section("STOPPING TRACKERS")

    try:
        stop_liquidation_tracker()
        print(colored("   ├─ ✓ Liquidation tracker stopped", "green"))
    except Exception as e:
        print(colored(f"   ├─ ✗ Error stopping liquidation tracker: {e}", "yellow"))

    try:
        stop_market_indicators()
        print(colored("   └─ ✓ Market indicators stopped", "green"))
    except Exception as e:
        print(colored(f"   └─ ✗ Error stopping market indicators: {e}", "yellow"))

    print()
    print(colored("   Integration test complete!", "green", attrs=['bold']))
    print()


if __name__ == "__main__":
    main()
