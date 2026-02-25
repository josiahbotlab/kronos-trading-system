"""
Polymarket Market Scanner

Scans active markets for mean reversion opportunities.
Uses the sampling-markets endpoint which returns active markets with liquidity.

Z-score calculation:
- Measures price deviation from "fair value" (0.50 for binary markets)
- Uses estimated standard deviation of ~0.15 for typical market price distribution
- Higher z-score = more extreme price, potential reversion opportunity

Usage:
    python -m polymarket_mr_bot.market_scanner
    python -m polymarket_mr_bot.market_scanner --top 20 --threshold 1.5
"""

import argparse
import sys
from datetime import datetime
from typing import Dict, List, Optional, Tuple

import requests
from termcolor import cprint

from .config import POLYMARKET_HOST, MIN_PRICE, MAX_PRICE


def fetch_sampling_markets(limit: int = 100) -> List[Dict]:
    """
    Fetch active markets from sampling-markets endpoint.
    These are markets with liquidity and order books enabled.
    """
    all_markets = []
    next_cursor = None

    while len(all_markets) < limit:
        try:
            url = f"{POLYMARKET_HOST}/sampling-markets"
            params = {}
            if next_cursor:
                params['next_cursor'] = next_cursor

            resp = requests.get(url, params=params, timeout=15)
            resp.raise_for_status()
            data = resp.json()

            markets = data.get('data', [])
            if not markets:
                break

            # Filter for active, non-closed markets accepting orders
            active = [
                m for m in markets
                if m.get('active')
                and not m.get('closed')
                and m.get('accepting_orders')
                and m.get('enable_order_book')
            ]
            all_markets.extend(active)

            next_cursor = data.get('next_cursor')
            if not next_cursor:
                break

        except Exception as e:
            cprint(f"Error fetching markets: {e}", "red")
            break

    return all_markets[:limit]


def calculate_deviation_metrics(price: float) -> Tuple[float, str, float]:
    """
    Calculate deviation metrics for a market price.

    For binary prediction markets:
    - 0.50 represents maximum uncertainty (fair value baseline)
    - Prices far from 0.50 indicate stronger directional conviction
    - Mean reversion assumes extreme prices will revert toward 0.50

    Returns:
        (z_score, direction, deviation_pct)
    """
    if price <= 0 or price >= 1:
        return 0, 'INVALID', 0

    # Distance from fair value (0.50)
    deviation = price - 0.50
    deviation_pct = abs(deviation) * 100

    # Normalize to approximate z-score
    # Using 0.15 as typical "1 std dev" for active prediction markets
    # (empirically, prices cluster around extremes, std dev ~0.15-0.20)
    estimated_std = 0.15
    z_score = abs(deviation) / estimated_std

    # Determine direction
    if deviation < -0.03:  # Price below 0.47
        direction = 'OVERSOLD'   # YES token cheap, might revert up
    elif deviation > 0.03:  # Price above 0.53
        direction = 'OVERBOUGHT' # YES token expensive, might revert down
    else:
        direction = 'NEUTRAL'

    return z_score, direction, deviation_pct


def scan_markets(top_n: int = 20, zscore_threshold: float = 1.5) -> List[Dict]:
    """
    Scan top N markets for mean reversion opportunities.

    Args:
        top_n: Number of markets to analyze
        zscore_threshold: Minimum z-score to flag as trading signal

    Returns:
        List of markets with signals exceeding threshold
    """
    cprint("\n" + "=" * 90, "cyan")
    cprint("  POLYMARKET MEAN REVERSION SCANNER", "cyan", attrs=['bold'])
    cprint(f"  Analyzing top {top_n} active markets | Signal threshold: {zscore_threshold}σ", "cyan")
    cprint("=" * 90, "cyan")

    # Fetch markets
    cprint("\n  Fetching active markets from Polymarket...", "white")
    markets = fetch_sampling_markets(limit=top_n * 2)

    if not markets:
        cprint("  No active markets found", "yellow")
        return []

    cprint(f"  Found {len(markets)} active markets with liquidity\n", "green")

    # Analyze each market
    results = []
    signals = []

    for market in markets:
        tokens = market.get('tokens', [])
        if len(tokens) < 2:
            continue

        # Get YES token (first token)
        yes_token = tokens[0]
        no_token = tokens[1]

        yes_price = yes_token.get('price')
        if yes_price is None:
            continue

        yes_price = float(yes_price)

        # Skip prices at extremes (likely near resolution)
        if yes_price < MIN_PRICE or yes_price > MAX_PRICE:
            continue

        # Calculate deviation metrics
        z_score, direction, deviation_pct = calculate_deviation_metrics(yes_price)

        result = {
            'question': market.get('question', 'Unknown')[:55],
            'condition_id': market.get('condition_id', ''),
            'market_slug': market.get('market_slug', ''),
            'token_id_yes': yes_token.get('token_id', ''),
            'token_id_no': no_token.get('token_id', ''),
            'yes_price': yes_price,
            'no_price': float(no_token.get('price', 1 - yes_price)),
            'z_score': z_score,
            'direction': direction,
            'deviation_pct': deviation_pct,
            'tags': market.get('tags', []),
            'end_date': market.get('end_date_iso', ''),
            'rewards': market.get('rewards', {}),
        }

        results.append(result)

        if z_score >= zscore_threshold and direction != 'NEUTRAL':
            signals.append(result)

        if len(results) >= top_n:
            break

    # Sort by z-score (most extreme first)
    results.sort(key=lambda x: x['z_score'], reverse=True)
    signals.sort(key=lambda x: x['z_score'], reverse=True)

    # Display all analyzed markets
    cprint("-" * 90, "cyan")
    cprint(f"  {'#':<3} {'YES':>6} {'NO':>6} {'Z-SCORE':>8} {'DIRECTION':>11} {'DEV%':>6}  {'MARKET QUESTION':<45}", "cyan")
    cprint("-" * 90, "cyan")

    for i, r in enumerate(results, 1):
        z = r['z_score']
        direction = r['direction']

        # Color and marker based on signal strength
        if z >= zscore_threshold and direction != 'NEUTRAL':
            if direction == 'OVERSOLD':
                color = "green"
                marker = "▲▲▲"  # Buy signal
            else:
                color = "red"
                marker = "▼▼▼"  # Sell signal
        elif z >= 1.0:
            color = "yellow"
            marker = " ◆ "
        else:
            color = "white"
            marker = "   "

        question = r['question'][:42] + "..." if len(r['question']) > 42 else r['question']

        cprint(
            f"{marker}{i:<3} {r['yes_price']*100:>5.1f}c {r['no_price']*100:>5.1f}c {z:>7.2f}σ {direction:>11} {r['deviation_pct']:>5.1f}%  {question}",
            color
        )

    # Summary
    cprint("\n" + "=" * 90, "cyan")
    cprint(f"  SCAN COMPLETE: {len(results)} markets analyzed", "cyan", attrs=['bold'])
    cprint("=" * 90, "cyan")

    # Display signals
    if signals:
        cprint(f"\n  ⚡ SIGNALS FOUND: {len(signals)} markets exceed {zscore_threshold}σ threshold\n", "green", attrs=['bold'])

        for i, s in enumerate(signals, 1):
            direction = s['direction']

            if direction == 'OVERSOLD':
                action = "BUY YES (price too low, expect reversion UP)"
                color = "green"
                icon = "📈"
            else:
                action = "SELL YES / BUY NO (price too high, expect reversion DOWN)"
                color = "red"
                icon = "📉"

            cprint(f"  {icon} Signal {i}: {direction} @ {s['yes_price']*100:.1f}c (z={s['z_score']:.2f}σ, dev={s['deviation_pct']:.1f}%)", color, attrs=['bold'])
            cprint(f"     Question: {s['question']}", "white")
            cprint(f"     Action: {action}", color)
            cprint(f"     Market: {s['market_slug'][:60]}", "white")
            cprint(f"     YES Token: {s['token_id_yes'][:50]}...", "white")
            cprint(f"     NO Token:  {s['token_id_no'][:50]}...", "white")
            if s['end_date']:
                cprint(f"     Expires: {s['end_date'][:10]}", "white")
            if s['tags']:
                cprint(f"     Tags: {', '.join(s['tags'][:5])}", "white")
            cprint("", "white")

        # Trading guidance
        cprint("  " + "-" * 86, "cyan")
        cprint("  MEAN REVERSION STRATEGY:", "cyan", attrs=['bold'])
        cprint("  • OVERSOLD (green): YES price is low → BUY YES, expect price to rise toward 50c", "green")
        cprint("  • OVERBOUGHT (red): YES price is high → SELL YES or BUY NO, expect price to fall toward 50c", "red")
        cprint("  • Higher z-score = stronger signal, but also higher risk if market has fundamental reason", "yellow")
        cprint("  " + "-" * 86 + "\n", "cyan")

    else:
        cprint(f"\n  No signals found exceeding {zscore_threshold}σ threshold", "yellow")
        cprint("  Try lowering the threshold (--threshold 1.0) or check back during higher volatility\n", "white")

    return signals


def main():
    parser = argparse.ArgumentParser(description="Polymarket Mean Reversion Scanner")
    parser.add_argument("--top", type=int, default=20, help="Number of markets to scan (default: 20)")
    parser.add_argument("--threshold", type=float, default=1.5, help="Z-score threshold for signals (default: 1.5)")

    args = parser.parse_args()

    signals = scan_markets(top_n=args.top, zscore_threshold=args.threshold)

    return 0 if signals else 1


if __name__ == "__main__":
    sys.exit(main())
