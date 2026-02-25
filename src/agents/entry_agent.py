"""
Entry Confirmation Agent

Moon Dev's philosophy: Entries matter less than exits, but confirmation helps.
- Volume spike confirms breakout strength
- Trend alignment improves win rate
- RSI filter avoids exhausted moves
- Multi-timeframe confluence adds confidence

Confirmation Checks:
1. Volume: Current volume > 1.5x 20-period average
2. Trend: LONG above 50 SMA, SHORT below 50 SMA
3. RSI: No LONG if RSI > 70, no SHORT if RSI < 30
4. Multi-timeframe: 1h breakout aligns with 4h trend (optional)

Usage:
    from src.agents.entry_agent import check_entry, EntryAgent

    # Quick check before entry
    should_enter, confidence, reasons = check_entry(
        'AAPL', 'LONG', 150.0, candles_1h, candles_4h
    )
    if should_enter and confidence >= 60:
        execute_trade()
"""

import csv
from datetime import datetime
from pathlib import Path
from typing import List, Tuple, Optional, Dict, Any

# Entry confirmation thresholds
VOLUME_MULTIPLIER = 1.5      # Volume must be 1.5x average
VOLUME_PERIOD = 20           # 20-period volume average
SMA_PERIOD = 50              # 50-period SMA for trend
RSI_PERIOD = 14              # 14-period RSI
RSI_OVERBOUGHT = 70          # RSI overbought level
RSI_OVERSOLD = 30            # RSI oversold level
MIN_CONFIDENCE = 60          # Minimum confidence to enter

# Confidence weights (must sum to 100)
WEIGHT_VOLUME = 25           # Volume confirmation weight
WEIGHT_TREND = 30            # Trend confirmation weight
WEIGHT_RSI = 25              # RSI filter weight
WEIGHT_MTF = 20              # Multi-timeframe weight

# Paths
CSV_DIR = Path(__file__).parent.parent.parent / 'csvs'
ENTRY_LOG_PATH = CSV_DIR / 'entry_log.csv'


class EntryAgent:
    """
    Entry confirmation agent that validates trade setups.

    Moon Dev says: "A good entry won't save a bad trade, but it helps."
    """

    def __init__(self, enable_mtf=True, min_confidence=MIN_CONFIDENCE):
        self.enable_mtf = enable_mtf
        self.min_confidence = min_confidence

        # Ensure CSV directory exists
        CSV_DIR.mkdir(parents=True, exist_ok=True)
        self._init_csv()

    def _init_csv(self):
        """Initialize entry log CSV with headers if needed."""
        if not ENTRY_LOG_PATH.exists():
            with open(ENTRY_LOG_PATH, 'w', newline='') as f:
                writer = csv.writer(f)
                writer.writerow([
                    'timestamp',
                    'symbol',
                    'direction',
                    'price',
                    'volume_ok',
                    'trend_ok',
                    'rsi_ok',
                    'mtf_ok',
                    'confidence',
                    'should_enter',
                    'reasons'
                ])

    def _log_decision(self, symbol, direction, price, volume_ok, trend_ok,
                      rsi_ok, mtf_ok, confidence, should_enter, reasons):
        """Log an entry decision to CSV."""
        with open(ENTRY_LOG_PATH, 'a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([
                datetime.now().isoformat(),
                symbol,
                direction,
                f"{price:.2f}",
                "YES" if volume_ok else "NO",
                "YES" if trend_ok else "NO",
                "YES" if rsi_ok else "NO",
                "YES" if mtf_ok else "N/A" if mtf_ok is None else "NO",
                confidence,
                "YES" if should_enter else "NO",
                "; ".join(reasons)
            ])

    def calculate_sma(self, candles, period=SMA_PERIOD):
        """Calculate Simple Moving Average from candles."""
        if len(candles) < period:
            # Use available data if not enough periods
            if candles:
                closes = [c['close'] for c in candles]
                return sum(closes) / len(closes)
            return None

        closes = [c['close'] for c in candles[-period:]]
        return sum(closes) / len(closes)

    def calculate_rsi(self, candles, period=RSI_PERIOD):
        """Calculate Relative Strength Index."""
        if len(candles) < period + 1:
            return 50  # Default to neutral if not enough data

        # Calculate price changes
        changes = []
        for i in range(1, len(candles)):
            changes.append(candles[i]['close'] - candles[i-1]['close'])

        if len(changes) < period:
            return 50

        # Use recent changes
        recent_changes = changes[-period:]

        gains = [c for c in recent_changes if c > 0]
        losses = [-c for c in recent_changes if c < 0]

        avg_gain = sum(gains) / period if gains else 0
        avg_loss = sum(losses) / period if losses else 0

        if avg_loss == 0:
            return 100 if avg_gain > 0 else 50

        rs = avg_gain / avg_loss
        rsi = 100 - (100 / (1 + rs))

        return rsi

    def calculate_average_volume(self, candles, period=VOLUME_PERIOD):
        """Calculate average volume over period."""
        if len(candles) < period:
            if candles:
                volumes = [c['volume'] for c in candles]
                return sum(volumes) / len(volumes)
            return None

        volumes = [c['volume'] for c in candles[-period:]]
        return sum(volumes) / len(volumes)

    def get_current_volume(self, candles):
        """Get most recent volume."""
        if candles:
            return candles[-1]['volume']
        return None

    def check_volume_confirmation(self, candles, symbol):
        """
        Check if current volume confirms the breakout.
        Volume should be > 1.5x 20-period average.
        """
        avg_volume = self.calculate_average_volume(candles, VOLUME_PERIOD)
        current_volume = self.get_current_volume(candles)

        if avg_volume is None or current_volume is None:
            return False, "No volume data available"

        threshold = avg_volume * VOLUME_MULTIPLIER

        if current_volume >= threshold:
            ratio = current_volume / avg_volume
            return True, f"Volume {ratio:.1f}x avg (${current_volume:,.0f} >= ${threshold:,.0f})"
        else:
            ratio = current_volume / avg_volume
            return False, f"Volume weak {ratio:.1f}x avg (${current_volume:,.0f} < ${threshold:,.0f})"

    def check_trend_confirmation(self, candles, current_price, direction):
        """
        Check if trade aligns with trend.
        LONG: Price above 50 SMA
        SHORT: Price below 50 SMA
        """
        sma = self.calculate_sma(candles, SMA_PERIOD)

        if sma is None:
            return False, "No SMA data available"

        if direction == 'LONG':
            if current_price > sma:
                diff_pct = ((current_price - sma) / sma) * 100
                return True, f"Price ${current_price:.2f} above SMA ${sma:.2f} (+{diff_pct:.1f}%)"
            else:
                diff_pct = ((sma - current_price) / sma) * 100
                return False, f"Price ${current_price:.2f} below SMA ${sma:.2f} (-{diff_pct:.1f}%)"
        else:  # SHORT
            if current_price < sma:
                diff_pct = ((sma - current_price) / sma) * 100
                return True, f"Price ${current_price:.2f} below SMA ${sma:.2f} (-{diff_pct:.1f}%)"
            else:
                diff_pct = ((current_price - sma) / sma) * 100
                return False, f"Price ${current_price:.2f} above SMA ${sma:.2f} (+{diff_pct:.1f}%)"

    def check_rsi_filter(self, candles, direction):
        """
        Check RSI filter.
        No LONG if RSI > 70 (overbought)
        No SHORT if RSI < 30 (oversold)
        """
        rsi = self.calculate_rsi(candles, RSI_PERIOD)

        if direction == 'LONG':
            if rsi > RSI_OVERBOUGHT:
                return False, f"RSI {rsi:.1f} overbought (>{RSI_OVERBOUGHT}) - avoid LONG"
            elif rsi < RSI_OVERSOLD:
                return True, f"RSI {rsi:.1f} oversold (<{RSI_OVERSOLD}) - great LONG setup"
            else:
                return True, f"RSI {rsi:.1f} neutral - OK for LONG"
        else:  # SHORT
            if rsi < RSI_OVERSOLD:
                return False, f"RSI {rsi:.1f} oversold (<{RSI_OVERSOLD}) - avoid SHORT"
            elif rsi > RSI_OVERBOUGHT:
                return True, f"RSI {rsi:.1f} overbought (>{RSI_OVERBOUGHT}) - great SHORT setup"
            else:
                return True, f"RSI {rsi:.1f} neutral - OK for SHORT"

    def check_mtf_alignment(self, candles_1h, candles_4h, direction):
        """
        Check multi-timeframe alignment.
        1h breakout should align with 4h trend.
        """
        if candles_4h is None or len(candles_4h) == 0:
            return None, "4h data not available"

        sma_4h = self.calculate_sma(candles_4h, SMA_PERIOD)

        if sma_4h is None:
            return None, "4h SMA not calculable"

        # Get current price from 4h candles
        current_price_4h = candles_4h[-1]['close'] if candles_4h else None

        if current_price_4h is None:
            return None, "No 4h price data"

        if direction == 'LONG':
            if current_price_4h > sma_4h:
                return True, f"4h trend bullish (price ${current_price_4h:.2f} > SMA ${sma_4h:.2f})"
            else:
                return False, f"4h trend bearish (price ${current_price_4h:.2f} < SMA ${sma_4h:.2f})"
        else:  # SHORT
            if current_price_4h < sma_4h:
                return True, f"4h trend bearish (price ${current_price_4h:.2f} < SMA ${sma_4h:.2f})"
            else:
                return False, f"4h trend bullish (price ${current_price_4h:.2f} > SMA ${sma_4h:.2f})"

    def check_entry(self, symbol, direction, current_price,
                    candles_1h, candles_4h=None) -> Tuple[bool, int, List[str]]:
        """
        Run all entry confirmations and calculate confidence score.

        Args:
            symbol: Trading symbol
            direction: 'LONG' or 'SHORT'
            current_price: Current market price
            candles_1h: 1-hour candle data
            candles_4h: 4-hour candle data (optional)

        Returns:
            tuple: (should_enter, confidence, reasons)
            - should_enter: bool
            - confidence: 0-100 score
            - reasons: list of confirmation results
        """
        reasons = []
        confidence = 0

        # Check 1: Volume confirmation
        volume_ok, volume_reason = self.check_volume_confirmation(candles_1h, symbol)
        reasons.append(f"VOLUME: {volume_reason}")
        if volume_ok:
            confidence += WEIGHT_VOLUME

        # Check 2: Trend confirmation
        trend_ok, trend_reason = self.check_trend_confirmation(
            candles_1h, current_price, direction
        )
        reasons.append(f"TREND: {trend_reason}")
        if trend_ok:
            confidence += WEIGHT_TREND

        # Check 3: RSI filter
        rsi_ok, rsi_reason = self.check_rsi_filter(candles_1h, direction)
        reasons.append(f"RSI: {rsi_reason}")
        if rsi_ok:
            confidence += WEIGHT_RSI

        # Check 4: Multi-timeframe (if enabled)
        mtf_ok = None
        if self.enable_mtf and candles_4h is not None:
            mtf_ok, mtf_reason = self.check_mtf_alignment(candles_1h, candles_4h, direction)
            reasons.append(f"MTF: {mtf_reason}")
            if mtf_ok:
                confidence += WEIGHT_MTF
        else:
            # If MTF disabled, redistribute weight
            if not self.enable_mtf:
                reasons.append("MTF: Disabled")
                # Add MTF weight to confidence if other checks pass
                if volume_ok and trend_ok and rsi_ok:
                    confidence += WEIGHT_MTF
            else:
                reasons.append("MTF: No 4h data")
                # Partial credit if other checks strong
                if volume_ok and trend_ok and rsi_ok:
                    confidence += WEIGHT_MTF // 2

        # Determine if we should enter
        should_enter = confidence >= self.min_confidence

        # Log the decision
        self._log_decision(
            symbol, direction, current_price,
            volume_ok, trend_ok, rsi_ok, mtf_ok,
            confidence, should_enter, reasons
        )

        return should_enter, confidence, reasons

    def get_confirmation_summary(self, symbol, direction, current_price,
                                  candles_1h, candles_4h=None) -> Dict[str, Any]:
        """Get detailed confirmation summary."""
        should_enter, confidence, reasons = self.check_entry(
            symbol, direction, current_price, candles_1h, candles_4h
        )

        return {
            'symbol': symbol,
            'direction': direction,
            'price': current_price,
            'should_enter': should_enter,
            'confidence': confidence,
            'min_confidence': self.min_confidence,
            'reasons': reasons,
            'rsi': self.calculate_rsi(candles_1h),
            'sma': self.calculate_sma(candles_1h),
            'volume_ratio': (
                self.get_current_volume(candles_1h) /
                self.calculate_average_volume(candles_1h)
                if self.calculate_average_volume(candles_1h) else None
            )
        }


# Global instance for simple function interface
_entry_agent = None


def get_entry_agent(enable_mtf=True, min_confidence=MIN_CONFIDENCE):
    """Get or create the global entry agent instance."""
    global _entry_agent
    if _entry_agent is None:
        _entry_agent = EntryAgent(enable_mtf=enable_mtf, min_confidence=min_confidence)
    return _entry_agent


def check_entry(symbol, direction, current_price,
                candles_1h, candles_4h=None) -> Tuple[bool, int, List[str]]:
    """
    Quick entry confirmation check.

    Args:
        symbol: Trading symbol
        direction: 'LONG' or 'SHORT'
        current_price: Current market price
        candles_1h: 1-hour candle data
        candles_4h: 4-hour candle data (optional)

    Returns:
        tuple: (should_enter, confidence, reasons)
    """
    agent = get_entry_agent()
    return agent.check_entry(symbol, direction, current_price, candles_1h, candles_4h)


def check_entry_simple(symbol, direction, current_price, candles_1h) -> bool:
    """
    Simple entry check - returns True/False only.
    """
    agent = get_entry_agent()
    should_enter, confidence, reasons = agent.check_entry(
        symbol, direction, current_price, candles_1h, None
    )
    return should_enter


def get_entry_summary(symbol, direction, current_price,
                      candles_1h, candles_4h=None) -> Dict[str, Any]:
    """Get detailed entry confirmation summary."""
    agent = get_entry_agent()
    return agent.get_confirmation_summary(
        symbol, direction, current_price, candles_1h, candles_4h
    )


if __name__ == "__main__":
    # Demo/test the entry agent
    from termcolor import cprint
    import random

    def generate_mock_candles(base_price, num_candles, trend='up'):
        """Generate mock candles with specified trend."""
        candles = []
        price = base_price * 0.95 if trend == 'up' else base_price * 1.05

        for i in range(num_candles):
            # Trend bias
            if trend == 'up':
                drift = random.uniform(-0.002, 0.006)
            elif trend == 'down':
                drift = random.uniform(-0.006, 0.002)
            else:
                drift = random.uniform(-0.004, 0.004)

            open_price = price
            close_price = price * (1 + drift)
            high = max(open_price, close_price) * (1 + random.uniform(0, 0.005))
            low = min(open_price, close_price) * (1 - random.uniform(0, 0.005))

            # Volume with occasional spikes
            base_volume = 500000
            if random.random() < 0.2:
                volume = base_volume * random.uniform(1.5, 3.0)
            else:
                volume = base_volume * random.uniform(0.5, 1.2)

            candles.append({
                'open': open_price,
                'high': high,
                'low': low,
                'close': close_price,
                'volume': volume
            })
            price = close_price

        return candles

    agent = EntryAgent(enable_mtf=True, min_confidence=60)

    cprint("\n=== Entry Agent Demo ===\n", "cyan")
    cprint(f"Volume threshold: {VOLUME_MULTIPLIER}x 20-period avg", "white")
    cprint(f"Trend: {SMA_PERIOD}-period SMA", "white")
    cprint(f"RSI: {RSI_PERIOD}-period, OB>{RSI_OVERBOUGHT}, OS<{RSI_OVERSOLD}", "white")
    cprint(f"Min confidence: {MIN_CONFIDENCE}%", "white")
    cprint(f"Weights: Vol={WEIGHT_VOLUME}, Trend={WEIGHT_TREND}, RSI={WEIGHT_RSI}, MTF={WEIGHT_MTF}", "white")

    # Test 1: Strong uptrend with volume
    cprint("\n--- Test 1: Strong Uptrend + Volume Spike ---", "yellow")
    candles_1h = generate_mock_candles(150.0, 60, trend='up')
    # Add volume spike to last candle
    candles_1h[-1]['volume'] = candles_1h[-1]['volume'] * 2.5
    candles_4h = generate_mock_candles(150.0, 60, trend='up')
    current_price = candles_1h[-1]['close']

    should_enter, confidence, reasons = agent.check_entry(
        'AAPL', 'LONG', current_price, candles_1h, candles_4h
    )

    color = "green" if should_enter else "red"
    cprint(f"Price: ${current_price:.2f}", "white")
    cprint(f"Should Enter: {should_enter} | Confidence: {confidence}%", color)
    for reason in reasons:
        cprint(f"  {reason}", "cyan")

    # Test 2: Overbought condition
    cprint("\n--- Test 2: Overbought RSI ---", "yellow")
    # Create strong uptrend that pushes RSI high
    candles_1h = generate_mock_candles(150.0, 60, trend='up')
    for i in range(10):
        candles_1h.append({
            'open': candles_1h[-1]['close'],
            'high': candles_1h[-1]['close'] * 1.02,
            'low': candles_1h[-1]['close'] * 0.995,
            'close': candles_1h[-1]['close'] * 1.015,
            'volume': 800000
        })
    current_price = candles_1h[-1]['close']

    should_enter, confidence, reasons = agent.check_entry(
        'AAPL', 'LONG', current_price, candles_1h, None
    )

    color = "green" if should_enter else "red"
    cprint(f"Price: ${current_price:.2f}", "white")
    cprint(f"Should Enter: {should_enter} | Confidence: {confidence}%", color)
    for reason in reasons:
        cprint(f"  {reason}", "cyan")

    # Test 3: Weak volume
    cprint("\n--- Test 3: Weak Volume ---", "yellow")
    candles_1h = generate_mock_candles(150.0, 60, trend='up')
    candles_1h[-1]['volume'] = candles_1h[-1]['volume'] * 0.3  # Low volume
    current_price = candles_1h[-1]['close']

    should_enter, confidence, reasons = agent.check_entry(
        'AAPL', 'LONG', current_price, candles_1h, None
    )

    color = "green" if should_enter else "red"
    cprint(f"Price: ${current_price:.2f}", "white")
    cprint(f"Should Enter: {should_enter} | Confidence: {confidence}%", color)
    for reason in reasons:
        cprint(f"  {reason}", "cyan")

    # Test 4: Counter-trend trade
    cprint("\n--- Test 4: Counter-Trend (LONG in Downtrend) ---", "yellow")
    candles_1h = generate_mock_candles(150.0, 60, trend='down')
    candles_4h = generate_mock_candles(150.0, 60, trend='down')
    candles_1h[-1]['volume'] = candles_1h[-1]['volume'] * 2.0
    current_price = candles_1h[-1]['close']

    should_enter, confidence, reasons = agent.check_entry(
        'AAPL', 'LONG', current_price, candles_1h, candles_4h
    )

    color = "green" if should_enter else "red"
    cprint(f"Price: ${current_price:.2f}", "white")
    cprint(f"Should Enter: {should_enter} | Confidence: {confidence}%", color)
    for reason in reasons:
        cprint(f"  {reason}", "cyan")

    # Test 5: Perfect SHORT setup
    cprint("\n--- Test 5: Perfect SHORT Setup ---", "yellow")
    candles_1h = generate_mock_candles(150.0, 60, trend='down')
    candles_4h = generate_mock_candles(150.0, 60, trend='down')
    candles_1h[-1]['volume'] = candles_1h[-1]['volume'] * 2.5
    current_price = candles_1h[-1]['close']

    should_enter, confidence, reasons = agent.check_entry(
        'AAPL', 'SHORT', current_price, candles_1h, candles_4h
    )

    color = "green" if should_enter else "red"
    cprint(f"Price: ${current_price:.2f}", "white")
    cprint(f"Should Enter: {should_enter} | Confidence: {confidence}%", color)
    for reason in reasons:
        cprint(f"  {reason}", "cyan")

    cprint(f"\nEntry log written to: {ENTRY_LOG_PATH}", "cyan")
