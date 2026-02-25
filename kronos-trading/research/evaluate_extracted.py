#!/usr/bin/env python3
"""
Kronos Strategy Evaluation Pipeline
=====================================
Compares LLM-extracted strategies against existing ones,
auto-generates backtest code for new high-confidence strategies,
runs robustness tests, and reports results to Telegram.

Usage:
    python -m research.evaluate_extracted                  # Compare + report
    python -m research.evaluate_extracted --generate       # Also generate code
    python -m research.evaluate_extracted --backtest       # Generate + run backtests
    python -m research.evaluate_extracted --full           # Full pipeline + Telegram
"""

import argparse
import json
import logging
import os
import signal
import sys
import textwrap
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from research.db import DB_PATH, init_research_db

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).parent.parent

# Known strategies (what we already have or have seen)
KNOWN_STRATEGIES = {
    # Kronos core
    "cascade_p99", "cascade_ride", "liq_bb_combo", "sma_crossover",
    "double_decay_reversal", "double_decay", "exhaustion_fade",
    # Alpaca / Moon Dev known
    "breakout_moondev", "gap_and_go", "short_overbought",
    # Also catch close variants
    "cascade p99", "cascade ride", "liquidation bollinger",
    "sma crossover", "double decay", "exhaustion fade",
    "breakout", "gap and go", "short overbought",
    # Social/sentiment (already extracted, not backtestable with price data)
    "social arbitrage", "tiktok social arbitrage",
    "tiktok trend scanning", "social media sentiment",
}

# Categories that are backtestable with our data (OHLCV + liquidation)
BACKTESTABLE_CATEGORIES = {"momentum", "reversal", "mean_reversion", "breakout", "scalping"}

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
log = logging.getLogger("evaluate")


# ---------------------------------------------------------------------------
# Step 1: Query & Compare
# ---------------------------------------------------------------------------
def get_high_confidence_strategies(conn, min_confidence: float = 0.7) -> list[dict]:
    """Get extracted strategies above confidence threshold."""
    rows = conn.execute(
        """SELECT es.*, t.title as video_title, t.channel, t.priority_tier
           FROM extracted_strategies es
           JOIN transcripts t ON es.source_video_id = t.video_id
           WHERE es.confidence >= ?
           ORDER BY es.confidence DESC""",
        (min_confidence,),
    ).fetchall()
    return [dict(r) for r in rows]


def is_known_strategy(strategy_name: str) -> bool:
    """Check if a strategy matches one we already have."""
    name_lower = strategy_name.lower().strip()
    for known in KNOWN_STRATEGIES:
        if known in name_lower or name_lower in known:
            return True
    return False


def is_backtestable(strategy: dict) -> bool:
    """Check if strategy can be backtested with our OHLCV+liquidation data."""
    category = (strategy.get("category") or "").lower()
    if category in BACKTESTABLE_CATEGORIES:
        return True

    # Check if parameters mention indicators we support
    params_str = json.dumps(strategy.get("parameters", {})).lower()
    desc = (strategy.get("description") or "").lower()
    combined = params_str + " " + desc

    backtestable_signals = [
        "sma", "ema", "rsi", "bollinger", "macd", "atr",
        "liquidation", "cascade", "stop loss", "take profit",
        "trailing stop", "breakout", "support", "resistance",
        "moving average", "crossover", "overbought", "oversold",
    ]
    return any(sig in combined for sig in backtestable_signals)


def find_new_strategies(conn, min_confidence: float = 0.7) -> list[dict]:
    """Find genuinely new, high-confidence, backtestable strategies."""
    all_high = get_high_confidence_strategies(conn, min_confidence)

    new_strategies = []
    for s in all_high:
        name = s["strategy_name"]
        if is_known_strategy(name):
            log.info(f"  KNOWN: {name} ({s['confidence']:.0%})")
            continue
        if not is_backtestable(s):
            log.info(f"  SKIP (not backtestable): {name} ({s['confidence']:.0%})")
            continue
        log.info(f"  NEW: {name} ({s['confidence']:.0%}) [{s['category']}]")
        new_strategies.append(s)

    return new_strategies


# ---------------------------------------------------------------------------
# Step 2: Generate Strategy Code
# ---------------------------------------------------------------------------
def sanitize_name(name: str) -> str:
    """Convert strategy name to valid Python identifier."""
    import re
    # Remove parentheses and content
    clean = re.sub(r"\([^)]*\)", "", name).strip()
    # Convert to snake_case
    clean = re.sub(r"[^a-zA-Z0-9]+", "_", clean).strip("_").lower()
    # Limit length
    return clean[:40]


def class_name(snake: str) -> str:
    """Convert snake_case to CamelCase."""
    return "".join(word.capitalize() for word in snake.split("_"))


def generate_strategy_code(strategy: dict) -> tuple[str, str, str]:
    """Generate BaseStrategy subclass code from extracted strategy.

    Returns:
        (filename, strategy_name, code_string)
    """
    name_snake = sanitize_name(strategy["strategy_name"])
    name_class = class_name(name_snake)
    params = json.loads(strategy.get("parameters") or "{}")
    description = strategy.get("description", "")
    category = strategy.get("category", "other")
    source_video = strategy.get("video_title", "Unknown")

    # Build default_params from extracted parameters
    # Extract numeric params where possible
    param_code_lines = []
    param_ranges = {}

    # Standard params every strategy needs
    default_params = {
        "trailing_stop_pct": 2.0,
        "take_profit_pct": 5.0,
        "max_hold_bars": 20,
        "cooldown_bars": 5,
        "entry_strength": 0.8,
        "max_history": 300,
    }

    # Try to extract meaningful numeric params from the LLM output
    for key, val in params.items():
        if isinstance(val, (int, float)):
            clean_key = sanitize_name(key)
            if clean_key:
                default_params[clean_key] = val
                # Build param ranges for robustness
                if isinstance(val, float):
                    param_ranges[clean_key] = [val * 0.5, val, val * 1.5, val * 2.0]
                elif isinstance(val, int) and val > 0:
                    param_ranges[clean_key] = [max(1, val // 2), val, val * 2]

    # Always add robustness ranges for standard params
    param_ranges.update({
        "trailing_stop_pct": [1.0, 1.5, 2.0, 3.0],
        "take_profit_pct": [3.0, 5.0, 8.0, 10.0],
        "max_hold_bars": [10, 20, 30],
    })

    for k, v in default_params.items():
        param_code_lines.append(f'            "{k}": {v!r},')

    param_block = "\n".join(param_code_lines)
    param_ranges_str = json.dumps(param_ranges, indent=4)

    # Determine which indicators are mentioned
    desc_lower = description.lower()
    use_rsi = "rsi" in desc_lower or "overbought" in desc_lower or "oversold" in desc_lower
    use_bb = "bollinger" in desc_lower or "bb " in desc_lower
    use_sma = "sma" in desc_lower or "moving average" in desc_lower
    use_ema = "ema" in desc_lower
    use_liq = "liquidation" in desc_lower or "cascade" in desc_lower
    is_reversal = category in ("reversal", "mean_reversion") or "reversal" in desc_lower

    # Build the on_candle logic based on what indicators were mentioned
    entry_logic_parts = []
    indicator_checks = []

    if use_liq:
        indicator_checks.append("""
        # Liquidation cascade detection
        liq_values = self.liq_usd(self.get_param("lookback_bars"))
        nonzero = liq_values[liq_values > 0]
        if len(nonzero) < 20:
            return Signal(direction=None)
        threshold = np.percentile(nonzero, self.get_param("liq_percentile"))
        cascade_active = candle.liquidation_usd >= threshold""")
        entry_logic_parts.append("cascade_active")
        default_params["lookback_bars"] = 300
        default_params["liq_percentile"] = 85
        default_params["liq_ratio_threshold"] = 0.6
        param_ranges["liq_percentile"] = [75, 80, 85, 90, 95]

    if use_rsi:
        indicator_checks.append("""
        # RSI filter
        current_rsi = self.rsi(self.get_param("rsi_period"))
        rsi_ready = current_rsi is not None""")
        if is_reversal:
            entry_logic_parts.append("rsi_ready and (current_rsi > 70 or current_rsi < 30)")
        else:
            entry_logic_parts.append("rsi_ready")
        default_params["rsi_period"] = 14
        param_ranges["rsi_period"] = [10, 14, 20]

    if use_bb:
        indicator_checks.append("""
        # Bollinger Band check
        bb = self.bollinger_bands(self.get_param("bb_period"), self.get_param("bb_std"))
        bb_ready = bb is not None""")
        if is_reversal:
            entry_logic_parts.append("bb_ready and (candle.close > bb[0] or candle.close < bb[2])")
        else:
            entry_logic_parts.append("bb_ready and (candle.close > bb[0] or candle.close < bb[2])")
        default_params["bb_period"] = 20
        default_params["bb_std"] = 2.0
        param_ranges["bb_period"] = [15, 20, 30]

    if use_sma or use_ema:
        indicator_checks.append("""
        # Moving average trend filter
        fast_ma = self.ema(self.get_param("fast_period")) if self.get_param("fast_period") else None
        slow_ma = self.sma(self.get_param("slow_period")) if self.get_param("slow_period") else None
        ma_ready = fast_ma is not None and slow_ma is not None""")
        entry_logic_parts.append("ma_ready")
        default_params["fast_period"] = 10
        default_params["slow_period"] = 30

    # If no specific indicators detected, use a generic liquidation-based approach
    if not indicator_checks:
        indicator_checks.append("""
        # Generic indicator setup (customize based on strategy logic)
        liq_values = self.liq_usd(200)
        nonzero = liq_values[liq_values > 0]
        if len(nonzero) < 20:
            return Signal(direction=None)
        threshold = np.percentile(nonzero, self.get_param("liq_percentile"))
        cascade_active = candle.liquidation_usd >= threshold""")
        entry_logic_parts.append("cascade_active")
        default_params["liq_percentile"] = 85
        param_ranges["liq_percentile"] = [75, 80, 85, 90, 95]

    indicator_block = "\n".join(indicator_checks)
    entry_condition = " and ".join(entry_logic_parts) if entry_logic_parts else "False"

    # Rebuild param block with final defaults
    param_code_lines = []
    for k, v in sorted(default_params.items()):
        param_code_lines.append(f'            "{k}": {v!r},')
    param_block = "\n".join(param_code_lines)

    # Direction logic
    if use_liq:
        direction_logic = """
            # Determine direction from liquidation imbalance
            total_liq = candle.liquidation_usd
            if total_liq > 0:
                ratio_thresh = self.get_param("liq_ratio_threshold")
                short_ratio = candle.short_liq_usd / total_liq
                long_ratio = candle.long_liq_usd / total_liq
                if short_ratio >= ratio_thresh:
                    direction = {dir_long}   # shorts rekt = bullish
                elif long_ratio >= ratio_thresh:
                    direction = {dir_short}  # longs rekt = bearish
                else:
                    return Signal(direction=None)
            else:
                return Signal(direction=None)""".format(
            dir_long="-1" if is_reversal else "1",
            dir_short="1" if is_reversal else "-1",
        )
    elif use_sma or use_ema:
        direction_logic = """
            # Direction from moving average crossover
            direction = 1 if fast_ma > slow_ma else -1"""
    elif use_rsi and is_reversal:
        direction_logic = """
            # Direction from RSI extremes (reversal)
            direction = -1 if current_rsi > 70 else 1 if current_rsi < 30 else None
            if direction is None:
                return Signal(direction=None)"""
    else:
        direction_logic = """
            # Direction from price action
            direction = 1 if candle.close > candle.open else -1"""

    code = f'''#!/usr/bin/env python3
"""
{strategy["strategy_name"]}
{"=" * len(strategy["strategy_name"])}
Auto-generated from: {source_video}
Category: {category}
Confidence: {strategy["confidence"]:.0%}

{description}

NOTE: This is auto-generated code from LLM-extracted strategy descriptions.
      Review and tune parameters before live trading.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from strategies.templates.base_strategy import BaseStrategy, CandleData, Signal
import numpy as np


class {name_class}(BaseStrategy):
    name = "{name_snake}"
    version = "0.1"

    def default_params(self) -> dict:
        return {{
{param_block}
        }}

    def on_init(self):
        self._in_trade = False
        self._trade_direction = 0
        self._bars_held = 0
        self._peak = 0.0
        self._trough = float("inf")
        self._cooldown = 0

    def on_candle(self, candle: CandleData) -> Signal:
        if self._cooldown > 0:
            self._cooldown -= 1

        # --- IN POSITION: manage exits ---
        if self._in_trade:
            self._bars_held += 1

            if self._trade_direction == 1:
                self._peak = max(self._peak, candle.high)
                stop = self._peak * (1 - self.get_param("trailing_stop_pct") / 100)
                if candle.low <= stop:
                    return self._exit("trailing_stop")
            else:
                self._trough = min(self._trough, candle.low)
                stop = self._trough * (1 + self.get_param("trailing_stop_pct") / 100)
                if candle.high >= stop:
                    return self._exit("trailing_stop")

            if self._bars_held >= self.get_param("max_hold_bars"):
                return self._exit("max_hold")

            return Signal(direction=None)

        # --- NO POSITION: check for entry ---
        if self._cooldown > 0:
            return Signal(direction=None)

        if len(self._candle_history) < self.get_param("max_history") // 2:
            return Signal(direction=None)
{indicator_block}

        # Entry condition
        if {entry_condition}:
{direction_logic}

            self._in_trade = True
            self._trade_direction = direction
            self._bars_held = 0
            self._peak = candle.high
            self._trough = candle.low

            return Signal(
                direction=direction,
                strength=self.get_param("entry_strength"),
                tag=f"{name_snake}_{{\'bull\' if direction == 1 else \'bear\'}}",
            )

        return Signal(direction=None)

    def _exit(self, reason: str) -> Signal:
        self._in_trade = False
        self._trade_direction = 0
        self._cooldown = self.get_param("cooldown_bars")
        return Signal(direction=0, tag=f"exit_{{reason}}")

    def on_trade(self, pnl: float, pnl_pct: float):
        self._in_trade = False
        self._trade_direction = 0


# Parameter ranges for robustness testing
PARAM_RANGES = {param_ranges_str}
'''

    filename = f"{name_snake}.py"
    return filename, name_snake, code


# ---------------------------------------------------------------------------
# Step 3: Run Backtests & Robustness
# ---------------------------------------------------------------------------
def run_backtest_and_robustness(strategy_path: Path, strategy_name: str) -> dict | None:
    """Run a strategy through backtesting and robustness suite.

    Returns dict with results or None on failure.
    """
    import importlib.util
    from strategies.templates.base_strategy import BaseStrategy

    # Import the generated strategy module
    spec = importlib.util.spec_from_file_location(strategy_name, strategy_path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)

    # Find the BaseStrategy subclass
    strategy_class = None
    for attr_name in dir(mod):
        attr = getattr(mod, attr_name)
        if (isinstance(attr, type)
                and issubclass(attr, BaseStrategy)
                and attr is not BaseStrategy):
            strategy_class = attr
            break

    if not strategy_class:
        log.error(f"No BaseStrategy subclass found in {strategy_path}")
        return None

    param_ranges = getattr(mod, "PARAM_RANGES", None)

    from core.backtester import Backtester
    from core.robustness import RobustnessTestSuite

    # Run backtest on BTC-USD 1h
    bt = Backtester(
        symbol="BTC-USD",
        timeframe="1h",
        initial_capital=10000.0,
        use_liquidation_data=True,
    )

    try:
        strategy = strategy_class()
        report = bt.run(strategy)
    except Exception as e:
        log.error(f"Backtest failed for {strategy_name}: {e}")
        return None

    result = {
        "name": strategy_name,
        "total_return_pct": report.total_return_pct,
        "max_drawdown_pct": report.max_drawdown_pct,
        "sharpe_ratio": report.sharpe_ratio,
        "total_trades": report.total_trades,
        "win_rate_pct": report.win_rate_pct,
        "profit_factor": report.profit_factor,
        "return_dd_ratio": report.return_dd_ratio,
    }

    # Only run robustness if the strategy is profitable
    if report.total_return_pct > 0 and report.total_trades >= 5:
        log.info(f"Running robustness suite for {strategy_name}...")
        suite = RobustnessTestSuite(bt, strategy_class)

        try:
            robust = suite.run_all(
                param_ranges=param_ranges,
                n_monte_carlo=50,
                n_walk_windows=4,
            )
            result["robustness"] = {
                "tests_passed": robust.tests_passed,
                "tests_total": robust.tests_total,
                "overall_pass": robust.overall_pass,
                "oos_pass": robust.oos.passed if robust.oos else False,
                "walk_forward_pass": robust.walk_forward.passed if robust.walk_forward else False,
                "param_sensitivity_pass": robust.param_sensitivity.passed if robust.param_sensitivity else False,
                "monte_carlo_pass": robust.monte_carlo.passed if robust.monte_carlo else False,
                "rolling_window_pass": robust.rolling_window.passed if robust.rolling_window else False,
            }
            result["robustness_summary"] = robust.summary()
        except Exception as e:
            log.error(f"Robustness failed for {strategy_name}: {e}")
            result["robustness"] = {"error": str(e)}
    else:
        result["robustness"] = {"skipped": "Not profitable or too few trades"}

    return result


# ---------------------------------------------------------------------------
# Step 4: Telegram Report
# ---------------------------------------------------------------------------
def send_telegram_report(new_strategies: list[dict], backtest_results: list[dict]):
    """Send extraction and backtest results to Telegram."""
    from execution.telegram_notifier import TelegramNotifier

    notifier = TelegramNotifier()
    if not notifier.enabled:
        log.warning("Telegram not configured, skipping report")
        return

    # Build summary message
    lines = ["<b>KRONOS Research Report</b>", ""]

    if not new_strategies:
        lines.append("No new high-confidence strategies found.")
        notifier.send("\n".join(lines))
        return

    lines.append(f"<b>{len(new_strategies)} new strategies extracted:</b>")
    for s in new_strategies:
        lines.append(
            f"  [{s['confidence']:.0%}] {s['strategy_name']} ({s['category']})"
        )
    lines.append("")

    if backtest_results:
        lines.append("<b>Backtest Results:</b>")
        for r in backtest_results:
            ret = r["total_return_pct"]
            emoji = "+" if ret > 0 else ""
            trades = r["total_trades"]
            sharpe = r["sharpe_ratio"]
            dd = r["max_drawdown_pct"]

            lines.append(
                f"  <b>{r['name']}</b>: {emoji}{ret:.1f}% | "
                f"DD {dd:.1f}% | Sharpe {sharpe:.2f} | {trades} trades"
            )

            robust = r.get("robustness", {})
            if "tests_passed" in robust:
                p = robust["tests_passed"]
                t = robust["tests_total"]
                status = "ROBUST" if robust["overall_pass"] else "NOT ROBUST"
                lines.append(f"    Robustness: {p}/{t} tests | {status}")
            elif "skipped" in robust:
                lines.append(f"    Robustness: Skipped ({robust['skipped']})")
    else:
        lines.append("No backtestable strategies to test.")

    msg = "\n".join(lines)
    notifier.send(msg)
    log.info("Telegram report sent")


# ---------------------------------------------------------------------------
# Main Pipeline
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description="Kronos Strategy Evaluation Pipeline")
    parser.add_argument("--min-confidence", type=float, default=0.7,
                        help="Minimum confidence threshold (default: 0.7)")
    parser.add_argument("--generate", action="store_true",
                        help="Generate strategy code for new strategies")
    parser.add_argument("--backtest", action="store_true",
                        help="Generate + run backtests")
    parser.add_argument("--full", action="store_true",
                        help="Full pipeline: compare, generate, backtest, robustness, Telegram")
    args = parser.parse_args()

    if args.full:
        args.generate = True
        args.backtest = True

    conn = init_research_db()

    # --- Step 1: Compare ---
    print("\n=== Step 1: Comparing Extracted vs Known Strategies ===\n")
    all_high = get_high_confidence_strategies(conn, args.min_confidence)
    print(f"Total high-confidence strategies (>={args.min_confidence:.0%}): {len(all_high)}")

    new_strategies = find_new_strategies(conn, args.min_confidence)
    print(f"\nNew backtestable strategies: {len(new_strategies)}")

    for s in new_strategies:
        print(f"  [{s['confidence']:.0%}] {s['strategy_name']} ({s['category']})")
        desc = s.get("description", "")
        if desc:
            print(f"        {desc[:120]}")

    if not new_strategies:
        print("\nNo new strategies to process.")
        if args.full:
            send_telegram_report([], [])
        conn.close()
        return

    # --- Step 2: Generate Code ---
    generated = []
    if args.generate:
        print("\n=== Step 2: Generating Strategy Code ===\n")
        gen_dir = PROJECT_ROOT / "strategies" / "generated"
        gen_dir.mkdir(parents=True, exist_ok=True)

        # Write __init__.py if missing
        init_file = gen_dir / "__init__.py"
        if not init_file.exists():
            init_file.write_text("# Auto-generated strategies\n")

        for s in new_strategies:
            filename, name, code = generate_strategy_code(s)
            filepath = gen_dir / filename
            filepath.write_text(code)
            generated.append((filepath, name, s))
            print(f"  Generated: strategies/generated/{filename}")

    # --- Step 3: Backtest + Robustness ---
    backtest_results = []
    if args.backtest and generated:
        print("\n=== Step 3: Running Backtests + Robustness ===\n")
        from strategies.templates.base_strategy import BaseStrategy

        for filepath, name, strategy_info in generated:
            print(f"\nTesting: {name}")
            result = run_backtest_and_robustness(filepath, name)
            if result:
                backtest_results.append(result)
                ret = result["total_return_pct"]
                trades = result["total_trades"]
                dd = result["max_drawdown_pct"]
                sharpe = result["sharpe_ratio"]
                print(f"  Return: {ret:+.2f}% | DD: {dd:.2f}% | "
                      f"Sharpe: {sharpe:.2f} | Trades: {trades}")

                robust = result.get("robustness", {})
                if "tests_passed" in robust:
                    print(f"  Robustness: {robust['tests_passed']}/{robust['tests_total']} "
                          f"{'PASS' if robust['overall_pass'] else 'FAIL'}")

    # --- Step 4: Telegram Report ---
    if args.full:
        print("\n=== Step 4: Sending Telegram Report ===\n")
        send_telegram_report(new_strategies, backtest_results)

    # --- Summary ---
    print("\n=== Summary ===")
    print(f"  Strategies analyzed: {len(all_high)}")
    print(f"  New strategies:      {len(new_strategies)}")
    print(f"  Code generated:      {len(generated)}")
    print(f"  Backtests run:       {len(backtest_results)}")
    profitable = [r for r in backtest_results if r["total_return_pct"] > 0]
    print(f"  Profitable:          {len(profitable)}/{len(backtest_results)}")
    robust_pass = [r for r in backtest_results
                   if r.get("robustness", {}).get("overall_pass")]
    print(f"  Passed robustness:   {len(robust_pass)}/{len(backtest_results)}")

    conn.close()


if __name__ == "__main__":
    main()
