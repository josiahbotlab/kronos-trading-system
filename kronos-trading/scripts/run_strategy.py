#!/usr/bin/env python3
"""
Kronos Strategy Runner
======================
CLI to run any strategy through backtesting and robustness tests.

Usage:
    # Run single strategy
    python run_strategy.py cascade_ride --symbol BTC/USDC:USDC --timeframe 1h

    # Run with robustness tests
    python run_strategy.py cascade_p99 --symbol BTC/USDC:USDC --timeframe 1h --robust

    # Compare all strategies on a symbol
    python run_strategy.py --compare --symbol BTC/USDC:USDC --timeframe 1h

    # Multi-symbol run
    python run_strategy.py cascade_ride --timeframe 1h --multi

    # List available strategies
    python run_strategy.py --list
"""

import argparse
import json
import logging
import sys
from datetime import datetime, timezone
from pathlib import Path

# Add project root
sys.path.insert(0, str(Path(__file__).parent.parent))

from core.backtester import Backtester
from core.metrics import compare_reports
from core.robustness import RobustnessTestSuite

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
log = logging.getLogger("runner")

# ---------------------------------------------------------------------------
# Strategy Registry
# ---------------------------------------------------------------------------
STRATEGIES = {}


def register_strategies():
    """Import and register all available strategies."""
    global STRATEGIES

    try:
        from strategies.momentum.cascade_ride import CascadeRide
        STRATEGIES["cascade_ride"] = CascadeRide
    except ImportError as e:
        log.warning(f"Could not load cascade_ride: {e}")

    try:
        from strategies.momentum.cascade_p99 import CascadeP99
        STRATEGIES["cascade_p99"] = CascadeP99
    except ImportError as e:
        log.warning(f"Could not load cascade_p99: {e}")

    try:
        from strategies.momentum.liq_bb_combo import LiquidationBollingerCombo
        STRATEGIES["liq_bb_combo"] = LiquidationBollingerCombo
    except ImportError as e:
        log.warning(f"Could not load liq_bb_combo: {e}")

    try:
        from strategies.momentum.sma_crossover import SMACrossover
        STRATEGIES["sma_crossover"] = SMACrossover
    except ImportError as e:
        log.warning(f"Could not load sma_crossover: {e}")

    try:
        from strategies.reversal.exhaustion_fade import ExhaustionFade
        STRATEGIES["exhaustion_fade"] = ExhaustionFade
    except ImportError as e:
        log.warning(f"Could not load exhaustion_fade: {e}")

    try:
        from strategies.reversal.double_decay import DoubleDecayReversal
        STRATEGIES["double_decay"] = DoubleDecayReversal
    except ImportError as e:
        log.warning(f"Could not load double_decay: {e}")


# Default param ranges for robustness tests
PARAM_RANGES = {
    "cascade_ride": {
        "liq_threshold_usd": [20000, 50000, 100000, 200000],
        "trailing_stop_pct": [1.0, 1.5, 2.0, 3.0],
        "take_profit_pct": [3.0, 5.0, 8.0, 10.0],
        "max_hold_bars": [12, 24, 48],
    },
    "cascade_p99": {
        "percentile": [95, 97, 99],
        "trailing_stop_pct": [1.0, 1.5, 2.0, 3.0],
        "take_profit_pct": [5.0, 8.0, 12.0],
        "max_hold_bars": [6, 12, 18],
    },
    "liq_bb_combo": {
        "liq_percentile": [70, 80, 90],
        "bb_period": [15, 20, 30],
        "bb_std": [1.5, 2.0, 2.5],
    },
    "exhaustion_fade": {
        "cascade_percentile": [90, 95],
        "decay_bars": [2, 3, 5],
        "stop_loss_pct": [1.0, 1.5, 2.0],
        "take_profit_pct": [2.0, 3.0, 5.0],
    },
    "double_decay": {
        "cascade_percentile": [85, 90, 95],
        "decay_1_ratio": [0.4, 0.5, 0.6],
        "decay_2_ratio": [0.15, 0.25, 0.35],
        "stop_loss_pct": [0.5, 1.0, 1.5],
    },
    "sma_crossover": {
        "fast_period": [5, 8, 10, 15, 20],
        "slow_period": [20, 30, 40, 50, 60],
    },
}

DEFAULT_SYMBOLS = [
    "BTC/USDC:USDC", "ETH/USDC:USDC", "SOL/USDC:USDC",
    "DOGE/USDC:USDC", "XRP/USDC:USDC", "LINK/USDC:USDC",
]


# ---------------------------------------------------------------------------
# Commands
# ---------------------------------------------------------------------------
def list_strategies():
    """List all available strategies."""
    print("\n📋 Available Strategies:")
    print("-" * 50)
    for name, cls in STRATEGIES.items():
        strategy = cls()
        print(f"  {name:25s} v{strategy.version}")
        params = strategy.default_params()
        key_params = {k: v for k, v in list(params.items())[:5]}
        print(f"    Params: {key_params}")
    print()


def run_single(args):
    """Run a single strategy."""
    if args.strategy not in STRATEGIES:
        print(f"Unknown strategy: {args.strategy}")
        list_strategies()
        return

    cls = STRATEGIES[args.strategy]

    # Parse any extra params from --params '{"key": value}'
    extra_params = {}
    if args.params:
        extra_params = json.loads(args.params)

    bt = Backtester(
        symbol=args.symbol,
        timeframe=args.timeframe,
        start_date=args.start,
        end_date=args.end,
        initial_capital=args.capital,
        fee_rate=args.fee,
        leverage=args.leverage,
        use_liquidation_data=True,
    )

    strategy = cls(**extra_params)
    report = bt.run(strategy)

    print(report.summary())

    # Save trades to file if requested
    if args.output:
        save_results(args.output, args.strategy, report, bt.trades)

    return report


def run_robust(args):
    """Run strategy with full robustness suite."""
    if args.strategy not in STRATEGIES:
        print(f"Unknown strategy: {args.strategy}")
        return

    cls = STRATEGIES[args.strategy]

    extra_params = {}
    if args.params:
        extra_params = json.loads(args.params)

    bt = Backtester(
        symbol=args.symbol,
        timeframe=args.timeframe,
        start_date=args.start,
        end_date=args.end,
        initial_capital=args.capital,
        fee_rate=args.fee,
        leverage=args.leverage,
        use_liquidation_data=True,
    )

    param_ranges = PARAM_RANGES.get(args.strategy, None)

    suite = RobustnessTestSuite(
        backtester=bt,
        strategy_class=cls,
        strategy_params=extra_params,
    )

    results = suite.run_all(
        param_ranges=param_ranges,
        n_monte_carlo=100,
        n_walk_windows=5,
    )

    print(results.summary())


def run_compare(args):
    """Compare all strategies on a single symbol."""
    bt = Backtester(
        symbol=args.symbol,
        timeframe=args.timeframe,
        start_date=args.start,
        end_date=args.end,
        initial_capital=args.capital,
        fee_rate=args.fee,
        leverage=args.leverage,
        use_liquidation_data=True,
    )

    reports = {}
    for name, cls in STRATEGIES.items():
        print(f"\nRunning {name}...")
        try:
            strategy = cls()
            reports[name] = bt.run(strategy)
        except Exception as e:
            log.warning(f"  {name} failed: {e}")

    if reports:
        print("\n" + compare_reports(reports))

        # Rank by return/DD ratio
        ranked = sorted(
            reports.items(),
            key=lambda x: x[1].return_dd_ratio,
            reverse=True,
        )
        print("\n🏆 Ranked by Return/Drawdown Ratio:")
        for i, (name, r) in enumerate(ranked, 1):
            print(
                f"  {i}. {name:25s} "
                f"R/DD: {r.return_dd_ratio:.2f}x | "
                f"Return: {r.total_return_pct:+.2f}% | "
                f"Sharpe: {r.sharpe_ratio:.2f}"
            )


def run_multi_symbol(args):
    """Run a strategy across multiple symbols."""
    if args.strategy not in STRATEGIES:
        print(f"Unknown strategy: {args.strategy}")
        return

    cls = STRATEGIES[args.strategy]
    symbols = DEFAULT_SYMBOLS

    bt = Backtester(
        timeframe=args.timeframe,
        start_date=args.start,
        end_date=args.end,
        initial_capital=args.capital,
        fee_rate=args.fee,
        leverage=args.leverage,
        use_liquidation_data=True,
    )

    results = bt.run_multi_symbol(cls, symbols)

    if results:
        print(f"\n📊 {args.strategy} across {len(results)} symbols:\n")
        print(compare_reports(results))

        # Summary
        profitable = sum(1 for r in results.values() if r.total_return_pct > 0)
        print(f"\n  Profitable: {profitable}/{len(results)} symbols")


def save_results(filepath: str, strategy_name: str, report, trades):
    """Save results to JSON."""
    output = {
        "strategy": strategy_name,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "metrics": {
            "total_return_pct": report.total_return_pct,
            "max_drawdown_pct": report.max_drawdown_pct,
            "sharpe_ratio": report.sharpe_ratio,
            "sortino_ratio": report.sortino_ratio,
            "win_rate_pct": report.win_rate_pct,
            "total_trades": report.total_trades,
            "profit_factor": report.profit_factor,
            "return_dd_ratio": report.return_dd_ratio,
        },
        "trades": [
            {
                "entry_time": t.entry_time,
                "exit_time": t.exit_time,
                "side": t.side,
                "pnl": t.pnl,
                "pnl_pct": t.pnl_pct,
                "tag": t.tag,
            }
            for t in trades
        ],
    }

    with open(filepath, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\n💾 Results saved to {filepath}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    register_strategies()

    parser = argparse.ArgumentParser(
        description="Kronos Strategy Runner",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    parser.add_argument("strategy", nargs="?", help="Strategy name")
    parser.add_argument("--symbol", default="BTC/USDC:USDC", help="Trading symbol")
    parser.add_argument("--timeframe", default="1h", help="Candle timeframe")
    parser.add_argument("--start", help="Start date (YYYY-MM-DD)")
    parser.add_argument("--end", help="End date (YYYY-MM-DD)")
    parser.add_argument("--capital", type=float, default=10000, help="Initial capital")
    parser.add_argument("--fee", type=float, default=0.0006, help="Fee rate")
    parser.add_argument("--leverage", type=float, default=1.0, help="Leverage")
    parser.add_argument("--params", help="Strategy params as JSON string")
    parser.add_argument("--output", help="Save results to JSON file")

    parser.add_argument("--list", action="store_true", help="List strategies")
    parser.add_argument("--robust", action="store_true", help="Run robustness tests")
    parser.add_argument("--compare", action="store_true", help="Compare all strategies")
    parser.add_argument("--multi", action="store_true", help="Run across multiple symbols")

    args = parser.parse_args()

    if args.list:
        list_strategies()
    elif args.compare:
        run_compare(args)
    elif args.multi:
        run_multi_symbol(args)
    elif args.robust:
        run_robust(args)
    elif args.strategy:
        run_single(args)
    else:
        parser.print_help()
        print()
        list_strategies()


if __name__ == "__main__":
    main()
