#!/usr/bin/env python3
"""
Run new extracted strategies through backtest + robustness pipeline.
Usage: python3 scripts/run_new_strategies.py
"""

import importlib.util
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from core.backtester import Backtester
from core.robustness import RobustnessTestSuite
from strategies.templates.base_strategy import BaseStrategy

STRATEGIES = [
    ("strategies/generated/bb_squeeze_breakout.py", "BbSqueezeBreakout"),
    ("strategies/generated/consecutive_down_reversal.py", "ConsecutiveDownReversal"),
    ("strategies/generated/kalman_bb_breakout.py", "KalmanBbBreakout"),
    ("strategies/generated/consolidation_breakout.py", "ConsolidationBreakout"),
    ("strategies/generated/parabolic_short.py", "ParabolicShort"),
]

PROJECT_ROOT = Path(__file__).parent.parent


def load_strategy(filepath: str, class_name: str):
    """Load a strategy class from file."""
    full_path = PROJECT_ROOT / filepath
    spec = importlib.util.spec_from_file_location(class_name, full_path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    cls = getattr(mod, class_name)
    param_ranges = getattr(mod, "PARAM_RANGES", None)
    return cls, param_ranges


def main():
    print("\n" + "=" * 70)
    print(f"  KRONOS v2.0 STRATEGY EVALUATION")
    print(f"  {len(STRATEGIES)} strategies (transcript + web research implementations)")
    print("=" * 70)

    results = []

    for filepath, class_name in STRATEGIES:
        strategy_cls, param_ranges = load_strategy(filepath, class_name)
        strat = strategy_cls()
        name = strat.name

        print(f"\n{'─' * 60}")
        print(f"  {name} ({class_name})")
        print(f"{'─' * 60}")

        # Backtest on BTC-USD 1h
        bt = Backtester(
            symbol="BTC-USD",
            timeframe="1h",
            initial_capital=10000.0,
            use_liquidation_data=True,
        )

        try:
            report = bt.run(strat)
        except Exception as e:
            print(f"  BACKTEST FAILED: {e}")
            results.append({"name": name, "error": str(e)})
            continue

        ret = report.total_return_pct
        dd = report.max_drawdown_pct
        sharpe = report.sharpe_ratio
        trades = report.total_trades
        wr = report.win_rate_pct
        pf = report.profit_factor

        print(f"  Return:    {ret:+.2f}%")
        print(f"  Max DD:    {dd:.2f}%")
        print(f"  Sharpe:    {sharpe:.2f}")
        print(f"  Trades:    {trades}")
        print(f"  Win Rate:  {wr:.1f}%")
        print(f"  PF:        {pf:.2f}")

        result = {
            "name": name,
            "class": class_name,
            "return_pct": ret,
            "max_dd_pct": dd,
            "sharpe": sharpe,
            "trades": trades,
            "win_rate": wr,
            "profit_factor": pf,
        }

        # Run robustness if profitable with enough trades
        if ret > 0 and trades >= 5:
            print(f"\n  Running robustness suite...")
            try:
                suite = RobustnessTestSuite(bt, strategy_cls)
                robust = suite.run_all(
                    param_ranges=param_ranges,
                    n_monte_carlo=50,
                    n_walk_windows=4,
                )
                result["robustness"] = {
                    "passed": robust.tests_passed,
                    "total": robust.tests_total,
                    "overall_pass": robust.overall_pass,
                    "summary": robust.summary(),
                }
                print(f"  Robustness: {robust.tests_passed}/{robust.tests_total} "
                      f"{'PASS' if robust.overall_pass else 'FAIL'}")
                print(f"  {robust.summary()}")
            except Exception as e:
                print(f"  ROBUSTNESS FAILED: {e}")
                result["robustness"] = {"error": str(e)}
        elif trades < 5:
            print(f"  Skipping robustness: too few trades ({trades})")
            result["robustness"] = {"skipped": "too few trades"}
        else:
            print(f"  Skipping robustness: not profitable ({ret:+.2f}%)")
            result["robustness"] = {"skipped": "not profitable"}

        results.append(result)

    # Summary
    print(f"\n{'=' * 70}")
    print("  SUMMARY")
    print(f"{'=' * 70}")
    print(f"\n{'Strategy':<28} {'Return':>8} {'DD':>8} {'Sharpe':>8} {'Trades':>7} {'WR':>6} {'Robust':>10}")
    print("-" * 80)
    for r in results:
        if "error" in r:
            print(f"{r['name']:<28} ERROR: {r['error'][:40]}")
            continue
        robust_str = "—"
        rob = r.get("robustness", {})
        if "passed" in rob:
            robust_str = f"{rob['passed']}/{rob['total']}" + (" PASS" if rob['overall_pass'] else " FAIL")
        elif "skipped" in rob:
            robust_str = "skip"
        print(f"{r['name']:<28} {r['return_pct']:>+7.2f}% {r['max_dd_pct']:>7.2f}% "
              f"{r['sharpe']:>7.2f} {r['trades']:>7} {r['win_rate']:>5.1f}% {robust_str:>10}")

    # Identify strategies to incubate
    incubation_candidates = [
        r for r in results
        if r.get("return_pct", 0) > 0
        and r.get("trades", 0) >= 5
        and r.get("robustness", {}).get("overall_pass", False)
    ]

    if incubation_candidates:
        print(f"\n  INCUBATION CANDIDATES ({len(incubation_candidates)}):")
        for r in incubation_candidates:
            print(f"    {r['name']}: {r['return_pct']:+.2f}% | "
                  f"{r['robustness']['passed']}/{r['robustness']['total']} robust")
    else:
        # Also show profitable even if not robust
        profitable = [r for r in results if r.get("return_pct", 0) > 0 and r.get("trades", 0) >= 5]
        if profitable:
            print(f"\n  PROFITABLE BUT NOT ROBUST ({len(profitable)}):")
            for r in profitable:
                print(f"    {r['name']}: {r['return_pct']:+.2f}%")
        else:
            print("\n  No profitable strategies found.")

    # Save results
    out_path = PROJECT_ROOT / "strategies" / "generated" / "evaluation_results.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\n  Results saved to: {out_path}")

    # Return for use by caller
    return results, incubation_candidates


if __name__ == "__main__":
    main()
