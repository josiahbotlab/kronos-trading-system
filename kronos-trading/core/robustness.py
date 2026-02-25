#!/usr/bin/env python3
"""
Kronos Robustness Test Suite
=============================
Moon Dev's 5 robustness tests to validate strategy edge.

Tests:
1. Out-of-Sample    - 70/30 split, check for decay
2. Walk-Forward     - Rolling windows, consistent performance
3. Parameter Sweep  - 500+ combos, want 80%+ profitable
4. Monte Carlo      - 100 sims removing 20% trades, want 100% survival
5. Rolling Window   - Quarterly breakdown, want all quarters profitable

A strategy must pass ALL 5 to be considered robust.

Usage:
    from core.robustness import RobustnessTestSuite
    from core.backtester import Backtester

    suite = RobustnessTestSuite(backtester, strategy_class, strategy_params)
    results = suite.run_all()
    print(results.summary())
"""

import logging
import random
from dataclasses import dataclass, field
from itertools import product
from typing import Any, Optional

import numpy as np

from core.backtester import Backtester
from core.metrics import Trade, PerformanceReport, calculate_metrics
from strategies.templates.base_strategy import BaseStrategy

log = logging.getLogger("robustness")


# ---------------------------------------------------------------------------
# Result types
# ---------------------------------------------------------------------------
@dataclass
class OOSResult:
    """Out-of-Sample test result."""
    in_sample: PerformanceReport = field(default_factory=PerformanceReport)
    out_of_sample: PerformanceReport = field(default_factory=PerformanceReport)
    return_decay_pct: float = 0.0  # how much OOS return decayed vs IS
    sharpe_decay_pct: float = 0.0
    passed: bool = False

    def summary(self) -> str:
        lines = [
            "--- Out-of-Sample Test ---",
            f"  In-Sample:     {self.in_sample.total_return_pct:+.2f}% | Sharpe {self.in_sample.sharpe_ratio:.2f}",
            f"  Out-of-Sample: {self.out_of_sample.total_return_pct:+.2f}% | Sharpe {self.out_of_sample.sharpe_ratio:.2f}",
            f"  Return Decay:  {self.return_decay_pct:.1f}%",
            f"  Sharpe Decay:  {self.sharpe_decay_pct:.1f}%",
            f"  Result:        {'✅ PASS' if self.passed else '❌ FAIL'}",
        ]
        return "\n".join(lines)


@dataclass
class WalkForwardResult:
    """Walk-Forward Analysis result."""
    window_results: list[PerformanceReport] = field(default_factory=list)
    profitable_windows: int = 0
    total_windows: int = 0
    win_rate_pct: float = 0.0
    avg_return_pct: float = 0.0
    consistency_score: float = 0.0  # std of returns across windows
    passed: bool = False

    def summary(self) -> str:
        lines = [
            "--- Walk-Forward Analysis ---",
            f"  Windows:        {self.total_windows}",
            f"  Profitable:     {self.profitable_windows}/{self.total_windows} ({self.win_rate_pct:.1f}%)",
            f"  Avg Return:     {self.avg_return_pct:+.2f}%",
            f"  Consistency:    {self.consistency_score:.2f} (lower = more consistent)",
            f"  Result:         {'✅ PASS' if self.passed else '❌ FAIL'}",
        ]
        return "\n".join(lines)


@dataclass
class ParamSensitivityResult:
    """Parameter Sensitivity test result."""
    total_combos: int = 0
    profitable_combos: int = 0
    profitable_pct: float = 0.0
    best_params: dict = field(default_factory=dict)
    best_return_pct: float = 0.0
    worst_return_pct: float = 0.0
    median_return_pct: float = 0.0
    passed: bool = False

    def summary(self) -> str:
        lines = [
            "--- Parameter Sensitivity ---",
            f"  Combos Tested:  {self.total_combos}",
            f"  Profitable:     {self.profitable_combos}/{self.total_combos} ({self.profitable_pct:.1f}%)",
            f"  Best Return:    {self.best_return_pct:+.2f}%",
            f"  Worst Return:   {self.worst_return_pct:+.2f}%",
            f"  Median Return:  {self.median_return_pct:+.2f}%",
            f"  Best Params:    {self.best_params}",
            f"  Result:         {'✅ PASS' if self.passed else '❌ FAIL'}",
        ]
        return "\n".join(lines)


@dataclass
class MonteCarloResult:
    """Monte Carlo simulation result."""
    simulations: int = 0
    surviving_sims: int = 0
    survival_rate_pct: float = 0.0
    median_return_pct: float = 0.0
    p5_return_pct: float = 0.0       # 5th percentile (worst case)
    p95_return_pct: float = 0.0      # 95th percentile (best case)
    max_drawdown_median: float = 0.0
    passed: bool = False

    def summary(self) -> str:
        lines = [
            "--- Monte Carlo Simulation ---",
            f"  Simulations:    {self.simulations}",
            f"  Survival Rate:  {self.surviving_sims}/{self.simulations} ({self.survival_rate_pct:.1f}%)",
            f"  Median Return:  {self.median_return_pct:+.2f}%",
            f"  5th Percentile: {self.p5_return_pct:+.2f}%",
            f"  95th Pctile:    {self.p95_return_pct:+.2f}%",
            f"  Median Max DD:  {self.max_drawdown_median:.2f}%",
            f"  Result:         {'✅ PASS' if self.passed else '❌ FAIL'}",
        ]
        return "\n".join(lines)


@dataclass
class RollingWindowResult:
    """Rolling Window (quarterly) test result."""
    quarter_results: list[tuple[str, PerformanceReport]] = field(default_factory=list)
    profitable_quarters: int = 0
    total_quarters: int = 0
    all_profitable: bool = False
    passed: bool = False

    def summary(self) -> str:
        lines = [
            "--- Rolling Window (Quarterly) ---",
            f"  Quarters:       {self.total_quarters}",
            f"  Profitable:     {self.profitable_quarters}/{self.total_quarters}",
            f"  All Profitable: {'Yes' if self.all_profitable else 'No'}",
        ]
        for label, r in self.quarter_results:
            status = "✅" if r.total_return_pct > 0 else "❌"
            lines.append(
                f"    {label}: {r.total_return_pct:+.2f}% | "
                f"{r.total_trades} trades | Sharpe {r.sharpe_ratio:.2f} {status}"
            )
        lines.append(f"  Result:         {'✅ PASS' if self.passed else '❌ FAIL'}")
        return "\n".join(lines)


@dataclass
class FullRobustnessResult:
    """Combined result of all 5 robustness tests."""
    oos: Optional[OOSResult] = None
    walk_forward: Optional[WalkForwardResult] = None
    param_sensitivity: Optional[ParamSensitivityResult] = None
    monte_carlo: Optional[MonteCarloResult] = None
    rolling_window: Optional[RollingWindowResult] = None
    tests_passed: int = 0
    tests_total: int = 5
    overall_pass: bool = False

    def summary(self) -> str:
        lines = [
            "=" * 55,
            "  KRONOS ROBUSTNESS REPORT",
            "=" * 55,
        ]
        if self.oos:
            lines.append(self.oos.summary())
        if self.walk_forward:
            lines.append(self.walk_forward.summary())
        if self.param_sensitivity:
            lines.append(self.param_sensitivity.summary())
        if self.monte_carlo:
            lines.append(self.monte_carlo.summary())
        if self.rolling_window:
            lines.append(self.rolling_window.summary())

        lines.extend([
            "",
            "=" * 55,
            f"  OVERALL: {self.tests_passed}/{self.tests_total} tests passed "
            f"{'✅ ROBUST' if self.overall_pass else '❌ NOT ROBUST'}",
            "=" * 55,
        ])
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# Robustness Test Suite
# ---------------------------------------------------------------------------
class RobustnessTestSuite:
    """
    Run Moon Dev's 5 robustness tests on a strategy.

    Args:
        backtester: Configured Backtester instance (symbol, timeframe, dates set)
        strategy_class: BaseStrategy subclass (not instance)
        strategy_params: Default params for the strategy
    """

    def __init__(
        self,
        backtester: Backtester,
        strategy_class: type,
        strategy_params: Optional[dict] = None,
    ):
        self.bt = backtester
        self.strategy_class = strategy_class
        self.params = strategy_params or {}

    def _make_strategy(self, **override_params) -> BaseStrategy:
        """Create a strategy instance with optional param overrides."""
        merged = {**self.params, **override_params}
        return self.strategy_class(**merged)

    def _run_on_trades(self, trades: list[Trade], capital: float) -> PerformanceReport:
        """Calculate metrics on a subset of trades."""
        return calculate_metrics(trades, initial_capital=capital)

    # -------------------------------------------------------------------
    # Test 1: Out-of-Sample
    # -------------------------------------------------------------------
    def test_out_of_sample(self, split_ratio: float = 0.7) -> OOSResult:
        """
        Split data 70/30. Run strategy on both halves.
        Check that OOS performance doesn't decay more than 50%.
        """
        log.info("Running Out-of-Sample test...")
        result = OOSResult()

        # Load all candles to find split point
        candles = self.bt.load_candles()
        if len(candles) < 20:
            log.warning("Not enough data for OOS test")
            return result

        split_idx = int(len(candles) * split_ratio)
        split_time = candles[split_idx].timestamp_ms

        # In-sample
        orig_end = self.bt.end_ms
        self.bt.end_ms = split_time
        result.in_sample = self.bt.run(self._make_strategy())

        # Out-of-sample
        self.bt.start_ms = split_time
        self.bt.end_ms = orig_end
        result.out_of_sample = self.bt.run(self._make_strategy())

        # Restore
        self.bt.start_ms = candles[0].timestamp_ms
        self.bt.end_ms = orig_end

        # Check decay
        is_ret = result.in_sample.total_return_pct
        oos_ret = result.out_of_sample.total_return_pct

        if is_ret != 0:
            result.return_decay_pct = ((is_ret - oos_ret) / abs(is_ret)) * 100
        if result.in_sample.sharpe_ratio != 0:
            result.sharpe_decay_pct = (
                (result.in_sample.sharpe_ratio - result.out_of_sample.sharpe_ratio)
                / abs(result.in_sample.sharpe_ratio)
            ) * 100

        # Pass if OOS is profitable and decay < 50%
        result.passed = (
            oos_ret > 0
            and result.return_decay_pct < 50
            and result.out_of_sample.total_trades >= 5
        )

        return result

    # -------------------------------------------------------------------
    # Test 2: Walk-Forward Analysis
    # -------------------------------------------------------------------
    def test_walk_forward(self, n_windows: int = 5) -> WalkForwardResult:
        """
        Divide data into N equal windows. Run strategy on each.
        Want majority profitable with consistent returns.
        """
        log.info(f"Running Walk-Forward Analysis ({n_windows} windows)...")
        result = WalkForwardResult()

        candles = self.bt.load_candles()
        if len(candles) < n_windows * 10:
            log.warning("Not enough data for walk-forward test")
            return result

        window_size = len(candles) // n_windows
        orig_start = self.bt.start_ms
        orig_end = self.bt.end_ms
        returns = []

        for i in range(n_windows):
            start_idx = i * window_size
            end_idx = min((i + 1) * window_size, len(candles)) - 1

            self.bt.start_ms = candles[start_idx].timestamp_ms
            self.bt.end_ms = candles[end_idx].timestamp_ms

            report = self.bt.run(self._make_strategy())
            result.window_results.append(report)
            returns.append(report.total_return_pct)

            if report.total_return_pct > 0:
                result.profitable_windows += 1

        # Restore
        self.bt.start_ms = orig_start
        self.bt.end_ms = orig_end

        result.total_windows = n_windows
        result.win_rate_pct = (result.profitable_windows / n_windows) * 100
        result.avg_return_pct = np.mean(returns) if returns else 0
        result.consistency_score = np.std(returns) if len(returns) > 1 else 0

        # Pass if 60%+ windows profitable
        result.passed = result.win_rate_pct >= 60

        return result

    # -------------------------------------------------------------------
    # Test 3: Parameter Sensitivity
    # -------------------------------------------------------------------
    def test_param_sensitivity(
        self,
        param_ranges: dict[str, list],
        max_combos: int = 500,
    ) -> ParamSensitivityResult:
        """
        Test strategy across many parameter combinations.
        Want 80%+ of combos to be profitable.

        Args:
            param_ranges: Dict of param_name -> list of values to test
                e.g. {"threshold": [10000, 50000, 100000], "period": [10, 20, 30]}
            max_combos: Cap on total combinations to test
        """
        log.info("Running Parameter Sensitivity test...")
        result = ParamSensitivityResult()

        # Generate all combos
        keys = list(param_ranges.keys())
        values = list(param_ranges.values())
        all_combos = list(product(*values))

        # Sample if too many
        if len(all_combos) > max_combos:
            random.shuffle(all_combos)
            all_combos = all_combos[:max_combos]

        returns = []
        best_return = float('-inf')
        best_params = {}

        for i, combo in enumerate(all_combos):
            params = dict(zip(keys, combo))
            merged = {**self.params, **params}

            try:
                strategy = self.strategy_class(**merged)
                report = self.bt.run(strategy)
                ret = report.total_return_pct
                returns.append(ret)

                if ret > best_return:
                    best_return = ret
                    best_params = params

                if ret > 0:
                    result.profitable_combos += 1
            except Exception as e:
                log.warning(f"Param combo {params} failed: {e}")
                continue

            if (i + 1) % 50 == 0:
                log.info(f"  Tested {i + 1}/{len(all_combos)} combos...")

        result.total_combos = len(returns)
        result.best_params = best_params
        result.best_return_pct = max(returns) if returns else 0
        result.worst_return_pct = min(returns) if returns else 0
        result.median_return_pct = float(np.median(returns)) if returns else 0

        if result.total_combos > 0:
            result.profitable_pct = (result.profitable_combos / result.total_combos) * 100

        # Pass if 80%+ profitable
        result.passed = result.profitable_pct >= 80

        return result

    # -------------------------------------------------------------------
    # Test 4: Monte Carlo Simulation
    # -------------------------------------------------------------------
    def test_monte_carlo(
        self,
        n_sims: int = 100,
        removal_pct: float = 0.20,
    ) -> MonteCarloResult:
        """
        Run strategy once, then simulate N times by randomly
        removing 20% of trades. Want 100% survival (all sims profitable).
        """
        log.info(f"Running Monte Carlo simulation ({n_sims} sims)...")
        result = MonteCarloResult(simulations=n_sims)

        # Run strategy once to get base trades
        base_report = self.bt.run(self._make_strategy())
        base_trades = self.bt.trades.copy()

        if len(base_trades) < 10:
            log.warning("Not enough trades for Monte Carlo")
            return result

        n_remove = max(1, int(len(base_trades) * removal_pct))
        sim_returns = []
        sim_drawdowns = []

        for _ in range(n_sims):
            # Randomly remove trades
            indices = list(range(len(base_trades)))
            remove_indices = set(random.sample(indices, n_remove))
            sim_trades = [t for i, t in enumerate(base_trades) if i not in remove_indices]

            # Recalculate metrics on remaining trades
            sim_report = self._run_on_trades(sim_trades, self.bt.initial_capital)
            sim_returns.append(sim_report.total_return_pct)
            sim_drawdowns.append(sim_report.max_drawdown_pct)

            if sim_report.total_return_pct > 0:
                result.surviving_sims += 1

        result.survival_rate_pct = (result.surviving_sims / n_sims) * 100
        result.median_return_pct = float(np.median(sim_returns))
        result.p5_return_pct = float(np.percentile(sim_returns, 5))
        result.p95_return_pct = float(np.percentile(sim_returns, 95))
        result.max_drawdown_median = float(np.median(sim_drawdowns))

        # Pass if 100% survival
        result.passed = result.survival_rate_pct >= 100

        return result

    # -------------------------------------------------------------------
    # Test 5: Rolling Window (Quarterly)
    # -------------------------------------------------------------------
    def test_rolling_window(self, window_days: int = 90) -> RollingWindowResult:
        """
        Break data into quarterly (90-day) windows.
        Want ALL quarters profitable.
        """
        log.info("Running Rolling Window (quarterly) test...")
        result = RollingWindowResult()

        candles = self.bt.load_candles()
        if not candles:
            return result

        orig_start = self.bt.start_ms
        orig_end = self.bt.end_ms

        window_ms = window_days * 86_400_000
        start = candles[0].timestamp_ms
        end = candles[-1].timestamp_ms
        current = start
        quarter_num = 1

        while current < end:
            window_end = min(current + window_ms, end)

            self.bt.start_ms = current
            self.bt.end_ms = window_end

            report = self.bt.run(self._make_strategy())

            # Label
            from datetime import datetime, timezone
            start_dt = datetime.fromtimestamp(current / 1000, tz=timezone.utc)
            label = f"Q{quarter_num} ({start_dt.strftime('%Y-%m-%d')})"

            result.quarter_results.append((label, report))

            if report.total_return_pct > 0 and report.total_trades >= 3:
                result.profitable_quarters += 1

            result.total_quarters += 1
            current = window_end
            quarter_num += 1

        # Restore
        self.bt.start_ms = orig_start
        self.bt.end_ms = orig_end

        result.all_profitable = result.profitable_quarters == result.total_quarters
        # Pass if all quarters profitable (with min trades)
        result.passed = result.all_profitable

        return result

    # -------------------------------------------------------------------
    # Run All Tests
    # -------------------------------------------------------------------
    def run_all(
        self,
        param_ranges: Optional[dict[str, list]] = None,
        n_monte_carlo: int = 100,
        n_walk_windows: int = 5,
        quarter_days: int = 90,
    ) -> FullRobustnessResult:
        """
        Run all 5 robustness tests.

        Args:
            param_ranges: For parameter sensitivity test. If None, skips test.
            n_monte_carlo: Number of Monte Carlo simulations
            n_walk_windows: Number of walk-forward windows
            quarter_days: Days per rolling window
        """
        full = FullRobustnessResult()

        # 1. Out-of-Sample
        full.oos = self.test_out_of_sample()
        if full.oos.passed:
            full.tests_passed += 1

        # 2. Walk-Forward
        full.walk_forward = self.test_walk_forward(n_walk_windows)
        if full.walk_forward.passed:
            full.tests_passed += 1

        # 3. Parameter Sensitivity
        if param_ranges:
            full.param_sensitivity = self.test_param_sensitivity(param_ranges)
            if full.param_sensitivity.passed:
                full.tests_passed += 1
        else:
            log.info("Skipping parameter sensitivity (no param_ranges provided)")
            full.tests_total -= 1

        # 4. Monte Carlo
        full.monte_carlo = self.test_monte_carlo(n_sims=n_monte_carlo)
        if full.monte_carlo.passed:
            full.tests_passed += 1

        # 5. Rolling Window
        full.rolling_window = self.test_rolling_window(quarter_days)
        if full.rolling_window.passed:
            full.tests_passed += 1

        full.overall_pass = full.tests_passed == full.tests_total

        return full
