#!/usr/bin/env python3
"""
Kronos Performance Metrics
==========================
All the math for evaluating trading strategies.
Used by the backtester and robustness test suite.

Metrics:
- Total return, CAGR
- Max drawdown, drawdown duration
- Sharpe ratio, Sortino ratio
- Win rate, profit factor
- Return/drawdown ratio
- Trade statistics
"""

import numpy as np
from dataclasses import dataclass, field
from typing import Optional


@dataclass
class Trade:
    """Single trade record."""
    entry_time: int          # timestamp ms
    exit_time: int           # timestamp ms
    symbol: str
    side: str                # "long" or "short"
    entry_price: float
    exit_price: float
    quantity: float
    pnl: float              # realized P&L in USD
    pnl_pct: float          # return as percentage
    fees: float = 0.0
    tag: str = ""            # strategy-specific label


@dataclass
class PerformanceReport:
    """Complete performance summary."""
    # Returns
    total_return_pct: float = 0.0
    total_return_usd: float = 0.0
    cagr_pct: float = 0.0

    # Risk
    max_drawdown_pct: float = 0.0
    max_drawdown_usd: float = 0.0
    max_drawdown_duration_hours: float = 0.0

    # Ratios
    sharpe_ratio: float = 0.0
    sortino_ratio: float = 0.0
    calmar_ratio: float = 0.0
    return_dd_ratio: float = 0.0
    profit_factor: float = 0.0

    # Trade stats
    total_trades: int = 0
    winning_trades: int = 0
    losing_trades: int = 0
    win_rate_pct: float = 0.0
    avg_win_pct: float = 0.0
    avg_loss_pct: float = 0.0
    avg_trade_pct: float = 0.0
    best_trade_pct: float = 0.0
    worst_trade_pct: float = 0.0
    avg_holding_hours: float = 0.0

    # Equity curve
    equity_curve: list = field(default_factory=list)
    drawdown_curve: list = field(default_factory=list)

    # Meta
    start_time: Optional[int] = None
    end_time: Optional[int] = None
    duration_days: float = 0.0
    initial_capital: float = 0.0

    def summary(self) -> str:
        """Pretty-print summary."""
        lines = [
            "=" * 50,
            "  PERFORMANCE REPORT",
            "=" * 50,
            f"  Period:           {self.duration_days:.0f} days",
            f"  Initial Capital:  ${self.initial_capital:,.0f}",
            "",
            f"  Total Return:     {self.total_return_pct:+.2f}% (${self.total_return_usd:+,.2f})",
            f"  CAGR:             {self.cagr_pct:.2f}%",
            f"  Max Drawdown:     {self.max_drawdown_pct:.2f}% (${self.max_drawdown_usd:,.2f})",
            f"  DD Duration:      {self.max_drawdown_duration_hours:.1f} hours",
            "",
            f"  Sharpe Ratio:     {self.sharpe_ratio:.2f}",
            f"  Sortino Ratio:    {self.sortino_ratio:.2f}",
            f"  Calmar Ratio:     {self.calmar_ratio:.2f}",
            f"  Return/DD:        {self.return_dd_ratio:.2f}x",
            f"  Profit Factor:    {self.profit_factor:.2f}",
            "",
            f"  Total Trades:     {self.total_trades}",
            f"  Win Rate:         {self.win_rate_pct:.1f}%",
            f"  Avg Win:          {self.avg_win_pct:+.2f}%",
            f"  Avg Loss:         {self.avg_loss_pct:+.2f}%",
            f"  Best Trade:       {self.best_trade_pct:+.2f}%",
            f"  Worst Trade:      {self.worst_trade_pct:+.2f}%",
            f"  Avg Holding:      {self.avg_holding_hours:.1f} hours",
            "=" * 50,
        ]
        return "\n".join(lines)


def calculate_metrics(
    trades: list[Trade],
    initial_capital: float = 10000.0,
    risk_free_rate: float = 0.0,
    periods_per_year: float = 365.25 * 24,  # hourly for crypto
) -> PerformanceReport:
    """
    Calculate full performance metrics from a list of trades.

    Args:
        trades: List of Trade objects (must be sorted by exit_time)
        initial_capital: Starting capital in USD
        risk_free_rate: Annual risk-free rate (default 0)
        periods_per_year: For annualization (default hourly for 24/7 crypto)
    """
    report = PerformanceReport(initial_capital=initial_capital)

    if not trades:
        return report

    # Sort trades by exit time
    trades = sorted(trades, key=lambda t: t.exit_time)

    report.start_time = trades[0].entry_time
    report.end_time = trades[-1].exit_time
    report.duration_days = (report.end_time - report.start_time) / (1000 * 86400)
    report.total_trades = len(trades)

    # --- Build equity curve ---
    equity = initial_capital
    peak = equity
    max_dd = 0.0
    max_dd_usd = 0.0
    dd_start_time = None
    max_dd_duration_ms = 0
    current_dd_start = None

    equity_curve = [(trades[0].entry_time, equity)]
    drawdown_curve = []

    for trade in trades:
        equity += trade.pnl
        equity_curve.append((trade.exit_time, equity))

        # Drawdown tracking
        if equity > peak:
            peak = equity
            if current_dd_start is not None:
                dd_duration = trade.exit_time - current_dd_start
                max_dd_duration_ms = max(max_dd_duration_ms, dd_duration)
            current_dd_start = None
        else:
            if current_dd_start is None:
                current_dd_start = trade.exit_time
            dd_pct = (peak - equity) / peak * 100
            dd_usd = peak - equity
            if dd_pct > max_dd:
                max_dd = dd_pct
                max_dd_usd = dd_usd
            drawdown_curve.append((trade.exit_time, -dd_pct))

    # Handle ongoing drawdown at end
    if current_dd_start is not None:
        dd_duration = trades[-1].exit_time - current_dd_start
        max_dd_duration_ms = max(max_dd_duration_ms, dd_duration)

    report.equity_curve = equity_curve
    report.drawdown_curve = drawdown_curve

    # --- Returns ---
    final_equity = equity
    report.total_return_usd = final_equity - initial_capital
    report.total_return_pct = (report.total_return_usd / initial_capital) * 100

    if report.duration_days > 1:
        years = report.duration_days / 365.25
        if years > 0.01 and final_equity > 0:
            try:
                report.cagr_pct = ((final_equity / initial_capital) ** (1 / years) - 1) * 100
            except (OverflowError, ValueError):
                report.cagr_pct = report.total_return_pct  # fallback

    # --- Drawdown ---
    report.max_drawdown_pct = max_dd
    report.max_drawdown_usd = max_dd_usd
    report.max_drawdown_duration_hours = max_dd_duration_ms / (1000 * 3600)

    # --- Trade statistics ---
    pnl_pcts = [t.pnl_pct for t in trades]
    winners = [t for t in trades if t.pnl > 0]
    losers = [t for t in trades if t.pnl <= 0]

    report.winning_trades = len(winners)
    report.losing_trades = len(losers)
    report.win_rate_pct = (len(winners) / len(trades)) * 100 if trades else 0

    if winners:
        report.avg_win_pct = np.mean([t.pnl_pct for t in winners])
    if losers:
        report.avg_loss_pct = np.mean([t.pnl_pct for t in losers])

    report.avg_trade_pct = np.mean(pnl_pcts) if pnl_pcts else 0
    report.best_trade_pct = max(pnl_pcts) if pnl_pcts else 0
    report.worst_trade_pct = min(pnl_pcts) if pnl_pcts else 0

    # Average holding time
    holding_hours = [(t.exit_time - t.entry_time) / (1000 * 3600) for t in trades]
    report.avg_holding_hours = np.mean(holding_hours) if holding_hours else 0

    # --- Profit Factor ---
    gross_profit = sum(t.pnl for t in winners)
    gross_loss = abs(sum(t.pnl for t in losers))
    report.profit_factor = gross_profit / gross_loss if gross_loss > 0 else float('inf')

    # --- Sharpe Ratio ---
    if len(pnl_pcts) > 1:
        returns_arr = np.array(pnl_pcts)
        mean_return = np.mean(returns_arr)
        std_return = np.std(returns_arr, ddof=1)

        # Estimate trades per year for annualization
        if report.duration_days > 0:
            trades_per_year = len(trades) / (report.duration_days / 365.25)
        else:
            trades_per_year = 252  # fallback

        if std_return > 0:
            report.sharpe_ratio = (mean_return / std_return) * np.sqrt(trades_per_year)

        # Sortino (only downside deviation)
        downside_returns = returns_arr[returns_arr < 0]
        if len(downside_returns) > 0:
            downside_std = np.std(downside_returns, ddof=1)
            if downside_std > 0:
                report.sortino_ratio = (mean_return / downside_std) * np.sqrt(trades_per_year)

    # --- Calmar Ratio ---
    if max_dd > 0 and report.duration_days >= 365:
        report.calmar_ratio = report.cagr_pct / max_dd

    # --- Return / Drawdown Ratio ---
    if max_dd > 0:
        report.return_dd_ratio = report.total_return_pct / max_dd

    return report


def compare_reports(reports: dict[str, PerformanceReport]) -> str:
    """Compare multiple strategy reports side by side."""
    if not reports:
        return "No reports to compare."

    names = list(reports.keys())
    header = f"{'Metric':25s}" + "".join(f"{n:>15s}" for n in names)
    sep = "-" * (25 + 15 * len(names))

    rows = [
        ("Total Return %",    lambda r: f"{r.total_return_pct:+.2f}%"),
        ("Max Drawdown %",    lambda r: f"{r.max_drawdown_pct:.2f}%"),
        ("Return/DD",         lambda r: f"{r.return_dd_ratio:.2f}x"),
        ("Sharpe",            lambda r: f"{r.sharpe_ratio:.2f}"),
        ("Sortino",           lambda r: f"{r.sortino_ratio:.2f}"),
        ("Profit Factor",     lambda r: f"{r.profit_factor:.2f}"),
        ("Win Rate",          lambda r: f"{r.win_rate_pct:.1f}%"),
        ("Total Trades",      lambda r: f"{r.total_trades}"),
        ("Avg Trade",         lambda r: f"{r.avg_trade_pct:+.2f}%"),
        ("Best Trade",        lambda r: f"{r.best_trade_pct:+.2f}%"),
        ("Worst Trade",       lambda r: f"{r.worst_trade_pct:+.2f}%"),
    ]

    lines = [header, sep]
    for label, fn in rows:
        line = f"{label:25s}" + "".join(f"{fn(reports[n]):>15s}" for n in names)
        lines.append(line)
    lines.append(sep)

    return "\n".join(lines)
