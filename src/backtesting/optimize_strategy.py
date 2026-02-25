"""
Strategy Parameter Optimizer

Moon Dev's optimization approach: Test all combinations to find the sweet spot.
- Tests Take Profit: 2%, 3%, 4%, 5%, 6%
- Tests Stop Loss: 1%, 1.5%, 2%, 2.5%, 3%
- Tests ATR Multiplier: 1.0, 1.5, 2.0, 2.5
- Generates heatmap of results

Usage:
    python src/backtesting/optimize_strategy.py
    python src/backtesting/optimize_strategy.py --symbol TSLA
    python src/backtesting/optimize_strategy.py --symbol NVDA --days 180
"""

import argparse
import os
import sys
from datetime import datetime, timedelta
from pathlib import Path
from itertools import product

import pandas as pd
import numpy as np
from dotenv import load_dotenv
from termcolor import cprint

sys.path.append('/Users/josiahgarcia/trading-bot')
load_dotenv()

# Backtesting library
try:
    from backtesting import Backtest, Strategy
    BACKTESTING_AVAILABLE = True
except ImportError:
    BACKTESTING_AVAILABLE = False
    cprint("Error: backtesting.py not installed. Run: pip install backtesting", "red")

# Alpaca imports
try:
    import alpaca_trade_api as tradeapi
    ALPACA_AVAILABLE = True
except ImportError:
    ALPACA_AVAILABLE = False

# Bokeh for heatmap
try:
    from bokeh.plotting import figure, output_file, save
    from bokeh.models import ColumnDataSource, LinearColorMapper, ColorBar, BasicTicker
    from bokeh.models import HoverTool, Title
    from bokeh.palettes import RdYlGn11
    from bokeh.transform import transform
    from bokeh.layouts import column
    BOKEH_AVAILABLE = True
except ImportError:
    BOKEH_AVAILABLE = False
    cprint("Warning: bokeh not available for heatmap", "yellow")

# Configuration
ALPACA_API_KEY = os.getenv("ALPACA_API_KEY")
ALPACA_SECRET_KEY = os.getenv("ALPACA_SECRET_KEY")
ALPACA_PAPER = os.getenv("ALPACA_PAPER", "true").lower() == "true"

# Optimization parameters
TAKE_PROFIT_RANGE = [2, 3, 4, 5, 6]           # Percentage
STOP_LOSS_RANGE = [1, 1.5, 2, 2.5, 3]         # Percentage
ATR_MULTIPLIER_RANGE = [1.0, 1.5, 2.0, 2.5]   # ATR multiplier

# Fixed parameters
ATR_PERIOD = 14
LOOKBACK_PERIOD = 24

# Paths
CSV_DIR = Path(__file__).parent.parent.parent / 'csvs'
HEATMAP_PATH = CSV_DIR / 'optimization_heatmap.html'
RESULTS_CSV_PATH = CSV_DIR / 'optimization_results.csv'


def fetch_historical_data(symbol='TSLA', days=365):
    """Fetch historical hourly data from Alpaca."""
    if not ALPACA_AVAILABLE:
        raise RuntimeError("alpaca-trade-api not installed")

    if not ALPACA_API_KEY or not ALPACA_SECRET_KEY:
        raise RuntimeError("ALPACA_API_KEY and ALPACA_SECRET_KEY required")

    base_url = "https://paper-api.alpaca.markets" if ALPACA_PAPER else "https://api.alpaca.markets"
    api = tradeapi.REST(ALPACA_API_KEY, ALPACA_SECRET_KEY, base_url, api_version='v2')

    end_date = datetime.now() - timedelta(days=15)  # Avoid SIP restrictions
    start_date = end_date - timedelta(days=days)

    start_str = start_date.strftime('%Y-%m-%d')
    end_str = end_date.strftime('%Y-%m-%d')

    cprint(f"Fetching {symbol} data from {start_str} to {end_str}...", "cyan")

    bars = api.get_bars(
        symbol,
        tradeapi.TimeFrame.Hour,
        start=start_str,
        end=end_str,
        limit=10000,
        feed='iex'
    ).df

    if bars.empty:
        raise ValueError(f"No data returned for {symbol}")

    bars = bars.reset_index()
    bars = bars.rename(columns={
        'timestamp': 'Date',
        'open': 'Open',
        'high': 'High',
        'low': 'Low',
        'close': 'Close',
        'volume': 'Volume'
    })

    bars['Date'] = pd.to_datetime(bars['Date'])
    bars = bars.set_index('Date')
    bars.index = bars.index.tz_localize(None)

    cprint(f"Fetched {len(bars)} hourly bars", "green")

    return bars[['Open', 'High', 'Low', 'Close', 'Volume']]


def calculate_atr(high, low, close, period=ATR_PERIOD):
    """Calculate Average True Range."""
    high = pd.Series(high)
    low = pd.Series(low)
    close = pd.Series(close)

    tr1 = high - low
    tr2 = abs(high - close.shift(1))
    tr3 = abs(low - close.shift(1))

    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr = tr.rolling(window=period).mean()

    return atr


class OptimizableBreakoutStrategy(Strategy):
    """
    Breakout strategy with optimizable parameters.
    """
    # Optimizable parameters
    take_profit_pct = 4.0    # Take profit percentage
    stop_loss_pct = 2.0      # Stop loss percentage
    atr_multiplier = 1.5     # ATR multiplier for dynamic stops

    # Fixed parameters
    lookback = LOOKBACK_PERIOD
    use_pct_stops = True     # Use percentage-based stops instead of ATR

    def init(self):
        """Initialize indicators."""
        self.rolling_high = self.I(
            lambda x: pd.Series(x).rolling(self.lookback).max(),
            self.data.High,
            name='Rolling High'
        )
        self.rolling_low = self.I(
            lambda x: pd.Series(x).rolling(self.lookback).min(),
            self.data.Low,
            name='Rolling Low'
        )
        self.atr = self.I(
            calculate_atr,
            self.data.High,
            self.data.Low,
            self.data.Close,
            ATR_PERIOD,
            name='ATR'
        )

    def next(self):
        """Execute strategy logic."""
        price = self.data.Close[-1]
        atr = self.atr[-1]
        high_24h = self.rolling_high[-2] if len(self.rolling_high) > 1 else self.rolling_high[-1]

        if pd.isna(atr) or atr <= 0:
            return

        # Exit logic for existing positions
        if self.position:
            return  # Let SL/TP handle exits

        # Entry logic
        if not self.position:
            buffer = high_24h * 0.001

            if price > high_24h + buffer:
                # Calculate stops based on percentage or ATR
                if self.use_pct_stops:
                    stop_loss = price * (1 - self.stop_loss_pct / 100)
                    take_profit = price * (1 + self.take_profit_pct / 100)
                else:
                    stop_distance = atr * self.atr_multiplier
                    stop_loss = price - stop_distance
                    take_profit = price + (stop_distance * (self.take_profit_pct / self.stop_loss_pct))

                self.buy(sl=stop_loss, tp=take_profit)


def run_single_backtest(data, tp_pct, sl_pct, atr_mult, cash=100000):
    """Run a single backtest with given parameters."""
    bt = Backtest(
        data,
        OptimizableBreakoutStrategy,
        cash=cash,
        commission=0.001,
        exclusive_orders=True,
        trade_on_close=True
    )

    stats = bt.run(
        take_profit_pct=tp_pct,
        stop_loss_pct=sl_pct,
        atr_multiplier=atr_mult
    )

    return {
        'take_profit_pct': tp_pct,
        'stop_loss_pct': sl_pct,
        'atr_multiplier': atr_mult,
        'return_pct': stats['Return [%]'],
        'win_rate': stats['Win Rate [%]'],
        'profit_factor': stats.get('Profit Factor', 0),
        'max_drawdown': stats['Max. Drawdown [%]'],
        'num_trades': stats['# Trades'],
        'sharpe_ratio': stats.get('Sharpe Ratio', 0),
        'final_equity': stats['Equity Final [$]']
    }


def run_optimization(symbol='TSLA', days=365, cash=100000):
    """
    Run full parameter optimization.
    """
    if not BACKTESTING_AVAILABLE:
        raise RuntimeError("backtesting.py not installed")

    # Fetch data
    data = fetch_historical_data(symbol, days)

    # Generate all parameter combinations
    combinations = list(product(
        TAKE_PROFIT_RANGE,
        STOP_LOSS_RANGE,
        ATR_MULTIPLIER_RANGE
    ))

    total = len(combinations)
    cprint(f"\nRunning {total} parameter combinations...\n", "cyan")

    results = []

    for i, (tp, sl, atr_mult) in enumerate(combinations):
        if (i + 1) % 20 == 0 or i == 0:
            cprint(f"Progress: {i + 1}/{total} ({(i+1)/total*100:.0f}%)", "white")

        try:
            result = run_single_backtest(data, tp, sl, atr_mult, cash)
            results.append(result)
        except Exception as e:
            cprint(f"Error with TP={tp}, SL={sl}, ATR={atr_mult}: {e}", "red")

    return pd.DataFrame(results)


def create_heatmap(results_df, metric='return_pct', atr_mult=1.5):
    """
    Create a heatmap of TP vs SL performance.
    """
    if not BOKEH_AVAILABLE:
        cprint("Bokeh not available, skipping heatmap", "yellow")
        return

    # Filter for specific ATR multiplier
    filtered = results_df[results_df['atr_multiplier'] == atr_mult].copy()

    if filtered.empty:
        cprint(f"No results for ATR multiplier {atr_mult}", "yellow")
        return

    # Create pivot table
    pivot = filtered.pivot_table(
        values=metric,
        index='stop_loss_pct',
        columns='take_profit_pct',
        aggfunc='mean'
    )

    # Prepare data for bokeh
    sl_values = [str(x) for x in pivot.index.tolist()]
    tp_values = [str(x) for x in pivot.columns.tolist()]

    # Create data source
    data_list = []
    for sl in pivot.index:
        for tp in pivot.columns:
            val = pivot.loc[sl, tp]
            data_list.append({
                'stop_loss': str(sl),
                'take_profit': str(tp),
                'value': val if not pd.isna(val) else 0
            })

    source = ColumnDataSource(pd.DataFrame(data_list))

    # Color mapper
    values = [d['value'] for d in data_list]
    min_val = min(values)
    max_val = max(values)

    # Ensure we have a valid range
    if min_val == max_val:
        min_val -= 1
        max_val += 1

    mapper = LinearColorMapper(
        palette=RdYlGn11,
        low=min_val,
        high=max_val
    )

    # Create figure
    output_file(str(HEATMAP_PATH))

    p = figure(
        title=f"Breakout Strategy Optimization - Return % (ATR Mult: {atr_mult}x)",
        x_range=tp_values,
        y_range=list(reversed(sl_values)),
        x_axis_label="Take Profit %",
        y_axis_label="Stop Loss %",
        width=700,
        height=500,
        tools="hover,save,reset",
        toolbar_location="above"
    )

    # Add title
    p.add_layout(Title(text="Moon Dev's Parameter Optimization", text_font_style="italic"), 'above')

    # Create heatmap rectangles
    p.rect(
        x='take_profit',
        y='stop_loss',
        width=1,
        height=1,
        source=source,
        fill_color=transform('value', mapper),
        line_color=None
    )

    # Add hover tool
    hover = p.select_one(HoverTool)
    hover.tooltips = [
        ('Take Profit', '@take_profit%'),
        ('Stop Loss', '@stop_loss%'),
        ('Return', '@value{0.2f}%')
    ]

    # Add color bar
    color_bar = ColorBar(
        color_mapper=mapper,
        ticker=BasicTicker(),
        label_standoff=12,
        border_line_color=None,
        location=(0, 0),
        title="Return %"
    )
    p.add_layout(color_bar, 'right')

    # Style
    p.xaxis.major_label_text_font_size = "12pt"
    p.yaxis.major_label_text_font_size = "12pt"
    p.title.text_font_size = "14pt"

    save(p)
    cprint(f"\nHeatmap saved to: {HEATMAP_PATH}", "green")


def create_multi_heatmaps(results_df):
    """Create heatmaps for each ATR multiplier."""
    if not BOKEH_AVAILABLE:
        cprint("Bokeh not available, skipping heatmaps", "yellow")
        return

    from bokeh.layouts import gridplot

    plots = []

    for atr_mult in ATR_MULTIPLIER_RANGE:
        filtered = results_df[results_df['atr_multiplier'] == atr_mult].copy()

        if filtered.empty:
            continue

        pivot = filtered.pivot_table(
            values='return_pct',
            index='stop_loss_pct',
            columns='take_profit_pct',
            aggfunc='mean'
        )

        sl_values = [str(x) for x in pivot.index.tolist()]
        tp_values = [str(x) for x in pivot.columns.tolist()]

        data_list = []
        for sl in pivot.index:
            for tp in pivot.columns:
                val = pivot.loc[sl, tp]
                data_list.append({
                    'stop_loss': str(sl),
                    'take_profit': str(tp),
                    'value': val if not pd.isna(val) else 0
                })

        source = ColumnDataSource(pd.DataFrame(data_list))

        values = [d['value'] for d in data_list]
        min_val = min(values) if values else -10
        max_val = max(values) if values else 10

        if min_val == max_val:
            min_val -= 1
            max_val += 1

        mapper = LinearColorMapper(palette=RdYlGn11, low=min_val, high=max_val)

        p = figure(
            title=f"ATR Multiplier: {atr_mult}x",
            x_range=tp_values,
            y_range=list(reversed(sl_values)),
            x_axis_label="Take Profit %",
            y_axis_label="Stop Loss %",
            width=400,
            height=350,
            tools="hover,save",
        )

        p.rect(
            x='take_profit',
            y='stop_loss',
            width=1,
            height=1,
            source=source,
            fill_color=transform('value', mapper),
            line_color=None
        )

        hover = p.select_one(HoverTool)
        hover.tooltips = [
            ('TP', '@take_profit%'),
            ('SL', '@stop_loss%'),
            ('Return', '@value{0.2f}%')
        ]

        color_bar = ColorBar(
            color_mapper=mapper,
            ticker=BasicTicker(),
            label_standoff=8,
            border_line_color=None,
            location=(0, 0),
            width=8
        )
        p.add_layout(color_bar, 'right')

        plots.append(p)

    # Arrange in grid
    output_file(str(HEATMAP_PATH))
    grid = gridplot([plots[:2], plots[2:]], merge_tools=False)
    save(grid)
    cprint(f"\nMulti-heatmap saved to: {HEATMAP_PATH}", "green")


def print_results(results_df, symbol):
    """Print optimization results summary."""
    cprint("\n" + "=" * 70, "cyan")
    cprint(f"  OPTIMIZATION RESULTS - {symbol}", "cyan")
    cprint("=" * 70, "cyan")

    # Best overall parameters
    best = results_df.loc[results_df['return_pct'].idxmax()]

    cprint(f"\n{'BEST PARAMETERS':-^70}", "green")
    cprint(f"  Take Profit:     {best['take_profit_pct']}%", "green")
    cprint(f"  Stop Loss:       {best['stop_loss_pct']}%", "green")
    cprint(f"  ATR Multiplier:  {best['atr_multiplier']}x", "green")
    cprint(f"\n  Return:          {best['return_pct']:+.2f}%", "green")
    cprint(f"  Win Rate:        {best['win_rate']:.1f}%", "white")
    cprint(f"  Profit Factor:   {best['profit_factor']:.2f}", "white")
    cprint(f"  Max Drawdown:    {best['max_drawdown']:.2f}%", "white")
    cprint(f"  Trades:          {best['num_trades']:.0f}", "white")
    cprint(f"  Sharpe Ratio:    {best['sharpe_ratio']:.2f}", "white")

    # Worst parameters
    worst = results_df.loc[results_df['return_pct'].idxmin()]

    cprint(f"\n{'WORST PARAMETERS':-^70}", "red")
    cprint(f"  Take Profit:     {worst['take_profit_pct']}%", "red")
    cprint(f"  Stop Loss:       {worst['stop_loss_pct']}%", "red")
    cprint(f"  ATR Multiplier:  {worst['atr_multiplier']}x", "red")
    cprint(f"  Return:          {worst['return_pct']:+.2f}%", "red")

    # Top 10 combinations
    cprint(f"\n{'TOP 10 PARAMETER COMBINATIONS':-^70}", "yellow")
    top10 = results_df.nlargest(10, 'return_pct')

    cprint(f"{'TP%':<6} {'SL%':<6} {'ATR':<6} {'Return':<10} {'WinRate':<10} {'PF':<8} {'Trades':<8}", "white")
    cprint("-" * 70, "white")

    for _, row in top10.iterrows():
        color = "green" if row['return_pct'] > 0 else "red"
        cprint(
            f"{row['take_profit_pct']:<6} {row['stop_loss_pct']:<6} {row['atr_multiplier']:<6} "
            f"{row['return_pct']:>+8.2f}% {row['win_rate']:>8.1f}% {row['profit_factor']:>6.2f} {row['num_trades']:>6.0f}",
            color
        )

    # Summary by ATR multiplier
    cprint(f"\n{'SUMMARY BY ATR MULTIPLIER':-^70}", "cyan")

    for atr_mult in ATR_MULTIPLIER_RANGE:
        subset = results_df[results_df['atr_multiplier'] == atr_mult]
        avg_return = subset['return_pct'].mean()
        best_return = subset['return_pct'].max()
        color = "green" if avg_return > 0 else "red"
        cprint(f"  ATR {atr_mult}x: Avg Return {avg_return:+.2f}% | Best {best_return:+.2f}%", color)

    cprint("\n" + "=" * 70, "cyan")


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description="Optimize Breakout Strategy Parameters")
    parser.add_argument("--symbol", type=str, default="TSLA", help="Symbol to optimize (default: TSLA)")
    parser.add_argument("--days", type=int, default=365, help="Days of history (default: 365)")
    parser.add_argument("--cash", type=int, default=100000, help="Starting cash (default: 100000)")
    args = parser.parse_args()

    cprint("\n" + "=" * 70, "cyan")
    cprint("  MOON DEV'S PARAMETER OPTIMIZER", "cyan")
    cprint("=" * 70, "cyan")
    cprint(f"  Symbol: {args.symbol}", "white")
    cprint(f"  Period: {args.days} days", "white")
    cprint(f"  Take Profit Range: {TAKE_PROFIT_RANGE}%", "white")
    cprint(f"  Stop Loss Range: {STOP_LOSS_RANGE}%", "white")
    cprint(f"  ATR Multipliers: {ATR_MULTIPLIER_RANGE}", "white")

    try:
        # Run optimization
        results_df = run_optimization(
            symbol=args.symbol,
            days=args.days,
            cash=args.cash
        )

        # Save results to CSV
        CSV_DIR.mkdir(parents=True, exist_ok=True)
        results_df.to_csv(RESULTS_CSV_PATH, index=False)
        cprint(f"\nResults saved to: {RESULTS_CSV_PATH}", "green")

        # Print summary
        print_results(results_df, args.symbol)

        # Create heatmaps
        create_multi_heatmaps(results_df)

        return results_df

    except Exception as e:
        cprint(f"\nError running optimization: {e}", "red")
        import traceback
        traceback.print_exc()
        return None


if __name__ == "__main__":
    main()
