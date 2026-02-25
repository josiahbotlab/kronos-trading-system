#!/usr/bin/env python3
"""
Kronos Trading System - Main Runner
Ties together: Executor → Risk Manager → Portfolio → Monitor

Usage:
    # Health check only
    python run_kronos.py --check
    
    # Start paper trading (testnet)
    python run_kronos.py --paper
    
    # Dashboard only (no trading)
    python run_kronos.py --dashboard
    
    # Initialize strategy incubation
    python run_kronos.py --incubate cascade_p99 100
    
    # Emergency close all
    python run_kronos.py --close-all

    # Research - scan YouTube transcripts
    python run_kronos.py scan
    python run_kronos.py scan --scheduled --interval 12

    # Research - ingest manual transcripts
    python run_kronos.py ingest
    python run_kronos.py ingest --watch

    # Research - extract strategies from transcripts
    python run_kronos.py extract
    python run_kronos.py extract --list
    python run_kronos.py extract --summary

Requires: config/kronos.json (copy from kronos.json.example)
"""

import argparse
import json
import logging
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

from execution.coinbase_executor import CoinbaseExecutor
from execution.risk_manager import RiskManager
from execution.portfolio import PortfolioManager, Signal
from monitoring.incubation_tracker import IncubationTracker
from monitoring.dashboard import PerformanceMonitor


def setup_logging(verbose: bool = False):
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )


def load_config(config_path: str = "config/kronos.json") -> dict:
    path = Path(config_path)
    if not path.exists():
        print(f"❌ Config not found: {config_path}")
        print(f"   Copy config/kronos.json.example → config/kronos.json")
        print(f"   Add your Coinbase API credentials")
        sys.exit(1)
    with open(path) as f:
        return json.load(f)


def init_system(config_path: str = "config/kronos.json"):
    """Initialize all system components."""
    config = load_config(config_path)

    executor = CoinbaseExecutor.from_config(config_path)
    risk_mgr = RiskManager(config_path)
    portfolio = PortfolioManager(executor, risk_mgr)
    tracker = IncubationTracker()
    monitor = PerformanceMonitor(
        portfolio=portfolio,
        risk_manager=risk_mgr,
        incubation_tracker=tracker,
    )

    return executor, risk_mgr, portfolio, tracker, monitor


# ─── Commands ────────────────────────────────────────────────────

def cmd_health_check(config_path: str):
    """Run health check on all components."""
    print("\n🦞 Kronos Health Check")
    print("=" * 50)
    
    # Executor
    try:
        executor = CoinbaseExecutor.from_config(config_path)
        health = executor.health_check()
        if health["connected"]:
            env = "PAPER" if health["testnet"] else "LIVE"
            print(f"  ✅ Coinbase {env} connected")
            print(f"     Account:   {health['address']}")
            print(f"     Equity:    ${health['equity']:,.2f}")
            print(f"     Available: ${health['available']:,.2f}")
            print(f"     BTC Price: ${health.get('btc_price', 0):,.2f}")
            print(f"     Positions: {health['positions']}")
        else:
            print(f"  ❌ Coinbase: {health.get('error')}")
    except Exception as e:
        print(f"  ❌ Coinbase: {e}")
    
    # Risk Manager
    try:
        risk = RiskManager(config_path)
        summary = risk.get_risk_summary(0)
        print(f"  ✅ Risk Manager loaded")
        print(f"     Kill switch: {'ON' if summary['kill_switch'] else 'OFF'}")
        print(f"     Strategies: {summary['strategies']['total']}")
    except Exception as e:
        print(f"  ❌ Risk Manager: {e}")
    
    # Incubation Tracker
    try:
        tracker = IncubationTracker()
        dash = tracker.get_dashboard()
        print(f"  ✅ Incubation Tracker")
        print(f"     Strategies: {dash['total_strategies']}")
        print(f"     Budget:     ${dash['total_budget_allocated']}")
    except Exception as e:
        print(f"  ❌ Incubation Tracker: {e}")
    
    print()


def cmd_paper_trading(config_path: str, interval: int = 60):
    """Start paper trading loop with monitoring."""
    executor, risk_mgr, portfolio, tracker, monitor = init_system(config_path)
    
    print("\n🧪 Starting Kronos Paper Trading")
    print(f"   Interval: {interval}s")
    print(f"   Press Ctrl+C to stop\n")
    
    # The main loop monitors positions and manages trades
    # Actual signals come from strategies (Week 3) connected to this system
    monitor.run_loop(interval_seconds=interval, dashboard=True)


def cmd_dashboard(config_path: str):
    """Show dashboard without trading."""
    try:
        _, risk_mgr, portfolio, tracker, monitor = init_system(config_path)
        monitor.print_dashboard()
    except Exception as e:
        print(f"Error loading dashboard: {e}")
        # Fallback to incubation-only dashboard
        try:
            tracker = IncubationTracker()
            tracker.print_dashboard()
        except Exception:
            print("No data available yet.")


def cmd_incubate(config_path: str, strategy: str, budget: float, backtest_file: str = None):
    """Start incubating a strategy."""
    tracker = IncubationTracker()
    risk = RiskManager(config_path)
    
    # Load backtest metrics if provided
    bt_metrics = None
    if backtest_file:
        try:
            with open(backtest_file) as f:
                bt_metrics = json.load(f)
            print(f"  Loaded backtest: {backtest_file}")
        except Exception as e:
            print(f"  Warning: Could not load backtest: {e}")
    
    # Allocate in both systems
    risk.allocate_strategy(strategy, budget)
    tracker.start_incubation(strategy, budget, bt_metrics)
    
    print(f"\n✅ Strategy '{strategy}' now incubating with ${budget} budget")
    if bt_metrics:
        print(f"   Backtest return: {bt_metrics.get('return_pct', 'N/A')}%")
        print(f"   Backtest Sharpe: {bt_metrics.get('sharpe', 'N/A')}")
    print(f"   30-day evaluation period started")


def cmd_close_all(config_path: str):
    """Emergency close all positions."""
    confirm = input("⚠️ Close ALL positions? This cannot be undone. Type 'yes': ")
    if confirm.lower() != "yes":
        print("Cancelled.")
        return

    executor = CoinbaseExecutor.from_config(config_path)
    risk = RiskManager(config_path)
    portfolio = PortfolioManager(executor, risk)

    closed = portfolio.close_all()
    print(f"\n🚨 Closed {closed} positions")


def cmd_scan(config_path: str, scheduled: bool, interval: float, max_videos: int,
             cookies: str | None = None):
    """Scan YouTube for new transcripts."""
    from research.transcript_scanner import TranscriptScanner
    scanner = TranscriptScanner(config_path=Path(config_path))
    scanner.max_videos = max_videos
    if cookies:
        scanner.cookies_path = Path(cookies)

    if scheduled:
        scanner.run_scheduled(interval_hours=interval)
    else:
        results = scanner.scan_channel()
        print(f"\nScanned {len(results)} new transcript(s)")
        for r in results:
            tier_label = {1: "HIGH", 2: "MED", 3: "LOW"}.get(r["priority_tier"], "?")
            print(f"  [{tier_label}] {r['title']}")
    scanner.shutdown()


def cmd_ingest(config_path: str, watch: bool):
    """Ingest manual transcript files."""
    from research.manual_ingest import ManualIngestor
    ingestor = ManualIngestor()

    if watch:
        ingestor.run_watch()
    else:
        count = ingestor.ingest_all()
        print(f"\nIngested {count} manual transcript(s)")
    ingestor.shutdown()


def cmd_evaluate(config_path: str, full: bool, generate: bool, backtest: bool,
                  min_confidence: float):
    """Evaluate extracted strategies: compare, generate code, backtest, report."""
    from research.evaluate_extracted import main as eval_main
    import sys
    argv = []
    if full:
        argv.append("--full")
    elif backtest:
        argv.append("--backtest")
    elif generate:
        argv.append("--generate")
    argv.extend(["--min-confidence", str(min_confidence)])
    sys.argv = ["evaluate"] + argv
    eval_main()


def cmd_extract(config_path: str, list_strategies: bool, summary: bool, min_confidence: float):
    """Extract strategies from transcripts."""
    from research.strategy_extractor import StrategyExtractor
    extractor = StrategyExtractor()

    if list_strategies:
        strategies = extractor.get_extracted_strategies(min_confidence=min_confidence)
        if not strategies:
            print("No strategies found.")
        else:
            for s in strategies:
                print(f"  [{s['confidence']:.0%}] {s['strategy_name']} ({s['category']})")
                if s.get("description"):
                    print(f"        {s['description'][:100]}")
    elif summary:
        extractor.print_summary()
    else:
        count = extractor.process_all_pending()
        print(f"\nProcessed {count} transcript(s)")
    extractor.shutdown()


def cmd_status(config_path: str):
    """Quick status check."""
    try:
        executor = CoinbaseExecutor.from_config(config_path)
        state = executor.get_account_state()
        
        print(f"\n💰 Equity: ${state.equity:,.2f}  |  Available: ${state.available_balance:,.2f}")
        
        if state.positions:
            print(f"\n{'Coin':>6} {'Side':>6} {'Size':>10} {'Entry':>10} {'PnL':>10}")
            print("-" * 48)
            for p in state.positions:
                side = "LONG" if p.size > 0 else "SHORT"
                print(f"{p.coin:>6} {side:>6} {abs(p.size):>10.4f} "
                      f"${p.entry_px:>9,.2f} ${p.unrealized_pnl:>+9,.2f}")
        else:
            print("  No open positions")
        print()
        
    except Exception as e:
        print(f"Error: {e}")


# ─── Main ────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Kronos Trading System")
    parser.add_argument("--config", default="config/kronos.json", help="Config file path")
    parser.add_argument("--verbose", "-v", action="store_true", help="Verbose logging")
    
    sub = parser.add_subparsers(dest="command")
    
    sub.add_parser("check", help="Health check")
    sub.add_parser("status", help="Quick account status")
    
    paper = sub.add_parser("paper", help="Start paper trading")
    paper.add_argument("--interval", type=int, default=60, help="Check interval in seconds")
    
    sub.add_parser("dashboard", help="Show dashboard")
    
    incubate = sub.add_parser("incubate", help="Start strategy incubation")
    incubate.add_argument("strategy", help="Strategy name")
    incubate.add_argument("budget", type=float, help="Budget in USD")
    incubate.add_argument("--backtest", help="Path to backtest results JSON")
    
    sub.add_parser("close-all", help="Emergency close all positions")

    # Research commands
    scan = sub.add_parser("scan", help="Scan YouTube for new transcripts")
    scan.add_argument("--scheduled", action="store_true", help="Run on periodic schedule")
    scan.add_argument("--interval", type=float, default=24, help="Hours between scans")
    scan.add_argument("--max-videos", type=int, default=50, help="Max videos per scan")
    scan.add_argument("--cookies", default=None, help="Path to cookies.txt for YouTube auth")

    ingest = sub.add_parser("ingest", help="Ingest manual transcript files")
    ingest.add_argument("--watch", action="store_true", help="Watch directory continuously")

    extract = sub.add_parser("extract", help="Extract strategies from transcripts via LLM")
    extract.add_argument("--list", action="store_true", help="List extracted strategies")
    extract.add_argument("--summary", action="store_true", help="Print extraction summary")
    extract.add_argument("--min-confidence", type=float, default=0.5,
                         help="Minimum confidence for --list")

    evaluate = sub.add_parser("evaluate", help="Evaluate extracted strategies: compare, generate, backtest")
    evaluate.add_argument("--full", action="store_true", help="Full pipeline + Telegram report")
    evaluate.add_argument("--generate", action="store_true", help="Generate strategy code")
    evaluate.add_argument("--backtest", action="store_true", help="Generate + backtest + robustness")
    evaluate.add_argument("--min-confidence", type=float, default=0.7,
                          help="Minimum confidence threshold")

    args = parser.parse_args()
    setup_logging(args.verbose)
    
    if args.command == "check":
        cmd_health_check(args.config)
    elif args.command == "status":
        cmd_status(args.config)
    elif args.command == "paper":
        cmd_paper_trading(args.config, args.interval)
    elif args.command == "dashboard":
        cmd_dashboard(args.config)
    elif args.command == "incubate":
        cmd_incubate(args.config, args.strategy, args.budget, args.backtest)
    elif args.command == "close-all":
        cmd_close_all(args.config)
    elif args.command == "scan":
        cmd_scan(args.config, args.scheduled, args.interval, args.max_videos,
                args.cookies)
    elif args.command == "ingest":
        cmd_ingest(args.config, args.watch)
    elif args.command == "extract":
        cmd_extract(args.config, getattr(args, "list", False),
                    args.summary, args.min_confidence)
    elif args.command == "evaluate":
        cmd_evaluate(args.config, args.full, args.generate, args.backtest,
                     args.min_confidence)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
