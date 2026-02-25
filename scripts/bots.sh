#!/bin/bash
# ─────────────────────────────────────────────────────────────
# Trading Bot Manager — load, unload, status for all bots
#
# Usage:
#   ./scripts/bots.sh status          # Show all bot statuses
#   ./scripts/bots.sh load            # Load all bots
#   ./scripts/bots.sh unload          # Unload all bots
#   ./scripts/bots.sh load momentum   # Load single bot
#   ./scripts/bots.sh unload breakout # Unload single bot
#   ./scripts/bots.sh logs momentum   # Tail log for a bot
#   ./scripts/bots.sh restart         # Unload + load all
# ─────────────────────────────────────────────────────────────

LAUNCH_DIR="$HOME/Library/LaunchAgents"
LOG_DIR="$HOME/trading-bot/logs"

# Ordered list of bots
ALL_BOTS="smartbot gapgo momentum breakout meanrev macd bbbounce analyzer"

# Map bot name -> launchd label
_label() {
    case "$1" in
        smartbot)  echo "com.tradingbot.smartbot" ;;
        gapgo)     echo "com.tradingbot.gapgo" ;;
        momentum)  echo "com.tradingbot.momentum" ;;
        breakout)  echo "com.tradingbot.breakout" ;;
        meanrev)   echo "com.tradingbot.meanrev" ;;
        macd)      echo "com.tradingbot.macd" ;;
        bbbounce)  echo "com.tradingbot.bbbounce" ;;
        analyzer)  echo "com.tradingbot.analyzer" ;;
        *)         echo "" ;;
    esac
}

# Map bot name -> log file
_logfile() {
    case "$1" in
        smartbot)  echo "$LOG_DIR/smart_bot.log" ;;
        gapgo)     echo "$LOG_DIR/gapgo_bot.log" ;;
        momentum)  echo "$LOG_DIR/momentum_bot.log" ;;
        breakout)  echo "$LOG_DIR/breakout_bot.log" ;;
        meanrev)   echo "$LOG_DIR/mean_reversion_bot.log" ;;
        macd)      echo "$LOG_DIR/macd_bot.log" ;;
        bbbounce)  echo "$LOG_DIR/bb_bounce_bot.log" ;;
        analyzer)  echo "$LOG_DIR/analyzer.log" ;;
        *)         echo "" ;;
    esac
}

_plist_path() {
    local label
    label=$(_label "$1")
    echo "$LAUNCH_DIR/${label}.plist"
}

_is_loaded() {
    local label
    label=$(_label "$1")
    launchctl list 2>/dev/null | grep -q "$label"
}

_status() {
    local name="$1"
    local label
    label=$(_label "$name")
    if _is_loaded "$name"; then
        local pid
        pid=$(launchctl list 2>/dev/null | grep "$label" | awk '{print $1}')
        if [[ "$pid" == "-" || -z "$pid" ]]; then
            printf "  %-12s  LOADED (idle)     %s\n" "$name" "$label"
        else
            printf "  %-12s  RUNNING (PID %s)  %s\n" "$name" "$pid" "$label"
        fi
    else
        printf "  %-12s  STOPPED           %s\n" "$name" "$label"
    fi
}

_load() {
    local name="$1"
    local plist
    plist=$(_plist_path "$name")
    if [[ ! -f "$plist" ]]; then
        echo "  ERROR: $plist not found"
        return 1
    fi
    if _is_loaded "$name"; then
        echo "  $name already loaded"
        return 0
    fi
    launchctl load "$plist"
    echo "  Loaded $name"
}

_unload() {
    local name="$1"
    local plist
    plist=$(_plist_path "$name")
    if ! _is_loaded "$name"; then
        echo "  $name not loaded"
        return 0
    fi
    launchctl unload "$plist"
    echo "  Unloaded $name"
}

_validate_target() {
    local target="$1"
    if [[ "$target" == "all" ]]; then
        return 0
    fi
    local label
    label=$(_label "$target")
    if [[ -z "$label" ]]; then
        echo "Unknown bot: $target" >&2
        echo "Available: $ALL_BOTS" >&2
        exit 1
    fi
}

_get_targets() {
    if [[ "$1" == "all" ]]; then
        echo $ALL_BOTS
    else
        echo "$1"
    fi
}

ACTION="${1:-status}"
TARGET="${2:-all}"

_validate_target "$TARGET"

case "$ACTION" in
    status)
        echo ""
        echo "  ════════════════════════════════════════════════════════════"
        echo "  TRADING BOT STATUS"
        echo "  ════════════════════════════════════════════════════════════"
        for name in $ALL_BOTS; do
            _status "$name"
        done
        echo "  ════════════════════════════════════════════════════════════"
        echo ""
        ;;

    load)
        echo "Loading bots..."
        for name in $(_get_targets "$TARGET"); do
            _load "$name"
        done
        ;;

    unload)
        echo "Unloading bots..."
        for name in $(_get_targets "$TARGET"); do
            _unload "$name"
        done
        ;;

    restart)
        echo "Restarting bots..."
        for name in $(_get_targets "$TARGET"); do
            _unload "$name"
        done
        sleep 1
        for name in $(_get_targets "$TARGET"); do
            _load "$name"
        done
        ;;

    logs)
        if [[ "$TARGET" == "all" ]]; then
            echo "Specify a bot name: ./scripts/bots.sh logs momentum"
            exit 1
        fi
        log_file=$(_logfile "$TARGET")
        if [[ -z "$log_file" ]]; then
            echo "Unknown bot: $TARGET"
            exit 1
        fi
        if [[ ! -f "$log_file" ]]; then
            echo "No log file yet: $log_file"
            exit 1
        fi
        echo "Tailing $log_file (Ctrl+C to stop)"
        tail -f "$log_file"
        ;;

    *)
        echo "Usage: $0 {status|load|unload|restart|logs} [bot_name|all]"
        echo ""
        echo "Bots: $ALL_BOTS"
        exit 1
        ;;
esac
