#!/bin/bash
# Kronos Engine Control
# Usage: ./engine.sh start|stop|status|logs|paper|deploy

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
VPS_HOST="agent@100.113.94.124"
VPS_DIR="/home/agent/kronos-trading"
SERVICE_NAME="kronos-engine"

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m'

case "$1" in
    paper)
        # Run paper trading locally (for testing)
        STRATEGY="${2:-liq_bb_combo}"
        SYMBOL="${3:-BTC/USDC:USDC}"
        TF="${4:-5m}"
        echo -e "${GREEN}[KRONOS]${NC} Starting paper trading: $STRATEGY on $SYMBOL ($TF)"
        cd "$PROJECT_DIR"
        python3 execution/live_engine.py \
            --strategy "$STRATEGY" \
            --symbol "$SYMBOL" \
            --timeframe "$TF" \
            --capital 1000
        ;;

    start)
        # Start engine on VPS
        STRATEGY="${2:-liq_bb_combo}"
        SYMBOL="${3:-BTC/USDC:USDC}"
        TF="${4:-5m}"
        echo -e "${GREEN}[KRONOS]${NC} Starting engine on VPS..."
        ssh "$VPS_HOST" "
            mkdir -p $VPS_DIR/logs
            cd $VPS_DIR
            # Update service with current params
            cat > ~/.config/systemd/user/$SERVICE_NAME.service << SEOF
[Unit]
Description=Kronos Paper Trading Engine
After=network.target

[Service]
Type=simple
WorkingDirectory=$VPS_DIR
ExecStart=/usr/bin/python3 execution/live_engine.py --strategy $STRATEGY --symbol $SYMBOL --timeframe $TF --capital 1000
Restart=always
RestartSec=30
StandardOutput=append:$VPS_DIR/logs/engine.log
StandardError=append:$VPS_DIR/logs/engine.log

[Install]
WantedBy=default.target
SEOF
            systemctl --user daemon-reload
            systemctl --user start $SERVICE_NAME
            systemctl --user enable $SERVICE_NAME
            echo 'Engine started'
            systemctl --user status $SERVICE_NAME --no-pager
        "
        ;;

    stop)
        echo -e "${YELLOW}[KRONOS]${NC} Stopping engine on VPS..."
        ssh "$VPS_HOST" "systemctl --user stop $SERVICE_NAME"
        echo -e "${GREEN}[KRONOS]${NC} Engine stopped"
        ;;

    restart)
        echo -e "${YELLOW}[KRONOS]${NC} Restarting engine on VPS..."
        ssh "$VPS_HOST" "systemctl --user restart $SERVICE_NAME"
        echo -e "${GREEN}[KRONOS]${NC} Engine restarted"
        ;;

    status)
        # Show both service status and trading status
        echo -e "${GREEN}[KRONOS]${NC} Engine status:"
        ssh "$VPS_HOST" "
            echo '--- Service ---'
            systemctl --user status $SERVICE_NAME --no-pager 2>/dev/null || echo 'Not running'
            echo ''
            echo '--- Trading ---'
            cd $VPS_DIR && python3 execution/live_engine.py --status 2>/dev/null || echo 'No trading data yet'
        "
        ;;

    logs)
        N="${2:-50}"
        echo -e "${GREEN}[KRONOS]${NC} Last $N log lines:"
        ssh "$VPS_HOST" "tail -n $N $VPS_DIR/logs/engine.log 2>/dev/null || journalctl --user -u $SERVICE_NAME -n $N --no-pager"
        ;;

    deploy)
        # Sync code and restart
        echo -e "${GREEN}[KRONOS]${NC} Deploying execution engine..."
        rsync -avz --exclude='*.db' --exclude='*.db-*' --exclude='logs/' --exclude='__pycache__/' \
            "$PROJECT_DIR/" "$VPS_HOST:$VPS_DIR/"
        echo -e "${GREEN}[KRONOS]${NC} Code synced. Restarting engine..."
        ssh "$VPS_HOST" "systemctl --user restart $SERVICE_NAME 2>/dev/null; echo 'Done'"
        ;;

    *)
        echo "Kronos Engine Control"
        echo ""
        echo "Usage: $0 <command> [args]"
        echo ""
        echo "Commands:"
        echo "  paper [strategy] [symbol] [tf]  - Run paper trading locally"
        echo "  start [strategy] [symbol] [tf]  - Start engine on VPS"
        echo "  stop                            - Stop engine on VPS"
        echo "  restart                         - Restart engine on VPS"
        echo "  status                          - Show engine status"
        echo "  logs [n]                        - Show last N log lines"
        echo "  deploy                          - Sync code + restart"
        echo ""
        echo "Examples:"
        echo "  $0 paper liq_bb_combo BTC/USDC:USDC 5m"
        echo "  $0 start cascade_ride,liq_bb_combo BTC/USDC:USDC 15m"
        echo "  $0 logs 100"
        ;;
esac
