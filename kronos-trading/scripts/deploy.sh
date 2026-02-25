#!/bin/bash
# ============================================================================
# Kronos Trading - Deploy to VPS
# ============================================================================
# Pushes code to Hetzner VPS and sets up systemd services
#
# Usage:
#   ./deploy.sh              # Full deploy (code + services)
#   ./deploy.sh --code-only  # Just sync code
#   ./deploy.sh --status     # Check service status
# ============================================================================

set -euo pipefail

VPS_HOST="agent@100.113.94.124"
VPS_DIR="/home/agent/kronos-trading"
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"

# Colors
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m'

log() { echo -e "${GREEN}[KRONOS]${NC} $1"; }
warn() { echo -e "${YELLOW}[WARN]${NC} $1"; }
err() { echo -e "${RED}[ERROR]${NC} $1"; }

# ---------------------------------------------------------------------------
# Check status
# ---------------------------------------------------------------------------
check_status() {
    log "Checking Kronos service status..."
    ssh "$VPS_HOST" "
        echo '=== Liquidation Collector ==='
        systemctl --user status kronos-liquidations 2>/dev/null || echo 'Not installed'
        echo ''
        echo '=== Price Collector ==='
        systemctl --user status kronos-prices 2>/dev/null || echo 'Not installed'
        echo ''
        echo '=== Position Collector ==='
        systemctl --user status kronos-positions 2>/dev/null || echo 'Not installed'
        echo ''
        echo '=== Database Sizes ==='
        ls -lh $VPS_DIR/data/*.db 2>/dev/null || echo 'No databases yet'
    "
}

# ---------------------------------------------------------------------------
# Sync code
# ---------------------------------------------------------------------------
sync_code() {
    log "Syncing code to VPS..."
    rsync -avz --delete \
        --exclude 'data/' \
        --exclude '__pycache__/' \
        --exclude '*.pyc' \
        --exclude '.git/' \
        --exclude 'venv/' \
        "$PROJECT_DIR/" "$VPS_HOST:$VPS_DIR/"
    log "Code synced ✅"
}

# ---------------------------------------------------------------------------
# Install dependencies
# ---------------------------------------------------------------------------
install_deps() {
    log "Installing Python dependencies..."
    ssh "$VPS_HOST" "
        cd $VPS_DIR
        pip3 install --break-system-packages -r requirements.txt 2>&1 | tail -5
    "
    log "Dependencies installed ✅"
}

# ---------------------------------------------------------------------------
# Setup systemd services (user-level, no root needed)
# ---------------------------------------------------------------------------
setup_services() {
    log "Setting up systemd services..."

    ssh "$VPS_HOST" "
        # Enable lingering so user services run without login
        loginctl enable-linger agent 2>/dev/null || true

        # Create user systemd dir
        mkdir -p ~/.config/systemd/user/

        # Copy service files (modify for user-level: remove User= line)
        for svc in kronos-liquidations kronos-prices kronos-positions; do
            sed '/^User=/d' $VPS_DIR/scripts/\${svc}.service > ~/.config/systemd/user/\${svc}.service
        done

        # Reload and enable
        systemctl --user daemon-reload
        systemctl --user enable kronos-liquidations kronos-prices kronos-positions
    "
    log "Services configured ✅"
}

# ---------------------------------------------------------------------------
# Start/restart services
# ---------------------------------------------------------------------------
start_services() {
    log "Starting Kronos collectors..."
    ssh "$VPS_HOST" "
        systemctl --user restart kronos-liquidations
        systemctl --user restart kronos-prices
        systemctl --user restart kronos-positions
        sleep 2
        echo ''
        echo '=== Service Status ==='
        systemctl --user is-active kronos-liquidations && echo 'Liquidations: ✅ RUNNING' || echo 'Liquidations: ❌ FAILED'
        systemctl --user is-active kronos-prices && echo 'Prices: ✅ RUNNING' || echo 'Prices: ❌ FAILED'
        systemctl --user is-active kronos-positions && echo 'Positions: ✅ RUNNING' || echo 'Positions: ❌ FAILED'
    "
}

# ---------------------------------------------------------------------------
# View logs
# ---------------------------------------------------------------------------
view_logs() {
    local service="${1:-kronos-liquidations}"
    ssh "$VPS_HOST" "journalctl --user -u $service -n 50 --no-pager"
}

# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
case "${1:-deploy}" in
    --status|-s)
        check_status
        ;;
    --code-only|-c)
        sync_code
        ;;
    --logs|-l)
        view_logs "${2:-kronos-liquidations}"
        ;;
    --restart|-r)
        start_services
        ;;
    deploy|*)
        log "🚀 Full deployment starting..."
        sync_code
        install_deps
        setup_services
        start_services
        echo ""
        log "🎉 Kronos collectors deployed and running!"
        log "Check status: ./deploy.sh --status"
        log "View logs:    ./deploy.sh --logs [service-name]"
        ;;
esac
