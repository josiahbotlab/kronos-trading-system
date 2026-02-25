#!/bin/bash
#
# Kronos Trading VPS Deployment Script
# =====================================
# Deploys updated Kronos code to Hetzner VPS and configures Coinbase API
#
# Usage:
#   ./deploy_to_vps.sh [COINBASE_API_KEY] [COINBASE_API_SECRET]
#
# Or set env vars:
#   export COINBASE_API_KEY=your_key
#   export COINBASE_API_SECRET=your_secret
#   ./deploy_to_vps.sh

set -e  # Exit on error

# Configuration
VPS_HOST="agent@100.113.94.124"
VPS_DIR="/home/agent/kronos-trading"
LOCAL_DIR="/Users/josiahgarcia/trading-bot/kronos-trading"
BACKUP_DIR="/home/agent/kronos-backups"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

echo -e "${BLUE}========================================${NC}"
echo -e "${BLUE}  Kronos Trading VPS Deployment${NC}"
echo -e "${BLUE}========================================${NC}"

# Get API credentials
if [ -n "$1" ] && [ -n "$2" ]; then
    COINBASE_KEY="$1"
    COINBASE_SECRET="$2"
elif [ -n "$COINBASE_API_KEY" ] && [ -n "$COINBASE_API_SECRET" ]; then
    COINBASE_KEY="$COINBASE_API_KEY"
    COINBASE_SECRET="$COINBASE_API_SECRET"
else
    echo -e "${RED}Error: Coinbase API credentials not provided${NC}"
    echo "Usage: $0 [COINBASE_API_KEY] [COINBASE_API_SECRET]"
    echo "Or set COINBASE_API_KEY and COINBASE_API_SECRET env vars"
    exit 1
fi

# Verify SSH connection
echo -e "\n${YELLOW}[1/7] Verifying SSH connection...${NC}"
if ! ssh -o ConnectTimeout=5 "$VPS_HOST" "echo 'SSH connection successful'" > /dev/null 2>&1; then
    echo -e "${RED}Error: Cannot connect to VPS at $VPS_HOST${NC}"
    echo "Please check your SSH configuration and VPS status"
    exit 1
fi
echo -e "${GREEN}✓ SSH connection verified${NC}"

# Create backup
echo -e "\n${YELLOW}[2/7] Creating backup of current deployment...${NC}"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
ssh "$VPS_HOST" "mkdir -p $BACKUP_DIR && cp -r $VPS_DIR $BACKUP_DIR/kronos-trading-$TIMESTAMP"
echo -e "${GREEN}✓ Backup created at $BACKUP_DIR/kronos-trading-$TIMESTAMP${NC}"

# Stop services
echo -e "\n${YELLOW}[3/7] Stopping systemd services...${NC}"
ssh "$VPS_HOST" "systemctl --user stop kronos-engine.service || true"
echo -e "${GREEN}✓ Services stopped${NC}"

# Upload updated code
echo -e "\n${YELLOW}[4/7] Uploading updated code...${NC}"
rsync -avz --delete \
    --exclude 'data/' \
    --exclude 'logs/' \
    --exclude '__pycache__/' \
    --exclude '*.pyc' \
    --exclude '.git/' \
    --exclude 'venv/' \
    "$LOCAL_DIR/" "$VPS_HOST:$VPS_DIR/"
echo -e "${GREEN}✓ Code uploaded${NC}"

# Update environment configuration
echo -e "\n${YELLOW}[5/7] Updating environment configuration...${NC}"
ssh "$VPS_HOST" bash << EOF
    # Create environment.d directory if it doesn't exist
    mkdir -p ~/.config/environment.d

    # Backup existing kronos.conf
    if [ -f ~/.config/environment.d/kronos.conf ]; then
        cp ~/.config/environment.d/kronos.conf ~/.config/environment.d/kronos.conf.backup
    fi

    # Read existing Telegram credentials if they exist
    if [ -f ~/.config/environment.d/kronos.conf ]; then
        # Extract existing Telegram vars
        EXISTING_TG_BOT=\$(grep '^KRONOS_TG_BOT_TOKEN=' ~/.config/environment.d/kronos.conf | cut -d'=' -f2- || echo "")
        EXISTING_TG_CHAT=\$(grep '^KRONOS_TG_CHAT_ID=' ~/.config/environment.d/kronos.conf | cut -d'=' -f2- || echo "")
    else
        EXISTING_TG_BOT=""
        EXISTING_TG_CHAT=""
    fi

    # Update or create kronos.conf with Coinbase credentials
    cat > ~/.config/environment.d/kronos.conf << ENVEOF
# Telegram notifications (preserved from existing config)
KRONOS_TG_BOT_TOKEN=\${EXISTING_TG_BOT}
KRONOS_TG_CHAT_ID=\${EXISTING_TG_CHAT}

# Coinbase API credentials
COINBASE_API_KEY=$COINBASE_KEY
COINBASE_API_SECRET=$COINBASE_SECRET

# Trading mode (paper trading enabled)
KRONOS_PAPER=true
ENVEOF

    echo "Environment configuration updated"
    echo "Telegram bot token: \${EXISTING_TG_BOT:0:20}..."
    echo "Telegram chat ID: \${EXISTING_TG_CHAT}"
EOF
echo -e "${GREEN}✓ Environment configured${NC}"

# Install/update dependencies
echo -e "\n${YELLOW}[6/7] Installing dependencies...${NC}"
ssh "$VPS_HOST" bash << 'EOF'
    cd /home/agent/kronos-trading
    pip3 install --user --upgrade pip > /dev/null 2>&1 || true
    pip3 install --user -r requirements.txt > /dev/null 2>&1 || true
EOF
echo -e "${GREEN}✓ Dependencies installed${NC}"

# Restart services and verify
echo -e "\n${YELLOW}[7/7] Starting services and verifying...${NC}"
ssh "$VPS_HOST" bash << 'EOF'
    # Copy systemd service file
    mkdir -p ~/.config/systemd/user
    cp /home/agent/kronos-trading/config/kronos-engine.service ~/.config/systemd/user/

    # Reload systemd
    systemctl --user daemon-reload

    # Enable and start the engine service
    systemctl --user enable kronos-engine.service
    systemctl --user start kronos-engine.service

    # Wait a moment for service to start
    sleep 3

    # Check service status
    if systemctl --user is-active --quiet kronos-engine.service; then
        echo "✓ kronos-engine.service is running"
    else
        echo "✗ kronos-engine.service failed to start"
        echo "Service logs:"
        journalctl --user -u kronos-engine.service -n 20 --no-pager
        exit 1
    fi
EOF

if [ $? -eq 0 ]; then
    echo -e "${GREEN}✓ Services started successfully${NC}"
else
    echo -e "${RED}✗ Service startup failed${NC}"
    exit 1
fi

# Display service status
echo -e "\n${BLUE}========================================${NC}"
echo -e "${GREEN}Deployment Complete!${NC}"
echo -e "${BLUE}========================================${NC}"
echo ""
echo "Service Status:"
ssh "$VPS_HOST" "systemctl --user status kronos-engine.service --no-pager -l | head -20"

echo ""
echo -e "${YELLOW}Next steps:${NC}"
echo "1. Monitor logs: ssh $VPS_HOST 'tail -f $VPS_DIR/logs/engine.log'"
echo "2. Check Telegram for startup notification"
echo "3. Verify paper trading is working"
echo ""
echo -e "${GREEN}Backup location: $BACKUP_DIR/kronos-trading-$TIMESTAMP${NC}"
