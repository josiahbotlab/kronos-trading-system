# Trading Bot Scheduler

Automatically starts the smart trading bot at market open (9:30 AM ET) every weekday.

## Quick Start

```bash
# Test run (starts immediately)
python src/scheduler/auto_start.py --test

# Normal mode (waits for 9:30 AM ET)
python src/scheduler/auto_start.py
```

## Background Execution

### Option 1: nohup (Simple)

Run the scheduler in the background with logging:

```bash
# Create logs directory
mkdir -p logs

# Start scheduler in background
nohup python src/scheduler/auto_start.py > logs/scheduler.log 2>&1 &

# Save the process ID
echo $! > logs/scheduler.pid
```

To stop:
```bash
kill $(cat logs/scheduler.pid)
```

To check if running:
```bash
ps aux | grep auto_start
```

To view logs:
```bash
tail -f logs/scheduler.log
```

### Option 2: launchd (macOS - Recommended)

Create a LaunchAgent for automatic startup on login.

1. Create the plist file:

```bash
cat > ~/Library/LaunchAgents/com.tradingbot.scheduler.plist << 'EOF'
<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0">
<dict>
    <key>Label</key>
    <string>com.tradingbot.scheduler</string>

    <key>ProgramArguments</key>
    <array>
        <string>/usr/bin/python3</string>
        <string>/Users/josiahgarcia/trading-bot/src/scheduler/auto_start.py</string>
    </array>

    <key>WorkingDirectory</key>
    <string>/Users/josiahgarcia/trading-bot</string>

    <key>RunAtLoad</key>
    <true/>

    <key>KeepAlive</key>
    <true/>

    <key>StandardOutPath</key>
    <string>/Users/josiahgarcia/trading-bot/logs/launchd_stdout.log</string>

    <key>StandardErrorPath</key>
    <string>/Users/josiahgarcia/trading-bot/logs/launchd_stderr.log</string>

    <key>EnvironmentVariables</key>
    <dict>
        <key>PATH</key>
        <string>/usr/local/bin:/usr/bin:/bin</string>
    </dict>
</dict>
</plist>
EOF
```

2. Load the agent:

```bash
# Create logs directory
mkdir -p ~/trading-bot/logs

# Load the launch agent
launchctl load ~/Library/LaunchAgents/com.tradingbot.scheduler.plist
```

3. Control commands:

```bash
# Start
launchctl start com.tradingbot.scheduler

# Stop
launchctl stop com.tradingbot.scheduler

# Unload (disable)
launchctl unload ~/Library/LaunchAgents/com.tradingbot.scheduler.plist

# Check status
launchctl list | grep tradingbot
```

4. View logs:

```bash
tail -f ~/trading-bot/logs/launchd_stdout.log
```

### Option 3: Screen/tmux (Interactive)

Keep the scheduler in a detachable terminal session:

```bash
# Using screen
screen -S tradingbot
python src/scheduler/auto_start.py
# Detach: Ctrl+A, then D
# Reattach: screen -r tradingbot

# Using tmux
tmux new -s tradingbot
python src/scheduler/auto_start.py
# Detach: Ctrl+B, then D
# Reattach: tmux attach -t tradingbot
```

## Configuration

Edit `auto_start.py` to change:

| Variable | Default | Description |
|----------|---------|-------------|
| `MARKET_OPEN` | `"09:30"` | Start time (Eastern) |
| `BOT_ARGS` | `--use-scanner --interval 30 --duration 390` | Bot arguments |

## Logs

All start/stop events are logged to:
- `csvs/scheduler_log.csv` - Event log (CSV format)
- `logs/scheduler.log` - Console output (when using nohup)

### Log CSV Format

| timestamp | event | details |
|-----------|-------|---------|
| 2026-01-15 09:30:00 | START | Args: --use-scanner --interval 30 --duration 390 |
| 2026-01-15 16:00:00 | STOP | Exit code: 0 |

## Timezone Note

The scheduler uses **local system time**. Ensure your system timezone is set correctly, or install `pytz` for explicit Eastern time handling:

```bash
pip install pytz
```

## Troubleshooting

**Bot doesn't start at 9:30:**
- Check system timezone matches Eastern Time
- Verify the scheduler is running: `ps aux | grep auto_start`

**Permission denied:**
- Make script executable: `chmod +x src/scheduler/auto_start.py`

**Module not found:**
- Activate your virtual environment before running
- Install dependencies: `pip install schedule termcolor`
