# Systemd Timer Installation

SSH to server and run:

```bash
# Copy unit files
cp ~/stock-trading/systemd/stock-*.service ~/.config/systemd/user/
cp ~/stock-trading/systemd/stock-*.timer ~/.config/systemd/user/

# Reload and enable
systemctl --user daemon-reload
systemctl --user enable --now stock-skill-update.timer
systemctl --user enable --now stock-daily-summary.timer

# Verify
systemctl --user list-timers | grep stock
```

## Timers

| Timer | Schedule | Description |
|-------|----------|-------------|
| stock-skill-update | Mon-Fri 21:30 UTC (4:30 PM ET) | Analyzes trades, updates skill file |
| stock-daily-summary | Mon-Fri 21:00 UTC (4:00 PM ET) | Sends daily Telegram summary |
| stock-bots | Mon-Fri 14:30 UTC (9:30 AM ET) | Starts trading bots (existing) |
