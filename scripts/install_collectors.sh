#!/usr/bin/env bash
# Install the Wave_Analysis data collectors as systemd *user* timers.
#
#   scripts/install_collectors.sh            # buoycam (hourly) + NDBC realtime (1st/15th)
#   scripts/install_collectors.sh --sync     # also the daily offsite copy (needs rclone)
#   scripts/install_collectors.sh --remove   # stop and remove all units
#
# Units are rendered from deploy/systemd/*.{service,timer} with @REPO@ set to
# this checkout. Logs: journalctl --user -u wave-analysis-buoycam
# To keep timers running while you are logged out, run once:
#   loginctl enable-linger "$USER"
set -euo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
UNIT_DIR="${XDG_CONFIG_HOME:-$HOME/.config}/systemd/user"
TIMERS=(wave-analysis-buoycam wave-analysis-ndbc-realtime)

if [[ "${1:-}" == "--remove" ]]; then
    for u in wave-analysis-buoycam wave-analysis-ndbc-realtime wave-analysis-offsite-sync; do
        systemctl --user disable --now "$u.timer" 2>/dev/null || true
        rm -f "$UNIT_DIR/$u.service" "$UNIT_DIR/$u.timer"
    done
    systemctl --user daemon-reload
    echo "removed"
    exit 0
fi
if [[ "${1:-}" == "--sync" ]]; then
    command -v rclone >/dev/null || { echo "rclone not found; install it first" >&2; exit 1; }
    [[ -f "${XDG_CONFIG_HOME:-$HOME/.config}/wave-analysis/sync.env" ]] || {
        echo "create ~/.config/wave-analysis/sync.env with WAVE_ANALYSIS_REMOTE=<remote:bucket>" >&2
        exit 1
    }
    TIMERS+=(wave-analysis-offsite-sync)
fi

[[ -x "$REPO/.venv/bin/wave-analysis" ]] || { echo "run 'uv sync' in $REPO first" >&2; exit 1; }
mkdir -p "$UNIT_DIR"
for u in "${TIMERS[@]}"; do
    for ext in service timer; do
        sed "s|@REPO@|$REPO|g" "$REPO/deploy/systemd/$u.$ext" > "$UNIT_DIR/$u.$ext"
    done
done
systemctl --user daemon-reload
for u in "${TIMERS[@]}"; do
    systemctl --user enable --now "$u.timer"
done
systemctl --user list-timers 'wave-analysis-*' --no-pager
if [[ "$(loginctl show-user "$USER" -p Linger --value 2>/dev/null)" != "yes" ]]; then
    echo
    echo "Note: lingering is off, so timers stop when you log out."
    echo "      Enable it with: loginctl enable-linger $USER"
fi
