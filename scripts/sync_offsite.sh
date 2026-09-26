#!/usr/bin/env bash
# Copy data/raw to an rclone remote (offsite copy of the raw archive).
#
#   WAVE_ANALYSIS_REMOTE=r2:wave-analysis-raw scripts/sync_offsite.sh [--dry-run]
#
# Raw files are append-only: pass 1 uses --immutable, so a file that already
# exists remotely with different content is reported as an error instead of
# being overwritten. Manifests (monthly CSVs that grow) are copied in pass 2.
# Nothing is ever deleted remotely ("copy", never "sync"). On success the time
# is written to data/raw/.offsite_last_sync, which the dashboard reports.
set -euo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
SRC="${WAVE_ANALYSIS_DATA:-$REPO/data}/raw"
: "${WAVE_ANALYSIS_REMOTE:?set WAVE_ANALYSIS_REMOTE, e.g. r2:wave-analysis-raw}"
DEST="$WAVE_ANALYSIS_REMOTE/raw"
COMMON=(--transfers 8 --checkers 16 --stats-one-line --stats 5m "$@")

rclone copy "$SRC" "$DEST" --immutable \
    --exclude "**/_manifests/**" --exclude "**/.archiver.lock" --exclude "**/.partial-*" \
    --exclude "/.offsite_last_sync" \
    "${COMMON[@]}"
rclone copy "$SRC" "$DEST" --include "**/_manifests/**" "${COMMON[@]}"
if [[ " $* " != *" --dry-run "* ]]; then
    date -u +%FT%TZ > "$SRC/.offsite_last_sync"
fi
echo "$(date -u +%FT%TZ) offsite copy to $DEST complete"
