#!/usr/bin/env bash
# Exchange data/raw with an rclone remote: the offsite copy of the raw archive,
# and the way several collector hosts share one archive.
#
#   WAVE_ANALYSIS_REMOTE=r2:wave-analysis-raw scripts/sync_offsite.sh [--dry-run]
#
# Pushes local files to the remote, then pulls the files other hosts pushed.
# Raw files are append-only: they are copied with --immutable, so a file that
# exists on both sides with a different size is reported as an error instead
# of being overwritten. They are compared by size only, because two hosts
# fetch the same image at different times (and reading S3 modification times
# costs a request per object); `rclone check` compares hashes. Ledgers
# (_manifests/*.csv) grow, so they are copied with --update: every ledger has
# a single writer (buoycam ledgers are named per host), and an older copy
# never replaces a newer one. Nothing is ever deleted on either side ("copy",
# never "sync"). On success the time is written to data/raw/.offsite_last_sync,
# which the dashboard reports.
#
# A collection may live on another drive behind a symlink (e.g. data/raw/
# webcoos_timex -> /mnt/Warehouse/...): --copy-links follows it in both
# directions. If that drive is not mounted, the exchange fails instead of
# skipping the collection.
#
# A host without room for some collections can leave them out of the pull
# (its own files in them are still pushed), e.g. in sync.env on a laptop:
#   WAVE_ANALYSIS_SYNC_PULL_SKIP="webcoos webcoos_timex webcoos_brt"
set -euo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
SRC="${WAVE_ANALYSIS_DATA:-$REPO/data}/raw"
: "${WAVE_ANALYSIS_REMOTE:?set WAVE_ANALYSIS_REMOTE, e.g. r2:wave-analysis-raw}"
DEST="$WAVE_ANALYSIS_REMOTE/raw"
COMMON=(--transfers 8 --checkers 16 --fast-list --copy-links --stats-one-line --stats 5m "$@")
# Patterns without a leading / match at any depth. Ledgers use --filter rules,
# which rclone applies in order after the skip rules (--include would come first).
RAW=(--immutable --size-only --exclude "**/_manifests/**" --exclude ".*.lock"
    --exclude ".partial-*" --exclude "*.partial" --exclude "/.offsite_last_sync")
LEDGERS=(--update --filter "+ **/_manifests/**" --filter "- **")
SKIP=()
read -ra SKIP_DIRS <<< "${WAVE_ANALYSIS_SYNC_PULL_SKIP:-}"
for d in "${SKIP_DIRS[@]}"; do
    [[ "$d" =~ ^[A-Za-z0-9_-]+$ ]] || { echo "bad WAVE_ANALYSIS_SYNC_PULL_SKIP entry: $d" >&2; exit 2; }
    SKIP+=(--filter "- /$d/**")
done

rclone copy "$SRC" "$DEST" "${RAW[@]}" "${COMMON[@]}"
rclone copy "$SRC" "$DEST" "${LEDGERS[@]}" "${COMMON[@]}"
# Nothing to pull before the first push (only possible in a dry run).
if rclone lsf --max-depth 1 "$DEST" >/dev/null 2>&1; then
    rclone copy "$DEST" "$SRC" "${SKIP[@]}" "${RAW[@]}" "${COMMON[@]}"
    rclone copy "$DEST" "$SRC" "${SKIP[@]}" "${LEDGERS[@]}" "${COMMON[@]}"
fi
if [[ " $* " != *" --dry-run "* ]]; then
    date -u +%FT%TZ > "$SRC/.offsite_last_sync"
fi
echo "$(date -u +%FT%TZ) offsite exchange with $DEST complete"
