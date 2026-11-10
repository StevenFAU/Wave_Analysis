#!/usr/bin/env bash
# Exchange data/raw with an rclone remote: the offsite copy of the raw archive,
# and the way several collector hosts share one archive.
#
#   WAVE_ANALYSIS_REMOTE=r2:wave-analysis-raw scripts/sync_offsite.sh [--dry-run]
#
# Checks the files written since the last successful exchange, pushes local
# files to the remote, then pulls the files other hosts pushed.
#
# A damaged file never reaches the remote. Before pushing, every file modified
# since the last successful exchange (less a day) is checked against its
# ledger's SHA-256 (`wave-analysis archive verify --since last-exchange`). A
# power cut can leave a file empty, or the right size and full of NUL bytes
# (2026-11-03); such files, and ledgers containing NUL bytes, are held back
# from every copy below, and the exchange fails at the end naming them.
#
# Raw files are append-only: they are copied with --immutable, so a file that
# exists on both sides with a different size is reported as an error instead
# of being overwritten. They are compared by size only, because two hosts
# fetch the same image at different times (and reading S3 modification times
# costs a request per object); `wave-analysis archive compare-remote` compares
# hashes. Such a conflict no longer stops the exchange: the other files are
# still pushed and pulled, the conflicting paths are written to
# data/interim/offsite/conflicts.txt, and the exchange fails at the end with
# the command that settles them (`wave-analysis archive repair`, which uses
# the ledgers to tell which copy is right).
#
# Ledgers (_manifests/*.csv) grow, so they are copied with --update: every
# ledger has a single writer (buoycam ledgers are named per host), and an
# older copy never replaces a newer one. Nothing is ever deleted on either
# side ("copy", never "sync"). Only when every step succeeded is the time
# written to data/raw/.offsite_last_sync, which the dashboard reports (the
# watchdog opens an issue when it is more than 30 h old).
#
# A collector appends to its ledger while it runs, and rclone rejects a file
# that changes during its upload. A long download (the WebCOOS history adds a
# row every two seconds for days) would fail every exchange that way. So the
# ledgers are uploaded from a copy made just before, which holds what was
# written up to that moment; the next exchange uploads the rest.
#
# A collection may live on another drive behind a symlink (e.g. data/raw/
# webcoos_timex -> /mnt/wavedata/...): --copy-links follows it in both
# directions. If that drive is not mounted, the exchange fails instead of
# skipping the collection.
#
# A host without room for some collections can leave them out of the pull
# (its own files in them are still pushed), e.g. in sync.env on a laptop:
#   WAVE_ANALYSIS_SYNC_PULL_SKIP="webcoos webcoos_timex webcoos_brt"
set -euo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DATA="${WAVE_ANALYSIS_DATA:-$REPO/data}"
SRC="$DATA/raw"
STATE="$DATA/interim/offsite"
: "${WAVE_ANALYSIS_REMOTE:?set WAVE_ANALYSIS_REMOTE, e.g. r2:wave-analysis-raw}"
DEST="$WAVE_ANALYSIS_REMOTE/raw"
PY="${WAVE_ANALYSIS_PYTHON:-$REPO/.venv/bin/python}"
COMMON=(--transfers 8 --checkers 16 --fast-list --copy-links --stats-one-line --stats 5m
    --stats-log-level NOTICE "$@")
# Patterns without a leading / match at any depth. Ledgers use --filter rules,
# which rclone applies after every --exclude and --exclude-from rule.
RAW=(--immutable --size-only --exclude "**/_manifests/**" --exclude ".*.lock"
    --exclude ".partial-*" --exclude "*.partial" --exclude ".*.part" --exclude "/.offsite_last_sync")
ONLY_LEDGERS=(--filter "+ **/_manifests/**" --filter "- **")
LEDGERS=(--update "${ONLY_LEDGERS[@]}")
SKIP=()
read -ra SKIP_DIRS <<< "${WAVE_ANALYSIS_SYNC_PULL_SKIP:-}"
for d in "${SKIP_DIRS[@]}"; do
    [[ "$d" =~ ^[A-Za-z0-9_-]+$ ]] || { echo "bad WAVE_ANALYSIS_SYNC_PULL_SKIP entry: $d" >&2; exit 2; }
    SKIP+=(--filter "- /$d/**")
done

mkdir -p "$STATE"
HELD="$STATE/held_back.rules"
CONFLICTS="$STATE/conflicts.txt"
LOG="$STATE/rclone.log"
SNAPSHOT="$(mktemp -d)"
trap 'rm -rf "$SNAPSHOT"' EXIT
failed=()

# 1. Check what was written since the last successful exchange.
rm -f "$HELD"
verify_rc=0
"$PY" -m wave_analysis.cli archive verify --root "$SRC" --since last-exchange \
    --problems-only --quiet --show 100 --out "$STATE/verify" --exclude-list "$HELD" ||
    verify_rc=$?
if [[ ! -f "$HELD" ]]; then
    echo "ERROR: the check before the exchange did not finish (exit $verify_rc); nothing was copied" >&2
    exit 2
fi
if [[ -s "$HELD" ]]; then
    echo "ERROR: $(wc -l < "$HELD") damaged file(s) held back from the exchange (listed above;" \
        "report $STATE/verify/problems.csv)" >&2
    failed+=("damaged files")
fi
EXCLUDE=(--exclude-from "$HELD")

# copy NAME ARGS...: one rclone copy; its log goes to the journal, and any
# "immutable file modified" conflicts are added to $CONFLICTS.
copy() {
    local name="$1" rc=0
    shift
    : > "$LOG"
    rclone copy "$@" --log-file "$LOG" || rc=$?
    cat "$LOG" >&2
    grep -oP 'ERROR : \K.+?(?=: Source and destination exist but do not match: immutable file modified)' \
        "$LOG" >> "$CONFLICTS" || true
    if ((rc != 0)); then
        echo "ERROR: $name failed (rclone exit $rc)" >&2
        failed+=("$name")
    fi
}

: > "$CONFLICTS"
# 2. Push this host's raw files.
copy "push" "$SRC" "$DEST" "${EXCLUDE[@]}" "${RAW[@]}" "${COMMON[@]}"
# 3. Push the ledgers, from a copy (see above).
rclone lsf -R --files-only --copy-links "${EXCLUDE[@]}" "${ONLY_LEDGERS[@]}" "$SRC" | while IFS= read -r f; do
    mkdir -p "$SNAPSHOT/$(dirname "$f")"
    cp -pL "$SRC/$f" "$SNAPSHOT/$f"
done
copy "ledger push" "$SNAPSHOT" "$DEST" "${LEDGERS[@]}" "${COMMON[@]}"
# 4. Pull other hosts' files. Nothing to pull before the first push (only possible in a dry run).
if rclone lsf --max-depth 1 "$DEST" >/dev/null 2>&1; then
    copy "pull" "$DEST" "$SRC" "${EXCLUDE[@]}" "${SKIP[@]}" "${RAW[@]}" "${COMMON[@]}"
    copy "ledger pull" "$DEST" "$SRC" "${EXCLUDE[@]}" "${SKIP[@]}" "${LEDGERS[@]}" "${COMMON[@]}"
fi

sort -u -o "$CONFLICTS" "$CONFLICTS"
if [[ -s "$CONFLICTS" ]]; then
    echo "ERROR: $(wc -l < "$CONFLICTS") file(s) differ between $SRC and $DEST, listed in $CONFLICTS." >&2
    echo "  The ledgers tell which copy is right. See what would change, then settle them:" >&2
    echo "    $REPO/.venv/bin/wave-analysis archive repair --from $CONFLICTS --dry-run" >&2
    echo "    $REPO/.venv/bin/wave-analysis archive repair --from $CONFLICTS" >&2
fi
if ((${#failed[@]})); then
    echo "$(date -u +%FT%TZ) offsite exchange with $DEST FAILED: ${failed[*]}" >&2
    exit 1
fi
if [[ " $* " != *" --dry-run "* ]]; then
    date -u +%FT%TZ > "$SRC/.offsite_last_sync"
fi
echo "$(date -u +%FT%TZ) offsite exchange with $DEST complete"
