#!/usr/bin/env bash
# Download the sealed storm set (scripts/webcoos_storm_dates.py, rule 7): the dates of
# configs/experiments/webcoos_video_storm_dates.csv in rank order, hourly clips of each
# date's cameras over its local day, until 24 dates have video (at least 6 clips from one
# camera), 120 GB has been fetched, or the list ends. A date without video is skipped.
#
# Each date is processed before the next: pair, frames, windows and physics v2 of the storm set
# (WEBCOOS_VIDEO_SET=storm scripts/webcoos_video_clips.py), then `offload` moves its raw clips to
# the GPU server, checked against the SHA-256 their ledger recorded, and frees them here. So the
# desktop holds about one date's clips at a time (all of them, if the server login is down:
# a failed offload is retried with the next date), and MIN_FREE_GB (default 40) stays free on
# the storm root's drive. At the end: pack, then ship the windows and the package
# (webcoos_video_storm_v1). A date counted once is not fetched again (<root>/_dates/<rank>).
#
# Same pace and the same lock as the video sample (one connection to WebCOOS;
# docs/datasets/webcoos.md): run it after scripts/webcoos_video_sample.sh has finished. The
# clips go to their own root, <main checkout>/data/raw/webcoos_video_storm, which the
# sample's pairing never reads. Needs GPU_HOST (the server in ~/.ssh/config) with its login
# open. Resumable: rerun.
#
#   GPU_HOST=<ssh-alias> systemd-run --user --collect --unit wave-analysis-webcoos-storm \
#       --setenv=GPU_HOST --working-directory="$PWD" "$PWD/scripts/webcoos_storm_sample.sh"
set -uo pipefail
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
MAIN="$(cd "$(git -C "$REPO" rev-parse --git-common-dir)/.." && pwd)"   # the lock lives beside the sample's root
WA="${WA:-$MAIN/.venv/bin/wave-analysis}"
PY="$MAIN/.venv/bin/python"
LIST="$REPO/configs/experiments/webcoos_video_storm_dates.csv"
OUT="${OUT:-$MAIN/data/raw/webcoos_video_storm}"
TARGET_DATES=24
MAX_GB=120
MIN_CLIPS=6
MIN_FREE_GB="${MIN_FREE_GB:-40}"
CUT="${CUT:-1}"   # CUT=0: download only (a test)
: "${GPU_HOST:?set GPU_HOST to the server name in ~/.ssh/config (offload)}"
[[ -d "$OUT/" ]] || { echo "no storm root $OUT" >&2; exit 1; }
export WEBCOOS_VIDEO_SET=storm
DONE="$OUT/_dates"
mkdir -p "$DONE"

clips() {  # camera, UTC start, UTC end: clips of the camera fetched in [start, end), from its ledgers
    "$PY" - "$OUT" "$1" "$2" "$3" <<'PY'
import glob, sys
import pandas as pd
out, cam, a, b = sys.argv[1:]
fs = glob.glob(f"{out}/_manifests/{cam}.*.csv")
s = pd.concat([pd.read_csv(f, usecols=["period", "status"]) for f in fs]) if fs else pd.DataFrame(columns=["period", "status"])
t = pd.to_datetime(s.loc[s["status"] == "verified", "period"], utc=True, format="ISO8601")
print(int(((t >= pd.Timestamp(a, tz="UTC")) & (t < pd.Timestamp(b, tz="UTC"))).sum()))
PY
}
fetched_gb() {  # GB fetched so far, from the ledgers (the clips themselves move to the server)
    "$PY" - "$OUT" <<'PY'
import glob, sys
import pandas as pd
fs = glob.glob(f"{sys.argv[1]}/_manifests/*.csv")
s = pd.concat([pd.read_csv(f, usecols=["size_bytes", "status"]) for f in fs]) if fs else pd.DataFrame(columns=["size_bytes", "status"])
print(round(float(s.loc[s["status"] == "verified", "size_bytes"].sum()) / 1e9, 1))
PY
}
cut_and_offload() {  # the storm set's clips so far: cut, then the cut raw clips to the server
    ((CUT)) || return 0
    (
        cd "$REPO" || exit 1
        C=(uv run python scripts/webcoos_video_clips.py)
        "${C[@]}" pair >/dev/null 2>&1 || { echo "pair failed"; exit 1; }
        "${C[@]}" frames --workers 4 2>&1 | tail -1
        "${C[@]}" windows --workers 4 2>&1 | tail -1
        "${C[@]}" physics --workers 4 2>&1 | tail -1
        # A failed offload (no server login) is not fatal: the clips wait here, the disk guard
        # keeps MIN_FREE_GB free, and the next offload sends every cut clip not yet on the server.
        "${C[@]}" offload 2>&1 | tail -1 || echo "offload failed (server login?): the clips stay here for now"
        exit 0
    )
}

n=0
while IFS=, read -r rank site day season band hs cameras gb; do
    [[ "$rank" == rank ]] && continue
    if [[ -f "$DONE/$rank" ]]; then  # counted in an earlier run
        grep -q '^video' "$DONE/$rank" && n=$((n + 1))
        ((n >= TARGET_DATES)) && break
        continue
    fi
    used=$(fetched_gb)
    if awk -v u="$used" -v g="$gb" -v m="$MAX_GB" 'BEGIN {exit !(u + g > m)}'; then
        echo "stopping before rank $rank: $used GB fetched, the next date would pass $MAX_GB GB"
        break
    fi
    # The local day's daylight lies in [day 08:00Z, day+1 04:00Z) in EST and EDT alike.
    a="${day}T08:00"
    b="$(date -u -d "$day + 1 day" +%F)T04:00"
    best=0
    for cam in ${cameras//;/ }; do
        log=$("$WA" webcoos historical "$cam" --product video --every 60min --start "$a" --end "$b" \
            --out "$OUT" --min-free-gb "$MIN_FREE_GB" --historical-approved 2>&1)
        rc=$?
        tail -1 <<<"$log"
        [[ "$log" == *"another historical run holds"* ]] && { echo "another download holds the lock: stopping; rerun when it has finished"; exit 5; }
        ((rc == 4)) && { echo "stopped by the disk guard (less than $MIN_FREE_GB GB free)"; exit 4; }
        k=$(clips "$cam" "$a" "$b")
        ((k > best)) && best=$k
    done
    ((best > 0)) && { cut_and_offload || { echo "cutting failed at rank $rank"; exit 6; }; }
    if ((best >= MIN_CLIPS)); then
        n=$((n + 1))
        echo "video $best" >"$DONE/$rank"
        echo "rank $rank $site $day: video (up to $best clips a camera); $n of $TARGET_DATES dates"
    else
        echo "none $best" >"$DONE/$rank"
        echo "rank $rank $site $day: no video ($best clips); skipped"
    fi
    ((n >= TARGET_DATES)) && break
done <"$LIST"
echo "storm set fetched: $n dates with video, $(fetched_gb) GB"
if ((CUT)); then
    cd "$REPO" || exit 1
    uv run python scripts/webcoos_video_clips.py offload | tail -1
    uv run python scripts/webcoos_video_clips.py pack | tail -2 &&
        uv run python scripts/webcoos_video_clips.py ship | tail -1 &&
        uv run python scripts/webcoos_video_clips.py ship --package | tail -1
fi
