#!/usr/bin/env bash
# Download the sealed storm set (scripts/webcoos_storm_dates.py, rule 7): the dates of
# configs/experiments/webcoos_video_storm_dates.csv in rank order, hourly clips of each
# date's cameras over its local day, until 24 dates have video (at least 6 clips from one
# camera), the storm root holds 120 GB, or the list ends. A date without video is skipped.
#
# Same pace and the same lock as the video sample (one connection to WebCOOS;
# docs/datasets/webcoos.md): run it after scripts/webcoos_video_sample.sh has finished. The
# clips go to their own root, <main checkout>/data/raw/webcoos_video_storm (a symlink to the
# drive that holds them), which the sample's pairing never reads. The disk guard keeps
# MIN_FREE_GB (default 200) free on that drive. Resumable: rerun; clips on disk are skipped.
#
#   systemd-run --user --collect --unit wave-analysis-webcoos-storm \
#       --working-directory="$PWD" "$PWD/scripts/webcoos_storm_sample.sh"
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
MIN_FREE_GB="${MIN_FREE_GB:-200}"
[[ -d "$OUT/" ]] || { echo "no storm root $OUT (a symlink to the drive that holds the storm clips)" >&2; exit 1; }

clips() {  # camera, UTC start, UTC end: clips of the camera on disk in [start, end), from its ledgers
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

n=0
while IFS=, read -r rank site day season band hs cameras gb; do
    [[ "$rank" == rank ]] && continue
    used=$(du -s --apparent-size -B1G "$OUT/" | cut -f1)
    if awk -v u="$used" -v g="$gb" -v m="$MAX_GB" 'BEGIN {exit !(u + g > m)}'; then
        echo "stopping before rank $rank: $used GB in the storm root, the next date would pass $MAX_GB GB"
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
    if ((best >= MIN_CLIPS)); then
        n=$((n + 1))
        echo "rank $rank $site $day: video (up to $best clips a camera); $n of $TARGET_DATES dates"
    else
        echo "rank $rank $site $day: no video ($best clips); skipped"
    fi
    ((n >= TARGET_DATES)) && break
done <"$LIST"
echo "storm set done: $n dates with video, $(du -s --apparent-size -B1G "$OUT/" | cut -f1) GB"
