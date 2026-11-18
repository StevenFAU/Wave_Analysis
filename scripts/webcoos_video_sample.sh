#!/usr/bin/env bash
# Download the WebCOOS video sample (docs/datasets/webcoos.md, "Video archive"):
# one clip per 30-min CDIP record in daylight, from the on-premise archive's mp4
# tree, at WebCOOS's pace (one connection, 1 s between requests, 20 MiB/s).
#
# 1. A quick test set first: per camera one rough and one typical day, picked from
#    the reference buoy's daily mean H_s within the video period (2024-03 on):
#    Currituck and Jennette (CDIP 433 / 243): 2026-09-24 (3.9 m) and 2024-08-04
#    (0.8 m); Oak Island (CDIP 200, to 2026-07-20): 2024-04-12 (2.7 m) and
#    2024-08-14 (0.9 m).
# 2. Then six sample weeks, about one per season, before CDIP 200's record ends.
#
# Resumable: files already on disk are skipped; rerun to continue.
#
#   systemd-run --user --collect --unit wave-analysis-webcoos-video \
#       --working-directory="$PWD" "$PWD/scripts/webcoos_video_sample.sh"
#   journalctl --user -u wave-analysis-webcoos-video -f
set -uo pipefail
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
WA="$REPO/.venv/bin/wave-analysis"
NORTH=(currituck_hampton_inn jennette_south)
OAK=(oakisland_east oakisland_west)
rc=0
get() {  # cameras..., then --start and --end
    "$WA" webcoos historical "$@" --product video --every 30min --historical-approved || rc=1
}
# 1. the test set
get "${NORTH[@]}" --start 2026-09-24 --end 2026-09-25
get "${NORTH[@]}" --start 2024-08-04 --end 2024-08-05
get "${OAK[@]}" --start 2024-04-12 --end 2024-04-13
get "${OAK[@]}" --start 2024-08-14 --end 2024-08-15
echo "test set done (rc=$rc)"
# 2. six sample weeks
for wk in 2024-04-08 2024-07-08 2024-10-07 2025-01-13 2025-04-14 2026-01-12; do
    end=$(date -u -d "$wk + 7 days" +%F)
    get "${NORTH[@]}" "${OAK[@]}" --start "$wk" --end "$end"
done
echo "video sample done (rc=$rc)"
exit "$rc"
