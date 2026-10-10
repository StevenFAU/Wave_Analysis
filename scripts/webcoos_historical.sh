#!/usr/bin/env bash
# Download the WebCOOS history agreed with WebCOOS (docs/datasets/webcoos.md):
# stills every 30 min for all years, then timex and brt every 30 min (2024 on),
# one product after another over a single connection. Resumable: rerun to
# continue. Extra arguments are passed to every run (e.g. --list-only).
# WEBCOOS_PRODUCTS limits the products (default "stills timex brt").
#
#   systemd-run --user --collect --unit wave-analysis-webcoos-historical \
#       --working-directory="$PWD" "$PWD/scripts/webcoos_historical.sh"
#   systemd-run ... --setenv=WEBCOOS_PRODUCTS=stills ...   # stills only
#   journalctl --user -u wave-analysis-webcoos-historical -f
set -uo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
CAMS=(currituck_hampton_inn currituck_sailfish jennette_north jennette_south
    oakisland_east oakisland_west masonboro_inlet cocoabeach)

rc=0
read -ra PRODUCTS <<< "${WEBCOOS_PRODUCTS:-stills timex brt}"
for product in "${PRODUCTS[@]}"; do
    # A product that fails (a listing that kept failing, or too little free
    # space on its drive) does not stop the others.
    "$REPO/.venv/bin/wave-analysis" webcoos historical "${CAMS[@]}" \
        --product "$product" --historical-approved "$@" || rc=1
done
exit "$rc"
