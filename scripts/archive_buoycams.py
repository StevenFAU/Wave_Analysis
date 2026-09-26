#!/usr/bin/env python
"""Archive new NDBC buoy-camera images and backfill the last 70 h.

uv run python scripts/archive_buoycams.py [--backfill-hours N]

Normally run by the hourly systemd timer (scripts/install_collectors.sh);
see docs/operations/data_collection.md.
"""

import sys

from wave_analysis.cli import main

if __name__ == "__main__":
    sys.exit(main(["ndbc", "cameras", *sys.argv[1:]]))
