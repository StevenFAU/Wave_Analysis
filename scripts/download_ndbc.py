#!/usr/bin/env python
"""Download NDBC files for stations (wrapper around ``wave-analysis ndbc download``).

uv run python scripts/download_ndbc.py 41010 41009 --years 2020-2023 --realtime
"""

import sys

from wave_analysis.cli import main

if __name__ == "__main__":
    sys.exit(main(["ndbc", "download", *sys.argv[1:]]))
