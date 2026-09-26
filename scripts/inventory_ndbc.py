#!/usr/bin/env python
"""Build the NDBC station/product/year availability inventory.

Thin wrapper around ``wave-analysis ndbc inventory`` (kept for the paths named
in the repository specification). Outputs go to ``data/registry/``.

    uv run python scripts/inventory_ndbc.py [--dirs swden swdir ...]
"""

import sys

from wave_analysis.cli import main

if __name__ == "__main__":
    sys.exit(main(["ndbc", "inventory", *sys.argv[1:]]))
