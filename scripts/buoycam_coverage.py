#!/usr/bin/env python
"""Report buoy-camera archive coverage: images per station and missing hours.

    uv run python scripts/buoycam_coverage.py [--days 3] [--root data/raw/ndbc/buoycam]

For each station, counts archived images in the last ``--days`` days, the
number of hours recorded upstream as never published (``not_found`` in the
ledger), and hours with neither an image nor a ledger record (true gaps:
the collector missed them and they may be past the 72 h retention).
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import pandas as pd

from wave_analysis.config import data_dir
from wave_analysis.sources.ndbc.buoycam import MANIFESTS_DIR, archive_index


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--days", type=float, default=3.0)
    ap.add_argument("--root", type=Path, default=data_dir("raw") / "ndbc" / "buoycam")
    args = ap.parse_args()

    now = pd.Timestamp.now(tz="UTC")
    start = (now - pd.Timedelta(days=args.days)).ceil("h")
    idx = archive_index(args.root)
    idx = idx[idx["time_utc"] >= start]
    rows = []
    for p in sorted((args.root / MANIFESTS_DIR).glob("*.csv")):
        with p.open(newline="", encoding="utf-8") as fh:
            rows += [r for r in csv.DictReader(fh) if r["product"] == "buoycam:image"]
    led = pd.DataFrame(rows, columns=["station_id", "period", "status"])
    led["time_utc"] = pd.to_datetime(led["period"], format="%Y-%m-%dT%H:%MZ", utc=True)
    missing = led[(led["status"] == "not_found") & (led["time_utc"] >= start)]

    # Hours expected per station: from its first archived hour in the window to now - 1 h.
    out = []
    for sid, g in idx.groupby("station_id"):
        hours = pd.date_range(
            g["time_utc"].min().floor("h"), (now - pd.Timedelta("50min")).floor("h"), freq="h"
        )
        have = set(g["time_utc"].dt.floor("h"))
        nf = set(missing.loc[missing["station_id"] == sid, "time_utc"].dt.floor("h"))
        out.append(
            {
                "station_id": sid,
                "images": len(g),
                "expected_hours": len(hours),
                "not_published": len(nf),
                "unexplained_gaps": sum(1 for h in hours if h not in have and h not in nf),
                "last_image_utc": g["time_utc"].max(),
            }
        )
    rep = pd.DataFrame(out).sort_values("station_id")
    pd.set_option("display.width", 140)
    print(f"window: {start} .. {now:%Y-%m-%d %H:%M}Z   stations: {len(rep)}")
    print(rep.to_string(index=False))
    print(
        f"\ntotal images {rep['images'].sum()}, not published {rep['not_published'].sum()}, "
        f"unexplained gaps {rep['unexplained_gaps'].sum()}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
