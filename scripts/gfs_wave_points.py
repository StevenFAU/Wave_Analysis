"""Build a station/issue/valid-time table on the buoy station inventory."""

import argparse
from pathlib import Path

import pandas as pd

from wave_analysis.sources.gfs_wave import read_cycle


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--labels", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("tars", nargs="+", type=Path)
    a = p.parse_args()
    if a.out.exists():
        raise FileExistsError(a.out)
    stations = set(pd.read_csv(a.labels, usecols=["station_id"], dtype=str).station_id)
    df = pd.DataFrame(row for path in a.tars for row in read_cycle(path, stations))
    if df.empty or df.duplicated(["station_id", "issue_utc", "lead_h"]).any():
        raise ValueError("no station matches or duplicate station/issue/lead")
    df.to_parquet(a.out, index=False)
    print(len(df), "rows;", df.station_id.nunique(), "of", len(stations), "requested stations")


if __name__ == "__main__":
    main()
