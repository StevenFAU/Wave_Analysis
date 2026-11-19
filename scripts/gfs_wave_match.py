"""Match retrospective GFS guidance to strip timestamps using nominal issue times."""

import argparse
from pathlib import Path

import pandas as pd


def match_strips(strips, points):
    strips = strips.copy()
    time = pd.to_datetime(strips["time_utc"], utc=True)
    if time.isna().any() or strips["sample_id"].duplicated().any():
        raise ValueError("missing capture time or duplicate strip identifier")
    strips["issue_utc"] = time.dt.floor("6h")
    # Resolve an exact half-hour tie to the earlier valid hour.
    strips["valid_utc"] = (time + pd.Timedelta(minutes=30) - pd.Timedelta(nanoseconds=1)).dt.floor(
        "h"
    )
    strips["offset_s"] = (strips["valid_utc"] - time).dt.total_seconds()
    keys = ["station_id", "issue_utc", "valid_utc"]
    result = strips.merge(points, on=keys, how="left", validate="many_to_one", indicator=True)
    result["bulletin_present"] = result.pop("_merge").eq("both")
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("labels", "points", "out"):
        parser.add_argument("--" + name, type=Path, required=True)
    args = parser.parse_args()
    if args.out.exists():
        raise FileExistsError(args.out)
    labels = pd.read_csv(
        args.labels,
        usecols=["sample_id", "station_id", "station_group", "cv_fold", "time_utc"],
        dtype={"station_id": str},
    )
    matched = match_strips(labels, pd.read_parquet(args.points))
    matched.to_parquet(args.out, index=False)
    print(matched.groupby("cv_fold").agg(strips=("sample_id", "size"), finite_hs=("hs_m", "count")))


if __name__ == "__main__":
    main()
