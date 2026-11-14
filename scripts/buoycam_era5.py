#!/usr/bin/env python
"""ERA5 significant wave height at every NDBC camera buoy: the physics baseline of the buoy-camera experiment.

    .venv/bin/python scripts/buoycam_era5.py download --start 2026-09 --end 2026-11
    .venv/bin/python scripts/buoycam_era5.py extract

``download`` requests one parameter (``swh``) over the whole globe, one
calendar month per request: three requests serve every buoy, where a box per
buoy would take about 240 (the CDS size limit counts variables x days x hours,
not area). Files go to ``data/raw/era5/global_swh/`` with their requests, and
every attempt to ``data/manifests/raw/era5.csv``. Months that can still change
upstream (ERA5T, or not yet complete) are requested again when re-run later,
as for the per-site downloads (``docs/datasets/era5.md``).

``extract`` takes, for each buoy, the grid point nearest its listed position
that has sea data (the same rule as the per-site layer), and writes its hourly
series to ``data/processed/era5/buoycam/era5_at_buoys.parquet`` and the chosen
points to ``points.csv``. The buoy-camera package interpolates the series to
each strip's time and corrects it with a line fitted on the training strips
(``scripts/buoycam_package.py``).
"""

from __future__ import annotations

import argparse
import fcntl
import json
import sys

import numpy as np
import pandas as pd

from wave_analysis.config import data_dir, portable_path
from wave_analysis.datasets import build_provenance, write_provenance
from wave_analysis.datasets.buoycam import station_positions
from wave_analysis.ingest.durable import atomic_open
from wave_analysis.ingest.manifest import EntryStatus, ManifestEntry, write_manifest
from wave_analysis.processing.geospatial import haversine_m
from wave_analysis.sources.era5 import (
    CITATION,
    EXPVER,
    PARAMETERS,
    CDSConfigError,
    ERA5WaveSource,
    months,
)
from wave_analysis.sources.ndbc.buoycam import LISTINGS_DIR

NAME = "global_swh"
GLOBE = [90.0, -180.0, -90.0, 180.0]
SWH = tuple(p for p in PARAMETERS if p.short == "swh")


def download(args: argparse.Namespace) -> int:
    dest = data_dir("raw") / "era5"
    manifest = data_dir("manifests") / "raw" / "era5.csv"
    dest.mkdir(parents=True, exist_ok=True)

    def log(msg: str) -> None:
        print(f"{pd.Timestamp.now(tz='UTC'):%Y-%m-%dT%H:%M:%SZ} {msg}", flush=True)

    def record(batch: list[ManifestEntry]) -> None:
        write_manifest(batch, manifest)
        for e in batch:
            size = f"{e.size_bytes / 1e6:,.1f} MB" if e.size_bytes is not None else "-"
            log(f"{e.station_id} {e.period}: {e.status.value} ({size}) {e.note}")

    with (dest / ".era5.lock").open("w") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            log(f"another run holds {dest / '.era5.lock'}; skipping")
            return 0
        src = ERA5WaveSource(log=log)
        try:
            entries = src.fetch_area(
                NAME, GLOBE, months(args.start, args.end), dest_root=dest,
                parameters=SWH, refresh=args.refresh, sink=record,
            )  # fmt: skip
        except CDSConfigError as exc:
            print(f"error: {exc}", file=sys.stderr)
            return 2
    log(f"{len(entries)} requests; cite: {CITATION}")
    return 1 if any(e.status == EntryStatus.FAILED for e in entries) else 0


def extract(args: argparse.Namespace) -> int:
    import xarray as xr

    raw = sorted((data_dir("raw") / "era5" / NAME).glob(f"*_{NAME}_*.nc"))
    if not raw:
        raise SystemExit("no global files; run the download first")
    archive = data_dir("raw") / "ndbc" / "buoycam"
    pos = station_positions(sorted((archive / LISTINGS_DIR).glob("*.json")))
    series, points = [], []
    for path in raw:
        with xr.open_dataset(path) as ds:
            pid = ds["swh"].attrs.get("GRIB_paramId")
            if pid is not None and int(pid) != SWH[0].param_id:
                raise SystemExit(f"{path.name}: swh has GRIB_paramId {pid}")
            lat = np.asarray(ds["latitude"].values, dtype=np.float64)
            lon = np.asarray(ds["longitude"].values, dtype=np.float64)
            # Land is NaN at every hour; a point with a value on any day is sea.
            daily = ds["swh"].isel(valid_time=slice(None, None, 24))
            sea = np.isfinite(daily.transpose("valid_time", "latitude", "longitude").to_numpy())
            sea = sea.any(axis=0)
            times = pd.DatetimeIndex(ds["valid_time"].values).tz_localize("UTC")
            expver = (
                np.broadcast_to(np.atleast_1d(ds["expver"].values).astype(str), (len(times),))
                if "expver" in ds.variables
                else np.full(len(times), "unknown")
            )
            glat, glon = np.meshgrid(lat, lon, indexing="ij")
            for r in pos.itertuples(index=False):
                d = np.asarray(haversine_m(r.latitude, r.longitude, glat, glon)) / 1000
                d[~sea] = np.inf
                i, j = np.unravel_index(int(np.argmin(d)), d.shape)
                v = ds["swh"].isel(latitude=int(i), longitude=int(j)).to_numpy()
                series.append(
                    pd.DataFrame(
                        {
                            "station_id": r.station_id,
                            "time_utc": times,
                            "era5_hs_m": v.astype(np.float64),
                            "source_qc_flag": [EXPVER.get(e, e) for e in expver],
                        }
                    )
                )
                points.append(
                    {
                        "station_id": r.station_id,
                        "month": path.stem.rsplit("_", 1)[-1],
                        "grid_latitude": float(lat[i]),
                        "grid_longitude": float(lon[j]),
                        "distance_km": round(float(d[i, j]), 1),
                    }
                )
    out = data_dir("processed") / "era5" / "buoycam"
    out.mkdir(parents=True, exist_ok=True)
    s = pd.concat(series, ignore_index=True).dropna(subset=["era5_hs_m"])
    s = s.sort_values(["station_id", "time_utc"]).drop_duplicates(["station_id", "time_utc"])
    s.to_parquet(out / "era5_at_buoys.parquet", index=False)
    pts = pd.DataFrame(points)
    moved = pts.groupby("station_id")[["grid_latitude", "grid_longitude"]].nunique().max(axis=1)
    if (moved > 1).any():
        raise SystemExit(f"grid point differs between months for {list(moved[moved > 1].index)}")
    pts = pts.drop(columns="month").drop_duplicates("station_id")
    with atomic_open(out / "points.csv", "w", newline="") as fh:
        pts.to_csv(fh, index=False)
    summary = {
        "buoys": len(pts),
        "hours": len(s),
        "first": str(s["time_utc"].min()),
        "last": str(s["time_utc"].max()),
        "by_flag": s["source_qc_flag"].value_counts().to_dict(),
        "distance_km": pts["distance_km"].describe().round(1).to_dict(),
        "farthest": pts.nlargest(5, "distance_km").to_dict("records"),
    }
    with atomic_open(out / "summary.json", "w") as fh:
        json.dump(summary, fh, indent=1, default=str)
    write_provenance(
        build_provenance(
            artifact_id="era5_at_ndbc_camera_buoys",
            inputs=raw,
            outputs=[out / "era5_at_buoys.parquet", out / "points.csv", out / "summary.json"],
            config={
                "rule": "nearest grid point with sea data",
                "files": [portable_path(p) for p in raw],
            },
            notes=CITATION,
        ),
        out / "provenance.yaml",
    )
    print(json.dumps(summary, indent=1, default=str))
    return 0


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    d = sub.add_parser("download")
    d.add_argument("--start", required=True, help="first month, YYYY-MM")
    d.add_argument("--end", required=True, help="last month, YYYY-MM")
    d.add_argument("--refresh", action="store_true", help="request every month again")
    d.set_defaults(func=download)
    e = sub.add_parser("extract")
    e.set_defaults(func=extract)
    args = ap.parse_args(argv)
    return int(args.func(args))


if __name__ == "__main__":
    sys.exit(main())
