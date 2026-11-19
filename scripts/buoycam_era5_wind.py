#!/usr/bin/env python
"""ERA5 10-m wind and gust at every NDBC camera buoy: the no-camera baseline of the virtual anemometer.

    scripts/buoycam_era5_wind.py download --package <buoy package> --start 2026-09 --end 2026-11
    scripts/buoycam_era5_wind.py extract --package <buoy package>

Test 5 of the wave-sensing report (bar in ``docs/experiments/wave_measurements.md``) asks
whether one camera strip gives the buoy's own wind better than ERA5 does. ERA5's wind is
an atmospheric field on its native 0.25 deg grid (the wave fields of
``scripts/buoycam_era5.py`` are on 0.5 deg), so this script follows that one's pattern
with three other parameters:

- ``u10``, ``v10``: the 10-m wind components (GRIB 165, 166), hourly instantaneous;
- ``i10fg``: the instantaneous 10-m wind gust (GRIB 228029), the largest gust of the
  hour before each stamp in ERA5's post-processing.

``download`` asks for all three over one box that holds every buoy of the package (its
strips' positions, 1 deg margin), one calendar month per request (``ERA5WaveSource``:
files ``data/raw/era5/buoys_wind/era5_waves_buoys_wind_<YYYY-MM>.nc``, each attempt a
row of ``data/manifests/raw/era5.csv``; ERA5T months are requested again on a later run).

``extract`` reads, for each buoy, the grid point nearest its position (its strips'
median latitude and longitude; atmospheric fields have values over land too, so the
distance is recorded); wind speed is sqrt(u10^2 + v10^2). Each strip gets the two hourly
values bracketing its time, interpolated linearly, as ERA5 heights are. The buoy's
anemometer is lower than 10 m (about 4-5 m on most hulls), so a comparison corrects ERA5
by a line fitted on the training strips. Writes
``data/processed/era5/buoycam_wind/{era5_wind_strips.csv,points.csv,provenance.yaml}``.
"""

from __future__ import annotations

import argparse
import fcntl
import sys

import numpy as np
import pandas as pd

from wave_analysis.config import data_dir, portable_path
from wave_analysis.datasets import build_provenance, write_provenance
from wave_analysis.ingest.manifest import EntryStatus, ManifestEntry, write_manifest
from wave_analysis.processing.geospatial import haversine_m
from wave_analysis.sources.era5 import (
    CITATION,
    CDSConfigError,
    ERA5WaveSource,
    WaveParameter,
    months,
)

NAME = "buoys_wind"
WIND = (
    WaveParameter("10m_u_component_of_wind", "u10", 165, "wind_u_10m", "era5_hourly"),
    WaveParameter("10m_v_component_of_wind", "v10", 166, "wind_v_10m", "era5_hourly"),
    WaveParameter("instantaneous_10m_wind_gust", "i10fg", 228029, "wind_gust_10m", "era5_hourly"),
)


def positions(package) -> pd.DataFrame:
    lab = pd.read_csv(
        package / "labels.csv",
        usecols=["station_id", "latitude", "longitude"],
        dtype={"station_id": str},
    )
    return lab.groupby("station_id")[["latitude", "longitude"]].median().reset_index()


def groups(pos: pd.DataFrame) -> list[tuple[str, pd.DataFrame]]:
    """One box west of the date line, one small box for buoys east of it (46071 at 179.8 E)."""
    east = pos.longitude > 0
    return [(NAME, pos[~east]), (NAME + "_dateline", pos[east])]


def box(pos: pd.DataFrame, margin: float = 1.0) -> list[float]:
    """``[north, west, south, east]`` on the 0.25 deg grid, holding every buoy."""
    lon = np.where(pos.longitude > 180, pos.longitude - 360, pos.longitude)

    def snap(v, up):
        return float((np.ceil if up else np.floor)(v * 4) / 4)

    return [
        snap(pos.latitude.max() + margin, True),
        snap(lon.min() - margin, False),
        snap(pos.latitude.min() - margin, False),
        min(180.0, snap(lon.max() + margin, True)),
    ]


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
        entries = []
        try:
            for name, sub in groups(positions(args.package)):
                area = box(sub)
                log(f"{name}: {len(sub)} buoys, box {area} (north, west, south, east)")
                entries += src.fetch_area(
                    name, area, months(args.start, args.end), dest_root=dest, parameters=WIND,
                    refresh=args.refresh, sink=record,
                )  # fmt: skip
        except CDSConfigError as exc:
            print(f"error: {exc}", file=sys.stderr)
            return 2
    log(f"{len(entries)} requests; cite: {CITATION}")
    return 1 if any(e.status == EntryStatus.FAILED for e in entries) else 0


def _read(files, pos, hourly, points) -> None:
    """Append each buoy's hourly ERA5 wind and gust at its nearest grid point."""
    import xarray as xr

    for path in files:
        with xr.open_dataset(path) as ds:
            for p in WIND:
                pid = ds[p.short].attrs.get("GRIB_paramId")
                if pid is not None and int(pid) != p.param_id:
                    raise SystemExit(f"{path.name}: {p.short} has GRIB_paramId {pid}")
            lat = np.asarray(ds["latitude"].values, dtype=np.float64)
            lon = np.asarray(ds["longitude"].values, dtype=np.float64)
            times = pd.DatetimeIndex(ds["valid_time"].values).tz_localize("UTC")
            glat, glon = np.meshgrid(lat, lon, indexing="ij")
            for r in pos.itertuples(index=False):
                blon = r.longitude - 360 if r.longitude > 180 else r.longitude
                d = np.asarray(haversine_m(r.latitude, blon, glat, glon)) / 1000
                i, j = np.unravel_index(int(np.argmin(d)), d.shape)
                points[r.station_id] = {
                    "station_id": r.station_id,
                    "grid_latitude": float(lat[i]),
                    "grid_longitude": float(lon[j]),
                    "distance_km": float(d[i, j]),
                }
                v = {
                    p.short: ds[p.short].isel(latitude=int(i), longitude=int(j)).to_numpy()
                    for p in WIND
                }
                hourly.append(
                    pd.DataFrame(
                        {
                            "station_id": r.station_id,
                            "time_utc": times,
                            "wind": np.hypot(v["u10"], v["v10"]),
                            "gust": v["i10fg"],
                        }
                    )
                )


def extract(args: argparse.Namespace) -> int:
    hourly, points, paths = [], {}, []
    for name, pos in groups(positions(args.package)):
        files = sorted((data_dir("raw") / "era5" / name).glob(f"*_{name}_*.nc"))
        if not files:
            raise SystemExit(f"no {name} files; run the download first")
        paths += files
        _read(files, pos, hourly, points)
    hourly = pd.concat(hourly).drop_duplicates(["station_id", "time_utc"], keep="last")
    lab = pd.read_csv(
        args.package / "labels.csv", usecols=["sample_id", "station_id", "time_utc"],
        dtype={"station_id": str},
    )  # fmt: skip
    lab["time_utc"] = pd.to_datetime(lab.time_utc, utc=True)
    rows = []
    for sid, g in lab.groupby("station_id"):
        h = hourly[hourly.station_id == sid].set_index("time_utc").sort_index()
        left, right = g.time_utc.dt.floor("h"), g.time_utc.dt.ceil("h")
        w = ((g.time_utc - left).dt.total_seconds() / 3600).to_numpy()
        out = g[["sample_id"]].copy()
        for col, name in [("wind", "era5_wind_mps"), ("gust", "era5_gust_mps")]:
            a = h[col].reindex(left).to_numpy()
            b = h[col].reindex(right).to_numpy()
            out[name] = (1 - w) * a + w * b
        rows.append(out)
    strips = pd.concat(rows)
    dest = data_dir("processed") / "era5" / "buoycam_wind"
    dest.mkdir(parents=True, exist_ok=True)
    strips.to_csv(dest / "era5_wind_strips.csv", index=False, float_format="%.3f")
    pd.DataFrame(points.values()).to_csv(dest / "points.csv", index=False)
    write_provenance(
        build_provenance(
            artifact_id="era5_wind_at_buoycam_strips",
            inputs=[*paths, args.package / "labels.csv"],
            outputs=[dest / "era5_wind_strips.csv", dest / "points.csv"],
            config={"parameters": [p.short for p in WIND], "grid": "0.25 deg, nearest point"},
            notes=f"cite: {CITATION}",
        ),
        dest / "provenance.yaml",
    )
    print(
        f"{strips.era5_wind_mps.notna().sum():,} of {len(strips):,} strips with ERA5 wind; "
        f"grid points {np.median([p['distance_km'] for p in points.values()]):.1f} km away "
        f"(median); written to {portable_path(dest)}"
    )
    return 0


def main(argv: list[str] | None = None) -> int:
    from pathlib import Path

    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    d = sub.add_parser("download")
    d.add_argument("--start", required=True)
    d.add_argument("--end", required=True)
    d.add_argument("--refresh", action="store_true")
    e = sub.add_parser("extract")
    for p in (d, e):
        p.add_argument("--package", type=Path, required=True)
    args = ap.parse_args(argv)
    return download(args) if args.cmd == "download" else extract(args)


if __name__ == "__main__":
    raise SystemExit(main())
