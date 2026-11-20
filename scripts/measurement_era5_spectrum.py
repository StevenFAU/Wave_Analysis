"""Prepare one ERA5 wave-spectra request and align its band energies to a buoy package."""

import argparse
import itertools
import json
from pathlib import Path

import numpy as np
import pandas as pd

from measurement_labels import EDGES
from wave_analysis.config import data_dir


def request(row, date):
    lat, lon = row.grid_latitude, row.grid_longitude
    return {
        "class": "ea",
        "expver": "1",
        "stream": "wave",
        "type": "an",
        "levtype": "sfc",
        "param": "140251",
        "date": date,
        "time": "00/to/23/by/1",
        "direction": "1/to/24",
        "frequency": "1/to/30",
        "area": f"{lat + 0.5}/{lon - 0.5}/{lat - 0.5}/{lon + 0.5}",
        "grid": "0.5/0.5",
        "format": "netcdf",
    }


def decode(nc, row, labels):
    import xarray as xr

    with xr.open_dataset(nc) as ds:
        names = [k for k, v in ds.data_vars.items() if {"direction", "frequency"}.issubset(v.dims)]
        if len(names) != 1:
            raise ValueError(
                "expected one log-density field with frequency and direction dimensions"
            )
        arr = ds[names[0]]
        if arr.sizes["frequency"] != 30 or arr.sizes["direction"] != 24:
            raise ValueError("need all 30 ECMWF frequencies and 24 directions in order")
        for dim, n in [("frequency", 30), ("direction", 24)]:
            if not np.array_equal(arr[dim].to_numpy(), np.arange(1, n + 1)):
                raise ValueError(f"unexpected encoded {dim} coordinate")
        lon = (row.grid_longitude + 360) % 360 if ds.longitude.min() >= 0 else row.grid_longitude
        arr = arr.sel(latitude=row.grid_latitude, longitude=lon)
        t = "valid_time" if "valid_time" in arr.dims else "time"
        log_density = arr.transpose(t, "frequency", "direction").to_numpy()
        hours = pd.to_datetime(arr[t].to_numpy(), utc=True)
    valid = np.isfinite(log_density).any(axis=(1, 2))
    sf = (
        (10.0 ** np.where(np.isfinite(log_density), log_density, -np.inf)).sum(axis=2)
        * 2
        * np.pi
        / 24
    )
    fc = 0.03453 * 1.1 ** np.arange(30)
    edges = np.r_[fc[0] / np.sqrt(1.1), fc * np.sqrt(1.1)]
    spec = pd.DataFrame({"time_utc": hours})
    for j, (low, high) in enumerate(itertools.pairwise(EDGES)):
        if low < edges[0] or high > edges[-1]:
            continue
        overlaps = np.maximum(0, np.minimum(high, edges[1:]) - np.maximum(low, edges[:-1]))
        spec[f"prior_band_{j}_m2"] = np.where(valid, sf @ overlaps, np.nan)
    # Only exact hours or bracketing observations one hour apart qualify.
    samples = labels.loc[
        labels.station_id.eq(str(row.station_id)), ["sample_id", "time_utc"]
    ].copy()
    samples["time_utc"] = pd.to_datetime(samples.time_utc, utc=True)
    left = samples.time_utc.dt.floor("h")
    right = samples.time_utc.dt.ceil("h")
    for col in spec.columns[1:]:
        lookup = spec.set_index("time_utc")[col]
        weight = (samples.time_utc - left).dt.total_seconds() / 3600
        samples[col] = (1 - weight) * left.map(lookup) + weight * right.map(lookup)
    return samples.dropna()


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--station", required=True)
    p.add_argument("--date", help="YYYY-MM-DD or a MARS date range")
    p.add_argument("--netcdf", type=Path, help="decode an already retrieved NetCDF")
    p.add_argument("--labels", type=Path)
    p.add_argument("--out", type=Path, required=True)
    a = p.parse_args()
    sites = pd.read_csv(
        data_dir("processed") / "era5/buoycam/points.csv", dtype={"station_id": str}
    )
    row = sites[sites.station_id.eq(a.station)].iloc[0]
    if a.out.exists():
        raise FileExistsError(a.out)
    if a.netcdf:
        if not a.labels:
            p.error("--labels required for decoding")
        d = pd.read_csv(
            a.labels, usecols=["station_id", "time_utc", "sample_id"], dtype={"station_id": str}
        )
        decode(a.netcdf, row, d).to_csv(a.out, index=False)
    else:
        if not a.date:
            p.error("--date required for request preparation")
        a.out.write_text(json.dumps(request(row, a.date), indent=2))


if __name__ == "__main__":
    main()
