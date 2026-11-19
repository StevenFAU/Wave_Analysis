"""Prepare a versioned buoy target table using the S3 station/time joins."""

from __future__ import annotations

import argparse
import hashlib
import itertools
import json
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd

from buoycam_spectral_labels import JOIN, WINDOW, spectral_table
from wave_analysis.config import data_dir
from wave_analysis.datasets.buoycam import spec_records
from wave_analysis.physics.dispersion import wavenumber
from wave_analysis.processing.spectra import frequency_bandwidths

# NDBC WPM band 1 is absent from the 46-band realtime product.
CENTERS = np.r_[
    0.0325 + 0.005 * np.arange(13), 0.10 + 0.01 * np.arange(26), 0.365 + 0.02 * np.arange(7)
]
WIDTHS = np.repeat([0.005, 0.01, 0.02], [13, 26, 7])
# Edges lie on WPM band boundaries. Report all energy in the observed range.
EDGES = np.array([0.030, 0.095, 0.155, 0.255, 0.495])
BANDS = [f"band_{i}_m2" for i in range(len(EDGES) - 1)]


def station_labels(raw, station, strips, depth):
    table, freq = spectral_table(raw, station)
    if freq is None or not np.allclose(freq, CENTERS, atol=0.00051, rtol=0):
        raise ValueError(f"{station}: expected the realtime 46-band WPM grid")
    summaries = spec_records(
        sorted((raw / station / "spec").glob(f"{station}.spec.*Z.gz")), station
    )
    # Reproduce S3's nominal record centres and nearest spectral stamp.
    joined = pd.merge_asof(
        summaries[["time_utc", "Hs_m"]],
        pd.DataFrame({"spectral_time": table.index}),
        left_on="time_utc",
        right_on="spectral_time",
        tolerance=JOIN,
        direction="nearest",
    ).dropna(subset=["spectral_time"])
    spec = table.loc[joined.spectral_time]
    density = spec["swden"][list(range(len(freq)))].to_numpy(float)
    valid = np.isfinite(density).all(axis=1) & (density >= 0).all(axis=1)
    spec, joined, density = spec.iloc[valid], joined.iloc[valid], density[valid]
    energy = density * WIDTHS
    mass = energy.sum(axis=1)
    phase_k = wavenumber(CENTERS)
    old_h = 4 * np.sqrt(density @ frequency_bandwidths(freq))
    new_h = 4 * np.sqrt(mass)
    ratio_use = joined.Hs_m.to_numpy() >= 0.5
    reference = joined.Hs_m.to_numpy()[ratio_use]
    check = {
        "station_id": station,
        "valid_records": len(mass),
        "ratio_records": int(ratio_use.sum()),
        "median_old_hs_ratio": np.median(old_h[ratio_use] / reference)
        if len(reference)
        else np.nan,
        "median_nominal_hs_ratio": np.median(new_h[ratio_use] / reference)
        if len(reference)
        else np.nan,
    }
    sep = spec["swden"]["sep"].to_numpy(float)
    wind = (sep[:, None] <= CENTERS) & (sep[:, None] > 0) & (sep[:, None] < 1)
    ew = energy * wind
    a = spec["swdir"].to_numpy(float) if "swdir" in spec else np.full_like(energy, np.nan)
    r = spec["swr1"].to_numpy(float) if "swr1" in spec else np.full_like(energy, np.nan)
    good = np.isfinite(a) & (a >= 0) & (a <= 360) & (r >= 0) & (r <= 1)
    ew_good = np.where(good, ew, 0.0)
    vector = np.where(good, r * np.exp(1j * np.deg2rad(a)), 0.0) * ew_good
    center = pd.DatetimeIndex(joined.time_utc)
    spectral_time = pd.DatetimeIndex(joined.spectral_time)
    bands = [((low <= CENTERS) & (high > CENTERS)) for low, high in itertools.pairwise(EDGES)]
    records = []
    for item in strips.itertuples():
        t = pd.Timestamp(item.time_utc)
        sl = slice(center.searchsorted(t - WINDOW), center.searchsorted(t + WINDOW, side="right"))
        times = center[sl]
        row = {"sample_id": item.sample_id, "n_complete_spectra": len(times)}
        if len(times):
            i = np.argmin(np.abs(times - t))
            avg = density[sl].mean(axis=0)
            e = avg * WIDTHS
            nz = mass[sl] > 0
            peaks = 1 / freq[density[sl][nz].argmax(axis=1)] if nz.any() else np.array([])
            row.update(
                {
                    "nearest_record_offset_s": (times[i] - t).total_seconds(),
                    "spectral_join_offset_s": (spectral_time[sl][i] - times[i]).total_seconds(),
                    "tp_nearest_s": 1 / freq[density[sl][i].argmax()] if nz[i] else np.nan,
                    "tp_window_spectrum_s": 1 / freq[avg.argmax()] if e.sum() else np.nan,
                    "tp_peak_range_s": np.ptp(peaks) if len(peaks) else np.nan,
                    "slope_rms": np.sqrt(e @ phase_k**2),
                    "slope_below_035": np.sqrt((e * (CENTERS <= 0.35 + 1e-8)) @ phase_k**2),
                    "slope_finite_depth": np.sqrt(e @ wavenumber(CENTERS, depth) ** 2)
                    if pd.notna(depth)
                    else np.nan,
                }
            )
            row.update({col: e[mask].sum() for col, mask in zip(BANDS, bands, strict=True)})
            total = ew[sl].sum()
            measured = ew_good[sl].sum()
            coverage = measured / total if total else np.nan
            row["direction_energy_coverage"] = coverage
            # All records in this window need a defined partition for a window-wide width.
            if measured and coverage >= 0.95 and wind[sl].any(axis=1).all():
                resultant = min(1.0, abs(vector[sl].sum()) / measured)
                row["wind_r1"] = resultant
                row["wind_spread_deg"] = np.rad2deg(np.sqrt(2 * (1 - resultant)))
        records.append(row)
    return pd.DataFrame(records), check


def hash_file(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def prepare_station(job):
    raw, sid, strips, depth, cache, version = job
    inputs = {}
    for product in ["spec", "swden", "swdir", "swr1"]:
        for snapshot in sorted((raw / sid / product).glob("*Z.gz")):
            inputs[str(snapshot.relative_to(raw))] = hash_file(snapshot)
    key = hashlib.sha256(
        json.dumps([version, str(depth), inputs], sort_keys=True).encode()
    ).hexdigest()
    dest = cache / f"{sid}-{key}"
    if not dest.exists():
        from tempfile import TemporaryDirectory

        with TemporaryDirectory(dir=cache) as tmp:
            part, check = station_labels(raw, sid, strips, depth)
            entry = Path(tmp) / "station"
            entry.mkdir()
            part.to_parquet(entry / "rows.parquet", index=False)
            (entry / "check.json").write_text(json.dumps(check))
            try:
                entry.rename(dest)
            except OSError:
                if not dest.is_dir():
                    raise
    print(f"Prepared station {sid}", flush=True)
    return dest, inputs


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--workers", type=int, default=2)
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--station", action="append", help="build a separate station subset for QA")
    ap.add_argument(
        "--depths", type=Path, help="CSV station_id,depth_m with documented water depths"
    )
    args = ap.parse_args()
    if args.out.exists():
        raise SystemExit(f"{args.out} already exists. Use a new version directory.")
    from tempfile import TemporaryDirectory

    src = data_dir("processed") / "packages/ndbc_buoycam_spectral_v3cv"
    path = src / "labels.csv"
    labels = pd.read_csv(path, dtype={"station_id": str}, low_memory=False)
    if args.station:
        if set(args.station) - set(labels.station_id):
            raise ValueError("unknown station")
        labels = labels[labels.station_id.isin(args.station)].copy()
    depths = (
        pd.read_csv(args.depths, dtype={"station_id": str}).set_index("station_id").depth_m
        if args.depths
        else {}
    )
    raw = data_dir("raw") / "ndbc"
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(dir=args.out.parent) as tmp:
        stage = Path(tmp) / "package"
        stage.mkdir()
        pieces, checks, inputs = [], [], {}
        cache = args.out.with_name(args.out.name + ".parts")
        cache.mkdir(exist_ok=True)
        version = hash_file(path) + hash_file(Path(__file__))
        jobs = [
            (raw, sid, group[["sample_id", "time_utc"]], depths.get(sid, np.nan), cache, version)
            for sid, group in labels.groupby("station_id")
        ]
        with ProcessPoolExecutor(max_workers=args.workers) as pool:
            for entry, source_hashes in pool.map(prepare_station, jobs):
                pieces.append(pd.read_parquet(entry / "rows.parquet"))
                checks.append(json.loads((entry / "check.json").read_text()))
                inputs.update(source_hashes)
        all_rows = labels.merge(pd.concat(pieces), on="sample_id", validate="1:1")
        target = "slope_rms"
        # Reuse the height trainer's 0.05 numeric log floor at a physically small slope of 0.0005.
        all_rows["slope_percent"] = 100 * all_rows[target]
        all_rows.to_csv(stage / "labels.csv", index=False)
        pd.DataFrame(checks).to_csv(stage / "bandwidth_checks.csv", index=False)
        (stage / "images").symlink_to((src / "images").resolve(), target_is_directory=True)
        (stage / "provenance.json").write_text(
            json.dumps(
                {
                    "source_package": src.name,
                    "label_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                    "raw_snapshot_sha256": inputs,
                    "station_subset": args.station,
                    "depths_file": str(args.depths) if args.depths else None,
                    "edge_hz": EDGES.tolist(),
                    "coverage_min": 0.95,
                    "window_min": 30,
                    "join_min": 15,
                    "peak_centers": "rounded file headers, as in S3",
                    "energy_centers": "nominal WPM",
                    "complete_density_only": True,
                    "n_strips": len(all_rows),
                },
                indent=2,
            )
        )
        stage.rename(args.out)
    print("Prepared", len(all_rows), "strips at", len(checks), "stations in", args.out)


if __name__ == "__main__":
    main()
