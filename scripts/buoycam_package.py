#!/usr/bin/env python
"""Build the training package of the NDBC buoy-camera sea-state experiment, and score the baselines that use no image.

    .venv/bin/python scripts/buoycam_package.py configs/experiments/ndbc_buoycam_sea_state_v0.yaml

Reads the pairing table (``scripts/buoycam_pairs.py``), keeps the strips the
config's filters allow, gives each a WMO sea-state class, makes the two splits
(unseen buoys, grouped by distance; later weeks of the same buoys) and audits
them for leakage. It scores the baselines (majority class; for the time split
also each buoy's own majority class) and writes the package:

    <package.out>/
      images/<station>/YYYY/MM/<strip>.jpg   strips without their caption band
      labels.csv                             one row per strip: class, label, splits, view stats
      baselines.csv                          per strip: each baseline's class for each split
      exclusions.csv                         strips left out and the rule that left each out
      stations.csv                           position, distance group and partition of each buoy
      summary.json                           class counts, leakage audits, baseline metrics
      config.yaml, provenance.yaml, SHA256SUMS, README.md

The caption (station ID, UTC time) is cut from every packaged image, so a
model cannot read where or when a strip was taken. A rerun reuses images
already in the package (checked against the previous labels.csv).
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from multiprocessing import Pool
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import PIL

from wave_analysis.config import load_yaml, repo_root
from wave_analysis.datasets import build_manifest, build_provenance, check_leakage, write_provenance
from wave_analysis.datasets.buoycam import (
    check_filters,
    crop_strip,
    spatial_groups,
    usable_views,
    view_columns,
)
from wave_analysis.datasets.sea_state import (
    default_workers,
    majority_baseline,
    near_boundary,
    sea_state_class,
    write_checksums,
)
from wave_analysis.datasets.splits import TEST, TRAIN, VAL, grouped_split, time_block_split
from wave_analysis.evaluation.sea_state import format_line, report
from wave_analysis.ingest.checksum import sha256_file
from wave_analysis.ingest.durable import atomic_open

SPLITS = ("station", "time")


def station_majority(
    y: np.ndarray, station: np.ndarray, train: np.ndarray, *, fallback: np.ndarray
) -> np.ndarray:
    """Each buoy's most frequent training class; ``fallback`` where a buoy has no training strip."""
    out = fallback.copy()
    for st in np.unique(station):
        tr = train & (station == st)
        if tr.any():
            out[station == st] = int(np.bincount(y[tr]).argmax())
    return out


def _crop(job: tuple[str, str, int]) -> tuple[str, str, int, int, int]:
    src, dest, quality = job
    sha, size, w, h = crop_strip(src, dest, quality=quality)
    return dest, sha, size, w, h


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("config")
    ap.add_argument("--out", help="package directory (default: package.out of the config)")
    ap.add_argument("--workers", type=int, default=default_workers())
    ap.add_argument("--no-images", action="store_true", help="labels, splits and baselines only")
    args = ap.parse_args(argv)

    root = repo_root()
    cfg = load_yaml(args.config)
    task, pkg, ims = cfg["task"], cfg["package"], cfg["images"]
    check_filters(ims)
    names = list(task["classes"]["names"])
    bounds = [float(b) for b in task["classes"]["bounds_m"]]
    samples_path = root / cfg["input"]["samples"]
    out = Path(args.out) if args.out else root / pkg["out"]

    raw = pd.read_parquet(samples_path)
    t = pd.to_datetime(raw["time_utc"], utc=True)
    period = (t >= pd.Timestamp(cfg["input"]["start_utc"])) & (
        t < pd.Timestamp(cfg["input"]["end_utc"])
    )
    raw = raw[period].reset_index(drop=True)
    raw["usable_views"] = usable_views(
        raw, min_mean=float(ims["min_view_mean"]), min_std=float(ims["min_view_std"])
    )
    raw["station_day"] = raw["station_id"] + "|" + raw["time_utc"].dt.strftime("%Y-%m-%d")
    built = build_manifest(
        raw, id_col=task["id_col"], target_cols=[task["label"]], filters=task["filters"]
    )
    m = built.manifest
    hs = m[task["label"]].to_numpy(dtype=float)
    m["class"] = sea_state_class(hs, bounds)
    m["class_name"] = [names[c] for c in m["class"]]
    m["near_boundary"] = near_boundary(hs, bounds, rel_tol=float(task["near_boundary_rel_tol"]))

    # ------------------------------------------------------------------ splits
    sp_cfg = cfg["split"]
    stations = (
        m.groupby("station_id")[["latitude", "longitude"]]
        .first()
        .reset_index()
        .sort_values("station_id")
    )
    group_of = spatial_groups(
        stations["station_id"], stations["latitude"], stations["longitude"],
        km=float(sp_cfg["station"]["group_km"]),
    )  # fmt: skip
    m["station_group"] = m["station_id"].map(group_of)
    splits = {
        "station": grouped_split(
            m, "station_group", sp_cfg["station"]["fractions"],
            seed=int(sp_cfg["station"]["seed"]), protocol="cross_station",
        ),
        "time": time_block_split(
            m, "time_utc", val_start=sp_cfg["time"]["val_start"],
            test_start=sp_cfg["time"]["test_start"], embargo=sp_cfg["time"]["embargo"],
        ),
    }  # fmt: skip
    audits = {}
    gap = sp_cfg["audit"]["min_gap"]
    for k, sp in splits.items():
        m[f"split_{k}"] = sp.labels.to_numpy()
        rep = (
            check_leakage(m, sp, group_cols=["station_id", "station_group"])
            if k == "station"
            else check_leakage(m, sp, time_col="time_utc", time_group_col="station_id", min_gap=gap)
        )
        audits[k] = {
            "ok": rep.ok,
            "shared_groups": {c: len(v) for c, v in rep.shared_groups.items()},
            "temporal_neighbours": rep.temporal_neighbours,
            "params": sp.params,
        }
    part_of = m.groupby("station_id")["split_station"].first()
    stations["station_group"] = stations["station_id"].map(group_of)
    stations["split_station"] = stations["station_id"].map(part_of)
    stations["strips"] = stations["station_id"].map(m["station_id"].value_counts())

    # ------------------------------------------------------------------ baselines
    y = m["class"].to_numpy()
    st = m["station_id"].to_numpy()
    base = pd.DataFrame({task["id_col"]: m[task["id_col"]]})
    for k in SPLITS:
        train = m[f"split_{k}"].to_numpy() == TRAIN
        base[f"majority_{k}"] = majority_baseline(y, train)
        if k in cfg["baselines"].get("station_majority", []):
            base[f"station_majority_{k}"] = station_majority(
                y, st, train, fallback=base[f"majority_{k}"].to_numpy()
            )
    boot = cfg["evaluation"]["bootstrap"]
    scores: dict[str, Any] = {}
    for k in SPLITS:
        part = m[f"split_{k}"].to_numpy()
        for p in (VAL, TEST):
            rows = part == p
            if not rows.any():
                continue
            for col in [c for c in base.columns if c.endswith(f"_{k}")]:
                pred = base[col].to_numpy()
                rep = report(
                    y[rows], pred[rows],
                    groups=m.loc[rows, boot["group_col"][k]].to_numpy(),
                    names=names, near_boundary=m.loc[rows, "near_boundary"].to_numpy(),
                    n_boot=int(boot["n_boot"]), alpha=float(boot["alpha"]), seed=int(boot["seed"]),
                )  # fmt: skip
                bname = col.removesuffix(f"_{k}")
                scores[f"{k}/{p}/{bname}"] = rep
                print(format_line(f"{k:>7} {p:<4} {bname:<16}", rep))

    # ------------------------------------------------------------------ package
    out.mkdir(parents=True, exist_ok=True)
    m["image"] = [
        f"images/{s}/{t:%Y}/{t:%m}/{Path(src).name}"
        for s, t, src in zip(m["station_id"], m["time_utc"], m["local_path"], strict=True)
    ]
    previous: dict[str, str] = {}
    if (out / "labels.csv").exists():
        prev = pd.read_csv(out / "labels.csv", usecols=["image", "sha256_package"])
        previous = dict(zip(prev["image"], prev["sha256_package"], strict=True))
    sha = pd.Series("", index=m.index, dtype=object)
    size = pd.Series(0, index=m.index, dtype="int64")
    jobs = []
    for i, (img, src) in enumerate(zip(m["image"], m["local_path"], strict=True)):
        dest = out / img
        if dest.exists() and img in previous and sha256_file(dest) == previous[img]:
            sha[i], size[i] = previous[img], dest.stat().st_size
        elif not args.no_images:
            jobs.append((str(root / src), str(dest), int(pkg["jpeg_quality"])))
    if jobs:
        print(f"cropping {len(jobs):,} strips with {args.workers} processes ...", flush=True)
        where = {str(out / img): i for i, img in enumerate(m["image"])}
        with Pool(args.workers) as pool:
            for n, (dest, h, nbytes, _w, _h) in enumerate(pool.imap_unordered(_crop, jobs, 64), 1):
                i = where[dest]
                sha[i], size[i] = h, nbytes
                if n % 10000 == 0:
                    print(f"  {n:,}/{len(jobs):,}", flush=True)
    m["sha256_package"], m["size_bytes_package"] = sha, size

    label_cols = [
        task["id_col"], "image", "sha256_package", "size_bytes_package", "width", "height",
        "station_id", "camera_code", "station_group", "latitude", "longitude", "time_utc",
        "station_day", "sun_elevation_deg", "sun_azimuth_deg", "illumination",
        *view_columns("view_mean"), *view_columns("view_std"), "usable_views",
        task["label"], "Hs_label_spread_m", "n_label_records", "Hs_nearest_m", "Hs_interp_m",
        "swell_height_m", "wind_wave_height_m", "mean_period_s", "mean_direction_deg",
        "steepness", "class", "class_name", "near_boundary", "split_station", "split_time",
        "local_path", "sha256",
    ]  # fmt: skip
    labels = m[label_cols].rename(columns={"local_path": "source_path", "sha256": "source_sha256"})
    labels["time_utc"] = pd.to_datetime(labels["time_utc"], utc=True).dt.strftime(
        "%Y-%m-%dT%H:%M:%SZ"
    )
    written = []
    for fname, frame in (
        ("labels.csv", labels),
        ("baselines.csv", base),
        ("exclusions.csv", built.exclusions[[task["id_col"], "excluded_by"]]),
        ("stations.csv", stations),
    ):
        with atomic_open(out / fname, "w", newline="") as fh:
            frame.to_csv(fh, index=False, float_format="%.6g")
        written.append(fname)
    shutil.copyfile(args.config, out / "config.yaml")
    written.append("config.yaml")

    def counts(frame: pd.DataFrame) -> dict[str, int]:
        return dict(
            zip(names, np.bincount(frame["class"], minlength=len(names)).tolist(), strict=True)
        )

    summary = {
        "name": cfg["name"],
        "strips": len(m),
        "stations": int(m["station_id"].nunique()),
        "station_groups": int(m["station_group"].nunique()),
        "first": str(m["time_utc"].min()),
        "last": str(m["time_utc"].max()),
        "excluded": len(built.exclusions),
        "excluded_by": built.exclusions["excluded_by"].value_counts().to_dict(),
        "classes": counts(m),
        "by_illumination": m["illumination"].value_counts().to_dict(),
        "near_boundary_share": float(m["near_boundary"].mean()),
        "majority_share": float(np.bincount(y).max() / len(y)),
        "by_split": {
            k: {
                str(p): {
                    "strips": int((m[f"split_{k}"] == p).sum()),
                    "stations": int(m.loc[m[f"split_{k}"] == p, "station_id"].nunique()),
                    "classes": counts(m[m[f"split_{k}"] == p]),
                }
                for p in sorted(m[f"split_{k}"].unique())
            }
            for k in SPLITS
        },
        "station_split_test_stations": sorted(
            stations.loc[stations["split_station"] == TEST, "station_id"]
        ),
        "station_split_val_stations": sorted(
            stations.loc[stations["split_station"] == VAL, "station_id"]
        ),
        "leakage_audit": audits,
        "baselines": scores,
    }
    with atomic_open(out / "summary.json", "w") as fh:
        json.dump(summary, fh, indent=1, default=str)
    written.append("summary.json")
    (out / "README.md").write_text(
        f"# {cfg['name']}\n\n{cfg['description'].strip()}\n\n"
        "Built by `scripts/buoycam_package.py` from the Wave_Analysis repository; "
        "see `provenance.yaml` for the commit and inputs, and "
        "`docs/experiments/ndbc_buoycam_sea_state.md` for the method.\n\n"
        "Images are NDBC buoy-camera strips (U.S. Government work) with the caption band cut.\n\n"
        "Check a copy: `sha256sum -c --quiet SHA256SUMS` (prints nothing when every file is intact).\n",
        encoding="utf-8",
    )
    written.append("README.md")
    images = list(m["image"]) if not args.no_images else []
    write_checksums(out, [*written, *images], out / "SHA256SUMS")
    write_provenance(
        build_provenance(
            artifact_id=cfg["name"],
            inputs=[samples_path, Path(args.config)],
            outputs=[out / f for f in (*written, "SHA256SUMS")],
            config=cfg,
            notes=(
                f"{len(m):,} strips; Pillow {PIL.__version__}; caption band cut, JPEG quality "
                f"{pkg['jpeg_quality']}; leakage audits "
                + ", ".join(f"{k} ok={a['ok']}" for k, a in audits.items())
            ),
        ),
        out / "provenance.yaml",
    )
    leak = ", ".join(f"{k}={a['ok']}" for k, a in audits.items())
    print(
        f"{cfg['name']}: {len(m):,} strips at {summary['stations']} buoys -> {out} "
        f"({int(size.sum()) / 1e9:.2f} GB of images); classes {summary['classes']}; "
        f"leakage ok: {leak}"
    )
    return 0 if all(a["ok"] for a in audits.values()) else 2


if __name__ == "__main__":
    sys.exit(main())
