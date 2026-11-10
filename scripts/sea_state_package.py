#!/usr/bin/env python
"""Build the training package of a sea-state experiment, and score the baselines that use no image.

    .venv/bin/python scripts/sea_state_package.py configs/experiments/currituck_sea_state_v0.yaml

Reads the pairing table named in the config, keeps the stills its filters
allow, gives each a WMO sea-state class, makes the two splits (whole local
days; a held-out year) and audits them for leakage. It then scores the
baselines (majority class, monthly climatology, and each available wave
nowcast with a linear correction fitted on each split's training stills) on
every test partition, and writes the package:

    <package.out>/
      images/YYYY/MM/DD/<still>.jpg   resized copies (long side package.long_side_px)
      labels.csv                      one row per still: class, label, splits, checksums
      baselines.csv                   per still: each baseline's class for each split
      exclusions.csv                  stills left out and the rule that left each out
      summary.json                    class counts, leakage audits, baseline metrics
      config.yaml                     the experiment config as used
      provenance.yaml                 git commit, inputs and their checksums
      SHA256SUMS                      every file; `sha256sum -c SHA256SUMS` checks a copy
      README.md

Only the package goes to the GPU cluster; the raw archive stays here. A rerun
reuses images already in the package (checked against the previous
labels.csv) and resizes only new ones. ``--limit N`` builds a small package
of N stills per split partition for smoke tests.
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
from PIL import Image

from wave_analysis.config import data_dir, load_yaml, repo_root
from wave_analysis.datasets import build_manifest, build_provenance, check_leakage, write_provenance
from wave_analysis.datasets.sea_state import (
    TEST_LATER,
    ResizeSpec,
    add_calendar,
    class_table,
    day_split,
    default_workers,
    linear_correction,
    majority_baseline,
    monthly_baseline,
    near_boundary,
    nowcast_at,
    package_path,
    resize_still,
    sea_state_class,
    write_checksums,
    year_split,
)
from wave_analysis.datasets.splits import TEST, TRAIN, VAL
from wave_analysis.evaluation.sea_state import format_line, report
from wave_analysis.ingest.checksum import sha256_file
from wave_analysis.ingest.durable import atomic_open

SPLITS = ("day", "year")


def load_nowcast(spec: dict[str, Any]) -> pd.Series | None:
    """A standardized nowcast series (UTC time index), or ``None`` if optional and absent."""
    path = data_dir("processed") / spec["path"]
    if not path.exists():
        if spec.get("optional"):
            return None
        raise FileNotFoundError(f"{path} (run 'wave-analysis wis|era5 standardize')")
    d = pd.read_parquet(path, columns=["time_utc", "variable", "value"])
    d = d[d["variable"] == spec["variable"]]
    return d.set_index(pd.to_datetime(d["time_utc"], utc=True))["value"].sort_index()


def _resize(job: tuple[str, str, int, int]) -> tuple[str, str, int, int, int]:
    src, dest, side, quality = job
    sha, size, w, h = resize_still(src, dest, ResizeSpec(side, quality))
    return dest, sha, size, w, h


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("config")
    ap.add_argument("--out", help="package directory (default: package.out of the config)")
    ap.add_argument("--limit", type=int, help="stills per partition (smoke-test package)")
    ap.add_argument("--workers", type=int, default=default_workers())
    ap.add_argument("--no-images", action="store_true", help="labels, splits and baselines only")
    args = ap.parse_args(argv)

    root = repo_root()
    cfg = load_yaml(args.config)
    task, pkg = cfg["task"], cfg["package"]
    names = list(task["classes"]["names"])
    bounds = [float(b) for b in task["classes"]["bounds_m"]]
    samples_path = root / cfg["input"]["samples"]
    out = Path(args.out) if args.out else root / pkg["out"]

    raw = pd.read_parquet(samples_path)
    raw = add_calendar(raw, tz=cfg["input"]["timezone"])
    built = build_manifest(
        raw, id_col=task["id_col"], target_cols=[task["label"]], filters=task["filters"]
    )
    m = built.manifest
    hs = m[task["label"]].to_numpy(dtype=float)
    m["class"] = sea_state_class(hs, bounds)
    m["class_name"] = [names[c] for c in m["class"]]
    m["near_boundary"] = near_boundary(hs, bounds, rel_tol=float(task["near_boundary_rel_tol"]))

    splits = {"day": day_split(m, cfg["split"]["day"]), "year": year_split(m, cfg["split"]["year"])}
    audits = {}
    for k, sp in splits.items():
        m[f"split_{k}"] = sp.labels.to_numpy()
        rep = check_leakage(m, sp, **cfg["split"].get("audit", {}))
        audits[k] = {
            "ok": rep.ok,
            "shared_groups": {c: len(v) for c, v in rep.shared_groups.items()},
            "temporal_neighbours": rep.temporal_neighbours,
            "params": sp.params,
        }

    if args.limit:
        rng = np.random.default_rng(int(cfg.get("seed", 0)))
        keep = np.zeros(len(m), bool)
        for k in SPLITS:
            for part in m[f"split_{k}"].unique():
                idx = np.flatnonzero(m[f"split_{k}"].to_numpy() == part)
                keep[rng.choice(idx, min(args.limit, idx.size), replace=False)] = True
        m = m.loc[keep].reset_index(drop=True)

    # ------------------------------------------------------------------ baselines
    y = m["class"].to_numpy()
    hs = m[task["label"]].to_numpy(dtype=float)
    month = m["local_month"].to_numpy()
    nowcasts = {
        k: load_nowcast(spec) for k, spec in cfg.get("baselines", {}).get("nowcasts", {}).items()
    }
    base = pd.DataFrame({task["id_col"]: m[task["id_col"]]})
    corrections: dict[str, dict[str, Any]] = {}
    for name, series in nowcasts.items():
        if series is None:
            continue
        base[f"{name}_hs_m"] = nowcast_at(m["time_utc"], series)
    for k in SPLITS:
        part = m[f"split_{k}"].to_numpy()
        train = part == TRAIN
        base[f"majority_{k}"] = majority_baseline(y, train)
        base[f"monthly_{k}"] = monthly_baseline(y, month, train)
        for name, series in nowcasts.items():
            if series is None:
                continue
            x = base[f"{name}_hs_m"].to_numpy()
            a, b = linear_correction(x, hs, train)
            base[f"{name}_{k}"] = sea_state_class(a + b * x, bounds)
            ok = train & np.isfinite(x) & np.isfinite(hs)
            corrections[f"{name}_{k}"] = {"a": a, "b": b, "n_train": int(ok.sum())}
    baseline_names = sorted(
        {c.rsplit("_", 1)[0] for c in base.columns if c.endswith(SPLITS) and "_hs_" not in c}
    )

    boot = cfg["evaluation"]["bootstrap"]
    scores: dict[str, Any] = {}
    for k in SPLITS:
        part = m[f"split_{k}"].to_numpy()
        for p in (VAL, TEST, TEST_LATER):
            rows = part == p
            if not rows.any():
                continue
            for bname in baseline_names:
                pred = base[f"{bname}_{k}"].to_numpy()
                avail = rows & (pred >= 0)
                rep = report(
                    y[avail], pred[avail],
                    groups=m.loc[avail, boot["group_col"]].to_numpy(),
                    names=names, near_boundary=m.loc[avail, "near_boundary"].to_numpy(),
                    n_boot=int(boot["n_boot"]), alpha=float(boot["alpha"]), seed=int(boot["seed"]),
                )  # fmt: skip
                rep["coverage"] = float(avail.sum() / rows.sum())
                scores[f"{k}/{p}/{bname}"] = rep
                print(format_line(f"{k:>4} {p:<10} {bname:<9}", rep))

    # ------------------------------------------------------------------ package
    out.mkdir(parents=True, exist_ok=True)
    m["image"] = [package_path(t, n) for t, n in zip(m["time_utc"], m["name"], strict=True)]
    previous: dict[str, str] = {}
    if (out / "labels.csv").exists():
        prev = pd.read_csv(out / "labels.csv", usecols=["image", "sha256"])
        previous = dict(zip(prev["image"], prev["sha256"], strict=True))
    sha = pd.Series("", index=m.index, dtype=object)
    size = pd.Series(0, index=m.index, dtype="int64")
    width = pd.Series(0, index=m.index, dtype="int64")
    height = pd.Series(0, index=m.index, dtype="int64")
    jobs = []
    for i, (img, src) in enumerate(zip(m["image"], m["local_path"], strict=True)):
        dest = out / img
        if dest.exists() and img in previous and sha256_file(dest) == previous[img]:
            with Image.open(dest) as im:
                width[i], height[i] = im.size
            sha[i], size[i] = previous[img], dest.stat().st_size
        elif not args.no_images:
            jobs.append(
                (str(root / src), str(dest), int(pkg["long_side_px"]), int(pkg["jpeg_quality"]))
            )
    if jobs:
        print(f"resizing {len(jobs):,} stills with {args.workers} processes ...", flush=True)
        where = {str(out / img): i for i, img in enumerate(m["image"])}
        with Pool(args.workers) as pool:
            for n, (dest, h, nbytes, w, hh) in enumerate(pool.imap_unordered(_resize, jobs, 64), 1):
                i = where[dest]
                sha[i], size[i], width[i], height[i] = h, nbytes, w, hh
                if n % 5000 == 0:
                    print(f"  {n:,}/{len(jobs):,}", flush=True)
    # "sha256" now describes the packaged copy; the source's is kept as source_sha256.
    m["sha256"], m["size_bytes_package"], m["width_px"], m["height_px"] = sha, size, width, height

    label_cols = [
        task["id_col"], "image", "sha256", "size_bytes_package", "width_px", "height_px",
        "time_utc", "local_day", "local_week", "local_month", "local_year",
        "illumination", "sun_elevation_deg", task["label"], "Hs_label_spread_m",
        "n_label_records", "Hs_nearest_m", "class", "class_name", "near_boundary",
        "split_day", "split_year", "local_path",
    ]  # fmt: skip
    labels = m[label_cols].rename(columns={"local_path": "source_path"})
    labels.insert(labels.columns.get_loc("source_path") + 1, "source_sha256", raw.set_index(
        task["id_col"]).loc[labels[task["id_col"]], "sha256"].to_numpy())  # fmt: skip
    labels["time_utc"] = pd.to_datetime(labels["time_utc"], utc=True).dt.strftime(
        "%Y-%m-%dT%H:%M:%SZ"
    )
    written = []
    for fname, frame in (
        ("labels.csv", labels),
        ("baselines.csv", base),
        ("exclusions.csv", built.exclusions),
    ):
        with atomic_open(out / fname, "w", newline="") as fh:
            frame.to_csv(fh, index=False, float_format="%.6g")
        written.append(fname)
    shutil.copyfile(args.config, out / "config.yaml")
    written.append("config.yaml")

    tables = {
        k: {
            p: class_table(m[m[f"split_{k}"] == p], "class", names, "local_year").to_dict("index")
            for p in m[f"split_{k}"].unique()
        }
        for k in SPLITS
    }
    summary = {
        "name": cfg["name"],
        "stills": len(m),
        "days": int(m["local_day"].nunique()),
        "excluded": len(built.exclusions),
        "excluded_by": built.exclusions["excluded_by"].value_counts().to_dict(),
        "classes": dict(zip(names, np.bincount(y, minlength=len(names)).tolist(), strict=True)),
        "class_days": {
            names[c]: int(m.loc[m["class"] == c, "local_day"].nunique()) for c in range(len(names))
        },
        "near_boundary_share": float(m["near_boundary"].mean()),
        "majority_share": float(np.bincount(y).max() / len(y)),
        "by_split": tables,
        "leakage_audit": audits,
        "nowcast_corrections": corrections,
        "nowcasts_available": {k: v is not None for k, v in nowcasts.items()},
        "baselines": scores,
        "limit_per_partition": args.limit,
    }
    with atomic_open(out / "summary.json", "w") as fh:
        json.dump(summary, fh, indent=1, default=str)
    written.append("summary.json")
    (out / "README.md").write_text(
        f"# {cfg['name']}\n\n{cfg['description'].strip()}\n\n"
        "Built by `scripts/sea_state_package.py` from the Wave_Analysis repository; "
        "see `provenance.yaml` for the commit and inputs, and "
        "`docs/experiments/currituck_sea_state.md` for the method.\n\n"
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
                f"{len(m):,} stills; Pillow {PIL.__version__}; images resized to long side "
                f"{pkg['long_side_px']} px, JPEG quality {pkg['jpeg_quality']}; leakage audits "
                + ", ".join(f"{k} ok={a['ok']}" for k, a in audits.items())
            ),
        ),
        out / "provenance.yaml",
    )
    total = int(size.sum())
    leak = ", ".join(f"{k}={a['ok']}" for k, a in audits.items())
    print(
        f"{cfg['name']}: {len(m):,} stills on {summary['days']:,} days -> {out} "
        f"({total / 1e9:.2f} GB of images); classes {summary['classes']}; leakage ok: {leak}"
    )
    return 0 if all(a["ok"] for a in audits.values()) else 2


if __name__ == "__main__":
    sys.exit(main())
