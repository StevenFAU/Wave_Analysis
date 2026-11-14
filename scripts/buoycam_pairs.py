#!/usr/bin/env python
"""Pair every archived NDBC buoy-camera strip with its own buoy's wave height.

    .venv/bin/python scripts/buoycam_pairs.py

One row per strip in the archive (``data/raw/ndbc/buoycam``), night frames
included; ``exclusion`` gives the reason a row has no usable label. The
camera and the wave sensor are on the same buoy (distance 0). For each strip:

* its geometry, checksum and the brightness and contrast of each of its six
  views (caption excluded; :func:`wave_analysis.datasets.buoycam.strip_stats`);
* the sun's elevation and azimuth at the buoy (median listed position);
* the label (ADR 0010): the mean of the buoy's ``.spec`` WVHT records whose
  nominal centre lies within 30 min of the image, with their spread and
  count, and the ``nearest`` and ``interpolate`` labels beside it; context
  from the nearest record (swell and wind-sea heights and periods, mean
  period and direction, steepness).

The ``.spec`` records come from the realtime snapshots saved twice a month
(``data/raw/ndbc/<station>/spec/``); they are NDBC's realtime values, not the
quality-controlled annual files. Writes
``data/processed/pairs/ndbc_buoycam/{samples.parquet,summary.json,provenance.yaml}``.
Method: ``docs/experiments/ndbc_buoycam_sea_state.md``.
"""

from __future__ import annotations

import argparse
import json
import sys
from multiprocessing import Pool
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from wave_analysis.config import data_dir, portable_path
from wave_analysis.datasets import build_provenance, write_provenance
from wave_analysis.datasets.buoycam import spec_records, station_positions, strip_stats
from wave_analysis.ingest.durable import atomic_open
from wave_analysis.processing.solar import illumination_category, solar_position
from wave_analysis.processing.synchronize import label_from_candidates, pair_window
from wave_analysis.sources.ndbc.buoycam import LISTINGS_DIR, archive_index

#: ADR 0010: records centred within this time of the image make the label.
LABEL_WINDOW = "30min"
#: ``interpolate``: the two records around the image must be this close (one missing record allowed).
INTERP_MAX_GAP = "60min"
CONTEXT = [
    "swell_height_m",
    "swell_period_s",
    "wind_wave_height_m",
    "wind_wave_period_s",
    "mean_period_s",
    "mean_direction_deg",
    "steepness",
    "snapshots_disagree",
]


def _stats(path: str) -> dict[str, Any]:
    return {"path": path, **strip_stats(path)}


def quantiles(s: pd.Series, q: tuple[float, ...] = (0.0, 0.05, 0.5, 0.95, 1.0)) -> dict[str, float]:
    """Rounded quantiles of a numeric series, for the summary."""
    s = s.dropna()
    return {f"q{int(x * 100)}": round(float(s.quantile(x)), 3) for x in q} if len(s) else {}


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--archive", type=Path, default=data_dir("raw") / "ndbc" / "buoycam")
    ap.add_argument("--ndbc", type=Path, default=data_dir("raw") / "ndbc")
    ap.add_argument("--out", type=Path, default=data_dir("processed") / "pairs" / "ndbc_buoycam")
    ap.add_argument("--workers", type=int, default=8)
    args = ap.parse_args(argv)

    # --- Images ----------------------------------------------------------------
    idx = archive_index(args.archive)
    idx["size_bytes"] = [Path(p).stat().st_size for p in idx["path"]]
    print(f"{len(idx):,} archived strips at {idx['station_id'].nunique()} stations", flush=True)
    with Pool(args.workers) as pool:
        stats = pd.DataFrame(pool.map(_stats, idx["path"].tolist(), chunksize=256))
    img = idx.merge(stats, on="path", how="left", validate="one_to_one")
    img["time_utc"] = pd.to_datetime(img["time_utc"], utc=True)
    # The camera code is part of the id: a redeployed buoy can briefly list two
    # cameras for the same hour (46006, 2026-09-26 to 09-28: old hull W67A, new W97A).
    img.insert(
        0,
        "sample_id",
        img["station_id"]
        + "-"
        + img["camera_code"]
        + "-"
        + img["time_utc"].dt.strftime("%Y-%m-%dT%H%MZ"),
    )
    if img["sample_id"].duplicated().any():
        raise SystemExit("two strips share a station, camera and time stamp")
    img["cameras_this_hour"] = img.groupby(["station_id", "time_utc"])["camera_code"].transform(
        "size"
    )

    # --- Positions and the sun --------------------------------------------------
    pos = station_positions(sorted((args.archive / LISTINGS_DIR).glob("*.json")))
    img = img.merge(pos, on="station_id", how="left")
    has_pos = img["latitude"].notna().to_numpy()
    el = np.full(len(img), np.nan)
    az = np.full(len(img), np.nan)
    if has_pos.any():
        e, a = solar_position(
            img.loc[has_pos, "time_utc"],
            img.loc[has_pos, "latitude"].to_numpy(),
            img.loc[has_pos, "longitude"].to_numpy(),
        )
        el[has_pos], az[has_pos] = e, a
    img["sun_elevation_deg"] = el.round(2)
    img["sun_azimuth_deg"] = az.round(2)
    img["illumination"] = np.where(has_pos, illumination_category(el), "unknown")

    # --- Wave records -------------------------------------------------------------
    recs = []
    files_used: list[Path] = []
    for st in sorted(img["station_id"].unique()):
        paths = sorted((args.ndbc / st / "spec").glob(f"{st}.spec.*Z.gz"))
        files_used += paths
        recs.append(spec_records(paths, st))
    rec = pd.concat([r for r in recs if len(r)], ignore_index=True)
    rec = rec[rec["Hs_m"].notna()].reset_index(drop=True)
    print(
        f"{len(rec):,} wave records with WVHT at {rec['station_id'].nunique()} stations "
        f"from {len(files_used)} snapshot files",
        flush=True,
    )

    # --- Pair, station by station (a record only labels its own buoy's strips) ----
    cands = []
    for st, g in img.groupby("station_id"):
        r = rec[rec["station_id"] == st]
        if r.empty:
            continue
        pw = pair_window(g, r, window=INTERP_MAX_GAP)
        cands.append(
            pd.concat(
                [
                    g.loc[pw["target_index"], ["sample_id"]].reset_index(drop=True),
                    pw.drop(columns=["target_index", "reference_index"]),
                    r.loc[pw["reference_index"], ["Hs_m", *CONTEXT]].reset_index(drop=True),
                ],
                axis=1,
            )
        )
    cand = pd.concat(cands, ignore_index=True)
    img = img.set_index("sample_id")

    def label(rule: str) -> pd.DataFrame:
        return label_from_candidates(
            cand,
            "Hs_m",
            rule=rule,  # type: ignore[arg-type]
            target="sample_id",
            max_offset=LABEL_WINDOW,
            max_gap=INTERP_MAX_GAP,
        ).reindex(img.index)

    win = label("window_mean")
    img["Hs_label_m"] = win["label"]
    img["Hs_label_spread_m"] = win["spread"]
    img["n_label_records"] = win["n_records"].fillna(0).astype(int)
    near = label("nearest")
    img["Hs_nearest_m"] = near["label"]
    img["nearest_offset_s"] = near["max_abs_offset_s"]
    img["Hs_interp_m"] = label("interpolate")["label"]
    within = cand[cand["offset_s"].abs() <= pd.Timedelta(LABEL_WINDOW).total_seconds()]
    first = within.sort_values(["sample_id", "rank"]).groupby("sample_id").head(1)
    first = first.set_index("sample_id")
    for c in CONTEXT:
        img[c] = first[c].reindex(img.index)
    img["nearest_record_offset_s"] = first["offset_s"].reindex(img.index)
    img = img.reset_index()

    # --- Exclusions (a row keeps the first reason that applies) ------------------
    img["exclusion"] = np.select(
        [
            img["size_bytes"].eq(0),
            ~img["readable"].fillna(False).astype(bool),
            img["latitude"].isna(),
            img["Hs_label_m"].isna(),
        ],
        ["empty_file", "unreadable", "no_position", "no_wave_record_within_30min"],
        default="",
    )
    img["exclusion"] = img["exclusion"].replace("", pd.NA).astype("string")
    img["path"] = [portable_path(p) for p in img["path"]]
    img = img.rename(columns={"path": "local_path"}).sort_values(["station_id", "time_utc"])

    out = args.out
    out.mkdir(parents=True, exist_ok=True)
    img.to_parquet(out / "samples.parquet", index=False)
    p = img[img["exclusion"].isna()]
    summary = {
        "strips": len(img),
        "stations": int(img["station_id"].nunique()),
        "first": str(img["time_utc"].min()),
        "last": str(img["time_utc"].max()),
        "exclusion": img["exclusion"].fillna("paired").value_counts().to_dict(),
        "paired_by_illumination": p["illumination"].value_counts().to_dict(),
        "geometry": img.groupby(["width", "height"])
        .size()
        .rename("n")
        .reset_index()
        .to_dict("records"),
        "stations_without_records": sorted(set(img["station_id"]) - set(rec["station_id"])),
        "stations_moved_over_5km": pos.loc[pos["moved_km"] > 5, ["station_id", "moved_km"]]
        .round(1)
        .to_dict("records"),
        "n_label_records": p["n_label_records"].value_counts().sort_index().to_dict(),
        "Hs_label_m": quantiles(p["Hs_label_m"]),
        "label_spread_m": quantiles(p["Hs_label_spread_m"]),
        "nearest_offset_min": quantiles(p["nearest_offset_s"] / 60),
        "label_minus_nearest_m": quantiles(p["Hs_label_m"] - p["Hs_nearest_m"]),
        "records_revised_between_snapshots": int(rec["snapshots_disagree"].sum()),
        "records": len(rec),
    }
    with atomic_open(out / "summary.json", "w") as fh:
        json.dump(summary, fh, indent=1, default=str)
    write_provenance(
        build_provenance(
            artifact_id="ndbc_buoycam_pairs",
            inputs=files_used,
            outputs=[out / "samples.parquet", out / "summary.json"],
            config={
                "label_window": LABEL_WINDOW,
                "interp_max_gap": INTERP_MAX_GAP,
                "label_rule": "window_mean",
                "archive": str(args.archive),
            },
            notes=f"{len(img):,} strips; images are listed by path and sha256, not copied",
        ),
        out / "provenance.yaml",
    )
    print(json.dumps(summary, indent=1, default=str))
    return 0


if __name__ == "__main__":
    sys.exit(main())
