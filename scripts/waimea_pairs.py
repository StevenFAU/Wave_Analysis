#!/usr/bin/env python
"""Pair every Waimea Bay image with the CDIP 106 records around its capture time.

Run after ``scripts/waimea_extract.py`` and ``scripts/waimea_timing_audit.py``.
For each image with a trusted capture time (``time_source == "caption"``)
every CDIP 106 record whose sample centre lies within +-3 h is kept with its
offset, so the rule that picks or combines records into a label can be
changed and its effect measured. The label of ADR 0010 (mean of the records
centred within +-30 min, with their spread) is added to the samples; the
nearest and interpolated values stay beside it for sensitivity analysis.

Outputs in ``data/processed/pairs/waimea_cdip106/``:

``samples.parquet``
    One row per indexed image (all 17,854), including the images PacIOOS no
    longer serves and those without a trusted time, with the reason in
    ``exclusion`` (``image_missing``, ``time_uncertain``,
    ``no_buoy_within_30min``; empty when paired). Also sun position, the offset
    to the other camera's image of the same hour, and label diagnostics:
    ``Hs_nearest_m`` (record with the nearest centre, if within 30 min),
    ``Hs_interp_m`` (linear in time between the two records around the image,
    if they are consecutive) and ``Hs_range_1h_m`` (spread of the records
    centred within 1 h: how much the label depends on timing). The label
    itself: ``Hs_label_m``, ``Hs_label_spread_m`` and ``n_label_records``.
    Peak period and direction (``Tp_label_s``, ``Dp_label_deg``) come from the
    nearest record: a peak can jump between swell systems, and a mean of two
    peaks may match neither. ``Tp_window_spread_s`` and
    ``Dp_window_spread_deg`` give their range over the label window, and
    ``peak_records_disagree`` marks images where that range is large.
``candidates.parquet``
    One row per (image, record) within the window: offsets, whether the image
    time falls inside the 1600-s sample, rank by distance in time, and the
    record's CDIP values and flags.
``summary.json`` and ``provenance.yaml``
    Counts and checks; copies are kept in ``data/manifests/processed/``.

Usage::

    uv run python scripts/waimea_pairs.py
"""

from __future__ import annotations

import json
import shutil
import sys

import numpy as np
import pandas as pd

from wave_analysis.config import data_dir
from wave_analysis.datasets.provenance import build_provenance, write_provenance
from wave_analysis.physics.dispersion import group_velocity
from wave_analysis.processing.geospatial import haversine_m
from wave_analysis.processing.solar import illumination_category, solar_position
from wave_analysis.processing.synchronize import label_from_candidates, pair_window
from wave_analysis.sources.pacioos import HST_TO_UTC_HOURS

NAME = "waimea_cdip106"
LAT, LON = 21.643578, -158.06339  # beachcam_003/004 (ERDDAP index; identical for both)
CONFIG = {
    "reference": "CDIP 106 (NDBC 51201), historic file, 2009-2013",
    "window": "3h",
    "nearest_max": "30min",
    "interp_max_gap": "40min",
    "range_window": "1h",
    "caption_mask_rows": [1232, 1254],
    "label_rule": "window_mean",  # ADR 0010
    "label_window": "30min",
    "peak_label_rule": "nearest",  # Tp and Dp (ADR 0010)
    "peak_disagree_tp_s": 3.0,  # >= 2 CDIP frequency bands at 9-17 s
    "peak_disagree_dp_deg": 60.0,
}
#: CDIP variables as columns: (variable, method) -> column.
RECORD_COLUMNS = {
    ("wave_significant_height", "provider_reported"): "Hs_m",
    ("wave_peak_period", "provider_reported"): "Tp_s",
    ("wave_mean_period_tm01", "provider_reported"): "Tm01_s",
    ("wave_mean_period_tm02", "provider_reported"): "Tm02_s",
    ("wave_from_direction_at_peak", "provider_reported"): "Dp_deg",
    ("wave_significant_height", "spectral_hm0"): "Hm0_spectrum_m",
    ("wave_mean_from_direction", "spectral_dm_kuik1988"): "Dm_deg",
}


def cdip_records() -> pd.DataFrame:
    """One row per CDIP 106 sample: window, values, flags, deployment, position."""
    d = data_dir("processed") / "cdip" / "106"
    parts = []
    for name in ("bulk", "spectral_bulk"):
        p = pd.read_parquet(d / f"{name}_2009-01-01_2014-01-01.parquet")
        p = p[[k in RECORD_COLUMNS for k in zip(p["variable"], p["method"], strict=True)]]
        parts.append(p)
    long = pd.concat(parts, ignore_index=True)
    long["column"] = [RECORD_COLUMNS[k] for k in zip(long["variable"], long["method"], strict=True)]
    if long.duplicated(["time_utc", "column"]).any():
        raise RuntimeError("CDIP 106 has more than one value per record and variable")
    wide = long.pivot(index="time_utc", columns="column", values="value")
    meta = (
        long[long["method"] == "provider_reported"]
        .groupby("time_utc")
        .agg(
            averaging_period_s=("averaging_period_s", "first"),
            deployment_id=("deployment_id", "first"),
            buoy_latitude=("latitude", "first"),
            buoy_longitude=("longitude", "first"),
            source_qc_flag=("source_qc_flag", "first"),
            n_flags=("source_qc_flag", "nunique"),
        )
    )
    if (meta.pop("n_flags") > 1).any():
        raise RuntimeError("CDIP flags differ between variables of one record")
    rec = meta.join(wide[list(dict.fromkeys(RECORD_COLUMNS.values()))]).reset_index()
    rec = rec.rename(columns={"time_utc": "start_utc"})
    rec["end_utc"] = rec["start_utc"] + pd.to_timedelta(rec["averaging_period_s"], unit="s")
    rec["reference_distance_m"] = haversine_m(
        LAT, LON, rec["buoy_latitude"].to_numpy(), rec["buoy_longitude"].to_numpy()
    )
    return rec


def image_table() -> pd.DataFrame:
    """All indexed images with their audit results (capture time, darkness)."""
    d = data_dir("interim") / "waimea_audit"
    idx = pd.read_parquet(d / "images.parquet")
    times = pd.read_parquet(d / "image_times.parquet")
    keep = ["name", "file_hst", "caption_hst", "time_source", "time_utc", "ocr_margin", "dark"]
    t = idx[["camera", "name", "url", "time_utc", "ledger_status", "sha256", "size_bytes"]]
    t = t.rename(columns={"time_utc": "index_time_utc"})
    t = t.merge(times[keep], on="name", how="left", validate="one_to_one")
    t["time_source"] = t["time_source"].fillna("none")
    if t["name"].duplicated().any():
        raise RuntimeError("image names are not unique")
    named = pd.to_datetime(t["name"].str.extract(r"\.(\d{8}T\d{6})")[0], format="%Y%m%dT%H%M%S")
    if (
        named + pd.Timedelta(hours=HST_TO_UTC_HOURS) != t["index_time_utc"].dt.tz_convert(None)
    ).any():
        raise RuntimeError("an index time is not the file-name time + 10 h")
    t["file_hst"] = named
    t.insert(0, "sample_id", t["name"].str.removesuffix(".jpg"))
    return t


def main() -> int:
    out = data_dir("processed") / "pairs" / NAME
    out.mkdir(parents=True, exist_ok=True)
    rec = cdip_records()
    img = image_table()
    s: dict[str, object] = {"config": CONFIG}

    # --- CDIP 106 checks -------------------------------------------------------
    gap = rec["start_utc"].diff().dt.total_seconds()
    # Consecutive records (30 min apart) differ by sampling noise plus real change;
    # if Hs were constant, median |d| = 0.6745 * sqrt(2) * sigma.
    hs_v = rec["Hs_m"].to_numpy()
    cons = ((gap > 1700) & (gap < 1900)).to_numpy()[1:]
    rel_d = (np.abs(np.diff(hs_v)) / ((hs_v[1:] + hs_v[:-1]) / 2))[cons]
    q = [0.5, 0.9, 0.99]
    s["cdip"] = {
        "records": len(rec),
        "sample_seconds": sorted(rec["averaging_period_s"].unique().tolist()),
        "record_spacing_s_top": {
            str(int(k)): int(v) for k, v in gap.value_counts().head(3).items()
        },
        "gaps_over_1h": int((gap > 3600).sum()),
        "abs_Hs_minus_Hm0_m_quantiles": {
            str(k): round(float(v), 4)
            for k, v in (rec["Hs_m"] - rec["Hm0_spectrum_m"]).abs().quantile([*q, 1]).items()
        },
        "consecutive_relative_dHs_quantiles": {
            str(k): round(float(np.quantile(rel_d, k)), 3) for k in q
        },
        "record_noise_relative_sigma_at_most": round(
            float(np.median(rel_d) / (0.6745 * np.sqrt(2))), 3
        ),
        "source_qc_flag": rec["source_qc_flag"].value_counts().to_dict(),
        "deployments": {
            dep: {
                "first": str(g["start_utc"].min()),
                "last": str(g["start_utc"].max()),
                "latitude": round(float(g["buoy_latitude"].iloc[0]), 6),
                "longitude": round(float(g["buoy_longitude"].iloc[0]), 6),
                "distance_to_camera_m": round(float(g["reference_distance_m"].iloc[0])),
            }
            for dep, g in rec.groupby("deployment_id")
        },
    }

    # --- Pair ------------------------------------------------------------------
    trusted = img[img["time_source"] == "caption"]
    pw = pair_window(trusted, rec, window=CONFIG["window"])
    cand = pd.concat(
        [
            img.loc[pw["target_index"], ["sample_id", "camera", "time_utc"]].reset_index(drop=True),
            pw.drop(columns=["target_index", "reference_index"]),
            rec.loc[pw["reference_index"]].reset_index(drop=True),
        ],
        axis=1,
    )
    lag = cand["reference_distance_m"] / group_velocity(1 / cand["Tp_s"].to_numpy())
    cand["propagation_lag_deep_s"] = lag.round(0)

    near_max = pd.Timedelta(CONFIG["nearest_max"]).total_seconds()
    nearest = cand[cand["rank"] == 0].set_index("sample_id")
    img = img.set_index("sample_id")
    img["nearest_offset_s"] = nearest["offset_s"]
    img["nearest_in_record"] = nearest["in_record"]
    img["deployment_id"] = nearest["deployment_id"]
    img["reference_distance_m"] = nearest["reference_distance_m"].round(0)
    ok_near = img["nearest_offset_s"].abs() <= near_max
    img["Hs_nearest_m"] = nearest["Hs_m"].where(ok_near)
    img["Tp_nearest_s"] = nearest["Tp_s"].where(ok_near)
    img["Dp_nearest_deg"] = nearest["Dp_deg"].where(ok_near)
    img["source_qc_flag"] = nearest["source_qc_flag"].where(ok_near)
    img["n_records_30min"] = cand[cand["offset_s"].abs() <= near_max].groupby("sample_id").size()
    img["n_records_30min"] = img["n_records_30min"].fillna(0).astype(int)

    # Linear interpolation between the last record centred at or before the image
    # and the first after, only when they are consecutive samples.
    max_gap = pd.Timedelta(CONFIG["interp_max_gap"]).total_seconds()
    before = cand[cand["offset_s"] >= 0].sort_values("offset_s").groupby("sample_id").first()
    after = cand[cand["offset_s"] < 0].sort_values("offset_s").groupby("sample_id").last()
    both = before[["offset_s", "Hs_m"]].join(
        after[["offset_s", "Hs_m"]], lsuffix="_b", rsuffix="_a", how="inner"
    )
    span = both["offset_s_b"] - both["offset_s_a"]
    w = both["offset_s_b"] / span
    interp = both["Hs_m_b"] + w * (both["Hs_m_a"] - both["Hs_m_b"])
    img["Hs_interp_m"] = interp.where(span <= max_gap)

    rng = pd.Timedelta(CONFIG["range_window"]).total_seconds()
    near_h = cand[cand["offset_s"].abs() <= rng].groupby("sample_id")["Hs_m"]
    img["Hs_range_1h_m"] = (near_h.max() - near_h.min()).where(near_h.size() >= 2)

    # The label (ADR 0010): mean of the records centred within the label window.
    def label(column: str, rule: str, circular: bool = False) -> pd.DataFrame:
        return label_from_candidates(
            cand,
            column,
            rule=rule,
            target="sample_id",
            max_offset=CONFIG["label_window"],
            circular=circular,
        )

    hs_label = label("Hs_m", CONFIG["label_rule"])
    img["Hs_label_m"] = hs_label["label"]
    img["Hs_label_spread_m"] = hs_label["spread"]
    img["n_label_records"] = hs_label["n_records"]
    img["n_label_records"] = img["n_label_records"].fillna(0).astype(int)
    # Peaks from one record; their range over the window shows a change of system.
    img["Tp_label_s"] = label("Tp_s", CONFIG["peak_label_rule"])["label"]
    img["Dp_label_deg"] = label("Dp_deg", CONFIG["peak_label_rule"], circular=True)["label"]
    img["Tp_window_spread_s"] = label("Tp_s", "window_mean")["spread"]
    img["Dp_window_spread_deg"] = label("Dp_deg", "window_mean", circular=True)["spread"]
    tp_jump = img["Tp_window_spread_s"] >= CONFIG["peak_disagree_tp_s"]
    dp_jump = img["Dp_window_spread_deg"] >= CONFIG["peak_disagree_dp_deg"]
    img["peak_records_disagree"] = tp_jump | dp_jump
    img = img.reset_index()

    el, az = solar_position(img["time_utc"], LAT, LON)
    has_t = img["time_utc"].notna().to_numpy()
    img["sun_elevation_deg"] = np.where(has_t, el, np.nan).round(2)
    img["sun_azimuth_deg"] = np.where(has_t, az, np.nan).round(2)
    img["illumination"] = np.where(has_t, illumination_category(el), None)

    # The other camera's image of the same named hour: how far apart they were taken.
    other = img[img["time_source"] == "caption"].pivot(
        index="file_hst", columns="camera", values="time_utc"
    )
    diff = (other["beachcam_003"] - other["beachcam_004"]).dt.total_seconds()
    sign = np.where(img["camera"] == "beachcam_003", 1.0, -1.0)
    img["other_camera_offset_s"] = img["file_hst"].map(diff).to_numpy() * sign

    img["exclusion"] = np.select(
        [
            img["time_source"] == "none",
            img["time_source"] == "uncertain",
            img["Hs_nearest_m"].isna(),
        ],
        ["image_missing", "time_uncertain", "no_buoy_within_30min"],
        default="",
    )
    img["exclusion"] = img["exclusion"].replace("", pd.NA).astype("string")

    # --- Checks on the pairs ---------------------------------------------------
    p = img[img["exclusion"].isna()]
    s["images"] = {
        "indexed": len(img),
        "exclusion": img["exclusion"].fillna("paired").value_counts().to_dict(),
        "paired_by_camera": p["camera"].value_counts().sort_index().to_dict(),
        "paired_by_year": p["time_utc"].dt.year.value_counts().sort_index().to_dict(),
        "paired_days": int(p["time_utc"].dt.tz_convert("Pacific/Honolulu").dt.date.nunique()),
    }
    off = p["nearest_offset_s"].abs() / 60
    s["timing"] = {
        "nearest_centre_offset_min_quantiles": {
            str(k): round(float(off.quantile(k)), 1) for k in q
        },
        "nearest_centre_offset_min_max": round(float(off.max()), 1),
        "image_inside_nearest_sample_fraction": round(float(p["nearest_in_record"].mean()), 3),
        "n_records_30min": p["n_records_30min"].value_counts().sort_index().to_dict(),
        "propagation_lag_deep_min_quantiles": {
            str(k): round(float(v / 60), 1)
            for k, v in cand.loc[cand["rank"] == 0, "propagation_lag_deep_s"].quantile(q).items()
        },
        "other_camera_offset_s_abs_quantiles": {
            str(k): round(float(v), 0)
            for k, v in img["other_camera_offset_s"].abs().dropna().quantile(q).items()
        },
        "other_camera_offset_over_2min_hours": int((diff.abs() > 120).sum()),
        "hours_with_both_cameras_trusted": int(diff.notna().sum()),
    }
    rel = p["Hs_range_1h_m"] / p["Hs_nearest_m"]
    dn = (p["Hs_nearest_m"] - p["Hs_interp_m"]).abs()
    s["label_sensitivity"] = {
        "Hs_range_1h_m_quantiles": {
            str(k): round(float(p["Hs_range_1h_m"].quantile(k)), 3) for k in q
        },
        "Hs_range_1h_relative_quantiles": {str(k): round(float(rel.quantile(k)), 3) for k in q},
        "abs_nearest_minus_interp_m_quantiles": {
            str(k): round(float(dn.quantile(k)), 3) for k in q
        },
        "interp_available_fraction": round(float(p["Hs_interp_m"].notna().mean()), 3),
    }
    dl = (p["Hs_label_m"] - p["Hs_nearest_m"]).abs()
    s["label"] = {
        "rule": CONFIG["label_rule"],
        "window": CONFIG["label_window"],
        "n_label_records": p["n_label_records"].value_counts().sort_index().to_dict(),
        "abs_label_minus_nearest_m_quantiles": {str(k): round(float(dl.quantile(k)), 3) for k in q},
        "label_spread_relative_quantiles": {
            str(k): round(float((p["Hs_label_spread_m"] / p["Hs_label_m"]).quantile(k)), 3)
            for k in q
        },
        "peak_rule": CONFIG["peak_label_rule"],
        "peak_disagree_by_Tp": int((p["Tp_window_spread_s"] >= CONFIG["peak_disagree_tp_s"]).sum()),
        "peak_disagree_by_Dp": int(
            (p["Dp_window_spread_deg"] >= CONFIG["peak_disagree_dp_deg"]).sum()
        ),
        "peak_records_disagree": int(p["peak_records_disagree"].sum()),
    }
    hs = p["Hs_nearest_m"]
    s["labels"] = {
        "Hs_m_quantiles": {str(k): round(float(hs.quantile(k)), 2) for k in [0.01, 0.5, 0.99]},
        "Hs_m_max": round(float(hs.max()), 2),
        "Hs_m_counts": {
            "<1": int((hs < 1).sum()),
            "1-2": int(((hs >= 1) & (hs < 2)).sum()),
            "2-3": int(((hs >= 2) & (hs < 3)).sum()),
            "3-4": int(((hs >= 3) & (hs < 4)).sum()),
            ">=4": int((hs >= 4).sum()),
        },
        "days_with_Hs_ge_4m": int(p.loc[hs >= 4, "file_hst"].dt.date.nunique()),
        "source_qc_flag": p["source_qc_flag"].value_counts().to_dict(),
        "illumination": p["illumination"].value_counts().to_dict(),
        "dark": int(p["dark"].sum()),
    }

    # --- Write -----------------------------------------------------------------
    cols = [
        "sample_id", "camera", "name", "url", "sha256", "size_bytes", "index_time_utc",
        "file_hst", "caption_hst", "time_source", "time_utc", "ocr_margin",
        "other_camera_offset_s", "sun_elevation_deg", "sun_azimuth_deg", "illumination",
        "dark", "deployment_id", "reference_distance_m", "nearest_offset_s",
        "nearest_in_record", "n_records_30min", "Hs_nearest_m", "Hs_interp_m",
        "Hs_range_1h_m", "Tp_nearest_s", "Dp_nearest_deg", "Hs_label_m", "Hs_label_spread_m",
        "n_label_records", "Tp_label_s", "Dp_label_deg", "Tp_window_spread_s",
        "Dp_window_spread_deg", "peak_records_disagree", "source_qc_flag", "exclusion",
    ]  # fmt: skip
    paths = {
        "samples": out / "samples.parquet",
        "candidates": out / "candidates.parquet",
        "summary": out / "summary.json",
    }
    img[cols].to_parquet(paths["samples"], index=False)
    cand.drop(columns=["buoy_latitude", "buoy_longitude"]).to_parquet(
        paths["candidates"], index=False
    )
    paths["summary"].write_text(json.dumps(s, indent=1, default=str) + "\n")
    d_in = data_dir("interim") / "waimea_audit"
    d_cdip = data_dir("processed") / "cdip" / "106"
    prov = build_provenance(
        artifact_id=NAME,
        inputs=[
            d_in / "images.parquet",
            d_in / "image_times.parquet",
            d_cdip / "bulk_2009-01-01_2014-01-01.parquet",
            d_cdip / "spectral_bulk_2009-01-01_2014-01-01.parquet",
        ],
        outputs=list(paths.values()),
        command="python scripts/waimea_pairs.py",
        config=CONFIG,
        notes=(
            "Labels by ADR 0010 (Hs window mean; Tp, Dp nearest record); "
            "see docs/datasets/pacioos_beachcam.md."
        ),
    )
    write_provenance(prov, out / "provenance.yaml")
    keep = data_dir("manifests") / "processed"
    shutil.copyfile(out / "provenance.yaml", keep / f"{NAME}.provenance.yaml")
    shutil.copyfile(paths["summary"], keep / f"{NAME}.summary.json")
    print(json.dumps(s, indent=1, default=str))
    return 0


if __name__ == "__main__":
    sys.exit(main())
