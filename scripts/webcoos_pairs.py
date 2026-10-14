#!/usr/bin/env python
"""Pair a WebCOOS camera's stills with the CDIP records around their capture times.

Run after ``scripts/webcoos_extract.py <camera>`` and after the reference
buoy's records have been standardized (``wave-analysis cdip standardize``, the
historic file and, for the current deployment, ``--product realtime``). The
method is that of the Waimea Bay pairing (``scripts/waimea_pairs.py``, ADR
0010): for each still every record whose sample centre lies within +-3 h is
kept with its offset, so the rule that picks or combines records into a label
can be changed and its effect measured. The label of ADR 0010 (mean of the
records centred within +-30 min, with their spread) is added to the samples;
the nearest and interpolated values stay beside it.

What differs from Waimea Bay:

* The capture time is the one in the file name (UTC, to the second): the
  stills have no time stamp in the picture and no EXIF. It is checked against
  the sun: the cameras switch to a monochrome mode in the dark.
* The reference buoy is not offshore of the camera but along the coast from
  it, so the travel time between them depends on the wave direction. Besides
  the Waimea estimate (distance over the deep-water group velocity, an upper
  bound) each record carries ``propagation_lag_directional_s``: the distance
  projected on the direction the peak waves travel in, over the group velocity
  at the buoy's depth. It is positive when the waves pass the buoy first. No
  lag is applied to the label (ADR 0010); ``Hs_label_lagged_m`` shows what
  applying it would change.
* The current deployment comes from CDIP's realtime file, which also holds
  records flagged bad. Only records with primary flag 1 (good) are used.

No image is dropped: ``samples.parquet`` has one row per still on disk or in a
ledger, night frames included, and ``exclusion`` gives the reason a row has no
label (``image_missing``, ``image_unreadable``, ``checksum_mismatch``,
``time_contradicts_light``, ``no_buoy_within_30min``; empty when paired).

Outputs in ``data/processed/pairs/webcoos_<camera>_cdip<station>/``:
``samples.parquet``, ``candidates.parquet`` (one row per still and record
within the window), ``summary.json`` and ``provenance.yaml``; copies of the
last two are kept in ``data/manifests/processed/``.

Usage::

    uv run python scripts/webcoos_pairs.py currituck_hampton_inn
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from wave_analysis.config import data_dir, repo_root
from wave_analysis.dashboard.catalog import load_camera_sites
from wave_analysis.datasets.provenance import build_provenance, write_provenance
from wave_analysis.physics.dispersion import group_velocity
from wave_analysis.processing.geospatial import haversine_m, initial_bearing_deg
from wave_analysis.processing.solar import illumination_category, solar_position
from wave_analysis.processing.synchronize import label_from_candidates, pair_window
from wave_analysis.sources.webcoos import ONPREM_POSITIONS

#: Camera -> its site in ``data/registry/camera_sites.yaml``, the CDIP station
#: used as reference (the site's primary reference there), the water depth at
#: the buoy (``metaWaterDepth`` of the CDIP files) and the local time zone
#: (for counting days).
PAIRINGS: dict[str, dict[str, object]] = {
    "currituck_hampton_inn": {
        "site_id": "webcoos_currituck",
        "cdip": "433",
        "reference_depth_m": 18.0,
        "timezone": "America/New_York",
    },
}
CONFIG: dict[str, object] = {
    "window": "3h",
    "nearest_max": "30min",
    "interp_max_gap": "40min",
    "range_window": "1h",
    "label_rule": "window_mean",  # ADR 0010
    "label_window": "30min",
    "peak_label_rule": "nearest",  # Tp and Dp (ADR 0010)
    "peak_disagree_tp_s": 3.0,
    "peak_disagree_dp_deg": 60.0,
    "good_primary_flag": "1",
    "monochrome_max_saturation": 0.01,
    # A frame whose light contradicts its time: night mode with the sun well up,
    # or colour deep in the night (the camera switches between -6 and -9 deg).
    "monochrome_max_sun_deg": -3.0,
    "colour_min_sun_deg": -12.0,
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
Q = [0.5, 0.9, 0.99]


def quantiles(values: pd.Series, q: list[float] = Q, digits: int = 3) -> dict[str, float]:
    """Quantiles of the values present, as a JSON-friendly mapping."""
    v = values.dropna()
    return {str(k): round(float(v.quantile(k)), digits) for k in q} if len(v) else {}


def cdip_records(station: str, lat: float, lon: float) -> tuple[pd.DataFrame, list[Path]]:
    """One row per CDIP sample of ``station``, from every standardized file of it.

    Returns the records (window, values, flags, deployment, position, distance
    and bearing to the camera) and the files read. A sample found in two files
    (a realtime snapshot and the historic file that later absorbed it) is taken
    from the historic file.
    """
    d = data_dir("processed") / "cdip" / station
    files = sorted(d.glob("bulk_*.parquet")) + sorted(d.glob("spectral_bulk_*.parquet"))
    if not files:
        raise FileNotFoundError(
            f"no standardized records in {d}; run 'wave-analysis cdip standardize'"
        )
    parts = []
    for path in files:
        p = pd.read_parquet(path)
        p = p[[k in RECORD_COLUMNS for k in zip(p["variable"], p["method"], strict=True)]]
        p = p.assign(
            column=[RECORD_COLUMNS[k] for k in zip(p["variable"], p["method"], strict=True)],
            realtime="_realtime_" in path.name,
        )
        if p.duplicated(["time_utc", "column"]).any():
            raise RuntimeError(f"{path.name} has more than one value per record and variable")
        parts.append(p)
    long = pd.concat(parts, ignore_index=True)
    # Archived records first, so that they win over a realtime copy.
    long = long.sort_values(["time_utc", "column", "realtime"], kind="stable")
    long = long.drop_duplicates(["time_utc", "column"], keep="first")
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
            source_file=("source_file", "first"),
        )
    )
    if (meta.pop("n_flags") > 1).any():
        raise RuntimeError("CDIP flags differ between variables of one record")
    rec = meta.join(wide[list(dict.fromkeys(RECORD_COLUMNS.values()))]).reset_index()
    rec = rec.rename(columns={"time_utc": "start_utc"})
    rec["end_utc"] = rec["start_utc"] + pd.to_timedelta(rec["averaging_period_s"], unit="s")
    b_lat, b_lon = rec["buoy_latitude"].to_numpy(), rec["buoy_longitude"].to_numpy()
    rec["reference_distance_m"] = haversine_m(lat, lon, b_lat, b_lon)
    rec["bearing_buoy_to_camera_deg"] = initial_bearing_deg(b_lat, b_lon, lat, lon)
    return rec, files


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("camera", choices=sorted(PAIRINGS))
    args = ap.parse_args(argv)
    cam = args.camera
    pairing = PAIRINGS[cam]
    station = str(pairing["cdip"])
    depth = float(pairing["reference_depth_m"])  # type: ignore[arg-type]
    name = f"webcoos_{cam}_cdip{station}"
    lat, lon = ONPREM_POSITIONS[cam]
    config = {"camera": cam, **pairing, "camera_latitude": lat, "camera_longitude": lon, **CONFIG}

    # The reference is chosen in the site registry; this script must not drift from it.
    sites = {
        s.site_id: s for s in load_camera_sites(repo_root() / "data/registry/camera_sites.yaml")
    }
    primary = [r for r in sites[str(pairing["site_id"])].references if r.role == "primary"]
    if [r.cdip_id for r in primary] != [station]:
        raise RuntimeError(
            f"{pairing['site_id']}: the registry's primary reference is not CDIP {station}"
        )

    out = data_dir("processed") / "pairs" / name
    out.mkdir(parents=True, exist_ok=True)
    audit = data_dir("interim") / "webcoos_audit" / cam / "images.parquet"
    img = pd.read_parquet(audit)
    if img["name"].duplicated().any():
        raise RuntimeError("image names are not unique")
    img.insert(0, "sample_id", img["name"].str.removesuffix(".jpg"))
    rec_all, record_files = cdip_records(station, lat, lon)
    s: dict[str, object] = {"config": config}

    # --- CDIP checks -----------------------------------------------------------
    first_image = img["time_utc"].min() - pd.Timedelta(CONFIG["window"])
    rec_all = rec_all[rec_all["start_utc"] >= first_image].reset_index(drop=True)
    good = rec_all["source_qc_flag"].str.split("/").str[0] == CONFIG["good_primary_flag"]
    rec = rec_all[good].reset_index(drop=True)
    gap = rec["start_utc"].diff().dt.total_seconds()
    # Consecutive records (30 min apart) differ by sampling noise plus real change;
    # if Hs were constant, median |d| = 0.6745 * sqrt(2) * sigma.
    hs_v = rec["Hs_m"].to_numpy()
    cons = ((gap > 1700) & (gap < 1900)).to_numpy()[1:]
    rel_d = (np.abs(np.diff(hs_v)) / ((hs_v[1:] + hs_v[:-1]) / 2))[cons]
    long_gaps = rec.assign(prev=rec["start_utc"].shift(), gap_s=gap)[gap > 3 * 86400]
    s["cdip"] = {
        "station": station,
        "files": [p.name for p in record_files],
        "records_from": str(first_image),
        "records": len(rec_all),
        "records_not_good_left_out": int((~good).sum()),
        "source_qc_flag": rec_all["source_qc_flag"].value_counts().to_dict(),
        "sample_seconds": {
            str(int(k)): int(v) for k, v in rec["averaging_period_s"].value_counts().items()
        },
        "record_spacing_s_top": {
            str(int(k)): int(v) for k, v in gap.value_counts().head(3).items()
        },
        "gaps_over_1h": int((gap > 3600).sum()),
        "gaps_over_3d": [
            {"from": str(r.prev), "to": str(r.start_utc), "days": round(r.gap_s / 86400, 1)}
            for r in long_gaps.itertuples()
        ],
        "abs_Hs_minus_Hm0_m_quantiles": quantiles(
            (rec["Hs_m"] - rec["Hm0_spectrum_m"]).abs(), [*Q, 1], 4
        ),
        "consecutive_relative_dHs_quantiles": {
            str(k): round(float(np.quantile(rel_d, k)), 3) for k in Q
        },
        "record_noise_relative_sigma_at_most": round(
            float(np.median(rel_d) / (0.6745 * np.sqrt(2))), 3
        ),
        "deployments": {
            dep: {
                "first": str(g["start_utc"].min()),
                "last": str(g["start_utc"].max()),
                "records": len(g),
                "sample_s": float(g["averaging_period_s"].iloc[0]),
                "latitude": round(float(g["buoy_latitude"].iloc[0]), 6),
                "longitude": round(float(g["buoy_longitude"].iloc[0]), 6),
                "distance_to_camera_m": round(float(g["reference_distance_m"].iloc[0])),
                "bearing_to_camera_deg": round(float(g["bearing_buoy_to_camera_deg"].iloc[0]), 1),
            }
            for dep, g in rec.groupby("deployment_id")
        },
    }

    # --- Images ----------------------------------------------------------------
    el, az = solar_position(img["time_utc"], lat, lon)
    img["sun_elevation_deg"] = np.round(el, 2)
    img["sun_azimuth_deg"] = np.round(az, 2)
    img["illumination"] = illumination_category(el)
    img["monochrome"] = (img["mean_saturation"] <= CONFIG["monochrome_max_saturation"]).where(
        img["mean_saturation"].notna()
    )
    readable = img["exists"] & img["features_ok"] & img["eoi"].fillna(False).astype(bool)
    img["time_contradicts_light"] = (
        img["monochrome"].eq(True) & (img["sun_elevation_deg"] > CONFIG["monochrome_max_sun_deg"])
    ) | (img["monochrome"].eq(False) & (img["sun_elevation_deg"] < CONFIG["colour_min_sun_deg"]))
    mismatch = img["sha256_matches_ledger"].eq(False) | img["ledger_sha256_differ"]

    # --- Pair ------------------------------------------------------------------
    pw = pair_window(img, rec, window=CONFIG["window"])
    cand = pd.concat(
        [
            img.loc[pw["target_index"], ["sample_id", "camera", "time_utc"]].reset_index(drop=True),
            pw.drop(columns=["target_index", "reference_index"]),
            rec.loc[pw["reference_index"]].reset_index(drop=True),
        ],
        axis=1,
    )
    f_peak = 1 / cand["Tp_s"].to_numpy()
    dist = cand["reference_distance_m"].to_numpy()
    cand["propagation_lag_deep_s"] = (dist / group_velocity(f_peak)).round(0)
    # Waves from Dp travel towards Dp + 180; the part of the buoy-to-camera
    # distance along that direction is what they cover between the two.
    towards = np.deg2rad(cand["Dp_deg"].to_numpy() + 180.0 - cand["bearing_buoy_to_camera_deg"])
    cand["propagation_lag_directional_s"] = (
        dist * np.cos(towards) / group_velocity(f_peak, depth)
    ).round(0)

    near_max = pd.Timedelta(CONFIG["nearest_max"]).total_seconds()
    nearest = cand[cand["rank"] == 0].set_index("sample_id")
    img = img.set_index("sample_id")
    img["nearest_offset_s"] = nearest["offset_s"]
    img["nearest_in_record"] = nearest["in_record"]
    ok_near = img["nearest_offset_s"].abs() <= near_max
    img["deployment_id"] = nearest["deployment_id"].where(ok_near)
    img["reference_distance_m"] = nearest["reference_distance_m"].round(0).where(ok_near)
    img["Hs_nearest_m"] = nearest["Hs_m"].where(ok_near)
    img["Tp_nearest_s"] = nearest["Tp_s"].where(ok_near)
    img["Dp_nearest_deg"] = nearest["Dp_deg"].where(ok_near)
    img["source_qc_flag"] = nearest["source_qc_flag"].where(ok_near)
    img["propagation_lag_directional_s"] = nearest["propagation_lag_directional_s"].where(ok_near)
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
    def label(
        column: str, rule: str, circular: bool = False, lag: str | None = None
    ) -> pd.DataFrame:
        return label_from_candidates(
            cand,
            column,
            rule=rule,  # type: ignore[arg-type]
            target="sample_id",
            max_offset=str(CONFIG["label_window"]),
            circular=circular,
            lag_s=lag,
        )

    hs_label = label("Hs_m", str(CONFIG["label_rule"]))
    img["Hs_label_m"] = hs_label["label"]
    img["Hs_label_spread_m"] = hs_label["spread"]
    img["n_label_records"] = hs_label["n_records"]
    img["n_label_records"] = img["n_label_records"].fillna(0).astype(int)
    # Peaks from one record; their range over the window shows a change of system.
    img["Tp_label_s"] = label("Tp_s", str(CONFIG["peak_label_rule"]))["label"]
    img["Dp_label_deg"] = label("Dp_deg", str(CONFIG["peak_label_rule"]), circular=True)["label"]
    img["Tp_window_spread_s"] = label("Tp_s", "window_mean")["spread"]
    img["Dp_window_spread_deg"] = label("Dp_deg", "window_mean", circular=True)["spread"]
    tp_jump = img["Tp_window_spread_s"] >= CONFIG["peak_disagree_tp_s"]
    dp_jump = img["Dp_window_spread_deg"] >= CONFIG["peak_disagree_dp_deg"]
    img["peak_records_disagree"] = tp_jump | dp_jump
    # Not the label: what it would be if the image were moved by the directional lag.
    lagged = label("Hs_m", str(CONFIG["label_rule"]), lag="propagation_lag_directional_s")
    img["Hs_label_lagged_m"] = lagged["label"]
    img = img.reset_index()

    img["exclusion"] = np.select(
        [
            ~img["exists"],
            ~readable,
            mismatch,
            img["time_contradicts_light"],
            img["Hs_nearest_m"].isna(),
        ],
        [
            "image_missing",
            "image_unreadable",
            "checksum_mismatch",
            "time_contradicts_light",
            "no_buoy_within_30min",
        ],
        default="",
    )
    img["exclusion"] = img["exclusion"].replace("", pd.NA).astype("string")

    # --- Checks on the images and the pairs ------------------------------------
    p = img[img["exclusion"].isna()]
    local_day = p["time_utc"].dt.tz_convert(str(pairing["timezone"])).dt.date
    held = img[img["exists"]]
    s["images"] = {
        "indexed": len(img),
        "on_disk": int(img["exists"].sum()),
        "first": str(held["time_utc"].min()),
        "last": str(held["time_utc"].max()),
        "decoded": int(readable.sum()),
        "sha256_matches_ledger": held["sha256_matches_ledger"]
        .map({True: "yes", False: "no"})
        .fillna("no_ledger_row")
        .value_counts()
        .to_dict(),
        "by_ledger": held["ledgers"].fillna("none").value_counts().to_dict(),
        "frame_sizes": (
            held["width"].astype("Int64").astype(str)
            + "x"
            + held["height"].astype("Int64").astype(str)
        )
        .value_counts()
        .to_dict(),
        "with_exif": int(held["has_exif"].fillna(False).sum()),
        "exclusion": img["exclusion"].fillna("paired").value_counts().to_dict(),
        "paired_by_year": p["time_utc"].dt.year.value_counts().sort_index().to_dict(),
        "paired_days": int(local_day.nunique()),
        "paired_by_illumination": p["illumination"].value_counts().to_dict(),
    }
    twilight = held[
        held["sun_elevation_deg"].between(-14, 2)
        & held["monochrome"].notna()
        & ~held["time_contradicts_light"]
    ]
    # Only days on which the camera recorded through the night show the switch.
    night_days = set(held.loc[held["monochrome"].eq(True), "time_utc"].dt.floor("D"))
    twilight = twilight[twilight["time_utc"].dt.floor("D").isin(night_days)]
    twilight = twilight.assign(side=np.where(twilight["sun_azimuth_deg"] < 180, "dawn", "dusk"))
    mono = held[held["monochrome"].eq(True)]
    colour = held[held["monochrome"].eq(False)]
    s["capture_time_check"] = {
        "monochrome_frames": len(mono),
        "monochrome_by_illumination": mono["illumination"].value_counts().to_dict(),
        "monochrome_sun_elevation_deg_max": round(float(mono["sun_elevation_deg"].max()), 2)
        if len(mono)
        else None,
        "colour_frames": len(colour),
        "colour_by_illumination": colour["illumination"].value_counts().to_dict(),
        "colour_sun_elevation_deg_min": round(float(colour["sun_elevation_deg"].min()), 2)
        if len(colour)
        else None,
        "colour_frames_with_sun_below_minus_6": int((colour["sun_elevation_deg"] < -6).sum()),
        "monochrome_frames_with_sun_above_0": int((mono["sun_elevation_deg"] > 0).sum()),
        "time_contradicts_light": sorted(held.loc[held["time_contradicts_light"], "name"]),
        # The switch to and from night mode, by side of the day: a clock that is
        # off moves the two apart by about 0.4 deg per minute.
        "switch_band_sun_elevation_deg": {
            side: {
                "frames": len(g),
                "highest_monochrome": round(
                    float(g.loc[g["monochrome"].eq(True), "sun_elevation_deg"].max()), 2
                ),
                "lowest_colour": round(
                    float(g.loc[g["monochrome"].eq(False), "sun_elevation_deg"].min()), 2
                ),
            }
            for side, g in twilight.groupby("side")
            if g["monochrome"].eq(True).any() and g["monochrome"].eq(False).any()
        },
    }
    off = p["nearest_offset_s"].abs() / 60
    near0 = cand[(cand["rank"] == 0) & cand["sample_id"].isin(p["sample_id"])]
    lag_dir = near0["propagation_lag_directional_s"] / 60
    s["timing"] = {
        "nearest_centre_offset_min_quantiles": quantiles(off, digits=1),
        "nearest_centre_offset_min_max": round(float(off.max()), 1),
        "image_inside_nearest_sample_fraction": round(float(p["nearest_in_record"].mean()), 3),
        "n_records_30min": p["n_records_30min"].value_counts().sort_index().to_dict(),
        "propagation_lag_deep_min_quantiles": quantiles(
            near0["propagation_lag_deep_s"] / 60, digits=1
        ),
        "propagation_lag_directional_min_quantiles": quantiles(
            lag_dir, [0.01, 0.1, 0.5, 0.9, 0.99], 1
        ),
        "propagation_lag_directional_abs_min_quantiles": quantiles(lag_dir.abs(), digits=1),
        "directional_lag_over_sample_length_fraction": round(
            float(
                (near0["propagation_lag_directional_s"].abs() > near0["averaging_period_s"]).mean()
            ),
            3,
        ),
    }
    rel = p["Hs_range_1h_m"] / p["Hs_nearest_m"]
    s["label_sensitivity"] = {
        "Hs_range_1h_m_quantiles": quantiles(p["Hs_range_1h_m"]),
        "Hs_range_1h_relative_quantiles": quantiles(rel),
        "abs_nearest_minus_interp_m_quantiles": quantiles(
            (p["Hs_nearest_m"] - p["Hs_interp_m"]).abs()
        ),
        "interp_available_fraction": round(float(p["Hs_interp_m"].notna().mean()), 3),
        "abs_label_minus_lagged_label_m_quantiles": quantiles(
            (p["Hs_label_m"] - p["Hs_label_lagged_m"]).abs()
        ),
        "abs_label_minus_lagged_label_relative_quantiles": quantiles(
            (p["Hs_label_m"] - p["Hs_label_lagged_m"]).abs() / p["Hs_label_m"]
        ),
        "lagged_label_available_fraction": round(float(p["Hs_label_lagged_m"].notna().mean()), 3),
    }
    s["label"] = {
        "rule": CONFIG["label_rule"],
        "window": CONFIG["label_window"],
        "n_label_records": p["n_label_records"].value_counts().sort_index().to_dict(),
        "abs_label_minus_nearest_m_quantiles": quantiles(
            (p["Hs_label_m"] - p["Hs_nearest_m"]).abs()
        ),
        "label_spread_relative_quantiles": quantiles(p["Hs_label_spread_m"] / p["Hs_label_m"]),
        "peak_rule": CONFIG["peak_label_rule"],
        "peak_disagree_by_Tp": int((p["Tp_window_spread_s"] >= CONFIG["peak_disagree_tp_s"]).sum()),
        "peak_disagree_by_Dp": int(
            (p["Dp_window_spread_deg"] >= CONFIG["peak_disagree_dp_deg"]).sum()
        ),
        "peak_records_disagree": int(p["peak_records_disagree"].sum()),
    }
    hs = p["Hs_label_m"]
    edges = [0, 0.5, 1, 1.5, 2, 3, 4, np.inf]
    names = ["<0.5", "0.5-1", "1-1.5", "1.5-2", "2-3", "3-4", ">=4"]
    s["labels"] = {
        "Hs_m_quantiles": quantiles(hs, [0.01, 0.5, 0.99], 2),
        "Hs_m_max": round(float(hs.max()), 2),
        "Hs_m_counts": pd.cut(hs, edges, labels=names, right=False)
        .value_counts()
        .sort_index()
        .to_dict(),
        "days_with_Hs_ge_2m": int(local_day[hs >= 2].nunique()),
        "days_with_Hs_ge_3m": int(local_day[hs >= 3].nunique()),
        "Tp_s_quantiles": quantiles(p["Tp_label_s"], [0.01, 0.5, 0.99], 1),
        "by_deployment": p["deployment_id"].value_counts().sort_index().to_dict(),
        "source_qc_flag": p["source_qc_flag"].value_counts().to_dict(),
        "monochrome": int(p["monochrome"].eq(True).sum()),
    }

    # --- Write -----------------------------------------------------------------
    cols = [
        "sample_id", "camera", "name", "local_path", "url", "sha256", "size_bytes", "ledgers",
        "width", "height", "time_utc", "sun_elevation_deg", "sun_azimuth_deg", "illumination",
        "mean_luma", "mean_saturation", "monochrome", "deployment_id", "reference_distance_m",
        "nearest_offset_s", "nearest_in_record", "n_records_30min", "Hs_nearest_m", "Hs_interp_m",
        "Hs_range_1h_m", "Tp_nearest_s", "Dp_nearest_deg", "Hs_label_m", "Hs_label_spread_m",
        "n_label_records", "Tp_label_s", "Dp_label_deg", "Tp_window_spread_s",
        "Dp_window_spread_deg", "peak_records_disagree", "propagation_lag_directional_s",
        "Hs_label_lagged_m", "source_qc_flag", "exclusion",
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
    prov = build_provenance(
        artifact_id=name,
        inputs=[audit, *record_files],
        outputs=list(paths.values()),
        command=f"python scripts/webcoos_pairs.py {cam}",
        config=config,
        notes=(
            "Labels by ADR 0010 (Hs window mean; Tp, Dp nearest record); "
            "see docs/datasets/webcoos.md."
        ),
    )
    write_provenance(prov, out / "provenance.yaml")
    keep = data_dir("manifests") / "processed"
    shutil.copyfile(out / "provenance.yaml", keep / f"{name}.provenance.yaml")
    shutil.copyfile(paths["summary"], keep / f"{name}.summary.json")
    print(json.dumps(s, indent=1, default=str))
    return 0


if __name__ == "__main__":
    sys.exit(main())
