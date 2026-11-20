"""Inspect local readiness and summarize paired out-of-fold predictions on common records."""

import argparse
import json
import shlex
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import pandas as pd

from measurement_buoy_probe import GEOMETRY, TARGETS, geometry, ridge, wind_features
from wave_analysis.config import data_dir


def review(args):
    pkg = data_dir("processed") / "packages/webcoos_video_v1"
    files = [
        ("clips.csv", "clips"),
        ("windows.csv", "whole_clip_index"),
        ("pixel_spectrum.csv", "pixel_spectrum"),
    ]
    report = {
        "scope": "local artifacts pulled so far",
        "checked_utc": datetime.now(UTC).isoformat(),
        "buoy_runs": [],
        "missing_fold_seeds": {},
        "upstream_summaries": {},
        "prepared": {},
        "shore": {},
    }
    for name, filename in [
        ("grouped_cv", "oof.json"),
        ("spectral", "spectral.json"),
        ("partition", "fusion.json"),
    ]:
        report["upstream_summaries"][name] = [str(p) for p in args.runs.glob("*/" + filename)]
    completed = set()
    for p in sorted(args.runs.glob("buoy-fold*/metrics.json")):
        info = json.loads(p.read_text())["run"]
        if (
            info.get("config") != "ndbc_buoycam_heights_v2cv"
            or info.get("limit_per_partition")
            or info.get("aux") != "none"
        ):
            continue
        if not all(
            (p.parent / f).is_file() for f in ["best.pt", "predictions.csv", "provenance.yaml"]
        ):
            continue
        arm = info["settings"]["backbone"]
        seed = info["seed"]
        fold = info["split"]
        completed.add((arm, fold, seed))
        report["buoy_runs"].append({"arm": arm, "fold": fold, "seed": seed, "path": str(p.parent)})
    for arm in ("convnext_tiny", "timm:convnext_tiny.dinov3_lvd1689m"):
        report["missing_fold_seeds"][arm] = [
            f"fold{k}/s{s}"
            for k in range(5)
            for s in range(3)
            if (arm, f"fold{k}", s) not in completed
        ]
    if (pkg / "clips.csv").exists():
        clips = pd.read_csv(pkg / "clips.csv")
        for f, label in files:
            if (pkg / f).exists():
                d = clips[clips.clip_id.isin(pd.read_csv(pkg / f, usecols=["clip_id"]).clip_id)]
                report["shore"][label] = {
                    "clips": len(d),
                    "local_dates": d.local_day.nunique(),
                    "cameras": d.camera.value_counts().to_dict(),
                }
    for name, required in {
        "buoys": ["labels.csv", "provenance.json", "bandwidth_checks.csv"],
        "duck": ["masks.npy", "waves.csv"],
        "yin": ["Set_A.zip", "figshare-v5.json", "loso_split_by_database_filename.csv"],
        "yin-index": ["clips.csv"],
        "buoys-depth": ["labels.csv", "bandwidth_checks.csv"],
        "crest-readers": ["reader-A.csv", "reader-B.csv"],
        "foam": ["views.csv"],
        "cdip-xyz": ["displacement.csv"],
        "crest-stacks": ["index.csv"],
        "portable-buoys": ["SHA256SUMS", "horizons.parquet", "labels.csv"],
        "yin-index-536": ["clips.csv"],
        "cdip-xyz-536": ["displacement.csv"],
    }.items():
        report["prepared"][name] = all((args.prep / name / f).is_file() for f in required)
    for name in ("stereo-screened.csv", "station-depths.csv", "era5-pilot-request.json"):
        report["prepared"][name] = (args.prep / name).is_file()
    return report


def plan(args):
    """Print future commands for the chosen winner on this machine."""
    snapshot = review(args)
    backbone = {"a": "convnext_tiny", "dinov3": "timm:convnext_tiny.dinov3_lvd1689m"}[args.arm]
    chosen = [r for r in snapshot["buoy_runs"] if r["arm"] == backbone and r["seed"] == args.seed]
    for k in range(5):
        matches = [r for r in chosen if r["fold"] == f"fold{k}"]
        if len(matches) != 1:
            print(
                f"# fold{k}: {len(matches)} matched upstream runs; resolve before launching this fold."
            )
            continue
        prefix = f"{args.arm}-s{args.seed}-fold{k}"
        pkg = str(args.prep / "buoys-depth")
        if args.stage == "probe":
            command = [
                "scripts/measurements",
                "probe",
                "--package",
                pkg,
                "--fold",
                str(k),
                "--checkpoint-run",
                matches[0]["path"],
                "--device",
                args.device,
                "--out",
                str(args.prep.parent / "measurement-probes" / prefix),
            ]
        else:
            command = [
                "scripts/measurements",
                "finetune",
                "--config",
                "configs/experiments/buoy_slope_v1.yaml",
                "--package",
                pkg,
                "--split",
                f"fold{k}",
                "--init-heights-run",
                matches[0]["path"],
                "--seed",
                str(args.seed),
                "--backbone",
                backbone,
                "--device",
                args.device,
                "--out",
                str(args.prep.parent / "measurement-tuning" / prefix),
            ]
        print(shlex.join(command))


def paired(y, a, b, group, mae=False):
    z = (
        pd.DataFrame({"y": y, "a": a, "b": b, "group": group})
        .replace([np.inf, -np.inf], np.nan)
        .dropna()
    )
    if not len(z):
        return {"rows": 0}
    z["a"] = (z.a - z.y).abs()
    z["b"] = (z.b - z.y).abs()
    if not mae:
        z[["a", "b"]] = z[["a", "b"]] ** 2
    by = z.groupby("group")[["a", "b"]]
    n = by.size().to_numpy()
    sums = by.sum().to_numpy()

    def metric(x):
        return x if mae else np.sqrt(x)

    observed = metric(sums.sum(0) / n.sum())
    rng = np.random.default_rng(0)
    sel = rng.integers(len(n), size=(2000, len(n)))
    boots = metric(sums[sel].sum(1) / n[sel].sum(1)[:, None])
    return {
        "rows": len(z),
        "groups": len(n),
        "candidate": observed[0],
        "comparator": observed[1],
        "delta": float(observed[0] - observed[1]),
        "delta_ci": np.quantile(boots[:, 0] - boots[:, 1], [0.025, 0.975]).tolist()
        if len(n) > 1
        else None,
        "equal_group_delta": float((metric(sums[:, 0] / n) - metric(sums[:, 1] / n)).mean()),
    }


def slope_bar(data, metadata):
    """Bar M-S1 (docs/experiments/wave_measurements.md): slope RMSE, all inputs vs wind + geometry."""
    y = "slope_rms"
    out = {
        "test_strips": len(data),
        "test_buoys": int(data.station_id.nunique()),
        "test_groups": int(data.station_group.nunique()),
        "eligible_strips": int(data.eligible_inputs.sum()),
        "eligible_buoys": int(data.loc[data.eligible_inputs, "station_id"].nunique()),
        "checkpoints": {m["fold"]: m.get("checkpoint_run") for m in metadata},
    }
    for prefix, name in [("", "primary"), ("valfit_", "valfit_sensitivity")]:
        a, b = data[f"{prefix}all_inputs__{y}"], data[f"{prefix}wind_geometry__{y}"]
        if a.notna().any():
            out[name] = paired(data[y], a, b, data.station_group)
    ok = data.eligible_inputs & data[y].notna()
    out["scored_strips"] = int(ok.sum())
    out["scored_buoys"] = int(data.loc[ok, "station_id"].nunique())
    rmse = {}
    for arm in ["all_inputs", "wind_geometry", "wind", "geometry", "image", "median"]:
        col = f"{arm}__{y}"
        if col in data:
            rmse[arm] = float(np.sqrt(np.mean((data.loc[ok, col] - data.loc[ok, y]) ** 2)))
    out["rmse_by_arm"] = rmse
    out["slope_sd_eligible"] = float(data.loc[ok, y].std())
    out["by_fold_delta"] = {
        int(k): float(
            np.sqrt(np.mean((g[f"all_inputs__{y}"] - g[y]) ** 2))
            - np.sqrt(np.mean((g[f"wind_geometry__{y}"] - g[y]) ** 2))
        )
        for k, g in data[ok].groupby("fold")
    }
    out["camera_alone_vs_wind_geometry"] = (
        paired(data[y], data[f"image__{y}"], data[f"wind_geometry__{y}"], data.station_group)
        if f"image__{y}" in data
        else None
    )
    out["wind_wave_check"] = (
        paired(
            data.wind_wave_label_m,
            data["all_inputs__wind_wave_label_m"],
            data["wind_geometry__wind_wave_label_m"],
            data.station_group,
        )
        if "all_inputs__wind_wave_label_m" in data
        else None
    )
    return out


def finetune_bar(args):
    """Bar M-S2 (docs/experiments/wave_measurements.md): the fine-tuned slope against the probe.

    Both cameras are stacked with wind + geometry by a ridge on each fold's validation
    strips (out of sample for both encoders) and scored on its test buoys.
    """
    y = "slope_rms"
    table = geometry(
        pd.read_csv(args.package / "labels.csv", dtype={"station_id": str}, low_memory=False),
        args.package,
    )
    table.loc[table[y] <= 0, y] = np.nan
    probes = {}
    for r in args.probes:
        m = json.loads((r / "run.json").read_text())
        if m["smoke"] or m.get("water_only"):
            raise ValueError(f"{r}: a full probe on whole views is required")
        probes[m["fold"]] = (m, pd.read_parquet(r / "predictions.parquet"))
    rows, folds = [], []
    for run in args.finetunes:
        info = json.loads((run / "metrics.json").read_text())["run"]
        k = int(info["split"].removeprefix("fold"))
        m, probe = probes[k]
        if (
            info["config"] != args.config_name
            or info.get("limit_per_partition")
            or Path(info.get("init_heights_run") or "").name != m["checkpoint_run"]
        ):
            raise ValueError(
                f"{run}: a full {args.config_name} run from fold {k}'s probe checkpoint"
            )
        ft = pd.read_csv(run / "predictions.csv")[["sample_id", "partition", "pred_slope_pct"]]
        t = table.merge(ft, on="sample_id", validate="1:1")
        if set(t.loc[t.partition.eq("test"), "sample_id"]) != set(probe.sample_id):
            raise ValueError(f"{run}: test strips differ from the probe's")
        camera = t.pred_slope_pct.to_numpy() / 100  # the fine-tune predicts percent
        x = np.column_stack([wind_features(t), t[GEOMETRY].to_numpy(), camera])
        ok = np.isfinite(x).all(axis=1)
        fit = ok & t.partition.eq("val").to_numpy() & t[y].notna().to_numpy()
        take = ok & t.partition.eq("test").to_numpy()
        out = t.loc[take, ["sample_id"]].copy()
        out["finetune_all_inputs"] = np.maximum(0, ridge(x[fit], t.loc[fit, y].to_numpy(), x[take]))
        out["finetune_camera"] = camera[take]
        folds.append(k)
        rows.append(probe.merge(out, on="sample_id", how="left", validate="1:1").assign(fold=k))
    if sorted(folds) != list(range(5)):
        raise ValueError("one fine-tune per fold")
    data = pd.concat(rows, ignore_index=True)
    data = data[data.eligible_inputs & data[y].notna()]
    g = data.station_group
    return {
        "scored_strips": int(data.finetune_all_inputs.notna().sum()),
        "bar_M_S2": paired(data[y], data.finetune_all_inputs, data[f"valfit_all_inputs__{y}"], g),
        "finetune_vs_wind_geometry": paired(
            data[y], data.finetune_all_inputs, data[f"valfit_wind_geometry__{y}"], g
        ),
        "finetune_camera_alone_vs_probe_camera": paired(
            data[y], data.finetune_camera, data[f"valfit_image__{y}"], g
        ),
    }


def direction_control(args):
    labels = pd.read_csv(args.package / "labels.csv", low_memory=False)
    oof = pd.read_csv(args.oof, usecols=["sample_id", "pred_dir_deg", "dir_agreement"])
    labels = labels.merge(oof, how="left", on="sample_id", validate="1:1", indicator=True)
    if len(oof) != labels["_merge"].eq("both").sum():
        raise ValueError("unmatched OOF predictions")
    y = np.deg2rad(labels.mwd_label_deg.to_numpy())
    clock = pd.to_datetime(labels.time_utc, utc=True)
    hour = 2 * np.pi * (clock.dt.hour + clock.dt.minute / 60) / 24
    lon, sun = [np.deg2rad(labels[col]) for col in ["longitude", "sun_azimuth_deg"]]
    x = np.column_stack(
        [
            labels.latitude,
            np.cos(lon),
            np.sin(lon),
            labels.sun_elevation_deg,
            np.sin(sun),
            np.cos(sun),
            np.sin(hour),
            np.cos(hour),
        ]
    )
    control = np.full(len(labels), np.nan)
    for k in range(5):
        roles = labels[f"split_fold{k}"]
        finite = np.isfinite(x).all(axis=1)
        train = roles.eq("train") & np.isfinite(y) & finite
        test = roles.eq("test") & finite
        if test.any():
            c, s = [ridge(x[train], fn(y[train]), x[test]) for fn in [np.cos, np.sin]]
            control[test] = np.rad2deg(np.arctan2(s, c))
    ref = labels.mwd_label_deg
    image_error = (labels.pred_dir_deg - ref + 180) % 360 - 180
    control_error = (control - ref + 180) % 360 - 180
    report = {}
    for name, mask in [
        ("all", labels.pred_dir_deg.notna()),
        ("low_seed_agreement", labels.dir_agreement < 0.5),
        ("broad_wind_band", labels.wind_r1 < 0.5),
    ]:
        report[name] = paired(
            np.zeros(mask.sum()),
            image_error[mask],
            control_error[mask],
            labels.station_group[mask],
            mae=True,
        )
    return report


def scores(args):
    if args.task == "direction":
        return direction_control(args)
    if args.task == "finetune":
        return finetune_bar(args)
    if args.task == "peaks":
        lab = pd.read_csv(
            args.package / "labels.csv",
            usecols=[
                "sample_id",
                "station_group",
                "tp_label_s",
                "tp_nearest_s",
                "tp_window_spectrum_s",
                "tp_peak_range_s",
            ],
        )
        pred = pd.read_csv(args.oof)
        data = lab.merge(pred, on="sample_id", validate="1:1")
        if len(data) != len(pred) or not len(data):
            raise ValueError("prediction IDs absent from package")
        # Match the same cases for all definitions.
        data = data.dropna(
            subset=["tp_label_s", "tp_nearest_s", "tp_window_spectrum_s", "pred_tp", "era5_tp"]
        )
        return {
            subset: {
                key: paired(d[key], d.pred_tp, d.era5_tp, d.station_group)
                for key in ["tp_label_s", "tp_nearest_s", "tp_window_spectrum_s"]
            }
            for subset, d in [("all", data), ("switch_gt_3s", data[data.tp_peak_range_s > 3])]
        }
    metadata = [json.loads((r / "run.json").read_text()) for r in args.probes]
    if (
        len(metadata) != 5
        or {r["fold"] for r in metadata} != set(range(5))
        or any(r["smoke"] for r in metadata)
        or len(
            {
                (
                    r["package_sha256"],
                    r["backbone"],
                    r["seed"],
                    r.get("water_only"),
                    r.get("prior_sha256"),
                    tuple(r.get("geometry") or ()),
                )
                for r in metadata
            }
        )
        != 1
    ):
        raise ValueError("provide one full visual probe per fold, same package, seed and recipe")
    data = pd.concat(
        [
            pd.read_parquet(r / "predictions.parquet").assign(fold=m["fold"])
            for r, m in zip(args.probes, metadata, strict=True)
        ],
        ignore_index=True,
    )
    if data.sample_id.duplicated().any():
        raise ValueError("duplicate test strips")
    report = {"bar_M_S1": slope_bar(data, metadata)}
    for name in [*TARGETS, "wind_spread_deg"]:
        y = data[name]
        a = data["all_inputs__" + name]
        b = data["wind_geometry__" + name]
        if name.startswith("band_"):
            y, a, b = [4 * np.sqrt(v) for v in (y, a, b)]
        report[name] = paired(y, a, b, data.station_group, mae=name == "wind_spread_deg")

    # Evaluate shape independently of total energy, using total variation distance.
    def shape_distance(arm):
        cols = [f"{arm}band_{i}_m2" for i in range(4)]
        e = data[cols].to_numpy()
        total = e.sum(1, keepdims=True)
        return np.divide(e, total, out=np.full_like(e, np.nan), where=total > 0)

    true_shape = shape_distance("")
    for arm, comparator in [
        ("all_inputs", "wind_geometry"),
        ("image_dictionary", "wind_dictionary"),
    ]:
        if f"{arm}__band_0_m2" in data:
            errors = [
                np.abs(shape_distance(a + "__") - true_shape).sum(1) / 2 for a in [arm, comparator]
            ]
            report[f"shape_{arm}_vs_{comparator}"] = paired(
                np.zeros(len(data)), *errors, data.station_group, mae=True
            )
    for j in range(4):
        target = f"band_{j}_m2"
        for arm, comp in [
            ("image_dictionary", "wind_dictionary"),
            ("all_inputs_corrected", "wind_geometry_corrected"),
        ]:
            ca, cb = f"{arm}__{target}", f"{comp}__{target}"
            if ca in data and cb in data:
                y, a, b = [4 * np.sqrt(data[c]) for c in [target, ca, cb]]
                report[f"{target}_{arm}_vs_{comp}"] = paired(y, a, b, data.station_group)
    return report


def main():
    p = argparse.ArgumentParser(description=__doc__)
    sub = p.add_subparsers(dest="task", required=True)
    r = sub.add_parser("status")
    r.add_argument("--runs", required=True, type=Path)
    r.add_argument("--prep", required=True, type=Path)
    q = sub.add_parser("plan", help="print commands only, after selecting a recipe")
    q.add_argument("--runs", type=Path, required=True)
    q.add_argument("--prep", type=Path, required=True)
    q.add_argument("--arm", choices=["a", "dinov3"], required=True)
    q.add_argument("--stage", choices=["probe", "finetune"], default="probe")
    q.add_argument("--seed", type=int, default=0)
    q.add_argument("--device", default="cuda")
    b = sub.add_parser("probes")
    b.add_argument("probes", nargs="+", type=Path)
    f = sub.add_parser("finetune", help="bar M-S2: fine-tuned slope against the frozen probe")
    f.add_argument("--package", type=Path, required=True)
    f.add_argument("--probes", nargs=5, type=Path, required=True)
    f.add_argument("--finetunes", nargs=5, type=Path, required=True)
    f.add_argument("--config-name", default="buoy_slope_v1", help="the fine-tune's config name")
    for task in ("peaks", "direction"):
        a = sub.add_parser(task)
        a.add_argument("--package", required=True, type=Path)
        a.add_argument("--oof", required=True, type=Path)
    args = p.parse_args()
    if args.task == "status":
        print(json.dumps(review(args), indent=2))
    elif args.task == "plan":
        plan(args)
    else:
        print(json.dumps(scores(args), indent=2))


if __name__ == "__main__":
    main()
