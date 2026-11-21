"""Prepare and score the two experiments in docs/experiments/hpc_buoy_followup.md."""

import argparse
import json
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np
import pandas as pd

from measurement_visibility import ridge_multi, wind_features
from wave_analysis.config import repo_root
from wave_analysis.datasets.splits import early_fit_late_test
from wave_analysis.evaluation.regression import r2
from wave_analysis.ingest.checksum import sha256_file
from wind_wave_reliability import score

DATES = ["2026-10-29T00:00Z", "2026-10-31T00:00Z"]
BACKBONE = "timm:convnext_tiny.dinov3_lvd1689m"
KEYS = ["sample_id", "station_id", "station_group", "cv_fold"]
LABEL = "wind_wave_label_m"
ARCHIVE_SHA = "76211456db0dd1e2175e6e48c00f36c55ed2ea5ba3a599b43853a1ae2a7da5a4"
ROOT = repo_root()
RUNS = ROOT / "runs"
PKGS = ROOT / "data/processed/packages"
LABELS = PKGS / "ndbc_buoycam_heights_v2cv/labels.csv"
PREP = RUNS / "buoy-cutoff-forecast-v1"


def read_labels(path):
    return pd.read_csv(path, dtype={"station_id": str}, low_memory=False)


def forecast(dest):
    from sensing_forecast_corrector import evaluate, prepare

    archive = RUNS / "forecast-fusion-nested-v1/rows.parquet"
    if sha256_file(archive) != ARCHIVE_SHA:
        raise ValueError("unexpected historical reference archive")
    source = {
        "labels": LABELS,
        "gfs": RUNS / "gfs-wave-strip-match-v1.parquet",
        "era5": ROOT / "data/processed/era5/buoycam/era5_at_buoys.parquet",
        "depths": RUNS / "claims-inputs/station-depths.csv",
    }
    labels = read_labels(LABELS)
    eligible = pd.read_parquet(
        archive,
        columns=[*KEYS, "observed"],
        filters=[
            ("baseline", "==", "model_wind"),
            ("arrangement", "==", "nested"),
            ("target", "==", "true_wind_wave_m"),
        ],
    ).merge(labels[["sample_id", "time_utc"]], on="sample_id", validate="1:1")
    eligible = eligible[pd.to_datetime(eligible.time_utc, utc=True) >= pd.Timestamp(DATES[1])]
    if (len(eligible), eligible.station_group.nunique(), eligible.station_id.nunique()) != (
        7855,
        39,
        67,
    ):
        raise ValueError("unexpected reference population")
    gfs = pd.read_parquet(source["gfs"])
    era = pd.read_parquet(source["era5"])
    depths = read_labels(source["depths"])
    times = pd.to_datetime(labels.time_utc, utc=True)
    cutoff = pd.Timestamp(DATES[0])
    blocks = []
    # Even the right endpoint used to interpolate ERA5 on an early strip must be early.
    for block, weather in (
        (labels[times < cutoff], era[pd.to_datetime(era.time_utc, utc=True) < cutoff]),
        (labels[times >= pd.Timestamp(DATES[1])], era),
    ):
        frame, cols = prepare(block, gfs[gfs.sample_id.isin(block.sample_id)], weather, depths)
        blocks.append(frame)
    data = pd.concat(blocks, ignore_index=True)
    predictions, coverage = [], []
    for k in range(5):
        fold = f"fold{k}"
        role = "split_" + fold
        frame = early_fit_late_test(data, role, *DATES)
        for partition, part in frame.groupby(role):
            coverage.append(
                {
                    "fold": k,
                    "role": partition,
                    "strips": len(part),
                    "groups": part.station_group.nunique(),
                    "first": part.time_utc.min(),
                    "last": part.time_utc.max(),
                }
            )
        fit = evaluate(frame, cols, [fold])
        predictions.append(fit[fit.target == LABEL])
    pred = pd.concat(predictions, ignore_index=True)
    for d in (pred, eligible):
        d["cv_fold"] = d.cv_fold.astype(str)
    paired = eligible.merge(
        pred[[*KEYS, "model_wind", "observed", "model_wind_arm", "model_wind_val_rmse"]],
        on=KEYS,
        how="left",
        validate="1:1",
        suffixes=("", "_fit"),
    )
    if not np.isfinite(paired.model_wind).all() or not np.allclose(
        paired.observed, paired.observed_fit, atol=1e-5, rtol=0
    ):
        raise ValueError("forecast failed to reproduce eligible rows and references")
    paired.drop(columns="observed_fit").rename(columns={"model_wind": "forecast"}).to_parquet(
        dest / "forecast.parquet", index=False
    )
    pd.DataFrame(coverage).to_csv(dest / "coverage.csv", index=False)
    return {
        "date_split": DATES,
        "source_sha256": {k: sha256_file(v) for k, v in source.items()},
        "eligibility_archive_sha256": ARCHIVE_SHA,
        "forecast_sha256": sha256_file(dest / "forecast.parquet"),
        "strips": len(paired),
    }


def comparison(dest, population, rows, arms, candidate, controls):
    if (
        rows.sample_id.duplicated().any()
        or not np.isfinite(rows[[*arms, "observed"]].to_numpy()).all()
    ):
        raise ValueError("incomplete or duplicate scored row")
    scores, contrasts = score(population, rows, arms, candidate, controls)
    scores.to_csv(dest / "rmse.csv", index=False)
    contrasts.to_csv(dest / "contrasts.csv", index=False)
    rows.to_parquet(dest / "predictions.parquet", index=False)
    primary = (
        contrasts.query("scope == 'all' and weighting == 'strip'")
        .set_index("comparator")
        .loc[controls[0]]
    )
    return {
        "strips": len(rows),
        "buoys": rows.station_id.nunique(),
        "groups": rows.station_group.nunique(),
        "pooled_r2": {arm: r2(rows.observed, rows[arm]) for arm in arms},
        "primary_comparator": controls[0],
        "primary_delta_rmse_m": primary.delta_m,
        "primary_95pct_m": [primary.lo_m, primary.hi_m],
        "primary_met": bool(primary.hi_m < 0),
    }


def heights(dest, prep, runs):
    meta = json.loads((prep / "summary.json").read_text())
    if (
        meta["date_split"] != DATES
        or meta["eligibility_archive_sha256"] != ARCHIVE_SHA
        or sha256_file(prep / "forecast.parquet") != meta["forecast_sha256"]
        or sha256_file(LABELS) != meta["source_sha256"]["labels"]
    ):
        raise ValueError("forecast preparation differs from the cutoff protocol")
    labels = read_labels(LABELS)
    rows = pd.read_parquet(prep / "forecast.parquet")
    pieces, hashes = [], {}
    for run in runs:
        m = json.loads((run / "metrics.json").read_text())["run"]
        fold = m["split"]
        if (
            fold not in [f"fold{k}" for k in range(5)]
            or fold in hashes
            or m.get("date_split") != DATES
            or m["config"] != "ndbc_buoycam_heights_v2cv"
            or m["settings"]["backbone"] != BACKBONE
            or not m["settings"]["pretrained"]
            or m["aux"] != "none"
            or m["seed"] != 0
            or m["lp_epochs"] != 2
            or m["backbone_lr_scale"] != 0.1
            or m["settings"]["epochs"] != 10
            or any(
                m.get(key) is not None
                for key in ("init_heights_run", "backbone_weights", "limit_per_partition")
            )
        ):
            raise ValueError(f"unexpected camera run settings: {run}")
        pred = read_labels(run / "predictions.csv")
        role = f"split_{fold}"
        expected = early_fit_late_test(labels, role, *DATES)
        expected = expected[expected[role].isin(["val", "test"])][
            ["sample_id", "station_id", role]
        ].rename(columns={role: "partition"})
        audit = expected.merge(
            pred,
            on=["sample_id", "station_id", "partition"],
            how="outer",
            validate="1:1",
            indicator=True,
        )
        if audit._merge.ne("both").any():
            raise ValueError(f"camera partitions differ from expected strips: {run}")
        pieces.append(audit.query("partition == 'test'").assign(cv_fold=fold[-1]))
        hashes[fold] = {"run": run.name, "predictions_sha256": sha256_file(run / "predictions.csv")}
    camera = pd.concat(pieces)[
        ["sample_id", "station_id", "cv_fold", "pred_wind_wave_m", "true_wind_wave_m"]
    ]
    rows = rows.merge(
        camera, on=["sample_id", "station_id", "cv_fold"], how="left", validate="1:1"
    ).rename(columns={"pred_wind_wave_m": "camera"})
    if not np.allclose(rows.observed, rows.true_wind_wave_m, atol=1.1e-5, rtol=0):
        raise ValueError("missing camera prediction or inconsistent wind-wave reference")
    rows["blend"] = (rows.camera + rows.forecast) / 2
    return comparison(
        dest, "cutoff", rows, ["forecast", "camera", "blend"], "blend", ["forecast"]
    ) | {"camera_runs": hashes, "forecast_sha256": meta["forecast_sha256"]}


def view_features(directory, labels, k, water, label_sha):
    meta = json.loads((directory / "run.json").read_text())
    if (
        meta["fold"] != k
        or meta["seed"] != 0
        or meta["backbone"] != BACKBONE
        or meta["smoke"]
        or meta["water_only"] != water
        or not meta["saved_features"]
        or meta["package_sha256"] != label_sha
    ):
        raise ValueError(f"wrong feature run: {directory}")
    rows = pd.read_csv(directory / "feature_rows.csv").set_index("sample_id")
    role = labels[f"split_fold{k}"]
    expected = role[role.isin(["train", "val", "test"])]
    if (
        not rows.index.is_unique
        or set(rows.index) != set(expected.index)
        or not np.array_equal(rows.role, expected.loc[rows.index])
    ):
        raise ValueError(f"unexpected sample ids or role in {directory}")
    features = np.load(directory / "features.npy", mmap_mode="r")
    if features.ndim != 2 or len(features) != len(rows):
        raise ValueError(f"invalid feature shape in {directory}")
    return features[rows.index.get_indexer(expected.index)].astype(np.float64), meta


def swell_fold(lab, whole, water, role):
    band = lab[[f"band_{j}_m2" for j in range(4)]].to_numpy(float)
    wind = wind_features(lab)
    ok = np.isfinite(band).all(1) & (band >= 0).all(1) & (band.sum(1) > 0)
    ok &= np.isfinite(wind).all(1) & np.isfinite(whole).all(1) & np.isfinite(water).all(1)
    fit, take = (role == "train") & ok, (role == "test") & ok
    if not fit.any() or not take.any():
        raise ValueError("empty eligible train or test partition")
    out = lab.loc[take, ["station_id", "station_group", "cv_fold"]].copy()
    out["observed"] = 4 * np.sqrt(band[take, 0])
    target = 4 * np.sqrt(band[fit, 0])
    for arm, x in (
        ("wind", wind),
        ("whole", whole),
        ("water", water),
        ("whole_wind", np.column_stack([whole, wind])),
        ("water_wind", np.column_stack([water, wind])),
    ):
        out[arm] = np.maximum(0, ridge_multi(x[fit], target, x[take]))
    return out.reset_index(), {
        "train_total": int((role == "train").sum()),
        "train_matched": int(fit.sum()),
        "test_total": int((role == "test").sum()),
        "test_matched": int(take.sum()),
    }


def swell(dest, whole_dir, water_dir):
    path = PKGS / "ndbc_buoycam_slope_v1/labels.csv"
    label_sha = sha256_file(path)
    labels = read_labels(path).set_index("sample_id")
    if not labels.index.is_unique:
        raise ValueError("duplicate swell label identifier")
    rows, coverage = [], []
    for k in range(5):
        role = labels[f"split_fold{k}"]
        if role.groupby(labels.station_group).nunique().gt(1).any():
            raise ValueError("geographic group crosses roles")
        lab = labels[role.isin(["train", "val", "test"])]
        whole, a = view_features(whole_dir / f"fold{k}", labels, k, False, label_sha)
        water, b = view_features(water_dir / f"fold{k}", labels, k, True, label_sha)
        if a["checkpoint_sha256"] != b["checkpoint_sha256"] or whole.shape != water.shape:
            raise ValueError(f"unmatched view checkpoints or feature shapes for fold {k}")
        predictions, sizes = swell_fold(lab, whole, water, lab[f"split_fold{k}"].to_numpy())
        rows.append(predictions)
        coverage.append({"fold": k, "checkpoint_sha256": a["checkpoint_sha256"], **sizes})
    pd.DataFrame(coverage).to_csv(dest / "coverage.csv", index=False)
    return comparison(
        dest,
        "swell",
        pd.concat(rows, ignore_index=True),
        ["wind", "whole", "water", "whole_wind", "water_wind"],
        "water_wind",
        ["wind", "whole_wind"],
    ) | {"label_sha256": label_sha, "whole_run": whole_dir.name, "water_run": water_dir.name}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    commands = p.add_subparsers(dest="cmd", required=True)
    for cmd in ("forecast", "heights", "swell"):
        sub = commands.add_parser(cmd)
        sub.add_argument(
            "--out",
            type=Path,
            default=PREP if cmd == "forecast" else RUNS / f"buoy-{cmd}-followup-v1",
        )
        if cmd == "heights":
            sub.add_argument("--forecast", type=Path, default=PREP)
            sub.add_argument("--runs", type=Path, nargs=5, required=True)
        if cmd == "swell":
            sub.add_argument("--whole", type=Path, default=RUNS / "claims-inputs/t5-s0")
            sub.add_argument("--water", type=Path, required=True)
    a = p.parse_args()
    if a.out.exists():
        raise FileExistsError(a.out)
    a.out.parent.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(dir=a.out.parent) as directory:
        dest = Path(directory) / "result"
        dest.mkdir()
        if a.cmd == "forecast":
            result = forecast(dest)
        elif a.cmd == "heights":
            result = heights(dest, a.forecast, a.runs)
        else:
            result = swell(dest, a.whole, a.water)
        (dest / "summary.json").write_text(json.dumps(result, indent=2) + "\n")
        dest.rename(a.out)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
