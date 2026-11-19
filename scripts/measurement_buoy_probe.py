"""Fit matched wind, geometry and frozen visual probes inside one registered buoy fold."""

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.cluster.vq import kmeans2
from scipy.linalg import solve

from wave_analysis.config import data_dir
from wave_analysis.datasets.buoycam import usable_view_mask

TARGETS = [
    "slope_rms",
    "slope_finite_depth",
    "slope_below_035",
    "tp_nearest_s",
    "wind_r1",
    *[f"band_{i}_m2" for i in range(4)],
    "wind_wave_label_m",
]


def ridge(x, y, xt, penalty=10.0):
    mean = x.mean(0)
    scale = x.std(0)
    scale[scale < 1e-8] = 1
    x = (x - mean) / scale
    yt = y - y.mean()
    if x.shape[1] > len(x):
        dual = solve(x @ x.T + penalty * np.eye(len(x)), yt, assume_a="pos")
        return ((xt - mean) / scale @ x.T) @ dual + y.mean()
    weights = solve(x.T @ x + penalty * np.eye(x.shape[1]), x.T @ yt, assume_a="pos")
    return (xt - mean) / scale @ weights + y.mean()


def horizon_path(package):
    local = package / "horizons.parquet"
    if local.is_file():
        return local
    return data_dir("processed") / "horizon/ndbc_buoycam_sea_state_v0/horizons.parquet"


def geometry(table, package):
    p = horizon_path(package)
    views = pd.read_parquet(p).query("ok").copy()
    views["roll2"] = views.angle_deg**2
    geo = (
        views.groupby("sample_id")
        .agg(n=("view", "size"), roll=("roll2", "mean"), row_sd=("row_frac", "std"))
        .query("n>=3")
    )
    geo["roll"] = np.sqrt(geo.roll)
    return table.merge(geo, on="sample_id", how="left", validate="1:1")


def wind_features(table):
    speed = table.wind_speed_mps.to_numpy()
    gust = table.wind_gust_mps.to_numpy()
    return np.column_stack([speed, gust, speed**2, gust**2, speed * gust])


def extract(pkg, table, run, device, batch, water_only):
    import torch
    from torch.utils.data import DataLoader

    from wave_analysis.models.sea_state import TrainSettings, build_transforms
    from wave_analysis.models.wave_heights import HeightsDataset, MultiViewRegressor

    # Only load a locally produced checkpoint with known provenance.
    ck = torch.load(run / "best.pt", map_location="cpu", weights_only=False)
    cfg = TrainSettings.from_config(ck["settings"])
    model = MultiViewRegressor(
        cfg.backbone, len(ck["targets"]), pretrained=False, distribution=bool(ck["distribution"])
    )
    model.load_state_dict(ck["state_dict"])
    model.to(device).eval()
    masks = usable_view_mask(table, min_mean=20, min_std=4, n_views=cfg.n_views)
    ds = HeightsDataset(
        pkg,
        table,
        cfg,
        build_transforms(cfg, cfg.view_size, train=False),
        masks,
        np.zeros((len(table), 1)),
    )
    if water_only:
        v = pd.read_parquet(horizon_path(pkg)).query("ok")
        h = (
            v.pivot(index="sample_id", columns="view", values="row_frac")
            .reindex(index=table.sample_id, columns=range(cfg.n_views))
            .to_numpy()
        )
    start = 0
    features = []
    with torch.inference_mode():
        for inputs, _ in DataLoader(ds, batch_size=batch):
            if water_only:
                views, mask = inputs
                for i, fracs in enumerate(h[start : start + len(views)]):
                    for j, frac in enumerate(fracs):
                        # Resize a below-horizon patch to eliminate the viewing angle cue from the sky boundary.
                        if not np.isfinite(frac) or frac + 0.08 >= 0.85:
                            mask[i, j] = False
                            continue
                        top = max(0, int((frac + 0.08) * views.shape[-2]))
                        bottom = int(0.9 * views.shape[-2])
                        views[i, j] = torch.nn.functional.interpolate(
                            views[i, j : j + 1, :, top:bottom, :],
                            size=views.shape[-2:],
                            mode="bilinear",
                            align_corners=False,
                        )[0]
                start += len(views)
                valid = mask.sum(1) >= 3
                mask[~valid, 0] = True
            _, pool = model.tokens(tuple(i.to(device) for i in inputs))
            if water_only:
                pool[~valid.to(device)] = float("nan")
            features.append(pool.cpu().numpy())
    return np.concatenate(features)


def dictionary_shapes(table, features, complete, test, results):
    """Four nonnegative spectral prototypes learned only from the training fold."""
    cols = [f"band_{j}_m2" for j in range(4)]
    energy = table[cols].to_numpy()
    total = energy.sum(1)
    fit = (~test) & complete & np.isfinite(total) & (total > 0)
    take = test & complete
    if fit.sum() < 8 or not take.any():
        return
    fractions = energy[fit] / total[fit, None]
    prototype, cluster = kmeans2(fractions, 4, minit="++", seed=0)
    for arm in ["wind", "image"]:
        if arm not in features:
            continue
        x = features[arm]
        weights = np.column_stack(
            [ridge(x[fit], (cluster == j).astype(float), x[take]) for j in range(4)]
        )
        weights = np.maximum(weights, 1e-8)
        weights /= weights.sum(1, keepdims=True)
        amount = np.maximum(0, ridge(x[fit], total[fit], x[take]))
        est = amount[:, None] * (weights @ prototype)
        for j, col in enumerate(cols):
            results.loc[table.index[take], f"{arm}_dictionary__{col}"] = est[:, j]


def spectral_correction(table, features, complete, test, results):
    """Positive multiplicative corrections on the three fully covered ERA5 bands."""
    for j in (1, 2, 3):
        target = f"band_{j}_m2"
        prior = table["prior_" + target].to_numpy()
        true = table[target].to_numpy()
        valid = (prior > 0) & np.isfinite(prior) & complete
        fit = valid & ~test & np.isfinite(true)
        take = valid & test
        if fit.sum() < 2 or not take.any():
            continue
        for arm in ["wind_geometry", "all_inputs"]:
            if arm not in features:
                continue
            x = np.column_stack((features[arm], np.log(np.maximum(prior, 1e-8))))
            residual = np.log(np.maximum(true[fit], 1e-8) / prior[fit])
            pred = ridge(x[fit], residual, x[take])
            results.loc[table.index[take], f"{arm}_corrected__{target}"] = prior[take] * np.exp(
                pred
            )


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--package", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--fold", type=int, required=True, choices=range(5))
    ap.add_argument(
        "--checkpoint-run", type=Path, help="matched upstream fold, omit for tabular-only baselines"
    )
    ap.add_argument("--prior-bands", type=Path, help="joined ERA5 band CSV, one row per sample_id")
    ap.add_argument(
        "--water-only",
        action="store_true",
        help="keep only the water region below each detected horizon",
    )
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--device", default="cpu")
    ap.add_argument(
        "--smoke", action="store_true", help="up to 12 rows per partition; plumbing only"
    )
    args = ap.parse_args()
    if args.out.exists():
        raise FileExistsError(args.out)
    table = geometry(
        pd.read_csv(args.package / "labels.csv", dtype={"station_id": str}, low_memory=False),
        args.package,
    )
    if args.prior_bands:
        table = table.merge(
            pd.read_csv(args.prior_bands),
            on="sample_id",
            how="left",
            validate="1:1",
            suffixes=("", "_prior"),
        )
    split = f"split_fold{args.fold}"
    if table.groupby("station_group")[split].nunique().max() != 1:
        raise ValueError("a station group crosses partitions")
    table = table[table[split].isin(["train", "test"])].copy()
    if args.smoke:
        table = pd.concat(
            [g.sample(min(12, len(g)), random_state=0) for _, g in table.groupby(split)]
        )
    table = table.reset_index(drop=True)
    w = wind_features(table)
    g = table[["roll", "row_sd"]].to_numpy()
    features = {"wind": w, "geometry": g, "wind_geometry": np.column_stack([w, g])}
    ckhash = None
    if args.checkpoint_run:
        info = json.loads((args.checkpoint_run / "metrics.json").read_text())["run"]
        if (
            info["config"] != "ndbc_buoycam_heights_v2cv"
            or info["split"] != f"fold{args.fold}"
            or info["aux"] != "none"
            or info.get("limit_per_partition") is not None
        ):
            raise ValueError(
                "a full images-only heights-v2cv checkpoint from the same fold is required"
            )
        with (args.checkpoint_run / "best.pt").open("rb") as fh:
            ckhash = hashlib.file_digest(fh, "sha256").hexdigest()
        image = extract(
            args.package, table, args.checkpoint_run, args.device, args.batch, args.water_only
        )
        features.update(image=image, all_inputs=np.column_stack([w, g, image]))
    # Primary comparisons use the same strips for all arms; keep exclusions in outputs.
    complete = np.isfinite(np.column_stack(list(features.values()))).all(axis=1)
    test = table[split].eq("test").to_numpy()
    results = table.loc[
        test,
        [
            "sample_id",
            "station_id",
            "station_group",
            "sun_elevation_deg",
            "usable_views",
            *TARGETS,
            "wind_spread_deg",
        ],
    ].copy()
    results["eligible_inputs"] = complete[test]
    for target in TARGETS:
        y = table[target].to_numpy()
        fit = ~test & complete & np.isfinite(y)
        take = test & complete
        if fit.sum() < 2 or not take.any():
            continue
        for arm, x in features.items():
            p = ridge(x[fit], y[fit], x[take])
            p = np.clip(p, 0, 1) if target == "wind_r1" else np.maximum(p, 0)
            results.loc[table.index[take], f"{arm}__{target}"] = p
            if target == "wind_r1":
                results.loc[table.index[take], f"{arm}__wind_spread_deg"] = np.rad2deg(
                    np.sqrt(2 * (1 - p))
                )
        results.loc[table.index[take], f"median__{target}"] = np.median(y[fit])
    dictionary_shapes(table, features, complete, test, results)
    if args.prior_bands:
        spectral_correction(table, features, complete, test, results)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    from tempfile import TemporaryDirectory

    with TemporaryDirectory(dir=args.out.parent) as temp:
        dest = Path(temp) / "probe"
        dest.mkdir()
        results.to_parquet(dest / "predictions.parquet", index=False)
        (dest / "run.json").write_text(
            json.dumps(
                {
                    "fold": args.fold,
                    "checkpoint_sha256": ckhash,
                    "backbone": info["settings"]["backbone"] if args.checkpoint_run else None,
                    "seed": info["seed"] if args.checkpoint_run else None,
                    "package_sha256": hashlib.sha256(
                        (args.package / "labels.csv").read_bytes()
                    ).hexdigest(),
                    "smoke": args.smoke,
                    "water_only": args.water_only,
                    "prior_bands": str(args.prior_bands) if args.prior_bands else None,
                    "prior_sha256": hashlib.sha256(args.prior_bands.read_bytes()).hexdigest()
                    if args.prior_bands
                    else None,
                    "arms": list(features),
                    "ridge_penalty": 10,
                    "n_fit_inputs": int((~test & complete).sum()),
                    "n_test": int(test.sum()),
                },
                indent=2,
            )
        )
        dest.rename(args.out)


if __name__ == "__main__":
    main()
