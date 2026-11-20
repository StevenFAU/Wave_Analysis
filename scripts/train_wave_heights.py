#!/usr/bin/env python
"""Train the buoy-camera wave-heights model: total, wind-wave and swell height in metres from one strip.

    python scripts/train_wave_heights.py --config configs/experiments/ndbc_buoycam_heights_v2.yaml \\
        --package <package dir> --split station --out runs/<name> [--aux horizon] [--seed N]

The model (``wave_analysis.models.wave_heights.MultiViewRegressor``) reads the six
views of a strip with one backbone; with ``--aux horizon`` each view's horizon
roll and pitch (the package's ``views.csv``, :mod:`wave_analysis.processing.horizon`)
join its image feature. Writes, in ``--out``:

* ``predictions.csv`` - every val/test strip: true and predicted heights (metres);
  with a distribution head (config ``model.distribution``), also each height's
  predicted mean and standard deviation of its natural log (``mu_<h>_log``,
  ``sigma_<h>_log``);
* ``metrics.json`` - the run (``run``) and, per partition and height, the
  regression report (bias, RMSE, R², Pearson, ...) with 95 % intervals of RMSE,
  R² and Pearson from a bootstrap over the config's groups;
* ``history.csv``, ``best.pt``, ``train.log``, ``provenance.yaml``.

Published measures (classes, thresholds) and comparisons with ERA5 are computed
from ``predictions.csv`` by ``scripts/compare_papers.py``.
"""

from __future__ import annotations

import argparse
import json
import platform
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

from wave_analysis.config import load_yaml, repo_root
from wave_analysis.datasets import build_provenance, write_provenance
from wave_analysis.datasets.buoycam import usable_view_mask
from wave_analysis.datasets.provenance import git_state
from wave_analysis.datasets.splits import TEST, TRAIN, VAL
from wave_analysis.evaluation.regression import pearson, r2, regression_report, rmse
from wave_analysis.evaluation.uncertainty import block_bootstrap_ci
from wave_analysis.models.sea_state import TrainSettings, build_transforms, input_size
from wave_analysis.models.wave_heights import (
    HeightsDataset,
    MultiViewRegressor,
    fit_heights,
    log_heights,
    predict_heights,
)

PARTITIONS = (VAL, TEST)
AUX_COLUMNS = ("roll_deg", "pitch_frac", "ok")


def aux_features(views: pd.DataFrame, table: pd.DataFrame, n_views: int) -> np.ndarray:
    """``(strips, n_views, 3)``: de-mounted roll (deg / 5), pitch (share of height x 20), clear flag; 0 where unclear."""
    v = views.copy()
    v["ok"] = v["ok"].astype(bool) & v["roll_deg"].notna() & v["pitch_frac"].notna()
    idx = pd.Series(np.arange(len(table)), index=table["sample_id"])
    v = v[v["sample_id"].isin(idx.index)]
    out = np.zeros((len(table), n_views, 3), dtype=np.float32)
    rows = idx.loc[v["sample_id"]].to_numpy()
    cols = v["view"].to_numpy(dtype=int)
    ok = v["ok"].to_numpy()
    out[rows[ok], cols[ok], 0] = v["roll_deg"].to_numpy()[ok] / 5.0
    out[rows[ok], cols[ok], 1] = v["pitch_frac"].to_numpy()[ok] * 20.0
    out[rows[ok], cols[ok], 2] = 1.0
    return out


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--config", required=True)
    ap.add_argument("--package", required=True)
    ap.add_argument("--split", required=True, help="station or time")
    ap.add_argument("--out", required=True)
    ap.add_argument(
        "--aux", choices=("none", "horizon"), default="none", help="per-view extra inputs"
    )
    ap.add_argument(
        "--backbone", help="override model.backbone (e.g. timm:convnext_tiny.dinov3_lvd1689m)"
    )
    ap.add_argument(
        "--lp-epochs",
        type=int,
        help="override model.lp_epochs (backbone frozen for the first N epochs)",
    )
    ap.add_argument(
        "--backbone-lr-scale",
        type=float,
        help="override model.backbone_lr_scale (the backbone's learning rate as a share of lr)",
    )
    ap.add_argument(
        "--backbone-weights",
        help="a timm state dict loaded into the backbone once built (strict), e.g. continued "
        "DINOv3 pretraining (scripts/dinov3_weights.py to-timm)",
    )
    ap.add_argument(
        "--init-heights-run",
        type=Path,
        help="transfer a full same-fold images-only encoder and view mixer",
    )
    ap.add_argument("--epochs", type=int)
    ap.add_argument("--batch-size", type=int)
    ap.add_argument("--num-workers", type=int)
    ap.add_argument("--amp", choices=("bf16", "fp16", "none"))
    ap.add_argument("--no-pretrained", action="store_true")
    ap.add_argument("--limit", type=int, help="strips per partition (smoke test)")
    ap.add_argument("--device", default="auto")
    ap.add_argument("--seed", type=int)
    args = ap.parse_args(argv)
    git_at_start = git_state(repo_root())

    cfg = load_yaml(args.config)
    mcfg = cfg["model"]
    settings = TrainSettings.from_config(
        mcfg, epochs=args.epochs, batch_size=args.batch_size, num_workers=args.num_workers,
        amp=args.amp, pretrained=False if args.no_pretrained else None, backbone=args.backbone,
    )  # fmt: skip
    seed = int(args.seed if args.seed is not None else cfg.get("seed", 0))
    torch.manual_seed(seed)
    torch.backends.cudnn.benchmark = torch.version.hip is None
    device = torch.device(
        ("cuda" if torch.cuda.is_available() else "cpu") if args.device == "auto" else args.device
    )
    if device.type == "cpu":
        settings.amp = "none"
    targets = list(mcfg["targets"])  # label columns; the first chooses the epoch
    tnames = list(mcfg.get("target_names", targets))
    units = list(mcfg.get("target_units", ["m"] * len(targets)))
    pkg, out = Path(args.package), Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    log_fh = (out / "train.log").open("a", encoding="utf-8")

    def log(msg: str) -> None:
        line = f"{time.strftime('%Y-%m-%d %H:%M:%S')} {msg}"
        print(line, flush=True)
        log_fh.write(line + "\n")
        log_fh.flush()

    labels = pd.read_csv(pkg / "labels.csv", dtype={"station_id": str})
    col = f"split_{args.split}"
    boot = cfg["evaluation"]["bootstrap"]
    gcol = (
        boot["group_col"][args.split] if isinstance(boot["group_col"], dict) else boot["group_col"]
    )
    table = labels[labels[col].isin([TRAIN, *PARTITIONS])].reset_index(drop=True)
    for c in mcfg.get("missing_if_nonpositive", []):
        # zero here means no measurement (an all-zero spectrum), not a calm sea
        table.loc[table[c] <= 0, c] = np.nan
    if args.limit:
        rng = np.random.default_rng(seed)
        keep = np.zeros(len(table), bool)
        for part in table[col].unique():
            idx = np.flatnonzero(table[col].to_numpy() == part)
            keep[rng.choice(idx, min(args.limit, idx.size), replace=False)] = True
        table = table.loc[keep].reset_index(drop=True)
    parts = {p: table[table[col] == p].reset_index(drop=True) for p in (TRAIN, *PARTITIONS)}
    size = input_size(settings, (int(table["height"].iloc[0]), int(table["width"].iloc[0])))
    views = pd.read_csv(pkg / "views.csv") if args.aux == "horizon" else None
    log(
        f"{cfg['name']} split={args.split} aux={args.aux} device={device} backbone={settings.backbone} "
        f"{size[0]}x{size[1]} px; targets {tnames}; "
        + ", ".join(f"{p} {len(t):,}" for p, t in parts.items())
    )

    def y_of(t: pd.DataFrame) -> np.ndarray:
        return np.column_stack([log_heights(t[c].to_numpy()) for c in targets])

    ytr = y_of(parts[TRAIN])
    mean, std = np.nanmean(ytr, axis=0), np.nanstd(ytr, axis=0)
    ims = cfg["images"]

    def loader(t: pd.DataFrame, train: bool) -> DataLoader[Any]:
        mask = usable_view_mask(
            t,
            min_mean=float(ims["min_view_mean"]),
            min_std=float(ims["min_view_std"]),
            n_views=settings.n_views,
        )
        aux = aux_features(views, t, settings.n_views) if views is not None else None
        ds = HeightsDataset(
            pkg, t, settings, build_transforms(settings, size, train=train), mask, y_of(t), aux
        )
        return DataLoader(
            ds, batch_size=settings.batch_size, shuffle=train, drop_last=train and len(t) > settings.batch_size,
            num_workers=settings.num_workers, pin_memory=device.type == "cuda",
            persistent_workers=settings.num_workers > 0,
        )  # fmt: skip

    arch = mcfg.get("mixer", {})
    # {loss: beta_nll, beta: 0.5}: a mean and spread beside the point
    dist = mcfg.get("distribution")
    if dist and dist.get("loss", "beta_nll") != "beta_nll":
        raise SystemExit(f"unknown distribution loss {dist['loss']!r}")
    model = MultiViewRegressor(
        settings.backbone, len(targets), aux_dim=3 if views is not None else 0,
        d_model=int(arch.get("d_model", 256)), heads=int(arch.get("heads", 4)),
        layers=int(arch.get("layers", 1)), dropout=float(arch.get("dropout", 0.1)),
        pretrained=settings.pretrained and args.init_heights_run is None, distribution=bool(dist),
    )  # fmt: skip
    if args.backbone_weights and args.init_heights_run:
        raise SystemExit("--backbone-weights and --init-heights-run both set the encoder; use one")
    if args.backbone_weights:
        state = torch.load(args.backbone_weights, map_location="cpu", weights_only=True)
        model.encoder.load_state_dict(state, strict=True)
        log(f"backbone weights from {args.backbone_weights}")
    if args.init_heights_run:
        previous = json.loads((args.init_heights_run / "metrics.json").read_text())["run"]
        if (
            previous["config"] != "ndbc_buoycam_heights_v2cv"
            or previous["split"] != args.split
            or previous["settings"]["backbone"] != settings.backbone
            or previous["aux"] != "none"
            or previous.get("limit_per_partition") is not None
            or previous["seed"] != seed
        ):
            raise ValueError(
                "transfer requires a full images-only run with matching fold, backbone and seed"
            )
        checkpoint = torch.load(
            args.init_heights_run / "best.pt", map_location="cpu", weights_only=False
        )
        keep = {
            k: v
            for k, v in checkpoint["state_dict"].items()
            if not k.startswith(("head.", "dist_head."))
        }
        missing, unexpected = model.load_state_dict(keep, strict=False)
        if unexpected or any(not k.startswith(("head.", "dist_head.")) for k in missing):
            raise ValueError("incompatible feature extractor or mixer")
    val_true = np.exp(y_of(parts[VAL]))
    lr_scale = args.backbone_lr_scale
    if lr_scale is None:
        lr_scale = float(mcfg.get("backbone_lr_scale", 1.0))
    result = fit_heights(
        model, loader(parts[TRAIN], True), loader(parts[VAL], False), val_true,
        settings=settings, mean=mean, std=std, device=device, log=log, names=tnames, units=units,
        lp_epochs=int(args.lp_epochs if args.lp_epochs is not None else mcfg.get("lp_epochs", 0)),
        backbone_lr_scale=lr_scale, dist_beta=float(dist.get("beta", 0.5)) if dist else 0.5,
        rewarm_epochs=float(mcfg.get("rewarm_epochs", 0)), grad_clip=mcfg.get("grad_clip"),
    )  # fmt: skip
    log(
        f"chosen epoch {result.best_epoch} (val RMSE {tnames[0]} {result.best_rmse_m:.3f} {units[0]})"
    )
    model.load_state_dict(result.state_dict)
    torch.save(
        {"state_dict": result.state_dict, "settings": settings.__dict__, "targets": targets,
         "mean": mean, "std": std, "aux": args.aux, "best_epoch": result.best_epoch,
         "distribution": dist},
        out / "best.pt",
    )  # fmt: skip
    pd.DataFrame(result.history).to_csv(out / "history.csv", index=False)

    rows, metrics = [], {}
    kw = {"n_boot": int(boot["n_boot"]), "alpha": float(boot["alpha"]), "seed": int(boot["seed"])}
    for p in PARTITIONS:
        t = parts[p]
        if t.empty:
            continue
        pred = predict_heights(
            model, loader(t, False), device, mean=mean, std=std, amp=settings.amp,
            with_distribution=bool(dist),
        )  # fmt: skip
        if dist:
            pred, mu_log, sigma_log = pred
        fr = pd.DataFrame({"sample_id": t["sample_id"], "partition": p, "station_id": t["station_id"],
                           "station_day": t["station_day"]})  # fmt: skip
        entry = {}
        for j, (c, nm, unit) in enumerate(zip(targets, tnames, units, strict=True)):
            truth = t[c].to_numpy(dtype=float)
            fr[f"true_{nm}_{unit}"], fr[f"pred_{nm}_{unit}"] = truth, pred[:, j]
            if dist:
                fr[f"mu_{nm}_log"], fr[f"sigma_{nm}_log"] = mu_log[:, j], sigma_log[:, j]
            ok = np.isfinite(truth)
            rep = regression_report(truth[ok], pred[ok, j])
            g = t.loc[ok, gcol].to_numpy()
            for mname, fn in (("rmse", rmse), ("r2", r2), ("pearson", pearson)):
                rep[f"{mname}_ci"] = list(
                    block_bootstrap_ci(truth[ok], pred[ok, j], fn, groups=g, **kw)[1:]
                )
            entry[nm] = rep
            log(f"{p:<5} {nm:<10} n={int(rep['n']):,}: RMSE {rep['rmse']:.3f} {unit} [{rep['rmse_ci'][0]:.3f}, "
                f"{rep['rmse_ci'][1]:.3f}], bias {rep['bias']:+.3f} {unit}, R2 {rep['r2']:.3f}, "
                f"Pearson {rep['pearson']:.3f}, slope {rep['slope']:.2f}")  # fmt: skip
        rows.append(fr)
        metrics[p] = entry
    pd.concat(rows).to_csv(out / "predictions.csv", index=False, float_format="%.5f")
    run_info = {
        "config": cfg["name"], "split": args.split, "task": "heights", "aux": args.aux, "targets": targets,
        "target_names": tnames, "settings": settings.__dict__, "input_size_hw": size,
        "best_epoch": result.best_epoch, "device": str(device), "backbone_lr_scale": lr_scale,
        "lp_epochs": int(args.lp_epochs if args.lp_epochs is not None else mcfg.get("lp_epochs", 0)),
        "distribution": dist, "gpu": torch.cuda.get_device_name(device) if device.type == "cuda" else None,
        "torch": torch.__version__, "limit_per_partition": args.limit, "seed": seed,
        "backbone_weights": args.backbone_weights,
    }  # fmt: skip
    run_info["init_heights_run"] = str(args.init_heights_run) if args.init_heights_run else None
    (out / "metrics.json").write_text(
        json.dumps({"run": run_info, "partitions": metrics}, indent=1, default=str)
    )
    inputs = (
        [pkg / "labels.csv", pkg / "SHA256SUMS", Path(args.config)]
        + ([pkg / "views.csv"] if views is not None else [])
        + ([Path(args.backbone_weights)] if args.backbone_weights else [])
    )
    if args.init_heights_run:
        inputs.extend([args.init_heights_run / "metrics.json", args.init_heights_run / "best.pt"])
    write_provenance(
        build_provenance(
            artifact_id=f"{cfg['name']}_{args.split}_heights_{args.aux}",
            inputs=[x for x in inputs if x.exists()],
            outputs=[out / f for f in ("predictions.csv", "metrics.json", "history.csv", "best.pt")],
            config=cfg, notes=f"torch {torch.__version__}; {run_info['gpu'] or platform.processor() or 'cpu'}",
            git=git_at_start,
        ),
        out / "provenance.yaml",
    )  # fmt: skip
    log(f"done: {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
