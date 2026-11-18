#!/usr/bin/env python
"""Train the stage-3 buoy-camera model: heights, periods, wind-sea spectrum shape and mean direction.

    python scripts/train_wave_spectral.py --config configs/experiments/ndbc_buoycam_spectral_v3cv.yaml \\
        --package <package dir> --split fold0 --out runs/<name> [--seed N] [recipe options]

The model is :class:`wave_analysis.models.wave_spectral.MultiViewSpectral`: the heights
model of ``train_wave_heights.py`` (point and distribution heads, here over heights and
periods) plus a wind-sea shape head and a sun-referenced direction head; the config's
``model.spectrum`` and ``model.direction`` name their label columns. The epoch with the
lowest validation RMSE of H_s is kept, as in stage 1. Writes, in ``--out``:

* ``predictions.csv`` - every val/test strip: true and predicted value, ``mu_<t>_log`` and
  ``sigma_<t>_log`` of every scalar target; the predicted direction relative to the sun
  (``pred_dir_rel_cos``/``sin``) and in degrees true (``pred_dir_deg``, the sun's azimuth
  added back) beside the buoy's (``true_dir_deg``); the predicted wind-sea shape
  (``pred_wsf_00`` ..);
* ``metrics.json``, ``history.csv``, ``best.pt``, ``train.log``, ``provenance.yaml``.

Scores against ERA5 come from ``scripts/buoycam_spectral_oof.py``.
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
from wave_analysis.evaluation.regression import rmse
from wave_analysis.models.sea_state import TrainSettings, build_transforms, input_size
from wave_analysis.models.wave_heights import HeightsDataset, log_heights
from wave_analysis.models.wave_spectral import (
    Layout,
    MultiViewSpectral,
    fit_spectral,
    predict_spectral,
)

PARTITIONS = (VAL, TEST)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--config", required=True)
    ap.add_argument("--package", required=True)
    ap.add_argument("--split", required=True, help="fold0 .. fold4")
    ap.add_argument("--out", required=True)
    ap.add_argument("--backbone")
    ap.add_argument("--lp-epochs", type=int)
    ap.add_argument("--backbone-lr-scale", type=float)
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
    targets, tnames = list(mcfg["targets"]), list(mcfg["target_names"])
    spec, direc = mcfg.get("spectrum") or {}, mcfg.get("direction") or {}
    n_bins = int(spec.get("bins", 0))
    shape_cols = [f"{spec['prefix']}{k:02d}" for k in range(n_bins)]
    dir_cols = [direc["cos"], direc["sin"]] if direc else []
    layout = Layout(len(targets), n_bins, bool(direc))
    pkg, out = Path(args.package), Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    log_fh = (out / "train.log").open("a", encoding="utf-8")

    def log(msg: str) -> None:
        line = f"{time.strftime('%Y-%m-%d %H:%M:%S')} {msg}"
        print(line, flush=True)
        log_fh.write(line + "\n")
        log_fh.flush()

    labels = pd.read_csv(pkg / "labels.csv", dtype={"station_id": str}, low_memory=False)
    col = f"split_{args.split}"
    boot = cfg["evaluation"]["bootstrap"]
    table = labels[labels[col].isin([TRAIN, *PARTITIONS])].reset_index(drop=True)
    if args.limit:
        rng = np.random.default_rng(0)
        keep = np.zeros(len(table), bool)
        for part in table[col].unique():
            idx = np.flatnonzero(table[col].to_numpy() == part)
            keep[rng.choice(idx, min(args.limit, idx.size), replace=False)] = True
        table = table.loc[keep].reset_index(drop=True)
    parts = {p: table[table[col] == p].reset_index(drop=True) for p in (TRAIN, *PARTITIONS)}
    size = input_size(settings, (int(table["height"].iloc[0]), int(table["width"].iloc[0])))
    log(f"{cfg['name']} split={args.split} device={device} backbone={settings.backbone} {size[0]}x{size[1]} px; "
        f"scalars {tnames}; shape bins {n_bins}; direction {bool(direc)}; "
        + ", ".join(f"{p} {len(t):,}" for p, t in parts.items()))  # fmt: skip

    def y_of(t: pd.DataFrame) -> np.ndarray:
        cols = [log_heights(t[c].to_numpy()) for c in targets]
        cols += [t[c].to_numpy(float) for c in shape_cols + dir_cols]
        return np.column_stack(cols)

    ytr = y_of(parts[TRAIN])[:, : layout.n_scalar]
    mean, std = np.nanmean(ytr, axis=0), np.nanstd(ytr, axis=0)
    ims = cfg["images"]

    def loader(t: pd.DataFrame, train: bool) -> DataLoader[Any]:
        mask = usable_view_mask(t, min_mean=float(ims["min_view_mean"]), min_std=float(ims["min_view_std"]),
                                n_views=settings.n_views)  # fmt: skip
        ds = HeightsDataset(
            pkg, t, settings, build_transforms(settings, size, train=train), mask, y_of(t)
        )
        return DataLoader(
            ds, batch_size=settings.batch_size, shuffle=train, drop_last=train and len(t) > settings.batch_size,
            num_workers=settings.num_workers, pin_memory=device.type == "cuda",
            persistent_workers=settings.num_workers > 0,
        )  # fmt: skip

    arch = mcfg.get("mixer", {})
    model = MultiViewSpectral(
        settings.backbone, layout.n_scalar, n_bins=n_bins, direction=bool(direc), n_views=settings.n_views,
        d_model=int(arch.get("d_model", 256)), heads=int(arch.get("heads", 4)),
        layers=int(arch.get("layers", 1)), dropout=float(arch.get("dropout", 0.1)),
        pretrained=settings.pretrained,
    )  # fmt: skip
    yv = y_of(parts[VAL])
    val_true = np.column_stack([np.exp(yv[:, : layout.n_scalar]), yv[:, layout.n_scalar :]])
    lr_scale = float(
        args.backbone_lr_scale
        if args.backbone_lr_scale is not None
        else mcfg.get("backbone_lr_scale", 1.0)
    )
    lp_epochs = int(args.lp_epochs if args.lp_epochs is not None else mcfg.get("lp_epochs", 0))
    dist = mcfg.get("distribution") or {}
    result = fit_spectral(
        model, loader(parts[TRAIN], True), loader(parts[VAL], False), val_true, layout,
        settings=settings, mean=mean, std=std, device=device, log=log, names=tnames,
        lp_epochs=lp_epochs, backbone_lr_scale=lr_scale, dist_beta=float(dist.get("beta", 0.5)),
        spectrum_weight=float(spec.get("weight", 1.0)), direction_weight=float(direc.get("weight", 1.0)),
    )  # fmt: skip
    log(f"chosen epoch {result.best_epoch} (val RMSE {tnames[0]} {result.best_rmse:.3f})")
    model.load_state_dict(result.state_dict)
    torch.save({"state_dict": result.state_dict, "settings": settings.__dict__, "targets": targets,
                "mean": mean, "std": std, "layout": layout.__dict__, "best_epoch": result.best_epoch},
               out / "best.pt")  # fmt: skip
    pd.DataFrame(result.history).to_csv(out / "history.csv", index=False)

    rows, metrics = [], {}
    for p in PARTITIONS:
        t = parts[p]
        if t.empty:
            continue
        pr = predict_spectral(
            model, loader(t, False), device, layout, mean=mean, std=std, amp=settings.amp
        )
        fr = pd.DataFrame({"sample_id": t["sample_id"], "partition": p, "station_id": t["station_id"],
                           "station_day": t["station_day"]})  # fmt: skip
        entry: dict[str, Any] = {}
        for j, (c, nm) in enumerate(zip(targets, tnames, strict=True)):
            truth = t[c].to_numpy(dtype=float)
            fr[f"true_{nm}"], fr[f"pred_{nm}"] = truth, pr["point"][:, j]
            fr[f"mu_{nm}_log"], fr[f"sigma_{nm}_log"] = pr["mu_log"][:, j], pr["sigma_log"][:, j]
            ok = np.isfinite(truth)
            entry[nm] = {"n": int(ok.sum()), "rmse": rmse(truth[ok], pr["point"][ok, j])}
        if "direction" in pr:
            dvec = pr["direction"]
            fr["pred_dir_rel_cos"], fr["pred_dir_rel_sin"] = dvec[:, 0], dvec[:, 1]
            rel = np.degrees(np.arctan2(dvec[:, 1], dvec[:, 0]))
            fr["pred_dir_deg"] = (rel + t["sun_azimuth_deg"].to_numpy(float)) % 360
            fr["true_dir_deg"] = (
                t[direc["degrees"]].to_numpy(float) if "degrees" in direc else np.nan
            )
            d = (fr["pred_dir_deg"] - fr["true_dir_deg"] + 180) % 360 - 180
            entry["direction_mae_deg"] = float(np.nanmean(np.abs(d)))
        if "shape" in pr:
            for k in range(n_bins):
                fr[f"pred_wsf_{k:02d}"] = pr["shape"][:, k]
        log(f"{p:<5} " + ", ".join(f"{nm} RMSE {v['rmse']:.3f}" for nm, v in entry.items() if isinstance(v, dict))
            + (f", direction MAE {entry['direction_mae_deg']:.1f} deg" if "direction_mae_deg" in entry else ""))  # fmt: skip
        rows.append(fr)
        metrics[p] = entry
    pd.concat(rows).to_csv(out / "predictions.csv", index=False, float_format="%.5f")
    run_info = {
        "config": cfg["name"], "split": args.split, "task": "spectral", "aux": "none", "targets": targets,
        "target_names": tnames, "settings": settings.__dict__, "input_size_hw": size, "layout": layout.__dict__,
        "best_epoch": result.best_epoch, "device": str(device), "backbone_lr_scale": lr_scale, "lp_epochs": lp_epochs,
        "distribution": dist, "spectrum": spec, "direction": direc,
        "gpu": torch.cuda.get_device_name(device) if device.type == "cuda" else None,
        "torch": torch.__version__, "limit_per_partition": args.limit, "seed": seed,
        "bootstrap_group_col": boot["group_col"],
    }  # fmt: skip
    (out / "metrics.json").write_text(
        json.dumps({"run": run_info, "partitions": metrics}, indent=1, default=str)
    )
    write_provenance(
        build_provenance(
            artifact_id=f"{cfg['name']}_{args.split}_spectral",
            inputs=[x for x in (pkg / "labels.csv", pkg / "SHA256SUMS", Path(args.config)) if x.exists()],
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
