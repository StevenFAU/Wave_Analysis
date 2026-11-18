#!/usr/bin/env python
"""Ridge probe of frozen DINOv3 ViT-B/16 features at the validation buoys: picks the pretraining checkpoint.

    python scripts/dinov3_probe.py --package <ndbc_buoycam_heights_v2 dir> \\
        --config configs/experiments/ndbc_buoycam_heights_v2.yaml --run-dir <pretraining output dir> \\
        --out <dir> [--limit N]

For the public weights (``public``) and every teacher checkpoint of the run
(``<run-dir>/eval/training_<i>/teacher_checkpoint.pth``, step ``i + 1``):

1. the frozen features of every strip of the station split's training and
   validation buoys, exactly as Bar B3 reads them (``HeightsDataset`` with the
   config's view size and evaluation transform; timm's
   ``vit_base_patch16_dinov3.lvd1689m``, patch tokens averaged, bf16), each strip
   the mean of its usable views;
2. a ridge regression (standardised features, alpha from 5-fold cross-validation
   grouped by buoy) of log H_s and log wind-wave height fitted on the training
   buoys, and its RMSE in metres at the validation buoys;
3. the spread of the features (mean over dimensions of their standard deviation
   across validation strips), to see a collapse.

The test buoys are never read. The checkpoint with the lowest validation RMSE of
H_s is the one for Bar P1 (``docs/experiments/domain_pretraining.md``). Writes
``probe.csv``, ``probe.json`` and, with ``--save-timm``, each checkpoint's backbone
as a timm state dict (``timm_<step>.pth``, for ``train_wave_heights.py
--backbone-weights``).
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

from wave_analysis.config import load_yaml
from wave_analysis.datasets.buoycam import usable_view_mask
from wave_analysis.datasets.splits import TRAIN, VAL
from wave_analysis.models.sea_state import TrainSettings, _autocast, build_transforms
from wave_analysis.models.wave_heights import HeightsDataset, log_heights
from wave_analysis.pretraining.dinov3_weights import load_teacher_backbone

TIMM_NAME = "vit_base_patch16_dinov3.lvd1689m"
TARGETS = {"hs": "Hs_label_m", "wind_wave": "wind_wave_label_m"}
ALPHAS = np.logspace(-2, 4, 13)


def checkpoints(run_dir: Path) -> list[tuple[str, Path]]:
    """``(step, teacher_checkpoint.pth)`` of a DINOv3 run, by step."""
    found = []
    for p in sorted((run_dir / "eval").glob("*/teacher_checkpoint.pth")):
        it = p.parent.name.rsplit("_", 1)[-1]
        if it.isdigit():
            found.append((str(int(it) + 1), p))
    return sorted(found, key=lambda x: int(x[0]))


def backbone(state: dict[str, torch.Tensor] | None, device: torch.device) -> torch.nn.Module:
    import timm

    net = timm.create_model(
        TIMM_NAME, pretrained=state is None, num_classes=0, dynamic_img_size=True,
        dynamic_img_pad=True,
    )  # fmt: skip
    if state is not None:
        net.load_state_dict(state, strict=True)
    return net.to(device).eval()


@torch.no_grad()
def strip_features(net: torch.nn.Module, loader: DataLoader, device: torch.device) -> np.ndarray:
    """``(strips, dim)``: per strip, the mean feature of its usable views."""
    out = []
    for (views, mask), _ in loader:
        b, v = views.shape[:2]
        with _autocast(device, "bf16"):
            f = net(views.to(device, non_blocking=True).flatten(0, 1)).float().reshape(b, v, -1)
        m = mask.to(device).unsqueeze(-1).float()
        out.append(((f * m).sum(1) / m.sum(1).clamp_min(1)).cpu().numpy())
    return np.concatenate(out)


def ridge_fit(x: np.ndarray, y: np.ndarray, alpha: float) -> tuple[np.ndarray, ...]:
    """Ridge on standardised features (closed form); returns what :func:`ridge_predict` needs."""
    mu, sd = x.mean(0), x.std(0)
    sd[sd == 0] = 1.0
    z = (x - mu) / sd
    ym = y.mean()
    coef = np.linalg.solve(z.T @ z + alpha * np.eye(z.shape[1]), z.T @ (y - ym))
    return mu, sd, coef, np.asarray(ym)


def ridge_predict(model: tuple[np.ndarray, ...], x: np.ndarray) -> np.ndarray:
    mu, sd, coef, ym = model
    return np.asarray(((x - mu) / sd) @ coef + ym)


def group_folds(groups: np.ndarray, k: int = 5) -> list[np.ndarray]:
    """``k`` boolean test masks; whole groups (buoys) dealt round-robin in sorted order."""
    names = sorted(set(groups.tolist()))
    return [np.isin(groups, names[i::k]) for i in range(k)]


def ridge_rmse(
    xtr: np.ndarray, ytr: np.ndarray, gtr: np.ndarray, xva: np.ndarray, yva: np.ndarray
) -> tuple[float, float]:
    """Validation RMSE (metres) of a ridge on log heights; alpha from grouped CV on the training buoys."""
    ok_tr, ok_va = np.isfinite(ytr), np.isfinite(yva)
    x, y, g = xtr[ok_tr].astype(np.float64), log_heights(ytr[ok_tr]), gtr[ok_tr]
    cv = []
    for alpha in ALPHAS:
        err = [
            np.mean((ridge_predict(ridge_fit(x[~te], y[~te], alpha), x[te]) - y[te]) ** 2)
            for te in group_folds(g)
        ]
        cv.append(float(np.mean(err)))
    best = float(ALPHAS[int(np.argmin(cv))])
    pred = np.exp(ridge_predict(ridge_fit(x, y, best), xva[ok_va].astype(np.float64)))
    return float(np.sqrt(np.mean((pred - yva[ok_va]) ** 2))), best


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--package", required=True, help="a package with height labels (v2)")
    ap.add_argument("--config", required=True, help="its config (view size, transform, light rule)")
    ap.add_argument("--run-dir", help="a DINOv3 pretraining output directory")
    ap.add_argument("--no-public", action="store_true", help="skip the public weights")
    ap.add_argument("--steps", help="comma-separated steps to probe (default: all)")
    ap.add_argument("--out", required=True)
    ap.add_argument("--save-timm", action="store_true")
    ap.add_argument(
        "--save-features", action="store_true", help="each source's strip features (.npy)"
    )
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--num-workers", type=int, default=8)
    ap.add_argument("--limit", type=int, help="strips per partition (smoke test)")
    ap.add_argument("--device", default="auto")
    args = ap.parse_args(argv)
    device = torch.device(
        ("cuda" if torch.cuda.is_available() else "cpu") if args.device == "auto" else args.device
    )
    pkg, out = Path(args.package), Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    cfg = load_yaml(args.config)
    settings = TrainSettings.from_config(cfg["model"])
    size = (int(settings.view_size[0]), int(settings.view_size[1]))
    ims = cfg["images"]
    labels = pd.read_csv(pkg / "labels.csv", dtype={"station_id": str})
    parts = {}
    for p in (TRAIN, VAL):
        t = labels[labels["split_station"] == p].reset_index(drop=True)
        if args.limit:
            t = t.sample(n=min(args.limit, len(t)), random_state=0).reset_index(drop=True)
        parts[p] = t
    tf = build_transforms(settings, size, train=False)

    def loader(t: pd.DataFrame) -> DataLoader:
        mask = usable_view_mask(
            t, min_mean=float(ims["min_view_mean"]), min_std=float(ims["min_view_std"]),
            n_views=settings.n_views,
        )  # fmt: skip
        ds = HeightsDataset(pkg, t, settings, tf, mask, np.zeros((len(t), 1), np.float32))
        return DataLoader(
            ds, batch_size=args.batch_size, num_workers=args.num_workers,
            pin_memory=device.type == "cuda",
        )  # fmt: skip

    sources: list[tuple[str, Path | None]] = [] if args.no_public else [("public", None)]
    if args.run_dir:
        sources += checkpoints(Path(args.run_dir))
    if args.steps:
        keep = set(args.steps.split(","))
        sources = [s for s in sources if s[0] in keep]
    if not sources:
        raise SystemExit("nothing to probe")
    print(f"probing {[s[0] for s in sources]} on {device}; " + ", ".join(
        f"{p} {len(t):,} strips" for p, t in parts.items()), flush=True)  # fmt: skip

    rows = []
    for name, path in sources:
        t0 = time.monotonic()
        state = None if path is None else load_teacher_backbone(path)
        if state is not None and args.save_timm:
            torch.save(state, out / f"timm_{name}.pth")
        net = backbone(state, device)
        feats = {p: strip_features(net, loader(t), device) for p, t in parts.items()}
        del net
        if args.save_features:
            for p, f in feats.items():
                np.save(out / f"features_{name}_{p}.npy", f.astype(np.float32))
                parts[p][["sample_id"]].to_csv(out / f"features_{p}_strips.csv", index=False)
        row: dict[str, float | str] = {
            "weights": name, "step": 0 if name == "public" else int(name),
            "feature_spread": float(feats[VAL].std(axis=0).mean()),
        }  # fmt: skip
        for tname, col in TARGETS.items():
            rmse, alpha = ridge_rmse(
                feats[TRAIN], parts[TRAIN][col].to_numpy(float),
                parts[TRAIN]["station_id"].to_numpy(),
                feats[VAL], parts[VAL][col].to_numpy(float),
            )  # fmt: skip
            row[f"val_rmse_{tname}_m"], row[f"alpha_{tname}"] = rmse, alpha
        row["seconds"] = time.monotonic() - t0
        rows.append(row)
        print(
            f"{name:>7}: val ridge RMSE H_s {row['val_rmse_hs_m']:.3f} m, wind-wave "
            f"{row['val_rmse_wind_wave_m']:.3f} m, feature spread {row['feature_spread']:.3f}, "
            f"{row['seconds']:.0f} s",
            flush=True,
        )
        pd.DataFrame(rows).to_csv(out / "probe.csv", index=False, float_format="%.5f")

    table = pd.DataFrame(rows)
    ckpts = table[table["weights"] != "public"]
    chosen = None if ckpts.empty else ckpts.loc[ckpts["val_rmse_hs_m"].idxmin()].to_dict()
    public = table[table["weights"] == "public"]
    summary = {
        "rule": "lowest validation ridge RMSE of H_s among the run's teacher checkpoints",
        "chosen": chosen,
        "public": None if public.empty else public.iloc[0].to_dict(),
        "package": str(pkg), "run_dir": args.run_dir, "limit_per_partition": args.limit,
        "strips": {p: len(t) for p, t in parts.items()},
    }  # fmt: skip
    (out / "probe.json").write_text(json.dumps(summary, indent=1, default=str))
    if chosen:
        print(
            f"chosen: step {chosen['weights']} (val ridge RMSE H_s {chosen['val_rmse_hs_m']:.3f} m)"
        )
    return 0


if __name__ == "__main__":
    sys.exit(main())
