"""Run an individual published-model comparison using the registered buoy split."""

import argparse
import hashlib
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from PIL import Image
from torch import nn
from torch.utils.data import DataLoader, Dataset
from torchvision.models import VGG16_Weights, vgg16
from torchvision.transforms import Compose, Normalize, Resize, ToTensor

from wave_analysis.config import load_yaml
from wave_analysis.datasets.buoycam import usable_view_mask
from wave_analysis.models.baselines.kim2023 import Kim2023SnapshotCNN
from wave_analysis.models.multimodal.yang2026 import Yang2026Fusion


class Strips(Dataset):
    def __init__(self, pkg, frame, arm, attitude):
        self.pkg = pkg
        self.arm = arm
        hs = frame.Hs_label_m.to_numpy(float)
        good = usable_view_mask(frame, min_mean=20, min_std=4, n_views=6)
        if arm == "choi":
            means = frame[[f"view_mean_{i}" for i in range(6)]].to_numpy()
            good = (means >= 30) & (means <= 190)
            valid = (hs < 1.52) | (hs > 3.05)
            target = (hs > 3.05).astype(float)
        elif arm == "kim":
            height = hs * 0.631
            valid = (height >= 0.75) & (height < 3.25)
            target = np.digitize(height, [1.25, 1.75, 2.25, 2.75])
        else:
            valid = np.isfinite(hs)
            target = hs
        self.angles = np.zeros((len(frame), 6, 3), dtype="float32")
        if arm.startswith("yang"):
            for i in range(6):
                v = attitude.xs(i, level="view").reindex(frame.sample_id)
                roll = v.roll_deg.to_numpy()
                pitch = v.pitch_frac.to_numpy()
                good[:, i] &= v.ok.eq(True).to_numpy() & np.isfinite(roll) & np.isfinite(pitch)
                self.angles[:, i, 0] = np.deg2rad(roll)
                self.angles[:, i, 1] = np.arctan(pitch)
            valid &= good.all(1)
        valid &= good.any(1) & np.isfinite(hs)
        self.frame = frame[valid].reset_index(drop=True)
        self.good = good[valid]
        self.angles = self.angles[valid]
        self.target = target[valid]
        self.transform = Compose(
            [Resize((224, 224)), ToTensor()]
            + ([] if arm == "kim" else [Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
        )
        if not len(self.frame):
            raise ValueError("empty eligible split")

    def __len__(self):
        return len(self.frame)

    def __getitem__(self, i):
        with Image.open(self.pkg / self.frame.image.iloc[i]) as raw:
            im = raw.convert("L" if self.arm == "kim" else "RGB")
            width = im.width // 6
            top = im.height // 2 if self.arm == "choi" else 0
            views = torch.stack(
                [
                    self.transform(im.crop((j * width, top, (j + 1) * width, im.height)))
                    for j in range(6)
                ]
            )
        return views, self.good[i], self.angles[i], self.target[i]


def model_for(arm):
    if arm == "kim":
        return Kim2023SnapshotCNN()
    if arm.startswith("yang"):
        return Yang2026Fusion(pretrained=True, use_attitude=arm == "yang")
    trunk = vgg16(weights=VGG16_Weights.IMAGENET1K_V1).features
    return nn.Sequential(trunk, nn.AdaptiveAvgPool2d(1), nn.Flatten(), nn.Linear(512, 2))


def epoch(model, loader, arm, device, amp, opt=None):
    """One pass; without ``opt``, returns strip predictions and (classification) per-view ones."""
    model.train(opt is not None)
    preds, view_preds = [], []
    with torch.set_grad_enabled(opt is not None):
        for images, mask, angles, y in loader:
            images, mask, angles, y = [a.to(device) for a in (images, mask, angles, y)]
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=amp == "bf16"):
                if arm.startswith("yang"):
                    pred = model(images, angles)
                    loss = nn.functional.mse_loss(pred.float(), y.float())
                else:
                    logits = model(images[mask])
                    weights = mask.sum(1)
                    ids = torch.arange(len(y), device=device).repeat_interleave(weights)
                    ce = nn.functional.cross_entropy(
                        logits.float(), y[ids].long(), reduction="none"
                    )
                    loss = (ce / weights[ids]).sum() / len(y)
                    probs = logits.float().softmax(1)
                    pred = torch.zeros((len(y), probs.shape[1]), device=device)
                    pred.index_add_(0, ids, probs)
                    pred = pred / weights[:, None]
                    per_view = torch.full((*mask.shape, probs.shape[1]), np.nan, device=device)
                    per_view[mask] = probs
            if opt is not None:
                opt.zero_grad()
                loss.backward()
                opt.step()
            else:
                preds.append(pred.detach().float().cpu().numpy())
                if not arm.startswith("yang"):
                    view_preds.append(per_view.cpu().numpy())
    if not preds:
        return None, None
    return np.concatenate(preds), np.concatenate(view_preds) if view_preds else None


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for k in ("package", "config", "out"):
        p.add_argument("--" + k, type=Path, required=True)
    p.add_argument("--arm", choices=["choi", "kim", "yang", "yang-no-attitude"], required=True)
    p.add_argument("--split", choices=["station", *[f"fold{i}" for i in range(5)]], required=True)
    p.add_argument("--seed", type=int, choices=[0, 1, 2], required=True)
    p.add_argument("--amp", choices=["bf16", "none"], default="bf16")
    p.add_argument("--num-workers", type=int, default=8)
    a = p.parse_args()
    if not torch.cuda.is_available():
        p.error("requires a training GPU on the shared server")
    torch.manual_seed(a.seed)
    cfg = load_yaml(a.config)
    df = pd.read_csv(a.package / "labels.csv", low_memory=False, dtype={"station_id": str})
    attitude = (
        pd.read_csv(a.package / "views.csv").set_index(["sample_id", "view"])
        if a.arm.startswith("yang")
        else None
    )
    split = "split_" + a.split
    if df.groupby("station_id")[split].nunique().max() != 1:
        raise ValueError("station leakage")
    a.out.mkdir(parents=True, exist_ok=True)
    if any((a.out / n).exists() for n in ("settings.json", "best.pt", "predictions.csv")):
        raise FileExistsError(a.out)
    parts = {
        role: Strips(a.package, df[df[split].eq(role)], a.arm, attitude)
        for role in ("train", "val", "test")
    }
    settings = {
        **vars(a),
        "recipe": cfg,
        "commit": os.environ.get("COMMIT"),
        "split_counts": {k: len(v) for k, v in parts.items()},
        "labels_sha256": hashlib.sha256((a.package / "labels.csv").read_bytes()).hexdigest(),
    }
    (a.out / "settings.json").write_text(json.dumps(settings, default=str, indent=2))
    loaders = {
        k: DataLoader(
            v, batch_size=cfg["batch_size"], num_workers=a.num_workers, shuffle=k == "train"
        )
        for k, v in parts.items()
    }
    model = model_for(a.arm).cuda()
    opt = torch.optim.AdamW(model.parameters(), lr=cfg["lr"], weight_decay=cfg["weight_decay"])
    best = np.inf
    for i in range(cfg["epochs"]):
        epoch(model, loaders["train"], a.arm, "cuda", a.amp, opt)
        pred, _ = epoch(model, loaders["val"], a.arm, "cuda", a.amp)
        y = parts["val"].target
        metric = (
            np.sqrt(np.mean((pred - y) ** 2)) if pred.ndim == 1 else np.mean(pred.argmax(1) != y)
        )
        print(f"epoch {i + 1}, validation error {metric:.6f}", flush=True)
        if metric < best:
            best = metric
            torch.save(model.state_dict(), a.out / "best.pt")
    model.load_state_dict(torch.load(a.out / "best.pt", weights_only=True))
    pred, per_view = epoch(model, loaders["test"], a.arm, "cuda", a.amp)
    out = parts["test"].frame[["sample_id", "station_group", "station_id"]].copy()
    out["target"] = parts["test"].target
    if pred.ndim == 1:
        out["pred_hs_m"] = pred
    else:
        for j, pr in enumerate(pred.T):
            out[f"prob_{j}"] = pr
        # each view's own probabilities (empty where the view was not usable), for
        # the per-view-position report
        for v in range(per_view.shape[1]):
            for j in range(per_view.shape[2]):
                out[f"view{v}_prob_{j}"] = per_view[:, v, j]
    out.to_csv(a.out / "predictions.csv", index=False)


if __name__ == "__main__":
    main()
