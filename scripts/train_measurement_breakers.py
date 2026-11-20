"""Site-held-out breaker classification and day-held-out lidar-target regression."""

import argparse
import json
import subprocess
import tempfile
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset, TensorDataset

from measurement_buoy_probe import ridge

CLASSES = ["Spilling", "Plunging", "Surging"]


def decode(source, member):
    # A seekable MP4 handles files whose sample table follows the video bytes.
    with tempfile.NamedTemporaryFile(suffix=".mp4") as f:
        with zipfile.ZipFile(source) as z:
            f.write(z.read(member))
            f.flush()
        command = [
            "ffmpeg",
            "-v",
            "error",
            "-threads",
            "1",
            "-i",
            f.name,
            "-an",
            "-vf",
            "fps=8/15,scale=128:128:force_original_aspect_ratio=decrease,pad=128:128:(ow-iw)/2:(oh-ih)/2",
            "-threads",
            "1",
            "-frames:v",
            "16",
            "-f",
            "rawvideo",
            "-pix_fmt",
            "rgb24",
            "pipe:1",
        ]
        raw = subprocess.run(command, capture_output=True, check=True).stdout
    frames = np.frombuffer(raw, np.uint8).reshape(-1, 128, 128, 3)
    if not len(frames):
        raise ValueError(f"zero decoded frames: {member}")
    return frames[np.minimum(np.arange(16), len(frames) - 1)].copy()


class YinClips(Dataset):
    def __init__(self, rows, source):
        self.rows = rows.reset_index(drop=True)
        self.source = source

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, i):
        row = self.rows.iloc[i]
        frames = decode(self.source, row.zip_member)
        return torch.from_numpy(frames).permute(0, 3, 1, 2).float() / 255, CLASSES.index(
            row.true_label
        )


class BreakerVideo(nn.Module):
    def __init__(self, arm, pretrained):
        super().__init__()
        from torchvision.models import ResNet18_Weights, resnet18
        from torchvision.models.video import R3D_18_Weights, r3d_18

        self.arm = arm
        if arm == "video":
            self.encoder = r3d_18(weights=R3D_18_Weights.DEFAULT if pretrained else None)
            self.encoder.fc = nn.Identity()
            self.encoder.requires_grad_(False)
            self.head = nn.Linear(512, 3)
            mean, std = [0.43216, 0.394666, 0.37645], [0.22803, 0.22145, 0.216989]
        else:
            self.encoder = resnet18(weights=ResNet18_Weights.DEFAULT if pretrained else None)
            self.encoder.fc = nn.Identity()
            self.encoder.requires_grad_(False)
            self.gru = nn.GRU(512, 64, batch_first=True)
            self.head = nn.Linear(64, 3)
            mean, std = [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]
        self.register_buffer("mean", torch.tensor(mean).view(1, 1, 3, 1, 1))
        self.register_buffer("std", torch.tensor(std).view(1, 1, 3, 1, 1))

    def forward(self, x):
        x = (x - self.mean) / self.std
        self.encoder.eval()
        with torch.no_grad():
            features = (
                self.encoder(x.transpose(1, 2))
                if self.arm == "video"
                else self.encoder(x.flatten(0, 1)).reshape(x.shape[0], x.shape[1], -1)
            )
        if self.arm == "gru":
            _, state = self.gru(features)
            features = state[0]
        return self.head(features)


def run_model(model, train, test, epochs, device, classification=False, batch=8):
    model.to(device)
    opt = torch.optim.AdamW((p for p in model.parameters() if p.requires_grad), lr=0.001)
    loss_fn = nn.CrossEntropyLoss() if classification else nn.SmoothL1Loss()
    for _ in range(epochs):
        model.train()
        for x, y in DataLoader(train, batch_size=batch, shuffle=True):
            loss = loss_fn(model(x.to(device)), y.to(device))
            opt.zero_grad()
            loss.backward()
            opt.step()
    model.eval()
    with torch.inference_mode():
        return np.concatenate(
            [model(x.to(device)).cpu().numpy() for x, _ in DataLoader(test, batch_size=batch)]
        )


def yin(args):
    rows = pd.read_csv(args.index / "clips.csv")
    official = pd.read_csv(args.source / "loso_split_by_database_filename.csv")
    roles = official[official.held_out_site.eq(args.site)][["filename", "split"]]
    data = rows.drop(columns="split").merge(roles, on="filename", validate="1:1")
    train = data[data.split.eq("train")]
    test = data[data.split.eq("test")]
    if train.empty or test.empty or set(train.site) & set(test.site):
        raise ValueError("invalid official held-out site")
    if args.smoke:
        train = train.sample(3, random_state=0)
        test = test.sample(2, random_state=0)
    source = args.source / "Set_A.zip"
    model = BreakerVideo(args.arm, not args.smoke)
    pred = run_model(
        model,
        YinClips(train, source),
        YinClips(test, source),
        args.epochs,
        args.device,
        True,
        args.batch,
    )
    out = test[["filename", "site", "true_label", "database_camera", "webcoos_overlaps"]].copy()
    out["prediction"] = np.array(CLASSES)[pred.argmax(1)]
    return out


def duck(args):
    data = pd.read_csv(args.index / "waves.csv")
    masks = np.load(args.index / "masks.npy", mmap_mode="r")
    train = data.index[data.utc_day != args.day]
    test = data.index[data.utc_day == args.day]
    if (
        not len(train)
        or not len(test)
        or set(data.loc[train, "run_id"]) & set(data.loc[test, "run_id"])
    ):
        raise ValueError("invalid whole-day holdout")
    if args.smoke:
        train = train[:8]
        test = test[:4]
    y = data[["height_m", "period_s"]].to_numpy(np.float32)
    mu = y[train].mean(0)
    std = np.maximum(1e-6, y[train].std(0))

    def dataset(idx):
        return TensorDataset(torch.tensor(masks[idx, None]), torch.tensor((y[idx] - mu) / std))

    model = nn.Sequential(
        nn.Conv2d(1, 12, 5, stride=2),
        nn.ReLU(),
        nn.Conv2d(12, 24, 5, stride=2),
        nn.ReLU(),
        nn.AdaptiveAvgPool2d((2, 2)),
        nn.Flatten(),
        nn.Linear(96, 2),
    )
    pred = (
        run_model(model, dataset(train), dataset(test), args.epochs, args.device, batch=args.batch)
        * std
        + mu
    )
    out = data.loc[test, ["wave_id", "run_id", "utc_day", "label", "height_m", "period_s"]].copy()
    x = data[["mask_fraction"]].to_numpy()
    for i, target in enumerate(["height_m", "period_s"]):
        out["mask_cnn_" + target] = pred[:, i]
        out["mask_fraction_" + target] = ridge(x[train], y[train, i], x[test])
        out["mean_" + target] = mu[i]
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    sub = ap.add_subparsers(dest="task", required=True)
    a = sub.add_parser("yin")
    a.add_argument("--source", type=Path, required=True)
    a.add_argument("--site", required=True)
    a.add_argument("--arm", choices=["video", "gru"], required=True)
    d = sub.add_parser("duck")
    d.add_argument("--day", required=True)
    for p in [a, d]:
        p.add_argument("--index", type=Path, required=True)
        p.add_argument("--out", type=Path, required=True)
        p.add_argument("--device", default="cpu")
        p.add_argument("--batch", type=int, default=8)
        p.add_argument("--epochs", type=int, default=10)
        p.add_argument("--smoke", action="store_true")
    args = ap.parse_args()
    if args.out.exists():
        raise FileExistsError(args.out)
    torch.manual_seed(0)
    if args.smoke:
        args.epochs = 1
    df = yin(args) if args.task == "yin" else duck(args)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(args.out, mode="x", index=False)
    metrics = {}
    if args.task == "yin":
        counts = pd.crosstab(df.true_label, df.prediction).reindex(
            index=CLASSES, columns=CLASSES, fill_value=0
        )
        f1 = 2 * np.diag(counts) / (counts.sum(0).to_numpy() + counts.sum(1).to_numpy()).clip(1)
        metrics = {
            "classes": CLASSES,
            "confusion": counts.to_numpy().tolist(),
            "macro_f1_present_classes": float(f1[counts.sum(1) > 0].mean()),
        }
    else:
        for target in ("height_m", "period_s"):
            metrics[target] = {
                arm: float((df[arm + "_" + target] - df[target]).abs().mean())
                for arm in ["mask_cnn", "mask_fraction", "mean"]
            }
    (args.out.with_suffix(".json")).write_text(
        json.dumps({"settings": vars(args), "metrics": metrics}, default=str, indent=2)
    )


if __name__ == "__main__":
    main()
