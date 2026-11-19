#!/usr/bin/env python
"""Breaker type at unseen sites: heads on cached frozen features, scored on Yin et al.'s split.

Design and bars: ``docs/experiments/breaker_types.md`` (fixed before any run).
Features: ``scripts/breaker_features.py`` (each clip decoded once).

``fit`` trains every arm on each of the six official leave-one-site-out folds (epoch
chosen on the fold's official validation clips, the held-out site only predicted) and
writes one prediction per clip, arm and seed. Arms:

* ``yin_crnn_b2hz`` (the comparator): Yin et al.'s CRNN (frozen ImageNet ResNet-18,
  their FC/BN encoder head, three-layer LSTM, their recipe) on their own 2 Hz frames
  (Set B, frames 1-59 as in their code);
* ``yin_crnn_a5fps``: the same on all 150 Set A frames (5 fps);
* ``dinov3_temporal`` (the candidate): frozen DINOv3 ViT-B/16 frames at 5 fps, a
  temporal convolution head (kernel 5 = 1 s, dilations 1, 2, 4: 5.8 s of context);
* ``dinov3_orderfree``: the same head with kernel 1: each frame alone, then pooled, so
  the clip's frame order (its motion) cannot be used;
* ``vjepa2_video`` / ``vjepa2_shuffled`` / ``vjepa2_still`` (bar Y-B3): frozen V-JEPA 2
  ViT-L tokens (75 per clip, 0.4 s apart, averaged over space) of the clip in order, of its
  frames shuffled, and of each window's middle frame repeated, with the temporal head;
  ``vjepa2_video_orderfree`` / ``vjepa2_shuffled_orderfree``: the same with kernel 1;
* ``iribarren_fixed`` / ``iribarren_fit``: no pixels: the surf-similarity number from
  ERA5 offshore H_s, T_p (Yin et al.'s per-clip file) and each site's beach slope (their
  Table 1), with Battjes' thresholds or a logistic regression fitted on training sites.

``random5`` reproduces Yin et al.'s own protocol (random stratified 5-fold, epoch and
score on the same fold) for the comparator, as a check of the re-implementation.

``score`` computes the bars: pooled out-of-site macro-F1 over present classes, paired
bootstrap over site-days stratified by site. ``--with`` adds another run's out-of-site
predictions (arms this run lacks), scored on the same draws.

    python scripts/train_breaker_heads.py fit --cache <features> --source <yin dir> --out <dir> --device cuda
    python scripts/train_breaker_heads.py score --run <dir> [--with <other run>/predictions.csv]
"""

from __future__ import annotations

import argparse
import json
import math
import os
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch import nn

CLASSES = ["Spilling", "Plunging", "Surging"]
G = 9.81
# Beach slope tan(beta) of each site: Yin et al. 2025, Table 1.
SLOPE = {"SiteA": 0.10, "SiteB": 0.10, "SiteC": 0.02, "SiteD": 0.15, "SiteE": 0.09, "SiteF": 0.06}
# Standard time of each site, for local dates (A, B: North Carolina; C: Texas;
# D: California; E, F: Fujian).
UTC_OFFSET_H = {"SiteA": -5, "SiteB": -5, "SiteC": -6, "SiteD": -8, "SiteE": 8, "SiteF": 8}
NEURAL = {
    "yin_crnn_b2hz": {"features": "b_resnet18", "frames": "yin", "head": "yin"},
    "yin_crnn_a5fps": {"features": "a_resnet18", "frames": "all", "head": "yin"},
    "dinov3_temporal": {"features": "a_dinov3", "frames": "all", "head": "conv", "kernel": 5},
    "dinov3_orderfree": {"features": "a_dinov3", "frames": "all", "head": "conv", "kernel": 1},
    # bar Y-B3: the V-JEPA 2 cache (breaker_features.py --encoders vjepa2)
    "vjepa2_video": {"features": "a_vjepa2_video", "frames": "all", "head": "conv", "kernel": 5},
    "vjepa2_shuffled": {
        "features": "a_vjepa2_shuffled",
        "frames": "all",
        "head": "conv",
        "kernel": 5,
    },
    "vjepa2_still": {"features": "a_vjepa2_still", "frames": "all", "head": "conv", "kernel": 5},
    "vjepa2_video_orderfree": {
        "features": "a_vjepa2_video",
        "frames": "all",
        "head": "conv",
        "kernel": 1,
    },
    "vjepa2_shuffled_orderfree": {
        "features": "a_vjepa2_shuffled",
        "frames": "all",
        "head": "conv",
        "kernel": 1,
    },
}
PHYSICS = ["iribarren_fixed", "iribarren_fit"]
BARS = {
    "Y-B1": ("dinov3_temporal", "yin_crnn_b2hz"),
    "Y-B2": ("dinov3_temporal", "dinov3_orderfree"),
    "Y-B3": ("vjepa2_video", "vjepa2_shuffled"),
}
BESIDE = [
    ("dinov3_temporal", "iribarren_fit"),
    ("yin_crnn_b2hz", "iribarren_fit"),
    ("yin_crnn_a5fps", "yin_crnn_b2hz"),
    ("dinov3_orderfree", "yin_crnn_b2hz"),
    ("vjepa2_video", "vjepa2_still"),
    ("vjepa2_video_orderfree", "vjepa2_shuffled_orderfree"),
    ("vjepa2_video", "dinov3_temporal"),
    ("vjepa2_video", "dinov3_orderfree"),
    ("vjepa2_video", "yin_crnn_b2hz"),
]
# Shared by every neural arm: Yin et al.'s batch, epochs and early stopping.
BATCH, MAX_EPOCHS, PATIENCE, LR = 50, 50, 5, 1e-3


def log(msg: str) -> None:
    print(f"{time.strftime('%Y-%m-%d %H:%M:%S')} {msg}", flush=True)


# ---------------------------------------------------------------- data


def local_date(site: pd.Series, timestamp_utc: pd.Series) -> pd.Series:
    t = pd.to_datetime(timestamp_utc, utc=True)
    return (t + pd.to_timedelta(site.map(UTC_OFFSET_H), unit="h")).dt.strftime("%Y-%m-%d")


def load_clips(cache: Path, source: Path) -> pd.DataFrame:
    from breaker_features import official_split

    clips = pd.read_csv(cache / "clips.csv")
    split = official_split(source / "loso_split_by_database_filename.csv")
    if not set(clips.filename) <= set(split.filename):
        raise ValueError("cache holds clips the split does not")
    clips["y"] = clips.label.map(CLASSES.index)
    clips["site_day"] = clips.site + " " + local_date(clips.site, clips.timestamp_utc)
    return clips


def loso_folds(clips: pd.DataFrame, source: Path) -> list[dict]:
    """The official folds as indices into the cache's clip order."""
    from breaker_features import official_split

    split = official_split(source / "loso_split_by_database_filename.csv")
    pos = pd.Series(np.arange(len(clips)), index=clips.filename)
    folds = []
    for fold, g in split.groupby("fold"):
        g = g[g.filename.isin(pos.index)]
        idx = {
            s: pos[g.filename[g.split.eq(s)]].to_numpy().copy() for s in ("train", "val", "test")
        }
        site = g.held_out_site.iloc[0]
        if set(clips.site.iloc[idx["test"]]) - {site} or site in set(
            clips.site.iloc[np.r_[idx["train"], idx["val"]]]
        ):
            raise ValueError(f"fold {fold}: the held-out site leaks")
        folds.append(dict(fold=int(fold), test_site=site, **idx))
    return folds


def random_folds(y: np.ndarray, k: int = 5, seed: int = 42) -> list[dict]:
    """Yin et al.'s protocol: stratified random k folds; the held-out fold is both val and test."""
    rng = np.random.default_rng(seed)
    fold_of = np.empty(len(y), int)
    for c in np.unique(y):
        i = np.flatnonzero(y == c)
        rng.shuffle(i)
        fold_of[i] = np.arange(len(i)) % k
    return [
        {
            "fold": f,
            "test_site": "random",
            "train": np.flatnonzero(fold_of != f),
            "val": np.flatnonzero(fold_of == f),
            "test": np.flatnonzero(fold_of == f),
        }
        for f in range(k)
    ]


def yin_frames(b_frames: str) -> np.ndarray:
    """Positions of Yin et al.'s frames in a clip's 60 Set B frames: those numbered 1-59
    (their notebook reads frame_{:06d}.jpg for range(1, 60))."""
    from breaker_features import frame_number

    numbers = np.array([frame_number(n) for n in b_frames.split()])
    keep = np.flatnonzero((numbers >= 1) & (numbers <= 59))
    if len(keep) != 59:
        raise ValueError(f"Set B frame numbers {numbers.min()}-{numbers.max()} do not hold 1-59")
    return keep


def features(cache: Path, arm: str, clips: pd.DataFrame) -> np.ndarray:
    spec = NEURAL[arm]
    x = np.load(cache / f"{spec['features']}.npy", mmap_mode="r")
    if spec["frames"] == "yin":
        keep = {tuple(yin_frames(s)) for s in clips.b_frames}
        if len(keep) != 1:
            raise ValueError("Set B frame numbering differs between clips")
        return np.ascontiguousarray(x[:, list(keep.pop())])
    return np.ascontiguousarray(x)


# ---------------------------------------------------------------- heads


class YinCRNN(nn.Module):
    """Yin et al.'s CRNN after the frozen ResNet-18 (their main_crnn.ipynb): per frame
    FC 1024-BN-ReLU, FC 768-BN-ReLU, dropout 0.3, FC 512; LSTM, 3 layers of 512; the last
    step's output through FC 256-ReLU, dropout 0.3, FC 3."""

    def __init__(self, d_in: int = 512, drop: float = 0.3) -> None:
        super().__init__()
        self.fc1, self.bn1 = nn.Linear(d_in, 1024), nn.BatchNorm1d(1024, momentum=0.01)
        self.fc2, self.bn2 = nn.Linear(1024, 768), nn.BatchNorm1d(768, momentum=0.01)
        self.fc3 = nn.Linear(768, 512)
        self.lstm = nn.LSTM(512, 512, num_layers=3, batch_first=True)
        self.fc4, self.fc5 = nn.Linear(512, 256), nn.Linear(256, len(CLASSES))
        self.drop = drop

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, t, _ = x.shape
        h = F.relu(self.bn1(self.fc1(x.reshape(b * t, -1))))
        h = F.relu(self.bn2(self.fc2(h)))
        h = self.fc3(F.dropout(h, self.drop, self.training)).view(b, t, -1)
        out, _ = self.lstm(h)
        z = F.dropout(F.relu(self.fc4(out[:, -1])), self.drop, self.training)
        return self.fc5(z)


class ConvHead(nn.Module):
    """Per-frame projection, three residual temporal convolutions, mean and max over time.
    kernel=1 makes every layer act on each frame alone: the output then does not depend
    on the frames' order."""

    def __init__(
        self, d_in: int, kernel: int, width: int = 256, dilations=(1, 2, 4), drop: float = 0.3
    ) -> None:
        super().__init__()
        self.inp = nn.Sequential(nn.Linear(d_in, width), nn.GELU(), nn.Dropout(drop))
        self.norms = nn.ModuleList(nn.LayerNorm(width) for _ in dilations)
        self.convs = nn.ModuleList(
            nn.Conv1d(
                width,
                width,
                kernel,
                dilation=d if kernel > 1 else 1,
                padding=(kernel - 1) // 2 * (d if kernel > 1 else 1),
            )
            for d in dilations
        )
        self.drop = nn.Dropout(drop)
        self.out = nn.Sequential(
            nn.LayerNorm(2 * width), nn.Dropout(drop), nn.Linear(2 * width, len(CLASSES))
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.inp(x)
        for norm, conv in zip(self.norms, self.convs, strict=True):
            h = h + self.drop(F.gelu(conv(norm(h).transpose(1, 2)).transpose(1, 2)))
        return self.out(torch.cat([h.mean(1), h.amax(1)], -1))


def make_head(arm: str, d_in: int) -> tuple[nn.Module, str]:
    spec = NEURAL[arm]
    if spec["head"] == "yin":
        return YinCRNN(d_in), "adam"
    return ConvHead(d_in, spec["kernel"]), "adamw"


# ---------------------------------------------------------------- metrics


def confusion(y: np.ndarray, p: np.ndarray) -> np.ndarray:
    return np.bincount(np.asarray(y) * 3 + np.asarray(p), minlength=9).reshape(3, 3)


def macro_f1(conf: np.ndarray) -> np.ndarray:
    """Macro-F1 over the classes present (rows: true, columns: predicted); works on stacks."""
    conf = np.asarray(conf, float)
    tp = np.diagonal(conf, axis1=-2, axis2=-1)
    true, pred = conf.sum(-1), conf.sum(-2)
    with np.errstate(invalid="ignore", divide="ignore"):
        f1 = np.where(true + pred > 0, 2 * tp / (true + pred), 0.0)
    present = true > 0
    return (f1 * present).sum(-1) / present.sum(-1)


def accuracy(conf: np.ndarray) -> np.ndarray:
    conf = np.asarray(conf, float)
    return np.diagonal(conf, axis1=-2, axis2=-1).sum(-1) / conf.sum((-1, -2))


# ---------------------------------------------------------------- training


def standardiser(x: torch.Tensor, train: np.ndarray) -> tuple[torch.Tensor, torch.Tensor]:
    """Per-feature mean and std over every frame of the training clips (float64 sums)."""
    s = torch.zeros(x.shape[-1], dtype=torch.float64, device=x.device)
    s2 = torch.zeros_like(s)
    n = 0
    for c in torch.as_tensor(train).split(256):
        v = x[c.to(x.device)].double().flatten(0, 1)
        s += v.sum(0)
        s2 += (v * v).sum(0)
        n += len(v)
    mean = s / n
    std = (s2 / n - mean**2).clamp_min(0).sqrt().clamp_min(1e-6)
    return mean.float(), std.float()


def predict(head: nn.Module, x: torch.Tensor, idx: np.ndarray, norm) -> torch.Tensor:
    head.eval()
    out = []
    with torch.inference_mode():
        for c in torch.as_tensor(idx).split(256):
            v = x[c.to(x.device)].float()
            if norm is not None:
                v = (v - norm[0]) / norm[1]
            out.append(head(v).float().softmax(-1))
    return torch.cat(out)


def fit_fold(
    arm: str, x: torch.Tensor, y: torch.Tensor, fold: dict, seed: int, max_epochs: int
) -> tuple[np.ndarray, int, list]:
    """Train on the fold's train clips; keep the epoch with the best validation accuracy;
    stop after more than PATIENCE epochs without a new best (Yin et al.'s rule)."""
    torch.manual_seed(seed)
    norm = standardiser(x, fold["train"]) if NEURAL[arm]["head"] == "conv" else None
    head, opt_name = make_head(arm, x.shape[-1])
    head = head.to(x.device)
    opt = (
        torch.optim.Adam(head.parameters(), lr=LR)
        if opt_name == "adam"
        else torch.optim.AdamW(head.parameters(), lr=LR, weight_decay=0.05)
    )
    gen = torch.Generator().manual_seed(seed)
    yv = y[torch.as_tensor(fold["val"]).to(y.device)].cpu().numpy()
    best, best_epoch, best_state, stale, history = -1.0, 0, None, 0, []
    for epoch in range(1, max_epochs + 1):
        t0 = time.time()
        head.train()
        order = torch.as_tensor(fold["train"])[torch.randperm(len(fold["train"]), generator=gen)]
        losses = []
        for c in order.split(BATCH):
            c = c.to(x.device)
            v = x[c].float()
            if norm is not None:
                v = (v - norm[0]) / norm[1]
            loss = F.cross_entropy(head(v), y[c])
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            losses.append(loss.item() * len(c))
        pv = predict(head, x, fold["val"], norm)
        conf = confusion(yv, pv.argmax(1).cpu().numpy())
        acc = float(accuracy(conf))
        history.append(
            {
                "epoch": epoch,
                "train_loss": sum(losses) / len(order),
                "val_loss": float(
                    F.nll_loss(
                        pv.clamp_min(1e-12).log(), y[torch.as_tensor(fold["val"]).to(y.device)]
                    )
                ),
                "val_acc": acc,
                "val_macro_f1": float(macro_f1(conf)),
                "seconds": round(time.time() - t0, 2),
            }
        )
        if acc > best:
            best, best_epoch, stale = acc, epoch, 0
            best_state = {k: v.detach().clone() for k, v in head.state_dict().items()}
        else:
            stale += 1
            if stale > PATIENCE:
                break
    head.load_state_dict(best_state)
    return predict(head, x, fold["test"], norm).cpu().numpy(), best_epoch, history


def physics_features(clips: pd.DataFrame, era5: pd.DataFrame) -> pd.DataFrame:
    e = clips[["filename", "site"]].merge(era5, on="filename", how="left", validate="1:1")
    if e[["wave_height", "wave_period", "wind_speed"]].isna().any().any():
        raise ValueError("ERA5 values missing for some clips")
    slope = e.site.map(SLOPE)
    L0 = G * e.wave_period**2 / (2 * math.pi)
    e["xi0"] = slope / np.sqrt(e.wave_height / L0)
    e["slope"] = slope
    return e


def battjes(xi0: np.ndarray) -> np.ndarray:
    """Battjes (1974) on the offshore surf-similarity number: spilling below 0.5, plunging
    0.5-3.3, surging (collapsing included, as Yin et al.) above 3.3."""
    return np.where(xi0 < 0.5, 0, np.where(xi0 <= 3.3, 1, 2))


def logistic_fit_predict(
    xtr: np.ndarray, ytr: np.ndarray, xte: np.ndarray, l2: float = 1e-3
) -> np.ndarray:
    """Multinomial logistic regression (L-BFGS, standardised inputs) -> class probabilities."""
    mean, std = xtr.mean(0), xtr.std(0) + 1e-9
    a = torch.tensor((xtr - mean) / std, dtype=torch.float64)
    b = torch.tensor(ytr)
    w = torch.zeros(a.shape[1], len(CLASSES), dtype=torch.float64, requires_grad=True)
    c = torch.zeros(len(CLASSES), dtype=torch.float64, requires_grad=True)
    opt = torch.optim.LBFGS(
        [w, c], max_iter=500, tolerance_grad=1e-10, line_search_fn="strong_wolfe"
    )

    def closure():
        opt.zero_grad()
        loss = F.cross_entropy(a @ w + c, b) + l2 * (w**2).sum()
        loss.backward()
        return loss

    opt.step(closure)
    with torch.no_grad():
        return (torch.tensor((xte - mean) / std) @ w + c).softmax(-1).numpy()


def physics_arm(arm: str, phys: pd.DataFrame, y: np.ndarray, fold: dict) -> np.ndarray:
    test = fold["test"]
    if arm == "iribarren_fixed":
        return np.eye(3)[battjes(phys.xi0.to_numpy()[test])]
    cols = np.column_stack(
        [
            np.log(phys.xi0),
            np.log(phys.wave_height),
            np.log(phys.wave_period),
            np.log(phys.slope),
            phys.wind_speed,
        ]
    )
    train = np.r_[fold["train"], fold["val"]]  # no epoch to choose: every training-site clip
    return logistic_fit_predict(cols[train], y[train], cols[test])


def default_arms(cache: Path) -> list[str]:
    """Every neural arm whose features the cache holds, and the physics arms."""
    return [a for a in NEURAL if (cache / f"{NEURAL[a]['features']}.npy").is_file()] + PHYSICS


def fit(args: argparse.Namespace) -> None:
    t_start = time.time()
    if not args.arms:
        args.arms = default_arms(args.cache)
    if unknown := set(args.arms) - set(NEURAL) - set(PHYSICS):
        raise ValueError(f"unknown arms {sorted(unknown)}")
    clips = load_clips(args.cache, args.source)
    y_np = clips.y.to_numpy().copy()
    era5 = pd.read_csv(args.source / "ERA5_offshore_wave_wind_by_clip.csv", encoding="utf-8-sig")
    phys = physics_features(clips, era5)
    protocols = {"loso": loso_folds(clips, args.source)}
    if args.folds:
        protocols["loso"] = [f for f in protocols["loso"] if f["fold"] in args.folds]
    if args.random5:
        protocols["random5"] = random_folds(y_np)
    args.out.mkdir(parents=True, exist_ok=False)
    rows, hist = [], []
    base = clips[["filename", "site", "site_day", "label"]]

    def record(protocol, arm, seed, fold, prob, best_epoch):
        r = base.iloc[fold["test"]].copy()
        r.insert(0, "protocol", protocol)
        r.insert(1, "arm", arm)
        r.insert(2, "seed", seed)
        r.insert(3, "fold", fold["fold"])
        r["pred"] = np.array(CLASSES)[prob.argmax(1)]
        for k, c in enumerate(CLASSES):
            r[f"p_{c}"] = prob[:, k].round(5)
        r["best_epoch"] = best_epoch
        rows.append(r)

    for arm in [a for a in PHYSICS if a in args.arms]:
        for fold in protocols["loso"]:
            record("loso", arm, 0, fold, physics_arm(arm, phys, y_np, fold), 0)
        log(f"{arm}: done")
    device = torch.device(args.device)
    y = torch.as_tensor(y_np, device=device)
    for arm in [a for a in NEURAL if a in args.arms]:
        x = torch.as_tensor(features(args.cache, arm, clips), device=device)
        log(f"{arm}: features {tuple(x.shape)} on {device}")
        jobs = [("loso", s, f) for s in args.seeds for f in protocols["loso"]]
        if "random5" in protocols and arm in args.random5:
            jobs += [("random5", 0, f) for f in protocols["random5"]]
        for protocol, seed, fold in jobs:
            t0 = time.time()
            prob, best_epoch, h = fit_fold(arm, x, y, fold, seed, args.max_epochs)
            record(protocol, arm, seed, fold, prob, best_epoch)
            hist += [dict(protocol=protocol, arm=arm, seed=seed, fold=fold["fold"], **e) for e in h]
            conf = confusion(y_np[fold["test"]], prob.argmax(1))
            log(
                f"{arm} {protocol} seed {seed} fold {fold['fold']} ({fold['test_site']}): epoch {best_epoch}/{len(h)}, "
                f"held-out macro-F1 {macro_f1(conf):.3f}, accuracy {accuracy(conf):.3f}, {time.time() - t0:.0f} s"
            )
        del x
        torch.cuda.empty_cache() if device.type == "cuda" else None
    pd.concat(rows).to_csv(args.out / "predictions.csv", index=False)
    pd.DataFrame(hist).to_csv(args.out / "history.csv", index=False)
    meta = (
        json.loads((args.cache / "meta.json").read_text())
        if (args.cache / "meta.json").is_file()
        else {}
    )
    run_info = {
        "settings": {
            k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items() if k != "func"
        },
        "recipe": {
            "batch": BATCH,
            "max_epochs": args.max_epochs,
            "patience": PATIENCE,
            "lr": LR,
            "selection": "best validation accuracy",
        },
        "cache_meta": meta,
        "commit": os.environ.get("COMMIT", ""),
        "seconds": round(time.time() - t_start),
    }
    (args.out / "run.json").write_text(json.dumps(run_info, indent=2, default=str))
    log(f"predictions written to {args.out}")


# ---------------------------------------------------------------- scoring


def site_day_draws(clusters: pd.DataFrame, n_draws: int = 2000, seed: int = 0) -> np.ndarray:
    """Multiplicity of each site-day in each bootstrap draw: site-days resampled with
    replacement within their own site (each site keeps its number of days)."""
    rng = np.random.default_rng(seed)
    w = np.zeros((n_draws, len(clusters)), np.float32)
    for _, members in clusters.groupby("site").indices.items():
        pick = rng.integers(len(members), size=(n_draws, len(members)))
        for d in range(n_draws):
            np.add.at(w[d], members[pick[d]], 1)
    return w


def score_tables(pred: pd.DataFrame, n_draws: int = 2000, seed: int = 0) -> dict:
    """Per arm (mean over seeds): pooled and equal-site macro-F1, accuracy, per-site
    confusion; per comparison: the difference with its site-day bootstrap interval."""
    pred = pred[pred.protocol.eq("loso")].copy()
    pred["y"] = pred.label.map(CLASSES.index)
    pred["p"] = pred.pred.map(CLASSES.index)
    pred["as"] = pred.arm + "#" + pred.seed.astype(str)
    keys = sorted(pred["as"].unique())
    n_clips = pred.groupby("as").size()
    if (n_clips != n_clips.iloc[0]).any():
        raise ValueError(f"arms cover different clips: {n_clips.to_dict()}")
    if "filename" in pred:
        covered = pred.groupby("as").filename.agg(frozenset)
        if any(c != covered.iloc[0] for c in covered):
            raise ValueError("arms cover different clips")
    clusters = (
        pred[["site", "site_day"]]
        .drop_duplicates()
        .sort_values(["site", "site_day"])
        .reset_index(drop=True)
    )
    cl = pd.Series(np.arange(len(clusters)), index=clusters.site_day)
    pred["c"] = cl[pred.site_day].to_numpy()
    k_of = {k: i for i, k in enumerate(keys)}
    conf = np.zeros((len(clusters), len(keys), 3, 3), np.float32)
    np.add.at(
        conf,
        (pred.c.to_numpy(), pred["as"].map(k_of).to_numpy(), pred.y.to_numpy(), pred.p.to_numpy()),
        1,
    )
    sites = sorted(clusters.site.unique())
    site_of = clusters.site.map({s: i for i, s in enumerate(sites)}).to_numpy()
    onehot = np.eye(len(sites), dtype=np.float32)[site_of]  # cluster -> site

    def stats(c):  # c: (..., clusters, keys, 3, 3) -> pooled f1, equal-site f1, accuracy per key
        pooled = c.sum(-4)
        per_site = np.einsum("...ckij,cs->...skij", c, onehot)
        return macro_f1(pooled), np.nanmean(macro_f1(per_site), -2), accuracy(pooled)

    pooled, equal, acc = stats(conf)
    arms = sorted(pred.arm.unique())
    by_arm = {a: [k_of[k] for k in keys if k.split("#")[0] == a] for a in arms}
    w = site_day_draws(clusters, n_draws, seed)
    boot = [np.zeros((n_draws, len(keys))) for _ in range(2)]
    for lo in range(0, n_draws, 100):
        bc = np.einsum("dc,ckij->dckij", w[lo : lo + 100], conf)
        p_, e_, _ = stats(bc)
        boot[0][lo : lo + 100], boot[1][lo : lo + 100] = p_, e_
    out = {
        "clips": int(n_clips.iloc[0]),
        "site_days": len(clusters),
        "site_days_per_site": clusters.site.value_counts().sort_index().to_dict(),
        "arms": {},
        "comparisons": {},
    }
    for a in arms:
        ks = by_arm[a]
        per_site = np.einsum("cij,cs->sij", conf[:, ks].sum(1), onehot)  # summed over seeds
        out["arms"][a] = {
            "seeds": [int(keys[k].split("#")[1]) for k in ks],
            "macro_f1_pooled": float(pooled[ks].mean()),
            "macro_f1_pooled_per_seed": [round(float(pooled[k]), 4) for k in ks],
            "macro_f1_equal_site": float(equal[ks].mean()),
            "accuracy_pooled": float(acc[ks].mean()),
            "per_site": {
                s: {
                    "macro_f1": float(macro_f1(per_site[i]) if per_site[i].sum() else np.nan),
                    "confusion_summed_over_seeds": per_site[i].astype(int).tolist(),
                }
                for i, s in enumerate(sites)
            },
            "confusion_pooled_summed_over_seeds": conf[:, ks].sum((0, 1)).astype(int).tolist(),
        }
    pairs = [(n, *ab) for n, ab in BARS.items()] + [(f"{a} - {b}", a, b) for a, b in BESIDE]
    for name, a, b in pairs:
        if a not in by_arm or b not in by_arm:
            continue
        r = {"candidate": a, "comparator": b}
        for label, obs, bt in (("pooled", pooled, boot[0]), ("equal_site", equal, boot[1])):
            d = bt[:, by_arm[a]].mean(1) - bt[:, by_arm[b]].mean(1)
            r[label] = {
                "delta": float(obs[by_arm[a]].mean() - obs[by_arm[b]].mean()),
                "ci95": np.quantile(d, [0.025, 0.975]).round(4).tolist(),
            }
        r["met"] = bool(r["pooled"]["ci95"][0] > 0) if name in BARS else None
        if len(by_arm[a]) == len(by_arm[b]) > 1:
            r["per_seed_pooled_delta"] = [
                round(float(pooled[i] - pooled[j]), 4)
                for i, j in zip(by_arm[a], by_arm[b], strict=True)
            ]
        out["comparisons"][name] = r
    return out


def random5_table(pred: pd.DataFrame) -> dict:
    """Yin et al.'s reported quantity: the mean over the five folds of the best validation accuracy."""
    r = pred[pred.protocol.eq("random5")]
    return {
        a: {
            "mean_fold_accuracy": float(
                g.groupby("fold").apply(lambda f: (f.label == f.pred).mean()).mean()
            ),
            "macro_f1": float(
                macro_f1(confusion(g.label.map(CLASSES.index), g.pred.map(CLASSES.index)))
            ),
        }
        for a, g in r.groupby("arm")
    }


def merge_predictions(pred: pd.DataFrame, others: list[Path]) -> tuple[pd.DataFrame, dict]:
    """Add the out-of-site predictions of arms this run lacks from other runs' files."""
    added = {}
    for path in others:
        o = pd.read_csv(path)
        o = o[o.protocol.eq("loso") & ~o.arm.isin(pred.arm.unique())]
        if o.empty:
            continue
        added[str(path)] = sorted(o.arm.unique())
        pred = pd.concat([pred, o[pred.columns]], ignore_index=True)
    return pred, added


def score(args: argparse.Namespace) -> None:
    pred = pd.read_csv(args.run / "predictions.csv")
    pred, added = merge_predictions(pred, getattr(args, "with_", []))
    for path, arms in added.items():
        print(f"from {path}: {', '.join(arms)}")
    out = score_tables(pred, args.draws)
    out["merged_from"] = added
    out["random5"] = random5_table(pred)
    (args.run / "score.json").write_text(json.dumps(out, indent=2))
    print(f"{out['clips']:,} clips, {out['site_days']} site-days")
    for a, r in out["arms"].items():
        sites = " ".join(f"{s[-1]} {v['macro_f1']:.2f}" for s, v in r["per_site"].items())
        print(
            f"{a:18s} macro-F1 pooled {r['macro_f1_pooled']:.3f} (seeds {r['macro_f1_pooled_per_seed']}), "
            f"equal-site {r['macro_f1_equal_site']:.3f}, accuracy {r['accuracy_pooled']:.3f} | {sites}"
        )
    for n, r in out["comparisons"].items():
        p = r["pooled"]
        print(
            f"{n}: {r['candidate']} - {r['comparator']}: pooled {p['delta']:+.3f} {p['ci95']}, "
            f"equal-site {r['equal_site']['delta']:+.3f} {r['equal_site']['ci95']}"
            + ("" if r["met"] is None else f" -> {'met' if r['met'] else 'not met'}")
        )
    for a, r in out["random5"].items():
        print(f"random 5-fold (Yin et al.'s protocol) {a}: accuracy {r['mean_fold_accuracy']:.3f}")


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    sub = ap.add_subparsers(dest="cmd", required=True)
    f = sub.add_parser("fit")
    f.add_argument("--cache", type=Path, required=True)
    f.add_argument("--source", type=Path, required=True, help="split CSV and ERA5 CSV")
    f.add_argument("--out", type=Path, required=True)
    f.add_argument("--device", default="cuda")
    f.add_argument(
        "--arms", nargs="+", default=None, help="default: every arm the cache has features for"
    )
    f.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2])
    f.add_argument("--folds", nargs="*", type=int, default=[])
    f.add_argument("--random5", nargs="*", default=["yin_crnn_b2hz", "dinov3_temporal"])
    f.add_argument("--max-epochs", type=int, default=MAX_EPOCHS)
    f.set_defaults(func=fit)
    s = sub.add_parser("score")
    s.add_argument("--run", type=Path, required=True)
    s.add_argument("--draws", type=int, default=2000)
    s.add_argument(
        "--with",
        dest="with_",
        type=Path,
        nargs="*",
        default=[],
        help="other runs' predictions.csv: their arms are scored beside, on the same draws",
    )
    s.set_defaults(func=score)
    args = ap.parse_args()
    args.func(args)


if __name__ == "__main__":
    import sys

    sys.path.insert(0, str(Path(__file__).resolve().parent))
    main()
