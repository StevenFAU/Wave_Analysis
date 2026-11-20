#!/usr/bin/env python
"""Shore-camera video vs single image on identical clips: frozen V-JEPA 2 and DINOv3, small trained heads.

Design fixed before any clip was scored: ``docs/experiments/webcoos_video.md``
(config ``configs/experiments/webcoos_video_v1.yaml``).

1. **Features** (GPU, once; cached in ``<out>/features/``, reused with
   ``--features-from``): each clip's 64 frames (16 s at 4 fps) as two square tiles through

   * V-JEPA 2 ViT-L, frozen, three arms: ``video`` (frames in order), ``still`` (the
     middle frame 64 times: the same encoder without motion) and ``shuffled`` (the
     frames in a fixed random order: appearance kept, motion destroyed);
   * DINOv3 ViT-L, frozen: ``image``, the middle frame.

   Tokens of four layers, average-pooled (V-JEPA 8 x 4 x 4 per tile, DINOv3 4 x 4).
2. **Heads**: per arm an attentive probe (primary) and an MLP on mean tokens, fitted on
   log H_s and log T_p, out of fold over folds of local dates, several seeds.
3. **Scores**: RMSE, R^2 and Pearson of H_s and T_p per arm; the bars (video minus
   still, 95 % interval from a bootstrap over dates); per camera; Kamagata's measures at
   Oak Island; Kim's five classes.

**Whole clips** (bars M3-M5, with ``--windows``; ``docs/experiments/webcoos_video.md``,
"Whole-clip bars"): every clip's K non-overlapping 16-s windows (cut by
``scripts/webcoos_video_clips.py windows``) go through the same frozen V-JEPA 2 (``video``
order) and DINOv3 (each window's middle frame), cached per clip in ``--window-cache``.
One attentive probe per encoder is trained on every window of the training clips, each
window carrying its clip's label, with the folds, inner validation dates, seeds and
hyperparameters above; the 25 (fold, seed) probes of an encoder learn side by side from
one pass over the cache per epoch (each probe takes its own training windows from each
read batch, in its own order, in mini-batches of the configured size). A clip is then
read two ways: ``window`` (the window holding its midpoint) and ``clip`` (all K windows:
energy mean of H_s, geometric mean of T_p). Beside them: the mean of log H_s, a learned
record pooler (``pooled``), the DINOv3 probe read the same two ways, Kim's classes from
1, 11 and K windows, and ridge heads on the pixel spectrum (the package's
``pixel_spectrum.csv``): the spectrum alone, the spectrum with the ``clip`` T_p (bar M5)
and the ``clip`` T_p alone (a recalibration control).

    python scripts/train_video_probe.py --config configs/experiments/webcoos_video_v1.yaml \\
        --package data/processed/packages/webcoos_video_v1 --out runs/<name> [--limit 16] \\
        [--windows <dir of <camera>/<clip>.zip> --window-cache <dir>]
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import os
import sys
import time
import zipfile
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from PIL import Image
from torch import nn

from wave_analysis.config import load_yaml
from wave_analysis.evaluation.regression import pearson, r2, rmse
from wave_analysis.evaluation.uncertainty import paired_block_bootstrap_ci
from wave_analysis.models.video_probe import (
    AttentiveProbe,
    MeanProbe,
    RecordPooler,
    StackedProbes,
    arm_order,
    centred_windows,
    energy_mean,
    geometric_mean,
    midpoint_window,
    normalise,
    pool_tokens,
    square_tiles,
)
from wave_analysis.processing.pixel_spectrum import BREAKING_FEATURES, FEATURES_V2
from wave_analysis.processing.pixel_spectrum import FEATURES as SPECTRUM_FEATURES

TARGETS = {"hs": ("Hs_label_m", "m"), "tp": ("Tp_label_s", "s")}
KIM_EDGES = [0.75, 1.25, 1.75, 2.25, 2.75, 3.25]  # average wave height (m), Kim et al. 2023
KIM_HAVE_OVER_HS = 0.631
KAMAGATA = {"hs_rmse_m": 0.586, "hs_r2": 0.107, "hs_pearson": 0.436, "tp_rmse_s": 1.42}


def log(msg: str) -> None:
    print(f"{time.strftime('%Y-%m-%d %H:%M:%S')} {msg}", flush=True)


class Clips(torch.utils.data.Dataset):
    def __init__(self, root: Path, clips: pd.DataFrame, n_frames: int) -> None:
        self.root, self.clips, self.n = root, clips.reset_index(drop=True), n_frames

    def __len__(self) -> int:
        return len(self.clips)

    def __getitem__(self, i: int) -> tuple[torch.Tensor, int]:
        c = self.clips.iloc[i]
        d = self.root / "frames" / c["camera"] / c["clip_id"]
        f = np.stack(
            [
                np.asarray(Image.open(d / f"{k:02d}.jpg").convert("RGB"))
                for k in range(1, self.n + 1)
            ]
        )
        t = square_tiles(f)  # [2, T, H, H, 3]
        return torch.from_numpy(t).permute(0, 1, 4, 2, 3).contiguous(), i  # [2, T, 3, H, H]


def _hooked(layers: list[nn.Module]) -> tuple[list[torch.Tensor], list[Any]]:
    store: list[torch.Tensor] = []

    def hook(_m: nn.Module, _i: Any, out: Any) -> None:
        store.append(out[0] if isinstance(out, tuple) else out)

    return store, [m.register_forward_hook(hook) for m in layers]


@torch.no_grad()
def extract(args: argparse.Namespace, cfg: dict, clips: pd.DataFrame, out: Path) -> None:
    """Write ``<out>/<arm>.npy`` ([clips, tiles, layers, tokens, dim], float16) for every arm."""
    out.mkdir(parents=True, exist_ok=True)
    dev = torch.device(args.device)
    amp = (
        torch.autocast("cuda", dtype=torch.bfloat16)
        if dev.type == "cuda"
        else torch.autocast("cpu", enabled=False)
    )
    n_frames, tile = int(cfg["data"]["frames"]), int(cfg["data"]["tile_px"])
    loader = torch.utils.data.DataLoader(Clips(args.package, clips, n_frames), batch_size=4,
                                         num_workers=args.num_workers, pin_memory=True)  # fmt: skip
    enc = cfg["encoders"]

    # V-JEPA 2: three arms from one model
    from transformers import VJEPA2Model

    vj = enc["vjepa2"]
    model = VJEPA2Model.from_pretrained(vj["hf_id"], torch_dtype=torch.bfloat16).to(dev).eval()
    patch, tub = int(model.config.patch_size), int(model.config.tubelet_size)
    grid = (n_frames // tub, tile // patch, tile // patch)
    pool = tuple(vj["pool"])
    n_tok = int(np.prod(pool))
    layers = [model.encoder.layer[i] for i in vj["layers"]]
    dim = int(model.config.hidden_size)
    mm = {a: np.lib.format.open_memmap(out / f"{a}.npy", "w+", np.float16,
                                       (len(clips), 2, len(layers), n_tok, dim)) for a in vj["arms"]}  # fmt: skip
    store, hooks = _hooked(layers)
    t0 = time.time()
    for b, (x, idx) in enumerate(loader):
        x = x.to(dev, non_blocking=True)  # [B, 2, T, 3, H, H] uint8
        bsz = x.shape[0]
        for arm in vj["arms"]:
            order = [arm_order(arm, n_frames, clips["clip_id"].iloc[int(i)]) for i in idx]
            xa = torch.stack([x[k][:, torch.as_tensor(o, device=dev)] for k, o in enumerate(order)])
            xa = normalise(xa.flatten(0, 1))  # [B*2, T, 3, H, H]
            store.clear()
            with amp:
                model(pixel_values_videos=xa.to(torch.bfloat16), skip_predictor=True)
            feats = torch.stack(
                [pool_tokens(h, grid, pool) for h in store], dim=1
            )  # [B*2, L, n, D]
            mm[arm][idx.numpy()] = feats.reshape(bsz, 2, *feats.shape[1:]).half().cpu().numpy()
        if b % 25 == 0:
            log(
                f"  V-JEPA 2: {min((b + 1) * 4, len(clips)):,}/{len(clips):,} clips, {time.time() - t0:.0f} s"
            )
    for h in hooks:
        h.remove()
    for m in mm.values():
        m.flush()
    del model
    torch.cuda.empty_cache()

    # DINOv3: the middle frame
    import timm

    dn = enc["dinov3"]
    model = timm.create_model(dn["timm"], pretrained=True, num_classes=0).to(dev).eval()
    pool = tuple(dn["pool"])
    n_tok = int(np.prod(pool))
    dim = int(model.num_features)
    patch = int(model.patch_embed.patch_size[0])
    arm = dn["arms"][0]
    m = np.lib.format.open_memmap(
        out / f"{arm}.npy", "w+", np.float16, (len(clips), 2, len(dn["layers"]), n_tok, dim)
    )
    for x, idx in loader:
        x = x.to(dev, non_blocking=True)[:, :, n_frames // 2]  # [B, 2, 3, H, H], the middle frame
        bsz = x.shape[0]
        with amp:
            inter = model.forward_intermediates(normalise(x.flatten(0, 1)), indices=list(dn["layers"]),
                                                output_fmt="NLC", intermediates_only=True)  # fmt: skip
        g = (1, tile // patch, tile // patch)
        feats = torch.stack([pool_tokens(h, g, pool) for h in inter], dim=1)
        m[idx.numpy()] = feats.reshape(bsz, 2, *feats.shape[1:]).half().cpu().numpy()
    m.flush()
    clips.to_csv(out / "clips.csv", index=False)
    log(f"features written to {out}")


def folds_by_date(days: np.ndarray, k: int) -> np.ndarray:
    """Fold of each clip: whole local dates, dealt round-robin after a fixed shuffle."""
    u = np.array(sorted(set(days)))
    np.random.default_rng(0).shuffle(u)
    k = min(k, len(u))
    fold_of = {d: i % k for i, d in enumerate(u)}
    return np.array([fold_of[d] for d in days])


def fit_head(kind: str, x: torch.Tensor, y: torch.Tensor, tr: np.ndarray, va: np.ndarray,
             hp: dict, seed: int) -> nn.Module:  # fmt: skip
    torch.manual_seed(seed)
    n_out = y.shape[1]
    _, p, n_layers, t, dim = x.shape
    head = (AttentiveProbe(n_layers, dim, p * t, n_out, d=int(hp["d"]), heads=int(hp["heads"]), dropout=float(hp["dropout"]))
            if kind == "attentive" else MeanProbe(n_layers, dim, n_out, d=int(hp["d"]), dropout=float(hp["dropout"])))  # fmt: skip
    head = head.to(x.device)
    opt = torch.optim.AdamW(
        head.parameters(), lr=float(hp["lr"]), weight_decay=float(hp["weight_decay"])
    )
    epochs, bs = int(hp["epochs"]), int(hp["batch_size"])
    steps = epochs * max(1, int(np.ceil(len(tr) / bs)))
    sched = torch.optim.lr_scheduler.OneCycleLR(
        opt, max_lr=float(hp["lr"]), total_steps=steps, pct_start=0.1
    )
    rng = np.random.default_rng(seed)
    best, best_state = np.inf, copy.deepcopy(head.state_dict())
    for _ in range(epochs):
        head.train()
        order = tr[rng.permutation(len(tr))]
        for s in range(0, len(order), bs):
            b = torch.as_tensor(order[s : s + bs], device=x.device)
            loss = nn.functional.smooth_l1_loss(head(x[b]), y[b])
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            sched.step()
        head.eval()
        with torch.no_grad():
            v = float(
                nn.functional.mse_loss(
                    head(x[torch.as_tensor(va, device=x.device)]),
                    y[torch.as_tensor(va, device=x.device)],
                )
            )
        if v < best:
            best, best_state = v, copy.deepcopy(head.state_dict())
    head.load_state_dict(best_state)
    return head.eval()


def out_of_fold(
    kind: str, feats: np.ndarray, clips: pd.DataFrame, cfg: dict, dev: torch.device
) -> np.ndarray:
    """Out-of-fold predictions (natural units) of every target, seeds averaged in log space."""
    hp = cfg["heads"]
    x = torch.from_numpy(np.array(feats)).to(dev)  # a copy: the cache is read-only
    ylog = np.log(clips[[TARGETS[t][0] for t in TARGETS]].to_numpy(dtype=float))
    fold = clips["fold"].to_numpy()
    days = clips["local_day"].to_numpy()
    pred = np.zeros_like(ylog)
    for f in sorted(set(fold)):
        te = np.flatnonzero(fold == f)
        trall = np.flatnonzero(fold != f)
        mu, sd = ylog[trall].mean(0), ylog[trall].std(0) + 1e-6
        y = torch.as_tensor((ylog - mu) / sd, dtype=torch.float32, device=dev)
        acc = np.zeros((len(te), ylog.shape[1]))
        for seed in hp["seeds"]:
            tr_days = np.array(sorted(set(days[trall])))
            if len(tr_days) >= 3:
                vday = np.random.default_rng(1000 * f + int(seed)).choice(tr_days)
                va, tr = trall[days[trall] == vday], trall[days[trall] != vday]
            else:  # smoke runs with too few dates: a random 15 % of clips
                perm = np.random.default_rng(int(seed)).permutation(trall)
                va, tr = perm[: max(1, len(perm) // 7)], perm[max(1, len(perm) // 7) :]
            head = fit_head(kind, x, y, tr, va, hp, int(seed))
            with torch.no_grad():
                acc += head(x[torch.as_tensor(te, device=dev)]).cpu().numpy() * sd + mu
        pred[te] = acc / len(hp["seeds"])
    return np.exp(pred)


def kim_accuracy(y: np.ndarray, p: np.ndarray) -> tuple[float, int]:
    ya, pa = KIM_HAVE_OVER_HS * y, KIM_HAVE_OVER_HS * p
    k = (ya >= KIM_EDGES[0]) & (ya < KIM_EDGES[-1])
    yk, pk = np.digitize(ya[k], KIM_EDGES[1:-1]), np.clip(np.digitize(pa[k], KIM_EDGES[1:-1]), 0, 4)
    return (float(np.mean(yk == pk)) if k.any() else float("nan")), int(k.sum())


def scores(d: pd.DataFrame, cols: list[str], cfg: dict) -> dict:
    b = cfg["evaluation"]["bootstrap"]
    kw = {"n_boot": int(b["n_boot"]), "alpha": float(b["alpha"]), "seed": int(b["seed"])}
    g = d["local_day"].to_numpy()
    res: dict[str, Any] = {"n_clips": len(d), "n_dates": int(d["local_day"].nunique()),
                           "by_camera_n": d["camera"].value_counts().to_dict(), "arms": {}}  # fmt: skip
    for c in cols:
        r: dict[str, Any] = {}
        for t, (lab, unit) in TARGETS.items():
            y, p = d[lab].to_numpy(), d[f"pred_{t}_{c}"].to_numpy()
            r[t] = {f"rmse_{unit}": rmse(y, p), "r2": r2(y, p), "pearson": pearson(y, p), f"bias_{unit}": float(np.mean(p - y)),
                    "by_camera_rmse": {cam: rmse(gg[lab], gg[f"pred_{t}_{c}"]) for cam, gg in d.groupby("camera")}}  # fmt: skip
        r["kim2023_accuracy"], r["kim2023_n"] = kim_accuracy(
            d["Hs_label_m"].to_numpy(), d[f"pred_hs_{c}"].to_numpy()
        )
        oak = d[d["camera"].str.startswith("oakisland")]
        if len(oak) > 2:
            y, p = oak["Hs_label_m"].to_numpy(), oak[f"pred_hs_{c}"].to_numpy()
            r["oak_island"] = {"n": len(oak), "hs_rmse_m": rmse(y, p), "hs_r2": r2(y, p), "hs_pearson": pearson(y, p),
                               "tp_rmse_s": rmse(oak["Tp_label_s"], oak[f"pred_tp_{c}"])}  # fmt: skip
        res["arms"][c] = r
    res["differences"] = {}
    for head in ("attentive", "mean"):
        for other in ("still", "shuffled", "image"):
            a, o = f"video_{head}", f"{other}_{head}"
            if a not in cols or o not in cols:
                continue
            for t, (lab, _unit) in TARGETS.items():
                y = d[lab].to_numpy()
                v, lo, hi = paired_block_bootstrap_ci(
                    y, d[f"pred_{t}_{a}"], d[f"pred_{t}_{o}"], rmse, groups=g, **kw
                )
                res["differences"][f"{a} minus {o}: {t}"] = [float(v), float(lo), float(hi)]
    prim = cfg["heads"]["primary"]
    m1 = res["differences"].get(f"video_{prim} minus still_{prim}: hs")
    m2 = res["differences"].get(f"video_{prim} minus still_{prim}: tp")
    res["bars"] = {"M1 (H_s)": bool(m1 and m1[2] < 0), "M2 (T_p)": bool(m2 and m2[2] < 0)}
    res["published"] = {
        "kamagata2026_held_out": KAMAGATA,
        "kim2023": {"snapshot": 0.783, "video_180s": 0.932, "split": "random frames"},
    }
    return res


def report(res: dict, primary: str) -> str:
    lines = [f"# Shore video vs single image: {res['n_clips']:,} clips, {res['n_dates']} dates "
             f"(folds of whole dates), cameras {res['by_camera_n']}\n",
             "| arm | H_s RMSE (m) | H_s R² | H_s Pearson | T_p RMSE (s) | T_p Pearson | Kim 5-class acc. |",
             "|---|---|---|---|---|---|---|"]  # fmt: skip
    for c, r in res["arms"].items():
        lines.append(f"| {c} | {r['hs']['rmse_m']:.3f} | {r['hs']['r2']:.3f} | {r['hs']['pearson']:.3f} | "
                     f"{r['tp']['rmse_s']:.2f} | {r['tp']['pearson']:.3f} | {r['kim2023_accuracy']:.3f} (n {r['kim2023_n']}) |")  # fmt: skip
    lines += [
        "",
        "Differences in RMSE (first minus second; 95 % interval, bootstrap over dates):",
        "",
    ]
    lines += [
        f"- {k}: {v[0]:+.3f} [{v[1]:+.3f}, {v[2]:+.3f}]" for k, v in res["differences"].items()
    ]
    lines += [
        "",
        f"Bars ({primary} probe): "
        + ", ".join(f"{k} {'met' if v else 'not met'}" for k, v in res["bars"].items()),
    ]
    oak = {c: r["oak_island"] for c, r in res["arms"].items() if "oak_island" in r}
    if oak:
        lines += ["", f"Oak Island (Kamagata 2026 held-out: H_s RMSE {KAMAGATA['hs_rmse_m']} m, R² {KAMAGATA['hs_r2']}, "
                      f"Pearson {KAMAGATA['hs_pearson']}, T_p RMSE {KAMAGATA['tp_rmse_s']} s):", ""]  # fmt: skip
        lines += [f"- {c}: n {o['n']}, H_s RMSE {o['hs_rmse_m']:.3f} m, R² {o['hs_r2']:.3f}, Pearson {o['hs_pearson']:.3f}, "
                  f"T_p RMSE {o['tp_rmse_s']:.2f} s" for c, o in oak.items()]  # fmt: skip
    return "\n".join(lines) + "\n"


# --- Whole clips: bars M3-M5 ---------------------------------------------------------

WINDOW_FEATURES_VERSION = 1  # bump when the window extraction changes: a new cache folder
WINDOW_ARMS = {"video": "vjepa2", "image": "dinov3"}  # cached per window; "vmean" feeds pooled
READ_BATCH = 256  # windows read from the cache at a time, shared by the probes


def window_cache_dir(cfg: dict, root: Path) -> Path:
    """The cache folder of these encoders and this frame recipe (a short hash of them)."""
    enc = cfg["encoders"]
    key = {"v": WINDOW_FEATURES_VERSION, "frames": cfg["data"]["frames"], "tile": cfg["data"]["tile_px"],
           "vjepa2": {k: enc["vjepa2"][k] for k in ("hf_id", "layers", "pool")},
           "dinov3": {k: enc["dinov3"][k] for k in ("timm", "layers", "pool")}}  # fmt: skip
    return root / hashlib.sha256(json.dumps(key, sort_keys=True).encode()).hexdigest()[:12]


def _feature_path(cache: Path, camera: str, clip_id: str, kind: str) -> Path:
    return cache / camera / f"{clip_id}.{kind}.npy"


class Windows(torch.utils.data.Dataset):
    """Window ``w`` of a clip: its 64 frames from the clip's zip, as two square tiles."""

    def __init__(self, items: list[tuple[str, int]], n_frames: int) -> None:
        self.items, self.n = items, n_frames
        self._zips: dict[str, zipfile.ZipFile] = {}

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, i: int) -> tuple[torch.Tensor, int]:
        path, w = self.items[i]
        z = self._zips.get(path)
        if z is None:
            for old in self._zips.values():  # windows come in clip order: keep one zip open
                old.close()
            self._zips = {path: (z := zipfile.ZipFile(path))}
        f = np.stack([np.asarray(Image.open(z.open(f"{w * self.n + k:04d}.jpg")).convert("RGB"))
                      for k in range(self.n)])  # fmt: skip
        return torch.from_numpy(square_tiles(f)).permute(0, 1, 4, 2, 3).contiguous(), i


def _save(path: Path, a: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + f".part{os.getpid()}")
    with open(tmp, "wb") as fh:
        np.save(fh, a)
    tmp.rename(path)


@torch.no_grad()
def extract_windows(args: argparse.Namespace, cfg: dict, clips: pd.DataFrame, cache: Path) -> None:
    """Cache per clip ``<clip>.video.npy`` [K, tiles, layers, 128, D] (V-JEPA 2, frames in order),
    ``<clip>.image.npy`` [K, tiles, layers, 16, D] (DINOv3, each window's middle frame) and
    ``<clip>.vmean.npy`` [K, tiles, layers, D] (the video tokens' mean), float16.
    Clips already cached are skipped, so a cache grows with the package."""
    todo = clips[[not all(_feature_path(cache, c, i, k).exists() for k in ("video", "image", "vmean"))
                  for c, i in zip(clips["camera"], clips["clip_id"], strict=True)]].reset_index(drop=True)  # fmt: skip
    log(f"window features in {cache}: {len(clips) - len(todo):,} clips cached, {len(todo):,} to extract "
        f"({int(todo['K'].sum()):,} windows)")  # fmt: skip
    if todo.empty:
        return
    dev = torch.device(args.device)
    amp = (
        torch.autocast("cuda", dtype=torch.bfloat16)
        if dev.type == "cuda"
        else torch.autocast("cpu", enabled=False)
    )
    n_frames, tile = int(cfg["data"]["frames"]), int(cfg["data"]["tile_px"])
    items = [(str(args.windows / c / f"{i}.zip"), w)
             for c, i, k in zip(todo["camera"], todo["clip_id"], todo["K"], strict=True) for w in range(int(k))]  # fmt: skip
    owner = [(j, w) for j, k in enumerate(todo["K"]) for w in range(int(k))]
    loader = torch.utils.data.DataLoader(Windows(items, n_frames), batch_size=4, num_workers=args.num_workers,
                                         pin_memory=True)  # fmt: skip
    import timm
    from transformers import VJEPA2Model

    enc = cfg["encoders"]
    vj = (
        VJEPA2Model.from_pretrained(enc["vjepa2"]["hf_id"], torch_dtype=torch.bfloat16)
        .to(dev)
        .eval()
    )
    grid_v = (
        n_frames // int(vj.config.tubelet_size),
        tile // int(vj.config.patch_size),
        tile // int(vj.config.patch_size),
    )
    pool_v = tuple(enc["vjepa2"]["pool"])
    store, hooks = _hooked([vj.encoder.layer[i] for i in enc["vjepa2"]["layers"]])
    dn = timm.create_model(enc["dinov3"]["timm"], pretrained=True, num_classes=0).to(dev).eval()
    patch_d = int(dn.patch_embed.patch_size[0])
    grid_d, pool_d = (1, tile // patch_d, tile // patch_d), tuple(enc["dinov3"]["pool"])
    acc: dict[int, dict[str, list]] = {}
    t0, n_done = time.time(), 0
    for b, (x, idx) in enumerate(loader):
        x = x.to(dev, non_blocking=True)  # [B, 2, T, 3, H, H] uint8
        bsz = x.shape[0]
        store.clear()
        with amp:
            vj(
                pixel_values_videos=normalise(x.flatten(0, 1)).to(torch.bfloat16),
                skip_predictor=True,
            )
            inter = dn.forward_intermediates(normalise(x[:, :, n_frames // 2].flatten(0, 1)),
                                             indices=list(enc["dinov3"]["layers"]), output_fmt="NLC",
                                             intermediates_only=True)  # fmt: skip
        fv = torch.stack([pool_tokens(h, grid_v, pool_v) for h in store], dim=1)  # [B*2, L, n, D]
        fv = fv.reshape(bsz, 2, *fv.shape[1:])
        fi = torch.stack([pool_tokens(h, grid_d, pool_d) for h in inter], dim=1)
        fi = fi.reshape(bsz, 2, *fi.shape[1:])
        out = {"video": fv.half().cpu().numpy(), "vmean": fv.mean(dim=3).half().cpu().numpy(),
               "image": fi.half().cpu().numpy()}  # fmt: skip
        for k, i in enumerate(idx.tolist()):
            j, w = owner[i]
            kk = int(todo["K"].iloc[j])
            a = acc.setdefault(j, {n: [None] * kk for n in out})
            for n in out:
                a[n][w] = out[n][k]
            if all(v is not None for v in a["video"]):
                c, cid = todo["camera"].iloc[j], todo["clip_id"].iloc[j]
                for n in out:
                    _save(_feature_path(cache, c, cid, n), np.stack(a[n]))
                del acc[j]
                n_done += 1
        if b % 250 == 0:
            log(
                f"  windows: {min((b + 1) * 4, len(items)):,}/{len(items):,} ({n_done:,} clips), {time.time() - t0:.0f} s"
            )
    for h in hooks:
        h.remove()
    del vj, dn
    torch.cuda.empty_cache()
    log(f"window features of {len(todo):,} clips written ({time.time() - t0:.0f} s)")


class WindowRows(torch.utils.data.Dataset):
    """Cached window features read in batches: an item is an array of global window ids."""

    def __init__(self, paths: list[str], clip_of: np.ndarray, w_of: np.ndarray) -> None:
        self.paths, self.clip_of, self.w_of = paths, clip_of, w_of
        self.shape = tuple(np.load(paths[0], mmap_mode="r").shape[1:])
        self._mm: dict[int, np.ndarray] = {}

    def __len__(self) -> int:
        return len(self.clip_of)

    def __getitem__(self, ids: np.ndarray) -> tuple[torch.Tensor, torch.Tensor]:
        out = np.empty((len(ids), *self.shape), np.float16)
        for k, g in enumerate(ids):
            c = int(self.clip_of[g])
            mm = self._mm.get(c)
            if mm is None:
                mm = self._mm[c] = np.load(self.paths[c], mmap_mode="r")
            out[k] = mm[self.w_of[g]]
        return torch.from_numpy(out), torch.as_tensor(ids)


def _val_split(
    fold_days: np.ndarray, trall: np.ndarray, f: int, seed: int
) -> tuple[np.ndarray, np.ndarray]:
    """The inner validation of ``out_of_fold``: one training date (or 15 % of clips in smoke tests)."""
    tr_days = np.array(sorted(set(fold_days[trall])))
    if len(tr_days) >= 3:
        vday = np.random.default_rng(1000 * f + int(seed)).choice(tr_days)
        return trall[fold_days[trall] == vday], trall[fold_days[trall] != vday]
    perm = np.random.default_rng(int(seed)).permutation(trall)
    return perm[: max(1, len(perm) // 7)], perm[max(1, len(perm) // 7) :]


class _Epoch(torch.utils.data.Sampler):
    """A sampler whose batches are set before each pass, so one loader's workers persist."""

    def __init__(self) -> None:
        self.batches: list[np.ndarray] = []

    def __iter__(self):  # type: ignore[no-untyped-def]
        return iter(self.batches)

    def __len__(self) -> int:
        return len(self.batches)


def _loader(rows: WindowRows, sampler: _Epoch, workers: int) -> torch.utils.data.DataLoader:
    return torch.utils.data.DataLoader(rows, batch_size=None, sampler=sampler, num_workers=workers,
                                       pin_memory=True, persistent_workers=workers > 0)  # fmt: skip


def _predict_chunked(
    probes: StackedProbes, n_h: int, x: torch.Tensor, chunk: int = 64
) -> torch.Tensor:
    """All probes on windows ``x``, ``chunk`` windows at a time (bounds the stacked activations)."""
    return torch.cat(
        [probes.predict(list(range(n_h)), x[s : s + chunk]) for s in range(0, len(x), chunk)], dim=1
    )


def window_probes_oof(rows: WindowRows, clip_of: np.ndarray, clips: pd.DataFrame, cfg: dict,
                      dev: torch.device, workers: int, name: str) -> np.ndarray:  # fmt: skip
    """Out-of-fold predictions (natural units) for every window, seeds averaged in log space.

    The 25 (fold, seed) attentive probes train in lock step (``StackedProbes``) on one
    stream of read batches: each takes its own training windows from every batch, in its
    own order, and steps whenever it holds a full mini-batch; the epoch's remainder is its
    last, partial mini-batch. Matmuls in TF32.
    """
    hp = cfg["heads"]
    bs, epochs = int(hp["batch_size"]), int(hp["epochs"])
    ylog = np.log(clips[[TARGETS[t][0] for t in TARGETS]].to_numpy(dtype=float))
    fold, days = clips["fold"].to_numpy(), clips["local_day"].to_numpy()
    n_w = len(clip_of)
    p, n_layers, t, dim = rows.shape
    ywin = torch.as_tensor(ylog[clip_of], dtype=torch.float32, device=dev)
    models, info = [], []
    for f in sorted(set(fold)):
        trall = np.flatnonzero(fold != f)
        mu, sd = ylog[trall].mean(0), ylog[trall].std(0) + 1e-6
        for seed in hp["seeds"]:
            va_c, tr_c = _val_split(days, trall, f, int(seed))
            torch.manual_seed(int(seed))
            models.append(AttentiveProbe(n_layers, dim, p * t, ylog.shape[1], d=int(hp["d"]), heads=int(hp["heads"]),
                                         dropout=float(hp["dropout"])).to(dev))  # fmt: skip
            tr = np.isin(clip_of, tr_c)
            info.append({"f": f, "seed": int(seed), "mu": mu, "sd": sd, "tr": tr, "va": np.isin(clip_of, va_c),
                         "rng": np.random.default_rng(int(seed)), "best": np.inf,
                         "steps": epochs * max(1, int(np.ceil(tr.sum() / bs)))})  # fmt: skip
    n_h = len(info)
    probes = StackedProbes(models, lr=float(hp["lr"]), weight_decay=float(hp["weight_decay"]),
                           total_steps=[h["steps"] for h in info])  # fmt: skip
    del models
    best = [probes.state(k) for k in range(n_h)]
    mu_t = torch.as_tensor(np.stack([h["mu"] for h in info]), dtype=torch.float32, device=dev)[
        :, None
    ]  # [H, 1, n]
    sd_t = torch.as_tensor(np.stack([h["sd"] for h in info]), dtype=torch.float32, device=dev)[
        :, None
    ]
    tr_m = np.stack([h["tr"] for h in info])  # [H, windows]
    va_m = np.stack([h["va"] for h in info])
    train_ids, val_ids = np.flatnonzero(tr_m.any(0)), np.flatnonzero(va_m.any(0))
    smp_tr, smp_va = _Epoch(), _Epoch()
    smp_va.batches = [val_ids[s : s + READ_BATCH] for s in range(0, len(val_ids), READ_BATCH)]
    ld_tr, ld_va = _loader(rows, smp_tr, min(workers, 16)), _loader(rows, smp_va, min(workers, 8))
    log(
        f"{name}: {n_h} probes on {n_w:,} windows of {len(clips):,} clips; {len(train_ids):,} train in some fold"
    )
    tf32 = torch.backends.cuda.matmul.allow_tf32
    torch.backends.cuda.matmul.allow_tf32 = True
    t0, n_steps = time.time(), 0
    for ep in range(epochs):
        perm = np.random.default_rng(10_000 + ep).permutation(train_ids)
        smp_tr.batches = [
            np.sort(perm[s : s + READ_BATCH]) for s in range(0, len(perm), READ_BATCH)
        ]
        used = np.zeros(n_h, dtype=np.int64)  # per probe: training windows stepped on
        carry_x: list[torch.Tensor | None] = [None] * n_h  # rows of a mini-batch not yet full
        carry_y: list[torch.Tensor | None] = [None] * n_h
        for xb, ids in ld_tr:
            ids_np = ids.numpy()
            xb, yb = xb.to(dev, non_blocking=True), ywin[ids.to(dev, non_blocking=True)]
            seq, n_c = [], np.zeros(n_h, dtype=np.int64)
            for k, h in enumerate(info):
                sel = np.flatnonzero(h["tr"][ids_np])
                seq.append(torch.as_tensor(sel[h["rng"].permutation(len(sel))], device=dev))
                n_c[k] = 0 if carry_x[k] is None else len(carry_x[k])
            n_full = (n_c + np.array([len(s) for s in seq])) // bs
            for j in range(int(n_full.max())):
                ready = np.flatnonzero(n_full > j)
                xs, ys = [], []
                for k in ready:
                    a, b = j * bs - n_c[k], (j + 1) * bs - n_c[k]  # positions in this batch's rows
                    if a < 0:
                        xs.append(torch.cat([carry_x[k], xb[seq[k][:b]]]))
                        ys.append(torch.cat([carry_y[k], yb[seq[k][:b]]]))
                    else:
                        xs.append(xb[seq[k][a:b]])
                        ys.append(yb[seq[k][a:b]])
                ri = torch.as_tensor(ready, device=dev)
                probes.step(
                    ready.tolist(), torch.stack(xs), (torch.stack(ys) - mu_t[ri]) / sd_t[ri]
                )
                used[ready] += bs
                n_steps += 1
            for k in range(n_h):
                start = n_full[k] * bs - n_c[k]
                if start < 0:  # no full mini-batch: the carry grows
                    carry_x[k] = (
                        xb[seq[k]] if carry_x[k] is None else torch.cat([carry_x[k], xb[seq[k]]])
                    )
                    carry_y[k] = (
                        yb[seq[k]] if carry_y[k] is None else torch.cat([carry_y[k], yb[seq[k]]])
                    )
                else:
                    rest = seq[k][start:]
                    carry_x[k], carry_y[k] = (xb[rest], yb[rest]) if len(rest) else (None, None)
        for k in range(n_h):  # the epoch's last, partial mini-batch of each probe
            if carry_x[k] is not None and len(carry_x[k]):
                probes.step([k], carry_x[k][None], ((carry_y[k] - mu_t[k]) / sd_t[k])[None])
                used[k] += len(carry_x[k])
        if not (used == tr_m.sum(1)).all():  # every probe saw each of its training windows once
            raise RuntimeError(f"{name}: windows used {used} != training windows {tr_m.sum(1)}")
        sse = torch.zeros(n_h, device=dev)
        cnt = np.zeros(n_h)
        for xb, ids in ld_va:
            ids_np = ids.numpy()
            mk = va_m[:, ids_np]  # [H, R]
            if not mk.any():
                continue
            out = _predict_chunked(probes, n_h, xb.to(dev, non_blocking=True))  # [H, R, n]
            y = (ywin[ids.to(dev)][None] - mu_t) / sd_t
            m_t = torch.as_tensor(mk, device=dev, dtype=torch.float32)[..., None]
            sse += (((out - y) ** 2) * m_t).sum(dim=(1, 2))
            cnt += mk.sum(1) * ylog.shape[1]
        v = sse.cpu().numpy() / np.maximum(cnt, 1)
        for k, h in enumerate(info):
            if v[k] < h["best"]:
                h["best"], best[k] = v[k], probes.state(k)
        if ep % 10 == 0 or ep == epochs - 1:
            log(f"  {name} epoch {ep + 1}/{epochs}: validation MSE {v.mean():.3f} (mean of probes), "
                f"{n_steps:,} lock steps, {time.time() - t0:.0f} s")  # fmt: skip
    if not (probes.t == np.array([h["steps"] for h in info])).all():
        raise RuntimeError(f"{name}: steps {probes.t} differ from the schedules' totals")
    for k in range(n_h):
        probes.load(k, best[k])
    pred = np.zeros((n_w, ylog.shape[1]))
    smp_all = _Epoch()
    smp_all.batches = [np.arange(s, min(n_w, s + READ_BATCH)) for s in range(0, n_w, READ_BATCH)]
    hf = np.array([h["f"] for h in info])
    with torch.no_grad():
        for xb, ids in _loader(rows, smp_all, min(workers, 8)):
            ids_np = ids.numpy()
            out = (
                _predict_chunked(probes, n_h, xb.to(dev, non_blocking=True)) * sd_t + mu_t
            )  # [H, R, n]
            wgt = (hf[:, None] == fold[clip_of[ids_np]][None]).astype(
                np.float32
            )  # each window: its fold's probes
            wgt /= wgt.sum(0, keepdims=True)
            pred[ids_np] = (
                torch.einsum("hr,hrn->rn", torch.as_tensor(wgt, device=dev), out).cpu().numpy()
            )
    torch.backends.cuda.matmul.allow_tf32 = tf32
    return np.exp(pred)


def fit_pooler(x: torch.Tensor, mask: torch.Tensor, y: torch.Tensor, tr: np.ndarray, va: np.ndarray,
               hp: dict, seed: int) -> nn.Module:  # fmt: skip
    """``fit_head`` for the record pooler (inputs with a window mask)."""
    torch.manual_seed(seed)
    _, w, p, n_layers, dim = x.shape
    head = RecordPooler(n_layers, dim, y.shape[1], max_windows=w, n_tiles=p, d=int(hp["d"]),
                        heads=int(hp["heads"]), dropout=float(hp["dropout"])).to(x.device)  # fmt: skip
    opt = torch.optim.AdamW(
        head.parameters(), lr=float(hp["lr"]), weight_decay=float(hp["weight_decay"])
    )
    epochs, bs = int(hp["epochs"]), int(hp["batch_size"])
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=float(hp["lr"]),
                                                total_steps=epochs * max(1, int(np.ceil(len(tr) / bs))), pct_start=0.1)  # fmt: skip
    rng = np.random.default_rng(seed)
    best, best_state = np.inf, copy.deepcopy(head.state_dict())
    vi = torch.as_tensor(va, device=x.device)
    for _ in range(epochs):
        head.train()
        order = tr[rng.permutation(len(tr))]
        for s in range(0, len(order), bs):
            b = torch.as_tensor(order[s : s + bs], device=x.device)
            loss = nn.functional.smooth_l1_loss(head(x[b], mask[b]), y[b])
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            sched.step()
        head.eval()
        with torch.no_grad():
            v = float(nn.functional.mse_loss(head(x[vi], mask[vi]), y[vi]))
        if v < best:
            best, best_state = v, copy.deepcopy(head.state_dict())
    head.load_state_dict(best_state)
    return head.eval()


def pooled_oof(
    vmean: np.ndarray, mask: np.ndarray, clips: pd.DataFrame, cfg: dict, dev: torch.device
) -> np.ndarray:
    """``out_of_fold`` for the record pooler on each clip's window mean tokens."""
    hp = cfg["heads"]
    x = torch.from_numpy(vmean).to(dev)
    mk = torch.from_numpy(mask).to(dev)
    ylog = np.log(clips[[TARGETS[t][0] for t in TARGETS]].to_numpy(dtype=float))
    fold, days = clips["fold"].to_numpy(), clips["local_day"].to_numpy()
    pred = np.zeros_like(ylog)
    for f in sorted(set(fold)):
        te, trall = np.flatnonzero(fold == f), np.flatnonzero(fold != f)
        mu, sd = ylog[trall].mean(0), ylog[trall].std(0) + 1e-6
        y = torch.as_tensor((ylog - mu) / sd, dtype=torch.float32, device=dev)
        acc = np.zeros((len(te), ylog.shape[1]))
        for seed in hp["seeds"]:
            va, tr = _val_split(days, trall, f, int(seed))
            head = fit_pooler(x, mk, y, tr, va, hp, int(seed))
            with torch.no_grad():
                ti = torch.as_tensor(te, device=dev)
                acc += head(x[ti], mk[ti]).cpu().numpy() * sd + mu
        pred[te] = acc / len(hp["seeds"])
    return np.exp(pred)


RIDGE_ALPHA = 1.0  # on standardised inputs; recipe fixed before scoring


def ridge_oof(
    x: np.ndarray, y: np.ndarray, fold: np.ndarray, alpha: float = RIDGE_ALPHA
) -> np.ndarray:
    """Out-of-fold ridge predictions of ``y`` from ``x`` [clips, features] over the same folds.

    Inputs standardised on each training fold (a missing value becomes the training
    mean), intercept not penalised.
    """
    pred = np.full(len(y), np.nan)
    for f in sorted(set(fold)):
        te, tr = fold == f, fold != f
        mu = np.nanmean(x[tr], axis=0)
        sd = np.nanstd(x[tr], axis=0)
        sd = np.where(sd > 0, sd, 1.0)
        z = np.nan_to_num((x - mu) / sd, nan=0.0)
        ym = y[tr].mean()
        a = z[tr].T @ z[tr] + alpha * np.eye(z.shape[1])
        beta = np.linalg.solve(a, z[tr].T @ (y[tr] - ym))
        pred[te] = z[te] @ beta + ym
    return pred


# --- Physics features v2: bars M6-M7 -------------------------------------------------
# docs/experiments/webcoos_video.md, "Physics features v2". Label-free spectrum (recipe 2)
# and breaking features per clip (the package's physics_v2.csv), fixed on the development
# dates; the bars are scored on the confirmation dates only (all final dates outside
# configs/experiments/webcoos_video_dev_dates.txt).

PHYSICS_FILE = "physics_v2.csv"
V2_SPECTRUM = [f"v2_{k}" for k in FEATURES_V2]
V2_BREAKING = [f"v2_{k}" for k in BREAKING_FEATURES]
PHYS_TARGETS = {"hs": ("Hs_label_m", "m"), "tp": ("Tp_label_s", "s"), "te": ("Te_label_s", "s")}
PHYS_ARMS = {  # target: arms
    "tp": ["spectrum", "spectrum_v2", "pixel_raw", "pixel_raw_v2", "clip", "clip_ridge", "clip_spectrum_v2",
           "spectrum_cam", "spectrum_v2_cam"],
    "te": ["spectrum", "spectrum_v2", "pixel_raw", "pixel_raw_v2", "clip_ridge", "clip_spectrum_v2"],
    "hs": ["clip", "clip_refit", "clip_breaking", "clip_refit_cam", "clip_breaking_cam"],
}  # fmt: skip
PHYS_DIFFS = [  # (first, second, target): RMSE of first minus second
    ("spectrum_v2", "spectrum", "tp"), ("clip_breaking", "clip_refit", "hs"),  # M6, M7
    ("pixel_raw_v2", "pixel_raw", "tp"), ("clip_spectrum_v2", "clip_ridge", "tp"),
    ("spectrum_v2_cam", "spectrum_cam", "tp"), ("clip_breaking_cam", "clip_refit_cam", "hs"),
    ("clip_refit", "clip", "hs"), ("spectrum_v2", "spectrum", "te"), ("pixel_raw_v2", "pixel_raw", "te"),
    ("clip_spectrum_v2", "clip_ridge", "te"),
]  # fmt: skip
PHYS_BARS = {
    "M6 (T_p)": ("spectrum_v2", "spectrum", "tp"),
    "M7 (H_s)": ("clip_breaking", "clip_refit", "hs"),
}


def _ridge_finite(x: np.ndarray, y: np.ndarray, fold: np.ndarray) -> np.ndarray:
    """``ridge_oof`` on the clips with a finite target (the others get NaN)."""
    ok = np.isfinite(y)
    out = np.full(len(y), np.nan)
    out[ok] = ridge_oof(x[ok], y[ok], fold[ok])
    return out


def physics_v2_arms(d: pd.DataFrame) -> pd.DataFrame:
    """Add the M6-M7 arms (``pred_<hs|tp|te>_<arm>``) to the whole-clip predictions ``d``.

    Ridge heads (``ridge_oof``: standardised inputs, alpha 1, the run's folds of dates) on
    log targets. T_p and T_e: ``spectrum`` (recipe v1's ten features), ``spectrum_v2``
    (v1's ten and the four recipe v2 adds), ``clip_ridge`` (log ``clip`` T_p alone),
    ``clip_spectrum_v2`` (log ``clip`` T_p and the fourteen); untrained ``pixel_raw``
    (v1's T_pix) and ``pixel_raw_v2`` (v2's T_pix, the peak of the cleaned spectrum). H_s: ``clip_refit`` (log ``clip`` H_s alone) and ``clip_breaking``
    (log ``clip`` H_s and the breaking features). ``_cam``: the same with a camera
    intercept (one indicator per camera).
    """
    d = d.copy()
    fold = d["fold"].to_numpy()
    cam = pd.get_dummies(d["camera"]).to_numpy(dtype=float)
    spec1 = d[list(SPECTRUM_FEATURES)].to_numpy(float)
    spec2 = np.hstack([spec1, d[V2_SPECTRUM].to_numpy(float)])
    brk = d[V2_BREAKING].to_numpy(float)
    ltp = np.log(d["pred_tp_clip"].to_numpy(float))[:, None]
    lhs = np.log(d["pred_hs_clip"].to_numpy(float))[:, None]
    for t in ("tp", "te"):
        lab = PHYS_TARGETS[t][0]
        if lab not in d:
            continue
        y = np.log(d[lab].to_numpy(float))
        for arm, x in (("spectrum", spec1), ("spectrum_v2", spec2), ("clip_ridge", ltp),
                       ("clip_spectrum_v2", np.hstack([ltp, spec2])), ("spectrum_cam", np.hstack([spec1, cam])),
                       ("spectrum_v2_cam", np.hstack([spec2, cam]))):  # fmt: skip
            if arm in PHYS_ARMS[t]:
                d[f"pred_{t}_{arm}"] = np.exp(_ridge_finite(x, y, fold))
        d[f"pred_{t}_pixel_raw"] = d["T_pix_s"].to_numpy(float)
        d[f"pred_{t}_pixel_raw_v2"] = d["v2_T_pix_s"].to_numpy(float)
    y = np.log(d["Hs_label_m"].to_numpy(float))
    for arm, x in (("clip_refit", lhs), ("clip_breaking", np.hstack([lhs, brk])),
                   ("clip_refit_cam", np.hstack([lhs, cam])), ("clip_breaking_cam", np.hstack([lhs, brk, cam]))):  # fmt: skip
        d[f"pred_hs_{arm}"] = np.exp(_ridge_finite(x, y, fold))
    return d


def _phys_scores_on(d: pd.DataFrame, cfg: dict) -> dict:
    b = cfg["evaluation"]["bootstrap"]
    kw = {"n_boot": int(b["n_boot"]), "alpha": float(b["alpha"]), "seed": int(b["seed"])}
    res: dict[str, Any] = {"n_clips": len(d), "n_dates": int(d["local_day"].nunique()),
                           "by_camera_n": d["camera"].value_counts().to_dict(), "arms": {}, "differences": {}}  # fmt: skip
    for t, arms in PHYS_ARMS.items():
        lab, unit = PHYS_TARGETS[t]
        if lab not in d:
            continue
        for a in arms:
            if f"pred_{t}_{a}" not in d:
                continue
            ok = np.isfinite(d[lab].to_numpy(float)) & np.isfinite(
                d[f"pred_{t}_{a}"].to_numpy(float)
            )
            g = d[ok]
            y, p = g[lab].to_numpy(float), g[f"pred_{t}_{a}"].to_numpy(float)
            r = {"n": int(ok.sum()), f"rmse_{unit}": rmse(y, p), "r2": r2(y, p), "pearson": pearson(y, p),
                 f"bias_{unit}": float(np.mean(p - y)),
                 "by_camera_rmse": {c: rmse(gg[lab], gg[f"pred_{t}_{a}"]) for c, gg in g.groupby("camera")}}  # fmt: skip
            oak = g[g["camera"].str.startswith("oakisland")]
            if len(oak) > 2:
                r["oak_island_rmse"] = rmse(oak[lab], oak[f"pred_{t}_{a}"])
            res["arms"].setdefault(t, {})[a] = r
    for a, o, t in PHYS_DIFFS:
        lab = PHYS_TARGETS[t][0]
        if lab not in d or f"pred_{t}_{a}" not in d or f"pred_{t}_{o}" not in d:
            continue
        ok = np.isfinite(d[[lab, f"pred_{t}_{a}", f"pred_{t}_{o}"]].to_numpy(float)).all(axis=1)
        g = d[ok]
        if g["local_day"].nunique() < 2:
            continue
        v, lo, hi = paired_block_bootstrap_ci(g[lab].to_numpy(), g[f"pred_{t}_{a}"], g[f"pred_{t}_{o}"], rmse,
                                              groups=g["local_day"].to_numpy(), **kw)  # fmt: skip
        res["differences"][f"{a} minus {o}: {t}"] = [float(v), float(lo), float(hi)]
    return res


def dev_dates(cfg: dict) -> tuple[set[str], str]:
    """The development dates of the physics-v2 bars, checked against their committed SHA-256."""
    pc = cfg["evaluation"]["physics_v2"]
    path = Path(__file__).resolve().parents[1] / pc["dev_dates"]
    raw = path.read_bytes()
    sha = hashlib.sha256(raw).hexdigest()
    if sha != pc["dev_dates_sha256"]:
        raise SystemExit(f"{path} has SHA-256 {sha}, not the committed {pc['dev_dates_sha256']}")
    return set(raw.decode().split()), sha


def physics_scores(d: pd.DataFrame, cfg: dict) -> dict:
    """M6-M7 on the confirmation dates (bars, if there are enough) and on the development dates (labelled so)."""
    dev, sha = dev_dates(cfg)
    need = int(cfg["evaluation"]["physics_v2"]["min_confirmation_dates"])
    conf, devd = d[~d["local_day"].isin(dev)], d[d["local_day"].isin(dev)]
    res: dict[str, Any] = {"dev_dates_sha256": sha, "min_confirmation_dates": need,
                           "development": _phys_scores_on(devd, cfg) if len(set(devd["local_day"])) > 1 else None,
                           "confirmation": _phys_scores_on(conf, cfg) if len(set(conf["local_day"])) > 1 else None}  # fmt: skip
    n_conf = int(conf["local_day"].nunique())
    if res["confirmation"] and n_conf >= need:
        res["bars"] = {k: bool(res["confirmation"]["differences"][f"{a} minus {o}: {t}"][2] < 0)
                       for k, (a, o, t) in PHYS_BARS.items()}  # fmt: skip
    else:
        res["bars"] = dict.fromkeys(
            PHYS_BARS, f"not scored: {n_conf} confirmation dates, {need} needed"
        )
    return res


def physics_report(res: dict) -> str:
    lines = []
    for key, title in (
        ("confirmation", "confirmation dates (the bars)"),
        ("development", "development dates (used to fix v2; not a test)"),
    ):
        r = res.get(key)
        if not r:
            continue
        lines += [f"\n## Physics v2, {title}: {r['n_clips']:,} clips, {r['n_dates']} dates\n",
                  "| target | arm | RMSE | R² | Pearson | bias | Oak Island RMSE |", "|---|---|---|---|---|---|---|"]  # fmt: skip
        for t, arms in r["arms"].items():
            unit = PHYS_TARGETS[t][1]
            for a, v in arms.items():
                lines.append(f"| {t} | {a} | {v[f'rmse_{unit}']:.3f} | {v['r2']:.3f} | {v['pearson']:.3f} | "
                             f"{v[f'bias_{unit}']:+.3f} | {v.get('oak_island_rmse', float('nan')):.3f} |")  # fmt: skip
        lines += [
            "",
            "Differences in RMSE (first minus second; 95 % interval, bootstrap over dates):",
            "",
        ]
        lines += [
            f"- {k}: {v[0]:+.3f} [{v[1]:+.3f}, {v[2]:+.3f}]" for k, v in r["differences"].items()
        ]
    lines += ["", "Bars M6-M7 (confirmation dates only): "
              + ", ".join(f"{k} {'met' if v is True else 'not met' if v is False else v}" for k, v in res["bars"].items())]  # fmt: skip
    return "\n".join(lines) + "\n"


def whole_clip(
    args: argparse.Namespace, cfg: dict, clips: pd.DataFrame, dev: torch.device
) -> pd.DataFrame:
    """Features, probes and readouts of the whole-clip arms: one row per clip, ``pred_<t>_<arm>`` columns."""
    cache = window_cache_dir(cfg, args.window_cache)
    if not args.no_extract:
        extract_windows(args, cfg, clips, cache)
    k = clips["K"].to_numpy().astype(int)
    clip_of = np.repeat(np.arange(len(clips)), k)
    w_of = np.concatenate([np.arange(n) for n in k])
    start = np.concatenate([[0], np.cumsum(k)[:-1]])
    d = clips.copy()
    mid = np.array([midpoint_window(dur, n) for dur, n in zip(clips["duration_s"], k, strict=True)])
    span = [centred_windows(m, n) for m, n in zip(mid, k, strict=True)]
    d["midpoint_window"] = mid
    for arm, enc in WINDOW_ARMS.items():
        paths = [
            str(_feature_path(cache, c, i, arm))
            for c, i in zip(clips["camera"], clips["clip_id"], strict=True)
        ]
        rows = WindowRows(paths, clip_of, w_of)
        t0 = time.time()
        pw = window_probes_oof(
            rows, clip_of, clips, cfg, dev, args.num_workers, f"{arm} windows ({enc})"
        )
        np.save(args.out / f"window_predictions_{arm}.npy", pw)
        sfx = "" if arm == "video" else "_image"
        hs, tp = pw[:, 0], pw[:, 1]
        per = [slice(s, s + n) for s, n in zip(start, k, strict=True)]
        d[f"pred_hs_window{sfx}"] = hs[start + mid]
        d[f"pred_tp_window{sfx}"] = tp[start + mid]
        d[f"pred_hs_clip{sfx}"] = [float(energy_mean(hs[q])) for q in per]
        d[f"pred_tp_clip{sfx}"] = [float(geometric_mean(tp[q])) for q in per]
        if arm == "video":
            d["pred_hs_clip_meanlog"] = [float(geometric_mean(hs[q])) for q in per]
            d["pred_tp_clip_meanlog"] = d["pred_tp_clip"]
            d["pred_hs_span11"] = [
                float(energy_mean(hs[s + a : s + b])) for s, (a, b) in zip(start, span, strict=True)
            ]
            d["pred_tp_span11"] = [
                float(geometric_mean(tp[s + a : s + b]))
                for s, (a, b) in zip(start, span, strict=True)
            ]
        log(f"{arm} windows: clip H_s RMSE {rmse(d['Hs_label_m'], d[f'pred_hs_clip{sfx}']):.3f} m, window "
            f"{rmse(d['Hs_label_m'], d[f'pred_hs_window{sfx}']):.3f} m; clip T_p RMSE "
            f"{rmse(d['Tp_label_s'], d[f'pred_tp_clip{sfx}']):.2f} s, window {rmse(d['Tp_label_s'], d[f'pred_tp_window{sfx}']):.2f} s "
            f"({time.time() - t0:.0f} s)")  # fmt: skip
    # pooled: the record pooler on each window's mean video tokens
    vm = [
        np.load(_feature_path(cache, c, i, "vmean"))
        for c, i in zip(clips["camera"], clips["clip_id"], strict=True)
    ]
    kmax = int(k.max())
    x = np.zeros((len(clips), kmax, *vm[0].shape[1:]), np.float16)
    mask = np.zeros((len(clips), kmax), bool)
    for j, a in enumerate(vm):
        x[j, : len(a)], mask[j, : len(a)] = a, True
    pp = pooled_oof(x, mask, clips, cfg, dev)
    d["pred_hs_pooled"], d["pred_tp_pooled"] = pp[:, 0], pp[:, 1]
    # ridge heads on the pixel spectrum (T_p only)
    fold, ytp = clips["fold"].to_numpy(), np.log(clips["Tp_label_s"].to_numpy(dtype=float))
    spec = clips[list(SPECTRUM_FEATURES)].to_numpy(dtype=float)
    lclip = np.log(d["pred_tp_clip"].to_numpy(dtype=float))[:, None]
    for arm, xx in (
        ("spectrum", spec),
        ("clip_spectrum", np.hstack([lclip, spec])),
        ("clip_ridge", lclip),
    ):
        d[f"pred_tp_{arm}"] = np.exp(ridge_oof(xx, ytp, fold))
        d[f"pred_hs_{arm}"] = np.nan
    d["pred_tp_pixel_raw"], d["pred_hs_pixel_raw"] = clips["T_pix_s"].to_numpy(dtype=float), np.nan
    if V2_SPECTRUM[0] in clips.columns and "physics_v2" in cfg["evaluation"]:
        d = physics_v2_arms(d)
    return d


WHOLE_ARMS = ["window", "clip", "clip_meanlog", "span11", "pooled", "window_image", "clip_image",
              "spectrum", "clip_spectrum", "clip_ridge", "pixel_raw"]  # fmt: skip
WHOLE_DIFFS = [  # (first, second, target): RMSE of first minus second
    ("clip", "window", "hs"), ("clip", "window", "tp"), ("clip_spectrum", "clip", "tp"),  # M3, M4, M5
    ("clip_meanlog", "window", "hs"), ("clip_meanlog", "clip", "hs"), ("pooled", "clip", "hs"),
    ("pooled", "clip", "tp"), ("clip_image", "window_image", "hs"), ("clip_image", "window_image", "tp"),
    ("clip_image", "clip", "hs"), ("clip_image", "clip", "tp"), ("clip_ridge", "clip", "tp"),
    ("clip_spectrum", "clip_ridge", "tp"), ("spectrum", "clip", "tp"), ("span11", "window", "hs"),
]  # fmt: skip
WHOLE_BARS = {"M3 (H_s)": ("clip", "window", "hs"), "M4 (T_p)": ("clip", "window", "tp"),
              "M5 (T_p)": ("clip_spectrum", "clip", "tp")}  # fmt: skip


def _whole_scores_on(d: pd.DataFrame, cfg: dict) -> dict:
    b = cfg["evaluation"]["bootstrap"]
    kw = {"n_boot": int(b["n_boot"]), "alpha": float(b["alpha"]), "seed": int(b["seed"])}
    g = d["local_day"].to_numpy()
    res: dict[str, Any] = {"n_clips": len(d), "n_dates": int(d["local_day"].nunique()),
                           "n_windows": int(d["K"].sum()), "by_camera_n": d["camera"].value_counts().to_dict(),
                           "arms": {}, "differences": {}}  # fmt: skip
    for c in WHOLE_ARMS:
        r: dict[str, Any] = {}
        for t, (lab, unit) in TARGETS.items():
            p = d[f"pred_{t}_{c}"].to_numpy(dtype=float)
            if np.isnan(p).all():
                continue
            y = d[lab].to_numpy()
            r[t] = {f"rmse_{unit}": rmse(y, p), "r2": r2(y, p), "pearson": pearson(y, p), f"bias_{unit}": float(np.mean(p - y)),
                    "by_camera_rmse": {cam: rmse(gg[lab], gg[f"pred_{t}_{c}"]) for cam, gg in d.groupby("camera")}}  # fmt: skip
        if "hs" in r:
            r["kim2023_accuracy"], r["kim2023_n"] = kim_accuracy(
                d["Hs_label_m"].to_numpy(), d[f"pred_hs_{c}"].to_numpy()
            )
        oak = d[d["camera"].str.startswith("oakisland")]
        if len(oak) > 2:
            o: dict[str, Any] = {
                "n": len(oak),
                "tp_rmse_s": rmse(oak["Tp_label_s"], oak[f"pred_tp_{c}"]),
            }
            if "hs" in r:
                y, p = oak["Hs_label_m"].to_numpy(), oak[f"pred_hs_{c}"].to_numpy()
                o |= {"hs_rmse_m": rmse(y, p), "hs_r2": r2(y, p), "hs_pearson": pearson(y, p)}
            r["oak_island"] = o
        res["arms"][c] = r
    for a, o, t in WHOLE_DIFFS:
        lab = TARGETS[t][0]
        v, lo, hi = paired_block_bootstrap_ci(
            d[lab].to_numpy(), d[f"pred_{t}_{a}"], d[f"pred_{t}_{o}"], rmse, groups=g, **kw
        )
        res["differences"][f"{a} minus {o}: {t}"] = [float(v), float(lo), float(hi)]
    res["bars"] = {
        k: bool(res["differences"][f"{a} minus {o}: {t}"][2] < 0)
        for k, (a, o, t) in WHOLE_BARS.items()
    }
    return res


def whole_clip_scores(d: pd.DataFrame, cfg: dict) -> dict:
    """Scores of the whole-clip arms on all clips and on the clips with K >= 15."""
    long = d[d["K"] >= 15]
    return {"all": _whole_scores_on(d, cfg), "k_ge_15": _whole_scores_on(long, cfg) if len(set(long["local_day"])) > 1 else None,
            "kim2023_windows": {"1 (midpoint)": "window", "11 (176 s)": "span11", "K (whole clip)": "clip"},
            "ridge_alpha": RIDGE_ALPHA, "spectrum_features": list(SPECTRUM_FEATURES)}  # fmt: skip


def whole_clip_report(res: dict) -> str:
    lines = []
    for key, title in (("all", "all clips"), ("k_ge_15", "clips with K >= 15")):
        r = res.get(key)
        if not r:
            continue
        lines += [f"\n## Whole clips, {title}: {r['n_clips']:,} clips, {r['n_windows']:,} windows, {r['n_dates']} dates\n",
                  "| arm | H_s RMSE (m) | H_s R² | H_s Pearson | T_p RMSE (s) | T_p R² | T_p Pearson | Kim 5-class acc. |",
                  "|---|---|---|---|---|---|---|---|"]  # fmt: skip
        for c, a in r["arms"].items():
            h = a.get("hs")
            hs = f"{h['rmse_m']:.3f} | {h['r2']:.3f} | {h['pearson']:.3f}" if h else "— | — | —"
            kim = f"{a['kim2023_accuracy']:.3f} (n {a['kim2023_n']})" if h else "—"
            lines.append(
                f"| {c} | {hs} | {a['tp']['rmse_s']:.2f} | {a['tp']['r2']:.3f} | {a['tp']['pearson']:.3f} | {kim} |"
            )
        lines += [
            "",
            "Differences in RMSE (first minus second; 95 % interval, bootstrap over dates):",
            "",
        ]
        lines += [
            f"- {k}: {v[0]:+.3f} [{v[1]:+.3f}, {v[2]:+.3f}]" for k, v in r["differences"].items()
        ]
        lines += [
            "",
            "Bars: " + ", ".join(f"{k} {'met' if v else 'not met'}" for k, v in r["bars"].items()),
        ]
        oak = {c: a["oak_island"] for c, a in r["arms"].items() if "oak_island" in a}
        if oak:
            lines += ["", f"Oak Island T_p RMSE (Kamagata 2026 held-out: {KAMAGATA['tp_rmse_s']} s): "
                      + ", ".join(f"{c} {o['tp_rmse_s']:.2f} s" for c, o in oak.items())]  # fmt: skip
    return "\n".join(lines) + "\n"


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--config", type=Path, required=True)
    ap.add_argument("--package", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument(
        "--split", default="days", help="accepted for the runner; folds are always by date"
    )
    ap.add_argument(
        "--amp",
        default="bf16",
        help="accepted for the runner; features always in bfloat16 on a GPU",
    )
    ap.add_argument("--num-workers", type=int, default=8)
    ap.add_argument(
        "--features-from", type=Path, help="reuse cached features (a run's features/ folder)"
    )
    ap.add_argument("--limit", type=int, help="first N clips only (smoke test)")
    ap.add_argument("--epochs", type=int, help="override the config's head epochs (smoke test)")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument(
        "--windows", type=Path, help="whole-clip windows, <camera>/<clip>.zip (bars M3-M5)"
    )
    ap.add_argument(
        "--window-cache",
        type=Path,
        help="per-clip window features (default <out>/features/windows)",
    )
    ap.add_argument("--no-extract", action="store_true", help="window features from the cache only")
    ap.add_argument("--whole-clip-only", action="store_true", help="skip the 64-frame arms (M1/M2)")
    ap.add_argument("--clip-arms-only", action="store_true",
                    help="with --windows: the same clips, but only the 64-frame arms (M1/M2), so the "
                         "two halves can run as two containers")  # fmt: skip
    args = ap.parse_args(argv)
    cfg = load_yaml(args.config)
    if args.epochs:
        cfg["heads"]["epochs"] = args.epochs
    args.out.mkdir(parents=True, exist_ok=True)

    clips = pd.read_csv(args.package / "clips.csv")
    clips = clips[(clips["sun_elevation_deg"] >= float(cfg["data"]["min_sun_elevation_deg"]))
                  & clips["Hs_label_m"].gt(0) & clips["Tp_label_s"].gt(0)].sort_values("clip_id").reset_index(drop=True)  # fmt: skip
    if args.windows:  # whole clips: the clips with windows, their K and pixel spectrum
        win = pd.read_csv(args.package / "windows.csv")[["clip_id", "K"]]
        spec = pd.read_csv(args.package / "pixel_spectrum.csv")[
            ["clip_id", "T_pix_s", *SPECTRUM_FEATURES]
        ]
        n0 = len(clips)
        clips = clips.merge(win, on="clip_id").merge(spec, on="clip_id")
        there = [
            (args.windows / c / f"{i}.zip").exists()
            for c, i in zip(clips["camera"], clips["clip_id"], strict=True)
        ]
        clips = clips[np.array(there, dtype=bool) & clips["K"].ge(1)].reset_index(drop=True)
        log(f"whole clips: {len(clips):,} of {n0:,} clips have windows in {args.windows}")
        if (args.package / PHYSICS_FILE).exists() and "physics_v2" in cfg["evaluation"]:
            phys = pd.read_csv(args.package / PHYSICS_FILE)
            keep = ["clip_id", *V2_SPECTRUM, *V2_BREAKING, "v2_T_pix_s"]
            clips = clips.merge(phys[keep], on="clip_id", how="left")  # the same clips
            log(
                f"physics v2: {int(clips[V2_SPECTRUM[0]].notna().sum()):,} of {len(clips):,} clips have features"
            )
        args.window_cache = args.window_cache or args.out / "features" / "windows"
    if args.limit:  # a smoke test: clips from as many dates as possible, so folds exist
        clips = (clips.assign(_k=clips.groupby("local_day").cumcount()).sort_values(["_k", "clip_id"])
                 .head(args.limit).drop(columns="_k").sort_values("clip_id").reset_index(drop=True))  # fmt: skip
    if clips["local_day"].nunique() < 2:
        raise SystemExit("clips from fewer than two dates: no fold can be held out")
    clips["fold"] = folds_by_date(clips["local_day"].to_numpy(), int(cfg["evaluation"]["folds"]))
    log(
        f"{len(clips):,} clips, {clips['local_day'].nunique()} dates, {clips['camera'].value_counts().to_dict()}"
    )

    dev = torch.device(args.device)
    d = clips.copy()
    res: dict[str, Any] = {}
    if not args.whole_clip_only:
        fdir = args.features_from or args.out / "features"
        if args.features_from:
            cached = pd.read_csv(fdir / "clips.csv")
            if list(cached["clip_id"]) != list(clips["clip_id"]):
                raise SystemExit(f"{fdir} holds features of other clips")
        else:
            extract(args, cfg, clips, fdir)
        arms = [a for e in cfg["encoders"].values() for a in e["arms"]]
        cols = []
        for arm in arms:
            feats = np.load(fdir / f"{arm}.npy", mmap_mode="r")
            for kind in ("attentive", "mean"):
                t0 = time.time()
                p = out_of_fold(kind, feats, clips, cfg, dev)
                c = f"{arm}_{kind}"
                cols.append(c)
                for j, t in enumerate(TARGETS):
                    d[f"pred_{t}_{c}"] = p[:, j]
                log(f"{c}: H_s RMSE {rmse(d['Hs_label_m'], d[f'pred_hs_{c}']):.3f} m, "
                    f"T_p RMSE {rmse(d['Tp_label_s'], d[f'pred_tp_{c}']):.2f} s ({time.time() - t0:.0f} s)")  # fmt: skip
        d.to_csv(args.out / "predictions.csv", index=False)
        res = scores(d, cols, cfg)
    if args.windows and not args.clip_arms_only:
        w = whole_clip(args, cfg, clips, dev)
        w.to_csv(args.out / "predictions_whole_clip.csv", index=False)
        res["whole_clip"] = whole_clip_scores(w, cfg)
        if V2_SPECTRUM[0] in w:
            res["physics_v2"] = physics_scores(w, cfg)
    res["run"] = {"config": str(args.config), "package": str(args.package), "limit": args.limit,
                  "features_from": str(args.features_from) if args.features_from else None,
                  "windows": str(args.windows) if args.windows else None,
                  "n_clips": len(clips), "clip_ids_sha256": hashlib.sha256("\n".join(clips["clip_id"]).encode()).hexdigest(),
                  "window_cache": str(window_cache_dir(cfg, args.window_cache)) if args.windows else None}  # fmt: skip
    (args.out / "metrics.json").write_text(json.dumps(res, indent=1, default=str))
    text = report(res, cfg["heads"]["primary"]) if "arms" in res else ""
    if "whole_clip" in res:
        text += whole_clip_report(res["whole_clip"])
    if "physics_v2" in res:
        text += physics_report(res["physics_v2"])
    (args.out / "report.md").write_text(text, encoding="utf-8")
    print(text)
    return 0


if __name__ == "__main__":
    sys.exit(main())
