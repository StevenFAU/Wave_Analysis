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

    python scripts/train_video_probe.py --config configs/experiments/webcoos_video_v1.yaml \\
        --package data/processed/packages/webcoos_video_v1 --out runs/<name> [--limit 16]
"""

from __future__ import annotations

import argparse
import copy
import json
import sys
import time
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
    arm_order,
    normalise,
    pool_tokens,
    square_tiles,
)

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
    args = ap.parse_args(argv)
    cfg = load_yaml(args.config)
    if args.epochs:
        cfg["heads"]["epochs"] = args.epochs
    args.out.mkdir(parents=True, exist_ok=True)

    clips = pd.read_csv(args.package / "clips.csv")
    clips = clips[(clips["sun_elevation_deg"] >= float(cfg["data"]["min_sun_elevation_deg"]))
                  & clips["Hs_label_m"].gt(0) & clips["Tp_label_s"].gt(0)].sort_values("clip_id").reset_index(drop=True)  # fmt: skip
    if args.limit:  # a smoke test: clips from as many dates as possible, so folds exist
        clips = (clips.assign(_k=clips.groupby("local_day").cumcount()).sort_values(["_k", "clip_id"])
                 .head(args.limit).drop(columns="_k").sort_values("clip_id").reset_index(drop=True))  # fmt: skip
    if clips["local_day"].nunique() < 2:
        raise SystemExit("clips from fewer than two dates: no fold can be held out")
    clips["fold"] = folds_by_date(clips["local_day"].to_numpy(), int(cfg["evaluation"]["folds"]))
    log(
        f"{len(clips):,} clips, {clips['local_day'].nunique()} dates, {clips['camera'].value_counts().to_dict()}"
    )

    fdir = args.features_from or args.out / "features"
    if args.features_from:
        cached = pd.read_csv(fdir / "clips.csv")
        if list(cached["clip_id"]) != list(clips["clip_id"]):
            raise SystemExit(f"{fdir} holds features of other clips")
    else:
        extract(args, cfg, clips, fdir)

    dev = torch.device(args.device)
    arms = [a for e in cfg["encoders"].values() for a in e["arms"]]
    d = clips.copy()
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
    res["run"] = {"config": str(args.config), "package": str(args.package), "limit": args.limit,
                  "features_from": str(args.features_from) if args.features_from else None}  # fmt: skip
    (args.out / "metrics.json").write_text(json.dumps(res, indent=1, default=str))
    text = report(res, cfg["heads"]["primary"])
    (args.out / "report.md").write_text(text, encoding="utf-8")
    print(text)
    return 0


if __name__ == "__main__":
    sys.exit(main())
