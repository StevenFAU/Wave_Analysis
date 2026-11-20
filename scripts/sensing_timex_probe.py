#!/usr/bin/env python
"""Test 9: frozen DINOv3 probes on a still, a time-exposure and a brightest-pixel image of the
same ten minutes, scored on a held-out year (GPU server).

Bar T9 and every setting: ``docs/experiments/sensing_timex_bars.md``; config
``configs/experiments/sensing_timex_v1.yaml``; package from ``scripts/sensing_timex_package.py``.

1. **Features** (once, cached in ``--feature-cache``): each image as two 256-px squares
   through DINOv3 ViT-L/16, tokens of four layers pooled to 4 x 4 per square (the shore
   video's ``image`` recipe), ``[triples, 2, 4, 16, 1024]`` float16 per input.
2. **Heads**: per arm (``still``, ``timex``, ``brt``; ``all3`` reported) the video's
   attentive probe on log H_s, trained on 2024, epoch chosen on the validation weeks,
   three seeds averaged in log space.
3. **Scores**: RMSE of H_s on 2025 (bars T9-A/B, 97.5 % day-bootstrap intervals) and 2026,
   per camera, WMO classes, against climatology.

    python scripts/sensing_timex_probe.py --config configs/experiments/sensing_timex_v1.yaml \\
        --package <pkg> --out runs/<name> --feature-cache <dir> [--limit 300 --epochs 2]
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from PIL import Image
from torch import nn

from wave_analysis.config import load_yaml
from wave_analysis.evaluation.uncertainty import paired_block_bootstrap_ci
from wave_analysis.models.video_probe import AttentiveProbe, normalise, pool_tokens, square_tiles

KINDS = ("still", "timex", "brt")


def log(msg: str) -> None:
    print(f"{time.strftime('%H:%M:%S')} {msg}", flush=True)


def rmse(y: np.ndarray, p: np.ndarray) -> float:
    return float(np.sqrt(np.mean((np.asarray(y, float) - np.asarray(p, float)) ** 2)))


class Images(torch.utils.data.Dataset):
    def __init__(self, pkg: Path, d: pd.DataFrame, kind: str) -> None:
        self.paths = [
            pkg / "images" / c / kind / f"{s}.jpg"
            for c, s in zip(d["camera"], d["stamp"], strict=True)
        ]

    def __len__(self) -> int:
        return len(self.paths)

    def __getitem__(self, i: int) -> tuple[torch.Tensor, int]:
        f = np.asarray(Image.open(self.paths[i]).convert("RGB"))[None]  # [1, H, W, 3]
        return torch.from_numpy(square_tiles(f)[:, 0]).permute(
            0, 3, 1, 2
        ).contiguous(), i  # [2, 3, H, H]


@torch.no_grad()
def extract(
    d: pd.DataFrame, cfg: dict, args: argparse.Namespace, cache: Path
) -> dict[str, np.ndarray]:
    import timm

    enc = cfg["encoder"]
    out = {}
    model = None
    for kind in KINDS:
        path = cache / f"{kind}.npy"
        if (cache / f"{kind}.done").exists():
            out[kind] = np.load(path, mmap_mode="r")
            continue
        if model is None:
            model = (
                timm.create_model(enc["timm"], pretrained=True, num_classes=0)
                .to(args.device)
                .eval()
            )
        patch, tile, pool = (
            int(model.patch_embed.patch_size[0]),
            int(enc["tile_px"]),
            tuple(enc["pool"]),
        )
        cache.mkdir(parents=True, exist_ok=True)
        mm = np.lib.format.open_memmap(path, "w+", np.float16, (len(d), 2, len(enc["layers"]), int(np.prod(pool)),
                                                                 int(model.num_features)))  # fmt: skip
        dl = torch.utils.data.DataLoader(Images(args.package, d, kind), batch_size=64,
                                         num_workers=args.num_workers, pin_memory=True)  # fmt: skip
        t0 = time.time()
        for b, (x, idx) in enumerate(dl):
            x = x.to(args.device, non_blocking=True)
            with torch.autocast(
                "cuda", dtype=torch.bfloat16, enabled=args.device.startswith("cuda")
            ):
                inter = model.forward_intermediates(normalise(x.flatten(0, 1)), indices=list(enc["layers"]),
                                                    output_fmt="NLC", intermediates_only=True)  # fmt: skip
            g = (1, tile // patch, tile // patch)
            f = torch.stack([pool_tokens(h, g, pool) for h in inter], dim=1)
            mm[idx.numpy()] = f.reshape(x.shape[0], 2, *f.shape[1:]).half().cpu().numpy()
            if b % 100 == 0:
                log(
                    f"  {kind}: {min((b + 1) * 64, len(d)):,}/{len(d):,} images, {time.time() - t0:.0f} s"
                )
        mm.flush()
        (cache / f"{kind}.done").write_text("ok\n")
        out[kind] = np.load(path, mmap_mode="r")
    return out


def fit_predict(x: torch.Tensor, y: np.ndarray, tr: np.ndarray, va: np.ndarray, pred_idx: np.ndarray,
                hp: dict, seed: int, dev: str, epochs: int) -> tuple[np.ndarray, int]:  # fmt: skip
    """The video run's ``fit_head`` (attentive, Huber, one-cycle, epoch by val MSE) with the
    features kept in CPU memory; returns standardised predictions for ``pred_idx`` and the epoch."""
    torch.manual_seed(seed)
    _, p, n_layers, t, dim = x.shape
    head = AttentiveProbe(n_layers, dim, p * t, 1, d=int(hp["d"]), heads=int(hp["heads"]),
                          dropout=float(hp["dropout"])).to(dev)  # fmt: skip
    opt = torch.optim.AdamW(
        head.parameters(), lr=float(hp["lr"]), weight_decay=float(hp["weight_decay"])
    )
    bs = int(hp["batch_size"])
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=float(hp["lr"]),
                                                total_steps=epochs * max(1, int(np.ceil(len(tr) / bs))),
                                                pct_start=0.1)  # fmt: skip
    yt = torch.as_tensor(y, dtype=torch.float32)
    rng = np.random.default_rng(seed)

    def predict(idx: np.ndarray) -> np.ndarray:
        head.eval()
        with torch.no_grad():
            return np.concatenate([head(x[idx[s : s + 1024]].to(dev)).float().cpu().numpy()[:, 0]
                                   for s in range(0, len(idx), 1024)])  # fmt: skip

    best, state, best_ep = np.inf, copy.deepcopy(head.state_dict()), 0
    for ep in range(epochs):
        head.train()
        order = tr[rng.permutation(len(tr))]
        for s in range(0, len(order), bs):
            b = order[s : s + bs]
            loss = nn.functional.smooth_l1_loss(
                head(x[b].to(dev, non_blocking=True))[:, 0], yt[b].to(dev)
            )
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            sched.step()
        v = float(np.mean((predict(va) - y[va]) ** 2))
        if v < best:
            best, state, best_ep = v, copy.deepcopy(head.state_dict()), ep + 1
    head.load_state_dict(state)
    return predict(pred_idx), best_ep


def classes(h: np.ndarray, edges: list[float]) -> np.ndarray:
    """WMO bands closed on the right: <= 0.5, <= 1.25, <= 2.5, above."""
    return np.searchsorted(np.asarray(edges), np.asarray(h, float), side="left")


def balanced_accuracy(y: np.ndarray, p: np.ndarray) -> float:
    return float(np.mean([np.mean(p[y == c] == c) for c in np.unique(y)]))


def macro_f1(y: np.ndarray, p: np.ndarray) -> float:
    f = []
    for c in np.unique(y):
        tp = np.sum((p == c) & (y == c))
        prec = tp / max(1, np.sum(p == c))
        rec = tp / max(1, np.sum(y == c))
        f.append(0.0 if tp == 0 else 2 * prec * rec / (prec + rec))
    return float(np.mean(f))


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--config", type=Path, required=True)
    ap.add_argument("--package", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument(
        "--split", default="year", help="accepted for the runner; the split is the package's"
    )
    ap.add_argument("--amp", default="bf16", help="accepted for the runner")
    ap.add_argument("--num-workers", type=int, default=8)
    ap.add_argument("--feature-cache", type=Path, required=True)
    ap.add_argument("--limit", type=int, help="first N triples per split (smoke test)")
    ap.add_argument("--epochs", type=int, help="override epochs (smoke test)")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args(argv)
    cfg = load_yaml(args.config)
    args.out.mkdir(parents=True, exist_ok=True)
    d = pd.read_csv(args.package / "triples.csv")
    if args.limit:
        d = d.groupby("split", group_keys=False).head(args.limit).reset_index(drop=True)
    content = hashlib.sha256((args.package / "SHA256SUMS").read_bytes()).hexdigest()[:12]
    key = hashlib.sha256(json.dumps({"enc": cfg["encoder"], "pkg": content, "n": len(d)}, sort_keys=True)
                         .encode()).hexdigest()[:12]  # fmt: skip
    log(
        f"{len(d):,} triples ({d['split'].value_counts().to_dict()}), package {content}, cache {key}"
    )
    feats = extract(d, cfg, args, args.feature_cache / key)
    hp, epochs = cfg["heads"], int(args.epochs or cfg["heads"]["epochs"])
    ylog = np.log(d["Hs_label_m"].to_numpy(float))
    split = d["split"].to_numpy()
    tr, va = np.flatnonzero(split == "train"), np.flatnonzero(split == "val")
    mu, sd = ylog[tr].mean(), ylog[tr].std() + 1e-6
    y = (ylog - mu) / sd
    allidx = np.arange(len(d))
    preds = d[["camera", "stamp", "local_day", "split", "Hs_label_m"]].copy()
    preds["pred_climatology"] = np.exp(mu)
    epochs_chosen: dict[str, list[int]] = {}
    for arm in [*KINDS, "all3"]:
        x = torch.from_numpy(np.concatenate([np.asarray(feats[k]) for k in KINDS], axis=1)
                             if arm == "all3" else np.asarray(feats[arm]))  # fmt: skip
        acc = np.zeros(len(d))
        epochs_chosen[arm] = []
        for seed in hp["seeds"]:
            p, ep = fit_predict(x, y, tr, va, allidx, hp, int(seed), args.device, epochs)
            acc += p
            epochs_chosen[arm].append(ep)
            log(f"{arm} seed {seed}: epoch {ep}")
        preds[f"pred_{arm}"] = np.exp(acc / len(hp["seeds"]) * sd + mu)
        del x
    preds.to_csv(args.out / "predictions.csv", index=False)
    ev = cfg["evaluation"]
    boot = {k: ev["bootstrap"][k] for k in ("n_boot", "alpha", "seed")}
    arms = [*KINDS, "all3", "climatology"]
    res: dict[str, Any] = {"package_content": content, "feature_key": key, "epochs_chosen": epochs_chosen,
                           "smoke": bool(args.limit or args.epochs), "parts": {}}  # fmt: skip
    for part in ["val", "test", "later"]:
        q = preds[preds["split"] == part]
        if len(q) == 0 or q["local_day"].nunique() < 2:
            continue
        yv, g = q["Hs_label_m"].to_numpy(), q["local_day"].to_numpy()
        r: dict[str, Any] = {"n": len(q), "days": int(q["local_day"].nunique()),
                             "rmse": {a: rmse(yv, q[f"pred_{a}"]) for a in arms}}  # fmt: skip
        for a in ["timex", "brt", "all3", "climatology"]:
            pt, lo, hi = paired_block_bootstrap_ci(
                yv, q[f"pred_{a}"], q["pred_still"], rmse, groups=g, **boot
            )
            r[f"{a} - still"] = {"diff": pt, "lo": lo, "hi": hi}
        yc = classes(yv, ev["wmo_edges_m"])
        r["wmo"] = {a: {"balanced_accuracy": balanced_accuracy(yc, classes(q[f"pred_{a}"], ev["wmo_edges_m"])),
                        "macro_f1": macro_f1(yc, classes(q[f"pred_{a}"], ev["wmo_edges_m"]))} for a in arms}  # fmt: skip
        r["by_camera"] = {c: {"n": len(v), **{a: rmse(v["Hs_label_m"], v[f"pred_{a}"]) for a in arms}}
                          for c, v in q.groupby("camera")}  # fmt: skip
        res["parts"][part] = r
    t = res["parts"].get("test")
    if t:
        res["bars"] = {
            "T9-A (timex - still, H_s RMSE, 2025)": "met"
            if t["timex - still"]["hi"] < 0
            else "not met",
            "T9-B (brt - still, H_s RMSE, 2025)": "met"
            if t["brt - still"]["hi"] < 0
            else "not met",
        }
        log(f"bars: {res['bars']}")
    (args.out / "metrics.json").write_text(json.dumps(res, indent=2, default=float) + "\n")
    log("done")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
