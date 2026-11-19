#!/usr/bin/env python
"""Test 3, video half: Kim et al. 2023's CNN-BiLSTM and Kamagata et al. 2026's SlowFast head
(with its physics loss) on the shore video's clips and date folds, against the project's
whole-clip read-out.

Bar T3-K / T3-S and every setting: ``docs/experiments/sensing_per_wave_bars.md`` ("Test 3
bar, video half"); config ``configs/experiments/sensing_shore_v1.yaml``.

* **Kim** (``Kim2023CNNBiLSTM``): the central 180 s of each clip as 360 grey frames at
  2 fps, read from the clip's window zip, centre-cropped square, 224 px, frozen ImageNet
  GoogLeNet features cached once in ``--kim-cache``; the BiLSTM (2,000 units) and its
  classifier trained on Kim's five classes of measured mean height.
* **Kamagata** (``SlowFastWaveHead`` + ``kamagata_physics_loss``): each 16-s window's
  cached V-JEPA 2 tokens (last layer, spatial mean: eight 2-s steps of 1,024), a linear
  map to 384, the paper's head; targets H_s, T_p, T_z, direction (H_max unlabelled);
  every window carries its clip's labels; clips read as the project reads them.

The package (``data/processed/packages/sensing_shore_v1``) holds ``labels.csv`` (the test
2 label table) and ``interim.csv`` (the interim video run's per-clip labels, folds and
out-of-fold estimates). The window zips and the feature cache are read, never written.

    python scripts/sensing_shore_test3.py --config configs/experiments/sensing_shore_v1.yaml \\
        --package <pkg> --out runs/<name> --windows <dir> --window-cache <dir> --kim-cache <dir> \\
        [--arms kim,kamagata] [--limit 40 --epochs 2]
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import sys
import time
import warnings
import zipfile
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from PIL import Image
from torch import nn

from wave_analysis.config import load_yaml
from wave_analysis.evaluation.regression import pearson, r2
from wave_analysis.evaluation.uncertainty import paired_block_bootstrap_ci
from wave_analysis.models.baselines.kim2023 import Kim2023CNNBiLSTM
from wave_analysis.models.layers.backbones import build_backbone
from wave_analysis.models.losses import kamagata_physics_loss
from wave_analysis.models.multimodal.slowfast_head import SlowFastWaveHead
from wave_analysis.models.video_probe import energy_mean, geometric_mean, midpoint_window

sys.path.insert(0, str(Path(__file__).resolve().parent))
import sensing_shore_test2 as t2

FPS = 4  # the window zips' frame rate
TARGETS = ["Hs", "Hmax", "Tp", "Tz", "direction"]  # SlowFastWaveHead.TARGETS order
IMAGENET = (
    np.array([0.485, 0.456, 0.406], np.float32),
    np.array([0.229, 0.224, 0.225], np.float32),
)


def log(msg: str) -> None:
    print(f"{time.strftime('%H:%M:%S')} {msg}", flush=True)


def rmse(y: np.ndarray, p: np.ndarray) -> float:
    return float(np.sqrt(np.mean((np.asarray(y, float) - np.asarray(p, float)) ** 2)))


def val_date(days: np.ndarray, candidates: np.ndarray, fold: int, seed: int) -> str:
    """The inner validation date of (fold, seed): drawn as the video run's ``_val_split`` does."""
    return str(np.random.default_rng(1000 * fold + seed).choice(np.array(sorted(set(candidates)))))


# --- data ---------------------------------------------------------------------------------


def load_clips(pkg: Path, limit: int | None) -> pd.DataFrame:
    it = pd.read_csv(pkg / "interim.csv")
    lab = pd.read_csv(pkg / "labels.csv")
    cols = ["clip_id"] + [c for c in lab.columns if c.startswith(("good_", "incl_unevaluated_"))]
    d = it.merge(lab[cols], on="clip_id", how="left", validate="1:1").sort_values("clip_id")
    d = d.reset_index(drop=True)
    if limit:  # smoke tests: a few clips from every date
        d = d.groupby("local_day", group_keys=False).head(max(1, limit // d["local_day"].nunique()))
        d = d.reset_index(drop=True)
    return d


def kim_frames(zpath: Path, duration_s: float, k: int, seconds: float, side: int) -> np.ndarray:
    """``[T, 3, side, side]``: the central ``seconds`` at 2 fps, grey, centre square, normalised."""
    n = 64 * k  # frames in the zip (the first K whole windows)
    span = round(seconds * FPS)
    c = round(duration_s / 2 * FPS)
    lo = int(np.clip(c - span // 2, 0, max(0, n - span)))
    idx = range(lo, min(n, lo + span), 2)
    out = []
    with zipfile.ZipFile(zpath) as z:
        for i in idx:
            im = Image.open(z.open(f"{i:04d}.jpg")).convert("L")
            w, h = im.size
            s = min(w, h)
            im = im.crop(((w - s) // 2, (h - s) // 2, (w - s) // 2 + s, (h - s) // 2 + s))
            g = np.asarray(im.resize((side, side), Image.BILINEAR), np.float32) / 255.0
            out.append(
                (np.repeat(g[None], 3, 0) - IMAGENET[0][:, None, None]) / IMAGENET[1][:, None, None]
            )
    return np.stack(out)


class KimFrames(torch.utils.data.Dataset):
    def __init__(self, rows: pd.DataFrame, windows: Path, seconds: float, side: int) -> None:
        self.rows, self.windows, self.seconds, self.side = (
            rows.reset_index(drop=True),
            windows,
            seconds,
            side,
        )

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, i: int) -> tuple[torch.Tensor, int]:
        r = self.rows.iloc[i]
        z = self.windows / r["camera"] / f"{r['clip_id']}.zip"
        return torch.from_numpy(
            kim_frames(z, float(r["duration_s"]), int(r["K"]), self.seconds, self.side)
        ), i


@torch.no_grad()
def kim_features(d: pd.DataFrame, args: argparse.Namespace, cfg: dict) -> dict[str, np.ndarray]:
    """GoogLeNet features ``[T, 1024]`` per clip, cached as ``<kim-cache>/<camera>/<clip>.npy``."""
    k = cfg["kim"]
    feats: dict[str, np.ndarray] = {}
    todo = []
    for r in d.itertuples():
        p = args.kim_cache / r.camera / f"{r.clip_id}.npy"
        if p.exists():
            feats[r.clip_id] = np.load(p)
        else:
            todo.append(r.Index)
    log(f"kim: {len(feats)} clips cached, {len(todo)} to extract")
    if todo:
        net, _ = build_backbone("googlenet", pretrained=True)
        net = net.to(args.device).eval()
        rows = d.loc[todo]
        dl = torch.utils.data.DataLoader(KimFrames(rows, args.windows, float(k["seconds"]), int(k["side_px"])),
                                         batch_size=None, num_workers=args.num_workers)  # fmt: skip
        t0 = time.time()
        for j, (x, i) in enumerate(dl):
            r = rows.iloc[int(i)]
            with torch.autocast(
                "cuda", dtype=torch.bfloat16, enabled=args.device.startswith("cuda")
            ):
                f = torch.cat([net(x[s : s + 128].to(args.device)) for s in range(0, len(x), 128)])
            a = f.float().cpu().numpy().astype(np.float16)
            p = args.kim_cache / r["camera"] / f"{r['clip_id']}.npy"
            p.parent.mkdir(parents=True, exist_ok=True)
            tmp = p.with_name(p.name + ".part")
            with open(tmp, "wb") as fh:
                np.save(fh, a)
            tmp.rename(p)
            feats[r["clip_id"]] = a
            if j % 100 == 0:
                log(f"kim features {j + 1}/{len(rows)} ({time.time() - t0:.0f} s)")
    return feats


def kamagata_inputs(d: pd.DataFrame, cache: Path, layer: int) -> list[np.ndarray]:
    """Per clip ``[K, 8, 1024]``: the window's V-JEPA 2 tokens of one layer, spatial and tile mean."""
    out = []
    t0 = time.time()
    for j, r in enumerate(d.itertuples()):
        a = np.load(
            cache / r.camera / f"{r.clip_id}.video.npy", mmap_mode="r"
        )  # [K, tiles, layers, 128, D]
        x = np.asarray(a[:, :, layer], np.float32)  # [K, tiles, 128, D]
        k, tiles, n, dim = x.shape
        out.append(x.reshape(k, tiles, 8, n // 8, dim).mean(axis=(1, 3)).astype(np.float16))
        if j % 200 == 0:
            log(f"kamagata inputs {j + 1}/{len(d)} ({time.time() - t0:.0f} s)")
    return out


# --- Kim ---------------------------------------------------------------------------------


def kim_logits(m: Kim2023CNNBiLSTM, x: torch.Tensor) -> torch.Tensor:
    """The class's own LSTM, dropout and classifier on cached frame features ``[B, T, 1024]``."""
    out, _ = m.lstm(x)
    return m.fc(m.dropout(out[:, -1]))


def kim_arm(
    d: pd.DataFrame, feats: dict[str, np.ndarray], cfg: dict, args: argparse.Namespace
) -> np.ndarray:
    """Out-of-fold class probabilities ``[clips, 5]`` (seeds averaged) for the eligible clips ``d``."""
    k = cfg["kim"]
    dev = args.device
    x = torch.from_numpy(np.stack([feats[c] for c in d["clip_id"]]).astype(np.float32)).to(dev)
    y = torch.as_tensor(d["kim_true"].to_numpy().copy(), device=dev)
    fold, days = d["fold"].to_numpy(), d["local_day"].to_numpy()
    prob = np.zeros((len(d), 5))
    for f in sorted(set(fold)):
        te, trall = np.flatnonzero(fold == f), np.flatnonzero(fold != f)
        for seed in cfg["seeds"]:
            vday = val_date(days, days[trall], int(f), int(seed))
            va, tr = trall[days[trall] == vday], trall[days[trall] != vday]
            torch.manual_seed(int(seed))
            m = Kim2023CNNBiLSTM(5, backbone="googlenet", pretrained=False,
                                 hidden_size=int(k["hidden"]), dropout=float(k["dropout"]))  # fmt: skip
            m.encoder = nn.Identity()  # frozen GoogLeNet features are cached
            m = m.to(dev)
            opt = torch.optim.Adam(m.parameters(), lr=float(k["lr"]))
            rng = np.random.default_rng(int(seed))
            best, state = np.inf, None
            for _ in range(int(args.epochs or k["epochs"])):
                m.train()
                order = tr[rng.permutation(len(tr))]
                for s in range(0, len(order), int(k["batch_size"])):
                    b = torch.as_tensor(order[s : s + int(k["batch_size"])], device=dev)
                    loss = nn.functional.cross_entropy(kim_logits(m, x[b]), y[b])
                    opt.zero_grad(set_to_none=True)
                    loss.backward()
                    nn.utils.clip_grad_norm_(m.parameters(), float(k["grad_clip"]))
                    opt.step()
                m.eval()
                with torch.no_grad():
                    v = float(nn.functional.cross_entropy(kim_logits(m, x[torch.as_tensor(va, device=dev)]),
                                                          y[torch.as_tensor(va, device=dev)]))  # fmt: skip
                if v < best:
                    best, state = v, copy.deepcopy(m.state_dict())
            m.load_state_dict(state)
            m.eval()
            with torch.no_grad():
                prob[te] += (
                    torch.softmax(kim_logits(m, x[torch.as_tensor(te, device=dev)]), -1)
                    .cpu()
                    .numpy()
                )
            log(
                f"kim fold {f} seed {seed}: {len(tr)} train, val {vday} ({len(va)}), best val CE {best:.3f}"
            )
    return prob / len(cfg["seeds"])


# --- Kamagata ----------------------------------------------------------------------------


class Kamagata(nn.Module):
    """A linear map to the paper's width, then the paper's SlowFast head."""

    def __init__(self, dim: int, width: int) -> None:
        super().__init__()
        self.proj = nn.Linear(dim, width)
        self.head = SlowFastWaveHead(width)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(self.proj(x))


def kamagata_loss(out: torch.Tensor, y: torch.Tensor, have: torch.Tensor, span: torch.Tensor,
                  lo: torch.Tensor, physics_weight: float) -> torch.Tensor:  # fmt: skip
    """MSE over the labelled scaled targets + the physics term on unscaled H_s (m) and T_p (s)."""
    mse = ((out - y) ** 2 * have).sum() / have.sum().clamp_min(1)
    nat = out * span + lo
    return mse + physics_weight * kamagata_physics_loss(
        nat[:, 0].clamp_min(1e-3), nat[:, 2].clamp_min(1e-3)
    )


def kamagata_arm(d: pd.DataFrame, xs: list[np.ndarray], cfg: dict, args: argparse.Namespace,
                 physics_weight: float) -> pd.DataFrame:  # fmt: skip
    """Out-of-fold per-clip estimates (``clip`` and midpoint ``window`` read-outs), seeds averaged."""
    kc = cfg["kamagata"]
    dev = args.device
    kk = np.array([len(a) for a in xs])
    clip_of = np.repeat(np.arange(len(d)), kk)
    x = torch.from_numpy(np.concatenate(xs).astype(np.float32)).to(dev)  # [windows, 8, 1024]
    y_clip = np.stack([d["Hs_label_m"], np.full(len(d), np.nan), d["Tp_label_s"], d["good_tz_s"],
                       d["Dp_label_deg"]], axis=1).astype(float)  # fmt: skip
    yw = y_clip[clip_of]
    fold, days = d["fold"].to_numpy(), d["local_day"].to_numpy()
    wfold, wday = fold[clip_of], days[clip_of]
    logsum = np.zeros((len(x), 4))  # log H_s, H_max, T_p, T_z per window, summed over seeds
    for f in sorted(set(fold)):
        te, trall = np.flatnonzero(wfold == f), np.flatnonzero(wfold != f)
        with warnings.catch_warnings():  # H_max has no label: an all-NaN column
            warnings.simplefilter("ignore", RuntimeWarning)
            lo, hi = np.nanmin(yw[trall], 0), np.nanmax(yw[trall], 0)
        lo, span = np.nan_to_num(lo), np.where(np.isfinite(hi - lo) & (hi > lo), hi - lo, 1.0)
        ys = torch.as_tensor((yw - lo) / span, dtype=torch.float32, device=dev)
        have = torch.isfinite(ys)
        ys = torch.nan_to_num(ys)
        lo_t, span_t = (torch.as_tensor(v, dtype=torch.float32, device=dev) for v in (lo, span))
        for seed in cfg["seeds"]:
            vday = val_date(days, days[np.flatnonzero(fold != f)], int(f), int(seed))
            va, tr = trall[wday[trall] == vday], trall[wday[trall] != vday]
            torch.manual_seed(int(seed))
            m = Kamagata(x.shape[-1], int(kc["width"])).to(dev)
            opt = torch.optim.AdamW(
                m.parameters(), lr=float(kc["lr"]), weight_decay=float(kc["weight_decay"])
            )
            rng = np.random.default_rng(int(seed))

            best, state = np.inf, None
            for _ in range(int(args.epochs or kc["epochs"])):
                m.train()
                order = tr[rng.permutation(len(tr))]
                for s in range(0, len(order), int(kc["batch_size"])):
                    b = torch.as_tensor(order[s : s + int(kc["batch_size"])], device=dev)
                    loss = kamagata_loss(m(x[b]), ys[b], have[b], span_t, lo_t, physics_weight)
                    opt.zero_grad(set_to_none=True)
                    loss.backward()
                    opt.step()
                m.eval()
                with torch.no_grad():
                    vb = torch.as_tensor(va, device=dev)
                    v = float(
                        sum(
                            kamagata_loss(
                                m(x[vb[s : s + 4096]]),
                                ys[vb[s : s + 4096]],
                                have[vb[s : s + 4096]],
                                span_t,
                                lo_t,
                                physics_weight,
                            )
                            * len(vb[s : s + 4096])
                            for s in range(0, len(vb), 4096)
                        )
                        / len(vb)
                    )
                if v < best:
                    best, state = v, copy.deepcopy(m.state_dict())
            m.load_state_dict(state)
            m.eval()
            with torch.no_grad():
                p = torch.cat(
                    [
                        m(x[torch.as_tensor(te[s : s + 4096], device=dev)])
                        for s in range(0, len(te), 4096)
                    ]
                )
                nat = (p * span_t + lo_t).clamp_min(1e-3)[:, :4].cpu().numpy()
            logsum[te] += np.log(nat)
            log(
                f"kamagata (physics {physics_weight}) fold {f} seed {seed}: val {vday}, best val loss {best:.4f}"
            )
    win = np.exp(logsum / len(cfg["seeds"]))
    out = pd.DataFrame({"clip_id": d["clip_id"]})
    starts = np.concatenate([[0], np.cumsum(kk)[:-1]])
    for t, col in [("hs", 0), ("tp", 2), ("tz", 3)]:
        clip_vals, mid_vals = [], []
        for i, (s, k) in enumerate(zip(starts, kk, strict=True)):
            w = win[s : s + k, col]
            clip_vals.append(float(energy_mean(w)) if t == "hs" else float(geometric_mean(w)))
            mid_vals.append(float(w[midpoint_window(float(d["duration_s"].iloc[i]), int(k))]))
        out[f"pred_{t}_clip"], out[f"pred_{t}_window"] = clip_vals, mid_vals
    return out


# --- scores ------------------------------------------------------------------------------


def diff(y, a, b, metric, g, boot) -> dict:  # type: ignore[no-untyped-def]
    point, lo, hi = paired_block_bootstrap_ci(y, a, b, metric, groups=g, **boot)
    return {"diff": point, "lo": lo, "hi": hi}


def verdict(dd: dict, *, lower_is_project_better: bool) -> str:
    if lower_is_project_better:  # RMSE: project minus paper
        return (
            "project ahead"
            if dd["hi"] < 0
            else ("paper ahead" if dd["lo"] > 0 else "no difference resolved")
        )
    return (
        "project ahead"
        if dd["lo"] > 0
        else ("paper ahead" if dd["hi"] < 0 else "no difference resolved")
    )


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--config", type=Path, required=True)
    ap.add_argument("--package", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument(
        "--split", default="days", help="accepted for the runner; folds are the interim run's"
    )
    ap.add_argument("--amp", default="bf16", help="accepted for the runner")
    ap.add_argument("--num-workers", type=int, default=8)
    ap.add_argument("--windows", type=Path, required=True, help="<camera>/<clip>.zip, read only")
    ap.add_argument(
        "--window-cache", type=Path, required=True, help="V-JEPA 2 window features, read only"
    )
    ap.add_argument(
        "--kim-cache", type=Path, required=True, help="this test's GoogLeNet frame features"
    )
    ap.add_argument("--arms", default="kim,kamagata")
    ap.add_argument("--limit", type=int, help="about N clips over all dates (smoke test)")
    ap.add_argument("--epochs", type=int, help="override epochs (smoke test)")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args(argv)
    cfg = load_yaml(args.config)
    args.out.mkdir(parents=True, exist_ok=True)
    arms = set(args.arms.split(","))
    boot = {k: cfg["bootstrap"][k] for k in ("n_boot", "alpha", "seed")}
    d = load_clips(args.package, args.limit)
    log(f"{len(d)} clips, {d['local_day'].nunique()} dates; seeds {cfg['seeds']}")
    res: dict[str, Any] = {
        "commit": (Path(__file__).resolve().parents[1] / "GIT_COMMIT").read_text().strip()
        if (Path(__file__).resolve().parents[1] / "GIT_COMMIT").exists() else None,
        "clip_list_sha256": hashlib.sha256("\n".join(d["clip_id"]).encode()).hexdigest(),
        "smoke": bool(args.limit or args.epochs),
    }  # fmt: skip
    # the project's arms of test 2 on the same clips (tier good): Kim classes of the clip read-out
    lab = d[d["good_mean_height_m"].notna()].reset_index(drop=True)
    o2 = t2.arms(lab, "good")
    preds = d[["clip_id", "camera", "local_day", "fold", "K", "Hs_label_m", "Tp_label_s", "good_tz_s",
               "pred_hs_clip", "pred_tp_clip"]].copy()  # fmt: skip
    preds = preds.merge(o2[["clip_id", "kim_in_range", "kim_true", "kim_clip", "kim_majority",
                            "pred_tz_spectrum", "pred_tz_climatology"]], on="clip_id", how="left")  # fmt: skip

    if "kamagata" in arms:
        xs = kamagata_inputs(d, args.window_cache, int(cfg["kamagata"]["layer_index"]))
        if any(len(a) != k for a, k in zip(xs, d["K"], strict=True)):
            raise ValueError("cached windows and the interim run's K disagree")
        for name, w in [
            ("kamagata", float(cfg["kamagata"]["physics_weight"])),
            ("kamagata_nophys", 0.0),
        ]:
            p = (
                kamagata_arm(d, xs, cfg, args, w)
                .add_prefix(f"{name}_")
                .rename(columns={f"{name}_clip_id": "clip_id"})
            )
            preds = preds.merge(p, on="clip_id")
        y, g = preds["Hs_label_m"].to_numpy(), preds["local_day"].to_numpy()
        clim = t2.per_fold(
            preds["fold"].to_numpy(),
            lambda tr, te: np.full(te.sum(), np.exp(np.mean(np.log(y[tr])))),
        )
        s: dict[str, Any] = {"n_clips": len(preds), "n_dates": int(preds["local_day"].nunique()), "hs_rmse": {
            "project_clip": rmse(y, preds["pred_hs_clip"]), "kamagata_clip": rmse(y, preds["kamagata_pred_hs_clip"]),
            "kamagata_window": rmse(y, preds["kamagata_pred_hs_window"]),
            "kamagata_nophys_clip": rmse(y, preds["kamagata_nophys_pred_hs_clip"]), "climatology": rmse(y, clim)}}  # fmt: skip
        s["T3-S (H_s RMSE, project clip - Kamagata clip)"] = dd = diff(
            y, preds["pred_hs_clip"], preds["kamagata_pred_hs_clip"], rmse, g, boot
        )
        s["T3-S verdict"] = verdict(dd, lower_is_project_better=True)
        s["kamagata physics - no physics (H_s RMSE)"] = diff(
            y, preds["kamagata_pred_hs_clip"], preds["kamagata_nophys_pred_hs_clip"], rmse, g, boot
        )
        s["kamagata clip - window (H_s RMSE)"] = diff(
            y, preds["kamagata_pred_hs_clip"], preds["kamagata_pred_hs_window"], rmse, g, boot
        )
        tp = preds["Tp_label_s"].to_numpy()
        s["tp_rmse"] = {"kamagata_clip": rmse(tp, preds["kamagata_pred_tp_clip"]),
                        "project_clip": rmse(tp, preds["pred_tp_clip"])}  # fmt: skip
        tzok = preds["good_tz_s"].notna() & preds["pred_tz_spectrum"].notna()
        tz = preds.loc[tzok]
        s["tz_rmse"] = {a: rmse(tz["good_tz_s"], tz[c]) for a, c in [
            ("kamagata_clip", "kamagata_pred_tz_clip"), ("spectrum", "pred_tz_spectrum"),
            ("climatology", "pred_tz_climatology")]}  # fmt: skip
        oak = preds[preds["camera"].str.startswith("oakisland")]
        s["oak_island"] = {a: {"rmse": rmse(oak["Hs_label_m"], oak[c]), "r2": float(r2(oak["Hs_label_m"], oak[c])),
                               "pearson": float(pearson(oak["Hs_label_m"], oak[c]))}
                           for a, c in [("project_clip", "pred_hs_clip"), ("kamagata_clip", "kamagata_pred_hs_clip")]}  # fmt: skip
        s["by_camera_hs_rmse"] = {cam: {"n": len(c), "project": rmse(c["Hs_label_m"], c["pred_hs_clip"]),
                                        "kamagata": rmse(c["Hs_label_m"], c["kamagata_pred_hs_clip"])}
                                  for cam, c in preds.groupby("camera")}  # fmt: skip
        res["kamagata"] = s
        log(f"T3-S: {dd} -> {s['T3-S verdict']}")

    if "kim" in arms:
        e = preds[(preds["kim_in_range"] == True) & preds["K"].ge(int(cfg["kim"]["min_windows"]))]  # noqa: E712
        e = e.reset_index(drop=True)
        dk = d.set_index("clip_id").loc[e["clip_id"]].reset_index()
        dk["kim_true"] = e["kim_true"].to_numpy()
        feats = kim_features(dk, args, cfg)
        prob = kim_arm(dk, feats, cfg, args)
        e["kim_kim"] = prob.argmax(1)
        for c in range(5):
            e[f"kim_prob_{c}"] = prob[:, c]
        yk, g = e["kim_true"].to_numpy(), e["local_day"].to_numpy()
        n_cls = len(set(yk))
        s = {"n_clips": len(e), "n_dates": int(e["local_day"].nunique()), "classes_present": n_cls,
             "assessable": t2.assessable(yk, g),
             "accuracy": {a: t2.accuracy(yk, e[f"kim_{a}"]) for a in ["clip", "kim", "majority"]},
             "within_one": {a: t2.within_one(yk, e[f"kim_{a}"]) for a in ["clip", "kim", "majority"]}}  # fmt: skip
        s["T3-K (accuracy, project clip - Kim)"] = dd = diff(
            yk, e["kim_clip"], e["kim_kim"], t2.accuracy, g, boot
        )
        s["T3-K verdict"] = (
            verdict(dd, lower_is_project_better=False)
            if s["assessable"]
            else "unassessable (coverage)"
        )
        s["kim - majority (accuracy)"] = diff(
            yk, e["kim_kim"], e["kim_majority"], t2.accuracy, g, boot
        )
        s["by_camera"] = {cam: {"n": len(c), **{a: t2.accuracy(c["kim_true"], c[f"kim_{a}"]) for a in ["clip", "kim", "majority"]}}
                          for cam, c in e.groupby("camera")}  # fmt: skip
        res["kim"] = s
        e.to_csv(args.out / "predictions_kim.csv", index=False)
        log(f"T3-K: {dd} -> {s['T3-K verdict']}")
    preds.to_csv(args.out / "predictions.csv", index=False)
    (args.out / "metrics.json").write_text(json.dumps(res, indent=2, default=float) + "\n")
    log("done")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
