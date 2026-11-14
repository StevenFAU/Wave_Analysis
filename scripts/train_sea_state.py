#!/usr/bin/env python
"""Train and evaluate a sea-state classifier on a package (on a GPU; a CPU run is a smoke test).

    python scripts/train_sea_state.py --config configs/experiments/currituck_sea_state_v0.yaml \\
        --package /path/to/currituck_sea_state_v0 --split year --out runs/year_convnext

The package comes from ``scripts/sea_state_package.py``. ``--split`` picks the
partition column (``day``: whole local days; ``year``: train 2022-2024, test
2025, later test 2026). The model is trained on ``train``, the epoch is chosen
on ``val``, and every test partition is predicted once with that epoch.

Outputs in ``--out``:

* ``predictions.csv`` - every still of val and the test partitions: true and
  predicted class, class probabilities.
* ``metrics.json`` - per partition, the model's metrics with day-bootstrap
  intervals, and each baseline's beside the model's **on the same stills**
  (a nowcast does not cover every still: WIS has no 2022 and ends in 2025).
* ``confusion_<partition>.svg`` / ``.csv`` - for the model and the WIS baseline.
* ``report.md`` - the tables above as text; ``history.csv`` - one row per epoch.
* ``best.pt`` - the chosen weights with the settings and class names.
* ``provenance.yaml`` - git commit (read at the start: a run takes hours and
  the checkout may move on), inputs (labels, baselines, the package's
  SHA256SUMS), software and the GPU.

Smoke test on a CPU: ``--limit 64 --epochs 1 --num-workers 2``.
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
from wave_analysis.datasets.provenance import git_state
from wave_analysis.datasets.sea_state import TEST_LATER
from wave_analysis.datasets.splits import TEST, TRAIN, VAL
from wave_analysis.evaluation.classification import classification_report
from wave_analysis.evaluation.sea_state import HEADLINE, confusion_svg, format_line, report
from wave_analysis.models.sea_state import (
    StillDataset,
    TrainSettings,
    build_model,
    build_transforms,
    class_weights,
    fit,
    input_size,
    predict,
)

PARTITIONS = (VAL, TEST, TEST_LATER)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--config", required=True)
    ap.add_argument("--package", required=True, help="package directory")
    ap.add_argument("--split", choices=("day", "year"), required=True)
    ap.add_argument("--out", required=True, help="run directory (created)")
    ap.add_argument("--backbone", help="override model.backbone")
    ap.add_argument("--input", choices=("sea_crop", "full_frame"), help="override model.input")
    ap.add_argument("--epochs", type=int)
    ap.add_argument("--batch-size", type=int)
    ap.add_argument("--lr", type=float)
    ap.add_argument("--num-workers", type=int)
    ap.add_argument(
        "--micro-batch",
        type=int,
        help="images per forward/backward pass; gradients are summed to the full batch "
        "(for a GPU with less memory; the update is unchanged)",
    )
    ap.add_argument("--amp", choices=("bf16", "fp16", "none"), help="override model.amp")
    ap.add_argument("--no-pretrained", action="store_true", help="random weights (offline tests)")
    ap.add_argument("--limit", type=int, help="stills per partition (smoke test)")
    ap.add_argument("--device", default="auto", help="auto, cuda, cuda:1 or cpu")
    ap.add_argument("--seed", type=int)
    args = ap.parse_args(argv)
    git_at_start = git_state(repo_root())

    cfg = load_yaml(args.config)
    settings = TrainSettings.from_config(
        cfg["model"],
        backbone=args.backbone,
        input=args.input,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        num_workers=args.num_workers,
        micro_batch=args.micro_batch,
        amp=args.amp,
        pretrained=False if args.no_pretrained else None,
    )
    seed = int(args.seed if args.seed is not None else cfg.get("seed", 0))
    torch.manual_seed(seed)
    # cuDNN's kernel autotuner on NVIDIA GPUs. On ROCm the same switch makes MIOpen
    # search kernels that it compiles at run time, and with PyTorch's ROCm wheels
    # that compile fails (its bundled compiler finds no C++ standard headers);
    # MIOpen's precompiled kernels work, and the desktop timings used them.
    torch.backends.cudnn.benchmark = torch.version.hip is None
    device = torch.device(
        ("cuda" if torch.cuda.is_available() else "cpu") if args.device == "auto" else args.device
    )
    if device.type == "cpu":
        settings.amp = "none"
    names = list(cfg["task"]["classes"]["names"])
    pkg = Path(args.package)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    log_fh = (out / "train.log").open("a", encoding="utf-8")

    def log(msg: str) -> None:
        line = f"{time.strftime('%Y-%m-%d %H:%M:%S')} {msg}"
        print(line, flush=True)
        log_fh.write(line + "\n")
        log_fh.flush()

    labels = pd.read_csv(pkg / "labels.csv")
    base = pd.read_csv(pkg / "baselines.csv")
    col = f"split_{args.split}"
    table = labels[labels[col].isin([TRAIN, *PARTITIONS])].reset_index(drop=True)
    if args.limit:
        rng = np.random.default_rng(seed)
        keep = np.zeros(len(table), bool)
        for part in table[col].unique():
            idx = np.flatnonzero(table[col].to_numpy() == part)
            keep[rng.choice(idx, min(args.limit, idx.size), replace=False)] = True
        table = table.loc[keep].reset_index(drop=True)
    parts = {p: table[table[col] == p].reset_index(drop=True) for p in (TRAIN, *PARTITIONS)}
    frame = (int(table["height_px"].iloc[0]), int(table["width_px"].iloc[0]))
    size = input_size(settings, frame)
    log(
        f"{cfg['name']} split={args.split} device={device} backbone={settings.backbone} "
        f"input={settings.input} {size[0]}x{size[1]} px; "
        + ", ".join(f"{p} {len(t):,}" for p, t in parts.items())
    )

    def loader(t: pd.DataFrame, train: bool) -> DataLoader[Any]:
        ds = StillDataset(pkg, t, settings, build_transforms(settings, size, train=train))
        return DataLoader(
            ds,
            batch_size=settings.batch_size,
            shuffle=train,
            drop_last=train and len(t) > settings.batch_size,
            num_workers=settings.num_workers,
            pin_memory=device.type == "cuda",
            persistent_workers=settings.num_workers > 0,
        )

    k = len(names)
    weights = (
        class_weights(parts[TRAIN]["class"], k)
        if settings.class_weights == "inverse_frequency"
        else torch.ones(k)
    )
    log(
        "class weights: "
        + ", ".join(f"{n} {w:.2f}" for n, w in zip(names, weights.tolist(), strict=True))
    )

    def score(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
        rep = classification_report(y_true, y_pred, n_classes=k)
        return {m: float(rep[m]) for m in HEADLINE[:4]}  # type: ignore[arg-type]

    model = build_model(settings.backbone, k, pretrained=settings.pretrained)
    result = fit(
        model,
        loader(parts[TRAIN], True),
        loader(parts[VAL], False),
        parts[VAL]["class"].to_numpy(),
        settings=settings,
        weights=weights,
        device=device,
        score=score,
        log=log,
    )
    log(f"chosen epoch {result.best_epoch} ({settings.select_on} on val {result.best_score:.3f})")
    model.load_state_dict(result.state_dict)
    torch.save(
        {
            "state_dict": result.state_dict,
            "settings": settings.__dict__,
            "class_names": names,
            "input_size_hw": size,
            "best_epoch": result.best_epoch,
        },
        out / "best.pt",
    )
    pd.DataFrame(result.history).to_csv(out / "history.csv", index=False)

    boot = cfg["evaluation"]["bootstrap"]
    rows, metrics, lines = [], {}, []
    for p in PARTITIONS:
        t = parts[p]
        if t.empty:
            continue
        pred, prob = predict(model, loader(t, False), device, amp=settings.amp)
        frame_p = pd.DataFrame(
            {
                "sample_id": t["sample_id"],
                "partition": p,
                "local_day": t["local_day"],
                "Hs_label_m": t["Hs_label_m"],
                "near_boundary": t["near_boundary"],
                "true": t["class"],
                "pred": pred,
            }
        )
        for j, nm in enumerate(names):
            frame_p[f"p_{nm}"] = prob[:, j]
        rows.append(frame_p)
        b = base.set_index("sample_id").loc[t["sample_id"]]
        kw: dict[str, Any] = {
            "names": names,
            "n_boot": int(boot["n_boot"]),
            "alpha": float(boot["alpha"]),
            "seed": int(boot["seed"]),
        }
        y, g, nb = t["class"].to_numpy(), t["local_day"].to_numpy(), t["near_boundary"].to_numpy()
        entry: dict[str, Any] = {"model": report(y, pred, groups=g, near_boundary=nb, **kw)}
        lines.append(format_line(f"{p:<10} model", entry["model"]))
        for bname in sorted(
            c.rsplit("_", 1)[0] for c in base.columns if c.endswith(f"_{args.split}")
        ):
            bp = b[f"{bname}_{args.split}"].to_numpy()
            ok = bp >= 0
            if not ok.any():
                continue
            entry[bname] = {
                "coverage": float(ok.mean()),
                "baseline": report(y[ok], bp[ok], groups=g[ok], near_boundary=nb[ok], **kw),
                "model_same_stills": report(
                    y[ok], pred[ok], groups=g[ok], near_boundary=nb[ok], **kw
                ),
            }
            lines.append(
                format_line(
                    f"{p:<10} {bname} ({ok.mean():.0%} of stills)", entry[bname]["baseline"]
                )
            )
            lines.append(
                format_line(f"{p:<10} model, same stills", entry[bname]["model_same_stills"])
            )
            if bname == "wis":
                (out / f"confusion_{p}_wis.svg").write_text(
                    confusion_svg(
                        entry[bname]["baseline"]["confusion_matrix"],
                        names,
                        title=f"{cfg['name']} {args.split}/{p}: WIS 44056 hindcast (corrected)",
                    )
                )
        metrics[p] = entry
        cm = entry["model"]["confusion_matrix"]
        pd.DataFrame(
            cm, index=[f"true_{n}" for n in names], columns=[f"pred_{n}" for n in names]
        ).to_csv(out / f"confusion_{p}.csv")
        (out / f"confusion_{p}.svg").write_text(
            confusion_svg(
                cm,
                names,
                title=f"{cfg['name']} {args.split}/{p}: {settings.backbone}, {settings.input}",
            )
        )
    pd.concat(rows).to_csv(out / "predictions.csv", index=False, float_format="%.5f")
    run_info = {
        "config": cfg["name"],
        "split": args.split,
        "settings": settings.__dict__,
        "input_size_hw": size,
        "best_epoch": result.best_epoch,
        "device": str(device),
        "gpu": torch.cuda.get_device_name(device) if device.type == "cuda" else None,
        "torch": torch.__version__,
        "hip": torch.version.hip,
        "limit_per_partition": args.limit,
        "seed": seed,
    }
    with (out / "metrics.json").open("w", encoding="utf-8") as fh:
        json.dump({"run": run_info, "partitions": metrics}, fh, indent=1, default=str)
    (out / "report.md").write_text(
        f"# {cfg['name']}, split {args.split}\n\n```\n" + "\n".join(lines) + "\n```\n",
        encoding="utf-8",
    )
    for line in lines:
        log(line)
    inputs = [pkg / "labels.csv", pkg / "baselines.csv", pkg / "SHA256SUMS", Path(args.config)]
    outputs = [out / f for f in ("predictions.csv", "metrics.json", "history.csv", "best.pt")]
    write_provenance(
        build_provenance(
            artifact_id=f"{cfg['name']}_{args.split}_{settings.backbone}_{settings.input}",
            inputs=[p for p in inputs if p.exists()],
            outputs=outputs,
            config=cfg,
            notes=f"torch {torch.__version__}; {run_info['gpu'] or platform.processor() or 'cpu'}",
            git=git_at_start,
        ),
        out / "provenance.yaml",
    )
    log(f"done: {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
