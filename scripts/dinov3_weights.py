#!/usr/bin/env python
"""DINOv3 ViT-B/16 weights between timm (the training scripts) and DINOv3's trainer.

    python scripts/dinov3_weights.py to-dinov3 --out init.pth           # timm's public weights -> trainer start
    python scripts/dinov3_weights.py to-timm eval/2500/teacher_checkpoint.pth --out timm_2500.pth
    python scripts/dinov3_weights.py check [--timm-weights timm_2500.pth]   # needs dinov3 importable

``to-dinov3`` writes ``{"teacher": {"backbone.<name>": tensor}}`` from timm's
``vit_base_patch16_dinov3.lvd1689m`` (the weights of Bar B3), for
``student.resume_from_teacher_chkpt``. ``to-timm`` writes the teacher backbone of a
DINOv3 checkpoint as a timm state dict, for ``train_wave_heights.py
--backbone-weights``. ``check`` builds both models from the same weights and
compares their averaged patch features on random 272 x 480 inputs (the B3 view
size); it fails above ``--tol``. Names: :mod:`wave_analysis.pretraining.dinov3_weights`.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch

from wave_analysis.pretraining.dinov3_weights import (
    VITB16_KWARGS,
    load_teacher_backbone,
    teacher_checkpoint,
    timm_to_dinov3,
)

TIMM_NAME = "vit_base_patch16_dinov3.lvd1689m"


def timm_model(weights: Path | None = None) -> torch.nn.Module:
    """The B3 backbone (``MultiViewRegressor``'s settings); ``weights`` replaces the public ones."""
    import timm

    net = timm.create_model(
        TIMM_NAME, pretrained=weights is None, num_classes=0, dynamic_img_size=True,
        dynamic_img_pad=True,
    )  # fmt: skip
    if weights is not None:
        net.load_state_dict(torch.load(weights, map_location="cpu", weights_only=True), strict=True)
    return net.eval()


def check(weights: Path | None, tol: float, device: str) -> float:
    """Largest difference between timm's and DINOv3's averaged patch features."""
    from dinov3.models.vision_transformer import vit_base

    t = timm_model(weights).to(device)
    d = vit_base(
        img_size=224, device="cpu", **{k: v for k, v in VITB16_KWARGS.items() if k != "patch_size"}
    )
    d.init_weights()
    missing, unexpected = d.load_state_dict(timm_to_dinov3(t.state_dict()), strict=False)
    if unexpected or set(missing) - {"mask_token", "rope_embed.periods"}:
        raise SystemExit(f"DINOv3 model: missing {missing}, unexpected {unexpected}")
    d = d.to(device).eval()
    gen = torch.Generator().manual_seed(0)
    x = torch.randn(2, 3, 272, 480, generator=gen).to(device)
    with torch.no_grad():
        a = d.forward_features(x)["x_norm_patchtokens"].mean(dim=1)
        b = t(x)
    diff = float((a - b).abs().max())
    print(f"max |DINOv3 - timm| = {diff:.2e} (features of size {float(b.abs().mean()):.3f})")
    if diff > tol:
        raise SystemExit(f"the two models differ by more than {tol}")
    return diff


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    a = sub.add_parser("to-dinov3", help="timm's public ViT-B/16 -> a trainer start checkpoint")
    a.add_argument("--out", required=True)
    b = sub.add_parser("to-timm", help="a DINOv3 teacher checkpoint -> a timm state dict")
    b.add_argument("checkpoint")
    b.add_argument("--out", required=True)
    c = sub.add_parser("check", help="timm and DINOv3 give the same features")
    c.add_argument("--timm-weights", help="a timm state dict (default: the public weights)")
    c.add_argument("--tol", type=float, default=1e-4)
    c.add_argument("--device", default="cpu")
    args = ap.parse_args(argv)

    if args.cmd == "to-dinov3":
        out = Path(args.out)
        out.parent.mkdir(parents=True, exist_ok=True)
        torch.save(teacher_checkpoint(timm_model().state_dict()), out)
        print(f"wrote {out}")
    elif args.cmd == "to-timm":
        out = Path(args.out)
        out.parent.mkdir(parents=True, exist_ok=True)
        torch.save(load_teacher_backbone(args.checkpoint), out)
        print(f"wrote {out}")
    else:
        check(Path(args.timm_weights) if args.timm_weights else None, args.tol, args.device)
    return 0


if __name__ == "__main__":
    sys.exit(main())
