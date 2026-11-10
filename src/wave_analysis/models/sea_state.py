"""Training a sea-state classifier on a package of stills (needs the ``ml`` extra: torch, torchvision).

The package comes from ``scripts/sea_state_package.py``: resized stills, a
labels table with the class and both splits of each still, and baseline
predictions. This module holds the pieces the training script puts together:

* :class:`StillDataset` reads stills and crops them to the configured input
  (the sea and surf band, or the full frame).
* :func:`build_transforms` - mild scale, rotation and colour changes for
  training; **no horizontal flip**: the camera is fixed, and a mirrored scene
  (shoreline, light direction, the beach on the other side) is not a scene it
  can see.
* :func:`build_model` - an ImageNet-pretrained torchvision backbone with a new
  four-class head.
* :func:`fit` - AdamW, a warm-up then cosine learning rate, class-weighted
  cross-entropy, bf16 autocast on a GPU, and selection of the epoch with the
  best validation score. The test partitions are predicted once, with that
  epoch.
"""

from __future__ import annotations

import math
import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from PIL import Image
from torch import nn
from torch.utils.data import DataLoader, Dataset

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


@dataclass
class TrainSettings:
    """The ``model`` section of an experiment config."""

    backbone: str = "convnext_tiny"
    pretrained: bool = True
    input: str = "sea_crop"
    sea_crop: tuple[float, float] = (0.27, 0.60)
    augment: dict[str, Any] = field(default_factory=dict)
    class_weights: str = "inverse_frequency"
    epochs: int = 15
    batch_size: int = 64
    lr: float = 3e-4
    weight_decay: float = 0.05
    warmup_epochs: int = 1
    select_on: str = "f1_macro"
    num_workers: int = 8
    amp: str = "bf16"

    @classmethod
    def from_config(cls, model: dict[str, Any], **overrides: Any) -> TrainSettings:
        """From the config's ``model`` mapping; ``overrides`` that are not ``None`` win."""
        known = {k: v for k, v in model.items() if k in cls.__dataclass_fields__}
        known.update({k: v for k, v in overrides.items() if v is not None})
        if "sea_crop" in known:
            known["sea_crop"] = tuple(float(x) for x in known["sea_crop"])
        return cls(**known)


def input_size(settings: TrainSettings, frame_hw: tuple[int, int]) -> tuple[int, int]:
    """``(height, width)`` of the network input for stills of ``frame_hw`` pixels."""
    h, w = frame_hw
    if settings.input == "full_frame":
        return h, w
    if settings.input == "sea_crop":
        top, bottom = settings.sea_crop
        return round(h * bottom) - round(h * top), w
    raise ValueError(f"unknown input {settings.input!r}: sea_crop or full_frame")


def crop(img: Image.Image, settings: TrainSettings) -> Image.Image:
    """The configured part of a still (rows between the ``sea_crop`` fractions)."""
    if settings.input == "full_frame":
        return img
    top, bottom = settings.sea_crop
    w, h = img.size
    return img.crop((0, round(h * top), w, round(h * bottom)))


def build_transforms(
    settings: TrainSettings, size_hw: tuple[int, int], *, train: bool
) -> Callable[[Image.Image], torch.Tensor]:
    """Image -> normalized tensor; with ``train``, the configured augmentations (never a flip)."""
    from torchvision.transforms import v2

    a = settings.augment
    if a.get("horizontal_flip"):
        raise ValueError(
            "horizontal_flip: the camera is fixed; a mirrored scene is not one it sees"
        )
    steps: list[Any] = [v2.ToImage()]
    if train:
        h, w = size_hw
        lo, hi = a.get("scale", (1.0, 1.0))
        ratio = w / h
        steps.append(
            v2.RandomResizedCrop(
                size_hw,
                scale=(float(lo), float(hi)),
                ratio=(ratio * 0.95, ratio * 1.05),
                antialias=True,
            )
        )
        if a.get("rotate_deg"):
            steps.append(v2.RandomRotation(float(a["rotate_deg"])))
        jitter = {k: float(a[k]) for k in ("brightness", "contrast", "saturation") if a.get(k)}
        if jitter:
            steps.append(v2.ColorJitter(**jitter))
    else:
        steps.append(v2.Resize(size_hw, antialias=True))
    steps += [v2.ToDtype(torch.float32, scale=True), v2.Normalize(IMAGENET_MEAN, IMAGENET_STD)]
    return v2.Compose(steps)  # type: ignore[no-any-return]


class StillDataset(Dataset[tuple[torch.Tensor, int]]):
    """Stills of a package: ``(image tensor, class)`` per row of ``table`` (columns ``image``, ``class``)."""

    def __init__(
        self,
        root: Path,
        table: pd.DataFrame,
        settings: TrainSettings,
        transform: Callable[[Image.Image], torch.Tensor],
    ) -> None:
        self.root = Path(root)
        self.paths = table["image"].tolist()
        self.labels = table["class"].astype(int).tolist()
        self.settings = settings
        self.transform = transform

    def __len__(self) -> int:
        return len(self.paths)

    def __getitem__(self, i: int) -> tuple[torch.Tensor, int]:
        with Image.open(self.root / self.paths[i]) as im:
            img = crop(im.convert("RGB"), self.settings)
        return self.transform(img), self.labels[i]


def build_model(name: str, n_classes: int, *, pretrained: bool = True) -> nn.Module:
    """A torchvision classifier with a new ``n_classes`` head (ImageNet weights if ``pretrained``)."""
    import torchvision.models as tvm

    if name == "convnext_tiny":
        net = tvm.convnext_tiny(weights=tvm.ConvNeXt_Tiny_Weights.DEFAULT if pretrained else None)
        head = net.classifier[2]
        assert isinstance(head, nn.Linear)
        net.classifier[2] = nn.Linear(head.in_features, n_classes)
        return net
    if name == "resnet50":
        res = tvm.resnet50(weights=tvm.ResNet50_Weights.DEFAULT if pretrained else None)
        res.fc = nn.Linear(res.fc.in_features, n_classes)
        return res
    if name == "resnet18":
        r18 = tvm.resnet18(weights=tvm.ResNet18_Weights.DEFAULT if pretrained else None)
        r18.fc = nn.Linear(r18.fc.in_features, n_classes)
        return r18
    raise ValueError(f"unknown backbone {name!r}: convnext_tiny, resnet50 or resnet18")


def class_weights(y: Sequence[int], n_classes: int) -> torch.Tensor:
    """Inverse class frequency of the training stills, scaled to mean 1 (absent classes: 0)."""
    counts = np.bincount(np.asarray(y, dtype=np.int64), minlength=n_classes).astype(float)
    w = np.divide(counts.sum() / n_classes, counts, out=np.zeros_like(counts), where=counts > 0)
    w *= n_classes / w.sum() if w.sum() else 1.0
    return torch.tensor(w, dtype=torch.float32)


def lr_lambda(step: int, *, warmup: int, total: int) -> float:
    """Linear warm-up to 1, then cosine decay to 0 (per optimizer step)."""
    if step < warmup:
        return (step + 1) / max(1, warmup)
    progress = (step - warmup) / max(1, total - warmup)
    return 0.5 * (1 + math.cos(math.pi * min(1.0, progress)))


@torch.no_grad()
def predict(
    model: nn.Module, loader: DataLoader[Any], device: torch.device, *, amp: str = "none"
) -> tuple[np.ndarray, np.ndarray]:
    """``(predicted class, class probabilities)`` for every still of ``loader``, in order."""
    model.eval()
    probs = []
    for x, _ in loader:
        x = x.to(device, non_blocking=True).to(memory_format=torch.channels_last)
        with _autocast(device, amp):
            logits = model(x)
        probs.append(torch.softmax(logits.float(), dim=1).cpu().numpy())
    p = np.concatenate(probs) if probs else np.zeros((0, 0))
    return p.argmax(axis=1) if p.size else np.zeros(0, int), p


def _autocast(device: torch.device, amp: str) -> Any:
    if device.type == "cuda" and amp == "bf16":
        return torch.autocast("cuda", dtype=torch.bfloat16)
    if device.type == "cuda" and amp == "fp16":
        return torch.autocast("cuda", dtype=torch.float16)
    return torch.autocast(device.type, enabled=False)


@dataclass
class FitResult:
    """Outcome of :func:`fit`."""

    best_epoch: int
    best_score: float
    history: list[dict[str, float]]
    state_dict: dict[str, torch.Tensor]


def fit(
    model: nn.Module,
    train_loader: DataLoader[Any],
    val_loader: DataLoader[Any],
    val_true: np.ndarray,
    *,
    settings: TrainSettings,
    weights: torch.Tensor,
    device: torch.device,
    score: Callable[[np.ndarray, np.ndarray], dict[str, float]],
    log: Callable[[str], object] = print,
) -> FitResult:
    """Train for ``settings.epochs`` and keep the weights of the epoch with the best validation score.

    ``score(y_true, y_pred)`` returns metrics; ``settings.select_on`` names the
    one that chooses the epoch.
    """
    model.to(device).to(memory_format=torch.channels_last)
    opt = torch.optim.AdamW(model.parameters(), lr=settings.lr, weight_decay=settings.weight_decay)
    steps = settings.epochs * max(1, len(train_loader))
    warm = settings.warmup_epochs * max(1, len(train_loader))
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: lr_lambda(s, warmup=warm, total=steps))
    loss_fn = nn.CrossEntropyLoss(weight=weights.to(device))
    best = (-math.inf, -1)
    best_state: dict[str, torch.Tensor] = {}
    history: list[dict[str, float]] = []
    for epoch in range(1, settings.epochs + 1):
        model.train()
        t0, total, n = time.monotonic(), 0.0, 0
        for x, y in train_loader:
            x = x.to(device, non_blocking=True).to(memory_format=torch.channels_last)
            y = y.to(device, non_blocking=True)
            with _autocast(device, settings.amp):
                loss = loss_fn(model(x), y)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            sched.step()
            total += float(loss.detach()) * len(y)
            n += len(y)
        pred, _ = predict(model, val_loader, device, amp=settings.amp)
        metrics = score(val_true, pred)
        row = {"epoch": epoch, "train_loss": total / max(1, n), "seconds": time.monotonic() - t0}
        row.update({f"val_{k}": float(v) for k, v in metrics.items()})
        history.append(row)
        chosen = float(metrics[settings.select_on])
        mark = ""
        if chosen > best[0]:
            best = (chosen, epoch)
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            mark = "  <- best"
        log(
            f"epoch {epoch:>2}/{settings.epochs}: loss {row['train_loss']:.4f}, "
            + ", ".join(f"val {k} {v:.3f}" for k, v in metrics.items())
            + f", {row['seconds']:.0f} s{mark}"
        )
    return FitResult(best[1], best[0], history, best_state)


__all__ = [
    "IMAGENET_MEAN",
    "IMAGENET_STD",
    "FitResult",
    "StillDataset",
    "TrainSettings",
    "build_model",
    "build_transforms",
    "class_weights",
    "crop",
    "fit",
    "input_size",
    "lr_lambda",
    "predict",
]
