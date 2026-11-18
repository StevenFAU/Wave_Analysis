"""Wave heights in metres from a buoy-camera strip: total, wind-wave and swell height at once.

The classifier (:mod:`wave_analysis.models.sea_state`) gives a sea-state class;
this model gives the heights themselves, so that every published measure can be
computed from one prediction (an RMSE in metres, a correlation, any set of
classes or thresholds). It reuses the classifier's view handling.

* :class:`HeightsDataset` - strips as ``((views, mask, aux), targets)``: the six
  views, which are usable, optional per-view extra inputs (the horizon's roll
  and pitch, :mod:`wave_analysis.processing.horizon`), and the log heights
  (NaN where a height is missing).
* :class:`MultiViewRegressor` - one backbone per view; a view's extra inputs
  join its image feature; a one-layer transformer over the usable views (the
  views are separate moments of the buoy's motion, so they can inform each
  other, as the frames of Yang et al. 2026); the masked mean goes to one output
  per height.
* :func:`fit_heights` / :func:`predict_heights` - a masked Huber loss on
  standardised log heights; the epoch with the lowest validation RMSE of the
  first height (the total) is kept.
"""

from __future__ import annotations

import math
import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from PIL import Image
from torch import nn
from torch.utils.data import DataLoader, Dataset

from wave_analysis.models.sea_state import (
    TrainSettings,
    _autocast,
    _pieces,
    _to_device,
    build_model,
    lr_lambda,
)

#: Heights below this are set to it before the logarithm (calm seas; NDBC reports 0.0-0.1 m).
FLOOR_M = 0.05


def log_heights(values: np.ndarray) -> np.ndarray:
    """Natural log of heights in metres, floored at :data:`FLOOR_M`; NaN stays NaN."""
    v = np.asarray(values, dtype=np.float64)
    return np.where(np.isfinite(v), np.log(np.maximum(v, FLOOR_M)), np.nan)


class HeightsDataset(Dataset[tuple[tuple[torch.Tensor, ...], torch.Tensor]]):
    """Strips as ``((views, mask[, aux]), log_heights)``; ``aux`` is ``(n_views, aux_dim)`` per strip."""

    def __init__(
        self,
        root: Path,
        table: pd.DataFrame,
        settings: TrainSettings,
        transform: Callable[[Image.Image], torch.Tensor],
        masks: np.ndarray,
        targets: np.ndarray,
        aux: np.ndarray | None = None,
    ) -> None:
        n = len(table)
        if masks.shape != (n, settings.n_views):
            raise ValueError(f"masks must be ({n}, {settings.n_views}), got {masks.shape}")
        if targets.shape[0] != n:
            raise ValueError("one row of targets per strip")
        if aux is not None and aux.shape[:2] != (n, settings.n_views):
            raise ValueError(f"aux must be ({n}, {settings.n_views}, k), got {aux.shape}")
        self.root = Path(root)
        self.paths = table["image"].tolist()
        self.masks = torch.as_tensor(masks, dtype=torch.bool)
        self.targets = torch.as_tensor(targets, dtype=torch.float32)
        self.aux = None if aux is None else torch.as_tensor(aux, dtype=torch.float32)
        self.n_views = settings.n_views
        self.transform = transform

    def __len__(self) -> int:
        return len(self.paths)

    def __getitem__(self, i: int) -> tuple[tuple[torch.Tensor, ...], torch.Tensor]:
        with Image.open(self.root / self.paths[i]) as im:
            strip = im.convert("RGB")
        w, h = strip.size
        step = w // self.n_views
        views = torch.stack(
            [
                self.transform(strip.crop((k * step, 0, (k + 1) * step, h)))
                for k in range(self.n_views)
            ]
        )
        x = (views, self.masks[i]) if self.aux is None else (views, self.masks[i], self.aux[i])
        return x, self.targets[i]


class MultiViewRegressor(nn.Module):
    """Per-view backbone (+ per-view extra inputs) -> transformer over usable views -> masked mean -> heights."""

    def __init__(
        self,
        backbone: str,
        n_outputs: int,
        *,
        aux_dim: int = 0,
        d_model: int = 256,
        heads: int = 4,
        layers: int = 1,
        dropout: float = 0.1,
        pretrained: bool = True,
    ) -> None:
        super().__init__()
        if backbone.startswith("timm:"):
            # Any timm backbone, e.g. "timm:convnext_tiny.dinov3_lvd1689m" (DINOv3) or
            # "timm:vit_base_patch14_reg4_dinov2.lvd142m" (DINOv2); ViTs take any view size.
            import timm

            name = backbone.removeprefix("timm:")
            kw = (
                {"dynamic_img_size": True, "dynamic_img_pad": True}
                if name.startswith(("vit", "eva"))
                else {}
            )
            net = timm.create_model(name, pretrained=pretrained, num_classes=0, **kw)
            feat = int(net.num_features)
        else:
            net = build_model(backbone, 1, pretrained=pretrained)
        if backbone.startswith("timm:"):
            pass
        elif backbone == "convnext_tiny":
            seq = getattr(net, "classifier")  # noqa: B009
            feat = seq[2].in_features
            seq[2] = nn.Identity()
        else:
            feat = getattr(net, "fc").in_features  # noqa: B009
            setattr(net, "fc", nn.Identity())  # noqa: B010
        self.encoder = net
        self.aux_dim = aux_dim
        self.aux_mlp = (
            nn.Sequential(nn.Linear(aux_dim, 32), nn.GELU(), nn.Linear(32, 32)) if aux_dim else None
        )
        self.proj = nn.Linear(feat + (32 if aux_dim else 0), d_model)
        enc = nn.TransformerEncoderLayer(
            d_model, heads, dim_feedforward=2 * d_model, dropout=dropout, batch_first=True
        )
        self.mixer = nn.TransformerEncoder(enc, layers) if layers else None
        self.head = nn.Sequential(nn.LayerNorm(d_model), nn.Linear(d_model, n_outputs))

    def forward(self, x: tuple[torch.Tensor, ...]) -> torch.Tensor:
        """Standardised log heights ``(B, n_outputs)``."""
        views, mask = x[0], x[1]
        b, v = views.shape[:2]
        flat = views.flatten(0, 1).contiguous(memory_format=torch.channels_last)
        f = self.encoder(flat).reshape(b, v, -1)
        if self.aux_mlp is not None:
            if len(x) < 3:
                raise ValueError("this model takes per-view extra inputs")
            f = torch.cat([f, self.aux_mlp(x[2].to(f.dtype))], dim=-1)
        h = self.proj(f)
        if self.mixer is not None:
            h = self.mixer(h, src_key_padding_mask=~mask)
        m = mask.to(h.dtype).unsqueeze(-1)
        pooled = (h * m).sum(dim=1) / m.sum(dim=1).clamp_min(1.0)
        out: torch.Tensor = self.head(pooled)
        return out


@dataclass
class HeightsFit:
    """Outcome of :func:`fit_heights`."""

    best_epoch: int
    best_rmse_m: float
    history: list[dict[str, float]]
    state_dict: dict[str, torch.Tensor]


@torch.no_grad()
def predict_heights(
    model: nn.Module,
    loader: DataLoader[Any],
    device: torch.device,
    *,
    mean: np.ndarray,
    std: np.ndarray,
    amp: str = "none",
) -> np.ndarray:
    """Heights in metres ``(N, n_outputs)`` for every strip of ``loader``, in order."""
    model.eval()
    out = []
    for x, _ in loader:
        x = _to_device(x, device)
        with _autocast(device, amp):
            z = model(x)
        out.append(z.float().cpu().numpy())
    z = np.concatenate(out) if out else np.zeros((0, len(mean)))
    return np.exp(z * std + mean)


def fit_heights(
    model: nn.Module,
    train_loader: DataLoader[Any],
    val_loader: DataLoader[Any],
    val_true_m: np.ndarray,
    *,
    settings: TrainSettings,
    mean: np.ndarray,
    std: np.ndarray,
    device: torch.device,
    huber_delta: float = 1.0,
    log: Callable[[str], object] = print,
    names: Sequence[str] = ("hs", "wind_wave", "swell"),
    lp_epochs: int = 0,
) -> HeightsFit:
    """Train for ``settings.epochs``; keep the epoch with the lowest validation RMSE of the first height.

    ``lp_epochs`` > 0: the backbone (``model.encoder``) stays frozen for the first
    ``lp_epochs`` epochs and only the rest learns (linear probing, then fine-tuning:
    Kumar et al. 2022's LP-FT, which keeps the pretrained features better under shift).
    """
    model.to(device).to(memory_format=torch.channels_last)
    opt = torch.optim.AdamW(model.parameters(), lr=settings.lr, weight_decay=settings.weight_decay)
    steps = settings.epochs * max(1, len(train_loader))
    warm = settings.warmup_epochs * max(1, len(train_loader))
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: lr_lambda(s, warmup=warm, total=steps))
    mu = torch.as_tensor(mean, dtype=torch.float32, device=device)
    sd = torch.as_tensor(std, dtype=torch.float32, device=device)
    loss_fn = nn.HuberLoss(reduction="none", delta=huber_delta)
    piece = settings.micro_batch or settings.batch_size
    best = (math.inf, -1)
    best_state: dict[str, torch.Tensor] = {}
    history: list[dict[str, float]] = []
    encoder = getattr(model, "encoder", None)
    for epoch in range(1, settings.epochs + 1):
        model.train()
        if encoder is not None and lp_epochs:
            frozen = epoch <= lp_epochs
            for prm in encoder.parameters():
                prm.requires_grad_(not frozen)
            if frozen:
                encoder.eval()  # also keeps its normalisation statistics fixed
        t0, total, n = time.monotonic(), 0.0, 0
        for x, y in train_loader:
            x = _to_device(x, device)
            y = y.to(device, non_blocking=True)
            z = (y - mu) / sd
            ok = torch.isfinite(z)
            denom = ok.sum().clamp_min(1)
            opt.zero_grad(set_to_none=True)
            batch_loss = torch.zeros((), device=device)
            for xs, zs, oks in zip(_pieces(x, piece), z.split(piece), ok.split(piece), strict=True):
                with _autocast(device, settings.amp):
                    pred = model(xs).float()
                loss = (loss_fn(pred, torch.nan_to_num(zs)) * oks).sum() / denom
                loss.backward()
                batch_loss += loss.detach()
            opt.step()
            sched.step()
            total += float(batch_loss) * len(y)
            n += len(y)
        pred_m = predict_heights(model, val_loader, device, mean=mean, std=std, amp=settings.amp)
        row: dict[str, float] = {
            "epoch": epoch,
            "train_loss": total / max(1, n),
            "seconds": time.monotonic() - t0,
        }
        for j, name in enumerate(names):
            ok_v = np.isfinite(val_true_m[:, j])
            row[f"val_rmse_{name}_m"] = float(
                np.sqrt(np.mean((pred_m[ok_v, j] - val_true_m[ok_v, j]) ** 2))
            )
        history.append(row)
        chosen = row[f"val_rmse_{names[0]}_m"]
        mark = ""
        if chosen < best[0]:
            best = (chosen, epoch)
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            mark = "  <- best"
        log(
            f"epoch {epoch:>2}/{settings.epochs}: loss {row['train_loss']:.4f}, "
            + ", ".join(f"val RMSE {nm} {row[f'val_rmse_{nm}_m']:.3f} m" for nm in names)
            + f", {row['seconds']:.0f} s{mark}"
        )
    return HeightsFit(best[1], best[0], history, best_state)


__all__ = [
    "FLOOR_M",
    "HeightsDataset",
    "HeightsFit",
    "MultiViewRegressor",
    "fit_heights",
    "log_heights",
    "predict_heights",
]
