"""Heights, periods, the wind-sea spectrum and the mean direction from one buoy-camera strip.

Stage 3 (rank 5 of the architecture review): the heights model
(:class:`~wave_analysis.models.wave_heights.MultiViewRegressor`, with its point and
distribution heads) gains three outputs.

* **Periods** join the heights as more log-scale targets of the same heads.
* **Wind-sea spectrum shape:** a head gives one logit per frequency bin; its softmax is
  each bin's share of the wind-sea energy (cross-entropy against the buoy's shape).
  Times the predicted wind-sea energy, (H_ww / 4)², it is a 1-D wind-sea spectrum.
* **Mean direction**, as cosine and sine of the direction the waves come from relative
  to the sun's azimuth. The caption carries no compass heading, so the sun, which the
  views show, is the only compass in the image. Unlike the heights, which pool the views
  as a set, this head gives each view a learned position embedding (the views keep a
  fixed order around the buoy) and pools them by learned attention. Loss: 1 - cos of
  the angle between prediction and truth.

The heights' path is unchanged: the new heads read the same tokens and add their losses
to the shared encoder, which is what "check H_s for harm" checks.
"""

from __future__ import annotations

import math
import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader

from wave_analysis.models.sea_state import TrainSettings, _autocast, _pieces, _to_device, lr_lambda
from wave_analysis.models.wave_heights import MultiViewRegressor, beta_nll, split_outputs


@dataclass(frozen=True)
class Layout:
    """Columns of a target row: ``n_scalar`` log values, ``n_bins`` shape shares, then cos and sin."""

    n_scalar: int
    n_bins: int
    direction: bool

    @property
    def width(self) -> int:
        """Columns of one target row."""
        return self.n_scalar + self.n_bins + (2 if self.direction else 0)


class MultiViewSpectral(MultiViewRegressor):
    """:class:`MultiViewRegressor` plus a wind-sea shape head and a direction head."""

    def __init__(
        self,
        backbone: str,
        n_scalar: int,
        *,
        n_bins: int = 46,
        direction: bool = True,
        n_views: int = 6,
        **kw: Any,
    ) -> None:
        super().__init__(backbone, n_scalar, distribution=True, **kw)
        d = self.proj.out_features
        self.n_bins, self.direction = n_bins, direction
        self.spec_head = nn.Sequential(nn.LayerNorm(d), nn.Linear(d, n_bins)) if n_bins else None
        if direction:
            self.view_embed = nn.Parameter(torch.randn(n_views, d) * 0.02)
            self.dir_mlp = nn.Sequential(nn.LayerNorm(d), nn.Linear(d, d), nn.GELU())
            self.dir_attn = nn.Linear(d, 1)
            self.dir_out = nn.Linear(d, 2)

    def forward(self, x: tuple[torch.Tensor, ...]) -> torch.Tensor:  # type: ignore[override]
        """``(B, 3 n_scalar + n_bins + 2)``: point, mean, raw variance, shape logits, cos/sin."""
        h, pooled = self.tokens(x)
        parts = [self.head(pooled)]
        assert self.dist_head is not None
        parts.append(self.dist_head(pooled))
        if self.spec_head is not None:
            parts.append(self.spec_head(pooled))
        if self.direction:
            u = self.dir_mlp(h + self.view_embed[: h.shape[1]].to(h.dtype))
            a = self.dir_attn(u).squeeze(-1).float().masked_fill(~x[1], float("-inf"))
            w = torch.softmax(a, dim=1).to(u.dtype).unsqueeze(-1)
            parts.append(self.dir_out((w * u).sum(dim=1)))
        return torch.cat(parts, dim=-1)


def split_spectral(
    out: torch.Tensor, layout: Layout
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor | None, torch.Tensor | None]:
    """Point, mean, variance (standardised logs), shape logits and the raw direction vector."""
    n = layout.n_scalar
    point, mu, var = split_outputs(out[:, : 3 * n], n)
    if mu is None or var is None:
        raise ValueError("the model has no distribution head")
    k = 3 * n
    logits = out[:, k : k + layout.n_bins] if layout.n_bins else None
    k += layout.n_bins
    direc = out[:, k : k + 2] if layout.direction else None
    return point, mu, var, logits, direc


def direction_loss(pred: torch.Tensor, cos_sin: torch.Tensor) -> torch.Tensor:
    """Per row, 1 - cos(angle between the predicted vector and the true unit vector)."""
    p = pred / pred.norm(dim=-1, keepdim=True).clamp_min(1e-6)
    return 1.0 - (p * cos_sin).sum(dim=-1)


@dataclass
class SpectralFit:
    """Outcome of :func:`fit_spectral`."""

    best_epoch: int
    best_rmse: float
    history: list[dict[str, float]]
    state_dict: dict[str, torch.Tensor]


@torch.no_grad()
def predict_spectral(
    model: nn.Module,
    loader: DataLoader[Any],
    device: torch.device,
    layout: Layout,
    *,
    mean: np.ndarray,
    std: np.ndarray,
    amp: str = "none",
) -> dict[str, np.ndarray]:
    """``point`` (natural units), ``mu_log``/``sigma_log``, ``shape`` (shares) and ``direction`` (unit cos, sin)."""
    model.eval()
    acc: dict[str, list[np.ndarray]] = {k: [] for k in ("point", "mu", "var", "shape", "dir")}
    for x, _ in loader:
        x = _to_device(x, device)
        with _autocast(device, amp):
            out = model(x)
        point, mu, var, logits, direc = split_spectral(out.float(), layout)
        acc["point"].append(point.cpu().numpy())
        acc["mu"].append(mu.cpu().numpy())
        acc["var"].append(var.cpu().numpy())
        if logits is not None:
            acc["shape"].append(torch.softmax(logits, dim=-1).cpu().numpy())
        if direc is not None:
            acc["dir"].append(
                (direc / direc.norm(dim=-1, keepdim=True).clamp_min(1e-6)).cpu().numpy()
            )
    res = {k: np.concatenate(v) for k, v in acc.items() if v}
    return {
        "point": np.exp(res["point"] * std + mean),
        "mu_log": res["mu"] * std + mean,
        "sigma_log": np.sqrt(res["var"]) * std,
        **({"shape": res["shape"]} if "shape" in res else {}),
        **({"direction": res["dir"]} if "dir" in res else {}),
    }


def fit_spectral(
    model: nn.Module,
    train_loader: DataLoader[Any],
    val_loader: DataLoader[Any],
    val_true: np.ndarray,
    layout: Layout,
    *,
    settings: TrainSettings,
    mean: np.ndarray,
    std: np.ndarray,
    device: torch.device,
    log: Callable[[str], object] = print,
    names: Sequence[str] = (),
    lp_epochs: int = 0,
    backbone_lr_scale: float = 1.0,
    dist_beta: float = 0.5,
    spectrum_weight: float = 1.0,
    direction_weight: float = 1.0,
    huber_delta: float = 1.0,
) -> SpectralFit:
    """As :func:`~wave_analysis.models.wave_heights.fit_heights`, with the shape and direction losses added.

    The scalar targets (``layout.n_scalar`` standardised logs) take the point head's
    Huber loss plus beta-NLL; the shape the cross-entropy of its softmax against the
    buoy's shares (rows with a shape); the direction ``direction_loss`` (rows with a
    direction). The epoch with the lowest validation RMSE of the first scalar (H_s) is kept.
    """
    model.to(device).to(memory_format=torch.channels_last)
    enc = getattr(model, "encoder", None)
    enc_ids = {id(q) for q in enc.parameters()} if enc is not None else set()
    groups = [
        {"params": [q for q in model.parameters() if id(q) in enc_ids], "lr": settings.lr * backbone_lr_scale},
        {"params": [q for q in model.parameters() if id(q) not in enc_ids], "lr": settings.lr},
    ]  # fmt: skip
    opt = torch.optim.AdamW([g for g in groups if g["params"]], weight_decay=settings.weight_decay)
    steps = settings.epochs * max(1, len(train_loader))
    warm = settings.warmup_epochs * max(1, len(train_loader))
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: lr_lambda(s, warmup=warm, total=steps))
    n = layout.n_scalar
    mu_t = torch.as_tensor(mean, dtype=torch.float32, device=device)
    sd_t = torch.as_tensor(std, dtype=torch.float32, device=device)
    huber = nn.HuberLoss(reduction="none", delta=huber_delta)
    piece = settings.micro_batch or settings.batch_size
    best = (math.inf, -1)
    best_state: dict[str, torch.Tensor] = {}
    history: list[dict[str, float]] = []
    for epoch in range(1, settings.epochs + 1):
        model.train()
        if enc is not None and lp_epochs:
            frozen = epoch <= lp_epochs
            for prm in enc.parameters():
                prm.requires_grad_(not frozen)
            if frozen:
                enc.eval()
        t0, total, count = time.monotonic(), 0.0, 0
        for x, y in train_loader:
            x = _to_device(x, device)
            y = y.to(device, non_blocking=True)
            z = (y[:, :n] - mu_t) / sd_t
            ok = torch.isfinite(z)
            denom = ok.sum().clamp_min(1)
            shp = y[:, n : n + layout.n_bins]
            ok_s = torch.isfinite(shp).all(dim=1)
            dirt = y[:, n + layout.n_bins :]
            ok_d = (
                torch.isfinite(dirt).all(dim=1)
                if layout.direction
                else torch.zeros(len(y), dtype=torch.bool, device=device)
            )
            n_s, n_d = ok_s.sum().clamp_min(1), ok_d.sum().clamp_min(1)
            opt.zero_grad(set_to_none=True)
            batch_loss = torch.zeros((), device=device)
            chunks = zip(_pieces(x, piece), z.split(piece), ok.split(piece), shp.split(piece),
                         ok_s.split(piece), dirt.split(piece), ok_d.split(piece), strict=True)  # fmt: skip
            for xs, zs, oks, ss, okss, ds, okds in chunks:
                with _autocast(device, settings.amp):
                    out = model(xs).float()
                point, mu, var, logits, direc = split_spectral(out, layout)
                zs0 = torch.nan_to_num(zs)
                loss = ((huber(point, zs0) + beta_nll(mu, var, zs0, dist_beta)) * oks).sum() / denom
                if logits is not None and okss.any():
                    ce = -(torch.nan_to_num(ss) * torch.log_softmax(logits, dim=-1)).sum(dim=-1)
                    loss = loss + spectrum_weight * (ce * okss).sum() / n_s
                if direc is not None and okds.any():
                    dl = direction_loss(direc, torch.nan_to_num(ds))
                    loss = loss + direction_weight * (dl * okds).sum() / n_d
                loss.backward()
                batch_loss += loss.detach()
            opt.step()
            sched.step()
            total += float(batch_loss) * len(y)
            count += len(y)
        pred = predict_spectral(
            model, val_loader, device, layout, mean=mean, std=std, amp=settings.amp
        )
        row: dict[str, float] = {"epoch": epoch, "train_loss": total / max(1, count),
                                 "seconds": time.monotonic() - t0}  # fmt: skip
        for j, name in enumerate(names[:n]):
            okv = np.isfinite(val_true[:, j])
            row[f"val_rmse_{name}"] = float(
                np.sqrt(np.mean((pred["point"][okv, j] - val_true[okv, j]) ** 2))
            )
        if "direction" in pred:
            t = val_true[:, n + layout.n_bins :]
            okv = np.isfinite(t).all(axis=1)
            cosang = np.clip((pred["direction"][okv] * t[okv]).sum(axis=1), -1, 1)
            row["val_mae_direction_deg"] = float(np.degrees(np.arccos(cosang)).mean())
        history.append(row)
        chosen = row[f"val_rmse_{names[0]}"]
        mark = ""
        if chosen < best[0]:
            best = (chosen, epoch)
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            mark = "  <- best"
        shown = [f"val_rmse_{nm}" for nm in names[:4]] + [k for k in row if k.endswith("_deg")]
        log(f"epoch {epoch:>2}/{settings.epochs}: loss {row['train_loss']:.4f}, "
            + ", ".join(f"{k.removeprefix('val_')} {row[k]:.3f}" for k in shown)
            + f", {row['seconds']:.0f} s{mark}")  # fmt: skip
    return SpectralFit(best[1], best[0], history, best_state)


__all__ = [
    "Layout",
    "MultiViewSpectral",
    "SpectralFit",
    "direction_loss",
    "fit_spectral",
    "predict_spectral",
    "split_spectral",
]
