"""Frozen-encoder features and small trained heads for shore-camera clips.

The pieces of ``scripts/train_video_probe.py`` that can be tested without weights:
how a clip's frames become encoder input (two square tiles, the arms' frame orders),
how encoder tokens are pooled to a fixed grid, and the two heads (an attentive probe
and an MLP on mean tokens). Choices and reasons: ``docs/experiments/webcoos_video.md``.
"""

from __future__ import annotations

import copy
import hashlib

import numpy as np
import torch
from torch import nn

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)
VIDEO_ARMS = ("video", "still", "shuffled")


def square_tiles(frames: np.ndarray) -> np.ndarray:
    """The left and right squares of frames ``[T, H, W, 3]`` (W >= H): ``[2, T, H, H, 3]``.

    Together they cover the whole frame for any aspect up to 2:1 (they overlap in the
    middle), so no camera is stretched or cut, and each is a square, which V-JEPA 2's
    position code needs.
    """
    h, w = frames.shape[1], frames.shape[2]
    if w < h:
        raise ValueError(f"frames narrower than high ({w} x {h})")
    return np.stack([frames[:, :, :h], frames[:, :, w - h :]])


def arm_order(arm: str, n: int, clip_id: str) -> np.ndarray:
    """Frame indices for an arm.

    ``video`` in order, ``still`` the middle frame n times, ``shuffled`` a fixed random
    order per clip (seeded by the clip's name, so reruns agree).
    """
    if arm == "video":
        return np.arange(n)
    if arm == "still":
        return np.full(n, n // 2)
    if arm == "shuffled":
        seed = int(hashlib.sha256(clip_id.encode()).hexdigest()[:8], 16)
        return np.random.default_rng(seed).permutation(n)
    raise ValueError(f"unknown arm {arm!r}")


def normalise(x: torch.Tensor) -> torch.Tensor:
    """uint8 ``[..., 3, H, W]`` to ImageNet-normalised float."""
    mean = torch.tensor(IMAGENET_MEAN, device=x.device).view(3, 1, 1)
    std = torch.tensor(IMAGENET_STD, device=x.device).view(3, 1, 1)
    return (x.float() / 255.0 - mean) / std


def pool_tokens(
    tokens: torch.Tensor, grid: tuple[int, int, int], out: tuple[int, int, int]
) -> torch.Tensor:
    """Average-pool ``[B, t*h*w, D]`` tokens on their (t, h, w) grid to ``out``: ``[B, prod(out), D]``."""
    b, n, d = tokens.shape
    t, h, w = grid
    if n != t * h * w:
        raise ValueError(f"{n} tokens is not a {grid} grid")
    x = tokens.reshape(b, t, h, w, d).permute(0, 4, 1, 2, 3)
    x = nn.functional.adaptive_avg_pool3d(x.float(), out)
    return x.flatten(2).transpose(1, 2)


class AttentiveProbe(nn.Module):
    """An attentive probe (V-JEPA 2's, smaller).

    One self-attention block over the pooled tokens of all tiles, then a learned query
    reads them out. Input ``[B, tiles, layers, tokens, D]``.
    """

    def __init__(self, n_layers: int, dim: int, n_tokens: int, n_out: int,
                 d: int = 256, heads: int = 4, dropout: float = 0.2) -> None:  # fmt: skip
        super().__init__()
        self.proj = nn.ModuleList(
            nn.Sequential(nn.LayerNorm(dim), nn.Linear(dim, d)) for _ in range(n_layers)
        )
        self.pos = nn.Parameter(torch.randn(n_tokens, d) * 0.02)
        self.block = nn.TransformerEncoderLayer(
            d, heads, 2 * d, dropout, batch_first=True, norm_first=True
        )
        self.query = nn.Parameter(torch.randn(1, 1, d) * 0.02)
        self.cross = nn.MultiheadAttention(d, heads, dropout=dropout, batch_first=True)
        self.norm = nn.LayerNorm(d)
        self.head = nn.Sequential(nn.Dropout(dropout), nn.Linear(d, n_out))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """``[B, tiles, layers, tokens, D]`` to ``[B, n_out]``."""
        b, p, n_layers, t, _ = x.shape
        z = torch.stack([self.proj[i](x[:, :, i].float()) for i in range(n_layers)]).mean(0)
        z = self.block(z.reshape(b, p * t, -1) + self.pos)
        o, _ = self.cross(self.query.expand(b, -1, -1), z, z)
        return self.head(self.norm(o[:, 0]))


class MeanProbe(nn.Module):
    """An MLP on each layer's mean token (over tiles and tokens), the layers concatenated."""

    def __init__(
        self, n_layers: int, dim: int, n_out: int, d: int = 256, dropout: float = 0.2
    ) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(n_layers * dim), nn.Linear(n_layers * dim, d), nn.GELU(),
            nn.Dropout(dropout), nn.Linear(d, n_out),
        )  # fmt: skip

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """``[B, tiles, layers, tokens, D]`` to ``[B, n_out]``."""
        return self.net(x.float().mean(dim=(1, 3)).flatten(1))


# --- Whole clips (bars M3-M5): K non-overlapping 16-s windows per clip -----------------


def midpoint_window(duration_s: float, k: int, span_s: float = 16.0) -> int:
    """Index of the window (counted from the clip's start) that holds the clip's midpoint."""
    if k < 1:
        raise ValueError("a clip needs at least one window")
    return int(min(k - 1, np.floor(duration_s / 2 / span_s)))


def centred_windows(mid: int, k: int, n: int = 11) -> tuple[int, int]:
    """``[lo, hi)``: the ``n`` windows centred on window ``mid`` (Kim's 180 s: 11 x 16 s = 176 s).

    Shifted to stay inside the clip; all ``k`` windows when ``k < n``.
    """
    lo = int(np.clip(mid - n // 2, 0, max(0, k - n)))
    return lo, min(k, lo + n)


def energy_mean(h: np.ndarray, axis: int | None = None) -> np.ndarray:
    """sqrt(mean(H_s^2)): the average of heights that averages their energy."""
    return np.sqrt(np.mean(np.square(h), axis=axis))


def geometric_mean(t: np.ndarray, axis: int | None = None) -> np.ndarray:
    """exp(mean(log T)): the average of periods (of positive quantities in general)."""
    return np.exp(np.mean(np.log(t), axis=axis))


class RecordPooler(nn.Module):
    """A learned record pooler (``pooled``): one attention block over a clip's windows.

    Input ``[B, W, tiles, layers, D]``: each window's mean token per tile and layer (the
    frozen encoder's cached tokens averaged), with a mask ``[B, W]`` of real windows.
    Each layer is projected to ``d`` and the layers averaged (as in
    :class:`AttentiveProbe`); every (window, tile) is one token, tagged with learned
    window-position and tile embeddings (EchoJEPA tags clips the same way); one
    self-attention block, then a learned query reads the record out.
    """

    def __init__(self, n_layers: int, dim: int, n_out: int, max_windows: int = 64, n_tiles: int = 2,
                 d: int = 256, heads: int = 4, dropout: float = 0.2) -> None:  # fmt: skip
        super().__init__()
        self.proj = nn.ModuleList(
            nn.Sequential(nn.LayerNorm(dim), nn.Linear(dim, d)) for _ in range(n_layers)
        )
        self.pos = nn.Parameter(torch.randn(max_windows, 1, d) * 0.02)
        self.tile = nn.Parameter(torch.randn(1, n_tiles, d) * 0.02)
        self.block = nn.TransformerEncoderLayer(
            d, heads, 2 * d, dropout, batch_first=True, norm_first=True
        )
        self.query = nn.Parameter(torch.randn(1, 1, d) * 0.02)
        self.cross = nn.MultiheadAttention(d, heads, dropout=dropout, batch_first=True)
        self.norm = nn.LayerNorm(d)
        self.head = nn.Sequential(nn.Dropout(dropout), nn.Linear(d, n_out))

    def forward(self, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """``[B, W, tiles, layers, D]`` and ``[B, W]`` (True = a real window) to ``[B, n_out]``."""
        b, w, p, n_layers, _ = x.shape
        z = torch.stack([self.proj[i](x[:, :, :, i].float()) for i in range(n_layers)]).mean(0)
        z = (z + self.pos[:w] + self.tile).reshape(b, w * p, -1)
        pad = ~mask.repeat_interleave(p, dim=1)
        z = self.block(z, src_key_padding_mask=pad)
        o, _ = self.cross(self.query.expand(b, -1, -1), z, z, key_padding_mask=pad)
        return self.head(self.norm(o[:, 0]))


class StackedProbes:
    """``n`` independent copies of a probe trained in lock step as one stacked model.

    Equivalent to training each copy alone with ``torch.optim.AdamW`` and
    ``OneCycleLR`` (same initialisation, batches, learning rates, momenta, decoupled weight
    decay and dropout rate), but every step runs the copies that have a batch through one
    vmapped forward and backward (``torch.func``), so a shared GPU sees one set of kernels
    instead of ``n``. A copy without a batch at a step is left untouched (no moment,
    step-count or schedule update). Each copy's learning rate and beta1 come from its own
    real ``OneCycleLR`` (on a dummy optimiser), so the schedule is PyTorch's exactly.
    """

    def __init__(self, models: list[nn.Module], lr: float, weight_decay: float, total_steps: list[int],
                 pct_start: float = 0.1, betas: tuple[float, float] = (0.9, 0.999), eps: float = 1e-8) -> None:  # fmt: skip
        self.n = len(models)
        self.base = copy.deepcopy(models[0]).to("meta")
        p, b = torch.func.stack_module_state(models)
        self.params = {k: v.detach().clone().requires_grad_(True) for k, v in p.items()}
        self.buffers = {k: v.detach().clone() for k, v in b.items()}
        self.m = {k: torch.zeros_like(v) for k, v in self.params.items()}
        self.v = {k: torch.zeros_like(v) for k, v in self.params.items()}
        self.t = np.zeros(self.n, dtype=np.int64)
        self.wd, self.b2, self.eps = float(weight_decay), float(betas[1]), float(eps)
        self._sched = []
        for tot in total_steps:
            dummy = torch.optim.AdamW([torch.zeros(1, requires_grad=True)], lr=lr, betas=betas,
                                      weight_decay=weight_decay)  # fmt: skip
            self._sched.append((dummy, torch.optim.lr_scheduler.OneCycleLR(
                dummy, max_lr=lr, total_steps=int(tot), pct_start=pct_start)))  # fmt: skip

    def _f(
        self, p: dict[str, torch.Tensor], b: dict[str, torch.Tensor], *x: torch.Tensor
    ) -> torch.Tensor:
        from torch.nn.attention import SDPBackend, sdpa_kernel

        fast = torch.backends.mha.get_fastpath_enabled()
        torch.backends.mha.set_fastpath_enabled(False)  # the fused eval path has no vmap rule
        try:
            with sdpa_kernel(SDPBackend.MATH):  # composite: vmap batches it rather than looping
                return torch.func.functional_call(self.base, (p, b), x)
        finally:
            torch.backends.mha.set_fastpath_enabled(fast)

    def _sub(self, idx: torch.Tensor) -> tuple[dict[str, torch.Tensor], dict[str, torch.Tensor]]:
        return {k: v[idx] for k, v in self.params.items()}, {
            k: v[idx] for k, v in self.buffers.items()
        }

    def step(self, idx: list[int], x: torch.Tensor, y: torch.Tensor, *extra: torch.Tensor) -> None:
        """One AdamW step of copies ``idx`` on batches ``x`` [len(idx), B, ...], targets ``y``.

        ``y`` is [len(idx), B, n_out]; the loss is the Huber loss averaged over each batch,
        as ``smooth_l1_loss``.
        """
        dev = next(iter(self.params.values())).device
        ii = torch.as_tensor(idx, device=dev)
        self.base.train()
        p, b = self._sub(ii)
        pred = torch.vmap(self._f, in_dims=(0, 0, 0, *([0] * len(extra))), randomness="different")(
            p, b, x, *extra
        )
        loss = nn.functional.smooth_l1_loss(pred, y, reduction="none").flatten(1).mean(1).sum()
        grads = torch.autograd.grad(loss, list(self.params.values()))
        lr = torch.tensor(
            [self._sched[i][0].param_groups[0]["lr"] for i in idx], device=dev, dtype=torch.float32
        )
        b1 = torch.tensor(
            [self._sched[i][0].param_groups[0]["betas"][0] for i in idx],
            device=dev,
            dtype=torch.float32,
        )
        t = torch.as_tensor(self.t[idx] + 1, device=dev, dtype=torch.float32)
        bc1, bc2 = 1 - b1**t, 1 - self.b2**t
        with torch.no_grad():
            for (k, prm), g in zip(self.params.items(), grads, strict=True):
                shape = (-1,) + (1,) * (prm.dim() - 1)
                lr_, b1_, bc1_, bc2_ = (z.view(shape) for z in (lr, b1, bc1, bc2))
                g = g[ii]
                m = self.m[k][ii].lerp_(g, 1 - b1_)
                v = self.v[k][ii].mul_(self.b2).addcmul_(g, g, value=1 - self.b2)
                q = prm[ii].mul_(1 - lr_ * self.wd)
                q -= (lr_ / bc1_) * m / (v.sqrt() / bc2_.sqrt() + self.eps)
                prm[ii], self.m[k][ii], self.v[k][ii] = q, m, v
        self.t[idx] += 1
        for i in idx:
            self._sched[i][0].step()  # the dummy (no gradients: a no-op), so the schedule may step
            self._sched[i][1].step()

    @torch.no_grad()
    def predict(self, idx: list[int], x: torch.Tensor, *extra: torch.Tensor) -> torch.Tensor:
        """Outputs of copies ``idx`` (eval mode) on the same inputs ``x`` [B, ...]: [len(idx), B, n_out]."""
        dev = next(iter(self.params.values())).device
        self.base.eval()
        p, b = self._sub(torch.as_tensor(idx, device=dev))
        nones = (None,) * (1 + len(extra))
        return torch.vmap(self._f, in_dims=(0, 0, *nones))(p, b, x, *extra)

    def state(self, i: int) -> dict[str, torch.Tensor]:
        """Copy ``i``'s parameters (a copy)."""
        return {k: v[i].detach().clone() for k, v in self.params.items()}

    def load(self, i: int, state: dict[str, torch.Tensor]) -> None:
        """Set copy ``i``'s parameters."""
        with torch.no_grad():
            for k, v in self.params.items():
                v[i] = state[k]
