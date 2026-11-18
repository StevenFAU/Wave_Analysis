"""Frozen-encoder features and small trained heads for shore-camera clips.

The pieces of ``scripts/train_video_probe.py`` that can be tested without weights:
how a clip's frames become encoder input (two square tiles, the arms' frame orders),
how encoder tokens are pooled to a fixed grid, and the two heads (an attentive probe
and an MLP on mean tokens). Choices and reasons: ``docs/experiments/webcoos_video.md``.
"""

from __future__ import annotations

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
