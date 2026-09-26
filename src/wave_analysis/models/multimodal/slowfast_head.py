r"""Reference re-implementation: Kamagata et al. (2026) SlowFast temporal encoder and multi-task head.

Kamagata, A. H. et al. (2026). HPC-Enabled Video-based Coastal Wave Parameter
Estimation Using V-JEPA and Deep Spatiotemporal Learning. arXiv:2607.11998.
https://arxiv.org/abs/2607.11998

Verified against the arXiv PDF (v1):

* input clips of 16 frames (stride 8 at 10 fps), 224x224;
* V-JEPA ``vit_small_patch16_224`` backbone giving per-frame 384-d features
  :math:`F \in \mathbb R^{B\times T\times 384}` (supplied externally here -
  any frame encoder producing ``(B, T, 384)`` can be plugged in);
* SlowFast transformer encoders, 2 layers, hidden 384, dropout 0.1; fast path
  on all 16 frames with 4 heads, slow path on every 4th frame (alpha = 4) with
  8 heads; outputs concatenated and linearly fused to 384-d;
* MLP head 384 -> 128 (GELU) -> 5, output :math:`\hat y = 5\,\mathrm{Softplus}(\cdot)`
  for :math:`[H_s, H_{\max}, T_p, T_z, \theta]`.

Improvement hooks (not in the paper, off by default): ``direction_as_sincos``
predicts direction as a unit vector. In the paper, one shared ``5 x Softplus``
output must span metres, seconds, and hundreds of degrees, and a scalar
direction has a 0/360 discontinuity. ``output_scale`` may also be given per
target so that each target has its own output range.
"""

from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F


def _pool_encoder(d: int, heads: int, layers: int, dropout: float) -> nn.TransformerEncoder:
    layer = nn.TransformerEncoderLayer(
        d, heads, 4 * d, dropout, batch_first=True, activation="gelu"
    )
    return nn.TransformerEncoder(layer, layers)


class SlowFastWaveHead(nn.Module):
    """SlowFast temporal encoder + multi-task regression head over frame embeddings."""

    TARGETS = ("Hs", "Hmax", "Tp", "Tz", "direction")

    def __init__(
        self,
        d_model: int = 384,
        *,
        alpha: int = 4,
        fast_heads: int = 4,
        slow_heads: int = 8,
        n_layers: int = 2,
        dropout: float = 0.1,
        hidden: int = 128,
        output_scale: float | tuple[float, ...] = 5.0,
        direction_as_sincos: bool = False,
    ) -> None:
        super().__init__()
        self.alpha = alpha
        self.fast = _pool_encoder(d_model, fast_heads, n_layers, dropout)
        self.slow = _pool_encoder(d_model, slow_heads, n_layers, dropout)
        self.fuse = nn.Linear(2 * d_model, d_model)
        self.direction_as_sincos = direction_as_sincos
        n_out = 4 + (2 if direction_as_sincos else 1)
        self.mlp = nn.Sequential(nn.Linear(d_model, hidden), nn.GELU(), nn.Linear(hidden, n_out))
        scale = torch.as_tensor(output_scale, dtype=torch.float32).expand(
            4 if direction_as_sincos else 5
        )
        self.register_buffer("scale", scale.clone())

    def forward(self, frame_features: torch.Tensor) -> torch.Tensor:
        """``(B, T, D)`` -> ``(B, 5)`` ordered as :attr:`TARGETS` (direction in degrees)."""
        fast = self.fast(frame_features).mean(dim=1)
        slow = self.slow(frame_features[:, :: self.alpha]).mean(dim=1)
        z = self.fuse(torch.cat([fast, slow], dim=-1))
        out = self.mlp(z)
        if not self.direction_as_sincos:
            return self.scale * F.softplus(out)  # type: ignore[operator]
        pos = self.scale * F.softplus(out[:, :4])  # type: ignore[operator]
        sc = F.normalize(out[:, 4:], dim=-1)
        deg = torch.remainder(torch.rad2deg(torch.atan2(sc[:, 0], sc[:, 1])), 360.0)
        return torch.cat([pos, deg.unsqueeze(-1)], dim=-1)
