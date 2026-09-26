r"""Reference re-implementation: Yang et al. (2026) image + attitude attention fusion.

Yang, X., Wang, H., Zhang, R., Sun, S. & Zhang, M. (2026). A deep learning
method for spatiotemporal significant wave height estimation with ship attitude
compensation. *Ocean Engineering*, 352, 124517.
https://doi.org/10.1016/j.oceaneng.2026.124517

Architecture (``docs/literature/papers/yang2026.md``):

* image branch: ResNet-18 per frame, :math:`f^{img}_t \in \mathbb R^d`;
* attitude branch: :math:`p_t = [\sin\phi,\cos\phi,\sin\theta,\cos\theta,\sin\psi,\cos\psi]`
  -> MLP :math:`\sigma(W_2\,\mathrm{ReLU}(W_1 p_t + b_1) + b_2)`;
* per-step fusion, sinusoidal positional encoding, multi-head self-attention
  encoder, temporal mean pooling, MLP regression head, MSE loss;
* sliding-window inference with overlapping windows averaged
  (:func:`sliding_window_predict`); the paper reports 7-9 frames as a good
  accuracy/latency trade-off.

The exact fusion operator (concatenation vs. summation before attention) is
exposed as ``fusion``; confirm against the paper for faithful replication.
"""

from __future__ import annotations

import math
from typing import Literal

import torch
from torch import nn

from wave_analysis.models.layers.backbones import build_backbone


class SinusoidalPositionalEncoding(nn.Module):
    """Vaswani et al. (2017) fixed positional encoding."""

    def __init__(self, d_model: int, max_len: int = 512) -> None:
        super().__init__()
        pos = torch.arange(max_len).unsqueeze(1)
        div = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(max_len, d_model)
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div[: d_model // 2])
        self.register_buffer("pe", pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Add encoding to ``(B, T, D)``."""
        return x + self.pe[: x.shape[1]]  # type: ignore[index]


def encode_attitude(roll_pitch_yaw_rad: torch.Tensor) -> torch.Tensor:
    """``(..., 3)`` radians -> ``(..., 6)`` sin/cos features, continuous across wrap-around."""
    return torch.stack(
        [f(roll_pitch_yaw_rad[..., i]) for i in range(3) for f in (torch.sin, torch.cos)], dim=-1
    )


class Yang2026Fusion(nn.Module):
    """ResNet-18 + attitude-MLP + transformer encoder regressor."""

    def __init__(
        self,
        *,
        backbone: str = "resnet18",
        pretrained: bool = False,
        d_model: int = 256,
        n_heads: int = 8,
        n_layers: int = 2,
        dropout: float = 0.1,
        fusion: Literal["concat", "sum"] = "concat",
        use_attitude: bool = True,
    ) -> None:
        super().__init__()
        self.encoder, d_img = build_backbone(backbone, pretrained=pretrained)
        self.img_proj = nn.Linear(d_img, d_model)
        self.use_attitude = use_attitude
        self.fusion = fusion
        self.att_mlp = nn.Sequential(
            nn.Linear(6, d_model), nn.ReLU(), nn.Linear(d_model, d_model), nn.Sigmoid()
        )
        fused_dim = 2 * d_model if (use_attitude and fusion == "concat") else d_model
        self.fuse_proj = nn.Linear(fused_dim, d_model)
        self.pos = SinusoidalPositionalEncoding(d_model)
        layer = nn.TransformerEncoderLayer(d_model, n_heads, 4 * d_model, dropout, batch_first=True)
        self.attn = nn.TransformerEncoder(layer, n_layers)
        self.head = nn.Sequential(
            nn.Linear(d_model, d_model // 2), nn.ReLU(), nn.Linear(d_model // 2, 1)
        )

    def forward(
        self, frames: torch.Tensor, attitude_rad: torch.Tensor | None = None
    ) -> torch.Tensor:
        """``frames`` ``(B, T, C, H, W)``; ``attitude_rad`` ``(B, T, 3)`` roll/pitch/yaw. Returns ``(B,)``."""
        b, t = frames.shape[:2]
        f = self.img_proj(self.encoder(frames.flatten(0, 1)).view(b, t, -1))
        if self.use_attitude:
            if attitude_rad is None:
                raise ValueError("attitude_rad is required when use_attitude=True")
            a = self.att_mlp(encode_attitude(attitude_rad))
            f = torch.cat([f, a], dim=-1) if self.fusion == "concat" else f + a
        z = self.attn(self.pos(self.fuse_proj(f)))
        return self.head(z.mean(dim=1)).squeeze(-1)


@torch.no_grad()
def sliding_window_predict(
    model: nn.Module,
    frames: torch.Tensor,
    attitude_rad: torch.Tensor | None,
    *,
    window: int = 8,
    step: int = 1,
) -> torch.Tensor:
    """Average overlapping window predictions over a long sequence.

    ``frames`` ``(T, C, H, W)``; returns one estimate per window, ordered in time
    (window ``i`` covers frames ``i*step : i*step + window``).
    """
    outs = []
    for s in range(0, frames.shape[0] - window + 1, step):
        f = frames[s : s + window].unsqueeze(0)
        a = attitude_rad[s : s + window].unsqueeze(0) if attitude_rad is not None else None
        outs.append(model(f, a) if a is not None else model(f))
    return torch.cat(outs)
