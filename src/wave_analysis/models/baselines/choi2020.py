"""Reference re-implementation: Choi et al. (2020) sequential-image Hs regression.

Choi, H. et al. (2020). Real-time significant wave height estimation from raw
ocean images based on 2D and 3D deep neural networks. *Ocean Engineering*, 201,
107129. https://doi.org/10.1016/j.oceaneng.2020.107129

Architecture (as documented in ``docs/literature/papers/choi2020.md``)::

    adjacent-frame differences  (T-1 frames from a T-frame clip; T = 16)
      -> VGG-16 convolutional trunk (ImageNet-pretrained)
      -> bidirectional ConvLSTM (256 filters, 3x3, stride 1)
      -> global average pooling (space and time)
      -> fully connected layers -> Hs

Details not verifiable from the abstract (FC widths, input resolution,
optimiser) are exposed as parameters with neutral defaults and must be set
from the paper when attempting a faithful replication.
"""

from __future__ import annotations

from itertools import pairwise

import torch
from torch import nn

from wave_analysis.models.layers.backbones import build_backbone
from wave_analysis.models.layers.convlstm import BiConvLSTM


class Choi2020Regressor(nn.Module):
    """Frame-difference + CNN + BiConvLSTM regressor."""

    def __init__(
        self,
        backbone: str = "vgg16",
        *,
        pretrained: bool = False,
        hidden_channels: int = 256,
        fc_dims: tuple[int, ...] = (256, 64),
        use_frame_differences: bool = True,
    ) -> None:
        super().__init__()
        self.encoder, c = build_backbone(backbone, pretrained=pretrained, keep_spatial=True)
        self.temporal = BiConvLSTM(c, hidden_channels, kernel_size=3)
        dims = (2 * hidden_channels, *fc_dims)
        layers: list[nn.Module] = []
        for a, b in pairwise(dims):
            layers += [nn.Linear(a, b), nn.ReLU(inplace=True)]
        layers.append(nn.Linear(dims[-1], 1))
        self.head = nn.Sequential(*layers)
        self.use_frame_differences = use_frame_differences

    def forward(self, clip: torch.Tensor) -> torch.Tensor:
        """``clip`` is ``(B, T, C, H, W)``; returns ``(B,)`` Hs estimates."""
        x = clip[:, 1:] - clip[:, :-1] if self.use_frame_differences else clip
        b, t = x.shape[:2]
        feats = self.encoder(x.flatten(0, 1))
        feats = feats.view(b, t, *feats.shape[1:])
        h = self.temporal(feats)  # (B, T, 2*hidden, h, w)
        pooled = h.mean(dim=(1, 3, 4))
        return self.head(pooled).squeeze(-1)
