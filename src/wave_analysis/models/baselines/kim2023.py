"""Reference re-implementations: Kim, Cho & Lee (2023) snapshot CNN and CNN-BiLSTM.

Kim, Y.-H., Cho, S. & Lee, P.-S. (2023). Wave height classification via deep
learning using monoscopic ocean videos. *Ocean Engineering*, 288, 116002.
https://doi.org/10.1016/j.oceaneng.2023.116002

* :class:`Kim2023SnapshotCNN` - 4 conv blocks (16-32-64-64; conv-BN-ReLU-maxpool)
  -> flatten -> FC -> softmax over wave-height classes; grayscale input.
* :class:`Kim2023CNNBiLSTM` - per-frame pretrained GoogLeNet features
  -> bidirectional LSTM (2000 hidden units) -> dropout 0.5 -> FC -> softmax.

The paper extracts frames every 0.5 s and finds 180-s sequences best; at
2 frames/s that is 360 time steps per clip.
"""

from __future__ import annotations

import torch
from torch import nn

from wave_analysis.models.layers.backbones import TinyCNN, build_backbone


class Kim2023SnapshotCNN(nn.Module):
    """Single-frame classifier (grayscale by default)."""

    def __init__(self, n_classes: int = 5, in_channels: int = 1, feature_dim: int = 64) -> None:
        super().__init__()
        self.encoder = TinyCNN(in_channels=in_channels, out_dim=feature_dim)
        self.fc = nn.Linear(feature_dim, n_classes)

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        """``(B, C, H, W)`` -> class logits ``(B, n_classes)``."""
        return self.fc(torch.relu(self.encoder(image)))


class Kim2023CNNBiLSTM(nn.Module):
    """Frame encoder + BiLSTM sequence classifier."""

    def __init__(
        self,
        n_classes: int = 5,
        *,
        backbone: str = "googlenet",
        pretrained: bool = False,
        hidden_size: int = 2000,
        dropout: float = 0.5,
        freeze_backbone: bool = True,
    ) -> None:
        super().__init__()
        self.encoder, d = build_backbone(backbone, pretrained=pretrained)
        if freeze_backbone:
            for p in self.encoder.parameters():
                p.requires_grad_(False)
        self.lstm = nn.LSTM(d, hidden_size, batch_first=True, bidirectional=True)
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(2 * hidden_size, n_classes)

    def forward(self, clip: torch.Tensor) -> torch.Tensor:
        """``(B, T, C, H, W)`` -> class logits; uses the last BiLSTM output."""
        b, t = clip.shape[:2]
        feats = self.encoder(clip.flatten(0, 1)).view(b, t, -1)
        out, _ = self.lstm(feats)
        return self.fc(self.dropout(out[:, -1]))
