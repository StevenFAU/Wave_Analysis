"""Reference re-implementations: Kim, Cho & Lee (2023) snapshot CNN and CNN-BiLSTM.

Kim, Y.-H., Cho, S. & Lee, P.-S. (2023). Wave height classification via deep
learning using monoscopic ocean videos. *Ocean Engineering*, 288, 116002.
https://doi.org/10.1016/j.oceaneng.2023.116002

* :class:`Kim2023SnapshotCNN` - 4 blocks of 3x3 convolution (same padding;
  16-32-64-64 filters), batch normalisation and ReLU; 4x4 max pooling with
  stride 4 after blocks 1-3 and none after block 4; flatten -> one fully
  connected layer -> softmax over wave-height classes; grayscale input
  (the paper's Fig. 5 and Table 3).
* :class:`Kim2023CNNBiLSTM` - per-frame pretrained GoogLeNet features
  -> bidirectional LSTM (2000 hidden units) -> dropout 0.5 -> FC -> softmax.

The paper extracts frames every 0.5 s and finds 180-s sequences best; at
2 frames/s that is 360 time steps per clip.
"""

from __future__ import annotations

import torch
from torch import nn

from wave_analysis.models.layers.backbones import build_backbone


class Kim2023SnapshotCNN(nn.Module):
    """Single-frame classifier as published (grayscale by default).

    The fully connected layer reads the flattened last feature map, so its
    width follows ``input_hw``: 224 x 224 pixels leave 64 maps of 3 x 3.
    """

    def __init__(
        self, n_classes: int = 5, in_channels: int = 1, input_hw: tuple[int, int] = (224, 224)
    ) -> None:
        super().__init__()
        layers: list[nn.Module] = []
        width = in_channels
        for i, filters in enumerate((16, 32, 64, 64)):
            layers += [
                nn.Conv2d(width, filters, 3, padding=1),
                nn.BatchNorm2d(filters),
                nn.ReLU(inplace=True),
            ]
            if i < 3:
                layers.append(nn.MaxPool2d(4, stride=4))
            width = filters
        self.features = nn.Sequential(*layers)
        h, w = input_hw
        flat = width * (h // 64) * (w // 64)
        if flat == 0:
            raise ValueError(f"input {input_hw} is too small for three 4 x 4 poolings")
        self.fc = nn.Linear(flat, n_classes)

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        """``(B, C, H, W)`` -> class logits ``(B, n_classes)``."""
        return self.fc(self.features(image).flatten(1))


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
