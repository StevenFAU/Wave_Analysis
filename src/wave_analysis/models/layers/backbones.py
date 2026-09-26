"""Frame-encoder backbones shared by the reference architectures.

Pretrained ImageNet weights (VGG-16, GoogLeNet, ResNet-18) come from
``torchvision`` when ``pretrained=True``; downloading weights is an explicit,
opt-in action. With ``pretrained=False`` the same topology is randomly
initialised (used in tests and for from-scratch ablations). A tiny CNN is
provided for fast smoke tests without torchvision.
"""

from __future__ import annotations

from itertools import pairwise

import torch
from torch import nn


class TinyCNN(nn.Module):
    """Minimal 4-block CNN (16-32-64-64 channels, as in Kim et al. 2023's snapshot model)."""

    def __init__(self, in_channels: int = 3, out_dim: int = 64, keep_spatial: bool = False) -> None:
        super().__init__()
        chans = [in_channels, 16, 32, 64, 64]
        layers: list[nn.Module] = []
        for a, b in pairwise(chans):
            layers += [
                nn.Conv2d(a, b, 3, padding=1),
                nn.BatchNorm2d(b),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(2),
            ]
        self.features = nn.Sequential(*layers)
        self.keep_spatial = keep_spatial
        self.out_channels = 64
        self.out_dim = out_dim
        self.proj = nn.Linear(64, out_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """``(B, C, H, W)`` -> ``(B, out_dim)`` or feature map if ``keep_spatial``."""
        f = self.features(x)
        if self.keep_spatial:
            return f
        return self.proj(f.mean(dim=(2, 3)))


def build_backbone(
    name: str, *, pretrained: bool = False, keep_spatial: bool = False
) -> tuple[nn.Module, int]:
    """Return ``(module, feature_dim_or_channels)``.

    ``keep_spatial=True`` returns the convolutional trunk (feature maps) for
    ConvLSTM-style models; otherwise a pooled feature vector.
    """
    if name == "tiny":
        m = TinyCNN(keep_spatial=keep_spatial)
        return m, (m.out_channels if keep_spatial else m.out_dim)
    try:
        import torchvision.models as tvm
    except ImportError as exc:  # pragma: no cover
        raise ImportError("torchvision is required for named backbones") from exc
    if name == "vgg16":
        net = tvm.vgg16(weights=tvm.VGG16_Weights.IMAGENET1K_V1 if pretrained else None)
        if keep_spatial:
            return net.features, 512
        return nn.Sequential(net.features, nn.AdaptiveAvgPool2d(1), nn.Flatten()), 512
    if name == "resnet18":
        net = tvm.resnet18(weights=tvm.ResNet18_Weights.IMAGENET1K_V1 if pretrained else None)
        trunk = nn.Sequential(*list(net.children())[:-2])
        if keep_spatial:
            return trunk, 512
        return nn.Sequential(trunk, nn.AdaptiveAvgPool2d(1), nn.Flatten()), 512
    if name == "googlenet":
        net = tvm.googlenet(
            weights=tvm.GoogLeNet_Weights.IMAGENET1K_V1 if pretrained else None,
            aux_logits=False,
            init_weights=not pretrained,
        )
        net.fc = nn.Identity()
        return net, 1024
    raise ValueError(f"unknown backbone {name!r}")
