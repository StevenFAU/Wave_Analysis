"""Stage-3 heads of the buoy-camera model (wave_analysis.models.wave_spectral)."""

from __future__ import annotations

import math

import pytest

torch = pytest.importorskip("torch")

from wave_analysis.models.wave_spectral import (  # noqa: E402
    Layout,
    MultiViewSpectral,
    direction_loss,
    split_spectral,
)


def test_spectral_outputs_and_view_order_matters_only_for_direction():
    torch.manual_seed(0)
    layout = Layout(n_scalar=4, n_bins=5, direction=True)
    net = MultiViewSpectral("resnet18", 4, n_bins=5, direction=True, pretrained=False).eval()
    views = torch.randn(2, 6, 3, 64, 96)
    mask = torch.ones(2, 6, dtype=torch.bool)
    out = net((views, mask))
    assert out.shape == (2, 3 * 4 + 5 + 2)
    point, _mu, var, logits, direc = split_spectral(out, layout)
    assert point.shape == (2, 4) and bool((var > 0).all())
    assert logits is not None and logits.shape == (2, 5)
    assert direc is not None and direc.shape == (2, 2)
    # Reversing the views leaves the heights (a set) unchanged and moves the direction.
    out_r = net((views.flip(1), mask))
    p_r, *_, d_r = split_spectral(out_r, layout)
    assert torch.allclose(p_r, point, atol=1e-4)
    assert d_r is not None and not torch.allclose(d_r, direc, atol=1e-4)


def test_direction_loss_is_one_minus_cosine():
    t = torch.tensor([[1.0, 0.0]])
    assert torch.allclose(direction_loss(torch.tensor([[2.0, 0.0]]), t), torch.tensor([0.0]))
    assert torch.allclose(direction_loss(torch.tensor([[0.0, 3.0]]), t), torch.tensor([1.0]))
    half = torch.tensor([[math.cos(math.pi / 3), math.sin(math.pi / 3)]])
    assert torch.allclose(direction_loss(half, t), torch.tensor([0.5]), atol=1e-6)
