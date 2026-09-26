"""Reference deep architectures and physics losses (skipped when torch is not installed)."""

from __future__ import annotations

import math

import pytest

torch = pytest.importorskip("torch")
pytestmark = pytest.mark.ml


@pytest.mark.ml
def test_convlstm_shapes():
    from wave_analysis.models.layers.convlstm import BiConvLSTM

    out = BiConvLSTM(4, 8)(torch.randn(2, 5, 4, 6, 6))
    assert out.shape == (2, 5, 16, 6, 6)


@pytest.mark.ml
def test_choi2020_forward():
    from wave_analysis.models.baselines.choi2020 import Choi2020Regressor

    m = Choi2020Regressor(backbone="tiny", hidden_channels=8, fc_dims=(8,))
    assert m(torch.randn(2, 4, 3, 32, 32)).shape == (2,)


@pytest.mark.ml
def test_kim2023_models():
    from wave_analysis.models.baselines.kim2023 import Kim2023CNNBiLSTM, Kim2023SnapshotCNN

    assert Kim2023SnapshotCNN()(torch.randn(3, 1, 64, 64)).shape == (3, 5)
    seq = Kim2023CNNBiLSTM(backbone="tiny", hidden_size=16)
    assert seq(torch.randn(2, 6, 3, 32, 32)).shape == (2, 5)


@pytest.mark.ml
def test_yang2026_fusion_and_sliding_window():
    from wave_analysis.models.multimodal.yang2026 import (
        Yang2026Fusion,
        encode_attitude,
        sliding_window_predict,
    )

    enc = encode_attitude(torch.tensor([[0.0, math.pi / 2, 2 * math.pi - 1e-6]]))
    assert enc.shape == (1, 6)
    assert torch.allclose(enc[0, 4:], torch.tensor([0.0, 1.0]), atol=1e-5)  # yaw ~ 360 deg ~ 0 deg
    m = Yang2026Fusion(backbone="tiny", d_model=32, n_heads=4).eval()
    assert m(torch.randn(2, 7, 3, 32, 32), torch.randn(2, 7, 3)).shape == (2,)
    with pytest.raises(ValueError):
        m(torch.randn(1, 7, 3, 32, 32))
    img_only = Yang2026Fusion(backbone="tiny", d_model=32, n_heads=4, use_attitude=False).eval()
    assert img_only(torch.randn(1, 7, 3, 32, 32)).shape == (1,)
    preds = sliding_window_predict(
        m, torch.randn(12, 3, 32, 32), torch.randn(12, 3), window=8, step=2
    )
    assert preds.shape == (3,)


@pytest.mark.ml
def test_slowfast_head():
    from wave_analysis.models.multimodal.slowfast_head import SlowFastWaveHead

    out = SlowFastWaveHead()(torch.randn(2, 16, 384))
    assert out.shape == (2, 5) and (out >= 0).all()
    out2 = SlowFastWaveHead(direction_as_sincos=True, output_scale=(3.0, 6.0, 20.0, 15.0))(
        torch.randn(2, 16, 384)
    )
    assert out2.shape == (2, 5) and ((out2[:, 4] >= 0) & (out2[:, 4] < 360)).all()


@pytest.mark.ml
def test_physics_losses():
    from wave_analysis.models import losses

    hs = torch.tensor([2.0])
    # Kamagata loss is zero exactly on Tp^2 = 6 Hs, i.e. fixed steepness 2*pi/(6 g) ~ 0.107.
    tp = torch.sqrt(6 * hs)
    assert losses.kamagata_physics_loss(hs, tp).item() == pytest.approx(0.0, abs=1e-4)
    s_implied = 2 * math.pi * 2.0 / (9.80665 * tp.item() ** 2)
    assert s_implied == pytest.approx(0.1068, abs=1e-3)
    # The same pair is penalised by the steepness band; a realistic swell is not.
    assert losses.steepness_band_loss(hs, tp).item() > 0
    assert losses.steepness_band_loss(hs, torch.tensor([10.0])).item() == 0.0
    assert losses.ordering_loss(
        torch.tensor([2.0, 3.0]), torch.tensor([3.0, 2.0])
    ).item() == pytest.approx(0.5)
    assert losses.circular_loss(torch.tensor([359.0]), torch.tensor([1.0])).item() < 1e-3
    S = torch.ones(1, 10) * (1.0 / 16) / 10 / 0.01
    assert losses.spectral_consistency_loss(
        torch.tensor([1.0]), S, torch.full((10,), 0.01)
    ).item() == pytest.approx(0.0, abs=1e-6)
