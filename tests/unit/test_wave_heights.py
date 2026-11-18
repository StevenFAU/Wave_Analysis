"""Wave-heights model (wave_analysis.models.wave_heights)."""

from __future__ import annotations

import numpy as np
import torch

from wave_analysis.models.wave_heights import FLOOR_M, MultiViewRegressor, log_heights


def test_log_heights_floor_and_nan():
    out = log_heights(np.array([0.0, 1.0, np.nan]))
    assert out[0] == np.log(FLOOR_M)
    assert out[1] == 0.0
    assert np.isnan(out[2])


def test_regressor_masks_views_and_takes_aux():
    torch.manual_seed(0)
    net = MultiViewRegressor("resnet18", 3, aux_dim=3, pretrained=False).eval()
    views = torch.randn(2, 6, 3, 64, 96)
    mask = torch.tensor([[True] * 6, [True, True, True, False, False, False]])
    aux = torch.zeros(2, 6, 3)
    out = net((views, mask, aux))
    assert out.shape == (2, 3)
    views2 = views.clone()
    views2[1, 3:] = torch.randn(3, 3, 64, 96)  # change only the masked views of strip 2
    out2 = net((views2, mask, aux))
    assert torch.allclose(out[1], out2[1], atol=1e-5)  # masked views do not change the output
