"""Wave-heights model (wave_analysis.models.wave_heights)."""

from __future__ import annotations

import numpy as np
import pytest
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


def test_distribution_head_shapes_and_positive_variance():
    from wave_analysis.models.wave_heights import split_outputs

    torch.manual_seed(0)
    net = MultiViewRegressor("resnet18", 3, pretrained=False, distribution=True).eval()
    views = torch.randn(2, 6, 3, 64, 96)
    mask = torch.ones(2, 6, dtype=torch.bool)
    out = net((views, mask))
    assert out.shape == (2, 9)
    point, mu, var = split_outputs(out, 3)
    assert point.shape == mu.shape == var.shape == (2, 3)
    assert bool((var > 0).all())
    p2, m2, v2 = split_outputs(out[:, :3], 3)
    assert m2 is None and v2 is None and torch.equal(p2, point)


def test_beta_nll_weights_by_variance_without_its_gradient():
    from wave_analysis.models.wave_heights import beta_nll

    mu = torch.tensor([0.0], requires_grad=True)
    var = torch.tensor([4.0], requires_grad=True)
    y = torch.tensor([1.0])
    plain = beta_nll(mu, var, y, beta=0.0)
    assert torch.allclose(plain, 0.5 * (torch.log(torch.tensor(4.0)) + 0.25))
    loss = beta_nll(mu, var, y, beta=0.5)
    assert torch.allclose(loss, plain.detach() * 2.0)  # var**0.5 = 2
    loss.sum().backward()
    # the mean's gradient is the NLL's, (mu - y) / var, times the weight var**beta
    assert torch.allclose(mu.grad, torch.tensor([(0.0 - 1.0) / 4.0 * 2.0]))


def test_backbone_rewarm_ramps_after_unfreezing():
    from wave_analysis.models.wave_heights import backbone_rewarm

    off = backbone_rewarm(20, 0)
    assert off(0) == off(25) == 1.0
    f = backbone_rewarm(20, 10)  # unfreezes at step 20, ramps over 10 steps
    assert f(19) == 1.0  # frozen: the factor does not matter
    assert f(20) == pytest.approx(0.1) and f(24) == pytest.approx(0.5)
    assert f(29) == f(100) == 1.0


def test_fit_heights_runs_with_rewarm_and_clipping():
    from torch.utils.data import DataLoader, TensorDataset

    from wave_analysis.models.sea_state import TrainSettings
    from wave_analysis.models.wave_heights import fit_heights

    class Tiny(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.encoder = torch.nn.Linear(4, 8)
            self.head = torch.nn.Linear(8, 1)

        def forward(self, x):
            return self.head(torch.relu(self.encoder(x[0])))

    torch.manual_seed(0)
    x = torch.randn(32, 4)
    y = torch.log(1 + x[:, :1].abs())
    loader = DataLoader(TensorDataset(x, y), batch_size=8)
    wrap = [((xb,), yb) for xb, yb in loader]
    settings = TrainSettings(epochs=3, warmup_epochs=1, lr=1e-2, amp="none", batch_size=8)
    fit = fit_heights(
        Tiny(), wrap, wrap, np.exp(y.numpy()), settings=settings, mean=np.zeros(1),
        std=np.ones(1), device=torch.device("cpu"), log=lambda _: None, names=("h",),
        lp_epochs=1, backbone_lr_scale=0.1, rewarm_epochs=1, grad_clip=0.5,
    )  # fmt: skip
    assert len(fit.history) == 3 and np.isfinite(fit.best_rmse_m)
