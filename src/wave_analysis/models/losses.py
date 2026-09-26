r"""Physics-informed loss terms (PyTorch; optional ``ml`` extra).

Kamagata et al. (2026) regularise multi-task predictions with

.. math::
    \mathcal L_{\text{phys}} = \frac1N\sum_i \left|\log(\hat T_{p,i}^2+\epsilon)
                                - \log(6\hat H_{s,i}+\epsilon)\right|,
    \quad \lambda_p = 0.1,\ \epsilon=10^{-4},

described as enforcing "approximate deep-water dispersion". It is reproduced
exactly in :func:`kamagata_physics_loss` for replication. Note, however, what
the constraint implies: :math:`T_p^2 = 6 H_s` is equivalent to a *fixed*
peak steepness

.. math::
    s_p = \frac{2\pi H_s}{g T_p^2} = \frac{2\pi}{6 g} \approx 0.107,

roughly twice the steepness of the steepest observed wind seas
(:math:`s_p \lesssim 0.05`-0.07) and ~3-10x typical swell-dominated coastal
values. For Hs = 2 m it pulls Tp towards ~3.5 s. It is thus a strong,
site-agnostic prior rather than a dispersion relation, and it is *consistent
with* (though not proven to cause) the negative Tp bias (-0.431 s) reported
in that paper.

This project instead provides constraints that are physically *necessary*
rather than empirically assumed:

* :func:`steepness_band_loss` - a hinge penalty only when predicted
  :math:`s_p` leaves a plausible band (default 0.005-0.07);
* :func:`ordering_loss` - :math:`H_{\max} \ge H_s` (always true for the same record);
* :func:`spectral_consistency_loss` - predicted :math:`H_s` must equal
  :math:`4\sqrt{m_0}` of the predicted spectrum;
* :func:`circular_loss` - direction error without the 0/360 discontinuity.

Which prior helps *cross-site generalisation* is an open empirical question for
this project (``docs/literature/research_gaps.md``, Q-G4).
"""

from __future__ import annotations

import math
from typing import TYPE_CHECKING

if TYPE_CHECKING:  # pragma: no cover
    import torch

G = 9.80665


def _torch():  # type: ignore[no-untyped-def]
    try:
        import torch
    except ImportError as exc:  # pragma: no cover
        raise ImportError("install the 'ml' extra: pip install 'wave-analysis[ml]'") from exc
    return torch


def kamagata_physics_loss(
    hs: torch.Tensor, tp: torch.Tensor, *, coeff: float = 6.0, eps: float = 1e-4
) -> torch.Tensor:
    """Exact reproduction of the Kamagata et al. (2026) physics regulariser (see module docs)."""
    torch = _torch()
    return torch.mean(torch.abs(torch.log(tp**2 + eps) - torch.log(coeff * hs + eps)))


def steepness_band_loss(
    hs: torch.Tensor,
    tp: torch.Tensor,
    *,
    s_min: float = 0.005,
    s_max: float = 0.07,
    eps: float = 1e-6,
) -> torch.Tensor:
    r"""Hinge penalty on log-steepness outside ``[s_min, s_max]``; zero inside the band.

    .. math:: \mathcal L = \overline{\mathrm{ReLU}(\log s_{\min} - \log s_p) + \mathrm{ReLU}(\log s_p - \log s_{\max})}
    """
    torch = _torch()
    log_s = torch.log(2 * math.pi * hs.clamp_min(eps) / (G * tp.clamp_min(eps) ** 2))
    return torch.mean(torch.relu(math.log(s_min) - log_s) + torch.relu(log_s - math.log(s_max)))


def ordering_loss(lower: torch.Tensor, upper: torch.Tensor) -> torch.Tensor:
    """Penalise violations of ``lower <= upper`` (e.g. Hs <= Hmax; Tm02 <= Tm01)."""
    torch = _torch()
    return torch.mean(torch.relu(lower - upper))


def spectral_consistency_loss(
    hs: torch.Tensor, spectrum: torch.Tensor, bandwidth: torch.Tensor
) -> torch.Tensor:
    """``|Hs - 4 sqrt(sum S df)|`` for models that predict both a spectrum and Hs."""
    torch = _torch()
    m0 = torch.sum(spectrum * bandwidth, dim=-1)
    return torch.mean(torch.abs(hs - 4.0 * torch.sqrt(m0.clamp_min(0))))


def circular_loss(pred_deg: torch.Tensor, true_deg: torch.Tensor) -> torch.Tensor:
    """``1 - cos(pred - true)``: smooth, periodic direction loss in [0, 2]."""
    torch = _torch()
    return torch.mean(1 - torch.cos(torch.deg2rad(pred_deg - true_deg)))
