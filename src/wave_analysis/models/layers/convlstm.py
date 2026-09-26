r"""Convolutional LSTM (Shi et al. 2015) and a bidirectional wrapper.

Gate equations (as used by Choi et al. 2020 for sequential-image Hs regression):

.. math::
    i_t &= \sigma(W_{xi} * x_t + W_{hi} * h_{t-1} + b_i) \\
    f_t &= \sigma(W_{xf} * x_t + W_{hf} * h_{t-1} + b_f) \\
    \tilde c_t &= \tanh(W_{xc} * x_t + W_{hc} * h_{t-1} + b_c) \\
    o_t &= \sigma(W_{xo} * x_t + W_{ho} * h_{t-1} + b_o) \\
    c_t &= f_t \odot c_{t-1} + i_t \odot \tilde c_t, \qquad h_t = o_t \odot \tanh(c_t)

Note: the project technical summary transcribes the candidate state with a
sigmoid; the standard ConvLSTM (Shi et al. 2015) uses tanh, implemented here.
Shi et al.'s peephole terms (:math:`W_{c\cdot} \circ c`) are omitted, as in most
modern implementations. The four gates are computed with one fused convolution.

Reference: Shi, X. et al. (2015). Convolutional LSTM network: a machine learning
approach for precipitation nowcasting. *NeurIPS 28*. https://arxiv.org/abs/1506.04214
"""

from __future__ import annotations

import torch
from torch import nn


class ConvLSTMCell(nn.Module):
    """One ConvLSTM cell with ``same`` padding."""

    def __init__(self, in_channels: int, hidden_channels: int, kernel_size: int = 3) -> None:
        super().__init__()
        self.hidden_channels = hidden_channels
        self.conv = nn.Conv2d(
            in_channels + hidden_channels,
            4 * hidden_channels,
            kernel_size,
            padding=kernel_size // 2,
        )

    def forward(
        self, x: torch.Tensor, state: tuple[torch.Tensor, torch.Tensor]
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Advance one step; ``x`` is ``(B, C, H, W)``."""
        h, c = state
        i, f, g, o = torch.chunk(self.conv(torch.cat([x, h], dim=1)), 4, dim=1)
        c = torch.sigmoid(f) * c + torch.sigmoid(i) * torch.tanh(g)
        h = torch.sigmoid(o) * torch.tanh(c)
        return h, c

    def init_state(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Zero state matching ``x`` spatial size."""
        b, _, hgt, wid = x.shape
        z = x.new_zeros(b, self.hidden_channels, hgt, wid)
        return z, z.clone()


class ConvLSTM(nn.Module):
    """Unidirectional ConvLSTM over ``(B, T, C, H, W)``; returns all hidden states."""

    def __init__(self, in_channels: int, hidden_channels: int, kernel_size: int = 3) -> None:
        super().__init__()
        self.cell = ConvLSTMCell(in_channels, hidden_channels, kernel_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Return hidden states ``(B, T, hidden, H, W)``."""
        state = self.cell.init_state(x[:, 0])
        outs = []
        for t in range(x.shape[1]):
            state = self.cell(x[:, t], state)
            outs.append(state[0])
        return torch.stack(outs, dim=1)


class BiConvLSTM(nn.Module):
    """Bidirectional ConvLSTM: forward and time-reversed passes concatenated on channels.

    Bidirectionality uses *future* frames within the clip, so it suits offline
    clip-level estimation; a strictly causal real-time system should use
    :class:`ConvLSTM` instead (technical summary §5.3).
    """

    def __init__(self, in_channels: int, hidden_channels: int, kernel_size: int = 3) -> None:
        super().__init__()
        self.fwd = ConvLSTM(in_channels, hidden_channels, kernel_size)
        self.bwd = ConvLSTM(in_channels, hidden_channels, kernel_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Return ``(B, T, 2*hidden, H, W)``."""
        f = self.fwd(x)
        b = self.bwd(torch.flip(x, dims=[1])).flip(dims=[1])
        return torch.cat([f, b], dim=2)
