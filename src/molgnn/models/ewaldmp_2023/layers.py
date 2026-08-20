"""Long-range aperiodic Ewald message-passing layers."""

from __future__ import annotations

import math

import torch
from torch import Tensor, nn


class ScaledSiLU(nn.Module):
    """SiLU with the second-moment scale used by the official implementation."""

    def forward(self, value: Tensor) -> Tensor:
        return torch.nn.functional.silu(value) / 0.6


class EwaldDense(nn.Module):
    """Bias-free dense layer with the source model's stable initialization."""

    def __init__(self, in_features: int, out_features: int, *, activate: bool) -> None:
        super().__init__()
        if min(in_features, out_features) < 2:
            raise ValueError("Ewald dense dimensions must be at least two")
        self.linear = nn.Linear(in_features, out_features, bias=False)
        self.activation = ScaledSiLU() if activate else nn.Identity()
        self.reset_parameters()

    def reset_parameters(self) -> None:
        nn.init.orthogonal_(self.linear.weight)
        with torch.no_grad():
            variance, mean = torch.var_mean(
                self.linear.weight, dim=1, unbiased=True, keepdim=True
            )
            standardized = (self.linear.weight - mean) / torch.sqrt(variance + 1.0e-6)
            self.linear.weight.copy_(standardized / math.sqrt(self.linear.in_features))

    def forward(self, value: Tensor) -> Tensor:
        return self.activation(self.linear(value))


class EwaldResidual(nn.Module):
    """Two-layer nonlinear residual block normalized by sqrt(2)."""

    def __init__(self, hidden_dim: int) -> None:
        super().__init__()
        self.layers = nn.Sequential(
            EwaldDense(hidden_dim, hidden_dim, activate=True),
            EwaldDense(hidden_dim, hidden_dim, activate=True),
        )

    def forward(self, value: Tensor) -> Tensor:
        return (value + self.layers(value)) / math.sqrt(2.0)


def fourier_message(
    hidden: Tensor,
    dot: Tensor,
    damping: Tensor,
    k_filter: Tensor,
    graph_batch: Tensor,
    num_graphs: int,
) -> Tensor:
    """Evaluate Eq. (13) through per-graph structure factors."""

    node_count, hidden_dim = hidden.shape
    if dot.ndim != 2 or dot.shape[0] != node_count:
        raise ValueError("dot must have shape [N, K]")
    if damping.shape != (node_count, 1):
        raise ValueError("damping must have shape [N, 1]")
    if k_filter.shape != (dot.shape[1], hidden_dim):
        raise ValueError("k_filter must have shape [K, H]")
    if graph_batch.shape != (node_count,) or graph_batch.dtype != torch.long:
        raise ValueError("graph_batch must be a long tensor with shape [N]")

    cosine = damping * torch.cos(dot)
    sine = damping * torch.sin(dot)
    real_contributions = hidden.unsqueeze(1) * cosine.unsqueeze(-1)
    imag_contributions = hidden.unsqueeze(1) * sine.unsqueeze(-1)
    factor_shape = (num_graphs, dot.shape[1], hidden_dim)
    structure_real = hidden.new_zeros(factor_shape).index_add(
        0, graph_batch, real_contributions
    )
    structure_imag = hidden.new_zeros(factor_shape).index_add(
        0, graph_batch, imag_contributions
    )

    filtered_real = structure_real.index_select(0, graph_batch) * k_filter
    filtered_imag = structure_imag.index_select(0, graph_batch) * k_filter
    return (
        filtered_real * cosine.unsqueeze(-1) + filtered_imag * sine.unsqueeze(-1)
    ).sum(dim=1)


class EwaldBlock(nn.Module):
    """One scalar long-range update with a shared Fourier downprojection."""

    def __init__(
        self,
        shared_down: EwaldDense,
        *,
        hidden_dim: int,
        downprojection_size: int,
        num_hidden: int,
        update_scale: float,
    ) -> None:
        super().__init__()
        if (
            isinstance(num_hidden, bool)
            or not isinstance(num_hidden, int)
            or num_hidden < 0
        ):
            raise ValueError("num_hidden must be a non-negative integer")
        if not math.isfinite(float(update_scale)) or update_scale < 0.0:
            raise ValueError("update_scale must be a non-negative finite number")
        self.shared_down = shared_down
        self.up = EwaldDense(downprojection_size, hidden_dim, activate=False)
        self.pre_residual = EwaldResidual(hidden_dim)
        self.update_layers = nn.ModuleList(
            [EwaldDense(hidden_dim, hidden_dim, activate=True)]
            + [EwaldResidual(hidden_dim) for _ in range(num_hidden)]
        )
        self.update_scale = float(update_scale)

    def forward(
        self,
        hidden: Tensor,
        dot: Tensor,
        damping: Tensor,
        k_rbf_values: Tensor,
        graph_batch: Tensor,
        num_graphs: int,
    ) -> Tensor:
        preprocessed = self.pre_residual(hidden)
        k_filter = self.up(self.shared_down(k_rbf_values))
        update = self.update_scale * fourier_message(
            preprocessed,
            dot,
            damping,
            k_filter,
            graph_batch,
            num_graphs,
        )
        for layer in self.update_layers:
            update = layer(update)
        return update


__all__ = [
    "EwaldBlock",
    "EwaldDense",
    "EwaldResidual",
    "ScaledSiLU",
    "fourier_message",
]
