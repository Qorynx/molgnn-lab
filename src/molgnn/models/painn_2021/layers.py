"""Scalar/vector message and mixing blocks from PaiNN."""

from __future__ import annotations

import torch
from torch import Tensor, nn
from torch_geometric.utils import scatter

from .constants import PAINN_EPS


class PaiNNDense(nn.Linear):
    """SchNetPack-style Dense layer with Xavier weights and zero bias."""

    def __init__(
        self,
        in_features: int,
        out_features: int,
        *,
        bias: bool = True,
        activation: nn.Module | None = None,
    ) -> None:
        super().__init__(in_features, out_features, bias=bias)
        self.activation = activation if activation is not None else nn.Identity()

    def reset_parameters(self) -> None:
        nn.init.xavier_uniform_(self.weight)
        if self.bias is not None:
            nn.init.zeros_(self.bias)

    def forward(self, value: Tensor) -> Tensor:
        return self.activation(super().forward(value))


class PaiNNInteraction(nn.Module):
    """One residual continuous-filter message block."""

    def __init__(self, hidden_dim: int) -> None:
        super().__init__()
        self.hidden_dim = hidden_dim
        self.context = nn.Sequential(
            PaiNNDense(hidden_dim, hidden_dim, activation=nn.SiLU()),
            PaiNNDense(hidden_dim, 3 * hidden_dim),
        )

    def forward(
        self,
        scalar: Tensor,
        vector: Tensor,
        filter_weight: Tensor,
        edge_index: Tensor,
        directions: Tensor,
    ) -> tuple[Tensor, Tensor]:
        source, target = edge_index
        context = self.context(scalar)
        filtered = filter_weight * context[source]
        delta_scalar, direction_gate, state_gate = filtered.split(
            self.hidden_dim, dim=-1
        )
        messages_vector = (
            directions.unsqueeze(-1) * direction_gate.unsqueeze(1)
            + vector[source] * state_gate.unsqueeze(1)
        )
        delta_scalar = scatter(
            delta_scalar, target, dim=0, dim_size=scalar.shape[0], reduce="sum"
        )
        messages_vector = scatter(
            messages_vector, target, dim=0, dim_size=scalar.shape[0], reduce="sum"
        )
        return scalar + delta_scalar, vector + messages_vector


class PaiNNMixing(nn.Module):
    """One residual intra-atomic scalar/vector update block."""

    def __init__(self, hidden_dim: int, epsilon: float = PAINN_EPS) -> None:
        super().__init__()
        if epsilon <= 0:
            raise ValueError("epsilon must be positive")
        self.hidden_dim = hidden_dim
        self.epsilon = float(epsilon)
        self.vector_projection = PaiNNDense(hidden_dim, 2 * hidden_dim, bias=False)
        self.context = nn.Sequential(
            PaiNNDense(2 * hidden_dim, hidden_dim, activation=nn.SiLU()),
            PaiNNDense(hidden_dim, 3 * hidden_dim),
        )

    def forward(self, scalar: Tensor, vector: Tensor) -> tuple[Tensor, Tensor]:
        projected = self.vector_projection(vector)
        vector_v, vector_w = projected.split(self.hidden_dim, dim=-1)
        vector_norm = torch.sqrt(vector_v.square().sum(dim=1) + self.epsilon)
        context = self.context(torch.cat((scalar, vector_norm), dim=-1))
        delta_scalar, delta_vector, scalar_vector = context.split(
            self.hidden_dim, dim=-1
        )
        delta_vector = delta_vector.unsqueeze(1) * vector_w
        scalar_vector = scalar_vector * (vector_v * vector_w).sum(dim=1)
        return scalar + delta_scalar + scalar_vector, vector + delta_vector


__all__ = ["PaiNNInteraction", "PaiNNMixing", "PaiNNDense"]
