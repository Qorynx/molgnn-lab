"""Core continuous-filter layers from SchNet (Schutt et al., 2017)."""

from __future__ import annotations

import math

import torch
from torch import Tensor, nn
from torch_geometric.utils import scatter


class ShiftedSoftplus(nn.Module):
    """``softplus(x) - log(2)`` used by the original implementation."""

    def forward(self, value: Tensor) -> Tensor:
        return torch.nn.functional.softplus(value) - math.log(2.0)


class GaussianRBF(nn.Module):
    """Fixed Gaussian distance expansion used to condition SchNet filters."""

    def __init__(self, cutoff: float, gap: float, gamma: float) -> None:
        super().__init__()
        if cutoff <= 0 or gap <= 0 or gamma <= 0:
            raise ValueError("cutoff, gap, and gamma must be positive")
        center_count = round(cutoff / gap)
        centers = torch.linspace(0.0, cutoff, center_count + 1)
        self.register_buffer("centers", centers)
        self.gamma = float(gamma)

    @property
    def num_rbf(self) -> int:
        return int(self.centers.numel())

    def forward(self, distances: Tensor) -> Tensor:
        if distances.ndim != 1:
            raise ValueError("distances must have shape [E]")
        return torch.exp(-self.gamma * (distances.unsqueeze(-1) - self.centers).square())


class ContinuousFilterConvolution(nn.Module):
    """Distance-conditioned aggregation from source atoms into target atoms."""

    def __init__(self, hidden_dim: int, num_filters: int, num_rbf: int) -> None:
        super().__init__()
        for value, name in (
            (hidden_dim, "hidden_dim"),
            (num_filters, "num_filters"),
            (num_rbf, "num_rbf"),
        ):
            _positive_int(value, name)
        self.input_projection = nn.Linear(hidden_dim, num_filters, bias=False)
        self.filter_network = nn.Sequential(
            nn.Linear(num_rbf, num_filters),
            ShiftedSoftplus(),
            nn.Linear(num_filters, num_filters),
            ShiftedSoftplus(),
        )
        self.output_projection = nn.Linear(num_filters, hidden_dim)
        self.activation = ShiftedSoftplus()

    def forward(self, features: Tensor, edge_index: Tensor, rbf: Tensor) -> Tensor:
        if features.ndim != 2:
            raise ValueError("features must have shape [N, F]")
        if edge_index.shape != (2, rbf.shape[0]):
            raise ValueError("edge_index and rbf must agree on edge count")

        source, target = edge_index
        projected = self.input_projection(features)
        filters = self.filter_network(rbf)
        messages = projected[source] * filters
        aggregated = scatter(
            messages,
            target,
            dim=0,
            dim_size=features.shape[0],
            reduce="sum",
        )
        return self.activation(self.output_projection(aggregated))


class InteractionBlock(nn.Module):
    """One residual continuous-filter interaction block."""

    def __init__(self, hidden_dim: int, num_filters: int, num_rbf: int) -> None:
        super().__init__()
        self.cfconv = ContinuousFilterConvolution(hidden_dim, num_filters, num_rbf)
        self.atomwise = nn.Linear(hidden_dim, hidden_dim)

    def forward(self, features: Tensor, edge_index: Tensor, rbf: Tensor) -> Tensor:
        return features + self.atomwise(self.cfconv(features, edge_index, rbf))


def _positive_int(value: int, field: str) -> None:
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise ValueError(f"{field} must be a positive integer")


__all__ = [
    "ContinuousFilterConvolution",
    "GaussianRBF",
    "InteractionBlock",
    "ShiftedSoftplus",
]
