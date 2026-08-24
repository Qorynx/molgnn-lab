"""Degree-specific convolution and normalization layers for Neural Fingerprint."""

from __future__ import annotations

from typing import Literal

import torch
from torch import Tensor, nn
from torch_geometric.utils import scatter


class LegacyFeatureNormalization(nn.Module):
    """Stateless feature normalization matching the official Autograd code.

    (x - mean) / (std + 1.0) without learned affine parameters or running stats.
    """

    def forward(self, x: Tensor) -> Tensor:
        if x.ndim != 2 or x.shape[0] == 0:
            return x
        mean = x.mean(dim=0, keepdim=True)
        std = x.std(dim=0, keepdim=True, unbiased=False)
        return (x - mean) / (std + 1.0)


class DegreeSpecificGraphConv(nn.Module):
    """Graph convolution layer with degree-specific neighbor projection matrices.

    For layer l:
        neighbor_i = sum_{j in N(i)} [h_j, bond_ji]
        h_i' = activation(h_i W_self + neighbor_i W_neighbor[d_i] + b)
    """

    def __init__(
        self,
        in_atom_dim: int,
        hidden_dim: int,
        bond_dim: int = 6,
        activation: Literal["relu", "tanh"] = "relu",
        normalization: Literal["legacy", "none"] = "legacy",
    ) -> None:
        super().__init__()
        self.in_atom_dim = in_atom_dim
        self.hidden_dim = hidden_dim
        self.bond_dim = bond_dim
        self.normalization_mode = normalization

        self.self_linear = nn.Linear(in_atom_dim, hidden_dim, bias=False)
        self.degree_linears = nn.ModuleList(
            [nn.Linear(in_atom_dim + bond_dim, hidden_dim, bias=False) for _ in range(6)]
        )
        self.overflow_linear = nn.Linear(in_atom_dim + bond_dim, hidden_dim, bias=False)
        self.bias = nn.Parameter(torch.zeros(hidden_dim))

        self.norm = LegacyFeatureNormalization() if normalization == "legacy" else None
        if activation == "relu":
            self.act = nn.ReLU()
        elif activation == "tanh":
            self.act = nn.Tanh()
        else:
            raise ValueError(f"unsupported activation: {activation!r}")

    def forward(
        self,
        x: Tensor,
        edge_index: Tensor,
        edge_attr: Tensor,
        *,
        num_nodes: int,
    ) -> Tensor:
        self_out = self.self_linear(x)
        num_edges = edge_index.shape[1]

        if num_edges > 0:
            source, target = edge_index
            neighbor_feat = torch.cat([x[source], edge_attr], dim=-1)
            neighbor_agg = scatter(
                neighbor_feat, target, dim=0, dim_size=num_nodes, reduce="sum"
            )
            in_degrees = scatter(
                torch.ones(num_edges, dtype=torch.long, device=x.device),
                target,
                dim=0,
                dim_size=num_nodes,
                reduce="sum",
            )
        else:
            neighbor_agg = torch.zeros(
                (num_nodes, self.in_atom_dim + self.bond_dim),
                dtype=x.dtype,
                device=x.device,
            )
            in_degrees = torch.zeros(num_nodes, dtype=torch.long, device=x.device)

        deg_out = torch.zeros((num_nodes, self.hidden_dim), dtype=x.dtype, device=x.device)
        for degree in range(6):
            mask = in_degrees == degree
            if mask.any():
                deg_out[mask] = self.degree_linears[degree](neighbor_agg[mask])

        overflow_mask = in_degrees >= 6
        if overflow_mask.any():
            deg_out[overflow_mask] = self.overflow_linear(neighbor_agg[overflow_mask])

        total = self_out + deg_out + self.bias
        if self.norm is not None:
            total = self.norm(total)
        return self.act(total)


__all__ = [
    "DegreeSpecificGraphConv",
    "LegacyFeatureNormalization",
]
