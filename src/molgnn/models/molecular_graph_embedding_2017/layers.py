"""Sparse layers for Coley et al.'s 2017 molecular graph embedding."""

from __future__ import annotations

import torch
from torch import Tensor, nn
from torch_geometric.utils import scatter


class ColeyGraphConv(nn.Module):
    """One self-inclusive atom/bond update from the 2017 architecture."""

    def __init__(self, input_dim: int, output_dim: int, bond_dim: int = 8) -> None:
        super().__init__()
        _positive_int(input_dim, "input_dim")
        _positive_int(output_dim, "output_dim")
        _positive_int(bond_dim, "bond_dim")
        self.atom_linear = nn.Linear(input_dim, output_dim)
        self.bond_linear = nn.Linear(bond_dim, output_dim, bias=False)
        self.reset_parameters()

    def reset_parameters(self) -> None:
        nn.init.xavier_uniform_(self.atom_linear.weight)
        nn.init.zeros_(self.atom_linear.bias)
        nn.init.xavier_uniform_(self.bond_linear.weight)

    def forward(self, x: Tensor, edge_index: Tensor, edge_attr: Tensor) -> Tensor:
        source, target = edge_index
        atom_values = self.atom_linear(x)
        atom_aggregate = atom_values + scatter(
            atom_values[source], target, dim=0, dim_size=x.shape[0], reduce="sum"
        )
        bond_aggregate = scatter(
            self.bond_linear(edge_attr), target, dim=0, dim_size=x.shape[0], reduce="sum"
        )
        return torch.tanh(atom_aggregate + bond_aggregate)


def _positive_int(value: int, field: str) -> None:
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise ValueError(f"{field} must be a positive integer")


__all__ = ["ColeyGraphConv"]
