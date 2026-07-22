"""Neural tensor and feature-attention layers used by HiGNN 2023."""

from __future__ import annotations

import torch
from torch import Tensor, nn
from torch.nn import functional as F
from torch_geometric.utils import scatter


class NTNConv(nn.Module):
    """Neural tensor message passing from the HiGNN reference implementation."""

    def __init__(self, hidden_dim: int, num_slices: int = 2, dropout: float = 0.2) -> None:
        super().__init__()
        _positive_int(hidden_dim, "hidden_dim")
        _positive_int(num_slices, "num_slices")
        if hidden_dim % num_slices:
            raise ValueError("hidden_dim must be divisible by num_slices")
        _dropout(dropout)

        self.hidden_dim = hidden_dim
        self.num_slices = num_slices
        self.dropout = float(dropout)
        self.node_projection = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.edge_projection = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.bilinear = nn.Bilinear(hidden_dim, hidden_dim, num_slices, bias=False)
        self.block = nn.Linear(3 * hidden_dim, num_slices)
        self.reset_parameters()

    def reset_parameters(self) -> None:
        nn.init.xavier_uniform_(self.node_projection.weight)
        nn.init.xavier_uniform_(self.edge_projection.weight)
        self.bilinear.reset_parameters()
        self.block.reset_parameters()

    def forward(self, x: Tensor, edge_index: Tensor, edge_attr: Tensor) -> Tensor:
        source, target = edge_index
        if source.numel() == 0:
            return x.new_zeros(x.shape)

        projected_x = self.node_projection(x)
        projected_edge = self.edge_projection(edge_attr)
        target_x = projected_x[target]
        source_x = projected_x[source]
        attention = torch.tanh(
            self.bilinear(target_x, source_x)
            + self.block(torch.cat((target_x, projected_edge, source_x), dim=-1))
        )
        attention = F.dropout(attention, p=self.dropout, training=self.training)

        values = torch.maximum(source_x, projected_edge).reshape(
            -1, self.num_slices, self.hidden_dim // self.num_slices
        )
        messages = (attention.unsqueeze(-1) * values).reshape(-1, self.hidden_dim)
        return scatter(messages, target, dim=0, dim_size=x.shape[0], reduce="sum")


class FeatureAttention(nn.Module):
    """Shared max/sum channel attention over a node grouping."""

    def __init__(self, hidden_dim: int, reduction: int = 4) -> None:
        super().__init__()
        _positive_int(hidden_dim, "hidden_dim")
        _positive_int(reduction, "reduction")
        if hidden_dim % reduction:
            raise ValueError("hidden_dim must be divisible by reduction")
        self.mlp = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // reduction, bias=False),
            nn.ReLU(),
            nn.Linear(hidden_dim // reduction, hidden_dim, bias=False),
        )
        self.reset_parameters()

    def reset_parameters(self) -> None:
        for layer in self.mlp:
            if isinstance(layer, nn.Linear):
                layer.reset_parameters()

    def forward(self, x: Tensor, group: Tensor) -> Tensor:
        num_groups = int(group.max().item()) + 1
        maximum = scatter(x, group, dim=0, dim_size=num_groups, reduce="max")
        total = scatter(x, group, dim=0, dim_size=num_groups, reduce="sum")
        weights = torch.sigmoid(self.mlp(maximum) + self.mlp(total))
        return x * weights[group]


def _positive_int(value: int, field: str) -> None:
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise ValueError(f"{field} must be a positive integer")


def _dropout(value: float) -> None:
    if isinstance(value, bool) or not isinstance(value, (float, int)) or not 0 <= value < 1:
        raise ValueError("dropout must be in [0, 1)")


__all__ = ["FeatureAttention", "NTNConv"]
