"""Building blocks for ResGAT 2024."""

from __future__ import annotations

import math
from typing import Optional

import torch
from torch import Tensor, nn
from torch.nn import Parameter
from torch.nn import functional as F
from torch.nn.init import _calculate_fan_in_and_fan_out, zeros_
from torch_geometric.nn import GATConv
from torch_geometric.nn.conv.message_passing import MessagePassing
from torch_geometric.typing import Adj, OptTensor
from torch_geometric.utils import softmax as pyg_softmax


def _glorot(tensor: Tensor | None) -> None:
    """In-place Xavier-uniform initialization (matches PyG's ``glorot_`` helper)."""

    if tensor is None:
        return
    fan_in, fan_out = _calculate_fan_in_and_fan_out(tensor)
    bound = math.sqrt(6.0 / (fan_in + fan_out))
    with torch.no_grad():
        tensor.uniform_(-bound, bound)


def _zeros(tensor: Tensor | None) -> None:
    """In-place zero initialization."""

    if tensor is None:
        return
    zeros_(tensor)


class GATEConv(MessagePassing):
    """Custom single-head GAT layer that fuses edge features (from the upstream code).

    Mirrors ``models.graph_residual_nw::GATEConv`` from
    2023-ResGAT (Nguyen-Vo et al., Memetic Computing 2024).  Uses an explicit
    message function with edge features concatenated to the source node feature
    before the linear projection, then a leaky-relu + softmax-attention score.
    This is structurally different from ``torch_geometric.nn.GATConv`` (which
    treats edge features as an attention-modulation term), so it is replicated
    here for a faithful port.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        edge_dim: int,
        dropout: float = 0.0,
    ) -> None:
        super().__init__(aggr="add", node_dim=0)
        _positive_int(in_channels, "in_channels")
        _positive_int(out_channels, "out_channels")
        _positive_int(edge_dim, "edge_dim")
        if not 0 <= dropout < 1:
            raise ValueError("dropout must be in [0, 1)")

        self.dropout = float(dropout)
        self.att_l = Parameter(torch.empty(1, out_channels))
        self.att_r = Parameter(torch.empty(1, in_channels))
        self.lin1 = nn.Linear(in_channels + edge_dim, out_channels, bias=False)
        self.lin2 = nn.Linear(out_channels, out_channels, bias=False)
        self.bias = Parameter(torch.empty(out_channels))
        self.reset_parameters()

    def reset_parameters(self) -> None:
        _glorot(self.att_l)
        _glorot(self.att_r)
        _glorot(self.lin1.weight)
        _glorot(self.lin2.weight)
        _zeros(self.bias)

    def forward(self, x: Tensor, edge_index: Adj, edge_attr: Tensor) -> Tensor:
        out = self.propagate(edge_index, x=x, edge_attr=edge_attr)
        out = out + self.bias
        return out

    def message(
        self,
        x_j: Tensor,
        x_i: Tensor,
        edge_attr: Tensor,
        index: Tensor,
        ptr: OptTensor,
        size_i: Optional[int],
    ) -> Tensor:
        x_j = F.leaky_relu_(self.lin1(torch.cat([x_j, edge_attr], dim=-1)))
        alpha_j = (x_j * self.att_l).sum(dim=-1)
        alpha_i = (x_i * self.att_r).sum(dim=-1)
        alpha = alpha_j + alpha_i
        alpha = F.leaky_relu_(alpha)
        alpha = pyg_softmax(alpha, index, ptr, size_i)
        alpha = F.dropout(alpha, p=self.dropout, training=self.training)
        return self.lin2(x_j) * alpha.unsqueeze(-1)


class ResGATBlock(nn.Module):
    """Two ``GATConv`` layers with ReLU between, consuming bond features via ``edge_dim``.

    Mirrors ``models.basic_block.py::BasicBlock`` from the upstream 2023-ResGAT
    paper. Uses PyG's standard ``GATConv`` (same as upstream).
    """

    def __init__(
        self,
        in_dim: int,
        out_dim: int,
        edge_dim: int,
        heads: int = 1,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()

        _positive_int(in_dim, "in_dim")
        _positive_int(out_dim, "out_dim")
        _positive_int(edge_dim, "edge_dim")
        _positive_int(heads, "heads")
        if not 0 <= dropout < 1:
            raise ValueError("dropout must be in [0, 1)")

        self.in_dim = in_dim
        self.out_dim = out_dim
        self.edge_dim = edge_dim
        self.heads = heads
        self.dropout_p = float(dropout)

        self.conv1 = GATConv(in_dim, out_dim, heads=heads, edge_dim=edge_dim)
        self.conv2 = GATConv(out_dim, out_dim, heads=heads, edge_dim=edge_dim)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(float(dropout)) if dropout > 0 else nn.Identity()

    def forward(self, x: Tensor, edge_index: Tensor, edge_attr: Tensor) -> Tensor:
        out = self.relu(self.conv1(x, edge_index, edge_attr))
        out = self.dropout(out)
        out = self.relu(self.conv2(out, edge_index, edge_attr))
        return out


def _positive_int(value: int, field: str) -> None:
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise ValueError(f"{field} must be a positive integer")


__all__ = ["GATEConv", "ResGATBlock"]
