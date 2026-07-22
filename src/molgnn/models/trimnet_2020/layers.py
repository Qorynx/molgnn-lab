"""Triplet-attention layers for TrimNet 2020."""

from __future__ import annotations

import torch
from torch import Tensor, nn
from torch.nn import functional as F
from torch_geometric.utils import scatter, softmax


class MultiHeadTripletAttention(nn.Module):
    """Aggregate atom-bond-atom messages with target-wise attention."""

    def __init__(self, hidden_dim: int, bond_dim: int, heads: int = 4) -> None:
        super().__init__()
        _positive_int(hidden_dim, "hidden_dim")
        _positive_int(bond_dim, "bond_dim")
        _positive_int(heads, "heads")

        self.hidden_dim = hidden_dim
        self.bond_dim = bond_dim
        self.heads = heads
        projected_dim = heads * hidden_dim
        self.node_weight = nn.Parameter(torch.empty(hidden_dim, projected_dim))
        self.edge_weight = nn.Parameter(torch.empty(bond_dim, projected_dim))
        self.attention_weight = nn.Parameter(torch.empty(1, heads, 3 * hidden_dim))
        self.scale_weight = nn.Parameter(torch.empty(projected_dim, hidden_dim))
        self.bias = nn.Parameter(torch.empty(hidden_dim))
        self.reset_parameters()

    def reset_parameters(self) -> None:
        """Match the initialization used by the official source."""

        nn.init.kaiming_uniform_(self.node_weight)
        nn.init.kaiming_uniform_(self.edge_weight)
        nn.init.kaiming_uniform_(self.attention_weight)
        nn.init.kaiming_uniform_(self.scale_weight)
        nn.init.zeros_(self.bias)

    def forward(self, x: Tensor, edge_index: Tensor, edge_attr: Tensor) -> Tensor:
        """Return one aggregated message vector per node."""

        _validate_layer_inputs(x, edge_index, edge_attr, self.hidden_dim, self.bond_dim)
        num_nodes = x.shape[0]
        projected_nodes = (x @ self.node_weight).view(num_nodes, self.heads, self.hidden_dim)
        source, target = edge_index

        if edge_index.shape[1]:
            projected_edges = (edge_attr @ self.edge_weight).view(-1, self.heads, self.hidden_dim)
            triplets = torch.cat(
                (projected_nodes[target], projected_edges, projected_nodes[source]),
                dim=-1,
            )
            scores = F.leaky_relu(
                (triplets * self.attention_weight).sum(dim=-1),
                negative_slope=0.2,
            )
            weights = softmax(scores, target, num_nodes=num_nodes)
            edge_messages = weights.unsqueeze(-1) * projected_edges * projected_nodes[source]
            messages = scatter(
                edge_messages,
                target,
                dim=0,
                dim_size=num_nodes,
                reduce="sum",
            )
        else:
            messages = x.new_zeros((num_nodes, self.heads, self.hidden_dim))

        return messages.reshape(num_nodes, -1) @ self.scale_weight + self.bias


class TripletMessageBlock(nn.Module):
    """Repeat one source-aligned triplet-attention/GRU/LayerNorm block."""

    def __init__(
        self,
        hidden_dim: int,
        bond_dim: int,
        heads: int = 4,
        num_timesteps: int = 3,
    ) -> None:
        super().__init__()
        _positive_int(num_timesteps, "num_timesteps")
        self.num_timesteps = num_timesteps
        self.attention = MultiHeadTripletAttention(hidden_dim, bond_dim, heads)
        self.gru = nn.GRU(hidden_dim, hidden_dim)
        self.layer_norm = nn.LayerNorm(hidden_dim)

    def forward(self, x: Tensor, edge_index: Tensor, edge_attr: Tensor) -> Tensor:
        """Return the normalized message state after recurrent updates."""

        message_state = x
        recurrent_hidden = x.unsqueeze(0)
        for _ in range(self.num_timesteps):
            message = F.celu(self.attention(message_state, edge_index, edge_attr))
            gru_output, recurrent_hidden = self.gru(message.unsqueeze(0), recurrent_hidden)
            message_state = self.layer_norm(gru_output.squeeze(0))
        return message_state


def _validate_layer_inputs(
    x: Tensor,
    edge_index: Tensor,
    edge_attr: Tensor,
    hidden_dim: int,
    bond_dim: int,
) -> None:
    if x.ndim != 2 or x.shape[1] != hidden_dim:
        raise ValueError(f"x must have shape [N, {hidden_dim}]")
    if edge_index.ndim != 2 or edge_index.shape[0] != 2:
        raise ValueError("edge_index must have shape [2, E]")
    if edge_index.dtype != torch.long:
        raise ValueError("edge_index must have dtype torch.long")
    if edge_attr.ndim != 2 or edge_attr.shape != (edge_index.shape[1], bond_dim):
        raise ValueError(f"edge_attr must have shape [E, {bond_dim}]")
    if edge_index.numel() and (edge_index.min() < 0 or edge_index.max() >= x.shape[0]):
        raise ValueError("edge_index contains an invalid node index")


def _positive_int(value: int, field: str) -> None:
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise ValueError(f"{field} must be a positive integer")


__all__ = ["MultiHeadTripletAttention", "TripletMessageBlock"]
