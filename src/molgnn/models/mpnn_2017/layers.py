"""Sparse layers for the typed-bond MPNN message-passing core."""

from __future__ import annotations

import torch
from torch import Tensor, nn
from torch.nn import functional as F
from torch_geometric.utils import scatter


class TypedEdgeMessage(nn.Module):
    """Aggregate independently parameterized incoming and outgoing messages."""

    def __init__(self, num_edge_types: int, hidden_dim: int) -> None:
        super().__init__()
        _positive_int(num_edge_types, "num_edge_types")
        _positive_int(hidden_dim, "hidden_dim")
        self.incoming_weights = nn.Parameter(
            torch.empty(num_edge_types, hidden_dim, hidden_dim)
        )
        self.outgoing_weights = nn.Parameter(
            torch.empty(num_edge_types, hidden_dim, hidden_dim)
        )
        self.message_bias = nn.Parameter(torch.empty(2 * hidden_dim))
        self.reset_parameters()

    def reset_parameters(self) -> None:
        nn.init.xavier_uniform_(self.incoming_weights)
        nn.init.xavier_uniform_(self.outgoing_weights)
        nn.init.zeros_(self.message_bias)

    def forward(
        self,
        hidden_states: Tensor,
        edge_index: Tensor,
        edge_type: Tensor,
    ) -> Tensor:
        """Return one concatenated incoming/outgoing message per node.

        For each sparse edge ``source -> target``, the incoming channel sends
        ``A_in[type] h_source`` to ``target``.  The outgoing channel mirrors
        the legacy directed-adjacency convention by sending
        ``A_out[type] h_target`` to ``source``.
        """

        source, target = edge_index
        incoming_edge_messages = torch.bmm(
            self.incoming_weights[edge_type], hidden_states[source].unsqueeze(-1)
        ).squeeze(-1)
        outgoing_edge_messages = torch.bmm(
            self.outgoing_weights[edge_type], hidden_states[target].unsqueeze(-1)
        ).squeeze(-1)
        incoming = scatter(
            incoming_edge_messages,
            target,
            dim=0,
            dim_size=hidden_states.shape[0],
            reduce="sum",
        )
        outgoing = scatter(
            outgoing_edge_messages,
            source,
            dim=0,
            dim_size=hidden_states.shape[0],
            reduce="sum",
        )
        return torch.cat((incoming, outgoing), dim=-1) + self.message_bias


class GRUUpdate(nn.Module):
    """Bias-free gated update with the source MPNN gate convention."""

    def __init__(self, hidden_dim: int) -> None:
        super().__init__()
        _positive_int(hidden_dim, "hidden_dim")
        message_dim = 2 * hidden_dim
        self.message_update = nn.Linear(message_dim, hidden_dim, bias=False)
        self.state_update = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.message_reset = nn.Linear(message_dim, hidden_dim, bias=False)
        self.state_reset = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.message_candidate = nn.Linear(message_dim, hidden_dim, bias=False)
        self.state_candidate = nn.Linear(hidden_dim, hidden_dim, bias=False)

    def forward(self, hidden_states: Tensor, messages: Tensor) -> Tensor:
        """Apply the source update ``(1-z) h + z h_tilde``."""

        update = torch.sigmoid(
            self.message_update(messages) + self.state_update(hidden_states)
        )
        reset = torch.sigmoid(
            self.message_reset(messages) + self.state_reset(hidden_states)
        )
        candidate = torch.tanh(
            self.message_candidate(messages)
            + self.state_candidate(reset * hidden_states)
        )
        return (1 - update) * hidden_states + update * candidate


class GatedGraphReadout(nn.Module):
    """Permutation-invariant gated sum over final node representations."""

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        num_hidden_layers: int,
        num_targets: int,
        dropout: float,
    ) -> None:
        super().__init__()
        self.gate_network = _FeedForwardNetwork(
            input_dim,
            hidden_dim,
            num_hidden_layers,
            num_targets,
            dropout,
        )
        self.value_network = _FeedForwardNetwork(
            input_dim,
            hidden_dim,
            num_hidden_layers,
            num_targets,
            dropout,
        )

    def forward(
        self, node_states: Tensor, graph_batch: Tensor, num_graphs: int
    ) -> Tensor:
        gate = torch.sigmoid(self.gate_network(node_states))
        values = self.value_network(node_states)
        return scatter(
            gate * values,
            graph_batch,
            dim=0,
            dim_size=num_graphs,
            reduce="sum",
        )


class _FeedForwardNetwork(nn.Module):
    """Source-style ReLU multilayer perceptron with a linear final layer."""

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        num_hidden_layers: int,
        output_dim: int,
        dropout: float,
    ) -> None:
        super().__init__()
        self.hidden_layers = nn.ModuleList()
        previous_dim = input_dim
        for _ in range(num_hidden_layers):
            self.hidden_layers.append(nn.Linear(previous_dim, hidden_dim))
            previous_dim = hidden_dim
        self.output_layer = nn.Linear(previous_dim, output_dim)
        self.dropout = float(dropout)

    def forward(self, values: Tensor) -> Tensor:
        for layer in self.hidden_layers:
            values = F.relu(layer(values))
            values = F.dropout(values, p=self.dropout, training=self.training)
        return self.output_layer(values)


def _positive_int(value: int, field: str) -> None:
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise ValueError(f"{field} must be a positive integer")


__all__ = ["GRUUpdate", "GatedGraphReadout", "TypedEdgeMessage"]
