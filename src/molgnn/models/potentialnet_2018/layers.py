"""Sparse, staged layers for the PotentialNet 2018 architecture."""

from __future__ import annotations

from collections.abc import Sequence

import torch
from torch import Tensor, nn
from torch.nn import functional as F
from torch_geometric.utils import scatter


class TypedMessageMLP(nn.Module):
    """Sum nonlinear, type-specific messages over a sparse directed graph.

    Each edge type owns an independent two-layer MLP.  Repeated sparse edges
    are intentional: they represent multiple active slices of PotentialNet's
    adjacency tensor for the same ordered atom pair.
    """

    def __init__(
        self, num_edge_types: int, state_dim: int, hidden_dim: int
    ) -> None:
        super().__init__()
        _positive_int(num_edge_types, "num_edge_types")
        _positive_int(state_dim, "state_dim")
        _positive_int(hidden_dim, "hidden_dim")
        self.num_edge_types = num_edge_types
        self.state_dim = state_dim
        self.hidden_dim = hidden_dim
        self.networks = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Linear(state_dim, hidden_dim),
                    nn.ReLU(),
                    nn.Linear(hidden_dim, state_dim),
                )
                for _ in range(num_edge_types)
            ]
        )

    def reset_parameters(self) -> None:
        """Reset every independently parameterized message network."""

        for network in self.networks:
            for layer in network:
                if isinstance(layer, nn.Linear):
                    layer.reset_parameters()

    def forward(
        self, hidden_states: Tensor, edge_index: Tensor, edge_type: Tensor
    ) -> Tensor:
        """Return the summed incoming message for every node.

        ``edge_index[0] -> edge_index[1]`` follows the usual PyG message
        direction.  The stage reuses this module at every recurrent step,
        which ties its message parameters within that stage.
        """

        edge_count = edge_index.shape[1]
        if edge_count == 0:
            return hidden_states.new_zeros((hidden_states.shape[0], self.state_dim))

        source, target = edge_index
        edge_messages = hidden_states.new_zeros((edge_count, self.state_dim))
        for type_index, network in enumerate(self.networks):
            positions = (edge_type == type_index).nonzero(as_tuple=False).flatten()
            if positions.numel() == 0:
                continue
            typed_states = hidden_states[source.index_select(0, positions)]
            edge_messages = edge_messages.index_copy(
                0, positions, network(typed_states)
            )
        return scatter(
            edge_messages,
            target,
            dim=0,
            dim_size=hidden_states.shape[0],
            reduce="sum",
        )


class StageGate(nn.Module):
    """PotentialNet's ``sigmoid(i([h, input])) * j(h)`` projection."""

    def __init__(self, input_dim: int, state_dim: int, gather_dim: int) -> None:
        super().__init__()
        _positive_int(input_dim, "input_dim")
        _positive_int(state_dim, "state_dim")
        _positive_int(gather_dim, "gather_dim")
        self.input_dim = input_dim
        self.state_dim = state_dim
        self.gather_dim = gather_dim
        self.gate_network = nn.Linear(state_dim + input_dim, gather_dim)
        self.value_network = nn.Linear(state_dim, gather_dim)

    def reset_parameters(self) -> None:
        """Reset both paper-specified gate networks."""

        self.gate_network.reset_parameters()
        self.value_network.reset_parameters()

    def forward(self, states: Tensor, stage_input: Tensor) -> Tensor:
        """Project recurrent state to the stage's gathered atom embedding."""

        gate = torch.sigmoid(self.gate_network(torch.cat((states, stage_input), dim=-1)))
        return gate * self.value_network(states)


class TypedRecurrentStage(nn.Module):
    """One PotentialNet recurrent stage with tied typed messages and a GRU."""

    def __init__(
        self,
        input_dim: int,
        state_dim: int,
        gather_dim: int,
        num_edge_types: int,
        num_steps: int,
        message_hidden_dim: int,
        dropout: float,
    ) -> None:
        super().__init__()
        for value, name in (
            (input_dim, "input_dim"),
            (state_dim, "state_dim"),
            (gather_dim, "gather_dim"),
            (num_edge_types, "num_edge_types"),
            (num_steps, "num_steps"),
            (message_hidden_dim, "message_hidden_dim"),
        ):
            _positive_int(value, name)
        _dropout(dropout)
        if state_dim < input_dim:
            raise ValueError("state_dim must be at least input_dim for zero-padded h0")

        self.input_dim = input_dim
        self.state_dim = state_dim
        self.gather_dim = gather_dim
        self.num_edge_types = num_edge_types
        self.num_steps = num_steps
        self.message_network = TypedMessageMLP(
            num_edge_types, state_dim, message_hidden_dim
        )
        self.gru = nn.GRUCell(state_dim, state_dim)
        self.dropout = nn.Dropout(float(dropout))
        self.gate = StageGate(input_dim, state_dim, gather_dim)

    def reset_parameters(self) -> None:
        """Reset all tied parameters for this stage."""

        self.message_network.reset_parameters()
        self.gru.reset_parameters()
        self.gate.reset_parameters()

    def forward(
        self, stage_input: Tensor, edge_index: Tensor, edge_type: Tensor
    ) -> Tensor:
        """Run zero-padded recurrent propagation then gated projection."""

        states = F.pad(stage_input, (0, self.state_dim - self.input_dim))
        for _ in range(self.num_steps):
            messages = self.message_network(states, edge_index, edge_type)
            states = self.gru(messages, states)
        return self.gate(self.dropout(states), stage_input)


class LigandReadout(nn.Module):
    """Ligand-only sum pooling followed by a ReLU MLP and raw output head."""

    def __init__(
        self,
        input_dim: int,
        hidden_dims: Sequence[int],
        num_targets: int,
        dropout: float,
    ) -> None:
        super().__init__()
        _positive_int(input_dim, "input_dim")
        _positive_int(num_targets, "num_targets")
        _dropout(dropout)
        widths = _positive_widths(hidden_dims, "hidden_dims")

        self.input_dim = input_dim
        self.hidden_dims = widths
        self.num_targets = num_targets
        self.dropout_probability = float(dropout)
        previous_dim = input_dim
        self.hidden_layers = nn.ModuleList()
        for width in widths:
            self.hidden_layers.append(nn.Linear(previous_dim, width))
            previous_dim = width
        self.output_layer = nn.Linear(previous_dim, num_targets)

    def reset_parameters(self) -> None:
        """Reset the graph-level predictor."""

        for layer in self.hidden_layers:
            layer.reset_parameters()
        self.output_layer.reset_parameters()

    def forward(
        self,
        node_states: Tensor,
        ligand_mask: Tensor,
        graph_batch: Tensor,
        num_graphs: int,
    ) -> Tensor:
        """Pool only ligand atoms, then return raw graph-level predictions."""

        ligand_embedding = scatter(
            node_states[ligand_mask],
            graph_batch[ligand_mask],
            dim=0,
            dim_size=num_graphs,
            reduce="sum",
        )
        hidden = ligand_embedding
        for layer_index, layer in enumerate(self.hidden_layers):
            if layer_index:
                hidden = F.dropout(
                    hidden, p=self.dropout_probability, training=self.training
                )
            hidden = F.relu(layer(hidden))
        return self.output_layer(hidden)


def _positive_int(value: int, field: str) -> None:
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise ValueError(f"{field} must be a positive integer")


def _positive_widths(values: Sequence[int], field: str) -> tuple[int, ...]:
    if isinstance(values, (str, bytes)):
        raise ValueError(f"{field} must be a non-empty sequence of positive integers")
    try:
        widths = tuple(values)
    except TypeError as exc:
        raise ValueError(
            f"{field} must be a non-empty sequence of positive integers"
        ) from exc
    if not widths:
        raise ValueError(f"{field} must be a non-empty sequence of positive integers")
    for width in widths:
        _positive_int(width, field)
    return widths


def _dropout(value: float) -> None:
    if isinstance(value, bool) or not isinstance(value, (float, int)) or not 0 <= value < 1:
        raise ValueError("dropout must be in [0, 1)")


__all__ = ["LigandReadout", "StageGate", "TypedMessageMLP", "TypedRecurrentStage"]
