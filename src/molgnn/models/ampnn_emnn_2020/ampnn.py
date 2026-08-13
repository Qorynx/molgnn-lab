"""Sparse source-faithful Attention GGNN (AMPNN) architecture."""

from __future__ import annotations

import math
from collections.abc import Sequence

import torch
from torch import Tensor, nn
from torch_geometric.data import Batch

from ..base import BaseMolecularModel
from ..contracts import validate_batched_molecular_graph
from .layers import GatedGraphGather, SELUFeedForward, vector_attention_aggregate


class AMPNN(BaseMolecularModel):
    """Attention GGNN with relation-specific vector message attention.

    The bundled ``ampnn_edge_types`` transform derives four relation labels
    (single, double, triple, aromatic) from canonical bond features.  The
    constructor intentionally permits another positive number of relations so
    an external featurizer may supply a compatible typed graph explicitly.
    Atom states start as the raw supplied ``x`` features; no atom encoder is
    inserted before the tied, bias-free GRUCell.
    """

    required_batch_fields = ("x", "edge_index", "ampnn_edge_type", "batch")

    def __init__(
        self,
        atom_dim: int,
        message_dim: int = 25,
        num_message_passing_steps: int = 3,
        message_hidden_dims: Sequence[int] = (200, 200, 200, 200),
        attention_hidden_dims: Sequence[int] = (200, 200, 200),
        gather_dim: int = 100,
        gather_gate_hidden_dims: Sequence[int] = (100, 100, 100),
        gather_value_hidden_dims: Sequence[int] = (100, 100, 100),
        predictor_hidden_dims: Sequence[int] = (100, 100),
        dropout: float = 0.0,
        num_targets: int = 1,
        num_edge_types: int = 4,
    ) -> None:
        super().__init__()
        for value, name in (
            (atom_dim, "atom_dim"),
            (message_dim, "message_dim"),
            (num_message_passing_steps, "num_message_passing_steps"),
            (gather_dim, "gather_dim"),
            (num_targets, "num_targets"),
            (num_edge_types, "num_edge_types"),
        ):
            _positive_int(value, name)
        _validate_dropout(dropout)

        self.atom_dim = atom_dim
        self.message_dim = message_dim
        self.num_message_passing_steps = num_message_passing_steps
        self.gather_dim = gather_dim
        self.num_targets = num_targets
        self.num_edge_types = num_edge_types
        self.dropout = float(dropout)

        self.message_networks = nn.ModuleList(
            [
                SELUFeedForward(
                    atom_dim,
                    message_hidden_dims,
                    message_dim,
                    dropout=dropout,
                    bias=False,
                )
                for _ in range(num_edge_types)
            ]
        )
        self.attention_networks = nn.ModuleList(
            [
                SELUFeedForward(
                    atom_dim,
                    attention_hidden_dims,
                    message_dim,
                    dropout=dropout,
                    bias=False,
                )
                for _ in range(num_edge_types)
            ]
        )
        self.gru = nn.GRUCell(message_dim, atom_dim, bias=False)
        self.graph_gather = GatedGraphGather(
            atom_dim,
            atom_dim,
            gather_dim,
            gate_hidden_dims=gather_gate_hidden_dims,
            value_hidden_dims=gather_value_hidden_dims,
            gate_dropout=dropout,
            value_dropout=dropout,
        )
        self.predictor = SELUFeedForward(
            gather_dim,
            predictor_hidden_dims,
            num_targets,
            dropout=dropout,
            bias=False,
        )

    def forward(self, batch: Batch) -> Tensor:
        """Return one raw prediction/logit vector for every graph in ``batch``."""

        x, edge_index, edge_type, graph_batch, num_graphs = self._batch_tensors(batch)
        hidden = x
        for _ in range(self.num_message_passing_steps):
            messages = self._aggregate_messages(hidden, edge_index, edge_type)
            hidden = self.gru(messages, hidden)
        graph_embeddings = self.graph_gather(hidden, x, graph_batch, num_graphs)
        return self.predictor(graph_embeddings)

    def _aggregate_messages(
        self, hidden: Tensor, edge_index: Tensor, edge_type: Tensor
    ) -> Tensor:
        """Return target-indexed AMPNN vector-attention messages.

        All relation FFNNs process every source state before the relation
        label selects one result.  This matches the dense reference's masked
        relation branches while keeping the PyG graph sparse.
        """

        source, target = edge_index
        if source.numel() == 0:
            return hidden.new_zeros((hidden.shape[0], self.message_dim))

        source_states = hidden[source]
        embeddings = _select_relation_outputs(
            self.message_networks, source_states, edge_type
        )
        energies = _select_relation_outputs(
            self.attention_networks, source_states, edge_type
        )
        return vector_attention_aggregate(
            embeddings,
            energies,
            target,
            hidden.shape[0],
        )

    def _batch_tensors(
        self, batch: Batch
    ) -> tuple[Tensor, Tensor, Tensor, Tensor, int]:
        names = self.required_batch_fields
        values = tuple(getattr(batch, name, None) for name in names)
        if not all(isinstance(value, Tensor) for value in values):
            raise ValueError(f"batch must provide {', '.join(names)} tensors")
        x, edge_index, edge_type, graph_batch = values
        assert isinstance(x, Tensor)
        assert isinstance(edge_index, Tensor)
        assert isinstance(edge_type, Tensor)
        assert isinstance(graph_batch, Tensor)

        edge_count = edge_index.shape[1] if edge_index.ndim == 2 else -1
        if x.ndim != 2 or x.shape[0] < 1 or x.shape[1] != self.atom_dim:
            raise ValueError(
                f"batch.x must have shape [N, {self.atom_dim}] with N >= 1"
            )
        if x.dtype != torch.float32 or not torch.isfinite(x).all():
            raise ValueError("batch.x must contain finite torch.float32 values")
        if (
            edge_index.ndim != 2
            or edge_index.shape[0] != 2
            or edge_index.dtype != torch.long
        ):
            raise ValueError(
                "batch.edge_index must have shape [2, E] and dtype torch.long"
            )
        if edge_type.shape != (edge_count,) or edge_type.dtype != torch.long:
            raise ValueError(
                "batch.ampnn_edge_type must have shape [E] and dtype torch.long"
            )
        if graph_batch.shape != (x.shape[0],) or graph_batch.dtype != torch.long:
            raise ValueError("batch.batch must have shape [N] and dtype torch.long")
        if edge_type.device != x.device:
            raise ValueError("batch.ampnn_edge_type must share the node device")
        if graph_batch.numel() == 0 or graph_batch.min() < 0:
            raise ValueError("batch.batch must contain non-negative graph indices")
        if edge_count and (edge_index.min() < 0 or edge_index.max() >= x.shape[0]):
            raise ValueError("batch.edge_index contains an invalid node index")
        num_graphs = validate_batched_molecular_graph(
            edge_index,
            graph_batch,
            num_nodes=x.shape[0],
            device=x.device,
            forbid_self_loops=True,
        )
        if edge_count and (
            edge_type.min() < 0 or edge_type.max() >= self.num_edge_types
        ):
            raise ValueError("batch.ampnn_edge_type contains an invalid edge type")
        return x, edge_index, edge_type, graph_batch, num_graphs


def _select_relation_outputs(
    networks: nn.ModuleList, source_states: Tensor, edge_type: Tensor
) -> Tensor:
    """Evaluate all source-style relation FFNNs, then select per edge."""

    relation_outputs = torch.stack(
        tuple(network(source_states) for network in networks), dim=1
    )
    edge_positions = torch.arange(edge_type.shape[0], device=edge_type.device)
    return relation_outputs[edge_positions, edge_type]


def _positive_int(value: int, field: str) -> None:
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise ValueError(f"{field} must be a positive integer")


def _validate_dropout(value: float) -> None:
    if (
        isinstance(value, bool)
        or not isinstance(value, (float, int))
        or not math.isfinite(float(value))
        or not 0 <= value < 1
    ):
        raise ValueError("dropout must be a finite value in [0, 1)")


__all__ = ["AMPNN"]
