"""Sparse, source-audited Edge Message Passing Neural Network (EMNN)."""

from __future__ import annotations

import math
from collections.abc import Sequence

import torch
from torch import Tensor, nn
from torch_geometric.data import Batch
from torch_geometric.utils import scatter

from ..base import BaseMolecularModel
from ..contracts import validate_batched_molecular_graph
from .layers import GatedGraphGather, SELUFeedForward


class EMNN(BaseMolecularModel):
    """Directed-edge EMNN with non-backtracking vector attention.

    Each covalent bond is represented by its paired directed edges.  A static
    oriented edge embedding participates in every message update, while the
    dynamic edge state is propagated from incoming directed edges with the
    immediate reverse edge excluded.  The implementation follows the paper's
    recurrent update ``GRUCell(message, previous_edge_memory)`` rather than
    the legacy source's omitted GRU hidden-state argument.
    """

    required_batch_fields = (
        "x",
        "edge_index",
        "edge_attr",
        "reverse_edge_index",
        "batch",
    )

    def __init__(
        self,
        atom_dim: int,
        bond_dim: int,
        edge_hidden_dim: int = 50,
        num_message_passing_steps: int = 3,
        edge_embedding_hidden_dims: Sequence[int] = (150, 150, 150),
        message_hidden_dims: Sequence[int] = (80, 80, 80),
        attention_hidden_dims: Sequence[int] = (80, 80, 80),
        gather_dim: int = 100,
        gather_gate_hidden_dims: Sequence[int] = (100, 100, 100),
        gather_value_hidden_dims: Sequence[int] = (100, 100, 100),
        predictor_hidden_dims: Sequence[int] = (100, 100),
        dropout: float = 0.0,
        num_targets: int = 1,
    ) -> None:
        super().__init__()
        for value, field in (
            (atom_dim, "atom_dim"),
            (bond_dim, "bond_dim"),
            (edge_hidden_dim, "edge_hidden_dim"),
            (num_message_passing_steps, "num_message_passing_steps"),
            (gather_dim, "gather_dim"),
            (num_targets, "num_targets"),
        ):
            _positive_int(value, field)
        _validate_dropout(dropout)

        edge_embedding_widths = _hidden_dims(
            edge_embedding_hidden_dims, "edge_embedding_hidden_dims"
        )
        message_widths = _hidden_dims(message_hidden_dims, "message_hidden_dims")
        attention_widths = _hidden_dims(attention_hidden_dims, "attention_hidden_dims")
        gather_gate_widths = _hidden_dims(
            gather_gate_hidden_dims, "gather_gate_hidden_dims"
        )
        gather_value_widths = _hidden_dims(
            gather_value_hidden_dims, "gather_value_hidden_dims"
        )
        predictor_widths = _hidden_dims(predictor_hidden_dims, "predictor_hidden_dims")

        self.atom_dim = atom_dim
        self.bond_dim = bond_dim
        self.edge_hidden_dim = edge_hidden_dim
        self.num_message_passing_steps = num_message_passing_steps
        self.edge_embedding_hidden_dims = edge_embedding_widths
        self.message_hidden_dims = message_widths
        self.attention_hidden_dims = attention_widths
        self.gather_dim = gather_dim
        self.gather_gate_hidden_dims = gather_gate_widths
        self.gather_value_hidden_dims = gather_value_widths
        self.predictor_hidden_dims = predictor_widths
        self.dropout = float(dropout)
        self.num_targets = num_targets

        self.edge_embedding = SELUFeedForward(
            atom_dim * 2 + bond_dim,
            edge_embedding_widths,
            edge_hidden_dim,
            dropout=self.dropout,
            bias=False,
        )
        self.message_network = SELUFeedForward(
            edge_hidden_dim,
            message_widths,
            edge_hidden_dim,
            dropout=self.dropout,
            bias=False,
        )
        self.attention_network = SELUFeedForward(
            edge_hidden_dim,
            attention_widths,
            edge_hidden_dim,
            dropout=self.dropout,
            bias=False,
        )
        self.gru = nn.GRUCell(edge_hidden_dim, edge_hidden_dim, bias=False)
        self.graph_readout = GatedGraphGather(
            edge_hidden_dim,
            edge_hidden_dim,
            gather_dim,
            gate_hidden_dims=gather_gate_widths,
            value_hidden_dims=gather_value_widths,
            gate_dropout=self.dropout,
            value_dropout=self.dropout,
        )
        self.predictor = SELUFeedForward(
            gather_dim,
            predictor_widths,
            num_targets,
            dropout=self.dropout,
            bias=False,
        )

    def forward(self, batch: Batch) -> Tensor:
        """Return raw regression predictions or binary-classification logits."""

        return self.predictor(self.fingerprint(batch))

    def fingerprint(self, batch: Batch) -> Tensor:
        """Return the source-backed gated graph representation before prediction."""

        x, edge_index, edge_attr, reverse_edge_index, graph_batch, num_graphs = (
            self._batch_tensors(batch)
        )
        source, _ = edge_index
        edge_count = edge_index.shape[1]
        edge_memories = x.new_zeros((edge_count, self.edge_hidden_dim))

        if edge_count:
            static_edges = torch.tanh(
                self.edge_embedding(
                    torch.cat((x[source], x[edge_index[1]], edge_attr), dim=-1)
                )
            )
            for _ in range(self.num_message_passing_steps):
                messages = self._edge_messages(
                    static_edges,
                    edge_memories,
                    edge_index,
                    reverse_edge_index,
                    num_nodes=x.shape[0],
                )
                edge_memories = self.gru(messages, edge_memories)
            node_states = scatter(
                edge_memories,
                source,
                dim=0,
                dim_size=x.shape[0],
                reduce="sum",
            )
        else:
            node_states = x.new_zeros((x.shape[0], self.edge_hidden_dim))

        # The legacy EMNN source passes its final edge-derived node state to
        # both GraphGather inputs.  The paper does not define a compatible
        # node-level h0 for an arbitrary edge-hidden width, so retain that
        # deliberate source-backed readout instead of adding a new projection.
        return self.graph_readout(node_states, node_states, graph_batch, num_graphs)

    def _edge_messages(
        self,
        static_edges: Tensor,
        edge_memories: Tensor,
        edge_index: Tensor,
        reverse_edge_index: Tensor,
        *,
        num_nodes: int | None = None,
    ) -> Tensor:
        """Aggregate static and non-backtracking predecessor edge messages.

        For edge ``(v, w)``, the predecessor set is every edge ending at
        ``v`` apart from ``(w, v)``.  Rather than materialising a line graph,
        this computes the exact vector-attention set in ``O(E * D)``: aggregate
        incoming dynamic terms by node, subtract the reverse term for each
        edge, and then combine the remaining set with its static embedding.

        The intermediate exponentials are max-shifted independently for each
        node and message coordinate.  When the excluded reverse edge is the
        unique incoming maximum, its group needs a second maximum: subtracting
        it after shifting by the global maximum would underflow all valid
        predecessors.  Selecting that second maximum preserves the exact
        non-backtracking softmax without raw ``exp(energy)`` overflow.
        """

        edge_count = edge_index.shape[1]
        if edge_count == 0:
            return static_edges
        if num_nodes is None:
            num_nodes = int(edge_index.max().item()) + 1

        source, target = edge_index
        static_embeddings = self.message_network(static_edges)
        static_energies = self.attention_network(static_edges)
        dynamic_embeddings = self.message_network(edge_memories)
        dynamic_energies = self.attention_network(edge_memories)

        incoming_maximum = scatter(
            dynamic_energies,
            target,
            dim=0,
            dim_size=num_nodes,
            reduce="max",
        )
        incoming_maximum_per_edge = incoming_maximum[target]
        is_incoming_maximum = dynamic_energies == incoming_maximum_per_edge
        maximum_count = scatter(
            is_incoming_maximum.to(dtype=dynamic_energies.dtype),
            target,
            dim=0,
            dim_size=num_nodes,
            reduce="sum",
        )
        shifted_dynamic_weights = torch.exp(
            dynamic_energies - incoming_maximum_per_edge
        )
        incoming_denominator = scatter(
            shifted_dynamic_weights,
            target,
            dim=0,
            dim_size=num_nodes,
            reduce="sum",
        )
        incoming_numerator = scatter(
            shifted_dynamic_weights * dynamic_embeddings,
            target,
            dim=0,
            dim_size=num_nodes,
            reduce="sum",
        )

        # Per-coordinate second maxima and their aggregates are only needed
        # where the excluded reverse edge is the unique global maximum.  Mask
        # global maxima before computing the group maximum; safe values avoid
        # evaluating ``exp(value - -inf)`` for groups that have no second item.
        no_maximum_energy = torch.full_like(dynamic_energies, -torch.inf)
        nonmaximum_energies = torch.where(
            is_incoming_maximum,
            no_maximum_energy,
            dynamic_energies,
        )
        second_maximum = scatter(
            nonmaximum_energies,
            target,
            dim=0,
            dim_size=num_nodes,
            reduce="max",
        )
        finite_second_maximum = torch.where(
            torch.isfinite(second_maximum),
            second_maximum,
            torch.zeros_like(second_maximum),
        )
        second_reference_per_edge = finite_second_maximum[target]
        safe_nonmaximum_energies = torch.where(
            is_incoming_maximum,
            second_reference_per_edge,
            dynamic_energies,
        )
        shifted_nonmaximum_weights = torch.where(
            is_incoming_maximum,
            torch.zeros_like(dynamic_energies),
            torch.exp(safe_nonmaximum_energies - second_reference_per_edge),
        )
        second_denominator = scatter(
            shifted_nonmaximum_weights,
            target,
            dim=0,
            dim_size=num_nodes,
            reduce="sum",
        )
        second_numerator = scatter(
            shifted_nonmaximum_weights * dynamic_embeddings,
            target,
            dim=0,
            dim_size=num_nodes,
            reduce="sum",
        )

        reverse_weights = shifted_dynamic_weights[reverse_edge_index]
        reverse_embeddings = dynamic_embeddings[reverse_edge_index]
        global_predecessor_denominator = incoming_denominator[source] - reverse_weights
        global_predecessor_numerator = incoming_numerator[source] - (
            reverse_weights * reverse_embeddings
        )

        reverse_is_unique_maximum = is_incoming_maximum[reverse_edge_index] & (
            maximum_count[source] == 1
        )
        predecessor_maximum = torch.where(
            reverse_is_unique_maximum,
            second_maximum[source],
            incoming_maximum[source],
        )
        predecessor_denominator = torch.where(
            reverse_is_unique_maximum,
            second_denominator[source],
            global_predecessor_denominator,
        ).clamp_min(0)
        predecessor_numerator = torch.where(
            reverse_is_unique_maximum,
            second_numerator[source],
            global_predecessor_numerator,
        )

        # Dynamic predecessor aggregates are represented relative to each
        # source node's max energy.  Rebase them together with the static item
        # at a per-edge coordinate-wise maximum before the final weighted sum.
        # With no predecessor (a degree-one atom), use the static item's energy
        # as the harmless reference so the result is exactly its embedding.
        has_predecessor = predecessor_denominator > 0
        safe_predecessor_maximum = torch.where(
            has_predecessor,
            predecessor_maximum,
            static_energies,
        )
        combined_maximum = torch.maximum(static_energies, safe_predecessor_maximum)
        predecessor_scale = torch.exp(safe_predecessor_maximum - combined_maximum)
        static_scale = torch.exp(static_energies - combined_maximum)
        denominator = predecessor_denominator * predecessor_scale + static_scale
        numerator = (
            predecessor_numerator * predecessor_scale + static_scale * static_embeddings
        )
        return numerator / denominator

    def _batch_tensors(
        self, batch: Batch
    ) -> tuple[Tensor, Tensor, Tensor, Tensor, Tensor, int]:
        """Fetch and validate the directed-edge molecular batch contract."""

        names = self.required_batch_fields
        values = tuple(getattr(batch, name, None) for name in names)
        if not all(isinstance(value, Tensor) for value in values):
            raise ValueError(f"batch must provide {', '.join(names)} tensors")
        x, edge_index, edge_attr, reverse_edge_index, graph_batch = values
        assert isinstance(x, Tensor)
        assert isinstance(edge_index, Tensor)
        assert isinstance(edge_attr, Tensor)
        assert isinstance(reverse_edge_index, Tensor)
        assert isinstance(graph_batch, Tensor)

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
        edge_count = edge_index.shape[1]
        if edge_attr.shape != (edge_count, self.bond_dim):
            raise ValueError(f"batch.edge_attr must have shape [E, {self.bond_dim}]")
        if edge_attr.dtype != torch.float32 or not torch.isfinite(edge_attr).all():
            raise ValueError("batch.edge_attr must contain finite torch.float32 values")
        if (
            reverse_edge_index.shape != (edge_count,)
            or reverse_edge_index.dtype != torch.long
        ):
            raise ValueError(
                "batch.reverse_edge_index must have shape [E] and dtype torch.long"
            )
        if graph_batch.shape != (x.shape[0],) or graph_batch.dtype != torch.long:
            raise ValueError("batch.batch must have shape [N] and dtype torch.long")
        if graph_batch.numel() == 0 or graph_batch.min() < 0:
            raise ValueError(
                "batch.batch must contain non-negative torch.long graph indices"
            )
        if edge_attr.device != x.device or reverse_edge_index.device != x.device:
            raise ValueError(
                "batch.x, edge_attr, and reverse_edge_index must share a device"
            )
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
            reverse_edge_index.min() < 0 or reverse_edge_index.max() >= edge_count
        ):
            raise ValueError("batch.reverse_edge_index contains an invalid edge index")
        if not torch.equal(
            reverse_edge_index[reverse_edge_index],
            torch.arange(edge_count, device=reverse_edge_index.device),
        ):
            raise ValueError("batch.reverse_edge_index must be an involution")
        if edge_count and not torch.equal(
            edge_index[:, reverse_edge_index], edge_index.flip(0)
        ):
            raise ValueError(
                "batch.reverse_edge_index must map each edge to its reverse"
            )
        if edge_count and not torch.equal(edge_attr, edge_attr[reverse_edge_index]):
            raise ValueError(
                "batch.edge_attr must match along reverse directed-edge pairs"
            )
        return x, edge_index, edge_attr, reverse_edge_index, graph_batch, num_graphs


def _hidden_dims(values: Sequence[int], field: str) -> tuple[int, ...]:
    if isinstance(values, (str, bytes)) or not isinstance(values, Sequence):
        raise ValueError(f"{field} must be a sequence of positive integers")
    normalized = tuple(values)
    for index, value in enumerate(normalized):
        _positive_int(value, f"{field}[{index}]")
    return normalized


def _validate_dropout(dropout: float) -> None:
    if (
        isinstance(dropout, bool)
        or not isinstance(dropout, (float, int))
        or not math.isfinite(float(dropout))
        or not 0 <= dropout < 1
    ):
        raise ValueError("dropout must be a finite value in [0, 1)")


def _positive_int(value: int, field: str) -> None:
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise ValueError(f"{field} must be a positive integer")


__all__ = ["EMNN"]
