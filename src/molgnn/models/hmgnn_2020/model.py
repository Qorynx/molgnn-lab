"""Order-two HMGNN molecular property predictor."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor, nn
from torch.nn import functional
from torch_geometric.data import Batch
from torch_geometric.utils import scatter

from ..base import BaseMolecularModel, ModelTrainingOutput
from .basis import AngleRBF, DistanceRBF, ShrinkDistanceRBF
from .constants import (
    HMGNN_CUTOFF,
    HMGNN_MAX_ATOMIC_NUMBER,
    HMGNN_NUM_ANGULAR,
    HMGNN_NUM_RADIAL,
)
from .layers import (
    Dense,
    HeterogeneousInteractionBlock,
    SafeBatchNorm1d,
    ShiftedSoftplus,
)


@dataclass(frozen=True)
class HMGNNComponents:
    prediction: Tensor
    order_predictions: Tensor
    attention: Tensor


class OrderOutputHead(nn.Module):
    def __init__(self, hidden_dim: int, num_targets: int, num_types: int) -> None:
        super().__init__()
        self.raw_output = Dense(hidden_dim, num_targets)
        self.scale = nn.Embedding(num_types, num_targets)
        self.shift = nn.Embedding(num_types, num_targets)
        nn.init.zeros_(self.scale.weight)
        nn.init.zeros_(self.shift.weight)

    def forward(
        self,
        states: Tensor,
        type_index: Tensor,
        graph_index: Tensor,
        num_graphs: int,
    ) -> tuple[Tensor, Tensor]:
        node_output = self.scale(type_index) * self.raw_output(states) + self.shift(type_index)
        graph_output = scatter(
            node_output, graph_index, dim=0, dim_size=num_graphs, reduce="sum"
        )
        graph_features = scatter(
            states, graph_index, dim=0, dim_size=num_graphs, reduce="sum"
        )
        return graph_output, graph_features


class HMGNN(BaseMolecularModel):
    """HMGNN with atom and distance-defined two-body graph orders."""

    required_batch_fields = (
        "atomic_number",
        "pos",
        "hmgnn_atom_edge_index",
        "hmgnn_body_atom_index",
        "hmgnn_body_edge_index",
        "batch",
    )

    def __init__(
        self,
        num_targets: int,
        hidden_dim: int = 128,
        num_interactions: int = 5,
        num_residual: int = 1,
        num_radial: int = HMGNN_NUM_RADIAL,
        num_angular: int = HMGNN_NUM_ANGULAR,
        cutoff: float = HMGNN_CUTOFF,
        max_atomic_number: int = HMGNN_MAX_ATOMIC_NUMBER,
    ) -> None:
        super().__init__()
        for value, name in (
            (num_targets, "num_targets"),
            (hidden_dim, "hidden_dim"),
            (num_interactions, "num_interactions"),
            (num_residual, "num_residual"),
            (num_radial, "num_radial"),
            (num_angular, "num_angular"),
            (max_atomic_number, "max_atomic_number"),
        ):
            _positive_int(value, name)
        if not isinstance(cutoff, (float, int)) or cutoff <= 0:
            raise ValueError("cutoff must be positive")
        if abs(float(cutoff) - HMGNN_CUTOFF) > 1e-12:
            raise ValueError(
                f"cutoff must be {HMGNN_CUTOFF}; topology is built by the registered transform"
            )

        self.num_targets = num_targets
        self.hidden_dim = hidden_dim
        self.max_atomic_number = max_atomic_number
        self.cutoff = float(cutoff)
        self.num_pair_types = max_atomic_number * (max_atomic_number + 1) // 2

        self.atom_embedding = nn.Embedding(max_atomic_number + 1, hidden_dim)
        self.pair_embedding = nn.Embedding(self.num_pair_types, hidden_dim)
        limit = 3.0**0.5
        nn.init.uniform_(self.atom_embedding.weight, -limit, limit)
        nn.init.uniform_(self.pair_embedding.weight, -limit, limit)

        self.atom_input = Dense(hidden_dim, hidden_dim, activation=ShiftedSoftplus())
        self.body_input = Dense(
            hidden_dim + num_radial, hidden_dim, activation=ShiftedSoftplus()
        )
        self.atom_distance_basis = ShrinkDistanceRBF(num_radial, self.cutoff)
        self.body_distance_basis = DistanceRBF(num_radial, self.cutoff)
        self.angle_basis = AngleRBF(num_angular)
        self.atom_edge_input = Dense(
            num_radial, hidden_dim, activation=ShiftedSoftplus()
        )
        self.body_edge_input = Dense(
            num_angular, hidden_dim, activation=ShiftedSoftplus()
        )
        self.interactions = nn.ModuleList(
            HeterogeneousInteractionBlock(hidden_dim, num_residual)
            for _ in range(num_interactions)
        )
        self.atom_output = OrderOutputHead(
            hidden_dim, num_targets, max_atomic_number + 1
        )
        self.body_output = OrderOutputHead(
            hidden_dim, num_targets, self.num_pair_types
        )
        self.fusion_norm = SafeBatchNorm1d(2 * hidden_dim)
        self.fusion_hidden = Dense(
            2 * hidden_dim, 2 * hidden_dim, activation=ShiftedSoftplus()
        )
        self.fusion_attention = nn.Linear(2 * hidden_dim, 2 * num_targets, bias=False)
        nn.init.uniform_(self.fusion_attention.weight, -limit, limit)

    def forward(self, batch: Batch) -> Tensor:
        return self.forward_components(batch).prediction

    def forward_training(self, batch: Batch) -> ModelTrainingOutput:
        components = self.forward_components(batch)
        return ModelTrainingOutput(
            components.prediction,
            (components.order_predictions[:, 0], components.order_predictions[:, 1]),
        )

    def forward_components(self, batch: Batch) -> HMGNNComponents:
        (
            atomic_number,
            pos,
            atom_edge_index,
            body_atom_index,
            body_edge_index,
            graph_batch,
            num_graphs,
        ) = self._validate_batch(batch)
        pair_type = self._pair_type(atomic_number[body_atom_index])
        body_batch = graph_batch[body_atom_index[0]]

        atom_states = self.atom_input(self.atom_embedding(atomic_number))
        body_distance = torch.linalg.vector_norm(
            pos[body_atom_index[0]] - pos[body_atom_index[1]], dim=-1
        )
        body_states = self.body_input(
            torch.cat(
                (self.pair_embedding(pair_type), self.body_distance_basis(body_distance)),
                dim=-1,
            )
        )
        atom_distance = torch.linalg.vector_norm(
            pos[atom_edge_index[0]] - pos[atom_edge_index[1]], dim=-1
        )
        atom_edge_features = self.atom_edge_input(
            self.atom_distance_basis(atom_distance)
        )
        angles = _body_angles(pos, body_atom_index, body_edge_index)
        body_edge_features = self.body_edge_input(self.angle_basis(angles))

        for interaction in self.interactions:
            atom_states, body_states = interaction(
                atom_states,
                body_states,
                atom_edge_index,
                atom_edge_features,
                body_atom_index,
                body_edge_index,
                body_edge_features,
            )
        atom_prediction, atom_features = self.atom_output(
            atom_states, atomic_number, graph_batch, num_graphs
        )
        body_prediction, body_features = self.body_output(
            body_states, pair_type, body_batch, num_graphs
        )
        fused_features = self.fusion_hidden(
            self.fusion_norm(torch.cat((atom_features, body_features), dim=-1))
        )
        attention = functional.softmax(
            functional.leaky_relu(self.fusion_attention(fused_features)).view(
                num_graphs, self.num_targets, 2
            ),
            dim=-1,
        )
        order_predictions = torch.stack((atom_prediction, body_prediction), dim=1)
        prediction = (attention * order_predictions.transpose(1, 2)).sum(dim=-1)
        return HMGNNComponents(prediction, order_predictions, attention)

    def _pair_type(self, pair_atomic_numbers: Tensor) -> Tensor:
        low = pair_atomic_numbers.min(dim=0).values
        high = pair_atomic_numbers.max(dim=0).values
        zero_based = low - 1
        return (
            zero_based * (2 * self.max_atomic_number + 1 - zero_based) // 2
            + high
            - low
        )

    def _validate_batch(
        self, batch: Batch
    ) -> tuple[Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, int]:
        values: list[Tensor] = []
        for field in self.required_batch_fields:
            value = getattr(batch, field, None)
            if not isinstance(value, Tensor):
                raise TypeError(f"HMGNN requires tensor batch field '{field}'")
            values.append(value)
        atomic_number, pos, atom_edges, body_atoms, body_edges, graph_batch = values
        node_count = atomic_number.numel()
        if (
            atomic_number.ndim != 1
            or atomic_number.dtype != torch.long
            or node_count < 1
            or bool((atomic_number < 1).any())
            or bool((atomic_number > self.max_atomic_number).any())
        ):
            raise ValueError("atomic_number must be positive long [N] within the configured vocabulary")
        if pos.shape != (node_count, 3) or not torch.is_floating_point(pos) or not bool(
            torch.isfinite(pos).all()
        ):
            raise ValueError("pos must be finite floating point [N, 3]")
        if graph_batch.shape != (node_count,) or graph_batch.dtype != torch.long:
            raise ValueError("batch must be long [N]")
        for index, name in (
            (atom_edges, "hmgnn_atom_edge_index"),
            (body_atoms, "hmgnn_body_atom_index"),
            (body_edges, "hmgnn_body_edge_index"),
        ):
            if index.ndim != 2 or index.shape[0] != 2 or index.dtype != torch.long:
                raise ValueError(f"{name} must be long [2, E]")
        body_count = body_atoms.shape[1]
        _index_range(body_atoms, node_count, "hmgnn_body_atom_index")
        _index_range(atom_edges, node_count, "hmgnn_atom_edge_index")
        _index_range(body_edges, body_count, "hmgnn_body_edge_index")
        if body_count and bool((body_atoms[0] >= body_atoms[1]).any()):
            raise ValueError("hmgnn_body_atom_index must contain canonical unordered pairs i < j")
        if atom_edges.shape[1] != 2 * body_count:
            raise ValueError("hmgnn_atom_edge_index must contain both directions of every body")
        expected_atom_edges = torch.stack(
            (
                torch.cat((body_atoms[0], body_atoms[1])),
                torch.cat((body_atoms[1], body_atoms[0])),
            )
        )
        actual_keys = atom_edges[0] * node_count + atom_edges[1]
        expected_keys = expected_atom_edges[0] * node_count + expected_atom_edges[1]
        if not torch.equal(actual_keys.sort().values, expected_keys.sort().values):
            raise ValueError("hmgnn_atom_edge_index does not match hmgnn_body_atom_index")
        if body_count and not torch.equal(
            graph_batch[body_atoms[0]], graph_batch[body_atoms[1]]
        ):
            raise ValueError("HMGNN bodies cannot cross graph boundaries")
        if body_edges.shape[1]:
            if bool((body_edges[0] == body_edges[1]).any()):
                raise ValueError("hmgnn_body_edge_index cannot contain self edges")
            source_pair = body_atoms[:, body_edges[0]].t()
            target_pair = body_atoms[:, body_edges[1]].t()
            shared = (
                source_pair[:, :, None] == target_pair[:, None, :]
            ).sum(dim=(1, 2))
            if bool((shared != 1).any()):
                raise ValueError("every HMGNN body edge must connect bodies sharing one atom")
        num_graphs = int(getattr(batch, "num_graphs", 0))
        if num_graphs < 1:
            num_graphs = int(graph_batch.max().item()) + 1
        return atomic_number, pos, atom_edges, body_atoms, body_edges, graph_batch, num_graphs


def _body_angles(pos: Tensor, body_atoms: Tensor, body_edges: Tensor) -> Tensor:
    if body_edges.shape[1] == 0:
        return pos.new_empty((0,))
    source_pair = body_atoms[:, body_edges[0]].t()
    target_pair = body_atoms[:, body_edges[1]].t()
    shared_source = (source_pair[:, :, None] == target_pair[:, None, :]).any(dim=2)
    shared_target = (target_pair[:, :, None] == source_pair[:, None, :]).any(dim=2)
    center = source_pair[shared_source]
    source_outer = source_pair[~shared_source]
    target_outer = target_pair[~shared_target]
    source_vector = pos[source_outer] - pos[center]
    target_vector = pos[target_outer] - pos[center]
    cosine = functional.cosine_similarity(source_vector, target_vector, dim=-1).clamp(
        -1.0, 1.0
    )
    return torch.acos(cosine)


def _index_range(index: Tensor, upper: int, name: str) -> None:
    if index.numel() and (upper < 1 or bool((index < 0).any()) or bool((index >= upper).any())):
        raise ValueError(f"{name} contains an out-of-range index")


def _positive_int(value: int, name: str) -> None:
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise ValueError(f"{name} must be a positive integer")


__all__ = ["HMGNN", "HMGNNComponents"]
