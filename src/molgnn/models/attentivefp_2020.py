"""AttentiveFP 2020 over the canonical sparse PyG graph contract."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor, nn
from torch.nn import functional as F
from torch_geometric.data import Batch
from torch_geometric.nn import global_add_pool
from torch_geometric.utils import scatter, softmax

from .base import BaseMolecularModel
from .contracts import validate_batched_molecular_graph


@dataclass(frozen=True)
class AttentiveFPTrace:
    """Attention and embedding snapshots returned by the diagnostic API."""

    atom_attention: tuple[Tensor, ...]
    molecule_attention: tuple[Tensor, ...]
    atom_embeddings: tuple[Tensor, ...]
    fingerprint: Tensor


class AttentiveFP(BaseMolecularModel):
    """AttentiveFP adapted to canonical ``x/edge_index/edge_attr/batch`` data.

    The atom attention and recurrent molecular readout follow the training
    implementation in the original source. Features, sparse batching, the
    binary head, and the shared training lifecycle are intentionally adapted
    to this project; this is not a paper-score reproduction.
    """

    required_batch_fields = ("x", "edge_index", "edge_attr", "batch")

    def __init__(
        self,
        atom_dim: int,
        bond_dim: int,
        hidden_dim: int = 200,
        num_atom_layers: int = 2,
        num_molecule_layers: int = 2,
        num_targets: int = 1,
        dropout: float = 0.2,
    ) -> None:
        super().__init__()
        for value, name in (
            (atom_dim, "atom_dim"),
            (bond_dim, "bond_dim"),
            (hidden_dim, "hidden_dim"),
            (num_atom_layers, "num_atom_layers"),
            (num_molecule_layers, "num_molecule_layers"),
            (num_targets, "num_targets"),
        ):
            _positive_int(value, name)
        if (
            isinstance(dropout, bool)
            or not isinstance(dropout, (float, int))
            or not 0 <= dropout < 1
        ):
            raise ValueError("dropout must be in [0, 1)")

        self.atom_dim = atom_dim
        self.bond_dim = bond_dim
        self.dropout = float(dropout)
        self.num_molecule_layers = num_molecule_layers
        self.atom_projection = nn.Linear(atom_dim, hidden_dim)
        self.neighbor_projection = nn.Linear(atom_dim + bond_dim, hidden_dim)
        self.atom_align = nn.ModuleList(
            [nn.Linear(2 * hidden_dim, 1) for _ in range(num_atom_layers)]
        )
        self.atom_attend = nn.ModuleList(
            [nn.Linear(hidden_dim, hidden_dim) for _ in range(num_atom_layers)]
        )
        self.atom_gru = nn.ModuleList(
            [nn.GRUCell(hidden_dim, hidden_dim) for _ in range(num_atom_layers)]
        )
        self.molecule_align = nn.Linear(2 * hidden_dim, 1)
        self.molecule_attend = nn.Linear(hidden_dim, hidden_dim)
        self.molecule_gru = nn.GRUCell(hidden_dim, hidden_dim)
        self.predictor = nn.Linear(hidden_dim, num_targets)

    def forward(self, batch: Batch) -> Tensor:
        """Return raw regression predictions or classification logits."""

        prediction, _ = self._forward_impl(batch, collect_trace=False)
        return prediction

    def forward_with_trace(self, batch: Batch) -> tuple[Tensor, AttentiveFPTrace]:
        """Return predictions plus per-step atom and molecule attention."""

        prediction, trace = self._forward_impl(batch, collect_trace=True)
        assert trace is not None
        return prediction, trace

    def _forward_impl(
        self, batch: Batch, *, collect_trace: bool
    ) -> tuple[Tensor, AttentiveFPTrace | None]:
        x, edge_index, edge_attr, graph_batch = _batch_tensors(
            batch, self.atom_dim, self.bond_dim
        )
        num_nodes = x.shape[0]
        num_graphs = int(graph_batch.max().item()) + 1
        source, target = edge_index

        atom_hidden = F.leaky_relu(self.atom_projection(x))
        atom_features = atom_hidden
        atom_embeddings: list[Tensor] = [atom_features]
        atom_attention: list[Tensor] = []
        neighbor_state = F.leaky_relu(
            self.neighbor_projection(torch.cat((x[source], edge_attr), dim=-1))
        )

        for layer, (align, attend, gru) in enumerate(
            zip(self.atom_align, self.atom_attend, self.atom_gru, strict=True)
        ):
            if layer:
                neighbor_state = atom_features[source]
            alignment_input = torch.cat((atom_features[target], neighbor_state), dim=-1)
            scores = F.leaky_relu(
                align(
                    F.dropout(alignment_input, p=self.dropout, training=self.training)
                )
            ).squeeze(-1)
            weights = _segmented_softmax(scores, target, num_nodes)
            values = attend(
                F.dropout(neighbor_state, p=self.dropout, training=self.training)
            )
            context = _scatter_context(weights, values, target, num_nodes)
            atom_hidden = gru(F.elu(context), atom_hidden)
            atom_features = F.relu(atom_hidden)
            atom_attention.append(weights)
            atom_embeddings.append(atom_features)

        fingerprint = global_add_pool(atom_features, graph_batch, size=num_graphs)
        molecule_attention: list[Tensor] = []
        molecule_state = fingerprint
        for _ in range(self.num_molecule_layers):
            query = F.relu(molecule_state)[graph_batch]
            alignment_input = torch.cat((query, atom_features), dim=-1)
            scores = F.leaky_relu(self.molecule_align(alignment_input)).squeeze(-1)
            weights = _segmented_softmax(scores, graph_batch, num_graphs)
            values = self.molecule_attend(
                F.dropout(atom_features, p=self.dropout, training=self.training)
            )
            context = _scatter_context(weights, values, graph_batch, num_graphs)
            molecule_state = self.molecule_gru(F.elu(context), molecule_state)
            molecule_attention.append(weights)

        prediction = self.predictor(
            F.dropout(molecule_state, p=self.dropout, training=self.training)
        )
        if not collect_trace:
            return prediction, None
        return prediction, AttentiveFPTrace(
            atom_attention=tuple(atom_attention),
            molecule_attention=tuple(molecule_attention),
            atom_embeddings=tuple(atom_embeddings),
            fingerprint=molecule_state,
        )


def _batch_tensors(
    batch: Batch, atom_dim: int, bond_dim: int
) -> tuple[Tensor, Tensor, Tensor, Tensor]:
    values = tuple(
        getattr(batch, name, None) for name in ("x", "edge_index", "edge_attr", "batch")
    )
    if not all(isinstance(value, Tensor) for value in values):
        raise ValueError(
            "batch must provide x, edge_index, edge_attr, and batch tensors"
        )
    x, edge_index, edge_attr, graph_batch = values
    assert isinstance(x, Tensor)
    assert isinstance(edge_index, Tensor)
    assert isinstance(edge_attr, Tensor)
    assert isinstance(graph_batch, Tensor)
    if x.ndim != 2 or x.shape[0] < 1 or x.shape[1] != atom_dim:
        raise ValueError(f"batch.x must have shape [N, {atom_dim}] with N >= 1")
    if not torch.is_floating_point(x):
        raise ValueError("batch.x must be a floating tensor")
    if (
        edge_index.ndim != 2
        or edge_index.shape[0] != 2
        or edge_index.dtype != torch.long
    ):
        raise ValueError("batch.edge_index must have shape [2, E] and dtype torch.long")
    if edge_attr.ndim != 2 or edge_attr.shape != (edge_index.shape[1], bond_dim):
        raise ValueError(f"batch.edge_attr must have shape [E, {bond_dim}]")
    if not torch.is_floating_point(edge_attr):
        raise ValueError("batch.edge_attr must be a floating tensor")
    if (
        graph_batch.ndim != 1
        or graph_batch.shape[0] != x.shape[0]
        or graph_batch.dtype != torch.long
    ):
        raise ValueError("batch.batch must have shape [N] and dtype torch.long")
    if graph_batch.numel() == 0 or graph_batch.min() < 0:
        raise ValueError("batch.batch must contain non-negative graph indices")
    if edge_index.numel() and (edge_index.min() < 0 or edge_index.max() >= x.shape[0]):
        raise ValueError("batch.edge_index contains an invalid node index")
    validate_batched_molecular_graph(
        edge_index,
        graph_batch,
        num_nodes=x.shape[0],
        device=x.device,
        forbid_self_loops=True,
    )
    return x, edge_index, edge_attr, graph_batch


def _segmented_softmax(scores: Tensor, index: Tensor, dim_size: int) -> Tensor:
    """Normalize one scalar score vector independently within each segment."""

    return softmax(scores, index, num_nodes=dim_size)


def _scatter_context(
    weights: Tensor, values: Tensor, index: Tensor, dim_size: int
) -> Tensor:
    """Compute a zero-filled segmented weighted sum, including empty groups."""

    return scatter(
        weights.unsqueeze(-1) * values, index, dim=0, dim_size=dim_size, reduce="sum"
    )


def _positive_int(value: int, field: str) -> None:
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise ValueError(f"{field} must be a positive integer")


__all__ = ["AttentiveFP", "AttentiveFPTrace"]
