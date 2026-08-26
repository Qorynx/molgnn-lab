"""MGCN multilevel encoder and atomwise graph readout (Lu et al., AAAI 2019)."""

from __future__ import annotations

import torch
from torch import Tensor, nn
from torch_geometric.data import Batch
from torch_geometric.utils import scatter

from ..base import BaseMolecularModel
from ..contracts import validate_batched_molecular_graph
from .constants import (
    MGCN_ETA,
    MGCN_HIDDEN_DIM,
    MGCN_MAX_ATOMIC_NUMBER,
    MGCN_NUM_LAYERS,
    MGCN_NUM_RBF,
    MGCN_RBF_BETA,
    MGCN_RBF_HIGH,
    MGCN_RBF_LOW,
)
from .layers import GaussianRBF, InteractionLayer, PairEmbedding


class MGCN(BaseMolecularModel):
    """Coordinate-dependent invariant MGCN over a complete directed graph.

    The initial atom state is a learned embedding of atomic number; the
    initial edge state is a shared learned embedding of the unordered
    element pair.  Interaction layers implement paper Eqs. (5)--(6) and
    concatenate all ``num_layers + 1`` levels before an atomwise two-layer
    Softplus readout that is summed per graph.  Predictions are returned
    raw with shape ``[B, num_targets]``.
    """

    required_batch_fields = (
        "atomic_number",
        "pos",
        "mgcn_edge_index",
        "batch",
    )

    def __init__(
        self,
        num_targets: int,
        hidden_dim: int = MGCN_HIDDEN_DIM,
        num_layers: int = MGCN_NUM_LAYERS,
        num_rbf: int = MGCN_NUM_RBF,
        rbf_low: float = MGCN_RBF_LOW,
        rbf_high: float = MGCN_RBF_HIGH,
        rbf_beta: float = MGCN_RBF_BETA,
        eta: float = MGCN_ETA,
        max_atomic_number: int = MGCN_MAX_ATOMIC_NUMBER,
        readout_hidden_dim: int | None = None,
    ) -> None:
        super().__init__()
        if isinstance(num_targets, bool) or not isinstance(num_targets, int) or num_targets < 1:
            raise ValueError("num_targets must be a positive integer")
        for value, name in (
            (hidden_dim, "hidden_dim"),
            (num_layers, "num_layers"),
            (num_rbf, "num_rbf"),
            (max_atomic_number, "max_atomic_number"),
        ):
            if isinstance(value, bool) or not isinstance(value, int) or value < 1:
                raise ValueError(f"{name} must be a positive integer")
        if eta < 0 or eta > 1:
            raise ValueError("eta must be in [0, 1]")
        if rbf_high <= rbf_low:
            raise ValueError("rbf_high must be greater than rbf_low")
        if rbf_beta <= 0:
            raise ValueError("rbf_beta must be positive")
        if readout_hidden_dim is not None and (
            isinstance(readout_hidden_dim, bool)
            or not isinstance(readout_hidden_dim, int)
            or readout_hidden_dim < 1
        ):
            raise ValueError("readout_hidden_dim must be a positive integer or None")

        self.num_targets = num_targets
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.num_rbf = num_rbf
        self.rbf_low = float(rbf_low)
        self.rbf_high = float(rbf_high)
        self.rbf_beta = float(rbf_beta)
        self.eta = float(eta)
        self.max_atomic_number = max_atomic_number
        self.readout_hidden_dim = (
            hidden_dim if readout_hidden_dim is None else readout_hidden_dim
        )

        self.atom_embedding = nn.Embedding(max_atomic_number + 1, hidden_dim)
        self.pair_embedding = PairEmbedding(hidden_dim, max_atomic_number)
        self.radial_basis = GaussianRBF(
            num_rbf, low=rbf_low, high=rbf_high, beta=rbf_beta
        )
        self.interactions = nn.ModuleList(
            InteractionLayer(hidden_dim, num_rbf, eta=eta)
            for _ in range(num_layers)
        )
        self.readout = nn.Sequential(
            nn.Linear((num_layers + 1) * hidden_dim, self.readout_hidden_dim),
            nn.Softplus(beta=1.0, threshold=20),
            nn.Linear(self.readout_hidden_dim, num_targets),
        )

    def forward(self, batch: Batch) -> Tensor:
        """Return raw graph-level predictions with shape ``[B, T]``."""

        atomic_number, pos, edge_index, graph_batch, num_graphs = self._validate_batch(
            batch
        )
        source, target = edge_index
        distances = torch.linalg.vector_norm(pos[source] - pos[target], dim=-1)
        rbf = self.radial_basis(distances)

        atom = self.atom_embedding(atomic_number)
        edge = self.pair_embedding(
            atomic_number[source], atomic_number[target]
        )

        levels = [atom]
        for layer in self.interactions:
            atom, edge = layer(atom, edge, rbf, edge_index)
            levels.append(atom)
        atom_multilevel = torch.cat(levels, dim=-1)

        atom_outputs = self.readout(atom_multilevel)
        return scatter(
            atom_outputs, graph_batch, dim=0, dim_size=num_graphs, reduce="sum"
        )

    def _validate_batch(
        self, batch: Batch
    ) -> tuple[Tensor, Tensor, Tensor, Tensor, int]:
        values = tuple(getattr(batch, name, None) for name in self.required_batch_fields)
        if not all(isinstance(value, Tensor) for value in values):
            raise ValueError(
                f"batch must provide {', '.join(self.required_batch_fields)} tensors"
            )
        atomic_number, pos, edge_index, graph_batch = values
        assert isinstance(atomic_number, Tensor)
        assert isinstance(pos, Tensor)
        assert isinstance(edge_index, Tensor)
        assert isinstance(graph_batch, Tensor)
        node_count = atomic_number.shape[0] if atomic_number.ndim == 1 else -1
        if (
            atomic_number.ndim != 1
            or node_count < 1
            or atomic_number.dtype != torch.long
        ):
            raise ValueError(
                "batch.atomic_number must have shape [N] and dtype torch.long with N >= 1"
            )
        if atomic_number.min() < 1 or atomic_number.max() > self.max_atomic_number:
            raise ValueError("batch.atomic_number contains an invalid element")
        if pos.shape != (node_count, 3) or pos.dtype != torch.float32:
            raise ValueError("batch.pos must have shape [N, 3] and dtype torch.float32")
        if not bool(torch.isfinite(pos).all()):
            raise ValueError("batch.pos must contain only finite values")
        if any(value.device != pos.device for value in values):
            raise ValueError("all MGCN batch tensors must share the position device")
        num_graphs = validate_batched_molecular_graph(
            edge_index,
            graph_batch,
            num_nodes=node_count,
            device=pos.device,
            edge_field="mgcn_edge_index",
            forbid_self_loops=True,
        )
        self._validate_complete_graph(pos, edge_index, num_graphs, graph_batch)
        return atomic_number, pos, edge_index, graph_batch, num_graphs

    def _validate_complete_graph(
        self, pos: Tensor, edge_index: Tensor, num_graphs: int, graph_batch: Tensor
    ) -> None:
        """Validate MGCN's complete directed reciprocal graph per sample.

        Every ordered pair ``(i, j)`` with ``i != j`` must appear exactly
        once, so each graph contributes exactly ``N*(N-1)`` directed edges
        with no duplicates, self-loops, or cross-graph edges.
        """
        source, target = edge_index
        node_count = pos.shape[0]
        if edge_index.numel() == 0:
            if node_count <= 1:
                return
            raise ValueError(
                "batch.mgcn_edge_index is empty for a multi-atom graph"
            )
        encoded = source * node_count + target
        if torch.unique(encoded).numel() != encoded.numel():
            raise ValueError("batch.mgcn_edge_index must not contain duplicate edges")
        if bool((source == target).any()):
            raise ValueError("batch.mgcn_edge_index must not contain self-loops")
        reverse = target * node_count + source
        if not bool(torch.isin(reverse, encoded).all()):
            raise ValueError("batch.mgcn_edge_index must contain reciprocal edges")
        # Per-graph completeness: graph g must hold exactly N_g*(N_g-1) edges.
        for graph_id in range(num_graphs):
            node_mask = graph_batch == graph_id
            edge_mask = graph_batch[source] == graph_id
            nodes_in_graph = int(node_mask.sum().item())
            edges_in_graph = int(edge_mask.sum().item())
            expected = nodes_in_graph * (nodes_in_graph - 1)
            if edges_in_graph != expected:
                raise ValueError(
                    "batch.mgcn_edge_index must be a complete directed graph; "
                    f"graph {graph_id} has {edges_in_graph} edges for "
                    f"{nodes_in_graph} nodes"
                )


def _positive_int(value: int, field: str) -> None:
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise ValueError(f"{field} must be a positive integer")


__all__ = ["MGCN"]