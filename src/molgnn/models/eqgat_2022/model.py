"""Graph-level EQGAT predictor over an explicit 3-D spatial graph."""

from __future__ import annotations

import torch
from torch import Tensor, nn
from torch.nn import functional
from torch_geometric.data import Batch
from torch_geometric.utils import scatter

from ..base import BaseMolecularModel
from ..contracts import validate_batched_molecular_graph
from .constants import (
    EQGAT_CUTOFF,
    EQGAT_EPS,
    EQGAT_MAX_ATOMIC_NUMBER,
    EQGAT_MAX_NEIGHBORS,
    EQGAT_NUM_RADIAL,
)
from .layers import DenseLayer, EQGATConv, GatedEquivariantBlock


class EQGAT(BaseMolecularModel):
    """EQGAT's graph-level scalar/vector architecture for 3-D molecules.

    The model consumes only its explicit radius graph ``eqgat_edge_index``.
    Coordinates remain live forward inputs, so the scalar output is invariant
    to rigid motions while the internal vector channels are SO(3)-equivariant.
    The cross-product term is intentionally retained; this is not an
    O(3)-equivariant/reflection-invariant architecture.
    """

    required_batch_fields = (
        "atomic_number",
        "pos",
        "eqgat_edge_index",
        "batch",
    )

    def __init__(
        self,
        num_targets: int,
        scalar_dim: int = 100,
        vector_dim: int = 16,
        depth: int = 3,
        num_radial: int = EQGAT_NUM_RADIAL,
        vector_aggr: str = "mean",
        graph_pooling: str = "mean",
        dropout: float = 0.0,
        max_atomic_number: int = EQGAT_MAX_ATOMIC_NUMBER,
    ) -> None:
        super().__init__()
        for value, name in (
            (num_targets, "num_targets"),
            (scalar_dim, "scalar_dim"),
            (vector_dim, "vector_dim"),
            (depth, "depth"),
            (num_radial, "num_radial"),
            (max_atomic_number, "max_atomic_number"),
        ):
            _positive_int(value, name)
        if vector_aggr not in {"mean", "sum"}:
            raise ValueError("vector_aggr must be either 'mean' or 'sum'")
        if graph_pooling not in {"mean", "sum"}:
            raise ValueError("graph_pooling must be either 'mean' or 'sum'")
        if not isinstance(dropout, (float, int)) or not 0.0 <= dropout < 1.0:
            raise ValueError("dropout must be in [0, 1)")

        self.num_targets = num_targets
        self.scalar_dim = scalar_dim
        self.vector_dim = vector_dim
        self.depth = depth
        self.num_radial = num_radial
        self.vector_aggr = vector_aggr
        self.graph_pooling = graph_pooling
        self.max_atomic_number = max_atomic_number
        self.cutoff = EQGAT_CUTOFF
        self.max_neighbors = EQGAT_MAX_NEIGHBORS

        self.atom_embedding = nn.Embedding(max_atomic_number + 1, scalar_dim)
        self.convs = nn.ModuleList(
            EQGATConv(
                scalar_dim,
                vector_dim,
                num_radial,
                self.cutoff,
                has_vector_input=index > 0,
                vector_aggr=vector_aggr,
                eps=EQGAT_EPS,
            )
            for index in range(depth)
        )
        self.scalarization = GatedEquivariantBlock(
            scalar_dim,
            vector_dim,
            scalar_dim,
            None,
            scalar_hidden=scalar_dim,
            vector_hidden=vector_dim,
            eps=EQGAT_EPS,
        )
        self.output_network = nn.Sequential(
            DenseLayer(scalar_dim, scalar_dim, activation=nn.SiLU()),
            nn.Dropout(float(dropout)),
            DenseLayer(scalar_dim, num_targets),
        )

    def forward(self, batch: Batch) -> Tensor:
        """Return graph-level raw predictions with shape ``[B, num_targets]``."""

        atomic_number, pos, edge_index, graph_batch, num_graphs = self._validate_batch(batch)
        scalar, vector = self._encode(atomic_number, pos, edge_index)
        scalar, _ = self.scalarization(scalar, vector)
        graph_features = scatter(
            scalar,
            graph_batch,
            dim=0,
            dim_size=num_graphs,
            reduce=self.graph_pooling,
        )
        return self.output_network(graph_features)

    def _encode(
        self,
        atomic_number: Tensor,
        pos: Tensor,
        edge_index: Tensor,
    ) -> tuple[Tensor, Tensor]:
        """Return final node scalar/vector states for structural tests and readout."""

        source, target = edge_index
        displacement = pos[source] - pos[target]
        distances = torch.linalg.vector_norm(displacement, dim=-1)
        directions = functional.normalize(displacement, dim=-1, eps=EQGAT_EPS)
        scalar = self.atom_embedding(atomic_number)
        vector = torch.zeros(
            (atomic_number.shape[0], 3, self.vector_dim),
            dtype=scalar.dtype,
            device=scalar.device,
        )
        for conv in self.convs:
            scalar, vector = conv(scalar, vector, edge_index, distances, directions)
        return scalar, vector

    def _validate_batch(
        self, batch: Batch
    ) -> tuple[Tensor, Tensor, Tensor, Tensor, int]:
        values = tuple(getattr(batch, field, None) for field in self.required_batch_fields)
        if not all(isinstance(value, Tensor) for value in values):
            raise ValueError(f"batch must provide {', '.join(self.required_batch_fields)} tensors")
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
            raise ValueError(
                "batch.atomic_number contains a value outside the configured vocabulary"
            )
        if pos.shape != (node_count, 3) or pos.dtype != torch.float32:
            raise ValueError("batch.pos must have shape [N, 3] and dtype torch.float32")
        if not bool(torch.isfinite(pos).all()):
            raise ValueError("batch.pos must contain only finite values")
        if any(value.device != pos.device for value in values):
            raise ValueError("all EQGAT batch tensors must share the position device")

        num_graphs = validate_batched_molecular_graph(
            edge_index,
            graph_batch,
            num_nodes=node_count,
            device=pos.device,
            edge_field="eqgat_edge_index",
            forbid_self_loops=True,
        )
        self._validate_spatial_edges(pos, edge_index)
        return atomic_number, pos, edge_index, graph_batch, num_graphs

    def _validate_spatial_edges(self, pos: Tensor, edge_index: Tensor) -> None:
        """Require the transform's unique, capped, strict-radius topology."""

        if edge_index.numel() == 0:
            return
        source, target = edge_index
        encoded = source * pos.shape[0] + target
        if torch.unique(encoded).numel() != encoded.numel():
            raise ValueError("batch.eqgat_edge_index must not contain duplicate edges")
        incoming = torch.bincount(target, minlength=pos.shape[0])
        if bool((incoming > self.max_neighbors).any()):
            raise ValueError(
                "batch.eqgat_edge_index exceeds EQGAT's maximum incoming-neighbor cap"
            )
        distances = torch.linalg.vector_norm(pos[source] - pos[target], dim=-1)
        if bool((distances >= self.cutoff).any()):
            raise ValueError("batch.eqgat_edge_index contains an edge outside the fixed cutoff")


def _positive_int(value: int, field: str) -> None:
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise ValueError(f"{field} must be a positive integer")


__all__ = ["EQGAT"]
