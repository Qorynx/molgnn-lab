"""Typed-bond MPNN over the project's sparse molecular graph contract."""

from __future__ import annotations

import torch
from torch import Tensor
from torch.nn import functional as F
from torch_geometric.data import Batch

from ..base import BaseMolecularModel
from ..contracts import validate_batched_molecular_graph
from .layers import GRUUpdate, GatedGraphReadout, TypedEdgeMessage


class MPNN(BaseMolecularModel):
    """A tied typed-bond MPNN with a gated graph-level readout.

    The model consumes sparse directed bonds and their discrete bond-type
    labels.  The bundled transform derives those labels from canonical bond
    features; callers using another featurizer may provide compatible labels
    directly.
    """

    required_batch_fields = ("x", "edge_index", "mpnn_edge_type", "batch")
    edge_index_field = "edge_index"
    edge_type_field = "mpnn_edge_type"

    def __init__(
        self,
        atom_dim: int,
        hidden_dim: int = 200,
        num_edge_types: int = 4,
        num_message_passing_steps: int = 6,
        readout_hidden_dim: int = 200,
        readout_num_hidden_layers: int = 1,
        num_targets: int = 1,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        for value, name in (
            (atom_dim, "atom_dim"),
            (hidden_dim, "hidden_dim"),
            (num_edge_types, "num_edge_types"),
            (num_message_passing_steps, "num_message_passing_steps"),
            (readout_hidden_dim, "readout_hidden_dim"),
            (num_targets, "num_targets"),
        ):
            _positive_int(value, name)
        _nonnegative_int(readout_num_hidden_layers, "readout_num_hidden_layers")
        if hidden_dim < atom_dim:
            raise ValueError("hidden_dim must be at least atom_dim for zero-padded h0")
        if (
            isinstance(dropout, bool)
            or not isinstance(dropout, (float, int))
            or not 0 <= dropout < 1
        ):
            raise ValueError("dropout must be in [0, 1)")

        self.atom_dim = atom_dim
        self.hidden_dim = hidden_dim
        self.num_edge_types = num_edge_types
        self.num_message_passing_steps = num_message_passing_steps
        self.message_function = TypedEdgeMessage(num_edge_types, hidden_dim)
        self.update_function = GRUUpdate(hidden_dim)
        self.graph_readout = GatedGraphReadout(
            hidden_dim + atom_dim,
            readout_hidden_dim,
            readout_num_hidden_layers,
            num_targets,
            float(dropout),
        )

    def forward(self, batch: Batch) -> Tensor:
        """Return raw regression predictions or classification logits."""

        x, edge_index, edge_type, graph_batch, num_graphs = self._batch_tensors(batch)
        hidden = F.pad(x, (0, self.hidden_dim - self.atom_dim))
        for _ in range(self.num_message_passing_steps):
            messages = self.message_function(hidden, edge_index, edge_type)
            hidden = self.update_function(hidden, messages)
        return self.graph_readout(
            torch.cat((hidden, x), dim=-1), graph_batch, num_graphs
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

        edge_index_field = self.edge_index_field
        edge_type_field = self.edge_type_field
        edge_count = edge_index.shape[1] if edge_index.ndim == 2 else -1
        if x.ndim != 2 or x.shape[0] < 1 or x.shape[1] != self.atom_dim:
            raise ValueError(f"batch.x must have shape [N, {self.atom_dim}] with N >= 1")
        if x.dtype != torch.float32:
            raise ValueError("batch.x must have dtype torch.float32")
        if (
            edge_index.ndim != 2
            or edge_index.shape[0] != 2
            or edge_index.dtype != torch.long
        ):
            raise ValueError(
                f"batch.{edge_index_field} must have shape [2, E] and dtype torch.long"
            )
        if edge_type.shape != (edge_count,) or edge_type.dtype != torch.long:
            raise ValueError(
                f"batch.{edge_type_field} must have shape [E] and dtype torch.long"
            )
        if edge_type.device != x.device:
            raise ValueError(f"batch.{edge_type_field} must share the node device")
        if graph_batch.shape != (x.shape[0],) or graph_batch.dtype != torch.long:
            raise ValueError("batch.batch must have shape [N] and dtype torch.long")
        if graph_batch.numel() == 0 or graph_batch.min() < 0:
            raise ValueError("batch.batch must contain non-negative graph indices")
        if edge_count and (edge_index.min() < 0 or edge_index.max() >= x.shape[0]):
            raise ValueError(f"batch.{edge_index_field} contains an invalid node index")
        num_graphs = validate_batched_molecular_graph(
            edge_index,
            graph_batch,
            num_nodes=x.shape[0],
            device=x.device,
            edge_field=edge_index_field,
            forbid_self_loops=True,
        )
        if edge_count and (edge_type.min() < 0 or edge_type.max() >= self.num_edge_types):
            raise ValueError(f"batch.{edge_type_field} contains an invalid edge type")
        return x, edge_index, edge_type, graph_batch, num_graphs


class MPNNDistanceBins3D(MPNN):
    """Typed MPNN over the explicit 3D distance-bin graph contract.

    This preserves the tied GGNN-style message/update/readout topology of
    :class:`MPNN`, but consumes the all-pairs graph derived from supplied
    molecular coordinates. The four covalent relation types and ten
    non-covalent distance bins form a fixed 14-class edge vocabulary.
    """

    required_batch_fields = (
        "x",
        "mpnn_3d_edge_index",
        "mpnn_3d_edge_type",
        "batch",
    )
    edge_index_field = "mpnn_3d_edge_index"
    edge_type_field = "mpnn_3d_edge_type"
    num_distance_bin_edge_types = 14

    def __init__(
        self,
        atom_dim: int,
        hidden_dim: int = 200,
        num_message_passing_steps: int = 6,
        readout_hidden_dim: int = 200,
        readout_num_hidden_layers: int = 1,
        num_targets: int = 1,
        dropout: float = 0.0,
    ) -> None:
        super().__init__(
            atom_dim=atom_dim,
            hidden_dim=hidden_dim,
            num_edge_types=self.num_distance_bin_edge_types,
            num_message_passing_steps=num_message_passing_steps,
            readout_hidden_dim=readout_hidden_dim,
            readout_num_hidden_layers=readout_num_hidden_layers,
            num_targets=num_targets,
            dropout=dropout,
        )


def _positive_int(value: int, field: str) -> None:
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise ValueError(f"{field} must be a positive integer")


def _nonnegative_int(value: int, field: str) -> None:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError(f"{field} must be a non-negative integer")


__all__ = ["MPNN", "MPNNDistanceBins3D"]
