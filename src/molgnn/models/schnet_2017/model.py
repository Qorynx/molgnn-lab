"""SchNet (Schutt et al., 2017) molecular property predictor."""

from __future__ import annotations

import torch
from torch import Tensor, nn
from torch_geometric.data import Batch
from torch_geometric.utils import scatter

from ..base import BaseMolecularModel
from ..contracts import validate_batched_molecular_graph
from .constants import (
    SCHNET_CUTOFF,
    SCHNET_MAX_ATOMIC_NUMBER,
    SCHNET_RBF_GAMMA,
    SCHNET_RBF_GAP,
)
from .layers import GaussianRBF, InteractionBlock, ShiftedSoftplus


class SchNet(BaseMolecularModel):
    """Continuous-filter convolutional network over a 3-D radius graph.

    This core follows the paper's molecule profile: nuclear-charge embeddings,
    three unshared interaction blocks, Gaussian distance filters, atomwise
    prediction, and sum readout.  It deliberately consumes only its explicit
    spatial view; canonical 2-D bonds and features remain available to other
    architectures but are not inputs to SchNet.
    """

    required_batch_fields = (
        "atomic_number",
        "pos",
        "schnet_edge_index",
        "batch",
    )

    def __init__(
        self,
        num_targets: int,
        hidden_dim: int = 64,
        num_interactions: int = 3,
        num_filters: int = 64,
        readout: str = "sum",
        max_atomic_number: int = SCHNET_MAX_ATOMIC_NUMBER,
    ) -> None:
        super().__init__()
        for value, name in (
            (num_targets, "num_targets"),
            (hidden_dim, "hidden_dim"),
            (num_interactions, "num_interactions"),
            (num_filters, "num_filters"),
            (max_atomic_number, "max_atomic_number"),
        ):
            _positive_int(value, name)
        if hidden_dim < 2:
            raise ValueError("hidden_dim must be at least 2 for SchNet's output head")
        if readout not in {"sum", "mean"}:
            raise ValueError("readout must be either 'sum' or 'mean'")

        self.num_targets = num_targets
        self.hidden_dim = hidden_dim
        self.num_interactions = num_interactions
        self.num_filters = num_filters
        self.readout = readout
        self.max_atomic_number = max_atomic_number
        self.cutoff = SCHNET_CUTOFF

        self.atom_embedding = nn.Embedding(max_atomic_number + 1, hidden_dim)
        self.rbf = GaussianRBF(SCHNET_CUTOFF, SCHNET_RBF_GAP, SCHNET_RBF_GAMMA)
        self.interactions = nn.ModuleList(
            InteractionBlock(hidden_dim, num_filters, self.rbf.num_rbf)
            for _ in range(num_interactions)
        )
        self.output_network = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            ShiftedSoftplus(),
            nn.Linear(hidden_dim // 2, num_targets),
        )
        final_linear = self.output_network[-1]
        assert isinstance(final_linear, nn.Linear)
        nn.init.zeros_(final_linear.weight)
        nn.init.zeros_(final_linear.bias)

    def forward(self, batch: Batch) -> Tensor:
        """Return raw graph-level predictions with shape ``[B, num_targets]``."""

        atomic_number, pos, edge_index, graph_batch, num_graphs = self._validate_batch(
            batch
        )
        source, target = edge_index
        distances = torch.linalg.vector_norm(pos[source] - pos[target], dim=-1)
        radial_basis = self.rbf(distances)

        features = self.atom_embedding(atomic_number)
        for interaction in self.interactions:
            features = interaction(features, edge_index, radial_basis)
        atom_outputs = self.output_network(features)
        return scatter(
            atom_outputs,
            graph_batch,
            dim=0,
            dim_size=num_graphs,
            reduce=self.readout,
        )

    def _validate_batch(
        self, batch: Batch
    ) -> tuple[Tensor, Tensor, Tensor, Tensor, int]:
        values = tuple(getattr(batch, name, None) for name in self.required_batch_fields)
        if not all(isinstance(value, Tensor) for value in values):
            names = ", ".join(self.required_batch_fields)
            raise ValueError(f"batch must provide {names} tensors")
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
            raise ValueError("all SchNet batch tensors must share the position device")

        num_graphs = validate_batched_molecular_graph(
            edge_index,
            graph_batch,
            num_nodes=node_count,
            device=pos.device,
            edge_field="schnet_edge_index",
            forbid_self_loops=True,
        )
        self._validate_radius_edges(pos, edge_index)
        return atomic_number, pos, edge_index, graph_batch, num_graphs

    def _validate_radius_edges(self, pos: Tensor, edge_index: Tensor) -> None:
        """Require the transform's complete, reciprocal fixed-radius topology."""

        if edge_index.numel() == 0:
            return
        source, target = edge_index
        encoded = source * pos.shape[0] + target
        if torch.unique(encoded).numel() != encoded.numel():
            raise ValueError("batch.schnet_edge_index must not contain duplicate edges")
        reverse = target * pos.shape[0] + source
        if not bool(torch.isin(reverse, encoded).all()):
            raise ValueError("batch.schnet_edge_index must contain reciprocal radius edges")
        distances = torch.linalg.vector_norm(pos[source] - pos[target], dim=-1)
        if bool((distances > self.cutoff).any()):
            raise ValueError("batch.schnet_edge_index contains an edge beyond the fixed cutoff")


def _positive_int(value: int, field: str) -> None:
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise ValueError(f"{field} must be a positive integer")


__all__ = ["SchNet"]
