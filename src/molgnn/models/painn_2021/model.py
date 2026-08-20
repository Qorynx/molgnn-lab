"""Project-facing graph-level PaiNN model."""

from __future__ import annotations

import torch
from torch import Tensor, nn
from torch.autograd import grad
from torch_geometric.data import Batch
from torch_geometric.utils import scatter

from ..base import BaseMolecularModel
from ..contracts import validate_batched_molecular_graph
from .constants import (
    PAINN_CUTOFF,
    PAINN_EPS,
    PAINN_MAX_ATOMIC_NUMBER,
    PAINN_NUM_RBF,
)
from .layers import PaiNNInteraction, PaiNNMixing, PaiNNDense
from .radial import CosineCutoff, build_radial_basis


class PaiNN(BaseMolecularModel):
    """PaiNN scalar/vector message-passing model over a 3-D radius graph."""

    required_batch_fields = (
        "atomic_number",
        "pos",
        "painn_edge_index",
        "batch",
    )

    def __init__(
        self,
        num_targets: int,
        hidden_dim: int = 128,
        num_interactions: int = 3,
        num_rbf: int = PAINN_NUM_RBF,
        cutoff: float = PAINN_CUTOFF,
        radial_basis: str = "bessel",
        readout: str = "sum",
        max_atomic_number: int = PAINN_MAX_ATOMIC_NUMBER,
        epsilon: float = PAINN_EPS,
    ) -> None:
        super().__init__()
        for value, name in (
            (num_targets, "num_targets"),
            (hidden_dim, "hidden_dim"),
            (num_interactions, "num_interactions"),
            (num_rbf, "num_rbf"),
            (max_atomic_number, "max_atomic_number"),
        ):
            _positive_int(value, name)
        if hidden_dim < 2:
            raise ValueError("hidden_dim must be at least 2")
        if readout not in {"sum", "mean"}:
            raise ValueError("readout must be either 'sum' or 'mean'")
        if radial_basis == "gaussian" and num_rbf < 2:
            raise ValueError("num_rbf must be at least 2 for Gaussian radial_basis")
        if cutoff <= 0 or epsilon <= 0:
            raise ValueError("cutoff and epsilon must be positive")

        self.num_targets = num_targets
        self.hidden_dim = hidden_dim
        self.num_interactions = num_interactions
        self.num_rbf = num_rbf
        self.cutoff = float(cutoff)
        self.radial_basis_name = radial_basis
        self.readout = readout
        self.max_atomic_number = max_atomic_number
        self.epsilon = float(epsilon)

        self.atom_embedding = nn.Embedding(max_atomic_number + 1, hidden_dim)
        self.radial_basis = build_radial_basis(
            radial_basis, cutoff=self.cutoff, num_rbf=num_rbf
        )
        self.cutoff_fn = CosineCutoff(self.cutoff)
        self.filter_net = PaiNNDense(
            num_rbf, num_interactions * 3 * hidden_dim
        )
        self.interactions = nn.ModuleList(
            PaiNNInteraction(hidden_dim) for _ in range(num_interactions)
        )
        self.mixing = nn.ModuleList(
            PaiNNMixing(hidden_dim, epsilon=self.epsilon)
            for _ in range(num_interactions)
        )
        self.output_network = nn.Sequential(
            PaiNNDense(hidden_dim, hidden_dim // 2, activation=nn.SiLU()),
            PaiNNDense(hidden_dim // 2, num_targets),
        )

    def forward(self, batch: Batch) -> Tensor:
        """Return raw graph-level predictions with shape ``[B, T]``."""

        atomic_number, pos, edge_index, graph_batch, num_graphs = self._validate_batch(
            batch
        )
        scalar, vector = self._encode(atomic_number, pos, edge_index)
        atom_outputs = self.output_network(scalar)
        return scatter(
            atom_outputs,
            graph_batch,
            dim=0,
            dim_size=num_graphs,
            reduce=self.readout,
        )

    def forward_with_forces(self, batch: Batch) -> tuple[Tensor, Tensor]:
        """Return predictions and conservative forces from live coordinates."""

        if not batch.pos.requires_grad:
            raise ValueError("batch.pos must require gradients to calculate forces")
        prediction = self(batch)
        coordinate_grad = grad(
            prediction.sum(),
            batch.pos,
            create_graph=self.training,
            retain_graph=True,
            allow_unused=True,
        )[0]
        forces = torch.zeros_like(batch.pos) if coordinate_grad is None else -coordinate_grad
        return prediction, forces

    def _encode(
        self, atomic_number: Tensor, pos: Tensor, edge_index: Tensor
    ) -> tuple[Tensor, Tensor]:
        source, target = edge_index
        displacement = pos[source] - pos[target]
        distances = torch.linalg.vector_norm(displacement, dim=-1)
        if distances.numel() and bool((distances <= self.epsilon).any()):
            raise ValueError("painn_edge_index contains a coincident atom pair")
        directions = displacement / distances.clamp_min(self.epsilon).unsqueeze(-1)
        radial = self.radial_basis(distances)
        filters = self.filter_net(radial)
        filters = filters * self.cutoff_fn(distances).unsqueeze(-1)
        filter_chunks = filters.split(3 * self.hidden_dim, dim=-1)

        scalar = self.atom_embedding(atomic_number)
        vector = torch.zeros(
            (atomic_number.shape[0], 3, self.hidden_dim),
            dtype=scalar.dtype,
            device=scalar.device,
        )
        for interaction, mixing, filter_weight in zip(
            self.interactions, self.mixing, filter_chunks
        ):
            scalar, vector = interaction(
                scalar, vector, filter_weight, edge_index, directions
            )
            scalar, vector = mixing(scalar, vector)
        return scalar, vector

    def _validate_batch(
        self, batch: Batch
    ) -> tuple[Tensor, Tensor, Tensor, Tensor, int]:
        values = tuple(getattr(batch, name, None) for name in self.required_batch_fields)
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
            raise ValueError("batch.atomic_number contains an invalid element")
        if pos.shape != (node_count, 3) or pos.dtype != torch.float32:
            raise ValueError("batch.pos must have shape [N, 3] and dtype torch.float32")
        if not bool(torch.isfinite(pos).all()):
            raise ValueError("batch.pos must contain only finite values")
        if any(value.device != pos.device for value in values):
            raise ValueError("all PaiNN batch tensors must share the position device")
        num_graphs = validate_batched_molecular_graph(
            edge_index,
            graph_batch,
            num_nodes=node_count,
            device=pos.device,
            edge_field="painn_edge_index",
            forbid_self_loops=True,
        )
        self._validate_spatial_edges(pos, edge_index)
        return atomic_number, pos, edge_index, graph_batch, num_graphs

    def _validate_spatial_edges(self, pos: Tensor, edge_index: Tensor) -> None:
        if edge_index.numel() == 0:
            return
        source, target = edge_index
        encoded = source * pos.shape[0] + target
        if torch.unique(encoded).numel() != encoded.numel():
            raise ValueError("batch.painn_edge_index must not contain duplicate edges")
        reverse = target * pos.shape[0] + source
        if not bool(torch.isin(reverse, encoded).all()):
            raise ValueError("batch.painn_edge_index must contain reciprocal radius edges")
        distances = torch.linalg.vector_norm(pos[source] - pos[target], dim=-1)
        if bool((distances >= self.cutoff).any()):
            raise ValueError("batch.painn_edge_index contains an edge outside the fixed cutoff")
        if bool((distances <= self.epsilon).any()):
            raise ValueError("batch.painn_edge_index contains a coincident atom pair")


def _positive_int(value: int, field: str) -> None:
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise ValueError(f"{field} must be a positive integer")


__all__ = ["PaiNN"]
