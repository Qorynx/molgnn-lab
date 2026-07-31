"""Architecture-only TrimNet 2020 molecular property model."""

from __future__ import annotations

import torch
from torch import Tensor, nn
from torch.nn import functional as F
from torch_geometric.data import Batch
from torch_geometric.nn import Set2Set

from ..base import BaseMolecularModel
from ..contracts import validate_batched_molecular_graph
from .layers import TripletMessageBlock, _positive_int


class TrimNet2020(BaseMolecularModel):
    """Source-aligned TrimNet adapted to the canonical PyG batch contract."""

    required_batch_fields = ("x", "edge_index", "edge_attr", "batch")

    def __init__(
        self,
        atom_dim: int,
        bond_dim: int,
        hidden_dim: int = 32,
        depth: int = 3,
        heads: int = 4,
        num_timesteps: int = 3,
        num_targets: int = 1,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        for value, field in (
            (atom_dim, "atom_dim"),
            (bond_dim, "bond_dim"),
            (hidden_dim, "hidden_dim"),
            (depth, "depth"),
            (heads, "heads"),
            (num_timesteps, "num_timesteps"),
            (num_targets, "num_targets"),
        ):
            _positive_int(value, field)
        if (
            isinstance(dropout, bool)
            or not isinstance(dropout, (float, int))
            or not 0 <= dropout < 1
        ):
            raise ValueError("dropout must be in [0, 1)")

        self.atom_dim = atom_dim
        self.bond_dim = bond_dim
        self.dropout = float(dropout)
        self.input_projection = nn.Linear(atom_dim, hidden_dim)
        self.blocks = nn.ModuleList(
            [
                TripletMessageBlock(hidden_dim, bond_dim, heads, num_timesteps)
                for _ in range(depth)
            ]
        )
        self.readout = Set2Set(hidden_dim, processing_steps=3)
        self.predictor = nn.Sequential(
            nn.Linear(2 * hidden_dim, 512),
            nn.LayerNorm(512),
            nn.ReLU(inplace=True),
            nn.Dropout(self.dropout),
            nn.Linear(512, num_targets),
        )

    def forward(self, batch: Batch) -> Tensor:
        """Return raw regression predictions or classification logits."""

        x, edge_index, edge_attr, graph_batch = _batch_tensors(
            batch, self.atom_dim, self.bond_dim
        )
        x = F.celu(self.input_projection(x))
        for block in self.blocks:
            x = x + F.dropout(
                block(x, edge_index, edge_attr),
                p=self.dropout,
                training=self.training,
            )
        graph_embedding = self.readout(x, graph_batch)
        graph_embedding = F.dropout(
            graph_embedding, p=self.dropout, training=self.training
        )
        return self.predictor(graph_embedding)


def _batch_tensors(
    batch: Batch, atom_dim: int, bond_dim: int
) -> tuple[Tensor, Tensor, Tensor, Tensor]:
    values = tuple(
        getattr(batch, field, None)
        for field in ("x", "edge_index", "edge_attr", "batch")
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
    if edge_index.ndim != 2 or edge_index.shape[0] != 2:
        raise ValueError("batch.edge_index must have shape [2, E]")
    if edge_index.dtype != torch.long:
        raise ValueError("batch.edge_index must have dtype torch.long")
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


__all__ = ["TrimNet2020"]
