"""Compact GCN baseline using the canonical atom graph contract."""

from __future__ import annotations

import torch
from torch import Tensor, nn
from torch_geometric.data import Batch
from torch_geometric.nn import GCNConv, global_mean_pool

from .base import BaseMolecularModel
from .contracts import validate_batched_molecular_graph


class GCNBaseline(BaseMolecularModel):
    """Stacked GCN layers over the project's canonical molecular graph.

    The standard runner supplies paired, loop-free molecular bonds, while this
    baseline itself preserves PyG ``GCNConv`` support for directed edges and
    supplied self-loops.  It only enforces safe disjoint batching.
    """

    required_batch_fields = ("x", "edge_index", "batch")

    def __init__(
        self,
        atom_dim: int,
        hidden_dim: int = 64,
        num_layers: int = 3,
        num_targets: int = 1,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        _positive_int(atom_dim, "atom_dim")
        _positive_int(hidden_dim, "hidden_dim")
        _positive_int(num_layers, "num_layers")
        _positive_int(num_targets, "num_targets")
        if (
            isinstance(dropout, bool)
            or not isinstance(dropout, (float, int))
            or not 0 <= dropout < 1
        ):
            raise ValueError("dropout must be in [0, 1)")
        self.atom_dim = atom_dim
        self.dropout = float(dropout)
        self.convs = nn.ModuleList(
            [GCNConv(atom_dim, hidden_dim)]
            + [GCNConv(hidden_dim, hidden_dim) for _ in range(num_layers - 1)]
        )
        self.predictor = nn.Linear(hidden_dim, num_targets)

    def forward(self, batch: Batch) -> Tensor:
        x = getattr(batch, "x", None)
        edge_index = getattr(batch, "edge_index", None)
        graph_batch = getattr(batch, "batch", None)
        if not isinstance(x, Tensor) or not isinstance(edge_index, Tensor):
            raise ValueError("batch must provide x and edge_index tensors")
        if x.ndim != 2 or x.shape[0] < 1 or x.shape[1] != self.atom_dim:
            raise ValueError(
                f"batch.x must have shape [N, {self.atom_dim}] with N >= 1"
            )
        if not torch.is_floating_point(x):
            raise ValueError("batch.x must be a floating tensor")
        if (
            edge_index.ndim != 2
            or edge_index.shape[0] != 2
            or edge_index.dtype != torch.long
        ):
            raise ValueError(
                "batch.edge_index must have shape [2, E] and dtype torch.long"
            )
        if graph_batch is None:
            graph_batch = torch.zeros(x.shape[0], dtype=torch.long, device=x.device)
        if not isinstance(graph_batch, Tensor):
            raise ValueError("batch.batch must be a torch.Tensor when provided")
        num_graphs = validate_batched_molecular_graph(
            edge_index,
            graph_batch,
            num_nodes=x.shape[0],
            device=x.device,
        )
        for convolution in self.convs:
            x = torch.relu(convolution(x, edge_index))
            x = nn.functional.dropout(x, p=self.dropout, training=self.training)
        pooled = global_mean_pool(x, graph_batch, size=num_graphs)
        return self.predictor(pooled)


def _positive_int(value: int, field: str) -> None:
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise ValueError(f"{field} must be a positive integer")


__all__ = ["GCNBaseline"]
