"""HiGNN 2023 hierarchical model for molecular property prediction."""

from __future__ import annotations

import torch
from torch import Tensor, nn
from torch.nn import functional as F
from torch_geometric.data import Batch
from torch_geometric.nn import GATConv
from torch_geometric.utils import scatter

from ..base import BaseMolecularModel
from .layers import FeatureAttention, NTNConv


class HiGNN(BaseMolecularModel):
    """Shared atom/fragment encoder with fragment-to-molecule attention."""

    required_batch_fields = (
        "x",
        "edge_index",
        "edge_attr",
        "brics_edge_index",
        "brics_edge_attr",
        "atom_to_fragment",
        "batch",
    )

    def __init__(
        self,
        atom_dim: int,
        bond_dim: int,
        hidden_dim: int = 64,
        num_layers: int = 3,
        num_slices: int = 2,
        dropout: float = 0.2,
        feature_reduction: int = 4,
        num_targets: int = 1,
    ) -> None:
        super().__init__()
        for value, name in (
            (atom_dim, "atom_dim"),
            (bond_dim, "bond_dim"),
            (hidden_dim, "hidden_dim"),
            (num_layers, "num_layers"),
            (num_slices, "num_slices"),
            (feature_reduction, "feature_reduction"),
            (num_targets, "num_targets"),
        ):
            _positive_int(value, name)
        if hidden_dim % num_slices:
            raise ValueError("hidden_dim must be divisible by num_slices")
        if hidden_dim % feature_reduction:
            raise ValueError("hidden_dim must be divisible by feature_reduction")
        if (
            isinstance(dropout, bool)
            or not isinstance(dropout, (float, int))
            or not 0 <= dropout < 1
        ):
            raise ValueError("dropout must be in [0, 1)")

        self.atom_dim = atom_dim
        self.bond_dim = bond_dim
        self.hidden_dim = hidden_dim
        self.dropout = float(dropout)
        self.atom_projection = nn.Linear(atom_dim, hidden_dim)
        self.bond_projection = nn.Linear(bond_dim, hidden_dim)
        self.convolutions = nn.ModuleList(
            [NTNConv(hidden_dim, num_slices, dropout) for _ in range(num_layers)]
        )
        self.gate = nn.Linear(3 * hidden_dim, hidden_dim)
        self.feature_attention = FeatureAttention(hidden_dim, feature_reduction)
        self.fragment_to_molecule = GATConv(
            (hidden_dim, hidden_dim),
            hidden_dim,
            heads=4,
            concat=False,
            negative_slope=0.01,
            dropout=dropout,
            add_self_loops=False,
        )
        self.predictor = nn.Linear(2 * hidden_dim, num_targets)
        self.reset_parameters()

    def reset_parameters(self) -> None:
        self.atom_projection.reset_parameters()
        self.bond_projection.reset_parameters()
        for convolution in self.convolutions:
            assert isinstance(convolution, NTNConv)
            convolution.reset_parameters()
        self.gate.reset_parameters()
        self.feature_attention.reset_parameters()
        self.fragment_to_molecule.reset_parameters()
        self.predictor.reset_parameters()

    def forward(self, batch: Batch) -> Tensor:
        """Return raw regression predictions or classification logits."""

        (
            x,
            edge_index,
            edge_attr,
            brics_edge_index,
            brics_edge_attr,
            atom_to_fragment,
            graph_batch,
            fragment_batch,
            num_graphs,
            num_fragments,
        ) = self._batch_tensors(batch)

        molecular_atoms = self._encode(x, edge_index, edge_attr, graph_batch)
        fragment_atoms = self._encode(x, brics_edge_index, brics_edge_attr, atom_to_fragment)
        molecular_embedding = F.relu(
            scatter(
                molecular_atoms,
                graph_batch,
                dim=0,
                dim_size=num_graphs,
                reduce="sum",
            )
        )
        fragment_embedding = F.relu(
            scatter(
                fragment_atoms,
                atom_to_fragment,
                dim=0,
                dim_size=num_fragments,
                reduce="sum",
            )
        )

        fragment_edge_index = torch.stack(
            (
                torch.arange(num_fragments, device=x.device),
                fragment_batch,
            )
        )
        hierarchical_embedding = F.relu(
            self.fragment_to_molecule(
                (fragment_embedding, molecular_embedding),
                fragment_edge_index,
                size=(num_fragments, num_graphs),
            )
        )
        fused = torch.cat((molecular_embedding, hierarchical_embedding), dim=-1)
        return self.predictor(F.dropout(fused, p=self.dropout, training=self.training))

    def _encode(
        self,
        raw_x: Tensor,
        edge_index: Tensor,
        raw_edge_attr: Tensor,
        attention_group: Tensor,
    ) -> Tensor:
        x = F.relu(self.atom_projection(raw_x))
        edge_attr = F.relu(self.bond_projection(raw_edge_attr))
        for convolution in self.convolutions:
            assert isinstance(convolution, NTNConv)
            message = F.relu(convolution(x, edge_index, edge_attr))
            beta = torch.sigmoid(self.gate(torch.cat((x, message, x - message), dim=-1)))
            x = beta * x + (1 - beta) * message
            x = self.feature_attention(x, attention_group)
        return x

    def _batch_tensors(
        self, batch: Batch
    ) -> tuple[Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, int, int]:
        names = (
            "x",
            "edge_index",
            "edge_attr",
            "brics_edge_index",
            "brics_edge_attr",
            "atom_to_fragment",
            "batch",
        )
        values = tuple(getattr(batch, name, None) for name in names)
        if not all(isinstance(value, Tensor) for value in values):
            raise ValueError(f"batch must provide {', '.join(names)} tensors")
        x, edge_index, edge_attr, brics_edge_index, brics_edge_attr, fragments, graph_batch = values
        assert isinstance(x, Tensor)
        assert isinstance(edge_index, Tensor)
        assert isinstance(edge_attr, Tensor)
        assert isinstance(brics_edge_index, Tensor)
        assert isinstance(brics_edge_attr, Tensor)
        assert isinstance(fragments, Tensor)
        assert isinstance(graph_batch, Tensor)

        if x.ndim != 2 or x.shape[0] < 1 or x.shape[1] != self.atom_dim:
            raise ValueError(f"batch.x must have shape [N, {self.atom_dim}]")
        if x.dtype != torch.float32 or not torch.isfinite(x).all():
            raise ValueError("batch.x must contain finite torch.float32 values")
        self._validate_edges(edge_index, edge_attr, "edge", x.shape[0])
        self._validate_edges(brics_edge_index, brics_edge_attr, "brics_edge", x.shape[0])
        if graph_batch.shape != (x.shape[0],) or graph_batch.dtype != torch.long:
            raise ValueError("batch.batch must have shape [N] and dtype torch.long")
        if fragments.shape != (x.shape[0],) or fragments.dtype != torch.long:
            raise ValueError("batch.atom_to_fragment must have shape [N] and dtype torch.long")
        if any(value.device != x.device for value in values if isinstance(value, Tensor)):
            raise ValueError("all HiGNN batch tensors must be on the same device")
        if graph_batch.min() < 0 or fragments.min() < 0:
            raise ValueError("batch and atom_to_fragment indices must be non-negative")

        graph_ids = torch.unique(graph_batch, sorted=True)
        fragment_ids = torch.unique(fragments, sorted=True)
        if not torch.equal(graph_ids, torch.arange(graph_ids.numel(), device=x.device)):
            raise ValueError("batch.batch graph indices must be contiguous from zero")
        if not torch.equal(fragment_ids, torch.arange(fragment_ids.numel(), device=x.device)):
            raise ValueError("batch.atom_to_fragment indices must be contiguous from zero")
        for indices, name in ((edge_index, "edge_index"), (brics_edge_index, "brics_edge_index")):
            if indices.shape[1] and not torch.equal(
                graph_batch[indices[0]], graph_batch[indices[1]]
            ):
                raise ValueError(f"batch.{name} must not connect different graphs")
        if brics_edge_index.shape[1] and not torch.equal(
            fragments[brics_edge_index[0]], fragments[brics_edge_index[1]]
        ):
            raise ValueError("batch.brics_edge_index must stay within each fragment")

        num_fragments = fragment_ids.numel()
        fragment_min = scatter(graph_batch, fragments, dim=0, dim_size=num_fragments, reduce="min")
        fragment_max = scatter(graph_batch, fragments, dim=0, dim_size=num_fragments, reduce="max")
        if not torch.equal(fragment_min, fragment_max):
            raise ValueError("each fragment must belong to exactly one graph")
        return (
            x,
            edge_index,
            edge_attr,
            brics_edge_index,
            brics_edge_attr,
            fragments,
            graph_batch,
            fragment_min,
            graph_ids.numel(),
            num_fragments,
        )

    def _validate_edges(
        self, edge_index: Tensor, edge_attr: Tensor, field: str, num_nodes: int
    ) -> None:
        if edge_index.ndim != 2 or edge_index.shape[0] != 2 or edge_index.dtype != torch.long:
            raise ValueError(f"batch.{field}_index must have shape [2, E] and dtype torch.long")
        if edge_attr.shape != (edge_index.shape[1], self.bond_dim):
            raise ValueError(f"batch.{field}_attr must have shape [E, {self.bond_dim}]")
        if edge_attr.dtype != torch.float32 or not torch.isfinite(edge_attr).all():
            raise ValueError(f"batch.{field}_attr must contain finite torch.float32 values")
        if edge_index.shape[1] and (edge_index.min() < 0 or edge_index.max() >= num_nodes):
            raise ValueError(f"batch.{field}_index contains an invalid node index")


def _positive_int(value: int, field: str) -> None:
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise ValueError(f"{field} must be a positive integer")


__all__ = ["HiGNN"]
