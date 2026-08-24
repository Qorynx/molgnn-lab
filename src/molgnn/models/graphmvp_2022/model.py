"""GraphMVP downstream encoder and its auxiliary SchNet representation."""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path

import torch
from torch import Tensor, nn
from torch_geometric.data import Batch
from torch_geometric.nn import global_add_pool, global_max_pool, global_mean_pool
from torch_geometric.utils import scatter

from ..base import BaseMolecularModel
from ..contracts import validate_batched_molecular_graph
from .layers import (
    GaussianSmearing,
    GraphMVPInteractionBlock,
    OGBAtomEncoder,
    OGBGINConv,
    ShiftedSoftplus,
    SimpleGINConv,
)

_POOL_FUNCTIONS: dict[str, Callable[..., Tensor]] = {
    "mean": global_mean_pool,
    "add": global_add_pool,
    "max": global_max_pool,
}


class GraphMVPEncoder(nn.Module):
    """Checkpoint-shaped GIN encoder for one GraphMVP feature profile."""

    def __init__(
        self,
        *,
        feature_profile: str = "simple",
        num_layers: int = 5,
        hidden_dim: int = 300,
        jk: str = "last",
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        if feature_profile not in {"simple", "ogb_full"}:
            raise ValueError("feature_profile must be 'simple' or 'ogb_full'")
        if isinstance(num_layers, bool) or not isinstance(num_layers, int) or num_layers < 1:
            raise ValueError("num_layers must be a positive integer")
        if isinstance(hidden_dim, bool) or not isinstance(hidden_dim, int) or hidden_dim < 1:
            raise ValueError("hidden_dim must be a positive integer")
        if jk not in {"last", "concat", "sum", "max"}:
            raise ValueError("jk must be last, concat, sum, or max")
        if isinstance(dropout, bool) or not 0 <= float(dropout) < 1:
            raise ValueError("dropout must be in [0, 1)")

        self.feature_profile = feature_profile
        self.num_layers = num_layers
        self.hidden_dim = hidden_dim
        self.JK = jk
        self.drop_ratio = float(dropout)
        if feature_profile == "simple":
            # Names and dimensions intentionally match the official source.
            self.x_embedding1 = nn.Embedding(120, hidden_dim)
            self.x_embedding2 = nn.Embedding(4, hidden_dim)
            nn.init.xavier_uniform_(self.x_embedding1.weight)
            nn.init.xavier_uniform_(self.x_embedding2.weight)
            self.gnns = nn.ModuleList(SimpleGINConv(hidden_dim) for _ in range(num_layers))
        else:
            self.atom_encoder = OGBAtomEncoder(hidden_dim)
            self.gnns = nn.ModuleList(OGBGINConv(hidden_dim) for _ in range(num_layers))
        self.batch_norms = nn.ModuleList(
            nn.BatchNorm1d(hidden_dim) for _ in range(num_layers)
        )

    @property
    def output_dim(self) -> int:
        return self.hidden_dim * (self.num_layers + 1) if self.JK == "concat" else self.hidden_dim

    def forward(self, atom_attr: Tensor, edge_index: Tensor, bond_attr: Tensor) -> Tensor:
        if self.feature_profile == "simple":
            if atom_attr.ndim != 2 or atom_attr.shape[1] != 2 or atom_attr.dtype != torch.long:
                raise ValueError("simple GraphMVP atom attributes must have shape [N, 2] and dtype long")
            if bond_attr.shape != (edge_index.shape[1], 2) or bond_attr.dtype != torch.long:
                raise ValueError("simple GraphMVP bond attributes must have shape [E, 2] and dtype long")
            if atom_attr.numel() and (atom_attr[:, 0].min() < 0 or atom_attr[:, 0].max() >= 120 or atom_attr[:, 1].min() < 0 or atom_attr[:, 1].max() >= 4):
                raise ValueError("simple GraphMVP atom indices are outside their vocabulary")
            h = self.x_embedding1(atom_attr[:, 0]) + self.x_embedding2(atom_attr[:, 1])
        else:
            if atom_attr.ndim != 2 or atom_attr.shape[1] != 9 or atom_attr.dtype != torch.long:
                raise ValueError("ogb_full GraphMVP atom attributes must have shape [N, 9] and dtype long")
            if bond_attr.shape != (edge_index.shape[1], 3) or bond_attr.dtype != torch.long:
                raise ValueError("ogb_full GraphMVP bond attributes must have shape [E, 3] and dtype long")
            h = self.atom_encoder(atom_attr)

        if edge_index.ndim != 2 or edge_index.shape[0] != 2 or edge_index.dtype != torch.long:
            raise ValueError("edge_index must have shape [2, E] and dtype long")
        h_list = [h]
        for layer, convolution in enumerate(self.gnns):
            h = convolution(h_list[-1], edge_index, bond_attr)
            h = self.batch_norms[layer](h)
            if layer == self.num_layers - 1:
                h = nn.functional.dropout(h, self.drop_ratio, self.training)
            else:
                h = nn.functional.dropout(torch.relu(h), self.drop_ratio, self.training)
            h_list.append(h)
        if self.JK == "last":
            return h_list[-1]
        stacked = torch.stack(h_list, dim=0)
        if self.JK == "concat":
            return torch.cat(h_list, dim=1)
        if self.JK == "sum":
            return stacked.sum(dim=0)
        return stacked.max(dim=0).values


class GraphMVP(BaseMolecularModel):
    """Graph-level GraphMVP predictor for 2-D downstream fine-tuning."""

    required_batch_fields = (
        "edge_index",
        "graphmvp_simple_atom_attr",
        "graphmvp_simple_bond_attr",
        "graphmvp_ogb_atom_attr",
        "graphmvp_ogb_bond_attr",
        "batch",
    )

    def __init__(
        self,
        atom_dim: int,
        bond_dim: int,
        num_targets: int,
        *,
        feature_profile: str = "simple",
        hidden_dim: int = 300,
        num_layers: int = 5,
        jk: str = "last",
        dropout: float = 0.0,
        pooling: str = "mean",
        pretrained_checkpoint: str | None = None,
    ) -> None:
        super().__init__()
        del atom_dim, bond_dim
        if isinstance(num_targets, bool) or not isinstance(num_targets, int) or num_targets < 1:
            raise ValueError("num_targets must be a positive integer")
        if pooling not in _POOL_FUNCTIONS:
            raise ValueError("pooling must be mean, add, or max")
        self.num_targets = num_targets
        self.feature_profile = feature_profile
        self.encoder = GraphMVPEncoder(
            feature_profile=feature_profile,
            num_layers=num_layers,
            hidden_dim=hidden_dim,
            jk=jk,
            dropout=dropout,
        )
        self.pooling = pooling
        self.pool = _POOL_FUNCTIONS[pooling]
        self.graph_pred_linear = nn.Linear(self.encoder.output_dim, num_targets)
        self.initialization = "scratch"
        self.checkpoint_info: dict[str, object] | None = None
        if pretrained_checkpoint:
            from .checkpoint import load_graphmvp_encoder

            self.checkpoint_info = load_graphmvp_encoder(
                self.encoder, Path(pretrained_checkpoint)
            )
            self.initialization = "pretrained"

    def forward(self, batch: Batch) -> Tensor:
        values = tuple(getattr(batch, name, None) for name in self.required_batch_fields)
        if not all(isinstance(value, Tensor) for value in values):
            raise ValueError("batch is missing GraphMVP tensors")
        edge_index = values[0]
        simple_atom, simple_bond = values[1], values[2]
        ogb_atom, ogb_bond = values[3], values[4]
        graph_batch = values[5]
        assert isinstance(edge_index, Tensor)
        assert isinstance(simple_atom, Tensor) and isinstance(simple_bond, Tensor)
        assert isinstance(ogb_atom, Tensor) and isinstance(ogb_bond, Tensor)
        assert isinstance(graph_batch, Tensor)
        node_count = simple_atom.shape[0] if self.feature_profile == "simple" else ogb_atom.shape[0]
        if graph_batch.shape != (node_count,) or graph_batch.dtype != torch.long:
            raise ValueError("batch.batch must assign every GraphMVP atom")
        num_graphs = validate_batched_molecular_graph(
            edge_index, graph_batch, num_nodes=node_count, device=graph_batch.device
        )
        if self.feature_profile == "simple":
            node_repr = self.encoder(simple_atom, edge_index, simple_bond)
        else:
            node_repr = self.encoder(ogb_atom, edge_index, ogb_bond)
        pooled = self.pool(node_repr, graph_batch, size=num_graphs)
        return self.graph_pred_linear(pooled)


class GraphMVPSchNetEncoder(nn.Module):
    """The SchNet representation used only by GraphMVP pretraining."""

    def __init__(
        self,
        *,
        hidden_dim: int = 300,
        num_filters: int = 128,
        num_interactions: int = 6,
        num_gaussians: int = 51,
        cutoff: float = 10.0,
        readout: str = "mean",
    ) -> None:
        super().__init__()
        if readout not in {"mean", "add"}:
            raise ValueError("readout must be mean or add")
        if cutoff <= 0 or num_gaussians < 2:
            raise ValueError("cutoff must be positive and num_gaussians must be at least 2")
        self.hidden_dim = hidden_dim
        self.cutoff = float(cutoff)
        self.readout = readout
        self.embedding = nn.Embedding(119, hidden_dim)
        self.distance_expansion = GaussianSmearing(0.0, cutoff, num_gaussians)
        self.interactions = nn.ModuleList(
            GraphMVPInteractionBlock(hidden_dim, num_filters, num_gaussians, cutoff)
            for _ in range(num_interactions)
        )
        self.lin1 = nn.Linear(hidden_dim, hidden_dim)
        self.act = ShiftedSoftplus()
        self.lin2 = nn.Linear(hidden_dim, hidden_dim)

    def forward(
        self,
        atomic_number_index: Tensor,
        pos: Tensor,
        batch: Tensor | None = None,
        edge_index: Tensor | None = None,
    ) -> Tensor:
        if atomic_number_index.ndim != 1 or atomic_number_index.dtype != torch.long:
            raise ValueError("GraphMVP SchNet expects zero-based atomic indices [N]")
        if pos.shape != (atomic_number_index.shape[0], 3) or pos.dtype != torch.float32:
            raise ValueError("GraphMVP SchNet pos must have shape [N, 3] and float32 dtype")
        batch = torch.zeros_like(atomic_number_index) if batch is None else batch
        if batch.shape != atomic_number_index.shape or batch.dtype != torch.long:
            raise ValueError("GraphMVP SchNet batch must assign each atom")
        if atomic_number_index.numel() and (atomic_number_index.min() < 0 or atomic_number_index.max() >= 119):
            raise ValueError("GraphMVP SchNet atomic index is outside 0..118")
        if edge_index is None:
            edge_index = _radius_edges(pos, batch, self.cutoff)
        source, target = edge_index
        distance = torch.linalg.vector_norm(pos[source] - pos[target], dim=-1)
        rbf = self.distance_expansion(distance)
        h = self.embedding(atomic_number_index)
        for interaction in self.interactions:
            h = h + interaction(h, edge_index, distance, rbf)
        h = self.lin2(self.act(self.lin1(h)))
        return scatter(h, batch, dim=0, reduce=self.readout)


def _radius_edges(pos: Tensor, batch: Tensor, cutoff: float) -> Tensor:
    node_count = pos.shape[0]
    nodes = torch.arange(node_count, device=pos.device, dtype=torch.long)
    source = nodes.repeat_interleave(node_count)
    target = nodes.repeat(node_count)
    distance = torch.linalg.vector_norm(pos[source] - pos[target], dim=-1)
    keep = (source != target) & (batch[source] == batch[target]) & (distance <= cutoff)
    return torch.stack((source[keep], target[keep]), dim=0)


__all__ = ["GraphMVP", "GraphMVPEncoder", "GraphMVPSchNetEncoder"]
