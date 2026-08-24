"""Modern PyG layers matching GraphMVP's two legacy GIN profiles."""

from __future__ import annotations

import math

import torch
from torch import Tensor, nn
from torch_geometric.nn import MessagePassing
from torch_geometric.utils import add_self_loops

NUM_SIMPLE_ATOMS = 120
NUM_SIMPLE_CHIRALITY = 4
NUM_SIMPLE_BONDS = 6
NUM_SIMPLE_DIRECTIONS = 3

# ``ogb.utils.features`` tables used by the legacy regression tree.  The
# final bucket in each categorical feature is the source's ``misc`` bucket.
OGB_ATOM_DIMS = (119, 4, 12, 12, 10, 6, 6, 2, 2)
OGB_BOND_DIMS = (5, 6, 2)


class SimpleGINConv(MessagePassing):
    """Bond-aware GINConv from the GraphMVP classification source."""

    def __init__(self, emb_dim: int) -> None:
        super().__init__(aggr="add")
        self.mlp = nn.Sequential(
            nn.Linear(emb_dim, 2 * emb_dim),
            nn.ReLU(),
            nn.Linear(2 * emb_dim, emb_dim),
        )
        self.edge_embedding1 = nn.Embedding(NUM_SIMPLE_BONDS, emb_dim)
        self.edge_embedding2 = nn.Embedding(NUM_SIMPLE_DIRECTIONS, emb_dim)
        nn.init.xavier_uniform_(self.edge_embedding1.weight)
        nn.init.xavier_uniform_(self.edge_embedding2.weight)

    def forward(self, x: Tensor, edge_index: Tensor, edge_attr: Tensor) -> Tensor:
        edge_index, _ = add_self_loops(edge_index, num_nodes=x.shape[0])
        self_loop_attr = torch.zeros(
            (x.shape[0], 2), dtype=torch.long, device=edge_attr.device
        )
        self_loop_attr[:, 0] = 4
        full_edge_attr = torch.cat((edge_attr.to(torch.long), self_loop_attr), dim=0)
        edge_embedding = self.edge_embedding1(full_edge_attr[:, 0]) + self.edge_embedding2(
            full_edge_attr[:, 1]
        )
        return self.propagate(edge_index, x=x, edge_attr=edge_embedding)

    def message(self, x_j: Tensor, edge_attr: Tensor) -> Tensor:
        return x_j + edge_attr

    def update(self, aggr_out: Tensor) -> Tensor:
        return self.mlp(aggr_out)


class CategoricalEncoder(nn.Module):
    """Sum one embedding table per OGB categorical feature column."""

    def __init__(self, dimensions: tuple[int, ...], emb_dim: int, *, name: str) -> None:
        super().__init__()
        self._embedding_name = f"{name}_embedding_list"
        setattr(self, self._embedding_name, nn.ModuleList(
            [nn.Embedding(size, emb_dim) for size in dimensions]
        ))
        for embedding in self.embedding_list:
            nn.init.xavier_uniform_(embedding.weight)

    @property
    def embedding_list(self) -> nn.ModuleList:
        return getattr(self, self._embedding_name)

    def forward(self, values: Tensor) -> Tensor:
        if values.ndim != 2 or values.shape[1] != len(self.embedding_list):
            raise ValueError(
                f"categorical features must have shape [N, {len(self.embedding_list)}]"
            )
        result = torch.zeros(
            (values.shape[0], self.embedding_list[0].embedding_dim),
            dtype=self.embedding_list[0].weight.dtype,
            device=values.device,
        )
        for column, embedding in enumerate(self.embedding_list):
            indices = values[:, column].to(torch.long)
            if indices.numel() and (
                int(indices.min()) < 0 or int(indices.max()) >= embedding.num_embeddings
            ):
                raise ValueError(f"categorical feature column {column} is outside its vocabulary")
            result = result + embedding(indices)
        return result


class OGBAtomEncoder(CategoricalEncoder):
    def __init__(self, emb_dim: int) -> None:
        super().__init__(OGB_ATOM_DIMS, emb_dim, name="atom")


class OGBBondEncoder(CategoricalEncoder):
    def __init__(self, emb_dim: int) -> None:
        super().__init__(OGB_BOND_DIMS, emb_dim, name="bond")


class OGBGINConv(MessagePassing):
    """GIN variant from ``models_complete_feature/molecule_gnn_model.py``."""

    def __init__(self, emb_dim: int) -> None:
        super().__init__(aggr="add")
        self.mlp = nn.Sequential(
            nn.Linear(emb_dim, 2 * emb_dim),
            nn.BatchNorm1d(2 * emb_dim),
            nn.ReLU(),
            nn.Linear(2 * emb_dim, emb_dim),
        )
        self.eps = nn.Parameter(torch.zeros(1))
        self.bond_encoder = OGBBondEncoder(emb_dim)

    def forward(self, x: Tensor, edge_index: Tensor, edge_attr: Tensor) -> Tensor:
        edge_embedding = self.bond_encoder(edge_attr)
        propagated = self.propagate(edge_index, x=x, edge_attr=edge_embedding)
        return self.mlp((1.0 + self.eps) * x + propagated)

    def message(self, x_j: Tensor, edge_attr: Tensor) -> Tensor:
        return torch.relu(x_j + edge_attr)


class ShiftedSoftplus(nn.Module):
    def forward(self, value: Tensor) -> Tensor:
        return torch.nn.functional.softplus(value) - math.log(2.0)


class GaussianSmearing(nn.Module):
    """Gaussian distance expansion used by the legacy GraphMVP SchNet."""

    def __init__(self, start: float, stop: float, num_gaussians: int) -> None:
        super().__init__()
        offset = torch.linspace(start, stop, num_gaussians)
        self.coeff = -0.5 / float(offset[1] - offset[0]) ** 2
        self.register_buffer("offset", offset)

    def forward(self, distance: Tensor) -> Tensor:
        return torch.exp(self.coeff * (distance.view(-1, 1) - self.offset.view(1, -1)).square())


class GraphMVPContinuousFilter(MessagePassing):
    def __init__(self, hidden_dim: int, num_filters: int, num_gaussians: int, cutoff: float) -> None:
        super().__init__(aggr="add")
        self.lin1 = nn.Linear(hidden_dim, num_filters, bias=False)
        self.lin2 = nn.Linear(num_filters, hidden_dim)
        self.filter_network = nn.Sequential(
            nn.Linear(num_gaussians, num_filters),
            ShiftedSoftplus(),
            nn.Linear(num_filters, num_filters),
        )
        self.cutoff = float(cutoff)

    def forward(self, x: Tensor, edge_index: Tensor, distance: Tensor, rbf: Tensor) -> Tensor:
        envelope = 0.5 * (torch.cos(distance * math.pi / self.cutoff) + 1.0)
        filters = self.filter_network(rbf) * envelope.view(-1, 1)
        return self.lin2(self.propagate(edge_index, x=self.lin1(x), W=filters))

    def message(self, x_j: Tensor, W: Tensor) -> Tensor:
        return x_j * W


class GraphMVPInteractionBlock(nn.Module):
    def __init__(self, hidden_dim: int, num_filters: int, num_gaussians: int, cutoff: float) -> None:
        super().__init__()
        self.conv = GraphMVPContinuousFilter(hidden_dim, num_filters, num_gaussians, cutoff)
        self.act = ShiftedSoftplus()
        self.lin = nn.Linear(hidden_dim, hidden_dim)

    def forward(self, x: Tensor, edge_index: Tensor, distance: Tensor, rbf: Tensor) -> Tensor:
        x = self.conv(x, edge_index, distance, rbf)
        return self.lin(self.act(x))


__all__ = [
    "OGB_ATOM_DIMS",
    "OGB_BOND_DIMS",
    "GaussianSmearing",
    "GraphMVPInteractionBlock",
    "OGBAtomEncoder",
    "OGBBondEncoder",
    "OGBGINConv",
    "SimpleGINConv",
]
