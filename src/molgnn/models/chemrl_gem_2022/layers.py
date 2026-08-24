"""PyTorch layers corresponding to PaddleHelix's GeoGNN blocks."""

from __future__ import annotations

import torch
from torch import Tensor, nn
from torch_geometric.utils import scatter

from .constants import RBF_GAMMA


class RBF(nn.Module):
    """Fixed Gaussian RBF without persistent buffers for checkpoint parity."""

    def __init__(self, centers: Tensor, gamma: float = RBF_GAMMA) -> None:
        super().__init__()
        self.centers = centers.detach().clone()
        self.gamma = float(gamma)

    def forward(self, values: Tensor) -> Tensor:
        if values.ndim != 1:
            raise ValueError("RBF input must have shape [N]")
        centers = self.centers.to(device=values.device, dtype=values.dtype)
        return torch.exp(-self.gamma * (values.unsqueeze(-1) - centers).square())


class AtomEmbedding(nn.Module):
    """Sum one legacy categorical embedding per atom feature column."""

    def __init__(self, sizes: tuple[int, ...], embed_dim: int) -> None:
        super().__init__()
        self.embed_list = nn.ModuleList([nn.Embedding(size, embed_dim) for size in sizes])
        for embedding in self.embed_list:
            nn.init.xavier_uniform_(embedding.weight)

    def forward(self, values: Tensor) -> Tensor:
        if values.ndim != 2 or values.shape[1] != len(self.embed_list):
            raise ValueError(f"categorical features must have shape [N, {len(self.embed_list)}]")
        result = self.embed_list[0](values[:, 0])
        for column, embedding in enumerate(self.embed_list[1:], start=1):
            result = result + embedding(values[:, column])
        return result


class BondEmbedding(AtomEmbedding):
    """Same sum encoder as the source, named separately for state-key parity."""


class BondFloatRBF(nn.Module):
    """RBF plus one source-compatible linear projection per float feature."""

    def __init__(self, centers: Tensor, embed_dim: int) -> None:
        super().__init__()
        self.rbf_list = nn.ModuleList([RBF(centers)])
        self.linear_list = nn.ModuleList([nn.Linear(int(centers.numel()), embed_dim)])

    def forward(self, values: Tensor) -> Tensor:
        return self.linear_list[0](self.rbf_list[0](values))


class BondAngleFloatRBF(BondFloatRBF):
    """RBF encoder for the line-graph angle feature."""


class GIN(nn.Module):
    """Edge-featured GIN: sum of ``source node + edge`` at each target."""

    def __init__(self, hidden_size: int) -> None:
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(hidden_size, hidden_size * 2),
            nn.ReLU(),
            nn.Linear(hidden_size * 2, hidden_size),
        )

    def forward(self, edge_index: Tensor, node_feat: Tensor, edge_feat: Tensor) -> Tensor:
        if edge_index.ndim != 2 or edge_index.shape[0] != 2:
            raise ValueError("edge_index must have shape [2, E]")
        if edge_feat.shape != (edge_index.shape[1], node_feat.shape[1]):
            raise ValueError("edge features must align with edge_index and node width")
        if edge_index.shape[1] == 0:
            aggregate = torch.zeros_like(node_feat)
        else:
            source, target = edge_index
            aggregate = scatter(
                node_feat[source] + edge_feat,
                target,
                dim=0,
                dim_size=node_feat.shape[0],
                reduce="sum",
            )
        return self.mlp(aggregate)


class GraphNorm(nn.Module):
    """Legacy GEM normalization: divide by sqrt(nodes per graph)."""

    def forward(self, features: Tensor, graph_batch: Tensor) -> Tensor:
        counts = scatter(
            torch.ones((features.shape[0], 1), dtype=features.dtype, device=features.device),
            graph_batch,
            dim=0,
            reduce="sum",
        )
        return features / counts.clamp_min(1.0).sqrt()[graph_batch]


class GeoGNNBlock(nn.Module):
    """GIN + LayerNorm + source GraphNorm + residual connection."""

    def __init__(self, embed_dim: int, dropout_rate: float, last_act: bool) -> None:
        super().__init__()
        self.embed_dim = embed_dim
        self.last_act = bool(last_act)
        self.gnn = GIN(embed_dim)
        self.norm = nn.LayerNorm(embed_dim)
        self.graph_norm = GraphNorm()
        self.act = nn.ReLU()
        self.dropout = nn.Dropout(float(dropout_rate))

    def forward(
        self,
        edge_index: Tensor,
        node_hidden: Tensor,
        edge_hidden: Tensor,
        graph_batch: Tensor,
    ) -> Tensor:
        out = self.gnn(edge_index, node_hidden, edge_hidden)
        out = self.norm(out)
        out = self.graph_norm(out, graph_batch)
        if self.last_act:
            out = self.act(out)
        return self.dropout(out) + node_hidden


class PredictionMLP(nn.Module):
    """Two-layer source MLP used by geometry pretraining heads."""

    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int, dropout: float) -> None:
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.Dropout(dropout),
            nn.LeakyReLU(),
            nn.Linear(hidden_dim, output_dim),
        )

    def forward(self, values: Tensor) -> Tensor:
        return self.mlp(values)


class DownstreamMLP(nn.Module):
    """Source ``down_mlp2/3`` head used after graph LayerNorm."""

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        output_dim: int,
        layer_num: int,
        dropout: float,
    ) -> None:
        super().__init__()
        if layer_num not in {2, 3}:
            raise ValueError("ChemRL-GEM downstream layer_num must be 2 or 3")
        layers: list[nn.Module] = []
        for layer_id in range(layer_num):
            if layer_id == 0:
                layers.append(nn.Linear(input_dim, hidden_dim))
            elif layer_id < layer_num - 1:
                layers.append(nn.Linear(hidden_dim, hidden_dim))
            else:
                layers.append(nn.Linear(hidden_dim, output_dim))
            if layer_id < layer_num - 1:
                layers.append(nn.Dropout(dropout))
                layers.append(nn.LeakyReLU())
        self.mlp = nn.Sequential(*layers)

    def forward(self, values: Tensor) -> Tensor:
        return self.mlp(values)


__all__ = [
    "AtomEmbedding",
    "BondAngleFloatRBF",
    "BondEmbedding",
    "BondFloatRBF",
    "DownstreamMLP",
    "GIN",
    "GeoGNNBlock",
    "GraphNorm",
    "PredictionMLP",
    "RBF",
]
