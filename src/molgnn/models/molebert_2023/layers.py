"""Bond-aware GIN layers used by Mole-BERT."""

from __future__ import annotations

import torch
from torch import Tensor, nn
from torch_geometric.nn import MessagePassing
from torch_geometric.utils import add_self_loops

NUM_ATOM_TYPES = 120
NUM_CHIRALITY_TAGS = 4
NUM_BOND_TYPES = 6
NUM_BOND_DIRECTIONS = 3
MASK_ATOM = 119
MASK_BOND = 5
SELF_LOOP_BOND = 4


class MoleBERTGINConv(MessagePassing):
    """GIN with additive bond type/direction embeddings and self-loops."""

    def __init__(self, emb_dim: int, out_dim: int | None = None) -> None:
        super().__init__(aggr="add")
        out_dim = emb_dim if out_dim is None else out_dim
        self.mlp = nn.Sequential(
            nn.Linear(emb_dim, 2 * emb_dim),
            nn.ReLU(),
            nn.Linear(2 * emb_dim, out_dim),
        )
        self.edge_embedding1 = nn.Embedding(NUM_BOND_TYPES, emb_dim)
        self.edge_embedding2 = nn.Embedding(NUM_BOND_DIRECTIONS, emb_dim)
        nn.init.xavier_uniform_(self.edge_embedding1.weight)
        nn.init.xavier_uniform_(self.edge_embedding2.weight)

    def forward(self, x: Tensor, edge_index: Tensor, edge_attr: Tensor) -> Tensor:
        edge_index, _ = add_self_loops(edge_index, num_nodes=x.shape[0])
        self_loop_attr = torch.zeros(
            x.shape[0], 2, dtype=edge_attr.dtype, device=edge_attr.device
        )
        self_loop_attr[:, 0] = SELF_LOOP_BOND
        full_edge_attr = torch.cat((edge_attr, self_loop_attr), dim=0).to(torch.long)
        edge_embedding = self.edge_embedding1(full_edge_attr[:, 0]) + self.edge_embedding2(
            full_edge_attr[:, 1]
        )
        return self.propagate(edge_index, x=x, edge_attr=edge_embedding)

    def message(self, x_j: Tensor, edge_attr: Tensor) -> Tensor:
        return x_j + edge_attr

    def update(self, aggr_out: Tensor) -> Tensor:
        return self.mlp(aggr_out)


__all__ = [
    "MASK_ATOM",
    "MASK_BOND",
    "NUM_ATOM_TYPES",
    "NUM_BOND_DIRECTIONS",
    "NUM_BOND_TYPES",
    "NUM_CHIRALITY_TAGS",
    "MoleBERTGINConv",
]
