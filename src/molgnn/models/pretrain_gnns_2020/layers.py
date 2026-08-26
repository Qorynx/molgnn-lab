"""Source-compatible molecular GIN for Pretrain-GNNs (Hu et al., ICLR 2020).

Provenance: ``OFFICIAL CODE`` ``chem/model.py`` at the core chemistry
revision ``6e69a06692b6870fbb381f336131314a42a0e983``. Module and parameter
names mirror the official classes so the released 57-tensor checkpoints load
without key remapping:

- ``x_embedding1`` over 120 atomic-number slots (mask token ``119``);
- ``x_embedding2`` over chirality tags (runtime 4 rows, official checkpoint
  carries 3 and is adapted by the loader);
- per layer a ``GINConv`` (self-loop bond type ``4``, bond embedding added to
  the source message, sum aggregation, MLP ``H -> 2H -> H``) followed by
  ``BatchNorm1d``;
- ReLU + dropout on every layer except the last, which keeps dropout only so
  ContextPred dot products stay sign-free;
- JK aggregation ``last/concat/sum/max``.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import Tensor, nn
from torch_geometric.nn import MessagePassing
from torch_geometric.utils import add_self_loops

NUM_ATOM_TYPE = 120  # including the extra mask token at index 119
RUNTIME_CHIRALITY_TAG = 4  # paper categories; official checkpoints carry 3
OFFICIAL_CHIRALITY_TAG = 3
NUM_BOND_TYPE = 6  # including self-loop type 4 and masked type 5
NUM_REAL_BOND_TYPES = 4  # single/double/triple/aromatic are the output classes
NUM_BOND_DIRECTION = 3
SELF_LOOP_BOND_TYPE = 4
MASK_ATOM_TOKEN = 119
MASK_BOND_TYPE = 5  # corruption-only token; never an output class

JK_MODES = ("last", "concat", "sum", "max")


def jk_output_dim(num_layer: int, emb_dim: int, JK: str) -> int:
    """Return the encoder's node-representation width for a JK mode.

    ``concat`` stacks the input embedding and every layer output, so its width
    is ``(num_layer + 1) * emb_dim``; all other modes return ``emb_dim``. The
    same contract must be applied to every downstream/pretraining head.
    """

    if JK not in JK_MODES:
        raise ValueError(f"invalid JK mode {JK!r}; expected one of {JK_MODES}")
    if emb_dim < 1:
        raise ValueError("emb_dim must be positive")
    return (num_layer + 1) * emb_dim if JK == "concat" else emb_dim


class GINConv(MessagePassing):
    """Official edge-aware GIN convolution with hard-coded self-loops."""

    def __init__(self, emb_dim: int, aggr: str = "add") -> None:
        super().__init__(aggr=aggr)
        self.mlp = nn.Sequential(
            nn.Linear(emb_dim, 2 * emb_dim),
            nn.ReLU(),
            nn.Linear(2 * emb_dim, emb_dim),
        )
        self.edge_embedding1 = nn.Embedding(NUM_BOND_TYPE, emb_dim)
        self.edge_embedding2 = nn.Embedding(NUM_BOND_DIRECTION, emb_dim)
        nn.init.xavier_uniform_(self.edge_embedding1.weight.data)
        nn.init.xavier_uniform_(self.edge_embedding2.weight.data)

    def forward(self, x: Tensor, edge_index: Tensor, edge_attr: Tensor) -> Tensor:
        looped_edge_index, _ = add_self_loops(edge_index, num_nodes=x.size(0))
        self_loop_attr = torch.zeros(x.size(0), 2, device=edge_attr.device).to(edge_attr.dtype)
        self_loop_attr[:, 0] = SELF_LOOP_BOND_TYPE
        full_attr = torch.cat((edge_attr, self_loop_attr), dim=0)
        edge_embeddings = self.edge_embedding1(full_attr[:, 0])
        edge_embeddings = edge_embeddings + self.edge_embedding2(full_attr[:, 1])
        return self.propagate(looped_edge_index, x=x, edge_attr=edge_embeddings)

    def message(self, x_j: Tensor, edge_attr: Tensor) -> Tensor:
        # OFFICIAL CODE adds the bond embedding to the source-node message.
        return x_j + edge_attr

    def update(self, aggr_out: Tensor) -> Tensor:
        return self.mlp(aggr_out)


class MolecularGNN(nn.Module):
    """Official ``GNN`` encoder returning node representations."""

    def __init__(
        self,
        num_layer: int,
        emb_dim: int,
        JK: str = "last",
        drop_ratio: float = 0,
        chirality_vocab: int = RUNTIME_CHIRALITY_TAG,
    ) -> None:
        super().__init__()
        self.num_layer = num_layer
        self.drop_ratio = drop_ratio
        self.JK = JK
        if self.num_layer < 2:
            raise ValueError("Number of GNN layers must be greater than 1.")
        if JK not in JK_MODES:
            raise ValueError(f"invalid JK mode {JK!r}; expected one of {JK_MODES}")

        self.x_embedding1 = nn.Embedding(NUM_ATOM_TYPE, emb_dim)
        self.x_embedding2 = nn.Embedding(chirality_vocab, emb_dim)
        nn.init.xavier_uniform_(self.x_embedding1.weight.data)
        nn.init.xavier_uniform_(self.x_embedding2.weight.data)

        self.gnns = nn.ModuleList([GINConv(emb_dim) for _ in range(num_layer)])
        self.batch_norms = nn.ModuleList(
            [nn.BatchNorm1d(emb_dim) for _ in range(num_layer)]
        )

    def forward(self, atom_attr: Tensor, edge_index: Tensor, bond_attr: Tensor) -> Tensor:
        x = self.x_embedding1(atom_attr[:, 0]) + self.x_embedding2(atom_attr[:, 1])

        h_list = [x]
        for layer in range(self.num_layer):
            h = self.gnns[layer](h_list[layer], edge_index, bond_attr)
            norm = self.batch_norms[layer]
            if self.training and h.shape[0] == 1:
                # BatchNorm cannot estimate variance from one sample.  Use
                # its running statistics for singleton graphs while keeping
                # the source-compatible training path for normal batches.
                h = F.batch_norm(
                    h,
                    norm.running_mean,
                    norm.running_var,
                    norm.weight,
                    norm.bias,
                    training=False,
                    momentum=0.0,
                    eps=norm.eps,
                )
            else:
                h = norm(h)
            if layer == self.num_layer - 1:
                # Remove relu for the last layer; dropout stays.
                h = F.dropout(h, self.drop_ratio, training=self.training)
            else:
                h = F.dropout(F.relu(h), self.drop_ratio, training=self.training)
            h_list.append(h)

        if self.JK == "concat":
            node_representation = torch.cat(h_list, dim=1)
        elif self.JK == "last":
            node_representation = h_list[-1]
        elif self.JK == "max":
            stacked = torch.stack(h_list, dim=0)
            node_representation = torch.max(stacked, dim=0)[0]
        elif self.JK == "sum":
            stacked = torch.stack(h_list, dim=0)
            node_representation = torch.sum(stacked, dim=0)
        else:
            raise ValueError(f"invalid JK mode {self.JK!r}")
        return node_representation


__all__ = [
    "JK_MODES",
    "MASK_ATOM_TOKEN",
    "MASK_BOND_TYPE",
    "NUM_ATOM_TYPE",
    "NUM_BOND_DIRECTION",
    "NUM_BOND_TYPE",
    "NUM_REAL_BOND_TYPES",
    "OFFICIAL_CHIRALITY_TAG",
    "RUNTIME_CHIRALITY_TAG",
    "SELF_LOOP_BOND_TYPE",
    "GINConv",
    "MolecularGNN",
    "jk_output_dim",
]
