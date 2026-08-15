"""Custom message-passing layers ported from MolCLR (Wang et al., 2022)."""

from __future__ import annotations

import math

import torch
from torch import Tensor, nn
from torch_geometric.nn import MessagePassing
from torch_geometric.utils import add_self_loops, degree


class GINEConv(MessagePassing):
    """GIN layer with Hu et al. edge-feature fusion, as used by MolCLR.

    MolCLR's GIN backbone computes ``h_v = MLP(SUM_{u in N(v) U {v}} (h_u + e_uv))``
    where the self-loop is injected through the message path (no separate
    combine step).  Upstream encodes the discrete bond type / direction with
    two small ``nn.Embedding`` tables; this port instead projects the lab's
    continuous 14-dim bond features with a single ``nn.Linear`` and fills the
    added self-loop rows with zeros, since a self-loop carries no bond
    information.
    """

    def __init__(self, emb_dim: int, edge_dim: int) -> None:
        super().__init__()
        self.emb_dim = emb_dim
        self.edge_dim = edge_dim
        # Edge-feature projection replacing the upstream bond-type and
        # bond-direction embedding tables.
        self.edge_embedding = nn.Linear(edge_dim, emb_dim)
        # MLP applied after aggregation; the wrapper supplies the surrounding
        # BatchNorm, so no BatchNorm lives inside the layer.
        self.mlp = nn.Sequential(
            nn.Linear(emb_dim, 2 * emb_dim),
            nn.ReLU(inplace=True),
            nn.Linear(2 * emb_dim, emb_dim),
        )

    def forward(self, x: Tensor, edge_index: Tensor, edge_attr: Tensor) -> Tensor:
        # Self-loops reuse the message path for the node's own features; their
        # edge_attr rows are zero-filled because there is no bond to describe.
        edge_index = add_self_loops(edge_index, num_nodes=x.size(0))[0]
        self_loop_attr = torch.zeros(
            x.size(0), self.edge_dim, dtype=edge_attr.dtype, device=edge_attr.device
        )
        edge_attr = torch.cat([edge_attr, self_loop_attr], dim=0)
        edge_embeddings = self.edge_embedding(edge_attr)
        return self.propagate(edge_index, x=x, edge_attr=edge_embeddings)

    def message(self, x_j: Tensor, edge_attr: Tensor) -> Tensor:
        # The Hu et al. modification: fuse the edge embedding additively into
        # the source node features before aggregation.
        return x_j + edge_attr

    def update(self, aggr_out: Tensor) -> Tensor:
        return self.mlp(aggr_out)


class GCNConv(MessagePassing):
    """MolCLR's GCN backbone layer with edge-feature fusion.

    Follows the upstream ``gcn_finetune.py`` GCNConv: a learned transform
    ``x @ W``, symmetric degree normalization (including the injected
    self-loops), an additive edge-feature fusion identical to the GIN layer,
    and a per-layer bias.  The upstream sparse-matrix fast path
    (``torch_sparse.matmul``) is omitted: PyG's default message + aggregate
    path applies the same normalized ``x_j + edge_attr`` messages with the
    ``sum`` aggregation, which is behaviourally equivalent without requiring
    the optional sparse library.
    """

    def __init__(self, emb_dim: int, edge_dim: int) -> None:
        super().__init__()
        self.emb_dim = emb_dim
        self.edge_dim = edge_dim
        self.weight = nn.Parameter(torch.empty(emb_dim, emb_dim))
        self.bias = nn.Parameter(torch.empty(emb_dim))
        # Edge-feature projection replacing the upstream bond-type and
        # bond-direction embedding tables.
        self.edge_embedding = nn.Linear(edge_dim, emb_dim)
        self.reset_parameters()

    def reset_parameters(self) -> None:
        # Mirrors the upstream ``gcn_finetune.GCNConv.reset_parameters``
        # (gcn_finetune.py:55-60): Glorot-style uniform on the learned
        # transform, zero-initialised bias.  Earlier revisions of this port
        # used uniform(-stdv, stdv) for the bias; zeros match the upstream
        # and avoid leaking activation asymmetry into layer 0.
        stdv = math.sqrt(6.0 / (self.emb_dim + self.emb_dim))
        self.weight.data.uniform_(-stdv, stdv)
        self.bias.data.fill_(0)

    def forward(self, x: Tensor, edge_index: Tensor, edge_attr: Tensor) -> Tensor:
        edge_index = add_self_loops(edge_index, num_nodes=x.size(0))[0]
        self_loop_attr = torch.zeros(
            x.size(0), self.edge_dim, dtype=edge_attr.dtype, device=edge_attr.device
        )
        edge_attr = torch.cat([edge_attr, self_loop_attr], dim=0)
        edge_embeddings = self.edge_embedding(edge_attr)
        edge_index, edge_weight = self.gcn_norm(edge_index, x.size(0))
        x = x @ self.weight
        out = self.propagate(
            edge_index, x=x, edge_attr=edge_embeddings, edge_weight=edge_weight
        )
        return out + self.bias

    def gcn_norm(self, edge_index: Tensor, num_nodes: int) -> tuple[Tensor, Tensor]:
        """Return ``(edge_index, norm)`` with symmetric degree normalization."""
        row, col = edge_index
        deg = degree(col, num_nodes=num_nodes)
        deg_inv_sqrt = deg.pow(-0.5)
        deg_inv_sqrt[deg_inv_sqrt == float("inf")] = 0
        edge_weight = deg_inv_sqrt[row] * deg_inv_sqrt[col]
        return edge_index, edge_weight

    def message(self, x_j: Tensor, edge_attr: Tensor, edge_weight: Tensor) -> Tensor:
        # Degree-normalized edge-feature fusion, matching the upstream message
        # ``norm * (x_j + edge_attr)``.
        return edge_weight.view(-1, 1) * (x_j + edge_attr)


__all__ = ["GINEConv", "GCNConv"]
