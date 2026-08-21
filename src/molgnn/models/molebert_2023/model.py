"""Mole-BERT downstream encoder and graph-level predictor."""

from __future__ import annotations

from pathlib import Path

import torch
from torch import Tensor, nn
from torch_geometric.data import Batch
from torch_geometric.nn import global_mean_pool

from ..base import BaseMolecularModel
from ..contracts import validate_batched_molecular_graph
from .checkpoint import load_molebert_encoder
from .layers import (
    NUM_ATOM_TYPES,
    NUM_CHIRALITY_TAGS,
    MoleBERTGINConv,
)


class MoleBERTEncoder(nn.Module):
    """Checkpoint-compatible bond-aware GIN node encoder."""

    def __init__(
        self,
        num_layers: int = 5,
        hidden_dim: int = 300,
        jk: str = "last",
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        if num_layers < 1 or hidden_dim < 1:
            raise ValueError("num_layers and hidden_dim must be positive")
        if jk not in {"last", "concat", "sum", "max"}:
            raise ValueError("jk must be last, concat, sum, or max")
        if not 0 <= float(dropout) < 1:
            raise ValueError("dropout must be in [0, 1)")
        self.num_layers = num_layers
        self.hidden_dim = hidden_dim
        self.JK = jk
        self.drop_ratio = float(dropout)
        self.x_embedding1 = nn.Embedding(NUM_ATOM_TYPES, hidden_dim)
        self.x_embedding2 = nn.Embedding(NUM_CHIRALITY_TAGS, hidden_dim)
        nn.init.xavier_uniform_(self.x_embedding1.weight)
        nn.init.xavier_uniform_(self.x_embedding2.weight)
        self.gnns = nn.ModuleList(
            [MoleBERTGINConv(hidden_dim, hidden_dim) for _ in range(num_layers)]
        )
        self.batch_norms = nn.ModuleList(
            [nn.BatchNorm1d(hidden_dim) for _ in range(num_layers)]
        )

    def forward(self, atom_attr: Tensor, edge_index: Tensor, bond_attr: Tensor) -> Tensor:
        if atom_attr.ndim != 2 or atom_attr.shape[1] != 2 or atom_attr.dtype != torch.long:
            raise ValueError("Mole-BERT atom attributes must have shape [N, 2] and dtype long")
        if edge_index.ndim != 2 or edge_index.shape[0] != 2 or edge_index.dtype != torch.long:
            raise ValueError("edge_index must have shape [2, E] and dtype long")
        if bond_attr.shape != (edge_index.shape[1], 2) or bond_attr.dtype != torch.long:
            raise ValueError("Mole-BERT bond attributes must have shape [E, 2] and dtype long")
        if atom_attr.numel() and (atom_attr[:, 0].min() < 0 or atom_attr[:, 0].max() >= NUM_ATOM_TYPES):
            raise ValueError("Mole-BERT atom indices are outside the supported range")
        if bond_attr.numel() and (bond_attr[:, 0].min() < 0 or bond_attr[:, 0].max() >= 6 or bond_attr[:, 1].min() < 0 or bond_attr[:, 1].max() >= 3):
            raise ValueError("Mole-BERT bond indices are outside the supported range")
        h = self.x_embedding1(atom_attr[:, 0]) + self.x_embedding2(atom_attr[:, 1])
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
        if self.JK == "concat":
            return torch.cat(h_list, dim=1)
        stacked = torch.stack(h_list, dim=0)
        return stacked.sum(dim=0) if self.JK == "sum" else stacked.max(dim=0).values


class MoleBERT(BaseMolecularModel):
    """Mole-BERT graph predictor for masked multi-task regression/classification."""

    required_batch_fields = ("edge_index", "molebert_atom_attr", "molebert_bond_attr", "batch")

    def __init__(
        self,
        atom_dim: int,
        bond_dim: int,
        num_targets: int,
        hidden_dim: int = 300,
        num_layers: int = 5,
        jk: str = "last",
        dropout: float = 0.0,
        pooling: str = "mean",
        pretrained_checkpoint: str | None = None,
    ) -> None:
        super().__init__()
        del atom_dim, bond_dim
        if num_targets < 1:
            raise ValueError("num_targets must be positive")
        if pooling != "mean":
            raise ValueError("Mole-BERT currently supports mean pooling only")
        self.num_targets = num_targets
        self.encoder = MoleBERTEncoder(num_layers, hidden_dim, jk, dropout)
        encoder_dim = hidden_dim * (num_layers + 1) if jk == "concat" else hidden_dim
        self.graph_pred_linear = nn.Linear(encoder_dim, num_targets)
        self.initialization = "scratch"
        self.checkpoint_info: dict[str, object] | None = None
        if pretrained_checkpoint:
            self.checkpoint_info = load_molebert_encoder(self.encoder, Path(pretrained_checkpoint))
            self.initialization = "pretrained"

    def forward(self, batch: Batch) -> Tensor:
        atom_attr = getattr(batch, "molebert_atom_attr", None)
        bond_attr = getattr(batch, "molebert_bond_attr", None)
        edge_index = getattr(batch, "edge_index", None)
        graph_batch = getattr(batch, "batch", None)
        if not all(isinstance(value, Tensor) for value in (atom_attr, bond_attr, edge_index, graph_batch)):
            raise ValueError("batch is missing Mole-BERT tensors")
        assert isinstance(atom_attr, Tensor) and isinstance(bond_attr, Tensor)
        assert isinstance(edge_index, Tensor) and isinstance(graph_batch, Tensor)
        if graph_batch.shape != (atom_attr.shape[0],) or graph_batch.dtype != torch.long:
            raise ValueError("batch.batch must assign every atom")
        num_graphs = validate_batched_molecular_graph(
            edge_index, graph_batch, num_nodes=atom_attr.shape[0], device=atom_attr.device
        )
        node_representation = self.encoder(atom_attr, edge_index, bond_attr)
        pooled = global_mean_pool(node_representation, graph_batch, size=num_graphs)
        return self.graph_pred_linear(pooled)


__all__ = ["MoleBERT", "MoleBERTEncoder"]
