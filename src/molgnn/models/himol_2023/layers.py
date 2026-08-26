"""Checkpoint-compatible hierarchical GIN layers for HiMol."""

from __future__ import annotations

import torch
from torch import Tensor, nn
from torch.nn import functional as F

NUM_ATOM_TYPES = 121
NUM_DEGREES = 11
NUM_EDGE_TYPES = 7
NUM_EDGE_META = 3
SELF_LOOP_EDGE_TYPE = 4
JK_MODES = ("last", "concat", "sum", "max")


class HiMolGINConv(nn.Module):
    """Author-source GIN update modernized without ``torch_scatter``."""

    def __init__(self, emb_dim: int) -> None:
        super().__init__()
        if emb_dim < 1:
            raise ValueError("emb_dim must be positive")
        self.mlp = nn.Sequential(
            nn.Linear(emb_dim, 2 * emb_dim),
            nn.ReLU(),
            nn.Linear(2 * emb_dim, emb_dim),
        )
        self.edge_embedding1 = nn.Embedding(NUM_EDGE_TYPES, emb_dim)
        self.edge_embedding2 = nn.Embedding(NUM_EDGE_META, emb_dim)
        nn.init.xavier_uniform_(self.edge_embedding1.weight)
        nn.init.xavier_uniform_(self.edge_embedding2.weight)

    def forward(self, x: Tensor, edge_index: Tensor, edge_attr: Tensor) -> Tensor:
        if x.ndim != 2:
            raise ValueError("x must have shape [N, F]")
        if edge_index.ndim != 2 or edge_index.shape[0] != 2:
            raise ValueError("edge_index must have shape [2, E]")
        if edge_attr.shape != (edge_index.shape[1], 2):
            raise ValueError("edge_attr must have shape [E, 2]")
        num_nodes = x.shape[0]
        loops = torch.arange(num_nodes, dtype=torch.long, device=edge_index.device)
        loop_index = torch.stack((loops, loops), dim=0)
        loop_attr = torch.zeros(
            (num_nodes, 2), dtype=torch.long, device=edge_attr.device
        )
        loop_attr[:, 0] = SELF_LOOP_EDGE_TYPE
        complete_index = torch.cat((edge_index, loop_index), dim=1)
        complete_attr = torch.cat((edge_attr, loop_attr), dim=0)
        edge_embedding = self.edge_embedding1(complete_attr[:, 0])
        edge_embedding = edge_embedding + self.edge_embedding2(complete_attr[:, 1])
        source, target = complete_index
        messages = x[source] + edge_embedding
        aggregated = x.new_zeros(x.shape)
        aggregated.index_add_(0, target, messages)
        return self.mlp(aggregated)


class HiMolEncoder(nn.Module):
    """Atom/motif/graph-node GIN encoder used by pretraining and fine-tuning."""

    def __init__(
        self,
        num_layer: int = 5,
        emb_dim: int = 512,
        JK: str = "last",
        drop_ratio: float = 0.5,
    ) -> None:
        super().__init__()
        if num_layer < 2:
            raise ValueError("num_layer must be at least 2")
        if emb_dim < 1:
            raise ValueError("emb_dim must be positive")
        if JK not in JK_MODES:
            raise ValueError(f"JK must be one of {JK_MODES}")
        if not 0 <= float(drop_ratio) < 1:
            raise ValueError("drop_ratio must be in [0, 1)")
        self.num_layer = int(num_layer)
        self.emb_dim = int(emb_dim)
        self.JK = JK
        self.drop_ratio = float(drop_ratio)
        self.x_embedding1 = nn.Embedding(NUM_ATOM_TYPES, emb_dim)
        self.x_embedding2 = nn.Embedding(NUM_DEGREES, emb_dim)
        nn.init.xavier_uniform_(self.x_embedding1.weight)
        nn.init.xavier_uniform_(self.x_embedding2.weight)
        self.gnns = nn.ModuleList([HiMolGINConv(emb_dim) for _ in range(num_layer)])
        self.batch_norms = nn.ModuleList(
            [nn.BatchNorm1d(emb_dim) for _ in range(num_layer)]
        )

    @property
    def output_dim(self) -> int:
        return (
            self.emb_dim * (self.num_layer + 1) if self.JK == "concat" else self.emb_dim
        )

    def forward(
        self, node_attr: Tensor, edge_index: Tensor, edge_attr: Tensor
    ) -> Tensor:
        _validate_inputs(node_attr, edge_index, edge_attr)
        hidden = self.x_embedding1(node_attr[:, 0]) + self.x_embedding2(node_attr[:, 1])
        history = [hidden]
        for layer, convolution in enumerate(self.gnns):
            hidden = convolution(history[-1], edge_index, edge_attr)
            hidden = self.batch_norms[layer](hidden)
            if layer != self.num_layer - 1:
                hidden = F.relu(hidden)
            hidden = F.dropout(hidden, self.drop_ratio, training=self.training)
            history.append(hidden)
        if self.JK == "last":
            return history[-1]
        if self.JK == "concat":
            return torch.cat(history, dim=1)
        stacked = torch.stack(history, dim=0)
        if self.JK == "sum":
            return stacked.sum(dim=0)
        return stacked.max(dim=0).values


def _validate_inputs(node_attr: Tensor, edge_index: Tensor, edge_attr: Tensor) -> None:
    if node_attr.ndim != 2 or node_attr.shape[1] != 2 or node_attr.dtype != torch.long:
        raise ValueError("HiMol node_attr must have shape [N, 2] and dtype long")
    if node_attr.shape[0] < 1:
        raise ValueError("HiMol graph must contain at least one hierarchy node")
    if (
        edge_index.ndim != 2
        or edge_index.shape[0] != 2
        or edge_index.dtype != torch.long
    ):
        raise ValueError("HiMol edge_index must have shape [2, E] and dtype long")
    if edge_attr.shape != (edge_index.shape[1], 2) or edge_attr.dtype != torch.long:
        raise ValueError("HiMol edge_attr must have shape [E, 2] and dtype long")
    if bool((node_attr[:, 0] < 0).any()) or bool(
        (node_attr[:, 0] >= NUM_ATOM_TYPES).any()
    ):
        raise ValueError("HiMol atom/token ids are outside the supported range")
    if bool((node_attr[:, 1] < 0).any()) or bool(
        (node_attr[:, 1] >= NUM_DEGREES).any()
    ):
        raise ValueError("HiMol degree ids are outside the supported range")
    if edge_attr.numel():
        if bool((edge_attr[:, 0] < 0).any()) or bool(
            (edge_attr[:, 0] >= NUM_EDGE_TYPES).any()
        ):
            raise ValueError("HiMol edge type ids are outside the supported range")
        if bool((edge_attr[:, 1] < 0).any()) or bool(
            (edge_attr[:, 1] >= NUM_EDGE_META).any()
        ):
            raise ValueError("HiMol edge metadata ids are outside the supported range")
    if edge_index.numel() and (
        int(edge_index.min()) < 0 or int(edge_index.max()) >= node_attr.shape[0]
    ):
        raise ValueError("HiMol edge_index references a missing hierarchy node")


__all__ = [
    "JK_MODES",
    "NUM_ATOM_TYPES",
    "NUM_DEGREES",
    "NUM_EDGE_META",
    "NUM_EDGE_TYPES",
    "HiMolEncoder",
    "HiMolGINConv",
]
