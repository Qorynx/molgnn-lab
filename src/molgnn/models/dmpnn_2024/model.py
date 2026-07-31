"""Chemprop 2024 directed bond message passing over sparse PyG batches."""

from __future__ import annotations

import math
from itertools import pairwise

import torch
from torch import Tensor, nn
from torch.nn import functional as F
from torch_geometric.data import Batch
from torch_geometric.nn import global_add_pool, global_mean_pool
from torch_geometric.utils import scatter

from ..base import BaseMolecularModel
from ..contracts import validate_batched_molecular_graph


class DMPNN(BaseMolecularModel):
    """Bond-centric D-MPNN with reverse-edge exclusion and a raw-output FFN.

    ``mean`` is the aggregation default stated in the 2024 Chemprop paper.
    Chemprop 2.2.3's CLI instead defaults to ``norm`` with
    ``aggregation_norm=100``; that executable preset remains available here
    without changing the directed-bond encoder.
    """

    required_batch_fields = (
        "x",
        "edge_index",
        "edge_attr",
        "reverse_edge_index",
        "batch",
    )

    def __init__(
        self,
        atom_dim: int,
        bond_dim: int,
        hidden_dim: int = 300,
        depth: int = 3,
        dropout: float = 0.0,
        bias: bool = False,
        aggregation: str = "mean",
        aggregation_norm: float = 100.0,
        ffn_hidden_dim: int = 300,
        ffn_num_hidden_layers: int = 1,
        batch_norm: bool = False,
        num_targets: int = 1,
    ) -> None:
        super().__init__()
        for value, name in (
            (atom_dim, "atom_dim"),
            (bond_dim, "bond_dim"),
            (hidden_dim, "hidden_dim"),
            (depth, "depth"),
            (ffn_hidden_dim, "ffn_hidden_dim"),
            (ffn_num_hidden_layers, "ffn_num_hidden_layers"),
            (num_targets, "num_targets"),
        ):
            _positive_int(value, name)
        if (
            isinstance(dropout, bool)
            or not isinstance(dropout, (float, int))
            or not 0 <= dropout < 1
        ):
            raise ValueError("dropout must be in [0, 1)")
        if not isinstance(bias, bool):
            raise ValueError("bias must be a boolean")
        if not isinstance(batch_norm, bool):
            raise ValueError("batch_norm must be a boolean")
        if aggregation not in {"mean", "sum", "norm"}:
            raise ValueError("aggregation must be one of: mean, sum, norm")
        if (
            isinstance(aggregation_norm, bool)
            or not isinstance(aggregation_norm, (float, int))
            or not math.isfinite(float(aggregation_norm))
            or aggregation_norm <= 0
        ):
            raise ValueError("aggregation_norm must be a positive finite number")

        self.atom_dim = atom_dim
        self.bond_dim = bond_dim
        self.hidden_dim = hidden_dim
        self.depth = depth
        self.dropout = float(dropout)
        self.aggregation = aggregation
        self.aggregation_norm = float(aggregation_norm)

        self.W_i = nn.Linear(atom_dim + bond_dim, hidden_dim, bias=bias)
        self.W_h = nn.Linear(hidden_dim, hidden_dim, bias=bias)
        self.W_o = nn.Linear(atom_dim + hidden_dim, hidden_dim)
        self.batch_norm = nn.BatchNorm1d(hidden_dim) if batch_norm else nn.Identity()

        dimensions = (
            [hidden_dim] + [ffn_hidden_dim] * ffn_num_hidden_layers + [num_targets]
        )
        self.ffn_layers = nn.ModuleList(
            [
                nn.Linear(input_dim, output_dim)
                for input_dim, output_dim in pairwise(dimensions)
            ]
        )

    def forward(self, batch: Batch) -> Tensor:
        """Return raw regression predictions or classification logits."""

        hidden = self.fingerprint(batch)
        for layer in self.ffn_layers[:-1]:
            hidden = F.dropout(
                F.relu(layer(hidden)), p=self.dropout, training=self.training
            )
        return self.ffn_layers[-1](hidden)

    def fingerprint(self, batch: Batch) -> Tensor:
        """Return graph embeddings before the feed-forward predictor."""

        x, edge_index, edge_attr, reverse_edge_index, graph_batch = self._batch_tensors(
            batch
        )
        source, target = edge_index
        edge_initial = self.W_i(torch.cat((x[source], edge_attr), dim=-1))
        edge_hidden = F.relu(edge_initial)

        for _ in range(1, self.depth):
            incoming = scatter(
                edge_hidden,
                target,
                dim=0,
                dim_size=x.shape[0],
                reduce="sum",
            )
            message = incoming[source] - edge_hidden[reverse_edge_index]
            edge_hidden = F.dropout(
                F.relu(edge_initial + self.W_h(message)),
                p=self.dropout,
                training=self.training,
            )

        incoming = scatter(
            edge_hidden,
            target,
            dim=0,
            dim_size=x.shape[0],
            reduce="sum",
        )
        atom_hidden = F.dropout(
            F.relu(self.W_o(torch.cat((x, incoming), dim=-1))),
            p=self.dropout,
            training=self.training,
        )
        if self.aggregation == "mean":
            graph_hidden = global_mean_pool(atom_hidden, graph_batch)
        else:
            graph_hidden = global_add_pool(atom_hidden, graph_batch)
            if self.aggregation == "norm":
                graph_hidden = graph_hidden / self.aggregation_norm
        return self._apply_batch_norm(graph_hidden)

    def _apply_batch_norm(self, hidden: Tensor) -> Tensor:
        if not isinstance(self.batch_norm, nn.BatchNorm1d):
            return hidden
        if self.training and hidden.shape[0] == 1:
            return F.batch_norm(
                hidden,
                self.batch_norm.running_mean,
                self.batch_norm.running_var,
                self.batch_norm.weight,
                self.batch_norm.bias,
                training=False,
                eps=self.batch_norm.eps,
            )
        return self.batch_norm(hidden)

    def _batch_tensors(
        self, batch: Batch
    ) -> tuple[Tensor, Tensor, Tensor, Tensor, Tensor]:
        names = ("x", "edge_index", "edge_attr", "reverse_edge_index", "batch")
        values = tuple(getattr(batch, name, None) for name in names)
        if not all(isinstance(value, Tensor) for value in values):
            raise ValueError(f"batch must provide {', '.join(names)} tensors")
        x, edge_index, edge_attr, reverse_edge_index, graph_batch = values
        assert isinstance(x, Tensor)
        assert isinstance(edge_index, Tensor)
        assert isinstance(edge_attr, Tensor)
        assert isinstance(reverse_edge_index, Tensor)
        assert isinstance(graph_batch, Tensor)

        edge_count = edge_index.shape[1] if edge_index.ndim == 2 else -1
        if x.ndim != 2 or x.shape[0] < 1 or x.shape[1] != self.atom_dim:
            raise ValueError(
                f"batch.x must have shape [N, {self.atom_dim}] with N >= 1"
            )
        if x.dtype != torch.float32:
            raise ValueError("batch.x must have dtype torch.float32")
        if (
            edge_index.ndim != 2
            or edge_index.shape[0] != 2
            or edge_index.dtype != torch.long
        ):
            raise ValueError(
                "batch.edge_index must have shape [2, E] and dtype torch.long"
            )
        if edge_attr.ndim != 2 or edge_attr.shape != (edge_count, self.bond_dim):
            raise ValueError(f"batch.edge_attr must have shape [E, {self.bond_dim}]")
        if edge_attr.dtype != torch.float32:
            raise ValueError("batch.edge_attr must have dtype torch.float32")
        if (
            reverse_edge_index.shape != (edge_count,)
            or reverse_edge_index.dtype != torch.long
        ):
            raise ValueError(
                "batch.reverse_edge_index must have shape [E] and dtype torch.long"
            )
        if graph_batch.ndim != 1 or graph_batch.shape[0] != x.shape[0]:
            raise ValueError("batch.batch must have shape [N]")
        if (
            graph_batch.dtype != torch.long
            or graph_batch.numel() == 0
            or graph_batch.min() < 0
        ):
            raise ValueError(
                "batch.batch must contain non-negative torch.long graph indices"
            )
        if edge_count and (edge_index.min() < 0 or edge_index.max() >= x.shape[0]):
            raise ValueError("batch.edge_index contains an invalid node index")
        validate_batched_molecular_graph(
            edge_index,
            graph_batch,
            num_nodes=x.shape[0],
            device=x.device,
            forbid_self_loops=True,
        )
        if edge_count and (
            reverse_edge_index.min() < 0 or reverse_edge_index.max() >= edge_count
        ):
            raise ValueError("batch.reverse_edge_index contains an invalid edge index")
        if not torch.equal(
            reverse_edge_index[reverse_edge_index],
            torch.arange(edge_count, device=reverse_edge_index.device),
        ):
            raise ValueError("batch.reverse_edge_index must be an involution")
        if edge_count and not torch.equal(
            edge_index[:, reverse_edge_index], edge_index.flip(0)
        ):
            raise ValueError(
                "batch.reverse_edge_index must map each edge to its reverse"
            )
        return x, edge_index, edge_attr, reverse_edge_index, graph_batch


def _positive_int(value: int, field: str) -> None:
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise ValueError(f"{field} must be a positive integer")


__all__ = ["DMPNN"]
