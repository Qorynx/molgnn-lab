"""Coley et al. 2017 graph convolution and learned fingerprint model."""

from __future__ import annotations

import torch
from torch import Tensor, nn
from torch.nn import functional as F
from torch_geometric.data import Batch
from torch_geometric.utils import scatter

from ..base import BaseMolecularModel
from .layers import ColeyGraphConv


class MolecularGraphEmbedding(BaseMolecularModel):
    """Radius-wise learned molecular fingerprint followed by a tanh predictor."""

    required_batch_fields = ("mge_x", "edge_index", "mge_edge_attr", "batch")

    def __init__(
        self,
        depth: int = 5,
        message_dim: int = 32,
        fingerprint_dim: int = 512,
        predictor_hidden_dim: int = 50,
        dropout: float = 0.0,
        num_targets: int = 1,
        input_atom_dim: int = 32,
        input_bond_dim: int = 8,
    ) -> None:
        super().__init__()
        for value, name in (
            (depth, "depth"),
            (message_dim, "message_dim"),
            (fingerprint_dim, "fingerprint_dim"),
            (predictor_hidden_dim, "predictor_hidden_dim"),
            (num_targets, "num_targets"),
            (input_atom_dim, "input_atom_dim"),
            (input_bond_dim, "input_bond_dim"),
        ):
            _positive_int(value, name)
        if (
            isinstance(dropout, bool)
            or not isinstance(dropout, (float, int))
            or not 0 <= dropout < 1
        ):
            raise ValueError("dropout must be in [0, 1)")

        self.depth = depth
        self.dropout = float(dropout)
        self.fingerprint_dim = fingerprint_dim
        self.input_atom_dim = input_atom_dim
        self.input_bond_dim = input_bond_dim
        state_dims = [input_atom_dim] + [message_dim] * depth
        self.convolutions = nn.ModuleList(
            [
                ColeyGraphConv(state_dims[layer], message_dim, input_bond_dim)
                for layer in range(depth)
            ]
        )
        self.fingerprint_projections = nn.ModuleList(
            [nn.Linear(state_dim, fingerprint_dim) for state_dim in state_dims]
        )
        self.hidden = nn.Linear(fingerprint_dim, predictor_hidden_dim)
        self.predictor = nn.Linear(predictor_hidden_dim, num_targets)
        self.reset_parameters()

    def reset_parameters(self) -> None:
        for convolution in self.convolutions:
            assert isinstance(convolution, ColeyGraphConv)
            convolution.reset_parameters()
        for projection in self.fingerprint_projections:
            assert isinstance(projection, nn.Linear)
            _reset_linear(projection)
        _reset_linear(self.hidden)
        _reset_linear(self.predictor)

    def forward(self, batch: Batch) -> Tensor:
        """Return raw regression predictions or classification logits."""

        fingerprint = self.fingerprint(batch)
        hidden = torch.tanh(self.hidden(fingerprint))
        hidden = F.dropout(hidden, p=self.dropout, training=self.training)
        return self.predictor(hidden)

    def fingerprint(self, batch: Batch) -> Tensor:
        """Return the atom- and depth-summed learned molecular fingerprint."""

        x, edge_index, edge_attr, graph_batch, num_graphs = self._batch_tensors(batch)
        fingerprint = x.new_zeros((num_graphs, self.fingerprint_dim))
        for depth, projection in enumerate(self.fingerprint_projections):
            assert isinstance(projection, nn.Linear)
            atom_fingerprints = torch.softmax(projection(x), dim=-1)
            fingerprint = fingerprint + scatter(
                atom_fingerprints,
                graph_batch,
                dim=0,
                dim_size=num_graphs,
                reduce="sum",
            )
            if depth < self.depth:
                convolution = self.convolutions[depth]
                assert isinstance(convolution, ColeyGraphConv)
                x = convolution(x, edge_index, edge_attr)
        return fingerprint

    def _batch_tensors(self, batch: Batch) -> tuple[Tensor, Tensor, Tensor, Tensor, int]:
        names = ("mge_x", "edge_index", "mge_edge_attr", "batch")
        values = tuple(getattr(batch, name, None) for name in names)
        if not all(isinstance(value, Tensor) for value in values):
            raise ValueError(f"batch must provide {', '.join(names)} tensors")
        x, edge_index, edge_attr, graph_batch = values
        assert isinstance(x, Tensor)
        assert isinstance(edge_index, Tensor)
        assert isinstance(edge_attr, Tensor)
        assert isinstance(graph_batch, Tensor)

        edge_count = edge_index.shape[1] if edge_index.ndim == 2 else -1
        if x.ndim != 2 or x.shape[0] < 1 or x.shape[1] != self.input_atom_dim:
            raise ValueError(f"batch.mge_x must have shape [N, {self.input_atom_dim}]")
        if x.dtype != torch.float32 or not torch.isfinite(x).all():
            raise ValueError("batch.mge_x must contain finite torch.float32 values")
        if edge_index.ndim != 2 or edge_index.shape[0] != 2 or edge_index.dtype != torch.long:
            raise ValueError("batch.edge_index must have shape [2, E] and dtype torch.long")
        if edge_attr.shape != (edge_count, self.input_bond_dim):
            raise ValueError(f"batch.mge_edge_attr must have shape [E, {self.input_bond_dim}]")
        if edge_attr.dtype != torch.float32 or not torch.isfinite(edge_attr).all():
            raise ValueError("batch.mge_edge_attr must contain finite torch.float32 values")
        if graph_batch.shape != (x.shape[0],) or graph_batch.dtype != torch.long:
            raise ValueError("batch.batch must have shape [N] and dtype torch.long")
        if graph_batch.numel() == 0 or graph_batch.min() < 0:
            raise ValueError("batch.batch must contain non-negative graph indices")
        if edge_count and (edge_index.min() < 0 or edge_index.max() >= x.shape[0]):
            raise ValueError("batch.edge_index contains an invalid node index")
        if edge_count and not torch.equal(graph_batch[edge_index[0]], graph_batch[edge_index[1]]):
            raise ValueError("batch.edge_index must not connect different graphs")

        graph_ids = torch.unique(graph_batch, sorted=True)
        expected_ids = torch.arange(graph_ids.numel(), device=graph_batch.device)
        if not torch.equal(graph_ids, expected_ids):
            raise ValueError("batch.batch graph indices must be contiguous from zero")
        return x, edge_index, edge_attr, graph_batch, graph_ids.numel()


def _reset_linear(layer: nn.Linear) -> None:
    nn.init.xavier_uniform_(layer.weight)
    if layer.bias is not None:
        nn.init.zeros_(layer.bias)


def _positive_int(value: int, field: str) -> None:
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise ValueError(f"{field} must be a positive integer")


__all__ = ["MolecularGraphEmbedding"]
