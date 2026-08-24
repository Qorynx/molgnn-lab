"""Neural Fingerprint architecture (Duvenaud et al. NeurIPS 2015)."""

from __future__ import annotations

import math
from typing import Literal

import torch
from torch import Tensor, nn
from torch_geometric.data import Batch
from torch_geometric.nn import global_add_pool

from ..base import BaseMolecularModel
from ..contracts import validate_batched_molecular_graph
from .layers import DegreeSpecificGraphConv, LegacyFeatureNormalization


class NeuralFingerprint(BaseMolecularModel):
    """Convolutional Networks on Graphs for Learning Molecular Fingerprints.

    Features degree-specific neighbor projection matrices and radius-wise
    (depth + 1) softmax fingerprint contributions pooled across atoms.
    """

    required_batch_fields = (
        "neural_fp_x",
        "edge_index",
        "neural_fp_edge_attr",
        "batch",
    )

    def __init__(
        self,
        in_atom_dim: int = 62,
        in_bond_dim: int = 6,
        depth: int = 4,
        hidden_dim: int = 20,
        fingerprint_dim: int = 128,
        predictor_hidden_dim: int | None = 100,
        num_targets: int = 1,
        activation: Literal["relu", "tanh"] = "relu",
        normalization: Literal["legacy", "none"] = "legacy",
        init_scale: float | None = math.exp(-4),
    ) -> None:
        super().__init__()
        _positive_int(in_atom_dim, "in_atom_dim")
        _positive_int(in_bond_dim, "in_bond_dim")
        _positive_int(depth, "depth")
        _positive_int(hidden_dim, "hidden_dim")
        _positive_int(fingerprint_dim, "fingerprint_dim")
        _positive_int(num_targets, "num_targets")
        if predictor_hidden_dim is not None:
            _positive_int(predictor_hidden_dim, "predictor_hidden_dim")
        if activation not in {"relu", "tanh"}:
            raise ValueError(f"unsupported activation: {activation!r}")
        if normalization not in {"legacy", "none"}:
            raise ValueError(f"unsupported normalization: {normalization!r}")

        self.in_atom_dim = in_atom_dim
        self.in_bond_dim = in_bond_dim
        self.depth = depth
        self.hidden_dim = hidden_dim
        self.fingerprint_dim = fingerprint_dim
        self.predictor_hidden_dim = predictor_hidden_dim
        self.num_targets = num_targets
        self.activation_name = activation
        self.normalization_mode = normalization

        self.convs = nn.ModuleList(
            [
                DegreeSpecificGraphConv(
                    in_atom_dim=in_atom_dim if layer == 0 else hidden_dim,
                    hidden_dim=hidden_dim,
                    bond_dim=in_bond_dim,
                    activation=activation,
                    normalization=normalization,
                )
                for layer in range(depth)
            ]
        )

        self.out_linears = nn.ModuleList(
            [
                nn.Linear(
                    in_atom_dim if layer == 0 else hidden_dim,
                    fingerprint_dim,
                )
                for layer in range(depth + 1)
            ]
        )

        if predictor_hidden_dim is not None:
            self.predictor_fc1 = nn.Linear(fingerprint_dim, predictor_hidden_dim)
            if normalization == "legacy":
                self.predictor_norm = LegacyFeatureNormalization()
            else:
                self.predictor_norm = None
            if activation == "relu":
                self.predictor_act = nn.ReLU()
            else:
                self.predictor_act = nn.Tanh()
            self.predictor_fc2 = nn.Linear(predictor_hidden_dim, num_targets)
        else:
            self.predictor_fc = nn.Linear(fingerprint_dim, num_targets)

        if init_scale is not None and init_scale > 0:
            self._init_weights(init_scale)

    def _init_weights(self, std: float) -> None:
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.normal_(p, mean=0.0, std=std)
            else:
                nn.init.normal_(p, mean=0.0, std=std)

    def fingerprint(self, batch: Batch) -> Tensor:
        """Compute the pooled radius 0..depth learned fingerprint vector [B, fp_dim]."""
        x, edge_index, edge_attr, graph_batch, num_graphs = self._validate_and_get_batch(
            batch
        )
        all_fps: list[Tensor] = []

        # Radius 0 contribution
        atom_p0 = torch.softmax(self.out_linears[0](x), dim=-1)
        all_fps.append(global_add_pool(atom_p0, graph_batch, size=num_graphs))

        # Radius 1..depth contributions
        h = x
        num_nodes = x.shape[0]
        for layer, conv in enumerate(self.convs):
            h = conv(h, edge_index, edge_attr, num_nodes=num_nodes)
            atom_p = torch.softmax(self.out_linears[layer + 1](h), dim=-1)
            all_fps.append(global_add_pool(atom_p, graph_batch, size=num_graphs))

        return torch.stack(all_fps, dim=0).sum(dim=0)

    def forward(self, batch: Batch) -> Tensor:
        """Return raw prediction tensor with shape [batch_size, num_targets]."""
        fp = self.fingerprint(batch)
        if self.predictor_hidden_dim is not None:
            h = self.predictor_fc1(fp)
            if self.predictor_norm is not None:
                h = self.predictor_norm(h)
            h = self.predictor_act(h)
            return self.predictor_fc2(h)
        return self.predictor_fc(fp)

    def _validate_and_get_batch(
        self, batch: Batch
    ) -> tuple[Tensor, Tensor, Tensor, Tensor, int]:
        x = getattr(batch, "neural_fp_x", None)
        edge_index = getattr(batch, "edge_index", None)
        edge_attr = getattr(batch, "neural_fp_edge_attr", None)
        graph_batch = getattr(batch, "batch", None)

        if not isinstance(x, Tensor) or not isinstance(edge_index, Tensor):
            raise ValueError(
                "batch must provide neural_fp_x and edge_index tensors"
            )
        if not isinstance(edge_attr, Tensor):
            raise ValueError("batch must provide neural_fp_edge_attr tensor")
        if x.ndim != 2 or x.shape[0] < 1 or x.shape[1] != self.in_atom_dim:
            raise ValueError(
                f"batch.neural_fp_x must have shape [N, {self.in_atom_dim}] with N >= 1"
            )
        if not torch.is_floating_point(x):
            raise ValueError("batch.neural_fp_x must be a floating tensor")
        if (
            edge_index.ndim != 2
            or edge_index.shape[0] != 2
            or edge_index.dtype != torch.long
        ):
            raise ValueError(
                "batch.edge_index must have shape [2, E] and dtype torch.long"
            )
        if (
            edge_attr.ndim != 2
            or edge_attr.shape[0] != edge_index.shape[1]
            or edge_attr.shape[1] != self.in_bond_dim
        ):
            raise ValueError(
                f"batch.neural_fp_edge_attr must have shape [E, {self.in_bond_dim}]"
            )
        if graph_batch is None:
            graph_batch = torch.zeros(x.shape[0], dtype=torch.long, device=x.device)
        if not isinstance(graph_batch, Tensor):
            raise ValueError("batch.batch must be a torch.Tensor when provided")

        num_graphs = validate_batched_molecular_graph(
            edge_index,
            graph_batch,
            num_nodes=x.shape[0],
            device=x.device,
        )
        return x, edge_index, edge_attr, graph_batch, num_graphs


def _positive_int(value: int, field: str) -> None:
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise ValueError(f"{field} must be a positive integer")


__all__ = ["NeuralFingerprint"]
