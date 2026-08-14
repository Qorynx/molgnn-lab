"""A sparse, independently usable implementation of the 2016 Weave model."""

from __future__ import annotations

from collections.abc import Sequence

import torch
from torch import Tensor, nn
from torch.nn import functional as F
from torch_geometric.data import Batch

from ..base import BaseMolecularModel
from ..contracts import validate_batched_molecular_graph
from .layers import GaussianHistogramReadout, SafeBatchNorm1d, WeaveModule


class Weave(BaseMolecularModel):
    """Two-state molecular graph convolution with Gaussian-histogram pooling.

    ``weave_pair_index`` represents ordered atom pairs.  Its first endpoint
    receives the ``P -> A`` sum, while the ``A -> P`` branch evaluates both
    endpoint orders and sums them.  This model deliberately does not inspect
    the canonical covalent ``edge_index`` at forward time; the registered
    Weave transform is responsible for preparing its pair graph.
    """

    required_batch_fields = (
        "x",
        "weave_pair_index",
        "weave_pair_attr",
        "batch",
    )

    def __init__(
        self,
        atom_dim: int,
        pair_dim: int = 22,
        hidden_dim: int = 50,
        num_weave_modules: int = 2,
        graph_feature_dim: int = 128,
        predictor_hidden_dims: Sequence[int] = (2000, 100),
        gaussian_bins: int = 11,
        dropout: float = 0.25,
        batch_norm: bool = True,
        num_targets: int = 1,
    ) -> None:
        super().__init__()
        for value, name in (
            (atom_dim, "atom_dim"),
            (pair_dim, "pair_dim"),
            (hidden_dim, "hidden_dim"),
            (num_weave_modules, "num_weave_modules"),
            (graph_feature_dim, "graph_feature_dim"),
            (gaussian_bins, "gaussian_bins"),
            (num_targets, "num_targets"),
        ):
            _positive_int(value, name)
        if (
            isinstance(dropout, bool)
            or not isinstance(dropout, (float, int))
            or not 0 <= dropout < 1
        ):
            raise ValueError("dropout must be in [0, 1)")
        if not isinstance(batch_norm, bool):
            raise ValueError("batch_norm must be a boolean")
        predictor_widths = _positive_widths(
            predictor_hidden_dims, "predictor_hidden_dims"
        )

        self.atom_dim = atom_dim
        self.pair_dim = pair_dim
        self.hidden_dim = hidden_dim
        self.num_weave_modules = num_weave_modules
        self.graph_feature_dim = graph_feature_dim
        self.gaussian_bins = gaussian_bins
        self.predictor_hidden_dims = predictor_widths
        self.dropout = float(dropout)
        self.batch_norm_enabled = batch_norm
        self.num_targets = num_targets

        modules: list[WeaveModule] = [
            WeaveModule(
                atom_in_dim=atom_dim,
                pair_in_dim=pair_dim,
                hidden_dim=hidden_dim,
                atom_out_dim=hidden_dim,
                pair_out_dim=hidden_dim,
                batch_norm=batch_norm,
            )
        ]
        modules.extend(
            WeaveModule(
                atom_in_dim=hidden_dim,
                pair_in_dim=hidden_dim,
                hidden_dim=hidden_dim,
                atom_out_dim=hidden_dim,
                pair_out_dim=hidden_dim,
                batch_norm=batch_norm,
            )
            for _ in range(num_weave_modules - 1)
        )
        self.weave_modules = nn.ModuleList(modules)

        # The provided DeepChem implementation applies tanh before the final
        # batch normalization; the paper's histogram then receives a
        # standardized atom representation without a ReLU bottleneck.
        self.final_atom_projection = nn.Linear(hidden_dim, graph_feature_dim)
        self.final_atom_norm: nn.Module
        self.final_atom_norm = (
            SafeBatchNorm1d(graph_feature_dim) if batch_norm else nn.Identity()
        )
        self.histogram_readout = GaussianHistogramReadout(
            graph_feature_dim, gaussian_bins
        )

        predictor_dims = (
            self.histogram_readout.output_dim,
            *predictor_widths,
            num_targets,
        )
        self.predictor_layers = nn.ModuleList(
            [
                nn.Linear(input_dim, output_dim)
                for input_dim, output_dim in zip(
                    predictor_dims[:-1], predictor_dims[1:], strict=True
                )
            ]
        )

    def forward(self, batch: Batch) -> Tensor:
        """Return raw regression values or binary-classification logits."""

        graph_features = self.fingerprint(batch)
        hidden = graph_features
        for layer in self.predictor_layers[:-1]:
            hidden = F.dropout(
                F.relu(layer(hidden)), p=self.dropout, training=self.training
            )
        return self.predictor_layers[-1](hidden)

    def fingerprint(self, batch: Batch) -> Tensor:
        """Return the Gaussian-histogram graph representation before the MLP."""

        atom_states, pair_states, pair_index, graph_batch, num_graphs = (
            self._batch_tensors(batch)
        )
        for module in self.weave_modules:
            atom_states, pair_states = module(atom_states, pair_states, pair_index)

        atom_states = torch.tanh(self.final_atom_projection(atom_states))
        atom_states = self.final_atom_norm(atom_states)
        return self.histogram_readout(atom_states, graph_batch, num_graphs)

    def _batch_tensors(
        self, batch: Batch
    ) -> tuple[Tensor, Tensor, Tensor, Tensor, int]:
        """Fetch and validate the architecture-specific sparse pair contract."""

        names = self.required_batch_fields
        values = tuple(getattr(batch, name, None) for name in names)
        if not all(isinstance(value, Tensor) for value in values):
            raise ValueError(f"batch must provide {', '.join(names)} tensors")
        x, pair_index, pair_attr, graph_batch = values
        assert isinstance(x, Tensor)
        assert isinstance(pair_index, Tensor)
        assert isinstance(pair_attr, Tensor)
        assert isinstance(graph_batch, Tensor)

        if x.ndim != 2 or x.shape[0] < 1 or x.shape[1] != self.atom_dim:
            raise ValueError(f"batch.x must have shape [N, {self.atom_dim}] with N >= 1")
        if x.dtype != torch.float32 or not torch.isfinite(x).all():
            raise ValueError("batch.x must contain finite torch.float32 values")
        if (
            pair_index.ndim != 2
            or pair_index.shape[0] != 2
            or pair_index.dtype != torch.long
        ):
            raise ValueError(
                "batch.weave_pair_index must have shape [2, Q] and dtype torch.long"
            )
        pair_count = pair_index.shape[1]
        if pair_attr.shape != (pair_count, self.pair_dim):
            raise ValueError(
                f"batch.weave_pair_attr must have shape [Q, {self.pair_dim}]"
            )
        if pair_attr.dtype != torch.float32 or not torch.isfinite(pair_attr).all():
            raise ValueError(
                "batch.weave_pair_attr must contain finite torch.float32 values"
            )
        if graph_batch.shape != (x.shape[0],) or graph_batch.dtype != torch.long:
            raise ValueError("batch.batch must have shape [N] and dtype torch.long")
        if (
            pair_index.device != x.device
            or pair_attr.device != x.device
            or graph_batch.device != x.device
        ):
            raise ValueError(
                "batch.x, weave pair tensors, and graph assignments must share a device"
            )
        if pair_count and (
            pair_index.min() < 0 or pair_index.max() >= x.shape[0]
        ):
            raise ValueError("batch.weave_pair_index contains an invalid node index")

        num_graphs = validate_batched_molecular_graph(
            pair_index,
            graph_batch,
            num_nodes=x.shape[0],
            device=x.device,
            edge_field="weave_pair_index",
            forbid_self_loops=False,
        )
        return x, pair_attr, pair_index, graph_batch, num_graphs


def _positive_int(value: int, field: str) -> None:
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise ValueError(f"{field} must be a positive integer")


def _positive_widths(values: Sequence[int], field: str) -> tuple[int, ...]:
    if isinstance(values, (str, bytes)):
        raise ValueError(f"{field} must be a non-empty sequence of positive integers")
    try:
        widths = tuple(values)
    except TypeError as exc:
        raise ValueError(
            f"{field} must be a non-empty sequence of positive integers"
        ) from exc
    if not widths:
        raise ValueError(f"{field} must be a non-empty sequence of positive integers")
    for width in widths:
        _positive_int(width, field)
    return widths


WeaveModel = Weave


__all__ = ["Weave", "WeaveModel"]
