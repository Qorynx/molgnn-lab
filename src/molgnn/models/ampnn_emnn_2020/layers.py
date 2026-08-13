"""Sparse source-faithful building blocks for AMPNN and EMNN.

The original implementation represents a molecular batch as padded dense
tensors.  These layers retain its feed-forward, vector-attention, and gated
readout semantics while operating directly on PyG's sparse graph contract.
"""

from __future__ import annotations

import math
from collections.abc import Sequence

import torch
from torch import Tensor, nn
from torch_geometric.utils import scatter, softmax


class SELUFeedForward(nn.Module):
    """The source FFNN: AlphaDropout, bias-free Linear layers, and SELU.

    ``hidden_dims`` gives the hidden layers explicitly.  This avoids the
    source implementation's ambiguous ``depth`` convention while preserving
    its exact layer order: dropout before every linear layer, SELU only after
    hidden layers, and normal weight initialization with
    ``std=sqrt(1 / fan_in)``.
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dims: Sequence[int],
        output_dim: int,
        *,
        dropout: float = 0.0,
        bias: bool = False,
    ) -> None:
        super().__init__()
        _positive_int(input_dim, "input_dim")
        _positive_int(output_dim, "output_dim")
        if not isinstance(bias, bool):
            raise ValueError("bias must be a boolean")
        _validate_dropout(dropout)
        normalized_hidden_dims = _hidden_dims(hidden_dims)

        layer_dims = (input_dim, *normalized_hidden_dims, output_dim)
        layers: list[nn.Module] = []
        for layer_index, (in_features, out_features) in enumerate(
            zip(layer_dims, layer_dims[1:])
        ):
            layers.append(nn.AlphaDropout(float(dropout)))
            linear = nn.Linear(in_features, out_features, bias=bias)
            nn.init.normal_(linear.weight, std=math.sqrt(1.0 / in_features))
            layers.append(linear)
            if layer_index < len(layer_dims) - 2:
                layers.append(nn.SELU())

        self.input_dim = input_dim
        self.output_dim = output_dim
        self.hidden_dims = normalized_hidden_dims
        self.dropout = float(dropout)
        self.network = nn.Sequential(*layers)

    def forward(self, values: Tensor) -> Tensor:
        """Apply the source FFNN to a floating ``[..., input_dim]`` tensor."""

        if not isinstance(values, Tensor):
            raise ValueError("values must be a tensor")
        if values.ndim < 1 or values.shape[-1] != self.input_dim:
            raise ValueError(
                f"values must have final dimension {self.input_dim}, got {tuple(values.shape)}"
            )
        if not torch.is_floating_point(values):
            raise ValueError("values must be a floating tensor")
        return self.network(values)


def coordinatewise_segment_softmax(
    energies: Tensor,
    index: Tensor,
    num_segments: int,
) -> Tensor:
    """Normalize each attention coordinate independently within a segment.

    For ``energies`` of shape ``[E, D]``, each output coordinate ``d`` is a
    softmax over only the edges with the same ``index`` value.  This is the
    vector-valued attention used by AMPNN and EMNN, not scalar GAT attention.
    """

    _validate_segment_inputs(energies, index, num_segments, "energies")
    return softmax(energies, index, num_nodes=num_segments)


def vector_attention_aggregate(
    embeddings: Tensor,
    energies: Tensor,
    index: Tensor,
    num_segments: int,
) -> Tensor:
    """Return the sparse coordinate-wise attention-weighted segment sums."""

    _validate_segment_inputs(embeddings, index, num_segments, "embeddings")
    _validate_segment_inputs(energies, index, num_segments, "energies")
    if embeddings.shape != energies.shape:
        raise ValueError("embeddings and energies must have the same shape")
    if embeddings.device != energies.device:
        raise ValueError("embeddings and energies must be on the same device")

    weights = coordinatewise_segment_softmax(energies, index, num_segments)
    return scatter(
        weights * embeddings,
        index,
        dim=0,
        dim_size=num_segments,
        reduce="sum",
    )


class VectorAttentionAggregation(nn.Module):
    """Parameter-free sparse aggregation for AMPNN/EMNN vector attention."""

    def forward(
        self,
        embeddings: Tensor,
        energies: Tensor,
        index: Tensor,
        num_segments: int,
    ) -> Tensor:
        """Aggregate edge or edge-memory embeddings into sparse segments."""

        return vector_attention_aggregate(embeddings, energies, index, num_segments)


class GatedGraphGather(nn.Module):
    """Source GraphGather semantics over sparse PyG node batches.

    ``hidden_nodes`` and ``input_nodes`` may use different feature widths.
    AMPNN supplies final hidden states together with raw atom features, while
    EMNN may deliberately supply its derived node representation to both
    inputs to retain the source implementation's readout behavior.
    """

    def __init__(
        self,
        hidden_node_dim: int,
        input_node_dim: int,
        output_dim: int,
        *,
        gate_hidden_dims: Sequence[int],
        value_hidden_dims: Sequence[int],
        gate_dropout: float = 0.0,
        value_dropout: float = 0.0,
    ) -> None:
        super().__init__()
        _positive_int(hidden_node_dim, "hidden_node_dim")
        _positive_int(input_node_dim, "input_node_dim")
        _positive_int(output_dim, "output_dim")
        self.hidden_node_dim = hidden_node_dim
        self.input_node_dim = input_node_dim
        self.output_dim = output_dim
        self.gate_network = SELUFeedForward(
            hidden_node_dim + input_node_dim,
            gate_hidden_dims,
            output_dim,
            dropout=gate_dropout,
            bias=False,
        )
        self.value_network = SELUFeedForward(
            hidden_node_dim,
            value_hidden_dims,
            output_dim,
            dropout=value_dropout,
            bias=False,
        )

    def forward(
        self,
        hidden_nodes: Tensor,
        input_nodes: Tensor,
        graph_batch: Tensor,
        num_graphs: int,
    ) -> Tensor:
        """Compute ``sum_v sigmoid(q([h_v, x_v])) * p(h_v)`` per graph."""

        _validate_gather_inputs(
            hidden_nodes,
            input_nodes,
            graph_batch,
            num_graphs,
            hidden_node_dim=self.hidden_node_dim,
            input_node_dim=self.input_node_dim,
        )
        gate_input = torch.cat((hidden_nodes, input_nodes), dim=-1)
        gate = torch.sigmoid(self.gate_network(gate_input))
        values = self.value_network(hidden_nodes)
        return scatter(
            gate * values,
            graph_batch,
            dim=0,
            dim_size=num_graphs,
            reduce="sum",
        )


def _hidden_dims(hidden_dims: Sequence[int]) -> tuple[int, ...]:
    if isinstance(hidden_dims, (str, bytes)) or not isinstance(hidden_dims, Sequence):
        raise ValueError("hidden_dims must be a sequence of positive integers")
    normalized = tuple(hidden_dims)
    for index, hidden_dim in enumerate(normalized):
        _positive_int(hidden_dim, f"hidden_dims[{index}]")
    return normalized


def _validate_dropout(dropout: float) -> None:
    if (
        isinstance(dropout, bool)
        or not isinstance(dropout, (float, int))
        or not math.isfinite(float(dropout))
        or not 0 <= dropout < 1
    ):
        raise ValueError("dropout must be a finite value in [0, 1)")


def _validate_segment_inputs(
    values: Tensor,
    index: Tensor,
    num_segments: int,
    values_name: str,
) -> None:
    if not isinstance(values, Tensor):
        raise ValueError(f"{values_name} must be a tensor")
    if values.ndim != 2:
        raise ValueError(f"{values_name} must have shape [E, D]")
    if not torch.is_floating_point(values):
        raise ValueError(f"{values_name} must be a floating tensor")
    if not isinstance(index, Tensor) or index.ndim != 1 or index.dtype != torch.long:
        raise ValueError("index must have shape [E] and dtype torch.long")
    if index.shape[0] != values.shape[0]:
        raise ValueError("index must have one entry per value row")
    if index.device != values.device:
        raise ValueError("index and values must be on the same device")
    _nonnegative_int(num_segments, "num_segments")
    if index.numel() and (index.min().item() < 0 or index.max().item() >= num_segments):
        raise ValueError("index contains a segment outside [0, num_segments)")


def _validate_gather_inputs(
    hidden_nodes: Tensor,
    input_nodes: Tensor,
    graph_batch: Tensor,
    num_graphs: int,
    *,
    hidden_node_dim: int,
    input_node_dim: int,
) -> None:
    if not isinstance(hidden_nodes, Tensor) or (
        hidden_nodes.ndim != 2 or hidden_nodes.shape[1] != hidden_node_dim
    ):
        raise ValueError(f"hidden_nodes must have shape [N, {hidden_node_dim}]")
    if not torch.is_floating_point(hidden_nodes):
        raise ValueError("hidden_nodes must be a floating tensor")
    if not isinstance(input_nodes, Tensor) or (
        input_nodes.ndim != 2 or input_nodes.shape[1] != input_node_dim
    ):
        raise ValueError(f"input_nodes must have shape [N, {input_node_dim}]")
    if not torch.is_floating_point(input_nodes):
        raise ValueError("input_nodes must be a floating tensor")
    if input_nodes.shape[0] != hidden_nodes.shape[0]:
        raise ValueError("hidden_nodes and input_nodes must have the same row count")
    if input_nodes.device != hidden_nodes.device:
        raise ValueError("hidden_nodes and input_nodes must be on the same device")
    if (
        not isinstance(graph_batch, Tensor)
        or graph_batch.ndim != 1
        or graph_batch.dtype != torch.long
        or graph_batch.shape[0] != hidden_nodes.shape[0]
    ):
        raise ValueError("graph_batch must have shape [N] and dtype torch.long")
    if graph_batch.device != hidden_nodes.device:
        raise ValueError("graph_batch and hidden_nodes must be on the same device")
    _nonnegative_int(num_graphs, "num_graphs")
    if graph_batch.numel() and (
        graph_batch.min().item() < 0 or graph_batch.max().item() >= num_graphs
    ):
        raise ValueError("graph_batch contains a graph outside [0, num_graphs)")


def _nonnegative_int(value: int, field: str) -> None:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError(f"{field} must be a non-negative integer")


def _positive_int(value: int, field: str) -> None:
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise ValueError(f"{field} must be a positive integer")


__all__ = [
    "GatedGraphGather",
    "SELUFeedForward",
    "VectorAttentionAggregation",
    "coordinatewise_segment_softmax",
    "vector_attention_aggregate",
]
