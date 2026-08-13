"""Sparse PyTorch building blocks for the GPS++ hybrid architecture.

The original GPS++ implementation operates on fixed-size packed graphs.  This
module retains its architectural order while adapting the operations to the
project's sparse PyG batching boundary:

* the local MPNN updates an edge state, then independent receiver and sender
  node channels, then the graph-level state;
* the global branch performs *per graph* biased self-attention, so nodes from
  separate PyG graphs can never attend to one another; and
* a hybrid block evaluates local and global branches from the same pre-block
  node state, normalizes their residual states separately, adds them, then
  applies the residual FFN.

``edge_index`` follows PyG's ``[sender, receiver]`` convention.  In
particular, the source-compatible edge MLP input order is
``[receiver, sender, edge, global]``.  The node MLP receives two independent
directional aggregate channels: the receiver-side sum of
``[edge proposal, sender node]`` and the sender-side sum of
``[edge proposal, receiver node]``.  This is the sparse equivalent of the
legacy GPS++ InteractionNetwork configuration with direct-neighbour
aggregation, ``scatter_to='both'``, and concatenation.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch
from torch import Tensor, nn
from torch_geometric.utils import scatter


class LayerNormMLP(nn.Module):
    """GPS++'s two-dense-layer MLP with hidden LayerNorm and GELU.

    The reference configuration uses a dense layer, LayerNorm, GELU, and a
    final dense layer for the local edge/node/global update functions.  The
    activation dropout is intentionally between the GELU and final projection.
    """

    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        *,
        hidden_dim: int | None = None,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        _positive_int(input_dim, "input_dim")
        _positive_int(output_dim, "output_dim")
        if hidden_dim is None:
            hidden_dim = 4 * output_dim
        _positive_int(hidden_dim, "hidden_dim")
        _validate_dropout(dropout, "dropout")

        self.input_dim = input_dim
        self.output_dim = output_dim
        self.hidden_dim = hidden_dim
        self.dropout_rate = float(dropout)
        self.linear1 = nn.Linear(input_dim, hidden_dim)
        self.hidden_norm = nn.LayerNorm(hidden_dim)
        self.activation = nn.GELU()
        self.dropout = nn.Dropout(float(dropout))
        self.linear2 = nn.Linear(hidden_dim, output_dim)

    def forward(self, values: Tensor) -> Tensor:
        """Project a floating tensor whose final dimension is ``input_dim``."""

        _validate_feature_tensor(values, self.input_dim, "values")
        return self.linear2(
            self.dropout(self.activation(self.hidden_norm(self.linear1(values))))
        )


class FeedForward(nn.Module):
    """The GPS++ transformer FFN: dense, GELU, dropout, dense.

    LayerNorm for this path is applied after its residual addition by
    :class:`GPSPlusPlusBlock`, matching the GPS++ block topology.
    """

    def __init__(
        self,
        node_dim: int,
        *,
        hidden_dim: int | None = None,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        _positive_int(node_dim, "node_dim")
        if hidden_dim is None:
            hidden_dim = 4 * node_dim
        _positive_int(hidden_dim, "hidden_dim")
        _validate_dropout(dropout, "dropout")

        self.node_dim = node_dim
        self.hidden_dim = hidden_dim
        self.dropout_rate = float(dropout)
        self.linear1 = nn.Linear(node_dim, hidden_dim)
        self.activation = nn.GELU()
        self.dropout = nn.Dropout(float(dropout))
        self.linear2 = nn.Linear(hidden_dim, node_dim)

    def forward(self, nodes: Tensor) -> Tensor:
        """Return one FFN proposal per sparse node."""

        _validate_feature_tensor(nodes, self.node_dim, "nodes")
        return self.linear2(self.dropout(self.activation(self.linear1(nodes))))


class GraphDropout(nn.Module):
    """Graph-wise stochastic depth with source-compatible inverted scaling.

    In training mode, all rows belonging to one graph receive the same binary
    keep/drop decision.  The kept graph values are divided by ``1 - rate``;
    therefore the operation preserves the expected activation magnitude.
    ``batch=None`` is the graph-state case, where the leading dimension already
    indexes graphs.
    """

    def __init__(self, rate: float = 0.0) -> None:
        super().__init__()
        _validate_dropout(rate, "rate")
        self.rate = float(rate)

    def forward(
        self,
        values: Tensor,
        batch: Tensor | None = None,
        *,
        num_graphs: int | None = None,
    ) -> Tensor:
        """Apply a shared Bernoulli mask to each graph's rows."""

        if not isinstance(values, Tensor) or values.ndim < 2:
            raise ValueError("values must be a tensor with shape [R, ...]")
        if not torch.is_floating_point(values):
            raise ValueError("values must be a floating tensor")
        if self.rate == 0.0 or not self.training:
            return values

        if batch is None:
            graph_count = values.shape[0]
            if num_graphs is not None and num_graphs != graph_count:
                raise ValueError(
                    "num_graphs must equal values.shape[0] when batch is None"
                )
            graph_mask = _graph_dropout_mask(
                values, graph_count=graph_count, rate=self.rate
            )
            return values * graph_mask

        _validate_batch_index(batch, values.shape[0], values.device, "batch")
        if num_graphs is None:
            graph_count = int(batch.max().item()) + 1 if batch.numel() else 0
        else:
            _nonnegative_int(num_graphs, "num_graphs")
            graph_count = num_graphs
        if batch.numel() and int(batch.max().item()) >= graph_count:
            raise ValueError("batch contains a graph ID outside [0, num_graphs)")

        graph_mask = _graph_dropout_mask(
            values, graph_count=graph_count, rate=self.rate
        )
        return values * graph_mask[batch]


class _RowDropout(nn.Module):
    """Reference-style dropout with one feature-shared mask per sparse row."""

    def __init__(self, rate: float = 0.0) -> None:
        super().__init__()
        _validate_dropout(rate, "rate")
        self.rate = float(rate)

    def forward(self, values: Tensor) -> Tensor:
        if not isinstance(values, Tensor) or values.ndim != 2:
            raise ValueError("values must have shape [R, D]")
        if not torch.is_floating_point(values):
            raise ValueError("values must be a floating tensor")
        if self.rate == 0.0 or not self.training:
            return values
        keep_rate = 1.0 - self.rate
        mask = values.new_empty((values.shape[0], 1)).bernoulli_(keep_rate)
        return values * mask.div_(keep_rate)


@dataclass(frozen=True)
class LocalMPNNOutput:
    """Local GPS++ proposals and their residual state updates."""

    node_proposal: Tensor
    edge_proposal: Tensor
    global_proposal: Tensor
    nodes: Tensor
    edges: Tensor
    globals: Tensor


class LocalMPNN(nn.Module):
    """Source-oriented local GPS++ InteractionNetwork over a sparse batch.

    ``edge_index[0]`` is the sender and ``edge_index[1]`` is the receiver.
    The edge MLP gets ``[x_receiver, x_sender, edge, global]``.  Its proposal
    is aggregated independently in both directions:

    ``sum_receiver([edge proposal, x_sender])`` and
    ``sum_sender([edge proposal, x_receiver])``.

    The node MLP consumes both aggregate channels, the original node state,
    and the graph state.  The global MLP then consumes the original global
    state plus sums of the *proposed* node and edge states.  Each returned
    ``nodes``, ``edges``, and ``globals`` value is its corresponding residual
    update after graph-wise stochastic depth.
    """

    def __init__(
        self,
        node_dim: int,
        edge_dim: int,
        global_dim: int,
        *,
        node_hidden_dim: int | None = None,
        edge_hidden_dim: int | None = None,
        global_hidden_dim: int | None = None,
        dropout: float = 0.0,
        node_dropout: float = 0.0,
        edge_dropout: float = 0.0,
        global_dropout: float = 0.0,
        graph_dropout: float = 0.0,
    ) -> None:
        super().__init__()
        for value, name in (
            (node_dim, "node_dim"),
            (edge_dim, "edge_dim"),
            (global_dim, "global_dim"),
        ):
            _positive_int(value, name)
        _validate_dropout(dropout, "dropout")
        _validate_dropout(node_dropout, "node_dropout")
        _validate_dropout(edge_dropout, "edge_dropout")
        _validate_dropout(global_dropout, "global_dropout")
        _validate_dropout(graph_dropout, "graph_dropout")
        for value, name in (
            (node_hidden_dim, "node_hidden_dim"),
            (edge_hidden_dim, "edge_hidden_dim"),
            (global_hidden_dim, "global_hidden_dim"),
        ):
            if value is not None:
                _positive_int(value, name)

        self.node_dim = node_dim
        self.edge_dim = edge_dim
        self.global_dim = global_dim
        self.dropout_rate = float(dropout)
        self.node_dropout_rate = float(node_dropout)
        self.edge_dropout_rate = float(edge_dropout)
        self.global_dropout_rate = float(global_dropout)
        self.graph_dropout_rate = float(graph_dropout)
        self.edge_model = LayerNormMLP(
            2 * node_dim + edge_dim + global_dim,
            edge_dim,
            hidden_dim=edge_hidden_dim,
            dropout=dropout,
        )
        self.node_model = LayerNormMLP(
            2 * (edge_dim + node_dim) + node_dim + global_dim,
            node_dim,
            hidden_dim=node_hidden_dim,
            dropout=dropout,
        )
        self.global_model = LayerNormMLP(
            global_dim + node_dim + edge_dim,
            global_dim,
            hidden_dim=global_hidden_dim,
            dropout=dropout,
        )
        self.node_graph_dropout = GraphDropout(graph_dropout)
        self.edge_graph_dropout = GraphDropout(graph_dropout)
        self.global_graph_dropout = GraphDropout(graph_dropout)
        # Graphcore's source masks complete node/edge/global latent rows
        # (noise shape ``[..., rows, 1]``), not individual coordinates.
        self.node_output_dropout = _RowDropout(node_dropout)
        self.edge_output_dropout = _RowDropout(edge_dropout)
        self.global_output_dropout = _RowDropout(global_dropout)

    def forward(
        self,
        nodes: Tensor,
        edges: Tensor,
        globals_: Tensor,
        edge_index: Tensor,
        graph_batch: Tensor,
    ) -> LocalMPNNOutput:
        """Return local proposals followed by their graph-drop residual states."""

        num_graphs = _validate_local_inputs(
            nodes,
            edges,
            globals_,
            edge_index,
            graph_batch,
            node_dim=self.node_dim,
            edge_dim=self.edge_dim,
            global_dim=self.global_dim,
        )
        sender, receiver = edge_index
        edge_batch = graph_batch[sender]

        edge_input = torch.cat(
            (
                nodes[receiver],
                nodes[sender],
                edges,
                globals_[edge_batch],
            ),
            dim=-1,
        )
        # The reference configuration's edge dropout is before scatter, so
        # both directional node aggregates and the edge residual observe the
        # same dropped proposal.
        edge_proposal = self.edge_output_dropout(self.edge_model(edge_input))

        receiver_messages = torch.cat((edge_proposal, nodes[sender]), dim=-1)
        sender_messages = torch.cat((edge_proposal, nodes[receiver]), dim=-1)
        incoming = scatter(
            receiver_messages,
            receiver,
            dim=0,
            dim_size=nodes.shape[0],
            reduce="sum",
        )
        outgoing = scatter(
            sender_messages,
            sender,
            dim=0,
            dim_size=nodes.shape[0],
            reduce="sum",
        )
        node_input = torch.cat(
            (incoming, outgoing, nodes, globals_[graph_batch]), dim=-1
        )
        node_proposal = self.node_model(node_input)

        proposed_nodes_by_graph = scatter(
            node_proposal,
            graph_batch,
            dim=0,
            dim_size=num_graphs,
            reduce="sum",
        )
        proposed_edges_by_graph = scatter(
            edge_proposal,
            edge_batch,
            dim=0,
            dim_size=num_graphs,
            reduce="sum",
        )
        global_proposal = self.global_model(
            torch.cat(
                (globals_, proposed_nodes_by_graph, proposed_edges_by_graph), dim=-1
            )
        )

        updated_nodes = nodes + self.node_output_dropout(
            self.node_graph_dropout(node_proposal, graph_batch, num_graphs=num_graphs)
        )
        updated_edges = edges + self.edge_graph_dropout(
            edge_proposal, edge_batch, num_graphs=num_graphs
        )
        updated_globals = globals_ + self.global_output_dropout(
            self.global_graph_dropout(global_proposal, num_graphs=num_graphs)
        )
        return LocalMPNNOutput(
            node_proposal=node_proposal,
            edge_proposal=edge_proposal,
            global_proposal=global_proposal,
            nodes=updated_nodes,
            edges=updated_edges,
            globals=updated_globals,
        )


class BiasedSelfAttention(nn.Module):
    """GPS++ all-pairs shortest-path-biased self-attention without padding.

    ``pair_index`` must enumerate each ordered pair in each graph exactly once,
    using ``[query/source, key/target]`` rows.  It is intentionally sparse at
    the PyG batch boundary; dense ``[heads, n, n]`` attention bias is built
    only for one graph at a time.  That makes cross-graph attention impossible
    even when multiple graphs are concatenated in a batch.

    ``spd`` uses ``-1`` for an unreachable pair, ``0`` for self pairs, and
    positive unweighted topological distances otherwise.  Values above
    ``max_spd`` share the final distance bucket.
    """

    def __init__(
        self,
        node_dim: int,
        num_heads: int,
        max_spd: int,
        *,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        _positive_int(node_dim, "node_dim")
        _positive_int(num_heads, "num_heads")
        _positive_int(max_spd, "max_spd")
        _validate_dropout(dropout, "dropout")
        if node_dim % num_heads:
            raise ValueError("node_dim must be divisible by num_heads")

        self.node_dim = node_dim
        self.num_heads = num_heads
        self.max_spd = max_spd
        self.head_dim = node_dim // num_heads
        self.scale = self.head_dim**-0.5
        self.dropout_rate = float(dropout)
        self.query_projection = nn.Linear(node_dim, node_dim)
        self.key_projection = nn.Linear(node_dim, node_dim)
        self.value_projection = nn.Linear(node_dim, node_dim)
        self.output_projection = nn.Linear(node_dim, node_dim)
        # Index 0 is the unreachable bucket; index 1 is distance 0.
        self.spatial_bias = nn.Embedding(max_spd + 2, num_heads)
        self.attention_dropout = nn.Dropout(float(dropout))

    def forward(
        self,
        nodes: Tensor,
        graph_batch: Tensor,
        pair_index: Tensor,
        spd: Tensor,
    ) -> Tensor:
        """Return graph-local attention proposals with shape ``[N, node_dim]``."""

        num_graphs, ptr = _validate_attention_inputs(
            nodes,
            graph_batch,
            pair_index,
            spd,
            node_dim=self.node_dim,
        )
        del num_graphs  # ``ptr`` captures every non-empty graph exactly once.

        source, target = pair_index
        pair_graph = graph_batch[source]
        if not torch.equal(pair_graph, graph_batch[target]):
            raise ValueError("pair_index must not contain a cross-graph pair")
        bias_by_pair = self.spatial_bias(_spd_bucket(spd, self.max_spd)).to(
            dtype=nodes.dtype
        )

        output = torch.empty_like(nodes)
        for graph_index, (start, stop) in enumerate(
            zip(ptr[:-1].tolist(), ptr[1:].tolist(), strict=True)
        ):
            node_count = stop - start
            pair_mask = pair_graph == graph_index
            if int(pair_mask.sum().item()) != node_count * node_count:
                raise ValueError(
                    "pair_index must enumerate every ordered node pair exactly once "
                    "for each graph"
                )
            local_query = source[pair_mask] - start
            local_key = target[pair_mask] - start
            _validate_complete_pair_grid(local_query, local_key, node_count)

            dense_bias = nodes.new_zeros((self.num_heads, node_count, node_count))
            dense_bias[:, local_query, local_key] = bias_by_pair[pair_mask].transpose(
                0, 1
            )
            graph_nodes = nodes[start:stop]
            query = (
                self.query_projection(graph_nodes)
                .reshape(node_count, self.num_heads, self.head_dim)
                .transpose(0, 1)
            )
            key = (
                self.key_projection(graph_nodes)
                .reshape(node_count, self.num_heads, self.head_dim)
                .transpose(0, 1)
            )
            value = (
                self.value_projection(graph_nodes)
                .reshape(node_count, self.num_heads, self.head_dim)
                .transpose(0, 1)
            )
            weights = torch.softmax(
                torch.matmul(query, key.transpose(-2, -1)) * self.scale + dense_bias,
                dim=-1,
            )
            weights = self.attention_dropout(weights)
            context = (
                torch.matmul(weights, value)
                .transpose(0, 1)
                .reshape(node_count, self.node_dim)
            )
            output[start:stop] = self.output_projection(context)
        return output


@dataclass(frozen=True)
class GPSPlusPlusBlockOutput:
    """Post-block sparse node, edge, and global latent states."""

    nodes: Tensor
    edges: Tensor
    globals: Tensor


class GPSPlusPlusBlock(nn.Module):
    """One parallel-local/global GPS++ block followed by an FFN.

    The local MPNN and attention paths both read the same incoming node state.
    The local path's intrinsic residual state and the attention path's explicit
    residual state are normalized independently.  Their normalized values are
    added before the final FFN residual and LayerNorm.  Edge and global states
    come only from the local MPNN, as in GPS++.
    """

    def __init__(
        self,
        node_dim: int,
        edge_dim: int,
        global_dim: int,
        num_heads: int,
        max_spd: int,
        *,
        node_hidden_dim: int | None = None,
        edge_hidden_dim: int | None = None,
        global_hidden_dim: int | None = None,
        ffn_hidden_dim: int | None = None,
        dropout: float = 0.0,
        node_dropout: float = 0.0,
        edge_dropout: float = 0.0,
        global_dropout: float = 0.0,
        ffn_dropout: float = 0.0,
        local_output_dropout: float = 0.0,
        attention_output_dropout: float = 0.0,
        ffn_output_dropout: float = 0.0,
        attention_dropout: float = 0.0,
        graph_dropout: float = 0.0,
        local_graph_dropout: float = 0.0,
    ) -> None:
        super().__init__()
        for value, name in (
            (node_dim, "node_dim"),
            (edge_dim, "edge_dim"),
            (global_dim, "global_dim"),
            (num_heads, "num_heads"),
            (max_spd, "max_spd"),
        ):
            _positive_int(value, name)
        for value, name in (
            (node_hidden_dim, "node_hidden_dim"),
            (edge_hidden_dim, "edge_hidden_dim"),
            (global_hidden_dim, "global_hidden_dim"),
            (ffn_hidden_dim, "ffn_hidden_dim"),
        ):
            if value is not None:
                _positive_int(value, name)
        _validate_dropout(dropout, "dropout")
        _validate_dropout(node_dropout, "node_dropout")
        _validate_dropout(edge_dropout, "edge_dropout")
        _validate_dropout(global_dropout, "global_dropout")
        _validate_dropout(ffn_dropout, "ffn_dropout")
        _validate_dropout(local_output_dropout, "local_output_dropout")
        _validate_dropout(attention_output_dropout, "attention_output_dropout")
        _validate_dropout(ffn_output_dropout, "ffn_output_dropout")
        _validate_dropout(attention_dropout, "attention_dropout")
        _validate_dropout(graph_dropout, "graph_dropout")
        _validate_dropout(local_graph_dropout, "local_graph_dropout")

        self.node_dim = node_dim
        self.edge_dim = edge_dim
        self.global_dim = global_dim
        self.local_mpnn = LocalMPNN(
            node_dim,
            edge_dim,
            global_dim,
            node_hidden_dim=node_hidden_dim,
            edge_hidden_dim=edge_hidden_dim,
            global_hidden_dim=global_hidden_dim,
            dropout=dropout,
            node_dropout=node_dropout,
            edge_dropout=edge_dropout,
            global_dropout=global_dropout,
            graph_dropout=local_graph_dropout,
        )
        self.attention = BiasedSelfAttention(
            node_dim,
            num_heads,
            max_spd,
            dropout=attention_dropout,
        )
        self.local_norm = nn.LayerNorm(node_dim)
        self.attention_norm = nn.LayerNorm(node_dim)
        self.feed_forward = FeedForward(
            node_dim, hidden_dim=ffn_hidden_dim, dropout=ffn_dropout
        )
        self.ffn_norm = nn.LayerNorm(node_dim)
        self.attention_graph_dropout = GraphDropout(graph_dropout)
        self.ffn_graph_dropout = GraphDropout(graph_dropout)
        self.local_output_dropout = nn.Dropout(float(local_output_dropout))
        self.attention_output_dropout = nn.Dropout(float(attention_output_dropout))
        self.ffn_output_dropout = nn.Dropout(float(ffn_output_dropout))

    def forward(
        self,
        nodes: Tensor,
        edges: Tensor,
        globals_: Tensor,
        edge_index: Tensor,
        graph_batch: Tensor,
        pair_index: Tensor,
        spd: Tensor,
    ) -> GPSPlusPlusBlockOutput:
        """Apply the source-ordered GPS++ local/global/FFN computation."""

        # Local and global paths deliberately receive the same pre-block nodes.
        local = self.local_mpnn(nodes, edges, globals_, edge_index, graph_batch)
        local_nodes = self.local_norm(self.local_output_dropout(local.nodes))

        attention_proposal = self.attention(nodes, graph_batch, pair_index, spd)
        attention_proposal = self.attention_graph_dropout(
            self.attention_output_dropout(attention_proposal), graph_batch
        )
        attention_nodes = self.attention_norm(nodes + attention_proposal)
        combined_nodes = local_nodes + attention_nodes

        ffn_proposal = self.feed_forward(combined_nodes)
        ffn_proposal = self.ffn_graph_dropout(
            self.ffn_output_dropout(ffn_proposal), graph_batch
        )
        output_nodes = self.ffn_norm(combined_nodes + ffn_proposal)
        return GPSPlusPlusBlockOutput(
            nodes=output_nodes,
            edges=local.edges,
            globals=local.globals,
        )


def _graph_dropout_mask(values: Tensor, *, graph_count: int, rate: float) -> Tensor:
    """Create a broadcastable inverted-dropout mask with one entry per graph."""

    shape = (graph_count,) + (1,) * (values.ndim - 1)
    keep_rate = 1.0 - rate
    return values.new_empty(shape).bernoulli_(keep_rate).div_(keep_rate)


def _validate_feature_tensor(values: Tensor, width: int, name: str) -> None:
    if (
        not isinstance(values, Tensor)
        or values.ndim < 2
        or values.shape[-1] != width
        or not torch.is_floating_point(values)
    ):
        raise ValueError(
            f"{name} must be a floating tensor with final dimension {width}"
        )


def _validate_batch_index(
    batch: Tensor,
    row_count: int,
    device: torch.device,
    name: str,
) -> None:
    if (
        not isinstance(batch, Tensor)
        or batch.ndim != 1
        or batch.dtype != torch.long
        or batch.shape[0] != row_count
    ):
        raise ValueError(f"{name} must have shape [{row_count}] and dtype torch.long")
    if batch.device != device:
        raise ValueError(f"{name} and values must share a device")
    if batch.numel() and int(batch.min().item()) < 0:
        raise ValueError(f"{name} must contain non-negative graph IDs")


def _validate_local_inputs(
    nodes: Tensor,
    edges: Tensor,
    globals_: Tensor,
    edge_index: Tensor,
    graph_batch: Tensor,
    *,
    node_dim: int,
    edge_dim: int,
    global_dim: int,
) -> int:
    _validate_feature_tensor(nodes, node_dim, "nodes")
    if nodes.ndim != 2 or nodes.shape[0] < 1:
        raise ValueError(f"nodes must have non-empty shape [N, {node_dim}]")
    _validate_feature_tensor(edges, edge_dim, "edges")
    if edges.ndim != 2:
        raise ValueError(f"edges must have shape [E, {edge_dim}]")
    if edges.device != nodes.device:
        raise ValueError("nodes and edges must share a device")
    _validate_batch_index(graph_batch, nodes.shape[0], nodes.device, "graph_batch")
    num_graphs, _ = _graph_ptr(graph_batch)

    if (
        not isinstance(edge_index, Tensor)
        or edge_index.ndim != 2
        or edge_index.shape != (2, edges.shape[0])
        or edge_index.dtype != torch.long
    ):
        raise ValueError("edge_index must have shape [2, E] and dtype torch.long")
    if edge_index.device != nodes.device:
        raise ValueError("edge_index and nodes must share a device")
    if edge_index.numel() and (
        int(edge_index.min().item()) < 0
        or int(edge_index.max().item()) >= nodes.shape[0]
    ):
        raise ValueError("edge_index contains a node outside [0, N)")
    if edge_index.numel() and not torch.equal(
        graph_batch[edge_index[0]], graph_batch[edge_index[1]]
    ):
        raise ValueError("edge_index must not contain a cross-graph edge")

    _validate_feature_tensor(globals_, global_dim, "globals_")
    if globals_.ndim != 2 or globals_.shape[0] != num_graphs:
        raise ValueError(f"globals_ must have shape [{num_graphs}, {global_dim}]")
    if globals_.device != nodes.device:
        raise ValueError("globals_ and nodes must share a device")
    return num_graphs


def _validate_attention_inputs(
    nodes: Tensor,
    graph_batch: Tensor,
    pair_index: Tensor,
    spd: Tensor,
    *,
    node_dim: int,
) -> tuple[int, Tensor]:
    _validate_feature_tensor(nodes, node_dim, "nodes")
    if nodes.ndim != 2 or nodes.shape[0] < 1:
        raise ValueError(f"nodes must have non-empty shape [N, {node_dim}]")
    _validate_batch_index(graph_batch, nodes.shape[0], nodes.device, "graph_batch")
    num_graphs, ptr = _graph_ptr(graph_batch)
    if (
        not isinstance(pair_index, Tensor)
        or pair_index.ndim != 2
        or pair_index.shape[0] != 2
        or pair_index.dtype != torch.long
    ):
        raise ValueError("pair_index must have shape [2, P] and dtype torch.long")
    if pair_index.device != nodes.device:
        raise ValueError("pair_index and nodes must share a device")
    if pair_index.numel() == 0:
        raise ValueError("pair_index must enumerate at least the singleton pair")
    if (
        int(pair_index.min().item()) < 0
        or int(pair_index.max().item()) >= nodes.shape[0]
    ):
        raise ValueError("pair_index contains a node outside [0, N)")
    if (
        not isinstance(spd, Tensor)
        or spd.ndim != 1
        or spd.dtype != torch.long
        or spd.shape[0] != pair_index.shape[1]
    ):
        raise ValueError("spd must have shape [P] and dtype torch.long")
    if spd.device != nodes.device:
        raise ValueError("spd and nodes must share a device")
    if int(spd.min().item()) < -1:
        raise ValueError(
            "spd values must be -1 or non-negative shortest-path distances"
        )
    return num_graphs, ptr


def _graph_ptr(graph_batch: Tensor) -> tuple[int, Tensor]:
    """Validate the standard contiguous PyG graph order and return its ptr."""

    if graph_batch.numel() == 0:
        raise ValueError("graph_batch must contain at least one node")
    num_graphs = int(graph_batch.max().item()) + 1
    counts = torch.bincount(graph_batch, minlength=num_graphs)
    if bool((counts == 0).any()):
        raise ValueError("graph_batch graph IDs must be contiguous from zero")
    expected = torch.repeat_interleave(
        torch.arange(num_graphs, dtype=torch.long, device=graph_batch.device), counts
    )
    if not torch.equal(graph_batch, expected):
        raise ValueError("graph_batch must group graph rows contiguously in PyG order")
    ptr = torch.cat((counts.new_zeros(1), counts.cumsum(dim=0)))
    return num_graphs, ptr


def _validate_complete_pair_grid(
    local_query: Tensor,
    local_key: Tensor,
    node_count: int,
) -> None:
    if local_query.numel() != node_count * node_count:
        raise ValueError("pair_index does not contain a complete ordered pair grid")
    if (
        int(local_query.min().item()) < 0
        or int(local_key.min().item()) < 0
        or int(local_query.max().item()) >= node_count
        or int(local_key.max().item()) >= node_count
    ):
        raise ValueError("pair_index pairs must lie inside their graph's node range")
    pair_ids = local_query * node_count + local_key
    if torch.unique(pair_ids).numel() != node_count * node_count:
        raise ValueError("pair_index must not omit or duplicate an ordered node pair")


def _spd_bucket(spd: Tensor, max_spd: int) -> Tensor:
    """Map ``-1``/distance labels to the shortest-path embedding vocabulary."""

    return spd.clamp(min=-1, max=max_spd) + 1


def _positive_int(value: int, name: str) -> None:
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise ValueError(f"{name} must be a positive integer")


def _nonnegative_int(value: int, name: str) -> None:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError(f"{name} must be a non-negative integer")


def _validate_dropout(value: float, name: str) -> None:
    if (
        isinstance(value, bool)
        or not isinstance(value, (float, int))
        or not math.isfinite(float(value))
        or not 0.0 <= float(value) < 1.0
    ):
        raise ValueError(f"{name} must be a finite value in [0, 1)")


__all__ = [
    "BiasedSelfAttention",
    "FeedForward",
    "GPSPlusPlusBlock",
    "GPSPlusPlusBlockOutput",
    "GraphDropout",
    "LayerNormMLP",
    "LocalMPNN",
    "LocalMPNNOutput",
]
