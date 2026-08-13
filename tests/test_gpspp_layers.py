"""Topology-sensitive unit tests for sparse GPS++ layer primitives."""

from __future__ import annotations

import torch
from torch import Tensor, nn

from molgnn.models.gpspp_2023.layers import (
    BiasedSelfAttention,
    GPSPlusPlusBlock,
    GraphDropout,
    LocalMPNN,
    LocalMPNNOutput,
)


class _FirstColumn(nn.Module):
    """Capture an MLP input while using its first column as a one-wide output."""

    def __init__(self) -> None:
        super().__init__()
        self.last_input: Tensor | None = None

    def forward(self, values: Tensor) -> Tensor:
        self.last_input = values.detach().clone()
        return values[:, :1]


class _CaptureZeros(nn.Module):
    """Capture an MLP input and emit a zero proposal of configurable width."""

    def __init__(self, output_dim: int) -> None:
        super().__init__()
        self.output_dim = output_dim
        self.last_input: Tensor | None = None

    def forward(self, values: Tensor) -> Tensor:
        self.last_input = values.detach().clone()
        return values.new_zeros((values.shape[0], self.output_dim))


def test_local_mpnn_keeps_source_aligned_directional_channels_separate() -> None:
    local = LocalMPNN(node_dim=1, edge_dim=1, global_dim=1).eval()
    edge_model = _FirstColumn()
    node_model = _CaptureZeros(output_dim=1)
    local.edge_model = edge_model
    local.node_model = node_model
    local.global_model = _CaptureZeros(output_dim=1)

    nodes = torch.tensor([[2.0], [5.0]])
    edges = torch.tensor([[7.0]])
    globals_ = torch.tensor([[11.0]])
    edge_index = torch.tensor([[0], [1]], dtype=torch.long)
    graph_batch = torch.zeros(2, dtype=torch.long)

    output = local(nodes, edges, globals_, edge_index, graph_batch)

    # GPS++ uses [receiver, sender, edge, global] for its edge update.
    assert torch.equal(edge_model.last_input, torch.tensor([[5.0, 2.0, 7.0, 11.0]]))
    # Node 0 gets only the sender-side channel [edge proposal, receiver node],
    # while node 1 gets only the receiver-side [edge proposal, sender node].
    assert torch.equal(
        node_model.last_input,
        torch.tensor(
            [
                [0.0, 0.0, 5.0, 5.0, 2.0, 11.0],
                [5.0, 2.0, 0.0, 0.0, 5.0, 11.0],
            ]
        ),
    )
    assert torch.equal(output.edge_proposal, torch.tensor([[5.0]]))
    assert torch.equal(output.nodes, nodes)


def test_biased_attention_is_graph_local_and_uses_shortest_path_bias() -> None:
    attention = BiasedSelfAttention(1, 1, max_spd=2).eval()
    with torch.no_grad():
        for projection in (
            attention.query_projection,
            attention.key_projection,
        ):
            projection.weight.zero_()
            projection.bias.zero_()
        attention.value_projection.weight.fill_(1.0)
        attention.value_projection.bias.zero_()
        attention.output_projection.weight.fill_(1.0)
        attention.output_projection.bias.zero_()
        attention.spatial_bias.weight.zero_()

    nodes = torch.tensor([[0.0], [10.0], [50.0]])
    graph_batch = torch.tensor([0, 0, 1], dtype=torch.long)
    pair_index = torch.tensor(
        [[0, 0, 1, 1, 2], [0, 1, 0, 1, 2]], dtype=torch.long
    )
    spd = torch.tensor([0, 1, 1, 0, 0], dtype=torch.long)

    neutral = attention(nodes, graph_batch, pair_index, spd)
    assert torch.allclose(neutral, torch.tensor([[5.0], [5.0], [50.0]]))

    changed_other_graph = nodes.clone()
    changed_other_graph[2] = 5000.0
    isolated = attention(changed_other_graph, graph_batch, pair_index, spd)
    assert torch.allclose(isolated[:2], neutral[:2])
    assert torch.equal(isolated[2], torch.tensor([5000.0]))

    with torch.no_grad():
        # Bucket 2 represents topological distance one, so query node 0 now
        # strongly favors its non-self key node 1.
        attention.spatial_bias.weight[2, 0] = 8.0
    biased = attention(nodes, graph_batch, pair_index, spd)
    assert biased[0, 0] > neutral[0, 0] + 4.0
    assert biased[1, 0] < neutral[1, 0] - 4.0


class _StubLocalMPNN(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.seen_nodes: Tensor | None = None

    def forward(
        self,
        nodes: Tensor,
        edges: Tensor,
        globals_: Tensor,
        edge_index: Tensor,
        graph_batch: Tensor,
    ) -> LocalMPNNOutput:
        del edge_index, graph_batch
        self.seen_nodes = nodes.detach().clone()
        return LocalMPNNOutput(
            node_proposal=torch.full_like(nodes, 10.0),
            edge_proposal=torch.zeros_like(edges),
            global_proposal=torch.zeros_like(globals_),
            nodes=nodes + 10.0,
            edges=edges + 3.0,
            globals=globals_ + 4.0,
        )


class _StubAttention(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.seen_nodes: Tensor | None = None

    def forward(
        self,
        nodes: Tensor,
        graph_batch: Tensor,
        pair_index: Tensor,
        spd: Tensor,
    ) -> Tensor:
        del graph_batch, pair_index, spd
        self.seen_nodes = nodes.detach().clone()
        return nodes + 1.0


class _StubFFN(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.seen_nodes: Tensor | None = None

    def forward(self, nodes: Tensor) -> Tensor:
        self.seen_nodes = nodes.detach().clone()
        return torch.full_like(nodes, 7.0)


class _BatchIdentity(nn.Module):
    def forward(self, values: Tensor, graph_batch: Tensor | None = None) -> Tensor:
        del graph_batch
        return values


def test_hybrid_block_runs_local_and_attention_in_parallel_before_ffn() -> None:
    block = GPSPlusPlusBlock(
        node_dim=1,
        edge_dim=1,
        global_dim=1,
        num_heads=1,
        max_spd=2,
    ).eval()
    local = _StubLocalMPNN()
    attention = _StubAttention()
    ffn = _StubFFN()
    block.local_mpnn = local
    block.attention = attention
    block.feed_forward = ffn
    block.local_norm = nn.Identity()
    block.attention_norm = nn.Identity()
    block.ffn_norm = nn.Identity()
    block.local_output_dropout = nn.Identity()
    block.attention_output_dropout = nn.Identity()
    block.ffn_output_dropout = nn.Identity()
    block.attention_graph_dropout = _BatchIdentity()
    block.ffn_graph_dropout = _BatchIdentity()

    nodes = torch.tensor([[2.0], [5.0]])
    edges = torch.tensor([[1.0]])
    globals_ = torch.tensor([[3.0]])
    output = block(
        nodes,
        edges,
        globals_,
        torch.tensor([[0], [1]], dtype=torch.long),
        torch.zeros(2, dtype=torch.long),
        torch.tensor([[0, 0, 1, 1], [0, 1, 0, 1]], dtype=torch.long),
        torch.tensor([0, 1, 1, 0], dtype=torch.long),
    )

    assert torch.equal(local.seen_nodes, nodes)
    assert torch.equal(attention.seen_nodes, nodes)
    # local: x + 10; attention residual: x + (x + 1); then add both.
    expected_combined = 3 * nodes + 11.0
    assert torch.equal(ffn.seen_nodes, expected_combined)
    assert torch.equal(output.nodes, expected_combined + 7.0)
    assert torch.equal(output.edges, edges + 3.0)
    assert torch.equal(output.globals, globals_ + 4.0)


def test_graph_dropout_is_shared_within_graph_and_singleton_zero_edge_inputs_work() -> None:
    torch.manual_seed(7)
    dropout = GraphDropout(rate=0.5).train()
    values = torch.ones((4, 2))
    graph_batch = torch.tensor([0, 0, 1, 1], dtype=torch.long)
    dropped = dropout(values, graph_batch)
    assert torch.equal(dropped[0], dropped[1])
    assert torch.equal(dropped[2], dropped[3])
    assert set(dropped.flatten().tolist()).issubset({0.0, 2.0})
    assert torch.equal(dropout.eval()(values, graph_batch), values)

    local = LocalMPNN(node_dim=2, edge_dim=3, global_dim=1).eval()
    local_output = local(
        torch.randn((1, 2)),
        torch.empty((0, 3)),
        torch.randn((1, 1)),
        torch.empty((2, 0), dtype=torch.long),
        torch.zeros(1, dtype=torch.long),
    )
    attention = BiasedSelfAttention(2, 1, max_spd=2).eval()
    attention_output = attention(
        local_output.nodes,
        torch.zeros(1, dtype=torch.long),
        torch.zeros((2, 1), dtype=torch.long),
        torch.zeros(1, dtype=torch.long),
    )
    assert local_output.edges.shape == (0, 3)
    assert local_output.globals.shape == (1, 1)
    assert attention_output.shape == (1, 2)
