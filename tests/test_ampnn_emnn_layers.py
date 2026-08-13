"""Numerical invariants for the private AMPNN/EMNN shared layers."""

import math

import torch
from torch import nn

from molgnn.models.ampnn_emnn_2020.layers import (
    GatedGraphGather,
    SELUFeedForward,
    vector_attention_aggregate,
)


def test_vector_attention_is_coordinatewise_within_each_segment() -> None:
    embeddings = torch.tensor(
        [[1.0, 10.0], [3.0, 30.0], [7.0, 70.0]], dtype=torch.float32
    )
    energies = torch.tensor(
        [[0.0, math.log(3.0)], [math.log(3.0), 0.0], [0.0, 0.0]],
        dtype=torch.float32,
    )
    target = torch.tensor([0, 0, 1], dtype=torch.long)

    actual = vector_attention_aggregate(embeddings, energies, target, 2)

    # For target 0 the first coordinate weighs rows (0, 1) as (1/4, 3/4),
    # whereas the second coordinate weighs them as (3/4, 1/4).
    expected = torch.tensor([[2.5, 15.0], [7.0, 70.0]])
    assert torch.allclose(actual, expected, atol=1e-6)


def test_gated_graph_gather_matches_source_gate_value_formula() -> None:
    gather = GatedGraphGather(
        hidden_node_dim=2,
        input_node_dim=1,
        output_dim=1,
        gate_hidden_dims=(),
        value_hidden_dims=(),
    ).eval()
    gate_linear = gather.gate_network.network[1]
    value_linear = gather.value_network.network[1]
    assert isinstance(gate_linear, nn.Linear)
    assert isinstance(value_linear, nn.Linear)
    with torch.no_grad():
        gate_linear.weight.copy_(torch.tensor([[1.0, -1.0, 0.5]]))
        value_linear.weight.copy_(torch.tensor([[0.2, 0.3]]))

    hidden_nodes = torch.tensor([[1.0, 2.0], [3.0, 1.0], [2.0, -1.0]])
    input_nodes = torch.tensor([[4.0], [0.0], [2.0]])
    graph_batch = torch.tensor([0, 0, 1], dtype=torch.long)

    actual = gather(hidden_nodes, input_nodes, graph_batch, 2)
    gate = torch.sigmoid(
        torch.cat((hidden_nodes, input_nodes), dim=-1) @ gate_linear.weight.T
    )
    values = hidden_nodes @ value_linear.weight.T
    expected = torch.stack(
        ((gate * values)[:2].sum(dim=0), (gate * values)[2:].sum(dim=0))
    )

    assert torch.allclose(actual, expected, atol=1e-6)


def test_selu_feed_forward_preserves_source_layer_and_bias_profile() -> None:
    network = SELUFeedForward(2, (3,), 1, dropout=0.2)

    layers = tuple(network.network)
    assert isinstance(layers[0], nn.AlphaDropout)
    assert isinstance(layers[1], nn.Linear)
    assert isinstance(layers[2], nn.SELU)
    assert isinstance(layers[3], nn.AlphaDropout)
    assert isinstance(layers[4], nn.Linear)
    assert all(layer.bias is None for layer in layers if isinstance(layer, nn.Linear))

    values = torch.randn(4, 2, requires_grad=True)
    output = network(values)
    output.square().mean().backward()
    assert output.shape == (4, 1)
    assert values.grad is not None
