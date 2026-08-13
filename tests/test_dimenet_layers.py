"""Topology-sensitive tests for DimeNet directed-edge layer primitives."""

import torch

from molgnn.models.dimenet_2020.layers import (
    EmbeddingBlock,
    InteractionBlock,
    OutputBlock,
)


def test_embedding_uses_source_then_target_atom_embeddings_and_linear_rbf() -> None:
    block = EmbeddingBlock(num_radial=1, hidden_dim=1, max_atomic_number=8).eval()
    with torch.no_grad():
        block.atom_embedding.weight.zero_()
        block.atom_embedding.weight[1] = 2.0
        block.atom_embedding.weight[6] = 5.0
        block.radial_projection.weight.fill_(3.0)
        block.message_projection.weight.copy_(torch.tensor([[1.0, 10.0, 100.0]]))
        block.message_projection.bias.zero_()

    atomic_number = torch.tensor([1, 6], dtype=torch.long)
    rbf = torch.tensor([[0.1]], dtype=torch.float32)
    actual = block(
        atomic_number,
        rbf,
        torch.tensor([0], dtype=torch.long),
        torch.tensor([1], dtype=torch.long),
    )

    assert torch.allclose(actual, torch.nn.functional.silu(torch.tensor([[82.0]])))
    assert block.radial_projection.bias is None


def test_interaction_applies_target_edge_rbf_at_idx_ji_not_idx_kj() -> None:
    block = InteractionBlock(
        hidden_dim=1,
        num_bilinear=1,
        num_spherical=1,
        num_radial=1,
        num_before_skip=0,
        num_after_skip=0,
    ).eval()
    with torch.no_grad():
        block.radial_projection.weight.fill_(1.0)
        block.spherical_projection.weight.fill_(1.0)
        block.incoming_projection.weight.fill_(1.0)
        block.incoming_projection.bias.zero_()
        block.target_projection.weight.zero_()
        block.target_projection.bias.zero_()
        block.bilinear.fill_(1.0)
        block.final_projection.weight.fill_(1.0)
        block.final_projection.bias.zero_()

    messages = torch.tensor([[1.0], [2.0]])
    target_rbf = torch.tensor([[2.0], [7.0]])
    sbf = torch.tensor([[3.0]])
    output = block(
        messages,
        target_rbf,
        sbf,
        torch.tensor([0], dtype=torch.long),
        torch.tensor([1], dtype=torch.long),
    )

    incoming = torch.nn.functional.silu(torch.tensor(1.0))
    aggregate = 3 * incoming * 7
    expected_target = 2 + torch.nn.functional.silu(aggregate)
    assert torch.allclose(output[1], expected_target.reshape(1), atol=1e-6)


def test_output_block_sums_messages_at_target_atoms() -> None:
    block = OutputBlock(
        num_radial=1,
        hidden_dim=1,
        num_targets=1,
        num_dense_output=0,
        output_initializer="glorot_orthogonal",
    ).eval()
    with torch.no_grad():
        block.radial_projection.weight.fill_(1.0)
        block.output_projection.weight.fill_(1.0)

    actual = block(
        torch.tensor([[2.0], [3.0], [5.0]]),
        torch.tensor([[1.0], [2.0], [1.0]]),
        torch.tensor([1, 1, 2], dtype=torch.long),
        3,
    )

    assert torch.equal(actual, torch.tensor([[0.0], [8.0], [5.0]]))
