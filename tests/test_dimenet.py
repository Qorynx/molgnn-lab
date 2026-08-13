"""End-to-end invariants for the coordinate-backed DimeNet-2020 core."""

from __future__ import annotations

import pytest
import torch
from torch_geometric.data import Batch

from molgnn.data import MolecularData
from molgnn.models.dimenet_2020 import DimeNet2020
from molgnn.transforms import add_dimenet_inputs


def _raw_sample(
    positions: torch.Tensor,
    atomic_numbers: torch.Tensor,
    *,
    sample_id: int,
) -> MolecularData:
    node_count = positions.shape[0]
    return MolecularData(
        x=torch.zeros((node_count, 1), dtype=torch.float32),
        edge_index=torch.empty((2, 0), dtype=torch.long),
        edge_attr=torch.empty((0, 1), dtype=torch.float32),
        y=torch.zeros((1, 1), dtype=torch.float32),
        y_mask=torch.ones((1, 1), dtype=torch.bool),
        sample_id=torch.tensor([sample_id], dtype=torch.long),
        atomic_number=atomic_numbers.to(dtype=torch.long),
        pos=positions.to(dtype=torch.float32),
    )


def _triangle(sample_id: int = 0) -> MolecularData:
    return _raw_sample(
        torch.tensor(
            [[0.0, 0.0, 0.0], [1.1, 0.1, 0.0], [0.2, 1.3, 0.4]],
            dtype=torch.float32,
        ),
        torch.tensor([6, 7, 8]),
        sample_id=sample_id,
    )


def _model(*, output_initializer: str = "glorot_orthogonal") -> DimeNet2020:
    torch.manual_seed(17)
    return DimeNet2020(
        hidden_dim=8,
        num_blocks=2,
        num_bilinear=2,
        num_spherical=2,
        num_radial=2,
        num_before_skip=1,
        num_after_skip=1,
        num_dense_output=1,
        output_initializer=output_initializer,
        num_targets=2,
    ).eval()


def _batch(*samples: MolecularData) -> Batch:
    return Batch.from_data_list([add_dimenet_inputs(sample) for sample in samples])


def test_forward_preserves_coordinate_gradients_and_state_dict_round_trip() -> None:
    batch = _batch(_triangle())
    batch.pos = batch.pos.detach().clone().requires_grad_(True)
    model = _model()

    output = model(batch)
    first = torch.autograd.grad(output.sum(), batch.pos, create_graph=True)[0]
    second = torch.autograd.grad(first.square().sum(), batch.pos)[0]
    restored = _model()
    restored.load_state_dict(model.state_dict())

    assert output.shape == (1, 2)
    assert torch.isfinite(output).all()
    assert torch.isfinite(first).all()
    assert torch.isfinite(second).all()
    assert torch.allclose(restored(batch), output)


def test_prediction_is_rigid_motion_and_atom_order_invariant() -> None:
    model = _model()
    original = _triangle()
    prediction = model(_batch(original))
    rotation = torch.tensor(
        [[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]],
        dtype=torch.float32,
    )
    rigid = _raw_sample(
        original.pos @ rotation.T + torch.tensor([2.0, -3.0, 0.5]),
        original.atomic_number,
        sample_id=1,
    )
    inverted = _raw_sample(-original.pos, original.atomic_number, sample_id=2)
    permutation = torch.tensor([2, 0, 1], dtype=torch.long)
    relabeled = _raw_sample(
        original.pos[permutation],
        original.atomic_number[permutation],
        sample_id=3,
    )

    assert torch.allclose(model(_batch(rigid)), prediction, atol=1e-5)
    assert torch.allclose(model(_batch(inverted)), prediction, atol=1e-5)
    assert torch.allclose(model(_batch(relabeled)), prediction, atol=1e-5)


def test_edge_triplet_order_and_companion_batch_do_not_change_prediction() -> None:
    model = _model()
    first = add_dimenet_inputs(_triangle())
    edge_count = first.dimenet_edge_index.shape[1]
    permutation = torch.tensor([4, 0, 5, 2, 1, 3], dtype=torch.long)
    inverse = torch.empty_like(permutation)
    inverse[permutation] = torch.arange(edge_count, dtype=torch.long)
    reordered = first.clone()
    reordered.dimenet_edge_index = first.dimenet_edge_index[:, permutation]
    reordered.dimenet_triplet_edge_index = inverse[first.dimenet_triplet_edge_index]
    companion = _raw_sample(
        torch.tensor([[0.0, 0.0, 0.0], [1.5, 0.0, 0.0]], dtype=torch.float32),
        torch.tensor([1, 6]),
        sample_id=1,
    )

    alone = model(Batch.from_data_list([first]))
    reordered_prediction = model(Batch.from_data_list([reordered]))
    mixed = model(Batch.from_data_list([first, add_dimenet_inputs(companion)]))

    assert torch.allclose(reordered_prediction, alone, atol=1e-6)
    assert torch.allclose(mixed[:1], alone, atol=1e-6)


def test_paper_literal_angle_cosine_is_taken_at_the_middle_atom() -> None:
    pos = torch.tensor(
        [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [1.0, 1.0, 0.0]],
        dtype=torch.float32,
    )
    # Edge 0 is k -> j (0 -> 1); edge 1 is j -> i (1 -> 2).
    edge_index = torch.tensor([[0, 1], [1, 2]], dtype=torch.long)
    distances = torch.linalg.vector_norm(
        pos[edge_index[1]] - pos[edge_index[0]], dim=-1
    )
    cosines = DimeNet2020._triplet_angle_cosines(
        pos,
        edge_index,
        distances,
        torch.tensor([0], dtype=torch.long),
        torch.tensor([1], dtype=torch.long),
    )

    assert torch.allclose(cosines, torch.tensor([0.0]), atol=1e-6)


def test_collinear_triplets_keep_second_coordinate_derivatives_finite() -> None:
    batch = _batch(
        _raw_sample(
            torch.tensor(
                [[0.0, 0.0, 0.0], [1.1, 0.0, 0.0], [2.4, 0.0, 0.0]],
                dtype=torch.float32,
            ),
            torch.tensor([6, 7, 8]),
            sample_id=5,
        )
    )
    batch.pos = batch.pos.detach().clone().requires_grad_(True)
    output = _model()(batch)
    first = torch.autograd.grad(output.sum(), batch.pos, create_graph=True)[0]
    second = torch.autograd.grad(first.square().sum(), batch.pos)[0]

    assert torch.isfinite(first).all()
    assert torch.isfinite(second).all()


def test_model_rejects_malformed_triplets_and_supports_a_single_atom() -> None:
    model = _model()
    valid = _batch(_triangle())
    invalid = valid.clone()
    invalid.dimenet_triplet_edge_index = valid.dimenet_triplet_edge_index.clone()
    invalid.dimenet_triplet_edge_index[:, 0] = torch.tensor([0, 0])
    with pytest.raises(ValueError, match="must encode k -> j -> i paths"):
        model(invalid)

    single = _raw_sample(
        torch.zeros((1, 3), dtype=torch.float32),
        torch.tensor([6]),
        sample_id=9,
    )
    prediction = model(_batch(single))
    assert prediction.shape == (1, 2)
    assert torch.isfinite(prediction).all()


def test_zero_output_initialization_preserves_the_paper_default() -> None:
    prediction = _model(output_initializer="zeros")(_batch(_triangle()))

    assert torch.equal(prediction, torch.zeros_like(prediction))
