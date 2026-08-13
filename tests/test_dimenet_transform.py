"""Invariant checks for DimeNet's coordinate topology boundary."""

from __future__ import annotations

import pytest
import torch
from torch_geometric.data import Batch

from molgnn.data import MolecularData
from molgnn.models.dimenet_2020.data import DimeNetData
from molgnn.transforms import TransformError, add_dimenet_inputs


def _sample(sample_id: int = 0) -> MolecularData:
    return MolecularData(
        x=torch.zeros((3, 2), dtype=torch.float32),
        edge_index=torch.empty((2, 0), dtype=torch.long),
        edge_attr=torch.empty((0, 1), dtype=torch.float32),
        y=torch.zeros((1, 1), dtype=torch.float32),
        y_mask=torch.ones((1, 1), dtype=torch.bool),
        sample_id=torch.tensor([sample_id], dtype=torch.long),
        atomic_number=torch.tensor([6, 7, 8], dtype=torch.long),
        pos=torch.tensor(
            [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [3.0, 0.0, 0.0]],
            dtype=torch.float32,
        ),
    )


def test_transform_builds_deterministic_radius_edges_and_nonbacktracking_triplets() -> None:
    transformed = add_dimenet_inputs(_sample())

    assert transformed.dimenet_edge_index.tolist() == [
        [0, 0, 1, 1, 2, 2],
        [1, 2, 0, 2, 0, 1],
    ]
    assert torch.equal(transformed.dimenet_edge_index, add_dimenet_inputs(_sample()).dimenet_edge_index)

    edge_index = transformed.dimenet_edge_index
    triplet_index = transformed.dimenet_triplet_edge_index
    assert triplet_index.shape == (2, 6)
    incoming, outgoing = triplet_index
    assert torch.equal(edge_index[1, incoming], edge_index[0, outgoing])
    assert not bool((edge_index[0, incoming] == edge_index[1, outgoing]).any())


def test_transform_uses_an_inclusive_fixed_cutoff() -> None:
    data = _sample()
    data.pos[:, 0] = torch.tensor([0.0, 5.0, 11.0])

    transformed = add_dimenet_inputs(data)

    assert transformed.dimenet_edge_index.tolist() == [[0, 1], [1, 0]]


def test_transform_rejects_invalid_or_coincident_coordinate_inputs() -> None:
    data = _sample()
    del data.pos
    with pytest.raises(TransformError, match="requires pos"):
        add_dimenet_inputs(data)

    data = _sample()
    data.pos = data.pos.double()
    with pytest.raises(TransformError, match=r"shape \[N, 3\] finite float32"):
        add_dimenet_inputs(data)

    data = _sample()
    data.pos[2] = data.pos[0]
    with pytest.raises(TransformError, match="coincident atoms"):
        add_dimenet_inputs(data)

    data = _sample()
    data.batch = torch.zeros(3, dtype=torch.long)
    with pytest.raises(TransformError, match="already batched"):
        add_dimenet_inputs(data)


def test_molecular_data_batches_triplet_edge_ids_by_radius_edge_count() -> None:
    first = add_dimenet_inputs(_sample(0))
    second = add_dimenet_inputs(_sample(1))

    batch = Batch.from_data_list([first, second])
    first_edges = first.dimenet_edge_index.shape[1]
    first_triplets = first.dimenet_triplet_edge_index.shape[1]

    assert torch.equal(
        batch.dimenet_edge_index[:, first_edges:],
        second.dimenet_edge_index + first.x.shape[0],
    )
    assert torch.equal(
        batch.dimenet_triplet_edge_index[:, first_triplets:],
        second.dimenet_triplet_edge_index + first_edges,
    )


def test_standalone_dimenet_data_offsets_triplet_edge_ids_by_edge_count() -> None:
    first = DimeNetData(
        x=torch.zeros((3, 1), dtype=torch.float32),
        dimenet_edge_index=torch.tensor(
            [[0, 1, 1, 2], [1, 0, 2, 1]], dtype=torch.long
        ),
        dimenet_triplet_edge_index=torch.tensor([[0, 3], [2, 1]], dtype=torch.long),
    )
    second = first.clone()

    batch = Batch.from_data_list([first, second])

    assert torch.equal(
        batch.dimenet_triplet_edge_index[:, 2:],
        second.dimenet_triplet_edge_index + first.dimenet_edge_index.shape[1],
    )
