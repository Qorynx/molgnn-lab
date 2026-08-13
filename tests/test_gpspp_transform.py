"""Invariant checks for GPS++'s topological all-pairs input boundary."""

from __future__ import annotations

import pytest
import torch
from torch_geometric.data import Batch

from molgnn.data import MolecularData
from molgnn.transforms import TransformError, add_gpspp_inputs


def _sample(
    num_nodes: int,
    undirected_edges: tuple[tuple[int, int], ...],
    *,
    sample_id: int = 0,
) -> MolecularData:
    directed_edges: list[tuple[int, int]] = []
    attrs: list[list[float]] = []
    for edge_id, (source, target) in enumerate(undirected_edges):
        feature = [float(edge_id + 1), float(edge_id % 2)]
        directed_edges.extend(((source, target), (target, source)))
        attrs.extend((feature, feature))

    if directed_edges:
        edge_index = torch.tensor(directed_edges, dtype=torch.long).t().contiguous()
        edge_attr = torch.tensor(attrs, dtype=torch.float32)
    else:
        edge_index = torch.empty((2, 0), dtype=torch.long)
        edge_attr = torch.empty((0, 2), dtype=torch.float32)
    return MolecularData(
        x=torch.arange(num_nodes * 3, dtype=torch.float32).reshape(num_nodes, 3),
        edge_index=edge_index,
        edge_attr=edge_attr,
        y=torch.zeros((1, 1), dtype=torch.float32),
        y_mask=torch.ones((1, 1), dtype=torch.bool),
        sample_id=torch.tensor([sample_id], dtype=torch.long),
    )


def test_transform_builds_source_major_chain_and_disconnected_shortest_paths() -> None:
    transformed = add_gpspp_inputs(_sample(4, ((0, 1), (1, 2))))

    assert transformed.gpspp_pair_index.tolist() == [
        [0, 0, 0, 0, 1, 1, 1, 1, 2, 2, 2, 2, 3, 3, 3, 3],
        [0, 1, 2, 3, 0, 1, 2, 3, 0, 1, 2, 3, 0, 1, 2, 3],
    ]
    assert transformed.gpspp_spd.reshape(4, 4).tolist() == [
        [0, 1, 2, -1],
        [1, 0, 1, -1],
        [2, 1, 0, -1],
        [-1, -1, -1, 0],
    ]


def test_transform_uses_the_shorter_route_in_a_cycle() -> None:
    transformed = add_gpspp_inputs(
        _sample(4, ((0, 1), (1, 2), (2, 3), (3, 0)))
    )

    assert transformed.gpspp_spd.reshape(4, 4).tolist() == [
        [0, 1, 2, 1],
        [1, 0, 1, 2],
        [2, 1, 0, 1],
        [1, 2, 1, 0],
    ]


def test_transform_clones_input_and_rejects_batched_samples() -> None:
    data = _sample(2, ((0, 1),))
    original_edge_index = data.edge_index.clone()
    original_edge_attr = data.edge_attr.clone()

    transformed = add_gpspp_inputs(data)

    assert transformed is not data
    assert not hasattr(data, "gpspp_pair_index")
    assert not hasattr(data, "gpspp_spd")
    assert torch.equal(data.edge_index, original_edge_index)
    assert torch.equal(data.edge_attr, original_edge_attr)

    data.batch = torch.zeros(data.x.shape[0], dtype=torch.long)
    with pytest.raises(TransformError, match="already batched"):
        add_gpspp_inputs(data)


def test_transform_rejects_invalid_directed_covalent_contract() -> None:
    missing_reverse = _sample(2, ((0, 1),))
    missing_reverse.edge_index = missing_reverse.edge_index[:, :1]
    missing_reverse.edge_attr = missing_reverse.edge_attr[:1]
    with pytest.raises(TransformError, match="reciprocal"):
        add_gpspp_inputs(missing_reverse)

    self_loop = _sample(2, ((0, 1),))
    self_loop.edge_index[:, 0] = torch.tensor([0, 0], dtype=torch.long)
    with pytest.raises(TransformError, match="self-loops"):
        add_gpspp_inputs(self_loop)

    duplicate = _sample(2, ((0, 1),))
    duplicate.edge_index = torch.cat((duplicate.edge_index, duplicate.edge_index[:, :1]), dim=1)
    duplicate.edge_attr = torch.cat((duplicate.edge_attr, duplicate.edge_attr[:1]), dim=0)
    with pytest.raises(TransformError, match="duplicate"):
        add_gpspp_inputs(duplicate)

    mismatched_reverse = _sample(2, ((0, 1),))
    mismatched_reverse.edge_attr[1, 0] = 99.0
    with pytest.raises(TransformError, match="matching edge_attr"):
        add_gpspp_inputs(mismatched_reverse)


def test_pyg_batch_offsets_pair_indices_and_preserves_graph_isolation() -> None:
    first = add_gpspp_inputs(_sample(2, ((0, 1),), sample_id=0))
    second = add_gpspp_inputs(_sample(3, ((0, 1), (1, 2)), sample_id=1))

    batch = Batch.from_data_list([first, second])
    first_pair_count = first.gpspp_pair_index.shape[1]

    assert torch.equal(
        batch.gpspp_pair_index[:, first_pair_count:],
        second.gpspp_pair_index + first.x.shape[0],
    )
    assert torch.equal(
        batch.gpspp_spd,
        torch.cat((first.gpspp_spd, second.gpspp_spd)),
    )
    pair_graph_ids = batch.batch[batch.gpspp_pair_index]
    assert torch.equal(pair_graph_ids[0], pair_graph_ids[1])
