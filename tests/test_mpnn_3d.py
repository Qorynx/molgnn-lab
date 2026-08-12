"""Regression checks for the explicit coordinate-backed MPNN variant."""

import pytest
import torch
from torch_geometric.data import Batch
from torch_geometric.data.data import BaseData

from molgnn.data import MolecularData
from molgnn.featurizer import featurize_smiles
from molgnn.models.mpnn_2017 import MPNNDistanceBins3D
from molgnn.transforms import TransformError, add_mpnn_3d_distance_bins_inputs


def _three_dimensional_sample(sample_id: int, *, terminal_x: float = 3.5):
    data = featurize_smiles(
        "CCO",
        targets=[0.0],
        target_mask=[True],
        sample_id=sample_id,
    )
    data.pos = torch.tensor(
        [[0.0, 0.0, 0.0], [1.5, 0.0, 0.0], [terminal_x, 0.0, 0.0]],
        dtype=torch.float32,
    )
    return add_mpnn_3d_distance_bins_inputs(data)


def test_transform_builds_all_pairs_with_bond_precedence() -> None:
    data = _three_dimensional_sample(0)

    assert data.mpnn_3d_edge_index.tolist() == [[0, 0, 1, 1, 2, 2], [1, 2, 0, 2, 0, 1]]
    # C-C and C-O retain bond type 0; the C...O pair at 3.5 Angstrom is bin 3 + 4.
    assert data.mpnn_3d_edge_type.tolist() == [0, 7, 0, 0, 7, 0]


def test_transform_requires_finite_three_dimensional_coordinates() -> None:
    data = featurize_smiles(
        "CC", targets=[0.0], target_mask=[True], sample_id=0
    )
    with pytest.raises(TransformError, match="requires pos"):
        add_mpnn_3d_distance_bins_inputs(data)

    data.pos = torch.tensor([[0.0, 0.0], [1.0, 0.0]], dtype=torch.float32)
    with pytest.raises(TransformError, match=r"shape \[N, 3\]"):
        add_mpnn_3d_distance_bins_inputs(data)


def test_transform_accepts_the_builtin_coordinate_source_bond_profile() -> None:
    data = MolecularData(
        x=torch.zeros((3, 44), dtype=torch.float32),
        edge_index=torch.tensor([[0, 1], [1, 0]], dtype=torch.long),
        # The bundled PDBBind source uses the four bond types plus in-ring.
        edge_attr=torch.tensor(
            [[1, 0, 0, 0, 1], [1, 0, 0, 0, 1]], dtype=torch.float32
        ),
        y=torch.zeros((1, 1), dtype=torch.float32),
        y_mask=torch.ones((1, 1), dtype=torch.bool),
        sample_id=torch.tensor([0], dtype=torch.long),
        pos=torch.tensor(
            [[0.0, 0.0, 0.0], [1.5, 0.0, 0.0], [4.5, 0.0, 0.0]],
            dtype=torch.float32,
        ),
    )

    transformed = add_mpnn_3d_distance_bins_inputs(data)
    batch = Batch.from_data_list(list[BaseData]([transformed, transformed.clone()]))
    model = MPNNDistanceBins3D(
        atom_dim=44,
        hidden_dim=48,
        num_message_passing_steps=1,
        readout_hidden_dim=4,
        readout_num_hidden_layers=1,
    ).eval()

    assert transformed.mpnn_3d_edge_type.tolist() == [0, 9, 0, 6, 9, 6]
    assert model(batch).shape == (2, 1)


def test_model_batches_distance_bin_graphs_and_rejects_cross_graph_edges() -> None:
    batch = Batch.from_data_list(
        list[BaseData]([
            _three_dimensional_sample(0),
            _three_dimensional_sample(1, terminal_x=4.0),
        ])
    )
    model = MPNNDistanceBins3D(
        atom_dim=153,
        hidden_dim=160,
        num_message_passing_steps=1,
        readout_hidden_dim=4,
        readout_num_hidden_layers=1,
    ).eval()

    output = model(batch)
    assert output.shape == (2, 1)
    assert torch.isfinite(output).all()
    assert batch.mpnn_3d_edge_index.max().item() == batch.x.shape[0] - 1

    invalid = batch.clone()
    invalid.mpnn_3d_edge_index = batch.mpnn_3d_edge_index.clone()
    invalid.mpnn_3d_edge_index[1, 0] = 3
    with pytest.raises(
        ValueError,
        match=r"mpnn_3d_edge_index must not connect different graphs",
    ):
        model(invalid)
