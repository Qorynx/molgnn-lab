"""Structural input contracts shared by the molecular model forwards."""

from collections.abc import Callable

import pytest
import torch
from torch_geometric.data import Batch
from torch_geometric.data.data import BaseData

from molgnn.data import MolecularData
from molgnn.featurizer import featurize_smiles
from molgnn.models.attentivefp_2020 import AttentiveFP
from molgnn.models.contracts import validate_batched_molecular_graph
from molgnn.models.dmpnn_2024 import DMPNN
from molgnn.models.gcn_baseline import GCNBaseline
from molgnn.models.hignn_2023 import HiGNN
from molgnn.models.himnet_2026 import HimNet
from molgnn.models.molecular_graph_embedding_2017 import MolecularGraphEmbedding
from molgnn.models.mpnn_2017 import MPNN
from molgnn.models.potentialnet_2018 import PotentialNet
from molgnn.models.trimnet_2020 import TrimNet2020
from molgnn.transforms import (
    add_brics_fragments,
    add_coley_2017_features,
    add_himnet_inputs,
    add_mpnn_edge_types,
    add_potentialnet_inputs,
    add_reverse_edge_index,
)


def _canonical_batch(*smiles: str) -> Batch:
    samples = [
        featurize_smiles(value, targets=[0.0], target_mask=[True], sample_id=index)
        for index, value in enumerate(smiles)
    ]
    return Batch.from_data_list(list[BaseData](samples))


def _potentialnet_batch() -> Batch:
    samples = []
    for sample_id in range(2):
        sample = MolecularData(
            x=torch.zeros((3, 44), dtype=torch.float32),
            edge_index=torch.tensor([[0, 1], [1, 0]], dtype=torch.long),
            edge_attr=torch.tensor(
                [[1, 0, 0, 0, 1], [1, 0, 0, 0, 1]], dtype=torch.float32
            ),
            y=torch.zeros((1, 1), dtype=torch.float32),
            y_mask=torch.ones((1, 1), dtype=torch.bool),
            sample_id=torch.tensor([sample_id], dtype=torch.long),
            pos=torch.tensor(
                [[0.0, 0.0, 0.0], [1.5, 0.0, 0.0], [4.5, 0.0, 0.0]],
                dtype=torch.float32,
            ),
            ligand_mask=torch.tensor([True, True, False]),
        )
        samples.append(add_potentialnet_inputs(sample))
    return Batch.from_data_list(list[BaseData](samples))


def test_shared_contract_accepts_paired_loop_free_disjoint_graphs() -> None:
    num_graphs = validate_batched_molecular_graph(
        torch.tensor([[0, 1, 2, 3], [1, 0, 3, 2]]),
        torch.tensor([0, 0, 1, 1]),
        num_nodes=4,
        device=torch.device("cpu"),
    )

    assert num_graphs == 2


@pytest.mark.parametrize(
    ("edge_index", "graph_batch", "num_nodes", "message"),
    [
        (
            torch.tensor([[0, 1, 2, 3], [2, 0, 3, 2]]),
            torch.tensor([0, 0, 1, 1]),
            4,
            "different graphs",
        ),
        (
            torch.empty((2, 0), dtype=torch.long),
            torch.tensor([1]),
            1,
            "contiguous",
        ),
    ],
)
def test_shared_contract_rejects_structurally_invalid_batches(
    edge_index: torch.Tensor,
    graph_batch: torch.Tensor,
    num_nodes: int,
    message: str,
) -> None:
    with pytest.raises(ValueError, match=message):
        validate_batched_molecular_graph(
            edge_index,
            graph_batch,
            num_nodes=num_nodes,
            device=torch.device("cpu"),
        )


def test_shared_contract_can_reject_supplied_self_loops_when_a_model_requires_it() -> (
    None
):
    with pytest.raises(ValueError, match="self-loops"):
        validate_batched_molecular_graph(
            torch.tensor([[0], [0]]),
            torch.tensor([0]),
            num_nodes=1,
            device=torch.device("cpu"),
            forbid_self_loops=True,
        )


def test_gcn_keeps_pyg_directed_and_self_loop_support() -> None:
    model = GCNBaseline(153, hidden_dim=4).eval()
    batch = _canonical_batch("CC")
    batch.edge_index = torch.tensor([[0, 0], [1, 0]], dtype=torch.long)

    assert model(batch).shape == (1, 1)


def _model_cases() -> tuple[tuple[str, Callable[[], object], Callable[[], Batch]], ...]:
    return (
        (
            "gcn",
            lambda: GCNBaseline(153, hidden_dim=4),
            lambda: _canonical_batch("CC", "CC"),
        ),
        (
            "attentivefp",
            lambda: AttentiveFP(153, 14, hidden_dim=4, dropout=0),
            lambda: _canonical_batch("CC", "CC"),
        ),
        (
            "dmpnn",
            lambda: DMPNN(153, 14, hidden_dim=4, depth=2),
            lambda: Batch.from_data_list(
                list[BaseData](
                    [
                        add_reverse_edge_index(sample)
                        for sample in _canonical_batch("CC", "CC").to_data_list()
                    ]
                )
            ),
        ),
        (
            "hignn",
            lambda: HiGNN(
                153,
                14,
                hidden_dim=4,
                num_layers=1,
                num_slices=2,
                feature_reduction=2,
                dropout=0,
            ),
            lambda: Batch.from_data_list(
                list[BaseData](
                    [
                        add_brics_fragments(sample)
                        for sample in _canonical_batch("CC", "CC").to_data_list()
                    ]
                )
            ),
        ),
        (
            "mge",
            lambda: MolecularGraphEmbedding(
                depth=1,
                message_dim=4,
                fingerprint_dim=8,
                predictor_hidden_dim=4,
            ),
            lambda: Batch.from_data_list(
                list[BaseData](
                    [
                        add_coley_2017_features(sample)
                        for sample in _canonical_batch("CC", "CC").to_data_list()
                    ]
                )
            ),
        ),
        (
            "mpnn",
            lambda: MPNN(
                153,
                hidden_dim=160,
                num_message_passing_steps=1,
                readout_hidden_dim=4,
                readout_num_hidden_layers=1,
            ),
            lambda: Batch.from_data_list(
                list[BaseData](
                    [
                        add_mpnn_edge_types(sample)
                        for sample in _canonical_batch("CC", "CC").to_data_list()
                    ]
                )
            ),
        ),
        (
            "trimnet",
            lambda: TrimNet2020(
                153,
                14,
                hidden_dim=4,
                depth=1,
                heads=2,
                num_timesteps=1,
                dropout=0,
            ),
            lambda: _canonical_batch("CC", "CC"),
        ),
    )


@pytest.mark.parametrize(("_name", "make_model", "make_batch"), _model_cases())
def test_each_model_rejects_cross_graph_messages(
    _name: str,
    make_model: Callable[[], object],
    make_batch: Callable[[], Batch],
) -> None:
    model = make_model()
    batch = make_batch()
    batch.edge_index = batch.edge_index.clone()
    batch.edge_index[1, 0] = 2

    with pytest.raises(ValueError, match="different graphs"):
        model(batch)  # type: ignore[operator]


def test_hignn_also_rejects_cross_graph_brics_edges() -> None:
    model = HiGNN(
        153,
        14,
        hidden_dim=4,
        num_layers=1,
        num_slices=2,
        feature_reduction=2,
        dropout=0,
    )
    samples = _canonical_batch("CC", "CC").to_data_list()
    batch = Batch.from_data_list(
        list[BaseData]([add_brics_fragments(sample) for sample in samples])
    )
    batch.brics_edge_index = batch.brics_edge_index.clone()
    batch.brics_edge_index[1, 0] = 2

    with pytest.raises(
        ValueError, match=r"brics_edge_index must not connect different graphs"
    ):
        model(batch)


def test_himnet_accepts_its_prepared_contract_and_rejects_cross_graph_edges() -> None:
    samples = [
        add_himnet_inputs(sample)
        for sample in _canonical_batch("CC", "CC").to_data_list()
    ]
    batch = Batch.from_data_list(list[BaseData](samples))
    model = HimNet(
        153,
        14,
        hidden_dim=8,
        depth=1,
        interaction_heads=2,
        fusion_heads=2,
        dropout=0.0,
    )
    assert model(batch).shape == (2, 1)

    first_nodes = samples[0].himnet_x.shape[0]
    batch.himnet_edge_index = batch.himnet_edge_index.clone()
    batch.himnet_edge_index[1, 0] = first_nodes
    with pytest.raises(ValueError, match=r"himnet_edge_index must not connect different graphs"):
        model(batch)


def test_potentialnet_rejects_cross_graph_spatial_edges() -> None:
    batch = _potentialnet_batch()
    model = PotentialNet(
        atom_dim=44,
        bond_hidden_dim=48,
        spatial_hidden_dim=48,
        gather_dim=48,
        num_bond_steps=1,
        num_spatial_steps=1,
        readout_hidden_dims=(8,),
    )
    assert model(batch).shape == (2, 1)

    invalid = batch.clone()
    invalid.potentialnet_stage2_edge_index = (
        batch.potentialnet_stage2_edge_index.clone()
    )
    invalid.potentialnet_stage2_edge_index[1, 0] = 3
    with pytest.raises(
        ValueError,
        match=r"potentialnet_stage2_edge_index must not connect different graphs",
    ):
        model(invalid)
