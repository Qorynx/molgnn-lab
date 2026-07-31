"""Structural input contracts shared by the molecular model forwards."""

from collections.abc import Callable

import pytest
import torch
from torch_geometric.data import Batch
from torch_geometric.data.data import BaseData

from molgnn.featurizer import featurize_smiles
from molgnn.models.attentivefp_2020 import AttentiveFP
from molgnn.models.contracts import validate_batched_molecular_graph
from molgnn.models.dmpnn_2024 import DMPNN
from molgnn.models.gcn_baseline import GCNBaseline
from molgnn.models.hignn_2023 import HiGNN
from molgnn.models.molecular_graph_embedding_2017 import MolecularGraphEmbedding
from molgnn.models.trimnet_2020 import TrimNet2020
from molgnn.transforms import (
    add_brics_fragments,
    add_coley_2017_features,
    add_reverse_edge_index,
)


def _canonical_batch(*smiles: str) -> Batch:
    samples = [
        featurize_smiles(value, targets=[0.0], target_mask=[True], sample_id=index)
        for index, value in enumerate(smiles)
    ]
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
