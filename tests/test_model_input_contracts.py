"""Structural input contracts shared by the molecular model forwards."""

from collections.abc import Callable

import pytest
import torch
from torch_geometric.data import Batch
from torch_geometric.data.data import BaseData

from molgnn.data import MolecularData
from molgnn.featurizer import featurize_smiles
from molgnn.models.ampnn_emnn_2020 import AMPNN, EMNN
from molgnn.models.attentivefp_2020 import AttentiveFP
from molgnn.models.chemrl_gem_2022 import ChemRLGEM, ChemRLGEMPretrainer
from molgnn.models.chemrl_gem_2022.pretraining import build_geometry_pretraining_targets
from molgnn.models.contracts import validate_batched_molecular_graph
from molgnn.models.dimenet_2020 import DimeNet2020
from molgnn.models.dmpnn_2024 import DMPNN
from molgnn.models.gcn_baseline import GCNBaseline
from molgnn.models.graphmvp_2022 import GraphMVP
from molgnn.models.hignn_2023 import HiGNN
from molgnn.models.himnet_2026 import HimNet
from molgnn.models.molecular_graph_embedding_2017 import MolecularGraphEmbedding
from molgnn.models.mpnn_2017 import MPNN
from molgnn.models.potentialnet_2018 import PotentialNet
from molgnn.models.resgat_2024 import ResGAT
from molgnn.models.trimnet_2020 import TrimNet2020
from molgnn.models.weave_2016 import Weave
from molgnn.transforms import (
    add_ampnn_edge_types,
    add_chemrl_gem_inputs,
    add_brics_fragments,
    add_coley_2017_features,
    add_dimenet_inputs,
    add_graphmvp_inputs,
    add_himnet_inputs,
    add_mpnn_edge_types,
    add_potentialnet_inputs,
    add_reverse_edge_index,
    add_weave_inputs,
)


def _canonical_batch(*smiles: str) -> Batch:
    samples = [
        featurize_smiles(value, targets=[0.0], target_mask=[True], sample_id=index)
        for index, value in enumerate(smiles)
    ]
    return Batch.from_data_list(list[BaseData](samples))


def test_graphmvp_profiles_train_and_batch_without_geometry() -> None:
    samples = [
        add_graphmvp_inputs(
            featurize_smiles(value, targets=[0.0], target_mask=[True], sample_id=index)
        )
        for index, value in enumerate(("CCO", "c1ccccc1"))
    ]
    batch = Batch.from_data_list(list[BaseData](samples))
    for profile in ("simple", "ogb_full"):
        model = GraphMVP(
            atom_dim=153,
            bond_dim=14,
            num_targets=1,
            feature_profile=profile,
            hidden_dim=8,
            num_layers=2,
            dropout=0.0,
        )
        prediction = model(batch)
        assert prediction.shape == (2, 1)
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
        optimizer.zero_grad(set_to_none=True)
        prediction.sum().backward()
        optimizer.step()


def test_chemrl_gem_builds_line_graph_and_trains_on_batched_geometry() -> None:
    samples = [
        add_chemrl_gem_inputs(
            featurize_smiles(value, targets=[0.0], target_mask=[True], sample_id=index)
        )
        for index, value in enumerate(("CCO", "c1ccccc1", "C"))
    ]
    batch = Batch.from_data_list(list[BaseData](samples))
    model = ChemRLGEM(
        atom_dim=153,
        bond_dim=14,
        num_targets=2,
        hidden_dim=8,
        num_layers=2,
        dropout=0.0,
    )
    prediction = model(batch)
    assert prediction.shape == (3, 2)
    assert batch.chemrl_gem_angle_edge_index.numel() == 2 * sum(
        sample.chemrl_gem_angle_edge_index.shape[1] for sample in samples
    )
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    optimizer.zero_grad(set_to_none=True)
    prediction.square().mean().backward()
    optimizer.step()


def test_chemrl_gem_pretraining_heads_take_one_step() -> None:
    samples = [
        add_chemrl_gem_inputs(
            featurize_smiles(value, targets=[0.0], target_mask=[True], sample_id=index)
        )
        for index, value in enumerate(("CCO", "C"))
    ]
    batch = Batch.from_data_list(list[BaseData](samples))
    targets = build_geometry_pretraining_targets(batch)
    targets["Cm_node_i"] = torch.arange(batch.num_nodes)
    targets["Cm_context_id"] = torch.zeros(batch.num_nodes, dtype=torch.long)
    targets["Fg_label"] = torch.zeros((batch.num_graphs, 494), dtype=torch.float32)
    model = ChemRLGEMPretrainer(
        embed_dim=8,
        layer_num=2,
        hidden_size=8,
        dropout=0.0,
    )
    losses = model(batch, targets)
    assert {"Cm_loss", "Fg_loss", "Bar_loss", "Blr_loss", "Adc_loss", "loss"} <= set(losses)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    optimizer.zero_grad(set_to_none=True)
    losses["loss"].backward()
    optimizer.step()


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


def _dimenet_batch() -> Batch:
    samples = []
    for sample_id, shift in enumerate((0.0, 8.0)):
        samples.append(
            add_dimenet_inputs(
                MolecularData(
                    x=torch.zeros((3, 1), dtype=torch.float32),
                    edge_index=torch.empty((2, 0), dtype=torch.long),
                    edge_attr=torch.empty((0, 1), dtype=torch.float32),
                    y=torch.zeros((1, 1), dtype=torch.float32),
                    y_mask=torch.ones((1, 1), dtype=torch.bool),
                    sample_id=torch.tensor([sample_id], dtype=torch.long),
                    atomic_number=torch.tensor([6, 7, 8], dtype=torch.long),
                    pos=torch.tensor(
                        [[shift, 0.0, 0.0], [shift + 1.0, 0.0, 0.0], [shift, 1.0, 0.0]],
                        dtype=torch.float32,
                    ),
                )
            )
        )
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


def test_resgat_enforces_its_fc_width_contract() -> None:
    with pytest.raises(
        ValueError, match="final embed_sizes entry must be at least 2"
    ):
        ResGAT(153, 14, 1, hidden_dim=1, num_blocks=(1,))


def test_resgat_applies_relu_after_both_hidden_fc_layers() -> None:
    model = ResGAT(
        153,
        14,
        1,
        hidden_dim=4,
        num_blocks=(1,),
        embed_sizes=(4,),
    ).eval()
    with torch.no_grad():
        model.task_heads[0].fc2.weight.zero_()
        model.task_heads[0].fc2.bias.fill_(-1.0)

    head_inputs: list[torch.Tensor] = []
    handle = model.task_heads[0].out.register_forward_pre_hook(
        lambda _module, inputs: head_inputs.append(inputs[0].detach().clone())
    )
    try:
        output = model(_canonical_batch("CC", "CC"))
    finally:
        handle.remove()

    assert output.shape == (2, 1)
    assert len(head_inputs) == 1
    assert torch.equal(head_inputs[0], torch.zeros_like(head_inputs[0]))


def test_resgat_uses_independent_single_head_predictors_for_each_task() -> None:
    model = ResGAT(
        153,
        14,
        2,
        hidden_dim=4,
        num_blocks=(1,),
        embed_sizes=(4,),
    ).eval()

    assert model.block_sets[0][0].conv1.heads == 1
    assert model.block_sets[0][0].conv2.heads == 1
    assert model.task_heads[0].fc1.weight.data_ptr() != (
        model.task_heads[1].fc1.weight.data_ptr()
    )
    assert model(_canonical_batch("CC", "CC")).shape == (2, 2)


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
            "ampnn",
            lambda: AMPNN(
                153,
                message_dim=4,
                num_message_passing_steps=1,
                message_hidden_dims=(4,),
                attention_hidden_dims=(4,),
                gather_dim=4,
                gather_gate_hidden_dims=(4,),
                gather_value_hidden_dims=(4,),
                predictor_hidden_dims=(4,),
                dropout=0.0,
            ),
            lambda: Batch.from_data_list(
                list[BaseData](
                    [
                        add_ampnn_edge_types(sample)
                        for sample in _canonical_batch("CC", "CC").to_data_list()
                    ]
                )
            ),
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
            "emnn",
            lambda: EMNN(
                153,
                14,
                edge_hidden_dim=4,
                num_message_passing_steps=1,
                edge_embedding_hidden_dims=(4,),
                message_hidden_dims=(4,),
                attention_hidden_dims=(4,),
                gather_dim=4,
                gather_gate_hidden_dims=(4,),
                gather_value_hidden_dims=(4,),
                predictor_hidden_dims=(4,),
                dropout=0.0,
            ),
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


def test_dimenet_rejects_cross_graph_radius_edges() -> None:
    batch = _dimenet_batch()
    model = DimeNet2020(
        hidden_dim=4,
        num_blocks=1,
        num_bilinear=2,
        num_spherical=2,
        num_radial=2,
        num_before_skip=1,
        num_after_skip=1,
        num_dense_output=1,
    )
    assert model(batch).shape == (2, 1)

    invalid = batch.clone()
    invalid.dimenet_edge_index = batch.dimenet_edge_index.clone()
    invalid.dimenet_edge_index[1, 0] = 3
    with pytest.raises(
        ValueError, match=r"dimenet_edge_index must not connect different graphs"
    ):
        model(invalid)


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
    with pytest.raises(
        ValueError, match=r"himnet_edge_index must not connect different graphs"
    ):
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


def test_weave_rejects_cross_graph_ordered_pairs() -> None:
    samples = [
        add_weave_inputs(sample)
        for sample in _canonical_batch("CC", "CC").to_data_list()
    ]
    batch = Batch.from_data_list(list[BaseData](samples))
    model = Weave(
        atom_dim=153,
        hidden_dim=8,
        num_weave_modules=1,
        graph_feature_dim=8,
        predictor_hidden_dims=(8,),
        dropout=0.0,
    )
    assert model(batch).shape == (2, 1)

    invalid = batch.clone()
    invalid.weave_pair_index = batch.weave_pair_index.clone()
    invalid.weave_pair_index[1, 0] = 2
    with pytest.raises(
        ValueError, match=r"weave_pair_index must not connect different graphs"
    ):
        model(invalid)
