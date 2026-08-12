"""Invariant checks for the full staged PotentialNet runtime path."""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
from rdkit import Chem
from rdkit.Chem import AllChem
from torch_geometric.data import Batch

from molgnn.config import DataConfig, TaskConfig
from molgnn.cli import main
from molgnn.data import MolecularData
from molgnn.dataset_sources import load_dataset
from molgnn.datasets.pdbbind import PDBBindComplexDataset
from molgnn.featurizer import featurize_smiles
from molgnn.models.potentialnet_2018 import PotentialNet, TypedMessageMLP
from molgnn.models.registration import register_builtin_models
from molgnn.registry import get_model_spec
from molgnn.runner import _effective_graph_transform
from molgnn.transforms import add_potentialnet_inputs


def _sample(sample_id: int = 0) -> MolecularData:
    data = MolecularData(
        x=torch.arange(132, dtype=torch.float32).reshape(3, 44) / 100,
        edge_index=torch.tensor([[0, 1], [1, 0]], dtype=torch.long),
        # A ring bond deliberately has two active types per directed edge.
        edge_attr=torch.tensor([[1, 0, 0, 0, 1], [1, 0, 0, 0, 1]], dtype=torch.float32),
        y=torch.zeros((1, 2), dtype=torch.float32),
        y_mask=torch.ones((1, 2), dtype=torch.bool),
        sample_id=torch.tensor([sample_id], dtype=torch.long),
        pos=torch.tensor(
            [[0.0, 0.0, 0.0], [1.5, 0.0, 0.0], [4.5, 0.0, 0.0]],
            dtype=torch.float32,
        ),
        ligand_mask=torch.tensor([True, True, False]),
    )
    return add_potentialnet_inputs(data)


def _model(*, spatial_mode: str = "auto") -> PotentialNet:
    return PotentialNet(
        atom_dim=44,
        bond_hidden_dim=48,
        spatial_hidden_dim=48,
        gather_dim=48,
        num_bond_steps=1,
        num_spatial_steps=1,
        readout_hidden_dims=(16,),
        num_targets=2,
        dropout=0.0,
        spatial_mode=spatial_mode,
    ).eval()


def _two_dimensional_sample(sample_id: int = 0) -> MolecularData:
    """Build the normal CSV/SMILES path with no coordinates."""

    return add_potentialnet_inputs(
        featurize_smiles(
            "CCO",
            targets=[float(sample_id)],
            target_mask=[True],
            sample_id=sample_id,
        )
    )


def _two_dimensional_model(*, spatial_mode: str = "auto") -> PotentialNet:
    return PotentialNet(
        atom_dim=153,
        bond_hidden_dim=160,
        spatial_hidden_dim=160,
        gather_dim=160,
        num_bond_steps=1,
        num_spatial_steps=1,
        readout_hidden_dims=(16,),
        num_targets=1,
        dropout=0.0,
        spatial_mode=spatial_mode,
    ).eval()


def test_transform_retains_multi_relation_edges_and_dgl_bin_boundaries() -> None:
    data = _sample()

    assert data.potentialnet_bond_edge_index.shape == (2, 4)
    assert data.potentialnet_bond_edge_type.tolist() == [0, 4, 0, 4]
    # Directed spatial neighbours have the source profile's right-inclusive
    # bins: 1.5 Å -> 0 and 4.5 Å -> 3.
    assert set(data.potentialnet_stage2_edge_type.tolist()) >= {0, 3, 4, 8}
    assert not torch.any(
        data.potentialnet_stage2_edge_index[0] == data.potentialnet_stage2_edge_index[1]
    )
    assert data.potentialnet_use_spatial.tolist() == [True]


def test_smiles_transform_uses_the_bond_only_potentialnet_branch() -> None:
    batch = Batch.from_data_list(
        [_two_dimensional_sample(0), _two_dimensional_sample(1)]
    )
    auto_model = _two_dimensional_model()
    disabled_model = _two_dimensional_model(spatial_mode="disabled")
    disabled_model.load_state_dict(auto_model.state_dict())

    assert batch.ligand_mask.tolist() == [True] * batch.x.shape[0]
    assert batch.potentialnet_stage2_edge_index.shape == (2, 0)
    assert batch.potentialnet_stage2_edge_type.shape == (0,)
    assert batch.potentialnet_use_spatial.tolist() == [False, False]
    assert auto_model(batch).shape == (2, 1)
    assert torch.equal(auto_model(batch), disabled_model(batch))


def test_potentialnet_rejects_mixed_2d_and_3d_batches_and_mode_conflicts() -> None:
    three_dimensional = _sample()
    two_dimensional = three_dimensional.clone()
    del two_dimensional.pos
    two_dimensional = add_potentialnet_inputs(two_dimensional)
    mixed = Batch.from_data_list([two_dimensional, three_dimensional])

    with pytest.raises(ValueError, match="homogeneous"):
        _model()(mixed)
    with pytest.raises(
        ValueError, match="requires paired PotentialNet Stage 2 tensors"
    ):
        _model(spatial_mode="required")(Batch.from_data_list([two_dimensional]))
    with pytest.raises(ValueError, match="does not accept"):
        _model(spatial_mode="disabled")(Batch.from_data_list([three_dimensional]))


def test_auto_2d_accepts_absent_optional_fields_and_custom_preparation() -> None:
    prepared = _two_dimensional_sample()
    del prepared.potentialnet_stage2_edge_index
    del prepared.potentialnet_stage2_edge_type
    del prepared.potentialnet_use_spatial

    assert _two_dimensional_model()(Batch.from_data_list([prepared])).shape == (1, 1)

    register_builtin_models()
    plan = SimpleNamespace(
        spec=get_model_spec("potentialnet"), graph_transform=add_potentialnet_inputs
    )
    assert _effective_graph_transform(plan, [prepared]) is None


def test_optional_stage2_fields_must_be_paired_and_well_formed_when_present() -> None:
    batch = Batch.from_data_list([_two_dimensional_sample()])
    model = _two_dimensional_model()

    partial = batch.clone()
    del partial.potentialnet_stage2_edge_type
    with pytest.raises(ValueError, match="provided together"):
        model(partial)

    malformed = batch.clone()
    malformed.potentialnet_stage2_edge_index = torch.empty((2, 0))
    with pytest.raises(ValueError, match="dtype torch.long"):
        model(malformed)


def test_nonlinear_typed_messages_sum_parallel_active_relations() -> None:
    messages = TypedMessageMLP(num_edge_types=2, state_dim=1, hidden_dim=1).eval()
    with torch.no_grad():
        first_a, _, first_b = messages.networks[0]
        second_a, _, second_b = messages.networks[1]
        first_a.weight.fill_(1.0)
        first_a.bias.zero_()
        first_b.weight.fill_(2.0)
        first_b.bias.zero_()
        second_a.weight.fill_(1.0)
        second_a.bias.zero_()
        second_b.weight.fill_(3.0)
        second_b.bias.zero_()

    output = messages(
        torch.tensor([[2.0], [0.0]]),
        torch.tensor([[0, 0], [1, 1]], dtype=torch.long),
        torch.tensor([0, 1], dtype=torch.long),
    )

    assert torch.allclose(output, torch.tensor([[0.0], [10.0]]))


def test_model_has_two_distinct_tied_stages_and_is_edge_order_invariant() -> None:
    data = _sample()
    model = _model()

    assert len(model.stage1.message_network.networks) == 5
    assert len(model.stage2.message_network.networks) == 9
    assert model.stage1.gate.gate_network.in_features == 44 + 48
    assert model.stage2.gate.gate_network.in_features == 48 + 48

    original = model(Batch.from_data_list([data]))
    permuted = data.clone()
    bond_order = torch.arange(data.potentialnet_bond_edge_type.numel() - 1, -1, -1)
    stage2_order = torch.arange(data.potentialnet_stage2_edge_type.numel() - 1, -1, -1)
    permuted.potentialnet_bond_edge_index = data.potentialnet_bond_edge_index[
        :, bond_order
    ]
    permuted.potentialnet_bond_edge_type = data.potentialnet_bond_edge_type[bond_order]
    permuted.potentialnet_stage2_edge_index = data.potentialnet_stage2_edge_index[
        :, stage2_order
    ]
    permuted.potentialnet_stage2_edge_type = data.potentialnet_stage2_edge_type[
        stage2_order
    ]
    reordered = model(Batch.from_data_list([permuted]))

    assert original.shape == (1, 2)
    assert torch.allclose(original, reordered, atol=1e-6)


def test_model_rejects_cross_complex_spatial_edges_and_backpropagates() -> None:
    batch = Batch.from_data_list([_sample(0), _sample(1)])
    model = _model()
    output = model(batch)
    output.square().mean().backward()
    assert output.shape == (2, 2)
    assert any(parameter.grad is not None for parameter in model.parameters())

    first_node_count = 3
    invalid = batch.clone()
    invalid.potentialnet_stage2_edge_index = (
        batch.potentialnet_stage2_edge_index.clone()
    )
    invalid.potentialnet_stage2_edge_index[1, 0] = first_node_count
    with pytest.raises(ValueError, match="must not connect different graphs"):
        model(invalid)


def test_pdbbind_manifest_source_builds_a_3d_complex_and_loader_result(
    tmp_path: Path,
) -> None:
    ligand_path = tmp_path / "ligand.sdf"
    ligand = Chem.AddHs(Chem.MolFromSmiles("CO"))
    assert AllChem.EmbedMolecule(ligand, randomSeed=7) == 0
    Chem.MolToMolFile(ligand, str(ligand_path))
    protein_path = tmp_path / "pocket.pdb"
    protein_path.write_text(
        "ATOM      1  C   GLY A   1       4.000   0.000   0.000  1.00  0.00           C  \n"
        "ATOM      2  O   GLY A   1       5.200   0.000   0.000  1.00  0.00           O  \n"
        "TER\nEND\n",
        encoding="utf-8",
    )
    manifest = tmp_path / "complexes.csv"
    manifest.write_text(
        "complex_id,ligand_file,pocket_file,target,split\n"
        "toy,ligand.sdf,pocket.pdb,1.25,train\n",
        encoding="utf-8",
    )

    dataset = PDBBindComplexDataset(
        manifest,
        ligand_path_column="ligand_file",
        protein_path_column="pocket_file",
        target_columns=("target",),
        id_column="complex_id",
        split_column="split",
        strip_hydrogens=True,
    )
    sample = dataset[0]
    assert sample.x.shape[1] == 44
    assert sample.pos.shape == (sample.x.shape[0], 3)
    assert sample.ligand_mask.dtype == torch.bool
    assert sample.ligand_mask.any() and (~sample.ligand_mask).any()
    assert dataset.smiles[0]
    assert dataset.fingerprint()

    result = load_dataset(
        DataConfig(
            path=manifest,
            smiles_column="unused",
            target_columns=("target",),
            id_column="complex_id",
            split="predefined",
            split_ratios=(0.8, 0.1, 0.1),
            split_column="split",
            invalid_smiles="error",
            source="pdbbind_complex",
            ligand_path_column="ligand_file",
            protein_path_column="pocket_file",
            strip_hydrogens=True,
        ),
        TaskConfig(
            type="regression",
            loss="mse",
            metrics=("rmse",),
            target_scaling=True,
        ),
    )
    transformed = add_potentialnet_inputs(result.dataset[0])
    assert result.feature_schema.atom_dim == 44
    assert transformed.potentialnet_stage2_edge_type.numel()
    assert result.summary.artifact_fields["skipped_invalid_complexes"] == 0


def test_potentialnet_cli_runs_through_the_common_artifact_pipeline(
    tmp_path: Path,
) -> None:
    ligand_path = tmp_path / "ligand.sdf"
    ligand = Chem.AddHs(Chem.MolFromSmiles("CO"))
    assert AllChem.EmbedMolecule(ligand, randomSeed=11) == 0
    Chem.MolToMolFile(ligand, str(ligand_path))
    protein_path = tmp_path / "pocket.pdb"
    protein_path.write_text(
        "ATOM      1  C   GLY A   1       4.000   0.000   0.000  1.00  0.00           C  \n"
        "ATOM      2  O   GLY A   1       5.200   0.000   0.000  1.00  0.00           O  \n"
        "TER\nEND\n",
        encoding="utf-8",
    )
    manifest = tmp_path / "complexes.csv"
    manifest.write_text(
        "complex_id,ligand_file,pocket_file,target,split\n"
        "train,ligand.sdf,pocket.pdb,1.0,train\n"
        "validation,ligand.sdf,pocket.pdb,2.0,validation\n"
        "test,ligand.sdf,pocket.pdb,3.0,test\n",
        encoding="utf-8",
    )
    config = tmp_path / "potentialnet.yaml"
    config.write_text(
        "\n".join(
            [
                "experiment:",
                "  name: potentialnet_smoke",
                "  seed: 31",
                f"  output_dir: {tmp_path.as_posix()}",
                "data:",
                "  source: pdbbind_complex",
                f"  path: {manifest.as_posix()}",
                "  ligand_path_column: ligand_file",
                "  protein_path_column: pocket_file",
                "  id_column: complex_id",
                "  target_columns: [target]",
                "  split: predefined",
                "  split_column: split",
                "  invalid_smiles: error",
                "  strip_hydrogens: true",
                "model:",
                "  name: potentialnet",
                "  parameters: {bond_hidden_dim: 48, spatial_hidden_dim: 48, gather_dim: 48, num_bond_steps: 1, num_spatial_steps: 1, readout_hidden_dims: [8], dropout: 0.0}",
                "training:",
                "  epochs: 1",
                "  batch_size: 1",
                "  learning_rate: 0.001",
                "  weight_decay: 0.0",
                "  patience: 1",
                "  monitor: val_loss",
                "  monitor_mode: min",
                "  device: cpu",
                "  num_workers: 0",
                "task:",
                "  type: regression",
                "  loss: mse",
                "  metrics: [rmse]",
                "  target_scaling: true",
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    assert main(["train", "--config", str(config)]) == 0
    run_dir = tmp_path / "potentialnet_smoke" / "seed_031"
    status = json.loads((run_dir / "status.json").read_text(encoding="utf-8"))
    assert status["status"] == "completed"
    assert (run_dir / "test_predictions.csv").is_file()


def test_potentialnet_csv_smiles_cli_uses_the_normal_2d_workflow(
    tmp_path: Path,
) -> None:
    dataset = tmp_path / "molecules.csv"
    dataset.write_text(
        "smiles,target,split\nCC,1.0,train\nCO,2.0,validation\nCCC,3.0,test\n",
        encoding="utf-8",
    )
    config = tmp_path / "potentialnet_2d.yaml"
    config.write_text(
        "\n".join(
            [
                "experiment:",
                "  name: potentialnet_2d_smoke",
                "  seed: 37",
                f"  output_dir: {tmp_path.as_posix()}",
                "data:",
                f"  path: {dataset.as_posix()}",
                "  source: csv_smiles",
                "  smiles_column: smiles",
                "  target_columns: [target]",
                "  split: predefined",
                "  split_column: split",
                "  invalid_smiles: error",
                "model:",
                "  name: potentialnet",
                "  parameters: {bond_hidden_dim: 160, spatial_hidden_dim: 160, gather_dim: 160, num_bond_steps: 1, num_spatial_steps: 1, readout_hidden_dims: [8], dropout: 0.0}",
                "training:",
                "  epochs: 1",
                "  batch_size: 1",
                "  learning_rate: 0.001",
                "  weight_decay: 0.0",
                "  patience: 1",
                "  monitor: val_loss",
                "  monitor_mode: min",
                "  device: cpu",
                "  num_workers: 0",
                "task:",
                "  type: regression",
                "  loss: mse",
                "  metrics: [rmse]",
                "  target_scaling: true",
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    assert main(["train", "--config", str(config)]) == 0
    run_dir = tmp_path / "potentialnet_2d_smoke" / "seed_037"
    status = json.loads((run_dir / "status.json").read_text(encoding="utf-8"))
    assert status["status"] == "completed"
    assert (run_dir / "test_predictions.csv").is_file()
