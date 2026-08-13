"""Invariant and common-runtime checks for the sparse 2016 Weave model."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
import torch
from torch_geometric.data import Batch

from molgnn.cli import main
from molgnn.featurizer import featurize_smiles
from molgnn.models.weave_2016 import GaussianHistogramReadout, Weave, WeaveModule
from molgnn.transforms import TransformError, add_weave_inputs


def _sample(smiles: str, sample_id: int = 0):
    return add_weave_inputs(
        featurize_smiles(
            smiles,
            targets=[float(sample_id)],
            target_mask=[True],
            sample_id=sample_id,
        )
    )


def _model(*, num_targets: int = 1, batch_norm: bool = True) -> Weave:
    return Weave(
        atom_dim=153,
        hidden_dim=8,
        num_weave_modules=1,
        graph_feature_dim=8,
        predictor_hidden_dims=(8,),
        dropout=0.0,
        batch_norm=batch_norm,
        num_targets=num_targets,
    )


def _pair_row(data, source: int, target: int) -> torch.Tensor:
    rows = ((data.weave_pair_index[0] == source) & (data.weave_pair_index[1] == target)).nonzero(
        as_tuple=False
    ).flatten()
    assert rows.numel() == 1
    return data.weave_pair_attr[rows.item()]


def test_weave_module_preserves_all_four_atom_pair_paths() -> None:
    module = WeaveModule(
        atom_in_dim=1,
        pair_in_dim=1,
        hidden_dim=1,
        batch_norm=False,
    ).eval()
    with torch.no_grad():
        for layer in (
            module.atom_to_atom,
            module.pair_to_atom,
            module.update_atom,
            module.pair_to_pair,
            module.atom_to_pair,
            module.update_pair,
        ):
            layer.bias.zero_()
        module.atom_to_atom.weight.fill_(1.0)
        module.pair_to_atom.weight.fill_(1.0)
        module.update_atom.weight.copy_(torch.tensor([[1.0, 1.0]]))
        module.pair_to_pair.weight.fill_(1.0)
        module.atom_to_pair.weight.copy_(torch.tensor([[1.0, 2.0]]))
        module.update_pair.weight.copy_(torch.tensor([[0.0, 1.0]]))

    atoms = torch.tensor([[1.0], [2.0]])
    pairs = torch.tensor([[3.0], [4.0]])
    pair_index = torch.tensor([[0, 1], [1, 0]], dtype=torch.long)

    next_atoms, next_pairs = module(atoms, pairs, pair_index)

    # P -> A sums onto the first endpoint: [1, 2] + [3, 4].  A -> P is
    # f(a, b) + f(b, a), so both reverse records receive 9.
    assert torch.equal(next_atoms, torch.tensor([[4.0], [6.0]]))
    assert torch.equal(next_pairs, torch.tensor([[9.0], [9.0]]))


def test_transform_builds_sparse_ordered_pairs_with_canonical_features() -> None:
    data = _sample("CCC")

    assert data.weave_pair_index.shape == (2, 9)
    assert data.weave_pair_attr.shape == (9, 22)
    assert torch.equal(
        data.weave_pair_index,
        torch.tensor(
            [[0, 0, 0, 1, 1, 1, 2, 2, 2], [0, 1, 2, 0, 1, 2, 0, 1, 2]]
        ),
    )
    self_pair = _pair_row(data, 0, 0)
    direct_pair = _pair_row(data, 0, 1)
    two_hop_pair = _pair_row(data, 0, 2)
    assert not self_pair.any()
    assert torch.equal(direct_pair[:14], data.edge_attr[0])
    assert direct_pair[15:].tolist() == [1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
    assert not two_hop_pair[:15].any()
    assert two_hop_pair[15:].tolist() == [0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0]
    assert torch.equal(_pair_row(data, 0, 1), _pair_row(data, 1, 0))


def test_transform_marks_same_ring_pairs_and_rejects_noncanonical_inputs() -> None:
    benzene = _sample("c1ccccc1")
    assert _pair_row(benzene, 0, 0)[14].item() == 1.0
    assert _pair_row(benzene, 0, 1)[14].item() == 1.0
    assert _pair_row(benzene, 0, 2)[14].item() == 1.0

    malformed = featurize_smiles(
        "CC", targets=[0.0], target_mask=[True], sample_id=0
    )
    malformed.x = malformed.x.clone()
    malformed.x[0].zero_()
    with pytest.raises(TransformError, match="source-atom features"):
        add_weave_inputs(malformed)


def test_gaussian_histogram_normalizes_each_atom_feature_and_sums_graphs() -> None:
    readout = GaussianHistogramReadout(input_dim=1)
    atoms = torch.tensor([[-1.0], [0.0], [1.0]])
    memberships = readout.gaussian_histogram(atoms).reshape(3, 1, 11)
    output = readout(atoms, torch.tensor([0, 0, 1]), num_graphs=2)

    assert torch.allclose(memberships.sum(dim=-1), torch.ones((3, 1)))
    assert torch.allclose(output[0], memberships[:2].reshape(2, 11).sum(dim=0))
    assert torch.allclose(output[1], memberships[2].reshape(11))


def test_model_is_pair_order_and_atom_permutation_invariant() -> None:
    data = _sample("CCO")
    model = _model().eval()
    original = model(Batch.from_data_list([data]))

    reordered = data.clone()
    order = torch.arange(data.weave_pair_index.shape[1] - 1, -1, -1)
    reordered.weave_pair_index = data.weave_pair_index[:, order]
    reordered.weave_pair_attr = data.weave_pair_attr[order]
    assert torch.allclose(original, model(Batch.from_data_list([reordered])), atol=1e-6)

    permuted = data.clone()
    new_to_old = torch.tensor([2, 0, 1], dtype=torch.long)
    old_to_new = torch.empty_like(new_to_old)
    old_to_new[new_to_old] = torch.arange(new_to_old.numel())
    permuted.x = data.x[new_to_old]
    permuted.weave_pair_index = old_to_new[data.weave_pair_index]
    assert torch.allclose(original, model(Batch.from_data_list([permuted])), atol=1e-6)


def test_model_batches_pairs_handles_single_atom_and_returns_raw_multitask_values() -> None:
    samples = [_sample("C", 0), _sample("CCO", 1)]
    batch = Batch.from_data_list(samples)
    model = _model(num_targets=12)
    output = model(batch)
    output.square().mean().backward()

    assert batch.weave_pair_index.max().item() == batch.x.shape[0] - 1
    assert output.shape == (2, 12)
    assert any(parameter.grad is not None for parameter in model.parameters())

    singleton = _model().train()
    assert singleton(Batch.from_data_list([_sample("C")])).shape == (1, 1)


def test_model_rejects_pair_edges_between_batched_molecules() -> None:
    batch = Batch.from_data_list([_sample("CC", 0), _sample("CC", 1)])
    invalid = batch.clone()
    invalid.weave_pair_index = batch.weave_pair_index.clone()
    invalid.weave_pair_index[1, 0] = 2

    with pytest.raises(ValueError, match="must not connect different graphs"):
        _model().eval()(invalid)


def test_weave_runs_through_the_common_csv_artifact_pipeline(tmp_path: Path) -> None:
    dataset = tmp_path / "molecules.csv"
    dataset.write_text(
        "smiles,target,split\nCC,1.0,train\nCO,2.0,validation\nCCC,3.0,test\n",
        encoding="utf-8",
    )
    config = tmp_path / "weave.yaml"
    config.write_text(
        "\n".join(
            [
                "experiment:",
                "  name: weave_smoke",
                "  seed: 41",
                f"  output_dir: {tmp_path.as_posix()}",
                "data:",
                "  source: csv_smiles",
                f"  path: {dataset.as_posix()}",
                "  smiles_column: smiles",
                "  target_columns: [target]",
                "  split: predefined",
                "  split_column: split",
                "  invalid_smiles: error",
                "model:",
                "  name: weave",
                "  parameters: {hidden_dim: 8, num_weave_modules: 1, graph_feature_dim: 8, predictor_hidden_dims: [8], dropout: 0.0}",
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
    run_dir = tmp_path / "weave_smoke" / "seed_041"
    assert json.loads((run_dir / "status.json").read_text(encoding="utf-8"))["status"] == "completed"
    assert (run_dir / "test_predictions.csv").is_file()
