"""Shared task-lifecycle coverage for the AMPNN and EMNN architectures."""

from __future__ import annotations

import csv
import json
from pathlib import Path

import pytest
import torch
from torch import Tensor, nn
from torch.nn import functional as F
from torch_geometric.data import Batch

from molgnn.cli import main
from molgnn.featurizer import featurize_smiles
from molgnn.models.ampnn_emnn_2020.ampnn import AMPNN
from molgnn.models.ampnn_emnn_2020.emnn import EMNN
from molgnn.tasks import BinaryClassificationTaskAdapter
from molgnn.transforms.ampnn import add_ampnn_edge_types
from molgnn.transforms.directed_edges import add_reverse_edge_index


class _RawThreeTargetHead(nn.Module):
    """Return deliberately unbounded logits to test the public model contract."""

    def forward(self, graph_embeddings: Tensor) -> Tensor:
        values = graph_embeddings.new_tensor((-2.0, 0.25, 3.0))
        return values.expand(graph_embeddings.shape[0], -1)


@pytest.mark.parametrize("model_name", ("ampnn", "emnn"))
def test_models_expose_unbounded_multitask_logits(model_name: str) -> None:
    model = _small_model(model_name, num_targets=3).eval()
    model.predictor = _RawThreeTargetHead()

    logits = model(_prepared_batch(model_name))
    expected = torch.tensor(((-2.0, 0.25, 3.0), (-2.0, 0.25, 3.0)))

    assert logits.shape == (2, 3)
    assert torch.equal(logits, expected)
    assert bool((logits < 0).any())
    assert bool((logits > 1).any())

    targets = torch.tensor(((0.0, 1.0, 1.0), (1.0, 0.0, 1.0)))
    mask = torch.tensor(((True, False, True), (True, True, False)))
    loss = BinaryClassificationTaskAdapter().loss(logits, targets, mask)
    assert loss is not None
    assert torch.allclose(
        loss,
        F.binary_cross_entropy_with_logits(logits[mask], targets[mask]),
    )


@pytest.mark.parametrize("model_name", ("ampnn", "emnn"))
def test_multitarget_models_round_trip_state_dict(model_name: str) -> None:
    torch.manual_seed(29)
    batch = _prepared_batch(model_name)
    original = _small_model(model_name, num_targets=3).eval()
    expected = original(batch)

    restored = _small_model(model_name, num_targets=3).eval()
    restored.load_state_dict(original.state_dict())

    actual = restored(batch)
    assert expected.shape == (2, 3)
    assert torch.allclose(actual, expected, atol=1e-6)


def test_ampnn_runs_masked_binary_multitask_csv_lifecycle(tmp_path: Path) -> None:
    dataset = tmp_path / "binary_multitask.csv"
    dataset.write_text(
        "smiles,activity_a,activity_b,activity_c,split\n"
        "CC,0,1,0,train\n"
        "CO,1,,1,train\n"
        "CCC,,0,1,train\n"
        "CCO,1,0,,validation\n"
        "C=O,0,,1,validation\n"
        "CCN,,1,0,test\n"
        "c1ccccc1,1,0,,test\n",
        encoding="utf-8",
    )
    config = _write_config(
        tmp_path,
        name="ampnn_binary_multitask",
        model_name="ampnn",
        dataset=dataset,
        target_columns=("activity_a", "activity_b", "activity_c"),
        parameters=(
            "{message_dim: 8, num_message_passing_steps: 1, "
            "message_hidden_dims: [8], attention_hidden_dims: [8], gather_dim: 8, "
            "gather_gate_hidden_dims: [8], gather_value_hidden_dims: [8], "
            "predictor_hidden_dims: [8], dropout: 0.0}"
        ),
        task_lines=(
            "  type: binary_classification",
            "  loss: bce_with_logits",
            "  metrics: [accuracy]",
            "  target_scaling: false",
        ),
    )

    assert main(["train", "--config", str(config)]) == 0

    run_dir = tmp_path / "ampnn_binary_multitask" / "seed_019"
    _assert_completed(run_dir)
    rows = _prediction_rows(run_dir)
    assert len(rows) == 2
    assert any(None in json.loads(row["target"]) for row in rows)
    for row in rows:
        probabilities = json.loads(row["prediction"])
        assert len(probabilities) == 3
        assert all(0.0 <= value <= 1.0 for value in probabilities)


def test_emnn_runs_masked_multitarget_regression_lifecycle(tmp_path: Path) -> None:
    dataset = tmp_path / "regression_multitask.csv"
    dataset.write_text(
        "smiles,solubility,energy,split\n"
        "CC,1.0,10.0,train\n"
        "CO,2.0,,train\n"
        "CCC,,30.0,train\n"
        "CCO,4.0,40.0,validation\n"
        "C=O,5.0,,validation\n"
        "CCN,6.0,60.0,test\n"
        "c1ccccc1,,70.0,test\n",
        encoding="utf-8",
    )
    config = _write_config(
        tmp_path,
        name="emnn_regression_multitask",
        model_name="emnn",
        dataset=dataset,
        target_columns=("solubility", "energy"),
        parameters=(
            "{edge_hidden_dim: 8, num_message_passing_steps: 1, "
            "edge_embedding_hidden_dims: [8], message_hidden_dims: [8], "
            "attention_hidden_dims: [8], gather_dim: 8, "
            "gather_gate_hidden_dims: [8], gather_value_hidden_dims: [8], "
            "predictor_hidden_dims: [8], dropout: 0.0}"
        ),
        task_lines=(
            "  type: regression",
            "  loss: mse",
            "  metrics: [rmse]",
            "  target_scaling: true",
        ),
    )

    assert main(["train", "--config", str(config)]) == 0

    run_dir = tmp_path / "emnn_regression_multitask" / "seed_019"
    _assert_completed(run_dir)
    checkpoint = torch.load(
        run_dir / "best.ckpt",
        map_location="cpu",
        weights_only=False,
    )
    scaler = checkpoint["target_scaler_state"]
    assert scaler is not None
    assert len(scaler["mean"]) == 2
    assert len(scaler["scale"]) == 2

    rows = _prediction_rows(run_dir)
    assert len(rows) == 2
    assert any(None in json.loads(row["target"]) for row in rows)
    assert all(len(json.loads(row["prediction"])) == 2 for row in rows)


def _small_model(model_name: str, *, num_targets: int) -> AMPNN | EMNN:
    if model_name == "ampnn":
        return AMPNN(
            atom_dim=153,
            message_dim=8,
            num_message_passing_steps=1,
            message_hidden_dims=(8,),
            attention_hidden_dims=(8,),
            gather_dim=8,
            gather_gate_hidden_dims=(8,),
            gather_value_hidden_dims=(8,),
            predictor_hidden_dims=(8,),
            dropout=0.0,
            num_targets=num_targets,
        )
    if model_name == "emnn":
        return EMNN(
            atom_dim=153,
            bond_dim=14,
            edge_hidden_dim=8,
            num_message_passing_steps=1,
            edge_embedding_hidden_dims=(8,),
            message_hidden_dims=(8,),
            attention_hidden_dims=(8,),
            gather_dim=8,
            gather_gate_hidden_dims=(8,),
            gather_value_hidden_dims=(8,),
            predictor_hidden_dims=(8,),
            dropout=0.0,
            num_targets=num_targets,
        )
    raise AssertionError(f"unknown model {model_name!r}")


def _prepared_batch(model_name: str) -> Batch:
    samples = [
        featurize_smiles(
            "CC",
            targets=[0.0, 1.0, 0.0],
            target_mask=[True, True, True],
            sample_id=0,
        ),
        featurize_smiles(
            "C=O",
            targets=[1.0, 0.0, 1.0],
            target_mask=[True, True, True],
            sample_id=1,
        ),
    ]
    if model_name == "ampnn":
        return Batch.from_data_list(
            [add_ampnn_edge_types(sample) for sample in samples]
        )
    if model_name == "emnn":
        return Batch.from_data_list(
            [add_reverse_edge_index(sample) for sample in samples]
        )
    raise AssertionError(f"unknown model {model_name!r}")


def _write_config(
    tmp_path: Path,
    *,
    name: str,
    model_name: str,
    dataset: Path,
    target_columns: tuple[str, ...],
    parameters: str,
    task_lines: tuple[str, ...],
) -> Path:
    config = tmp_path / f"{model_name}_task.yaml"
    targets = ", ".join(target_columns)
    config.write_text(
        "\n".join(
            (
                "experiment:",
                f"  name: {name}",
                "  seed: 19",
                f"  output_dir: {tmp_path.as_posix()}",
                "data:",
                "  source: csv_smiles",
                f"  path: {dataset.as_posix()}",
                "  smiles_column: smiles",
                f"  target_columns: [{targets}]",
                "  split: predefined",
                "  split_column: split",
                "  invalid_smiles: error",
                "model:",
                f"  name: {model_name}",
                f"  parameters: {parameters}",
                "training:",
                "  epochs: 1",
                "  batch_size: 2",
                "  learning_rate: 0.001",
                "  weight_decay: 0.0",
                "  patience: 1",
                "  monitor: val_loss",
                "  monitor_mode: min",
                "  device: cpu",
                "  num_workers: 0",
                "task:",
                *task_lines,
            )
        )
        + "\n",
        encoding="utf-8",
    )
    return config


def _assert_completed(run_dir: Path) -> None:
    status = json.loads((run_dir / "status.json").read_text(encoding="utf-8"))
    assert status["status"] == "completed"
    assert (run_dir / "best.ckpt").is_file()
    assert (run_dir / "test_predictions.csv").is_file()


def _prediction_rows(run_dir: Path) -> list[dict[str, str]]:
    with (run_dir / "test_predictions.csv").open(
        newline="", encoding="utf-8"
    ) as stream:
        return list(csv.DictReader(stream))
