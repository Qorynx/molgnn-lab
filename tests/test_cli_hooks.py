"""Regression coverage for explicit local ``molgnn train`` hooks."""

from __future__ import annotations

import json
from pathlib import Path

import yaml

from molgnn import runner
from molgnn.cli import main
from molgnn.config import load_config


def _write(path: Path, content: str) -> Path:
    path.write_text(content, encoding="utf-8")
    return path


def _write_config(tmp_path: Path, *, name: str) -> Path:
    project_root = Path.cwd()
    fixture = project_root / "tests" / "fixtures" / "tiny_regression.csv"
    return _write(
        tmp_path / f"{name}.yaml",
        "\n".join(
            [
                "extends: " + (project_root / "configs" / "base.yaml").as_posix(),
                "experiment:",
                f"  name: {name}",
                "  seed: 17",
                f"  output_dir: {tmp_path.as_posix()}",
                "data:",
                f"  path: {fixture.as_posix()}",
                "  split_ratios: [0.7, 0.2, 0.1]",
                "model:",
                "  name: gcn_baseline",
                "  parameters: {hidden_dim: 8, num_layers: 1, dropout: 0.0}",
                "training:",
                "  epochs: 1",
                "  batch_size: 4",
                "  patience: 1",
                "  device: cpu",
            ]
        )
        + "\n",
    )


def test_train_rejects_invalid_hook_specs_without_traceback(
    tmp_path: Path, capsys: object
) -> None:
    config = _write_config(tmp_path, name="invalid_hook")
    noncallable_hook = _write(tmp_path / "noncallable.py", "value = 1\n")
    cases = (
        ("--featurizer", "not-a-hook", "must use"),
        (
            "--training-strategy",
            f"{tmp_path / 'missing.py'}:fit",
            "hook file",
        ),
        ("--featurizer", f"{noncallable_hook}:value", "is not callable"),
    )

    for option, selector, expected_message in cases:
        assert main(["train", "--config", str(config), option, selector]) == 2
        captured = capsys.readouterr()
        assert "Train error:" in captured.err
        assert expected_message in captured.err
        assert "Traceback" not in captured.out + captured.err


def test_train_uses_file_hooks_and_records_runtime_references(tmp_path: Path) -> None:
    config = _write_config(tmp_path, name="file_hooks")
    marker = tmp_path / "hook_calls.txt"
    featurizer_hook = _write(
        tmp_path / "custom_featurizer.py",
        f"""from pathlib import Path

from molgnn.data import MolecularData
from molgnn.featurizer import featurize_smiles

_MARKER = Path({str(marker)!r})


def build_sample(smiles, *, targets, target_mask, sample_id):
    with _MARKER.open("a", encoding="utf-8") as stream:
        stream.write("featurizer\\n")
    sample = featurize_smiles(
        smiles,
        targets=targets,
        target_mask=target_mask,
        sample_id=sample_id,
    )
    if not isinstance(sample, MolecularData):
        raise TypeError("custom featurizer must return MolecularData")
    return sample
""",
    )
    strategy_hook = _write(
        tmp_path / "custom_strategy.py",
        f"""from pathlib import Path

import torch

from molgnn.trainer import StrategyResult, fit as default_fit

_MARKER = Path({str(marker)!r})


def fit(model, loaders, task_adapter, training, *, device, target_names, on_epoch):
    with _MARKER.open("a", encoding="utf-8") as stream:
        stream.write("training_strategy\\n")
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=training.learning_rate,
        weight_decay=training.weight_decay,
    )
    fit_result = default_fit(
        model,
        loaders,
        optimizer,
        task_adapter,
        epochs=training.epochs,
        patience=training.patience,
        monitor=training.monitor,
        monitor_mode=training.monitor_mode,
        device=device,
        target_names=target_names,
        callbacks=(on_epoch,),
    )
    return StrategyResult(
        fit_result=fit_result,
        optimizer_state_dict=optimizer.state_dict(),
    )
""",
    )
    featurizer_spec = f"{featurizer_hook}:build_sample"
    strategy_spec = f"{strategy_hook}:fit"

    assert (
        main(
            [
                "train",
                "--config",
                str(config),
                "--featurizer",
                featurizer_spec,
                "--training-strategy",
                strategy_spec,
            ]
        )
        == 0
    )

    calls = marker.read_text(encoding="utf-8").splitlines()
    assert "featurizer" in calls
    assert calls.count("training_strategy") == 1

    run_dir = tmp_path / "file_hooks" / "seed_017"
    assert (
        json.loads((run_dir / "status.json").read_text(encoding="utf-8"))["status"]
        == "completed"
    )
    assert (run_dir / "best.ckpt").is_file()
    run_results = json.loads((run_dir / "run_results.json").read_text(encoding="utf-8"))
    assert run_results["schema_version"] == 1
    assert run_results["split"]
    assert run_results["training"]["epochs"]
    first_epoch = run_results["training"]["epochs"][0]
    assert set(first_epoch) == {"epoch", "loss", "metrics"}
    assert first_epoch["loss"]
    assert first_epoch["metrics"]
    assert not (run_dir / "split.csv").exists()
    assert not (run_dir / "loss_history.csv").exists()
    assert not (run_dir / "metrics_history.csv").exists()
    assert (run_dir / "test_predictions.csv").is_file()
    aggregate = json.loads(
        (tmp_path / "file_hooks" / "aggregate_metrics.json").read_text(
            encoding="utf-8"
        )
    )
    assert aggregate["schema_version"] == 1
    assert aggregate["artifact_type"] == "evaluation_aggregate"
    assert aggregate["status"] == "completed"
    assert aggregate["protocol"]["aggregation_unit"] == "independent_seed_run"
    assert aggregate["protocol"]["epoch_values_aggregated"] is False
    assert aggregate["protocol"]["standard_deviation"] == {
        "ddof": 1,
        "minimum_run_count": 2,
        "type": "sample",
        "value_when_undefined": None,
    }
    model_aggregate = aggregate["models"]["gcn_baseline"]
    assert model_aggregate["completed_run_count"] == 1
    assert model_aggregate["checkpoint_selection"]["artifact"] == "best.ckpt"
    assert all(metric["std"] is None for metric in model_aggregate["metrics"].values())
    resolved = yaml.safe_load((run_dir / "config.yaml").read_text(encoding="utf-8"))
    assert resolved["runtime_hooks"] == {
        "featurizer": featurizer_spec,
        "training_strategy": strategy_spec,
    }


def test_train_without_hooks_keeps_default_runtime_metadata(
    tmp_path: Path, monkeypatch
) -> None:
    config = _write_config(tmp_path, name="default_hooks")
    original_load_dataset = runner.load_dataset
    loaded_sources: list[str] = []

    def capture_dataset_source(data_config, task_config, **kwargs):
        loaded_sources.append(data_config.source)
        return original_load_dataset(data_config, task_config, **kwargs)

    monkeypatch.setattr(runner, "load_dataset", capture_dataset_source)

    assert main(["train", "--config", str(config)]) == 0

    resolved = yaml.safe_load(
        (tmp_path / "default_hooks" / "seed_017" / "config.yaml").read_text(
            encoding="utf-8"
        )
    )
    assert "runtime_hooks" not in resolved
    assert resolved["data"]["source"] == "csv_smiles"
    assert resolved["dataset_source"] == "csv_smiles"
    assert loaded_sources == ["csv_smiles"]


def test_paper_style_aggregate_uses_sample_std_across_seed_runs(
    tmp_path: Path,
) -> None:
    config = load_config(_write_config(tmp_path, name="aggregate_unit"))
    rows = [
        {
            "seed": seed,
            "split_seed": seed,
            "status": "completed",
            "checkpoint_epoch": 4,
            "sample_count": 10,
            "rmse": value,
        }
        for seed, value in ((1, 1.0), (2, 2.0), (3, 3.0))
    ]

    aggregate = runner._aggregate_experiment_metrics(
        config,
        ("gcn_baseline",),
        {"gcn_baseline": rows},
        (),
        {"gcn_baseline": config.training},
        configured_seeds=(1, 2, 3),
    )

    rmse = aggregate["models"]["gcn_baseline"]["metrics"]["rmse"]
    assert rmse == {
        "count": 3,
        "mean": 2.0,
        "std": 1.0,
        "min": 1.0,
        "max": 3.0,
        "values_by_run": [
            {"seed": 1, "split_seed": 1, "value": 1.0},
            {"seed": 2, "split_seed": 2, "value": 2.0},
            {"seed": 3, "split_seed": 3, "value": 3.0},
        ],
    }
