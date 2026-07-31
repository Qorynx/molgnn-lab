"""Regression coverage for explicit local ``molgnn train`` hooks."""

from __future__ import annotations

import json
from pathlib import Path

import yaml

from molgnn.cli import main


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
    assert (run_dir / "test_predictions.csv").is_file()
    resolved = yaml.safe_load((run_dir / "config.yaml").read_text(encoding="utf-8"))
    assert resolved["runtime_hooks"] == {
        "featurizer": featurizer_spec,
        "training_strategy": strategy_spec,
    }


def test_train_without_hooks_keeps_default_runtime_metadata(tmp_path: Path) -> None:
    config = _write_config(tmp_path, name="default_hooks")

    assert main(["train", "--config", str(config)]) == 0

    resolved = yaml.safe_load(
        (tmp_path / "default_hooks" / "seed_017" / "config.yaml").read_text(
            encoding="utf-8"
        )
    )
    assert "runtime_hooks" not in resolved
