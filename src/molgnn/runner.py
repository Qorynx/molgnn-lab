"""Single shared experiment lifecycle for every registered architecture."""

from __future__ import annotations

import csv
import hashlib
import math
import random
import time
from dataclasses import replace
from pathlib import Path
from typing import Any

import numpy as np
import torch

from .artifacts import RunPaths
from .checkpointing import restore_model, save_checkpoint_atomic
from .config import ResolvedConfig, to_serializable_dict
from .dataset import CsvMoleculeDataset, build_dataloaders
from .evaluator import evaluate
from .models.registration import register_builtin_models
from .registry import (
    BuildContext,
    build_model,
    get_model_spec,
    validate_required_batch_fields,
)
from .splits import make_split, split_rows
from .tasks import TargetScalerState, build_task_adapter, fit_target_scaler
from .trainer import EpochRecord, fit, resolve_device
from .transforms import get_graph_transform, register_builtin_transforms


class RunnerError(RuntimeError):
    """Raised for orchestration failures after lifecycle artifacts are created."""


def run_experiment(config: ResolvedConfig) -> Path:
    """Run the first configured seed and return its artifact directory."""
    return _run_experiment(config, config.experiment.seed)


def run_experiments(config: ResolvedConfig) -> tuple[Path, ...]:
    """Run all configured seeds sequentially and return their artifact directories."""
    completed: list[Path] = []
    summary_rows: list[dict[str, object]] = []
    failed: list[dict[str, object]] = []
    for seed in config.experiment.seeds:
        try:
            run_dir = _run_experiment(config, seed)
        except Exception as exc:
            failed.append(
                {
                    "seed": seed,
                    "status": "failed",
                    "error_type": type(exc).__name__,
                    "error_message": str(exc)[:500],
                }
            )
            summary_rows.append(failed[-1])
            continue
        completed.append(run_dir)
        summary_rows.append(_summary_row(run_dir, seed))

    if completed or failed:
        root_paths = RunPaths.create(
            config.experiment.output_dir,
            config.experiment.name,
            config.experiment.seeds[0],
            initialize_status=False,
        )
        root_paths.write_summary(summary_rows)
        root_paths.write_aggregate_metrics(
            _aggregate_metrics(config.experiment.seeds, summary_rows, failed)
        )
    if failed:
        seeds = ", ".join(str(item["seed"]) for item in failed)
        raise RunnerError(f"one or more seeds failed: {seeds}")
    return tuple(completed)


def _run_experiment(config: ResolvedConfig, seed: int) -> Path:
    """Run one configured seed and return its artifact directory."""
    config = replace(config, experiment=replace(config.experiment, seed=seed))
    paths: RunPaths | None = None
    try:
        paths = RunPaths.create(
            config.experiment.output_dir,
            config.experiment.name,
            config.experiment.seed,
        )
        paths.mark_running()
        _seed_everything(config.experiment.seed)

        dataset = CsvMoleculeDataset(
            config.data.path,
            smiles_column=config.data.smiles_column,
            target_columns=config.data.target_columns,
            task_type=config.task.type,
            invalid_smiles=config.data.invalid_smiles,
            id_column=config.data.id_column,
            split_column=config.data.split_column,
        )
        print(
            f"dataset: valid={dataset.summary.valid_rows} "
            f"skipped={dataset.summary.skipped_invalid_smiles}"
        )
        schema = dataset.feature_schema
        metadata = {
            "dataset_fingerprint": _dataset_fingerprint(dataset.path),
            "feature_schema_version": schema.version,
            "dataset_summary": {
                "source_rows": dataset.summary.source_rows,
                "valid_rows": dataset.summary.valid_rows,
                "skipped_invalid_smiles": dataset.summary.skipped_invalid_smiles,
            },
        }
        serializable_config = to_serializable_dict(config)
        serializable_config.update(metadata)
        root_config = to_serializable_dict(
            replace(config, experiment=replace(config.experiment, seed=config.experiment.seeds[0]))
        )
        root_config.update(metadata)
        paths.write_experiment_config(root_config)
        paths.write_config(serializable_config)
        split_seed = config.experiment.seeds[0]
        splits = make_split(dataset, config.data, split_seed)
        paths.write_split_rows(split_rows(splits, dataset))
        print(
            f"split: train={len(splits.train)} validation={len(splits.validation)} "
            f"test={len(splits.test)}"
        )

        scaler: TargetScalerState | None = None
        if config.task.type == "regression" and config.task.target_scaling:
            scaler = fit_target_scaler(dataset, splits.train)
        register_builtin_models()
        register_builtin_transforms()
        model_spec = get_model_spec(config.model.name)
        graph_transform = get_graph_transform(model_spec.graph_transform_name)
        loaders = build_dataloaders(
            dataset,
            splits,
            config.training.batch_size,
            config.experiment.seed,
            config.training.num_workers,
            graph_transform,
        )
        validate_required_batch_fields(next(iter(loaders.train_eval)), model_spec)
        context = BuildContext(
            atom_dim=schema.atom_dim,
            bond_dim=schema.bond_dim,
            num_targets=config.num_targets,
            feature_schema_version=schema.version,
        )
        model = build_model(config.model.name, config.model.parameters, context)
        adapter = build_task_adapter(config.task, scaler)
        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=config.training.learning_rate,
            weight_decay=config.training.weight_decay,
        )
        resolved_device = resolve_device(config.training.device)
        print(f"device: {resolved_device}")
        started = time.perf_counter()

        def on_epoch(record: EpochRecord) -> None:
            paths.append_loss_history(record)
            paths.append_metrics_history(
                record,
                learning_rate=config.training.learning_rate,
                epoch_seconds=time.perf_counter() - started,
            )
            print(
                f"epoch {record.epoch_number}: train={record.train_eval_loss:.6g} "
                f"val={record.val_loss:.6g} monitor={record.monitor:.6g} "
                f"best={'yes' if record.is_best else 'no'}"
            )

        fit_result = fit(
            model,
            loaders,
            optimizer,
            adapter,
            epochs=config.training.epochs,
            patience=config.training.patience,
            monitor=config.training.monitor,
            monitor_mode=config.training.monitor_mode,
            device=resolved_device,
            target_names=config.data.target_columns,
            callbacks=(on_epoch,),
        )
        print(f"best epoch: {fit_result.best_epoch + 1} monitor={fit_result.best_value:.6g}")
        save_checkpoint_atomic(
            paths.best_checkpoint,
            epoch=fit_result.best_epoch,
            monitor_name=config.training.monitor,
            monitor_value=fit_result.best_value,
            model_state_dict=fit_result.best_state_dict,
            optimizer_state_dict=optimizer.state_dict(),
            resolved_config=serializable_config,
            feature_schema_version=schema.version,
            target_scaler_state=None if scaler is None else scaler.to_dict(),
        )
        last_record = fit_result.history[-1]
        last_monitor = (
            last_record.monitor if math.isfinite(last_record.monitor) else fit_result.best_value
        )
        save_checkpoint_atomic(
            paths.last_checkpoint,
            epoch=last_record.epoch,
            monitor_name=config.training.monitor,
            monitor_value=last_monitor,
            model_state_dict=model.state_dict(),
            optimizer_state_dict=optimizer.state_dict(),
            resolved_config=serializable_config,
            feature_schema_version=schema.version,
            target_scaler_state=None if scaler is None else scaler.to_dict(),
            rng_state=_rng_state(),
        )
        restore_model(
            model,
            paths.best_checkpoint,
            expected_feature_schema_version=schema.version,
        )
        test_result = evaluate(
            model,
            loaders.test,
            adapter,
            fit_result.device,
            return_predictions=True,
            target_names=config.data.target_columns,
        )
        test_row: dict[str, Any] = {
            "epoch": fit_result.best_epoch + 1,
            "loss": test_result.loss,
            **test_result.metrics,
        }
        paths.write_test_history(
            test_row,
            checkpoint_epoch=fit_result.best_epoch + 1,
            sample_count=test_result.sample_count,
        )
        paths.write_test_predictions(
            test_result,
            target_names=config.data.target_columns,
            seed=config.experiment.seed,
            checkpoint_epoch=fit_result.best_epoch + 1,
        )
        paths.mark_completed()
        print(f"run: {paths.seed_dir}")
        return paths.seed_dir
    except Exception as exc:
        if paths is not None:
            paths.mark_failed(exc)
        raise


def _dataset_fingerprint(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _rng_state() -> dict[str, object]:
    python_state = random.getstate()
    numpy_state = np.random.get_state()
    return {
        "python": python_state,
        "numpy": numpy_state,
        "torch": torch.get_rng_state(),
        "torch_cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
    }


def _summary_row(path: Path, seed: int) -> dict[str, object]:
    test_history = path / "test_history.csv"
    with test_history.open(newline="", encoding="utf-8") as stream:
        rows = list(csv.DictReader(stream))
    if not rows:
        raise RunnerError(f"test history is empty for seed {seed}")
    row = rows[-1]
    summary: dict[str, object] = {"seed": seed, "status": "completed"}
    for key, value in row.items():
        if key in {"checkpoint_epoch", "sample_count"}:
            summary[key] = int(str(value))
        elif key not in {"epoch"} and value not in {None, ""}:
            try:
                summary[key] = float(str(value))
            except ValueError:
                summary[key] = value
    return summary


def _aggregate_metrics(
    seeds: tuple[int, ...],
    rows: list[dict[str, object]],
    failed: list[dict[str, object]],
) -> dict[str, object]:
    completed = [row for row in rows if row.get("status") == "completed"]
    metric_names = sorted(
        {
            key
            for row in completed
            for key, value in row.items()
            if key not in {"seed", "status", "checkpoint_epoch", "sample_count"}
            and isinstance(value, (int, float))
            and math.isfinite(float(value))
        }
    )
    metrics: dict[str, object] = {}
    for name in metric_names:
        values = [float(str(row[name])) for row in completed if name in row]
        mean = sum(values) / len(values)
        variance = sum((value - mean) ** 2 for value in values) / len(values)
        metrics[name] = {
            "mean": mean,
            "std": math.sqrt(variance),
            "min": min(values),
            "max": max(values),
            "values": values,
        }
    return {
        "configured_seeds": list(seeds),
        "completed_seeds": [row["seed"] for row in completed],
        "failed_seeds": failed,
        "valid_seed_count": len(completed),
        "metrics": metrics,
    }


def _seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


__all__ = ["RunnerError", "run_experiment", "run_experiments"]
