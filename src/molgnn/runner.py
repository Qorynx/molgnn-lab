"""Shared single-model and dataset-driven benchmark orchestration."""

from __future__ import annotations

import csv
import gc
import hashlib
import math
import random
import time
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import Tensor
from torch_geometric.loader import DataLoader as PyGDataLoader

from .artifacts import RunPaths
from .checkpointing import restore_model, save_checkpoint_atomic
from .config import (
    ModelConfig,
    ResolvedConfig,
    TrainingConfig,
    resolve_training_config,
    to_serializable_dict,
)
from .dataset import (
    CsvMoleculeDataset,
    PreparedDataset,
    build_dataloaders,
    prepare_model_samples,
)
from .evaluator import evaluate
from .models.registration import register_builtin_models
from .registry import (
    BuildContext,
    ModelSpec,
    build_model,
    get_model_spec,
    resolve_benchmark_models,
    resolve_model_parameters,
    validate_required_batch_fields,
)
from .splits import SplitIndices, make_split, split_rows
from .tasks import TargetScalerState, build_task_adapter, fit_target_scaler
from .trainer import EpochRecord, fit, resolve_device
from .transforms import GraphTransform, get_graph_transform, register_builtin_transforms


class RunnerError(RuntimeError):
    """Raised for orchestration failures after lifecycle artifacts are created."""


@dataclass(frozen=True)
class ResolvedModelRun:
    """Effective model and training configuration for one seed run."""

    model_name: str
    parameters: Mapping[str, object]
    training: TrainingConfig
    seed: int


@dataclass(frozen=True)
class RunFailure:
    """One isolated model-seed failure returned by benchmark orchestration."""

    model_name: str
    seed: int
    stage: str
    error_type: str
    error_message: str


@dataclass(frozen=True)
class BenchmarkResult:
    """Phase-2 benchmark outcome; root summaries are added in Phase 3."""

    completed: tuple[Path, ...]
    failed: tuple[RunFailure, ...]
    summary_path: Path | None = None
    leaderboard_path: Path | None = None


@dataclass(frozen=True)
class _SharedBenchmarkData:
    dataset: CsvMoleculeDataset
    splits: SplitIndices
    scaler: TargetScalerState | None
    context: BuildContext
    metadata: Mapping[str, object]
    split_records: tuple[Mapping[str, object], ...]


@dataclass(frozen=True)
class _ResolvedModelPlan:
    spec: ModelSpec
    parameters: Mapping[str, object]
    training: TrainingConfig
    graph_transform: GraphTransform | None


def run_benchmark(config: ResolvedConfig) -> BenchmarkResult:
    """Run selected models in model-outer, seed-inner order.

    Dataset loading, split creation, and target-scaler fitting happen once.
    Model-specific transformed samples are prepared once and reused by every
    seed for that model. Failures are returned instead of aborting later models.
    """
    register_builtin_models()
    register_builtin_transforms()
    specs = resolve_benchmark_models(config.models)
    _validate_selected_overrides(config, specs)
    shared = _prepare_shared_data(config)

    plans: list[_ResolvedModelPlan] = []
    failures: list[RunFailure] = []
    for spec in specs:
        try:
            plan = _resolve_model_plan(config, spec, shared.context)
            _preflight_model(plan, shared)
        except Exception as exc:
            failures.extend(
                _failures_for_model(
                    spec.name, config.experiment.seeds, "preflight", exc
                )
            )
            continue
        plans.append(plan)

    completed: list[Path] = []
    benchmark_root = config.experiment.output_dir / config.experiment.name
    for plan in plans:
        prepared: PreparedDataset | None = None
        try:
            prepared = prepare_model_samples(
                shared.dataset,
                plan.graph_transform,
            )
        except Exception as exc:
            failures.extend(
                _failures_for_model(
                    plan.spec.name,
                    config.experiment.seeds,
                    "prepare",
                    exc,
                )
            )
            _cleanup_resources(plan.training.device)
            continue

        try:
            for seed in config.experiment.seeds:
                resolved_run = ResolvedModelRun(
                    model_name=plan.spec.name,
                    parameters=plan.parameters,
                    training=plan.training,
                    seed=seed,
                )
                effective_config = _effective_model_config(
                    config,
                    resolved_run,
                    output_dir=benchmark_root,
                    experiment_name=plan.spec.name,
                )
                try:
                    paths = RunPaths.create(benchmark_root, plan.spec.name, seed)
                    run_dir = _run_model_seed(
                        effective_config,
                        plan.spec,
                        resolved_run,
                        prepared,
                        shared,
                        paths,
                    )
                except Exception as exc:
                    failures.append(_run_failure(plan.spec.name, seed, "run", exc))
                else:
                    completed.append(run_dir)
                finally:
                    _cleanup_resources(plan.training.device)
        finally:
            del prepared
            _cleanup_resources(plan.training.device)

    return BenchmarkResult(completed=tuple(completed), failed=tuple(failures))


def run_experiment(config: ResolvedConfig) -> Path:
    """Compatibility wrapper running the first configured model and seed."""
    completed = _run_legacy_experiments(
        config,
        (config.experiment.seed,),
        write_summary=False,
    )
    return completed[0]


def run_experiments(config: ResolvedConfig) -> tuple[Path, ...]:
    """Compatibility wrapper retaining the current single-model artifact layout."""
    return _run_legacy_experiments(
        config,
        config.experiment.seeds,
        write_summary=True,
    )


def _run_legacy_experiments(
    config: ResolvedConfig,
    seeds: Sequence[int],
    *,
    write_summary: bool,
) -> tuple[Path, ...]:
    register_builtin_models()
    register_builtin_transforms()
    shared = _prepare_shared_data(config)
    spec = get_model_spec(config.model.name)
    plan = _resolve_model_plan(
        config,
        spec,
        shared.context,
        compatibility_parameters=config.model.parameters,
    )
    _preflight_model(plan, shared)
    prepared = prepare_model_samples(shared.dataset, plan.graph_transform)
    completed: list[Path] = []
    summary_rows: list[dict[str, object]] = []
    failed_rows: list[dict[str, object]] = []

    try:
        for seed in seeds:
            resolved_run = ResolvedModelRun(
                model_name=spec.name,
                parameters=plan.parameters,
                training=plan.training,
                seed=seed,
            )
            effective_config = _effective_model_config(
                config,
                resolved_run,
                output_dir=config.experiment.output_dir,
                experiment_name=config.experiment.name,
            )
            try:
                paths = RunPaths.create(
                    config.experiment.output_dir,
                    config.experiment.name,
                    seed,
                )
                run_dir = _run_model_seed(
                    effective_config,
                    spec,
                    resolved_run,
                    prepared,
                    shared,
                    paths,
                )
            except Exception as exc:
                failed_row: dict[str, object] = {
                    "seed": seed,
                    "status": "failed",
                    "error_type": type(exc).__name__,
                    "error_message": str(exc)[:500],
                }
                failed_rows.append(failed_row)
                summary_rows.append(failed_row)
            else:
                completed.append(run_dir)
                summary_rows.append(_summary_row(run_dir, seed))
            finally:
                _cleanup_resources(plan.training.device)
    finally:
        del prepared
        _cleanup_resources(plan.training.device)

    if write_summary and summary_rows:
        root_paths = RunPaths.create(
            config.experiment.output_dir,
            config.experiment.name,
            config.experiment.seeds[0],
            initialize_status=False,
        )
        root_paths.write_summary(summary_rows)
        root_paths.write_aggregate_metrics(
            _aggregate_metrics(tuple(seeds), summary_rows, failed_rows)
        )
    if failed_rows:
        failed_seeds = ", ".join(str(item["seed"]) for item in failed_rows)
        raise RunnerError(f"one or more seeds failed: {failed_seeds}")
    return tuple(completed)


def _prepare_shared_data(config: ResolvedConfig) -> _SharedBenchmarkData:
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
    split_seed = config.experiment.seeds[0]
    splits = make_split(dataset, config.data, split_seed)
    print(
        f"split: train={len(splits.train)} validation={len(splits.validation)} "
        f"test={len(splits.test)}"
    )
    scaler: TargetScalerState | None = None
    if config.task.type == "regression" and config.task.target_scaling:
        scaler = fit_target_scaler(dataset, splits.train)
    schema = dataset.feature_schema
    context = BuildContext(
        atom_dim=schema.atom_dim,
        bond_dim=schema.bond_dim,
        num_targets=config.num_targets,
        feature_schema_version=schema.version,
    )
    metadata: dict[str, object] = {
        "dataset_fingerprint": _dataset_fingerprint(dataset.path),
        "feature_schema_version": schema.version,
        "dataset_summary": {
            "source_rows": dataset.summary.source_rows,
            "valid_rows": dataset.summary.valid_rows,
            "skipped_invalid_smiles": dataset.summary.skipped_invalid_smiles,
        },
    }
    records = tuple(dict(row) for row in split_rows(splits, dataset))
    return _SharedBenchmarkData(
        dataset=dataset,
        splits=splits,
        scaler=scaler,
        context=context,
        metadata=metadata,
        split_records=records,
    )


def _resolve_model_plan(
    config: ResolvedConfig,
    spec: ModelSpec,
    context: BuildContext,
    *,
    compatibility_parameters: Mapping[str, object] | None = None,
) -> _ResolvedModelPlan:
    override = config.model_overrides.get(spec.name)
    parameters = dict(compatibility_parameters or {})
    training_overrides: Mapping[str, object] = {}
    if override is not None:
        parameters.update(override.parameters)
        training_overrides = override.training
    resolved_parameters = resolve_model_parameters(spec, parameters, context)
    training = resolve_training_config(config.training, training_overrides, config.task)
    return _ResolvedModelPlan(
        spec=spec,
        parameters=resolved_parameters,
        training=training,
        graph_transform=get_graph_transform(spec.graph_transform_name),
    )


def _preflight_model(
    plan: _ResolvedModelPlan,
    shared: _SharedBenchmarkData,
) -> None:
    try:
        representative_count = min(plan.training.batch_size, len(shared.splits.train))
        indices = shared.splits.train[:representative_count]
        samples = []
        for index in indices:
            sample = shared.dataset[index]
            samples.append(
                sample if plan.graph_transform is None else plan.graph_transform(sample)
            )
        batch = next(iter(PyGDataLoader(samples, batch_size=representative_count)))
        validate_required_batch_fields(batch, plan.spec)
        model = build_model(plan.spec.name, plan.parameters, shared.context)
        model.eval()
        with torch.inference_mode():
            predictions = model(batch)
        targets = getattr(batch, "y", None)
        if not isinstance(predictions, Tensor) or not isinstance(targets, Tensor):
            raise RunnerError(
                f"model '{plan.spec.name}' preflight must produce a Tensor "
                "and receive batch.y"
            )
        expected_shape = (targets.shape[0], shared.context.num_targets)
        if tuple(predictions.shape) != expected_shape:
            raise RunnerError(
                f"model '{plan.spec.name}' output shape {tuple(predictions.shape)} "
                f"does not match {expected_shape}"
            )
    finally:
        _cleanup_resources("cpu")


def _effective_model_config(
    config: ResolvedConfig,
    resolved_run: ResolvedModelRun,
    *,
    output_dir: Path,
    experiment_name: str,
) -> ResolvedConfig:
    return replace(
        config,
        experiment=replace(
            config.experiment,
            name=experiment_name,
            seed=resolved_run.seed,
            output_dir=output_dir,
        ),
        model=ModelConfig(
            name=resolved_run.model_name,
            parameters=resolved_run.parameters,
        ),
        training=resolved_run.training,
    )


def _run_model_seed(
    config: ResolvedConfig,
    model_spec: ModelSpec,
    resolved_run: ResolvedModelRun,
    prepared: PreparedDataset,
    shared: _SharedBenchmarkData,
    paths: RunPaths,
) -> Path:
    """Run one preflighted model seed against shared prepared data."""
    try:
        paths.mark_running()
        _seed_everything(resolved_run.seed)
        serializable_config = to_serializable_dict(config)
        serializable_config.update(shared.metadata)
        root_config = to_serializable_dict(
            replace(
                config,
                experiment=replace(
                    config.experiment,
                    seed=config.experiment.seeds[0],
                ),
            )
        )
        root_config.update(shared.metadata)
        paths.write_experiment_config(root_config)
        paths.write_config(serializable_config)
        paths.write_split_rows(shared.split_records)

        loaders = build_dataloaders(
            prepared,
            shared.splits,
            resolved_run.training.batch_size,
            resolved_run.seed,
            resolved_run.training.num_workers,
        )
        validate_required_batch_fields(next(iter(loaders.train_eval)), model_spec)
        model = build_model(
            resolved_run.model_name,
            resolved_run.parameters,
            shared.context,
        )
        adapter = build_task_adapter(config.task, shared.scaler)
        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=resolved_run.training.learning_rate,
            weight_decay=resolved_run.training.weight_decay,
        )
        resolved_device = resolve_device(resolved_run.training.device)
        print(f"model: {resolved_run.model_name} seed: {resolved_run.seed}")
        print(f"device: {resolved_device}")
        started = time.perf_counter()

        def on_epoch(record: EpochRecord) -> None:
            paths.append_loss_history(record)
            paths.append_metrics_history(
                record,
                learning_rate=resolved_run.training.learning_rate,
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
            epochs=resolved_run.training.epochs,
            patience=resolved_run.training.patience,
            monitor=resolved_run.training.monitor,
            monitor_mode=resolved_run.training.monitor_mode,
            device=resolved_device,
            target_names=config.data.target_columns,
            callbacks=(on_epoch,),
        )
        print(
            f"best epoch: {fit_result.best_epoch + 1} "
            f"monitor={fit_result.best_value:.6g}"
        )
        schema_version = shared.context.feature_schema_version
        save_checkpoint_atomic(
            paths.best_checkpoint,
            epoch=fit_result.best_epoch,
            monitor_name=resolved_run.training.monitor,
            monitor_value=fit_result.best_value,
            model_state_dict=fit_result.best_state_dict,
            optimizer_state_dict=optimizer.state_dict(),
            resolved_config=serializable_config,
            feature_schema_version=schema_version,
            target_scaler_state=(
                None if shared.scaler is None else shared.scaler.to_dict()
            ),
        )
        last_record = fit_result.history[-1]
        last_monitor = (
            last_record.monitor
            if math.isfinite(last_record.monitor)
            else fit_result.best_value
        )
        save_checkpoint_atomic(
            paths.last_checkpoint,
            epoch=last_record.epoch,
            monitor_name=resolved_run.training.monitor,
            monitor_value=last_monitor,
            model_state_dict=model.state_dict(),
            optimizer_state_dict=optimizer.state_dict(),
            resolved_config=serializable_config,
            feature_schema_version=schema_version,
            target_scaler_state=(
                None if shared.scaler is None else shared.scaler.to_dict()
            ),
            rng_state=_rng_state(),
        )
        restore_model(
            model,
            paths.best_checkpoint,
            expected_feature_schema_version=schema_version,
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
        paths.write_test_predictions(test_result)
        paths.mark_completed()
        print(f"run: {paths.seed_dir}")
        return paths.seed_dir
    except Exception as exc:
        paths.mark_failed(exc)
        raise


def _validate_selected_overrides(
    config: ResolvedConfig,
    specs: Sequence[ModelSpec],
) -> None:
    selected = {spec.name for spec in specs}
    inactive = sorted(set(config.model_overrides) - selected)
    if inactive:
        names = ", ".join(inactive)
        raise RunnerError(f"model override(s) do not match selected models: {names}")


def _failures_for_model(
    model_name: str,
    seeds: Sequence[int],
    stage: str,
    error: Exception,
) -> tuple[RunFailure, ...]:
    return tuple(_run_failure(model_name, seed, stage, error) for seed in seeds)


def _run_failure(
    model_name: str,
    seed: int,
    stage: str,
    error: Exception,
) -> RunFailure:
    return RunFailure(
        model_name=model_name,
        seed=seed,
        stage=stage,
        error_type=type(error).__name__,
        error_message=str(error)[:500],
    )


def _cleanup_resources(device: str) -> None:
    gc.collect()
    if torch.cuda.is_available() and device in {"cuda", "auto"}:
        torch.cuda.empty_cache()


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
        "torch_cuda": (
            torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None
        ),
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
        values = [
            float(str(row[name]))
            for row in completed
            if name in row and math.isfinite(float(str(row[name])))
        ]
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


__all__ = [
    "BenchmarkResult",
    "ResolvedModelRun",
    "RunFailure",
    "RunnerError",
    "run_benchmark",
    "run_experiment",
    "run_experiments",
]
