"""Shared single-model and dataset-driven benchmark orchestration."""

from __future__ import annotations

import csv
import gc
import math
import random
import time
from collections.abc import Callable, Mapping, Sequence
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
    DataLoaders,
    MolecularDataset,
    PreparedDataset,
    build_dataloaders,
    prepare_model_samples,
)
from .dataset_sources import DatasetSourceError, DatasetSourceResult, load_dataset
from .evaluator import evaluate
from .featurizer import FeatureSchema
from .hooks import LoadedHook
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
from .tasks import TaskAdapter, TargetScalerState, build_task_adapter, fit_target_scaler
from .trainer import EpochRecord, StrategyResult, fit, resolve_device
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
class _SharedDatasetData:
    """Dataset state shared by every model and training seed."""

    dataset: MolecularDataset
    feature_schema: FeatureSchema
    context: BuildContext
    metadata: Mapping[str, object]


@dataclass(frozen=True)
class _SeedSplitData:
    """One reproducible split and training-target scaler for a run seed."""

    split_seed: int | None
    splits: SplitIndices
    scaler: TargetScalerState | None
    split_records: tuple[Mapping[str, object], ...]


@dataclass(frozen=True)
class _ResolvedModelPlan:
    spec: ModelSpec
    parameters: Mapping[str, object]
    training: TrainingConfig
    graph_transform: GraphTransform | None


def run_benchmark(config: ResolvedConfig) -> BenchmarkResult:
    """Run selected models in model-outer, seed-inner order.

    Dataset loading happens once. Split/scaler state is reused according to the
    configured split-seed policy, while model-specific transformed samples are
    prepared once and reused by every seed for that model. Failures are returned
    instead of aborting later models.
    """
    register_builtin_models()
    register_builtin_transforms()
    specs = resolve_benchmark_models(config.models)
    _validate_selected_overrides(config, specs)
    shared = _prepare_shared_dataset(config)
    seed_splits = _prepare_seed_splits(config, shared, config.experiment.seeds)
    preflight_split = seed_splits[config.experiment.seeds[0]]

    plans: list[_ResolvedModelPlan] = []
    failures: list[RunFailure] = []
    for spec in specs:
        try:
            plan = _resolve_model_plan(config, spec, shared.context)
            _preflight_model(plan, shared, preflight_split, plan.graph_transform)
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
        graph_transform = plan.graph_transform
        try:
            prepared = prepare_model_samples(
                shared.dataset,
                graph_transform,
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
                seed_split = seed_splits[seed]
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
                        seed_split,
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


def run_experiments(
    config: ResolvedConfig,
    *,
    featurizer: LoadedHook | None = None,
    training_strategy: LoadedHook | None = None,
) -> tuple[Path, ...]:
    """Run the CLI lifecycle with optional local featurizer/training hooks."""
    return _run_legacy_experiments(
        config,
        config.experiment.seeds,
        write_summary=True,
        featurizer=featurizer,
        training_strategy=training_strategy,
    )


def _run_legacy_experiments(
    config: ResolvedConfig,
    seeds: Sequence[int],
    *,
    write_summary: bool,
    featurizer: LoadedHook | None = None,
    training_strategy: LoadedHook | None = None,
) -> tuple[Path, ...]:
    register_builtin_models()
    register_builtin_transforms()
    shared = _prepare_shared_dataset(
        config,
        featurizer=featurizer,
        training_strategy=training_strategy,
    )
    seed_splits = _prepare_seed_splits(config, shared, seeds)
    spec = get_model_spec(config.model.name)
    plan = _resolve_model_plan(
        config,
        spec,
        shared.context,
        compatibility_parameters=config.model.parameters,
    )
    graph_transform = (
        plan.graph_transform
        if featurizer is None
        else _effective_graph_transform(plan, shared.dataset)
    )
    _preflight_model(plan, shared, seed_splits[seeds[0]], graph_transform)
    prepared = prepare_model_samples(shared.dataset, graph_transform)
    completed: list[Path] = []
    summary_rows: list[dict[str, object]] = []
    failed_rows: list[dict[str, object]] = []

    try:
        for seed in seeds:
            seed_split = seed_splits[seed]
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
                    seed_split,
                    paths,
                    training_strategy=training_strategy,
                )
            except Exception as exc:
                failed_row: dict[str, object] = {
                    "seed": seed,
                    "split_seed": seed_split.split_seed,
                    "status": "failed",
                    "error_type": type(exc).__name__,
                    "error_message": str(exc)[:500],
                }
                failed_rows.append(failed_row)
                summary_rows.append(failed_row)
            else:
                completed.append(run_dir)
                summary_rows.append(_summary_row(run_dir, seed, seed_split.split_seed))
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


def _prepare_shared_dataset(
    config: ResolvedConfig,
    *,
    featurizer: LoadedHook | None = None,
    training_strategy: LoadedHook | None = None,
) -> _SharedDatasetData:
    try:
        source_result = load_dataset(
            config.data,
            config.task,
            featurizer=None if featurizer is None else featurizer.callback,
            feature_schema_version=None if featurizer is None else featurizer.reference,
        )
    except DatasetSourceError as exc:
        raise RunnerError(
            f"could not load dataset source '{config.data.source}': {exc}"
        ) from exc
    _validate_source_result_targets(source_result, config)
    dataset = source_result.dataset
    print(
        f"dataset: valid={source_result.summary.valid_rows} "
        f"skipped={source_result.summary.skipped_rows}"
    )
    schema = source_result.feature_schema
    context = BuildContext(
        atom_dim=schema.atom_dim,
        bond_dim=schema.bond_dim,
        num_targets=config.num_targets,
        feature_schema_version=schema.version,
    )
    metadata: dict[str, object] = {
        "dataset_source": config.data.source,
        "dataset_fingerprint": source_result.fingerprint,
        "feature_schema_version": schema.version,
        "dataset_summary": source_result.summary.to_artifact_dict(),
    }
    if source_result.metadata:
        metadata["dataset_source_metadata"] = dict(source_result.metadata)
    hook_references = {
        name: hook.reference
        for name, hook in (
            ("featurizer", featurizer),
            ("training_strategy", training_strategy),
        )
        if hook is not None
    }
    if hook_references:
        metadata["runtime_hooks"] = hook_references
    return _SharedDatasetData(
        dataset=dataset,
        feature_schema=schema,
        context=context,
        metadata=metadata,
    )


def _prepare_seed_splits(
    config: ResolvedConfig,
    shared: _SharedDatasetData,
    run_seeds: Sequence[int],
) -> dict[int, _SeedSplitData]:
    """Create one split/scaler state per effective split seed and reuse it."""

    cached: dict[int | None, _SeedSplitData] = {}
    resolved: dict[int, _SeedSplitData] = {}
    for run_seed in run_seeds:
        split_seed = _resolve_split_seed(config, run_seed)
        split_data = cached.get(split_seed)
        if split_data is None:
            split_data = _prepare_seed_split(config, shared, run_seed, split_seed)
            cached[split_seed] = split_data
        resolved[run_seed] = split_data
    return resolved


def _resolve_split_seed(config: ResolvedConfig, run_seed: int) -> int | None:
    """Return the split RNG seed, or ``None`` for a fixed predefined split."""

    if config.data.split == "predefined":
        return None
    if config.data.split_seed_mode == "first_experiment_seed":
        return config.experiment.seeds[0]
    return run_seed


def _prepare_seed_split(
    config: ResolvedConfig,
    shared: _SharedDatasetData,
    run_seed: int,
    split_seed: int | None,
) -> _SeedSplitData:
    """Build one validated partition and train-only scaler for a run seed."""

    effective_seed = run_seed if split_seed is None else split_seed
    splits = make_split(shared.dataset, config.data, effective_seed)
    split_label = "predefined" if split_seed is None else str(split_seed)
    print(
        f"split (seed={split_label}): train={len(splits.train)} "
        f"validation={len(splits.validation)} test={len(splits.test)}"
    )
    scaler: TargetScalerState | None = None
    if config.task.type == "regression" and config.task.target_scaling:
        scaler = fit_target_scaler(
            shared.dataset,
            splits.train,
            feature_schema=shared.feature_schema,
            num_targets=shared.context.num_targets,
        )
    return _SeedSplitData(
        split_seed=split_seed,
        splits=splits,
        scaler=scaler,
        split_records=tuple(dict(row) for row in split_rows(splits, shared.dataset)),
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
    shared: _SharedDatasetData,
    seed_split: _SeedSplitData,
    graph_transform: GraphTransform | None,
) -> None:
    try:
        representative_count = min(
            plan.training.batch_size, len(seed_split.splits.train)
        )
        indices = seed_split.splits.train[:representative_count]
        samples = []
        for index in indices:
            sample = shared.dataset[index]
            samples.append(
                sample if graph_transform is None else graph_transform(sample)
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


def _effective_graph_transform(
    plan: _ResolvedModelPlan,
    dataset: MolecularDataset,
) -> GraphTransform | None:
    """Skip a bundled derivation when a featurizer supplies its required output.

    Optional model fields intentionally do not participate in this decision:
    a valid prepared 2D PotentialNet sample, for example, has no spatial
    Stage-2 inputs.  The selected model validates any optional-field grouping.
    """

    transform = plan.graph_transform
    if transform is None:
        return None
    if plan.spec.transform_output_fields:
        derived_fields = tuple(
            field
            for field in plan.spec.transform_output_fields
            if field in plan.spec.required_batch_fields
        )
    else:
        derived_fields = tuple(
            field
            for field in plan.spec.required_batch_fields
            if field not in {"x", "edge_index", "edge_attr", "batch"}
        )
    if not derived_fields:
        return transform

    availability = {
        field: all(
            isinstance(getattr(sample, field, None), Tensor) for sample in dataset
        )
        for field in derived_fields
    }
    if all(availability.values()):
        return None
    if not any(availability.values()):
        return transform
    fields = ", ".join(field for field, present in availability.items() if present)
    raise RunnerError(
        f"custom featurizer provides only part of model '{plan.spec.name}' derived "
        f"fields: {fields}; provide all or let the bundled transform derive all"
    )


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
    shared: _SharedDatasetData,
    seed_split: _SeedSplitData,
    paths: RunPaths,
    *,
    training_strategy: LoadedHook | None = None,
) -> Path:
    """Run one preflighted model seed against shared prepared data."""
    try:
        paths.mark_running()
        _seed_everything(resolved_run.seed)
        serializable_config = to_serializable_dict(config)
        serializable_config.update(shared.metadata)
        serializable_config["resolved_split_seed"] = seed_split.split_seed
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
        paths.write_split_rows(seed_split.split_records)

        loaders = build_dataloaders(
            prepared,
            seed_split.splits,
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
        adapter = build_task_adapter(config.task, seed_split.scaler)
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

        strategy_result = _fit_with_strategy(
            training_strategy,
            model,
            loaders,
            adapter,
            resolved_run.training,
            device=resolved_device,
            target_names=config.data.target_columns,
            on_epoch=on_epoch,
        )
        fit_result = strategy_result.fit_result
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
            optimizer_state_dict=strategy_result.optimizer_state_dict,
            resolved_config=serializable_config,
            feature_schema_version=schema_version,
            target_scaler_state=(
                None if seed_split.scaler is None else seed_split.scaler.to_dict()
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
            optimizer_state_dict=strategy_result.optimizer_state_dict,
            resolved_config=serializable_config,
            feature_schema_version=schema_version,
            target_scaler_state=(
                None if seed_split.scaler is None else seed_split.scaler.to_dict()
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


def _fit_with_strategy(
    strategy: LoadedHook | None,
    model: torch.nn.Module,
    loaders: DataLoaders,
    task_adapter: TaskAdapter,
    training: TrainingConfig,
    *,
    device: torch.device,
    target_names: Sequence[str],
    on_epoch: Callable[[EpochRecord], None],
) -> StrategyResult:
    """Run the default fit loop or one caller-provided strategy callback."""

    if strategy is None:
        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=training.learning_rate,
            weight_decay=training.weight_decay,
        )
        return StrategyResult(
            fit_result=fit(
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
            ),
            optimizer_state_dict=optimizer.state_dict(),
        )

    try:
        result = strategy.callback(
            model,
            loaders,
            task_adapter,
            training,
            device=device,
            target_names=target_names,
            on_epoch=on_epoch,
        )
    except Exception as exc:
        raise RunnerError(
            f"Training strategy '{strategy.reference}' failed: {exc}"
        ) from exc
    if not isinstance(result, StrategyResult):
        raise RunnerError(
            f"Training strategy '{strategy.reference}' must return "
            "molgnn.trainer.StrategyResult"
        )
    return result


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


def _validate_source_result_targets(
    source_result: DatasetSourceResult,
    config: ResolvedConfig,
) -> None:
    if source_result.summary.num_targets != config.num_targets:
        raise RunnerError(
            f"dataset source '{config.data.source}' reports "
            f"{source_result.summary.num_targets} target(s), but config declares "
            f"{config.num_targets}"
        )


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


def _summary_row(
    path: Path,
    seed: int,
    split_seed: int | None,
) -> dict[str, object]:
    test_history = path / "test_history.csv"
    with test_history.open(newline="", encoding="utf-8") as stream:
        rows = list(csv.DictReader(stream))
    if not rows:
        raise RunnerError(f"test history is empty for seed {seed}")
    row = rows[-1]
    summary: dict[str, object] = {
        "seed": seed,
        "split_seed": split_seed,
        "status": "completed",
    }
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
            if key
            not in {"seed", "split_seed", "status", "checkpoint_epoch", "sample_count"}
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
