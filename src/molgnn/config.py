"""Typed YAML configuration loading and validation."""

from __future__ import annotations

import copy
import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, fields, is_dataclass
from pathlib import Path
from types import MappingProxyType
from typing import Any, Literal, cast

import yaml


class ConfigError(ValueError):
    """Raised when an experiment configuration is invalid."""


@dataclass(frozen=True)
class ExperimentConfig:
    name: str
    seed: int
    output_dir: Path
    seeds: tuple[int, ...] = ()


@dataclass(frozen=True)
class DataConfig:
    path: Path
    smiles_column: str
    target_columns: tuple[str, ...]
    id_column: str | None
    split: Literal["random", "scaffold", "predefined"]
    split_ratios: tuple[float, float, float]
    split_column: str | None
    invalid_smiles: Literal["error", "skip"]


@dataclass(frozen=True)
class ModelConfig:
    """Temporary single-model view retained until the benchmark runner lands."""

    name: str
    parameters: Mapping[str, object]


@dataclass(frozen=True)
class ModelOverrideConfig:
    """Optional architecture and training overrides for one registered model."""

    parameters: Mapping[str, object]
    training: Mapping[str, object]


@dataclass(frozen=True)
class TrainingConfig:
    epochs: int
    batch_size: int
    learning_rate: float
    weight_decay: float
    patience: int
    monitor: str
    monitor_mode: Literal["min", "max"]
    device: Literal["cpu", "cuda", "auto"]
    num_workers: int


@dataclass(frozen=True)
class TaskConfig:
    type: Literal["regression", "binary_classification"]
    loss: Literal["mse", "bce_with_logits"]
    metrics: tuple[str, ...]
    target_scaling: bool


@dataclass(frozen=True)
class ResolvedConfig:
    experiment: ExperimentConfig
    data: DataConfig
    model: ModelConfig
    training: TrainingConfig
    task: TaskConfig
    models: tuple[str, ...] | None
    model_overrides: Mapping[str, ModelOverrideConfig]

    @property
    def num_targets(self) -> int:
        """Return the number of target columns declared by the dataset config."""
        return len(self.data.target_columns)


_SECTION_KEYS: dict[str, frozenset[str]] = {
    "experiment": frozenset({"name", "seed", "seeds", "output_dir"}),
    "data": frozenset(
        {
            "path",
            "smiles_column",
            "target_columns",
            "id_column",
            "split",
            "split_ratios",
            "split_column",
            "invalid_smiles",
        }
    ),
    "model": frozenset({"name", "parameters"}),
    "training": frozenset(
        {
            "epochs",
            "batch_size",
            "learning_rate",
            "weight_decay",
            "patience",
            "monitor",
            "monitor_mode",
            "device",
            "num_workers",
        }
    ),
    "task": frozenset({"type", "loss", "metrics", "target_scaling"}),
}
_MODEL_OVERRIDE_KEYS = frozenset({"parameters", "training"})
_CONTEXT_PARAMETER_KEYS = frozenset(
    {"atom_dim", "bond_dim", "num_targets", "feature_schema_version"}
)
_TOP_LEVEL_KEYS = frozenset({"extends", "models", "model_overrides", *_SECTION_KEYS})

_DEFAULT_MODEL: dict[str, object] = {"name": "gcn_baseline", "parameters": {}}

_DEFAULT_CONFIG: dict[str, Any] = {
    "experiment": {"name": "molgnn_experiment", "seed": 0, "output_dir": "runs"},
    "data": {
        "path": "data/data.csv",
        "smiles_column": "smiles",
        "target_columns": ["target"],
        "id_column": None,
        "split": "random",
        "split_ratios": [0.8, 0.1, 0.1],
        "split_column": None,
        "invalid_smiles": "error",
    },
    "training": {
        "epochs": 200,
        "batch_size": 64,
        "learning_rate": 0.001,
        "weight_decay": 1e-6,
        "patience": 30,
        "monitor": "val_loss",
        "monitor_mode": "min",
        "device": "auto",
        "num_workers": 0,
    },
    "task": {
        "type": "regression",
        "loss": "mse",
        "target_scaling": True,
    },
}

_REGRESSION_METRICS = frozenset({"rmse", "mae", "r2"})
_CLASSIFICATION_METRICS = frozenset(
    {"roc_auc", "prc_auc", "accuracy", "balanced_accuracy"}
)


def _read_yaml(path: Path) -> dict[str, Any]:
    try:
        with path.open("r", encoding="utf-8") as stream:
            value = yaml.safe_load(stream)
    except OSError as exc:
        raise ConfigError(f"Cannot read config file '{path}': {exc}") from exc
    except yaml.YAMLError as exc:
        raise ConfigError(f"Invalid YAML in '{path}': {exc}") from exc

    if value is None:
        return {}
    if not isinstance(value, Mapping):
        raise ConfigError(f"Config root must be a mapping: '{path}'")
    return dict(value)


def _validate_top_level_keys(raw: Mapping[str, Any], *, source: Path) -> None:
    unknown = set(raw) - _TOP_LEVEL_KEYS
    if unknown:
        names = ", ".join(sorted(str(name) for name in unknown))
        raise ConfigError(f"Unknown config key(s) in '{source}': {names}")


def _validate_section_keys(
    raw: Mapping[str, Any], *, section: str, source: Path
) -> None:
    unknown = set(raw) - _SECTION_KEYS[section]
    if unknown:
        names = ", ".join(sorted(str(name) for name in unknown))
        raise ConfigError(
            f"Unknown key(s) in section '{section}' of '{source}': {names}"
        )


def _validate_raw_mapping(raw: Mapping[str, Any], *, source: Path) -> None:
    _validate_top_level_keys(raw, source=source)
    for section in _SECTION_KEYS:
        value = raw.get(section)
        if value is not None:
            if not isinstance(value, Mapping):
                raise ConfigError(
                    f"Section '{section}' must be a mapping in '{source}'"
                )
            _validate_section_keys(value, section=section, source=source)
    if "models" in raw and raw["models"] is not None:
        _strings(raw["models"], field="models")
    if "model_overrides" in raw:
        _validate_raw_model_overrides(raw["model_overrides"], source=source)


def _validate_raw_model_overrides(value: object, *, source: Path) -> None:
    if not isinstance(value, Mapping):
        raise ConfigError(f"'model_overrides' must be a mapping in '{source}'")
    for raw_name, raw_override in value.items():
        name = _string(raw_name, field="model_overrides model name")
        if not isinstance(raw_override, Mapping):
            raise ConfigError(f"'model_overrides.{name}' must be a mapping")
        unknown = set(raw_override) - _MODEL_OVERRIDE_KEYS
        if unknown:
            names = ", ".join(sorted(str(item) for item in unknown))
            raise ConfigError(f"Unknown key(s) in 'model_overrides.{name}': {names}")
        parameters = raw_override.get("parameters", {})
        if not isinstance(parameters, Mapping):
            raise ConfigError(f"'model_overrides.{name}.parameters' must be a mapping")
        training = raw_override.get("training", {})
        if not isinstance(training, Mapping):
            raise ConfigError(f"'model_overrides.{name}.training' must be a mapping")
        unknown_training = set(training) - _SECTION_KEYS["training"]
        if unknown_training:
            names = ", ".join(sorted(str(item) for item in unknown_training))
            raise ConfigError(
                f"Unknown key(s) in 'model_overrides.{name}.training': {names}"
            )


def _merge_raw(base: Mapping[str, Any], override: Mapping[str, Any]) -> dict[str, Any]:
    merged: dict[str, Any] = copy.deepcopy(dict(base))
    for section in _SECTION_KEYS:
        value = override.get(section)
        if value is None:
            continue
        previous = merged.get(section, {})
        if not isinstance(previous, Mapping):
            previous = {}
        merged[section] = {**dict(previous), **dict(value)}
    if "models" in override and "model" not in override:
        merged.pop("model", None)
    if "model" in override and "models" not in override:
        merged.pop("models", None)
        merged.pop("model_overrides", None)
    if "models" in override:
        merged["models"] = copy.deepcopy(override["models"])
    if "model_overrides" in override:
        merged["model_overrides"] = _merge_model_overrides(
            merged.get("model_overrides", {}), override["model_overrides"]
        )
    return merged


def _merge_model_overrides(base: object, override: object) -> dict[str, Any]:
    base_mapping = dict(base) if isinstance(base, Mapping) else {}
    override_mapping = dict(override) if isinstance(override, Mapping) else {}
    merged = copy.deepcopy(base_mapping)
    for raw_name, raw_override in override_mapping.items():
        name = str(raw_name)
        previous = merged.get(name, {})
        previous_mapping = dict(previous) if isinstance(previous, Mapping) else {}
        current_mapping = (
            dict(raw_override) if isinstance(raw_override, Mapping) else {}
        )
        model_override = copy.deepcopy(previous_mapping)
        for section in _MODEL_OVERRIDE_KEYS:
            if section not in current_mapping:
                continue
            previous_section = model_override.get(section, {})
            previous_values = (
                dict(previous_section) if isinstance(previous_section, Mapping) else {}
            )
            current_section = current_mapping[section]
            current_values = (
                dict(current_section) if isinstance(current_section, Mapping) else {}
            )
            model_override[section] = {**previous_values, **current_values}
        merged[name] = model_override
    return merged


def _load_raw_with_defaults(path: Path) -> dict[str, Any]:
    raw = _read_yaml(path)
    _validate_raw_mapping(raw, source=path)

    extends = raw.get("extends")
    if extends is not None:
        if not isinstance(extends, str) or not extends.strip():
            raise ConfigError("'extends' must be a non-empty relative YAML path")
        base_path = (path.parent / extends).resolve()
        base_raw = _read_yaml(base_path)
        _validate_raw_mapping(base_raw, source=base_path)
        if "extends" in base_raw:
            raise ConfigError("Only one config inheritance level is supported")
        merged = _merge_raw(_DEFAULT_CONFIG, base_raw)
    else:
        merged = copy.deepcopy(_DEFAULT_CONFIG)

    resolved = _merge_raw(merged, raw)
    task = resolved.get("task")
    if isinstance(task, dict) and "metrics" not in task:
        task["metrics"] = (
            ["roc_auc", "prc_auc", "accuracy", "balanced_accuracy"]
            if task.get("type") == "binary_classification"
            else ["rmse", "mae", "r2"]
        )
    return resolved


def _required_mapping(raw: Mapping[str, Any], section: str) -> dict[str, Any]:
    value = raw.get(section)
    if not isinstance(value, Mapping):
        raise ConfigError(f"Missing section '{section}'")
    return dict(value)


def _string(value: object, *, field: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ConfigError(f"'{field}' must be a non-empty string")
    return value.strip()


def _optional_string(value: object, *, field: str) -> str | None:
    if value is None:
        return None
    return _string(value, field=field)


def _integer(value: object, *, field: str, minimum: int | None = None) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise ConfigError(f"'{field}' must be an integer")
    if minimum is not None and value < minimum:
        raise ConfigError(f"'{field}' must be >= {minimum}")
    return value


def _number(value: object, *, field: str, minimum: float | None = None) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ConfigError(f"'{field}' must be a number")
    converted = float(value)
    if not math.isfinite(converted):
        raise ConfigError(f"'{field}' must be finite")
    if minimum is not None and converted < minimum:
        raise ConfigError(f"'{field}' must be >= {minimum}")
    return converted


def _strings(value: object, *, field: str) -> tuple[str, ...]:
    if isinstance(value, (str, bytes)) or not isinstance(value, Sequence):
        raise ConfigError(f"'{field}' must be a non-empty list of strings")
    result = tuple(_string(item, field=field) for item in value)
    if not result:
        raise ConfigError(f"'{field}' must not be empty")
    if len(set(result)) != len(result):
        raise ConfigError(f"'{field}' must not contain duplicates")
    return result


def _integers(value: object, *, field: str) -> tuple[int, ...]:
    if isinstance(value, (str, bytes)) or not isinstance(value, Sequence):
        raise ConfigError(f"'{field}' must be a non-empty list of integers")
    result = tuple(_integer(item, field=field) for item in value)
    if not result:
        raise ConfigError(f"'{field}' must not be empty")
    if len(set(result)) != len(result):
        raise ConfigError(f"'{field}' must not contain duplicates")
    return result


def _path(value: object, *, field: str, config_dir: Path) -> Path:
    raw = _string(value, field=field)
    path = Path(raw)
    return path if path.is_absolute() else (config_dir / path).resolve()


def _ratios(value: object) -> tuple[float, float, float]:
    if (
        isinstance(value, (str, bytes))
        or not isinstance(value, Sequence)
        or len(value) != 3
    ):
        raise ConfigError("'data.split_ratios' must contain exactly three numbers")
    ratios = tuple(
        _number(item, field="data.split_ratios", minimum=0.0) for item in value
    )
    if not math.isclose(sum(ratios), 1.0, rel_tol=0.0, abs_tol=1e-8):
        raise ConfigError("'data.split_ratios' must sum to 1.0")
    return cast(tuple[float, float, float], ratios)


def _model_overrides(
    value: object,
    *,
    selected_models: tuple[str, ...] | None,
) -> Mapping[str, ModelOverrideConfig]:
    if value is None:
        return MappingProxyType({})
    if not isinstance(value, Mapping):
        raise ConfigError("'model_overrides' must be a mapping")
    resolved: dict[str, ModelOverrideConfig] = {}
    for raw_name, raw_override in value.items():
        name = _string(raw_name, field="model_overrides model name")
        if not isinstance(raw_override, Mapping):
            raise ConfigError(f"'model_overrides.{name}' must be a mapping")
        parameters = raw_override.get("parameters", {})
        training = raw_override.get("training", {})
        assert isinstance(parameters, Mapping)
        assert isinstance(training, Mapping)
        parameter_names = tuple(
            _string(key, field=f"model_overrides.{name}.parameters")
            for key in parameters
        )
        managed = sorted(set(parameter_names) & _CONTEXT_PARAMETER_KEYS)
        if managed:
            names = ", ".join(managed)
            raise ConfigError(
                f"model parameter(s) managed by BuildContext cannot be overridden: {names}"
            )
        resolved[name] = ModelOverrideConfig(
            parameters=MappingProxyType(copy.deepcopy(dict(parameters))),
            training=MappingProxyType(copy.deepcopy(dict(training))),
        )
    if selected_models is not None:
        inactive = sorted(set(resolved) - set(selected_models))
        if inactive:
            names = ", ".join(inactive)
            raise ConfigError(
                f"model override(s) are not selected by 'models': {names}"
            )
    return MappingProxyType(resolved)


def _legacy_model(raw: Mapping[str, Any]) -> ModelConfig | None:
    value = raw.get("model")
    if value is None:
        return None
    if not isinstance(value, Mapping):
        raise ConfigError("Section 'model' must be a mapping")
    parameters = value.get("parameters", {})
    if not isinstance(parameters, Mapping):
        raise ConfigError("'model.parameters' must be a mapping")
    return ModelConfig(
        name=_string(value.get("name", _DEFAULT_MODEL["name"]), field="model.name"),
        parameters=MappingProxyType(copy.deepcopy(dict(parameters))),
    )


def _training_config_from_mapping(raw: Mapping[str, object]) -> TrainingConfig:
    monitor_mode = _string(raw["monitor_mode"], field="training.monitor_mode")
    if monitor_mode not in {"min", "max"}:
        raise ConfigError("'training.monitor_mode' must be min or max")
    device = _string(raw["device"], field="training.device")
    if device not in {"cpu", "cuda", "auto"}:
        raise ConfigError("'training.device' must be cpu, cuda, or auto")
    return TrainingConfig(
        epochs=_integer(raw["epochs"], field="training.epochs", minimum=1),
        batch_size=_integer(raw["batch_size"], field="training.batch_size", minimum=1),
        learning_rate=_number(
            raw["learning_rate"], field="training.learning_rate", minimum=0.0
        ),
        weight_decay=_number(
            raw["weight_decay"], field="training.weight_decay", minimum=0.0
        ),
        patience=_integer(raw["patience"], field="training.patience", minimum=1),
        monitor=_string(raw["monitor"], field="training.monitor"),
        monitor_mode=cast(Literal["min", "max"], monitor_mode),
        device=cast(Literal["cpu", "cuda", "auto"], device),
        num_workers=_integer(
            raw["num_workers"], field="training.num_workers", minimum=0
        ),
    )


def _validate_training_monitor(
    training: TrainingConfig, metrics: Sequence[str]
) -> None:
    allowed_monitor_names = {"loss", "train_loss", "val_loss", *metrics}
    allowed_monitor_names.update({f"train_{metric}" for metric in metrics})
    allowed_monitor_names.update({f"val_{metric}" for metric in metrics})
    if training.monitor not in allowed_monitor_names:
        names = ", ".join(sorted(allowed_monitor_names))
        raise ConfigError(
            f"Unsupported training.monitor '{training.monitor}'. Choose one of: {names}"
        )


def resolve_training_config(
    base: TrainingConfig,
    overrides: Mapping[str, object],
    task: TaskConfig,
) -> TrainingConfig:
    """Apply and validate one model's training overrides."""
    if not isinstance(base, TrainingConfig):
        raise ConfigError("base training config must be a TrainingConfig")
    if not isinstance(overrides, Mapping):
        raise ConfigError("training overrides must be a mapping")
    unknown = set(overrides) - _SECTION_KEYS["training"]
    if unknown:
        names = ", ".join(sorted(str(name) for name in unknown))
        raise ConfigError(f"Unknown training override key(s): {names}")
    raw = {field.name: getattr(base, field.name) for field in fields(base)}
    raw.update(overrides)
    training = _training_config_from_mapping(raw)
    _validate_training_monitor(training, task.metrics)
    return training


def load_config(path: Path) -> ResolvedConfig:
    """Load, merge, resolve, and validate one experiment YAML file."""
    config_path = path.expanduser().resolve()
    raw = _load_raw_with_defaults(config_path)

    experiment_raw = _required_mapping(raw, "experiment")
    data_raw = _required_mapping(raw, "data")
    training_raw = _required_mapping(raw, "training")
    task_raw = _required_mapping(raw, "task")

    configured_models = raw.get("models")
    models = (
        None
        if configured_models is None
        else _strings(configured_models, field="models")
    )
    legacy_model = _legacy_model(raw)
    raw_model_overrides = raw.get("model_overrides", {})
    if legacy_model is not None and "models" in raw:
        raise ConfigError("legacy 'model' cannot be combined with 'models'")
    if legacy_model is not None and raw_model_overrides:
        raise ConfigError("legacy 'model' cannot be combined with 'model_overrides'")
    if legacy_model is not None:
        models = (legacy_model.name,)
        model_overrides: Mapping[str, ModelOverrideConfig] = MappingProxyType(
            {
                legacy_model.name: ModelOverrideConfig(
                    parameters=legacy_model.parameters,
                    training=MappingProxyType({}),
                )
            }
        )
        compatibility_model = legacy_model
    else:
        model_overrides = _model_overrides(
            raw_model_overrides,
            selected_models=models,
        )
        compatibility_name = models[0] if models else str(_DEFAULT_MODEL["name"])
        compatibility_override = model_overrides.get(compatibility_name)
        compatibility_parameters = (
            {} if compatibility_override is None else compatibility_override.parameters
        )
        compatibility_model = ModelConfig(
            name=compatibility_name,
            parameters=MappingProxyType(dict(compatibility_parameters)),
        )

    experiment_name = _string(experiment_raw["name"], field="experiment.name")
    if (
        "/" in experiment_name
        or "\\" in experiment_name
        or ".." in Path(experiment_name).parts
    ):
        raise ConfigError("'experiment.name' must not contain path traversal")
    configured_seed = _integer(experiment_raw["seed"], field="experiment.seed")
    configured_seeds = experiment_raw.get("seeds")
    seeds = (
        _integers(configured_seeds, field="experiment.seeds")
        if configured_seeds is not None
        else (configured_seed,)
    )
    experiment = ExperimentConfig(
        name=experiment_name,
        seed=seeds[0],
        output_dir=_path(
            experiment_raw["output_dir"],
            field="experiment.output_dir",
            config_dir=config_path.parent,
        ),
        seeds=seeds,
    )

    target_columns = _strings(data_raw["target_columns"], field="data.target_columns")
    split = _string(data_raw["split"], field="data.split")
    if split not in {"random", "scaffold", "predefined"}:
        raise ConfigError("'data.split' must be random, scaffold, or predefined")
    split_column = _optional_string(
        data_raw.get("split_column"), field="data.split_column"
    )
    if split == "predefined" and split_column is None:
        raise ConfigError(
            "'data.split_column' is required when data.split is predefined"
        )
    invalid_smiles = _string(data_raw["invalid_smiles"], field="data.invalid_smiles")
    if invalid_smiles not in {"error", "skip"}:
        raise ConfigError("'data.invalid_smiles' must be error or skip")

    data = DataConfig(
        path=_path(data_raw["path"], field="data.path", config_dir=config_path.parent),
        smiles_column=_string(data_raw["smiles_column"], field="data.smiles_column"),
        target_columns=target_columns,
        id_column=_optional_string(data_raw.get("id_column"), field="data.id_column"),
        split=cast(Literal["random", "scaffold", "predefined"], split),
        split_ratios=_ratios(data_raw["split_ratios"]),
        split_column=split_column,
        invalid_smiles=cast(Literal["error", "skip"], invalid_smiles),
    )

    training = _training_config_from_mapping(training_raw)

    task_type = _string(task_raw["type"], field="task.type")
    loss = _string(task_raw["loss"], field="task.loss")
    metrics = _strings(task_raw["metrics"], field="task.metrics")
    target_scaling = task_raw["target_scaling"]
    if not isinstance(target_scaling, bool):
        raise ConfigError("'task.target_scaling' must be a boolean")
    if task_type not in {"regression", "binary_classification"}:
        raise ConfigError("'task.type' must be regression or binary_classification")
    expected_loss = "mse" if task_type == "regression" else "bce_with_logits"
    if loss != expected_loss:
        raise ConfigError(f"'{task_type}' requires task.loss='{expected_loss}'")
    allowed_metrics = (
        _REGRESSION_METRICS if task_type == "regression" else _CLASSIFICATION_METRICS
    )
    unknown_metrics = set(metrics) - allowed_metrics
    if unknown_metrics:
        names = ", ".join(sorted(unknown_metrics))
        raise ConfigError(f"Unsupported metric(s) for {task_type}: {names}")
    if task_type == "binary_classification" and target_scaling:
        raise ConfigError(
            "'task.target_scaling' must be false for binary_classification"
        )
    task = TaskConfig(
        type=cast(Literal["regression", "binary_classification"], task_type),
        loss=cast(Literal["mse", "bce_with_logits"], loss),
        metrics=metrics,
        target_scaling=target_scaling,
    )

    _validate_training_monitor(training, metrics)

    return ResolvedConfig(
        experiment=experiment,
        data=data,
        model=compatibility_model,
        training=training,
        task=task,
        models=models,
        model_overrides=model_overrides,
    )


def to_serializable_dict(config: ResolvedConfig) -> dict[str, Any]:
    """Convert a resolved config to YAML/JSON-safe built-in values."""

    def convert(value: object) -> object:
        if is_dataclass(value):
            return {
                field.name: convert(getattr(value, field.name))
                for field in fields(value)
            }
        if isinstance(value, Path):
            return str(value)
        if isinstance(value, Mapping):
            return {str(key): convert(item) for key, item in value.items()}
        if isinstance(value, tuple):
            return [convert(item) for item in value]
        if isinstance(value, list):
            return [convert(item) for item in value]
        return value

    return cast(dict[str, Any], convert(config))
