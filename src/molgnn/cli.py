"""Command-line entry points for the MolGNN project."""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Sequence
from pathlib import Path
from typing import TYPE_CHECKING

from . import __version__

if TYPE_CHECKING:
    from .registry import ModelSpec


def build_parser() -> argparse.ArgumentParser:
    """Build the top-level CLI parser without importing heavy dependencies."""
    parser = argparse.ArgumentParser(
        prog="molgnn",
        description="Architecture-only framework for molecular and complex GNNs.",
    )
    parser.add_argument(
        "--version", action="version", version=f"%(prog)s {__version__}"
    )

    subparsers = parser.add_subparsers(dest="command", metavar="COMMAND")

    validate_parser = subparsers.add_parser(
        "validate-config",
        help="Validate an experiment YAML configuration.",
    )
    validate_parser.add_argument("--config", required=True, help="Path to YAML config.")
    validate_parser.set_defaults(handler=_validate_config)

    train_parser = subparsers.add_parser(
        "train",
        help="Run one configured experiment.",
    )
    train_parser.add_argument("--config", required=True, help="Path to YAML config.")
    train_parser.add_argument(
        "--featurizer",
        metavar="SPEC",
        help="Custom featurizer as module_or_path.py:top_level_callable.",
    )
    train_parser.add_argument(
        "--training-strategy",
        metavar="SPEC",
        help="Custom training strategy as module_or_path.py:top_level_callable.",
    )
    train_parser.set_defaults(handler=_train)

    benchmark_parser = subparsers.add_parser(
        "benchmark",
        help="Run the selected multi-model benchmark from a YAML config.",
    )
    benchmark_parser.add_argument(
        "--config", required=True, help="Path to YAML config."
    )
    benchmark_parser.set_defaults(handler=_benchmark)

    describe_parser = subparsers.add_parser(
        "describe-model",
        help="Show the runtime integration contract for one built-in model.",
    )
    describe_parser.add_argument(
        "--model", required=True, help="Registered model name."
    )
    describe_parser.add_argument(
        "--format",
        dest="output_format",
        choices=("text", "json"),
        default="text",
        help="Output format (default: text).",
    )
    describe_parser.set_defaults(handler=_describe_model)

    return parser


def _missing_config(args: argparse.Namespace, *, command: str) -> int:
    """Report a missing experiment configuration."""
    print(
        f"{command} error: config file does not exist: {args.config}", file=sys.stderr
    )
    return 2


def _train(args: argparse.Namespace) -> int:
    """Load a resolved config and execute the shared experiment runner."""
    from .config import ConfigError, load_config
    from .hooks import HookError, load_hook
    from .runner import run_experiments

    config_path = Path(args.config)
    if not config_path.exists():
        return _missing_config(args, command="Train")
    try:
        config = load_config(config_path)
        featurizer_spec = getattr(args, "featurizer", None)
        training_strategy_spec = getattr(args, "training_strategy", None)
        featurizer = (
            load_hook(featurizer_spec, option="--featurizer")
            if featurizer_spec
            else None
        )
        training_strategy = (
            load_hook(training_strategy_spec, option="--training-strategy")
            if training_strategy_spec
            else None
        )
        run_dirs = run_experiments(
            config,
            featurizer=featurizer,
            training_strategy=training_strategy,
        )
    except (ConfigError, HookError, ValueError, RuntimeError, OSError) as exc:
        print(f"Train error: {exc}", file=sys.stderr)
        return 2
    for run_dir in run_dirs:
        print(f"Completed: {run_dir}")
    return 0


def _benchmark(args: argparse.Namespace) -> int:
    """Load one multi-model config and run its benchmark lifecycle."""

    from .config import ConfigError, load_config
    from .runner import run_benchmark

    config_path = Path(args.config)
    if not config_path.exists():
        return _missing_config(args, command="Benchmark")
    try:
        result = run_benchmark(load_config(config_path))
    except (ConfigError, ValueError, RuntimeError, OSError) as exc:
        print(f"Benchmark error: {exc}", file=sys.stderr)
        return 2

    for run_dir in result.completed:
        print(f"Completed: {run_dir}")
    for failure in result.failed:
        print(
            "Failed: "
            f"model={failure.model_name} seed={failure.seed} stage={failure.stage} "
            f"{failure.error_type}: {failure.error_message}",
            file=sys.stderr,
        )
    return 1 if result.failed else 0


def _validate_config(args: argparse.Namespace) -> int:
    """Validate a YAML config and print a concise result."""
    from .config import ConfigError, load_config

    try:
        config = load_config(Path(args.config))
    except ConfigError as exc:
        print(f"Config error: {exc}", file=sys.stderr)
        return 2
    print(
        f"Valid config: {args.config} "
        f"(experiment={config.experiment.name}, seeds={list(config.experiment.seeds)}, "
        f"targets={config.num_targets})"
    )
    return 0


def _describe_model(args: argparse.Namespace) -> int:
    """Print a model's runtime integration contract."""
    from .models.registration import register_builtin_models
    from .registry import RegistryError, get_model_spec

    try:
        # Normal package import stays lightweight; architecture modules are only
        # imported when a user explicitly asks to inspect a model.
        register_builtin_models()
        spec = get_model_spec(args.model)
    except RegistryError as exc:
        print(f"Model description error: {exc}", file=sys.stderr)
        return 2

    description = _runtime_model_contract(spec)
    if args.output_format == "json":
        print(json.dumps(description, indent=2, sort_keys=True))
    else:
        print(_format_model_description(description))
    return 0


def _runtime_model_contract(spec: ModelSpec) -> dict[str, object]:
    """Convert runtime registry metadata to a stable CLI-friendly mapping."""
    return {
        "name": spec.name,
        "required_batch_fields": list(spec.required_batch_fields),
        "optional_batch_fields": list(spec.optional_batch_fields),
        "graph_transform_name": spec.graph_transform_name,
        "prediction_reducer_name": spec.prediction_reducer_name,
        "geometry_requirement": spec.geometry_requirement,
        "geometry_role": spec.geometry_role,
        "benchmark_enabled": spec.benchmark_enabled,
        "benchmark_order": spec.benchmark_order,
    }


def _format_model_description(description: dict[str, object]) -> str:
    """Render one runtime contract in a concise, stable text layout."""
    lines = [f"Model: {description['name']}"]
    lines.append(
        "Required batch fields: "
        + _format_inline_items(description["required_batch_fields"])
    )
    lines.append(
        "Optional batch fields: "
        + _format_inline_items(description["optional_batch_fields"])
    )
    lines.append(
        "Graph transform: "
        + _format_optional_value(description["graph_transform_name"])
    )
    lines.append(f"Prediction reducer: {description['prediction_reducer_name']}")
    lines.append(f"Geometry requirement: {description['geometry_requirement']}")
    lines.append(f"Geometry role: {description['geometry_role']}")
    benchmark_enabled = description["benchmark_enabled"]
    if not isinstance(benchmark_enabled, bool):
        raise TypeError("benchmark_enabled must be a boolean")
    lines.append(f"Benchmark enabled: {str(benchmark_enabled).lower()}")
    lines.append(f"Benchmark order: {description['benchmark_order']}")
    return "\n".join(lines)


def _format_inline_items(values: object) -> str:
    """Render an item list without leaking Python list syntax into text output."""
    if not isinstance(values, list) or not values:
        return "<none>"
    return ", ".join(str(value) for value in values)


def _format_optional_value(value: object) -> str:
    """Render absent optional runtime fields consistently."""
    return "<none>" if value is None else str(value)


def main(argv: Sequence[str] | None = None) -> int:
    """Run the CLI and return a process exit code."""
    parser = build_parser()
    args = parser.parse_args(argv)
    handler = getattr(args, "handler", None)
    if handler is None:
        parser.print_help()
        return 0
    try:
        return int(handler(args))
    except Exception as exc:
        print(f"Unexpected error: {exc}", file=sys.stderr)
        return 1
