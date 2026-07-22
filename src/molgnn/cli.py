"""Command-line entry points for the MolGNN project."""

from __future__ import annotations

import argparse
import sys
from collections.abc import Sequence
from pathlib import Path

from . import __version__


def build_parser() -> argparse.ArgumentParser:
    """Build the top-level CLI parser without importing heavy dependencies."""
    parser = argparse.ArgumentParser(
        prog="molgnn",
        description="Architecture-only benchmark framework for 2D molecular GNNs.",
    )
    parser.add_argument("--version", action="version", version=f"%(prog)s {__version__}")

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
    train_parser.set_defaults(handler=_train)

    return parser


def _missing_config(args: argparse.Namespace) -> int:
    """Report a missing experiment configuration."""
    print(f"Train error: config file does not exist: {args.config}", file=sys.stderr)
    return 2


def _train(args: argparse.Namespace) -> int:
    """Load a resolved config and execute the shared experiment runner."""
    from .config import ConfigError, load_config
    from .runner import run_experiments

    config_path = Path(args.config)
    if not config_path.exists():
        return _missing_config(args)
    try:
        config = load_config(config_path)
        run_dirs = run_experiments(config)
    except (ConfigError, ValueError, RuntimeError, OSError) as exc:
        print(f"Train error: {exc}", file=sys.stderr)
        return 2
    for run_dir in run_dirs:
        print(f"Completed: {run_dir}")
    return 0


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
