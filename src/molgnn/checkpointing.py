"""Atomic, schema-validated checkpoint persistence."""

from __future__ import annotations

import os
import pickle
import tempfile
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import torch
from torch import nn
from torch.optim import Optimizer


class CheckpointError(RuntimeError):
    """Raised for corrupt, incomplete, or incompatible checkpoints."""


CHECKPOINT_FORMAT_VERSION = 1
_REQUIRED_KEYS = frozenset(
    {
        "format_version",
        "epoch",
        "monitor_name",
        "monitor_value",
        "model_state_dict",
        "optimizer_state_dict",
        "resolved_config",
        "feature_schema_version",
        "target_scaler_state",
    }
)


def save_checkpoint_atomic(
    path: str | Path,
    *,
    epoch: int,
    monitor_name: str,
    monitor_value: float,
    model_state_dict: Mapping[str, Any],
    optimizer_state_dict: Mapping[str, Any],
    resolved_config: Mapping[str, Any],
    feature_schema_version: str,
    target_scaler_state: Mapping[str, Any] | None = None,
    rng_state: Mapping[str, Any] | None = None,
) -> Path:
    """Write a versioned checkpoint through a same-directory atomic replace."""
    checkpoint_path = Path(path).expanduser().resolve()
    if isinstance(epoch, bool) or not isinstance(epoch, int) or epoch < 0:
        raise CheckpointError("checkpoint epoch must be a non-negative integer")
    if not isinstance(monitor_name, str) or not monitor_name.strip():
        raise CheckpointError("checkpoint monitor_name must be non-empty")
    if not isinstance(feature_schema_version, str) or not feature_schema_version.strip():
        raise CheckpointError("checkpoint feature_schema_version must be non-empty")
    if not torch.isfinite(torch.tensor(float(monitor_value))):
        raise CheckpointError("checkpoint monitor_value must be finite")
    payload: dict[str, Any] = {
        "format_version": CHECKPOINT_FORMAT_VERSION,
        "epoch": epoch,
        "monitor_name": monitor_name,
        "monitor_value": float(monitor_value),
        "model_state_dict": dict(model_state_dict),
        "optimizer_state_dict": dict(optimizer_state_dict),
        "resolved_config": dict(resolved_config),
        "feature_schema_version": feature_schema_version,
        "target_scaler_state": None if target_scaler_state is None else dict(target_scaler_state),
        "rng_state": None if rng_state is None else dict(rng_state),
    }
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="wb",
            dir=checkpoint_path.parent,
            prefix=f".{checkpoint_path.name}.",
            suffix=".tmp",
            delete=False,
        ) as stream:
            temporary_path = Path(stream.name)
            torch.save(payload, stream)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary_path, checkpoint_path)
    except (
        OSError,
        RuntimeError,
        pickle.PickleError,
        pickle.UnpicklingError,
        TypeError,
        ValueError,
        AttributeError,
    ) as exc:
        if temporary_path is not None:
            temporary_path.unlink(missing_ok=True)
        raise CheckpointError(f"failed to save checkpoint '{checkpoint_path}': {exc}") from exc
    return checkpoint_path


def load_checkpoint(
    path: str | Path,
    *,
    expected_feature_schema_version: str | None = None,
) -> dict[str, Any]:
    """Load and validate one checkpoint payload."""
    checkpoint_path = Path(path).expanduser().resolve()
    if not checkpoint_path.is_file():
        raise CheckpointError(f"checkpoint does not exist: '{checkpoint_path}'")
    try:
        payload = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    except (
        OSError,
        RuntimeError,
        EOFError,
        ImportError,
        AttributeError,
        pickle.PickleError,
        TypeError,
        ValueError,
    ) as exc:
        raise CheckpointError(f"cannot load checkpoint '{checkpoint_path}': {exc}") from exc
    validated = _validate_payload(payload)
    if (
        expected_feature_schema_version is not None
        and validated["feature_schema_version"] != expected_feature_schema_version
    ):
        raise CheckpointError(
            "checkpoint feature schema mismatch: "
            f"expected {expected_feature_schema_version!r}, "
            f"got {validated['feature_schema_version']!r}"
        )
    return validated


def restore_model(
    model: nn.Module,
    checkpoint: Mapping[str, Any] | str | Path,
    *,
    optimizer: Optimizer | None = None,
    expected_feature_schema_version: str | None = None,
) -> dict[str, Any]:
    """Restore model and optional optimizer state, returning the payload."""
    payload = (
        load_checkpoint(checkpoint, expected_feature_schema_version=expected_feature_schema_version)
        if isinstance(checkpoint, (str, Path))
        else _validate_payload(checkpoint)
    )
    try:
        model.load_state_dict(payload["model_state_dict"])
        if optimizer is not None:
            optimizer.load_state_dict(payload["optimizer_state_dict"])
    except (RuntimeError, ValueError, TypeError) as exc:
        raise CheckpointError(
            f"checkpoint state is incompatible with model/optimizer: {exc}"
        ) from exc
    return payload


def _validate_payload(payload: object) -> dict[str, Any]:
    if not isinstance(payload, Mapping):
        raise CheckpointError("checkpoint payload must be a mapping")
    missing = _REQUIRED_KEYS - set(payload)
    if missing:
        names = ", ".join(sorted(missing))
        raise CheckpointError(f"checkpoint is missing required key(s): {names}")
    if payload["format_version"] != CHECKPOINT_FORMAT_VERSION:
        raise CheckpointError(
            f"unsupported checkpoint format_version: {payload['format_version']!r}"
        )
    if isinstance(payload["epoch"], bool) or not isinstance(payload["epoch"], int):
        raise CheckpointError("checkpoint epoch must be an integer")
    if not isinstance(payload["monitor_name"], str):
        raise CheckpointError("checkpoint monitor_name must be a string")
    if not isinstance(payload["feature_schema_version"], str):
        raise CheckpointError("checkpoint feature_schema_version must be a string")
    if not isinstance(payload["model_state_dict"], Mapping):
        raise CheckpointError("checkpoint model_state_dict must be a mapping")
    if not isinstance(payload["optimizer_state_dict"], Mapping):
        raise CheckpointError("checkpoint optimizer_state_dict must be a mapping")
    if not isinstance(payload["resolved_config"], Mapping):
        raise CheckpointError("checkpoint resolved_config must be a mapping")
    if payload["target_scaler_state"] is not None and not isinstance(
        payload["target_scaler_state"], Mapping
    ):
        raise CheckpointError("checkpoint target_scaler_state must be a mapping or null")
    try:
        monitor_value = float(payload["monitor_value"])
    except (TypeError, ValueError) as exc:
        raise CheckpointError("checkpoint monitor_value must be numeric") from exc
    if not torch.isfinite(torch.tensor(monitor_value)):
        raise CheckpointError("checkpoint monitor_value must be finite")
    return dict(payload)


__all__ = [
    "CHECKPOINT_FORMAT_VERSION",
    "CheckpointError",
    "load_checkpoint",
    "restore_model",
    "save_checkpoint_atomic",
]
