"""Strict, allowlisted conversion of official Paddle ChemRL-GEM weights."""

from __future__ import annotations

import pickle
from collections import OrderedDict
from collections.abc import Mapping
from pathlib import Path

import numpy as np
import torch
from numpy.core.multiarray import _reconstruct, scalar
from torch import Tensor, nn


class ChemRLGEMCheckpointError(ValueError):
    """Raised when an official GEM encoder state is not compatible."""


class _SafePaddleUnpickler(pickle.Unpickler):
    """Allow only NumPy reconstruction primitives used by ``.pdparams``."""

    def find_class(self, module: str, name: str) -> object:
        if module == "numpy" and name in {"dtype", "ndarray"}:
            return getattr(np, name)
        if module in {"numpy.core.multiarray", "numpy._core.multiarray"} and name in {
            "_reconstruct",
            "scalar",
        }:
            return _reconstruct if name == "_reconstruct" else scalar
        if module == "collections" and name == "OrderedDict":
            return OrderedDict
        raise ChemRLGEMCheckpointError(f"checkpoint contains disallowed pickle global {module}.{name}")


def _read_paddle_state(path: Path) -> dict[str, np.ndarray]:
    try:
        with path.open("rb") as stream:
            payload = _SafePaddleUnpickler(stream).load()
    except ChemRLGEMCheckpointError:
        raise
    except Exception as exc:
        raise ChemRLGEMCheckpointError(f"could not safely read Paddle checkpoint: {exc}") from exc
    if not isinstance(payload, Mapping):
        raise ChemRLGEMCheckpointError("Paddle checkpoint must contain a mapping")
    result: dict[str, np.ndarray] = {}
    for key, value in payload.items():
        if not isinstance(key, str) or not isinstance(value, np.ndarray):
            raise ChemRLGEMCheckpointError("Paddle checkpoint contains a non-NumPy tensor")
        if value.dtype not in (np.float32, np.float64):
            raise ChemRLGEMCheckpointError(f"unsupported dtype for tensor {key!r}: {value.dtype}")
        result[key] = value
    if not result:
        raise ChemRLGEMCheckpointError("Paddle checkpoint is empty")
    return result


def _is_paddle_linear_weight(key: str) -> bool:
    return key.endswith(".mlp.0.weight") or key.endswith(".mlp.2.weight") or key.endswith(".linear_list.0.weight")


def convert_chemrl_gem_state(
    encoder: nn.Module,
    path: str | Path,
) -> tuple[dict[str, Tensor], dict[str, object]]:
    """Convert a Paddle encoder state with exact key and shape checks."""

    checkpoint_path = Path(path).expanduser().resolve()
    if not checkpoint_path.is_file():
        raise ChemRLGEMCheckpointError(f"checkpoint does not exist: {checkpoint_path}")
    if checkpoint_path.suffix.lower() == ".pdparams":
        source = _read_paddle_state(checkpoint_path)
        converted: dict[str, Tensor] = {}
        target = encoder.state_dict()
        if set(source) != set(target):
            missing = sorted(set(target) - set(source))
            unexpected = sorted(set(source) - set(target))
            raise ChemRLGEMCheckpointError(
                f"incompatible encoder keys; missing={missing}, unexpected={unexpected}"
            )
        for key, target_tensor in target.items():
            value = torch.from_numpy(np.asarray(source[key])).to(dtype=target_tensor.dtype)
            if _is_paddle_linear_weight(key):
                value = value.t().contiguous()
            if tuple(value.shape) != tuple(target_tensor.shape):
                raise ChemRLGEMCheckpointError(
                    f"shape mismatch for {key}: source={tuple(value.shape)}, target={tuple(target_tensor.shape)}"
                )
            converted[key] = value
        return converted, {
            "path": str(checkpoint_path),
            "format": "paddle_state_dict_pickle_numpy",
            "transposed_linear_weights": sum(_is_paddle_linear_weight(key) for key in target),
            "tensor_count": len(converted),
        }

    try:
        payload = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    except Exception as exc:
        raise ChemRLGEMCheckpointError(f"could not load converted checkpoint: {exc}") from exc
    if isinstance(payload, Mapping) and "state_dict" in payload:
        payload = payload["state_dict"]
    if not isinstance(payload, Mapping) or not all(isinstance(key, str) and isinstance(value, Tensor) for key, value in payload.items()):
        raise ChemRLGEMCheckpointError("converted checkpoint must contain a Torch state dictionary")
    target = encoder.state_dict()
    if set(payload) != set(target):
        raise ChemRLGEMCheckpointError("converted checkpoint keys do not exactly match the encoder")
    converted = {key: value.detach().to(dtype=target[key].dtype, device="cpu") for key, value in payload.items()}
    for key, value in converted.items():
        if tuple(value.shape) != tuple(target[key].shape):
            raise ChemRLGEMCheckpointError(f"shape mismatch for converted tensor {key}")
    return converted, {"path": str(checkpoint_path), "format": "torch_state_dict", "tensor_count": len(converted)}


def load_chemrl_gem_encoder(encoder: nn.Module, path: str | Path) -> dict[str, object]:
    """Load one class/regression encoder checkpoint strictly into ``encoder``."""

    state, info = convert_chemrl_gem_state(encoder, path)
    try:
        encoder.load_state_dict(state, strict=True)
    except RuntimeError as exc:
        raise ChemRLGEMCheckpointError(f"strict encoder load failed: {exc}") from exc
    return info


def convert_chemrl_gem_checkpoint(
    encoder: nn.Module,
    source_path: str | Path,
    output_path: str | Path,
) -> dict[str, object]:
    """Convert and persist a safe Torch state dict for repeatable local use."""

    state, info = convert_chemrl_gem_state(encoder, source_path)
    destination = Path(output_path).expanduser().resolve()
    destination.parent.mkdir(parents=True, exist_ok=True)
    torch.save(state, destination)
    return {**info, "output_path": str(destination), "output_format": "torch_state_dict"}


__all__ = [
    "ChemRLGEMCheckpointError",
    "convert_chemrl_gem_checkpoint",
    "convert_chemrl_gem_state",
    "load_chemrl_gem_encoder",
]

