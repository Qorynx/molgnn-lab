"""Safe encoder checkpoint loading for HiMol."""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path

import torch
from torch import Tensor, nn


class HiMolCheckpointError(ValueError):
    """Raised when a checkpoint is absent or incompatible with HiMol."""


def bundled_checkpoint_path(variant: str) -> Path:
    if variant != "zinc250k":
        raise HiMolCheckpointError("pretrained_variant must be 'none' or 'zinc250k'")
    return (
        Path(__file__).resolve().parents[4]
        / "pretrained"
        / "himol_2023"
        / "pretrain.pth"
    )


def load_himol_encoder(encoder: nn.Module, path: str | Path) -> dict[str, object]:
    checkpoint_path = Path(path).expanduser().resolve()
    if not checkpoint_path.is_file():
        raise HiMolCheckpointError(f"checkpoint does not exist: {checkpoint_path}")
    try:
        payload = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    except Exception as exc:
        raise HiMolCheckpointError(f"could not load checkpoint safely: {exc}") from exc
    state = _extract_state(payload)
    expected = encoder.state_dict()
    shape_errors = {
        key: (tuple(value.shape), tuple(expected[key].shape))
        for key, value in state.items()
        if key in expected and value.shape != expected[key].shape
    }
    if shape_errors:
        raise HiMolCheckpointError(
            f"checkpoint tensor shapes are incompatible: {shape_errors}"
        )
    missing, unexpected = encoder.load_state_dict(state, strict=False)
    if missing or unexpected:
        raise HiMolCheckpointError(
            f"incompatible encoder checkpoint; missing={tuple(missing)}, "
            f"unexpected={tuple(unexpected)}"
        )
    return {
        "path": str(checkpoint_path),
        "tensor_count": len(state),
        "hierarchy_profile": "paper_bidirectional",
    }


def _extract_state(payload: object) -> dict[str, Tensor]:
    if not isinstance(payload, Mapping):
        raise HiMolCheckpointError("checkpoint must contain a state dictionary")
    for nested_key in ("state_dict", "model_state_dict", "encoder", "gnn", "model"):
        nested = payload.get(nested_key)
        if isinstance(nested, Mapping):
            try:
                return _extract_state(nested)
            except HiMolCheckpointError:
                pass
    state: dict[str, Tensor] = {}
    for key, value in payload.items():
        if not isinstance(key, str) or not isinstance(value, Tensor):
            continue
        clean = key
        for prefix in ("module.", "model.", "encoder.", "gnn."):
            clean = clean.removeprefix(prefix)
        state[clean] = value
    if not state:
        raise HiMolCheckpointError("checkpoint state dictionary is empty")
    return state


__all__ = ["HiMolCheckpointError", "bundled_checkpoint_path", "load_himol_encoder"]
