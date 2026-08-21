"""Safe loading and conversion for Mole-BERT encoder checkpoints."""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path

import torch
from torch import Tensor, nn


class MoleBERTCheckpointError(ValueError):
    """Raised when an encoder state is not compatible with Mole-BERT."""


def _state_dict(payload: object) -> dict[str, Tensor]:
    if isinstance(payload, Mapping) and "state_dict" in payload:
        payload = payload["state_dict"]
    if not isinstance(payload, Mapping):
        raise MoleBERTCheckpointError("checkpoint must contain a state dictionary")
    result: dict[str, Tensor] = {}
    for key, value in payload.items():
        if not isinstance(key, str) or not isinstance(value, Tensor):
            continue
        clean = key.removeprefix("module.").removeprefix("gnn.")
        result[clean] = value
    if not result:
        raise MoleBERTCheckpointError("checkpoint state dictionary is empty")
    return result


def load_molebert_encoder(encoder: nn.Module, path: str | Path) -> dict[str, object]:
    """Load a source or project encoder state, adapting source's 3-tag table."""

    checkpoint_path = Path(path).expanduser().resolve()
    if not checkpoint_path.is_file():
        raise MoleBERTCheckpointError(f"checkpoint does not exist: {checkpoint_path}")
    try:
        payload = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    except Exception as exc:
        raise MoleBERTCheckpointError(f"could not load checkpoint: {exc}") from exc
    state = _state_dict(payload)
    target = encoder.state_dict()
    source_chiral = state.get("x_embedding2.weight")
    target_chiral = target.get("x_embedding2.weight")
    adapted = False
    if isinstance(source_chiral, Tensor) and isinstance(target_chiral, Tensor):
        if source_chiral.shape == target_chiral.shape:
            pass
        elif source_chiral.ndim == 2 and target_chiral.ndim == 2 and source_chiral.shape[0] == 3 and target_chiral.shape[0] == 4 and source_chiral.shape[1] == target_chiral.shape[1]:
            replacement = target_chiral.clone()
            replacement[:3] = source_chiral
            replacement[3] = source_chiral[0]
            state["x_embedding2.weight"] = replacement
            adapted = True
        else:
            raise MoleBERTCheckpointError(
                f"incompatible chirality embedding shape {tuple(source_chiral.shape)}"
            )
    missing, unexpected = encoder.load_state_dict(state, strict=False)
    unexpected = tuple(unexpected)
    missing = tuple(missing)
    allowed_missing = ("x_embedding2.weight",) if adapted else ()
    remaining_missing = tuple(name for name in missing if name not in allowed_missing)
    if remaining_missing or unexpected:
        raise MoleBERTCheckpointError(
            f"incompatible encoder checkpoint; missing={remaining_missing}, unexpected={unexpected}"
        )
    return {
        "path": str(checkpoint_path),
        "adapted_chirality": adapted,
        "tensor_count": len(state),
    }


__all__ = ["MoleBERTCheckpointError", "load_molebert_encoder"]
