"""Safe GraphMVP encoder checkpoint loading and legacy conversion."""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path

import torch
from torch import Tensor, nn


class GraphMVPCheckpointError(ValueError):
    """Raised when a GraphMVP checkpoint cannot match an encoder profile."""


def _extract_state(payload: object) -> dict[str, Tensor]:
    if isinstance(payload, Mapping):
        for key in ("state_dict", "model", "molecule_model", "encoder"):
            nested = payload.get(key)
            if isinstance(nested, Mapping):
                try:
                    return _extract_state(nested)
                except GraphMVPCheckpointError:
                    pass
        state = payload
    else:
        raise GraphMVPCheckpointError("checkpoint must contain a state dictionary")
    result: dict[str, Tensor] = {}
    for key, value in state.items():
        if not isinstance(key, str) or not isinstance(value, Tensor):
            continue
        clean = key
        for prefix in ("module.", "model.", "molecule_model.", "encoder."):
            clean = clean.removeprefix(prefix)
        result[clean] = value
    if not result:
        raise GraphMVPCheckpointError("checkpoint state dictionary is empty")
    return result


def load_graphmvp_encoder(encoder: nn.Module, path: str | Path) -> dict[str, object]:
    """Load a source/project encoder and adapt the known 3->4 chirality table."""

    checkpoint_path = Path(path).expanduser().resolve()
    if not checkpoint_path.is_file():
        raise GraphMVPCheckpointError(f"checkpoint does not exist: {checkpoint_path}")
    try:
        payload = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    except Exception as exc:
        raise GraphMVPCheckpointError(f"could not load checkpoint safely: {exc}") from exc
    state = _extract_state(payload)
    target = encoder.state_dict()
    adapted = False
    source_chiral = state.get("x_embedding2.weight")
    target_chiral = target.get("x_embedding2.weight")
    if isinstance(source_chiral, Tensor) and isinstance(target_chiral, Tensor):
        if source_chiral.shape == target_chiral.shape:
            pass
        elif (
            source_chiral.ndim == 2
            and target_chiral.ndim == 2
            and source_chiral.shape[0] == 3
            and target_chiral.shape[0] == 4
            and source_chiral.shape[1] == target_chiral.shape[1]
        ):
            replacement = target_chiral.clone()
            replacement[:3] = source_chiral
            replacement[3] = source_chiral[0]
            state["x_embedding2.weight"] = replacement
            adapted = True
        else:
            raise GraphMVPCheckpointError(
                f"incompatible chirality embedding shape {tuple(source_chiral.shape)}"
            )
    missing, unexpected = encoder.load_state_dict(state, strict=False)
    allowed_missing = ("x_embedding2.weight",) if adapted else ()
    remaining_missing = tuple(name for name in missing if name not in allowed_missing)
    if remaining_missing or unexpected:
        raise GraphMVPCheckpointError(
            f"incompatible {getattr(encoder, 'feature_profile', 'unknown')} checkpoint; "
            f"missing={remaining_missing}, unexpected={tuple(unexpected)}"
        )
    return {
        "path": str(checkpoint_path),
        "adapted_chirality": adapted,
        "tensor_count": len(state),
        "feature_profile": getattr(encoder, "feature_profile", "unknown"),
    }


__all__ = ["GraphMVPCheckpointError", "load_graphmvp_encoder"]
