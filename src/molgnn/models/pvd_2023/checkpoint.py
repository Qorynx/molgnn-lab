"""Safe conversion/loading for the official Lightning denoising checkpoint."""

from __future__ import annotations

import hashlib
from collections.abc import Mapping
from pathlib import Path

import torch
from torch import Tensor, nn

CHECKPOINT_DIR = Path(__file__).resolve().parents[4] / "pretrained" / "pvd_2023"
DEFAULT_CHECKPOINT = CHECKPOINT_DIR / "denoised-pcqm4mv2.weights.pt"
OFFICIAL_LEGACY_SHA256 = (
    "F6B387BA3632E03D273939257969161B2615A7FBF41E9F514FD25E1B5D345E66"
)
CONVERTED_SHA256 = (
    "76CA33BC4E784D8F4E74F454FEF0562D1D68433D7B3D02BBB53938771207AA5D"
)

_EXPECTED_HPARAMETERS: dict[str, object] = {
    "model": "equivariant-transformer",
    "embedding_dimension": 256,
    "num_layers": 8,
    "num_rbf": 64,
    "rbf_type": "expnorm",
    "trainable_rbf": False,
    "activation": "silu",
    "attn_activation": "silu",
    "neighbor_embedding": True,
    "num_heads": 8,
    "distance_influence": "both",
    "cutoff_lower": 0.0,
    "cutoff_upper": 5.0,
    "max_z": 100,
    "max_num_neighbors": 32,
    "layernorm_on_vec": "whitened",
    "output_model_noise": "VectorOutput",
    "position_noise_scale": 0.04,
    "denoising_only": True,
}


class PVDCheckpointError(ValueError):
    """Raised when a PVD checkpoint is unsafe or incompatible."""


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest().upper()


def _safe_load_legacy(path: Path) -> Mapping[str, object]:
    if _sha256(path) != OFFICIAL_LEGACY_SHA256:
        raise PVDCheckpointError("legacy checkpoint SHA-256 does not match official artifact")
    # Lightning serialized callback state. Placeholder types are allowlisted
    # only so ``weights_only=True`` can reconstruct inert dictionaries; no
    # callback package or arbitrary unpickling is used.
    early_stopping = type("EarlyStopping", (), {})
    early_stopping.__module__ = "pytorch_lightning.callbacks.early_stopping"
    model_checkpoint = type("ModelCheckpoint", (), {})
    model_checkpoint.__module__ = "pytorch_lightning.callbacks.model_checkpoint"
    try:
        with torch.serialization.safe_globals((early_stopping, model_checkpoint)):
            payload = torch.load(path, map_location="cpu", weights_only=True)
    except Exception as exc:
        raise PVDCheckpointError(f"could not safely load legacy checkpoint: {exc}") from exc
    if not isinstance(payload, Mapping):
        raise PVDCheckpointError("legacy checkpoint must contain a mapping")
    hyperparameters = payload.get("hyper_parameters")
    if not isinstance(hyperparameters, Mapping):
        raise PVDCheckpointError("legacy checkpoint is missing hyper_parameters")
    mismatches = {
        key: (hyperparameters.get(key), expected)
        for key, expected in _EXPECTED_HPARAMETERS.items()
        if hyperparameters.get(key) != expected
    }
    if mismatches:
        raise PVDCheckpointError(f"legacy checkpoint profile mismatch: {mismatches}")
    return payload


def _convert_source_state(source: Mapping[object, object]) -> dict[str, Tensor]:
    converted: dict[str, Tensor] = {}
    for key, value in source.items():
        if not isinstance(key, str) or not isinstance(value, Tensor):
            continue
        if key.startswith("model.representation_model."):
            clean = "encoder." + key.removeprefix("model.representation_model.")
        elif key.startswith("model.output_model_noise."):
            clean = "noise_head." + key.removeprefix("model.output_model_noise.")
        elif key.startswith("model.pos_normalizer."):
            clean = "position_normalizer." + key.removeprefix("model.pos_normalizer.")
        else:
            continue
        converted[clean] = value.detach().cpu()
    if not converted:
        raise PVDCheckpointError("legacy checkpoint has no transferable tensors")
    return converted


def _read_state(path: Path) -> tuple[dict[str, Tensor], dict[str, object]]:
    if not path.is_file():
        raise PVDCheckpointError(f"checkpoint does not exist: {path}")
    if path.suffix.lower() == ".ckpt":
        payload = _safe_load_legacy(path)
        source = payload.get("state_dict")
        if not isinstance(source, Mapping):
            raise PVDCheckpointError("legacy checkpoint is missing state_dict")
        return _convert_source_state(source), {
            "format": "official_lightning",
            "source_sha256": OFFICIAL_LEGACY_SHA256,
            "global_step": int(payload.get("global_step", -1)),
        }
    try:
        payload = torch.load(path, map_location="cpu", weights_only=True)
    except Exception as exc:
        raise PVDCheckpointError(f"could not safely load converted checkpoint: {exc}") from exc
    metadata: dict[str, object] = {"format": "pvd_weight_only"}
    if isinstance(payload, Mapping) and isinstance(payload.get("metadata"), Mapping):
        metadata.update(dict(payload["metadata"]))
    state: object = payload.get("state_dict") if isinstance(payload, Mapping) else None
    if not isinstance(state, Mapping):
        raise PVDCheckpointError("converted checkpoint must contain state_dict")
    if not all(isinstance(key, str) and isinstance(value, Tensor) for key, value in state.items()):
        raise PVDCheckpointError("converted state_dict must contain only named tensors")
    return dict(state), metadata


def _strict_subset(
    module: nn.Module,
    state: Mapping[str, Tensor],
    *,
    prefix: str,
) -> int:
    selected = {
        key.removeprefix(prefix): value
        for key, value in state.items()
        if key.startswith(prefix)
    }
    target = module.state_dict()
    if set(selected) != set(target):
        missing = sorted(set(target) - set(selected))
        unexpected = sorted(set(selected) - set(target))
        raise PVDCheckpointError(
            f"incompatible {prefix.rstrip('.')} keys; "
            f"missing={missing}, unexpected={unexpected}"
        )
    for key, value in selected.items():
        if value.shape != target[key].shape:
            raise PVDCheckpointError(
                f"shape mismatch for {prefix}{key}: "
                f"source={tuple(value.shape)}, target={tuple(target[key].shape)}"
            )
    module.load_state_dict(selected, strict=True)
    return len(selected)


def resolve_pvd_checkpoint(
    *,
    variant: str = "none",
    checkpoint_path: str | Path | None = None,
) -> tuple[Path | None, str]:
    if variant not in {"none", "pcqm4mv2"}:
        raise PVDCheckpointError("pretrained_variant must be none or pcqm4mv2")
    if checkpoint_path is not None and variant != "none":
        raise PVDCheckpointError("variant and checkpoint_path are mutually exclusive")
    if checkpoint_path is not None:
        return Path(checkpoint_path).expanduser().resolve(), "explicit"
    if variant == "pcqm4mv2":
        return DEFAULT_CHECKPOINT, variant
    return None, "scratch"


def load_pvd_pretrained(
    model: nn.Module,
    *,
    variant: str = "none",
    checkpoint_path: str | Path | None = None,
    include_noise_head: bool = False,
) -> dict[str, object]:
    path, resolved_variant = resolve_pvd_checkpoint(
        variant=variant, checkpoint_path=checkpoint_path
    )
    if path is None:
        return {"variant": "scratch", "tensor_count": 0}
    if resolved_variant == "pcqm4mv2" and CONVERTED_SHA256 is not None:
        if _sha256(path) != CONVERTED_SHA256:
            raise PVDCheckpointError("converted checkpoint SHA-256 mismatch")
    state, metadata = _read_state(path)
    encoder = getattr(model, "encoder", None)
    noise_head = getattr(model, "noise_head", None)
    if not isinstance(encoder, nn.Module):
        raise PVDCheckpointError("target model has no TorchMD-ET encoder")
    tensor_count = _strict_subset(encoder, state, prefix="encoder.")
    if include_noise_head:
        if not isinstance(noise_head, nn.Module):
            raise PVDCheckpointError("target model has no denoising head")
        tensor_count += _strict_subset(noise_head, state, prefix="noise_head.")
    return {
        **metadata,
        "path": str(path),
        "variant": resolved_variant,
        "tensor_count": tensor_count,
        "noise_head_loaded": include_noise_head,
    }


def load_pvd_pretrainer(pretrainer: nn.Module, path: str | Path) -> dict[str, object]:
    model = getattr(pretrainer, "model", None)
    normalizer = getattr(pretrainer, "position_normalizer", None)
    if not isinstance(model, nn.Module) or not isinstance(normalizer, nn.Module):
        raise PVDCheckpointError("target is not a PVD pretrainer")
    state, metadata = _read_state(Path(path).expanduser().resolve())
    encoder_count = _strict_subset(model.encoder, state, prefix="encoder.")
    noise_count = _strict_subset(model.noise_head, state, prefix="noise_head.")
    norm_count = _strict_subset(normalizer, state, prefix="position_normalizer.")
    return {
        **metadata,
        "path": str(Path(path).expanduser().resolve()),
        "tensor_count": encoder_count + noise_count + norm_count,
        "noise_head_loaded": True,
        "normalizer_loaded": True,
    }


def convert_official_pvd_checkpoint(
    source_path: str | Path,
    output_path: str | Path,
) -> dict[str, object]:
    """Strip Lightning/optimizer state into an auditable weight-only payload."""

    source = Path(source_path).expanduser().resolve()
    destination = Path(output_path).expanduser().resolve()
    payload = _safe_load_legacy(source)
    source_state = payload.get("state_dict")
    if not isinstance(source_state, Mapping):
        raise PVDCheckpointError("legacy checkpoint is missing state_dict")
    state = _convert_source_state(source_state)
    metadata = {
        "format_version": 1,
        "model": "pvd_torchmd_et",
        "variant": "pcqm4mv2",
        "source_commit": "2d81667c4daf519a1bfd33f1e1257eefa527db61",
        "source_sha256": OFFICIAL_LEGACY_SHA256,
        "global_step": int(payload.get("global_step", -1)),
        "tensor_count": len(state),
        "profile": dict(_EXPECTED_HPARAMETERS),
    }
    destination.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"metadata": metadata, "state_dict": state}, destination)
    return {
        **metadata,
        "output_path": str(destination),
        "output_sha256": _sha256(destination),
    }


__all__ = [
    "CONVERTED_SHA256",
    "DEFAULT_CHECKPOINT",
    "OFFICIAL_LEGACY_SHA256",
    "PVDCheckpointError",
    "convert_official_pvd_checkpoint",
    "load_pvd_pretrained",
    "load_pvd_pretrainer",
    "resolve_pvd_checkpoint",
]
