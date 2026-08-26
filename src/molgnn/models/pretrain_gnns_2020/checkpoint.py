"""Strict loading of the official Pretrain-GNNs checkpoints.

The four released chemistry artifacts are pure 57-tensor encoder state dicts
(``torch.save(model.gnn.state_dict())``), so loading is safe under
``weights_only=True`` after SHA-256 verification. The official
``x_embedding2`` carries three chirality rows while the paper defines four
categories; the loader expands it to the runtime vocabulary by copying the
``unspecified`` row into a new ``other`` row (row index 3). Scratch remains
the default initialization everywhere.
"""

from __future__ import annotations

import hashlib
from pathlib import Path

import torch

from .layers import (
    NUM_ATOM_TYPE,
    OFFICIAL_CHIRALITY_TAG,
    RUNTIME_CHIRALITY_TAG,
)

CHECKPOINT_DIR = Path(__file__).resolve().parents[4] / "pretrained" / "pretrain_gnns_2020"

EXPECTED_CHECKSUMS: dict[str, str] = {
    "contextpred": "9538CEF4C19EA734BE8490A8EDA8E17F9CB5DF76A77EF069C6A8D5F5B8195564",
    "masking": "77DD7F4EEEB16200BE1D001411972D0EDB04BD406023CA8AB4EAA18F772AA9E4",
    "supervised_contextpred": "107197F159E9A26ED9A026B655560AC341D2D7DDEC4A8D2540E83C9F87E256AC",
    "supervised_masking": "375CD40AF9F21D2A92ED1ACBDEA9EFAD14254C36703BB0E3A7E433E09E624CE1",
}
EXPECTED_TENSOR_COUNT = 57


class CheckpointError(ValueError):
    """Raised when an official checkpoint cannot be consumed safely."""


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest().upper()


def resolve_checkpoint_path(
    variant: str | None = None, checkpoint_path: str | Path | None = None
) -> tuple[Path | None, str]:
    """Resolve an explicit path or a named pinned variant."""

    if checkpoint_path is not None:
        if variant not in {None, "none"}:
            raise CheckpointError(
                "variant and checkpoint_path are mutually exclusive"
            )
        return Path(checkpoint_path), "explicit"
    if variant is not None and variant != "none":
        if variant not in EXPECTED_CHECKSUMS:
            raise CheckpointError(f"unknown pretrained variant {variant!r}")
        return CHECKPOINT_DIR / f"{variant}.pth", variant
    return None, "scratch"


def _validate_and_adapt(state: dict[str, torch.Tensor], emb_dim: int) -> dict[str, torch.Tensor]:
    non_tensors = [key for key, value in state.items() if not isinstance(value, torch.Tensor)]
    if non_tensors:
        raise CheckpointError(f"non-tensor entries in checkpoint: {non_tensors[:4]}")
    if len(state) != EXPECTED_TENSOR_COUNT:
        raise CheckpointError(
            f"expected {EXPECTED_TENSOR_COUNT} tensors, found {len(state)}"
        )
    expected_shapes = {
        "x_embedding1.weight": (NUM_ATOM_TYPE, emb_dim),
        "x_embedding2.weight": (OFFICIAL_CHIRALITY_TAG, emb_dim),
        "gnns.0.mlp.0.weight": (2 * emb_dim, emb_dim),
        "gnns.0.mlp.2.weight": (emb_dim, 2 * emb_dim),
        "gnns.4.edge_embedding1.weight": (6, emb_dim),
        "gnns.4.edge_embedding2.weight": (3, emb_dim),
        "batch_norms.4.running_mean": (emb_dim,),
    }
    for key, shape in expected_shapes.items():
        value = state.get(key)
        if not isinstance(value, torch.Tensor) or tuple(value.shape) != shape:
            found = None if not isinstance(value, torch.Tensor) else tuple(value.shape)
            raise CheckpointError(f"unexpected schema at '{key}': expected {shape}, found {found}")

    adapted = dict(state)
    official_rows = state["x_embedding2.weight"]
    # The `other` row copies `unspecified` (row 0): CHI_OTHER never occurs in
    # the source featurization practice, and this preserves row parity for
    # the three supported ids exactly.
    adapted["x_embedding2.weight"] = torch.cat(
        [official_rows, official_rows[0:1]],
        dim=0,
    )
    return adapted


def load_pretrained_encoder(
    encoder,
    *,
    variant: str | None = None,
    checkpoint_path: str | Path | None = None,
    expected_emb_dim: int | None = None,
) -> dict[str, object]:
    """Verify, adapt, and strict-load one official encoder checkpoint."""

    path, resolved_variant = resolve_checkpoint_path(variant, checkpoint_path)
    if path is None:
        raise CheckpointError("no checkpoint requested")
    if not path.is_file():
        raise CheckpointError(f"checkpoint file does not exist: {path}")
    actual_sha256 = _sha256(path)
    if resolved_variant != "explicit":
        expected = EXPECTED_CHECKSUMS[resolved_variant]
    else:
        expected = next(
            (value for value in EXPECTED_CHECKSUMS.values() if value == actual_sha256),
            actual_sha256,
        )
        resolved_variant = next(
            (name for name, value in EXPECTED_CHECKSUMS.items() if value == actual_sha256),
            "explicit-unmatched",
        )
    if actual_sha256 != expected:
        raise CheckpointError(
            f"checksum mismatch: expected {expected}, got {actual_sha256}"
        )

    state = torch.load(path, map_location="cpu", weights_only=True)
    if not isinstance(state, dict):
        raise CheckpointError("checkpoint must contain a state-dict mapping")

    probe = state.get("x_embedding1.weight")
    if not isinstance(probe, torch.Tensor) or probe.dim() != 2:
        raise CheckpointError("checkpoint is missing the atomic-number embedding")
    inferred_dim = int(probe.shape[1])
    if expected_emb_dim is not None and inferred_dim != expected_emb_dim:
        raise CheckpointError(
            f"checkpoint embedding dim {inferred_dim} != model dim {expected_emb_dim}"
        )
    adapted = _validate_and_adapt(state, inferred_dim)

    encoder_chirality = getattr(encoder.x_embedding2, "num_embeddings", None)
    if encoder_chirality == OFFICIAL_CHIRALITY_TAG:
        adapted["x_embedding2.weight"] = adapted["x_embedding2.weight"][
            :OFFICIAL_CHIRALITY_TAG
        ]
    elif encoder_chirality != RUNTIME_CHIRALITY_TAG:
        raise CheckpointError(
            f"encoder chirality vocabulary {encoder_chirality} is unsupported"
        )

    try:
        encoder.load_state_dict(adapted, strict=True)
    except RuntimeError as exc:
        raise CheckpointError(f"strict load failed: {exc}") from exc
    return {
        "variant": resolved_variant,
        "sha256": actual_sha256,
        "loaded_tensors": len(adapted),
        "chirality_adapted": True,
    }


def load_pretrained_encoder_for_variant(encoder, variant: str) -> dict[str, object]:
    return load_pretrained_encoder(encoder, variant=variant)


__all__ = [
    "EXPECTED_CHECKSUMS",
    "CheckpointError",
    "load_pretrained_encoder",
    "load_pretrained_encoder_for_variant",
    "resolve_checkpoint_path",
]
