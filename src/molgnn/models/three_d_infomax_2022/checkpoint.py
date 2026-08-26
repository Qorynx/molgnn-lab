"""Safe conversion and strict loading of official 3D Infomax checkpoints.

The released artifact (``runs/PNA_qmugs_NTXentMultiplePositives_.../
best_checkpoint_35epochs.pt``) is a legacy pickle that stores a handful of
NumPy objects inside the optimizer/scheduler sections. The one-time
converter below loads it with ``torch.load(..., weights_only=True)`` plus a
minimal, pinned allowlist for exactly those NumPy globals — never
``weights_only=False`` — verifies the SHA-256 checksum and the expected
tensor schema, and exports a pure-tensor PNA-encoder state dict.

Runtime loading consumes only converted encoder-only artifacts; Net3D,
optimizer, scheduler and the pretraining readout projection are dropped.
Artifact license status: ``UNKNOWN`` (internal-use policy).
"""

from __future__ import annotations

import hashlib
from pathlib import Path

import torch
from torch import Tensor

EXPECTED_SOURCE_SHA256 = (
    "109587CA4C794F7E4CD2D8EB0E577C81BD794B531332970C6074ECF3B185BAAB"
)
EXPECTED_PNA_TENSOR_COUNT = 168
EXPECTED_NET3D_TENSOR_COUNT = 26
ENCODER_KEY_PREFIX = "node_gnn."


class CheckpointError(ValueError):
    """Raised when an official checkpoint cannot be consumed safely."""


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest().upper()


def _allowlisted_load(path: Path) -> dict[str, object]:
    """Load the legacy artifact under ``weights_only=True`` with pinned globals."""

    import numpy as np
    from numpy._core.multiarray import _reconstruct as numpy_reconstruct
    from numpy._core.multiarray import scalar as numpy_scalar
    from numpy.dtypes import (
        BoolDType,
        Float16DType,
        Float32DType,
        Float64DType,
        Int8DType,
        Int16DType,
        Int32DType,
        Int64DType,
        UInt8DType,
        UInt16DType,
        UInt32DType,
        UInt64DType,
    )

    # NumPy >= 2 resolves these classes under numpy._core while the pickle
    # stream references the historical numpy.core module paths.
    allowlist: list[object] = [
        (numpy_scalar, "numpy.core.multiarray.scalar"),
        (numpy_reconstruct, "numpy.core.multiarray._reconstruct"),
        (np.ndarray, "numpy.ndarray"),
        (np.dtype, "numpy.dtype"),
        BoolDType,
        Float16DType,
        Float32DType,
        Float64DType,
        Int8DType,
        Int16DType,
        Int32DType,
        Int64DType,
        UInt8DType,
        UInt16DType,
        UInt32DType,
        UInt64DType,
    ]
    torch.serialization.add_safe_globals(allowlist)
    try:
        payload = torch.load(path, map_location="cpu", weights_only=True)
    except Exception as exc:
        raise CheckpointError(f"checkpoint could not be loaded safely: {exc}") from exc
    if not isinstance(payload, dict):
        raise CheckpointError("checkpoint must contain a mapping")
    return payload


def _validate_official_payload(payload: dict[str, object]) -> tuple[dict[str, Tensor], dict[str, Tensor]]:
    for section in ("model_state_dict", "model3d_state_dict"):
        if not isinstance(payload.get(section), dict):
            raise CheckpointError(f"checkpoint is missing '{section}'")
    pna = payload["model_state_dict"]
    net3d = payload["model3d_state_dict"]
    assert isinstance(pna, dict) and isinstance(net3d, dict)
    non_tensor = [key for key, value in pna.items() if not isinstance(value, Tensor)]
    if non_tensor:
        raise CheckpointError(f"non-tensor entries in model_state_dict: {non_tensor[:4]}")
    if len(pna) != EXPECTED_PNA_TENSOR_COUNT:
        raise CheckpointError(
            f"expected {EXPECTED_PNA_TENSOR_COUNT} PNA tensors, found {len(pna)}"
        )
    if len(net3d) != EXPECTED_NET3D_TENSOR_COUNT:
        raise CheckpointError(
            f"expected {EXPECTED_NET3D_TENSOR_COUNT} Net3D tensors, found {len(net3d)}"
        )
    expected_shapes = {
        "node_gnn.atom_encoder.atom_embedding_list.0.weight": (119, 200),
        "node_gnn.atom_encoder.atom_embedding_list.4.weight": (10, 200),
        "node_gnn.bond_encoder.bond_embedding_list.0.weight": (5, 200),
        "node_gnn.mp_layers.6.posttrans.fully_connected.0.linear.weight": (200, 2600),
        "node_gnn.mp_layers.6.pretrans.fully_connected.1.linear.weight": (200, 200),
        "output.fully_connected.1.linear.weight": (256, 200),
    }
    for key, shape in expected_shapes.items():
        value = pna.get(key)
        if not isinstance(value, Tensor) or tuple(value.shape) != shape:
            found = None if not isinstance(value, Tensor) else tuple(value.shape)
            raise CheckpointError(
                f"unexpected schema at '{key}': expected {shape}, found {found}"
            )
    net3d_probe = net3d.get("edge_input.fully_connected.0.linear.weight")
    if not isinstance(net3d_probe, Tensor) or tuple(net3d_probe.shape) != (20, 9):
        raise CheckpointError("unexpected Net3D edge-input schema")
    return pna, net3d


def convert_official_checkpoint(
    source_path: str | Path,
    *,
    expected_sha256: str = EXPECTED_SOURCE_SHA256,
) -> dict[str, Tensor]:
    """Verify and decode the official artifact into an encoder-only state.

    Only the PNA encoder tensors survive; the pretraining readout head, Net3D
    weights, optimizer and scheduler are intentionally discarded.
    """

    source = Path(source_path)
    if not source.is_file():
        raise CheckpointError(f"checkpoint file does not exist: {source}")
    actual_sha256 = _sha256(source)
    if actual_sha256 != expected_sha256.upper():
        raise CheckpointError(
            f"checksum mismatch: expected {expected_sha256.upper()}, got {actual_sha256}"
        )
    payload = _allowlisted_load(source)
    pna, _ = _validate_official_payload(payload)
    encoder: dict[str, Tensor] = {}
    for key, value in pna.items():
        if not key.startswith(ENCODER_KEY_PREFIX):
            continue  # pretraining-only readout/output head
        encoder[key[len(ENCODER_KEY_PREFIX) :]] = value.detach().cpu().clone()
    return encoder


MANIFEST_TEMPLATE = """model: three_d_infomax_2022
paper: "3D Infomax for better molecular property prediction (ICML 2022)"
source_repository: "HannesStark/3DInfomax"
source_checkout_revision: "5cd32629c690e119bcae8726acedefdb0aa037fc"
download_url: "bundled with the official repository checkout"
downloaded_at: "2026-08-26"
original_artifact:
  path: "{original_name}"
  size_bytes: {size_bytes}
  sha256: "{sha256}"
converted_artifact:
  path: "{converted_name}"
  tensor_count: {tensor_count}
  scope: "PNA encoder only (atom/bond encoders + propagation layers); Net3D, optimizer, scheduler and pretraining head removed"
profile:
  hidden_dim: 200
  propagation_depth: 7
  aggregators: [mean, max, min, std]
  scalers: [identity, amplification, attenuation]
  avg_d_log_hardcoded: 1.0
  pretrain_target_dim: 256
net3d_profile:
  hidden_dim: 20
  fourier_encodings: 4
  propagation_depth: 1
  reduce_func: mean
  readout_aggregators: [min, max, mean]
license: "UNKNOWN"
license_notes: >-
  Neither the official repository nor the run directory carries a LICENSE
  file. Per internal project policy this blocks nothing locally, but no
  external redistribution right is inferred.
safety_notes: >-
  Load the original artifact only through molgnn's allowlisted converter
  (weights_only=True). Runtime code must consume the converted pure-tensor
  encoder artifact exclusively.
"""


def pin_pretrained_artifacts(source_path: str | Path, output_dir: str | Path) -> dict[str, object]:
    """Copy the verified original next to its converted encoder artifact."""

    import shutil

    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    source = Path(source_path)
    sha256 = _sha256(source)
    original_name = "best_checkpoint_35epochs.pt"
    converted_name = "pna_encoder_qmugs.pt"

    encoder = convert_official_checkpoint(source)
    torch.save(
        {
            "format_version": 1,
            "source_sha256": sha256,
            "scope": "three_d_infomax_pna_encoder",
            "encoder_state": encoder,
        },
        output / converted_name,
    )
    copied = output / original_name
    if not copied.exists() or _sha256(copied) != sha256:
        shutil.copyfile(source, copied)

    manifest = MANIFEST_TEMPLATE.format(
        original_name=original_name,
        size_bytes=source.stat().st_size,
        sha256=sha256,
        converted_name=converted_name,
        tensor_count=len(encoder),
    )
    (output / "manifest.yaml").write_text(manifest, encoding="utf-8")
    readme = (
        "# 3D Infomax official QMugs pretrained checkpoint\n\n"
        f"`{original_name}` pins the official `NTXentMultiplePositives` QMugs run\n"
        "(35 epochs), verified against the manifest SHA-256. "
        f"`{converted_name}` is the converted,\n"
        "pure-tensor PNA encoder used by the runtime loader; regenerate it with\n"
        "`molgnn.models.three_d_infomax_2022.checkpoint.pin_pretrained_artifacts`.\n"
        "License status is UNKNOWN (see manifest.yaml).\n"
    )
    (output / "README.md").write_text(readme, encoding="utf-8")
    return {
        "source": str(source),
        "sha256": sha256,
        "encoder_tensors": len(encoder),
        "converted_path": str(output / converted_name),
        "original_path": str(copied),
        "manifest_path": str(output / "manifest.yaml"),
    }


def load_pretrained_encoder(model, path: str | Path) -> dict[str, object]:
    """Strictly load a converted encoder-only artifact.

    Accepts either a :class:`~molgnn.models.three_d_infomax_2022.ThreeDInfomax`
    predictor or its ``PNAGNN`` encoder directly.
    """

    location = Path(path)
    if not location.is_file():
        raise CheckpointError(f"converted checkpoint does not exist: {location}")
    payload = torch.load(location, map_location="cpu", weights_only=True)
    if not isinstance(payload, dict) or payload.get("scope") != "three_d_infomax_pna_encoder":
        raise CheckpointError(f"not a converted 3D Infomax encoder artifact: {location}")
    state = payload.get("encoder_state")
    if not isinstance(state, dict):
        raise CheckpointError("converted artifact is missing 'encoder_state'")
    target_module = getattr(model, "node_gnn", model)
    target = target_module.state_dict()
    mismatched = {
        key: (tuple(value.shape), tuple(target[key].shape))
        for key, value in state.items()
        if key in target and tuple(value.shape) != tuple(target[key].shape)
    }
    if mismatched:
        details = next(iter(mismatched.items()))
        raise CheckpointError(f"shape mismatch for '{details[0]}': {details[1]}")
    unexpected = sorted(set(state) - set(target))
    missing = sorted(set(target) - set(state))
    if unexpected or missing:
        raise CheckpointError(
            f"incompatible encoder keys; unexpected={unexpected[:4]}, missing={missing[:4]}"
        )
    target_module.load_state_dict(state, strict=True)
    return {"loaded_tensors": len(state), "source_sha256": payload.get("source_sha256")}


__all__ = [
    "EXPECTED_SOURCE_SHA256",
    "CheckpointError",
    "convert_official_checkpoint",
    "load_pretrained_encoder",
    "pin_pretrained_artifacts",
]
