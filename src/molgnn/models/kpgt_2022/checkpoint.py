"""Explicit pretrained-backbone loading for KPGT.

Official checkpoints (Figshare ``base.pth``) store a plain
``LiGhTPredictor.state_dict()`` from the DGL implementation; the downstream
predictor heads are replaced at fine-tuning time in the official workflow,
so only backbone weights are consumed here. Scratch initialization stays the
default: loading happens exclusively through an explicit path. The internal
official artifact is stored at ``pretrained/kpgt_2022/base.pth`` and
documented by its adjacent manifest.
"""

from __future__ import annotations

from pathlib import Path

import torch
from torch import Tensor

from .constants import kpgt_vocab_size
from .model import KPGT


class CheckpointError(ValueError):
    """Raised when a pretrained checkpoint cannot be consumed safely."""


_PREDICTOR_KEY_PREFIXES = (
    # Official pretraining heads plus this port's downstream head; all are
    # (re)initialized at prediction time and never consumed from checkpoints.
    "node_predictor.",
    "fp_predictor.",
    "md_predictor.",
    "predictor.",
)


def _strip_module_prefix(state: dict[str, Tensor]) -> dict[str, Tensor]:
    return {key.removeprefix("module."): value for key, value in state.items()}


def _load_raw_state(path: str | Path) -> dict[str, Tensor]:
    location = Path(path)
    if not location.is_file():
        raise CheckpointError(f"checkpoint file does not exist: {location}")
    try:
        payload = torch.load(location, map_location="cpu", weights_only=True)
    except Exception as exc:
        raise CheckpointError(f"checkpoint could not be loaded safely: {exc}") from exc
    if not isinstance(payload, dict):
        raise CheckpointError("checkpoint must contain a state-dict mapping")
    if isinstance(payload.get("model_state_dict"), dict):
        payload = payload["model_state_dict"]
    elif isinstance(payload.get("state_dict"), dict):
        payload = payload["state_dict"]
    state = _strip_module_prefix(payload)
    non_tensors = [key for key, value in state.items() if not isinstance(value, Tensor)]
    if non_tensors:
        raise CheckpointError(f"checkpoint contains non-tensor entries: {non_tensors[:4]}")
    return state


def infer_checkpoint_profile(state: dict[str, Tensor]) -> dict[str, int]:
    """Derive architecture dimensions from official tensor shapes."""

    def shape(key: str) -> tuple[int, ...]:
        value = state.get(key)
        if not isinstance(value, Tensor):
            raise CheckpointError(f"checkpoint is missing required tensor '{key}'")
        return tuple(value.shape)

    qkv = shape("model.mol_T_layers.0.qkv.weight")
    d_g_feats = qkv[1]
    n_heads = shape("model.dist_attn_layer.2.weight")[0]
    path_length_embedding = shape("model.path_len_emb.weight")[0]
    d_trip_path = shape("model.trip_fortrans.0.out_proj.weight")[0]
    d_node_feats = shape("node_emb.in_proj.weight")[1]
    d_edge_feats = shape("edge_emb.in_proj.weight")[1]
    d_fp_feats = shape("triplet_emb.fp_proj.in_proj.weight")[1]
    d_md_feats = shape("triplet_emb.md_proj.in_proj.weight")[1]

    mol_layers = sorted(
        {
            int(key.split(".")[2])
            for key in state
            if key.startswith("model.mol_T_layers.")
        }
    )
    if mol_layers != list(range(len(mol_layers))):
        raise CheckpointError("checkpoint transformer layers are not contiguous")

    trip_positions = sorted(
        {int(key.split(".")[2]) for key in state if key.startswith("model.trip_fortrans.")}
    )
    if trip_positions != list(range(len(trip_positions))):
        raise CheckpointError("checkpoint path projections are not contiguous")

    profile = {
        "d_g_feats": d_g_feats,
        "n_heads": n_heads,
        "path_length": path_length_embedding - 1,
        "d_hpath_ratio": d_g_feats // d_trip_path if d_trip_path else 0,
        "n_mol_layers": len(mol_layers),
        # FFN holds in_proj/out_proj plus one Linear per extra hidden layer.
        "n_ffn_dense_layers": len(
            [
                key
                for key in state
                if key.startswith("model.mol_T_layers.0.node_out_layer.ffn.")
            ]
        )
        // 2,
        "d_node_feats": d_node_feats,
        "d_edge_feats": d_edge_feats,
        "d_fp_feats": d_fp_feats,
        "d_md_feats": d_md_feats,
    }
    if profile["path_length"] != len(trip_positions):
        raise CheckpointError("checkpoint path-length metadata is inconsistent")
    if d_g_feats % d_trip_path or profile["d_hpath_ratio"] * d_trip_path != d_g_feats:
        raise CheckpointError("checkpoint hidden/path ratio is inconsistent")
    return profile


def load_pretrained_backbone(model: KPGT, path: str | Path) -> dict[str, object]:
    """Load official backbone weights into a downstream :class:`KPGT`.

    Only an explicit caller-provided ``path`` is read. The loader strips
    ``module.`` prefixes, validates every remaining key and shape against the
    target model, skips the three official pretraining heads (replaced at
    downstream time), and refuses partial backbone loads.
    """

    state = _load_raw_state(path)
    profile = infer_checkpoint_profile(state)
    expected = {
        "d_g_feats": model.d_g_feats,
        "path_length": model.path_length,
    }
    mismatched = {
        name: (provided, wanted)
        for name, wanted in expected.items()
        for provided in [profile[name]]
        if provided != wanted
    }
    if mismatched:
        details = ", ".join(f"{name}: checkpoint={a} model={b}" for name, (a, b) in mismatched.items())
        raise CheckpointError(f"checkpoint profile does not match model ({details})")

    target = model.state_dict()
    skipped: list[str] = []
    matched: dict[str, Tensor] = {}
    for key, value in state.items():
        if key.startswith(_PREDICTOR_KEY_PREFIXES):
            skipped.append(key)
            continue
        target_value = target.get(key)
        if target_value is None:
            raise CheckpointError(f"checkpoint tensor '{key}' has no counterpart in the model")
        if tuple(target_value.shape) != tuple(value.shape):
            raise CheckpointError(
                f"shape mismatch for '{key}': checkpoint {tuple(value.shape)} "
                f"vs model {tuple(target_value.shape)}"
            )
        matched[key] = value

    missing_backbone = [
        key for key in target if key not in matched and not key.startswith("predictor.")
    ]
    if missing_backbone:
        raise CheckpointError(
            f"checkpoint is missing backbone tensor(s): {missing_backbone[:8]}"
        )

    incompatible = model.load_state_dict(matched, strict=False)
    unexpected = [key for key in incompatible.unexpected_keys]
    if unexpected:
        raise CheckpointError(f"unexpected tensors after load: {unexpected[:8]}")
    return {
        "loaded_tensors": len(matched),
        "skipped_head_tensors": len(skipped),
        "profile": profile,
        "expected_vocab_size": kpgt_vocab_size(),
    }


__all__ = ["CheckpointError", "infer_checkpoint_profile", "load_pretrained_backbone"]
