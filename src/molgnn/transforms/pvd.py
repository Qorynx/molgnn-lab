"""Coordinate-derived graph for pre-training via denoising TorchMD-ET."""

from __future__ import annotations

import torch
from torch import Tensor

from ..data import MolecularData
from ..models.pvd_2023.constants import (
    PVD_CUTOFF_LOWER,
    PVD_CUTOFF_UPPER,
    PVD_MAX_ATOMIC_NUMBER,
    PVD_MAX_NUM_NEIGHBORS,
)
from ..models.pvd_2023.geometry import build_pvd_radius_graph
from .base import TransformError, geometry_is_proxy, with_shared_geometry


def add_pvd_inputs(data: MolecularData) -> MolecularData:
    """Attach TorchMD-ET's looped, capped directed radius graph."""

    sample = _sample_id(data)
    if getattr(data, "batch", None) is not None:
        raise TransformError(
            f"sample {sample} is already batched; PVD inputs must be derived before batching"
        )
    atomic_number = getattr(data, "atomic_number", None)
    pos = getattr(data, "pos", None)
    if atomic_number is None and pos is None:
        data = with_shared_geometry(data)
        atomic_number = data.atomic_number
        pos = data.pos
    _validate_inputs(atomic_number, pos, sample=sample)
    assert isinstance(atomic_number, Tensor)
    assert isinstance(pos, Tensor)

    transformed = data.clone()
    transformed.atomic_number = atomic_number
    transformed.pos = pos
    transformed.pvd_edge_index = build_pvd_radius_graph(
        pos,
        cutoff_lower=PVD_CUTOFF_LOWER,
        cutoff_upper=PVD_CUTOFF_UPPER,
        max_num_neighbors=PVD_MAX_NUM_NEIGHBORS,
        loop=True,
    )
    transformed.pvd_geometry_is_proxy = torch.tensor(
        [geometry_is_proxy(data)], dtype=torch.bool, device=pos.device
    )
    return transformed


def _validate_inputs(
    atomic_number: object,
    pos: object,
    *,
    sample: int | str,
) -> None:
    if not isinstance(atomic_number, Tensor):
        raise TransformError(f"sample {sample} requires atomic_number")
    if (
        atomic_number.ndim != 1
        or atomic_number.numel() < 1
        or atomic_number.dtype != torch.long
        or bool((atomic_number <= 0).any())
        or bool((atomic_number > PVD_MAX_ATOMIC_NUMBER).any())
    ):
        raise TransformError(
            f"sample {sample} atomic_number must be positive long [N] and <= {PVD_MAX_ATOMIC_NUMBER}"
        )
    if not isinstance(pos, Tensor):
        raise TransformError(f"sample {sample} requires pos")
    if (
        pos.shape != (atomic_number.shape[0], 3)
        or pos.dtype != torch.float32
        or pos.device != atomic_number.device
        or not bool(torch.isfinite(pos).all())
    ):
        raise TransformError(
            f"sample {sample} pos must be finite float32 [N, 3] on the atomic_number device"
        )


def _sample_id(data: MolecularData) -> int | str:
    value = getattr(data, "sample_id", None)
    if isinstance(value, Tensor) and value.numel():
        return int(value.flatten()[0].item())
    return "<unknown>"


__all__ = ["add_pvd_inputs"]
