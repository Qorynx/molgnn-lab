"""ViSNet's coordinate-derived radius graph."""

from __future__ import annotations

import torch
from torch import Tensor

from ..data import MolecularData
from ..models.visnet_2023.constants import (
    VISNET_CUTOFF,
    VISNET_EPS,
    VISNET_MAX_ATOMIC_NUMBER,
    VISNET_MAX_NEIGHBORS,
)
from .base import TransformError, geometry_is_proxy, with_shared_geometry


def add_visnet_inputs(data: MolecularData) -> MolecularData:
    """Attach ViSNet's capped, self-looped spatial graph to one sample.

    Shared 3-D coordinates are preserved and only ViSNet's capped spatial
    topology is derived here.
    """

    sample = _sample_id(data)
    if getattr(data, "batch", None) is not None:
        raise TransformError(
            f"sample {sample} is already batched; ViSNet inputs must be derived before batching"
        )
    atomic_number = getattr(data, "atomic_number", None)
    pos = getattr(data, "pos", None)
    if atomic_number is None and pos is None:
        data = with_shared_geometry(data)
        atomic_number = data.atomic_number
        pos = data.pos
    _validate_native_inputs(atomic_number, pos, sample=sample)
    assert isinstance(atomic_number, Tensor)
    assert isinstance(pos, Tensor)

    transformed = data.clone()
    transformed.atomic_number = atomic_number
    transformed.pos = pos
    transformed.visnet_edge_index = _radius_edge_index(pos)
    transformed.visnet_geometry_is_proxy = torch.tensor(
        [geometry_is_proxy(data)], dtype=torch.bool, device=pos.device
    )
    return transformed


def _validate_native_inputs(
    atomic_number: object, pos: object, *, sample: int | str
) -> None:
    if not isinstance(atomic_number, Tensor):
        raise TransformError(
            f"sample {sample} requires atomic_number for native ViSNet geometry"
        )
    if (
        atomic_number.ndim != 1
        or atomic_number.numel() < 1
        or atomic_number.dtype != torch.long
        or bool((atomic_number < 1).any())
        or bool((atomic_number > VISNET_MAX_ATOMIC_NUMBER).any())
    ):
        raise TransformError(
            "sample "
            f"{sample} atomic_number must be a non-empty long tensor of elements 1--{VISNET_MAX_ATOMIC_NUMBER}"
        )
    if not isinstance(pos, Tensor):
        raise TransformError(f"sample {sample} requires pos for native ViSNet geometry")
    if (
        pos.shape != (atomic_number.shape[0], 3)
        or pos.dtype != torch.float32
        or pos.device != atomic_number.device
        or not bool(torch.isfinite(pos).all())
    ):
        raise TransformError(
            f"sample {sample} pos must have shape [N, 3] finite float32 on the atomic_number device"
        )
    if _has_coincident_distinct_atoms(pos):
        raise TransformError(f"sample {sample} contains coincident distinct atoms")


def _radius_edge_index(pos: Tensor) -> Tensor:
    """Return source-to-target ViSNet neighbors, including one self-loop/node.

    The source uses ``radius_graph(loop=True, max_num_neighbors=32)``.  This
    local equivalent fixes the source-id tie break and limits every target's
    incoming edges including its loop.  With a cap, directed neighborhoods need
    not be reciprocal.
    """

    node_count = pos.shape[0]
    source_ids = torch.arange(node_count, dtype=torch.long, device=pos.device)
    source_parts: list[Tensor] = []
    target_parts: list[Tensor] = []
    chunk_size = 256
    for start in range(0, node_count, chunk_size):
        stop = min(start + chunk_size, node_count)
        distances = torch.cdist(pos[start:stop], pos, p=2)
        for local_target in range(stop - start):
            target = start + local_target
            candidate_distances = distances[local_target]
            keep = candidate_distances < VISNET_CUTOFF
            candidates = source_ids[keep]
            ordered = torch.argsort(candidate_distances[keep], stable=True)
            selected = candidates[ordered[:VISNET_MAX_NEIGHBORS]]
            # The target itself is distance zero, therefore first under the
            # deterministic sort and always retained by the positive cap.
            source_parts.append(selected)
            target_parts.append(torch.full_like(selected, target))
    return torch.stack((torch.cat(source_parts), torch.cat(target_parts)), dim=0)


def _has_coincident_distinct_atoms(pos: Tensor) -> bool:
    node_count = pos.shape[0]
    if node_count < 2:
        return False
    node_ids = torch.arange(node_count, dtype=torch.long, device=pos.device)
    chunk_size = 256
    for start in range(0, node_count, chunk_size):
        stop = min(start + chunk_size, node_count)
        distances = torch.cdist(pos[start:stop], pos, p=2)
        for local_target in range(stop - start):
            target = start + local_target
            if bool(
                ((node_ids != target) & (distances[local_target] <= VISNET_EPS)).any()
            ):
                return True
    return False


def _sample_id(data: MolecularData) -> int | str:
    value = getattr(data, "sample_id", None)
    if isinstance(value, Tensor) and value.numel():
        return int(value.flatten()[0].item())
    return "<unknown>"


__all__ = ["add_visnet_inputs"]
