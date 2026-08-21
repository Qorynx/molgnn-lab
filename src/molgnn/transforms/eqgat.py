"""EQGAT's coordinate-derived radius graph."""

from __future__ import annotations

import torch
from torch import Tensor

from ..data import MolecularData
from ..models.eqgat_2022.constants import EQGAT_CUTOFF, EQGAT_MAX_NEIGHBORS
from .base import TransformError, geometry_is_proxy, with_shared_geometry


def add_eqgat_inputs(data: MolecularData) -> MolecularData:
    """Attach EQGAT's capped directed radius graph to one unbatched sample.

    The shared geometry provider supplies coordinates. This transform derives
    only EQGAT's capped directed radius graph.
    """

    sample = _sample_id(data)
    if getattr(data, "batch", None) is not None:
        raise TransformError(
            f"sample {sample} is already batched; EQGAT inputs must be derived before batching"
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
    transformed.eqgat_edge_index = _radius_edge_index(pos)
    transformed.eqgat_geometry_is_proxy = torch.tensor(
        [geometry_is_proxy(data)], dtype=torch.bool, device=pos.device
    )
    return transformed


def _validate_native_inputs(
    atomic_number: object, pos: object, *, sample: int | str
) -> None:
    if not isinstance(atomic_number, Tensor):
        raise TransformError(
            f"sample {sample} requires atomic_number for native EQGAT geometry"
        )
    if (
        atomic_number.ndim != 1
        or atomic_number.numel() < 1
        or atomic_number.dtype != torch.long
        or bool((atomic_number <= 0).any())
    ):
        raise TransformError(
            f"sample {sample} atomic_number must be a non-empty positive long tensor with shape [N]"
        )
    if not isinstance(pos, Tensor):
        raise TransformError(f"sample {sample} requires pos for native EQGAT geometry")
    if (
        pos.shape != (atomic_number.shape[0], 3)
        or pos.dtype != torch.float32
        or pos.device != atomic_number.device
        or not bool(torch.isfinite(pos).all())
    ):
        raise TransformError(
            f"sample {sample} pos must have shape [N, 3] finite float32 on the atomic_number device"
        )


def _radius_edge_index(pos: Tensor) -> Tensor:
    """Return nearest-first directed edges ``j -> i`` in EQGAT's radius.

    The official ATOM3D transform limits each target to 32 neighbors.  We
    compute distances in small target chunks so topology construction does not
    retain an ``N x N x 3`` displacement tensor, then use source-ID order as a
    deterministic tie break for equal distances.
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
            keep = (source_ids != target) & (distances[local_target] < EQGAT_CUTOFF)
            candidates = source_ids[keep]
            if candidates.numel() == 0:
                continue
            candidate_distances = distances[local_target, keep]
            order = torch.argsort(candidate_distances, stable=True)
            selected = candidates[order[:EQGAT_MAX_NEIGHBORS]]
            source_parts.append(selected)
            target_parts.append(torch.full_like(selected, target))
    if not source_parts:
        return torch.empty((2, 0), dtype=torch.long, device=pos.device)
    return torch.stack((torch.cat(source_parts), torch.cat(target_parts)), dim=0)


def _sample_id(data: MolecularData) -> int | str:
    value = getattr(data, "sample_id", None)
    if isinstance(value, Tensor) and value.numel():
        return int(value.flatten()[0].item())
    return "<unknown>"


__all__ = ["add_eqgat_inputs"]
