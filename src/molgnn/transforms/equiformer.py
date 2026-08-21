"""Equiformer's coordinate-derived radius topology.

The 2023 Equiformer consumes coordinates through its own radius graph.  This
module deliberately leaves the framework's canonical 2-D bond graph intact;
it only adds the model-local data needed by Equiformer.
"""

from __future__ import annotations

import torch
from torch import Tensor

from ..data import MolecularData
from .base import TransformError, geometry_is_proxy, with_shared_geometry

# These are the main QM9 settings in the author implementation.  They are
# kept local so this input boundary remains importable before optional e3nn is
# installed; the model validates the same public defaults independently.
_EQUIFORMER_CUTOFF = 5.0
_EQUIFORMER_MAX_NEIGHBORS = 1000
_EQUIFORMER_MAX_ATOMIC_NUMBER = 118


def add_equiformer_inputs(data: MolecularData) -> MolecularData:
    """Attach Equiformer's directed, reciprocal radius graph to one sample.

    Shared ``atomic_number`` plus ``pos`` values are preserved. Geometry
    generation is deliberately outside this model-specific transform.

    ``equiformer_edge_index`` stores topology only.  Distances and spherical
    harmonics must be recomputed from ``pos`` inside the model so coordinate
    derivatives remain available to Equiformer.
    """

    sample = _sample_id(data)
    if getattr(data, "batch", None) is not None:
        raise TransformError(
            f"sample {sample} is already batched; Equiformer inputs must be derived before batching"
        )

    atomic_number = getattr(data, "atomic_number", None)
    pos = getattr(data, "pos", None)
    if atomic_number is None and pos is None:
        data = with_shared_geometry(data)
        atomic_number = data.atomic_number
        pos = data.pos
    _validate_native_inputs(data, atomic_number, pos, sample=sample)
    assert isinstance(atomic_number, Tensor)
    assert isinstance(pos, Tensor)

    transformed = data.clone()
    transformed.atomic_number = atomic_number
    transformed.pos = pos
    transformed.equiformer_edge_index = _radius_edge_index(pos, sample=sample)
    transformed.equiformer_geometry_is_proxy = torch.tensor(
        [geometry_is_proxy(data)], dtype=torch.bool, device=pos.device
    )
    return transformed


def _validate_native_inputs(
    data: MolecularData,
    atomic_number: object,
    pos: object,
    *,
    sample: int | str,
) -> None:
    if not isinstance(atomic_number, Tensor):
        raise TransformError(
            f"sample {sample} requires atomic_number for native Equiformer geometry"
        )
    if (
        atomic_number.ndim != 1
        or atomic_number.numel() < 1
        or atomic_number.dtype != torch.long
        or bool((atomic_number < 1).any())
        or bool((atomic_number > _EQUIFORMER_MAX_ATOMIC_NUMBER).any())
    ):
        raise TransformError(
            f"sample {sample} atomic_number must be a non-empty long tensor with values in [1, 118]"
        )
    if not isinstance(pos, Tensor):
        raise TransformError(
            f"sample {sample} requires pos for native Equiformer geometry"
        )
    if (
        pos.shape != (atomic_number.shape[0], 3)
        or pos.dtype != torch.float32
        or pos.device != atomic_number.device
        or not bool(torch.isfinite(pos).all())
    ):
        raise TransformError(
            f"sample {sample} pos must have shape [N, 3] finite float32 on the atomic_number device"
        )

    x = getattr(data, "x", None)
    if not isinstance(x, Tensor) or x.ndim != 2 or x.shape[0] != atomic_number.shape[0]:
        raise TransformError(
            f"sample {sample} canonical x must have one row for each Equiformer atom"
        )


def _radius_edge_index(pos: Tensor, *, sample: int | str) -> Tensor:
    """Build deterministic, reciprocal ``j -> i`` edges with ``distance < 5``.

    The author source sets ``max_num_neighbors=1000``.  We select the nearest
    candidates for each target (source-ID is the stable tie break), then keep
    only mutual selections.  For ordinary molecular graphs this is identical
    to the uncapped reciprocal radius graph; when the cap is reached it keeps
    the project contract of reciprocal edges while never exceeding the source
    limit for any target.
    """

    node_count = pos.shape[0]
    source_ids = torch.arange(node_count, dtype=torch.long, device=pos.device)
    source_parts: list[Tensor] = []
    target_parts: list[Tensor] = []
    cutoff = pos.new_tensor(_EQUIFORMER_CUTOFF)

    # Avoid retaining an N x N x 3 displacement tensor while preparing the
    # topology.  This preprocessing operation is intentionally outside the
    # differentiable model path.
    chunk_size = 256
    for start in range(0, node_count, chunk_size):
        stop = min(start + chunk_size, node_count)
        distances = torch.cdist(pos[start:stop], pos, p=2)
        for local_target in range(stop - start):
            target = start + local_target
            non_self = source_ids != target
            target_distances = distances[local_target]
            if bool((target_distances[non_self] == 0).any()):
                raise TransformError(f"sample {sample} contains coincident atoms")
            candidates = source_ids[non_self & (target_distances < cutoff)]
            if candidates.numel() == 0:
                continue
            candidate_distances = target_distances[candidates]
            order = torch.argsort(candidate_distances, stable=True)
            selected = candidates[order[:_EQUIFORMER_MAX_NEIGHBORS]]
            source_parts.append(selected)
            target_parts.append(torch.full_like(selected, target))

    if not source_parts:
        return torch.empty((2, 0), dtype=torch.long, device=pos.device)

    source = torch.cat(source_parts)
    target = torch.cat(target_parts)
    keys = source * node_count + target
    reverse_keys = target * node_count + source
    sorted_keys = torch.sort(keys).values
    reverse_positions = torch.searchsorted(sorted_keys, reverse_keys)
    in_bounds = reverse_positions < sorted_keys.numel()
    reciprocal = torch.zeros_like(in_bounds)
    reciprocal[in_bounds] = (
        sorted_keys[reverse_positions[in_bounds]] == reverse_keys[in_bounds]
    )
    return torch.stack((source[reciprocal], target[reciprocal]), dim=0).contiguous()


def _sample_id(data: MolecularData) -> int | str:
    value = getattr(data, "sample_id", None)
    if isinstance(value, Tensor) and value.numel():
        return int(value.reshape(-1)[0].item())
    return "<unknown>"


__all__ = ["add_equiformer_inputs"]
