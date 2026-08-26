"""MGCN's coordinate-derived complete directed atom graph."""

from __future__ import annotations

import torch
from torch import Tensor

from ..data import MolecularData
from .base import TransformError, geometry_is_proxy, with_shared_geometry


def add_mgcn_inputs(data: MolecularData) -> MolecularData:
    """Attach MGCN's complete directed atom graph to one unbatched sample.

    Shared ``atomic_number`` and ``pos`` are preserved exactly while the
    paper's complete graph (no radius cutoff, no self-loops) is derived
    independently.  The canonical 2-D bond graph is left untouched.
    """

    sample = _sample_id(data)
    if getattr(data, "batch", None) is not None:
        raise TransformError(
            f"sample {sample} is already batched; MGCN inputs must be derived before batching"
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
    transformed.mgcn_edge_index = _complete_edge_index(
        atomic_number.shape[0], device=pos.device
    )
    transformed.mgcn_geometry_is_proxy = torch.tensor(
        [geometry_is_proxy(data)], dtype=torch.bool, device=pos.device
    )
    return transformed


def _validate_native_inputs(
    atomic_number: object, pos: object, *, sample: int | str
) -> None:
    if not isinstance(atomic_number, Tensor):
        raise TransformError(
            f"sample {sample} requires atomic_number for native MGCN geometry"
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
        raise TransformError(f"sample {sample} requires pos for native MGCN geometry")
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


def _complete_edge_index(node_count: int, device: torch.device) -> Tensor:
    """Return deterministic source-major directed edges for every pair.

    Produces ``N*(N-1)`` directed edges ``(i, j)`` with ``i != j`` in
    source-major order: for each source ``i`` the targets run ``0..N-1``
    excluding ``i``.  This is the paper's complete undirected graph made
    directed for message passing.
    """
    nodes = torch.arange(node_count, dtype=torch.long, device=device)
    source = nodes.repeat_interleave(node_count)
    target = nodes.repeat(node_count)
    keep = source != target
    return torch.stack((source[keep], target[keep]), dim=0)


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
            if bool(((node_ids != target) & (distances[local_target] <= 1.0e-8)).any()):
                return True
    return False


def _sample_id(data: MolecularData) -> int | str:
    value = getattr(data, "sample_id", None)
    if isinstance(value, Tensor) and value.numel():
        return int(value.flatten()[0].item())
    return "<unknown>"


__all__ = ["add_mgcn_inputs"]