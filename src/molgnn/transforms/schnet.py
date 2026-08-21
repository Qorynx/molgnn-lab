"""SchNet's coordinate-derived spatial graph."""

from __future__ import annotations

import torch
from torch import Tensor

from ..data import MolecularData
from ..models.schnet_2017.constants import SCHNET_CUTOFF
from .base import TransformError, geometry_is_proxy, with_shared_geometry


def add_schnet_inputs(data: MolecularData) -> MolecularData:
    """Attach SchNet's directed fixed-radius graph to one unbatched sample.

    The shared geometry provider supplies ``atomic_number`` + ``pos``. This
    transform preserves them and derives only SchNet's radius graph.
    """

    sample = _sample_id(data)
    if getattr(data, "batch", None) is not None:
        raise TransformError(
            f"sample {sample} is already batched; SchNet inputs must be derived before batching"
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

    edge_index = _radius_edge_index(pos)
    transformed = data.clone()
    transformed.atomic_number = atomic_number
    transformed.pos = pos
    transformed.schnet_edge_index = edge_index
    transformed.schnet_geometry_is_proxy = torch.tensor(
        [geometry_is_proxy(data)], dtype=torch.bool, device=pos.device
    )
    return transformed


def _validate_native_inputs(
    atomic_number: object, pos: object, *, sample: int | str
) -> None:
    if not isinstance(atomic_number, Tensor):
        raise TransformError(
            f"sample {sample} requires atomic_number for native SchNet geometry"
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
        raise TransformError(f"sample {sample} requires pos for native SchNet geometry")
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
    """Return deterministic source-major directed edges within SchNet's cutoff."""

    node_count = pos.shape[0]
    nodes = torch.arange(node_count, dtype=torch.long, device=pos.device)
    source = nodes.repeat_interleave(node_count)
    target = nodes.repeat(node_count)
    distances = torch.linalg.vector_norm(pos[source] - pos[target], dim=-1)
    keep = (source != target) & (distances <= SCHNET_CUTOFF)
    return torch.stack((source[keep], target[keep]), dim=0)


def _sample_id(data: MolecularData) -> int | str:
    value = getattr(data, "sample_id", None)
    if isinstance(value, Tensor) and value.numel():
        return int(value.flatten()[0].item())
    return "<unknown>"


__all__ = ["add_schnet_inputs"]
