"""Order-two heterogeneous molecular graph construction for HMGNN."""

from __future__ import annotations

import torch
from torch import Tensor

from ..data import MolecularData
from ..models.hmgnn_2020.constants import HMGNN_CUTOFF
from .base import TransformError, geometry_is_proxy, with_shared_geometry


def add_hmgnn_inputs(data: MolecularData) -> MolecularData:
    """Attach HMGNN atom/body topology to one unbatched molecular sample.

    The canonical bond graph remains untouched. Shared coordinates are used
    to derive only HMGNN's atom/body topology.
    """

    sample = _sample_id(data)
    if getattr(data, "batch", None) is not None:
        raise TransformError(
            f"sample {sample} is already batched; HMGNN inputs must be derived before batching"
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

    body_atom_index = _body_atom_index(pos, sample=sample)
    transformed = data.clone()
    transformed.atomic_number = atomic_number
    transformed.pos = pos
    transformed.hmgnn_body_atom_index = body_atom_index
    transformed.hmgnn_atom_edge_index = _atom_edge_index(body_atom_index)
    transformed.hmgnn_body_edge_index = _body_edge_index(body_atom_index)
    transformed.hmgnn_geometry_is_proxy = torch.tensor(
        [geometry_is_proxy(data)], dtype=torch.bool, device=pos.device
    )
    return transformed


def _body_atom_index(pos: Tensor, *, sample: int | str) -> Tensor:
    node_count = pos.shape[0]
    if node_count < 2:
        return torch.empty((2, 0), dtype=torch.long, device=pos.device)
    pairs = torch.triu_indices(node_count, node_count, offset=1, device=pos.device)
    distances = torch.linalg.vector_norm(pos[pairs[0]] - pos[pairs[1]], dim=-1)
    if bool((distances <= 0).any()):
        raise TransformError(f"sample {sample} contains coincident atoms")
    return pairs[:, distances < HMGNN_CUTOFF].contiguous()


def _atom_edge_index(body_atom_index: Tensor) -> Tensor:
    if body_atom_index.shape[1] == 0:
        return body_atom_index.clone()
    first, second = body_atom_index
    return torch.stack(
        (torch.cat((first, second)), torch.cat((second, first))), dim=0
    ).contiguous()


def _body_edge_index(body_atom_index: Tensor) -> Tensor:
    body_count = body_atom_index.shape[1]
    if body_count < 2:
        return torch.empty((2, 0), dtype=torch.long, device=body_atom_index.device)
    body_ids = torch.arange(body_count, device=body_atom_index.device)
    source_parts: list[Tensor] = []
    target_parts: list[Tensor] = []
    for atom in range(int(body_atom_index.max().item()) + 1):
        incident = body_ids[(body_atom_index == atom).any(dim=0)]
        degree = incident.numel()
        if degree < 2:
            continue
        source = incident.repeat_interleave(degree)
        target = incident.repeat(degree)
        keep = source != target
        source_parts.append(source[keep])
        target_parts.append(target[keep])
    if not source_parts:
        return torch.empty((2, 0), dtype=torch.long, device=body_atom_index.device)
    return torch.stack(
        (torch.cat(source_parts), torch.cat(target_parts)), dim=0
    ).contiguous()


def _validate_native_inputs(
    atomic_number: object, pos: object, *, sample: int | str
) -> None:
    if not isinstance(atomic_number, Tensor):
        raise TransformError(
            f"sample {sample} requires atomic_number for native HMGNN geometry"
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
        raise TransformError(f"sample {sample} requires pos for native HMGNN geometry")
    if (
        pos.shape != (atomic_number.shape[0], 3)
        or pos.dtype != torch.float32
        or pos.device != atomic_number.device
        or not bool(torch.isfinite(pos).all())
    ):
        raise TransformError(
            f"sample {sample} pos must have shape [N, 3] finite float32 on the atomic_number device"
        )


def _sample_id(data: MolecularData) -> int | str:
    value = getattr(data, "sample_id", None)
    if isinstance(value, Tensor) and value.numel():
        return int(value.flatten()[0].item())
    return "<unknown>"


__all__ = ["add_hmgnn_inputs"]
