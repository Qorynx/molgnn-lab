"""Coordinate-derived radius and triplet topology for DimeNet-2020."""

from __future__ import annotations

import torch
from torch import Tensor

from ..data import MolecularData
from ..models.dimenet_2020.constants import DIMENET_CUTOFF
from .base import TransformError, with_shared_geometry


def add_dimenet_inputs(data: MolecularData) -> MolecularData:
    """Attach DimeNet's directed radius graph and edge-indexed triplets.

    The input is one *unbatched* coordinate-backed molecular sample.  The
    transform leaves the framework's ordinary graph fields untouched and adds
    a separate spatial topology:

    - ``dimenet_edge_index`` has directed radius edges ``j -> i`` for every
      pair satisfying ``0 < ||r_j-r_i|| <= 5.0``;
    - ``dimenet_triplet_edge_index`` has edge IDs ``[k -> j, j -> i]`` for
      every non-backtracking path ``k -> j -> i``.

    Distances and angles are intentionally not cached here.  The model
    recomputes them from ``pos`` during its forward pass, which keeps its
    coordinate-gradient contract intact.
    """

    sample = _sample_id(data)
    if getattr(data, "batch", None) is not None:
        raise TransformError(
            f"sample {sample} is already batched; DimeNet inputs must be "
            "derived before PyG batching"
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

    dimenet_edge_index = _radius_edge_index(pos, sample=sample)
    dimenet_triplet_edge_index = _triplet_edge_index(
        dimenet_edge_index, num_nodes=atomic_number.shape[0]
    )

    transformed = data.clone()
    transformed.dimenet_edge_index = dimenet_edge_index
    transformed.dimenet_triplet_edge_index = dimenet_triplet_edge_index
    return transformed


def _validate_inputs(
    atomic_number: object,
    pos: object,
    *,
    sample: int | str,
) -> None:
    if not isinstance(atomic_number, Tensor):
        raise TransformError(f"sample {sample} requires atomic_number for DimeNet")
    if (
        atomic_number.ndim != 1
        or atomic_number.numel() < 1
        or atomic_number.dtype != torch.long
    ):
        raise TransformError(
            f"sample {sample} atomic_number must be a non-empty long tensor with shape [N]"
        )
    if bool((atomic_number <= 0).any()):
        raise TransformError(f"sample {sample} atomic_number values must be positive")
    if not isinstance(pos, Tensor):
        raise TransformError(f"sample {sample} requires pos for DimeNet")
    if (
        pos.shape != (atomic_number.shape[0], 3)
        or pos.dtype != torch.float32
        or pos.device != atomic_number.device
        or not bool(torch.isfinite(pos).all())
    ):
        raise TransformError(
            f"sample {sample} pos must have shape [N, 3] finite float32 on the atomic_number device"
        )


def _radius_edge_index(pos: Tensor, *, sample: int | str) -> Tensor:
    """Return source-major deterministic directed edges in the fixed radius."""

    node_count = pos.shape[0]
    nodes = torch.arange(node_count, dtype=torch.long, device=pos.device)
    source = nodes.repeat_interleave(node_count)
    target = nodes.repeat(node_count)
    pair_distances = torch.cdist(pos, pos, p=2).reshape(-1)
    cutoff = pos.new_tensor(DIMENET_CUTOFF)
    non_self = source != target
    if bool(((pair_distances == 0) & non_self).any()):
        raise TransformError(f"sample {sample} pos must not contain coincident atoms")
    keep = non_self & (pair_distances <= cutoff)
    return torch.stack((source[keep], target[keep]), dim=0).contiguous()


def _triplet_edge_index(edge_index: Tensor, *, num_nodes: int) -> Tensor:
    """Build deterministic non-backtracking ``[k -> j, j -> i]`` edge IDs."""

    edge_count = edge_index.shape[1]
    if edge_count == 0:
        return torch.empty((2, 0), dtype=torch.long, device=edge_index.device)

    source, target = edge_index
    incoming_parts: list[Tensor] = []
    outgoing_parts: list[Tensor] = []
    for center in range(num_nodes):
        incoming = torch.nonzero(target == center, as_tuple=False).flatten()
        outgoing = torch.nonzero(source == center, as_tuple=False).flatten()
        if incoming.numel() == 0 or outgoing.numel() == 0:
            continue

        # Every column below represents k -> j -> i.  The direct reversal
        # k == i is deliberately excluded, as required by the paper.
        non_backtracking = source[incoming][:, None] != target[outgoing][None, :]
        pairs = torch.nonzero(non_backtracking, as_tuple=False)
        if pairs.numel() == 0:
            continue
        incoming_parts.append(incoming[pairs[:, 0]])
        outgoing_parts.append(outgoing[pairs[:, 1]])

    if not incoming_parts:
        return torch.empty((2, 0), dtype=torch.long, device=edge_index.device)
    return torch.stack(
        (torch.cat(incoming_parts), torch.cat(outgoing_parts)), dim=0
    ).contiguous()


def _sample_id(data: MolecularData) -> int | str:
    value = getattr(data, "sample_id", None)
    if isinstance(value, Tensor) and value.numel():
        return int(value.reshape(-1)[0].item())
    return "<unknown>"


__all__ = ["add_dimenet_inputs"]
