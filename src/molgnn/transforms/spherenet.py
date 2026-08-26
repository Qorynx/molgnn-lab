"""Coordinate-derived radius, triplet and torsion topology for SphereNet.

The transform is the single place that turns a coordinate-backed molecular
sample into SphereNet's discrete spatial index contract.  It owns only the
topology: distances, angles, signed torsions and bases are recomputed from
``pos`` inside the model's forward pass so coordinate gradients stay intact.

Semantics (paper: Liu et al., *Spherical Message Passing for 3D Molecular
Graphs*, ICLR 2022; official DIG snapshot commit ``21476b0``):

- ``spherenet_edge_index`` holds directed radius edges ``j -> i`` (source
  ``j`` is the neighbor, target ``i`` is the receiving atom), loop-free and
  reciprocal, with every neighbor inside the fixed cutoff.
- ``spherenet_triplet_edge_index[0]`` is the incoming edge id ``k -> j`` and
  ``[1]`` is the current edge id ``j -> i`` for every non-backtracking path
  ``k -> j -> i`` (``k != i``).
- ``spherenet_torsion_pair_index[0]`` is a base triplet id and ``[1]`` a
  candidate incoming-edge id ``k_n -> j``.  For a fixed triplet ``k -> j -> i``
  every incoming neighbor ``k_n`` except ``i`` is a candidate; the reference
  ``k_n == k`` is retained as the cyclic ``2*pi`` fallback.
"""

from __future__ import annotations

import torch
from torch import Tensor

from ..data import MolecularData
from ..models.spherenet_2022.constants import SPHERENET_CUTOFF
from .base import TransformError, geometry_is_proxy, with_shared_geometry


def add_spherenet_inputs(data: MolecularData) -> MolecularData:
    """Attach SphereNet's radius/triplet/torsion index contract.

    The input is one *unbatched* coordinate-backed molecular sample.  Native
    ``atomic_number`` and ``pos`` are preserved exactly when both are valid;
    when both are missing the shared deterministic ETKDGv3 provider is used.
    Partial or non-finite native geometry is rejected.  The canonical 2-D
    ``x``, ``edge_index`` and ``edge_attr`` are never modified.
    """

    sample = _sample_id(data)
    if getattr(data, "batch", None) is not None:
        raise TransformError(
            f"sample {sample} is already batched; SphereNet inputs must be "
            "derived before PyG batching"
        )

    atomic_number = getattr(data, "atomic_number", None)
    pos = getattr(data, "pos", None)
    if atomic_number is None and pos is None:
        data = with_shared_geometry(data)
        atomic_number = data.atomic_number
        pos = data.pos
    if (atomic_number is None) != (pos is None):
        raise TransformError(
            f"sample {sample} has partial native geometry; SphereNet requires "
            "both atomic_number and pos, or neither"
        )
    _validate_inputs(atomic_number, pos, sample=sample)
    assert isinstance(atomic_number, Tensor)
    assert isinstance(pos, Tensor)

    edge_index = _radius_edge_index(pos, sample=sample)
    triplet_edge_index = _triplet_edge_index(
        edge_index, num_nodes=atomic_number.shape[0]
    )
    torsion_pair_index = _torsion_pair_index(
        edge_index, triplet_edge_index, num_nodes=atomic_number.shape[0]
    )

    transformed = data.clone()
    transformed.spherenet_edge_index = edge_index
    transformed.spherenet_triplet_edge_index = triplet_edge_index
    transformed.spherenet_torsion_pair_index = torsion_pair_index
    transformed.spherenet_geometry_is_proxy = torch.tensor(
        [geometry_is_proxy(transformed)],
        dtype=torch.bool,
        device=pos.device,
    )
    return transformed


def _validate_inputs(
    atomic_number: object,
    pos: object,
    *,
    sample: int | str,
) -> None:
    if not isinstance(atomic_number, Tensor):
        raise TransformError(f"sample {sample} requires atomic_number for SphereNet")
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
        raise TransformError(f"sample {sample} requires pos for SphereNet")
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
    """Return source-major deterministic directed edges inside the cutoff."""

    node_count = pos.shape[0]
    nodes = torch.arange(node_count, dtype=torch.long, device=pos.device)
    source = nodes.repeat_interleave(node_count)
    target = nodes.repeat(node_count)
    pair_distances = torch.cdist(pos, pos, p=2).reshape(-1)
    cutoff = pos.new_tensor(SPHERENET_CUTOFF)
    non_self = source != target
    if bool(((pair_distances == 0) & non_self).any()):
        raise TransformError(f"sample {sample} pos must not contain coincident atoms")
    keep = non_self & (pair_distances <= cutoff)
    return torch.stack((source[keep], target[keep]), dim=0).contiguous()


def _triplet_edge_index(
    edge_index: Tensor, *, num_nodes: int
) -> Tensor:
    """Build deterministic non-backtracking ``[k -> j, j -> i]`` edge ids.

    Triplets are enumerated per current edge ``j -> i`` in edge order, with
    incoming edges ``k -> j`` in edge order, excluding ``k == i``.
    """

    edge_count = edge_index.shape[1]
    if edge_count == 0:
        return torch.empty((2, 0), dtype=torch.long, device=edge_index.device)

    source, target = edge_index
    incoming_parts: list[Tensor] = []
    current_parts: list[Tensor] = []
    for current in range(edge_count):
        center = int(source[current])
        incoming = torch.nonzero(target == center, as_tuple=False).flatten()
        if incoming.numel() == 0:
            continue
        # k -> j edges whose source k is not the current target i.
        backtracking = source[incoming] == int(target[current])
        keep = incoming[~backtracking]
        if keep.numel() == 0:
            continue
        incoming_parts.append(keep)
        current_parts.append(
            torch.full((keep.numel(),), current, dtype=torch.long, device=edge_index.device)
        )

    if not incoming_parts:
        return torch.empty((2, 0), dtype=torch.long, device=edge_index.device)
    return torch.stack(
        (torch.cat(incoming_parts), torch.cat(current_parts)), dim=0
    ).contiguous()


def _torsion_pair_index(
    edge_index: Tensor,
    triplet_edge_index: Tensor,
    *,
    num_nodes: int,
) -> Tensor:
    """Build ``[triplet_id, candidate_incoming_edge_id]`` torsion pairs.

    For every base triplet ``k -> j -> i``, every incoming edge ``k_n -> j``
    with ``k_n != i`` is a candidate (the self-reference ``k_n == k`` stays in
    the set).  Candidates are enumerated in edge order.
    """

    if triplet_edge_index.shape[1] == 0:
        return torch.empty((2, 0), dtype=torch.long, device=edge_index.device)

    source, target = edge_index
    # incoming_by_center[c] lists edge ids of k_n -> c in edge order.
    incoming_by_center: list[list[int]] = [[] for _ in range(num_nodes)]
    for edge_id in range(edge_index.shape[1]):
        incoming_by_center[int(target[edge_id])].append(edge_id)

    triplet_ids: list[Tensor] = []
    candidate_ids: list[Tensor] = []
    num_triplets = triplet_edge_index.shape[1]
    for triplet_id in range(num_triplets):
        idx_ji = int(triplet_edge_index[1, triplet_id])
        center = int(source[idx_ji])  # j
        current_target = int(target[idx_ji])  # i
        candidates = [
            edge_id
            for edge_id in incoming_by_center[center]
            if int(source[edge_id]) != current_target
        ]
        if not candidates:
            continue
        triplet_ids.append(
            torch.full(
                (len(candidates),),
                triplet_id,
                dtype=torch.long,
                device=edge_index.device,
            )
        )
        candidate_ids.append(
            torch.tensor(candidates, dtype=torch.long, device=edge_index.device)
        )

    if not triplet_ids:
        return torch.empty((2, 0), dtype=torch.long, device=edge_index.device)
    return torch.stack((torch.cat(triplet_ids), torch.cat(candidate_ids)), dim=0).contiguous()


def _sample_id(data: MolecularData) -> int | str:
    value = getattr(data, "sample_id", None)
    if isinstance(value, Tensor) and value.numel():
        return int(value.reshape(-1)[0].item())
    return "<unknown>"


__all__ = ["add_spherenet_inputs"]
