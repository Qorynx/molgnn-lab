"""GemNet's directed radius, triplet, and quadruplet topology."""

from __future__ import annotations

import torch
from torch import Tensor

from ..data import MolecularData
from ..models.gemnet_2021.constants import GEMNET_CUTOFF, GEMNET_INTERACTION_CUTOFF
from .base import TransformError, geometry_is_proxy, with_shared_geometry


def add_gemnet_t_inputs(data: MolecularData) -> MolecularData:
    """Attach the coordinate-backed graph required by GemNet-T."""

    return _add_gemnet_inputs(data, include_quadruplets=False)


def add_gemnet_q_inputs(data: MolecularData) -> MolecularData:
    """Attach the coordinate-backed graphs required by GemNet-Q."""

    return _add_gemnet_inputs(data, include_quadruplets=True)


def _add_gemnet_inputs(
    data: MolecularData, *, include_quadruplets: bool
) -> MolecularData:
    sample = _sample_id(data)
    if getattr(data, "batch", None) is not None:
        raise TransformError(
            f"sample {sample} is already batched; GemNet inputs must be derived before batching"
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

    edge_index, reverse_edge_index = _radius_edges(pos, GEMNET_CUTOFF)
    triplet_edge_index = _triplets(edge_index, atomic_number.shape[0])

    transformed = data.clone()
    transformed.atomic_number = atomic_number
    transformed.pos = pos
    transformed.gemnet_edge_index = edge_index
    transformed.gemnet_reverse_edge_index = reverse_edge_index
    transformed.gemnet_triplet_edge_index = triplet_edge_index
    transformed.gemnet_geometry_is_proxy = torch.tensor(
        [geometry_is_proxy(data)], dtype=torch.bool, device=pos.device
    )

    if include_quadruplets:
        interaction_edge_index, _ = _radius_edges(pos, GEMNET_INTERACTION_CUTOFF)
        quadruplet_edges, quadruplet_interactions = _quadruplets(
            edge_index,
            interaction_edge_index,
            atomic_number.shape[0],
        )
        transformed.gemnet_interaction_edge_index = interaction_edge_index
        transformed.gemnet_quadruplet_edge_index = quadruplet_edges
        transformed.gemnet_quadruplet_interaction_index = quadruplet_interactions
    return transformed


def _radius_edges(pos: Tensor, cutoff: float) -> tuple[Tensor, Tensor]:
    """Return undirected pairs followed by their exact reverse directions."""

    node_count = pos.shape[0]
    sources: list[Tensor] = []
    targets: list[Tensor] = []
    for source in range(node_count - 1):
        distances = torch.linalg.vector_norm(pos[source + 1 :] - pos[source], dim=-1)
        relative_targets = torch.nonzero(distances <= cutoff, as_tuple=False).flatten()
        if relative_targets.numel():
            target = relative_targets + source + 1
            sources.append(torch.full_like(target, source))
            targets.append(target)
    if not sources:
        empty_edges = torch.empty((2, 0), dtype=torch.long, device=pos.device)
        empty_reverse = torch.empty((0,), dtype=torch.long, device=pos.device)
        return empty_edges, empty_reverse

    source = torch.cat(sources)
    target = torch.cat(targets)
    half = source.shape[0]
    edge_index = torch.stack(
        (torch.cat((source, target)), torch.cat((target, source))), dim=0
    ).contiguous()
    edge_ids = torch.arange(half, dtype=torch.long, device=pos.device)
    reverse = torch.cat((edge_ids + half, edge_ids))
    return edge_index, reverse


def _incoming_edges(edge_index: Tensor, num_nodes: int) -> list[Tensor]:
    target = edge_index[1]
    return [
        torch.nonzero(target == node, as_tuple=False).flatten()
        for node in range(num_nodes)
    ]


def _triplets(edge_index: Tensor, num_nodes: int) -> Tensor:
    """Build sorted edge-ID pairs ``[c -> a, b -> a]`` with ``b != c``."""

    incoming = _incoming_edges(edge_index, num_nodes)
    source, target = edge_index
    reduce_parts: list[Tensor] = []
    expand_parts: list[Tensor] = []
    for reduce_edge in range(edge_index.shape[1]):
        candidates = incoming[int(target[reduce_edge])]
        candidates = candidates[source[candidates] != source[reduce_edge]]
        if candidates.numel():
            reduce_parts.append(torch.full_like(candidates, reduce_edge))
            expand_parts.append(candidates)
    if not reduce_parts:
        return torch.empty((2, 0), dtype=torch.long, device=edge_index.device)
    return torch.stack((torch.cat(reduce_parts), torch.cat(expand_parts)), dim=0)


def _quadruplets(
    edge_index: Tensor,
    interaction_edge_index: Tensor,
    num_nodes: int,
) -> tuple[Tensor, Tensor]:
    """Build sorted references for all distinct ``c -> a - b <- d`` tuples."""

    incoming = _incoming_edges(edge_index, num_nodes)
    edge_source, _ = edge_index
    interaction_source, interaction_target = interaction_edge_index
    records: list[tuple[int, int, int]] = []
    for interaction_id in range(interaction_edge_index.shape[1]):
        b = int(interaction_source[interaction_id])
        a = int(interaction_target[interaction_id])
        ca_edges = incoming[a]
        db_edges = incoming[b]
        for ca_edge_tensor in ca_edges:
            ca_edge = int(ca_edge_tensor)
            c = int(edge_source[ca_edge])
            if c == b:
                continue
            for db_edge_tensor in db_edges:
                db_edge = int(db_edge_tensor)
                d = int(edge_source[db_edge])
                if d == a or d == c:
                    continue
                records.append((ca_edge, db_edge, interaction_id))

    if not records:
        empty_pair = torch.empty((2, 0), dtype=torch.long, device=edge_index.device)
        empty_map = torch.empty((0,), dtype=torch.long, device=edge_index.device)
        return empty_pair, empty_map

    records.sort()
    record_tensor = torch.tensor(records, dtype=torch.long, device=edge_index.device)
    return record_tensor[:, :2].T.contiguous(), record_tensor[:, 2].contiguous()


def _validate_native_inputs(
    atomic_number: object, pos: object, *, sample: int | str
) -> None:
    if not isinstance(atomic_number, Tensor):
        raise TransformError(
            f"sample {sample} requires atomic_number for native GemNet geometry"
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
        raise TransformError(f"sample {sample} requires pos for native GemNet geometry")
    if (
        pos.shape != (atomic_number.shape[0], 3)
        or pos.dtype != torch.float32
        or pos.device != atomic_number.device
        or not bool(torch.isfinite(pos).all())
    ):
        raise TransformError(
            f"sample {sample} pos must have shape [N, 3] finite float32 on the atomic_number device"
        )
    if pos.shape[0] > 1:
        distances = torch.cdist(pos, pos)
        distinct = ~torch.eye(pos.shape[0], dtype=torch.bool, device=pos.device)
        if bool((distances[distinct] <= 1.0e-8).any()):
            raise TransformError(f"sample {sample} contains coincident distinct atoms")


def _sample_id(data: MolecularData) -> int | str:
    value = getattr(data, "sample_id", None)
    if isinstance(value, Tensor) and value.numel():
        return int(value.flatten()[0].item())
    return "<unknown>"


__all__ = ["add_gemnet_q_inputs", "add_gemnet_t_inputs"]
