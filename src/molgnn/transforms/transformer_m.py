"""Packed structural inputs for the Transformer-M architecture."""

from __future__ import annotations

from collections import deque

import torch
from torch import Tensor

from ..data import MolecularData
from ..featurizer import ATOM_FEATURE_BLOCKS, CANONICAL_FEATURE_SCHEMA_V1
from ..models.transformer_m_2023.constants import BOND_TYPE_COUNT, UNREACHABLE_SPD
from .base import TransformError


def add_transformer_m_inputs(data: MolecularData) -> MolecularData:
    """Attach categorical atoms, all pairs, SPD, and sparse path records.

    Transformer-M uses dense atom attention, but PyG batches variable-sized
    molecules sparsely.  The transform therefore stores all ordered atom pairs
    plus one record for each edge on one shortest path.  The model reconstructs
    its per-graph dense attention bias without changing shared batching code.

    Coordinates are deliberately not created or modified.  If ``data.pos``
    exists it is consumed by the model as the explicit optional 3D channel.
    """

    sample = _sample_id(data)
    if getattr(data, "batch", None) is not None:
        raise TransformError(
            f"sample {sample} is already batched; Transformer-M inputs must be derived before batching"
        )
    x = _tensor(data, "x", sample)
    edge_index = _tensor(data, "edge_index", sample)
    edge_attr = _tensor(data, "edge_attr", sample)
    _validate_canonical_graph(x, edge_index, edge_attr, sample)

    categorical_x = _categorical_atom_features(x, sample)
    bond_types = _bond_types(edge_attr, edge_index.shape[1], sample)
    node_count = x.shape[0]
    in_degree = torch.bincount(edge_index[1], minlength=node_count)
    out_degree = torch.bincount(edge_index[0], minlength=node_count)
    pair_index = _pair_index(node_count, device=x.device)
    distances, path_index, path_step, path_type = _shortest_path_records(
        edge_index.detach().cpu(), bond_types.detach().cpu(), node_count
    )
    spatial_pos = torch.tensor(
        [
            UNREACHABLE_SPD
            if distance is None
            else min(distance + 1, UNREACHABLE_SPD - 1)
            for distance in distances
        ],
        dtype=torch.long,
        device=x.device,
    )

    transformed = data.clone()
    transformed.transformer_m_x = categorical_x
    transformed.transformer_m_in_degree = in_degree.to(device=x.device)
    transformed.transformer_m_out_degree = out_degree.to(device=x.device)
    transformed.transformer_m_pair_index = pair_index
    transformed.transformer_m_spatial_pos = spatial_pos
    transformed.transformer_m_path_index = _long_tensor(path_index, (2, 0), x.device)
    transformed.transformer_m_path_step = _long_tensor(path_step, (0,), x.device)
    transformed.transformer_m_path_type = _long_tensor(path_type, (0,), x.device)
    return transformed


def _validate_canonical_graph(
    x: Tensor, edge_index: Tensor, edge_attr: Tensor, sample: int | str
) -> None:
    if (
        x.ndim != 2
        or x.shape[0] < 1
        or x.shape[1] != CANONICAL_FEATURE_SCHEMA_V1.atom_dim
        or x.dtype != torch.float32
        or not torch.isfinite(x).all()
    ):
        raise TransformError(
            f"sample {sample} must provide canonical float32 x with shape "
            f"[N, {CANONICAL_FEATURE_SCHEMA_V1.atom_dim}]"
        )
    if (
        edge_index.ndim != 2
        or edge_index.shape[0] != 2
        or edge_index.dtype != torch.long
        or edge_index.device != x.device
    ):
        raise TransformError(f"sample {sample} edge_index must be long on the x device")
    if (
        edge_attr.shape != (edge_index.shape[1], CANONICAL_FEATURE_SCHEMA_V1.bond_dim)
        or edge_attr.dtype != torch.float32
        or edge_attr.device != x.device
        or not torch.isfinite(edge_attr).all()
    ):
        raise TransformError(f"sample {sample} must provide canonical float32 edge_attr")
    if edge_index.shape[1]:
        if edge_index.min() < 0 or edge_index.max() >= x.shape[0]:
            raise TransformError(f"sample {sample} edge_index contains an invalid node")
        if bool((edge_index[0] == edge_index[1]).any()):
            raise TransformError(f"sample {sample} edge_index must not contain self-loops")
        pairs = edge_index[0] * x.shape[0] + edge_index[1]
        if torch.unique(pairs).numel() != edge_index.shape[1]:
            raise TransformError(f"sample {sample} edge_index must not contain duplicate edges")
        reverse_pairs = edge_index[1] * x.shape[0] + edge_index[0]
        sorted_pairs, permutation = torch.sort(pairs)
        reverse_positions = torch.searchsorted(sorted_pairs, reverse_pairs)
        if not bool(
            (
                (reverse_positions < sorted_pairs.numel())
                & (sorted_pairs[reverse_positions.clamp_max(sorted_pairs.numel() - 1)] == reverse_pairs)
            ).all()
        ):
            raise TransformError(f"sample {sample} edge_index must contain reciprocal edges")
        reverse_edges = permutation[reverse_positions]
        if not torch.equal(edge_attr, edge_attr[reverse_edges]):
            raise TransformError(f"sample {sample} reciprocal edges must have matching edge_attr")


def _categorical_atom_features(x: Tensor, sample: int | str) -> Tensor:
    columns: list[Tensor] = []
    offset = 0
    for _, width in ATOM_FEATURE_BLOCKS:
        block = x[:, offset : offset + width]
        if width == 1:
            if not bool(((block == 0) | (block == 1)).all()):
                raise TransformError(f"sample {sample} binary atom features must be 0/1")
            columns.append(block[:, 0].to(dtype=torch.long))
        else:
            if not bool(((block == 0) | (block == 1)).all()) or not bool(
                (block.sum(dim=-1) == 1).all()
            ):
                raise TransformError(f"sample {sample} categorical atom features must be one-hot")
            columns.append(torch.argmax(block, dim=-1).to(dtype=torch.long))
        offset += width
    return torch.stack(columns, dim=-1)


def _bond_types(edge_attr: Tensor, edge_count: int, sample: int | str) -> Tensor:
    block = edge_attr[:, :BOND_TYPE_COUNT]
    if block.shape != (edge_count, BOND_TYPE_COUNT) or not bool(
        ((block == 0) | (block == 1)).all()
    ) or not bool((block.sum(dim=-1) == 1).all()):
        raise TransformError(f"sample {sample} bond type features must be one-hot")
    return torch.argmax(block, dim=-1).to(dtype=torch.long)


def _shortest_path_records(
    edge_index: Tensor, bond_types: Tensor, node_count: int
) -> tuple[list[int | None], list[list[int]], list[int], list[int]]:
    adjacency: list[list[tuple[int, int]]] = [[] for _ in range(node_count)]
    for (source, target), bond_type in zip(edge_index.t().tolist(), bond_types.tolist(), strict=True):
        adjacency[source].append((target, bond_type))

    distances: list[int | None] = []
    path_sources: list[int] = []
    path_targets: list[int] = []
    path_steps: list[int] = []
    path_types: list[int] = []
    for source in range(node_count):
        distance = [-1] * node_count
        parent = [-1] * node_count
        parent_bond = [-1] * node_count
        distance[source] = 0
        queue: deque[int] = deque([source])
        while queue:
            current = queue.popleft()
            for neighbor, bond_type in adjacency[current]:
                if distance[neighbor] != -1:
                    continue
                distance[neighbor] = distance[current] + 1
                parent[neighbor] = current
                parent_bond[neighbor] = bond_type
                queue.append(neighbor)

        for target in range(node_count):
            distance_value = distance[target]
            distances.append(None if distance_value < 0 else distance_value)
            if target == source or distance_value < 0:
                continue
            reversed_types: list[int] = []
            current = target
            while current != source:
                reversed_types.append(parent_bond[current])
                current = parent[current]
            path_types_for_pair = list(reversed(reversed_types))
            for step, bond_type in enumerate(path_types_for_pair):
                path_sources.append(source)
                path_targets.append(target)
                path_steps.append(step)
                path_types.append(bond_type + 1)

    return (
        distances,
        [path_sources, path_targets],
        path_steps,
        path_types,
    )


def _pair_index(node_count: int, *, device: torch.device) -> Tensor:
    nodes = torch.arange(node_count, dtype=torch.long, device=device)
    return torch.stack(
        (nodes.repeat_interleave(node_count), nodes.repeat(node_count)), dim=0
    )


def _long_tensor(value: list[list[int]] | list[int], empty_shape: tuple[int, ...], device: torch.device) -> Tensor:
    if isinstance(value, list) and value and isinstance(value[0], list):
        if value[0]:
            return torch.tensor(value, dtype=torch.long, device=device)
        return torch.empty(empty_shape, dtype=torch.long, device=device)
    if value:
        return torch.tensor(value, dtype=torch.long, device=device)
    return torch.empty(empty_shape, dtype=torch.long, device=device)


def _tensor(data: MolecularData, name: str, sample: int | str) -> Tensor:
    value = getattr(data, name, None)
    if not isinstance(value, Tensor):
        raise TransformError(f"sample {sample} is missing tensor field {name}")
    return value


def _sample_id(data: MolecularData) -> int | str:
    value = getattr(data, "sample_id", None)
    if isinstance(value, Tensor) and value.numel():
        return int(value.flatten()[0].item())
    return "<unknown>"


__all__ = ["add_transformer_m_inputs"]
