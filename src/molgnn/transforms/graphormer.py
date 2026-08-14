"""Packed structural inputs for the 2D Graphormer architecture."""

from __future__ import annotations

from collections import deque

import torch
from torch import Tensor

from ..data import MolecularData
from ..featurizer import (
    ATOM_FEATURE_BLOCKS,
    BOND_FEATURE_BLOCKS,
    CANONICAL_FEATURE_SCHEMA_V1,
)
from ..models.graphormer_2021.constants import (
    BOND_FEATURE_COUNT,
    UNREACHABLE_SPD,
)
from .base import TransformError


def add_graphormer_inputs(data: MolecularData) -> MolecularData:
    """Attach Graphormer's categorical, all-pairs, and shortest-path inputs.

    The transform keeps dense Graphormer preparation model-specific: all atom
    pairs and per-hop edge features are stored sparsely and reconstructed into
    one dense attention-bias matrix per graph by the model.  It neither creates
    coordinates nor adds a 3D channel.
    """

    sample = _sample_id(data)
    if getattr(data, "batch", None) is not None:
        raise TransformError(
            f"sample {sample} is already batched; Graphormer inputs must be derived before batching"
        )
    x = _tensor(data, "x", sample)
    edge_index = _tensor(data, "edge_index", sample)
    edge_attr = _tensor(data, "edge_attr", sample)
    _validate_canonical_graph(x, edge_index, edge_attr, sample)

    categorical_x = _categorical_features(x, ATOM_FEATURE_BLOCKS, sample, "atom")
    categorical_edge_attr = _categorical_features(
        edge_attr, BOND_FEATURE_BLOCKS, sample, "bond"
    )
    node_count = x.shape[0]
    pair_index = _pair_index(node_count, x.device)
    distances, path_index, path_step, path_edge_type = _shortest_path_records(
        edge_index.detach().cpu(), categorical_edge_attr.detach().cpu(), node_count
    )
    spatial_pos = torch.tensor(
        [UNREACHABLE_SPD if distance is None else distance for distance in distances],
        dtype=torch.long,
        device=x.device,
    )

    transformed = data.clone()
    transformed.graphormer_x = categorical_x
    transformed.graphormer_in_degree = torch.bincount(edge_index[1], minlength=node_count)
    transformed.graphormer_out_degree = torch.bincount(edge_index[0], minlength=node_count)
    transformed.graphormer_pair_index = pair_index
    transformed.graphormer_spatial_pos = spatial_pos
    transformed.graphormer_path_index = _index_tensor(path_index, x.device)
    transformed.graphormer_path_step = torch.tensor(
        path_step, dtype=torch.long, device=x.device
    )
    transformed.graphormer_path_edge_type = _edge_type_tensor(path_edge_type, x.device)
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
        matches_reverse = (reverse_positions < sorted_pairs.numel()) & (
            sorted_pairs[reverse_positions.clamp_max(sorted_pairs.numel() - 1)]
            == reverse_pairs
        )
        if not bool(matches_reverse.all()):
            raise TransformError(f"sample {sample} edge_index must contain reciprocal edges")
        reverse_edges = permutation[reverse_positions]
        if not torch.equal(edge_attr, edge_attr[reverse_edges]):
            raise TransformError(f"sample {sample} reciprocal edges must have matching edge_attr")


def _categorical_features(
    values: Tensor,
    blocks: tuple[tuple[str, int], ...],
    sample: int | str,
    kind: str,
) -> Tensor:
    columns: list[Tensor] = []
    offset = 0
    for _, width in blocks:
        block = values[:, offset : offset + width]
        if width == 1:
            if not bool(((block == 0) | (block == 1)).all()):
                raise TransformError(f"sample {sample} binary {kind} features must be 0/1")
            columns.append(block[:, 0].to(dtype=torch.long))
        else:
            if not bool(((block == 0) | (block == 1)).all()) or not bool(
                (block.sum(dim=-1) == 1).all()
            ):
                raise TransformError(
                    f"sample {sample} categorical {kind} features must be one-hot"
                )
            columns.append(torch.argmax(block, dim=-1).to(dtype=torch.long))
        offset += width
    return torch.stack(columns, dim=-1)


def _shortest_path_records(
    edge_index: Tensor, edge_features: Tensor, node_count: int
) -> tuple[list[int | None], list[list[int]], list[int], list[list[int]]]:
    adjacency: list[list[tuple[int, list[int]]]] = [[] for _ in range(node_count)]
    for (source, target), edge_feature in zip(
        edge_index.t().tolist(), edge_features.tolist(), strict=True
    ):
        adjacency[source].append((target, edge_feature))
    for neighbors in adjacency:
        neighbors.sort(key=lambda item: item[0])

    distances: list[int | None] = []
    path_sources: list[int] = []
    path_targets: list[int] = []
    path_steps: list[int] = []
    path_edge_types: list[list[int]] = []
    for source in range(node_count):
        distance = [-1] * node_count
        parent = [-1] * node_count
        parent_edge: list[list[int] | None] = [None] * node_count
        distance[source] = 0
        queue: deque[int] = deque([source])
        while queue:
            current = queue.popleft()
            for neighbor, edge_feature in adjacency[current]:
                if distance[neighbor] != -1:
                    continue
                distance[neighbor] = distance[current] + 1
                parent[neighbor] = current
                parent_edge[neighbor] = edge_feature
                queue.append(neighbor)

        for target in range(node_count):
            distance_value = distance[target]
            distances.append(None if distance_value < 0 else distance_value)
            if target == source or distance_value < 0:
                continue
            reverse_edges: list[list[int]] = []
            current = target
            while current != source:
                edge_feature = parent_edge[current]
                if edge_feature is None:
                    raise AssertionError("reachable node is missing a shortest-path edge")
                reverse_edges.append(edge_feature)
                current = parent[current]
            for step, edge_feature in enumerate(reversed(reverse_edges)):
                path_sources.append(source)
                path_targets.append(target)
                path_steps.append(step)
                path_edge_types.append(edge_feature)

    return distances, [path_sources, path_targets], path_steps, path_edge_types


def _pair_index(node_count: int, device: torch.device) -> Tensor:
    nodes = torch.arange(node_count, dtype=torch.long, device=device)
    return torch.stack((nodes.repeat_interleave(node_count), nodes.repeat(node_count)))


def _index_tensor(value: list[list[int]], device: torch.device) -> Tensor:
    if not value[0]:
        return torch.empty((2, 0), dtype=torch.long, device=device)
    return torch.tensor(value, dtype=torch.long, device=device)


def _edge_type_tensor(value: list[list[int]], device: torch.device) -> Tensor:
    if not value:
        return torch.empty((0, BOND_FEATURE_COUNT), dtype=torch.long, device=device)
    return torch.tensor(value, dtype=torch.long, device=device)


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


__all__ = ["add_graphormer_inputs"]
