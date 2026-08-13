"""Topological structural inputs for the 2D GPS++ architecture.

GPS++ uses an all-pairs shortest-path embedding as a bias for its global
self-attention branch.  This transform derives that topology from the
canonical directed covalent graph before PyG batching.  It deliberately does
not inspect SMILES, construct coordinates, or derive any 3D features.
"""

from __future__ import annotations

import torch
from torch import Tensor

from ..data import MolecularData
from .base import TransformError


def add_gpspp_inputs(data: MolecularData) -> MolecularData:
    """Attach source-major all-pairs indices and topological distances.

    The input must be one unbatched canonical molecular graph with paired,
    loop-free directed covalent edges.  The result adds:

    - ``gpspp_pair_index`` with shape ``[2, N * N]`` in source-major order;
    - ``gpspp_spd`` with shape ``[N * N]`` where self pairs are ``0``,
      connected pairs carry their positive unweighted shortest-path length,
      and disconnected pairs carry ``-1``.

    ``gpspp_pair_index`` intentionally uses PyG's normal ``*_index`` batching
    convention, so each graph's node IDs are offset automatically by
    :class:`torch_geometric.data.Batch`.
    """

    sample = _sample_id(data)
    if getattr(data, "batch", None) is not None:
        raise TransformError(
            f"sample {sample} is already batched; GPS++ inputs must be derived "
            "before PyG batching"
        )

    x = _tensor(data, "x", sample=sample)
    edge_index = _tensor(data, "edge_index", sample=sample)
    edge_attr = _tensor(data, "edge_attr", sample=sample)
    _validate_canonical_graph(x, edge_index, edge_attr, sample=sample)

    node_count = x.shape[0]
    pair_index = _source_major_pair_index(node_count, device=x.device)
    shortest_paths = _all_pairs_shortest_paths(
        edge_index, num_nodes=node_count
    ).reshape(-1)

    transformed = data.clone()
    transformed.gpspp_pair_index = pair_index
    transformed.gpspp_spd = shortest_paths
    return transformed


def _validate_canonical_graph(
    x: Tensor,
    edge_index: Tensor,
    edge_attr: Tensor,
    *,
    sample: int | str,
) -> None:
    """Validate the strict undirected-covalent contract GPS++ relies on."""

    if x.ndim != 2 or x.shape[0] < 1 or x.dtype != torch.float32:
        raise TransformError(
            f"sample {sample} x must be a non-empty float32 tensor with shape [N, F]"
        )
    if not bool(torch.isfinite(x).all()):
        raise TransformError(f"sample {sample} x must contain only finite values")

    if (
        edge_index.ndim != 2
        or edge_index.shape[0] != 2
        or edge_index.dtype != torch.long
    ):
        raise TransformError(f"sample {sample} edge_index must have dtype long and shape [2, E]")
    if edge_index.device != x.device:
        raise TransformError(f"sample {sample} edge_index must share the x device")

    if edge_attr.ndim != 2 or edge_attr.shape[0] != edge_index.shape[1]:
        raise TransformError(
            f"sample {sample} edge_attr must have shape [E, F] aligned with edge_index"
        )
    if edge_attr.dtype != torch.float32:
        raise TransformError(f"sample {sample} edge_attr must have dtype float32")
    if edge_attr.device != x.device:
        raise TransformError(f"sample {sample} edge_attr must share the x device")
    if not bool(torch.isfinite(edge_attr).all()):
        raise TransformError(
            f"sample {sample} edge_attr must contain only finite values"
        )

    edge_count = edge_index.shape[1]
    if edge_count == 0:
        return
    if edge_index.min() < 0 or edge_index.max() >= x.shape[0]:
        raise TransformError(f"sample {sample} edge_index contains an invalid node")

    source, target = edge_index
    if bool((source == target).any()):
        raise TransformError(f"sample {sample} edge_index must not contain self-loops")

    pair_ids = source * x.shape[0] + target
    sorted_pair_ids, permutation = torch.sort(pair_ids)
    if bool((sorted_pair_ids[1:] == sorted_pair_ids[:-1]).any()):
        raise TransformError(
            f"sample {sample} edge_index must not contain duplicate directed covalent edges"
        )

    reverse_pair_ids = target * x.shape[0] + source
    reverse_positions = torch.searchsorted(sorted_pair_ids, reverse_pair_ids)
    safe_reverse_positions = reverse_positions.clamp_max(edge_count - 1)
    reverse_found = (reverse_positions < edge_count) & (
        sorted_pair_ids[safe_reverse_positions] == reverse_pair_ids
    )
    if not bool(reverse_found.all()):
        raise TransformError(
            f"sample {sample} edge_index must contain one reciprocal directed covalent edge "
            "for every edge"
        )

    reverse_edges = permutation[reverse_positions]
    if not torch.equal(edge_attr, edge_attr[reverse_edges]):
        raise TransformError(
            f"sample {sample} reciprocal directed covalent edges must have matching edge_attr"
        )


def _source_major_pair_index(num_nodes: int, *, device: torch.device) -> Tensor:
    nodes = torch.arange(num_nodes, dtype=torch.long, device=device)
    return torch.stack((nodes.repeat_interleave(num_nodes), nodes.repeat(num_nodes))).contiguous()


def _all_pairs_shortest_paths(edge_index: Tensor, *, num_nodes: int) -> Tensor:
    """Return unweighted shortest paths for the undirected covalent topology."""

    adjacency = torch.zeros(
        (num_nodes, num_nodes), dtype=torch.bool, device=edge_index.device
    )
    if edge_index.numel():
        adjacency[edge_index[0], edge_index[1]] = True
        # Reciprocity was validated above; retaining this assignment makes the
        # topology explicit and keeps this helper correct in isolation.
        adjacency[edge_index[1], edge_index[0]] = True

    distances = torch.full(
        (num_nodes, num_nodes), -1, dtype=torch.long, device=edge_index.device
    )
    for source in range(num_nodes):
        visited = torch.zeros(num_nodes, dtype=torch.bool, device=edge_index.device)
        frontier = torch.zeros(num_nodes, dtype=torch.bool, device=edge_index.device)
        visited[source] = True
        distances[source, source] = 0
        frontier[source] = True

        for distance in range(1, num_nodes):
            next_frontier = adjacency[frontier].any(dim=0) & ~visited
            if not bool(next_frontier.any()):
                break
            distances[source, next_frontier] = distance
            visited |= next_frontier
            frontier = next_frontier

    return distances


def _tensor(data: MolecularData, name: str, *, sample: int | str) -> Tensor:
    value = getattr(data, name, None)
    if not isinstance(value, Tensor):
        raise TransformError(f"sample {sample} is missing tensor field '{name}'")
    return value


def _sample_id(data: MolecularData) -> int | str:
    value = getattr(data, "sample_id", None)
    if isinstance(value, Tensor) and value.numel():
        return int(value.reshape(-1)[0].item())
    return "<unknown>"


__all__ = ["add_gpspp_inputs"]
