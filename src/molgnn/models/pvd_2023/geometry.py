"""Model-owned radius graph construction for the released TorchMD-ET profile."""

from __future__ import annotations

import torch
from torch import Tensor


def build_pvd_radius_graph(
    pos: Tensor,
    batch: Tensor | None = None,
    *,
    cutoff_lower: float = 0.0,
    cutoff_upper: float = 5.0,
    max_num_neighbors: int = 32,
    loop: bool = True,
) -> Tensor:
    """Return directed source-to-target radius edges without cross-graph edges.

    The official implementation calls ``torch_cluster.radius_graph`` with
    ``loop=True`` and a 32-neighbor cap.  This pure-PyTorch implementation
    keeps the same contract while choosing nearest neighbors with a stable
    source-index tie break when the cap is active.
    """

    if pos.ndim != 2 or pos.shape[1] != 3 or not pos.is_floating_point():
        raise ValueError("pos must be a floating tensor with shape [N, 3]")
    if pos.shape[0] < 1 or not bool(torch.isfinite(pos).all()):
        raise ValueError("pos must contain at least one finite coordinate")
    if cutoff_lower < 0 or cutoff_upper <= cutoff_lower:
        raise ValueError("cutoffs must satisfy 0 <= cutoff_lower < cutoff_upper")
    if (
        isinstance(max_num_neighbors, bool)
        or not isinstance(max_num_neighbors, int)
        or max_num_neighbors < 1
    ):
        raise ValueError("max_num_neighbors must be a positive integer")

    if batch is None:
        batch = torch.zeros(pos.shape[0], dtype=torch.long, device=pos.device)
    if batch.shape != (pos.shape[0],) or batch.dtype != torch.long:
        raise ValueError("batch must have shape [N] and dtype torch.long")
    if batch.device != pos.device or bool((batch < 0).any()):
        raise ValueError("batch must be non-negative and share the pos device")
    graph_ids = torch.unique(batch, sorted=True)
    expected = torch.arange(graph_ids.numel(), device=batch.device)
    if not torch.equal(graph_ids, expected):
        raise ValueError("batch graph indices must be contiguous from zero")

    source_parts: list[Tensor] = []
    target_parts: list[Tensor] = []
    for graph_id in graph_ids:
        node_index = torch.nonzero(batch == graph_id, as_tuple=False).flatten()
        graph_pos = pos[node_index]
        distances = torch.cdist(graph_pos, graph_pos)
        node_count = int(node_index.numel())
        local_ids = torch.arange(node_count, device=pos.device)
        for target in range(node_count):
            valid = (distances[:, target] >= cutoff_lower) & (
                distances[:, target] < cutoff_upper
            )
            if not loop:
                valid &= local_ids != target
            sources = local_ids[valid]
            if sources.numel() > max_num_neighbors:
                # Stable sort retains ascending source indices for exact ties.
                order = torch.argsort(
                    distances[sources, target], stable=True
                )[:max_num_neighbors]
                sources = sources[order]
            if sources.numel():
                source_parts.append(node_index[sources])
                target_parts.append(
                    node_index.new_full((sources.numel(),), int(node_index[target]))
                )
    if not source_parts:
        return torch.empty((2, 0), dtype=torch.long, device=pos.device)
    return torch.stack((torch.cat(source_parts), torch.cat(target_parts)), dim=0)


__all__ = ["build_pvd_radius_graph"]
