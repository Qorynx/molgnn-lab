"""Shared structural input contracts for molecular graph architectures."""

from __future__ import annotations

import torch
from torch import Tensor


def validate_batched_molecular_graph(
    edge_index: Tensor,
    graph_batch: Tensor,
    *,
    num_nodes: int,
    device: torch.device,
    edge_field: str = "edge_index",
    forbid_self_loops: bool = False,
) -> int:
    """Validate the sparse topology shared by the molecular model inputs.

    ``graph_batch`` must identify a dense, disjoint graph partition.  In
    particular, an edge may never connect two samples in a batch.  Edge
    directionality is deliberately left to the caller: this is a shared PyG
    batch boundary, not a generic requirement that every graph be molecular
    or bidirected.  The standard runner builds a paired, loop-free molecular
    graph from SMILES; D-MPNN additionally validates its explicit reverse-edge
    map.  Models whose source equations exclude supplied self-loops request
    that check through ``forbid_self_loops``.

    Returns:
        The number of graphs in the dense batch partition.
    """

    if num_nodes < 1:
        raise ValueError("a molecular graph batch must contain at least one node")
    if (
        edge_index.ndim != 2
        or edge_index.shape[0] != 2
        or edge_index.dtype != torch.long
    ):
        raise ValueError(
            f"batch.{edge_field} must have shape [2, E] and dtype torch.long"
        )
    if graph_batch.shape != (num_nodes,) or graph_batch.dtype != torch.long:
        raise ValueError("batch.batch must have shape [N] and dtype torch.long")
    if edge_index.device != device or graph_batch.device != device:
        raise ValueError(
            "batch edge indices and graph assignments must share the node device"
        )
    if graph_batch.min() < 0:
        raise ValueError("batch.batch must contain non-negative graph indices")

    graph_ids = torch.unique(graph_batch, sorted=True)
    expected_graph_ids = torch.arange(graph_ids.numel(), device=device)
    if not torch.equal(graph_ids, expected_graph_ids):
        raise ValueError("batch.batch graph indices must be contiguous from zero")

    edge_count = edge_index.shape[1]
    if not edge_count:
        return int(graph_ids.numel())

    source, target = edge_index
    if (
        source.min() < 0
        or target.min() < 0
        or source.max() >= num_nodes
        or target.max() >= num_nodes
    ):
        raise ValueError(f"batch.{edge_field} contains an invalid node index")
    if not torch.equal(graph_batch[source], graph_batch[target]):
        raise ValueError(f"batch.{edge_field} must not connect different graphs")
    if forbid_self_loops and torch.any(source == target):
        raise ValueError(f"batch.{edge_field} must not contain self-loops")
    return int(graph_ids.numel())


__all__ = ["validate_batched_molecular_graph"]
