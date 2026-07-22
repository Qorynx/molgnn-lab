"""Directed-edge metadata derived from the canonical paired bond ordering."""

from __future__ import annotations

import torch
from torch import Tensor

from ..data import MolecularData
from .base import TransformError


def add_reverse_edge_index(data: MolecularData) -> MolecularData:
    """Return a clone with the reverse directed-edge mapping required by D-MPNN."""

    edge_index = getattr(data, "edge_index", None)
    edge_attr = getattr(data, "edge_attr", None)
    if not isinstance(edge_index, Tensor) or edge_index.ndim != 2 or edge_index.shape[0] != 2:
        raise TransformError(f"sample {_sample_id(data)} has invalid edge_index")
    if not isinstance(edge_attr, Tensor) or edge_attr.ndim != 2:
        raise TransformError(f"sample {_sample_id(data)} has invalid edge_attr")

    edge_count = edge_index.shape[1]
    if edge_attr.shape[0] != edge_count:
        raise TransformError(f"sample {_sample_id(data)} has mismatched edge feature count")
    if edge_count % 2:
        raise TransformError(f"sample {_sample_id(data)} has odd directed-edge count {edge_count}")

    reverse = torch.arange(edge_count, dtype=torch.long, device=edge_index.device)
    reverse = reverse.reshape(-1, 2).flip(1).reshape(-1)
    if edge_count:
        first = torch.arange(0, edge_count, 2, device=edge_index.device)
        second = first + 1
        reversed_orientation = (edge_index[0, first] == edge_index[1, second]) & (
            edge_index[1, first] == edge_index[0, second]
        )
        matching_features = (edge_attr[first] == edge_attr[second]).all(dim=1)
        valid_pairs = reversed_orientation & matching_features
        if not bool(valid_pairs.all()):
            pair = int((~valid_pairs).nonzero(as_tuple=False)[0].item())
            edge = pair * 2
            raise TransformError(
                f"sample {_sample_id(data)} has invalid reverse pair at edges {edge}/{edge + 1}"
            )
    if not torch.equal(reverse[reverse], torch.arange(edge_count, device=reverse.device)):
        raise TransformError(f"sample {_sample_id(data)} reverse mapping is not an involution")

    transformed = data.clone()
    transformed.reverse_edge_index = reverse
    return transformed


def _sample_id(data: MolecularData) -> int | str:
    value = getattr(data, "sample_id", None)
    if isinstance(value, Tensor) and value.numel():
        return int(value.flatten()[0].item())
    return "<unknown>"


__all__ = ["add_reverse_edge_index"]
