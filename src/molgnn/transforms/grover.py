"""GROVER-specific directed-bond and feature adaptation preparation."""

from __future__ import annotations

import torch
from torch import Tensor

from ..data import MolecularData
from ..featurizer import CANONICAL_FEATURE_SCHEMA_V1
from .base import TransformError


def add_grover_inputs(data: MolecularData) -> MolecularData:
    """Attach explicit canonical feature adaptation and reverse-bond mapping.

    GROVER's official atom recipe has extra fields that are not present in the
    shared featurizer.  This first port therefore preserves canonical atom
    features as ``grover_f_atoms`` and constructs official-shaped directed-bond
    inputs ``[source_atom_features, bond_features]``.  Neighbor lists are
    rebuilt inside the model from ``edge_index`` and the local reverse map.
    """

    sample = _sample_id(data)
    if getattr(data, "batch", None) is not None:
        raise TransformError(
            f"sample {sample} is already batched; GROVER inputs must be derived before batching"
        )
    x = _tensor(data, "x", sample)
    edge_index = _tensor(data, "edge_index", sample)
    edge_attr = _tensor(data, "edge_attr", sample)
    _validate_graph(x, edge_index, edge_attr, sample)

    source = edge_index[0]
    transformed = data.clone()
    transformed.grover_f_atoms = x.clone()
    transformed.grover_f_bonds = torch.cat((x[source], edge_attr), dim=-1)
    # The name deliberately avoids PyG's generic ``*_index`` increment rule:
    # values are local directed-edge indices and must remain local per graph.
    transformed.grover_reverse_bond = _reverse_bond_map(edge_index, edge_attr, sample)
    return transformed


def _validate_graph(
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
            f"sample {sample} must provide canonical finite float32 x with shape "
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
        raise TransformError(f"sample {sample} must provide canonical finite float32 edge_attr")
    if edge_index.shape[1] and (
        edge_index.min() < 0 or edge_index.max() >= x.shape[0]
    ):
        raise TransformError(f"sample {sample} edge_index contains an invalid node")
    if edge_index.shape[1] and bool((edge_index[0] == edge_index[1]).any()):
        raise TransformError(f"sample {sample} edge_index must not contain self-loops")


def _reverse_bond_map(edge_index: Tensor, edge_attr: Tensor, sample: int | str) -> Tensor:
    edge_count = edge_index.shape[1]
    reverse = torch.full(
        (edge_count,), -1, dtype=torch.long, device=edge_index.device
    )
    pair_to_edge: dict[tuple[int, int], int] = {}
    for edge_id, (source, target) in enumerate(edge_index.t().tolist()):
        pair = (source, target)
        if pair in pair_to_edge:
            raise TransformError(f"sample {sample} edge_index contains duplicate edges")
        pair_to_edge[pair] = edge_id
    for edge_id, (source, target) in enumerate(edge_index.t().tolist()):
        reverse_id = pair_to_edge.get((target, source))
        if reverse_id is None:
            raise TransformError(f"sample {sample} edge_index must contain reciprocal edges")
        if not torch.equal(edge_attr[edge_id], edge_attr[reverse_id]):
            raise TransformError(
                f"sample {sample} reciprocal edges must have matching edge_attr"
            )
        reverse[edge_id] = reverse_id
    if not torch.equal(
        reverse[reverse], torch.arange(edge_count, device=reverse.device)
    ):
        raise TransformError(f"sample {sample} reverse bond map is not an involution")
    return reverse


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


__all__ = ["add_grover_inputs"]
