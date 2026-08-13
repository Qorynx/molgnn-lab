"""Canonical typed-bond view required by the sparse AMPNN model."""

from __future__ import annotations

import torch
from torch import Tensor

from ..data import MolecularData
from ..featurizer import BOND_TYPE_VOCAB, CANONICAL_FEATURE_SCHEMA_V1
from .base import TransformError

AMPNN_EDGE_TYPE_COUNT = len(BOND_TYPE_VOCAB)
_CANONICAL_TYPED_BOND_DIM = AMPNN_EDGE_TYPE_COUNT + 1


def add_ampnn_edge_types(data: MolecularData) -> MolecularData:
    """Return a clone with source-style four-way AMPNN bond labels.

    The bundled AMPNN implementation selects separate message and attention
    networks for single, double, triple, and aromatic covalent bonds.  The
    canonical featurizer stores those four values followed by an unknown
    bucket; unknown relations are deliberately rejected rather than silently
    being mapped to one of the source relation networks.
    """

    edge_index = getattr(data, "edge_index", None)
    edge_attr = getattr(data, "edge_attr", None)
    sample = _sample_id(data)
    if (
        not isinstance(edge_index, Tensor)
        or edge_index.ndim != 2
        or edge_index.shape[0] != 2
        or edge_index.dtype != torch.long
    ):
        raise TransformError(f"sample {sample} has invalid edge_index")
    if not isinstance(edge_attr, Tensor) or edge_attr.ndim != 2:
        raise TransformError(f"sample {sample} has invalid edge_attr")

    edge_count = edge_index.shape[1]
    expected_shape = (edge_count, CANONICAL_FEATURE_SCHEMA_V1.bond_dim)
    if edge_attr.shape != expected_shape or edge_attr.dtype != torch.float32:
        raise TransformError(
            f"sample {sample} must provide canonical float32 edge_attr"
        )
    if edge_attr.device != edge_index.device:
        raise TransformError(f"sample {sample} edge features must share edge device")

    bond_type_block = edge_attr[:, :_CANONICAL_TYPED_BOND_DIM]
    if not torch.isfinite(bond_type_block).all() or not bool(
        ((bond_type_block == 0) | (bond_type_block == 1)).all()
    ):
        raise TransformError(f"sample {sample} has invalid canonical bond types")
    if not bool((bond_type_block.sum(dim=-1) == 1).all()):
        raise TransformError(f"sample {sample} bond types must be one-hot")
    if bool(bond_type_block[:, AMPNN_EDGE_TYPE_COUNT].any()):
        raise TransformError(f"sample {sample} has an unsupported bond type")

    transformed = data.clone()
    transformed.ampnn_edge_type = torch.argmax(
        bond_type_block[:, :AMPNN_EDGE_TYPE_COUNT], dim=-1
    ).to(dtype=torch.long)
    return transformed


def _sample_id(data: MolecularData) -> int | str:
    value = getattr(data, "sample_id", None)
    if isinstance(value, Tensor) and value.numel():
        return int(value.flatten()[0].item())
    return "<unknown>"


__all__ = ["AMPNN_EDGE_TYPE_COUNT", "add_ampnn_edge_types"]
