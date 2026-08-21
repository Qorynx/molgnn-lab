"""2D and coordinate-backed typed views for the MPNN architectures."""

from __future__ import annotations

import torch
from torch import Tensor

from ..data import MolecularData
from ..featurizer import BOND_TYPE_VOCAB, CANONICAL_FEATURE_SCHEMA_V1
from .base import TransformError

MPNN_DISTANCE_BIN_BOUNDARIES = (2.0, 2.5, 3.0, 3.5, 4.0, 4.5, 5.0, 5.5, 6.0)
MPNN_DISTANCE_BIN_COUNT = len(MPNN_DISTANCE_BIN_BOUNDARIES) + 1
MPNN_3D_EDGE_TYPE_COUNT = len(BOND_TYPE_VOCAB) + MPNN_DISTANCE_BIN_COUNT
MPNN_3D_TYPED_BOND_DIM = len(BOND_TYPE_VOCAB) + 1


def add_mpnn_edge_types(data: MolecularData) -> MolecularData:
    """Return a clone with sparse 2D bond labels for the typed MPNN."""

    edge_index = getattr(data, "edge_index", None)
    edge_attr = getattr(data, "edge_attr", None)
    if (
        not isinstance(edge_index, Tensor)
        or edge_index.ndim != 2
        or edge_index.shape[0] != 2
    ):
        raise TransformError(f"sample {_sample_id(data)} has invalid edge_index")
    if not isinstance(edge_attr, Tensor) or edge_attr.ndim != 2:
        raise TransformError(f"sample {_sample_id(data)} has invalid edge_attr")

    edge_types = _canonical_bond_types(data, edge_index=edge_index, edge_attr=edge_attr)

    transformed = data.clone()
    transformed.mpnn_edge_type = edge_types
    return transformed


def add_mpnn_3d_distance_bins_inputs(data: MolecularData) -> MolecularData:
    """Attach the all-pairs distance-bin view for ``MPNNDistanceBins3D``.

    Coordinates must be supplied by the dataset as finite ``[N, 3]`` float32
    positions in Angstrom. Covalent pairs retain their four-way bond label;
    every other ordered atom pair receives one of ten distance-bin labels.
    The covalent labels may come from canonical 14-wide features or a five-wide
    typed-bond profile whose first four columns are single/double/triple/aromatic.
    """

    sample = _sample_id(data)
    if getattr(data, "batch", None) is not None:
        raise TransformError(
            f"sample {sample} is already batched; MPNN 3D inputs must be derived before batching"
        )
    x = getattr(data, "x", None)
    edge_index = getattr(data, "edge_index", None)
    edge_attr = getattr(data, "edge_attr", None)
    pos = getattr(data, "pos", None)
    if not isinstance(x, Tensor) or x.ndim != 2 or x.shape[0] < 1:
        raise TransformError(
            f"sample {sample} must provide a non-empty node feature matrix"
        )
    if x.dtype != torch.float32:
        raise TransformError(f"sample {sample} x must have dtype torch.float32")
    if (
        not isinstance(edge_index, Tensor)
        or edge_index.ndim != 2
        or edge_index.shape[0] != 2
    ):
        raise TransformError(f"sample {sample} has invalid edge_index")
    if edge_index.dtype != torch.long or edge_index.device != x.device:
        raise TransformError(
            f"sample {sample} edge_index must be long on the node device"
        )
    if not isinstance(edge_attr, Tensor) or edge_attr.ndim != 2:
        raise TransformError(f"sample {sample} has invalid edge_attr")
    if not isinstance(pos, Tensor):
        raise TransformError(f"sample {sample} requires pos for MPNN 3D distance bins")
    if (
        pos.shape != (x.shape[0], 3)
        or pos.dtype != torch.float32
        or pos.device != x.device
        or not torch.isfinite(pos).all()
    ):
        raise TransformError(
            f"sample {sample} pos must have shape [N, 3] finite float32"
        )

    edge_count = edge_index.shape[1]
    if edge_count and (edge_index.min() < 0 or edge_index.max() >= x.shape[0]):
        raise TransformError(f"sample {sample} edge_index contains an invalid node")
    if edge_count and bool((edge_index[0] == edge_index[1]).any()):
        raise TransformError(f"sample {sample} edge_index must not contain self-loops")
    bond_types = _mpnn_3d_bond_types(data, edge_index=edge_index, edge_attr=edge_attr)
    bond_type_matrix = _reciprocal_bond_type_matrix(
        edge_index,
        bond_types,
        num_nodes=x.shape[0],
        sample=sample,
    )

    nodes = torch.arange(x.shape[0], dtype=torch.long, device=x.device)
    source = nodes.repeat_interleave(x.shape[0])
    target = nodes.repeat(x.shape[0])
    keep = source != target
    source = source[keep]
    target = target[keep]
    mpnn_3d_edge_index = torch.stack((source, target), dim=0)

    distances = torch.linalg.vector_norm(pos[source] - pos[target], dim=-1)
    boundaries = pos.new_tensor(MPNN_DISTANCE_BIN_BOUNDARIES)
    distance_types = torch.bucketize(distances, boundaries, right=False).to(torch.long)
    bond_pair_types = bond_type_matrix[source, target]
    mpnn_3d_edge_type = torch.where(
        bond_pair_types >= 0,
        bond_pair_types,
        distance_types + len(BOND_TYPE_VOCAB),
    )

    transformed = data.clone()
    transformed.mpnn_3d_edge_index = mpnn_3d_edge_index
    transformed.mpnn_3d_edge_type = mpnn_3d_edge_type
    return transformed


def _canonical_bond_types(
    data: MolecularData, *, edge_index: Tensor, edge_attr: Tensor
) -> Tensor:
    """Extract the four supported canonical covalent bond labels."""

    sample = _sample_id(data)
    edge_count = edge_index.shape[1]
    expected_shape = (edge_count, CANONICAL_FEATURE_SCHEMA_V1.bond_dim)
    if edge_attr.shape != expected_shape or edge_attr.dtype != torch.float32:
        raise TransformError(
            f"sample {sample} must provide canonical float32 edge_attr"
        )
    if edge_attr.device != edge_index.device:
        raise TransformError(f"sample {sample} edge features must share edge device")

    type_width = MPNN_3D_TYPED_BOND_DIM
    bond_type_block = edge_attr[:, :type_width]
    if not torch.isfinite(bond_type_block).all() or not bool(
        ((bond_type_block == 0) | (bond_type_block == 1)).all()
    ):
        raise TransformError(f"sample {sample} has invalid canonical bond types")
    if not bool((bond_type_block.sum(dim=-1) == 1).all()):
        raise TransformError(f"sample {sample} bond types must be one-hot")

    unknown_column = bond_type_block[:, len(BOND_TYPE_VOCAB)]
    if bool(unknown_column.any()):
        raise TransformError(f"sample {sample} has an unsupported bond type")
    return torch.argmax(bond_type_block[:, : len(BOND_TYPE_VOCAB)], dim=-1).to(
        dtype=torch.long
    )


def _mpnn_3d_bond_types(
    data: MolecularData, *, edge_index: Tensor, edge_attr: Tensor
) -> Tensor:
    """Extract four bond labels from supported coordinate-aware source schemas."""

    edge_count = edge_index.shape[1]
    if edge_attr.shape == (edge_count, CANONICAL_FEATURE_SCHEMA_V1.bond_dim):
        return _canonical_bond_types(data, edge_index=edge_index, edge_attr=edge_attr)

    sample = _sample_id(data)
    if (
        edge_attr.shape != (edge_count, MPNN_3D_TYPED_BOND_DIM)
        or edge_attr.dtype != torch.float32
    ):
        raise TransformError(
            f"sample {sample} must provide canonical [E, "
            f"{CANONICAL_FEATURE_SCHEMA_V1.bond_dim}] or typed [E, "
            f"{MPNN_3D_TYPED_BOND_DIM}] float32 edge_attr"
        )
    if edge_attr.device != edge_index.device:
        raise TransformError(f"sample {sample} edge features must share edge device")
    if not torch.isfinite(edge_attr).all() or not bool(
        ((edge_attr == 0) | (edge_attr == 1)).all()
    ):
        raise TransformError(f"sample {sample} typed bond features must be binary")
    bond_type_block = edge_attr[:, : len(BOND_TYPE_VOCAB)]
    if not bool((bond_type_block.sum(dim=-1) == 1).all()):
        raise TransformError(f"sample {sample} typed bond types must be one-hot")
    return torch.argmax(bond_type_block, dim=-1).to(dtype=torch.long)


def _reciprocal_bond_type_matrix(
    edge_index: Tensor,
    edge_types: Tensor,
    *,
    num_nodes: int,
    sample: int | str,
) -> Tensor:
    """Validate paired canonical bonds and return their dense label lookup."""

    edge_count = edge_index.shape[1]
    if edge_count:
        encoded_pairs = edge_index[0] * num_nodes + edge_index[1]
        if torch.unique(encoded_pairs).numel() != edge_count:
            raise TransformError(
                f"sample {sample} edge_index must not contain duplicate edges"
            )
    bond_type_matrix = torch.full(
        (num_nodes, num_nodes),
        -1,
        dtype=torch.long,
        device=edge_index.device,
    )
    bond_type_matrix[edge_index[0], edge_index[1]] = edge_types
    if not torch.equal(bond_type_matrix, bond_type_matrix.T):
        raise TransformError(
            f"sample {sample} edge_index must provide reciprocal edges with matching bond types"
        )
    return bond_type_matrix


def _sample_id(data: MolecularData) -> int | str:
    value = getattr(data, "sample_id", None)
    if isinstance(value, Tensor) and value.numel():
        return int(value.flatten()[0].item())
    return "<unknown>"


__all__ = [
    "MPNN_3D_EDGE_TYPE_COUNT",
    "MPNN_3D_TYPED_BOND_DIM",
    "MPNN_DISTANCE_BIN_BOUNDARIES",
    "MPNN_DISTANCE_BIN_COUNT",
    "add_mpnn_3d_distance_bins_inputs",
    "add_mpnn_edge_types",
]
