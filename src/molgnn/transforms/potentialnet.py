"""2D/3D typed graph derivation for the unified PotentialNet contract.

The transform always derives the covalent branch.  When a sample carries
valid 3D coordinates it additionally derives the spatial branch; otherwise it
marks the sample as bond-only and PotentialNet bypasses Stage 2.  It never
creates coordinates or a covalent-only substitute for the spatial stage.
"""

from __future__ import annotations

import torch
from torch import Tensor

from ..data import MolecularData
from ..featurizer import BOND_TYPE_VOCAB, CANONICAL_FEATURE_SCHEMA_V1
from .base import TransformError


POTENTIALNET_BOND_EDGE_TYPE_COUNT = 5
POTENTIALNET_DISTANCE_BINS = (1.5, 2.5, 3.5, 4.5)
POTENTIALNET_MAX_SPATIAL_NEIGHBORS = 4
POTENTIALNET_STAGE2_EDGE_TYPE_COUNT = (
    len(POTENTIALNET_DISTANCE_BINS) + POTENTIALNET_BOND_EDGE_TYPE_COUNT
)
POTENTIALNET_USE_SPATIAL_FIELD = "potentialnet_use_spatial"


def add_potentialnet_inputs(data: MolecularData) -> MolecularData:
    """Attach the 2D/3D PotentialNet view to one unbatched graph sample.

    Canonical 14-wide bond features and the five-channel DGL-LifeSci complex
    profile are both accepted.  In either case every active relation expands
    into a parallel typed sparse edge.  Missing ``pos`` is a valid 2D input:
    Stage 1 is retained and the output flag disables Stage 2.
    """

    x = _tensor(data, "x")
    edge_index = _tensor(data, "edge_index")
    edge_attr = _tensor(data, "edge_attr")
    sample = _sample_id(data)
    if getattr(data, "batch", None) is not None:
        raise TransformError(
            f"sample {sample} is already batched; PotentialNet inputs must be "
            "derived before PyG batching"
        )
    if x.ndim != 2 or x.shape[0] < 1 or x.dtype != torch.float32:
        raise TransformError(f"sample {sample} must provide non-empty float32 x")
    node_count = x.shape[0]
    if (
        edge_index.ndim != 2
        or edge_index.shape[0] != 2
        or edge_index.dtype != torch.long
    ):
        raise TransformError(f"sample {sample} has invalid edge_index")
    if edge_index.device != x.device:
        raise TransformError(f"sample {sample} edge_index must share the node device")
    edge_count = edge_index.shape[1]
    if edge_count and (edge_index.min() < 0 or edge_index.max() >= node_count):
        raise TransformError(f"sample {sample} edge_index contains an invalid node")
    if edge_count and bool((edge_index[0] == edge_index[1]).any()):
        raise TransformError(f"sample {sample} edge_index must not contain self-loops")
    relations = _bond_relations(edge_attr, edge_count, sample=sample)
    if relations.device != x.device:
        raise TransformError(
            f"sample {sample} edge features must share the node device"
        )

    ligand_mask = _ligand_mask(data, node_count, x.device, sample=sample)
    bond_edge_index, bond_edge_type = _expand_active_relations(edge_index, relations)
    pos = getattr(data, "pos", None)
    if pos is None:
        stage2_edge_index = torch.empty((2, 0), dtype=torch.long, device=x.device)
        stage2_edge_type = torch.empty((0,), dtype=torch.long, device=x.device)
        use_spatial = False
    else:
        if not isinstance(pos, Tensor):
            raise TransformError(f"sample {sample} pos must be a torch.Tensor")
        _validate_pos(pos, node_count, x.device, sample=sample)
        spatial_edge_index, spatial_edge_type = _spatial_edges(pos)
        stage2_edge_index = torch.cat((spatial_edge_index, bond_edge_index), dim=1)
        stage2_edge_type = torch.cat(
            (
                spatial_edge_type,
                bond_edge_type + len(POTENTIALNET_DISTANCE_BINS),
            ),
            dim=0,
        )
        use_spatial = True

    transformed = data.clone()
    transformed.ligand_mask = ligand_mask
    transformed.potentialnet_bond_edge_index = bond_edge_index
    transformed.potentialnet_bond_edge_type = bond_edge_type
    transformed.potentialnet_stage2_edge_index = stage2_edge_index
    transformed.potentialnet_stage2_edge_type = stage2_edge_type
    transformed.potentialnet_use_spatial = torch.tensor(
        [use_spatial], dtype=torch.bool, device=x.device
    )
    return transformed


def _bond_relations(edge_attr: Tensor, edge_count: int, *, sample: int | str) -> Tensor:
    """Return the five multi-hot PotentialNet covalent relation channels."""

    if edge_attr.ndim != 2 or edge_attr.shape[0] != edge_count:
        raise TransformError(f"sample {sample} has mismatched edge feature count")
    if edge_attr.dtype != torch.float32:
        raise TransformError(f"sample {sample} edge_attr must have dtype torch.float32")
    if not torch.isfinite(edge_attr).all() or not bool(
        ((edge_attr == 0) | (edge_attr == 1)).all()
    ):
        raise TransformError(
            f"sample {sample} covalent relation features must be binary"
        )

    if edge_attr.shape[1] == POTENTIALNET_BOND_EDGE_TYPE_COUNT:
        relations = edge_attr
    elif edge_attr.shape[1] == CANONICAL_FEATURE_SCHEMA_V1.bond_dim:
        type_width = len(BOND_TYPE_VOCAB) + 1
        bond_type = edge_attr[:, :type_width]
        if not bool((bond_type.sum(dim=-1) == 1).all()):
            raise TransformError(
                f"sample {sample} canonical bond types must be one-hot"
            )
        if bool(bond_type[:, len(BOND_TYPE_VOCAB)].any()):
            raise TransformError(
                f"sample {sample} has an unsupported canonical bond type"
            )
        ring_column = type_width + 1  # bond_type then conjugated, then in_ring
        relations = torch.cat(
            (
                bond_type[:, : len(BOND_TYPE_VOCAB)],
                edge_attr[:, ring_column : ring_column + 1],
            ),
            dim=-1,
        )
    else:
        raise TransformError(
            f"sample {sample} edge_attr must be canonical [E, "
            f"{CANONICAL_FEATURE_SCHEMA_V1.bond_dim}] or PotentialNet [E, "
            f"{POTENTIALNET_BOND_EDGE_TYPE_COUNT}]"
        )
    if edge_count and not bool((relations.sum(dim=-1) >= 1).all()):
        raise TransformError(
            f"sample {sample} every covalent edge needs a relation type"
        )
    return relations


def _ligand_mask(
    data: MolecularData, node_count: int, device: torch.device, *, sample: int | str
) -> Tensor:
    """Use all atoms for 2D ligands unless a complex source provides a mask."""

    value = getattr(data, "ligand_mask", None)
    if value is None:
        return torch.ones(node_count, dtype=torch.bool, device=device)
    if (
        not isinstance(value, Tensor)
        or value.shape != (node_count,)
        or value.dtype != torch.bool
        or value.device != device
    ):
        raise TransformError(f"sample {sample} ligand_mask must have shape [N] bool")
    if not bool(value.any()):
        raise TransformError(f"sample {sample} must contain at least one ligand atom")
    return value


def _validate_pos(
    pos: Tensor, node_count: int, device: torch.device, *, sample: int | str
) -> None:
    if (
        pos.shape != (node_count, 3)
        or pos.dtype != torch.float32
        or pos.device != device
        or not torch.isfinite(pos).all()
    ):
        raise TransformError(
            f"sample {sample} pos must have shape [N, 3] finite float32"
        )


def _expand_active_relations(
    edge_index: Tensor, edge_attr: Tensor
) -> tuple[Tensor, Tensor]:
    """Turn each multi-hot relation row into parallel typed sparse edges."""

    active = edge_attr.nonzero(as_tuple=False)
    if not active.numel():
        return (
            torch.empty((2, 0), dtype=torch.long, device=edge_index.device),
            torch.empty((0,), dtype=torch.long, device=edge_index.device),
        )
    edge_rows = active[:, 0]
    relation_types = active[:, 1].to(dtype=torch.long)
    return edge_index[:, edge_rows].contiguous(), relation_types.contiguous()


def _spatial_edges(pos: Tensor) -> tuple[Tensor, Tensor]:
    """Construct deterministic directed radius/KNN edges for one complex."""

    node_count = pos.shape[0]
    cutoff = POTENTIALNET_DISTANCE_BINS[-1]
    boundary = pos.new_tensor(POTENTIALNET_DISTANCE_BINS[:-1])
    source_parts: list[Tensor] = []
    target_parts: list[Tensor] = []
    type_parts: list[Tensor] = []
    all_nodes = torch.arange(node_count, device=pos.device)
    for target in range(node_count):
        distances = torch.linalg.vector_norm(pos - pos[target], dim=-1)
        candidate = (all_nodes != target) & (distances <= cutoff)
        sources = all_nodes[candidate]
        if not sources.numel():
            continue
        source_distances = distances[candidate]
        order = torch.argsort(source_distances, stable=True)
        order = order[:POTENTIALNET_MAX_SPATIAL_NEIGHBORS]
        sources = sources[order]
        source_distances = source_distances[order]
        source_parts.append(sources)
        target_parts.append(torch.full_like(sources, target))
        type_parts.append(torch.bucketize(source_distances, boundary, right=False))

    if not source_parts:
        return (
            torch.empty((2, 0), dtype=torch.long, device=pos.device),
            torch.empty((0,), dtype=torch.long, device=pos.device),
        )
    return (
        torch.stack((torch.cat(source_parts), torch.cat(target_parts)), dim=0),
        torch.cat(type_parts).to(dtype=torch.long),
    )


def _tensor(data: MolecularData, name: str) -> Tensor:
    value = getattr(data, name, None)
    if not isinstance(value, Tensor):
        raise TransformError(
            f"sample {_sample_id(data)} is missing tensor field '{name}'"
        )
    return value


def _sample_id(data: MolecularData) -> int | str:
    value = getattr(data, "sample_id", None)
    if isinstance(value, Tensor) and value.numel():
        return int(value.reshape(-1)[0].item())
    return "<unknown>"


__all__ = [
    "POTENTIALNET_BOND_EDGE_TYPE_COUNT",
    "POTENTIALNET_DISTANCE_BINS",
    "POTENTIALNET_MAX_SPATIAL_NEIGHBORS",
    "POTENTIALNET_STAGE2_EDGE_TYPE_COUNT",
    "POTENTIALNET_USE_SPATIAL_FIELD",
    "add_potentialnet_inputs",
]
