"""GraphMVP's model-local categorical 2-D views.

The legacy implementation used integer atom / bond tables instead of the
project's canonical one-hot tensors.  This transform keeps that view explicit
and leaves the shared molecular features untouched.  The ``ogb_full`` view is
the dependency-free equivalent of the feature vector consumed by OGB's
``AtomEncoder`` / ``BondEncoder`` in the regression source tree.
"""

from __future__ import annotations

import torch
from torch import Tensor

from ..data import MolecularData
from .base import TransformError
from .molebert import add_molebert_inputs


def _argmax_block(values: Tensor, start: int, width: int, name: str) -> Tensor:
    block = values[:, start : start + width]
    if block.ndim != 2 or block.shape[1] != width:
        raise TransformError(f"canonical {name} block is missing or has the wrong width")
    if block.shape[0] and not torch.allclose(
        block.sum(dim=1), torch.ones(block.shape[0], device=block.device)
    ):
        raise TransformError(f"canonical {name} block is not one-hot")
    return block.argmax(dim=1).to(torch.long)


def _canonical_indices(data: MolecularData) -> tuple[Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor]:
    if not isinstance(data.x, Tensor) or data.x.ndim != 2 or data.x.shape[1] < 153:
        raise TransformError("GraphMVP requires the canonical atom feature schema")
    # canonical_2d_v1 block widths: 119, 7, 6, 8, 1, 1, 6, 5
    atomic = _argmax_block(data.x, 0, 119, "atomic_number")
    degree = _argmax_block(data.x, 119, 7, "degree")
    formal_charge = _argmax_block(data.x, 126, 6, "formal_charge")
    hybridization = _argmax_block(data.x, 132, 8, "hybridization")
    aromatic = data.x[:, 140].round().to(torch.long)
    in_ring = data.x[:, 141].round().to(torch.long)
    total_h = _argmax_block(data.x, 142, 6, "total_hydrogens")
    chirality = _argmax_block(data.x, 148, 5, "chiral_tag")
    return atomic, degree, formal_charge, hybridization, aromatic, in_ring, total_h, chirality


def _ogb_atom_attributes(data: MolecularData) -> Tensor:
    atomic, degree, formal_charge, hybridization, aromatic, in_ring, total_h, chirality = _canonical_indices(data)

    # OGB feature tables include a final ``misc`` bucket.  The canonical
    # schema already uses the same index for common values; for fields whose
    # compact vocabulary is smaller, preserve the unknown bucket and derive
    # degree from connectivity when possible.
    if degree.numel():
        degree = degree.clone()
        degree_unknown = degree == 6
        if bool(degree_unknown.any()) and data.edge_index.numel():
            degree_counts = torch.bincount(
                data.edge_index[0], minlength=data.x.shape[0]
            ).to(torch.long)
            degree[degree_unknown] = degree_counts[degree_unknown].clamp_max(10)
        degree[degree == 6] = 11

    # canonical formal-charge IDs represent -2..2; OGB uses -5..5 + misc.
    formal_charge = torch.where(
        formal_charge < 5, formal_charge + 3, torch.full_like(formal_charge, 11)
    )
    # OGB's six-value table is SP, SP2, SP3, SP3D, SP3D2, misc;
    # canonical ``S``/``SP2D``/unknown therefore map to misc.
    hybridization_map = torch.tensor(
        (5, 0, 1, 5, 2, 3, 4, 5), dtype=torch.long, device=hybridization.device
    )
    hybridization = hybridization_map[hybridization.clamp_max(7)]
    # Radical electrons are not part of canonical_2d_v1.  The source datasets
    # use zero for ordinary molecular graphs; reserve the OGB zero bucket.
    radical_electrons = torch.zeros_like(atomic)
    return torch.stack(
        (
            atomic.clamp_max(118),
            chirality.clamp_max(3),
            degree,
            formal_charge,
            torch.where(total_h == 5, torch.full_like(total_h, 9), total_h),
            radical_electrons,
            hybridization,
            aromatic.clamp(0, 1),
            in_ring.clamp(0, 1),
        ),
        dim=1,
    ).contiguous()


def _ogb_bond_attributes(data: MolecularData) -> Tensor:
    if not isinstance(data.edge_attr, Tensor) or data.edge_attr.ndim != 2 or data.edge_attr.shape[1] < 14:
        raise TransformError("GraphMVP requires the canonical bond feature schema")
    # canonical_2d_v1 bond blocks: type 5, conjugated 1, ring 1, stereo 7.
    bond_type = _argmax_block(data.edge_attr, 0, 5, "bond_type").clamp_max(4)
    conjugated = data.edge_attr[:, 5].round().to(torch.long).clamp(0, 1)
    stereo = _argmax_block(data.edge_attr, 7, 7, "bond_stereo").clamp_max(5)
    return torch.stack((bond_type, stereo, conjugated), dim=1).contiguous()


def add_graphmvp_inputs(data: MolecularData) -> MolecularData:
    """Attach GraphMVP's simple and OGB-compatible integer graph views."""

    if not isinstance(data, MolecularData):
        raise TransformError("GraphMVP transform requires MolecularData")
    transformed = data.clone()
    try:
        # The simple profile is the exact atom/bond contract used by the
        # classification source.  Reusing the proven stereo alignment keeps
        # explicit-H QM9 graphs safe while exposing GraphMVP-owned names.
        transformed = add_molebert_inputs(transformed)
    except (TransformError, ValueError) as exc:
        raise TransformError(str(exc)) from exc
    transformed.graphmvp_simple_atom_attr = transformed.molebert_atom_attr.clone()
    transformed.graphmvp_simple_bond_attr = transformed.molebert_bond_attr.clone()
    transformed.graphmvp_ogb_atom_attr = _ogb_atom_attributes(transformed)
    transformed.graphmvp_ogb_bond_attr = _ogb_bond_attributes(transformed)
    return transformed


__all__ = ["add_graphmvp_inputs"]
