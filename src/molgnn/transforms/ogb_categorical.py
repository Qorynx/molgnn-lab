"""Neutral OGB categorical atom/bond views shared by model-local transforms.

The mapping reproduces the integer feature vectors consumed by OGB's
``AtomEncoder`` / ``BondEncoder`` (as used by the 3D Infomax official code,
``commons/mol_encoder.py``) without introducing the ``ogb`` dependency.
Vocabulary sizes are pinned to the official tables and were verified against
the released QMugs checkpoint embedding shapes:

- atoms: [119, 4, 12, 12, 10, 6, 6, 2, 2]
- bonds: [5, 6, 2]

Column order (atoms): atomic number, chirality, degree, formal charge, total
H, radical electrons, hybridization, aromatic, in ring.  Column order
(bonds): bond type, stereo, conjugated.  The final bucket of every column is
the OGB ``misc`` slot.

This module is pure data derivation: it neither mutates samples nor adds any
model-specific branching to the shared featurizer.
"""

from __future__ import annotations

import torch
from rdkit import Chem
from torch import Tensor

from ..data import MolecularData
from .base import TransformError

ATOM_FEATURE_VOCAB_SIZES: tuple[int, ...] = (119, 4, 12, 12, 10, 6, 6, 2, 2)
BOND_FEATURE_VOCAB_SIZES: tuple[int, ...] = (5, 6, 2)


def _argmax_block(values: Tensor, start: int, width: int, name: str) -> Tensor:
    block = values[:, start : start + width]
    if block.ndim != 2 or block.shape[1] != width:
        raise TransformError(f"canonical {name} block is missing or has the wrong width")
    if block.shape[0] and not torch.allclose(
        block.sum(dim=1), torch.ones(block.shape[0], device=block.device)
    ):
        raise TransformError(f"canonical {name} block is not one-hot")
    return block.argmax(dim=1).to(torch.long)


def canonical_atom_block_indices(
    data: MolecularData,
) -> tuple[Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor]:
    """Split the canonical one-hot atom schema into categorical columns."""

    if not isinstance(data.x, Tensor) or data.x.ndim != 2 or data.x.shape[1] < 153:
        raise TransformError("OGB categorical view requires the canonical atom feature schema")
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


def categorical_atom_attrs_from_canonical(data: MolecularData) -> Tensor:
    """Derive the [N, 9] OGB atom ids from the canonical one-hot schema.

    Radical electrons do not exist in ``canonical_2d_v1`` and map to the
    ordinary zero bucket used by the source molecular datasets; callers that
    require exact radical fidelity should derive attributes from RDKit
    directly instead.
    """

    atomic, degree, formal_charge, hybridization, aromatic, in_ring, total_h, chirality = (
        canonical_atom_block_indices(data)
    )

    # OGB degree table covers 0-10 plus misc.  The compact canonical block
    # only distinguishes 0-5 + unknown, so recover exact degrees from
    # connectivity when the block ran out of buckets.
    if degree.numel():
        degree = degree.clone()
        degree_unknown = degree == 6
        if bool(degree_unknown.any()) and isinstance(data.edge_index, Tensor) and data.edge_index.numel():
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


def categorical_bond_attrs_from_canonical(data: MolecularData) -> Tensor:
    """Derive the [E, 3] OGB bond ids from the canonical bond schema."""

    if not isinstance(data.edge_attr, Tensor) or data.edge_attr.ndim != 2 or data.edge_attr.shape[1] < 14:
        raise TransformError("OGB categorical view requires the canonical bond feature schema")
    # canonical_2d_v1 bond blocks: type 5, conjugated 1, ring 1, stereo 7.
    bond_type = _argmax_block(data.edge_attr, 0, 5, "bond_type").clamp_max(4)
    conjugated = data.edge_attr[:, 5].round().to(torch.long).clamp(0, 1)
    canonical_stereo = _argmax_block(data.edge_attr, 7, 7, "bond_stereo")
    # canonical: NONE, ANY, Z, E, CIS, TRANS, unknown
    # OGB:       NONE, Z, E, CIS, TRANS, ANY
    stereo_map = torch.tensor((0, 5, 1, 2, 3, 4, 5), device=canonical_stereo.device)
    stereo = stereo_map[canonical_stereo]
    return torch.stack((bond_type, stereo, conjugated), dim=1).contiguous()


_HYBRIDIZATION_TO_OGB = {
    Chem.rdchem.HybridizationType.SP: 0,
    Chem.rdchem.HybridizationType.SP2: 1,
    Chem.rdchem.HybridizationType.SP3: 2,
    Chem.rdchem.HybridizationType.SP3D: 3,
    Chem.rdchem.HybridizationType.SP3D2: 4,
}
_CHIRALITY_TO_OGB = {
    Chem.rdchem.ChiralType.CHI_UNSPECIFIED: 0,
    Chem.rdchem.ChiralType.CHI_TETRAHEDRAL_CW: 1,
    Chem.rdchem.ChiralType.CHI_TETRAHEDRAL_CCW: 2,
}


def parse_aligned_mol(smiles: str, expected_atom_count: int | None) -> Chem.Mol | None:
    """Parse ``smiles`` and align explicit hydrogens with a reference count.

    Returns an RDKit molecule whose atom count matches
    ``expected_atom_count`` when possible (``Chem.AddHs`` supplies the
    explicit-H variant needed by native QM9 graphs), otherwise the plain
    implicit-H parse. ``None`` means no candidate matched the count.
    """

    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    if expected_atom_count is None or mol.GetNumAtoms() == expected_atom_count:
        return mol
    expanded = Chem.AddHs(mol)
    if expanded.GetNumAtoms() == expected_atom_count:
        return expanded
    return None


def categorical_atom_attrs_from_mol(mol: Chem.Mol) -> Tensor:
    """Derive the [N, 9] OGB atom ids directly from one RDKit molecule.

    This is the lossless path (radical electrons included); callers must
    verify atom-order alignment against the sample graph before use.
    """

    columns: list[list[int]] = []
    for atom in mol.GetAtoms():
        atomic_number = atom.GetAtomicNum()
        atomic = atomic_number - 1 if 1 <= atomic_number <= 118 else 118
        chirality = _CHIRALITY_TO_OGB.get(atom.GetChiralTag(), 3)
        degree = atom.GetDegree()
        degree_id = degree if degree <= 10 else 11
        charge = atom.GetFormalCharge()
        charge_id = charge + 5 if -5 <= charge <= 5 else 11
        total_h = atom.GetTotalNumHs()
        total_h_id = total_h if total_h <= 8 else 9
        radical = atom.GetNumRadicalElectrons()
        radical_id = radical if radical <= 4 else 5
        hybridization = _HYBRIDIZATION_TO_OGB.get(atom.GetHybridization(), 5)
        columns.append(
            [
                atomic,
                chirality,
                degree_id,
                charge_id,
                total_h_id,
                radical_id,
                hybridization,
                int(atom.GetIsAromatic()),
                int(atom.IsInRing()),
            ]
        )
    if not columns:
        return torch.empty((0, 9), dtype=torch.long)
    return torch.tensor(columns, dtype=torch.long)


def canonical_atomic_ids(data: MolecularData) -> Tensor:
    """Canonical atomic-number column used for alignment verification."""

    atomic, *_ = canonical_atom_block_indices(data)
    return atomic.clamp_max(118)


__all__ = [
    "ATOM_FEATURE_VOCAB_SIZES",
    "BOND_FEATURE_VOCAB_SIZES",
    "canonical_atom_block_indices",
    "canonical_atomic_ids",
    "categorical_atom_attrs_from_canonical",
    "categorical_atom_attrs_from_mol",
    "categorical_bond_attrs_from_canonical",
    "parse_aligned_mol",
]
