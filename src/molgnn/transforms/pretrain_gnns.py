"""Model-local categorical view for the Pretrain-GNNs molecular GIN.

Hu et al. (ICLR 2020) pretrain and fine-tune a GIN over the minimal
categorical schema of the official ``chem/loader.py``:

- atom: ``(atomic-number index, chirality index)`` with the atomic index
  equal to ``Z - 1`` and chirality following RDKit's
  ``UNSPECIFIED/CW/CCW/OTHER`` enumeration (paper's four categories);
- bond: ``(bond-type index, bond-direction index)`` where types map
  ``SINGLE/DOUBLE/TRIPLE/AROMATIC`` to ``0..3`` and directions follow RDKit's
  ``NONE/ENDUPRIGHT/ENDDOWNRIGHT`` enumeration. Each physical bond contributes
  two identical rows (begin->end, end->begin) in the official bidirected
  ``edge_index`` order.

Self-loops (bond type ``4``) and mask tokens are added by the model/pretrainer,
never by this transform. The transform is read-only: canonical ``x``,
``edge_index``, and ``edge_attr`` stay untouched.

Unrepresentable input is rejected with :class:`TransformError` (unsupported
atomic-number bucket, unknown chirality tag, unknown bond direction) instead
of silently clamping or falling back.
"""

from __future__ import annotations

import torch
from rdkit import Chem
from torch import Tensor

from ..data import MolecularData
from .base import TransformError
from .ogb_categorical import canonical_atom_block_indices, parse_aligned_mol

CHIRALITY_TO_INDEX = {
    Chem.rdchem.ChiralType.CHI_UNSPECIFIED: 0,
    Chem.rdchem.ChiralType.CHI_TETRAHEDRAL_CW: 1,
    Chem.rdchem.ChiralType.CHI_TETRAHEDRAL_CCW: 2,
    Chem.rdchem.ChiralType.CHI_OTHER: 3,
}
BOND_TYPE_TO_INDEX = {
    Chem.rdchem.BondType.SINGLE: 0,
    Chem.rdchem.BondType.DOUBLE: 1,
    Chem.rdchem.BondType.TRIPLE: 2,
    Chem.rdchem.BondType.AROMATIC: 3,
}
BOND_DIRECTION_TO_INDEX = {
    Chem.rdchem.BondDir.NONE: 0,
    Chem.rdchem.BondDir.ENDUPRIGHT: 1,
    Chem.rdchem.BondDir.ENDDOWNRIGHT: 2,
}


def _chirality_index(atom: Chem.Atom) -> int:
    chirality = CHIRALITY_TO_INDEX.get(atom.GetChiralTag())
    if chirality is None:
        raise TransformError(
            f"unsupported chirality tag {atom.GetChiralTag()!r} for Pretrain-GNNs"
        )
    return chirality


def _direction_index(bond: Chem.Bond) -> int:
    direction = BOND_DIRECTION_TO_INDEX.get(bond.GetBondDir())
    if direction is None:
        raise TransformError(
            f"unsupported bond direction {bond.GetBondDir()!r} for Pretrain-GNNs"
        )
    return direction


def atom_attrs_from_mol(mol: Chem.Mol) -> Tensor:
    """Official two-column atom ids from one aligned RDKit molecule."""

    columns: list[list[int]] = []
    for atom in mol.GetAtoms():
        atomic_number = atom.GetAtomicNum()
        if not 1 <= atomic_number <= 118:
            raise TransformError(
                f"atomic number {atomic_number} is outside the official table"
            )
        columns.append([atomic_number - 1, _chirality_index(atom)])
    if not columns:
        return torch.empty((0, 2), dtype=torch.long)
    return torch.tensor(columns, dtype=torch.long)


def bond_attrs_from_mol(
    mol: Chem.Mol, edge_index: Tensor | None = None
) -> Tensor:
    """Official two-column bond ids (type, direction).

    When ``edge_index`` is supplied, rows are emitted in *that exact order*.
    This is important because PyG permits arbitrary edge ordering and the
    official source's RDKit bond iteration order is not an alignment
    contract.  Without an explicit edge order, retain the source-compatible
    bidirected RDKit order for callers that only need the molecular table.
    """

    def encode(bond: Chem.Bond) -> list[int]:
        bond_type = BOND_TYPE_TO_INDEX.get(bond.GetBondType())
        if bond_type is None:
            raise TransformError(
                f"unsupported bond type {bond.GetBondType()!r} for Pretrain-GNNs"
            )
        return [bond_type, _direction_index(bond)]

    if edge_index is not None:
        if edge_index.ndim != 2 or edge_index.shape[0] != 2:
            raise TransformError("edge_index must have shape [2, E]")
        rows: list[list[int]] = []
        num_atoms = mol.GetNumAtoms()
        for source, target in edge_index.t().tolist():
            source = int(source)
            target = int(target)
            if not (0 <= source < num_atoms and 0 <= target < num_atoms):
                raise TransformError("edge_index contains an atom outside the molecule")
            bond = mol.GetBondBetweenAtoms(source, target)
            if bond is None:
                raise TransformError(
                    f"edge_index contains a non-bonded pair ({source}, {target})"
                )
            rows.append(encode(bond))
        if not rows:
            return torch.empty((0, 2), dtype=torch.long)
        return torch.tensor(rows, dtype=torch.long)

    rows: list[list[int]] = []
    for bond in mol.GetBonds():
        encoded = encode(bond)
        rows.append(encoded)
        rows.append(encoded.copy())
    if not rows:
        return torch.empty((0, 2), dtype=torch.long)
    return torch.tensor(rows, dtype=torch.long)


def _aligned_smiles_attrs(data: MolecularData, smiles: str) -> tuple[Tensor, Tensor] | None:
    """RDKit-derived attrs when they provably match the sample atom order."""

    expected = int(data.x.shape[0])
    mol = parse_aligned_mol(smiles, expected)
    if mol is None or mol.GetNumAtoms() != expected:
        return None
    atom_attrs = atom_attrs_from_mol(mol)
    atomic_reference, *_ = canonical_atom_block_indices(data)
    # id == Z - 1 for Z in 1..118; canonical bucket 118 is the unknown
    # element and cannot be represented by Pretrain-GNNs, so reject instead
    # of clamping it into a valid element.
    if bool((atomic_reference >= 118).any()):
        raise TransformError(
            "canonical atom schema uses an unknown atomic-number bucket that "
            "Pretrain-GNNs cannot encode"
        )
    if not torch.equal(atom_attrs[:, 0], atomic_reference):
        return None
    # Second alignment signal: RDKit degrees must equal connectivity degrees.
    expected_degrees = torch.tensor(
        [atom.GetDegree() for atom in mol.GetAtoms()], dtype=torch.long
    )
    if data.edge_index.numel():
        connectivity_degrees = torch.bincount(data.edge_index[0], minlength=expected)
    else:
        connectivity_degrees = torch.zeros(expected, dtype=torch.long)
    if not torch.equal(expected_degrees, connectivity_degrees):
        return None
    bond_attrs = bond_attrs_from_mol(mol, data.edge_index)
    return atom_attrs, bond_attrs


def _canonical_fallback_attrs(data: MolecularData) -> tuple[Tensor, Tensor]:
    """Canonical-schema fallback when SMILES alignment is unavailable.

    Atomic numbers and bond types are lossless; chirality keeps the official
    four-category mapping; bond direction is not part of the canonical schema
    and defaults to ``NONE`` (index 0).

    Raises :class:`TransformError` when the canonical data contains categories
    outside the Pretrain-GNNs vocabulary (unknown atomic-number bucket in the
    canonical 119-way one-hot, or unknown chirality tag).
    """

    atomic, _, _, _, aromatic, in_ring, total_h, chiral_tag = (
        canonical_atom_block_indices(data)
    )
    del aromatic, in_ring, total_h
    # Canonical atomic bucket 118 is the unknown element; Pretrain-GNNs
    # supports Z=1..118 (ids 0..117) and cannot represent a true unknown.
    if bool((atomic >= 118).any()):
        raise TransformError(
            "canonical atom schema uses an unknown atomic-number bucket that "
            "Pretrain-GNNs cannot encode"
        )
    # Canonical chiral block has five buckets (4 known + 1 unknown); the
    # unknown bucket is not representable by the official paper vocabulary.
    if bool((chiral_tag >= 4).any()):
        raise TransformError(
            "canonical chirality tag uses an unknown bucket that Pretrain-GNNs "
            "cannot encode"
        )
    atom_attrs = torch.stack((atomic, chiral_tag), dim=1).contiguous()

    from .ogb_categorical import _argmax_block

    bond_type_block = _argmax_block(data.edge_attr, 0, 5, "bond_type")
    if bool((bond_type_block == 4).any()):
        raise TransformError(
            "canonical bond type uses the unknown bucket; Pretrain-GNNs cannot "
            "encode it without SMILES alignment"
        )
    direction = torch.zeros_like(bond_type_block)
    bond_attrs = torch.stack((bond_type_block, direction), dim=1).contiguous()
    return atom_attrs, bond_attrs


def add_pretrain_gnns_inputs(data: MolecularData) -> MolecularData:
    """Attach ``pretrain_gnns_atom_attr [N,2]`` and ``..._bond_attr [E,2]``."""

    transformed = data.clone()
    attrs: tuple[Tensor, Tensor] | None = None
    smiles = getattr(transformed, "smiles", None)
    if isinstance(smiles, str) and smiles:
        attrs = _aligned_smiles_attrs(transformed, smiles)
    if attrs is None:
        attrs = _canonical_fallback_attrs(transformed)
    atom_attrs, bond_attrs = attrs

    num_atoms = int(transformed.x.shape[0])
    num_edges = int(transformed.edge_index.shape[1])
    if atom_attrs.shape != (num_atoms, 2) or bond_attrs.shape != (num_edges, 2):
        raise TransformError("Pretrain-GNNs attribute shapes do not align with the graph")
    transformed.pretrain_gnns_atom_attr = atom_attrs.contiguous()
    transformed.pretrain_gnns_bond_attr = bond_attrs.contiguous()
    return transformed


__all__ = [
    "add_pretrain_gnns_inputs",
    "atom_attrs_from_mol",
    "bond_attrs_from_mol",
]
