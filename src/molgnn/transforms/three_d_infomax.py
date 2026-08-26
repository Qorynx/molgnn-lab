"""Model-local OGB categorical view for the 3D Infomax PNA predictor.

3D Infomax (Stärk et al., ICML 2022) fine-tunes/inferences with a pure 2-D
PNA over OGB categorical atom/bond features; coordinates are never read at
downstream time.  This transform derives the model-owned integer view:

- ``three_d_infomax_atom_attr`` ``LongTensor[N, 9]``
- ``three_d_infomax_bond_attr`` ``LongTensor[E, 3]``

Atom attributes prefer the lossless RDKit path from ``data.smiles`` (radical
electrons included, explicit-H QM9 handled via ``Chem.AddHs``).  Every
smiles-derived candidate is verified against the canonical graph by
element-wise atomic number and degree agreement before it is accepted;
otherwise the verified canonical-schema mapping is used so atom order always
matches ``edge_index``.  Bond attributes are derived from the canonical bond
schema, which is lossless for the three consumed columns.
"""

from __future__ import annotations

import torch
from torch import Tensor

from ..data import MolecularData
from .base import TransformError
from .ogb_categorical import (
    ATOM_FEATURE_VOCAB_SIZES,
    BOND_FEATURE_VOCAB_SIZES,
    canonical_atomic_ids,
    categorical_atom_attrs_from_canonical,
    categorical_atom_attrs_from_mol,
    categorical_bond_attrs_from_canonical,
    parse_aligned_mol,
)


def _aligned_smiles_attrs(data: MolecularData, smiles: str) -> Tensor | None:
    """RDKit-derived attrs when they provably match the sample's atom order."""

    expected = int(data.x.shape[0])
    mol = parse_aligned_mol(smiles, expected)
    if mol is None:
        return None
    if mol.GetNumAtoms() != expected:
        return None
    attrs = categorical_atom_attrs_from_mol(mol)
    atomic_reference = canonical_atomic_ids(data)
    if not torch.equal(attrs[:, 0], atomic_reference):
        return None
    # Second alignment signal: connectivity degrees must agree element-wise.
    if data.edge_index.numel():
        degrees = torch.bincount(data.edge_index[0], minlength=expected)
    else:
        degrees = torch.zeros(expected, dtype=torch.long)
    if not torch.equal(attrs[:, 2].clamp_max(10), degrees.clamp_max(10)):
        return None
    return attrs


def add_three_d_infomax_inputs(data: MolecularData) -> MolecularData:
    """Attach the 3D Infomax PNA categorical view to one sample."""

    transformed = data.clone()
    atom_attrs: Tensor | None = None
    smiles = getattr(transformed, "smiles", None)
    if isinstance(smiles, str) and smiles:
        try:
            atom_attrs = _aligned_smiles_attrs(transformed, smiles)
        except TransformError:
            atom_attrs = None
    if atom_attrs is None:
        atom_attrs = categorical_atom_attrs_from_canonical(transformed)
    bond_attrs = categorical_bond_attrs_from_canonical(transformed)

    if tuple(atom_attrs.shape[1:]) != (9,) or tuple(bond_attrs.shape[1:]) != (3,):
        raise TransformError("3D Infomax categorical views have unexpected widths")
    for column, size in enumerate(ATOM_FEATURE_VOCAB_SIZES):
        if atom_attrs.shape[0] and (
            int(atom_attrs[:, column].min()) < 0 or int(atom_attrs[:, column].max()) >= size
        ):
            raise TransformError(
                f"3D Infomax atom attribute column {column} exceeds its vocabulary"
            )
    for column, size in enumerate(BOND_FEATURE_VOCAB_SIZES):
        if bond_attrs.shape[0] and (
            int(bond_attrs[:, column].min()) < 0 or int(bond_attrs[:, column].max()) >= size
        ):
            raise TransformError(
                f"3D Infomax bond attribute column {column} exceeds its vocabulary"
            )

    transformed.three_d_infomax_atom_attr = atom_attrs.contiguous()
    transformed.three_d_infomax_bond_attr = bond_attrs.contiguous()
    return transformed


__all__ = ["add_three_d_infomax_inputs"]
