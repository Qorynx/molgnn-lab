"""GraphMVP's model-local categorical 2-D views.

The legacy implementation used integer atom / bond tables instead of the
project's canonical one-hot tensors.  This transform keeps that view explicit
and leaves the shared molecular features untouched.  The ``ogb_full`` view is
the dependency-free equivalent of the feature vector consumed by OGB's
``AtomEncoder`` / ``BondEncoder`` in the regression source tree; the mapping
lives in the neutral :mod:`molgnn.transforms.ogb_categorical` helper so the
3D Infomax transform can share the exact same verified logic.
"""

from __future__ import annotations

from ..data import MolecularData
from .base import TransformError
from .molebert import add_molebert_inputs
from .ogb_categorical import (
    categorical_atom_attrs_from_canonical,
    categorical_bond_attrs_from_canonical,
)


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
    transformed.graphmvp_ogb_atom_attr = categorical_atom_attrs_from_canonical(
        transformed
    )
    transformed.graphmvp_ogb_bond_attr = categorical_bond_attrs_from_canonical(
        transformed
    )
    return transformed


__all__ = ["add_graphmvp_inputs"]
