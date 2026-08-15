"""Constants for the local canonical-feature GROVER adaptation."""

from __future__ import annotations

# The official atom recipe contains extra chemistry flags that are not part of
# molgnn-lab's canonical schema.  The first port therefore names its identity
# adaptation explicitly instead of silently claiming checkpoint compatibility.
ADAPTATION_NAME = "canonical_x_and_src_atom_plus_bond_attr"

__all__ = ["ADAPTATION_NAME"]
