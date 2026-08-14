"""Runtime constants for the canonical Transformer-M adaptation."""

from __future__ import annotations

# The transform maps the canonical feature blocks to one categorical id per
# block.  The widths are the canonical 2D schema's block widths, including the
# final unknown bucket for categorical blocks.
ATOM_FEATURE_VOCAB_SIZES = (119, 7, 6, 8, 2, 2, 6, 5)
ATOM_FEATURE_COUNT = len(ATOM_FEATURE_VOCAB_SIZES)

# Canonical bond types are single, double, triple, aromatic, and unknown.
BOND_TYPE_COUNT = 5

# The official collator uses 510 as its unreachable/too-distant SPD sentinel.
UNREACHABLE_SPD = 510

__all__ = [
    "ATOM_FEATURE_COUNT",
    "ATOM_FEATURE_VOCAB_SIZES",
    "BOND_TYPE_COUNT",
    "UNREACHABLE_SPD",
]
