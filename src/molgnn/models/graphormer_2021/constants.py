"""Constants for the canonical 2D Graphormer adaptation."""

from __future__ import annotations

# One categorical id is derived from every canonical feature block.  The
# categorical widths include their respective unknown buckets.
ATOM_FEATURE_VOCAB_SIZES = (119, 7, 6, 8, 2, 2, 6, 5)
BOND_FEATURE_VOCAB_SIZES = (5, 2, 2, 7)
ATOM_FEATURE_COUNT = len(ATOM_FEATURE_VOCAB_SIZES)
BOND_FEATURE_COUNT = len(BOND_FEATURE_VOCAB_SIZES)

# Graphormer's official preprocessing reserves 510 for disconnected pairs.
UNREACHABLE_SPD = 510

# ``convert_to_single_emb`` in the official implementation uses this offset.
FEATURE_EMBEDDING_OFFSET = 512

__all__ = [
    "ATOM_FEATURE_COUNT",
    "ATOM_FEATURE_VOCAB_SIZES",
    "BOND_FEATURE_COUNT",
    "BOND_FEATURE_VOCAB_SIZES",
    "FEATURE_EMBEDDING_OFFSET",
    "UNREACHABLE_SPD",
]
