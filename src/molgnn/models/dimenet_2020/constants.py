"""Fixed public constants for the bundled DimeNet-2020 profile."""

from __future__ import annotations

# The original DimeNet reference configuration uses a 5 Å interaction radius
# and a sixth-order polynomial envelope.  These are shared with the topology
# transform so a prepared graph cannot silently disagree with its model.
DIMENET_CUTOFF = 5.0
DIMENET_ENVELOPE_P = 6
DIMENET_MAX_ATOMIC_NUMBER = 94

__all__ = [
    "DIMENET_CUTOFF",
    "DIMENET_ENVELOPE_P",
    "DIMENET_MAX_ATOMIC_NUMBER",
]
