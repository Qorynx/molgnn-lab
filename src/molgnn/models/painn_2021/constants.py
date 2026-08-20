"""Fixed defaults for the paper-backed PaiNN core."""

from __future__ import annotations

PAINN_CUTOFF = 5.0
PAINN_NUM_RBF = 20
PAINN_MAX_ATOMIC_NUMBER = 118
PAINN_EPS = 1.0e-8

__all__ = [
    "PAINN_CUTOFF",
    "PAINN_EPS",
    "PAINN_MAX_ATOMIC_NUMBER",
    "PAINN_NUM_RBF",
]
