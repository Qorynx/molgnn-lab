"""Fixed defaults for the paper-backed MGCN core.

MGCN (Lu et al., AAAI 2019) is a coordinate-dependent invariant model: it
uses pair distances over a complete directed atom graph but no directional
vectors, angles, equivariant states, or conformer ensemble.
"""

from __future__ import annotations

MGCN_HIDDEN_DIM = 128
MGCN_NUM_LAYERS = 4
MGCN_ETA = 0.8
MGCN_NUM_RBF = 5
MGCN_RBF_LOW = 0.0
MGCN_RBF_HIGH = 5.0
MGCN_RBF_BETA = 1.0
MGCN_MAX_ATOMIC_NUMBER = 118
MGCN_EPS = 1.0e-8

__all__ = [
    "MGCN_EPS",
    "MGCN_ETA",
    "MGCN_HIDDEN_DIM",
    "MGCN_MAX_ATOMIC_NUMBER",
    "MGCN_NUM_LAYERS",
    "MGCN_NUM_RBF",
    "MGCN_RBF_BETA",
    "MGCN_RBF_HIGH",
    "MGCN_RBF_LOW",
]
