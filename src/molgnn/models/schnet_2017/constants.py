"""Fixed constants for the paper-literal SchNet core profile."""

from __future__ import annotations

# Schutt et al. (2017), Section 4: 0--30 Å Gaussian centers at 0.1 Å spacing
# and gamma=10.  The transform imports the cutoff from here so the prepared
# spatial graph and the model cannot silently use different radii.
SCHNET_CUTOFF = 30.0
SCHNET_RBF_GAP = 0.1
SCHNET_RBF_GAMMA = 10.0
SCHNET_MAX_ATOMIC_NUMBER = 100

__all__ = [
    "SCHNET_CUTOFF",
    "SCHNET_MAX_ATOMIC_NUMBER",
    "SCHNET_RBF_GAMMA",
    "SCHNET_RBF_GAP",
]
