"""Fixed constants for the bundled EQGAT ATOM3D/LBA profile."""

from __future__ import annotations

# The official ATOM3D experiments use these values for the ligand-binding
# affinity task.  The spatial transform imports the same constants so its
# prepared radius graph cannot disagree with the model.
EQGAT_CUTOFF = 4.5
EQGAT_MAX_NEIGHBORS = 32
EQGAT_MAX_ATOMIC_NUMBER = 118
EQGAT_NUM_RADIAL = 32
EQGAT_POLYNOMIAL_CUTOFF_P = 6
EQGAT_EPS = 1e-6

__all__ = [
    "EQGAT_CUTOFF",
    "EQGAT_EPS",
    "EQGAT_MAX_ATOMIC_NUMBER",
    "EQGAT_MAX_NEIGHBORS",
    "EQGAT_NUM_RADIAL",
    "EQGAT_POLYNOMIAL_CUTOFF_P",
]
