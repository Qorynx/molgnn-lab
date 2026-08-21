"""Shared-coordinate validation for the EGNN architecture.

EGNN's positional message term (paper Eq. 4) consumes pairwise squared
distances between atom coordinates, so every input graph must carry a
``[N, 3]`` float32 coordinate tensor. The shared geometry provider supplies
that tensor; this model transform only validates EGNN's runtime contract.

.. warning::

   ETKDG geometry used for SMILES-only benchmarks is **not** what the paper
   does: EGNN's authors trained and evaluated on QM9, whose ``data.pos`` ships
   with DFT-optimised geometries
   (B3LYP/6-31G(2df,p)).  An MMFF-relaxed conformer is a much weaker
   proxy for a low-energy 3-D structure and will materially depress the
   model's performance on 2-D-only benchmarks (BACE, BBBP, …).  For a
   faithful evaluation, swap to a dataset that ships with real 3-D
   coordinates (QM9, MD17, GEOM-DRUGS, …) and replace this transform with
   one that reads the precomputed conformers from disk.
"""

from __future__ import annotations

import torch
from torch import Tensor

from ..data import MolecularData
from .base import TransformError, with_shared_geometry


def add_egnn_inputs(data: MolecularData) -> MolecularData:
    """Clone a canonical graph and attach a ``pos`` ``[N, 3]`` float32 tensor.

    The function never mutates the input ``data`` and never generates or
    replaces coordinates.

    .. warning::

       See the module-level warning: shared ETKDG coordinates are a benchmark
       proxy, not the DFT-optimised QM9 geometry assumed by the paper.
    """

    sample = _sample_id(data)
    if getattr(data, "batch", None) is not None:
        raise TransformError(
            f"sample {sample} is already batched; EGNN inputs must be derived before batching"
        )

    x = getattr(data, "x", None)
    if not isinstance(x, Tensor) or x.ndim != 2 or x.shape[0] < 1:
        raise TransformError(f"sample {sample} has invalid x")
    if x.dtype != torch.float32:
        raise TransformError(f"sample {sample} x must be float32")
    if not torch.isfinite(x).all():
        raise TransformError(f"sample {sample} x must contain only finite values")

    pos = getattr(data, "pos", None)
    if pos is None:
        data = with_shared_geometry(data)
        pos = data.pos
    if not isinstance(pos, Tensor):
        raise TransformError(f"sample {sample} requires pos for EGNN")
    if (
        pos.shape != (x.shape[0], 3)
        or pos.dtype != torch.float32
        or pos.device != x.device
        or not bool(torch.isfinite(pos).all())
    ):
        raise TransformError(
            f"sample {sample} pos must have shape [N, 3] finite float32 on the node device"
        )
    transformed = data.clone()
    transformed.pos = pos
    return transformed


def _sample_id(data: MolecularData) -> int | str:
    value = getattr(data, "sample_id", None)
    if isinstance(value, Tensor) and value.numel():
        return int(value.flatten()[0].item())
    return "<unknown>"


__all__ = ["add_egnn_inputs"]
