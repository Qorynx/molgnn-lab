"""Dummy-node derivation for the MAT architecture.

MAT (Maziarka et al., 2020) augments the Transformer encoder with a
disconnected "dummy" node per molecule (Li et al., 2017; Clark et al.,
2019) and with inter-atomic distance matrices (paper Eq. 2). The
canonical featurizer emits 2-D atom and bond features only; this transform
closes the model-specific gap by

  1. consuming the shared coordinate tensor so the model can later build a
     pairwise distance matrix;
  2. prepending a dummy atom at index 0 with a zero feature vector
     marked by ``x[0, 0] = 1`` (matches the upstream ``featurize_mol``
     convention of setting the dummy's first feature to ``1.0``);
  3. attaching the 3-D coordinates as ``data.pos`` — the dummy gets
     ``pos = (1e6, 1e6, 1e6)`` so the model sees "very far" distances to
     every real atom, matching the upstream's ``1e6``-padding convention.

No edge is added from the dummy to any real atom — the dummy stays
disconnected, which is what the paper intends.

.. warning::

   The upstream trains and evaluates on 7 MoleculeNet benchmarks with RDKit
   conformers. The shared benchmark geometry policy is recorded with every
   run and may still differ from upstream preprocessing details.
"""

from __future__ import annotations

import torch
from torch import Tensor

from ..data import MolecularData
from .base import TransformError, with_shared_geometry

# Mirror the upstream ``featurize_mol`` padding for the dummy node's
# distance-to-self / distance-to-others — large enough that softmax(-D)
# puts effectively zero weight on the dummy in the attention kernel.
_DUMMY_DISTANCE = 1.0e6


def add_mat_inputs(data: MolecularData) -> MolecularData:
    """Clone a canonical graph and attach a ``pos`` ``[N+1, 3]`` float32 tensor.

    Prepends a dummy atom at index 0 (matching the upstream convention of
    "add_dummy_node=True") and stores the resulting 3-D coordinates on
    ``data.pos``.  Edge index and edge attributes are left untouched —
    the dummy stays disconnected.
    """

    sample = _sample_id(data)
    if getattr(data, "batch", None) is not None:
        raise TransformError(
            f"sample {sample} is already batched; MAT inputs must be derived before batching"
        )

    x = getattr(data, "x", None)
    if not isinstance(x, Tensor) or x.ndim != 2 or x.shape[0] < 1:
        raise TransformError(f"sample {sample} has invalid x")
    if x.dtype != torch.float32:
        raise TransformError(f"sample {sample} x must be float32")
    if not torch.isfinite(x).all():
        raise TransformError(f"sample {sample} x must contain only finite values")

    real_pos = getattr(data, "pos", None)
    if real_pos is None:
        data = with_shared_geometry(data)
        real_pos = data.pos
    if (
        not isinstance(real_pos, Tensor)
        or real_pos.shape != (x.shape[0], 3)
        or real_pos.dtype != torch.float32
        or real_pos.device != x.device
        or not bool(torch.isfinite(real_pos).all())
    ):
        raise TransformError(
            f"sample {sample} requires finite float32 pos with shape [N, 3]"
        )

    device = x.device
    dummy_pos = torch.full((1, 3), _DUMMY_DISTANCE, dtype=torch.float32, device=device)
    pos = torch.cat((dummy_pos, real_pos), dim=0)

    # Prepend the dummy node: zero vector with the first feature set to 1
    # so the encoder can recognise the dummy atom by inspection.  This
    # matches the upstream's ``m[0, 0] = 1`` convention.
    dummy_x = torch.zeros(1, x.shape[1], dtype=torch.float32, device=device)
    dummy_x[0, 0] = 1.0
    padded_x = torch.cat((dummy_x, x), dim=0)

    transformed = data.clone()
    transformed.x = padded_x
    transformed.pos = pos
    return transformed


def _sample_id(data: MolecularData) -> int | str:
    value = getattr(data, "sample_id", None)
    if isinstance(value, Tensor) and value.numel():
        return int(value.flatten()[0].item())
    return "<unknown>"


__all__ = ["add_mat_inputs"]
