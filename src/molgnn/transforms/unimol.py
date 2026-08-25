"""3-D conformer + [CLS]-atom derivation for the Uni-Mol architecture.

Uni-Mol (Zhou et al., ICLR 2023) is a dense 3-D Transformer: its backbone
consumes per-atom features ``x``, 3-D coordinates ``pos``, and the graph
assignment ``batch`` only — it never reads ``edge_index`` / ``edge_attr``.
The canonical featurizer emits 2-D atom and bond features only, so this
transform closes the model-specific gap by

  1. consuming the shared coordinate tensor (via ``with_shared_geometry``)
     so the model can later build a pairwise distance matrix;
  2. prepending a ``[CLS]`` atom at index 0 with a zero feature vector and
     a far-away sentinel coordinate (``pos.mean + 1e6``), mirroring the
     upstream Uni-Mol convention of a leading ``<s>`` / ``[CLS]`` token
     whose representation is read out for the graph-level prediction.

The ``[CLS]`` atom participates in the dense attention through two dummy
self-loop edges (kept so the sparse graph stays a valid molecular graph);
the model itself ignores the edge structure entirely.

.. warning::

   The upstream trains and evaluates on pre-computed PubChem 3-D
   conformers.  Here the shared geometry policy synthesises one
   deterministic ETKDGv3 conformer per SMILES (with MMFF/UFF fallback),
   so BACE / BBBP-class downstream numbers are a lower bound on Uni-Mol's
   true capability.  Numerical verification against the paper's equations
   is the right correctness check.
"""

from __future__ import annotations

import torch
from torch import Tensor

from ..data import MolecularData
from .base import TransformError, with_shared_geometry

# Far-away sentinel for the [CLS] atom's coordinates, mirroring MAT's
# ``_DUMMY_DISTANCE``: large enough that the Gaussian RBF distance kernel
# puts effectively zero weight on the [CLS] atom's own position when it
# attends to real atoms (and vice versa).
_CLS_SENTINEL_DISTANCE = 1.0e6


def add_unimol_inputs(data: MolecularData) -> MolecularData:
    """Clone a canonical graph and attach a ``[CLS]`` atom plus ``pos``.

    Guarantees ``pos: float32 [N+1, 3]`` (via the shared geometry provider)
    and prepends a ``[CLS]`` atom at index 0 with a zero feature vector and
    a far-away sentinel coordinate.  ``edge_index`` / ``edge_attr`` are
    shifted by one node and given two dummy ``[CLS]`` self-loop edges so the
    sparse graph stays valid; ``batch`` is extended with the first atom's
    graph id.  Graph-level labels (``y`` / ``y_mask``) are left untouched —
    they are per-graph, not per-node, so the node-level ``[CLS]`` addition
    does not affect them.
    """

    sample = _sample_id(data)
    if getattr(data, "batch", None) is not None:
        raise TransformError(
            f"sample {sample} is already batched; Uni-Mol inputs must be derived before batching"
        )

    x = getattr(data, "x", None)
    if not isinstance(x, Tensor) or x.ndim != 2 or x.shape[0] < 1:
        raise TransformError(f"sample {sample} has invalid x")
    if x.dtype != torch.float32:
        raise TransformError(f"sample {sample} x must be float32")
    if not torch.isfinite(x).all():
        raise TransformError(f"sample {sample} x must contain only finite values")

    data = with_shared_geometry(data)
    pos = getattr(data, "pos", None)
    if (
        not isinstance(pos, Tensor)
        or pos.shape != (x.shape[0], 3)
        or pos.dtype != torch.float32
        or pos.device != x.device
        or not bool(torch.isfinite(pos).all())
    ):
        raise TransformError(
            f"sample {sample} requires finite float32 pos with shape [N, 3]"
        )

    device = x.device
    num_nodes = x.shape[0]

    # [CLS] atom: zero feature vector (a learnable bias after the input
    # projection, like the upstream's fixed [CLS] token embedding) and a
    # far-away sentinel coordinate so it stays effectively disconnected in
    # the distance-derived attention bias.
    cls_x = torch.zeros(1, x.shape[1], dtype=torch.float32, device=device)
    cls_pos = pos.mean(dim=0, keepdim=True) + _CLS_SENTINEL_DISTANCE

    # Shift every original node index by +1 (the [CLS] atom now owns index 0)
    # and prepend two dummy [CLS] self-loop edges so the sparse graph remains
    # a valid molecular graph.  Two directed self-loops mimic one bidirected
    # bond, matching the canonical two-directed-edges-per-bond layout.
    edge_index = getattr(data, "edge_index", None)
    edge_attr = getattr(data, "edge_attr", None)
    if not isinstance(edge_index, Tensor) or edge_index.ndim != 2 or edge_index.shape[0] != 2:
        raise TransformError(f"sample {sample} has invalid edge_index")
    if not isinstance(edge_attr, Tensor) or edge_attr.shape[0] != edge_index.shape[1]:
        raise TransformError(f"sample {sample} has invalid edge_attr")

    shifted_edges = edge_index + 1
    cls_edges = torch.zeros((2, 2), dtype=torch.long, device=device)
    padded_edge_index = torch.cat((cls_edges, shifted_edges), dim=1)
    cls_edge_attr = torch.zeros(
        (2, edge_attr.shape[1]), dtype=torch.float32, device=device
    )
    padded_edge_attr = torch.cat((cls_edge_attr, edge_attr), dim=0)

    # The [CLS] atom belongs to the same graph as the first real atom.
    batch = getattr(data, "batch", None)
    if batch is None:
        batch = torch.zeros(num_nodes, dtype=torch.long, device=device)
    cls_batch = batch[0:1]
    padded_batch = torch.cat((cls_batch, batch), dim=0)

    transformed = data.clone()
    transformed.x = torch.cat((cls_x, x), dim=0)
    transformed.pos = torch.cat((cls_pos, pos), dim=0)
    transformed.edge_index = padded_edge_index
    transformed.edge_attr = padded_edge_attr
    transformed.batch = padded_batch
    return transformed


def _sample_id(data: MolecularData) -> int | str:
    value = getattr(data, "sample_id", None)
    if isinstance(value, Tensor) and value.numel():
        return int(value.flatten()[0].item())
    return "<unknown>"


__all__ = ["add_unimol_inputs"]
