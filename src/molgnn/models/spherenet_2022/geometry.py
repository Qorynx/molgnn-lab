"""Angle and cyclic torsion computation from prepared SphereNet indices.

The transform owns only the discrete index relationships (``edge_index``,
``triplet_edge_index``, ``torsion_pair_index``).  Distances, interior angles
and cyclic dihedral angles are recomputed from ``pos`` in the model forward
pass so that coordinate gradients stay intact for force/energy differentials.
"""

from __future__ import annotations

import math

import torch
from torch import Tensor

_TORSION_EPS = 1e-8


def compute_distances(pos: Tensor, edge_index: Tensor) -> Tensor:
    """Return ``[E]`` directed edge distances ``||pos[i] - pos[j]||``.

    ``edge_index`` has source ``j``, target ``i`` (``j -> i``).  The result
    is non-negative and finite because the transform rejects coincident atoms.
    """

    j, i = edge_index
    return (pos[i] - pos[j]).pow(2).sum(dim=-1).sqrt()


def compute_angles(
    pos: Tensor, edge_index: Tensor, triplet_edge_index: Tensor
) -> Tensor:
    """Return ``[Q]`` interior angles ``theta`` in ``[0, pi]``.

    For each triplet ``k -> j -> i`` (``triplet_edge_index[0] = k->j``,
    ``[1] = j->i``), the angle is the angle between the vectors ``j -> k``
    and ``j -> i``.
    """

    idx_kj, idx_ji = triplet_edge_index
    j_atom = edge_index[0, idx_ji]  # center j (source of j->i)
    pos_ji = pos[edge_index[1, idx_ji]] - pos[j_atom]
    pos_jk = pos[edge_index[0, idx_kj]] - pos[j_atom]
    dot = (pos_ji * pos_jk).sum(dim=-1)
    cross_norm = torch.linalg.cross(pos_ji, pos_jk).norm(dim=-1)
    return torch.atan2(cross_norm, dot)


def compute_torsions(
    pos: Tensor,
    edge_index: Tensor,
    triplet_edge_index: Tensor,
    torsion_pair_index: Tensor,
) -> Tensor:
    """Return ``[Q]`` cyclic dihedral angles ``phi`` in ``(0, 2*pi]``.

    For each base triplet ``k -> j -> i``, all incoming edges ``k_n -> j``
    with ``k_n != i`` are candidates.  Every candidate's signed dihedral from
    the reference plane ``(j, i, k)`` to the candidate plane ``(j, i, k_n)``
    is computed, wrapped to ``(0, 2*pi]``, and the *minimum* positive value is
    selected per triplet (the official DIG ``reduce='min'`` behavior).

    A degenerate plane (collinear reference or candidate) yields a finite
    ``2*pi`` fallback; the ``torch.where`` guard prevents NaN gradient
    propagation through exact ``atan2(0, 0)``.
    """

    idx_kj, idx_ji = triplet_edge_index
    triplet_id, candidate_edge = torsion_pair_index

    # Per-triplet geometry [Q, 3].
    j_atom = edge_index[0, idx_ji]  # [Q] center atom per triplet
    pos_ji = pos[edge_index[1, idx_ji]] - pos[j_atom]  # [Q, 3]
    pos_j0 = pos[edge_index[0, idx_kj]] - pos[j_atom]  # [Q, 3]

    # Per-candidate geometry [R, 3].
    candidate_j_atom = j_atom[triplet_id]  # [R]
    pos_ji_c = pos_ji[triplet_id]  # [R, 3]
    pos_j0_c = pos_j0[triplet_id]  # [R, 3]
    pos_jk_c = pos[edge_index[0, candidate_edge]] - pos[candidate_j_atom]  # [R, 3]

    n1 = torch.linalg.cross(pos_ji_c, pos_j0_c)
    n2 = torch.linalg.cross(pos_ji_c, pos_jk_c)

    a = (n1 * n2).sum(dim=-1)
    b = (torch.linalg.cross(n1, n2) * pos_ji_c).sum(dim=-1) / (
        pos_ji_c.norm(dim=-1).clamp(min=_TORSION_EPS)
    )

    magnitude = torch.hypot(a, b)
    degenerate = magnitude < _TORSION_EPS
    a_safe = torch.where(degenerate, torch.ones_like(a), a)
    b_safe = torch.where(degenerate, torch.zeros_like(b), b)
    torsion1 = torch.atan2(b_safe, a_safe)
    torsion1 = torch.where(torsion1 <= 0, torsion1 + 2 * math.pi, torsion1)

    return _scatter_min(torsion1, triplet_id, dim_size=int(idx_kj.shape[0]))


def _scatter_min(values: Tensor, index: Tensor, dim_size: int) -> Tensor:
    """Segment min with a ``torch_scatter``-free implementation."""

    out = values.new_full((dim_size,), float("inf"))
    out.scatter_reduce_(0, index, values, reduce="amin", include_self=False)
    return out


__all__ = [
    "compute_angles",
    "compute_distances",
    "compute_torsions",
]