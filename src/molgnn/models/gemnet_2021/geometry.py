"""Differentiable invariant geometry used by GemNet."""

from __future__ import annotations

import torch
from torch import Tensor


def angle_cosine(first: Tensor, second: Tensor, *, epsilon: float) -> Tensor:
    """Return stable cosine values for matching nonzero vectors."""

    denominator = torch.linalg.vector_norm(first, dim=-1) * torch.linalg.vector_norm(
        second, dim=-1
    )
    cosine = (first * second).sum(dim=-1) / denominator.clamp_min(epsilon)
    return cosine.clamp(-1.0, 1.0)


def quadruplet_geometry(
    pos: Tensor,
    edge_index: Tensor,
    interaction_edge_index: Tensor,
    quadruplet_edge_index: Tensor,
    quadruplet_interaction_index: Tensor,
    *,
    epsilon: float,
) -> tuple[Tensor, Tensor, Tensor, Tensor]:
    """Return cosines for ``phi_cab``, ``phi_abd``, and the plane angle."""

    ca_edge, db_edge = quadruplet_edge_index
    interaction = quadruplet_interaction_index
    edge_source, edge_target = edge_index
    interaction_source, interaction_target = interaction_edge_index

    c = edge_source[ca_edge]
    a = edge_target[ca_edge]
    d = edge_source[db_edge]
    b = edge_target[db_edge]
    expected_b = interaction_source[interaction]
    expected_a = interaction_target[interaction]
    if not torch.equal(b, expected_b) or not torch.equal(a, expected_a):
        raise ValueError("GemNet quadruplet references disagree with the interaction edge")

    vector_ac = pos[c] - pos[a]
    vector_ab = pos[b] - pos[a]
    vector_ba = -vector_ab
    vector_bd = pos[d] - pos[b]
    cosine_cab = angle_cosine(vector_ac, vector_ab, epsilon=epsilon)
    cosine_abd = angle_cosine(vector_ba, vector_bd, epsilon=epsilon)

    ac_plane = _vector_rejection(vector_ac, vector_ab, epsilon=epsilon)
    bd_plane = _vector_rejection(vector_bd, vector_ba, epsilon=epsilon)
    ac_norm = torch.linalg.vector_norm(ac_plane, dim=-1)
    bd_norm = torch.linalg.vector_norm(bd_plane, dim=-1)
    valid = (ac_norm > epsilon) & (bd_norm > epsilon)
    denominator = (ac_norm * bd_norm).clamp_min(epsilon)
    cosine_plane = ((ac_plane * bd_plane).sum(dim=-1) / denominator).clamp(-1.0, 1.0)
    sine_plane = (
        torch.linalg.vector_norm(torch.linalg.cross(ac_plane, bd_plane, dim=-1), dim=-1)
        / denominator
    ).clamp(0.0, 1.0)
    cosine_plane = torch.where(valid, cosine_plane, torch.ones_like(cosine_plane))
    sine_plane = torch.where(valid, sine_plane, torch.zeros_like(sine_plane))
    return cosine_cab, cosine_abd, cosine_plane, sine_plane


def _vector_rejection(vector: Tensor, axis: Tensor, *, epsilon: float) -> Tensor:
    squared_norm = axis.square().sum(dim=-1, keepdim=True)
    projection = (vector * axis).sum(dim=-1, keepdim=True) / squared_norm.clamp_min(
        epsilon
    )
    return vector - projection * axis


__all__ = ["angle_cosine", "quadruplet_geometry"]
