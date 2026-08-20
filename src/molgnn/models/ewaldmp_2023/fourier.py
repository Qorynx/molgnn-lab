"""Aperiodic Fourier helpers for Ewald message passing."""

from __future__ import annotations

import math

import torch
from torch import Tensor


def build_k_voxel_grid(k_cutoff: float, delta_k: float) -> Tensor:
    """Return one non-zero representative from each point-symmetric k pair."""

    _positive_finite(k_cutoff, "k_cutoff")
    _positive_finite(delta_k, "delta_k")
    ratio = float(k_cutoff) / float(delta_k)
    extent = round(ratio)
    if extent < 1 or not math.isclose(ratio, extent, rel_tol=1.0e-6, abs_tol=1.0e-6):
        raise ValueError("k_cutoff / delta_k must be a positive integer")

    axis = torch.arange(-extent, extent + 1, dtype=torch.float32)
    indices = torch.cartesian_prod(axis, axis, axis)
    # Lexicographic ordering places zero at the center. Keep only one member of
    # each +/- pair and deliberately omit the zero frequency, as in the source.
    indices = indices[indices.shape[0] // 2 + 1 :]
    grid = indices * float(delta_k)
    squared_norm = grid.square().sum(dim=-1)
    tolerance = torch.finfo(grid.dtype).eps * max(1.0, float(k_cutoff) ** 2) * 8.0
    return grid[squared_norm <= float(k_cutoff) ** 2 + tolerance]


def build_k_rbf(
    k_grid: Tensor,
    *,
    num_rbf: int,
    k_cutoff: float,
) -> Tensor:
    """Evaluate the official Gaussian-plus-polynomial Fourier radial basis."""

    if isinstance(num_rbf, bool) or not isinstance(num_rbf, int) or num_rbf < 2:
        raise ValueError("num_rbf must be an integer greater than one")
    _positive_finite(k_cutoff, "k_cutoff")
    if k_grid.ndim != 2 or k_grid.shape[1] != 3 or not torch.is_floating_point(k_grid):
        raise ValueError("k_grid must be a floating tensor with shape [K, 3]")
    if not bool(torch.isfinite(k_grid).all()):
        raise ValueError("k_grid must contain only finite values")

    k_offset = 0.1 if num_rbf <= 48 else 0.25
    effective_cutoff = float(k_cutoff) + k_offset
    scaled_norm = torch.linalg.vector_norm(k_grid, dim=-1) / effective_cutoff
    centers = torch.linspace(
        0.0,
        1.0,
        num_rbf,
        dtype=k_grid.dtype,
        device=k_grid.device,
    )
    spacing = centers[1] - centers[0]
    gaussian = torch.exp(
        -0.5 * ((scaled_norm.unsqueeze(-1) - centers) / spacing).square()
    )

    # Fifth-order smooth polynomial envelope used by the official GemNet basis.
    envelope = (
        1.0
        - 21.0 * scaled_norm.pow(5)
        + 35.0 * scaled_norm.pow(6)
        - 15.0 * scaled_norm.pow(7)
    )
    envelope = torch.where(scaled_norm < 1.0, envelope, torch.zeros_like(envelope))
    return gaussian * envelope.unsqueeze(-1)


def canonicalize_positions(pos: Tensor, graph_batch: Tensor, num_graphs: int) -> Tensor:
    """Center each graph and rotate non-trivial structures into an SVD frame."""

    if pos.ndim != 2 or pos.shape[1] != 3 or not torch.is_floating_point(pos):
        raise ValueError("pos must be a floating tensor with shape [N, 3]")
    if graph_batch.shape != (pos.shape[0],) or graph_batch.dtype != torch.long:
        raise ValueError("graph_batch must be a long tensor with shape [N]")
    if (
        isinstance(num_graphs, bool)
        or not isinstance(num_graphs, int)
        or num_graphs < 1
    ):
        raise ValueError("num_graphs must be a positive integer")
    if pos.device != graph_batch.device:
        raise ValueError("pos and graph_batch must share a device")

    centers = pos.new_zeros((num_graphs, 3)).index_add(0, graph_batch, pos)
    counts = torch.bincount(graph_batch, minlength=num_graphs).to(dtype=pos.dtype)
    if bool((counts == 0).any()):
        raise ValueError("every graph must contain at least one atom")
    centered = pos - (centers / counts.unsqueeze(-1)).index_select(0, graph_batch)

    positions: list[Tensor] = []
    indices: list[Tensor] = []
    for graph_index in range(num_graphs):
        node_index = torch.nonzero(graph_batch == graph_index, as_tuple=False).flatten()
        graph_pos = centered.index_select(0, node_index)
        if graph_pos.shape[0] > 2:
            # torch.linalg.svd is the maintained equivalent of source torch.svd.
            _, _, vh = torch.linalg.svd(graph_pos, full_matrices=False)
            graph_pos = graph_pos @ vh.transpose(-2, -1)
        positions.append(graph_pos)
        indices.append(node_index)

    joined_index = torch.cat(indices)
    joined_pos = torch.cat(positions, dim=0)
    return joined_pos.index_select(0, torch.argsort(joined_index))


def voxel_damping(pos: Tensor, delta_k: float) -> Tensor:
    """Return the aperiodic voxel-average sinc damping with shape [N, 1]."""

    _positive_finite(delta_k, "delta_k")
    if pos.ndim != 2 or pos.shape[1] != 3 or not torch.is_floating_point(pos):
        raise ValueError("pos must be a floating tensor with shape [N, 3]")
    return torch.sinc(0.5 * float(delta_k) * pos).prod(dim=-1, keepdim=True)


def _positive_finite(value: float, name: str) -> None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(f"{name} must be a positive finite number")
    if not math.isfinite(float(value)) or float(value) <= 0.0:
        raise ValueError(f"{name} must be a positive finite number")


__all__ = [
    "build_k_rbf",
    "build_k_voxel_grid",
    "canonicalize_positions",
    "voxel_damping",
]
