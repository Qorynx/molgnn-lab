"""Gaussian RBF, unordered pair embedding, and the paper interaction layer."""

from __future__ import annotations

import torch
from torch import Tensor, nn
from torch_geometric.utils import scatter

from .constants import (
    MGCN_ETA,
    MGCN_NUM_RBF,
    MGCN_RBF_BETA,
    MGCN_RBF_HIGH,
    MGCN_RBF_LOW,
)


class GaussianRBF(nn.Module):
    """Paper Eq. (2): Gaussian radial basis expansion over a fixed range.

    ``exp(-beta * (d - mu_k)^2)`` with ``K`` linearly spaced centers between
    ``low`` and ``high``.
    """

    def __init__(
        self,
        num_rbf: int = MGCN_NUM_RBF,
        low: float = MGCN_RBF_LOW,
        high: float = MGCN_RBF_HIGH,
        beta: float = MGCN_RBF_BETA,
    ) -> None:
        super().__init__()
        if num_rbf < 1:
            raise ValueError("num_rbf must be at least 1")
        if high <= low:
            raise ValueError("rbf_high must be greater than rbf_low")
        if beta <= 0:
            raise ValueError("rbf_beta must be positive")
        self.num_rbf = num_rbf
        self.low = low
        self.high = high
        self.beta = beta
        centers = torch.linspace(low, high, num_rbf)
        self.register_buffer("centers", centers)

    def forward(self, distances: Tensor) -> Tensor:
        """Expand distances with shape ``[E]`` to ``[E, num_rbf]``."""
        diff = distances.unsqueeze(-1) - self.centers.unsqueeze(0)
        return torch.exp(-self.beta * diff.square())


class PairEmbedding(nn.Module):
    """Collision-free unordered element-pair embedding.

    Each unordered pair ``(Z_i, Z_j)`` maps to a unique integer via the
    Cantor pairing function, producing the same id for ``(Z_i, Z_j)`` and
    ``(Z_j, Z_i)``.  The embedding vocabulary is sized to cover all pairs
    up to ``max_atomic_number``.
    """

    def __init__(self, hidden_dim: int, max_atomic_number: int) -> None:
        super().__init__()
        # Cantor upper bound: max_pair_id = max_Z^2 + (0 - 1)^2 // 4
        max_pairs = max_atomic_number * max_atomic_number + 1
        self.embedding = nn.Embedding(max_pairs, hidden_dim)

    @staticmethod
    def pair_id(a: Tensor, b: Tensor) -> Tensor:
        """Return a unique id for the unordered pair ``(a, b)``.

        Uses the Cantor pairing function for unordered pairs:
        ``k = a * b + (abs(a - b) - 1)^2 // 4``
        """
        diff = (a - b).abs() - 1
        clamped = diff.clamp(min=0)
        return a * b + clamped.square() // 4

    def forward(self, source_atomic: Tensor, target_atomic: Tensor) -> Tensor:
        """Embed edges from source and target atomic numbers.

        Args:
            source_atomic: ``[E]`` atomic numbers of source nodes.
            target_atomic: ``[E]`` atomic numbers of target nodes.

        Returns:
            ``[E, hidden_dim]`` edge embeddings.
        """
        ids = self.pair_id(source_atomic, target_atomic)
        return self.embedding(ids)


class InteractionLayer(nn.Module):
    """One MGCN interaction step implementing paper Eqs. (5)--(6) with
    synchronous update timing.

    The node message (Eq. (6)) consumes the level-``l`` edge state, while
    the edge update (Eq. (5)) produces level ``l+1``.  The new edge state
    is **not** fed back into the same layer's node update.
    """

    def __init__(
        self, hidden_dim: int, rbf_dim: int, eta: float = MGCN_ETA
    ) -> None:
        super().__init__()
        if eta < 0 or eta > 1:
            raise ValueError("eta must be in [0, 1]")
        self.eta = eta
        self.edge_update = nn.Linear(hidden_dim, hidden_dim)
        self.atom_proj = nn.Linear(hidden_dim, hidden_dim)
        self.dist_proj = nn.Linear(rbf_dim, hidden_dim)
        self.edge_proj = nn.Linear(hidden_dim, hidden_dim)
        self.message_linear = nn.Linear(hidden_dim, hidden_dim)

    def forward(
        self,
        atom: Tensor,
        edge: Tensor,
        rbf: Tensor,
        edge_index: Tensor,
    ) -> tuple[Tensor, Tensor]:
        """Apply one interaction layer.

        Args:
            atom: ``[N, hidden_dim]`` current atom states.
            edge: ``[E, hidden_dim]`` current edge states (level ``l``).
            rbf: ``[E, num_rbf]`` expanded distances.
            edge_index: ``[2, E]`` directed edges ``j -> i`` (source, target).

        Returns:
            ``(atom_next, edge_next)`` where ``atom_next`` is ``[N, hidden_dim]``
            and ``edge_next`` is ``[E, hidden_dim]``.
        """
        source, target = edge_index

        # Eq. (5): edge_next = eta * edge + (1 - eta) * W_ue(atom_s * atom_t)
        pair_product = atom[source] * atom[target]
        edge_update = self.edge_update(pair_product)
        edge_next = self.eta * edge + (1 - self.eta) * edge_update

        # Eq. (6): message = tanh(W_uv(M_atom(atom_s) * M_dist(rbf) + M_edge(edge)))
        atom_feat = self.atom_proj(atom[source])
        dist_feat = self.dist_proj(rbf)
        edge_feat = self.edge_proj(edge)
        message = torch.tanh(self.message_linear(atom_feat * dist_feat + edge_feat))

        # Eq. (4): atom_next_i = sum_{j != i} message_ji
        atom_next = scatter(
            message, target, dim=0, dim_size=atom.shape[0], reduce="sum"
        )

        return atom_next, edge_next


__all__ = ["GaussianRBF", "InteractionLayer", "PairEmbedding"]