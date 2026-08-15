"""Building blocks for the MAT 2020 architecture.

MAT (Maziarka et al., 2020) is the Molecule Attention Transformer. Its core
innovation — paper Equation 2 — augments the scaled-dot-product self-
attention with the molecular graph adjacency ``A`` and an inter-atomic
distance kernel ``g(D)``::

    A^(i) = λ_a · ρ(Q K^T / √d_k) + λ_d · g(D) + λ_g · A    (Eq. 2)

The two are mixed through a learnable (or frozen) convex combination
``λ = (λ_a, λ_d, λ_g)``.  The kernel ``g`` is either ``softmax(-D)`` (per
row) or the element-wise ``exp(-D)``; the upstream defaults to ``softmax``.

The layer stack follows the standard Transformer encoder pattern
(Vaswani et al., 2017): ``N`` blocks of (multi-head self-attention →
position-wise feed-forward) wrapped in pre-norm residual connections
plus a final LayerNorm.  An artificial dummy node sits at row 0 of every
input matrix (added by the ``add_mat_inputs`` graph transform) so the
model can effectively down-weight irrelevant inputs by attending to it.
"""

from __future__ import annotations

import math

import torch
from torch import Tensor, nn
import torch.nn.functional as F


# Module-level constants — kept here rather than on the class so the
# supported options are easy to enumerate from outside the class.
_DISTANCE_KERNELS = ("softmax", "exp")
_AGGREGATION_MODES = ("mean", "sum", "dummy_node")


class LayerNorm(nn.Module):
    """Standard LayerNorm with elementwise affine parameters.

    Mirrors the upstream ``transformer.LayerNorm``; PyG / ``torch.nn`` ship
    their own LayerNorm but matching the upstream formulation exactly keeps
    the port's behaviour identical.
    """

    def __init__(self, features: int, eps: float = 1.0e-6) -> None:
        super().__init__()
        self.a_2 = nn.Parameter(torch.ones(features))
        self.b_2 = nn.Parameter(torch.zeros(features))
        self.eps = eps

    def forward(self, x: Tensor) -> Tensor:
        mean = x.mean(-1, keepdim=True)
        std = x.std(-1, keepdim=True)
        return self.a_2 * (x - mean) / (std + self.eps) + self.b_2


class PositionwiseFeedForward(nn.Module):
    """Two-layer position-wise MLP with configurable inner activation.

    Mirrors the upstream ``transformer.PositionwiseFeedForward``: the
    intermediate activation is always ``F.leaky_relu(x, negative_slope=...)``
    (the slope defaults to ``0.0`` so this becomes ReLU); the final layer
    uses the user-selected ``dense_output_nonlinearity`` ('relu' /
    'tanh' / 'none').  The lab's port exposes the cleaner
    ``output_activation`` strings (matching the upstream choices) so the
    upstream paper's defaults map to ``output_activation='relu'``.
    """

    def __init__(
        self,
        d_model: int,
        n_dense: int = 2,
        dropout: float = 0.1,
        leaky_relu_slope: float = 0.0,
        output_activation: str = "relu",
    ) -> None:
        super().__init__()
        if n_dense < 1:
            raise ValueError("n_dense must be >= 1")
        if not 0 <= dropout < 1:
            raise ValueError("dropout must be in [0, 1)")
        if output_activation not in ("relu", "tanh", "none"):
            raise ValueError(
                f"output_activation must be one of ('relu', 'tanh', 'none'); "
                f"got {output_activation!r}"
            )

        self.d_model = d_model
        self.n_dense = n_dense
        self.leaky_relu_slope = leaky_relu_slope

        self.linears = nn.ModuleList(
            nn.Linear(d_model, d_model) for _ in range(n_dense)
        )
        self.dropouts = nn.ModuleList(
            nn.Dropout(dropout) for _ in range(n_dense)
        )

        if output_activation == "tanh":
            self._final_activation: nn.Module = nn.Tanh()
        else:
            # "relu" and "none" both pass the upstream's leaky-relu(x, slope)
            # expression; for "none" the slope is 0 → identity in spirit, but
            # we keep leaky-relu(x, slope=leaky_relu_slope) so the field name
            # matches the paper.
            self._final_activation = _LeakyRelu(leaky_relu_slope)

    def forward(self, x: Tensor) -> Tensor:
        # Apply leaky-relu + dropout on every intermediate layer and the
        # configured activation + dropout on the final layer.
        for linear, dropout in zip(self.linears[:-1], self.dropouts[:-1]):
            x = dropout(F.leaky_relu(linear(x), negative_slope=self.leaky_relu_slope))
        x = self.linears[-1](x)
        x = self._final_activation(x)
        return self.dropouts[-1](x)


class _LeakyRelu(nn.Module):
    """Thin wrapper so we can store the final activation as ``nn.Module``."""

    def __init__(self, negative_slope: float) -> None:
        super().__init__()
        self.negative_slope = float(negative_slope)

    def forward(self, x: Tensor) -> Tensor:
        return F.leaky_relu(x, negative_slope=self.negative_slope)


class SublayerConnection(nn.Module):
    """Pre-norm residual: ``x + Dropout(sublayer(LayerNorm(x)))`` (Eq. 1)."""

    def __init__(self, size: int, dropout: float) -> None:
        super().__init__()
        self.norm = LayerNorm(size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: Tensor, sublayer: nn.Module) -> Tensor:
        return x + self.dropout(sublayer(self.norm(x)))


class MoleculeMultiHeadAttention(nn.Module):
    """Modified multi-head self-attention with adjacency and distance bias (Eq. 2).

    Per-head attention for atom ``i`` over atom ``j`` is a convex
    combination of three (per-graph, per-head) ``N x N`` matrices:

      * ``λ_a · ρ(Q K^T / √d_k)`` — the standard softmax-scaled attention
        (Vaswani et al., 2017);
      * ``λ_d · g(D)`` — a distance matrix kernel, either
        ``g(d) = exp(-d)`` (element-wise) or the row-wise
        ``g(d) = softmax(-d)``;
      * ``λ_g · A`` — the row-normalised graph adjacency (one self-loop
        on each atom, ``+eps`` in the denominator avoids divide-by-zero on
        the disconnected dummy node).

    The lambdas are stored as either a frozen tuple of three floats or
    three learnable ``softmax_attention / softmax_distance / softmax_adjacency``
    parameters (the upstream's ``trainable_lambda=True`` branch); we expose
    that switch via the same flag for parity.
    """

    def __init__(
        self,
        h: int,
        d_model: int,
        dropout: float = 0.1,
        lambda_attention: float = 0.3,
        lambda_distance: float = 0.3,
        trainable_lambda: bool = False,
        distance_matrix_kernel: str = "softmax",
        attn_eps: float = 1.0e-6,
    ) -> None:
        super().__init__()
        if d_model % h != 0:
            raise ValueError(f"d_model ({d_model}) must be divisible by h ({h})")
        if not 0 <= dropout < 1:
            raise ValueError("dropout must be in [0, 1)")
        if distance_matrix_kernel not in _DISTANCE_KERNELS:
            raise ValueError(
                f"distance_matrix_kernel must be one of {_DISTANCE_KERNELS}; "
                f"got {distance_matrix_kernel!r}"
            )

        self.d_k = d_model // h
        self.h = h
        self.trainable_lambda = trainable_lambda
        self.attn_eps = float(attn_eps)

        # Frozen tuple of three lambdas (paper default: 0.3, 0.3, 0.4). The
        # third is derived so the three sum to one — the paper requires
        # ``λ_a + λ_d + λ_g = 1``.
        lambda_adjacency = 1.0 - lambda_attention - lambda_distance
        if trainable_lambda:
            # Upstream style: store three learnable scalars and softmax
            # them at forward time so the constraint holds automatically.
            lambdas_tensor = torch.tensor(
                [lambda_attention, lambda_distance, lambda_adjacency],
                requires_grad=True,
            )
            self.lambdas = nn.Parameter(lambdas_tensor)
        else:
            self.lambdas = (lambda_attention, lambda_distance, lambda_adjacency)

        # Four linear projections: Q, K, V, and the output projection.
        # Matches the upstream ``clones(nn.Linear(d_model, d_model), 4)``.
        self.linears = nn.ModuleList(
            nn.Linear(d_model, d_model) for _ in range(4)
        )
        self.dropout = nn.Dropout(dropout)

        if distance_matrix_kernel == "softmax":
            self._distance_kernel: nn.Module = _DistanceSoftmax()
        else:
            self._distance_kernel = _DistanceExp()

    def forward(
        self,
        query: Tensor,
        key: Tensor,
        value: Tensor,
        adj_matrix: Tensor,
        distances_matrix: Tensor,
        mask: Tensor | None = None,
    ) -> tuple[Tensor, Tensor, Tensor]:
        """Return ``(out, p_weighted, p_attn)`` matching the upstream contract.

        Args:
            query, key, value: ``(B, H, N, d_k)`` per-head projections.
            adj_matrix: ``(B, N, N)`` adjacency with one self-loop per atom.
            distances_matrix: ``(B, N, N)`` pairwise inter-atomic distances.
            mask: ``(B, N)`` boolean, ``True`` for real atoms, ``False`` for
                padding / dummy padding positions.  The first row / column
                of every per-graph slice is the dummy atom — its entries in
                ``mask`` are ``True`` because the dummy is a real (if
                disconnected) atom.

        Returns:
            ``out``: ``(B, N, d_model)`` post-projection per-atom features.
            ``p_weighted``: ``(B, H, N, N)`` combined attention probabilities.
            ``p_attn``: ``(B, H, N, N)`` plain softmax-attention probabilities
                (before mixing with adjacency and distance).
        """

        nbatches = query.size(0)
        # 1) Project to Q, K, V and reshape ``(B, N, d_model)`` into
        # ``(B, H, N, d_k)``.  Matches the upstream
        # ``[l(x).view(nbatches, -1, self.h, self.d_k).transpose(1, 2) for ...]``.
        query, key, value = [
            l(x).view(nbatches, -1, self.h, self.d_k).transpose(1, 2)
            for l, x in zip(self.linears[:3], (query, key, value))
        ]
        # Scaled dot-product attention scores: ``(B, H, N, N)``.
        scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(self.d_k)
        if mask is not None:
            # Mask padded positions across query rows.  The upstream
            # ``attention()`` helper broadcasts via
            # ``mask.unsqueeze(1).repeat(1, H, N, 1)`` which produces a
            # wrong-shape tensor (21 = 3·7 instead of 3·4·7·7) because
            # ``repeat`` on a 3-D tensor with 4 sizes prepends a new dim
            # rather than broadcasting.  This port uses
            # ``mask[:, :, None, None]``-style broadcasting which expands
            # correctly to ``(B, 1, 1, N)``.
            scores = scores.masked_fill(~mask[:, None, None, :], float("-inf"))
        p_attn = F.softmax(scores, dim=-1)

        # Row-normalised adjacency (Eq. 2 denominator).  ``+ eps`` prevents
        # divide-by-zero for the disconnected dummy node.
        adj_normalised = adj_matrix / (adj_matrix.sum(dim=-1, keepdim=True) + self.attn_eps)
        # ``(B, 1, N, N)`` so it broadcasts across heads.
        p_adj = adj_normalised.unsqueeze(1)

        # Distance kernel applied per graph (``(B, N, N)`` → ``(B, 1, N, N)``).
        p_dist = self._distance_kernel(distances_matrix).unsqueeze(1)

        # Convex combination of the three streams.
        if self.trainable_lambda:
            # Softmax the trainable lambdas so the three weights sum to one.
            softmax_lambdas = F.softmax(self.lambdas, dim=-1)
            lambda_a, lambda_d, lambda_g = (
                softmax_lambdas[0],
                softmax_lambdas[1],
                softmax_lambdas[2],
            )
        else:
            lambda_a, lambda_d, lambda_g = self.lambdas
        p_weighted = lambda_a * p_attn + lambda_d * p_dist + lambda_g * p_adj
        p_weighted = self.dropout(p_weighted)

        # Apply the mixed attention to the value projections.
        out = torch.matmul(p_weighted, value)
        # Concatenate heads and apply the final linear.
        out = out.transpose(1, 2).contiguous().view(nbatches, -1, self.h * self.d_k)
        return self.linears[-1](out), p_weighted, p_attn


class _DistanceSoftmax(nn.Module):
    """Per-row softmax(-D).  Matches the upstream's ``distance_matrix_kernel='softmax'``."""

    def forward(self, distances_matrix: Tensor) -> Tensor:
        return F.softmax(-distances_matrix, dim=-1)


class _DistanceExp(nn.Module):
    """Element-wise ``exp(-D)``.  Matches the upstream's ``distance_matrix_kernel='exp'``."""

    def forward(self, distances_matrix: Tensor) -> Tensor:
        return torch.exp(-distances_matrix)


class EncoderLayer(nn.Module):
    """One Transformer-encoder block: attention → feed-forward, both pre-normed.

    Follows Figure 1 of Vaswani et al. (2017) and the upstream
    ``transformer.EncoderLayer``: the input is wrapped in two
    ``SublayerConnection`` residual blocks (sublayer = attention or
    feed-forward) so the residual stream keeps the input-shape invariant.
    """

    def __init__(
        self,
        size: int,
        self_attn: MoleculeMultiHeadAttention,
        feed_forward: PositionwiseFeedForward,
        dropout: float,
    ) -> None:
        super().__init__()
        self.self_attn = self_attn
        self.feed_forward = feed_forward
        self.sublayer = nn.ModuleList(
            SublayerConnection(size, dropout) for _ in range(2)
        )
        self.size = size

    def forward(
        self,
        x: Tensor,
        adj_matrix: Tensor,
        distances_matrix: Tensor,
        mask: Tensor,
    ) -> Tensor:
        # Pre-norm attention block.  ``self_attn`` returns ``(out, _, _)``
        # but only the output is used here (the upstream stores
        # ``self.attn`` / ``self.self_attn`` for interpretability but does
        # not consume them downstream).
        x = self.sublayer[0](
            x,
            lambda x: self.self_attn(
                x,
                x,
                x,
                adj_matrix,
                distances_matrix,
                mask=mask,
            )[0],
        )
        # Pre-norm feed-forward block.
        x = self.sublayer[1](x, self.feed_forward)
        return x


__all__ = [
    "EncoderLayer",
    "LayerNorm",
    "MoleculeMultiHeadAttention",
    "PositionwiseFeedForward",
    "SublayerConnection",
    "_AGGREGATION_MODES",
    "_DISTANCE_KERNELS",
]