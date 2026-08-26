"""DGT transformer layers: dense multi-head attention and dual-graph blocks.

DGT runs two independent transformer streams on the atom graph and the bond
(line) graph.  ``DGTAttention`` is the dense multi-head attention with a
structural positional bias; ``DGTLayer`` applies one block to both streams with
separate weights and no cross-stream coupling.
"""

from __future__ import annotations

import torch
from torch import Tensor, nn
from torch_geometric.utils import to_dense_batch


def activation(name: str) -> nn.Module:
    """Small activation factory (no YACS/config system)."""

    lowered = name.lower()
    if lowered == "gelu":
        return nn.GELU()
    if lowered == "relu":
        return nn.ReLU()
    if lowered == "silu" or lowered == "swish":
        return nn.SiLU()
    if lowered == "mish":
        return nn.Mish()
    raise ValueError(f"unsupported activation {name!r}")


class DGTAttention(nn.Module):
    """Multi-head attention over a dense batch with dense PE bias.

    ``head_dim = dim_h // num_heads``.  Scores are merged across heads before
    softmax, so the score tensor has shape ``[B, N, N, dim_h]``.
    """

    def __init__(
        self,
        dim_h: int,
        num_heads: int,
        *,
        attn_dropout: float = 0.0,
    ) -> None:
        super().__init__()
        if dim_h % num_heads != 0:
            raise ValueError("dim_h must be divisible by num_heads")
        self.dim_h = dim_h
        self.num_heads = num_heads
        self.head_dim = dim_h // num_heads
        self.Q = nn.Linear(dim_h, dim_h, bias=False)
        self.K = nn.Linear(dim_h, dim_h, bias=False)
        self.V = nn.Linear(dim_h, dim_h, bias=False)
        self.out_proj = nn.Linear(dim_h, dim_h)
        self.attn_dropout = nn.Dropout(p=attn_dropout)

    def forward(
        self,
        h: Tensor,
        e_att: Tensor,
        e_val: Tensor,
        attn_mask: Tensor,
    ) -> Tensor:
        """Attend over a padded dense batch.

        Args:
            h: ``[B, N, dim_h]`` node features.
            e_att: ``[B, N, N, dim_h]`` attention bias.
            e_val: ``[B, N, N, dim_h]`` value bias.
            attn_mask: ``[B, N, N]`` boolean mask (False = padding).

        Returns:
            ``[B, N, dim_h]`` updated features.
        """

        batch, num_nodes, _ = h.shape
        q = self.Q(h).view(batch, num_nodes, self.num_heads, self.head_dim)
        k = self.K(h).view(batch, num_nodes, self.num_heads, self.head_dim)
        v = self.V(h)

        scaling = float(self.head_dim) ** -0.5
        scores = torch.einsum("bihk,bjhk->bijh", q, k * scaling).unsqueeze(-1)

        # Mask padding (keep -1e24 to avoid NaN in softmax).
        mask = attn_mask.view(batch, num_nodes, num_nodes, 1, 1)
        scores = scores - 1e24 * (~mask)

        # Bias with the dense structural encodings.
        e_att_heads = e_att.view(
            batch, num_nodes, num_nodes, self.num_heads, self.head_dim
        )
        scores = scores + e_att_heads
        scores = scores.reshape(
            batch, num_nodes, num_nodes, self.num_heads * self.head_dim
        )

        scores = torch.softmax(scores, dim=2)

        # Dropout connections (OFFICIAL CODE): the dropout is applied to the
        # padding/validity mask itself and broadcast over head/feature
        # channels, so every feature of one connection (i, j) shares a mask.
        connection_dropout = self.attn_dropout(mask.float()).squeeze(-1)
        scores = scores * connection_dropout

        # h[b,i,k] = sum_j scores[b,i,j,k] * (v[b,j,k] + e_val[b,i,j,k])
        output = torch.einsum("bijk,bjk->bik", scores, v)
        output = output + (scores * e_val).sum(2)
        return self.out_proj(output)


class DGTLayer(nn.Module):
    """One transformer block applied to the atom and bond streams.

    The two streams are processed independently with separate weights.  The
    dense attention biases and masks are read from ``batch`` (populated once
    per forward pass by ``DGTEmbedder``).
    """

    def __init__(
        self,
        dim_h: int,
        num_heads: int,
        *,
        act: str = "gelu",
        dropout: float = 0.0,
        attn_dropout: float = 0.0,
        batch_norm: bool = True,
    ) -> None:
        super().__init__()
        if not batch_norm:
            raise ValueError("DGT requires batch_norm=True in this port")
        self.dim_h = dim_h
        self.num_heads = num_heads
        self.activation = activation(act)
        self.dropout_n = nn.Dropout(p=dropout)
        self.dropout_e = nn.Dropout(p=dropout)
        self.ff_dropout1_n = nn.Dropout(p=dropout)
        self.ff_dropout2_n = nn.Dropout(p=dropout)
        self.ff_dropout1_e = nn.Dropout(p=dropout)
        self.ff_dropout2_e = nn.Dropout(p=dropout)

        # Atom stream.
        self.self_attn_n = DGTAttention(
            dim_h, num_heads, attn_dropout=attn_dropout
        )
        self.norm1_n = nn.BatchNorm1d(dim_h)
        self.ff_linear1_n = nn.Linear(dim_h, dim_h * 2)
        self.ff_linear2_n = nn.Linear(dim_h * 2, dim_h)
        self.norm2_n = nn.BatchNorm1d(dim_h)

        # Bond stream (identical structure, separate weights).
        self.self_attn_e = DGTAttention(
            dim_h, num_heads, attn_dropout=attn_dropout
        )
        self.norm1_e = nn.BatchNorm1d(dim_h)
        self.ff_linear1_e = nn.Linear(dim_h, dim_h * 2)
        self.ff_linear2_e = nn.Linear(dim_h * 2, dim_h)
        self.norm2_e = nn.BatchNorm1d(dim_h)

    @staticmethod
    def _apply_norm(norm: nn.BatchNorm1d, x: Tensor) -> Tensor:
        """BatchNorm with a train-mode fallback for single-row streams.

        ``nn.BatchNorm1d`` requires more than one row in training mode.  A
        one-row stream (single-atom molecule, single-bond bond graph) is
        normalized with the running statistics instead — numerically the same
        as an eval-mode pass for that call.  Empty streams pass through and
        two-or-more-row streams keep the plain BatchNorm behavior.
        """

        if x.shape[0] == 0:
            return x
        if x.shape[0] == 1 and norm.training:
            return (x - norm.running_mean) / torch.sqrt(norm.running_var + norm.eps)
        return norm(x)

    def forward(self, batch):
        # --- Atom stream ---
        h_n = batch.x
        h_n_in = h_n
        h_n_dense, _ = to_dense_batch(h_n, batch.batch)
        h_n = self.self_attn_n(
            h_n_dense,
            e_att=batch.edge_attention,
            e_val=batch.edge_values,
            attn_mask=batch.attn_mask,
        )[batch.mask]
        h_n = self.dropout_n(h_n)
        h_n = h_n + h_n_in
        h_n = self._apply_norm(self.norm1_n, h_n)
        h_n = h_n + self._n_ff_block(h_n)
        h_n = self._apply_norm(self.norm2_n, h_n)
        batch.x = h_n

        # --- Bond stream ---
        h_e = batch.e
        if h_e.numel() == 0:
            # No undirected bonds (single atoms): keep the empty state and
            # skip BatchNorm, which cannot run on an empty batch.
            batch.e = h_e
            return batch
        h_e_in = h_e
        h_e_dense, _ = to_dense_batch(
            h_e, batch.dgt_e_batch, batch_size=batch.mask.size(0)
        )
        h_e = self.self_attn_e(
            h_e_dense,
            e_att=batch.e2e_edge_attention,
            e_val=batch.e2e_edge_values,
            attn_mask=batch.e_attn_mask,
        )[batch.e_mask]
        h_e = self.dropout_e(h_e)
        h_e = h_e + h_e_in
        h_e = self._apply_norm(self.norm1_e, h_e)
        h_e = h_e + self._e_ff_block(h_e)
        h_e = self._apply_norm(self.norm2_e, h_e)
        batch.e = h_e

        return batch

    def _n_ff_block(self, x: Tensor) -> Tensor:
        x = self.ff_dropout1_n(self.activation(self.ff_linear1_n(x)))
        return self.ff_dropout2_n(self.ff_linear2_n(x))

    def _e_ff_block(self, x: Tensor) -> Tensor:
        x = self.ff_dropout1_e(self.activation(self.ff_linear1_e(x)))
        return self.ff_dropout2_e(self.ff_linear2_e(x))


__all__ = ["DGTAttention", "DGTLayer", "activation"]
