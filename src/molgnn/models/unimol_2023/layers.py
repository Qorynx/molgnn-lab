"""Custom Transformer layers ported from Uni-Mol (Zhou et al., ICLR 2023).

The backbone is a pre-LN Transformer whose self-attention is augmented with
an *additive pair bias* derived from 3-D inter-atomic distances (paper
Eq. 1-2):

    q_ij^{l+1} = q_ij^l + Q_i^T K_j / sqrt(d)          (Eq. 1)
    Attn(Q, K, V) = softmax(Q K^T / sqrt(d) + q_ij) V  (Eq. 2)

The pair bias is produced by a Gaussian RBF over pairwise distances
(:class:`GaussianLayer`) projected to one value per attention head
(:class:`NonLinearHead`).  Each :class:`TransformerEncoderLayer` adds the
running pair representation to the attention logits and returns its own
attention weights, which :class:`TransformerEncoderWithPair` accumulates
into the updated pair representation.
"""

from __future__ import annotations

import math

import torch
from torch import Tensor, nn

# Upstream Uni-Mol uses GELU in the FFN; hard-code it to match.
_ACTIVATION = nn.GELU


class GaussianLayer(nn.Module):
    """K Gaussian RBF kernels over pairwise distances.

    Faithful port of the upstream ``GaussianLayer`` (``unimol.py``): learned
    per-kernel ``means`` / ``stds`` (initialised ``uniform(0, 3)``) and a
    per-edge-type affine ``mul`` / ``bias``.  Because the lab has no atom-type
    vocabulary, the edge-type axis collapses to a single value (1 edge type),
    so ``mul`` / ``bias`` are ``[1, 1]`` embeddings indexed at row 0.
    """

    def __init__(self, K: int = 128) -> None:
        super().__init__()
        self.K = K
        # One row of K kernels; the single edge-type row for the affine map.
        self.means = nn.Embedding(1, K)
        self.stds = nn.Embedding(1, K)
        self.mul = nn.Embedding(1, 1)
        self.bias = nn.Embedding(1, 1)
        nn.init.uniform_(self.means.weight, 0, 3)
        nn.init.uniform_(self.stds.weight, 0, 3)
        nn.init.constant_(self.bias.weight, 0)
        nn.init.constant_(self.mul.weight, 1)

    def forward(self, x: Tensor) -> Tensor:
        """Map a distance matrix ``[bsz, N, N]`` to ``[bsz, N, N, K]``.

        ``x`` carries the pairwise distances with an explicit trailing
        single-edge-type axis (``[bsz, N, N, 1]``); the affine ``mul`` /
        ``bias`` scale and shift the raw distance before the Gaussian kernel.
        """

        mul = self.mul.weight[0].type_as(x)
        bias = self.bias.weight[0].type_as(x)
        x = mul * x + bias  # [bsz, N, N, 1]
        mean = self.means.weight.view(-1)  # [K]
        std = self.stds.weight.view(-1).abs() + 1e-5  # [K]
        x = x - mean[None, None, None, :]
        x = x / std[None, None, None, :]
        # Normalised Gaussian PDF (upstream ``gaussian`` helper).
        return torch.exp(-0.5 * x * x) / (math.sqrt(2.0 * math.pi) * std[None, None, None, :])


class NonLinearHead(nn.Module):
    """Two-layer MLP with GELU between (upstream ``NonLinearHead``).

    ``Linear(in_dim, in_dim) -> gelu -> Linear(in_dim, out_dim)`` — the
    hidden width defaults to the input width, matching the upstream.
    """

    def __init__(self, in_dim: int, out_dim: int) -> None:
        super().__init__()
        self.linear1 = nn.Linear(in_dim, in_dim)
        self.linear2 = nn.Linear(in_dim, out_dim)
        self.activation = _ACTIVATION()

    def forward(self, x: Tensor) -> Tensor:
        x = self.linear1(x)
        x = self.activation(x)
        x = self.linear2(x)
        return x


class TransformerEncoderLayer(nn.Module):
    """Pre-LN Transformer encoder layer with additive pair bias.

    Mirrors the upstream ``TransformerEncoderLayer`` (pre-LN by default):
    LayerNorm -> self-attention (with the pair bias added to the attention
    logits) -> residual -> LayerNorm -> FFN (GELU, 4x ratio) -> residual.
    """

    def __init__(
        self,
        embed_dim: int,
        ffn_embed_dim: int,
        num_heads: int,
        dropout: float = 0.1,
        attention_dropout: float = 0.1,
        activation_dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.embed_dim = embed_dim
        self.attention_heads = num_heads

        # PyG-style ``batch_first=True`` so we operate on [bsz, N, embed_dim].
        self.self_attn = nn.MultiheadAttention(
            embed_dim, num_heads, dropout=attention_dropout, batch_first=True
        )
        self.self_attn_layer_norm = nn.LayerNorm(embed_dim)
        self.fc1 = nn.Linear(embed_dim, ffn_embed_dim)
        self.fc2 = nn.Linear(ffn_embed_dim, embed_dim)
        self.final_layer_norm = nn.LayerNorm(embed_dim)
        self.dropout = nn.Dropout(dropout)
        self.attn_dropout = nn.Dropout(attention_dropout)
        self.act_dropout = nn.Dropout(activation_dropout)
        self.activation = _ACTIVATION()

    def forward(
        self,
        x: Tensor,
        attn_bias: Tensor,
        key_padding_mask: Tensor | None = None,
    ) -> tuple[Tensor, Tensor]:
        """Return ``(x, attn_weights)``.

        ``attn_bias`` is the additive pair bias ``[bsz*num_heads, N, N]``
        added to the attention logits.  ``attn_weights`` are the per-head
        post-softmax attention probabilities ``[bsz*num_heads, N, N]``.
        """

        x_norm = self.self_attn_layer_norm(x)
        attn_output, attn_weights = self.self_attn(
            x_norm,
            x_norm,
            x_norm,
            attn_mask=attn_bias,
            key_padding_mask=key_padding_mask,
            need_weights=True,
            average_attn_weights=False,
        )
        attn_output = self.attn_dropout(attn_output)
        x = x + attn_output

        x_norm = self.final_layer_norm(x)
        ffn = self.fc2(self.act_dropout(self.activation(self.fc1(x_norm))))
        x = x + self.dropout(ffn)
        return x, attn_weights


class TransformerEncoderWithPair(nn.Module):
    """Stack of pre-LN encoder layers threading an additive pair bias.

    Each layer receives the running pair bias (initial RBF bias plus the
    accumulated attention weights from previous layers) and returns its own
    attention weights, which are accumulated into the pair representation.
    """

    def __init__(
        self,
        embed_dim: int,
        ffn_embed_dim: int,
        layers: int,
        num_heads: int,
        dropout: float = 0.1,
        attention_dropout: float = 0.1,
        activation_dropout: float = 0.0,
        post_ln: bool = False,
    ) -> None:
        super().__init__()
        self.embed_dim = embed_dim
        self.attention_heads = num_heads
        # Pre-LN stacks apply a final LayerNorm after the last layer; post-LN
        # stacks already normalise inside each layer, so no final norm.
        self.final_layer_norm = nn.LayerNorm(embed_dim) if not post_ln else None
        self.layers = nn.ModuleList(
            [
                TransformerEncoderLayer(
                    embed_dim=embed_dim,
                    ffn_embed_dim=ffn_embed_dim,
                    num_heads=num_heads,
                    dropout=dropout,
                    attention_dropout=attention_dropout,
                    activation_dropout=activation_dropout,
                )
                for _ in range(layers)
            ]
        )

    def forward(self, x: Tensor, pair_bias: Tensor) -> tuple[Tensor, Tensor]:
        """Return ``(x, pair_re)``.

        ``pair_bias`` is ``[bsz*num_heads, N, N]``; ``pair_re`` accumulates
        the per-layer attention weights into the updated pair representation.
        """

        bsz, seq_len, _ = x.shape
        num_heads = self.attention_heads
        # Accumulator starts as a zero bias in the same layout as pair_bias.
        pair_re = torch.zeros_like(pair_bias)
        for layer in self.layers:
            x, attn = layer(x, pair_bias + pair_re)
            # attn is [bsz*num_heads, N, N]; fold it back into the running
            # pair representation (per-head view documents the structure).
            pair_re = pair_re + attn.view(bsz, num_heads, seq_len, seq_len).reshape(
                pair_bias.shape
            )
        if self.final_layer_norm is not None:
            x = self.final_layer_norm(x)
        return x, pair_re


class LinearHead(nn.Module):
    """Dropout + Linear supervised head (replaces upstream ClassificationHead).

    The upstream uses a tanh-pooler ``ClassificationHead`` paired with
    cross-entropy on ``[N, 2]`` logits; the lab pairs with ``bce_with_logits``
    on ``[N, 1]`` raw logits, so a plain ``dropout -> Linear`` head suffices.
    """

    def __init__(self, embed_dim: int, out_dim: int, dropout_rate: float = 0.0) -> None:
        super().__init__()
        self.dropout = nn.Dropout(dropout_rate)
        self.out_proj = nn.Linear(embed_dim, out_dim)

    def forward(self, x: Tensor) -> Tensor:
        x = self.dropout(x)
        return self.out_proj(x)


__all__ = [
    "GaussianLayer",
    "LinearHead",
    "NonLinearHead",
    "TransformerEncoderLayer",
    "TransformerEncoderWithPair",
]
