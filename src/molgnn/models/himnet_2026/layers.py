"""Local HimNet encoders and attention blocks.

The modules in this file intentionally know nothing about RDKit, dataset
loading, model registration, or the training loop.  They operate only on the
already batched hierarchical tensors supplied by :class:`HimNet`.
"""

from __future__ import annotations

import math

import torch
from torch import Tensor, nn
from torch.nn import functional as F
from torch_geometric.utils import scatter


class SegmentedSelfAttention(nn.Module):
    """Self-attention independently over contiguous graph segments.

    The upstream HimNet implementation performs dense attention within each
    molecule.  This sparse-batch version preserves that operation while
    refusing to let one sample attend to another sample's nodes or edges.
    """

    def __init__(
        self,
        feature_dim: int,
        *,
        num_heads: int = 1,
        dropout: float = 0.0,
        bias: bool = True,
        output_projection: bool = True,
    ) -> None:
        super().__init__()
        _positive_int(feature_dim, "feature_dim")
        _positive_int(num_heads, "num_heads")
        _dropout(dropout)
        if feature_dim % num_heads:
            raise ValueError("feature_dim must be divisible by num_heads")
        if not isinstance(bias, bool):
            raise ValueError("bias must be a boolean")

        self.feature_dim = feature_dim
        self.num_heads = num_heads
        self.head_dim = feature_dim // num_heads
        self.query = nn.Linear(feature_dim, feature_dim, bias=bias)
        self.key = nn.Linear(feature_dim, feature_dim, bias=bias)
        self.value = nn.Linear(feature_dim, feature_dim, bias=bias)
        self.out_proj: nn.Module
        self.out_proj = (
            nn.Linear(feature_dim, feature_dim, bias=bias)
            if output_projection
            else nn.Identity()
        )
        self.dropout = nn.Dropout(float(dropout))

    def forward(self, x: Tensor, group: Tensor) -> Tensor:
        """Apply one shared projection set independently to every segment."""

        if x.ndim != 2 or x.shape[1] != self.feature_dim:
            raise ValueError(f"x must have shape [N, {self.feature_dim}]")
        if group.shape != (x.shape[0],) or group.dtype != torch.long:
            raise ValueError("group must have shape [N] and dtype torch.long")
        if x.shape[0] == 0:
            return x
        if group.device != x.device:
            raise ValueError("x and group must be on the same device")
        if group.min() < 0:
            raise ValueError("group must contain non-negative indices")

        num_groups = int(group.max().item()) + 1
        counts = torch.bincount(group, minlength=num_groups)
        expected = torch.repeat_interleave(
            torch.arange(num_groups, device=x.device), counts
        )
        if not torch.equal(group, expected):
            raise ValueError("group indices must form contiguous graph segments")

        chunks = torch.split(x, counts.tolist())
        return torch.cat([self._attend(chunk) for chunk in chunks], dim=0)

    def _attend(self, x: Tensor) -> Tensor:
        length = x.shape[0]
        query = self.query(x).view(length, self.num_heads, self.head_dim).transpose(0, 1)
        key = self.key(x).view(length, self.num_heads, self.head_dim).transpose(0, 1)
        value = self.value(x).view(length, self.num_heads, self.head_dim).transpose(0, 1)
        scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(self.head_dim)
        weights = self.dropout(F.softmax(scores, dim=-1))
        context = torch.matmul(weights, value)
        context = context.transpose(0, 1).contiguous().view(length, self.feature_dim)
        return self.out_proj(context)


class HierarchicalDirectedEncoder(nn.Module):
    """HimNet's D-MPNN and edge-attention branch over hierarchy edges."""

    def __init__(
        self,
        atom_dim: int,
        bond_dim: int,
        hidden_dim: int,
        depth: int,
        dropout: float,
    ) -> None:
        super().__init__()
        for value, name in (
            (atom_dim, "atom_dim"),
            (bond_dim, "bond_dim"),
            (hidden_dim, "hidden_dim"),
            (depth, "depth"),
        ):
            _positive_int(value, name)
        _dropout(dropout)

        self.atom_dim = atom_dim
        self.bond_dim = bond_dim
        self.hidden_dim = hidden_dim
        self.depth = depth
        self.raw_node_attention = SegmentedSelfAttention(
            atom_dim,
            num_heads=1,
            dropout=0.0,
            bias=False,
            output_projection=False,
        )
        self.raw_node_norm = nn.LayerNorm(atom_dim)

        # The source implementation's ``f_bonds`` concatenates source-node
        # features with a relation feature.  The canonical project stores the
        # two parts separately, so that concatenation happens explicitly here.
        self.W_i = nn.Linear(atom_dim + bond_dim, hidden_dim, bias=False)
        self.W_h = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.W_o = nn.Linear(atom_dim + hidden_dim, hidden_dim)
        self.edge_attention = SegmentedSelfAttention(
            hidden_dim,
            num_heads=1,
            dropout=0.0,
            bias=False,
            output_projection=False,
        )
        # The legacy source accidentally reassigns W_a, making the final
        # linear layer shared by edge attention and node readout.  Preserve
        # that executable parameter-sharing invariant deliberately.
        self.W_a = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.W_b = nn.Linear(hidden_dim, hidden_dim)
        self.W_alpha = nn.Linear(2 * hidden_dim, 1)
        self.dropout = nn.Dropout(float(dropout))

    def forward(
        self,
        x: Tensor,
        edge_index: Tensor,
        edge_attr: Tensor,
        reverse_edge_index: Tensor,
        node_batch: Tensor,
    ) -> Tensor:
        """Return one directed-branch representation per molecule."""

        source, target = edge_index
        enhanced_x = self.raw_node_norm(x + self.raw_node_attention(x, node_batch))
        edge_initial = self.W_i(torch.cat((x[source], edge_attr), dim=-1))
        message = F.relu(edge_initial)
        edge_batch = node_batch[source]

        for _ in range(1, self.depth):
            incoming = scatter(message, target, dim=0, dim_size=x.shape[0], reduce="sum")
            dmpnn_message = self.W_h(incoming[source] - message[reverse_edge_index])
            attended_message = self.W_a(self.edge_attention(message, edge_batch))
            gate = torch.sigmoid(
                self.W_alpha(torch.cat((dmpnn_message, attended_message), dim=-1))
            )
            message = self.dropout(
                F.relu(
                    edge_initial
                    + gate * dmpnn_message
                    + (1.0 - gate) * attended_message
                )
            )

        incoming = scatter(message, target, dim=0, dim_size=x.shape[0], reduce="sum")
        node_hidden = self.dropout(
            F.relu(self.W_o(torch.cat((enhanced_x, incoming), dim=-1)))
        )
        return self._self_attentive_readout(node_hidden, node_batch)

    def _self_attentive_readout(self, node_hidden: Tensor, node_batch: Tensor) -> Tensor:
        """Apply source-style dense node readout independently per graph."""

        num_graphs = int(node_batch.max().item()) + 1
        counts = torch.bincount(node_batch, minlength=num_graphs)
        embeddings: list[Tensor] = []
        for nodes in torch.split(node_hidden, counts.tolist()):
            attention = F.softmax(torch.matmul(self.W_a(nodes), nodes.transpose(0, 1)), dim=1)
            attended = torch.matmul(attention, nodes)
            attended = self.dropout(F.relu(self.W_b(attended)))
            embeddings.append((nodes + attended).mean(dim=0))
        return torch.stack(embeddings, dim=0)


class HierarchicalInteractionEncoder(nn.Module):
    """Shared multi-head attention over atom, motif, and global hierarchy nodes."""

    def __init__(
        self,
        atom_dim: int,
        hidden_dim: int,
        num_heads: int,
        dropout: float,
    ) -> None:
        super().__init__()
        _positive_int(atom_dim, "atom_dim")
        _positive_int(hidden_dim, "hidden_dim")
        _positive_int(num_heads, "num_heads")
        _dropout(dropout)
        if hidden_dim % num_heads:
            raise ValueError("hidden_dim must be divisible by num_heads")

        self.atom_projection = nn.Linear(atom_dim, hidden_dim)
        self.cross_attention = SegmentedSelfAttention(
            hidden_dim,
            num_heads=num_heads,
            dropout=dropout,
            bias=True,
            output_projection=True,
        )
        self.layer_norm = nn.LayerNorm(hidden_dim)

    def forward(self, x: Tensor, node_batch: Tensor) -> Tensor:
        projected = self.atom_projection(x)
        enhanced = self.layer_norm(projected + self.cross_attention(projected, node_batch))
        num_graphs = int(node_batch.max().item()) + 1
        return scatter(enhanced, node_batch, dim=0, dim_size=num_graphs, reduce="mean")


class ConsensusFingerprintEncoder(nn.Module):
    """Five-view fingerprint consensus branch from the HimNet source."""

    atom_pairs_dim = 2048
    maccs_dim = 167
    morgan_bits_dim = 2048
    morgan_counts_dim = 2048
    pharmacophore_dim = 27
    fingerprint_dim = (
        atom_pairs_dim
        + maccs_dim
        + morgan_bits_dim
        + morgan_counts_dim
        + pharmacophore_dim
    )
    _view_names = (
        "atom_pairs",
        "maccs",
        "morgan_bits",
        "morgan_counts",
        "pharmacophore",
    )

    def __init__(
        self,
        hidden_dim: int,
        *,
        dropout: float,
        similarity_threshold: float,
    ) -> None:
        super().__init__()
        _positive_int(hidden_dim, "hidden_dim")
        _dropout(dropout)
        if (
            isinstance(similarity_threshold, bool)
            or not isinstance(similarity_threshold, (float, int))
            or not math.isfinite(float(similarity_threshold))
            or not 0.0 <= float(similarity_threshold) <= 1.0
        ):
            raise ValueError("similarity_threshold must be a finite number in [0, 1]")

        self.hidden_dim = hidden_dim
        self.similarity_threshold = float(similarity_threshold)
        self.fp_encoders = nn.ModuleDict(
            {
                "atom_pairs": _fingerprint_encoder(self.atom_pairs_dim, 512, hidden_dim, dropout),
                "maccs": _fingerprint_encoder(self.maccs_dim, 256, hidden_dim, dropout),
                "morgan_bits": _fingerprint_encoder(
                    self.morgan_bits_dim, 512, hidden_dim, dropout
                ),
                "morgan_counts": _fingerprint_encoder(
                    self.morgan_counts_dim, 512, hidden_dim, dropout
                ),
                "pharmacophore": _fingerprint_encoder(
                    self.pharmacophore_dim, 128, hidden_dim, dropout
                ),
            }
        )
        self.enhancement_layer = nn.Sequential(nn.Linear(hidden_dim, hidden_dim), nn.Sigmoid())
        self.weight_generator = nn.Linear(5 * hidden_dim, 5)
        self.fusion_layer = nn.Linear(2 * hidden_dim, hidden_dim)
        self.register_buffer(
            "_pair_indices",
            torch.tensor(
                ((0, 1), (0, 2), (0, 3), (0, 4), (1, 2), (1, 3), (1, 4), (2, 3), (2, 4), (3, 4)),
                dtype=torch.long,
            ),
            persistent=False,
        )

    def forward(self, fp_features: Tensor) -> Tensor:
        """Encode five fingerprint views and fuse their hard-threshold consensus."""

        if fp_features.ndim != 2 or fp_features.shape[1] != self.fingerprint_dim:
            raise ValueError(
                f"fp_features must have shape [B, {self.fingerprint_dim}]"
            )
        if fp_features.dtype != torch.float32:
            raise ValueError("fp_features must have dtype torch.float32")

        views = self._split_views(fp_features)
        encoded = torch.stack(
            [self.fp_encoders[name](view) for name, view in zip(self._view_names, views, strict=True)],
            dim=1,
        )
        normalized = F.normalize(encoded, dim=-1)
        first = self._pair_indices[:, 0]
        second = self._pair_indices[:, 1]
        first_normalized = normalized[:, first]
        second_normalized = normalized[:, second]
        pair_similarity = (first_normalized * second_normalized).sum(dim=-1)
        valid_pair = pair_similarity > self.similarity_threshold

        pair_feature = (encoded[:, first] + encoded[:, second]) / 2.0
        common_mask = (
            first_normalized * second_normalized > self.similarity_threshold
        ).to(encoded.dtype)
        pair_feature = pair_feature * common_mask

        # ``softmax`` is applied only across retained pairs.  The sentinel is
        # irrelevant for rows with no pair because those rows use the exact
        # source fallback (the unweighted mean of all five views).
        masked_similarity = pair_similarity.masked_fill(~valid_pair, -1.0e9)
        pair_weight = F.softmax(masked_similarity, dim=1)
        consensus = (pair_feature * pair_weight.unsqueeze(-1)).sum(dim=1)
        fallback = encoded.mean(dim=1)
        common = torch.where(valid_pair.any(dim=1, keepdim=True), consensus, fallback)

        view_weight = F.softmax(self.weight_generator(encoded.flatten(start_dim=1)), dim=1)
        weighted_views = (encoded * view_weight.unsqueeze(-1)).sum(dim=1)
        enhanced_common = common * self.enhancement_layer(common)
        return self.fusion_layer(torch.cat((weighted_views, enhanced_common), dim=-1))

    def _split_views(self, fp_features: Tensor) -> tuple[Tensor, Tensor, Tensor, Tensor, Tensor]:
        start = 0
        result: list[Tensor] = []
        for width in (
            self.atom_pairs_dim,
            self.maccs_dim,
            self.morgan_bits_dim,
            self.morgan_counts_dim,
            self.pharmacophore_dim,
        ):
            result.append(fp_features[:, start : start + width])
            start += width
        return tuple(result)  # type: ignore[return-value]


class TwoViewAttentionFusion(nn.Module):
    """Source-style four-head self-attention over graph and fingerprint tokens."""

    def __init__(self, hidden_dim: int, *, num_heads: int, dropout: float) -> None:
        super().__init__()
        _positive_int(hidden_dim, "hidden_dim")
        _positive_int(num_heads, "num_heads")
        _dropout(dropout)
        if hidden_dim % num_heads:
            raise ValueError("hidden_dim must be divisible by num_heads")

        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.head_dim = hidden_dim // num_heads
        self.query = nn.Linear(hidden_dim, hidden_dim)
        self.key = nn.Linear(hidden_dim, hidden_dim)
        self.value = nn.Linear(hidden_dim, hidden_dim)
        self.out_proj = nn.Linear(hidden_dim, hidden_dim)
        self.dropout = nn.Dropout(float(dropout))

    def forward(self, tokens: Tensor) -> Tensor:
        """Fuse exactly two ``[graph, fingerprint]`` tokens per sample."""

        if tokens.ndim != 3 or tokens.shape[1:] != (2, self.hidden_dim):
            raise ValueError(f"tokens must have shape [B, 2, {self.hidden_dim}]")
        batch_size = tokens.shape[0]
        query = self.query(tokens).view(batch_size, 2, self.num_heads, self.head_dim)
        key = self.key(tokens).view(batch_size, 2, self.num_heads, self.head_dim)
        value = self.value(tokens).view(batch_size, 2, self.num_heads, self.head_dim)
        query = query.transpose(1, 2)
        key = key.transpose(1, 2)
        value = value.transpose(1, 2)
        scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(self.head_dim)
        weights = self.dropout(F.softmax(scores, dim=-1))
        context = torch.matmul(weights, value)
        context = context.transpose(1, 2).contiguous().view(batch_size, 2, self.hidden_dim)
        return self.out_proj(context).mean(dim=1)


def _fingerprint_encoder(
    input_dim: int, intermediate_dim: int, hidden_dim: int, dropout: float
) -> nn.Sequential:
    return nn.Sequential(
        nn.Linear(input_dim, intermediate_dim),
        nn.ReLU(),
        nn.Dropout(dropout),
        nn.Linear(intermediate_dim, hidden_dim),
    )


def _positive_int(value: int, field: str) -> None:
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise ValueError(f"{field} must be a positive integer")


def _dropout(value: float) -> None:
    if isinstance(value, bool) or not isinstance(value, (float, int)) or not 0 <= value < 1:
        raise ValueError("dropout must be in [0, 1)")


__all__ = [
    "ConsensusFingerprintEncoder",
    "HierarchicalDirectedEncoder",
    "HierarchicalInteractionEncoder",
    "SegmentedSelfAttention",
    "TwoViewAttentionFusion",
]
