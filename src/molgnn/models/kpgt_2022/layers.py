"""PyTorch/PyG port of the official KPGT LiGhT building blocks.

Provenance: ``OFFICIAL CODE src/model/light.py`` at revision
``47dc1646c70b2138a157de481d24a1ac35d174cd`` (DGL implementation). The
module/attribute names intentionally mirror the official classes so that
pretrained checkpoints load without key remapping. Where the paper differs,
the official behavior is kept: pre-LN placement, ``hidden_dim**-0.5``
attention scale, and destination-wise edge softmax over the sparse path
graph (never complete attention).
"""

from __future__ import annotations

import torch
from torch import Tensor, nn


def init_params(module: nn.Module) -> None:
    """Official initialization: normal(0, 0.02) weights, zero biases."""

    if isinstance(module, nn.Linear):
        module.weight.data.normal_(mean=0.0, std=0.02)
        if module.bias is not None:
            module.bias.data.zero_()
    if isinstance(module, nn.Embedding):
        module.weight.data.normal_(mean=0.0, std=0.02)


class MLP(nn.Module):
    """Official MLP: input projection, optional hidden layers, output."""

    def __init__(
        self,
        d_in_feats: int,
        d_out_feats: int,
        n_dense_layers: int,
        activation: nn.Module,
        d_hidden_feats: int | None = None,
    ) -> None:
        super().__init__()
        self.n_dense_layers = n_dense_layers
        self.d_hidden_feats = d_out_feats if d_hidden_feats is None else d_hidden_feats
        self.dense_layer_list = nn.ModuleList()
        self.in_proj = nn.Linear(d_in_feats, self.d_hidden_feats)
        for _ in range(self.n_dense_layers - 2):
            self.dense_layer_list.append(nn.Linear(self.d_hidden_feats, self.d_hidden_feats))
        self.out_proj = nn.Linear(self.d_hidden_feats, d_out_feats)
        self.act = activation

    def forward(self, feats: Tensor) -> Tensor:
        feats = self.act(self.in_proj(feats))
        for index in range(self.n_dense_layers - 2):
            feats = self.act(self.dense_layer_list[index](feats))
        return self.out_proj(feats)


class Residual(nn.Module):
    """Official residual block around the FFN with pre-LN on the branch."""

    def __init__(
        self,
        d_in_feats: int,
        d_out_feats: int,
        n_ffn_dense_layers: int,
        feat_drop: float,
        activation: nn.Module,
    ) -> None:
        super().__init__()
        self.norm = nn.LayerNorm(d_in_feats)
        self.in_proj = nn.Linear(d_in_feats, d_out_feats)
        self.ffn = MLP(
            d_out_feats,
            d_out_feats,
            n_ffn_dense_layers,
            activation,
            d_hidden_feats=d_out_feats * 4,
        )
        self.feat_dropout = nn.Dropout(feat_drop)

    def forward(self, x: Tensor, y: Tensor) -> Tensor:
        x = x + self.feat_dropout(self.in_proj(y))
        branch = self.norm(x)
        branch = self.ffn(branch)
        branch = self.feat_dropout(branch)
        return x + branch


def destination_softmax(scores: Tensor, dst: Tensor) -> Tensor:
    """Softmax over incoming edges per destination node, per head.

    Replicates DGL's ``edge_softmax`` semantics (destination-wise groups)
    with stable numerics. ``scores`` is ``[E, H]`` and ``dst`` is ``[E]``;
    every destination owns at least its self-loop so denominators stay > 0.
    """

    num_destinations = int(dst.max().item()) + 1
    expanded_dst = dst.unsqueeze(-1).expand_as(scores)
    max_per_dst = scores.new_full((num_destinations, scores.shape[1]), float("-inf"))
    max_per_dst = max_per_dst.scatter_reduce(
        0, expanded_dst, scores, reduce="amax", include_self=True
    )
    exponentiated = torch.exp(scores - max_per_dst.index_select(0, dst))
    denominator = scores.new_zeros(num_destinations, scores.shape[1])
    denominator.index_add_(0, dst, exponentiated)
    return exponentiated / denominator.index_select(0, dst)


class TripletTransformer(nn.Module):
    """One pre-LN attention/FFN block over line-nodes with structural bias."""

    def __init__(
        self,
        d_feats: int,
        d_hpath_ratio: int,
        path_length: int,
        n_heads: int,
        n_ffn_dense_layers: int,
        feat_drop: float,
        attn_drop: float,
        activation: nn.Module,
    ) -> None:
        super().__init__()
        self.d_feats = d_feats
        self.d_trip_path = d_feats // d_hpath_ratio
        self.path_length = path_length
        self.n_heads = n_heads
        # OFFICIAL CODE keeps the hidden-dim scale even though it deviates
        # from the standard per-head scaling used by the paper formulation.
        self.scale = d_feats ** (-0.5)

        self.attention_norm = nn.LayerNorm(d_feats)
        self.qkv = nn.Linear(d_feats, d_feats * 3)
        self.node_out_layer = Residual(
            d_feats, d_feats, n_ffn_dense_layers, feat_drop, activation
        )

        self.feat_dropout = nn.Dropout(feat_drop)
        self.attn_dropout = nn.Dropout(attn_drop)
        self.act = activation

    def forward(
        self,
        edge_index: Tensor,
        triplet_h: Tensor,
        dist_attn: Tensor,
        path_attn: Tensor,
    ) -> Tensor:
        src, dst = edge_index[0], edge_index[1]
        normalized = self.attention_norm(triplet_h)
        qkv = (
            self.qkv(normalized)
            .reshape(-1, 3, self.n_heads, self.d_feats // self.n_heads)
            .permute(1, 0, 2, 3)
        )
        query, key, value = qkv[0] * self.scale, qkv[1], qkv[2]

        node_attention = (query[src] * key[dst]).sum(dim=-1)
        attention = node_attention + dist_attn + path_attn
        softmax = self.attn_dropout(destination_softmax(attention, dst))

        messages = value[src].view(-1, self.n_heads, self.d_feats // self.n_heads)
        messages = (messages * softmax.unsqueeze(-1)).view(-1, self.d_feats)
        aggregated = triplet_h.new_zeros(triplet_h.shape)
        aggregated.index_add_(0, dst, messages)
        return self.node_out_layer(triplet_h, aggregated)


class AtomEmbedding(nn.Module):
    """Project both atoms of each line-node; virtual second slot for isolates."""

    def __init__(self, d_atom_feats: int, d_g_feats: int, input_drop: float) -> None:
        super().__init__()
        self.in_proj = nn.Linear(d_atom_feats, d_g_feats)
        self.virtual_atom_emb = nn.Embedding(1, d_g_feats)
        self.input_dropout = nn.Dropout(input_drop)
        self.placeholder_indicator = -1

    def forward(self, begin_end: Tensor, indicators: Tensor) -> Tensor:
        pair_node_h = self.in_proj(begin_end)
        isolated = indicators == self.placeholder_indicator
        if bool(isolated.any()):
            pair_node_h[isolated, 1, :] = self.virtual_atom_emb.weight[0]
        return torch.sum(self.input_dropout(pair_node_h), dim=-2)


class BondEmbedding(nn.Module):
    """Project bond features; learned embedding replaces placeholder rows."""

    def __init__(self, d_bond_feats: int, d_g_feats: int, input_drop: float) -> None:
        super().__init__()
        self.in_proj = nn.Linear(d_bond_feats, d_g_feats)
        # Attribute name keeps the official typo for checkpoint compatibility.
        self.virutal_bond_emb = nn.Embedding(1, d_g_feats)
        self.input_dropout = nn.Dropout(input_drop)
        self.placeholder_indicator = -1

    def forward(self, bond_attr: Tensor, indicators: Tensor) -> Tensor:
        edge_h = self.in_proj(bond_attr)
        placeholder = indicators == self.placeholder_indicator
        if bool(placeholder.any()):
            edge_h[placeholder] = self.virutal_bond_emb.weight[0]
        return self.input_dropout(edge_h)


class TripletEmbedding(nn.Module):
    """Fuse projected atom-pair and bond states; knowledge nodes override."""

    def __init__(
        self,
        d_g_feats: int,
        d_fp_feats: int,
        d_md_feats: int,
        activation: nn.Module,
    ) -> None:
        super().__init__()
        self.in_proj = MLP(d_g_feats * 2, d_g_feats, 2, activation)
        self.fp_proj = MLP(d_fp_feats, d_g_feats, 2, activation)
        self.md_proj = MLP(d_md_feats, d_g_feats, 2, activation)
        self.fp_indicator = 1
        self.md_indicator = 2

    def forward(
        self,
        node_h: Tensor,
        edge_h: Tensor,
        fp_nodes: Tensor,
        md_nodes: Tensor,
        indicators: Tensor,
    ) -> Tensor:
        triplet_h = self.in_proj(torch.cat([node_h, edge_h], dim=-1))
        fingerprint_nodes = indicators == self.fp_indicator
        descriptor_nodes = indicators == self.md_indicator
        if bool(fingerprint_nodes.any()):
            triplet_h[fingerprint_nodes] = fp_nodes[fingerprint_nodes]
        if bool(descriptor_nodes.any()):
            triplet_h[descriptor_nodes] = md_nodes[descriptor_nodes]
        return triplet_h


__all__ = [
    "MLP",
    "AtomEmbedding",
    "BondEmbedding",
    "Residual",
    "TripletEmbedding",
    "TripletTransformer",
    "destination_softmax",
    "init_params",
]
