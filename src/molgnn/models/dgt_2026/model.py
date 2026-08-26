"""DGT (Dual Graph Transformer) model: embedder, readout and top-level class.

Liu et al., *Enhancing molecular property prediction of transformer models
with dual graph representation*, Nature Communications 2026.
"""

from __future__ import annotations

import torch
from torch import Tensor, nn
from torch_geometric.data import Batch
from torch_geometric.utils import add_self_loops, scatter, to_dense_adj, to_dense_batch

from ..base import BaseMolecularModel
from .layers import DGTLayer, activation


class DGTEmbedder(nn.Module):
    """Project raw features, fuse atom/bond states, and build dense biases.

    The dense structural biases (SPDE embeddings + RWSE projections) are
    computed here once per forward pass and stored on the batch, where every
    ``DGTLayer`` reads them.
    """

    def __init__(
        self,
        atom_dim: int,
        bond_dim: int,
        dim_h: int,
        *,
        spd_max_length: int = 8,
        rwse_steps: int = 16,
    ) -> None:
        super().__init__()
        self.atom_proj = nn.Linear(atom_dim, dim_h)
        self.bond_proj = nn.Linear(bond_dim, dim_h)
        self.edge_mlp = nn.Sequential(
            nn.Linear(2 * dim_h, dim_h),
            nn.Mish(),
            nn.Linear(dim_h, dim_h),
        )
        self.node_mlp = nn.Sequential(
            nn.Linear(2 * dim_h, dim_h),
            nn.Mish(),
            nn.Linear(dim_h, dim_h),
        )
        # SPD lengths are shifted by 1; index 0 is the self-connection and the
        # non-connected bucket is never used, so the vocabulary is +2 wide.
        self.spde_encoder = nn.Embedding(spd_max_length + 2, dim_h)
        self.e2e_spde_encoder = nn.Embedding(spd_max_length + 2, dim_h)
        self.rwse_encoder = nn.Linear(rwse_steps, dim_h)
        self.e2e_rwse_encoder = nn.Linear(rwse_steps, dim_h)
        self.spd_max_length = spd_max_length
        self.rwse_steps = rwse_steps

    def forward(self, batch):
        # Project raw features to dim_h.
        h_n = self.atom_proj(batch.x)  # [N, dim_h]
        undirected_mask = batch.edge_index[0] < batch.edge_index[1]
        undirected = batch.edge_index[:, undirected_mask]
        h_e = self.bond_proj(batch.edge_attr[undirected_mask])  # [M', dim_h]

        # NodeEdgeEncoder fusion.
        # bond_fused[e] = edge_mlp(cat(bond_feat[e], x[i] + x[j]))
        fused_e = self.edge_mlp(
            torch.cat([h_e, h_n[undirected[0]] + h_n[undirected[1]]], dim=-1)
        )
        # atom_fused[i] = node_mlp(cat(x[i], sum_{j->i} bond_feat[j->i]))
        # Both bond directions exist in the canonical graph, so scattering the
        # full directed bond projections by the target atom is symmetric and
        # permutation-equivariant.
        atom_bond_sum = scatter(
            self.bond_proj(batch.edge_attr),
            batch.edge_index[1],
            dim=0,
            dim_size=batch.x.shape[0],
            reduce="sum",
        )
        fused_n = self.node_mlp(torch.cat([h_n, atom_bond_sum], dim=-1))
        batch.x = fused_n
        batch.e = fused_e

        self._build_attn_biases(batch)
        return batch

    def _dense_pairwise_rwse(
        self,
        flat_rwse: Tensor,
        node_batch: Tensor,
        num_graphs: int,
        max_nodes: int,
    ) -> Tensor:
        """Reshape concatenated per-graph ``[N*N, K]`` blocks to dense form.

        The transform stores, for every graph in row-major ``(i, j)`` order,
        the exact ``P^k[i, j]`` values (official ``rw_landing_all``).  This
        helper places each graph's block into
        ``[B, max_nodes, max_nodes, K]`` without mixing graphs.
        """

        steps = int(flat_rwse.shape[1])
        dense = flat_rwse.new_zeros((num_graphs, max_nodes, max_nodes, steps))
        counts = torch.bincount(node_batch, minlength=num_graphs)
        start = 0
        for graph_index, count in enumerate(counts.tolist()):
            count = int(count)
            block_rows = count * count
            if block_rows:
                dense[graph_index, :count, :count] = flat_rwse[
                    start : start + block_rows
                ].reshape(count, count, steps)
            start += block_rows
        return dense

    def _build_attn_biases(self, batch) -> None:
        num_graphs = int(batch.batch.max()) + 1

        # --- Atom graph ---
        _, mask = to_dense_batch(batch.x, batch.batch)  # [B, N_max]
        num_nodes = int(mask.shape[1])
        # The transform stores SPDE up to length 8 and RWSE up to 16 steps
        # (paper defaults); clamp/truncate to this model's chosen budget.
        spd_lengths = batch.dgt_spd_lengths.clamp(max=self.spd_max_length) + 1
        spd_index, spd_lengths = add_self_loops(
            batch.dgt_spd_index,
            spd_lengths,
            num_nodes=batch.x.shape[0],
            fill_value=0,
        )
        spd_emb = self.spde_encoder(spd_lengths)  # [P', dim_h]
        spd_dense = to_dense_adj(
            spd_index, batch=batch.batch, edge_attr=spd_emb
        )  # [B, N_max, N_max, dim_h]
        # True pairwise RWSE: bias[b, i, j] = Linear(P^k[b, i, j]).
        pairwise_rwse = self._dense_pairwise_rwse(
            batch.dgt_rwse[:, : self.rwse_steps], batch.batch, num_graphs, num_nodes
        )
        rwse_bias = self.rwse_encoder(pairwise_rwse)  # [B, N_max, N_max, dim_h]
        bias = spd_dense + rwse_bias
        attn_mask = mask.unsqueeze(1) * mask.unsqueeze(2)
        batch.edge_attention = bias * attn_mask.unsqueeze(-1)
        batch.edge_values = bias * attn_mask.unsqueeze(-1)
        batch.mask = mask
        batch.attn_mask = attn_mask

        # --- Bond graph ---
        _, e_mask = to_dense_batch(
            batch.e, batch.dgt_e_batch, batch_size=num_graphs
        )
        e_num_nodes = int(e_mask.shape[1])
        e2e_spd_lengths = (
            batch.dgt_e2e_spd_lengths.clamp(max=self.spd_max_length) + 1
        )
        e2e_spd_index, e2e_spd_lengths = add_self_loops(
            batch.dgt_e2e_spd_index,
            e2e_spd_lengths,
            num_nodes=batch.e.shape[0],
            fill_value=0,
        )
        e2e_spd_emb = self.e2e_spde_encoder(e2e_spd_lengths)
        e2e_spd_dense = to_dense_adj(
            e2e_spd_index, batch=batch.dgt_e_batch, edge_attr=e2e_spd_emb
        )
        # True pairwise RWSE on the bond graph as well.
        e2e_pairwise_rwse = self._dense_pairwise_rwse(
            batch.dgt_e2e_rwse[:, : self.rwse_steps],
            batch.dgt_e_batch,
            num_graphs,
            e_num_nodes,
        )
        e2e_rwse_bias = self.e2e_rwse_encoder(
            e2e_pairwise_rwse
        )  # [B, E_max, E_max, dim_h]
        e2e_bias = e2e_spd_dense + e2e_rwse_bias
        e_attn_mask = e_mask.unsqueeze(1) * e_mask.unsqueeze(2)
        batch.e2e_edge_attention = e2e_bias * e_attn_mask.unsqueeze(-1)
        batch.e2e_edge_values = e2e_bias * e_attn_mask.unsqueeze(-1)
        batch.e_mask = e_mask
        batch.e_attn_mask = e_attn_mask


class LineGraphReadout(nn.Module):
    """Dual-stream graph readout (atom pool + bond pool, then combine).

    With ``head_layers=2`` and ``dim_h=128`` the widths are ``128 -> 64 ->
    32`` per stream, concatenated to ``64`` before the final projection.
    Graphs without bonds contribute zeros from the bond stream.
    """

    def __init__(
        self,
        dim_h: int,
        num_targets: int,
        *,
        head_layers: int = 2,
        act: str = "gelu",
        graph_pooling: str = "add",
    ) -> None:
        super().__init__()
        if dim_h % (2 ** head_layers) != 0:
            raise ValueError(
                f"dim_h={dim_h} must be divisible by 2**head_layers={2 ** head_layers}"
            )
        if graph_pooling not in ("add", "sum", "mean"):
            raise ValueError("graph_pooling must be 'add'/'sum' or 'mean'")
        self.head_layers = head_layers
        self.graph_pooling = graph_pooling
        self.activation = activation(act)
        self.atom_fc = nn.ModuleList(
            [
                nn.Linear(dim_h // 2 ** layer, dim_h // 2 ** (layer + 1))
                for layer in range(head_layers)
            ]
        )
        self.bond_fc = nn.ModuleList(
            [
                nn.Linear(dim_h // 2 ** layer, dim_h // 2 ** (layer + 1))
                for layer in range(head_layers)
            ]
        )
        self.out_layer = nn.Linear(
            2 * dim_h // 2 ** head_layers, num_targets
        )

    def _pool(self, x: Tensor, index: Tensor, dim_size: int) -> Tensor:
        if self.graph_pooling == "mean":
            return scatter(x, index, dim=0, dim_size=dim_size, reduce="mean")
        return scatter(x, index, dim=0, dim_size=dim_size, reduce="sum")

    def forward(self, batch) -> Tensor:
        num_graphs = int(batch.batch.max()) + 1

        h_n = batch.x
        for layer in self.atom_fc:
            h_n = self.activation(layer(h_n))
        graph_feature = self._pool(h_n, batch.batch, num_graphs)

        h_e = batch.e
        for layer in self.bond_fc:
            h_e = self.activation(layer(h_e))
        graph_edge_feature = self._pool(h_e, batch.dgt_e_batch, num_graphs)

        combined = torch.cat([graph_feature, graph_edge_feature], dim=1)
        return self.out_layer(combined)


class DGT2026(BaseMolecularModel):
    """Dual Graph Transformer for 2-D molecular property prediction."""

    required_batch_fields = (
        "x",
        "edge_attr",
        "edge_index",
        "dgt_e2e_edge_index",
        "dgt_e2e_node_index",
        "dgt_e_batch",
        "dgt_spd_index",
        "dgt_spd_lengths",
        "dgt_e2e_spd_index",
        "dgt_e2e_spd_lengths",
        "dgt_rwse",
        "dgt_e2e_rwse",
        "batch",
    )

    def __init__(
        self,
        *,
        atom_dim: int = 153,
        bond_dim: int = 14,
        dim_h: int = 128,
        num_heads: int = 16,
        num_layers: int = 4,
        dropout: float = 0.0,
        attn_dropout: float = 0.3,
        batch_norm: bool = True,
        act: str = "gelu",
        graph_pooling: str = "add",
        head_layers: int = 2,
        spd_max_length: int = 8,
        rwse_steps: int = 16,
        num_targets: int = 1,
    ) -> None:
        super().__init__()
        if dim_h % num_heads != 0:
            raise ValueError("dim_h must be divisible by num_heads")
        if num_targets < 1:
            raise ValueError("num_targets must be positive")
        if not batch_norm:
            raise ValueError("DGT requires batch_norm=True in this port")

        self.dim_h = dim_h
        self.num_heads = num_heads
        self.num_layers = num_layers

        self.embedder = DGTEmbedder(
            atom_dim,
            bond_dim,
            dim_h,
            spd_max_length=spd_max_length,
            rwse_steps=rwse_steps,
        )
        self.layers = nn.ModuleList(
            [
                DGTLayer(
                    dim_h,
                    num_heads,
                    act=act,
                    dropout=dropout,
                    attn_dropout=attn_dropout,
                    batch_norm=batch_norm,
                )
                for _ in range(num_layers)
            ]
        )
        self.readout = LineGraphReadout(
            dim_h,
            num_targets,
            head_layers=head_layers,
            act=act,
            graph_pooling=graph_pooling,
        )

    def forward(self, batch: Batch) -> Tensor:
        missing = [
            name
            for name in self.required_batch_fields
            if not isinstance(getattr(batch, name, None), Tensor)
        ]
        if missing:
            raise ValueError(
                f"batch is missing tensor field(s): {', '.join(missing)}"
            )
        if batch.x.ndim != 2:
            raise ValueError("batch.x must have shape [N, atom_dim]")
        if batch.edge_index.shape[0] != 2:
            raise ValueError("batch.edge_index must have shape [2, E]")

        # The embedder writes node/bond embeddings and dense biases onto the
        # batch; work on a clone so repeated calls with the same batch are safe.
        batch = batch.clone()
        batch = self.embedder(batch)
        for layer in self.layers:
            batch = layer(batch)
        return self.readout(batch)


__all__ = ["DGT2026", "DGTEmbedder", "LineGraphReadout"]
