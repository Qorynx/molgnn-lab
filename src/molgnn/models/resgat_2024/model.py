"""Architecture-only ResGAT 2024 molecular property predictor."""

from __future__ import annotations

import torch
from torch import Tensor, nn
from torch.nn import functional as F
from torch_geometric.data import Batch
from torch_geometric.nn import GATConv, global_max_pool

from ..base import BaseMolecularModel
from ..contracts import validate_batched_molecular_graph
from .layers import GATEConv, ResGATBlock


class ResGAT(BaseMolecularModel):
    """Residual Graph Attention Network (Nguyen-Vo et al., Memetic Computing 2024).

    Architecture: an input ``GATConv`` projects atom features into a hidden
    representation; ``num_blocks`` block sets (each with multiple ``ResGATBlock``s)
    progressively transform the hidden state; a global max-pool and a 3-layer MLP
    head produce one scalar per task per graph.

    Two shortcut types (per the paper):
      * Within a block set, the residual is the identity (dimensions match).
      * The first block of a block set uses a ``GATConv`` shortcut that handles
        the dimension change between block sets.
    """

    required_batch_fields = ("x", "edge_index", "edge_attr", "batch")

    def __init__(
        self,
        atom_dim: int,
        bond_dim: int,
        num_targets: int,
        hidden_dim: int = 64,
        num_blocks: tuple[int, ...] = (2, 2, 2),
        embed_sizes: tuple[int, ...] | None = None,
        heads: int = 1,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()

        if len(num_blocks) == 0:
            raise ValueError("num_blocks must be a non-empty tuple")
        for value, name in (
            (atom_dim, "atom_dim"),
            (bond_dim, "bond_dim"),
            (num_targets, "num_targets"),
            (hidden_dim, "hidden_dim"),
            (heads, "heads"),
        ):
            _positive_int(value, name)
        for n in num_blocks:
            _positive_int(n, "num_blocks entry")
        if embed_sizes is None:
            embed_sizes = (hidden_dim,) * len(num_blocks)
        if len(embed_sizes) != len(num_blocks):
            raise ValueError("embed_sizes must have the same length as num_blocks")
        for v in embed_sizes:
            _positive_int(v, "embed_sizes entry")
        if not 0 <= dropout < 1:
            raise ValueError("dropout must be in [0, 1)")

        self.atom_dim = atom_dim
        self.bond_dim = bond_dim
        self.num_targets = num_targets
        self.hidden_dim = hidden_dim
        self.num_blocks = tuple(num_blocks)
        self.embed_sizes = tuple(embed_sizes)
        self.heads = heads
        self.dropout_p = float(dropout)

        # Input projection: custom GATEConv (matches upstream; fuses edge features
        # into the projection instead of treating them as an attention modulation).
        self.input_conv = GATEConv(
            atom_dim, hidden_dim, edge_dim=bond_dim, dropout=self.dropout_p
        )
        self.input_dropout = (
            nn.Dropout(self.dropout_p) if dropout > 0 else nn.Identity()
        )

        # Build block sets and their shortcuts.  Shortcut uses PyG GATConv with
        # ``add_self_loops=False`` to match the upstream configuration.
        self.block_sets = nn.ModuleList()
        self.shortcut_gats = nn.ModuleList()
        prev_dim = hidden_dim
        for seg_idx, (num_blk, out_dim) in enumerate(zip(num_blocks, embed_sizes)):
            blocks = nn.ModuleList()
            for blk_idx in range(num_blk):
                in_blk = prev_dim if blk_idx == 0 else out_dim
                blocks.append(
                    ResGATBlock(
                        in_blk,
                        out_dim,
                        edge_dim=bond_dim,
                        heads=heads,
                        dropout=self.dropout_p,
                    )
                )
            self.block_sets.append(blocks)
            if prev_dim != out_dim:
                self.shortcut_gats.append(
                    GATConv(
                        prev_dim,
                        out_dim,
                        heads=heads,
                        edge_dim=bond_dim,
                        add_self_loops=False,
                    )
                )
            else:
                self.shortcut_gats.append(nn.Identity())
            prev_dim = out_dim

        self.relu = nn.ReLU()

        # Final head: global max-pool then a 3-layer MLP.
        final_dim = embed_sizes[-1]
        self.head_dropout = (
            nn.Dropout(self.dropout_p) if dropout > 0 else nn.Identity()
        )
        self.head_fc1 = nn.Linear(final_dim, final_dim)
        self.head_fc2 = nn.Linear(final_dim, final_dim // 2)
        self.head_out = nn.Linear(final_dim // 2, num_targets)

    def forward(self, batch: Batch) -> Tensor:
        """Return raw regression values or binary classification logits."""

        x, edge_index, edge_attr, graph_batch, num_graphs = self._validate_batch(
            batch
        )

        # Initial projection.
        x = self.input_dropout(x)
        x = self.relu(self.input_conv(x, edge_index, edge_attr))

        # Block sets with between-block / within-block residuals.
        for blocks, shortcut in zip(self.block_sets, self.shortcut_gats):
            for blk_idx, block in enumerate(blocks):
                residual = shortcut(x, edge_index, edge_attr) if blk_idx == 0 and not isinstance(shortcut, nn.Identity) else x
                x = self.relu(residual + block(x, edge_index, edge_attr))

        # Global max-pool and head.  The head mirrors the upstream's
        # ``Classification_Module`` exactly: only ``fc1`` has a ReLU; ``fc2`` and
        # the final ``out`` are linear.  No activation on the regression output;
        # the BCE-with-logits loss expects raw logits for classification.
        graph_repr = global_max_pool(x, graph_batch, size=num_graphs)
        h = self.head_dropout(F.relu(self.head_fc1(graph_repr)))
        h = self.head_dropout(self.head_fc2(h))
        return self.head_out(h)

    def _validate_batch(
        self, batch: Batch
    ) -> tuple[Tensor, Tensor, Tensor, Tensor, int]:
        """Validate tensors, fetch them, and return canonical inputs."""

        names = self.required_batch_fields
        values = tuple(getattr(batch, name, None) for name in names)
        if not all(isinstance(value, Tensor) for value in values):
            raise ValueError(f"batch must provide {', '.join(names)} tensors")
        x, edge_index, edge_attr, graph_batch = values
        assert isinstance(x, Tensor)
        assert isinstance(edge_index, Tensor)
        assert isinstance(edge_attr, Tensor)
        assert isinstance(graph_batch, Tensor)

        if x.ndim != 2 or x.shape[0] < 1 or x.shape[1] != self.atom_dim:
            raise ValueError(f"batch.x must have shape [N, {self.atom_dim}] with N >= 1")
        if x.dtype != torch.float32 or not torch.isfinite(x).all():
            raise ValueError("batch.x must contain finite torch.float32 values")
        if (
            edge_index.ndim != 2
            or edge_index.shape[0] != 2
            or edge_index.dtype != torch.long
        ):
            raise ValueError(
                "batch.edge_index must have shape [2, E] and dtype torch.long"
            )
        if edge_attr.shape != (edge_index.shape[1], self.bond_dim):
            raise ValueError(
                f"batch.edge_attr must have shape [E, {self.bond_dim}]"
            )
        if edge_attr.dtype != torch.float32 or not torch.isfinite(edge_attr).all():
            raise ValueError(
                "batch.edge_attr must contain finite torch.float32 values"
            )
        if edge_index.shape[1] and (
            edge_index.min().item() < 0 or edge_index.max().item() >= x.shape[0]
        ):
            raise ValueError("batch.edge_index contains an invalid node index")
        if graph_batch.shape != (x.shape[0],) or graph_batch.dtype != torch.long:
            raise ValueError("batch.batch must have shape [N] and dtype torch.long")
        if any(value.device != x.device for value in values):
            raise ValueError("all ResGAT batch tensors must be on the same device")

        num_graphs = validate_batched_molecular_graph(
            edge_index,
            graph_batch,
            num_nodes=x.shape[0],
            device=x.device,
            edge_field="edge_index",
            forbid_self_loops=False,
        )
        return x, edge_index, edge_attr, graph_batch, num_graphs


def _positive_int(value: int, field: str) -> None:
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise ValueError(f"{field} must be a positive integer")


__all__ = ["ResGAT"]
