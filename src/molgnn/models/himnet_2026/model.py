"""Architecture-only HimNet molecular property predictor."""

from __future__ import annotations

import math

import torch
from torch import Tensor, nn
from torch.nn import functional as F
from torch_geometric.data import Batch

from ..base import BaseMolecularModel
from ..contracts import validate_batched_molecular_graph
from .layers import (
    ConsensusFingerprintEncoder,
    HierarchicalDirectedEncoder,
    HierarchicalInteractionEncoder,
    TwoViewAttentionFusion,
)


class HimNet(BaseMolecularModel):
    """Hierarchical interaction model with graph and fingerprint fusion.

    The model consumes a precomputed unified hierarchy.  RDKit parsing,
    BRICS decomposition, and fingerprint generation belong to the
    ``himnet_inputs`` transform rather than this forward path.
    """

    required_batch_fields = (
        "himnet_x",
        "himnet_edge_index",
        "himnet_edge_attr",
        "himnet_reverse_edge_index",
        "himnet_node_batch",
        "himnet_node_type",
        "himnet_fp",
    )

    def __init__(
        self,
        atom_dim: int,
        bond_dim: int,
        hidden_dim: int = 512,
        depth: int = 7,
        dropout: float = 0.1,
        interaction_heads: int = 8,
        fusion_heads: int = 4,
        similarity_threshold: float = 0.6,
        num_targets: int = 1,
    ) -> None:
        super().__init__()
        for value, name in (
            (atom_dim, "atom_dim"),
            (bond_dim, "bond_dim"),
            (hidden_dim, "hidden_dim"),
            (depth, "depth"),
            (interaction_heads, "interaction_heads"),
            (fusion_heads, "fusion_heads"),
            (num_targets, "num_targets"),
        ):
            _positive_int(value, name)
        _dropout(dropout)
        if hidden_dim % interaction_heads:
            raise ValueError("hidden_dim must be divisible by interaction_heads")
        if hidden_dim % fusion_heads:
            raise ValueError("hidden_dim must be divisible by fusion_heads")
        if (
            isinstance(similarity_threshold, bool)
            or not isinstance(similarity_threshold, (float, int))
            or not math.isfinite(float(similarity_threshold))
            or not 0.0 <= float(similarity_threshold) <= 1.0
        ):
            raise ValueError("similarity_threshold must be a finite number in [0, 1]")

        self.atom_dim = atom_dim
        self.bond_dim = bond_dim
        self.hidden_dim = hidden_dim
        self.depth = depth
        self.dropout_probability = float(dropout)
        self.interaction_heads = interaction_heads
        self.fusion_heads = fusion_heads
        self.similarity_threshold = float(similarity_threshold)
        self.num_targets = num_targets

        self.directed_encoder = HierarchicalDirectedEncoder(
            atom_dim,
            bond_dim,
            hidden_dim,
            depth,
            dropout,
        )
        self.interaction_encoder = HierarchicalInteractionEncoder(
            atom_dim,
            hidden_dim,
            interaction_heads,
            dropout,
        )
        self.output_layer = nn.Linear(hidden_dim, hidden_dim)
        self.dropout = nn.Dropout(float(dropout))
        self.fingerprint_encoder = ConsensusFingerprintEncoder(
            hidden_dim,
            dropout=dropout,
            similarity_threshold=similarity_threshold,
        )
        self.feature_fusion = TwoViewAttentionFusion(
            hidden_dim,
            num_heads=fusion_heads,
            dropout=dropout,
        )
        self.predictor = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(float(dropout)),
            nn.Linear(hidden_dim, num_targets),
        )

    def forward(self, batch: Batch) -> Tensor:
        """Return raw regression values or binary-classification logits."""

        (
            x,
            edge_index,
            edge_attr,
            reverse_edge_index,
            node_batch,
            _node_type,
            fp_features,
            num_graphs,
        ) = self._batch_tensors(batch)
        directed_embedding = self.directed_encoder(
            x,
            edge_index,
            edge_attr,
            reverse_edge_index,
            node_batch,
        )
        interaction_embedding = self.interaction_encoder(x, node_batch)
        if directed_embedding.shape != (num_graphs, self.hidden_dim):
            raise RuntimeError("directed HimNet encoder returned an invalid graph shape")
        if interaction_embedding.shape != (num_graphs, self.hidden_dim):
            raise RuntimeError("interaction HimNet encoder returned an invalid graph shape")

        graph_embedding = self.dropout(
            F.relu(self.output_layer(directed_embedding + interaction_embedding))
        )
        fingerprint_embedding = self.fingerprint_encoder(fp_features)
        fused_embedding = self.feature_fusion(
            torch.stack((graph_embedding, fingerprint_embedding), dim=1)
        )
        return self.predictor(fused_embedding)

    def _batch_tensors(
        self, batch: Batch
    ) -> tuple[Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, int]:
        names = self.required_batch_fields
        values = tuple(getattr(batch, name, None) for name in names)
        if not all(isinstance(value, Tensor) for value in values):
            raise ValueError(f"batch must provide {', '.join(names)} tensors")
        (
            x,
            edge_index,
            edge_attr,
            reverse_edge_index,
            node_batch,
            node_type,
            fp_features,
        ) = values
        assert isinstance(x, Tensor)
        assert isinstance(edge_index, Tensor)
        assert isinstance(edge_attr, Tensor)
        assert isinstance(reverse_edge_index, Tensor)
        assert isinstance(node_batch, Tensor)
        assert isinstance(node_type, Tensor)
        assert isinstance(fp_features, Tensor)

        if x.ndim != 2 or x.shape[0] < 1 or x.shape[1] != self.atom_dim:
            raise ValueError(f"batch.himnet_x must have shape [N, {self.atom_dim}] with N >= 1")
        if x.dtype != torch.float32 or not torch.isfinite(x).all():
            raise ValueError("batch.himnet_x must contain finite torch.float32 values")
        if (
            edge_index.ndim != 2
            or edge_index.shape[0] != 2
            or edge_index.dtype != torch.long
        ):
            raise ValueError(
                "batch.himnet_edge_index must have shape [2, E] and dtype torch.long"
            )
        edge_count = edge_index.shape[1]
        if edge_attr.shape != (edge_count, self.bond_dim) or edge_attr.dtype != torch.float32:
            raise ValueError(
                f"batch.himnet_edge_attr must have shape [E, {self.bond_dim}] "
                "and dtype torch.float32"
            )
        if not torch.isfinite(edge_attr).all():
            raise ValueError("batch.himnet_edge_attr must contain only finite values")
        if (
            reverse_edge_index.shape != (edge_count,)
            or reverse_edge_index.dtype != torch.long
        ):
            raise ValueError("batch.himnet_reverse_edge_index must have shape [E]")
        if node_batch.shape != (x.shape[0],) or node_batch.dtype != torch.long:
            raise ValueError("batch.himnet_node_batch must have shape [N] and dtype torch.long")
        if node_type.shape != (x.shape[0],) or node_type.dtype != torch.long:
            raise ValueError("batch.himnet_node_type must have shape [N] and dtype torch.long")
        if any(value.device != x.device for value in values):
            raise ValueError("all HimNet batch tensors must be on the same device")

        num_graphs = validate_batched_molecular_graph(
            edge_index,
            node_batch,
            num_nodes=x.shape[0],
            device=x.device,
            edge_field="himnet_edge_index",
            forbid_self_loops=True,
        )
        if edge_count and (
            reverse_edge_index.min() < 0 or reverse_edge_index.max() >= edge_count
        ):
            raise ValueError("batch.himnet_reverse_edge_index contains an invalid edge index")
        if not torch.equal(
            reverse_edge_index[reverse_edge_index],
            torch.arange(edge_count, device=x.device),
        ):
            raise ValueError("batch.himnet_reverse_edge_index must be an involution")
        if edge_count and not torch.equal(
            edge_index[:, reverse_edge_index], edge_index.flip(0)
        ):
            raise ValueError(
                "batch.himnet_reverse_edge_index must map each edge to its reverse"
            )

        self._validate_node_layout(node_batch, node_type, edge_index, num_graphs)
        expected_fp_shape = (num_graphs, ConsensusFingerprintEncoder.fingerprint_dim)
        if fp_features.shape != expected_fp_shape or fp_features.dtype != torch.float32:
            raise ValueError(
                "batch.himnet_fp must have shape "
                f"[{num_graphs}, {ConsensusFingerprintEncoder.fingerprint_dim}] "
                "and dtype torch.float32"
            )
        if not torch.isfinite(fp_features).all():
            raise ValueError("batch.himnet_fp must contain only finite values")
        return (
            x,
            edge_index,
            edge_attr,
            reverse_edge_index,
            node_batch,
            node_type,
            fp_features,
            num_graphs,
        )

    @staticmethod
    def _validate_node_layout(
        node_batch: Tensor,
        node_type: Tensor,
        edge_index: Tensor,
        num_graphs: int,
    ) -> None:
        """Require the hierarchy order and relation types emitted by the transform."""

        if node_type.min() < 0 or node_type.max() > 2:
            raise ValueError("batch.himnet_node_type values must be atom=0, motif=1, or global=2")
        counts = torch.bincount(node_batch, minlength=num_graphs)
        expected_batch = torch.repeat_interleave(
            torch.arange(num_graphs, device=node_batch.device), counts
        )
        if not torch.equal(node_batch, expected_batch):
            raise ValueError("batch.himnet_node_batch must form contiguous graph segments")

        motif_present = torch.zeros(num_graphs, dtype=torch.bool, device=node_batch.device)
        for graph_index, types in enumerate(torch.split(node_type, counts.tolist())):
            atom_count = int((types == 0).sum().item())
            motif_count = int((types == 1).sum().item())
            if atom_count < 1:
                raise ValueError("each HimNet graph must contain at least one atom node")
            expected_types = torch.cat(
                (
                    torch.zeros(atom_count, dtype=torch.long, device=types.device),
                    torch.ones(motif_count, dtype=torch.long, device=types.device),
                    torch.full((1,), 2, dtype=torch.long, device=types.device),
                )
            )
            if not torch.equal(types, expected_types):
                raise ValueError(
                    "each HimNet graph must order atom nodes, motif nodes, then one global node"
                )
            motif_present[graph_index] = motif_count > 0

        if edge_index.shape[1] == 0:
            return
        source, target = edge_index
        source_type = node_type[source]
        target_type = node_type[target]
        atom_atom = (source_type == 0) & (target_type == 0)
        motif_motif = (source_type == 1) & (target_type == 1)
        atom_motif = (source_type == 0) & (target_type == 1)
        motif_atom = (source_type == 1) & (target_type == 0)
        motif_global = (source_type == 1) & (target_type == 2)
        global_motif = (source_type == 2) & (target_type == 1)
        atom_global = (source_type == 0) & (target_type == 2)
        global_atom = (source_type == 2) & (target_type == 0)
        fallback_global_relation = (atom_global | global_atom) & ~motif_present[node_batch[source]]
        allowed = (
            atom_atom
            | motif_motif
            | atom_motif
            | motif_atom
            | motif_global
            | global_motif
            | fallback_global_relation
        )
        if not bool(allowed.all()):
            raise ValueError("batch.himnet_edge_index contains an invalid hierarchy relation")


def _positive_int(value: int, field: str) -> None:
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise ValueError(f"{field} must be a positive integer")


def _dropout(value: float) -> None:
    if isinstance(value, bool) or not isinstance(value, (float, int)) or not 0 <= value < 1:
        raise ValueError("dropout must be in [0, 1)")


__all__ = ["HimNet"]
