"""LiGhT encoder and the KPGT downstream architecture (PyTorch/PyG port).

Provenance: ``OFFICIAL CODE src/model/light.py`` revision
``47dc1646c70b2138a157de481d24a1ac35d174cd``. Paper/source differences are
resolved toward the official code so checkpoints stay loadable:

- two knowledge nodes (fingerprint ``1`` and descriptor ``2``) instead of a
  single paper knowledge token;
- sparse directed path graph capped at ``path_length=5`` line-nodes, not
  complete attention;
- distance bias from a learned path-length embedding plus position-specific
  path MLPs, computed once from the initial node states;
- only fingerprint edges receive the special virtual-path embedding while
  descriptor edges keep the generic path-length bias (official asymmetry);
- readout concatenates fingerprint-node, descriptor-node, and mean real
  line-node states; predictor uses GELU and returns raw ``[B, T]``.

The model never reads or creates coordinates: pure 2-D topology.
"""

from __future__ import annotations

import torch
from torch import Tensor, nn
from torch_geometric.data import Batch

from ..base import BaseMolecularModel
from .constants import (
    D_EDGE_FEATS,
    D_NODE_FEATS,
    DESCRIPTOR_DIM,
    DESCRIPTOR_NODE_INDICATOR,
    FINGERPRINT_DIM,
    FINGERPRINT_NODE_INDICATOR,
)
from .layers import (
    MLP,
    AtomEmbedding,
    BondEmbedding,
    TripletEmbedding,
    TripletTransformer,
    init_params,
)


class LiGhTEncoder(nn.Module):
    """Official LiGhT backbone: structural biases plus Transformer layers."""

    def __init__(
        self,
        d_g_feats: int = 768,
        d_hpath_ratio: int = 12,
        path_length: int = 5,
        n_mol_layers: int = 12,
        n_heads: int = 12,
        n_ffn_dense_layers: int = 2,
        feat_drop: float = 0.0,
        attn_drop: float = 0.0,
        activation: nn.Module | None = None,
    ) -> None:
        super().__init__()
        if activation is None:
            activation = nn.GELU()
        self.n_mol_layers = n_mol_layers
        self.n_heads = n_heads
        self.path_length = path_length
        self.d_g_feats = d_g_feats
        self.d_trip_path = d_g_feats // d_hpath_ratio

        # Retained for exact OFFICIAL CODE checkpoint compatibility. LiGhT
        # defines this embedding even though masking is applied by its wrapper.
        self.mask_emb = nn.Embedding(1, d_g_feats)
        self.path_len_emb = nn.Embedding(path_length + 1, d_g_feats)
        self.virtual_path_emb = nn.Embedding(1, d_g_feats)
        self.self_loop_emb = nn.Embedding(1, d_g_feats)
        self.dist_attn_layer = nn.Sequential(
            nn.Linear(d_g_feats, d_g_feats),
            activation,
            nn.Linear(d_g_feats, n_heads),
        )
        self.trip_fortrans = nn.ModuleList(
            [
                MLP(d_g_feats, self.d_trip_path, 2, activation)
                for _ in range(self.path_length)
            ]
        )
        self.path_attn_layer = nn.Sequential(
            nn.Linear(self.d_trip_path, self.d_trip_path),
            activation,
            nn.Linear(self.d_trip_path, n_heads),
        )
        self.mol_T_layers = nn.ModuleList(
            [
                TripletTransformer(
                    d_g_feats,
                    d_hpath_ratio,
                    path_length,
                    n_heads,
                    n_ffn_dense_layers,
                    feat_drop,
                    attn_drop,
                    activation,
                )
                for _ in range(n_mol_layers)
            ]
        )

    def featurize_path(
        self, path_index: Tensor, virtual_path: Tensor, self_loop: Tensor
    ) -> Tensor:
        """Distance-bias embeddings with official virtual/self-loop overrides."""

        counts = (path_index >= 0).sum(dim=-1)
        feats = self.path_len_emb(counts)
        # OFFICIAL CODE stores both knowledge-edge markers in a BoolTensor,
        # so fingerprint and descriptor edges both receive this override.
        if bool(virtual_path.any()):
            feats[virtual_path] = self.virtual_path_emb.weight[0]
        if bool(self_loop.any()):
            feats[self_loop] = self.self_loop_emb.weight[0]
        return feats

    def init_path(self, triplet_h: Tensor, path_index: Tensor) -> Tensor:
        """Position-specific mean over at most ``path_length`` path tokens."""

        valid = path_index >= 0
        slots: list[Tensor] = []
        for index in range(self.path_length):
            table = torch.cat(
                [
                    self.trip_fortrans[index](triplet_h),
                    triplet_h.new_zeros((1, self.d_trip_path)),
                ],
                dim=0,
            )
            gathered = table.index_select(0, path_index[:, index].clamp_min(0))
            slots.append(gathered * valid[:, index].unsqueeze(-1).to(triplet_h.dtype))
        stacked = torch.stack(slots, dim=-1)
        counts = valid.sum(dim=-1, keepdim=True).to(triplet_h.dtype)
        return stacked.sum(dim=-1) / counts

    def structural_biases(
        self,
        triplet_h: Tensor,
        path_index: Tensor,
        virtual_path: Tensor,
        self_loop: Tensor,
    ) -> tuple[Tensor, Tensor]:
        """Shared per-head distance/path attention biases for all layers."""

        dist_h = self.featurize_path(path_index, virtual_path, self_loop)
        path_h = self.init_path(triplet_h, path_index)
        return self.dist_attn_layer(dist_h), self.path_attn_layer(path_h)

    def forward(
        self,
        triplet_h: Tensor,
        edge_index: Tensor,
        path_index: Tensor,
        virtual_path: Tensor,
        self_loop: Tensor,
    ) -> Tensor:
        dist_attn, path_attn = self.structural_biases(
            triplet_h, path_index, virtual_path, self_loop
        )
        for layer in self.mol_T_layers:
            triplet_h = layer(edge_index, triplet_h, dist_attn, path_attn)
        return triplet_h


def build_kpgt_predictor(
    input_dim: int,
    num_targets: int,
    num_layers: int,
    dropout: float,
    hidden_dim: int | None = None,
) -> nn.Sequential:
    """Official downstream predictor: GELU MLP returning raw values."""

    if num_layers < 1:
        raise ValueError("predictor_num_layers must be >= 1")
    effective_hidden = input_dim if hidden_dim is None else hidden_dim
    if num_layers == 1:
        return nn.Sequential(nn.Linear(input_dim, num_targets))
    blocks: list[nn.Module] = [
        nn.Linear(input_dim, effective_hidden),
        nn.Dropout(dropout),
        nn.GELU(),
    ]
    for _ in range(num_layers - 2):
        blocks.extend(
            (
                nn.Linear(effective_hidden, effective_hidden),
                nn.Dropout(dropout),
                nn.GELU(),
            )
        )
    blocks.append(nn.Linear(effective_hidden, num_targets))
    return nn.Sequential(*blocks)


class KPGT(BaseMolecularModel):
    """Downstream KPGT: LiGhT encoder plus knowledge-guided representation.

    Scratch initialization is the default; a pretrained backbone is loaded
    only when an explicit ``pretrained_checkpoint`` path is provided.
    """

    required_batch_fields = (
        "kpgt_begin_end",
        "kpgt_bond_attr",
        "kpgt_node_indicator",
        "kpgt_triplet_label",
        "kpgt_attention_edge_index",
        "kpgt_path_index",
        "kpgt_virtual_path",
        "kpgt_self_loop",
        "kpgt_fingerprint",
        "kpgt_descriptor",
        "kpgt_token_count",
    )

    def __init__(
        self,
        *,
        num_targets: int = 1,
        d_node_feats: int = D_NODE_FEATS,
        d_edge_feats: int = D_EDGE_FEATS,
        d_g_feats: int = 768,
        d_fp_feats: int = FINGERPRINT_DIM,
        d_md_feats: int = DESCRIPTOR_DIM,
        d_hpath_ratio: int = 12,
        n_mol_layers: int = 12,
        path_length: int = 5,
        n_heads: int = 12,
        n_ffn_dense_layers: int = 2,
        input_drop: float = 0.0,
        attn_drop: float = 0.1,
        feat_drop: float = 0.1,
        predictor_hidden_dim: int = 256,
        predictor_num_layers: int = 2,
        predictor_dropout: float = 0.0,
        pretrained_checkpoint: str | None = None,
    ) -> None:
        super().__init__()
        _positive_int(num_targets, "num_targets")
        for value, name in (
            (d_node_feats, "d_node_feats"),
            (d_edge_feats, "d_edge_feats"),
            (d_g_feats, "d_g_feats"),
            (d_fp_feats, "d_fp_feats"),
            (d_md_feats, "d_md_feats"),
            (d_hpath_ratio, "d_hpath_ratio"),
            (n_mol_layers, "n_mol_layers"),
            (path_length, "path_length"),
            (n_heads, "n_heads"),
            (n_ffn_dense_layers, "n_ffn_dense_layers"),
            (predictor_hidden_dim, "predictor_hidden_dim"),
            (predictor_num_layers, "predictor_num_layers"),
        ):
            _positive_int(value, name)
        if d_g_feats % n_heads:
            raise ValueError("d_g_feats must be divisible by n_heads")
        for value, name in (
            (input_drop, "input_drop"),
            (attn_drop, "attn_drop"),
            (feat_drop, "feat_drop"),
            (predictor_dropout, "predictor_dropout"),
        ):
            _probability(value, name)

        self.d_g_feats = d_g_feats
        self.path_length = path_length
        self.num_targets = num_targets
        # Fixed official downstream/pretraining contract: exactly one
        # fingerprint node and one descriptor node per graph.
        self.knowledge_nodes_per_graph = 2

        # Attribute names mirror the official module for checkpoint loading.
        self.node_emb = AtomEmbedding(d_node_feats, d_g_feats, input_drop)
        self.edge_emb = BondEmbedding(d_edge_feats, d_g_feats, input_drop)
        self.triplet_emb = TripletEmbedding(d_g_feats, d_fp_feats, d_md_feats, nn.GELU())
        self.mask_emb = nn.Embedding(1, d_g_feats)
        self.model = LiGhTEncoder(
            d_g_feats=d_g_feats,
            d_hpath_ratio=d_hpath_ratio,
            path_length=path_length,
            n_mol_layers=n_mol_layers,
            n_heads=n_heads,
            n_ffn_dense_layers=n_ffn_dense_layers,
            feat_drop=feat_drop,
            attn_drop=attn_drop,
        )
        self.predictor = build_kpgt_predictor(
            d_g_feats * 3,
            num_targets,
            predictor_num_layers,
            predictor_dropout,
            predictor_hidden_dim,
        )
        self.apply(init_params)

        if pretrained_checkpoint is not None:
            from .checkpoint import load_pretrained_backbone

            load_pretrained_backbone(self, pretrained_checkpoint)

    def validated_fields(self, batch: Batch) -> dict[str, Tensor]:
        """Extract and shape-check the KPGT tensor fields of one batch."""

        fields = {name: getattr(batch, name, None) for name in self.required_batch_fields}
        missing = [name for name, value in fields.items() if not isinstance(value, Tensor)]
        if missing:
            raise ValueError(
                f"batch is missing KPGT tensor field(s): {', '.join(missing)}"
            )
        typed = {name: value for name, value in fields.items()}
        indicators: Tensor = typed["kpgt_node_indicator"]
        token_count: Tensor = typed["kpgt_token_count"]
        begin_end: Tensor = typed["kpgt_begin_end"]
        bond_attr: Tensor = typed["kpgt_bond_attr"]
        total_nodes = indicators.shape[0]
        if begin_end.shape != (total_nodes, 2, begin_end.shape[-1]):
            raise ValueError("batch.kpgt_begin_end must align with kpgt_node_indicator")
        if bond_attr.shape[0] != total_nodes:
            raise ValueError("batch.kpgt_bond_attr must align with kpgt_node_indicator")
        block_sizes = token_count + self.knowledge_nodes_per_graph
        expected_nodes = int(block_sizes.sum().item())
        if expected_nodes != total_nodes:
            raise ValueError(
                "batch KPGT line-nodes do not align with kpgt_token_count "
                f"(expected {expected_nodes} nodes, got {total_nodes})"
            )
        return typed

    def node_layout(
        self, token_count: Tensor, total_nodes: int
    ) -> tuple[Tensor, Tensor]:
        """Node-to-graph ids and the real-line-node mask for one batch."""

        batch_size = token_count.shape[0]
        block_sizes = token_count + self.knowledge_nodes_per_graph
        node_graph_ids = torch.repeat_interleave(
            torch.arange(batch_size, device=token_count.device), block_sizes
        )
        starts = torch.cumsum(block_sizes, dim=0) - block_sizes
        positions = torch.arange(total_nodes, device=token_count.device)
        real_mask = (positions - starts[node_graph_ids]) < token_count[node_graph_ids]
        return node_graph_ids, real_mask

    def embed_triplet_states(
        self,
        fields: dict[str, Tensor],
        fp_nodes: Tensor,
        md_nodes: Tensor,
    ) -> Tensor:
        """Run the input embeddings up to the initial line-node states."""

        indicators = fields["kpgt_node_indicator"]
        node_h = self.node_emb(fields["kpgt_begin_end"], indicators)
        edge_h = self.edge_emb(fields["kpgt_bond_attr"], indicators)
        return self.triplet_emb(node_h, edge_h, fp_nodes, md_nodes, indicators)

    def project_knowledge(self, fields: dict[str, Tensor], node_graph_ids: Tensor) -> tuple[Tensor, Tensor]:
        """Project per-graph fingerprints/descriptors onto every line-node."""

        fp_nodes = self.triplet_emb.fp_proj(fields["kpgt_fingerprint"])
        md_nodes = self.triplet_emb.md_proj(fields["kpgt_descriptor"])
        return fp_nodes[node_graph_ids], md_nodes[node_graph_ids]

    def forward(self, batch: Batch) -> Tensor:
        fields = self.validated_fields(batch)
        indicators = fields["kpgt_node_indicator"]
        token_count = fields["kpgt_token_count"]
        total_nodes = indicators.shape[0]
        batch_size = token_count.shape[0]
        node_graph_ids, real_mask = self.node_layout(token_count, total_nodes)

        fp_nodes, md_nodes = self.project_knowledge(fields, node_graph_ids)
        triplet_h = self.embed_triplet_states(fields, fp_nodes, md_nodes)
        states = self.model(
            triplet_h,
            fields["kpgt_attention_edge_index"],
            fields["kpgt_path_index"],
            fields["kpgt_virtual_path"],
            fields["kpgt_self_loop"],
        )

        fingerprint_states = states[indicators == FINGERPRINT_NODE_INDICATOR]
        descriptor_states = states[indicators == DESCRIPTOR_NODE_INDICATOR]
        if fingerprint_states.shape[0] != batch_size or descriptor_states.shape[0] != batch_size:
            raise ValueError(
                "each graph must provide exactly one fingerprint and one descriptor node"
            )
        real_sum = states.new_zeros((batch_size, self.d_g_feats))
        real_sum.index_add_(0, node_graph_ids[real_mask], states[real_mask])
        real_mean = real_sum / token_count.unsqueeze(-1).to(states.dtype)
        graph_features = torch.cat(
            (fingerprint_states, descriptor_states, real_mean), dim=-1
        )
        return self.predictor(graph_features)


def _positive_int(value: object, name: str) -> None:
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise ValueError(f"{name} must be a positive integer")


def _probability(value: object, name: str) -> None:
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not 0 <= value < 1:
        raise ValueError(f"{name} must be a probability in [0, 1)")


__all__ = ["KPGT", "LiGhTEncoder", "build_kpgt_predictor"]
