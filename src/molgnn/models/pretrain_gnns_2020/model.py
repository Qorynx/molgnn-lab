"""Downstream Pretrain-GNNs predictor: GIN encoder, pooling, linear head."""

from __future__ import annotations

import torch
from torch import Tensor, nn
from torch_geometric.data import Batch

from ..base import BaseMolecularModel
from .layers import JK_MODES, MolecularGNN, jk_output_dim


class PretrainGNNs(BaseMolecularModel):
    """Official ``GNN_graphpred`` contract with a raw multi-task head.

    Scratch initialization is the default. A pretrained encoder loads only
    when ``pretrained_variant`` names one of the pinned official artifacts or
    ``pretrained_checkpoint`` gives an explicit path; both never load the
    auxiliary pretraining heads because the official checkpoints contain the
    pure encoder state dict.
    """

    PRETRAINED_VARIANTS = (
        "none",
        "contextpred",
        "masking",
        "supervised_contextpred",
        "supervised_masking",
    )

    required_batch_fields = (
        "pretrain_gnns_atom_attr",
        "pretrain_gnns_bond_attr",
        "edge_index",
        "batch",
    )

    def __init__(
        self,
        *,
        num_targets: int = 1,
        num_layer: int = 5,
        emb_dim: int = 300,
        JK: str = "last",
        drop_ratio: float = 0.0,
        graph_pooling: str = "mean",
        pretrained_variant: str = "none",
        pretrained_checkpoint: str | None = None,
    ) -> None:
        super().__init__()
        if num_layer < 2:
            raise ValueError("Number of GNN layers must be greater than 1.")
        if emb_dim < 1 or num_targets < 1:
            raise ValueError("emb_dim and num_targets must be positive")
        if drop_ratio < 0 or drop_ratio >= 1:
            raise ValueError("drop_ratio must be in [0, 1)")
        if JK not in JK_MODES:
            raise ValueError(f"JK must be one of {JK_MODES}")
        if graph_pooling not in {"sum", "mean", "max"}:
            raise ValueError("graph_pooling must be sum, mean, or max")
        if pretrained_variant not in self.PRETRAINED_VARIANTS:
            raise ValueError(
                f"pretrained_variant must be one of {self.PRETRAINED_VARIANTS}"
            )
        if pretrained_variant != "none" and pretrained_checkpoint is not None:
            raise ValueError(
                "pretrained_variant and pretrained_checkpoint are mutually exclusive"
            )

        self.num_layer = num_layer
        self.emb_dim = emb_dim
        self.num_tasks = num_targets
        # Attribute names mirror the official GNN_graphpred module.
        self.gnn = MolecularGNN(num_layer, emb_dim, JK=JK, drop_ratio=drop_ratio)
        if graph_pooling == "sum":
            self.pool = self._pool_sum
        elif graph_pooling == "mean":
            self.pool = self._pool_mean
        else:
            self.pool = self._pool_max
        # JK concat stacks num_layer+1 node representations; the head must
        # match the encoder's actual output width (official GNN_graphpred).
        self.graph_pred_linear = nn.Linear(
            jk_output_dim(num_layer, emb_dim, JK), num_targets
        )
        self.pretrained_metadata: dict[str, object] = {
            "variant": "none",
            "sha256": None,
            "loaded_tensors": 0,
            "chirality_adapted": False,
        }

        if pretrained_checkpoint is not None:
            from .checkpoint import load_pretrained_encoder

            self.pretrained_metadata = load_pretrained_encoder(
                self.gnn, checkpoint_path=pretrained_checkpoint
            )
        elif pretrained_variant != "none":
            from .checkpoint import load_pretrained_encoder_for_variant

            self.pretrained_metadata = load_pretrained_encoder_for_variant(
                self.gnn, pretrained_variant
            )

    @staticmethod
    def _pool_sum(node_features: Tensor, batch: Tensor) -> Tensor:
        pooled = node_features.new_zeros((int(batch.max().item()) + 1, node_features.shape[1]))
        return pooled.index_add_(0, batch, node_features)

    @staticmethod
    def _pool_mean(node_features: Tensor, batch: Tensor) -> Tensor:
        pooled = node_features.new_zeros((int(batch.max().item()) + 1, node_features.shape[1]))
        pooled.index_add_(0, batch, node_features)
        counts = (
            torch.bincount(batch, minlength=pooled.shape[0]).unsqueeze(-1).to(node_features.dtype)
        )
        return pooled / counts.clamp_min(1.0)

    @staticmethod
    def _pool_max(node_features: Tensor, batch: Tensor) -> Tensor:
        # global_max_pool semantics: pure per-graph scatter max, preserving
        # negative maxima (no zero clamp).
        num_graphs = int(batch.max().item()) + 1
        pooled = node_features.new_full((num_graphs, node_features.shape[1]), float("-inf"))
        pooled.scatter_reduce_(
            0,
            batch.unsqueeze(-1).expand_as(node_features),
            node_features,
            reduce="amax",
            include_self=True,
        )
        return pooled

    def encode_nodes(self, atom_attr: Tensor, edge_index: Tensor, bond_attr: Tensor) -> Tensor:
        return self.gnn(atom_attr, edge_index, bond_attr)

    def forward(self, batch: Batch) -> Tensor:
        atom_attr = getattr(batch, "pretrain_gnns_atom_attr", None)
        bond_attr = getattr(batch, "pretrain_gnns_bond_attr", None)
        edge_index = getattr(batch, "edge_index", None)
        graph_batch = getattr(batch, "batch", None)
        missing = [
            name
            for name, value in (
                ("pretrain_gnns_atom_attr", atom_attr),
                ("pretrain_gnns_bond_attr", bond_attr),
                ("edge_index", edge_index),
                ("batch", graph_batch),
            )
            if not isinstance(value, Tensor)
        ]
        if missing:
            raise ValueError(f"batch is missing tensor field(s): {', '.join(missing)}")
        node_representation = self.encode_nodes(atom_attr, edge_index, bond_attr)
        pooled = self.pool(node_representation, graph_batch)
        # Raw values/logits; sigmoid belongs to the task adapter.
        return self.graph_pred_linear(pooled)


__all__ = ["PretrainGNNs"]
