"""HiMol hierarchical encoder and downstream molecular predictor."""

from __future__ import annotations

import torch
from torch import Tensor, nn
from torch_geometric.data import Batch

from ..base import BaseMolecularModel
from .checkpoint import bundled_checkpoint_path, load_himol_encoder
from .layers import HiMolEncoder


class HiMol(BaseMolecularModel):
    """Paper-profile HMGNN with learned graph-token readout."""

    required_batch_fields = (
        "himol_node_attr",
        "himol_edge_index",
        "himol_edge_attr",
        "himol_batch",
        "himol_atom_node_index",
        "himol_graph_node_index",
        "himol_atom_target",
        "himol_bond_index",
        "himol_bond_target",
        "himol_num_atoms",
        "himol_num_bonds",
    )
    PRETRAINED_VARIANTS = ("none", "zinc250k")

    def __init__(
        self,
        atom_dim: int,
        bond_dim: int,
        num_targets: int,
        num_layer: int = 5,
        emb_dim: int = 512,
        JK: str = "last",
        drop_ratio: float = 0.5,
        pretrained_variant: str = "none",
        pretrained_checkpoint: str | None = None,
    ) -> None:
        super().__init__()
        del atom_dim, bond_dim
        if num_targets < 1:
            raise ValueError("num_targets must be positive")
        if pretrained_variant not in self.PRETRAINED_VARIANTS:
            raise ValueError(
                f"pretrained_variant must be one of {self.PRETRAINED_VARIANTS}"
            )
        if pretrained_variant != "none" and pretrained_checkpoint is not None:
            raise ValueError(
                "pretrained_variant and pretrained_checkpoint are mutually exclusive"
            )
        self.num_targets = int(num_targets)
        self.gnn = HiMolEncoder(num_layer, emb_dim, JK, drop_ratio)
        self.graph_pred_linear = nn.Sequential(
            nn.Linear(self.gnn.output_dim, max(1, self.gnn.output_dim // 2)),
            nn.ELU(),
            nn.Linear(max(1, self.gnn.output_dim // 2), num_targets),
        )
        self.pretrained_metadata: dict[str, object] = {
            "variant": "none",
            "path": None,
            "tensor_count": 0,
            "hierarchy_profile": "paper_bidirectional",
        }
        checkpoint = pretrained_checkpoint
        if pretrained_variant != "none":
            checkpoint = str(bundled_checkpoint_path(pretrained_variant))
        if checkpoint is not None:
            self.pretrained_metadata = load_himol_encoder(self.gnn, checkpoint)
            self.pretrained_metadata["variant"] = pretrained_variant

    def encode_nodes(self, batch: Batch) -> Tensor:
        node_attr, edge_index, edge_attr, _, _ = _runtime_fields(batch)
        return self.gnn(node_attr, edge_index, edge_attr)

    def encode_graph(self, batch: Batch) -> Tensor:
        node_attr, _, _, graph_batch, graph_node_index = _runtime_fields(batch)
        node_representation = self.encode_nodes(batch)
        num_graphs = _validate_graph_indices(
            graph_batch, graph_node_index, num_nodes=node_attr.shape[0]
        )
        if graph_node_index.shape[0] != num_graphs:
            raise ValueError("HiMol requires exactly one graph node per molecule")
        return node_representation[graph_node_index]

    def forward(self, batch: Batch) -> Tensor:
        return self.graph_pred_linear(self.encode_graph(batch))


def _runtime_fields(batch: Batch) -> tuple[Tensor, Tensor, Tensor, Tensor, Tensor]:
    values = tuple(
        getattr(batch, name, None)
        for name in (
            "himol_node_attr",
            "himol_edge_index",
            "himol_edge_attr",
            "himol_batch",
            "himol_graph_node_index",
        )
    )
    if not all(isinstance(value, Tensor) for value in values):
        raise ValueError("batch is missing HiMol hierarchy tensors")
    node_attr, edge_index, edge_attr, graph_batch, graph_node_index = values
    assert isinstance(node_attr, Tensor) and isinstance(edge_index, Tensor)
    assert isinstance(edge_attr, Tensor) and isinstance(graph_batch, Tensor)
    assert isinstance(graph_node_index, Tensor)
    if graph_batch.dtype != node_attr.dtype or graph_batch.shape != (
        node_attr.shape[0],
    ):
        raise ValueError("himol_batch must be a long graph id for every hierarchy node")
    return node_attr, edge_index, edge_attr, graph_batch, graph_node_index


def _validate_graph_indices(
    graph_batch: Tensor, graph_nodes: Tensor, *, num_nodes: int
) -> int:
    if graph_batch.dtype != torch.long or graph_nodes.dtype != torch.long:
        raise ValueError("HiMol batch and graph-node indices must have dtype long")
    if graph_nodes.ndim != 1 or graph_nodes.numel() < 1:
        raise ValueError("himol_graph_node_index must contain one index per graph")
    if int(graph_nodes.min()) < 0 or int(graph_nodes.max()) >= num_nodes:
        raise ValueError("himol_graph_node_index references a missing node")
    num_graphs = int(graph_batch.max().item()) + 1
    expected = torch.arange(num_graphs, dtype=torch.long, device=graph_nodes.device)
    if not (graph_batch[graph_nodes] == expected).all():
        raise ValueError("HiMol graph nodes do not align with himol_batch")
    return num_graphs


__all__ = ["HiMol"]
