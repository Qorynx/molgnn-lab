"""Project-facing TorchMD-ET model from pre-training via denoising."""

from __future__ import annotations

from pathlib import Path

import torch
from torch import Tensor, nn
from torch_geometric.data import Batch
from torch_geometric.utils import scatter

from ..base import BaseMolecularModel
from ..contracts import validate_batched_molecular_graph
from .constants import (
    PVD_CUTOFF_LOWER,
    PVD_CUTOFF_UPPER,
    PVD_HIDDEN_CHANNELS,
    PVD_MAX_ATOMIC_NUMBER,
    PVD_MAX_NUM_NEIGHBORS,
    PVD_NUM_HEADS,
    PVD_NUM_LAYERS,
    PVD_NUM_RBF,
)
from .layers import (
    EquivariantLayerNorm,
    EquivariantMultiHeadAttention,
    EquivariantScalarHead,
    EquivariantVectorHead,
    NeighborEmbedding,
)
from .radial import ExpNormalSmearing


class TorchMDETEncoder(nn.Module):
    """Scalar/vector equivariant Transformer matching the released checkpoint."""

    def __init__(
        self,
        *,
        hidden_channels: int = PVD_HIDDEN_CHANNELS,
        num_layers: int = PVD_NUM_LAYERS,
        num_rbf: int = PVD_NUM_RBF,
        num_heads: int = PVD_NUM_HEADS,
        cutoff_lower: float = PVD_CUTOFF_LOWER,
        cutoff_upper: float = PVD_CUTOFF_UPPER,
        max_atomic_number: int = PVD_MAX_ATOMIC_NUMBER,
        max_num_neighbors: int = PVD_MAX_NUM_NEIGHBORS,
        trainable_rbf: bool = False,
        neighbor_embedding: bool = True,
        distance_influence: str = "both",
        vector_layer_norm: str | None = "whitened",
    ) -> None:
        super().__init__()
        for value, name in (
            (hidden_channels, "hidden_channels"),
            (num_layers, "num_layers"),
            (num_rbf, "num_rbf"),
            (num_heads, "num_heads"),
            (max_atomic_number, "max_atomic_number"),
            (max_num_neighbors, "max_num_neighbors"),
        ):
            if isinstance(value, bool) or not isinstance(value, int) or value < 1:
                raise ValueError(f"{name} must be a positive integer")
        if hidden_channels % num_heads:
            raise ValueError("hidden_channels must be divisible by num_heads")
        if cutoff_lower < 0 or cutoff_upper <= cutoff_lower:
            raise ValueError("cutoffs must satisfy 0 <= lower < upper")
        if distance_influence not in {"keys", "values", "both", "none"}:
            raise ValueError("unsupported distance_influence")
        if vector_layer_norm not in {None, "whitened"}:
            raise ValueError("vector_layer_norm must be None or 'whitened'")

        self.hidden_channels = hidden_channels
        self.num_layers = num_layers
        self.num_rbf = num_rbf
        self.num_heads = num_heads
        self.cutoff_lower = float(cutoff_lower)
        self.cutoff_upper = float(cutoff_upper)
        self.max_z = max_atomic_number + 1
        self.max_num_neighbors = max_num_neighbors
        self.distance_influence = distance_influence
        self.vector_layer_norm = vector_layer_norm

        # Module names intentionally follow the released TorchMD-ET state dict.
        self.embedding = nn.Embedding(self.max_z, hidden_channels)
        self.distance_expansion = ExpNormalSmearing(
            cutoff_lower,
            cutoff_upper,
            num_rbf,
            trainable=trainable_rbf,
        )
        self.neighbor_embedding = (
            NeighborEmbedding(
                hidden_channels,
                num_rbf,
                cutoff_lower,
                cutoff_upper,
                self.max_z,
            )
            if neighbor_embedding
            else None
        )
        self.attention_layers = nn.ModuleList(
            EquivariantMultiHeadAttention(
                hidden_channels,
                num_rbf,
                distance_influence,
                num_heads,
                cutoff_lower,
                cutoff_upper,
            )
            for _ in range(num_layers)
        )
        self.out_norm = nn.LayerNorm(hidden_channels)
        self.out_norm_vec = (
            EquivariantLayerNorm(hidden_channels)
            if vector_layer_norm == "whitened"
            else None
        )
        self.reset_parameters()

    def reset_parameters(self) -> None:
        self.embedding.reset_parameters()
        self.distance_expansion.reset_parameters()
        if self.neighbor_embedding is not None:
            self.neighbor_embedding.reset_parameters()
        for attention in self.attention_layers:
            attention.reset_parameters()
        self.out_norm.reset_parameters()
        if self.out_norm_vec is not None:
            self.out_norm_vec.reset_parameters()

    def forward(
        self,
        atomic_number: Tensor,
        pos: Tensor,
        edge_index: Tensor,
    ) -> tuple[Tensor, Tensor]:
        if atomic_number.ndim != 1 or atomic_number.dtype != torch.long:
            raise ValueError("atomic_number must have shape [N] and dtype long")
        if pos.shape != (atomic_number.shape[0], 3) or not pos.is_floating_point():
            raise ValueError("pos must have shape [N, 3] and floating dtype")
        if edge_index.ndim != 2 or edge_index.shape[0] != 2:
            raise ValueError("pvd_edge_index must have shape [2, E]")
        if edge_index.dtype != torch.long:
            raise ValueError("pvd_edge_index must have dtype long")
        if any(value.device != atomic_number.device for value in (pos, edge_index)):
            raise ValueError("PVD tensors must share one device")
        if atomic_number.numel() and (
            atomic_number.min() < 0 or atomic_number.max() >= self.max_z
        ):
            raise ValueError(f"atomic_number must be in [0, {self.max_z - 1}]")

        x = self.embedding(atomic_number)
        source, target = edge_index
        edge_vec = pos[source] - pos[target]
        distances = torch.linalg.vector_norm(edge_vec, dim=-1)
        directions = torch.zeros_like(edge_vec)
        non_self = source != target
        if bool(non_self.any()):
            nonzero_distance = distances[non_self]
            if bool((nonzero_distance <= 1.0e-8).any()):
                raise ValueError("PVD graph contains coincident distinct atoms")
            directions[non_self] = (
                edge_vec[non_self] / nonzero_distance.unsqueeze(-1)
            )
        edge_attr = self.distance_expansion(distances)
        if self.neighbor_embedding is not None:
            x = self.neighbor_embedding(
                atomic_number, x, edge_index, distances, edge_attr
            )
        vec = x.new_zeros((x.shape[0], 3, self.hidden_channels))
        for attention in self.attention_layers:
            scalar_update, vector_update = attention(
                x, vec, edge_index, distances, edge_attr, directions
            )
            x = x + scalar_update
            vec = vec + vector_update
        x = self.out_norm(x)
        if self.out_norm_vec is not None:
            vec = self.out_norm_vec(vec)
        return x, vec


class PVDTorchMDET(BaseMolecularModel):
    """Graph predictor backed by the released denoising TorchMD-ET encoder."""

    required_batch_fields = (
        "atomic_number",
        "pos",
        "pvd_edge_index",
        "batch",
    )

    def __init__(
        self,
        atom_dim: int,
        bond_dim: int,
        num_targets: int,
        *,
        hidden_channels: int = PVD_HIDDEN_CHANNELS,
        num_layers: int = PVD_NUM_LAYERS,
        num_rbf: int = PVD_NUM_RBF,
        num_heads: int = PVD_NUM_HEADS,
        cutoff_lower: float = PVD_CUTOFF_LOWER,
        cutoff_upper: float = PVD_CUTOFF_UPPER,
        max_atomic_number: int = PVD_MAX_ATOMIC_NUMBER,
        max_num_neighbors: int = PVD_MAX_NUM_NEIGHBORS,
        trainable_rbf: bool = False,
        neighbor_embedding: bool = True,
        distance_influence: str = "both",
        vector_layer_norm: str | None = "whitened",
        readout: str = "sum",
        pretrained_variant: str = "none",
        pretrained_checkpoint: str | None = None,
    ) -> None:
        super().__init__()
        del atom_dim, bond_dim
        if isinstance(num_targets, bool) or not isinstance(num_targets, int) or num_targets < 1:
            raise ValueError("num_targets must be a positive integer")
        if readout not in {"sum", "mean"}:
            raise ValueError("readout must be sum or mean")
        if (
            float(cutoff_lower) != PVD_CUTOFF_LOWER
            or float(cutoff_upper) != PVD_CUTOFF_UPPER
            or max_num_neighbors != PVD_MAX_NUM_NEIGHBORS
        ):
            raise ValueError(
                "cutoff/max_num_neighbors must match the fixed pvd_inputs transform"
            )
        if max_atomic_number != PVD_MAX_ATOMIC_NUMBER:
            raise ValueError(
                "max_atomic_number must match the official max_z=100 profile"
            )
        self.num_targets = num_targets
        self.readout = readout
        self.encoder = TorchMDETEncoder(
            hidden_channels=hidden_channels,
            num_layers=num_layers,
            num_rbf=num_rbf,
            num_heads=num_heads,
            cutoff_lower=cutoff_lower,
            cutoff_upper=cutoff_upper,
            max_atomic_number=max_atomic_number,
            max_num_neighbors=max_num_neighbors,
            trainable_rbf=trainable_rbf,
            neighbor_embedding=neighbor_embedding,
            distance_influence=distance_influence,
            vector_layer_norm=vector_layer_norm,
        )
        self.property_head = EquivariantScalarHead(hidden_channels, num_targets)
        self.noise_head = EquivariantVectorHead(hidden_channels)
        self.initialization = "scratch"
        self.checkpoint_info: dict[str, object] | None = None
        if pretrained_variant != "none" or pretrained_checkpoint is not None:
            from .checkpoint import load_pvd_pretrained

            self.checkpoint_info = load_pvd_pretrained(
                self,
                variant=pretrained_variant,
                checkpoint_path=(
                    Path(pretrained_checkpoint)
                    if pretrained_checkpoint is not None
                    else None
                ),
                include_noise_head=False,
            )
            self.initialization = "pretrained"

    def _validated_inputs(
        self, batch: Batch
    ) -> tuple[Tensor, Tensor, Tensor, Tensor, int]:
        values = tuple(getattr(batch, field, None) for field in self.required_batch_fields)
        if not all(isinstance(value, Tensor) for value in values):
            raise ValueError("batch is missing PVD TorchMD-ET tensors")
        atomic_number, pos, edge_index, graph_batch = values
        assert isinstance(atomic_number, Tensor)
        assert isinstance(pos, Tensor)
        assert isinstance(edge_index, Tensor)
        assert isinstance(graph_batch, Tensor)
        if atomic_number.ndim != 1 or atomic_number.dtype != torch.long:
            raise ValueError("batch.atomic_number must have shape [N] and dtype long")
        if pos.shape != (atomic_number.shape[0], 3) or pos.dtype != torch.float32:
            raise ValueError("batch.pos must have shape [N, 3] and dtype float32")
        if not bool(torch.isfinite(pos).all()):
            raise ValueError("batch.pos must be finite")
        num_graphs = validate_batched_molecular_graph(
            edge_index,
            graph_batch,
            num_nodes=atomic_number.shape[0],
            device=atomic_number.device,
            edge_field="pvd_edge_index",
            forbid_self_loops=False,
        )
        if edge_index.shape[1] < atomic_number.shape[0]:
            raise ValueError("pvd_edge_index must include one self-loop per atom")
        source, target = edge_index
        self_nodes = source[source == target]
        if self_nodes.numel() != atomic_number.shape[0] or not torch.equal(
            torch.sort(self_nodes).values,
            torch.arange(atomic_number.shape[0], device=atomic_number.device),
        ):
            raise ValueError("pvd_edge_index must contain exactly one self-loop per atom")
        return atomic_number, pos, edge_index, graph_batch, num_graphs

    def encode_batch(
        self, batch: Batch
    ) -> tuple[Tensor, Tensor, Tensor, int]:
        atomic_number, pos, edge_index, graph_batch, num_graphs = self._validated_inputs(batch)
        scalar, vector = self.encoder(atomic_number, pos, edge_index)
        return scalar, vector, graph_batch, num_graphs

    def predict_noise(self, batch: Batch) -> Tensor:
        scalar, vector, _, _ = self.encode_batch(batch)
        return self.noise_head(scalar, vector)

    def forward(self, batch: Batch) -> Tensor:
        scalar, vector, graph_batch, num_graphs = self.encode_batch(batch)
        atomwise = self.property_head(scalar, vector)
        return scatter(
            atomwise,
            graph_batch,
            dim=0,
            dim_size=num_graphs,
            reduce=self.readout,
        )


__all__ = ["PVDTorchMDET", "TorchMDETEncoder"]
