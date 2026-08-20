"""Project-facing ViSNet graph-level scalar predictor."""

from __future__ import annotations

import torch
from torch import Tensor, nn
from torch.autograd import grad
from torch_geometric.data import Batch
from torch_geometric.utils import scatter

from ..base import BaseMolecularModel
from ..contracts import validate_batched_molecular_graph
from .constants import (
    VISNET_CUTOFF,
    VISNET_DEFAULT_HIDDEN_DIM,
    VISNET_DEFAULT_LMAX,
    VISNET_DEFAULT_NUM_HEADS,
    VISNET_DEFAULT_NUM_LAYERS,
    VISNET_EPS,
    VISNET_MAX_ATOMIC_NUMBER,
    VISNET_MAX_NEIGHBORS,
    VISNET_NUM_RBF,
)
from .geometry import VecLayerNorm, spherical_harmonics
from .layers import Dense, EdgeEmbedding, GatedEquivariantBlock, NeighborEmbedding, ViSMP, build_activation
from .radial import build_radial_basis


class ViSNet(BaseMolecularModel):
    """ViSNet's RGC and vector-scalar message-passing core over 3-D inputs."""

    required_batch_fields = (
        "atomic_number",
        "pos",
        "visnet_edge_index",
        "batch",
    )

    def __init__(
        self,
        num_targets: int,
        hidden_dim: int = VISNET_DEFAULT_HIDDEN_DIM,
        num_layers: int = VISNET_DEFAULT_NUM_LAYERS,
        num_heads: int = VISNET_DEFAULT_NUM_HEADS,
        num_rbf: int = VISNET_NUM_RBF,
        lmax: int = VISNET_DEFAULT_LMAX,
        rbf_type: str = "expnorm",
        trainable_rbf: bool = False,
        activation: str = "silu",
        attn_activation: str = "silu",
        vecnorm_type: str = "none",
        trainable_vecnorm: bool = False,
        output_head: str = "equivariant",
        readout: str = "sum",
        max_atomic_number: int = VISNET_MAX_ATOMIC_NUMBER,
    ) -> None:
        super().__init__()
        for value, field in (
            (num_targets, "num_targets"),
            (hidden_dim, "hidden_dim"),
            (num_layers, "num_layers"),
            (num_heads, "num_heads"),
            (num_rbf, "num_rbf"),
            (max_atomic_number, "max_atomic_number"),
        ):
            _positive_int(value, field)
        if hidden_dim < 2:
            raise ValueError("hidden_dim must be at least 2")
        if hidden_dim % num_heads != 0:
            raise ValueError("hidden_dim must be divisible by num_heads")
        if lmax not in {1, 2}:
            raise ValueError("lmax must be either 1 or 2")
        if rbf_type not in {"expnorm", "gauss"}:
            raise ValueError("rbf_type must be either 'expnorm' or 'gauss'")
        if vecnorm_type not in {"none", "rms", "max_min"}:
            raise ValueError("vecnorm_type must be one of 'none', 'rms', or 'max_min'")
        if output_head not in {"equivariant", "scalar"}:
            raise ValueError("output_head must be either 'equivariant' or 'scalar'")
        if readout not in {"sum", "mean"}:
            raise ValueError("readout must be either 'sum' or 'mean'")

        self.num_targets = num_targets
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.num_rbf = num_rbf
        self.lmax = lmax
        self.rbf_type = rbf_type
        self.trainable_rbf = bool(trainable_rbf)
        self.vecnorm_type = vecnorm_type
        self.trainable_vecnorm = bool(trainable_vecnorm)
        self.output_head_name = output_head
        self.readout = readout
        self.max_atomic_number = max_atomic_number
        self.cutoff = VISNET_CUTOFF
        self.max_neighbors = VISNET_MAX_NEIGHBORS
        self.representation_dim = (lmax + 1) ** 2 - 1

        self.atom_embedding = nn.Embedding(max_atomic_number + 1, hidden_dim)
        self.distance_expansion = build_radial_basis(
            rbf_type,
            cutoff=self.cutoff,
            num_rbf=num_rbf,
            trainable=trainable_rbf,
        )
        self.neighbor_embedding = NeighborEmbedding(
            hidden_dim, num_rbf, self.cutoff, max_atomic_number
        )
        self.edge_embedding = EdgeEmbedding(num_rbf, hidden_dim)
        self.blocks = nn.ModuleList(
            ViSMP(
                hidden_dim,
                num_heads,
                activation=activation,
                attn_activation=attn_activation,
                cutoff=self.cutoff,
                vecnorm_type=vecnorm_type,
                trainable_vecnorm=trainable_vecnorm,
                last_layer=index == num_layers - 1,
            )
            for index in range(num_layers)
        )
        self.output_norm = nn.LayerNorm(hidden_dim)
        self.vector_output_norm = VecLayerNorm(
            hidden_dim,
            trainable=trainable_vecnorm,
            norm_type=vecnorm_type,
            eps=VISNET_EPS,
        )
        if output_head == "equivariant":
            self.output_blocks = nn.ModuleList(
                (
                    GatedEquivariantBlock(
                        hidden_dim,
                        hidden_dim // 2,
                        activation=activation,
                        scalar_activation=True,
                    ),
                    GatedEquivariantBlock(
                        hidden_dim // 2,
                        num_targets,
                        activation=activation,
                    ),
                )
            )
            self.scalar_output = None
        else:
            self.output_blocks = nn.ModuleList()
            self.scalar_output = nn.Sequential(
                Dense(hidden_dim, hidden_dim // 2, activation=build_activation(activation)),
                Dense(hidden_dim // 2, num_targets),
            )

    def forward(self, batch: Batch) -> Tensor:
        """Return raw graph predictions with shape ``[B, num_targets]``."""

        atomic_number, pos, edge_index, graph_batch, num_graphs = self._validate_batch(batch)
        scalar, vector = self._encode(atomic_number, pos, edge_index)
        if self.scalar_output is None:
            for block in self.output_blocks:
                scalar, vector = block(scalar, vector)
            # Preserve the source's no-unused-vector-parameter DDP behavior.
            atom_output = scalar + vector.sum() * 0.0
        else:
            atom_output = self.scalar_output(scalar)
        return scatter(
            atom_output,
            graph_batch,
            dim=0,
            dim_size=num_graphs,
            reduce=self.readout,
        )

    def forward_with_forces(
        self, batch: Batch, *, target_index: int | None = None
    ) -> tuple[Tensor, Tensor]:
        """Return scalar predictions and conservative coordinate forces.

        The shared graph-level trainer intentionally does not consume force
        labels.  This helper leaves force-supervised losses available to an
        explicit caller without mutating ``batch.pos``.
        """

        if not batch.pos.requires_grad:
            raise ValueError("batch.pos must require gradients to calculate forces")
        prediction = self(batch)
        if target_index is None:
            if self.num_targets != 1:
                raise ValueError("target_index is required when num_targets is not 1")
            energy = prediction[:, 0].sum()
        else:
            if (
                isinstance(target_index, bool)
                or not isinstance(target_index, int)
                or not 0 <= target_index < self.num_targets
            ):
                raise ValueError("target_index must select one output target")
            energy = prediction[:, target_index].sum()
        coordinate_grad = grad(
            energy,
            batch.pos,
            create_graph=self.training,
            retain_graph=True,
            allow_unused=True,
        )[0]
        return prediction, -torch.zeros_like(batch.pos) if coordinate_grad is None else -coordinate_grad

    def _encode(
        self, atomic_number: Tensor, pos: Tensor, edge_index: Tensor
    ) -> tuple[Tensor, Tensor]:
        source, target = edge_index
        displacement = pos[source] - pos[target]
        non_self = source != target
        non_self_index = non_self.nonzero(as_tuple=False).flatten()
        distances = displacement.new_zeros((edge_index.shape[1],))
        direction_xyz = torch.zeros_like(displacement)
        if non_self_index.numel():
            non_self_displacement = displacement[non_self_index]
            non_self_distances = torch.linalg.vector_norm(non_self_displacement, dim=-1)
            distances = distances.index_copy(0, non_self_index, non_self_distances)
            direction_xyz = direction_xyz.index_copy(
                0,
                non_self_index,
                non_self_displacement / non_self_distances.clamp_min(VISNET_EPS).unsqueeze(-1),
            )
        directions = spherical_harmonics(direction_xyz, self.lmax)
        radial = self.distance_expansion(distances)
        scalar = self.atom_embedding(atomic_number)
        scalar = self.neighbor_embedding(atomic_number, scalar, edge_index, distances, radial)
        vector = scalar.new_zeros((scalar.shape[0], self.representation_dim, self.hidden_dim))
        edge_attr = self.edge_embedding(scalar, edge_index, radial)
        for block in self.blocks:
            scalar_update, vector_update, edge_update = block(
                scalar, vector, edge_index, distances, edge_attr, directions
            )
            scalar = scalar + scalar_update
            vector = vector + vector_update
            if edge_update is not None:
                edge_attr = edge_attr + edge_update
        return self.output_norm(scalar), self.vector_output_norm(vector)

    def _validate_batch(
        self, batch: Batch
    ) -> tuple[Tensor, Tensor, Tensor, Tensor, int]:
        values = tuple(getattr(batch, name, None) for name in self.required_batch_fields)
        if not all(isinstance(value, Tensor) for value in values):
            raise ValueError(f"batch must provide {', '.join(self.required_batch_fields)} tensors")
        atomic_number, pos, edge_index, graph_batch = values
        assert isinstance(atomic_number, Tensor)
        assert isinstance(pos, Tensor)
        assert isinstance(edge_index, Tensor)
        assert isinstance(graph_batch, Tensor)
        node_count = atomic_number.shape[0] if atomic_number.ndim == 1 else -1
        if (
            atomic_number.ndim != 1
            or node_count < 1
            or atomic_number.dtype != torch.long
            or atomic_number.min() < 1
            or atomic_number.max() > self.max_atomic_number
        ):
            raise ValueError("batch.atomic_number must contain valid element IDs with shape [N]")
        if pos.shape != (node_count, 3) or pos.dtype != torch.float32:
            raise ValueError("batch.pos must have shape [N, 3] and dtype torch.float32")
        if not bool(torch.isfinite(pos).all()):
            raise ValueError("batch.pos must contain only finite values")
        if any(value.device != pos.device for value in values):
            raise ValueError("all ViSNet batch tensors must share the position device")
        num_graphs = validate_batched_molecular_graph(
            edge_index,
            graph_batch,
            num_nodes=node_count,
            device=pos.device,
            edge_field="visnet_edge_index",
        )
        self._validate_spatial_edges(pos, edge_index)
        return atomic_number, pos, edge_index, graph_batch, num_graphs

    def _validate_spatial_edges(self, pos: Tensor, edge_index: Tensor) -> None:
        source, target = edge_index
        if source.numel() == 0:
            raise ValueError("batch.visnet_edge_index must include one self-loop per atom")
        encoded = source * pos.shape[0] + target
        if torch.unique(encoded).numel() != encoded.numel():
            raise ValueError("batch.visnet_edge_index must not contain duplicate edges")
        self_loops = source == target
        self_loop_count = torch.bincount(source[self_loops], minlength=pos.shape[0])
        if not bool(torch.equal(self_loop_count, torch.ones_like(self_loop_count))):
            raise ValueError("batch.visnet_edge_index must include exactly one self-loop per atom")
        incoming = torch.bincount(target, minlength=pos.shape[0])
        if bool((incoming > self.max_neighbors).any()):
            raise ValueError("batch.visnet_edge_index exceeds the fixed incoming-neighbor cap")
        non_self = ~self_loops
        if not bool(non_self.any()):
            return
        distances = torch.linalg.vector_norm(pos[source[non_self]] - pos[target[non_self]], dim=-1)
        if bool((distances >= self.cutoff).any()):
            raise ValueError("batch.visnet_edge_index contains an edge outside the fixed cutoff")
        if bool((distances <= VISNET_EPS).any()):
            raise ValueError("batch.visnet_edge_index contains a coincident atom pair")


def _positive_int(value: int, field: str) -> None:
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise ValueError(f"{field} must be a positive integer")


__all__ = ["ViSNet"]
