"""e3nn-backed Equiformer core, imported lazily by the public wrapper."""

from __future__ import annotations

import math

import torch
from e3nn import nn as e3nn_nn
from e3nn import o3
from torch import Tensor, nn
from torch_geometric.data import Batch
from torch_geometric.utils import scatter

from ..base import BaseMolecularModel
from ..contracts import validate_batched_molecular_graph
from .constants import (
    EQUIFORMER_AVG_DEGREE,
    EQUIFORMER_AVG_NUM_NODES,
    EQUIFORMER_CUTOFF,
    EQUIFORMER_EPS,
    EQUIFORMER_MAX_ATOMIC_NUMBER,
    EQUIFORMER_MAX_NEIGHBORS,
    EQUIFORMER_NUM_RADIAL,
)
from .layers import (
    EdgeDegreeEmbedding,
    EquiformerBlock,
    EquivariantLayerNorm,
    EquivariantLinear,
)
from .radial import GaussianRadialBasis


class EquiformerCore(BaseMolecularModel):
    """Nonlinear-message SE(3) Equiformer over an explicit radius graph."""

    required_batch_fields = (
        "atomic_number",
        "pos",
        "equiformer_edge_index",
        "batch",
    )

    def __init__(
        self,
        num_targets: int,
        scalar_channels: int = 128,
        vector_channels: int = 64,
        tensor_channels: int = 32,
        num_layers: int = 6,
        num_radial: int = EQUIFORMER_NUM_RADIAL,
        radial_hidden_dim: int = 64,
        head_scalar_channels: int = 32,
        head_vector_channels: int = 16,
        head_tensor_channels: int = 8,
        num_heads: int = 4,
        ffn_multiplier: int = 3,
        feature_scalar_channels: int = 512,
        attention_dropout: float = 0.2,
        max_atomic_number: int = EQUIFORMER_MAX_ATOMIC_NUMBER,
        average_degree: float = EQUIFORMER_AVG_DEGREE,
        average_num_nodes: float = EQUIFORMER_AVG_NUM_NODES,
    ) -> None:
        super().__init__()
        for value, name in (
            (num_targets, "num_targets"),
            (scalar_channels, "scalar_channels"),
            (vector_channels, "vector_channels"),
            (tensor_channels, "tensor_channels"),
            (num_layers, "num_layers"),
            (num_radial, "num_radial"),
            (radial_hidden_dim, "radial_hidden_dim"),
            (head_scalar_channels, "head_scalar_channels"),
            (head_vector_channels, "head_vector_channels"),
            (head_tensor_channels, "head_tensor_channels"),
            (num_heads, "num_heads"),
            (ffn_multiplier, "ffn_multiplier"),
            (feature_scalar_channels, "feature_scalar_channels"),
            (max_atomic_number, "max_atomic_number"),
        ):
            _positive_int(value, name)
        if not isinstance(attention_dropout, (float, int)) or not 0.0 <= attention_dropout < 1.0:
            raise ValueError("attention_dropout must be in [0, 1)")
        for value, name in (
            (average_degree, "average_degree"),
            (average_num_nodes, "average_num_nodes"),
        ):
            if not isinstance(value, (float, int)) or not math.isfinite(value) or value <= 0.0:
                raise ValueError(f"{name} must be a finite positive number")

        self.num_targets = num_targets
        self.max_atomic_number = max_atomic_number
        self.cutoff = EQUIFORMER_CUTOFF
        self.max_neighbors = EQUIFORMER_MAX_NEIGHBORS
        self.average_degree = float(average_degree)
        self.average_num_nodes = float(average_num_nodes)
        self.irreps_node = o3.Irreps(
            f"{scalar_channels}x0e+{vector_channels}x1e+{tensor_channels}x2e"
        )
        self.irreps_edge = o3.Irreps("1x0e+1x1e+1x2e")
        self.irreps_head = o3.Irreps(
            f"{head_scalar_channels}x0e+{head_vector_channels}x1e+{head_tensor_channels}x2e"
        )
        self.irreps_middle = o3.Irreps(
            f"{scalar_channels * ffn_multiplier}x0e+"
            f"{vector_channels * ffn_multiplier}x1e+"
            f"{tensor_channels * ffn_multiplier}x2e"
        )
        self.irreps_feature = o3.Irreps(f"{feature_scalar_channels}x0e")

        # A learned scalar lookup is equivalent to the source's one-hot scalar
        # linear map, while allowing element types beyond QM9's H/C/N/O/F.
        self.atom_embedding = nn.Embedding(max_atomic_number + 1, scalar_channels)
        self.rbf = GaussianRadialBasis(num_radial, self.cutoff)
        radial_channels = (num_radial, radial_hidden_dim, radial_hidden_dim)
        self.edge_degree_embedding = EdgeDegreeEmbedding(
            self.irreps_node,
            self.irreps_edge,
            radial_channels=radial_channels,
            average_degree=self.average_degree,
        )
        blocks: list[EquiformerBlock] = []
        for index in range(num_layers):
            block_output = (
                self.irreps_feature if index == num_layers - 1 else self.irreps_node
            )
            blocks.append(
                EquiformerBlock(
                    self.irreps_node,
                    block_output,
                    self.irreps_edge,
                    self.irreps_head,
                    self.irreps_middle,
                    num_heads=num_heads,
                    radial_channels=radial_channels,
                    attention_dropout=float(attention_dropout),
                )
            )
        self.blocks = nn.ModuleList(blocks)
        self.final_norm = EquivariantLayerNorm(self.irreps_feature)
        self.head_first = EquivariantLinear(self.irreps_feature, self.irreps_feature)
        self.head_activation = e3nn_nn.Activation(
            self.irreps_feature,
            [torch.nn.functional.silu],
        )
        self.head_last = EquivariantLinear(
            self.irreps_feature,
            o3.Irreps(f"{num_targets}x0e"),
        )
        self.apply(_initialize_source_linear_layers)

    def forward(self, batch: Batch) -> Tensor:
        atomic_number, pos, edge_index, graph_batch, num_graphs = self._validate_batch(batch)
        source, target = edge_index
        edge_vectors = pos[source] - pos[target]
        distances = torch.linalg.vector_norm(edge_vectors, dim=-1)
        edge_attributes = o3.spherical_harmonics(
            self.irreps_edge,
            edge_vectors,
            normalize=True,
            normalization="component",
        )
        edge_scalars = self.rbf(distances)

        scalar = self.atom_embedding(atomic_number)
        node_features = torch.cat(
            (
                scalar,
                scalar.new_zeros((scalar.shape[0], self.irreps_node.dim - scalar.shape[1])),
            ),
            dim=-1,
        )
        node_features = node_features + self.edge_degree_embedding(
            node_features,
            source,
            target,
            edge_attributes,
            edge_scalars,
        )
        node_attr = node_features.new_ones((node_features.shape[0], 1))
        for block in self.blocks:
            node_features = block(
                node_features,
                node_attr,
                source,
                target,
                edge_attributes,
                edge_scalars,
            )
        atom_outputs = self.head_last(
            self.head_activation(self.head_first(self.final_norm(node_features)))
        )
        return scatter(
            atom_outputs,
            graph_batch,
            dim=0,
            dim_size=num_graphs,
            reduce="sum",
        ) / math.sqrt(self.average_num_nodes)

    def _validate_batch(
        self, batch: Batch
    ) -> tuple[Tensor, Tensor, Tensor, Tensor, int]:
        values = tuple(getattr(batch, field, None) for field in self.required_batch_fields)
        if not all(isinstance(value, Tensor) for value in values):
            raise ValueError(
                "batch must provide " + ", ".join(self.required_batch_fields) + " tensors"
            )
        atomic_number, pos, edge_index, graph_batch = values
        assert isinstance(atomic_number, Tensor)
        assert isinstance(pos, Tensor)
        assert isinstance(edge_index, Tensor)
        assert isinstance(graph_batch, Tensor)
        node_count = atomic_number.shape[0] if atomic_number.ndim == 1 else -1
        if atomic_number.ndim != 1 or node_count < 1 or atomic_number.dtype != torch.long:
            raise ValueError("batch.atomic_number must have shape [N] and dtype torch.long")
        if atomic_number.min() < 1 or atomic_number.max() > self.max_atomic_number:
            raise ValueError("batch.atomic_number contains a value outside the configured vocabulary")
        if pos.shape != (node_count, 3) or pos.dtype != torch.float32:
            raise ValueError("batch.pos must have shape [N, 3] and dtype torch.float32")
        if not bool(torch.isfinite(pos).all()):
            raise ValueError("batch.pos must contain only finite values")
        if any(value.device != pos.device for value in values):
            raise ValueError("all Equiformer batch tensors must share the position device")
        num_graphs = validate_batched_molecular_graph(
            edge_index,
            graph_batch,
            num_nodes=node_count,
            device=pos.device,
            edge_field="equiformer_edge_index",
            forbid_self_loops=True,
        )
        self._validate_spatial_edges(pos, edge_index)
        return atomic_number, pos, edge_index, graph_batch, num_graphs

    def _validate_spatial_edges(self, pos: Tensor, edge_index: Tensor) -> None:
        if edge_index.numel() == 0:
            return
        source, target = edge_index
        node_count = pos.shape[0]
        encoded = source * node_count + target
        if torch.unique(encoded).numel() != encoded.numel():
            raise ValueError("batch.equiformer_edge_index must not contain duplicate edges")
        reverse = target * node_count + source
        if not bool(torch.isin(reverse, encoded).all()):
            raise ValueError("batch.equiformer_edge_index must contain reciprocal radius edges")
        incoming = torch.bincount(target, minlength=node_count)
        if bool((incoming > self.max_neighbors).any()):
            raise ValueError("batch.equiformer_edge_index exceeds the maximum incoming-neighbor cap")
        distances = torch.linalg.vector_norm(pos[source] - pos[target], dim=-1)
        if bool((distances <= EQUIFORMER_EPS).any()):
            raise ValueError("batch.equiformer_edge_index contains coincident atoms")
        if bool((distances >= self.cutoff).any()):
            raise ValueError("batch.equiformer_edge_index contains an edge outside the strict cutoff")


def _positive_int(value: int, field: str) -> None:
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise ValueError(f"{field} must be a positive integer")


def _initialize_source_linear_layers(module: nn.Module) -> None:
    """Keep the author source's zero-bias Linear/LayerNorm initialization."""

    if isinstance(module, nn.Linear) and module.bias is not None:
        nn.init.zeros_(module.bias)
    elif isinstance(module, nn.LayerNorm):
        nn.init.ones_(module.weight)
        nn.init.zeros_(module.bias)


__all__ = ["EquiformerCore"]
