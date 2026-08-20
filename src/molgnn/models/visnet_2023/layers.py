"""Vector-scalar interactive layers from the paper-backed ViSNet core."""

from __future__ import annotations

from collections.abc import Callable

import torch
from torch import Tensor, nn
from torch_geometric.utils import scatter

from .constants import VISNET_CUTOFF, VISNET_EPS
from .geometry import VecLayerNorm, vector_rejection
from .radial import CosineCutoff


class Dense(nn.Linear):
    """AI2BMD-style Linear layer with Xavier weights and zero bias."""

    def __init__(
        self,
        in_features: int,
        out_features: int,
        *,
        bias: bool = True,
        activation: nn.Module | None = None,
    ) -> None:
        super().__init__(in_features, out_features, bias=bias)
        self.activation = activation if activation is not None else nn.Identity()

    def reset_parameters(self) -> None:
        nn.init.xavier_uniform_(self.weight)
        if self.bias is not None:
            nn.init.zeros_(self.bias)

    def forward(self, value: Tensor) -> Tensor:
        return self.activation(super().forward(value))


def build_activation(name: str) -> nn.Module:
    """Construct the activation names accepted by the supplied ViSNet code."""

    factories: dict[str, Callable[[], nn.Module]] = {
        "silu": nn.SiLU,
        "tanh": nn.Tanh,
        "sigmoid": nn.Sigmoid,
        "swish": nn.SiLU,
        "ssp": ShiftedSoftplus,
    }
    try:
        return factories[name]()
    except KeyError as exc:
        supported = ", ".join(sorted(factories))
        raise ValueError(f"unsupported activation '{name}'; choose one of {supported}") from exc


class ShiftedSoftplus(nn.Module):
    """Softplus shifted so zero maps to zero."""

    def forward(self, value: Tensor) -> Tensor:
        return torch.nn.functional.softplus(value) - torch.log(value.new_tensor(2.0))


class NeighborEmbedding(nn.Module):
    """Initial scalar neighborhood embedding; self loops are intentionally excluded."""

    def __init__(
        self, hidden_channels: int, num_rbf: int, cutoff: float, max_atomic_number: int
    ) -> None:
        super().__init__()
        self.hidden_channels = hidden_channels
        self.embedding = nn.Embedding(max_atomic_number + 1, hidden_channels)
        self.distance_projection = Dense(num_rbf, hidden_channels)
        self.combine = Dense(hidden_channels * 2, hidden_channels)
        self.cutoff = CosineCutoff(cutoff)

    def forward(
        self,
        atomic_number: Tensor,
        scalar: Tensor,
        edge_index: Tensor,
        distances: Tensor,
        radial: Tensor,
    ) -> Tensor:
        source, target = edge_index
        non_self = source != target
        if not bool(non_self.any()):
            neighbor_sum = scalar.new_zeros(scalar.shape)
        else:
            source = source[non_self]
            target = target[non_self]
            distances = distances[non_self]
            radial = radial[non_self]
            filter_weight = self.distance_projection(radial) * self.cutoff(distances).unsqueeze(-1)
            neighbor_messages = self.embedding(atomic_number)[source] * filter_weight
            neighbor_sum = scatter(
                neighbor_messages,
                target,
                dim=0,
                dim_size=scalar.shape[0],
                reduce="sum",
            )
        return self.combine(torch.cat((scalar, neighbor_sum), dim=-1))


class EdgeEmbedding(nn.Module):
    """Pairwise edge initialization without an aggregation step."""

    def __init__(self, num_rbf: int, hidden_channels: int) -> None:
        super().__init__()
        self.edge_projection = Dense(num_rbf, hidden_channels)

    def forward(self, scalar: Tensor, edge_index: Tensor, radial: Tensor) -> Tensor:
        source, target = edge_index
        return (scalar[target] + scalar[source]) * self.edge_projection(radial)


class ViSMP(nn.Module):
    """One ViSNet vector-scalar interactive message-passing block."""

    def __init__(
        self,
        hidden_channels: int,
        num_heads: int,
        *,
        activation: str,
        attn_activation: str,
        cutoff: float = VISNET_CUTOFF,
        vecnorm_type: str = "none",
        trainable_vecnorm: bool = False,
        last_layer: bool = False,
    ) -> None:
        super().__init__()
        if hidden_channels % num_heads != 0:
            raise ValueError("hidden_channels must be divisible by num_heads")
        self.hidden_channels = hidden_channels
        self.num_heads = num_heads
        self.head_dim = hidden_channels // num_heads
        self.last_layer = last_layer

        self.layernorm = nn.LayerNorm(hidden_channels)
        self.vec_layernorm = VecLayerNorm(
            hidden_channels,
            trainable=trainable_vecnorm,
            norm_type=vecnorm_type,
            eps=VISNET_EPS,
        )
        self.activation = build_activation(activation)
        self.attn_activation = build_activation(attn_activation)
        self.cutoff = CosineCutoff(cutoff)

        self.vec_projection = Dense(hidden_channels, hidden_channels * 3, bias=False)
        self.q_projection = Dense(hidden_channels, hidden_channels)
        self.k_projection = Dense(hidden_channels, hidden_channels)
        self.v_projection = Dense(hidden_channels, hidden_channels)
        self.dk_projection = Dense(hidden_channels, hidden_channels)
        self.dv_projection = Dense(hidden_channels, hidden_channels)
        self.scalar_projection = Dense(hidden_channels, hidden_channels * 2)
        self.output_projection = Dense(hidden_channels, hidden_channels * 3)
        if last_layer:
            self.edge_projection = None
            self.source_vector_projection = None
            self.target_vector_projection = None
        else:
            self.edge_projection = Dense(hidden_channels, hidden_channels)
            self.source_vector_projection = Dense(hidden_channels, hidden_channels, bias=False)
            self.target_vector_projection = Dense(hidden_channels, hidden_channels, bias=False)

    def forward(
        self,
        scalar: Tensor,
        vector: Tensor,
        edge_index: Tensor,
        distances: Tensor,
        edge_attr: Tensor,
        directions: Tensor,
    ) -> tuple[Tensor, Tensor, Tensor | None]:
        normalized_scalar = self.layernorm(scalar)
        normalized_vector = self.vec_layernorm(vector)
        q = self.q_projection(normalized_scalar).view(-1, self.num_heads, self.head_dim)
        k = self.k_projection(normalized_scalar).view(-1, self.num_heads, self.head_dim)
        value = self.v_projection(normalized_scalar).view(-1, self.num_heads, self.head_dim)
        dk = self.activation(self.dk_projection(edge_attr)).view(-1, self.num_heads, self.head_dim)
        dv = self.activation(self.dv_projection(edge_attr)).view(-1, self.num_heads, self.head_dim)

        vec1, vec2, vec3 = self.vec_projection(normalized_vector).split(
            self.hidden_channels, dim=-1
        )
        angle_features = (vec1 * vec2).sum(dim=1)
        scalar_message, vector_message = self._message(
            q, k, value, dk, dv, normalized_vector, edge_index, distances, directions
        )

        edge_update = self._edge_update(normalized_vector, edge_index, directions, edge_attr)
        out1, out2, out3 = self.output_projection(scalar_message).split(
            self.hidden_channels, dim=-1
        )
        scalar_update = angle_features * out2 + out3
        vector_update = vec3 * out1.unsqueeze(1) + vector_message
        return scalar_update, vector_update, edge_update

    def _message(
        self,
        q: Tensor,
        k: Tensor,
        value: Tensor,
        dk: Tensor,
        dv: Tensor,
        vector: Tensor,
        edge_index: Tensor,
        distances: Tensor,
        directions: Tensor,
    ) -> tuple[Tensor, Tensor]:
        source, target = edge_index
        # ViSNet intentionally uses an activated dot-product gate, not softmax.
        attention = (q[target] * k[source] * dk).sum(dim=-1)
        attention = self.attn_activation(attention) * self.cutoff(distances).unsqueeze(-1)
        weighted_value = (value[source] * dv * attention.unsqueeze(-1)).reshape(
            -1, self.hidden_channels
        )
        scalar_gate, direction_gate = self.activation(
            self.scalar_projection(weighted_value)
        ).split(self.hidden_channels, dim=-1)
        vector_messages = (
            vector[source] * scalar_gate.unsqueeze(1)
            + directions.unsqueeze(-1) * direction_gate.unsqueeze(1)
        )
        scalar_messages = scatter(
            weighted_value,
            target,
            dim=0,
            dim_size=vector.shape[0],
            reduce="sum",
        )
        vector_messages = scatter(
            vector_messages,
            target,
            dim=0,
            dim_size=vector.shape[0],
            reduce="sum",
        )
        return scalar_messages, vector_messages

    def _edge_update(
        self, vector: Tensor, edge_index: Tensor, directions: Tensor, edge_attr: Tensor
    ) -> Tensor | None:
        if self.last_layer:
            return None
        assert self.edge_projection is not None
        assert self.source_vector_projection is not None
        assert self.target_vector_projection is not None
        source, target = edge_index
        target_rejection = vector_rejection(
            self.target_vector_projection(vector[target]), directions
        )
        source_rejection = vector_rejection(
            self.source_vector_projection(vector[source]), -directions
        )
        dihedral_features = (target_rejection * source_rejection).sum(dim=1)
        return self.activation(self.edge_projection(edge_attr)) * dihedral_features


class GatedEquivariantBlock(nn.Module):
    """PaiNN-style equivariant output block used by ViSNet's paper head."""

    def __init__(
        self,
        hidden_channels: int,
        out_channels: int,
        *,
        activation: str = "silu",
        scalar_activation: bool = False,
    ) -> None:
        super().__init__()
        self.out_channels = out_channels
        self.vec1_projection = Dense(hidden_channels, hidden_channels, bias=False)
        self.vec2_projection = Dense(hidden_channels, out_channels, bias=False)
        self.update_network = nn.Sequential(
            Dense(hidden_channels * 2, hidden_channels, activation=build_activation(activation)),
            Dense(hidden_channels, out_channels * 2),
        )
        self.scalar_activation = build_activation(activation) if scalar_activation else nn.Identity()

    def forward(self, scalar: Tensor, vector: Tensor) -> tuple[Tensor, Tensor]:
        vector_norm = torch.linalg.vector_norm(self.vec1_projection(vector), dim=-2)
        projected_vector = self.vec2_projection(vector)
        scalar, gate = self.update_network(torch.cat((scalar, vector_norm), dim=-1)).split(
            self.out_channels, dim=-1
        )
        return self.scalar_activation(scalar), gate.unsqueeze(1) * projected_vector


__all__ = [
    "Dense",
    "EdgeEmbedding",
    "GatedEquivariantBlock",
    "NeighborEmbedding",
    "ViSMP",
    "build_activation",
]
