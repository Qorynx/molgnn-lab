"""TorchMD-ET layers used by the official denoising implementation.

The equations and parameter layout follow the MIT-licensed source at
``shehzaidi/pre-training-via-denoising`` while replacing the legacy
``torch_scatter``/``MessagePassing`` dependency with current PyG scatter.
"""

from __future__ import annotations

import torch
from torch import Tensor, nn
from torch_geometric.utils import scatter

from .radial import CosineCutoff


class NeighborEmbedding(nn.Module):
    def __init__(
        self,
        hidden_channels: int,
        num_rbf: int,
        cutoff_lower: float,
        cutoff_upper: float,
        max_z: int,
    ) -> None:
        super().__init__()
        self.embedding = nn.Embedding(max_z, hidden_channels)
        self.distance_proj = nn.Linear(num_rbf, hidden_channels)
        self.combine = nn.Linear(2 * hidden_channels, hidden_channels)
        self.cutoff = CosineCutoff(cutoff_lower, cutoff_upper)
        self.reset_parameters()

    def reset_parameters(self) -> None:
        self.embedding.reset_parameters()
        nn.init.xavier_uniform_(self.distance_proj.weight)
        nn.init.zeros_(self.distance_proj.bias)
        nn.init.xavier_uniform_(self.combine.weight)
        nn.init.zeros_(self.combine.bias)

    def forward(
        self,
        z: Tensor,
        x: Tensor,
        edge_index: Tensor,
        edge_weight: Tensor,
        edge_attr: Tensor,
    ) -> Tensor:
        source, target = edge_index
        mask = source != target
        source = source[mask]
        target = target[mask]
        weights = self.distance_proj(edge_attr[mask])
        weights = weights * self.cutoff(edge_weight[mask]).unsqueeze(-1)
        messages = self.embedding(z[source]) * weights
        neighbors = scatter(
            messages,
            target,
            dim=0,
            dim_size=x.shape[0],
            reduce="sum",
        )
        return self.combine(torch.cat((x, neighbors), dim=-1))


class EquivariantMultiHeadAttention(nn.Module):
    def __init__(
        self,
        hidden_channels: int,
        num_rbf: int,
        distance_influence: str,
        num_heads: int,
        cutoff_lower: float,
        cutoff_upper: float,
    ) -> None:
        super().__init__()
        if hidden_channels % num_heads:
            raise ValueError("hidden_channels must be divisible by num_heads")
        if distance_influence not in {"keys", "values", "both", "none"}:
            raise ValueError("distance_influence must be keys, values, both, or none")
        self.distance_influence = distance_influence
        self.num_heads = int(num_heads)
        self.hidden_channels = int(hidden_channels)
        self.head_dim = hidden_channels // num_heads

        self.layernorm = nn.LayerNorm(hidden_channels)
        self.act = nn.SiLU()
        self.attn_activation = nn.SiLU()
        self.cutoff = CosineCutoff(cutoff_lower, cutoff_upper)
        self.q_proj = nn.Linear(hidden_channels, hidden_channels)
        self.k_proj = nn.Linear(hidden_channels, hidden_channels)
        self.v_proj = nn.Linear(hidden_channels, 3 * hidden_channels)
        self.o_proj = nn.Linear(hidden_channels, 3 * hidden_channels)
        self.vec_proj = nn.Linear(hidden_channels, 3 * hidden_channels, bias=False)
        self.dk_proj = (
            nn.Linear(num_rbf, hidden_channels)
            if distance_influence in {"keys", "both"}
            else None
        )
        self.dv_proj = (
            nn.Linear(num_rbf, 3 * hidden_channels)
            if distance_influence in {"values", "both"}
            else None
        )
        self.reset_parameters()

    def reset_parameters(self) -> None:
        self.layernorm.reset_parameters()
        for layer in (self.q_proj, self.k_proj, self.v_proj, self.o_proj):
            nn.init.xavier_uniform_(layer.weight)
            nn.init.zeros_(layer.bias)
        nn.init.xavier_uniform_(self.vec_proj.weight)
        for layer in (self.dk_proj, self.dv_proj):
            if layer is not None:
                nn.init.xavier_uniform_(layer.weight)
                nn.init.zeros_(layer.bias)

    def forward(
        self,
        x: Tensor,
        vec: Tensor,
        edge_index: Tensor,
        distances: Tensor,
        edge_attr: Tensor,
        directions: Tensor,
    ) -> tuple[Tensor, Tensor]:
        source, target = edge_index
        normalized = self.layernorm(x)
        q = self.q_proj(normalized).reshape(-1, self.num_heads, self.head_dim)
        k = self.k_proj(normalized).reshape(-1, self.num_heads, self.head_dim)
        value = self.v_proj(normalized).reshape(
            -1, self.num_heads, 3 * self.head_dim
        )

        projected = self.vec_proj(vec)
        vec1, vec2, vec3 = torch.split(
            projected, self.hidden_channels, dim=-1
        )
        vec_dot = (vec1 * vec2).sum(dim=1)
        vec_heads = vec.reshape(-1, 3, self.num_heads, self.head_dim)

        distance_keys = (
            self.act(self.dk_proj(edge_attr)).reshape(
                -1, self.num_heads, self.head_dim
            )
            if self.dk_proj is not None
            else None
        )
        distance_values = (
            self.act(self.dv_proj(edge_attr)).reshape(
                -1, self.num_heads, 3 * self.head_dim
            )
            if self.dv_proj is not None
            else None
        )

        attention = q[target] * k[source]
        if distance_keys is not None:
            attention = attention * distance_keys
        attention = self.attn_activation(attention.sum(dim=-1))
        attention = attention * self.cutoff(distances).unsqueeze(-1)

        edge_value = value[source]
        if distance_values is not None:
            edge_value = edge_value * distance_values
        scalar_value, vector_value1, vector_value2 = torch.split(
            edge_value, self.head_dim, dim=-1
        )
        scalar_message = scalar_value * attention.unsqueeze(-1)
        vector_message = (
            vec_heads[source] * vector_value1.unsqueeze(1)
            + directions.unsqueeze(-1).unsqueeze(-1)
            * vector_value2.unsqueeze(1)
        )
        aggregated_scalar = scatter(
            scalar_message,
            target,
            dim=0,
            dim_size=x.shape[0],
            reduce="sum",
        ).reshape(-1, self.hidden_channels)
        aggregated_vector = scatter(
            vector_message,
            target,
            dim=0,
            dim_size=x.shape[0],
            reduce="sum",
        ).reshape(-1, 3, self.hidden_channels)

        out1, out2, out3 = torch.split(
            self.o_proj(aggregated_scalar), self.hidden_channels, dim=-1
        )
        scalar_update = vec_dot * out2 + out3
        vector_update = vec3 * out1.unsqueeze(1) + aggregated_vector
        return scalar_update, vector_update


class EquivariantLayerNorm(nn.Module):
    """Whitening-based rotation-equivariant vector normalization from source."""

    def __init__(
        self,
        normalized_shape: int,
        eps: float = 1.0e-5,
        elementwise_linear: bool = True,
    ) -> None:
        super().__init__()
        self.normalized_shape = (int(normalized_shape),)
        self.eps = float(eps)
        self.elementwise_linear = bool(elementwise_linear)
        if elementwise_linear:
            self.weight = nn.Parameter(torch.empty(self.normalized_shape))
        else:
            self.register_parameter("weight", None)
        self.reset_parameters()

    def reset_parameters(self) -> None:
        if self.weight is not None:
            nn.init.ones_(self.weight)

    def _symmetric_inverse_sqrt(self, matrix: Tensor) -> Tensor:
        # Preserve the source's rank-aware SVD path. This is deliberately more
        # specific than a generic LayerNorm because it is part of the released
        # checkpoint profile.
        _, singular_values, vectors_h = torch.linalg.svd(
            matrix, full_matrices=False
        )
        vectors = vectors_h.transpose(-2, -1)
        good = (
            singular_values
            > singular_values.max(dim=-1, keepdim=True).values
            * singular_values.shape[-1]
            * torch.finfo(singular_values.dtype).eps
        )
        components = good.sum(dim=-1)
        common = int(components.max().item())
        unbalanced = bool((components.min() != common).item())
        if common < singular_values.shape[-1]:
            singular_values = singular_values[..., :common]
            vectors = vectors[..., :common]
            if unbalanced:
                good = good[..., :common]
        if unbalanced:
            singular_values = singular_values.where(
                good,
                torch.zeros((), device=matrix.device, dtype=matrix.dtype),
            )
        return (
            vectors
            * (singular_values + self.eps).rsqrt().unsqueeze(-2)
        ) @ vectors.transpose(-2, -1)

    def forward(self, value: Tensor) -> Tensor:
        original_dtype = value.dtype
        centered = value.to(torch.float64)
        centered = centered - centered.mean(dim=-1, keepdim=True)
        covariance = (
            centered @ centered.transpose(-1, -2)
        ) / self.normalized_shape[0]
        regularizer = torch.diag(
            torch.tensor(
                (1.0, 2.0, 3.0),
                dtype=covariance.dtype,
                device=covariance.device,
            )
        ).unsqueeze(0)
        whitened = self._symmetric_inverse_sqrt(
            covariance + self.eps * regularizer
        ) @ centered
        whitened = whitened.to(original_dtype)
        if self.weight is not None:
            whitened = whitened * self.weight.reshape(1, 1, -1)
        return whitened


class GatedEquivariantBlock(nn.Module):
    def __init__(
        self,
        hidden_channels: int,
        out_channels: int,
        *,
        intermediate_channels: int | None = None,
        scalar_activation: bool = False,
    ) -> None:
        super().__init__()
        intermediate = hidden_channels if intermediate_channels is None else intermediate_channels
        self.out_channels = int(out_channels)
        self.vec1_proj = nn.Linear(hidden_channels, hidden_channels, bias=False)
        self.vec2_proj = nn.Linear(hidden_channels, out_channels, bias=False)
        self.update_net = nn.Sequential(
            nn.Linear(2 * hidden_channels, intermediate),
            nn.SiLU(),
            nn.Linear(intermediate, 2 * out_channels),
        )
        self.act = nn.SiLU() if scalar_activation else None
        self.reset_parameters()

    def reset_parameters(self) -> None:
        nn.init.xavier_uniform_(self.vec1_proj.weight)
        nn.init.xavier_uniform_(self.vec2_proj.weight)
        for index in (0, 2):
            layer = self.update_net[index]
            assert isinstance(layer, nn.Linear)
            nn.init.xavier_uniform_(layer.weight)
            nn.init.zeros_(layer.bias)

    def forward(self, x: Tensor, vec: Tensor) -> tuple[Tensor, Tensor]:
        vec_norm = torch.linalg.vector_norm(self.vec1_proj(vec), dim=-2)
        projected_vec = self.vec2_proj(vec)
        scalar, gate = torch.split(
            self.update_net(torch.cat((x, vec_norm), dim=-1)),
            self.out_channels,
            dim=-1,
        )
        vector = projected_vec * gate.unsqueeze(1)
        if self.act is not None:
            scalar = self.act(scalar)
        return scalar, vector


class EquivariantScalarHead(nn.Module):
    def __init__(self, hidden_channels: int, out_channels: int) -> None:
        super().__init__()
        half = max(1, hidden_channels // 2)
        self.output_network = nn.ModuleList(
            (
                GatedEquivariantBlock(
                    hidden_channels, half, scalar_activation=True
                ),
                GatedEquivariantBlock(half, out_channels),
            )
        )

    def forward(self, x: Tensor, vec: Tensor) -> Tensor:
        for layer in self.output_network:
            x, vec = layer(x, vec)
        return x + vec.sum() * 0.0


class EquivariantVectorHead(nn.Module):
    def __init__(self, hidden_channels: int) -> None:
        super().__init__()
        half = max(1, hidden_channels // 2)
        self.output_network = nn.ModuleList(
            (
                GatedEquivariantBlock(
                    hidden_channels, half, scalar_activation=True
                ),
                GatedEquivariantBlock(half, 1),
            )
        )

    def forward(self, x: Tensor, vec: Tensor) -> Tensor:
        for layer in self.output_network:
            x, vec = layer(x, vec)
        return vec.squeeze(-1)


__all__ = [
    "EquivariantLayerNorm",
    "EquivariantMultiHeadAttention",
    "EquivariantScalarHead",
    "EquivariantVectorHead",
    "GatedEquivariantBlock",
    "NeighborEmbedding",
]
