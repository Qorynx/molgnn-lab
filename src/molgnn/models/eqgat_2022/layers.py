"""Core scalar/vector layers from EQGAT (Le, Noe, and Clevert, 2022)."""

from __future__ import annotations

import math

import torch
from torch import Tensor, nn
from torch_geometric.utils import scatter, softmax

from .constants import EQGAT_EPS, EQGAT_POLYNOMIAL_CUTOFF_P


class DenseLayer(nn.Linear):
    """Official EQGAT linear layer with zero-initialized biases."""

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
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if self.bias is not None:
            nn.init.zeros_(self.bias)

    def forward(self, value: Tensor) -> Tensor:
        return self.activation(super().forward(value))


class PolynomialCutoff(nn.Module):
    """Sixth-order DimeNet envelope used by the official EQGAT source."""

    def __init__(
        self,
        cutoff: float,
        p: int = EQGAT_POLYNOMIAL_CUTOFF_P,
    ) -> None:
        super().__init__()
        if cutoff <= 0:
            raise ValueError("cutoff must be positive")
        if p < 2:
            raise ValueError("p must be at least 2")
        self.cutoff = float(cutoff)
        self.p = int(p)

    def forward(self, distances: Tensor) -> Tensor:
        scaled = distances / self.cutoff
        p = float(self.p)
        value = 1.0
        value = value - ((p + 1.0) * (p + 2.0) / 2.0) * scaled.pow(p)
        value = value + p * (p + 2.0) * scaled.pow(p + 1.0)
        value = value - (p * (p + 1.0) / 2.0) * scaled.pow(p + 2.0)
        return value * (scaled < 1.0).to(dtype=distances.dtype)


class BesselExpansion(nn.Module):
    """Fixed radial Bessel expansion from the official EQGAT implementation."""

    def __init__(self, cutoff: float, num_radial: int) -> None:
        super().__init__()
        if cutoff <= 0:
            raise ValueError("cutoff must be positive")
        _positive_int(num_radial, "num_radial")
        self.cutoff = float(cutoff)
        self.num_radial = num_radial
        frequency = math.pi * torch.arange(1, num_radial + 1, dtype=torch.float32)
        self.register_buffer("frequency", frequency)

    def forward(self, distances: Tensor) -> Tensor:
        if distances.ndim != 1:
            raise ValueError("distances must have shape [E]")
        angles = distances.unsqueeze(-1) / self.cutoff * self.frequency
        numerator = torch.sin(angles)
        denominator = torch.where(distances == 0.0, torch.ones_like(distances), distances)
        return numerator / denominator.unsqueeze(-1) * math.sqrt(2.0 / self.cutoff)


class GatedEquivariantBlock(nn.Module):
    """EQGAT's scalar/vector gated update and scalarization block."""

    def __init__(
        self,
        scalar_in: int,
        vector_in: int,
        scalar_out: int,
        vector_out: int | None,
        *,
        scalar_hidden: int | None = None,
        vector_hidden: int | None = None,
        eps: float = EQGAT_EPS,
        use_mlp: bool = False,
    ) -> None:
        super().__init__()
        for value, name in (
            (scalar_in, "scalar_in"),
            (vector_in, "vector_in"),
            (scalar_out, "scalar_out"),
        ):
            _positive_int(value, name)
        if vector_out is not None:
            _positive_int(vector_out, "vector_out")
        if eps <= 0:
            raise ValueError("eps must be positive")

        self.scalar_in = scalar_in
        self.vector_in = vector_in
        self.scalar_out = scalar_out
        self.vector_out = 0 if vector_out is None else vector_out
        self.scalar_hidden = scalar_hidden or max(scalar_in, scalar_out)
        self.vector_hidden = vector_hidden or max(vector_in, self.vector_out)
        self.eps = float(eps)
        self.use_mlp = use_mlp

        self.vector_projection = DenseLayer(
            vector_in, self.vector_hidden + self.vector_out, bias=False
        )
        scalar_width = self.vector_hidden + scalar_in
        output_width = self.vector_out + scalar_out
        if use_mlp:
            self.scalar_projection = nn.Sequential(
                DenseLayer(scalar_width, scalar_in, activation=nn.SiLU()),
                DenseLayer(scalar_in, output_width),
            )
            self.vector_output_projection = DenseLayer(
                self.vector_out, self.vector_out, bias=False
            )
        else:
            self.scalar_projection = DenseLayer(scalar_width, output_width)
            self.vector_output_projection = None

    def forward(self, scalar: Tensor, vector: Tensor) -> tuple[Tensor, Tensor]:
        projected = self.vector_projection(vector)
        if self.vector_out:
            vector_norm_source, vector_out = projected.split(
                (self.vector_hidden, self.vector_out), dim=-1
            )
        else:
            vector_norm_source = projected
            vector_out = vector

        vector_norm = torch.clamp(
            vector_norm_source.square().sum(dim=1), min=self.eps
        ).sqrt()
        scalar_output = self.scalar_projection(torch.cat((scalar, vector_norm), dim=-1))
        if not self.vector_out:
            return scalar_output, vector_out

        gate, scalar_output = scalar_output.split((self.vector_out, self.scalar_out), dim=-1)
        vector_out = gate.unsqueeze(1) * vector_out
        if self.vector_output_projection is not None:
            vector_out = self.vector_output_projection(vector_out)
        return scalar_output, vector_out


class EQGATConv(nn.Module):
    """One residual scalar/vector EQGAT attention layer on edges ``j -> i``."""

    def __init__(
        self,
        scalar_dim: int,
        vector_dim: int,
        num_radial: int,
        cutoff: float,
        *,
        has_vector_input: bool,
        vector_aggr: str = "mean",
        eps: float = EQGAT_EPS,
    ) -> None:
        super().__init__()
        for value, name in (
            (scalar_dim, "scalar_dim"),
            (vector_dim, "vector_dim"),
            (num_radial, "num_radial"),
        ):
            _positive_int(value, name)
        if vector_aggr not in {"mean", "sum"}:
            raise ValueError("vector_aggr must be either 'mean' or 'sum'")

        self.scalar_dim = scalar_dim
        self.vector_dim = vector_dim
        self.has_vector_input = has_vector_input
        self.vector_aggr = vector_aggr
        self.rbf = BesselExpansion(cutoff, num_radial)
        self.cutoff = PolynomialCutoff(cutoff)
        self.scalar_projection = DenseLayer(scalar_dim, scalar_dim)
        if has_vector_input:
            self.vector_projection: nn.Module = DenseLayer(
                vector_dim, vector_dim, bias=False
            )
            vector_gate_count = 3
        else:
            self.vector_projection = nn.Identity()
            vector_gate_count = 1
        self.edge_network = nn.Sequential(
            DenseLayer(2 * scalar_dim + num_radial, scalar_dim, activation=nn.SiLU()),
            DenseLayer(scalar_dim, scalar_dim + vector_gate_count * vector_dim),
        )
        self.update = GatedEquivariantBlock(
            scalar_dim,
            vector_dim,
            scalar_dim,
            vector_dim,
            scalar_hidden=scalar_dim,
            vector_hidden=vector_dim,
            eps=eps,
            use_mlp=True,
        )

    def forward(
        self,
        scalar: Tensor,
        vector: Tensor,
        edge_index: Tensor,
        distances: Tensor,
        directions: Tensor,
    ) -> tuple[Tensor, Tensor]:
        _validate_layer_inputs(scalar, vector, edge_index, distances, directions)
        source, target = edge_index
        projected_scalar = self.scalar_projection(scalar)
        edge_features = torch.cat(
            (
                scalar[target],
                scalar[source],
                self.cutoff(distances).unsqueeze(-1) * self.rbf(distances),
            ),
            dim=-1,
        )
        attention_logits, vector_gates = self.edge_network(edge_features).split(
            (self.scalar_dim, self.edge_network[-1].out_features - self.scalar_dim),
            dim=-1,
        )
        attention = softmax(attention_logits, target, num_nodes=scalar.shape[0])
        scalar_messages = attention * projected_scalar[source]

        vector_gates = vector_gates.unsqueeze(1)
        if self.has_vector_input:
            direction_gate, vector_gate, cross_gate = vector_gates.split(
                self.vector_dim, dim=-1
            )
            direction_messages = directions.unsqueeze(-1) * direction_gate
            projected_vector = self.vector_projection(vector)
            assert isinstance(projected_vector, Tensor)
            vector_messages = (
                direction_messages
                + vector_gate * projected_vector[source]
                + cross_gate
                * torch.linalg.cross(vector[target], projected_vector[source], dim=1)
            )
        else:
            vector_messages = directions.unsqueeze(-1) * vector_gates

        scalar_update = scatter(
            scalar_messages,
            target,
            dim=0,
            dim_size=scalar.shape[0],
            reduce="sum",
        )
        vector_update = scatter(
            vector_messages,
            target,
            dim=0,
            dim_size=scalar.shape[0],
            reduce=self.vector_aggr,
        )
        scalar = scalar + scalar_update
        vector = vector + vector_update
        scalar_update, vector_update = self.update(scalar, vector)
        return scalar + scalar_update, vector + vector_update


def _validate_layer_inputs(
    scalar: Tensor,
    vector: Tensor,
    edge_index: Tensor,
    distances: Tensor,
    directions: Tensor,
) -> None:
    if scalar.ndim != 2 or vector.shape != (scalar.shape[0], 3, vector.shape[-1]):
        raise ValueError("scalar and vector must have shapes [N, Fs] and [N, 3, Fv]")
    if edge_index.shape != (2, distances.shape[0]):
        raise ValueError("edge_index and distances must agree on edge count")
    if directions.shape != (distances.shape[0], 3):
        raise ValueError("directions must have shape [E, 3]")


def _positive_int(value: int, field: str) -> None:
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise ValueError(f"{field} must be a positive integer")


__all__ = [
    "BesselExpansion",
    "DenseLayer",
    "EQGATConv",
    "GatedEquivariantBlock",
    "PolynomialCutoff",
]
