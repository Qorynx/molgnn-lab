"""Independent PyTorch implementation of GemNet's neural building blocks."""

from __future__ import annotations

import math
from collections.abc import Mapping

import torch
from torch import Tensor, nn
from torch_geometric.utils import scatter


class ScaledSiLU(nn.Module):
    def forward(self, values: Tensor) -> Tensor:
        return torch.nn.functional.silu(values) / 0.6


class Dense(nn.Module):
    def __init__(
        self,
        in_features: int,
        out_features: int,
        *,
        activation: bool = False,
        bias: bool = False,
    ) -> None:
        super().__init__()
        self.linear = nn.Linear(in_features, out_features, bias=bias)
        self.activation = ScaledSiLU() if activation else nn.Identity()
        self.reset_parameters()

    def reset_parameters(self) -> None:
        he_orthogonal_(self.linear.weight)
        if self.linear.bias is not None:
            nn.init.zeros_(self.linear.bias)

    def forward(self, values: Tensor) -> Tensor:
        return self.activation(self.linear(values))


class ResidualLayer(nn.Module):
    def __init__(self, hidden_dim: int) -> None:
        super().__init__()
        self.first = Dense(hidden_dim, hidden_dim, activation=True)
        self.second = Dense(hidden_dim, hidden_dim, activation=True)

    def forward(self, values: Tensor) -> Tensor:
        return (values + self.second(self.first(values))) / math.sqrt(2)


class ScaleFactor(nn.Module):
    """Fixed architecture scale; unit is the explicit uncalibrated fallback."""

    def __init__(self, value: float = 1.0) -> None:
        super().__init__()
        if not math.isfinite(float(value)) or value <= 0:
            raise ValueError("scale factor must be finite and positive")
        self.register_buffer("value", torch.tensor(float(value)))

    def forward(self, values: Tensor) -> Tensor:
        return values * self.value.to(dtype=values.dtype)


class BasisDownProjection(nn.Module):
    """Project radial order ``n`` before grouped angular aggregation."""

    def __init__(self, num_orders: int, num_radial: int, output_dim: int) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.empty(num_orders, num_radial, output_dim))
        self.reset_parameters()

    def reset_parameters(self) -> None:
        he_orthogonal_(self.weight)

    def forward(self, radial: Tensor) -> Tensor:
        if radial.ndim != 3 or radial.shape[1:3] != self.weight.shape[:2]:
            raise ValueError("radial basis shape does not match the down projection")
        return torch.einsum("elr,lrd->edl", radial, self.weight)


class EfficientBilinear(nn.Module):
    """Grouped form of GemNet's basis-conditioned bilinear aggregation."""

    def __init__(self, message_dim: int, basis_dim: int, output_dim: int) -> None:
        super().__init__()
        self.message_dim = message_dim
        self.basis_dim = basis_dim
        self.output_dim = output_dim
        self.weight = nn.Parameter(torch.empty(message_dim, basis_dim, output_dim))
        self.reset_parameters()

    def reset_parameters(self) -> None:
        he_orthogonal_(self.weight)

    def forward(
        self,
        radial_down: Tensor,
        angular: Tensor,
        incoming: Tensor,
        reduce_edge: Tensor,
    ) -> Tensor:
        edge_count, basis_dim, order_count = radial_down.shape
        pair_count = reduce_edge.shape[0]
        if basis_dim != self.basis_dim:
            raise ValueError("radial down-projection width does not match bilinear basis_dim")
        if angular.shape != (pair_count, order_count):
            raise ValueError("angular basis has an invalid grouped shape")
        if incoming.shape != (pair_count, self.message_dim):
            raise ValueError("incoming messages have an invalid grouped shape")
        if reduce_edge.dtype != torch.long or reduce_edge.ndim != 1:
            raise ValueError("reduce_edge must be a one-dimensional long tensor")
        if pair_count == 0:
            return radial_down.new_zeros((edge_count, self.output_dim))
        if reduce_edge.min() < 0 or reduce_edge.max() >= edge_count:
            raise ValueError("reduce_edge contains an invalid edge index")
        if bool((reduce_edge[1:] < reduce_edge[:-1]).any()):
            raise ValueError("reduce_edge must be sorted for grouped GemNet aggregation")

        counts = torch.bincount(reduce_edge, minlength=edge_count)
        max_count = int(counts.max().item())
        starts = torch.cumsum(counts, dim=0) - counts
        slots = torch.arange(pair_count, device=reduce_edge.device) - starts[reduce_edge]
        angular_dense = angular.new_zeros((edge_count, order_count, max_count))
        message_dense = incoming.new_zeros((edge_count, max_count, self.message_dim))
        angular_dense[reduce_edge, :, slots] = angular
        message_dense[reduce_edge, slots] = incoming

        angular_messages = torch.bmm(angular_dense, message_dense)
        combined = torch.bmm(radial_down, angular_messages)
        return torch.einsum("edf,fdo->eo", combined, self.weight)


class EdgeEmbedding(nn.Module):
    def __init__(self, atom_dim: int, edge_input_dim: int, edge_dim: int) -> None:
        super().__init__()
        self.dense = Dense(2 * atom_dim + edge_input_dim, edge_dim, activation=True)

    def forward(
        self, atoms: Tensor, edge_values: Tensor, source: Tensor, target: Tensor
    ) -> Tensor:
        return self.dense(torch.cat((atoms[target], atoms[source], edge_values), dim=-1))


class TripletInteraction(nn.Module):
    def __init__(
        self,
        edge_dim: int,
        triplet_dim: int,
        rbf_dim: int,
        cbf_dim: int,
        bilinear_dim: int,
        *,
        scales: Mapping[str, float] | None = None,
        prefix: str,
    ) -> None:
        super().__init__()
        self.edge_dense = Dense(edge_dim, edge_dim, activation=True)
        self.rbf_up = Dense(rbf_dim, edge_dim)
        self.rbf_scale = ScaleFactor(_scale(scales, f"{prefix}.rbf"))
        self.down = Dense(edge_dim, triplet_dim, activation=True)
        self.bilinear = EfficientBilinear(triplet_dim, cbf_dim, bilinear_dim)
        self.sum_scale = ScaleFactor(_scale(scales, f"{prefix}.sum"))
        self.up_ca = Dense(bilinear_dim, edge_dim, activation=True)
        self.up_ac = Dense(bilinear_dim, edge_dim, activation=True)

    def forward(
        self,
        messages: Tensor,
        rbf: Tensor,
        radial_down: Tensor,
        angular: Tensor,
        reduce_edge: Tensor,
        expand_edge: Tensor,
        reverse_edge: Tensor,
    ) -> Tensor:
        filtered = self.edge_dense(messages)
        filtered = self.rbf_scale(filtered * self.rbf_up(rbf))
        incoming = self.down(filtered)[expand_edge]
        aggregated = self.sum_scale(
            self.bilinear(radial_down, angular, incoming, reduce_edge)
        )
        return (self.up_ca(aggregated) + self.up_ac(aggregated)[reverse_edge]) / math.sqrt(2)


class QuadrupletInteraction(nn.Module):
    def __init__(
        self,
        edge_dim: int,
        quadruplet_dim: int,
        rbf_dim: int,
        cbf_dim: int,
        sbf_dim: int,
        bilinear_dim: int,
        *,
        scales: Mapping[str, float] | None = None,
        prefix: str,
    ) -> None:
        super().__init__()
        self.edge_dense = Dense(edge_dim, edge_dim, activation=True)
        self.rbf_up = Dense(rbf_dim, edge_dim)
        self.rbf_scale = ScaleFactor(_scale(scales, f"{prefix}.rbf"))
        self.down = Dense(edge_dim, quadruplet_dim, activation=True)
        self.cbf_up = Dense(cbf_dim, quadruplet_dim)
        self.cbf_scale = ScaleFactor(_scale(scales, f"{prefix}.cbf"))
        self.bilinear = EfficientBilinear(quadruplet_dim, sbf_dim, bilinear_dim)
        self.sum_scale = ScaleFactor(_scale(scales, f"{prefix}.sum"))
        self.up_ca = Dense(bilinear_dim, edge_dim, activation=True)
        self.up_ac = Dense(bilinear_dim, edge_dim, activation=True)

    def forward(
        self,
        messages: Tensor,
        rbf: Tensor,
        cbf: Tensor,
        radial_down: Tensor,
        angular: Tensor,
        reduce_edge: Tensor,
        expand_edge: Tensor,
        reverse_edge: Tensor,
    ) -> Tensor:
        filtered = self.edge_dense(messages)
        filtered = self.rbf_scale(filtered * self.rbf_up(rbf))
        incoming = self.down(filtered)[expand_edge]
        incoming = self.cbf_scale(incoming * self.cbf_up(cbf))
        aggregated = self.sum_scale(
            self.bilinear(radial_down, angular, incoming, reduce_edge)
        )
        return (self.up_ca(aggregated) + self.up_ac(aggregated)[reverse_edge]) / math.sqrt(2)


class AtomUpdate(nn.Module):
    def __init__(
        self,
        atom_dim: int,
        edge_dim: int,
        rbf_dim: int,
        num_residual: int,
        *,
        scale: float,
    ) -> None:
        super().__init__()
        self.rbf_up = Dense(rbf_dim, edge_dim)
        self.sum_scale = ScaleFactor(scale)
        self.input = Dense(edge_dim, atom_dim, activation=True)
        self.residual = nn.ModuleList(ResidualLayer(atom_dim) for _ in range(num_residual))

    def forward(
        self, atoms: Tensor, messages: Tensor, rbf: Tensor, target: Tensor
    ) -> Tensor:
        values = messages * self.rbf_up(rbf)
        values = scatter(values, target, dim=0, dim_size=atoms.shape[0], reduce="sum")
        values = self.input(self.sum_scale(values))
        for layer in self.residual:
            values = layer(values)
        return values


class OutputBlock(nn.Module):
    def __init__(
        self,
        atom_dim: int,
        edge_dim: int,
        rbf_dim: int,
        num_targets: int,
        num_residual: int,
        *,
        scale: float,
        output_initializer: str,
    ) -> None:
        super().__init__()
        self.rbf_up = Dense(rbf_dim, edge_dim)
        self.sum_scale = ScaleFactor(scale)
        self.input = Dense(edge_dim, atom_dim, activation=True)
        self.residual = nn.ModuleList(ResidualLayer(atom_dim) for _ in range(num_residual))
        self.output = Dense(atom_dim, num_targets)
        if output_initializer == "zeros":
            nn.init.zeros_(self.output.linear.weight)
        elif output_initializer != "he_orthogonal":
            raise ValueError("output_initializer must be 'he_orthogonal' or 'zeros'")

    def forward(
        self, messages: Tensor, rbf: Tensor, target: Tensor, num_nodes: int
    ) -> Tensor:
        values = messages * self.rbf_up(rbf)
        values = scatter(values, target, dim=0, dim_size=num_nodes, reduce="sum")
        values = self.input(self.sum_scale(values))
        for layer in self.residual:
            values = layer(values)
        return self.output(values)


class InteractionBlock(nn.Module):
    def __init__(
        self,
        atom_dim: int,
        edge_dim: int,
        triplet_dim: int,
        quadruplet_dim: int,
        rbf_dim: int,
        cbf_dim: int,
        sbf_dim: int,
        bilinear_triplet_dim: int,
        bilinear_quadruplet_dim: int,
        num_before_skip: int,
        num_after_skip: int,
        num_concat: int,
        num_atom: int,
        *,
        use_quadruplets: bool,
        scales: Mapping[str, float] | None,
        prefix: str,
    ) -> None:
        super().__init__()
        self.use_quadruplets = use_quadruplets
        self.skip = Dense(edge_dim, edge_dim, activation=True)
        self.triplet = TripletInteraction(
            edge_dim,
            triplet_dim,
            rbf_dim,
            cbf_dim,
            bilinear_triplet_dim,
            scales=scales,
            prefix=f"{prefix}.triplet",
        )
        self.quadruplet = (
            QuadrupletInteraction(
                edge_dim,
                quadruplet_dim,
                rbf_dim,
                cbf_dim,
                sbf_dim,
                bilinear_quadruplet_dim,
                scales=scales,
                prefix=f"{prefix}.quadruplet",
            )
            if use_quadruplets
            else None
        )
        self.before_skip = nn.ModuleList(
            ResidualLayer(edge_dim) for _ in range(num_before_skip)
        )
        self.after_skip = nn.ModuleList(
            ResidualLayer(edge_dim) for _ in range(num_after_skip)
        )
        self.atom_update = AtomUpdate(
            atom_dim,
            edge_dim,
            rbf_dim,
            num_atom,
            scale=_scale(scales, f"{prefix}.atom_sum"),
        )
        self.concat = EdgeEmbedding(atom_dim, edge_dim, edge_dim)
        self.concat_residual = nn.ModuleList(
            ResidualLayer(edge_dim) for _ in range(num_concat)
        )

    def forward(
        self,
        atoms: Tensor,
        messages: Tensor,
        source: Tensor,
        target: Tensor,
        reverse_edge: Tensor,
        rbf_triplet: Tensor,
        triplet_radial: Tensor,
        triplet_angular: Tensor,
        triplet_reduce: Tensor,
        triplet_expand: Tensor,
        rbf_atom: Tensor,
        *,
        rbf_quadruplet: Tensor | None = None,
        quadruplet_cbf: Tensor | None = None,
        quadruplet_radial: Tensor | None = None,
        quadruplet_angular: Tensor | None = None,
        quadruplet_reduce: Tensor | None = None,
        quadruplet_expand: Tensor | None = None,
    ) -> tuple[Tensor, Tensor]:
        skip = self.skip(messages)
        triplet = self.triplet(
            messages,
            rbf_triplet,
            triplet_radial,
            triplet_angular,
            triplet_reduce,
            triplet_expand,
            reverse_edge,
        )
        if self.use_quadruplets:
            if any(
                value is None
                for value in (
                    rbf_quadruplet,
                    quadruplet_cbf,
                    quadruplet_radial,
                    quadruplet_angular,
                    quadruplet_reduce,
                    quadruplet_expand,
                )
            ):
                raise ValueError("GemNet-Q interaction is missing quadruplet basis data")
            assert self.quadruplet is not None
            assert rbf_quadruplet is not None
            assert quadruplet_cbf is not None
            assert quadruplet_radial is not None
            assert quadruplet_angular is not None
            assert quadruplet_reduce is not None
            assert quadruplet_expand is not None
            quadruplet = self.quadruplet(
                messages,
                rbf_quadruplet,
                quadruplet_cbf,
                quadruplet_radial,
                quadruplet_angular,
                quadruplet_reduce,
                quadruplet_expand,
                reverse_edge,
            )
            update = (skip + triplet + quadruplet) / math.sqrt(3)
        else:
            update = (skip + triplet) / math.sqrt(2)

        for layer in self.before_skip:
            update = layer(update)
        messages = (messages + update) / math.sqrt(2)
        for layer in self.after_skip:
            messages = layer(messages)

        atoms = (atoms + self.atom_update(atoms, messages, rbf_atom, target)) / math.sqrt(2)
        edge_update = self.concat(atoms, messages, source, target)
        for layer in self.concat_residual:
            edge_update = layer(edge_update)
        messages = (messages + edge_update) / math.sqrt(2)
        return atoms, messages


@torch.no_grad()
def he_orthogonal_(tensor: Tensor) -> None:
    if tensor.ndim < 2:
        raise ValueError("GemNet weights must have at least two dimensions")
    if tensor.ndim == 2:
        nn.init.orthogonal_(tensor)
        dimensions = (1,)
        fan_in = tensor.shape[1]
    else:
        nn.init.normal_(tensor)
        dimensions = tuple(range(tensor.ndim - 1))
        fan_in = math.prod(tensor.shape[:-1])
    mean = tensor.mean(dim=dimensions, keepdim=True)
    variance = tensor.var(dim=dimensions, unbiased=False, keepdim=True)
    tensor.copy_((tensor - mean) / torch.sqrt(variance + 1.0e-6) / math.sqrt(fan_in))


def _scale(scales: Mapping[str, float] | None, name: str) -> float:
    return 1.0 if scales is None else float(scales.get(name, 1.0))


__all__ = [
    "BasisDownProjection",
    "Dense",
    "EdgeEmbedding",
    "EfficientBilinear",
    "InteractionBlock",
    "OutputBlock",
    "ResidualLayer",
    "ScaleFactor",
    "ScaledSiLU",
    "he_orthogonal_",
]
