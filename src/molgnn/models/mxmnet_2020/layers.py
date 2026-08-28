"""Multiplex global/local message-passing modules for MXMNet."""

from __future__ import annotations

from itertools import pairwise

import torch
from torch import Tensor, nn
from torch_geometric.utils import scatter


class DenseMLP(nn.Module):
    """Source-compatible affine-plus-SiLU stack."""

    def __init__(self, channels: tuple[int, ...]) -> None:
        super().__init__()
        if len(channels) < 2 or any(value < 1 for value in channels):
            raise ValueError("channels must contain at least two positive widths")
        operations: list[nn.Module] = []
        for input_dim, output_dim in pairwise(channels):
            operations.extend((nn.Linear(input_dim, output_dim), nn.SiLU()))
        self.layers = nn.Sequential(*operations)

    def forward(self, values: Tensor) -> Tensor:
        return self.layers(values)


class ResidualUnit(nn.Module):
    """Two-affine SiLU residual block used throughout the author model."""

    def __init__(self, hidden_dim: int) -> None:
        super().__init__()
        self.mlp = DenseMLP((hidden_dim, hidden_dim, hidden_dim))

    def forward(self, values: Tensor) -> Tensor:
        return values + self.mlp(values)


class GlobalMessagePassing(nn.Module):
    """Two distance-filtered global passes with the source update ordering."""

    def __init__(self, hidden_dim: int) -> None:
        super().__init__()
        _positive_int(hidden_dim, "hidden_dim")
        self.hidden_dim = hidden_dim
        self.cross_layer = DenseMLP((hidden_dim, hidden_dim))
        self.residual1 = ResidualUnit(hidden_dim)
        self.residual2 = ResidualUnit(hidden_dim)
        self.residual3 = ResidualUnit(hidden_dim)
        self.update = DenseMLP((hidden_dim, hidden_dim))
        self.edge_mlp = DenseMLP((3 * hidden_dim, hidden_dim))
        self.radial_linear = nn.Linear(hidden_dim, hidden_dim, bias=False)

    def forward(self, atom: Tensor, radial: Tensor, edge_index: Tensor) -> Tensor:
        _validate_hidden(atom, self.hidden_dim, "atom")
        _validate_edge_features(radial, edge_index, self.hidden_dim, atom)
        residual = atom
        atom = self.cross_layer(atom)
        atom = self._propagate(atom, radial, edge_index)
        atom = self.residual1(atom)
        atom = self.update(atom) + residual
        atom = self.residual2(atom)
        atom = self.residual3(atom)
        return self._propagate(atom, radial, edge_index)

    def _propagate(self, atom: Tensor, radial: Tensor, edge_index: Tensor) -> Tensor:
        source, target = edge_index
        edge_hidden = self.edge_mlp(
            torch.cat((atom[target], atom[source], radial), dim=-1)
        )
        messages = self.radial_linear(radial) * edge_hidden
        # The official code appends self-loops after all real edges.  Writing
        # the same contribution explicitly removes ordering assumptions.
        return atom + scatter(
            messages,
            target,
            dim=0,
            dim_size=atom.shape[0],
            reduce="sum",
        )


class LocalMessagePassing(nn.Module):
    """Two directed-angle edge updates followed by an atom/output update."""

    def __init__(self, hidden_dim: int, num_targets: int) -> None:
        super().__init__()
        _positive_int(hidden_dim, "hidden_dim")
        _positive_int(num_targets, "num_targets")
        self.hidden_dim = hidden_dim
        self.num_targets = num_targets
        self.cross_layer = DenseMLP((hidden_dim, hidden_dim))
        self.edge_incoming = DenseMLP((3 * hidden_dim, hidden_dim))
        self.edge_target1 = DenseMLP((3 * hidden_dim, hidden_dim))
        self.edge_target2 = DenseMLP((hidden_dim, hidden_dim))
        self.edge_sibling = DenseMLP((hidden_dim, hidden_dim))
        self.spherical1 = DenseMLP((hidden_dim, hidden_dim, hidden_dim))
        self.spherical2 = DenseMLP((hidden_dim, hidden_dim, hidden_dim))
        self.radial1 = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.radial2 = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.radial_output = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.residual1 = ResidualUnit(hidden_dim)
        self.residual2 = ResidualUnit(hidden_dim)
        self.residual3 = ResidualUnit(hidden_dim)
        self.update = DenseMLP((hidden_dim, hidden_dim))
        self.output_mlp = DenseMLP((hidden_dim, hidden_dim, hidden_dim, hidden_dim))
        self.output_linear = nn.Linear(hidden_dim, num_targets)

    def forward(
        self,
        atom: Tensor,
        radial: Tensor,
        spherical_two_hop: Tensor,
        spherical_one_hop: Tensor,
        two_hop_edge_index: Tensor,
        one_hop_edge_index: Tensor,
        edge_index: Tensor,
    ) -> tuple[Tensor, Tensor]:
        _validate_hidden(atom, self.hidden_dim, "atom")
        edge_count = _validate_edge_features(radial, edge_index, self.hidden_dim, atom)
        _validate_pair_features(
            spherical_two_hop,
            two_hop_edge_index,
            edge_count,
            self.hidden_dim,
            atom,
            "spherical_two_hop",
        )
        _validate_pair_features(
            spherical_one_hop,
            one_hop_edge_index,
            edge_count,
            self.hidden_dim,
            atom,
            "spherical_one_hop",
        )

        residual = atom
        atom = self.cross_layer(atom)
        source, target = edge_index
        edge = torch.cat((atom[target], atom[source], radial), dim=-1)

        incoming_edge, base_edge = two_hop_edge_index
        incoming = self.edge_incoming(edge) * self.radial1(radial)
        incoming = incoming[incoming_edge] * self.spherical1(spherical_two_hop)
        incoming = scatter(
            incoming,
            base_edge,
            dim=0,
            dim_size=edge_count,
            reduce="sum",
        )
        edge = self.edge_target1(edge) + incoming

        sibling_edge, base_edge = one_hop_edge_index
        sibling = self.edge_sibling(edge) * self.radial2(radial)
        sibling = sibling[sibling_edge] * self.spherical2(spherical_one_hop)
        sibling = scatter(
            sibling,
            base_edge,
            dim=0,
            dim_size=edge_count,
            reduce="sum",
        )
        edge = self.edge_target2(edge) + sibling

        atom = scatter(
            self.radial_output(radial) * edge,
            target,
            dim=0,
            dim_size=atom.shape[0],
            reduce="sum",
        )
        atom = self.residual1(atom)
        atom = self.update(atom) + residual
        atom = self.residual2(atom)
        atom = self.residual3(atom)
        output = self.output_linear(self.output_mlp(atom))
        return atom, output


def _validate_hidden(values: Tensor, width: int, name: str) -> None:
    if (
        not isinstance(values, Tensor)
        or values.ndim != 2
        or values.shape[1] != width
        or not torch.is_floating_point(values)
        or not bool(torch.isfinite(values).all())
    ):
        raise ValueError(f"{name} must have shape [N, {width}] and be finite floating")


def _validate_edge_features(
    radial: Tensor, edge_index: Tensor, width: int, reference: Tensor
) -> int:
    if (
        not isinstance(edge_index, Tensor)
        or edge_index.ndim != 2
        or edge_index.shape[0] != 2
        or edge_index.dtype != torch.long
    ):
        raise ValueError("edge_index must have shape [2, E] and dtype torch.long")
    edge_count = edge_index.shape[1]
    if (
        not isinstance(radial, Tensor)
        or radial.shape != (edge_count, width)
        or not torch.is_floating_point(radial)
        or radial.device != reference.device
        or edge_index.device != reference.device
        or not bool(torch.isfinite(radial).all())
    ):
        raise ValueError(
            f"radial must have shape [{edge_count}, {width}] and be finite"
        )
    return edge_count


def _validate_pair_features(
    spherical: Tensor,
    pair_index: Tensor,
    edge_count: int,
    width: int,
    reference: Tensor,
    name: str,
) -> None:
    if (
        not isinstance(pair_index, Tensor)
        or pair_index.ndim != 2
        or pair_index.shape[0] != 2
        or pair_index.dtype != torch.long
    ):
        raise ValueError(f"{name} edge table must have shape [2, Q] and be long")
    pair_count = pair_index.shape[1]
    if (
        not isinstance(spherical, Tensor)
        or spherical.shape != (pair_count, width)
        or not torch.is_floating_point(spherical)
        or spherical.device != reference.device
        or pair_index.device != reference.device
        or not bool(torch.isfinite(spherical).all())
    ):
        raise ValueError(
            f"{name} must have shape [{pair_count}, {width}] and be finite"
        )
    if pair_count and (pair_index.min() < 0 or pair_index.max() >= edge_count):
        raise ValueError(f"{name} edge table contains an invalid edge ID")


def _positive_int(value: int, name: str) -> None:
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise ValueError(f"{name} must be a positive integer")


__all__ = [
    "DenseMLP",
    "GlobalMessagePassing",
    "LocalMessagePassing",
    "ResidualUnit",
]
