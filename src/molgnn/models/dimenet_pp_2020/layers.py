"""Directed-edge neural blocks for the DimeNet++ architecture."""

from __future__ import annotations

import torch
from torch import Tensor, nn
from torch_geometric.utils import scatter

from ..dimenet_2020.layers import (
    ResidualLayer,
    _nonnegative_int,
    _positive_int,
    _validate_features,
    _validate_hidden,
    _validate_triplet_indices,
    glorot_orthogonal_,
)


class InteractionPPBlock(nn.Module):
    """One directional ``k -> j -> i`` interaction/update block for DimeNet++.

    DimeNet++ replaces the bilinear interaction tensor with elementwise
    Hadamard products and basis MLPs.  Target edge RBF is gathered using
    ``idx_ji`` (target distance ``d_ji``) while SBF uses incoming edge distance
    ``d_kj`` and directed angle between incoming and target vectors.
    """

    def __init__(
        self,
        hidden_dim: int,
        interaction_dim: int,
        basis_dim: int,
        num_spherical: int,
        num_radial: int,
        *,
        num_before_skip: int = 1,
        num_after_skip: int = 2,
    ) -> None:
        super().__init__()
        for value, name in (
            (hidden_dim, "hidden_dim"),
            (interaction_dim, "interaction_dim"),
            (basis_dim, "basis_dim"),
            (num_spherical, "num_spherical"),
            (num_radial, "num_radial"),
        ):
            _positive_int(value, name)
        _nonnegative_int(num_before_skip, "num_before_skip")
        _nonnegative_int(num_after_skip, "num_after_skip")

        self.hidden_dim = hidden_dim
        self.interaction_dim = interaction_dim
        self.basis_dim = basis_dim
        self.num_spherical = num_spherical
        self.num_radial = num_radial
        self.num_before_skip = num_before_skip
        self.num_after_skip = num_after_skip

        # Transformations of Bessel and spherical basis representations (bias-free, no activation)
        self.dense_rbf1 = nn.Linear(num_radial, basis_dim, bias=False)
        self.dense_rbf2 = nn.Linear(basis_dim, hidden_dim, bias=False)
        self.dense_sbf1 = nn.Linear(
            num_spherical * num_radial, basis_dim, bias=False
        )
        self.dense_sbf2 = nn.Linear(basis_dim, interaction_dim, bias=False)

        # Dense transformations of input messages
        self.dense_ji = nn.Linear(hidden_dim, hidden_dim)
        self.dense_kj = nn.Linear(hidden_dim, hidden_dim)

        # Projections for triplet interactions
        self.down_projection = nn.Linear(hidden_dim, interaction_dim, bias=False)
        self.up_projection = nn.Linear(interaction_dim, hidden_dim, bias=False)

        # Residual layers before and after skip connection
        self.before_skip = nn.ModuleList(
            ResidualLayer(hidden_dim) for _ in range(num_before_skip)
        )
        self.final_projection = nn.Linear(hidden_dim, hidden_dim)
        self.after_skip = nn.ModuleList(
            ResidualLayer(hidden_dim) for _ in range(num_after_skip)
        )
        self.activation = nn.SiLU()
        self.reset_parameters()

    def reset_parameters(self) -> None:
        glorot_orthogonal_(self.dense_rbf1.weight)
        glorot_orthogonal_(self.dense_rbf2.weight)
        glorot_orthogonal_(self.dense_sbf1.weight)
        glorot_orthogonal_(self.dense_sbf2.weight)
        glorot_orthogonal_(self.dense_ji.weight)
        nn.init.zeros_(self.dense_ji.bias)
        glorot_orthogonal_(self.dense_kj.weight)
        nn.init.zeros_(self.dense_kj.bias)
        glorot_orthogonal_(self.down_projection.weight)
        glorot_orthogonal_(self.up_projection.weight)
        for layer in self.before_skip:
            layer.reset_parameters()
        glorot_orthogonal_(self.final_projection.weight)
        nn.init.zeros_(self.final_projection.bias)
        for layer in self.after_skip:
            layer.reset_parameters()

    def forward(
        self,
        messages: Tensor,
        target_rbf: Tensor,
        sbf: Tensor,
        idx_kj: Tensor,
        idx_ji: Tensor,
    ) -> Tensor:
        """Update directed-edge messages from triplets using DimeNet++ interaction."""
        _validate_hidden(messages, self.hidden_dim, "messages")
        edge_count = messages.shape[0]
        _validate_features(
            target_rbf, edge_count, self.num_radial, "target_rbf", messages
        )
        triplet_count = _validate_triplet_indices(
            idx_kj, idx_ji, edge_count, messages.device
        )
        _validate_features(
            sbf,
            triplet_count,
            self.num_spherical * self.num_radial,
            "sbf",
            messages,
        )

        m_ji = self.activation(self.dense_ji(messages))
        m_kj = self.activation(self.dense_kj(messages))
        rbf_transformed = self.dense_rbf2(self.dense_rbf1(target_rbf))
        sbf_transformed = self.dense_sbf2(self.dense_sbf1(sbf))

        if idx_kj.numel() > 0:
            m_kj_triplet = m_kj[idx_kj]
            rbf_triplet = rbf_transformed[idx_ji]
            triplet_h = m_kj_triplet * rbf_triplet
            triplet_h = self.activation(self.down_projection(triplet_h))
            triplet_h = triplet_h * sbf_transformed
            aggregated = scatter(
                triplet_h,
                idx_ji,
                dim=0,
                dim_size=edge_count,
                reduce="sum",
            )
        else:
            aggregated = messages.new_zeros((edge_count, self.interaction_dim))

        aggregated = self.activation(self.up_projection(aggregated))
        hidden = m_ji + aggregated
        for layer in self.before_skip:
            hidden = layer(hidden)
        hidden = self.activation(self.final_projection(hidden)) + messages
        for layer in self.after_skip:
            hidden = layer(hidden)
        return hidden


class OutputPPBlock(nn.Module):
    """Convert directed messages into additive atom-wise target contributions."""

    def __init__(
        self,
        num_radial: int,
        hidden_dim: int,
        output_dim: int,
        num_targets: int,
        *,
        num_dense_output: int = 3,
        output_initializer: str = "zeros",
    ) -> None:
        super().__init__()
        for value, name in (
            (num_radial, "num_radial"),
            (hidden_dim, "hidden_dim"),
            (output_dim, "output_dim"),
            (num_targets, "num_targets"),
        ):
            _positive_int(value, name)
        _nonnegative_int(num_dense_output, "num_dense_output")
        if output_initializer not in {"zeros", "glorot_orthogonal"}:
            raise ValueError(
                "output_initializer must be 'zeros' or 'glorot_orthogonal'"
            )
        self.num_radial = num_radial
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.num_targets = num_targets
        self.num_dense_output = num_dense_output
        self.output_initializer = output_initializer

        self.radial_projection = nn.Linear(num_radial, hidden_dim, bias=False)
        self.up_projection = nn.Linear(hidden_dim, output_dim, bias=False)
        self.dense_layers = nn.ModuleList(
            nn.Linear(output_dim, output_dim) for _ in range(num_dense_output)
        )
        self.output_projection = nn.Linear(output_dim, num_targets, bias=False)
        self.activation = nn.SiLU()
        self.reset_parameters()

    def reset_parameters(self) -> None:
        glorot_orthogonal_(self.radial_projection.weight)
        glorot_orthogonal_(self.up_projection.weight)
        for layer in self.dense_layers:
            glorot_orthogonal_(layer.weight)
            nn.init.zeros_(layer.bias)
        if self.output_initializer == "zeros":
            nn.init.zeros_(self.output_projection.weight)
        else:
            glorot_orthogonal_(self.output_projection.weight)

    def forward(
        self,
        messages: Tensor,
        rbf: Tensor,
        target: Tensor,
        num_nodes: int,
    ) -> Tensor:
        """Return ``[N, T]`` additive values after summing incoming edges."""
        _validate_hidden(messages, self.hidden_dim, "messages")
        edge_count = messages.shape[0]
        _validate_features(rbf, edge_count, self.num_radial, "rbf", messages)
        if (
            not isinstance(target, Tensor)
            or target.ndim != 1
            or target.dtype != torch.long
            or target.shape[0] != edge_count
        ):
            raise ValueError("target must have shape [E] and dtype torch.long")
        if target.device != messages.device:
            raise ValueError("target and messages must share a device")
        _positive_int(num_nodes, "num_nodes")
        if target.numel() and (target.min() < 0 or target.max() >= num_nodes):
            raise ValueError("target contains a node index outside [0, num_nodes)")

        values = self.radial_projection(rbf) * messages
        values = scatter(values, target, dim=0, dim_size=num_nodes, reduce="sum")
        values = self.up_projection(values)
        for layer in self.dense_layers:
            values = self.activation(layer(values))
        return self.output_projection(values)


__all__ = [
    "InteractionPPBlock",
    "OutputPPBlock",
]
