"""Heterogeneous order interaction and fusion layers for HMGNN."""

from __future__ import annotations

import math

import torch
from torch import Tensor, nn
from torch.nn import functional
from torch_geometric.utils import scatter


class ShiftedSoftplus(nn.Module):
    def forward(self, x: Tensor) -> Tensor:
        return functional.softplus(x) - math.log(2.0)


class Dense(nn.Module):
    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        *,
        activation: nn.Module | None = None,
        bias: bool = True,
    ) -> None:
        super().__init__()
        self.linear = nn.Linear(input_dim, output_dim, bias=bias)
        self.activation = activation
        glorot_orthogonal_(self.linear.weight)
        if self.linear.bias is not None:
            nn.init.zeros_(self.linear.bias)

    def forward(self, x: Tensor) -> Tensor:
        x = self.linear(x)
        return self.activation(x) if self.activation is not None else x


class ResidualMLP(nn.Module):
    def __init__(self, hidden_dim: int, depth: int) -> None:
        super().__init__()
        self.layers = nn.ModuleList(
            Dense(hidden_dim, hidden_dim, activation=ShiftedSoftplus())
            for _ in range(depth)
        )

    def forward(self, x: Tensor) -> Tensor:
        residual = x
        for layer in self.layers:
            x = layer(x)
        return residual + x


class OrderInteraction(nn.Module):
    """One same-order gated update plus one cross-order incidence update."""

    def __init__(self, hidden_dim: int, num_residual: int) -> None:
        super().__init__()
        self.source_projection = Dense(
            hidden_dim, hidden_dim, activation=ShiftedSoftplus()
        )
        self.edge_gate = Dense(hidden_dim, hidden_dim, bias=False)
        self.foreign_projection = Dense(
            hidden_dim, hidden_dim, activation=ShiftedSoftplus()
        )
        self.destination_projection = Dense(
            hidden_dim, hidden_dim, activation=ShiftedSoftplus()
        )
        self.mixing = Dense(3 * hidden_dim, hidden_dim, activation=ShiftedSoftplus())
        self.interaction_residuals = nn.ModuleList(
            ResidualMLP(hidden_dim, 2) for _ in range(num_residual)
        )
        self.output_residuals = nn.ModuleList(
            ResidualMLP(hidden_dim, 2) for _ in range(num_residual)
        )

    def forward(
        self,
        states: Tensor,
        same_edge_index: Tensor,
        same_edge_features: Tensor,
        foreign_messages: Tensor,
    ) -> Tensor:
        source, target = same_edge_index
        messages = self.source_projection(states[source]) * self.edge_gate(
            same_edge_features
        )
        domestic = scatter(
            messages, target, dim=0, dim_size=states.shape[0], reduce="sum"
        )
        update = self.mixing(
            torch.cat((self.destination_projection(states), domestic, foreign_messages), dim=-1)
        )
        for residual in self.interaction_residuals:
            update = residual(update)
        states = states + update
        for residual in self.output_residuals:
            states = residual(states)
        return states


class HeterogeneousInteractionBlock(nn.Module):
    """Simultaneously update atom and two-body representations."""

    def __init__(self, hidden_dim: int, num_residual: int) -> None:
        super().__init__()
        self.atom_interaction = OrderInteraction(hidden_dim, num_residual)
        self.body_interaction = OrderInteraction(hidden_dim, num_residual)

    def forward(
        self,
        atom_states: Tensor,
        body_states: Tensor,
        atom_edge_index: Tensor,
        atom_edge_features: Tensor,
        body_atom_index: Tensor,
        body_edge_index: Tensor,
        body_edge_features: Tensor,
    ) -> tuple[Tensor, Tensor]:
        atom_foreign = scatter(
            self.atom_interaction.foreign_projection(body_states).repeat_interleave(2, dim=0),
            body_atom_index.t().reshape(-1),
            dim=0,
            dim_size=atom_states.shape[0],
            reduce="sum",
        )
        body_foreign = scatter(
            self.body_interaction.foreign_projection(
                atom_states[body_atom_index.reshape(-1)]
            ),
            torch.arange(
                body_states.shape[0], device=body_states.device, dtype=torch.long
            ).repeat(2),
            dim=0,
            dim_size=body_states.shape[0],
            reduce="sum",
        )
        new_atoms = self.atom_interaction(
            atom_states,
            atom_edge_index,
            atom_edge_features,
            atom_foreign,
        )
        new_bodies = self.body_interaction(
            body_states,
            body_edge_index,
            body_edge_features,
            body_foreign,
        )
        return new_atoms, new_bodies
class SafeBatchNorm1d(nn.BatchNorm1d):
    """BatchNorm with running-stat fallback for a one-graph training batch."""

    def forward(self, x: Tensor) -> Tensor:
        if self.training and x.shape[0] == 1:
            return functional.batch_norm(
                x,
                self.running_mean,
                self.running_var,
                self.weight,
                self.bias,
                False,
                self.momentum,
                self.eps,
            )
        return super().forward(x)


@torch.no_grad()
def glorot_orthogonal_(weight: Tensor, scale: float = 2.0) -> None:
    """Match the scaled orthogonal initializer used by the author source."""

    nn.init.orthogonal_(weight)
    variance = torch.var(weight, unbiased=False)
    weight.mul_(torch.sqrt(weight.new_tensor(scale) / ((sum(weight.shape)) * variance)))


__all__ = [
    "Dense",
    "HeterogeneousInteractionBlock",
    "ResidualMLP",
    "SafeBatchNorm1d",
    "ShiftedSoftplus",
    "glorot_orthogonal_",
]
