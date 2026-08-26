"""Input, interaction and output blocks for SphereNet.

Every block is an independent ``nn.Module`` with its own parameters.  The
architecture follows the official DIG source (commit ``21476b0``) with
modernizations: ``torch_geometric.utils.scatter``, explicit ``dim_size``,
``torch.nn.ModuleList``, and tensor-local dtype/device.
"""

from __future__ import annotations

import math

import torch
from torch import Tensor, nn
from torch_geometric.nn.inits import glorot_orthogonal
from torch_geometric.utils import scatter


def swish(x: Tensor) -> Tensor:
    return x * torch.sigmoid(x)


class ResidualLayer(nn.Module):
    """Two-layer SiLU residual block (DIG ``ResidualLayer``)."""

    def __init__(self, hidden_channels: int) -> None:
        super().__init__()
        self.lin1 = nn.Linear(hidden_channels, hidden_channels)
        self.lin2 = nn.Linear(hidden_channels, hidden_channels)
        self.reset_parameters()

    def reset_parameters(self) -> None:
        glorot_orthogonal(self.lin1.weight, scale=2.0)
        self.lin1.bias.data.fill_(0)
        glorot_orthogonal(self.lin2.weight, scale=2.0)
        self.lin2.bias.data.fill_(0)

    def forward(self, x: Tensor) -> Tensor:
        return x + swish(self.lin2(swish(self.lin1(x))))


class EmbeddingBlock(nn.Module):
    """Learned atomic-number embedding, radial projection, edge message creation.

    Produces the two initial edge states (``e1``, ``e2``) as in the DIG
    ``init`` module.
    """

    def __init__(
        self,
        num_radial: int,
        hidden_channels: int,
        max_atomic_number: int,
    ) -> None:
        super().__init__()
        self.emb = nn.Embedding(max_atomic_number + 1, hidden_channels)
        self.lin_rbf_0 = nn.Linear(num_radial, hidden_channels)
        self.lin = nn.Linear(3 * hidden_channels, hidden_channels)
        self.lin_rbf_1 = nn.Linear(num_radial, hidden_channels, bias=False)
        self.reset_parameters()

    def reset_parameters(self) -> None:
        self.emb.weight.data.uniform_(-math.sqrt(3), math.sqrt(3))
        self.lin_rbf_0.reset_parameters()
        self.lin.reset_parameters()
        glorot_orthogonal(self.lin_rbf_1.weight, scale=2.0)

    def forward(
        self, atomic_number: Tensor, rbf: Tensor, edge_index: Tensor
    ) -> tuple[Tensor, Tensor]:
        """Return ``(e1, e2)`` per directed edge ``j -> i``.

        Args:
            atomic_number: ``[N]`` learnt atomic-number index.
            rbf: ``[E, N]`` radial distance basis.
            edge_index: ``[2, E]`` source ``j``, target ``i``.
        """

        z = self.emb(atomic_number)
        j, i = edge_index
        rbf_proj = swish(self.lin_rbf_0(rbf))
        e1 = swish(self.lin(torch.cat([z[i], z[j], rbf_proj], dim=-1)))
        e2 = self.lin_rbf_1(rbf) * e1
        return e1, e2


class InteractionBlock(nn.Module):
    """Spherical message passing with distance, angle and torsion gates.

    Each block has its own ``nn.ModuleList`` of residual layers, so every
    block carries distinct parameters.
    """

    def __init__(
        self,
        hidden_channels: int,
        int_emb_size: int,
        basis_emb_size_dist: int,
        basis_emb_size_angle: int,
        basis_emb_size_torsion: int,
        num_spherical: int,
        num_radial: int,
        num_before_skip: int,
        num_after_skip: int,
    ) -> None:
        super().__init__()
        self.lin_rbf1 = nn.Linear(num_radial, basis_emb_size_dist, bias=False)
        self.lin_rbf2 = nn.Linear(basis_emb_size_dist, hidden_channels, bias=False)
        self.lin_sbf1 = nn.Linear(
            num_spherical * num_radial, basis_emb_size_angle, bias=False
        )
        self.lin_sbf2 = nn.Linear(
            basis_emb_size_angle, int_emb_size, bias=False
        )
        self.lin_t1 = nn.Linear(
            num_spherical * num_spherical * num_radial,
            basis_emb_size_torsion,
            bias=False,
        )
        self.lin_t2 = nn.Linear(
            basis_emb_size_torsion, int_emb_size, bias=False
        )
        self.lin_rbf = nn.Linear(num_radial, hidden_channels, bias=False)

        self.lin_kj = nn.Linear(hidden_channels, hidden_channels)
        self.lin_ji = nn.Linear(hidden_channels, hidden_channels)

        self.lin_down = nn.Linear(hidden_channels, int_emb_size, bias=False)
        self.lin_up = nn.Linear(int_emb_size, hidden_channels, bias=False)

        self.layers_before_skip = nn.ModuleList(
            [ResidualLayer(hidden_channels) for _ in range(num_before_skip)]
        )
        self.lin = nn.Linear(hidden_channels, hidden_channels)
        self.layers_after_skip = nn.ModuleList(
            [ResidualLayer(hidden_channels) for _ in range(num_after_skip)]
        )
        self.reset_parameters()

    def reset_parameters(self) -> None:
        for weight in [
            self.lin_rbf1.weight,
            self.lin_rbf2.weight,
            self.lin_sbf1.weight,
            self.lin_sbf2.weight,
            self.lin_t1.weight,
            self.lin_t2.weight,
        ]:
            glorot_orthogonal(weight, scale=2.0)

        glorot_orthogonal(self.lin_kj.weight, scale=2.0)
        self.lin_kj.bias.data.fill_(0)
        glorot_orthogonal(self.lin_ji.weight, scale=2.0)
        self.lin_ji.bias.data.fill_(0)

        glorot_orthogonal(self.lin_down.weight, scale=2.0)
        glorot_orthogonal(self.lin_up.weight, scale=2.0)

        for res_layer in self.layers_before_skip:
            res_layer.reset_parameters()
        glorot_orthogonal(self.lin.weight, scale=2.0)
        self.lin.bias.data.fill_(0)
        for res_layer in self.layers_after_skip:
            res_layer.reset_parameters()

        glorot_orthogonal(self.lin_rbf.weight, scale=2.0)

    def forward(
        self,
        edge_state: tuple[Tensor, Tensor],
        rbf: Tensor,
        sbf: Tensor,
        t: Tensor,
        idx_kj: Tensor,
        idx_ji: Tensor,
    ) -> tuple[Tensor, Tensor]:
        """One interaction step.

        Args:
            edge_state: ``(e1, e2)`` per edge.
            rbf: ``[E, N]`` radial distance basis.
            sbf: ``[Q, L*N]`` circular basis (angle).
            t: ``[Q, L^2*N]`` full torsion basis.
            idx_kj: ``[Q]`` incoming edge id for each triplet.
            idx_ji: ``[Q]`` current edge id for each triplet.

        Returns:
            Updated ``(e1, e2)``.
        """

        x1, _ = edge_state

        x_ji = swish(self.lin_ji(x1))
        x_kj = swish(self.lin_kj(x1))

        # Distance gate on incoming edge.
        rbf_proj = self.lin_rbf2(self.lin_rbf1(rbf))
        x_kj = x_kj * rbf_proj

        # Down-project.
        x_kj = swish(self.lin_down(x_kj))

        # Angle gate (triplet gather).
        sbf_proj = self.lin_sbf2(self.lin_sbf1(sbf))
        x_kj = x_kj[idx_kj] * sbf_proj

        # Torsion gate.
        t_proj = self.lin_t2(self.lin_t1(t))
        x_kj = x_kj * t_proj

        # Scatter-sum to current edge.
        x_kj = scatter(x_kj, idx_ji, dim=0, dim_size=x1.size(0), reduce="sum")
        x_kj = swish(self.lin_up(x_kj))

        # Residual before skip.
        e1 = x_ji + x_kj
        for layer in self.layers_before_skip:
            e1 = layer(e1)
        # Outer skip.
        e1 = swish(self.lin(e1)) + x1
        for layer in self.layers_after_skip:
            e1 = layer(e1)
        # Update e2 = Linear(rbf) * e1.
        e2 = self.lin_rbf(rbf) * e1

        return e1, e2


class OutputBlock(nn.Module):
    """Atom-wise target projection from edge states.

    ``scatter`` sums ``e2`` over incoming edges per atom, up-projects to the
    output embedding width, applies dense SiLU layers, and returns a
    ``dim_size``-safe ``[num_nodes, num_targets]`` tensor.
    """

    def __init__(
        self,
        hidden_channels: int,
        out_emb_channels: int,
        out_channels: int,
        num_output_layers: int,
        output_init: str,
    ) -> None:
        super().__init__()
        self.output_init = output_init
        self.lin_up = nn.Linear(hidden_channels, out_emb_channels)
        self.lins = nn.ModuleList(
            [nn.Linear(out_emb_channels, out_emb_channels) for _ in range(num_output_layers)]
        )
        self.lin = nn.Linear(out_emb_channels, out_channels, bias=False)
        self.reset_parameters()

    def reset_parameters(self) -> None:
        glorot_orthogonal(self.lin_up.weight, scale=2.0)
        for lin in self.lins:
            glorot_orthogonal(lin.weight, scale=2.0)
            lin.bias.data.fill_(0)
        if self.output_init == "zeros":
            self.lin.weight.data.fill_(0)
        elif self.output_init == "glorot_orthogonal":
            glorot_orthogonal(self.lin.weight, scale=2.0)

    def forward(
        self, e2: Tensor, edge_index: Tensor, num_nodes: int
    ) -> Tensor:
        """Return ``[num_nodes, out_channels]`` atom-wise targets."""

        v = scatter(e2, edge_index[1], dim=0, dim_size=num_nodes, reduce="sum")
        v = self.lin_up(v)
        for lin in self.lins:
            v = swish(lin(v))
        v = self.lin(v)
        return v


__all__ = [
    "EmbeddingBlock",
    "InteractionBlock",
    "OutputBlock",
    "ResidualLayer",
    "swish",
]