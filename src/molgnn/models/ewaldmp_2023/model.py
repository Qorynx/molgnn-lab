"""Project-facing aperiodic EwaldMP model with a PaiNN backbone."""

from __future__ import annotations

import math

import torch
from torch import Tensor, nn
from torch_geometric.data import Batch
from torch_geometric.utils import scatter

from ..painn_2021.constants import (
    PAINN_CUTOFF,
    PAINN_EPS,
    PAINN_MAX_ATOMIC_NUMBER,
    PAINN_NUM_RBF,
)
from ..painn_2021.model import PaiNN
from .fourier import (
    build_k_rbf,
    build_k_voxel_grid,
    canonicalize_positions,
    voxel_damping,
)
from .layers import EwaldBlock, EwaldDense


class EwaldMP(PaiNN):
    """Aperiodic Ewald long-range messages injected into scalar PaiNN states."""

    required_batch_fields = PaiNN.required_batch_fields

    def __init__(
        self,
        num_targets: int,
        hidden_dim: int = 128,
        num_interactions: int = 3,
        num_rbf: int = PAINN_NUM_RBF,
        radial_basis: str = "bessel",
        readout: str = "sum",
        max_atomic_number: int = PAINN_MAX_ATOMIC_NUMBER,
        epsilon: float = PAINN_EPS,
        k_cutoff: float = 0.6,
        delta_k: float = 0.2,
        num_k_rbf: int = 128,
        downprojection_size: int = 8,
        num_ewald_hidden: int = 0,
        ewald_update_scale: float = 0.01,
    ) -> None:
        super().__init__(
            num_targets=num_targets,
            hidden_dim=hidden_dim,
            num_interactions=num_interactions,
            num_rbf=num_rbf,
            cutoff=PAINN_CUTOFF,
            radial_basis=radial_basis,
            readout=readout,
            max_atomic_number=max_atomic_number,
            epsilon=epsilon,
        )
        if (
            isinstance(downprojection_size, bool)
            or not isinstance(downprojection_size, int)
            or downprojection_size < 2
        ):
            raise ValueError("downprojection_size must be an integer of at least two")
        if (
            isinstance(num_k_rbf, bool)
            or not isinstance(num_k_rbf, int)
            or num_k_rbf < 2
        ):
            raise ValueError("num_k_rbf must be an integer of at least two")
        if (
            isinstance(num_ewald_hidden, bool)
            or not isinstance(num_ewald_hidden, int)
            or num_ewald_hidden < 0
        ):
            raise ValueError("num_ewald_hidden must be a non-negative integer")
        if isinstance(ewald_update_scale, bool) or not isinstance(
            ewald_update_scale, (int, float)
        ):
            raise TypeError("ewald_update_scale must be a non-negative finite number")
        if not math.isfinite(float(ewald_update_scale)) or ewald_update_scale < 0.0:
            raise ValueError("ewald_update_scale must be a non-negative finite number")

        k_grid = build_k_voxel_grid(k_cutoff, delta_k)
        k_rbf_values = build_k_rbf(
            k_grid,
            num_rbf=num_k_rbf,
            k_cutoff=k_cutoff,
        )
        self.register_buffer("k_grid", k_grid)
        self.register_buffer("k_rbf_values", k_rbf_values)
        self.k_cutoff = float(k_cutoff)
        self.delta_k = float(delta_k)
        self.num_k_rbf = num_k_rbf
        self.downprojection_size = downprojection_size
        self.num_ewald_hidden = num_ewald_hidden
        self.ewald_update_scale = float(ewald_update_scale)

        self.ewald_down = EwaldDense(num_k_rbf, downprojection_size, activate=False)
        self.ewald_blocks = nn.ModuleList(
            EwaldBlock(
                self.ewald_down,
                hidden_dim=hidden_dim,
                downprojection_size=downprojection_size,
                num_hidden=num_ewald_hidden,
                update_scale=self.ewald_update_scale,
            )
            for _ in range(num_interactions)
        )
        self.message_skip_factor = 1.0 / math.sqrt(3.0)

    def forward(self, batch: Batch) -> Tensor:
        """Return graph-level predictions with shape ``[B, T]``."""

        atomic_number, pos, edge_index, graph_batch, num_graphs = self._validate_batch(
            batch
        )
        scalar, _ = self._encode_ewald(
            atomic_number,
            pos,
            edge_index,
            graph_batch,
            num_graphs,
        )
        atom_outputs = self.output_network(scalar)
        return scatter(
            atom_outputs,
            graph_batch,
            dim=0,
            dim_size=num_graphs,
            reduce=self.readout,
        )

    def _encode_ewald(
        self,
        atomic_number: Tensor,
        pos: Tensor,
        edge_index: Tensor,
        graph_batch: Tensor,
        num_graphs: int,
    ) -> tuple[Tensor, Tensor]:
        source, target = edge_index
        displacement = pos[source] - pos[target]
        distances = torch.linalg.vector_norm(displacement, dim=-1)
        if distances.numel() and bool((distances <= self.epsilon).any()):
            raise ValueError("painn_edge_index contains a coincident atom pair")
        directions = displacement / distances.clamp_min(self.epsilon).unsqueeze(-1)
        radial = self.radial_basis(distances)
        filters = self.filter_net(radial)
        filters = filters * self.cutoff_fn(distances).unsqueeze(-1)
        filter_chunks = filters.split(3 * self.hidden_dim, dim=-1)

        canonical_pos = canonicalize_positions(pos, graph_batch, num_graphs)
        dot = canonical_pos @ self.k_grid.to(dtype=pos.dtype).transpose(0, 1)
        damping = voxel_damping(canonical_pos, self.delta_k)

        scalar = self.atom_embedding(atomic_number)
        vector = torch.zeros(
            (atomic_number.shape[0], 3, self.hidden_dim),
            dtype=scalar.dtype,
            device=scalar.device,
        )
        k_rbf_values = self.k_rbf_values.to(dtype=scalar.dtype)
        for interaction, mixing, filter_weight, ewald_block in zip(
            self.interactions,
            self.mixing,
            filter_chunks,
            self.ewald_blocks,
        ):
            ewald_update = ewald_block(
                scalar,
                dot,
                damping,
                k_rbf_values,
                graph_batch,
                num_graphs,
            )
            scalar, vector = interaction(
                scalar,
                vector,
                filter_weight,
                edge_index,
                directions,
            )
            scalar = (scalar + ewald_update) * self.message_skip_factor
            scalar, vector = mixing(scalar, vector)
        return scalar, vector


__all__ = ["EwaldMP"]
