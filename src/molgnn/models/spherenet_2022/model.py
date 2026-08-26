"""SphereNet2022: spherical message passing for 3-D molecular graphs.

Liu et al., *Spherical Message Passing for 3D Molecular Graphs*, ICLR 2022.
Official DIG snapshot commit ``21476b079c9226f38915dcd082b5c2ee0cddaac8``.
"""

from __future__ import annotations

import torch
from torch import Tensor, nn
from torch_geometric.data import Batch
from torch_geometric.utils import scatter

from ..base import BaseMolecularModel
from .basis import CircularBasis, RadialBasis, TorsionBasis
from .constants import (
    SPHERENET_CUTOFF,
    SPHERENET_DEFAULT_BASIS_DIM_ANGLE,
    SPHERENET_DEFAULT_BASIS_DIM_DISTANCE,
    SPHERENET_DEFAULT_BASIS_DIM_TORSION,
    SPHERENET_DEFAULT_ENVELOPE_EXPONENT,
    SPHERENET_DEFAULT_HIDDEN_DIM,
    SPHERENET_DEFAULT_INTERACTION_DIM,
    SPHERENET_DEFAULT_MAX_ATOMIC_NUMBER,
    SPHERENET_DEFAULT_NUM_AFTER_SKIP,
    SPHERENET_DEFAULT_NUM_BEFORE_SKIP,
    SPHERENET_DEFAULT_NUM_BLOCKS,
    SPHERENET_DEFAULT_NUM_OUTPUT_LAYERS,
    SPHERENET_DEFAULT_NUM_RADIAL,
    SPHERENET_DEFAULT_NUM_SPHERICAL,
    SPHERENET_DEFAULT_OUTPUT_DIM,
)
from .geometry import compute_angles, compute_distances, compute_torsions
from .layers import EmbeddingBlock, InteractionBlock, OutputBlock


class SphereNet2022(BaseMolecularModel):
    r"""SphereNet: spherical message passing for 3-D molecular graphs.

    Args:
        num_targets: Number of output targets.
        hidden_dim: Hidden embedding size (default: 128).
        interaction_dim: Embedding size inside interaction triplets (default: 64).
        output_dim: Embedding size in the output blocks (default: 256).
        basis_dim_distance: Bottleneck for the distance basis (default: 8).
        basis_dim_angle: Bottleneck for the angle basis (default: 8).
        basis_dim_torsion: Bottleneck for the torsion basis (default: 8).
        num_blocks: Number of interaction blocks (default: 4).
        num_spherical: Max degree ``L`` for spherical harmonics (default: 7).
        num_radial: Number of radial basis functions (default: 6).
        cutoff: Radius cutoff in Angstrom (default: 5.0).
        envelope_exponent: Shape of the smooth cutoff (default: 5).
        num_before_skip: Residual layers before the skip connection (default: 1).
        num_after_skip: Residual layers after the skip connection (default: 2).
        num_output_layers: Dense layers for the output block (default: 3).
        activation: Activation function; only ``"silu"`` is supported (default: ``"silu"``).
        output_initializer: ``"glorot_orthogonal"`` or ``"zeros"`` (default: ``"glorot_orthogonal"``).
        readout: Graph readout ``"sum"`` or ``"mean"`` (default: ``"sum"``).
        max_atomic_number: Upper bound for the atomic-number vocabulary (default: 118).
    """

    required_batch_fields = (
        "atomic_number",
        "pos",
        "spherenet_edge_index",
        "spherenet_triplet_edge_index",
        "spherenet_torsion_pair_index",
        "batch",
    )

    def __init__(
        self,
        *,
        num_targets: int = 1,
        hidden_dim: int = SPHERENET_DEFAULT_HIDDEN_DIM,
        interaction_dim: int = SPHERENET_DEFAULT_INTERACTION_DIM,
        output_dim: int = SPHERENET_DEFAULT_OUTPUT_DIM,
        basis_dim_distance: int = SPHERENET_DEFAULT_BASIS_DIM_DISTANCE,
        basis_dim_angle: int = SPHERENET_DEFAULT_BASIS_DIM_ANGLE,
        basis_dim_torsion: int = SPHERENET_DEFAULT_BASIS_DIM_TORSION,
        num_blocks: int = SPHERENET_DEFAULT_NUM_BLOCKS,
        num_spherical: int = SPHERENET_DEFAULT_NUM_SPHERICAL,
        num_radial: int = SPHERENET_DEFAULT_NUM_RADIAL,
        cutoff: float = SPHERENET_CUTOFF,
        envelope_exponent: int = SPHERENET_DEFAULT_ENVELOPE_EXPONENT,
        num_before_skip: int = SPHERENET_DEFAULT_NUM_BEFORE_SKIP,
        num_after_skip: int = SPHERENET_DEFAULT_NUM_AFTER_SKIP,
        num_output_layers: int = SPHERENET_DEFAULT_NUM_OUTPUT_LAYERS,
        activation: str = "silu",
        output_initializer: str = "glorot_orthogonal",
        readout: str = "sum",
        max_atomic_number: int = SPHERENET_DEFAULT_MAX_ATOMIC_NUMBER,
    ) -> None:
        super().__init__()
        if num_targets < 1:
            raise ValueError("num_targets must be positive")
        if activation != "silu":
            raise ValueError(f"unsupported activation {activation!r}; only 'silu' is supported")
        if output_initializer not in ("glorot_orthogonal", "zeros"):
            raise ValueError(
                f"output_initializer must be 'glorot_orthogonal' or 'zeros'; got {output_initializer!r}"
            )
        if readout not in ("sum", "mean"):
            raise ValueError("readout must be 'sum' or 'mean'")
        # The registered transform builds the radius graph at exactly
        # SPHERENET_CUTOFF; the model must not silently run on a different
        # topology than the one the transform produced.
        if abs(float(cutoff) - SPHERENET_CUTOFF) > 1e-6:
            raise ValueError(
                "cutoff must equal the transform's fixed "
                f"SPHERENET_CUTOFF={SPHERENET_CUTOFF}; got {cutoff}"
            )

        self.cutoff = cutoff
        self.num_blocks = num_blocks
        self.readout = readout

        # Bases.
        self.psi_dist = RadialBasis(
            num_radial, cutoff=cutoff, envelope_exponent=envelope_exponent
        )
        self.psi_angle = CircularBasis(
            num_spherical, num_radial, cutoff=cutoff
        )
        self.psi_torsion = TorsionBasis(
            num_spherical, num_radial, cutoff=cutoff
        )

        # Embedding block.
        self.init_e = EmbeddingBlock(
            num_radial, hidden_dim, max_atomic_number
        )

        # Output block for the initial state.
        self.init_v = OutputBlock(
            hidden_dim, output_dim, num_targets, num_output_layers, output_initializer
        )

        # Interaction blocks.
        self.blocks = nn.ModuleList([
            InteractionBlock(
                hidden_dim,
                interaction_dim,
                basis_dim_distance,
                basis_dim_angle,
                basis_dim_torsion,
                num_spherical,
                num_radial,
                num_before_skip,
                num_after_skip,
            )
            for _ in range(num_blocks)
        ])

        # Output blocks for each interaction depth.
        self.output_blocks = nn.ModuleList([
            OutputBlock(
                hidden_dim, output_dim, num_targets, num_output_layers, output_initializer
            )
            for _ in range(num_blocks)
        ])

        self.reset_parameters()

    def reset_parameters(self) -> None:
        self.psi_dist.reset_parameters()
        self.init_e.reset_parameters()
        self.init_v.reset_parameters()
        for block in self.blocks:
            block.reset_parameters()
        for block in self.output_blocks:
            block.reset_parameters()

    # --- public interface ---

    def forward(self, batch: Batch) -> Tensor:
        atomic_number = getattr(batch, "atomic_number", None)
        pos = getattr(batch, "pos", None)
        edge_index = getattr(batch, "spherenet_edge_index", None)
        triplet = getattr(batch, "spherenet_triplet_edge_index", None)
        torsion_pair = getattr(batch, "spherenet_torsion_pair_index", None)
        graph_batch = getattr(batch, "batch", None)

        missing = [
            name
            for name, value in (
                ("atomic_number", atomic_number),
                ("pos", pos),
                ("spherenet_edge_index", edge_index),
                ("spherenet_triplet_edge_index", triplet),
                ("spherenet_torsion_pair_index", torsion_pair),
                ("batch", graph_batch),
            )
            if not isinstance(value, Tensor)
        ]
        if missing:
            raise ValueError(
                f"batch is missing tensor field(s): {', '.join(missing)}"
            )

        # Validate shapes and indices.
        num_nodes = int(atomic_number.shape[0])
        num_edges = int(edge_index.shape[1])
        num_triplets = int(triplet.shape[1])
        if pos.shape != (num_nodes, 3) or pos.dtype != torch.float32:
            raise ValueError("pos must have shape [N, 3] float32")
        if edge_index.shape[0] != 2:
            raise ValueError("spherenet_edge_index must have shape [2, E]")
        if triplet.shape[0] != 2:
            raise ValueError("spherenet_triplet_edge_index must have shape [2, Q]")
        if torsion_pair.shape[0] != 2:
            raise ValueError("spherenet_torsion_pair_index must have shape [2, R]")
        if num_edges and (
            edge_index.min() < 0 or edge_index.max() >= num_nodes
        ):
            raise ValueError("spherenet_edge_index has out-of-range atom indices")
        if num_edges and bool(
            (graph_batch[edge_index[0]] != graph_batch[edge_index[1]]).any()
        ):
            raise ValueError(
                "spherenet_edge_index must not connect different graphs"
            )
        if num_triplets and (
            triplet.min() < 0 or triplet.max() >= num_edges
        ):
            raise ValueError("spherenet_triplet_edge_index has out-of-range edge indices")
        if torsion_pair.numel():
            if torsion_pair[0].min() < 0 or torsion_pair[0].max() >= num_triplets:
                raise ValueError(
                    "spherenet_torsion_pair_index row 0 has out-of-range triplet indices"
                )
            if torsion_pair[1].min() < 0 or torsion_pair[1].max() >= num_edges:
                raise ValueError(
                    "spherenet_torsion_pair_index row 1 has out-of-range edge indices"
                )

        # --- geometry (differentiable) ---
        dist = compute_distances(pos, edge_index)  # [E]
        angle = compute_angles(pos, edge_index, triplet)  # [Q]
        torsion = compute_torsions(pos, edge_index, triplet, torsion_pair)  # [Q]

        # --- bases ---
        rbf = self.psi_dist(dist)  # [E, N]
        sbf = self.psi_angle(
            dist[triplet[0]], torch.cos(angle)
        )  # [Q, L*N]
        t = self.psi_torsion(
            dist[triplet[0]], torch.cos(angle), torsion
        )  # [Q, L^2*N]

        # --- initial edge state ---
        e1, e2 = self.init_e(atomic_number, rbf, edge_index)

        # --- output for initial depth ---
        v = self.init_v(e2, edge_index, num_nodes)
        u = self._graph_readout(v, graph_batch)

        # --- interaction blocks ---
        for block, out_block in zip(self.blocks, self.output_blocks):
            e1, e2 = block(
                (e1, e2), rbf, sbf, t, triplet[0], triplet[1]
            )
            v = out_block(e2, edge_index, num_nodes)
            u = u + self._graph_readout(v, graph_batch)

        return u

    # --- internal helpers ---

    def _graph_readout(self, atomwise: Tensor, graph_batch: Tensor) -> Tensor:
        num_graphs = int(graph_batch.max().item()) + 1
        if self.readout == "mean":
            return scatter(
                atomwise, graph_batch, dim=0, dim_size=num_graphs, reduce="mean"
            )
        return scatter(
            atomwise, graph_batch, dim=0, dim_size=num_graphs, reduce="sum"
        )


__all__ = ["SphereNet2022"]