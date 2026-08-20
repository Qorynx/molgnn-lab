"""Lazy public wrapper for the optional-e3nn Equiformer implementation."""

from __future__ import annotations

from torch import Tensor
from torch_geometric.data import Batch

from ..base import BaseMolecularModel
from .constants import (
    EQUIFORMER_AVG_DEGREE,
    EQUIFORMER_AVG_NUM_NODES,
    EQUIFORMER_CUTOFF,
    EQUIFORMER_MAX_ATOMIC_NUMBER,
    EQUIFORMER_MAX_NEIGHBORS,
    EQUIFORMER_NUM_RADIAL,
)


class Equiformer(BaseMolecularModel):
    """SE(3)-equivariant graph attention Transformer for 3-D molecules.

    The e3nn import happens only when a caller constructs this model.  That
    keeps discovery and all unrelated architectures usable without the optional
    Equiformer extra installed.
    """

    required_batch_fields = (
        "atomic_number",
        "pos",
        "equiformer_edge_index",
        "batch",
    )

    def __init__(
        self,
        num_targets: int,
        scalar_channels: int = 128,
        vector_channels: int = 64,
        tensor_channels: int = 32,
        num_layers: int = 6,
        num_radial: int = EQUIFORMER_NUM_RADIAL,
        radial_hidden_dim: int = 64,
        head_scalar_channels: int = 32,
        head_vector_channels: int = 16,
        head_tensor_channels: int = 8,
        num_heads: int = 4,
        ffn_multiplier: int = 3,
        feature_scalar_channels: int = 512,
        attention_dropout: float = 0.2,
        max_atomic_number: int = EQUIFORMER_MAX_ATOMIC_NUMBER,
        average_degree: float = EQUIFORMER_AVG_DEGREE,
        average_num_nodes: float = EQUIFORMER_AVG_NUM_NODES,
    ) -> None:
        super().__init__()
        try:
            from .core import EquiformerCore
        except ModuleNotFoundError as error:
            if error.name == "e3nn" or error.name.startswith("e3nn."):
                raise RuntimeError(
                    "Equiformer requires optional dependency e3nn; install molgnn-lab[equiformer]."
                ) from error
            raise
        self.num_targets = num_targets
        self.max_atomic_number = max_atomic_number
        self.cutoff = EQUIFORMER_CUTOFF
        self.max_neighbors = EQUIFORMER_MAX_NEIGHBORS
        self.core = EquiformerCore(
            num_targets=num_targets,
            scalar_channels=scalar_channels,
            vector_channels=vector_channels,
            tensor_channels=tensor_channels,
            num_layers=num_layers,
            num_radial=num_radial,
            radial_hidden_dim=radial_hidden_dim,
            head_scalar_channels=head_scalar_channels,
            head_vector_channels=head_vector_channels,
            head_tensor_channels=head_tensor_channels,
            num_heads=num_heads,
            ffn_multiplier=ffn_multiplier,
            feature_scalar_channels=feature_scalar_channels,
            attention_dropout=attention_dropout,
            max_atomic_number=max_atomic_number,
            average_degree=average_degree,
            average_num_nodes=average_num_nodes,
        )

    def forward(self, batch: Batch) -> Tensor:
        return self.core(batch)


__all__ = ["Equiformer"]
