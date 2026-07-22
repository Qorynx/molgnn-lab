"""PyG sample contract for batching D-MPNN 2024 reverse-edge indices."""

from __future__ import annotations

from typing import Any

from torch import Tensor
from torch_geometric.data import Data


class DMPNNData(Data):
    """Standalone PyG data whose reverse-edge mapping batches by edge count."""

    reverse_edge_index: Tensor

    def __inc__(self, key: str, value: Any, *args: Any, **kwargs: Any) -> Any:
        if key == "reverse_edge_index":
            edge_index = getattr(self, "edge_index", None)
            if not isinstance(edge_index, Tensor):
                raise ValueError("DMPNNData requires an edge_index tensor")
            return edge_index.shape[1]
        return super().__inc__(key, value, *args, **kwargs)

    def __cat_dim__(self, key: str, value: Any, *args: Any, **kwargs: Any) -> Any:
        if key == "reverse_edge_index":
            return 0
        return super().__cat_dim__(key, value, *args, **kwargs)


__all__ = ["DMPNNData"]
