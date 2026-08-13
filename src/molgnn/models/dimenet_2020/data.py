"""PyG data contract for DimeNet's edge-indexed triplet references."""

from __future__ import annotations

from typing import Any

from torch import Tensor
from torch_geometric.data import Data


class DimeNetData(Data):
    """Standalone PyG data with correct batching for DimeNet triplets.

    ``dimenet_triplet_edge_index`` stores *edge IDs*, rather than atom IDs:
    row zero selects the incoming ``k -> j`` edge and row one selects the
    outgoing ``j -> i`` edge.  PyG's default ``*_index`` offset is therefore
    the node count, which would corrupt a batch.  This view offsets both rows
    by the local DimeNet edge count instead.
    """

    dimenet_edge_index: Tensor
    dimenet_triplet_edge_index: Tensor

    def __inc__(self, key: str, value: Any, *args: Any, **kwargs: Any) -> Any:
        if key == "dimenet_triplet_edge_index":
            edge_index = getattr(self, "dimenet_edge_index", None)
            if not isinstance(edge_index, Tensor):
                raise ValueError("DimeNetData requires a dimenet_edge_index tensor")
            return edge_index.shape[1]
        return super().__inc__(key, value, *args, **kwargs)

    def __cat_dim__(self, key: str, value: Any, *args: Any, **kwargs: Any) -> Any:
        if key == "dimenet_triplet_edge_index":
            return 1
        return super().__cat_dim__(key, value, *args, **kwargs)


__all__ = ["DimeNetData"]
