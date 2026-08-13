"""PyG sample contract for batching EMNN reverse-edge indices."""

from __future__ import annotations

from typing import Any

from torch import Tensor
from torch_geometric.data import Data


class EMNNData(Data):
    """Standalone PyG data whose reverse map batches by directed-edge count.

    The canonical runner emits :class:`molgnn.data.MolecularData`, which
    already applies this offset.  This class keeps the same invariant for
    library users building an EMNN batch directly from ``torch_geometric``
    samples.
    """

    reverse_edge_index: Tensor

    def __inc__(self, key: str, value: Any, *args: Any, **kwargs: Any) -> Any:
        if key == "reverse_edge_index":
            edge_index = getattr(self, "edge_index", None)
            if not isinstance(edge_index, Tensor):
                raise ValueError("EMNNData requires an edge_index tensor")
            return edge_index.shape[1]
        return super().__inc__(key, value, *args, **kwargs)

    def __cat_dim__(self, key: str, value: Any, *args: Any, **kwargs: Any) -> Any:
        if key == "reverse_edge_index":
            return 0
        return super().__cat_dim__(key, value, *args, **kwargs)


__all__ = ["EMNNData"]
