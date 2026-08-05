"""PyG sample contract for HimNet's unified hierarchical graph."""

from __future__ import annotations

from typing import Any

from torch import Tensor

from ...data import MolecularData


class HimNetData(MolecularData):
    """Molecular data with hierarchy-local indices that batch independently.

    The canonical ``x``/``edge_index`` fields continue to describe the base
    molecule.  The ``himnet_*`` fields describe the model-specific hierarchy,
    whose node count is generally larger than the number of atoms.
    """

    himnet_x: Tensor
    himnet_edge_index: Tensor
    himnet_edge_attr: Tensor
    himnet_reverse_edge_index: Tensor
    himnet_node_batch: Tensor
    himnet_node_type: Tensor
    himnet_fp: Tensor

    def __inc__(self, key: str, value: Any, *args: Any, **kwargs: Any) -> Any:
        """Offset hierarchy-local indices during PyG collation."""

        if key == "himnet_edge_index":
            x = getattr(self, "himnet_x", None)
            if not isinstance(x, Tensor):
                raise ValueError("HimNetData requires a himnet_x tensor")
            return x.shape[0]
        if key == "himnet_reverse_edge_index":
            edge_index = getattr(self, "himnet_edge_index", None)
            if not isinstance(edge_index, Tensor):
                raise ValueError("HimNetData requires a himnet_edge_index tensor")
            return edge_index.shape[1]
        if key == "himnet_node_batch":
            return 1
        return super().__inc__(key, value, *args, **kwargs)

    def __cat_dim__(self, key: str, value: Any, *args: Any, **kwargs: Any) -> Any:
        """Keep one-dimensional hierarchy metadata concatenated by rows."""

        if key in {"himnet_reverse_edge_index", "himnet_node_batch", "himnet_node_type"}:
            return 0
        return super().__cat_dim__(key, value, *args, **kwargs)


__all__ = ["HimNetData"]
