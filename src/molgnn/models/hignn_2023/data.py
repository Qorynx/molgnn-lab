"""PyG sample contract for batching HiGNN 2023 fragment identifiers."""

from __future__ import annotations

from typing import Any

from torch import Tensor
from torch_geometric.data import Data


class HiGNNData(Data):
    """Standalone PyG data whose local fragment IDs batch without collisions."""

    atom_to_fragment: Tensor

    def __inc__(self, key: str, value: Any, *args: Any, **kwargs: Any) -> Any:
        if key == "atom_to_fragment":
            return int(value.max().item()) + 1 if isinstance(value, Tensor) and value.numel() else 0
        return super().__inc__(key, value, *args, **kwargs)

    def __cat_dim__(self, key: str, value: Any, *args: Any, **kwargs: Any) -> Any:
        if key == "atom_to_fragment":
            return 0
        return super().__cat_dim__(key, value, *args, **kwargs)


__all__ = ["HiGNNData"]
