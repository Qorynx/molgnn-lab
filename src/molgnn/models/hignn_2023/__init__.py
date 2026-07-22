"""HiGNN 2023 architecture package."""

from .data import HiGNNData
from .layers import FeatureAttention, NTNConv
from .model import HiGNN

__all__ = ["FeatureAttention", "HiGNN", "HiGNNData", "NTNConv"]
