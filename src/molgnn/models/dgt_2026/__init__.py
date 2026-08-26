"""DGT 2026 (Dual Graph Transformer) architecture."""

from .layers import DGTAttention, DGTLayer, activation
from .model import DGT2026, DGTEmbedder, LineGraphReadout

__all__ = [
    "DGT2026",
    "DGTAttention",
    "DGTEmbedder",
    "DGTLayer",
    "LineGraphReadout",
    "activation",
]
