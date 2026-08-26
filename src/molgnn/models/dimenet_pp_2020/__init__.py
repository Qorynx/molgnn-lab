"""DimeNet++ (2020) architecture implementation."""

from __future__ import annotations

from .layers import InteractionPPBlock, OutputPPBlock
from .model import DimeNetPlusPlus2020

__all__ = [
    "DimeNetPlusPlus2020",
    "InteractionPPBlock",
    "OutputPPBlock",
]
