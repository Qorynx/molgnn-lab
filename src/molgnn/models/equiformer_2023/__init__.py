"""Equiformer (Liao and Smidt, 2023) public model boundary."""

from .constants import (
    EQUIFORMER_AVG_DEGREE,
    EQUIFORMER_AVG_NUM_NODES,
    EQUIFORMER_CUTOFF,
    EQUIFORMER_MAX_ATOMIC_NUMBER,
    EQUIFORMER_MAX_NEIGHBORS,
    EQUIFORMER_NUM_RADIAL,
)
from .model import Equiformer

__all__ = [
    "EQUIFORMER_AVG_DEGREE",
    "EQUIFORMER_AVG_NUM_NODES",
    "EQUIFORMER_CUTOFF",
    "EQUIFORMER_MAX_ATOMIC_NUMBER",
    "EQUIFORMER_MAX_NEIGHBORS",
    "EQUIFORMER_NUM_RADIAL",
    "Equiformer",
]
