"""Lazy public exports for independently importable architectures."""

from __future__ import annotations

from importlib import import_module
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from .attentivefp_2020 import AttentiveFP as AttentiveFP
    from .attentivefp_2020 import AttentiveFPTrace as AttentiveFPTrace
    from .base import BaseMolecularModel as BaseMolecularModel
    from .dmpnn_2024 import DMPNN as DMPNN
    from .dmpnn_2024 import DMPNNData as DMPNNData
    from .gcn_baseline import GCNBaseline as GCNBaseline
    from .hignn_2023 import FeatureAttention as FeatureAttention
    from .hignn_2023 import HiGNN as HiGNN
    from .hignn_2023 import HiGNNData as HiGNNData
    from .hignn_2023 import NTNConv as NTNConv
    from .molecular_graph_embedding_2017 import ColeyGraphConv as ColeyGraphConv
    from .molecular_graph_embedding_2017 import (
        MolecularGraphEmbedding as MolecularGraphEmbedding,
    )
    from .registration import register_builtin_models as register_builtin_models
    from .trimnet_2020 import TrimNet2020 as TrimNet2020

_EXPORTS = {
    "AttentiveFP": (".attentivefp_2020", "AttentiveFP"),
    "AttentiveFPTrace": (".attentivefp_2020", "AttentiveFPTrace"),
    "BaseMolecularModel": (".base", "BaseMolecularModel"),
    "ColeyGraphConv": (".molecular_graph_embedding_2017", "ColeyGraphConv"),
    "DMPNN": (".dmpnn_2024", "DMPNN"),
    "DMPNNData": (".dmpnn_2024", "DMPNNData"),
    "FeatureAttention": (".hignn_2023", "FeatureAttention"),
    "GCNBaseline": (".gcn_baseline", "GCNBaseline"),
    "HiGNN": (".hignn_2023", "HiGNN"),
    "HiGNNData": (".hignn_2023", "HiGNNData"),
    "MolecularGraphEmbedding": (
        ".molecular_graph_embedding_2017",
        "MolecularGraphEmbedding",
    ),
    "NTNConv": (".hignn_2023", "NTNConv"),
    "TrimNet2020": (".trimnet_2020", "TrimNet2020"),
    "register_builtin_models": (".registration", "register_builtin_models"),
}


def __getattr__(name: str) -> Any:
    """Load only the architecture explicitly requested by the caller."""

    try:
        module_name, symbol_name = _EXPORTS[name]
    except KeyError as exc:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}") from exc
    value = getattr(import_module(module_name, __name__), symbol_name)
    globals()[name] = value
    return value


__all__ = (
    "DMPNN",
    "AttentiveFP",
    "AttentiveFPTrace",
    "BaseMolecularModel",
    "ColeyGraphConv",
    "DMPNNData",
    "FeatureAttention",
    "GCNBaseline",
    "HiGNN",
    "HiGNNData",
    "MolecularGraphEmbedding",
    "NTNConv",
    "TrimNet2020",
    "register_builtin_models",
)
