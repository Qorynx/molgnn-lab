"""Lazy public exports for independently importable architectures."""

from __future__ import annotations

from importlib import import_module
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from .ampnn_emnn_2020.ampnn import AMPNN as AMPNN
    from .ampnn_emnn_2020.data import EMNNData as EMNNData
    from .ampnn_emnn_2020.emnn import EMNN as EMNN
    from .attentivefp_2020 import AttentiveFP as AttentiveFP
    from .attentivefp_2020 import AttentiveFPTrace as AttentiveFPTrace
    from .base import BaseMolecularModel as BaseMolecularModel
    from .dmpnn_2024 import DMPNN as DMPNN
    from .dmpnn_2024 import DMPNNData as DMPNNData
    from .fragnet_2026 import FragNet as FragNet
    from .gcn_baseline import GCNBaseline as GCNBaseline
    from .hignn_2023 import FeatureAttention as FeatureAttention
    from .hignn_2023 import HiGNN as HiGNN
    from .hignn_2023 import HiGNNData as HiGNNData
    from .himnet_2026 import HimNet as HimNet
    from .himnet_2026 import HimNetData as HimNetData
    from .hignn_2023 import NTNConv as NTNConv
    from .molecular_graph_embedding_2017 import ColeyGraphConv as ColeyGraphConv
    from .molecular_graph_embedding_2017 import (
        MolecularGraphEmbedding as MolecularGraphEmbedding,
    )
    from .mpnn_2017 import MPNN as MPNN
    from .mpnn_2017 import MPNNDistanceBins3D as MPNNDistanceBins3D
    from .potentialnet_2018 import PotentialNet as PotentialNet
    from .registration import register_builtin_models as register_builtin_models
    from .resgat_2024 import ResGAT as ResGAT
    from .trimnet_2020 import TrimNet2020 as TrimNet2020
    from .weave_2016 import Weave as Weave

_EXPORTS = {
    "AMPNN": (".ampnn_emnn_2020.ampnn", "AMPNN"),
    "AttentiveFP": (".attentivefp_2020", "AttentiveFP"),
    "AttentiveFPTrace": (".attentivefp_2020", "AttentiveFPTrace"),
    "BaseMolecularModel": (".base", "BaseMolecularModel"),
    "ColeyGraphConv": (".molecular_graph_embedding_2017", "ColeyGraphConv"),
    "DMPNN": (".dmpnn_2024", "DMPNN"),
    "DMPNNData": (".dmpnn_2024", "DMPNNData"),
    "EMNN": (".ampnn_emnn_2020.emnn", "EMNN"),
    "EMNNData": (".ampnn_emnn_2020.data", "EMNNData"),
    "FragNet": (".fragnet_2026", "FragNet"),
    "FeatureAttention": (".hignn_2023", "FeatureAttention"),
    "GCNBaseline": (".gcn_baseline", "GCNBaseline"),
    "HiGNN": (".hignn_2023", "HiGNN"),
    "HiGNNData": (".hignn_2023", "HiGNNData"),
    "HimNet": (".himnet_2026", "HimNet"),
    "HimNetData": (".himnet_2026", "HimNetData"),
    "MolecularGraphEmbedding": (
        ".molecular_graph_embedding_2017",
        "MolecularGraphEmbedding",
    ),
    "MPNN": (".mpnn_2017", "MPNN"),
    "MPNNDistanceBins3D": (".mpnn_2017", "MPNNDistanceBins3D"),
    "MVGNNcross": (".mvgnn_2020", "MVGNNcross"),
    "PotentialNet": (".potentialnet_2018", "PotentialNet"),
    "ResGAT": (".resgat_2024", "ResGAT"),
    "NTNConv": (".hignn_2023", "NTNConv"),
    "TrimNet2020": (".trimnet_2020", "TrimNet2020"),
    "Weave": (".weave_2016", "Weave"),
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
    "AMPNN",
    "DMPNN",
    "AttentiveFP",
    "AttentiveFPTrace",
    "BaseMolecularModel",
    "ColeyGraphConv",
    "DMPNNData",
    "EMNN",
    "EMNNData",
    "FragNet",
    "FeatureAttention",
    "GCNBaseline",
    "HiGNN",
    "HiGNNData",
    "HimNet",
    "HimNetData",
    "MolecularGraphEmbedding",
    "MPNN",
    "MPNNDistanceBins3D",
    "MVGNNcross",
    "PotentialNet",
    "ResGAT",
    "NTNConv",
    "TrimNet2020",
    "Weave",
    "register_builtin_models",
)
