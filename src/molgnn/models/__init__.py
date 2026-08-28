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
    from .chemrl_gem_2022 import ChemRLGEM as ChemRLGEM
    from .chemrl_gem_2022 import ChemRLGEMPretrainer as ChemRLGEMPretrainer
    from .chemrl_gem_2022 import GeoGNNEncoder as GeoGNNEncoder
    from .dgt_2026 import DGT2026 as DGT2026
    from .dimenet_2020 import DimeNet2020 as DimeNet2020
    from .dimenet_2020 import DimeNetData as DimeNetData
    from .dimenet_pp_2020 import DimeNetPlusPlus2020 as DimeNetPlusPlus2020
    from .dmpnn_2024 import DMPNN as DMPNN
    from .dmpnn_2024 import DMPNNData as DMPNNData
    from .egnn_2021 import EGNN as EGNN
    from .equiformer_2023 import Equiformer as Equiformer
    from .fragnet_2026 import FragNet as FragNet
    from .gcn_baseline import GCNBaseline as GCNBaseline
    from .gpspp_2023 import GPSPlusPlus as GPSPlusPlus
    from .graphmvp_2022 import GraphMVP as GraphMVP
    from .graphmvp_2022 import GraphMVPPretrainer as GraphMVPPretrainer
    from .graphormer_2021 import Graphormer as Graphormer
    from .grover_2021 import GROVER as GROVER
    from .hignn_2023 import FeatureAttention as FeatureAttention
    from .hignn_2023 import HiGNN as HiGNN
    from .hignn_2023 import HiGNNData as HiGNNData
    from .hignn_2023 import NTNConv as NTNConv
    from .himnet_2026 import HimNet as HimNet
    from .himnet_2026 import HimNetData as HimNetData
    from .himol_2023 import HiMol as HiMol
    from .himol_2023 import HiMolPretrainer as HiMolPretrainer
    from .mat_2020 import MAT as MAT
    from .mgcn_2019 import MGCN as MGCN
    from .molclr_2022.model import MolCLRGCN as MolCLRGCN
    from .molclr_2022.model import MolCLRGIN as MolCLRGIN
    from .molebert_2023 import MoleBERT as MoleBERT
    from .molecular_graph_embedding_2017 import ColeyGraphConv as ColeyGraphConv
    from .molecular_graph_embedding_2017 import (
        MolecularGraphEmbedding as MolecularGraphEmbedding,
    )
    from .mpnn_2017 import MPNN as MPNN
    from .mpnn_2017 import MPNNDistanceBins3D as MPNNDistanceBins3D
    from .mxmnet_2020 import MXMNet2020 as MXMNet2020
    from .neural_fingerprint_2015 import NeuralFingerprint as NeuralFingerprint
    from .potentialnet_2018 import PotentialNet as PotentialNet
    from .pretrain_gnns_2020 import PretrainGNNs as PretrainGNNs
    from .pvd_2023 import PVDPretrainer as PVDPretrainer
    from .pvd_2023 import PVDTorchMDET as PVDTorchMDET
    from .registration import register_builtin_models as register_builtin_models
    from .resgat_2024 import ResGAT as ResGAT
    from .spherenet_2022 import SphereNet2022 as SphereNet2022
    from .transformer_m_2023 import TransformerM as TransformerM
    from .trimnet_2020 import TrimNet2020 as TrimNet2020
    from .weave_2016 import Weave as Weave

_EXPORTS = {
    "AMPNN": (".ampnn_emnn_2020.ampnn", "AMPNN"),
    "AttentiveFP": (".attentivefp_2020", "AttentiveFP"),
    "ChemRLGEM": (".chemrl_gem_2022", "ChemRLGEM"),
    "ChemRLGEMPretrainer": (".chemrl_gem_2022", "ChemRLGEMPretrainer"),
    "GeoGNNEncoder": (".chemrl_gem_2022", "GeoGNNEncoder"),
    "AttentiveFPTrace": (".attentivefp_2020", "AttentiveFPTrace"),
    "BaseMolecularModel": (".base", "BaseMolecularModel"),
    "ColeyGraphConv": (".molecular_graph_embedding_2017", "ColeyGraphConv"),
    "DMPNN": (".dmpnn_2024", "DMPNN"),
    "DMPNNData": (".dmpnn_2024", "DMPNNData"),
    "DGT2026": (".dgt_2026", "DGT2026"),
    "DimeNet2020": (".dimenet_2020", "DimeNet2020"),
    "DimeNetData": (".dimenet_2020", "DimeNetData"),
    "DimeNetPlusPlus2020": (".dimenet_pp_2020", "DimeNetPlusPlus2020"),
    "EGNN": (".egnn_2021", "EGNN"),
    "EMNN": (".ampnn_emnn_2020.emnn", "EMNN"),
    "EMNNData": (".ampnn_emnn_2020.data", "EMNNData"),
    "Equiformer": (".equiformer_2023", "Equiformer"),
    "FeatureAttention": (".hignn_2023", "FeatureAttention"),
    "FragNet": (".fragnet_2026", "FragNet"),
    "GCNBaseline": (".gcn_baseline", "GCNBaseline"),
    "GPSPlusPlus": (".gpspp_2023", "GPSPlusPlus"),
    "GraphMVP": (".graphmvp_2022", "GraphMVP"),
    "GraphMVPPretrainer": (".graphmvp_2022", "GraphMVPPretrainer"),
    "Graphormer": (".graphormer_2021", "Graphormer"),
    "GROVER": (".grover_2021", "GROVER"),
    "HiGNN": (".hignn_2023", "HiGNN"),
    "HiGNNData": (".hignn_2023", "HiGNNData"),
    "HimNet": (".himnet_2026", "HimNet"),
    "HimNetData": (".himnet_2026", "HimNetData"),
    "HiMol": (".himol_2023", "HiMol"),
    "HiMolPretrainer": (".himol_2023", "HiMolPretrainer"),
    "MAT": (".mat_2020", "MAT"),
    "MGCN": (".mgcn_2019", "MGCN"),
    "MolCLRGCN": (".molclr_2022.model", "MolCLRGCN"),
    "MolCLRGIN": (".molclr_2022.model", "MolCLRGIN"),
    "MoleBERT": (".molebert_2023", "MoleBERT"),
    "MolecularGraphEmbedding": (
        ".molecular_graph_embedding_2017",
        "MolecularGraphEmbedding",
    ),
    "MPNN": (".mpnn_2017", "MPNN"),
    "MPNNDistanceBins3D": (".mpnn_2017", "MPNNDistanceBins3D"),
    "MXMNet2020": (".mxmnet_2020", "MXMNet2020"),
    "MVGNNcross": (".mvgnn_2020", "MVGNNcross"),
    "NTNConv": (".hignn_2023", "NTNConv"),
    "NeuralFingerprint": (".neural_fingerprint_2015", "NeuralFingerprint"),
    "PotentialNet": (".potentialnet_2018", "PotentialNet"),
    "PretrainGNNs": (".pretrain_gnns_2020", "PretrainGNNs"),
    "PVDPretrainer": (".pvd_2023", "PVDPretrainer"),
    "PVDTorchMDET": (".pvd_2023", "PVDTorchMDET"),
    "ResGAT": (".resgat_2024", "ResGAT"),
    "SphereNet2022": (".spherenet_2022", "SphereNet2022"),
    "TransformerM": (".transformer_m_2023", "TransformerM"),
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
    "DGT2026",
    "DMPNN",
    "EGNN",
    "EMNN",
    "GROVER",
    "MAT",
    "MGCN",
    "MPNN",
    "AttentiveFP",
    "AttentiveFPTrace",
    "BaseMolecularModel",
    "ChemRLGEM",
    "ChemRLGEMPretrainer",
    "ColeyGraphConv",
    "DMPNNData",
    "DimeNet2020",
    "DimeNetData",
    "DimeNetPlusPlus2020",
    "EMNNData",
    "Equiformer",
    "FeatureAttention",
    "FragNet",
    "GCNBaseline",
    "GPSPlusPlus",
    "GeoGNNEncoder",
    "GraphMVP",
    "GraphMVPPretrainer",
    "Graphormer",
    "HiGNN",
    "HiGNNData",
    "HiMol",
    "HiMolPretrainer",
    "HimNet",
    "HimNetData",
    "MPNNDistanceBins3D",
    "MVGNNcross",
    "MXMNet2020",
    "MolCLRGCN",
    "MolCLRGIN",
    "MoleBERT",
    "MolecularGraphEmbedding",
    "NTNConv",
    "NeuralFingerprint",
    "PotentialNet",
    "PretrainGNNs",
    "PVDPretrainer",
    "PVDTorchMDET",
    "ResGAT",
    "SphereNet2022",
    "TransformerM",
    "TrimNet2020",
    "Weave",
    "register_builtin_models",
)
