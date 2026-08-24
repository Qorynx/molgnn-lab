"""ChemRL-GEM GeoGNN encoder and downstream predictor."""

from __future__ import annotations

from pathlib import Path

import torch
from torch import Tensor, nn
from torch_geometric.data import Batch
from torch_geometric.nn import global_mean_pool

from ..base import BaseMolecularModel
from ..contracts import validate_batched_molecular_graph
from .constants import (
    ATOM_EMBED_SIZES,
    BOND_ANGLE_CENTERS,
    BOND_EMBED_SIZES,
    BOND_LENGTH_CENTERS,
    DEFAULT_DROPOUT,
    DEFAULT_EMBED_DIM,
    DEFAULT_LAYER_NUM,
)
from .layers import (
    AtomEmbedding,
    BondAngleFloatRBF,
    BondEmbedding,
    BondFloatRBF,
    DownstreamMLP,
    GeoGNNBlock,
)


class GeoGNNEncoder(nn.Module):
    """Source-compatible GeoGNN over GEM's two graph views."""

    def __init__(
        self,
        *,
        embed_dim: int = DEFAULT_EMBED_DIM,
        layer_num: int = DEFAULT_LAYER_NUM,
        dropout_rate: float = DEFAULT_DROPOUT,
        readout: str = "mean",
    ) -> None:
        super().__init__()
        _positive_int(embed_dim, "embed_dim")
        _positive_int(layer_num, "layer_num")
        if not 0 <= float(dropout_rate) < 1:
            raise ValueError("dropout_rate must be in [0, 1)")
        if readout != "mean":
            raise ValueError("ChemRL-GEM currently supports mean readout only")
        self.embed_dim = embed_dim
        self.layer_num = layer_num
        self.dropout_rate = float(dropout_rate)
        self.readout = readout

        self.init_atom_embedding = AtomEmbedding(ATOM_EMBED_SIZES, embed_dim)
        self.init_bond_embedding = BondEmbedding(BOND_EMBED_SIZES, embed_dim)
        self.init_bond_float_rbf = BondFloatRBF(BOND_LENGTH_CENTERS, embed_dim)
        self.bond_embedding_list = nn.ModuleList(
            [BondEmbedding(BOND_EMBED_SIZES, embed_dim) for _ in range(layer_num)]
        )
        self.bond_float_rbf_list = nn.ModuleList(
            [BondFloatRBF(BOND_LENGTH_CENTERS, embed_dim) for _ in range(layer_num)]
        )
        self.bond_angle_float_rbf_list = nn.ModuleList(
            [BondAngleFloatRBF(BOND_ANGLE_CENTERS, embed_dim) for _ in range(layer_num)]
        )
        self.atom_bond_block_list = nn.ModuleList(
            [
                GeoGNNBlock(embed_dim, dropout_rate, last_act=index != layer_num - 1)
                for index in range(layer_num)
            ]
        )
        self.bond_angle_block_list = nn.ModuleList(
            [
                GeoGNNBlock(embed_dim, dropout_rate, last_act=index != layer_num - 1)
                for index in range(layer_num)
            ]
        )

    @property
    def node_dim(self) -> int:
        return self.embed_dim

    @property
    def graph_dim(self) -> int:
        return self.embed_dim

    def forward(
        self,
        atom_attr: Tensor,
        atom_bond_edge_index: Tensor,
        bond_attr: Tensor,
        bond_length: Tensor,
        angle_edge_index: Tensor,
        bond_angle: Tensor,
        atom_batch: Tensor,
    ) -> tuple[Tensor, Tensor, Tensor]:
        _validate_encoder_inputs(
            atom_attr,
            atom_bond_edge_index,
            bond_attr,
            bond_length,
            angle_edge_index,
            bond_angle,
            atom_batch,
        )
        node_hidden = self.init_atom_embedding(atom_attr)
        edge_hidden = self.init_bond_embedding(bond_attr) + self.init_bond_float_rbf(bond_length)
        edge_batch = atom_batch[atom_bond_edge_index[0]]
        node_hidden_list = [node_hidden]
        edge_hidden_list = [edge_hidden]

        for layer_id in range(self.layer_num):
            node_hidden = self.atom_bond_block_list[layer_id](
                atom_bond_edge_index,
                node_hidden_list[layer_id],
                edge_hidden_list[layer_id],
                atom_batch,
            )
            # Official source re-embeds raw bond/length features per layer.
            cur_edge_hidden = self.bond_embedding_list[layer_id](bond_attr)
            cur_edge_hidden = cur_edge_hidden + self.bond_float_rbf_list[layer_id](bond_length)
            cur_angle_hidden = self.bond_angle_float_rbf_list[layer_id](bond_angle)
            edge_hidden = self.bond_angle_block_list[layer_id](
                angle_edge_index,
                cur_edge_hidden,
                cur_angle_hidden,
                edge_batch,
            )
            node_hidden_list.append(node_hidden)
            edge_hidden_list.append(edge_hidden)

        graph_repr = global_mean_pool(node_hidden_list[-1], atom_batch)
        return node_hidden_list[-1], edge_hidden_list[-1], graph_repr


ChemRLGEMEncoder = GeoGNNEncoder


class ChemRLGEM(BaseMolecularModel):
    """Graph-level ChemRL-GEM predictor with dynamic target width."""

    required_batch_fields = (
        "chemrl_gem_atom_attr",
        "chemrl_gem_edge_index",
        "chemrl_gem_bond_attr",
        "chemrl_gem_bond_length",
        "chemrl_gem_angle_edge_index",
        "chemrl_gem_bond_angle",
        "batch",
    )

    def __init__(
        self,
        atom_dim: int,
        bond_dim: int,
        num_targets: int,
        *,
        hidden_dim: int = DEFAULT_EMBED_DIM,
        num_layers: int = DEFAULT_LAYER_NUM,
        dropout: float = DEFAULT_DROPOUT,
        pooling: str = "mean",
        head_layers: int = 2,
        head_hidden_dim: int = 128,
        head_dropout: float = 0.2,
        pretrained_variant: str = "none",
        pretrained_checkpoint: str | None = None,
    ) -> None:
        super().__init__()
        del atom_dim, bond_dim
        _positive_int(num_targets, "num_targets")
        if pooling != "mean":
            raise ValueError("ChemRL-GEM currently supports mean pooling only")
        if pretrained_variant not in {"none", "classification", "regression"}:
            raise ValueError("pretrained_variant must be none, classification, or regression")
        self.num_targets = num_targets
        self.encoder = GeoGNNEncoder(
            embed_dim=hidden_dim,
            layer_num=num_layers,
            dropout_rate=dropout,
            readout=pooling,
        )
        _positive_int(head_layers, "head_layers")
        _positive_int(head_hidden_dim, "head_hidden_dim")
        if head_layers not in {2, 3}:
            raise ValueError("head_layers must be 2 or 3")
        if not 0 <= float(head_dropout) < 1:
            raise ValueError("head_dropout must be in [0, 1)")
        self.norm = nn.LayerNorm(hidden_dim)
        self.mlp = DownstreamMLP(
            hidden_dim,
            head_hidden_dim,
            num_targets,
            head_layers,
            float(head_dropout),
        )
        self.initialization = "scratch"
        self.checkpoint_info: dict[str, object] | None = None
        if pretrained_variant != "none" or pretrained_checkpoint:
            from .checkpoint import load_chemrl_gem_encoder

            path = Path(pretrained_checkpoint) if pretrained_checkpoint else _default_checkpoint(pretrained_variant)
            self.checkpoint_info = load_chemrl_gem_encoder(self.encoder, path)
            self.initialization = "pretrained"

    def forward(self, batch: Batch) -> Tensor:
        values = tuple(getattr(batch, name, None) for name in self.required_batch_fields)
        if not all(isinstance(value, Tensor) for value in values):
            raise ValueError("batch is missing ChemRL-GEM tensors")
        (
            atom_attr,
            edge_index,
            bond_attr,
            bond_length,
            angle_edge_index,
            bond_angle,
            graph_batch,
        ) = values
        assert all(isinstance(value, Tensor) for value in values)
        node_repr, _, graph_repr = self.encoder(
            atom_attr,
            edge_index,
            bond_attr,
            bond_length,
            angle_edge_index,
            bond_angle,
            graph_batch,
        )
        del node_repr
        return self.mlp(self.norm(graph_repr))


def _validate_encoder_inputs(
    atom_attr: Tensor,
    edge_index: Tensor,
    bond_attr: Tensor,
    bond_length: Tensor,
    angle_edge_index: Tensor,
    bond_angle: Tensor,
    atom_batch: Tensor,
) -> None:
    if atom_attr.ndim != 2 or atom_attr.shape[1] != 7 or atom_attr.dtype != torch.long:
        raise ValueError("ChemRL-GEM atom_attr must have shape [N, 7] and dtype long")
    if bond_attr.ndim != 2 or bond_attr.shape[1] != 3 or bond_attr.dtype != torch.long:
        raise ValueError("ChemRL-GEM bond_attr must have shape [E, 3] and dtype long")
    if bond_length.shape != (bond_attr.shape[0],) or bond_length.dtype != torch.float32:
        raise ValueError("ChemRL-GEM bond_length must have shape [E] and float32 dtype")
    if bond_angle.shape != (angle_edge_index.shape[1],) or bond_angle.dtype != torch.float32:
        raise ValueError("ChemRL-GEM bond_angle must align with angle_edge_index")
    if atom_batch.shape != (atom_attr.shape[0],) or atom_batch.dtype != torch.long:
        raise ValueError("batch.batch must assign every ChemRL-GEM atom")
    if any(value.device != atom_attr.device for value in (edge_index, bond_attr, bond_length, angle_edge_index, bond_angle, atom_batch)):
        raise ValueError("ChemRL-GEM tensors must share one device")
    validate_batched_molecular_graph(
        edge_index,
        atom_batch,
        num_nodes=atom_attr.shape[0],
        device=atom_attr.device,
        edge_field="chemrl_gem_edge_index",
    )
    edge_batch = atom_batch[edge_index[0]]
    validate_batched_molecular_graph(
        angle_edge_index,
        edge_batch,
        num_nodes=bond_attr.shape[0],
        device=atom_attr.device,
        edge_field="chemrl_gem_angle_edge_index",
    )
    if atom_attr.numel() and (
        atom_attr.min() < 0
        or atom_attr[:, 0].max() >= 124
        or atom_attr[:, 1].max() >= 22
        or atom_attr[:, 2].max() >= 17
        or atom_attr[:, 3].max() >= 9
        or atom_attr[:, 4].max() >= 15
        or atom_attr[:, 5].max() >= 7
        or atom_attr[:, 6].max() >= 13
    ):
        raise ValueError("ChemRL-GEM atom feature index is outside its legacy vocabulary")
    if bond_attr.numel() and (
        bond_attr.min() < 0
        or bond_attr[:, 0].max() >= 12
        or bond_attr[:, 1].max() >= 27
        or bond_attr[:, 2].max() >= 7
    ):
        raise ValueError("ChemRL-GEM bond feature index is outside its legacy vocabulary")
    if not bool(torch.isfinite(bond_length).all()) or not bool(torch.isfinite(bond_angle).all()):
        raise ValueError("ChemRL-GEM geometry features must be finite")


def _positive_int(value: int, name: str) -> None:
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise ValueError(f"{name} must be a positive integer")


def _default_checkpoint(variant: str) -> Path:
    root = Path(__file__).resolve().parents[4]
    filename = "class.pdparams" if variant == "classification" else "regr.pdparams"
    return root / "pretrained" / "chemrl_gem_2022" / "pretrain_models-chemrl_gem" / filename


__all__ = ["ChemRLGEM", "ChemRLGEMEncoder", "GeoGNNEncoder"]
