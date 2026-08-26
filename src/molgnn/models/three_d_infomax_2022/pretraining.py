"""Paired 2D/3D contrastive pretraining for 3D Infomax.

Everything here is model-owned: the collator builds complete directed
conformer graphs and validates atom alignment, the pretrainer mirrors the
official module layout (``node_gnn.*`` plus the pretraining readout head
``output.*``), and the tiny epoch/save/resume helpers stay outside the
shared Trainer lifecycle.

Conformer policy follows the audit plan: when a molecule owns fewer real
conformers than ``num_conformers``, the lowest-energy conformer is repeated
without noise (paper-faithful); the official Gaussian noise on duplicated
conformers is available opt-in through ``legacy_repeat_noise``.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import torch
from torch import Tensor, nn
from torch_geometric.data import Batch

from .layers import PNAGNN
from .model import ThreeDInfomax
from .net3d import Net3D
from .objectives import multi_positive_infomax_loss


@dataclass(frozen=True)
class PairedConformerBatch:
    """Model-owned paired batch consumed by :class:`ThreeDInfomaxPretrainer`."""

    batch_2d: Batch
    positions: Tensor  # [total_atoms_over_conformers, 3]
    complete_edge_index: Tensor  # [2, total_pair_edges]
    conformer_owner: Tensor  # [B * C]
    conformer_node_batch: Tensor  # [total_atoms_over_conformers]

    def to(self, device: torch.device | str) -> PairedConformerBatch:
        return PairedConformerBatch(
            batch_2d=self.batch_2d.to(device),
            positions=self.positions.to(device),
            complete_edge_index=self.complete_edge_index.to(device),
            conformer_owner=self.conformer_owner.to(device),
            conformer_node_batch=self.conformer_node_batch.to(device),
        )


def _complete_directed_edges(num_nodes: int, offset: int) -> Tensor:
    arange = torch.arange(num_nodes)
    sources = torch.repeat_interleave(arange, num_nodes - 1)
    destinations = torch.cat(
        [
            torch.cat([arange[:index], arange[index + 1 :]])
            for index in range(num_nodes)
        ]
    )
    return torch.stack((sources + offset, destinations + offset), dim=0)


def build_paired_conformer_batch(
    samples: list,
    conformer_sets: list[Tensor],
    *,
    num_conformers: int,
    legacy_repeat_noise: bool = False,
    generator: torch.Generator | None = None,
) -> PairedConformerBatch:
    """Collate molecules with their conformer coordinate sets.

    ``samples[i]`` must already carry the transformed
    ``three_d_infomax_*`` fields; ``conformer_sets[i]`` is a ``[K_i >= 1,
    n_i, 3]`` tensor ordered by ascending energy. The first ``C`` entries are
    used, repeating the lowest-energy conformer when ``K_i < C``.
    """

    if len(samples) != len(conformer_sets):
        raise ValueError("samples and conformer sets must align")
    if num_conformers < 1:
        raise ValueError("num_conformers must be >= 1")
    if len(samples) < 2:
        raise ValueError("paired pretraining requires at least two molecules per batch")

    batch_2d = Batch.from_data_list(samples)

    positions_blocks: list[Tensor] = []
    edge_blocks: list[Tensor] = []
    owners: list[int] = []
    node_batch_blocks: list[Tensor] = []
    atom_offset = 0
    for molecule_index, (sample, conformers) in enumerate(zip(samples, conformer_sets, strict=True)):
        atom_count = int(sample.x.shape[0])
        if conformers.ndim != 3 or conformers.shape[1] != atom_count:
            raise ValueError(
                f"molecule {molecule_index}: conformers must have shape [K, {atom_count}, 3]"
            )
        if not torch.isfinite(conformers).all():
            raise ValueError(f"molecule {molecule_index}: conformer coordinates must be finite")

        atomic_reference = sample.three_d_infomax_atom_attr[:, 0]
        expected_atomic = _atomic_from_canonical(sample)
        if not torch.equal(atomic_reference, expected_atomic):
            raise ValueError(
                f"molecule {molecule_index}: categorical atoms disagree with the canonical graph"
            )

        selected: list[Tensor] = []
        for conformer_index in range(num_conformers):
            source = (
                conformers[conformer_index]
                if conformer_index < conformers.shape[0]
                else conformers[0]
            )
            chosen = source.detach().clone()
            if (
                legacy_repeat_noise
                and conformer_index > 0
                and bool(torch.equal(source, conformers[0]))
            ):
                chosen = chosen + torch.randn(
                    chosen.shape,
                    generator=generator,
                    dtype=chosen.dtype,
                    device=chosen.device,
                ) * 0.05
            selected.append(chosen)
            positions_blocks.append(chosen)
            edge_blocks.append(_complete_directed_edges(atom_count, atom_offset))
            node_batch_blocks.append(torch.full((atom_count,), len(owners), dtype=torch.long))
            owners.append(molecule_index)
            atom_offset += atom_count

    return PairedConformerBatch(
        batch_2d=batch_2d,
        positions=torch.cat(positions_blocks, dim=0),
        complete_edge_index=torch.cat(edge_blocks, dim=1),
        conformer_owner=torch.tensor(owners, dtype=torch.long),
        conformer_node_batch=torch.cat(node_batch_blocks, dim=0),
    )


def _atomic_from_canonical(sample) -> Tensor:
    from ...transforms.ogb_categorical import canonical_atomic_ids

    return canonical_atomic_ids(sample)


class PNA2D(nn.Module):
    """Official pretraining PNA wrapper: ``node_gnn`` plus readout head."""

    def __init__(
        self,
        *,
        pretrain_target_dim: int = 256,
        readout_aggregators: tuple[str, ...] = ("min", "max", "mean"),
        readout_hidden_dim: int | None = None,
        readout_layers: int = 2,
        readout_batchnorm: bool = True,
        batch_norm_momentum: float = 0.93,
        **encoder_kwargs: object,
    ) -> None:
        super().__init__()
        official_encoder_defaults: dict[str, object] = {
            "hidden_dim": 200,
            "aggregators": ("mean", "max", "min", "std"),
            "scalers": ("identity", "amplification", "attenuation"),
            "residual": True,
            "mid_batch_norm": True,
            "last_batch_norm": True,
            "propagation_depth": 7,
            "dropout": 0.0,
            "posttrans_layers": 1,
            "pretrans_layers": 2,
        }
        for name, value in official_encoder_defaults.items():
            encoder_kwargs.setdefault(name, value)
        self.node_gnn = PNAGNN(  # type: ignore[arg-type]
            batch_norm_momentum=batch_norm_momentum,
            **encoder_kwargs,
        )
        hidden_dim = int(encoder_kwargs["hidden_dim"])  # type: ignore[arg-type]
        self.readout_aggregators = tuple(readout_aggregators)
        from .layers import MLP

        self.output = MLP(
            in_dim=hidden_dim * len(self.readout_aggregators),
            hidden_size=hidden_dim if readout_hidden_dim is None else int(readout_hidden_dim),  # type: ignore[arg-type]
            out_dim=int(pretrain_target_dim),
            mid_batch_norm=readout_batchnorm,
            layers=readout_layers,
            batch_norm_momentum=batch_norm_momentum,
        )

    def forward(self, batch_2d: Batch) -> Tensor:
        node_features = self.node_gnn(
            batch_2d.three_d_infomax_atom_attr,
            batch_2d.edge_index,
            batch_2d.three_d_infomax_bond_attr,
        )
        statistics_total = node_features.new_zeros(
            (int(batch_2d.batch.max().item()) + 1, node_features.shape[1])
        )
        statistics_total.index_add_(0, batch_2d.batch, node_features)
        counts = (
            torch.bincount(batch_2d.batch, minlength=statistics_total.shape[0])
            .unsqueeze(-1)
            .to(node_features.dtype)
        )
        minimum = node_features.new_full(statistics_total.shape, float("inf"))
        minimum.scatter_reduce_(
            0,
            batch_2d.batch.unsqueeze(-1).expand_as(node_features),
            node_features,
            reduce="amin",
            include_self=True,
        )
        maximum = node_features.new_full(statistics_total.shape, float("-inf"))
        maximum.scatter_reduce_(
            0,
            batch_2d.batch.unsqueeze(-1).expand_as(node_features),
            node_features,
            reduce="amax",
            include_self=True,
        )
        stats = {
            "mean": statistics_total / counts.clamp_min(1.0),
            "min": minimum,
            "max": maximum,
            "sum": statistics_total,
        }
        readout = torch.cat([stats[name] for name in self.readout_aggregators], dim=-1)
        return self.output(readout)


class ThreeDInfomaxPretrainer(nn.Module):
    """PNA + Net3D pair trained with the multi-positive objective."""

    def __init__(
        self,
        *,
        pna: PNA2D | None = None,
        net3d: Net3D | None = None,
        tau: float = 0.1,
        **pna_kwargs: object,
    ) -> None:
        super().__init__()
        if pna is None:
            pna_kwargs.pop("pretrain_target_dim", None)
            pna = PNA2D(pretrain_target_dim=256, **pna_kwargs)  # type: ignore[arg-type]
        self.pna = pna
        if net3d is None:
            net3d = Net3D()
        self.net3d = net3d
        self.tau = tau

    def forward(self, batch: PairedConformerBatch) -> tuple[Tensor, Tensor, Tensor]:
        z2d = self.pna(batch.batch_2d)
        z3d = self.net3d(
            batch.positions,
            batch.complete_edge_index,
            batch.conformer_node_batch,
            int(batch.conformer_owner.shape[0]),
        )
        loss = multi_positive_infomax_loss(z2d, z3d, batch.conformer_owner, tau=self.tau)
        return loss, z2d, z3d

    @torch.no_grad()
    def export_encoder_state_dict(self) -> dict[str, Tensor]:
        """Encoder-only state matching a downstream :class:`ThreeDInfomax`.

        The pretraining projection head is dropped; keys are re-rooted from
        ``node_gnn.*`` onto the downstream ``node_gnn.*`` attribute so
        ``load_state_dict`` matches strictly.
        """

        return {key: value.detach().cpu().clone() for key, value in self.pna.node_gnn.state_dict().items()}


def train_pretraining_epoch(
    pretrainer: ThreeDInfomaxPretrainer,
    batches: list[PairedConformerBatch],
    optimizer: torch.optim.Optimizer,
    device: torch.device | str = "cpu",
) -> dict[str, float]:
    """Run one tiny paired-pretraining epoch over explicit batches."""

    pretrainer.train()
    target_device = torch.device(device)
    losses = []
    for batch in batches:
        moved = batch.to(target_device)
        optimizer.zero_grad(set_to_none=True)
        loss, _, _ = pretrainer(moved)
        loss.backward()
        optimizer.step()
        losses.append(float(loss.item()))
    if not losses:
        raise ValueError("pretraining epoch requires at least one batch")
    return {"loss": sum(losses) / len(losses)}


def save_pretraining_checkpoint(
    path: str | Path,
    pretrainer: ThreeDInfomaxPretrainer,
    *,
    optimizer: torch.optim.Optimizer | None = None,
    step: int = 0,
) -> Path:
    payload: dict[str, object] = {
        "format_version": 1,
        "step": int(step),
        "pna": pretrainer.pna.state_dict(),
        "net3d": pretrainer.net3d.state_dict(),
    }
    if optimizer is not None:
        payload["optimizer"] = optimizer.state_dict()
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, target)
    return target


def resume_pretraining_checkpoint(
    path: str | Path,
    pretrainer: ThreeDInfomaxPretrainer,
    optimizer: torch.optim.Optimizer | None = None,
) -> dict[str, object]:
    payload = torch.load(path, map_location="cpu", weights_only=True)
    if not isinstance(payload, dict) or "pna" not in payload or "net3d" not in payload:
        raise ValueError(f"not a 3D Infomax pretraining checkpoint: {path}")
    pretrainer.pna.load_state_dict(payload["pna"])
    pretrainer.net3d.load_state_dict(payload["net3d"])
    if optimizer is not None and "optimizer" in payload:
        optimizer.load_state_dict(payload["optimizer"])
    return {"step": int(payload.get("step", 0))}


__all__ = [
    "PNA2D",
    "PairedConformerBatch",
    "ThreeDInfomaxPretrainer",
    "build_paired_conformer_batch",
    "export_downstream_encoder",
    "resume_pretraining_checkpoint",
    "save_pretraining_checkpoint",
    "train_pretraining_epoch",
]


def export_downstream_encoder(model: ThreeDInfomax) -> dict[str, Tensor]:
    """Encoder state of a downstream predictor (mirror of the export path)."""

    return {key: value.detach().cpu().clone() for key, value in model.node_gnn.state_dict().items()}
