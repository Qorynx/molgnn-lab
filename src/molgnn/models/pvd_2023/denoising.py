"""Model-owned pre-training-via-denoising lifecycle."""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F
from torch import Tensor, nn
from torch_geometric.data import Batch
from torch_geometric.utils import scatter

from .constants import PVD_PRETRAIN_NOISE_SCALE
from .geometry import build_pvd_radius_graph
from .model import PVDTorchMDET


class PositionNormalizer(nn.Module):
    """Running component-wise normalization used by the released checkpoint."""

    def __init__(self, epsilon: float = 1.0e-8) -> None:
        super().__init__()
        self.epsilon = float(epsilon)
        self.register_buffer("acc_sum", torch.zeros(3))
        self.register_buffer("acc_squared_sum", torch.zeros(3))
        self.register_buffer("acc_count", torch.zeros(1))
        self.register_buffer("num_accumulations", torch.zeros(1))

    @property
    def mean(self) -> Tensor:
        return self.acc_sum / self.acc_count.clamp_min(1.0)

    @property
    def std(self) -> Tensor:
        variance = self.acc_squared_sum / self.acc_count.clamp_min(1.0)
        return (variance - self.mean.square()).sqrt().clamp_min(self.epsilon)

    def update_statistics(self, value: Tensor) -> None:
        with torch.no_grad():
            detached = value.detach()
            self.acc_sum.add_(detached.sum(dim=0))
            self.acc_squared_sum.add_(detached.square().sum(dim=0))
            self.acc_count.add_(detached.shape[0])
            self.num_accumulations.add_(1)

    def forward(self, value: Tensor) -> Tensor:
        if value.ndim != 2 or value.shape[1] != 3:
            raise ValueError("position target must have shape [N, 3]")
        if self.training:
            self.update_statistics(value)
        return (value - self.mean) / self.std


@dataclass(frozen=True)
class PVDPretrainingLoss:
    total: Tensor
    denoising: Tensor
    property: Tensor | None
    noise_prediction: Tensor
    noise_target: Tensor


def sample_position_noise(
    pos: Tensor,
    graph_batch: Tensor,
    *,
    sigma: float,
    centering: str,
    generator: torch.Generator | None = None,
) -> Tensor:
    if sigma <= 0:
        raise ValueError("sigma must be positive")
    if centering not in {"source_raw", "paper_centered"}:
        raise ValueError("centering must be source_raw or paper_centered")
    noise = torch.randn(
        pos.shape,
        dtype=pos.dtype,
        device=pos.device,
        generator=generator,
    ) * sigma
    if centering == "paper_centered":
        graph_count = int(graph_batch.max().item()) + 1
        graph_mean = scatter(
            noise,
            graph_batch,
            dim=0,
            dim_size=graph_count,
            reduce="mean",
        )
        noise = noise - graph_mean[graph_batch]
    return noise


def corrupt_pvd_batch(
    batch: Batch,
    model: PVDTorchMDET,
    *,
    sigma: float = PVD_PRETRAIN_NOISE_SCALE,
    centering: str = "source_raw",
    noise: Tensor | None = None,
    generator: torch.Generator | None = None,
) -> tuple[Batch, Tensor]:
    """Clone, perturb, and rebuild spatial edges exactly after corruption."""

    pos = getattr(batch, "pos", None)
    graph_batch = getattr(batch, "batch", None)
    if not isinstance(pos, Tensor) or not isinstance(graph_batch, Tensor):
        raise ValueError("batch must contain pos and batch tensors")
    if noise is None:
        noise = sample_position_noise(
            pos,
            graph_batch,
            sigma=sigma,
            centering=centering,
            generator=generator,
        )
    elif noise.shape != pos.shape or noise.device != pos.device:
        raise ValueError("provided noise must match pos shape and device")
    noisy = batch.clone()
    noisy.pos = pos + noise
    encoder = model.encoder
    noisy.pvd_edge_index = build_pvd_radius_graph(
        noisy.pos,
        graph_batch,
        cutoff_lower=encoder.cutoff_lower,
        cutoff_upper=encoder.cutoff_upper,
        max_num_neighbors=encoder.max_num_neighbors,
        loop=True,
    )
    return noisy, noise


class PVDPretrainer(nn.Module):
    """Denoising objective kept outside the shared supervised trainer."""

    def __init__(
        self,
        model: PVDTorchMDET,
        *,
        sigma: float = PVD_PRETRAIN_NOISE_SCALE,
        denoising_weight: float = 1.0,
        property_weight: float = 0.0,
        centering: str = "source_raw",
    ) -> None:
        super().__init__()
        if sigma <= 0 or denoising_weight <= 0 or property_weight < 0:
            raise ValueError("invalid denoising/pretraining weights")
        if centering not in {"source_raw", "paper_centered"}:
            raise ValueError("centering must be source_raw or paper_centered")
        self.model = model
        self.sigma = float(sigma)
        self.denoising_weight = float(denoising_weight)
        self.property_weight = float(property_weight)
        self.centering = centering
        self.position_normalizer = PositionNormalizer()

    def compute_loss(
        self,
        batch: Batch,
        *,
        noise: Tensor | None = None,
        generator: torch.Generator | None = None,
    ) -> PVDPretrainingLoss:
        noisy_batch, noise_target = corrupt_pvd_batch(
            batch,
            self.model,
            sigma=self.sigma,
            centering=self.centering,
            noise=noise,
            generator=generator,
        )
        scalar, vector, graph_batch, num_graphs = self.model.encode_batch(noisy_batch)
        noise_prediction = self.model.noise_head(scalar, vector)
        normalized_target = self.position_normalizer(noise_target)
        denoising_loss = F.mse_loss(noise_prediction, normalized_target)

        property_loss: Tensor | None = None
        total = self.denoising_weight * denoising_loss
        if self.property_weight > 0:
            target = getattr(batch, "y", None)
            mask = getattr(batch, "y_mask", None)
            if not isinstance(target, Tensor):
                raise ValueError("property_weight requires batch.y")
            prediction = scatter(
                self.model.property_head(scalar, vector),
                graph_batch,
                dim=0,
                dim_size=num_graphs,
                reduce=self.model.readout,
            )
            if target.ndim == 1:
                target = target.unsqueeze(-1)
            if prediction.shape != target.shape:
                raise ValueError("property prediction and target shapes differ")
            valid = torch.isfinite(target)
            if isinstance(mask, Tensor):
                valid &= mask.to(dtype=torch.bool)
            if not bool(valid.any()):
                raise ValueError("property objective has no valid targets")
            property_loss = F.mse_loss(prediction[valid], target[valid])
            total = total + self.property_weight * property_loss
        return PVDPretrainingLoss(
            total=total,
            denoising=denoising_loss,
            property=property_loss,
            noise_prediction=noise_prediction,
            noise_target=noise_target,
        )


__all__ = [
    "PVDPretrainer",
    "PVDPretrainingLoss",
    "PositionNormalizer",
    "corrupt_pvd_batch",
    "sample_position_noise",
]
