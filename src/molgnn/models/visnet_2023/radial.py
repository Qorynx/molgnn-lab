"""Distance expansions and cutoff used by the AI2BMD ViSNet block."""

from __future__ import annotations

import math

import torch
from torch import Tensor, nn

from .constants import VISNET_CUTOFF, VISNET_NUM_RBF


class CosineCutoff(nn.Module):
    """Smooth compact-support cutoff from the ViSNet source."""

    def __init__(self, cutoff: float = VISNET_CUTOFF) -> None:
        super().__init__()
        if not isinstance(cutoff, (float, int)) or cutoff <= 0:
            raise ValueError("cutoff must be positive")
        self.cutoff = float(cutoff)

    def forward(self, distances: Tensor) -> Tensor:
        if distances.ndim != 1:
            raise ValueError("distances must have shape [E]")
        values = 0.5 * (torch.cos(math.pi * distances / self.cutoff) + 1.0)
        return values * (distances < self.cutoff).to(dtype=distances.dtype)


class ExpNormalSmearing(nn.Module):
    """TorchMD/AI2BMD exponential-normal radial expansion."""

    def __init__(
        self,
        cutoff: float = VISNET_CUTOFF,
        num_rbf: int = VISNET_NUM_RBF,
        trainable: bool = False,
    ) -> None:
        super().__init__()
        _positive_int(num_rbf, "num_rbf")
        if not isinstance(cutoff, (float, int)) or cutoff <= 0:
            raise ValueError("cutoff must be positive")
        self.cutoff = float(cutoff)
        self.num_rbf = int(num_rbf)
        self.trainable = bool(trainable)
        self.alpha = 5.0 / self.cutoff
        self.cutoff_fn = CosineCutoff(self.cutoff)
        means, betas = self._initial_parameters()
        if self.trainable:
            self.means = nn.Parameter(means)
            self.betas = nn.Parameter(betas)
        else:
            self.register_buffer("means", means)
            self.register_buffer("betas", betas)

    def _initial_parameters(self) -> tuple[Tensor, Tensor]:
        start = math.exp(-self.cutoff)
        means = torch.linspace(start, 1.0, self.num_rbf, dtype=torch.float32)
        beta = (2.0 / self.num_rbf * (1.0 - start)) ** -2
        return means, torch.full((self.num_rbf,), beta, dtype=torch.float32)

    def reset_parameters(self) -> None:
        means, betas = self._initial_parameters()
        self.means.data.copy_(means.to(device=self.means.device, dtype=self.means.dtype))
        self.betas.data.copy_(betas.to(device=self.betas.device, dtype=self.betas.dtype))

    def forward(self, distances: Tensor) -> Tensor:
        if distances.ndim != 1:
            raise ValueError("distances must have shape [E]")
        value = distances.unsqueeze(-1)
        means = self.means.to(dtype=distances.dtype)
        betas = self.betas.to(dtype=distances.dtype)
        return self.cutoff_fn(distances).unsqueeze(-1) * torch.exp(
            -betas * (torch.exp(-self.alpha * value) - means).square()
        )


class GaussianSmearing(nn.Module):
    """Optional Gaussian radial profile exposed by the supplied source."""

    def __init__(
        self,
        cutoff: float = VISNET_CUTOFF,
        num_rbf: int = VISNET_NUM_RBF,
        trainable: bool = False,
    ) -> None:
        super().__init__()
        _positive_int(num_rbf, "num_rbf")
        if num_rbf < 2:
            raise ValueError("num_rbf must be at least 2 for Gaussian smearing")
        if not isinstance(cutoff, (float, int)) or cutoff <= 0:
            raise ValueError("cutoff must be positive")
        self.cutoff = float(cutoff)
        self.num_rbf = int(num_rbf)
        self.trainable = bool(trainable)
        offsets, coeff = self._initial_parameters()
        if self.trainable:
            self.offset = nn.Parameter(offsets)
            self.coeff = nn.Parameter(coeff)
        else:
            self.register_buffer("offset", offsets)
            self.register_buffer("coeff", coeff)

    def _initial_parameters(self) -> tuple[Tensor, Tensor]:
        offset = torch.linspace(0.0, self.cutoff, self.num_rbf, dtype=torch.float32)
        coefficient = torch.tensor(
            -0.5 / float((offset[1] - offset[0]).square()), dtype=torch.float32
        )
        return offset, coefficient

    def reset_parameters(self) -> None:
        offset, coeff = self._initial_parameters()
        self.offset.data.copy_(offset.to(device=self.offset.device, dtype=self.offset.dtype))
        self.coeff.data.copy_(coeff.to(device=self.coeff.device, dtype=self.coeff.dtype))

    def forward(self, distances: Tensor) -> Tensor:
        if distances.ndim != 1:
            raise ValueError("distances must have shape [E]")
        offset = self.offset.to(dtype=distances.dtype)
        coeff = self.coeff.to(dtype=distances.dtype)
        return torch.exp(coeff * (distances.unsqueeze(-1) - offset).square())


def build_radial_basis(
    name: str,
    *,
    cutoff: float = VISNET_CUTOFF,
    num_rbf: int = VISNET_NUM_RBF,
    trainable: bool = False,
) -> nn.Module:
    """Build one of the explicitly supported source radial expansions."""

    if name == "expnorm":
        return ExpNormalSmearing(cutoff=cutoff, num_rbf=num_rbf, trainable=trainable)
    if name == "gauss":
        return GaussianSmearing(cutoff=cutoff, num_rbf=num_rbf, trainable=trainable)
    raise ValueError("rbf_type must be either 'expnorm' or 'gauss'")


def _positive_int(value: int, field: str) -> None:
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise ValueError(f"{field} must be a positive integer")


__all__ = [
    "CosineCutoff",
    "ExpNormalSmearing",
    "GaussianSmearing",
    "build_radial_basis",
]
