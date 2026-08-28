"""Radial basis and cutoff functions from the released TorchMD-ET source."""

from __future__ import annotations

import math

import torch
from torch import Tensor, nn


class CosineCutoff(nn.Module):
    def __init__(self, cutoff_lower: float = 0.0, cutoff_upper: float = 5.0) -> None:
        super().__init__()
        self.cutoff_lower = float(cutoff_lower)
        self.cutoff_upper = float(cutoff_upper)

    def forward(self, distances: Tensor) -> Tensor:
        if self.cutoff_lower > 0:
            cutoffs = 0.5 * (
                torch.cos(
                    math.pi
                    * (
                        2.0
                        * (distances - self.cutoff_lower)
                        / (self.cutoff_upper - self.cutoff_lower)
                        + 1.0
                    )
                )
                + 1.0
            )
            return (
                cutoffs
                * (distances < self.cutoff_upper).to(distances.dtype)
                * (distances > self.cutoff_lower).to(distances.dtype)
            )
        cutoffs = 0.5 * (
            torch.cos(distances * math.pi / self.cutoff_upper) + 1.0
        )
        return cutoffs * (distances < self.cutoff_upper).to(distances.dtype)


class ExpNormalSmearing(nn.Module):
    """PhysNet-style exponential-normal radial basis used by the checkpoint."""

    def __init__(
        self,
        cutoff_lower: float = 0.0,
        cutoff_upper: float = 5.0,
        num_rbf: int = 64,
        trainable: bool = False,
    ) -> None:
        super().__init__()
        if num_rbf < 1:
            raise ValueError("num_rbf must be positive")
        self.cutoff_lower = float(cutoff_lower)
        self.cutoff_upper = float(cutoff_upper)
        self.num_rbf = int(num_rbf)
        self.trainable = bool(trainable)
        self.cutoff_fn = CosineCutoff(0.0, cutoff_upper)
        self.alpha = 5.0 / (cutoff_upper - cutoff_lower)
        means, betas = self._initial_params()
        if trainable:
            self.means = nn.Parameter(means)
            self.betas = nn.Parameter(betas)
        else:
            self.register_buffer("means", means)
            self.register_buffer("betas", betas)

    def _initial_params(self) -> tuple[Tensor, Tensor]:
        start_value = torch.exp(
            torch.scalar_tensor(-self.cutoff_upper + self.cutoff_lower)
        )
        means = torch.linspace(start_value, 1.0, self.num_rbf)
        beta = (2.0 / self.num_rbf * (1.0 - start_value)) ** -2
        return means, torch.ones(self.num_rbf) * beta

    def reset_parameters(self) -> None:
        means, betas = self._initial_params()
        with torch.no_grad():
            self.means.copy_(means)
            self.betas.copy_(betas)

    def forward(self, distances: Tensor) -> Tensor:
        expanded = distances.unsqueeze(-1)
        return self.cutoff_fn(expanded) * torch.exp(
            -self.betas
            * (
                torch.exp(
                    self.alpha * (-expanded + self.cutoff_lower)
                )
                - self.means
            ).square()
        )


__all__ = ["CosineCutoff", "ExpNormalSmearing"]
