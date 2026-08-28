"""MXMNet-local radial and spherical basis functions.

The author code uses DimeNet's inverse-distance polynomial envelope, but its
normalization and call sites differ from this project's DimeNet profile.  The
construction is therefore kept local while the forward path remains pure
PyTorch and differentiable with respect to coordinates.
"""

from __future__ import annotations

import math
from functools import lru_cache

import torch
from torch import Tensor, nn


class MXMNetEnvelope(nn.Module):
    """Inverse-distance polynomial cutoff used by the official MXMNet code."""

    def __init__(self, exponent: int) -> None:
        super().__init__()
        _positive_int(exponent, "exponent")
        self.exponent = exponent
        self.a = -(exponent + 1) * (exponent + 2) / 2
        self.b = exponent * (exponent + 2)
        self.c = -exponent * (exponent + 1) / 2

    def forward(self, scaled_distances: Tensor) -> Tensor:
        _validate_positive_distances(scaled_distances, "scaled_distances")
        p = self.exponent
        x_p = scaled_distances.pow(p)
        x_p1 = x_p * scaled_distances
        values = (
            scaled_distances.reciprocal()
            + self.a * x_p
            + self.b * x_p1
            + self.c * x_p1 * scaled_distances
        )
        return torch.where(scaled_distances < 1, values, torch.zeros_like(values))


class MXMNetRadialBasis(nn.Module):
    """Trainable sine basis multiplied by MXMNet's cutoff envelope."""

    def __init__(self, num_radial: int, cutoff: float, exponent: int) -> None:
        super().__init__()
        _positive_int(num_radial, "num_radial")
        _positive_float(cutoff, "cutoff")
        self.num_radial = num_radial
        self.cutoff = float(cutoff)
        self.envelope = MXMNetEnvelope(exponent)
        self.frequencies = nn.Parameter(torch.empty(num_radial))
        self.reset_parameters()

    def reset_parameters(self) -> None:
        with torch.no_grad():
            self.frequencies.copy_(
                torch.arange(
                    1,
                    self.num_radial + 1,
                    dtype=self.frequencies.dtype,
                    device=self.frequencies.device,
                )
                * math.pi
            )

    def forward(self, distances: Tensor) -> Tensor:
        _validate_positive_distances(distances, "distances")
        scaled = distances / self.cutoff
        frequencies = self.frequencies.to(dtype=distances.dtype)
        return self.envelope(scaled).unsqueeze(-1) * torch.sin(
            scaled.unsqueeze(-1) * frequencies
        )


class MXMNetSphericalBasis(nn.Module):
    """Normalized spherical-Bessel/real-harmonic basis from the source."""

    def __init__(
        self,
        num_spherical: int,
        num_radial: int,
        cutoff: float,
        exponent: int,
    ) -> None:
        super().__init__()
        _positive_int(num_spherical, "num_spherical")
        _positive_int(num_radial, "num_radial")
        _positive_float(cutoff, "cutoff")
        self.num_spherical = num_spherical
        self.num_radial = num_radial
        self.cutoff = float(cutoff)
        self.envelope = MXMNetEnvelope(exponent)
        roots, normalizers = _cached_roots(num_spherical, num_radial)
        self.register_buffer("roots", roots.clone())
        self.register_buffer("normalizers", normalizers.clone())

    @property
    def output_dim(self) -> int:
        return self.num_spherical * self.num_radial

    def forward_from_cosine(self, distances: Tensor, cosines: Tensor) -> Tensor:
        _validate_positive_distances(distances, "distances")
        if (
            not isinstance(cosines, Tensor)
            or cosines.shape != distances.shape
            or not torch.is_floating_point(cosines)
        ):
            raise ValueError("cosines must be a floating tensor matching distances")
        if cosines.device != distances.device or not bool(
            torch.isfinite(cosines).all()
        ):
            raise ValueError("cosines must be finite and share the distance device")

        scaled = distances / self.cutoff
        envelope = self.envelope(scaled).unsqueeze(-1)
        blocks: list[Tensor] = []
        for order in range(self.num_spherical):
            roots = self.roots[order].to(dtype=distances.dtype)
            radial = _spherical_bessel(order, scaled.unsqueeze(-1) * roots)
            normalizer = self.normalizers[order].to(dtype=distances.dtype)
            angular = _real_spherical_harmonic_m0(order, cosines).unsqueeze(-1)
            blocks.append(envelope * normalizer * radial * angular)
        return torch.cat(blocks, dim=-1)


@lru_cache(maxsize=16)
def _cached_roots(num_spherical: int, num_radial: int) -> tuple[Tensor, Tensor]:
    try:
        import numpy as np
        from scipy.optimize import brentq
        from scipy.special import spherical_jn
    except ImportError as exc:  # pragma: no cover - optional construction extra
        raise RuntimeError(
            "MXMNet requires SciPy when constructing its spherical basis; "
            "install molgnn-lab[mxmnet]."
        ) from exc

    roots = np.empty((num_spherical, num_radial), dtype=np.float64)
    normalizers = np.empty_like(roots)
    for order in range(num_spherical):
        if order == 0:
            order_roots = np.arange(1, num_radial + 1, dtype=np.float64) * math.pi
        else:
            order_roots = _positive_roots(order, num_radial, brentq, spherical_jn, np)
        roots[order] = order_roots
        next_values = spherical_jn(order + 1, order_roots)
        # This is the source's unit-interval normalization.
        normalizers[order] = np.sqrt(2.0 / np.square(next_values))
    return (
        torch.from_numpy(roots.astype(np.float32)),
        torch.from_numpy(normalizers.astype(np.float32)),
    )


def _positive_roots(
    order: int,
    count: int,
    brentq: object,
    spherical_jn: object,
    np: object,
):
    upper = (count + order / 2 + 2) * math.pi
    roots: list[float] = []
    while len(roots) < count:
        grid = np.linspace(1e-7, upper, max(1_024, math.ceil(upper * 64)))
        values = spherical_jn(order, grid)
        changes = np.flatnonzero(values[:-1] * values[1:] < 0)
        roots = []
        for index in changes:
            root = float(
                brentq(
                    lambda value: spherical_jn(order, value),
                    float(grid[index]),
                    float(grid[index + 1]),
                )
            )
            if not roots or abs(root - roots[-1]) > 1e-8:
                roots.append(root)
            if len(roots) == count:
                break
        if len(roots) < count:
            upper *= 2
    return np.asarray(roots, dtype=np.float64)


def _spherical_bessel(order: int, values: Tensor) -> Tensor:
    small = values.abs() < 0.5
    series = _spherical_bessel_series(order, values)
    if values.numel() == 0:
        return values
    calculation_dtype = torch.float64 if values.dtype == torch.float32 else values.dtype
    x = values.to(dtype=calculation_dtype)
    safe_x = torch.where(small, torch.ones_like(x), x)
    start_order = max(
        order + 32, int(torch.ceil(safe_x.detach().abs().max()).item()) + 32
    )
    following = torch.zeros_like(safe_x)
    current = torch.ones_like(safe_x)
    selected: Tensor | None = current if order == start_order else None
    for degree in range(start_order, 0, -1):
        previous = (2 * degree + 1) * current / safe_x - following
        if degree - 1 == order:
            selected = previous
        following, current = current, previous
    assert selected is not None
    regular = selected * (torch.sin(safe_x) / safe_x) / current
    return torch.where(small, series, regular.to(dtype=values.dtype))


def _spherical_bessel_series(order: int, values: Tensor) -> Tensor:
    term = values.pow(order) / _odd_double_factorial(2 * order + 1)
    result = term
    squared = values.square()
    for index in range(1, 12):
        term = term * (-squared / (2 * index * (2 * order + 2 * index + 1)))
        result = result + term
    return result


def _real_spherical_harmonic_m0(order: int, cosine: Tensor) -> Tensor:
    if order == 0:
        legendre = torch.ones_like(cosine)
    elif order == 1:
        legendre = cosine
    else:
        previous = torch.ones_like(cosine)
        current = cosine
        for degree in range(2, order + 1):
            following = (
                (2 * degree - 1) * cosine * current - (degree - 1) * previous
            ) / degree
            previous, current = current, following
        legendre = current
    return math.sqrt((2 * order + 1) / (4 * math.pi)) * legendre


def _odd_double_factorial(value: int) -> float:
    result = 1
    for factor in range(value, 0, -2):
        result *= factor
    return float(result)


def _validate_positive_distances(values: Tensor, name: str) -> None:
    if (
        not isinstance(values, Tensor)
        or values.ndim != 1
        or not torch.is_floating_point(values)
    ):
        raise ValueError(f"{name} must have shape [Q] and be floating")
    if not bool(torch.isfinite(values).all()) or bool((values <= 0).any()):
        raise ValueError(f"{name} must contain only finite positive values")


def _positive_int(value: int, name: str) -> None:
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise ValueError(f"{name} must be a positive integer")


def _positive_float(value: float, name: str) -> None:
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(float(value))
        or float(value) <= 0
    ):
        raise ValueError(f"{name} must be a finite positive number")


__all__ = ["MXMNetEnvelope", "MXMNetRadialBasis", "MXMNetSphericalBasis"]
