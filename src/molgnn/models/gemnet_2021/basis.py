"""Fourier--Bessel and real spherical bases for GemNet."""

from __future__ import annotations

import math

import torch
from torch import Tensor, nn

from .constants import GEMNET_ENVELOPE_P


class PolynomialEnvelope(nn.Module):
    """Compact polynomial envelope used by the GemNet paper."""

    def __init__(self, exponent: int = GEMNET_ENVELOPE_P) -> None:
        super().__init__()
        _positive_int(exponent, "exponent")
        self.exponent = exponent
        self.a = -(exponent + 1) * (exponent + 2) / 2
        self.b = exponent * (exponent + 2)
        self.c = -exponent * (exponent + 1) / 2

    def forward(self, scaled_distance: Tensor) -> Tensor:
        value = (
            1
            + self.a * scaled_distance.pow(self.exponent)
            + self.b * scaled_distance.pow(self.exponent + 1)
            + self.c * scaled_distance.pow(self.exponent + 2)
        )
        return torch.where(scaled_distance < 1, value, torch.zeros_like(value))


class RadialBasis(nn.Module):
    """Trainable sine/Bessel edge basis from the reference profile."""

    def __init__(self, num_radial: int, cutoff: float) -> None:
        super().__init__()
        _positive_int(num_radial, "num_radial")
        _positive_float(cutoff, "cutoff")
        self.num_radial = num_radial
        self.cutoff = float(cutoff)
        self.envelope = PolynomialEnvelope()
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
        _validate_distances(distances)
        scaled = distances / self.cutoff
        arguments = scaled[:, None] * self.frequencies.to(distances.dtype)
        safe = distances.clamp_min(1.0e-8)
        regular = torch.sin(arguments) / safe[:, None]
        limit = self.frequencies.to(distances.dtype)[None, :] / self.cutoff
        radial = torch.where((distances < 1.0e-5)[:, None], limit, regular)
        return (
            math.sqrt(2 / self.cutoff)
            * self.envelope(scaled)[:, None]
            * radial
        )


class CircularBasis(nn.Module):
    """Decomposed radial and real ``m=0`` angular basis."""

    def __init__(self, num_spherical: int, num_radial: int, cutoff: float) -> None:
        super().__init__()
        _positive_int(num_spherical, "num_spherical")
        _positive_int(num_radial, "num_radial")
        _positive_float(cutoff, "cutoff")
        self.num_spherical = num_spherical
        self.num_radial = num_radial
        self.cutoff = float(cutoff)
        self.envelope = PolynomialEnvelope()
        roots, normalizers = _spherical_bessel_roots(
            num_spherical, num_radial, self.cutoff
        )
        self.register_buffer("roots", roots)
        self.register_buffer("normalizers", normalizers)

    @property
    def output_dim(self) -> int:
        return self.num_spherical * self.num_radial

    def radial_components(self, distances: Tensor) -> Tensor:
        _validate_distances(distances)
        scaled = distances / self.cutoff
        envelope = self.envelope(scaled)[:, None]
        blocks: list[Tensor] = []
        for order in range(self.num_spherical):
            arguments = scaled[:, None] * self.roots[order].to(distances.dtype)
            radial = _spherical_bessel(order, arguments)
            blocks.append(
                envelope
                * radial
                * self.normalizers[order].to(distances.dtype)[None, :]
            )
        return torch.stack(blocks, dim=1)

    def angular_components(self, cosines: Tensor) -> Tensor:
        return torch.stack(
            [_real_harmonic_m0(order, cosines) for order in range(self.num_spherical)],
            dim=-1,
        )

    def flattened(self, distances: Tensor, cosines: Tensor) -> Tensor:
        if distances.shape != cosines.shape:
            raise ValueError("distances and cosines must have matching shapes")
        radial = self.radial_components(distances)
        angular = self.angular_components(cosines)[:, :, None]
        return (radial * angular).flatten(1)


class TensorBasis(nn.Module):
    """Full real ``(l,m)`` basis for GemNet-Q quadruplets."""

    def __init__(self, num_spherical: int, num_radial: int, cutoff: float) -> None:
        super().__init__()
        self.circular = CircularBasis(num_spherical, num_radial, cutoff)
        self.num_spherical = num_spherical
        self.num_radial = num_radial

    @property
    def num_orders(self) -> int:
        return self.num_spherical**2

    def radial_components(self, distances: Tensor) -> Tensor:
        by_degree = self.circular.radial_components(distances)
        blocks = [
            by_degree[:, degree : degree + 1].expand(-1, 2 * degree + 1, -1)
            for degree in range(self.num_spherical)
        ]
        return torch.cat(blocks, dim=1)

    def angular_components(
        self,
        polar_cosine: Tensor,
        azimuth_cosine: Tensor,
        azimuth_sine: Tensor,
    ) -> Tensor:
        return _real_spherical_harmonics(
            polar_cosine,
            azimuth_cosine,
            azimuth_sine,
            self.num_spherical,
        )


def _real_spherical_harmonics(
    polar_cosine: Tensor,
    azimuth_cosine: Tensor,
    azimuth_sine: Tensor,
    num_spherical: int,
) -> Tensor:
    if polar_cosine.shape != azimuth_cosine.shape or polar_cosine.shape != azimuth_sine.shape:
        raise ValueError("spherical-angle tensors must have matching shapes")
    cos_multiple = [torch.ones_like(azimuth_cosine)]
    sin_multiple = [torch.zeros_like(azimuth_sine)]
    for _ in range(1, num_spherical):
        cos_multiple.append(
            cos_multiple[-1] * azimuth_cosine - sin_multiple[-1] * azimuth_sine
        )
        sin_multiple.append(
            sin_multiple[-1] * azimuth_cosine + cos_multiple[-2] * azimuth_sine
        )

    values: list[Tensor] = []
    for degree in range(num_spherical):
        for order in range(-degree, degree + 1):
            absolute_order = abs(order)
            legendre = _associated_legendre(degree, absolute_order, polar_cosine)
            normalizer = math.sqrt(
                (2 * degree + 1)
                / (4 * math.pi)
                * math.factorial(degree - absolute_order)
                / math.factorial(degree + absolute_order)
            )
            if order < 0:
                value = math.sqrt(2) * normalizer * legendre * sin_multiple[absolute_order]
            elif order > 0:
                value = math.sqrt(2) * normalizer * legendre * cos_multiple[absolute_order]
            else:
                value = normalizer * legendre
            values.append(value)
    if not values:
        return polar_cosine.new_empty((polar_cosine.shape[0], 0))
    return torch.stack(values, dim=-1)


def _associated_legendre(degree: int, order: int, cosine: Tensor) -> Tensor:
    sine = torch.sqrt((1 - cosine.square()).clamp_min(0))
    p_mm = torch.ones_like(cosine)
    factor = 1.0
    for _ in range(order):
        p_mm = -factor * sine * p_mm
        factor += 2.0
    if degree == order:
        return p_mm
    p_m1m = (2 * order + 1) * cosine * p_mm
    if degree == order + 1:
        return p_m1m
    previous, current = p_mm, p_m1m
    for current_degree in range(order + 2, degree + 1):
        following = (
            (2 * current_degree - 1) * cosine * current
            - (current_degree + order - 1) * previous
        ) / (current_degree - order)
        previous, current = current, following
    return current


def _real_harmonic_m0(order: int, cosine: Tensor) -> Tensor:
    return math.sqrt((2 * order + 1) / (4 * math.pi)) * _associated_legendre(
        order, 0, cosine
    )


def _spherical_bessel_roots(
    num_spherical: int, num_radial: int, cutoff: float
) -> tuple[Tensor, Tensor]:
    try:
        import numpy as np
        from scipy.optimize import brentq
        from scipy.special import spherical_jn
    except ImportError as exc:  # pragma: no cover - environment dependent
        raise RuntimeError(
            "GemNet requires SciPy when constructing its spherical basis; "
            "install molgnn-lab[gemnet]."
        ) from exc

    roots = np.empty((num_spherical, num_radial), dtype=np.float64)
    normalizers = np.empty_like(roots)
    for order in range(num_spherical):
        if order == 0:
            order_roots = np.arange(1, num_radial + 1, dtype=np.float64) * math.pi
        else:
            upper = (num_radial + order / 2 + 2) * math.pi
            found: list[float] = []
            while len(found) < num_radial:
                grid = np.linspace(1.0e-7, upper, max(1024, math.ceil(upper * 64)))
                values = spherical_jn(order, grid)
                changes = np.flatnonzero(values[:-1] * values[1:] < 0)
                found = [
                    float(
                        brentq(
                            lambda value, current_order=order: spherical_jn(
                                current_order, value
                            ),
                            grid[i],
                            grid[i + 1],
                        )
                    )
                    for i in changes[:num_radial]
                ]
                upper *= 2
            order_roots = np.asarray(found, dtype=np.float64)
        roots[order] = order_roots
        normalizers[order] = np.sqrt(
            2 / (cutoff**3 * np.square(spherical_jn(order + 1, order_roots)))
        )
    return torch.from_numpy(roots.astype("float32")), torch.from_numpy(
        normalizers.astype("float32")
    )


def _spherical_bessel(order: int, values: Tensor) -> Tensor:
    small = values.abs() < 0.5
    series = _spherical_bessel_series(order, values)
    if values.numel() == 0:
        return values
    calculation = values.to(torch.float64 if values.dtype == torch.float32 else values.dtype)
    safe = torch.where(small, torch.ones_like(calculation), calculation)
    start_order = max(order + 32, int(torch.ceil(safe.detach().abs().max()).item()) + 32)
    following = torch.zeros_like(safe)
    current = torch.ones_like(safe)
    selected: Tensor | None = current if order == start_order else None
    for degree in range(start_order, 0, -1):
        previous = (2 * degree + 1) * current / safe - following
        if degree - 1 == order:
            selected = previous
        following, current = current, previous
    assert selected is not None
    regular = selected * (torch.sin(safe) / safe) / current
    return torch.where(small, series, regular.to(values.dtype))


def _spherical_bessel_series(order: int, values: Tensor) -> Tensor:
    term = values.pow(order) / _odd_double_factorial(2 * order + 1)
    result = term
    squared = values.square()
    for index in range(1, 12):
        term = term * (-squared / (2 * index * (2 * order + 2 * index + 1)))
        result = result + term
    return result


def _odd_double_factorial(value: int) -> float:
    result = 1
    for factor in range(value, 0, -2):
        result *= factor
    return float(result)


def _validate_distances(distances: Tensor) -> None:
    if distances.ndim != 1 or not torch.is_floating_point(distances):
        raise ValueError("distances must be a one-dimensional floating tensor")
    if not bool(torch.isfinite(distances).all()) or bool((distances < 0).any()):
        raise ValueError("distances must contain finite non-negative values")


def _positive_int(value: int, name: str) -> None:
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise ValueError(f"{name} must be a positive integer")


def _positive_float(value: float, name: str) -> None:
    if isinstance(value, bool) or not isinstance(value, (int, float)) or value <= 0:
        raise ValueError(f"{name} must be positive")


__all__ = ["CircularBasis", "PolynomialEnvelope", "RadialBasis", "TensorBasis"]
