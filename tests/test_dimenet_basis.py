"""Mathematical invariants for DimeNet's paper-correct basis functions."""

import math

import torch

from molgnn.models.dimenet_2020.basis import (
    CutoffEnvelope,
    RadialBesselBasis,
    SphericalBesselBasis,
    _spherical_bessel,
)


def test_radial_bessel_matches_equation_seven_and_has_a_finite_origin_limit() -> None:
    basis = RadialBesselBasis(2, cutoff=5.0, envelope_p=6)
    distances = torch.tensor([0.0, 1.25, 5.0], dtype=torch.float32)
    actual = basis(distances)

    scaled = distances[1] / 5.0
    envelope = 1 - 28 * scaled**6 + 48 * scaled**7 - 21 * scaled**8
    expected = (
        math.sqrt(2 / 5.0)
        * envelope
        * torch.sin(torch.tensor([math.pi, 2 * math.pi]) * scaled)
        / distances[1]
    )

    assert torch.allclose(actual[1], expected, atol=1e-6)
    assert torch.allclose(
        actual[0], math.sqrt(2 / 5.0) * torch.tensor([math.pi / 5, 2 * math.pi / 5])
    )
    assert torch.equal(actual[2], torch.zeros(2))


def test_spherical_basis_uses_no_extra_inverse_distance_and_vanishes_at_cutoff() -> (
    None
):
    basis = SphericalBesselBasis(2, 2, cutoff=5.0, envelope_p=6)
    near = basis(
        torch.tensor([0.5, 1.0], dtype=torch.float32),
        torch.tensor([0.7, 0.7], dtype=torch.float32),
    )
    cutoff = basis(torch.tensor([5.0], dtype=torch.float32), torch.tensor([0.7]))

    # The l=0 spherical-Bessel component has a finite value at the origin;
    # halving an already-small distance must not generate a spurious 1/r rise.
    assert torch.isfinite(near).all()
    assert near[0, 0].abs() < near[1, 0].abs() * 1.2
    assert torch.equal(cutoff, torch.zeros_like(cutoff))


def test_spherical_basis_is_differentiable_in_distance_and_angle() -> None:
    basis = SphericalBesselBasis(3, 2)
    distances = torch.tensor([0.7, 1.3], requires_grad=True)
    angles = torch.tensor([0.2, 1.1], requires_grad=True)
    output = basis(distances, angles)
    first_distance, first_angle = torch.autograd.grad(
        output.square().sum(), (distances, angles), create_graph=True
    )
    second_distance = torch.autograd.grad(first_distance.sum(), distances)[0]

    assert torch.isfinite(first_distance).all()
    assert torch.isfinite(first_angle).all()
    assert torch.isfinite(second_distance).all()


def test_spherical_basis_direct_cosine_matches_the_angle_representation() -> None:
    basis = SphericalBesselBasis(3, 2)
    distances = torch.tensor([0.7, 1.3], dtype=torch.float32)
    angles = torch.tensor([0.2, 1.1], dtype=torch.float32)

    from_angles = basis(distances, angles)
    from_cosines = basis.forward_from_cosine(distances, torch.cos(angles))

    assert torch.allclose(from_cosines, from_angles, atol=1e-6)


def test_high_order_spherical_bessel_remains_stable_near_the_origin() -> None:
    # A naive upward recurrence produces values of order 10^2 here, although
    # j_6(0.1) is approximately 7.3975e-12.  DimeNet's default seven angular
    # orders rely on the stable branch for short radius edges.
    actual = _spherical_bessel(6, torch.tensor([0.1]))

    assert torch.allclose(actual, torch.tensor([7.397541e-12]), rtol=1e-5)


def test_cutoff_envelope_is_only_the_polynomial_multiplier() -> None:
    envelope = CutoffEnvelope(6)
    values = envelope(torch.tensor([0.0, 0.5, 1.0, 2.0]))

    assert torch.allclose(values, torch.tensor([1.0, 0.85546875, 0.0, 0.0]))
