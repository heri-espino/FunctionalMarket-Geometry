"""Mathematical and numerical invariants of Legendre Hilbert representations."""

import numpy as np
import pytest

from market_geometry.polynomial_geometry import (
    derivative_gram,
    evaluate_legendre,
    orthonormal_design,
    pairwise_polynomial_distance,
    project_onto_legendre,
)


def test_continuous_orthonormality_by_quadrature():
    nodes, weights = np.polynomial.legendre.leggauss(30)
    basis = orthonormal_design((nodes + 1) / 2, 10)
    gram = (basis.T * (weights / 2)) @ basis
    np.testing.assert_allclose(gram, np.eye(11), atol=2e-13)


def test_exact_polynomial_reconstruction():
    u = np.linspace(0, 1, 60)
    original = np.array([[1.3, 0.75, 0.0, -0.25], [-0.3, 0.1, 1.8, 0]])
    sample = evaluate_legendre(original, u)
    computed = project_onto_legendre(sample, 3)
    np.testing.assert_allclose(computed, original, atol=1e-12)


def test_l2_equals_coefficient_euclidean_distance():
    rng = np.random.default_rng(12)
    a = rng.normal(size=(7, 8))
    actual = pairwise_polynomial_distance(a, 0)
    expected = np.linalg.norm(a[:, None, :] - a[None, :, :], axis=-1)
    np.testing.assert_allclose(actual, expected, atol=1e-12)


def test_sobolev_derivative_is_exact_for_linear_function():
    # f(u)=u, f'=1, coefficient vector is (1/2,1/sqrt(12)).
    # Norm f^2 = 1/3, integral derivative^2 = 1.
    a = np.array([[0.5, 1.0 / np.sqrt(12)], [0, 0]])
    actual = pairwise_polynomial_distance(a, 2.0)[0, 1] ** 2
    assert actual == pytest.approx(1 / 3 + 2.0, abs=1e-12)
    assert derivative_gram(0)[0, 0] == pytest.approx(0)


def test_distance_invariance_to_common_coefficient_shift_and_triangle():
    rng = np.random.default_rng(14)
    a = rng.normal(size=(6, 5))
    d = pairwise_polynomial_distance(a, 0.1)
    shift = rng.normal(size=(1, 5))
    np.testing.assert_allclose(pairwise_polynomial_distance(a + shift, 0.1), d)
    assert np.all(d >= 0)
    np.testing.assert_allclose(d, d.T)
    for i in range(6):
        for j in range(6):
            for k in range(6):
                assert d[i, j] <= d[i, k] + d[k, j] + 1e-12


def test_polynomial_degree_is_finite_and_fixed():
    with pytest.raises(ValueError):
        project_onto_legendre(np.ones((2, 7)), 7)
    with pytest.raises(ValueError):
        orthonormal_design([1.2], 2)
    with pytest.raises(ValueError):
        pairwise_polynomial_distance(np.ones((2, 2)), -1)
