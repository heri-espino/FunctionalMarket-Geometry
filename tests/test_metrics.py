"""Unit tests for well-defined metric properties and intended caveats."""

import numpy as np
import pytest

from market_geometry import SplineSettings, correlation_distance, spline_metric, standardize_windows


def test_discrete_l2_equals_correlation_distance():
    rng = np.random.default_rng(21)
    z = standardize_windows(rng.standard_normal((5, 60)))
    direct = np.sqrt(np.mean((z[:, None, :] - z[None, :, :]) ** 2, axis=2))
    assert np.allclose(direct, correlation_distance(z), atol=1e-7)


def test_spline_metric_symmetric_triangle_inequality():
    z = standardize_windows(np.random.default_rng(9).standard_normal((6, 60)))
    d = spline_metric(z, SplineSettings(n_basis=26, derivative_weight=1.5))
    assert np.allclose(d, d.T)
    assert np.all(np.diag(d) == 0)
    assert np.all(np.isfinite(d)) and np.min(d) >= 0
    for i in range(6):
        for j in range(6):
            for k in range(6):
                assert d[i, k] <= d[i, j] + d[j, k] + 1e-6


def test_identical_paths_are_zero_distance():
    rng = np.random.default_rng(11)
    z = standardize_windows(np.stack([rng.normal(size=50)] * 2))
    assert spline_metric(z, SplineSettings(n_basis=20))[0, 1] < 1e-6


def test_bad_inputs_raise():
    with pytest.raises(ValueError):
        standardize_windows(np.ones((3, 25)))
    z = standardize_windows(np.random.default_rng(12).normal(size=(3, 20)))
    with pytest.raises(ValueError):
        spline_metric(z, SplineSettings(n_basis=30))
    with pytest.raises(ValueError):
        spline_metric(z, SplineSettings(n_basis=10, derivative_weight=-1))
