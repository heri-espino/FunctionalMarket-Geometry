"""Reproducibility and rolling-time boundary checks."""

import numpy as np

from market_geometry import (
    RollingSettings,
    SimulationSettings,
    SplineSettings,
    reference_deviation,
    rolling_distances,
    simulate_market,
)


def test_generator_seed_is_reproducible():
    config = SimulationSettings(n_steps=240, boundary=120, seed=77)
    x, labels, groups = simulate_market(config)
    y, labels2, groups2 = simulate_market(config)
    assert np.array_equal(x, y)
    assert np.array_equal(labels, labels2) and np.array_equal(groups, groups2)
    assert not labels[119] and labels[120]


def test_rolling_uses_no_future_samples():
    x, _, _ = simulate_market(SimulationSettings(n_steps=180, boundary=90))
    cfg = RollingSettings(window=40, stride=20)
    s = SplineSettings(n_basis=20)
    ends, c, f = rolling_distances(x, cfg, s)
    x2 = x.copy()
    x2[:, 80:] += np.random.default_rng(3).normal(size=x2[:, 80:].shape) * 30
    ends2, c2, f2 = rolling_distances(x2, cfg, s)
    assert np.array_equal(ends, ends2)
    assert np.allclose(c[:3], c2[:3])
    assert np.allclose(f[:3], f2[:3])


def test_reference_estimates_only_calibration():
    mats = np.array([[[0, 1], [1, 0]], [[0, 3], [3, 0]], [[0, 8], [8, 0]]])
    scores, reference = reference_deviation(mats, np.array([True, True, False]))
    assert np.allclose(reference, [2.0])
    assert np.allclose(scores, [1.0, 1.0, 6.0])
