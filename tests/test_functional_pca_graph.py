"""Numerical correctness, leakage and degenerate-geometry regression tests."""

import importlib.util
from pathlib import Path

import numpy as np
import pytest

from market_geometry.functional_pca_graph import (
    fit_functional_pca,
    hilbert_coordinates,
    normalized_laplacian_eigenvalues,
    pairwise_euclidean,
    proximity_graph,
    reference_feature_score,
    rolling_graph_features,
)
from market_geometry.polynomial_geometry import pairwise_polynomial_distance


def test_hilbert_coordinate_distances_equal_exact_polynomial_metric():
    a = np.random.default_rng(12).normal(size=(8, 8))
    for alpha in (0, 0.01, 1.0):
        whitened = hilbert_coordinates(a, alpha)
        assert np.allclose(
            pairwise_euclidean(whitened), pairwise_polynomial_distance(a, alpha), atol=1e-7
        )


def test_fpca_reconstruction_and_contraction():
    rng = np.random.default_rng(13)
    x = rng.normal(size=(13, 8))
    full = fit_functional_pca(x, 8)
    limited = fit_functional_pca(x, 3)
    d_full = pairwise_euclidean(full.scores)
    d_raw = pairwise_euclidean(x)
    assert np.allclose(d_full, d_raw, atol=1e-7)
    assert np.all(pairwise_euclidean(limited.scores) <= d_raw + 1e-8)
    assert np.isclose(full.explained_variance_ratio.sum(), 1.0)
    assert np.all(np.diff(full.eigenvalues) <= 1e-9)


def test_symmetric_knn_and_zero_eigenvalue():
    x = np.arange(7.0)[:, None]
    d = pairwise_euclidean(x)
    w = proximity_graph(d, k=2, bandwidth=2.0)
    eig = normalized_laplacian_eigenvalues(w)
    assert np.allclose(w, w.T)
    assert np.all(w >= 0)
    assert np.allclose(np.diag(w), 0)
    assert eig[0] < 1e-9
    assert eig[1] > 0
    assert eig.max() <= 2 + 1e-9


def test_causal_graph_bandwidth_calibration():
    rng = np.random.default_rng(333)
    d = np.stack([pairwise_euclidean(rng.normal(size=(8, 5))) for _ in range(7)])
    mask = np.array([True, True, True, False, False, False, False])
    a = rolling_graph_features(d, mask, k=2)
    changed = d.copy()
    changed[3:] *= 10
    b = rolling_graph_features(changed, mask, k=2)
    assert a[3] == b[3]
    assert np.allclose(a[0][:3], b[0][:3])
    assert np.allclose(a[1][:3], b[1][:3])


def test_reference_score_uses_only_calibration():
    x = np.arange(35.0).reshape((7, 5))
    flag = np.array([True, True, True, False, False, False, False])
    base = reference_feature_score(x, flag)
    y = x.copy()
    y[3:] += 100
    modified = reference_feature_score(y, flag)
    assert np.allclose(base[:3], modified[:3])
    assert np.any(base[3:] != modified[3:])


def test_negative_inputs_and_invalid_neighbor_parameter():
    with pytest.raises(ValueError):
        hilbert_coordinates(np.ones((3, 4)), -1)
    with pytest.raises(ValueError):
        proximity_graph(np.zeros((3, 3)), k=3)
    with pytest.raises(ValueError):
        fit_functional_pca(np.ones((3, 2)), 4)


def _load_runner():
    path = (Path(__file__).resolve().parents[1]
            / "market_geometry/code/006-functional_pca_graphs/run.py")
    spec = importlib.util.spec_from_file_location("exp006_runner", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_rolling_features_have_no_future_lookahead():
    runner = _load_runner()
    rng = np.random.default_rng(42)
    panel = rng.normal(size=(12, 96))
    old = runner.rolling_features(panel, 48, 12)
    mutated = panel.copy()
    mutated[:, 75:] += rng.normal(0, 2, size=(12, 21))
    new = runner.rolling_features(mutated, 48, 12)
    assert np.array_equal(old[0], new[0])
    for name in old[1]:
        assert np.allclose(old[1][name][:2], new[1][name][:2])
    assert np.allclose(old[3][:2], new[3][:2])


def test_quick_run_synthetic_valid_metrics():
    runner = _load_runner()
    summary = runner.run(publish=False, quick=True, seeds=1)
    assert summary["n_independent_seeds"] == 1
    cases = {r["dgp"] for r in summary["aggregate"]}
    assert cases == {"ar1_switch", "ar2_switch", "ar1_null"}
    for row in summary["aggregate"]:
        if row["dgp"] == "ar1_null":
            assert row["mean_auroc"] is None
        else:
            assert 0 <= row["mean_auroc"] <= 1
