"""Invariants for the declared EXP-007 synthetic similarity targets."""

import numpy as np
import pytest

from market_geometry.similarity_targets import (
    pairwise_scores,
    prices_from_returns,
    signed_lag_similarity,
    simulate_case,
)


def test_prices_and_returns_scale_clone():
    case = simulate_case("level_clone", seed=14)
    np.testing.assert_allclose(case.prices[0] / case.prices[1], 10, rtol=0, atol=1e-12)
    np.testing.assert_allclose(
        np.diff(np.log(case.prices), axis=1), case.returns, atol=3e-15,
    )
    scores = pairwise_scores(case)
    assert scores["raw_price_rms"][0, 1] > 1000
    for method in (
        "rebased_price_rms", "return_rms", "pearson", "pls_h1",
        "legendre_h1", "max_lag3_correlation", "volatility_log_ratio",
    ):
        assert abs(scores[method][0, 1]) < 5e-7, method


def test_volatility_clone_retains_risk_difference():
    scores = pairwise_scores(simulate_case("volatility_clone", seed=20))
    assert abs(scores["volatility_log_ratio"][0, 1] - np.log(2)) < 1e-12
    assert scores["return_rms"][0, 1] > 0.005
    assert scores["pearson"][0, 1] < 1e-6


def test_lagged_clone_favors_explicit_lag_estimator():
    scores = pairwise_scores(simulate_case("lagged_factor", seed=100, length=128))
    assert scores["max_lag3_correlation"][0, 1] < 1e-6
    assert scores["pearson"][0, 1] > 0.3
    assert scores["max_lag3_correlation"][0, 1] < scores["max_lag3_correlation"][0, 2]


def test_raw_levels_reject_nonpositive():
    with pytest.raises(ValueError):
        prices_from_returns(np.ones((2, 30)), (0, 1000))


def test_pairwise_symmetry_and_diagonal():
    for name in ("level_clone", "shared_factor", "lagged_factor", "volatility_clone"):
        for distances in pairwise_scores(simulate_case(name, seed=11)).values():
            assert distances.shape == (3, 3)
            assert np.isfinite(distances).all()
            assert np.allclose(distances, distances.T)
            assert np.allclose(np.diag(distances), 0)
            assert np.all(distances >= 0)


def test_signed_lag_is_only_similarity():
    correlation = signed_lag_similarity(np.random.default_rng(5).normal(size=(3, 60)), 3)
    assert np.allclose(correlation, correlation.T)
    assert np.allclose(np.diag(correlation), 1)
