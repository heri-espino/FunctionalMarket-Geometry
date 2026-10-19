"""Exactly specified correlation and functional-Sobolev dissimilarities."""

from dataclasses import dataclass

import numpy as np
from scipy.interpolate import BSpline
from scipy.spatial.distance import squareform


@dataclass(frozen=True)
class SplineSettings:
    """Configuration for penalized cubic B-spline functional representations.

    Parameters
    ----------
    n_basis : int
        Number of B-spline basis functions, at least ``degree + 1``.
    degree : int
        Polynomial degree of the spline (3 by default).
    ridge : float
        Coefficient second-difference penalty, nonnegative.
    derivative_weight : float
        Nonnegative Sobolev derivative weight in squared **daily** units.
    """

    n_basis: int = 36
    degree: int = 3
    ridge: float = 0.02
    derivative_weight: float = 0.75


def standardize_windows(values: np.ndarray) -> np.ndarray:
    """Z-score the within-window returns of each asset.

    Parameters
    ----------
    values : ndarray of shape (n_assets, window_length)
        Finite asset returns; rows with near-zero variance are invalid.

    Returns
    -------
    ndarray
        Zero-mean, unit-population-variance rows.
    """
    a = np.asarray(values, dtype=float)
    if a.ndim != 2 or a.shape[1] < 5 or not np.isfinite(a).all():
        raise ValueError("Expected finite 2-D asset-by-time returns with >=5 observations")
    std = a.std(axis=1, ddof=0, keepdims=True)
    if np.any(std < 1e-12):
        raise ValueError("At least one asset has nearly constant returns")
    return (a - a.mean(axis=1, keepdims=True)) / std


def correlation_distance(z: np.ndarray) -> np.ndarray:
    """Compute ``sqrt(2*(1-rho))`` using window-standardized rows.

    Parameters
    ----------
    z : ndarray of shape (n_assets, window_length)
        Window-standardized returns; see :func:`standardize_windows`.

    Returns
    -------
    ndarray of shape (n_assets, n_assets)
        Chordal correlation distance. It is not a causal-dependence metric.
    """
    z = np.asarray(z, dtype=float)
    if z.ndim != 2 or z.shape[0] < 2:
        raise ValueError("Expected at least two assets")
    rho = np.clip(z @ z.T / z.shape[1], -1.0, 1.0)
    d = np.sqrt(np.maximum(0.0, 2.0 * (1.0 - rho)))
    np.fill_diagonal(d, 0.0)
    return d


def spline_metric(z: np.ndarray, settings: SplineSettings) -> np.ndarray:
    """Compute pairwise penalized-spline Sobolev distances.

    A B-spline is fitted to each standardized asset return path using a common
    basis and second-difference coefficient penalty. The squared metric is

    ``(beta_i-beta_j)' (G0 + lambda * G1) (beta_i-beta_j)``

    where G0 and G1 are normalized quadrature approximations to the continuous
    L2 and first-derivative Gram matrices. Derivatives are in trading-day units.
    Smoothing is an estimation step; the returned distance is a pseudometric
    on raw observations, and a metric on distinct fitted functions.

    Parameters
    ----------
    z : ndarray of shape (n_assets, window_length)
        Within-window standardized returns.
    settings : SplineSettings
        Spline fit and derivative weight.

    Returns
    -------
    ndarray
        Symmetric, finite, nonnegative pairwise functional distances.
    """
    z = np.asarray(z, dtype=float)
    n_assets, window = z.shape
    k, p = settings.n_basis, settings.degree
    if not (p >= 1 and p + 1 <= k <= window):
        raise ValueError("Require 1 <= degree < n_basis <= window")
    if settings.ridge < 0 or settings.derivative_weight < 0:
        raise ValueError("Penalties and metric weight must be nonnegative")
    interior = np.linspace(0, 1, k - p + 1)[1:-1]
    knots = np.r_[np.zeros(p + 1), interior, np.ones(p + 1)]
    basis = BSpline(knots, np.eye(k), p, extrapolate=False)
    x = np.linspace(0, 1, window)
    b = basis(x)
    difference = np.diff(np.eye(k), n=2, axis=0)
    penalty = difference.T @ difference
    coefficients = np.linalg.solve(b.T @ b + settings.ridge * penalty, b.T @ z.T).T
    grid = np.linspace(0, 1, max(4 * window, 201))
    eval_b = basis(grid)
    # The fitting coordinate u is [0,1]; t (trading days) = (window-1)*u.
    eval_d = basis.derivative(1)(grid) / (window - 1)
    weights = np.full(grid.size, 1.0 / (grid.size - 1))
    weights[[0, -1]] *= 0.5
    gram = eval_b.T @ (weights[:, None] * eval_b)
    gram += settings.derivative_weight * (eval_d.T @ (weights[:, None] * eval_d))
    gram = 0.5 * (gram + gram.T)
    inner = coefficients @ gram @ coefficients.T
    sqnorm = np.diag(inner)
    d2 = np.maximum(0.0, sqnorm[:, None] + sqnorm[None, :] - 2 * inner)
    d = np.sqrt(d2)
    np.fill_diagonal(d, 0.0)
    return d


def pairwise_vector(distance_matrix: np.ndarray) -> np.ndarray:
    """Return upper-triangular off-diagonal edge distances."""
    return squareform(distance_matrix, checks=False)
