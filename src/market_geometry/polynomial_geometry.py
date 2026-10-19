"""Orthogonal-polynomial representations and exact polynomial Sobolev metrics.

For u in [0, 1], phi_k(u) = sqrt(2k+1) P_k(2u-1), where P_k is
Legendre polynomial k. Thus integral phi_j phi_k du = delta_jk.
A finite-degree polynomial family is a Hilbert *subspace* of L2/H1.
No claims of novelty or superiority follow from this change of basis.
"""

from functools import lru_cache

import numpy as np
from numpy.polynomial.legendre import legder, leggauss, legval, legvander


def orthonormal_design(u: np.ndarray, degree: int) -> np.ndarray:
    """Return the [0,1] L2-orthonormal Legendre design matrix.

    Parameters
    ----------
    u : ndarray
        Coordinates in [0,1], with arbitrary shape.
    degree : int
        Largest polynomial degree (>=0).

    Returns
    -------
    ndarray
        Shape (n_coordinates, degree+1).
    """
    u = np.asarray(u, dtype=float).reshape(-1)
    if not isinstance(degree, (int, np.integer)) or degree < 0:
        raise ValueError("Degree must be a nonnegative integer")
    if not np.all(np.isfinite(u)) or np.any((u < 0) | (u > 1)):
        raise ValueError("u must be finite and lie within [0,1]")
    return legvander(2.0 * u - 1.0, degree) * np.sqrt(2 * np.arange(degree + 1) + 1)


def project_onto_legendre(trends: np.ndarray, degree: int) -> np.ndarray:
    """Discrete least-squares projection onto the polynomial subspace.

    This estimates coefficients from finite observations. It is not the
    exact continuous orthogonal projection, although its basis functions
    are continuously orthonormal. All fits use the same fixed grid/basis.
    """
    x = np.asarray(trends, dtype=float)
    if x.ndim != 2 or not np.isfinite(x).all() or x.shape[1] < 3:
        raise ValueError("Require finite 2D panel and at least three dates")
    if not isinstance(degree, (int, np.integer)) or not (0 <= degree < x.shape[1]):
        raise ValueError("Require integer 0 <= degree < sample count")
    basis = orthonormal_design(np.linspace(0.0, 1.0, x.shape[1]), degree)
    coeff = np.linalg.lstsq(basis, x.T, rcond=None)[0].T
    return coeff


def evaluate_legendre(coefficients: np.ndarray, u: np.ndarray) -> np.ndarray:
    """Evaluate polynomial coefficients as rows on coordinates u."""
    a = np.asarray(coefficients, dtype=float)
    if a.ndim != 2 or not np.isfinite(a).all() or a.shape[1] == 0:
        raise ValueError("Require finite 2D coefficient array")
    return a @ orthonormal_design(u, a.shape[1] - 1).T


@lru_cache(maxsize=48)
def derivative_gram(degree: int) -> np.ndarray:
    """Exact integral R_ij=integral_0^1 phi'_i(u) phi'_j(u) du.

    Gauss-Legendre quadrature integrates polynomial products up to
    degree 2p-2 exactly in exact arithmetic. Floating point precision
    gives a symmetric positive semidefinite approximation.
    """
    if degree < 0:
        raise ValueError("degree must be nonnegative")
    nodes, weights = leggauss(max(2, degree + 1))
    dphi = np.empty((len(nodes), degree + 1))
    for k in range(degree + 1):
        basis = np.zeros(k + 1)
        basis[-1] = 1
        # d/du P_k(2u-1) = 2 P'_k(2u-1)
        dphi[:, k] = 2 * np.sqrt(2 * k + 1) * legval(nodes, legder(basis))
    out = 0.5 * (dphi.T * weights) @ dphi
    return 0.5 * (out + out.T)


def pairwise_polynomial_distance(coefficients: np.ndarray, derivative_weight: float = 0.0) -> np.ndarray:
    r"""Exact continuous L2/H1 distances between represented polynomials.

    d^2(i,j) = Delta a' (I + alpha R) Delta a, with derivative in
    normalized u in [0,1]. alpha is nonnegative and dimensionless
    *relative to u*, not daily-time derivatives (unlike EXP-001).

    For alpha=0, coefficients' Euclidean distance is exact L2 distance.
    With alpha>0, it is an H1 metric on the finite polynomial space.
    """
    a = np.asarray(coefficients, dtype=float)
    if a.ndim != 2 or a.shape[0] < 2 or not np.isfinite(a).all():
        raise ValueError("Require finite matrix of >=2 coefficient rows")
    if not np.isfinite(derivative_weight) or derivative_weight < 0:
        raise ValueError("Derivative weight must be finite and nonnegative")
    gram = np.eye(a.shape[1]) + derivative_weight * derivative_gram(a.shape[1] - 1)
    differences = a[:, None, :] - a[None, :, :]
    d2 = np.einsum("ijk,kl,ijl->ij", differences, gram, differences, optimize=True)
    d = np.sqrt(np.maximum(0.0, d2))
    np.fill_diagonal(d, 0.0)
    return d
