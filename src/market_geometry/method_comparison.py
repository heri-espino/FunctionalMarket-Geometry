"""Nested held-out reconstruction benchmark for multiple *linear* trend estimators.

Every candidate is a symmetric hat matrix arising from penalized quadratic
least squares (or kernel-ridge regularization) on a prespecified fixed grid.
This permits exact one- and two-observation deletion identities. The inner
LOOCV choice is recomputed for every outer held-out observation, across both
hyperparameters and optionally method family; oracle latent curves never enter
selection. This is not chronological, causal financial forecasting validation.

The mean-difference Guerrero plug-in, LOESS, adaptive L1 trend filtering and
Kalman MLE are not represented by these fixed symmetric hat matrices; their
proper deleted-case fitting requires a separate comparison protocol.
"""

from dataclasses import dataclass
from functools import lru_cache

import numpy as np
from scipy.interpolate import BSpline
from scipy.linalg import cho_factor, cho_solve

from .guerrero import smoother
from .order_selection import inner_loo_scores, press_residuals


@dataclass(frozen=True)
class Candidate:
    """One prespecified fitted-function family and a tuned configuration."""

    family: str
    name: str
    hat: np.ndarray


def _symmetric_fit(design: np.ndarray, penalty: np.ndarray | None = None) -> np.ndarray:
    """Symmetric linear smoother from quadratic penalized basis regression."""
    b = np.asarray(design, dtype=float)
    if b.ndim != 2 or b.shape[0] < b.shape[1]:
        raise ValueError("Need full-rank or regularized tall design")
    p = np.zeros((b.shape[1], b.shape[1])) if penalty is None else np.asarray(penalty)
    gram = b.T @ b + p
    h = b @ np.linalg.solve(gram, b.T)
    return (h + h.T) / 2


@lru_cache(maxsize=64)
def bspline_basis(n: int, degree: int, n_basis: int) -> np.ndarray:
    """Nonperiodic clamped, equally-spaced interior-knot B-spline basis."""
    if n < 8 or degree < 1 or n_basis <= degree or n_basis >= n:
        raise ValueError("Invalid B-spline dimension")
    interiors = np.linspace(0, 1, n_basis - degree + 1)[1:-1]
    knots = np.r_[np.zeros(degree + 1), interiors, np.ones(degree + 1)]
    x = np.linspace(0, 1, n)
    return BSpline.design_matrix(x, knots, degree).toarray()


@lru_cache(maxsize=8)
def natural_cubic_penalty(n: int) -> np.ndarray:
    """Exact natural cubic smoothing-spline roughness matrix Q R^-1 Q^T.

    For observation times 0,1,...,n-1, Q has columns (1,-2,1),
    R is tridiagonal diag 2/3, offdiag 1/6. This reproduces the
    integrated squared-second-derivative penalty for natural cubic splines.
    """
    if n < 5:
        raise ValueError("Need n>=5")
    q = np.diff(np.eye(n), n=2, axis=0).T
    r = np.diag(np.repeat(2 / 3, n - 2))
    r += np.diag(np.repeat(1 / 6, n - 3), k=1)
    r += np.diag(np.repeat(1 / 6, n - 3), k=-1)
    k = q @ np.linalg.solve(r, q.T)
    return (k + k.T) / 2


def _ridge_hat_from_gram(k: np.ndarray, lam: float) -> np.ndarray:
    """Solve (I + lambda K)^{-1} for PSD roughness matrix K."""
    h = np.linalg.solve(np.eye(k.shape[0]) + lam * k, np.eye(k.shape[0]))
    return (h + h.T) / 2


def _gp_kernel_hat(n: int, length_scale: float, ridge: float) -> np.ndarray:
    """Fixed-hyperparameter RBF Gaussian kernel ridge / GP posterior mean."""
    t = np.linspace(0, 1, n)
    k = np.exp(-.5 * ((t[:, None] - t[None, :]) / length_scale) ** 2)
    chol = cho_factor(k + ridge * np.eye(n), lower=True, check_finite=True)
    h = k @ cho_solve(chol, np.eye(n))
    return (h + h.T) / 2


@lru_cache(maxsize=8)
def candidate_bank(n: int) -> tuple[Candidate, ...]:
    """Prespecified candidate families, all using the same time grid."""
    if n < 22:
        raise ValueError("n must be >=22")
    output: list[Candidate] = []
    for d in (1, 2, 3, 4):
        for s in (0.35, 0.65, 0.85, 0.96):
            h, _, _ = smoother(n, d, s)
            output.append(Candidate(f"PLS order {d}", f"d={d}, S={s:g}", h))

    for degree in (1, 3):
        for nb in (5, 9, 14):
            b = bspline_basis(n, degree, nb)
            h = _symmetric_fit(b)
            output.append(Candidate(f"B-spline degree {degree}", f"basis={nb}", h))

    for penalty_order in (1, 2, 3):
        for lam in (0.1, 3.0, 100.0):
            b = bspline_basis(n, 3, 14)
            diff = np.diff(np.eye(b.shape[1]), n=penalty_order, axis=0)
            h = _symmetric_fit(b, lam * (diff.T @ diff))
            output.append(Candidate(f"P-spline penalty {penalty_order}",
                                    f"14 cubic bases, lambda={lam:g}", h))

    roughness = natural_cubic_penalty(n)
    for lam in (0.03, 0.3, 3.0, 30.0, 300.0):
        output.append(Candidate("Natural cubic smoothing spline", f"lambda={lam:g}",
                                _ridge_hat_from_gram(roughness, lam)))

    for length_scale in (0.07, 0.15, 0.3):
        for ridge in (0.01, 0.1, 1.0):
            output.append(Candidate("RBF kernel ridge", f"length={length_scale:g}, ridge={ridge:g}",
                                    _gp_kernel_hat(n, length_scale, ridge)))

    time = np.linspace(-1, 1, n)
    for order in (0, 1, 2, 3):
        basis = np.polynomial.legendre.legvander(time, order)
        output.append(Candidate("Polynomial regression", f"degree={order}",
                                _symmetric_fit(basis)))

    for candidate in output:
        if not np.isfinite(candidate.hat).all() or candidate.hat.shape != (n, n):
            raise AssertionError("Invalid candidate hat matrix")
        if not np.allclose(candidate.hat, candidate.hat.T, atol=1e-8):
            raise AssertionError("Hat matrix not symmetric")
    return tuple(output)


def compare_nested_loo(y: np.ndarray) -> dict:
    """Outer LOO with *inner* LOO hyperparameter/family selection.

    Outcome is out-of-sample observed reconstruction RMSE, not future RMSE.
    For each outer j, candidate scores predict each other i after removing
    both i and j. The outer estimate never uses y_j. Inner best-of-all is
    tested on precisely the same outer j's as each individual family.
    """
    y = np.asarray(y, dtype=float)
    if y.ndim != 1 or not np.isfinite(y).all():
        raise ValueError("Expected finite 1D observed series")
    bank = candidate_bank(len(y))
    families = list(dict.fromkeys(c.family for c in bank))
    inner = np.stack([inner_loo_scores(y, c.hat) for c in bank])
    preds = np.stack([y - press_residuals(y, c.hat) for c in bank])
    chosen = {}
    for family in [*families, "All families (nested selection)"]:
        inds = np.arange(len(bank)) if family.startswith("All families") else np.asarray(
            [i for i, c in enumerate(bank) if c.family == family]
        )
        selected = inds[np.argmin(inner[inds], axis=0)]
        prediction = preds[selected, np.arange(len(y))]
        chosen[family] = {
            "outer_loo_prediction": prediction,
            "outer_rmse": float(np.sqrt(np.mean((y - prediction) ** 2))),
            "selected_candidates": [bank[i].name for i in selected],
            "selected_families": [bank[i].family for i in selected],
        }
    return {"by_family": chosen, "n_candidates": len(bank), "families": families}
