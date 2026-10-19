"""Finite-difference order comparison and *nested* LOOCV smoothing selection.

This module implements the Gaussian variance-ratio likelihood *motivated* by
Guerrero's (2007) stochastic PLS setup; the paper's published selection rule is
based on a user-specified smoothness index, NOT this MLE procedure.

For 1 <= d < n, D_d is the (n-d) by n difference operator, and
    y = trend + eta,     D_d trend = eps,     eta ⟂ eps,
    eta ~ N(0, sigma_eta^2 I), eps ~ N(0, sigma_eps^2 I).
Then z=D_d y is normal with covariance
    sigma_eps^2 * (I + lambda * D_d D_d.T),
where lambda=sigma_eta^2/sigma_eps^2. The Gaussian *profile* ML estimates
lambda, with sigma_eps^2 profiled. LOO missing indices are excluded from all
contrasts that touch them; this gives a valid but non-efficient likelihood.

Press residuals for one and two deleted observations are *exact* when the
linear smoother A_lambda=(I+lambda D_d.T D_d)^-1 and lambda are fixed.
Two-deletion identities enable inner-LOOCV selection for each held-out outer
index without using that index during hyperparameter selection.
"""

from dataclasses import dataclass
from functools import lru_cache

import numpy as np
from scipy.optimize import minimize_scalar

from .guerrero import lambda_from_s, smoother


@dataclass(frozen=True)
class OrderStudyConfig:
    """Prespecified candidate orders, smoothing grid, and likelihood bounds."""

    orders: tuple[int, ...] = (1, 2, 3, 4)
    s_grid: tuple[float, ...] = (0.20, 0.35, 0.50, 0.65, 0.75, 0.85, 0.92, 0.965, 0.99)
    fixed_s: float = 0.75
    ml_log_lambda_bounds: tuple[float, float] = (-12.0, 16.0)


@lru_cache(maxsize=12)
def differences(n: int, d: int) -> np.ndarray:
    """Construct d-th backward differences on an equally spaced grid."""
    if not 1 <= d < n:
        raise ValueError("Need 1 <= d < n")
    return np.diff(np.eye(n), n=d, axis=0)


def hat_matrix(n: int, d: int, smoothing: float) -> np.ndarray:
    """Return the zero-mean PLS hat matrix H(S)."""
    return smoother(n, d, float(smoothing))[0]


def press_residuals(y: np.ndarray, hat: np.ndarray) -> np.ndarray:
    """Exact scalar leave-one-out residuals for a fixed linear smoother."""
    values = np.asarray(y, dtype=float)
    if values.ndim != 1 or hat.shape != (len(values), len(values)):
        raise ValueError("Mismatched series and hat matrix")
    leverage = 1.0 - np.diag(hat)
    if np.any(leverage <= 1e-10):
        raise ValueError("Non-identifiable LOOCV (leverages too close to one)")
    return (values - hat @ values) / leverage


def two_delete_residuals(y: np.ndarray, hat: np.ndarray, heldout: int) -> np.ndarray:
    """Inner LOO residuals on the dataset excluding one *outer* index.

    Entry i!=heldout predicts y_i from observations excluding BOTH i and
    heldout; the outer value cancels algebraically. Entry heldout is NaN.
    """
    values = np.asarray(y, dtype=float)
    n = len(values)
    if values.ndim != 1 or hat.shape != (n, n) or not 0 <= heldout < n:
        raise ValueError("Mismatched two-deletion inputs")
    e = values - hat @ values
    a = 1.0 - np.diag(hat)
    j = heldout
    cross = hat[:, j]
    denom = a * a[j] - cross**2
    denom[j] = np.nan
    if np.any(denom[np.arange(n) != j] <= 1e-10):
        raise ValueError("Nearly singular two-point deletion")
    return (e * a[j] + cross * e[j]) / denom


def inner_loo_scores(y: np.ndarray, hat: np.ndarray) -> np.ndarray:
    """One inner-LOO RMSE-selection loss per possible outer heldout index."""
    e = y - hat @ y
    h = np.diag(hat)
    leverage = 1.0 - h
    cross = hat.copy()
    denom = leverage[:, None] * leverage[None, :] - cross**2
    np.fill_diagonal(denom, np.nan)
    if np.any(denom[~np.eye(len(y), dtype=bool)] <= 1e-10):
        raise ValueError("Nearly singular pair deletion")
    # [inner index i, outer heldout j]
    loo_e = (e[:, None] * leverage[None, :] + cross * e[None, :]) / denom
    return np.nanmean(loo_e**2, axis=0)


def inner_gcv_scores(y: np.ndarray, hat: np.ndarray) -> np.ndarray:
    """GCV criterion, refitted on n-1 observations for each omitted j.

    Uses the Sherman-Morrison downdate of an existing linear hat matrix.
    Inner GCV is a selector, not a claimed unbiased error estimate.
    """
    n = len(y)
    e = y - hat @ y
    v = 1.0 - np.diag(hat)
    if np.any(v < 1e-10):
        raise ValueError("LOO singularity")
    # R_ij is the residual of observed i after deleting j; diagonal excluded.
    residual = e[:, None] + hat * (e[None, :] / v[None, :])
    np.fill_diagonal(residual, 0.0)
    rss = np.sum(residual**2, axis=0)
    # diag fit for retained i after removing j: H_ii + H_ij²/(1-H_jj).
    leverage_sum = np.trace(hat) - np.diag(hat) + (
        np.sum(hat**2, axis=0) - np.diag(hat)**2
    ) / v
    frac = 1.0 - leverage_sum / (n - 1)
    return (rss / (n - 1)) / np.maximum(frac, 1e-9)**2


def observed_differences(y: np.ndarray, d: int, missing: int | None) -> tuple[np.ndarray, np.ndarray]:
    """Retain only differences whose d+1 observations are available."""
    dmat = differences(len(y), d)
    if missing is not None:
        if not 0 <= missing < len(y):
            raise ValueError("Missing index out of range")
        dmat = dmat[abs(dmat[:, missing]) < 1e-12]
    return dmat @ y, dmat


def gaussian_profile_ml_lambda(
    y: np.ndarray, d: int, *, missing: int | None = None,
    profile_mean: bool = False, log_bounds: tuple[float, float] = (-12, 16),
) -> tuple[float, bool, float]:
    """Profile Gaussian ML of the variance ratio λ using valid differences.

    With profile_mean=True a GLS-optimized constant mean μ is nuisance.
    The fitted PLS method in EXP-003 always assumes μ=0, so it uses False.
    Returns (lambda, optimum_is_boundary, negative_twice_loglik_up_to_const).
    The profile scale divisor is m (ML), *not* m-1 (REML).
    """
    v = np.asarray(y, dtype=float)
    if v.ndim != 1 or not np.all(np.isfinite(v)):
        raise ValueError("Expected finite one-dimensional observations")
    if not (1 <= d < len(v)):
        raise ValueError("Invalid difference order")
    z, operator = observed_differences(v, d, missing)
    if len(z) <= int(profile_mean):
        raise ValueError("Insufficient completed differences")
    eigenvalues, u = np.linalg.eigh(operator @ operator.T)
    eigenvalues = np.maximum(eigenvalues, 0.0)
    rotated_z = u.T @ z
    rotated_ones = (u.T @ np.ones(len(z))) if profile_mean else None
    m = len(z)

    def objective(loglam: float) -> float:
        lam = np.exp(loglam)
        eigen_variance = 1.0 + lam * eigenvalues
        weights = 1.0 / eigen_variance
        if profile_mean:
            mu = np.dot(weights * rotated_ones, rotated_z) / np.dot(
                weights * rotated_ones, rotated_ones
            )
            resid = rotated_z - mu * rotated_ones
        else:
            resid = rotated_z
        quad = float(np.dot(weights, resid**2))
        if quad <= np.finfo(float).tiny:
            return float('inf')
        return float(m * np.log(quad / m) + np.log(eigen_variance).sum())

    lower, upper = log_bounds
    if not np.isfinite([lower, upper]).all() or lower >= upper:
        raise ValueError("Invalid likelihood log bounds")
    # Multi-start on a modest grid, then local bounded refine: profile ML need
    # not be globally unimodal. Include declared endpoints.
    grid = np.linspace(lower, upper, 29)
    points = [(float(t), objective(float(t))) for t in grid]
    best = min(points, key=lambda pair: pair[1])
    values = np.asarray([q for _, q in points])
    # Refine each sampled local minimum, not just the lowest sampled cell.
    # Numerical screening is not a certificate of global optimality.
    interior = [k for k in range(1, len(grid)-1)
                if values[k] <= values[k-1] and values[k] <= values[k+1]]
    brackets = [(grid[k-1], grid[k+1]) for k in interior]
    brackets.extend(((grid[0], grid[1]), (grid[-2], grid[-1])))
    for low, high in brackets:
        opt = minimize_scalar(objective, bounds=(low, high), method="bounded",
                              options={"xatol": 1e-6})
        if opt.fun < best[1]:
            best = (float(opt.x), float(opt.fun))
    boundary = abs(best[0] - lower) < 1e-5 or abs(best[0] - upper) < 1e-5
    return float(np.exp(best[0])), bool(boundary), best[1]


def nested_loo_comparison(y: np.ndarray, config: OrderStudyConfig) -> dict:
    """Outer-LOO scores with internal smoothness selection on n-1 points.

    Fixed: predeclared S.
    Inner LOOCV: minimize two-deletion loss on retained observations.
    Inner GCV: minimize GCV on retained observations.
    ML: maximize profile Gaussian likelihood on differences that exclude outer j.
    The reported outer RMSE is calculated ONLY from outer-heldout predictions.
    """
    series = np.asarray(y, dtype=float)
    if series.ndim != 1 or not np.all(np.isfinite(series)):
        raise ValueError("Expected finite univariate series")
    n = len(series)
    methods = ("fixed", "inner_loocv", "inner_gcv", "gaussian_ml")
    res = {}
    if n < max(config.orders) + 4:
        raise ValueError("Series too short")
    for d in config.orders:
        s_candidates = np.asarray(config.s_grid, dtype=float)
        if not np.all((s_candidates > 0) & (s_candidates < 1)):
            raise ValueError("Use interior grid for stable two-deletion LOOCV")
        candidate_hats = [hat_matrix(n, d, float(s)) for s in s_candidates]
        i_cv = np.stack([inner_loo_scores(series, h) for h in candidate_hats])
        i_gcv = np.stack([inner_gcv_scores(series, h) for h in candidate_hats])
        cv_selected = np.argmin(i_cv, axis=0)
        gcv_selected = np.argmin(i_gcv, axis=0)
        fixed_hat = hat_matrix(n, d, config.fixed_s)
        fit = {key: np.zeros(n, dtype=float) for key in methods}
        selected_s = {key: np.zeros(n, dtype=float) for key in methods}
        selected_lambda = {key: np.zeros(n, dtype=float) for key in methods}
        ml_boundary = 0
        for j in range(n):
            ml_lam, boundary, _ = gaussian_profile_ml_lambda(
                series, d, missing=j, log_bounds=config.ml_log_lambda_bounds
            )
            ml_boundary += int(boundary)
            from .guerrero import _spectral
            eig, u, _ = _spectral(n, d)
            # ML chooses continuous lambda rather than the S grid.
            response = 1 / (1 + ml_lam * eig)
            h_ml = (u * response) @ u.T
            choices = {
                "fixed": (config.fixed_s, lambda_from_s(n, d, config.fixed_s), fixed_hat),
                "inner_loocv": (s_candidates[cv_selected[j]],
                                lambda_from_s(n, d, s_candidates[cv_selected[j]]),
                                candidate_hats[cv_selected[j]]),
                "inner_gcv": (s_candidates[gcv_selected[j]],
                              lambda_from_s(n, d, s_candidates[gcv_selected[j]]),
                              candidate_hats[gcv_selected[j]]),
                "gaussian_ml": (float(np.mean(ml_lam * eig[d:] / (1 + ml_lam * eig[d:]))),
                                ml_lam, h_ml),
            }
            for method, (s, lam, h) in choices.items():
                looe = press_residuals(series, h)[j]
                fit[method][j] = series[j] - looe
                selected_s[method][j] = float(s)
                selected_lambda[method][j] = float(lam)
        for method in methods:
            res[(d, method)] = {
                "outer_loo_prediction": fit[method],
                "selected_s": selected_s[method],
                "selected_lambda": selected_lambda[method],
                "rmse_observed": float(np.sqrt(np.mean((series - fit[method])**2))),
                "n_boundary_ml": ml_boundary if method == "gaussian_ml" else 0,
            }
    return res
