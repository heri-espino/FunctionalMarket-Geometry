"""Guerrero (2007)-style mean-difference PLS and controlled S selection.

S=(L-tr(H))/(L-d) follows the normalized coordinate of the user's
Trend-Estimation repository. Neither this coordinate nor the PLS fit is new.
"""

from dataclasses import dataclass
from functools import lru_cache

import numpy as np
from scipy.optimize import brentq


@dataclass(frozen=True)
class GuerreroSettings:
    """Shared smoothness grid and chronological selector specification."""

    order: int = 2
    fixed_s: float = 0.5
    derivative_weight: float = 0.75
    grid: tuple[float, ...] = (0.0, 0.1, 0.2, 0.35, 0.5, 0.65, 0.8, 0.9, 0.96, 1.0)
    cv_window: int = 40
    cv_stride: int = 5


@lru_cache(maxsize=8)
def _spectral(n: int, d: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Diagonalize the order-d difference Gram matrix; find D_d p = 1."""
    if not isinstance(d, (int, np.integer)) or not 1 <= d < n:
        raise ValueError("Require integer difference order 1 <= d < n")
    difference = np.diff(np.eye(n), n=d, axis=0)
    eigenvalues, eigenvectors = np.linalg.eigh(difference.T @ difference)
    eigenvalues[:d] = 0.0  # exact d-dimensional polynomial null-space
    eigenvalues[d:] = np.maximum(eigenvalues[d:], 0.0)
    # Discrete binomial basis: delta^d choose(j,d) = 1.
    from math import comb
    p = np.asarray([comb(j, d) if j >= d else 0 for j in range(n)], dtype=float)
    return eigenvalues, eigenvectors, p


def lambda_from_s(n: int, d: int, s: float) -> float:
    """Invert the strictly increasing smoothness S(lambda); endpoints exact."""
    if not np.isfinite(s) or not 0 <= s <= 1:
        raise ValueError("S outside [0,1]")
    if s == 0:
        return 0.0
    if s == 1:
        return float("inf")
    eig, _, _ = _spectral(n, d)

    def f(loglam: float) -> float:
        z = np.exp(loglam) * eig[d:]
        return float(np.mean(z / (1 + z)) - s)

    return float(np.exp(brentq(f, -45, 45)))


@lru_cache(maxsize=128)
def smoother(n: int, d: int, s: float) -> tuple[np.ndarray, np.ndarray, float]:
    """Return H, b=(I-H)p and the EDF of H at normalized S."""
    eig, u, p = _spectral(n, d)
    lam = lambda_from_s(n, d, float(s))
    response = np.zeros(n) if np.isinf(lam) else 1 / (1 + lam * eig)
    if np.isinf(lam):
        response[:d] = 1.0
    h = (u * response) @ u.T
    h = (h + h.T) / 2
    b = p - h @ p
    return h, b, float(response.sum())


def guerrero_mu(x: np.ndarray, d: int = 2) -> np.ndarray:
    """Per-series observed-difference plug-in mu_G=mean(D_d x)."""
    a = np.asarray(x, dtype=float)
    if a.ndim < 2 or a.shape[-1] <= d:
        raise ValueError("Expected panel with sufficient time points")
    return np.diff(a, n=d, axis=-1).mean(axis=-1)


def fit_panel(x: np.ndarray, s: float, *, d: int = 2, mean_mode: str = "zero") -> np.ndarray:
    """Fit a common-S PLS trend for each asset with no future data."""
    a = np.asarray(x, dtype=float)
    if a.ndim != 2 or not np.isfinite(a).all():
        raise ValueError("Expected finite asset-by-time panel")
    h, b, _ = smoother(a.shape[1], d, float(s))
    t = a @ h.T
    if mean_mode == "guerrero":
        t += guerrero_mu(a, d)[:, None] * b[None, :]
    elif mean_mode != "zero":
        raise ValueError("Unknown mean mode")
    return t


def functional_distance(x: np.ndarray, weight: float) -> np.ndarray:
    """Finite-grid Sobolev H1 distance; pseudometric on unfit data."""
    a = np.asarray(x, dtype=float)
    if a.ndim != 2 or a.shape[1] < 3 or weight < 0:
        raise ValueError("Invalid array or metric derivative weight")
    f = np.column_stack((a / np.sqrt(a.shape[1]), np.diff(a, axis=1) * np.sqrt(weight / (a.shape[1] - 1))))
    gram = f @ f.T
    norm = np.diag(gram)
    d2 = np.maximum(0, norm[:, None] + norm[None, :] - 2 * gram)
    res = np.sqrt(d2)
    np.fill_diagonal(res, 0)
    return res


def gcv_curve(x: np.ndarray, grid: tuple[float, ...], d: int, mean_mode: str) -> np.ndarray:
    """GCV with adjusted effective degrees of freedom for plug-in mu."""
    a = np.asarray(x, dtype=float)
    n = a.shape[1]
    dd = np.diff(np.eye(n), n=d, axis=0)
    c = dd.T @ np.ones(n - d) / (n - d)
    loss = []
    for s in grid:
        _, b, trace_h = smoother(n, d, float(s))
        df = trace_h + (float(c @ b) if mean_mode == "guerrero" else 0.0)
        ratio = 1 - df / n
        fit = fit_panel(a, s, d=d, mean_mode=mean_mode)
        loss.append(float(np.mean((a - fit) ** 2) / ratio**2) if ratio > 1e-8 else np.inf)
    return np.asarray(loss)


def forecast_cv_curve(
    raw: np.ndarray, grid: tuple[float, ...], d: int, mean_mode: str,
    train_length: int, stride: int,
) -> np.ndarray:
    """Completed historical one-step forecast losses inside a trailing window.

    IMPORTANT: use *raw returns*, not whole-window standardized observations:
    latter would leak inner-fold targets into the fold's normalization.
    """
    a = np.asarray(raw, dtype=float)
    if a.ndim != 2 or d != 2 or train_length < 6 or train_length >= a.shape[1] or stride < 1:
        raise ValueError("Invalid chronological CV configuration")
    origins = range(train_length, a.shape[1], stride)
    past = np.stack([a[:, t - train_length:t] for t in origins])
    target = np.stack([a[:, t] for t in origins])
    if mean_mode == "guerrero":
        mu = guerrero_mu(past, d)
    elif mean_mode == "zero":
        mu = np.zeros(past.shape[:2])
    else:
        raise ValueError("Unknown mean mode")
    errors = []
    for s in grid:
        h, b, _ = smoother(train_length, d, float(s))
        pred = np.einsum("qaw,w->qa", past, 2 * h[-1] - h[-2]) + mu * (2 * b[-1] - b[-2] + 1)
        errors.append(float(np.mean((pred - target) ** 2)))
    return np.asarray(errors)


def select_s(raw: np.ndarray, settings: GuerreroSettings, mean_mode: str, rule: str) -> tuple[float, np.ndarray]:
    """Return one shared S selected using only currently known observations."""
    if rule == "fixed":
        return settings.fixed_s, np.full(len(settings.grid), np.nan)
    if rule == "gcv":
        curve = gcv_curve(raw, settings.grid, settings.order, mean_mode)
    elif rule == "forecast_cv":
        curve = forecast_cv_curve(raw, settings.grid, settings.order, mean_mode, settings.cv_window, settings.cv_stride)
    else:
        raise ValueError("Unknown smoothness selection rule")
    return float(settings.grid[int(np.argmin(curve))]), curve


def lag_one_crosscorr(z: np.ndarray) -> np.ndarray:
    """Symmetric, signed cross-lag correlation feature matrix (not a metric)."""
    x, y = z[:, 1:], z[:, :-1]
    x = x - x.mean(axis=1, keepdims=True)
    y = y - y.mean(axis=1, keepdims=True)
    norm = np.maximum(1e-12, np.linalg.norm(x, axis=1)[:, None] * np.linalg.norm(y, axis=1)[None, :])
    r = np.clip((x @ y.T) / norm, -1, 1)
    r = 0.5 * (r + r.T)
    np.fill_diagonal(r, 0)
    return r
