"""Controlled synthetic market pairs with explicitly declared similarity targets.

Diagnostic counterexamples rather than realistic equity generators. Each call
uses an independent NumPy random generator; no real market data is accessed.
"""
from dataclasses import dataclass

import numpy as np

from .guerrero import fit_panel, functional_distance
from .metrics import correlation_distance, standardize_windows
from .polynomial_geometry import pairwise_polynomial_distance, project_onto_legendre


@dataclass(frozen=True)
class SyntheticCase:
    """A known-relationship panel with arrays shaped (assets, time)."""

    name: str
    returns: np.ndarray
    prices: np.ndarray
    latent_factor: np.ndarray


def prices_from_returns(returns: np.ndarray, initial_levels: tuple[float, ...]) -> np.ndarray:
    """Integrate log returns into strictly positive price paths, including time zero."""
    x = np.asarray(returns, dtype=float)
    p0 = np.asarray(initial_levels, dtype=float)
    if x.ndim != 2 or p0.shape != (x.shape[0],) or not np.isfinite(x).all():
        raise ValueError("Expected finite (assets,time) returns and matching initial prices")
    if np.any(p0 <= 0) or not np.isfinite(p0).all():
        raise ValueError("Initial levels must be finite and strictly positive")
    return p0[:, None] * np.exp(np.c_[np.zeros(x.shape[0]), np.cumsum(x, axis=1)])


def simulate_case(name: str, seed: int, length: int = 128) -> SyntheticCase:
    """Generate a controlled price-level, shared-factor, delayed, or volatility case."""
    if length < 32:
        raise ValueError("length must be at least 32")
    if name not in {"level_clone", "shared_factor", "lagged_factor", "volatility_clone"}:
        raise ValueError(f"Unknown synthetic case: {name}")
    rng = np.random.default_rng(seed)
    factor = rng.normal(0, 0.012, length)
    independent = rng.normal(0, 0.012, length)
    if name == "level_clone":
        x = np.vstack([factor, factor.copy(), independent])
    elif name == "shared_factor":
        noise = rng.normal(0, 0.0035, (2, length))
        x = np.vstack([factor + noise[0], factor + noise[1], independent])
    elif name == "lagged_factor":
        delayed = np.r_[rng.normal(0, 0.012, 2), factor[:-2]]
        x = np.vstack([factor, delayed, independent])
    else:
        x = np.vstack([factor, 2.0 * factor, independent])
    levels = (10000.0, 1000.0, 3500.0)
    return SyntheticCase(name, x, prices_from_returns(x, levels), factor)


def signed_lag_similarity(z: np.ndarray, max_lag: int = 3) -> np.ndarray:
    """Maximum signed sample correlation over symmetric relative lags.

    A similarity matrix, NOT a metric: lag maximization may violate the
    triangle inequality. Comparisons use overlapping time points only.
    """
    a = np.asarray(z, dtype=float)
    if a.ndim != 2 or max_lag < 0 or a.shape[1] <= 2 * max_lag + 5:
        raise ValueError("Invalid panel or maximum lag")
    n, _ = a.shape
    out = np.eye(n)
    for i in range(n):
        for j in range(i + 1, n):
            similarities = []
            for lag in range(-max_lag, max_lag + 1):
                if lag < 0:
                    u, v = a[i, :lag], a[j, -lag:]
                elif lag > 0:
                    u, v = a[i, lag:], a[j, :-lag]
                else:
                    u, v = a[i], a[j]
                u, v = u - u.mean(), v - v.mean()
                denominator = max(np.linalg.norm(u) * np.linalg.norm(v), 1e-14)
                similarities.append(float(u @ v / denominator))
            out[i, j] = out[j, i] = max(similarities)
    return out


def rms_pairwise(panel: np.ndarray) -> np.ndarray:
    """Root-mean-square distances between asset-by-time sample paths."""
    a = np.asarray(panel, dtype=float)
    if a.ndim != 2 or a.shape[0] < 2 or not np.isfinite(a).all():
        raise ValueError("Require finite asset-by-time panel")
    d = np.sqrt(np.mean((a[:, None, :] - a[None, :, :]) ** 2, axis=-1))
    np.fill_diagonal(d, 0.0)
    return d


def pairwise_scores(case: SyntheticCase) -> dict[str, np.ndarray]:
    """Return comparison distances with fixed prespecified tuning parameters.

    A distance on fitted functions can be a pseudometric on original series.
    Volatility contrast is a separate risk target, not an all-purpose score.
    """
    z = standardize_windows(case.returns)
    trend = fit_panel(z, 0.5, d=1, mean_mode="zero")
    poly = project_onto_legendre(trend, 7)
    rho = signed_lag_similarity(z, 3)
    volatility = np.std(case.returns, axis=1)
    return {
        "raw_price_rms": rms_pairwise(case.prices),
        "rebased_price_rms": rms_pairwise(case.prices / case.prices[:, :1]),
        "return_rms": rms_pairwise(case.returns),
        "pearson": correlation_distance(z),
        "pls_h1": functional_distance(trend, 0.20),
        "legendre_h1": pairwise_polynomial_distance(poly, derivative_weight=0.01),
        "max_lag3_correlation": np.sqrt(np.maximum(0, 2.0 * (1.0 - rho))),
        "volatility_log_ratio": np.abs(np.log(volatility[:, None] / volatility[None, :])),
    }
