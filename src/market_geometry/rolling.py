"""Causal rolling-window comparisons and descriptive score calibration."""

from dataclasses import dataclass

import numpy as np

from .metrics import SplineSettings, correlation_distance, pairwise_vector, spline_metric, standardize_windows


@dataclass(frozen=True)
class RollingSettings:
    """Rolling window length and stride (trading observations)."""

    window: int = 80
    stride: int = 10


def rolling_distances(
    returns: np.ndarray, config: RollingSettings, spline: SplineSettings
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Estimate distances using trailing returns only (no forward information).

    Parameters
    ----------
    returns : ndarray of shape (n_assets, n_days)
        Aligned asset log returns, synthetic or real.
    config : RollingSettings
        Window length and observation stride.
    spline : SplineSettings
        Functional metric and fitting configuration.

    Returns
    -------
    ends, correlation_distances, functional_distances : tuple of ndarrays
        ``ends`` are exclusive window end indices; remaining arrays have
        shape (n_windows, n_assets, n_assets).
    """
    v = np.asarray(returns, dtype=float)
    if v.ndim != 2 or config.stride <= 0 or config.window < 10 or config.window > v.shape[1]:
        raise ValueError("Invalid panel/window/stride")
    ends = np.arange(config.window, v.shape[1] + 1, config.stride)
    correlation, functional = [], []
    for end in ends:
        z = standardize_windows(v[:, end - config.window : end])
        correlation.append(correlation_distance(z))
        functional.append(spline_metric(z, spline))
    return ends, np.stack(correlation), np.stack(functional)


def reference_deviation(
    matrices: np.ndarray, calibration_mask: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Frobenius edge deviation from training-only mean network distances.

    Calibration uses only caller-designated windows. Outputs are descriptive
    anomaly scores, not calibrated probabilities or financial dependencies.

    Returns
    -------
    score : ndarray of shape (n_windows,)
        Root-mean-squared edge deviation from calibration mean.
    reference : ndarray of shape (n_edges,)
        Training-period mean pairwise-distance vector.
    """
    x = np.stack([pairwise_vector(d) for d in matrices], axis=0)
    mask = np.asarray(calibration_mask, dtype=bool)
    if mask.shape != (x.shape[0],) or mask.sum() < 2:
        raise ValueError("Calibration mask must identify at least two windows")
    center = x[mask].mean(axis=0)
    return np.sqrt(np.mean((x - center) ** 2, axis=1)), center
