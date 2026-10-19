"""Window-local functional PCA, weighted proximity graphs and graph spectra.

Finite Legendre coefficients describe functions in a Hilbert subspace. These
routines implement FPCA of the *cross-section of assets in one trailing window*,
not the separate theory of dynamic FPCA based on long-run covariance operators.
All transforms are deterministic, symmetric and use no future observations.
"""

from dataclasses import dataclass

import numpy as np

from .polynomial_geometry import derivative_gram


@dataclass(frozen=True)
class FunctionalPCA:
    """FPCA output for one asset-by-coefficient matrix.

    ``scores`` are n_assets x n_components; ``directions`` are orthonormal
    in the whitened finite-dimensional coefficient space. Eigenvector signs
    are arbitrary, so compare distances/spectra rather than raw score columns
    across independently fitted windows.
    """

    scores: np.ndarray
    explained_variance_ratio: np.ndarray
    eigenvalues: np.ndarray
    directions: np.ndarray


def hilbert_coordinates(coefficients: np.ndarray, alpha: float = 0.01) -> np.ndarray:
    """Map Legendre coefficients to Euclidean coordinates preserving H1 geometry.

    With Gram ``G = I + alpha*R``, Cholesky factor ``L`` satisfies G=L L^T.
    Return coefficients @ L, so pairwise squared Euclidean distance is
    exactly (a_i-a_j)^T G (a_i-a_j) up to floating-point error.
    """
    a = np.asarray(coefficients, dtype=float)
    if a.ndim != 2 or a.shape[0] < 2 or a.shape[1] < 1 or not np.isfinite(a).all():
        raise ValueError("Require finite (n_assets>=2, p>=1) coefficient matrix")
    if not np.isfinite(alpha) or alpha < 0:
        raise ValueError("alpha must be nonnegative and finite")
    gram = np.eye(a.shape[1]) + alpha * derivative_gram(a.shape[1] - 1)
    return a @ np.linalg.cholesky(gram)


def fit_functional_pca(features: np.ndarray, components: int) -> FunctionalPCA:
    """Estimate cross-sectional PCA using centered Hilbert coordinates.

    The components span the leading eigenvectors of the empirical asset
    covariance. They are fit independently on each trailing window. The
    number of eigen-directions with meaningful variance is at most n_assets-1.
    """
    x = np.asarray(features, dtype=float)
    if x.ndim != 2 or min(x.shape) < 2 or not np.isfinite(x).all():
        raise ValueError("Expected finite (n_assets>=2, n_features>=2)")
    if not isinstance(components, (int, np.integer)) or not 1 <= components <= min(x.shape):
        raise ValueError("components outside allowable range")
    centered = x - x.mean(axis=0, keepdims=True)
    _, singular_values, vt = np.linalg.svd(centered, full_matrices=False)
    variance = singular_values**2 / (x.shape[0] - 1)
    total = float(variance.sum())
    ratios = variance / total if total > 1e-20 else np.zeros_like(variance)
    return FunctionalPCA(
        scores=centered @ vt[:components].T,
        explained_variance_ratio=ratios,
        eigenvalues=variance,
        directions=vt[:components].T,
    )


def pairwise_euclidean(x: np.ndarray) -> np.ndarray:
    """Symmetric Euclidean distance matrix between feature rows."""
    a = np.asarray(x, dtype=float)
    if a.ndim != 2 or a.shape[0] < 2 or not np.isfinite(a).all():
        raise ValueError("Require at least two finite vectors")
    sq = np.sum(a * a, axis=1)
    d2 = np.maximum(0.0, sq[:, None] + sq[None, :] - 2 * a @ a.T)
    out = np.sqrt(d2)
    np.fill_diagonal(out, 0.0)
    return out


def proximity_graph(distances: np.ndarray, *, k: int = 3, bandwidth: float = 1.0) -> np.ndarray:
    """Gaussian-weighted, union-symmetrized kNN graph.

    ``bandwidth`` is meant to be pre-calibrated from past windows, rather than
    using event labels or information after an operational graph timestamp.
    Ties are broken reproducibly by asset index through stable mergesort.
    """
    d = np.asarray(distances, dtype=float)
    if d.ndim != 2 or d.shape[0] != d.shape[1] or d.shape[0] < 3:
        raise ValueError("Expected square distance matrix with >=3 assets")
    if not np.isfinite(d).all() or np.min(d) < -1e-10 or not np.allclose(d, d.T, atol=1e-8):
        raise ValueError("Distances must be finite, nonnegative and symmetric")
    if not np.allclose(np.diag(d), 0, atol=1e-8):
        raise ValueError("Distances must have zero diagonal")
    if not isinstance(k, (int, np.integer)) or not 1 <= k < len(d):
        raise ValueError("Require integer 1<=k<n_assets")
    if not np.isfinite(bandwidth) or bandwidth <= 0:
        raise ValueError("Require finite strictly positive bandwidth")
    work = d.copy()
    np.fill_diagonal(work, np.inf)
    neighbors = np.argsort(work, axis=1, kind="stable")[:, :k]
    mask = np.zeros_like(work, dtype=bool)
    mask[np.arange(len(d))[:, None], neighbors] = True
    mask |= mask.T
    weights = np.exp(-0.5 * (d / bandwidth) ** 2) * mask
    np.fill_diagonal(weights, 0.0)
    return weights


def normalized_laplacian_eigenvalues(weights: np.ndarray) -> np.ndarray:
    """Sorted eigenvalues of normalized Laplacian I-D^-1/2 W D^-1/2.

    Isolated vertices have zero diagonal under the normalized-Laplacian
    convention, contributing an extra zero eigenvalue. Graphs in this study
    are symmetrized kNN graphs and thus contain no isolated vertices for k>=1.
    """
    a = np.asarray(weights, dtype=float)
    if a.ndim != 2 or a.shape[0] != a.shape[1] or not np.isfinite(a).all():
        raise ValueError("Expected finite square adjacency")
    if np.min(a) < -1e-12 or not np.allclose(a, a.T, atol=1e-9):
        raise ValueError("Expected symmetric nonnegative adjacency")
    deg = a.sum(axis=1)
    invroot = np.where(deg > 1e-14, 1 / np.sqrt(np.maximum(deg, 1e-14)), 0.0)
    lap = np.eye(len(a)) - (invroot[:, None] * a * invroot[None, :])
    np.fill_diagonal(lap, np.where(deg > 1e-14, np.diag(lap), 0.0))
    return np.clip(np.linalg.eigvalsh(lap), 0.0, 2.0)


def historic_bandwidth(distances: np.ndarray, calibration_mask: np.ndarray) -> float:
    """Fix positive kernel bandwidth using calibration windows only."""
    values = np.asarray(distances, dtype=float)
    flag = np.asarray(calibration_mask, dtype=bool)
    if values.ndim != 3 or len(flag) != len(values) or flag.sum() < 2:
        raise ValueError("Require 3D distance series and >=2 calibration windows")
    upper = np.triu_indices(values.shape[1], 1)
    v = values[flag][:, upper[0], upper[1]]
    pos = v[np.isfinite(v) & (v > 1e-12)]
    if pos.size < 1:
        raise ValueError("Cannot calibrate a positive bandwidth")
    return float(np.median(pos))


def rolling_graph_features(distances: np.ndarray, calibration_mask: np.ndarray, k: int = 3):
    """Build graphs with fixed training-only scale; emit edges and spectra.

    Outputs: ``(edge_features, spectra, algebraic_connectivity, bandwidth)``.
    The edge vector retains asset identities; eigenvalues remove node labels.
    """
    values = np.asarray(distances, dtype=float)
    scale = historic_bandwidth(values, calibration_mask)
    upper = np.triu_indices(values.shape[1], 1)
    graphs = [proximity_graph(d, k=k, bandwidth=scale) for d in values]
    edges = np.stack([g[upper] for g in graphs])
    spectra = np.stack([normalized_laplacian_eigenvalues(g) for g in graphs])
    return edges, spectra, spectra[:, 1], scale


def reference_feature_score(features: np.ndarray, calibration_mask: np.ndarray) -> np.ndarray:
    """Euclidean deviation of feature rows from their *past* calibration mean."""
    x = np.asarray(features, dtype=float)
    cal = np.asarray(calibration_mask, dtype=bool)
    if x.ndim != 2 or x.shape[0] != len(cal) or cal.sum() < 2:
        raise ValueError("Invalid features / calibration")
    return np.sqrt(np.mean((x - x[cal].mean(axis=0)) ** 2, axis=1))
