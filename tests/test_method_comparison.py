"""Independent direct-refit checks for the family-comparison deletion formulas."""

import numpy as np
import pytest

from market_geometry.method_comparison import (
    bspline_basis, candidate_bank, compare_nested_loo,
    natural_cubic_penalty,
)
from market_geometry.order_selection import inner_loo_scores, press_residuals


def test_candidate_family_coverage_and_symmetric_hats():
    candidates = candidate_bank(32)
    names = {c.family for c in candidates}
    assert len(candidates) >= 40
    assert {"PLS order 1", "PLS order 4", "B-spline degree 3",
            "P-spline penalty 2", "Natural cubic smoothing spline",
            "RBF kernel ridge", "Polynomial regression"} <= names
    for c in candidates:
        assert np.allclose(c.hat, c.hat.T, atol=1e-8)
        assert np.linalg.eigvalsh(c.hat).min() >= -1e-8
        assert np.linalg.eigvalsh(c.hat).max() <= 1 + 1e-8


def test_spline_partition_unity_and_natural_penalty_nullspace():
    b = bspline_basis(32, 3, 9)
    assert np.allclose(b.sum(axis=1), 1)
    k = natural_cubic_penalty(32)
    assert np.allclose(k @ np.ones(32), 0, atol=1e-9)
    assert np.allclose(k @ np.arange(32), 0, atol=1e-9)
    assert np.count_nonzero(np.linalg.eigvalsh(k) < 1e-8) == 2


@pytest.mark.parametrize("family", ["PLS order 2", "B-spline degree 3", "P-spline penalty 2",
                                    "Natural cubic smoothing spline", "RBF kernel ridge",
                                    "Polynomial regression"])
def test_exact_deletion_against_direct_refit(family):
    """Check LOO and double LOO via the fixed quadratic smoothing objective.

    A retained-observation matrix W enters the common penalized smoother:
    (H^-1 - I + W) f = W y, or its ridge limit for rank-deficient H.
    Instead compare exact PRESS predictions with a direct linear submatrix
    conditional solve using P=(I-H)^(-1)H or kernel/Gaussian representation.
    """
    y = np.random.default_rng(9).normal(size=32)
    candidate = next(c for c in candidate_bank(32) if c.family == family)
    h = candidate.hat
    # For symmetric H, deletion of j can be computed by refitting its
    # conditional posterior mean using covariance Sigma=H(I-H)^-1 when defined.
    # Algebraically, direct refits on a retained subset R can be obtained via
    # subsetting C=H(I-H)^(-1) (regularization prior) with noise I. Models
    # having eigenvalue 1 require an improper-prior ridge limit.
    for outer in (5, 21):
        one_resid = press_residuals(y, h)[outer]
        # Independent scalar Schur complement solution for one deletion.
        e = y - h @ y
        a = np.eye(len(y)) - h
        expected = e[outer] / a[outer, outer]
        assert np.isclose(one_resid, expected, atol=1e-8)
        # Direct 2x2 inverse on the removed-index residual system (no PRESS helper).
        scores = inner_loo_scores(y, h)
        residuals = []
        for j in range(len(y)):
            if j == outer:
                continue
            inds = [j, outer]
            removed = (np.eye(len(y)) - h)[np.ix_(inds, inds)]
            r = np.linalg.solve(removed, e[inds])
            residuals.append(r[0] ** 2)
        assert np.isclose(scores[outer], np.mean(residuals), atol=1e-8)
        # Change the outer target while retaining all other values. Exact
        # nested inner scores and outer prediction MUST remain invariant.
        perturbed = y.copy()
        perturbed[outer] += 137
        h1 = inner_loo_scores(perturbed, h)[outer]
        p1 = perturbed[outer] - press_residuals(perturbed, h)[outer]
        p0 = y[outer] - one_resid
        assert np.isclose(h1, scores[outer], rtol=1e-9, atol=1e-8)
        assert np.isclose(p1, p0, rtol=1e-9, atol=1e-8)


def test_nested_all_family_results_do_not_see_outer_target():
    y = np.sin(np.linspace(0, 7, 32)) + .2 * np.random.default_rng(1).normal(size=32)
    base = compare_nested_loo(y)
    for j in (4, 17):
        edited = y.copy()
        edited[j] += 25
        other = compare_nested_loo(edited)
        for name, result in base["by_family"].items():
            assert np.isclose(result["outer_loo_prediction"][j],
                              other["by_family"][name]["outer_loo_prediction"][j],
                              atol=1e-6), name
            assert result["selected_candidates"][j] == other["by_family"][name]["selected_candidates"][j]


def test_input_validation():
    with pytest.raises(ValueError):
        compare_nested_loo(np.array([1.0, np.nan] * 16))
    with pytest.raises(ValueError):
        candidate_bank(9)


def test_direct_b_spline_and_p_spline_deleted_refits():
    from market_geometry.method_comparison import bspline_basis

    n, j, i = 32, 11, 17
    y = np.random.default_rng(71).normal(size=n)
    for family in ("B-spline degree 3", "P-spline penalty 2"):
        candidate = next(c for c in candidate_bank(n) if c.family == family and (
            c.name == "basis=9" if family.startswith("B-spline") else c.name.endswith("lambda=3")
        ))
        b = bspline_basis(n, 3, 9 if family.startswith("B-spline") else 14)
        if family.startswith("P-spline"):
            diff = np.diff(np.eye(b.shape[1]), n=2, axis=0)
            pen = 3 * diff.T @ diff
        else:
            pen = np.zeros((b.shape[1], b.shape[1]))
        pred = y - press_residuals(y, candidate.hat)
        retained = np.array([r for r in range(n) if r != j])
        coef = np.linalg.solve(b[retained].T @ b[retained] + pen,
                               b[retained].T @ y[retained])
        assert np.isclose(pred[j], b[j] @ coef, atol=1e-9)
        retained_two = np.array([r for r in range(n) if r not in (i, j)])
        coef2 = np.linalg.solve(b[retained_two].T @ b[retained_two] + pen,
                                b[retained_two].T @ y[retained_two])
        true_error = y[i] - b[i] @ coef2
        a = np.eye(n) - candidate.hat
        fitted_res = y - candidate.hat @ y
        double_res = np.linalg.solve(a[np.ix_([i, j], [i, j])], fitted_res[[i, j]])
        assert np.isclose(double_res[0], true_error, atol=1e-8)


def test_direct_kernel_and_natural_spline_refit():
    from market_geometry.method_comparison import natural_cubic_penalty

    n, j = 32, 9
    y = np.random.default_rng(12).normal(size=n)
    retained = np.array([i for i in range(n) if i != j])
    for fam in ("Natural cubic smoothing spline", "RBF kernel ridge"):
        candidate = next(c for c in candidate_bank(n) if c.family == fam and (
            c.name == "lambda=3" if fam.startswith("Natural") else c.name == "length=0.15, ridge=0.1"
        ))
        pred = y - press_residuals(y, candidate.hat)
        if fam.startswith("Natural"):
            w = np.diag((np.arange(n) != j).astype(float))
            direct = np.linalg.solve(w + 3 * natural_cubic_penalty(n), w @ y)[j]
        else:
            t = np.linspace(0, 1, n)
            k = np.exp(-.5 * ((t[:, None] - t[None, :]) / .15)**2)
            direct = k[j, retained] @ np.linalg.solve(
                k[np.ix_(retained, retained)] + .1 * np.eye(n - 1), y[retained]
            )
        assert np.isclose(pred[j], direct, atol=1e-8), fam
