# 2026-10-18 — Orthonormal polynomial Hilbert geometry, EXP-005

## Why

User proposed approximating already estimated stock trends by a polynomial, then experimenting in the Hilbert space of polynomial functions. This is scientifically different from EXP-004 polynomial regression against noisy samples, which had relatively poor observed-LOOCV RMSE. The first question is whether low-dimensional function coefficients can preserve temporal network changes.

## What changed

- Introduced `polynomial_geometry.py` for orthonormal shifted Legendre bases, fixed-grid least-squares coefficient estimation, exact polynomial $L^2$ coefficient norms and analytic/quadrature $H^1$ Gram matrices.
- Added synthetic-memory-switch EXP-005 with fully trailing observations, AR(1) and AR(2) group factor structures, per-seed method AUROC, two polynomial degrees and three PLS variants.
- Compared Pearson, signed lag-1 and lag-2 correlation feature matrices, and B-spline/Sobolev networks. Saved actual generated PNG/SVG figures, independent-seed CSV/JSON, and hash evidence with immutable run ID.
- Added tests proving/evaluating basis orthonormality, coefficient reconstruction, L2/Euclidean identity, Sobolev derivative exactness, metric properties and invalid parameter checks.

## Actual results

Eight seed panels per scenario with 12 synthetic assets, 480 timestamps, regime change at 240, 64-day trailing window and stride 12:

| Method | AR(1) mean AUROC | AR(2) mean AUROC |
|---|---:|---:|
| Pearson | 0.575 | 0.448 |
| Signed lag-1 | 1.000 | 0.092 |
| Signed lag-2 | 0.583 | 1.000 |
| B-spline/Sobolev | 0.980 | 0.849 |
| PLS(1), Legendre(7), H1 | 0.909 | 0.812 |
| PLS(2), Legendre(7), H1 | 0.916 | 0.801 |

These are retrospective scores computed against a known synthetic switch. Matching lag baselines perform best; the polynomial representation is not uniquely informative in these regimes. Look at the actual run CSV for all methods and across-seed variability. The AR(2) lag-1 score of 0.092 is a **fixed signed anomaly-score direction** result, not proof lag-1 contains absolutely no useful information or that reversing the detector would fail.

## Caveats / next steps

- No real data, no future price/risk prediction, no genuine out-of-time supervised validation, no posterior uncertainty over method selection.
- Degree and H1 weight are predetermined rather than LOOCV-selected; this is intentional to prevent selecting on scenario labels. The B-spline derivative is evaluated in daily units while Legendre derivatives are on normalized window time; a scale-matched derivative weight sensitivity study is still needed.
- Polynomial coordinate distances are exact *for the fitted polynomial*, not for the original returns; projection can lose high-frequency signals.
- Prior art includes 2025 Colombian-market FDA with spline Hilbert geometry, 2024 derivative FDA risk analysis, 2025 functional CAPM with basis expansions. No novelty established.
- Follow with richer DGPs, spectral and coherence controls, functional PCA, eigenvector consistency, and then nested chronological holdouts on financial data.
