# Canonical experiment notes

| Experiment | Subject | Status | Note |
|---|---|---|---|
| EXP-001 | Synthetic temporal-memory change, Pearson-vs-Sobolev network scores | Executed synthetic pilot, not confirmed | [Read mathematical design and archived figures](001-temporal_regime_pilot/README.md) |
| EXP-002 | Guerrero PLS smoothness vs B-splines, GCV/forecast-CV and lag-1 network | Replicated exploratory synthetic comparison | [Mathematics and completed-run figures](002-guerrero_smoothing/README.md) |
| EXP-003 | d=1..4 finite differences; nested outer LOOCV, GCV, fixed smoothness, Gaussian profile ML | Executed 4-seed synthetic study | [Derivations, RMSE, generated figures](003-order_lambda_loocv/README.md) |
| EXP-004 | PLS, B-spline, P-spline, natural smoothing splines, RBF kernel, polynomials; joint nested selection | Executed 6- and 30-seed comparisons | [Mathematical protocol and real plots](004-cross_family_methods/README.md) |

| EXP-005 | Orthogonal Legendre trend compression; $L^2/H^1$ metrics and lag-1/lag-2 baselines | Exploratory synthetic comparison | [Math and figures](005-orthogonal_polynomial_geometry/README.md) |
| EXP-007 | Controlled similarity targets: price scale, factor, lag and volatility | Reproducible simulation; remote archive generated on push | [Mathematics, code and figures](007-similarity_targets/README.md) |

[Mathematical notes](mathematics/001-hilbert_metrics.md) · [Research context and next experiments](../history/context/NEXT_STEPS.md)

Each future experiment requires matching numbered entries in `code/`, `figures/`, `results/`, and `notes/`. Notes must show exact mathematical definitions, baselines, generated figures, saved numerical evidence and limits. A proposed figure is a *table entry*, never a broken image link.

| EXP-006 | Window-local functional PCA, dynamic weighted kNN graphs and Laplacian spectra with direct lag controls and a no-switch null | Executed 8-seed synthetic study | [Mathematics and immutable figures](006-functional_pca_graphs/README.md) |
