# Runnable experiment implementations

| Number | Program | Output |
|---|---|---|
| 001 | [`001-temporal_regime_pilot/run.py`](001-temporal_regime_pilot/run.py) | Full `--publish`: immutable CSV/JSON/NPZ and PNG/SVG, appended actual note embeds; `--quick`: console-only smoke |
| 002 | [`002-guerrero_smoothing/run.py`](002-guerrero_smoothing/run.py) | Full `--publish`: 12 synthetic seeds, nine matched comparators, immutable seed-level CSV/JSON and PNG/SVG; `--quick`: console only |
| 003 | [`003-order_lambda_loocv/run.py`](003-order_lambda_loocv/run.py) | Full `--publish`: d=1..4, four selectors, nested outer LOOCV, Gaussian ML comparator, CSV/JSON and PNG/SVG; `--quick`: console only |
| 004 | [`004-cross_family_methods/run.py`](004-cross_family_methods/run.py) | Full `--publish`: 49 configs in 12 estimator families plus fully nested multi-family selector; immutable results and figures; `--quick`: console only |

| 005 | [`005-orthogonal_polynomial_geometry/run.py`](005-orthogonal_polynomial_geometry/run.py) | Fixed-order PLS/Legendre geometry; AR(1)/AR(2) regime network comparison and plots |

Reusable estimators belong in [`../../src/market_geometry/`](../../src/market_geometry/), not experiment notebooks.

- EXP-006: [Runnable record for functional PCA and dynamic graph spectra](006-functional_pca_graphs/) — matched simulations, no real assets.
