# Project state — 2026-10-18

**Project:** FunctionalMarket-Geometry (new, local ZIP; no GitHub repository created or pushed).

**Scientific status:** Method conception with two executed synthetic experiments (including a 12-seed Guerrero comparison); no generalization, novelty, causal-dependence detection or financial performance established.

**Implemented:** Deterministic stationary-variance Gaussian group-factor AR panel, rolling population-zscored return vectors, exact correlation-chord distances, common penalized B-spline geometry, daily-unit H1 first derivative metric, train-only reference network deviation, reserved pre/post evaluation and four figure exporters.

**Archiving:** EXP-001 run evidence under matching `market_geometry/{code,notes,figures,results}/001-temporal_regime_pilot/`, plus metadata and SHA-256 manifest; new run IDs not overwritable.

**Verified:** Exact tool-specific validation findings should be recorded in `VALIDATION.md` following the build; do not infer CI success from local passing tests.

**Key mathematical caveat:** Standardized raw Euclidean distances simply re-express Pearson correlations; the derivative metric's incremental value has not been established against lagged baselines.

**2026-10-18 (later):** Added EXP-002 to compare existing Guerrero/PLS trend estimators (zero vs plug-in mean; fixed, GCV and chronological CV smoothness) with B-spline Sobolev and lag-1 cross-correlation. New reproducible source `src/market_geometry/guerrero.py`; read EXP-002 note/results for execution state. No real data or confirmed novel result.

**2026-10-18 (EXP-003 completed):** Generalized Guerrero-compatible PLS to 1<=d<n, implemented nested outer-LOOCV for order d=1..4 with inner PRESS/GCV/fixed-S plus Gaussian profile-ML lambda. Archived 4-seed, 8-scenario-per-seed synthetic observations/latent diagnostics and actual figures in EXP-003. Correct paper attribution: 2007 selection method is smoothness-index based; ML is an independently implemented comparator under the Gaussian ratio model. No real market or regime-detection inference from EXP-003.

## 2026-10-18 EXP-004 extension

Cross-family observed nested-LOOCV reconstruction study added and executed, preserving EXP-001/002/003. See 004 note and run manifests. Twelve families/49 configs, including a genuinely nested joint all-family selector; 6- and 30-seed synthetic runs. No real data and no future forecasts. LOESS/L1/Kalman/wavelets have only been planned. Next critical work: chronological and downstream network evaluation with lagged and spectral controls.

## 2026-10-18 EXP-005 extension

Executed a bounded synthetic AR(1)/AR(2) memory-switch benchmark with orthonormal Legendre coefficients of raw and PLS-smoothed return paths, exact polynomial $L^2$ / $H^1$ distances and fixed lag-1/lag-2 and spline controls. Eight independently seeded panels per scenario. See the numbered EXP-005 notes, specific run's JSON/CSV, figures and SHA-256 manifest; 39 tests pass locally. Reference baselines beat polynomial geometry on the deliberately constructed lag switches. The correct research contribution, if any, would need to address richer dynamics, robust information gain and forward-in-time validation, not mere functional representation.


## 2026-10-18 EXP-006 — Functional PCA and graph spectra

Implemented cross-sectional FPCA of Hilbert-transformed degree-7 Legendre trends
and weighted kNN graph edge/spectral time series (fixed bandwidth calibrated from
pre-event windows only). An 8-seed AR1/AR2/null synthetic archive is present, with
six pairs of PNG/SVG figures, CSV/JSON/NPZ evidence and SHA-256 manifest.
The matched signed lag baseline remains best, graph eigenvalues are a lossy summary,
and null threshold exceedance is high. Full evidence is in EXP-006 notes; no paper
novelty, alarm calibration or financial forecasting superiority is established.

## EXP-007: simulation-first metric targets

Added the controlled synthetic simulator, lag/shape/risk comparisons, runner and invariance tests, preserving EXP-001–006. The GitHub workflow archives a 100-seed run to matching results and figures folders when it passes validation. No real equity risk prediction has been established.
