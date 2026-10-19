# 2026-10-18 — Difference orders, nested reconstruction and Gaussian ML

## Why

User asked to generalize from second differences to any d, compare fitted trends and leave-one-out RMSE, and compare against a purported Guerrero (2007) maximum-likelihood lambda estimator. The publisher abstract and the existing Trend-Estimation mathematics support Guerrero's PLS, variance ratio and controlled smoothness index, but **not a verified claim that Guerrero proposed the ML lambda selector**. Build an explicit comparator without misattribution.

## What changed

- Generalized the spectral difference solver to arbitrary positive d<n and binomial basis for nonzero observed-difference reference mean; retained d2 behavior in EXP-002.
- Introduced `order_selection.py` with outer deletion, inner two-deletion PRESS and corresponding GCV scores, plus profile Gaussian ML under the difference covariance; estimated lambda without seeing outer test targets.
- Added `market_geometry/{code,notes,figures,results}/003-order_lambda_loocv`, new unit tests for algebra and held-out invariance, updated entry points, notes, manuscript roadmap and handoff.
- Generated actual 4-seed x 4-shape x 2-noise experiment results using n=48, d=1..4 and four selectors, saving one panel of individual predictions, figures and hash manifests.

## What happened

See `market_geometry/notes/003-order_lambda_loocv/README.md` for actual archived tables and figures. Small synthetic differences favor d1 in this setup but are not generalizable; Gaussian ML and classical inner LOOCV/GCV are closely matched for d1. Some ML optima may occur at prespecified bounds. No real-stock data used.

## Why the experiment is not sufficient for a paper

Nested LOOCV here measures pointwise interpolation, not chronological future forecast error. The Gaussian ML comparator assumes independent Gaussian innovation/measurement noise and uses only fully observed differences when an outer index is omitted. The tested AR1 scenario violates that assumption. Orders were compared post hoc, not selected by outer-nested validation. Planned next: controlled dependence experiments, real stock series, chronological CV, stronger time series baselines and financial market regime evaluation.

## Numerical refinement and rerun

The ML bounded optimizer was improved to locally refine all sampled minima and both endpoint neighborhoods, not just the most promising original neighborhood. The **first** completed run was frozen; a **second** completed run under `run_20261018T221759917877Z_s20261020_n4` was generated and validated with the current code. The revised ML differences are small and not statistically meaningful on four seeds.
