# 2026-10-18 — EXP-007 simulation-first similarity targets

The initial goal is to specify which similarities a financial Hilbert-space metric should preserve, before tuning estimators. We compare price-scale clones, noisy common-factor paths, two-period lagged dependence, and proportional-volatility clones. Distinguish shape from risk; report retrieval rates only as descriptive statistics, not forecasting accuracy.

Added a reusable simulator and pairwise metric comparisons, an executable seeded runner, and numerical invariance tests. Fixed the metric choices before the simulation. The experiment is *not* a calibrated market DGP and does not establish superiority over lagged correlation.

Next: conditional covariance and stochastic-volatility designs, true graph recovery, negative controls, sequential calibration, and unseen future-risk prediction. Keep archives and source under the same numbered folders.
