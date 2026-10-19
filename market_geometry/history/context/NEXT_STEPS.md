# Next research steps — updated 2026-10-18

| Priority | Study | Question / evaluation |
|---|---|---|
| Completed exploratory | EXP-001, EXP-002 | A memory-switch simulation: lag-one correlation already explains much of the synthetic regime change. |
| Completed exploratory | EXP-003 | d=1..4 and fixed/LOOCV/GCV/Gaussian-ML smoothing selection, observation-reconstruction RMSE. |
| Completed exploratory | EXP-004 | 49 configurations in 12 smoothing families, plus joint inner selector. 30 independent synthetic seeds × 8 scenarios, nested outer observed LOOCV. |
| Completed exploratory | EXP-005 | Degree-3/7 Legendre coordinates of PLS trends, exact Hilbert/Sobolev metrics, synthetic AR(1)/AR(2) regime detection and Pearson/lag-1/lag-2/B-spline controls. Strong matching lag controls win. |
| P0 | EXP-006, chronological forecasting + market regimes | Compare future risk/regime targets, functional PCA or graph spectra across matched smoothing families; Pearson, lagged covariance and spectral/coherence controls; nested time-only validation. |
| P0 | EXP-007, nonlinear and locally adaptive estimators | Properly refit LOESS, l1 trend filtering, state-space/Kalman and wavelet methods in nested observed and rolling chronological folds; robust outliers and abrupt jumps. |
| P1 | EXP-008, model robustness + nulls | Extend iid/AR1 DGPs to stochastic volatility, rare shocks, negative controls, weaker regime changes, diverse window lengths and larger independent seeds. |
| P1 | EXP-009, real financial data | Adjusted log returns, transaction calendars, licenses, survivorship and timestamp controls, outer time-series validation. |
| P2 | Mathematical results/paper | Theory of functional metrics and their information gain vs lag/coherence; precise novelty check; external validation; uncertainty over independent seeds rather than overlapping folds. |

Older working roadmaps used EXP-004 for a not-yet-run spectral study; that number is now occupied by the completed cross-family benchmark, so the spectral/chronological comparison is carried forward as a proposed EXP-006. EXP-005 now denotes the completed Legendre experiment. Preserve archival notes; never rewrite historical observations as prior results.

Every new experiment creates matching numbered `code`, `notes`, `figures`, `results`; document mathematical formulas, actual outputs and negative findings before claiming a successful result.


## After EXP-006

1. Fit/calibrate alarm thresholds with training-only rolling chronological blocks,
   stationary null simulations without artificial factor restarts, empirical null
   distributions and dependence-aware uncertainty.
2. Test truly independent out-of-time and changed-DGP regimes, high-dimensional
   asynchronous assets, varying factor structure, volatility and heavy tails.
3. Add graph spectral alternatives (heat kernels, diffusion distances, graph
   signal total variation) alongside cross-spectrum/coherence and lagged networks.
4. For eventual real equities, lock universe and data vintage before observing
   outcomes, ensure corporate-action-adjusted aligned returns, and measure future
   risk/forecast value against strong simple and DCC/dynamic conditional baselines.
5. Compare whether EXP-004's reconstruction-RMSE-best smoothing also gives the
   most useful downstream functional geometry; optimize only nested by time.

### Following EXP-007

First verify exact shape invariances and negative controls; then build a realistic conditional-covariance simulation atlas with heavy tails, volatility clustering, dynamic lag and time-changing factor loadings. Compare learned network recovery and calibrated out-of-time risk forecasts against lagged-correlation/DCC baselines.