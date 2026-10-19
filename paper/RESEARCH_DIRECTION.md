# Research direction: functional geometry of financial dependence

**Working title:** *Functional Geometry of Financial Markets: Time-Varying Hilbert-Space Metrics and Regime Detection*

**Status:** 2026-10-18 conceptual framing plus one synthetic pilot. Publication potential is unverified; novelty and empirical gains have not been established.

## Central problem

Given aligned return panel $R\in\mathbb R^{n\times T}$, construct trailing-window functional embeddings $f_{i,w}\in H^1([0,1])$, compare pairwise distances $D_w(i,j)$ and identify regime changes using only pre-event calibration information. Evaluate whether these representations capture interpretable financial-market changes missed by **specified** contemporaneous-correlation-network procedures while comparing against more appropriate serial-dependence baselines.

## Hypotheses

**H0 (redundant geometry):** A proposed distance based only on normalized sampled L2 geometry is algebraically equivalent to contemporaneous correlation distance (true). More generally, added temporal features provide no incremental held-out evidence over comparably tuned autocorrelation/spectral baselines (to be tested).

**H1 (temporal information):** A positive-derivative Sobolev metric may respond to changes in short-range frequency/serial structure without changes in same-time population covariance. This does **not** imply uniqueness or improvement over existing serial methods.

**H2 (practical value):** In out-of-time empirical data, a pre-specified functional network score may add stable regime information conditional on contemporaneous- and lagged-dependence baselines. Outcome unknown.

## Model family

$$
\widehat f_{i,w}(u)=\sum_k \widehat\beta_{ik,w} B_k(u),\quad
\widehat\beta_{i,w}=\arg\min_\beta\|z_{i,w}-B\beta\|_2^2+\gamma\|D_2\beta\|_2^2.
$$

$$
d_{\lambda,w}^2(i,j)=\int_0^1\bigl[(\widehat f_i-\widehat f_j)^2+\lambda(\partial_t\widehat f_i-\partial_t\widehat f_j)^2\bigr]\,du.
$$

The first-difference derivative penalty is also expressible through lag-one autocovariances under stationarity (see mathematical Proposition 3); **this is not a novel dependence estimator**. Any incremental gain over lagged statistics is empirical and unverified.

Time-varying here means the **estimated distance matrix changes across trailing windows**, not that the kernel hyperparameter $\lambda$ is fitted with hindsight at each window. Alternative choices: cross-covariance operators, Hilbert-Schmidt norm, kernel mean embeddings, phase/amplitude decomposition, Wasserstein distances and lag-aware operators; *all require prior-art review*.

## Baseline ladder (do not skip)

1. Pearson correlation distance, rolling network distance score (implemented in EXP-001).
2. Ordinary time-series lag-1 autocorrelation and cross-correlation, lead–lag networks (planned; *necessary* for EXP-001).
3. Time-series power spectra, coherence, spectral densities; robust/volatility-standardized correlation networks (planned).
4. Distance correlation, mutual information estimates, DTW, DCC-GARCH / appropriate dynamic conditional correlation (planned, depending on objective).
5. Spline L2 with $\lambda=0$, the derivative-weighted variant, and raw standardized L2 controls (planned ablations).

## Validation hierarchy

- Known-ground-truth synthetic mechanisms with matched covariance and varied temporal structure; negative controls where derivative metric **should not** win.
- Repeated independent seeds and regime mechanisms: persistent, alternating, heavy tails, stochastic volatility, structural correlation, common shocks, asynchrony and missing data.
- Chronological train/validation/test on real assets, with event labels specified externally before evaluation. Embargo to limit overlapping-window leakage and multiple-comparison correction or conservative statistical testing.
- Bootstrap uncertainty respecting serial dependence, effect sizes, robustness to configuration, portfolio non-predictive diagnostics first. If predictive value is alleged, construct a dedicated out-of-sample protocol and compare transaction costs.

## Theoretical questions

- Prove Hilbert metric/pseudometric properties and exact raw L2–correlation equivalence (see mathematical notes).
- Characterize $\mathbb E[d_{\lambda,w}^2]$ under group-factor AR processes as a function of serial covariance and smoothing operator.
- Study stability of spline representations under weak dependence and irregular sampling; show bounds with explicit regularity assumptions.
- Operator/spectral kernel perspective may give a research contribution **only if genuinely distinct from prior literature**.

## Expected manuscript sections

Introduction; related functional and dynamic-network literature; mathematical construction; baselines and identifiability; controlled experiments; real financial panel with chronological validation; limitations; reproducibility statement.

## Related research

See [`literature/references/README.md`](../literature/references/README.md). Saart & Xia (2022) already study asset-return co-movement via functional time-series methods and functional PCA. Isogai (2017) studies dynamic correlation networks. Thus the project must not claim novelty for combining FDAs, financial assets and temporal networks alone.


## New smoothing ablation — EXP-002

Separately test the **representation estimator** (B-spline vs PLS) and the **smoothness selection rule** (fixed $S$, GCV, chronological forecast-CV). PLS uses Guerrero's observed difference mean as one option; it is a classical method and does not imply a new dependence estimator. See `market_geometry/notes/002-guerrero_smoothing/README.md`, with synthetic replicated seed-level evidence and a mandatory signed lag-one comparison.


## Order and lambda selection — EXP-003

Explore how finite-difference trend order d and smoothness selection interact with reconstruction of stock-return trends. Archive nested outer-LOOCV RMSE vs fixed controlled S, inner LOOCV, inner GCV and Gaussian profile ML. The latter follows a Gaussian variance-ratio assumption compatible with Guerrero's 2007 stochastic setup but is **not** claimed to be his proposed lambda selector. Distinguish this observation-reconstruction target from forecasting and from downstream financial-regime detection; none is automatically optimized by the other. See `market_geometry/notes/003-order_lambda_loocv/README.md`.

## Updated conceptual target after EXP-005

Orthonormal polynomial coefficients and $H^1$ Gram matrices are **classical functional data analysis**, not an original theorem. EXP-005 demonstrated compression plus regime-sensitivity in a synthetic experiment, but matched lag-1/lag-2 baselines detected the constructed changes perfectly. Potentially publishable work requires a decision-relevant target and a genuinely stronger comparator set: functional PCA or changing graph spectra, more complex noise/regime DGPs, null calibration, conditional information gains and out-of-time replication on actual equities. Report failure relative to basic lag models instead of omitting those controls. See `market_geometry/notes/005-orthogonal_polynomial_geometry/README.md` for source-verified related work.


## After EXP-006: FPCA and time-varying graph spectra

The window-local cross-sectional FPCA + kNN spectral-graph study is now
implemented and archived; no market data or retrospective tuning was used.
Exploratory evidence: rank-3 fPCA/PSS Sobolev graph edges identify the constructed
AR1/AR2 changes (mean AUROC .998/.980) but matched lag baselines achieve 1.0/1.0.
Graph Laplacian eigenvalue spectra contain less discriminative information than
labeled edge vectors in these simulations. Null-noise calibration is poor.
This **does not** establish novel mathematics or empirical advantage.
Closest examples: moving-window finance fPCA (Economics Letters 2014;
DOI 10.1016/j.econlet.2013.09.026) and published dynamic finance network
work (PLOS ONE 2025; DOI 10.1371/journal.pone.0319985). The methods require
specific value over careful lag/cross-spectrum benchmarks and future outcomes.
