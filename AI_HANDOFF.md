# AI handoff — FunctionalMarket-Geometry

Updated: 2026-10-18. Latest journal: [`2026-10-18-functional-pca-dynamic-graphs.md`](market_geometry/history/journal/2026-10/2026-10-18-functional-pca-dynamic-graphs.md).

## Project question

Can spline representations of aligned financial returns, equipped with time-varying Hilbert/Sobolev metrics, yield useful regime and relationship information *beyond* ordinary contemporary correlation networks and appropriate serial-dependence alternatives?

## Required reading

1. `AGENTS.md`
2. `market_geometry/notes/README.md`
3. `market_geometry/notes/mathematics/001-hilbert_metrics.md`
4. `market_geometry/notes/001-temporal_regime_pilot/README.md`
5. `market_geometry/history/journal/LATEST.md`
6. `market_geometry/history/context/STATE.md`
7. `paper/RESEARCH_DIRECTION.md` and `literature/references/README.md`

## Canonical implementation and evidence

- Library: `src/market_geometry/{metrics,simulation,rolling}.py`.
- Six numbered research experiments under `market_geometry/code/001-*`, `002-*`, `003-*`, `004-*`.
- Six matching numbered experiment directories: `code/`, `results/`, `figures/`, `notes/`.
- Smoke (untracked, only console): `python market_geometry/code/001-temporal_regime_pilot/run.py --quick`.
- Full archived: `python market_geometry/code/001-temporal_regime_pilot/run.py --publish`.
- The initial ZIP contains an actually generated and archived pilot run. Inspect exact run folders; **do not assume replicated validation**.

## Mathematics and methodological issues

- Raw standardized discrete L2 distance = sqrt(2(1-Pearson correlation)): no novel pairwise information.
- Positive derivative weight in an H1 distance changes the metric; B-spline projections can discard information; a metric on distinct fitted functions is only a pseudometric on raw data.
- The synthetic AR group-factor design varies serial covariance while holding population contemporaneous covariance constant. Finite-sample Pearson correlations can still vary. Lagged correlation *can already detect this change*.
- The current baseline compares descriptive network deviation scores, not inferred causal relationships or returns prediction.
- The earliest pilot intentionally uses a known regime-change boundary to identify evaluation segments; future unsupervised systems cannot use oracle boundaries.

## Immediate next tasks

- Run multiple independent synthetic seeds, report mean, uncertainty and negative findings.
- Add fair time-lagged correlation, spectral/coherence, cross-correlation and dynamic time-warping baselines; tune only on development data.
- Ablate spline basis, smoothing, derivative weight, window and stride. Report when derivative penalties amplify noise or fail.
- Real-stock ingestion with licensed/open adjusted close, temporal universe controls, log returns, source provenance and strict out-of-time validation.
- Formalize hypotheses and testability before claiming novelty. Record all steps in matching numbered experiment records.

**No remote push was made.** This is a locally generated ZIP, ready to be committed to an as-yet-uncreated user repository.

## EXP-002 — Guerrero smoothing comparison (2026-10-18)

- **Canonical note:** `market_geometry/notes/002-guerrero_smoothing/README.md`.
- **Code:** `src/market_geometry/guerrero.py`; `market_geometry/code/002-guerrero_smoothing/run.py`.
- Distinguish Guerrero's plug-in mean `mu_G=mean(D_d y)` from normalized smoothness `S` and from the newer, separately proposed chronological forecast-CV rule. See original `heri-espino/Trend-Estimation` research code and notes before modifying.
- Compare 9 matched pipelines, including a signed lag-one autocorrelation feature baseline; performance on a constructed AR-switch does not imply unique functional information.
- Selection inside windows uses raw-return historical forecasts to avoid inner-fold standardization leakage. Scoring uses known synthetic boundaries only after method selection.
- Full archived comparison: `python market_geometry/code/002-guerrero_smoothing/run.py --publish --seeds 12`. Read exact archive and scientific note before making claims.

## EXP-003 — finite-difference order and likelihood comparison (2026-10-18)

- Note + actual generated figures: `market_geometry/notes/003-order_lambda_loocv/README.md`.
- Model: `src/market_geometry/order_selection.py`, independently generalized `src/market_geometry/guerrero.py`.
- Program: `market_geometry/code/003-order_lambda_loocv/run.py`; `--quick` / `--publish`; all existing EXP-001/002 artifacts remain intact.
- Order d is integer 1<=d<n; pilot compares d=1,2,3,4. For matching predictors all selectors use mu=0. Guerrero's observed-difference mu appears in a **separate** descriptive gallery, not rated as exact LOO with mu reestimated.
- Evaluate **outer LOOCV** of observed sample imputation, with nested inner LOOCV, nested GCV, fixed Guerrero smoothness S, or standard Gaussian profile ML lambda derived from the stochastic assumption. No target heldout observation enters the selector. This is not forward forecasting or market regime detection.
- **Attribution correction:** original 2007 method uses the *controlled smoothness index*. We do NOT state Guerrero proposed the Gaussian ML rule; it is an additional standard variance-component comparator.
- The actual synthetic run uses 4 independent seeds, four trend shapes, two noise designs, n=48; see the run manifests and limitations. Interpret results across **independent seeds** and avoid presenting tiny differences as significant.
- Next: assess performance with real market data, blocked forecast-CV, and downstream functional network regime metrics; source-check actual historical lambda MLE procedures rather than assuming attribution.

## EXP-004 — matched smoothing families (2026-10-18)

- Canonical design + actual two archived runs: `market_geometry/notes/004-cross_family_methods/README.md`.
- Library: `src/market_geometry/method_comparison.py`; runner `market_geometry/code/004-cross_family_methods/run.py`.
- 49 candidate configs across 12 named families + one joint selector, compared via shared nested outer LOOCV. Explicit direct-deletion tests verify matching refits. 30-seed run is the main exploratory evidence; the 6-seed archive remains unchanged.
- Synthetic DGP matches EXP-003 four trend shapes and two noise processes, n=48; latent truth is diagnostic only. Winning on observed reconstruction does not imply winning on latent recovery, future forecasting or market regimes.
- Future: actual chronological h-step forecast validation, properly cross-validated Guerrero mu plug-in, LOESS, L1 trend filtering, Gaussian-process uncertainty, Kalman state-space, spectral/lagged baselines, and actual equities.
- EXP-005 has now been executed as orthonormal Legendre coefficient geometry under fixed PLS smoothing. Chronological forecasting remains an open EXP-006+ task. Historical proposed numbering is preserved in journal; read actual code and run evidence.

## EXP-005 — Orthogonal Legendre polynomial market geometry (2026-10-18)

- Note / actual generated figures: `market_geometry/notes/005-orthogonal_polynomial_geometry/README.md`. Core reusable formulas: `src/market_geometry/polynomial_geometry.py`. Runner: `market_geometry/code/005-orthogonal_polynomial_geometry/run.py`.
- The finite polynomial subspace is an ordinary Hilbert space; Legendre orthonormality gives exact polynomial $L^2$ coefficient distance, and the derivative Gram matrix gives $H^1$ distance on normalized window time. Discrete LS coefficient fitting is not the exact continuous projection.
- Fixed PLS smoothness $S=0.5$, order $d=1,2$, optional Guerrero plug-in mean, Legendre degrees 3/7, $H^1$ weight 0.01. These are exploratory rather than selected by LOOCV.
- Compare Pearson and **both signed lag-1 and lag-2** cross-asset feature matrices, plus B-spline Sobolev. The AR(2) switch illustrates why lag-1 alone is inadequate. Eight independent seeds per DGP, immutable output.
- Result: both spline and polynomial approaches discriminate the constructed changes, but the relevant lag baseline has AUROC 1.000; there is no demonstrated incremental method benefit. No real financial data, no future-risk predictions, and no independent regime-seed validation for selecting a winner.
- Next: out-of-time, nested hyperparameter selection with real or more realistic null/weak-shift controls; functional PCA or graph spectra beyond predictionless retrospective AUROC.


## EXP-006 — window-local functional PCA + graph spectra (2026-10-18)

- Reusable library: `src/market_geometry/functional_pca_graph.py`; runner:
  `market_geometry/code/006-functional_pca_graphs/run.py`, canonical notes:
  `market_geometry/notes/006-functional_pca_graphs/README.md`.
- Fixed PLS d=1 S=0.5, Legendre p=7, H1 alpha=.01, FPCA r=2/3/5,
  weighted k=3 union kNN graphs, calibration-median fixed Gaussian bandwidth.
- Compare direct Pearson/lagged correlation, their graph edges/spectra,
  full coefficient graph, and raw/PLS FPCA graphs. FPCA here is per-window
  cross-sectional PCA; signs/axes cannot be directly compared across windows.
- 8-seed synthetic AR1/AR2/null run: FPCA rank3 graph edge AUROC .998/.980,
  versus correctly matched direct lag AUROC **1.000/1.000** (AR1 lag1 and AR2 lag2). Null quantiles show
  large false-alarm diagnostics; **do not interpret as calibrated rates**.
- Full run archives, SHA256, actual PNG and SVGs, seed results, tests; no real
  stocks or predictive claims. See full note before quoting numbers.

## EXP-007 — Controlled simulation targets (2026-10-18)

- [Canonical note](market_geometry/notes/007-similarity_targets/README.md); library: src/market_geometry/similarity_targets.py; runner: market_geometry/code/007-similarity_targets/run.py.
- Four controls: same returns at different levels; noisy shared factor; exact two-day delayed factor; return path scaled twofold in volatility.
- Separate shape invariance, lag-aware dependence and volatility/risk: a shape match is not necessarily a risk match.
- Fixed methods and seed-range, no oracle tuning; use 100 independent seeds starting 42000. Pair-retrieval is descriptive, not held-out forecasting.
- EXP-007 creates immutable run artifacts and appends real figures to its note only after execution.
- Existing EXP-001–006 results must not be modified. Future research: conditional covariance, heavy tails, volatility clustering and chronological regime recovery.