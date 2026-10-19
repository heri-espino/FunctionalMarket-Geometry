# FunctionalMarket-Geometry

**Working paper:** *Functional Geometry of Financial Markets: Time-Varying Hilbert-Space Metrics and Regime Detection*

**Question:** Can a time-varying functional metric reveal temporal patterns in financial co-movement that contemporaneous correlation networks fail to describe?

**Research status (2026-10-18):** Exploratory methodological project, **not** an established new theory or investment model. The ZIP includes *six completed, reproducible, synthetic-data experiments*; no real stock prices are bundled and no financial outperformance is claimed.

## Repository map

The structure mirrors [ProximityGraphs-Isomap](https://github.com/heri-espino/ProximityGraphs-Isomap): [numbered experiment notes](market_geometry/notes/README.md), matching [`code/`](market_geometry/code/README.md), [`results/`](market_geometry/results/README.md), [`figures/`](market_geometry/figures/README.md), a reusable [`src/market_geometry`](src/market_geometry), mathematical notes, literature, a manuscript folder, and preserved research history.

```text
FunctionalMarket-Geometry/
├── AGENTS.md, AI_HANDOFF.md, README.md
├── environment.yml, pyproject.toml
├── src/market_geometry/                 # Reusable research library
├── market_geometry/
│   ├── code/001-temporal_regime_pilot/
│   ├── code/002-guerrero_smoothing/
│   ├── code/003-order_lambda_loocv/
│   ├── code/004-cross_family_methods/
│   ├── code/005-orthogonal_polynomial_geometry/
│   ├── code/006-functional_pca_graphs/
│   ├── notes/001-temporal_regime_pilot/   # Mathematics + actual figure embeds
│   ├── notes/002-guerrero_smoothing/
│   ├── notes/003-order_lambda_loocv/
│   ├── notes/004-cross_family_methods/
│   ├── notes/005-orthogonal_polynomial_geometry/
│   ├── notes/006-functional_pca_graphs/
│   ├── notes/mathematics/
│   ├── figures/001-temporal_regime_pilot/runs/<RUN_ID>/
│   ├── results/001-temporal_regime_pilot/runs/<RUN_ID>/
│   └── history/{journal,context}/
├── paper/                              # Research direction + manuscript outline
├── literature/{pdf,markdown,references}/
├── tests/, scripts/, examples/, docs/, .github/workflows/
└── .gitignore
```

## Quick start (Windows PowerShell + Miniforge)

```powershell
conda env create -f environment.yml
conda activate functional-market-geometry
python -m pip install -e .
.\scripts\run_all.ps1 -Quick
```

Or with a standard Python 3.11+ environment:

```bash
python -m pip install -e '.[dev]'
python -m pytest -q
python market_geometry/code/001-temporal_regime_pilot/run.py --quick
python market_geometry/code/001-temporal_regime_pilot/run.py --publish
```

`--quick` prints a synthetic smoke-test summary and does **not** change experiment records. `--publish` runs the full 720-observation synthetic pilot, writes immutable CSV/JSON/NPZ with SHA-256 evidence and PNG/SVG figures, and appends actual figure references to its canonical note. **The ZIP already contains the first run, generated and checked in this workspace.** Repeating `--publish` makes a new run rather than overwriting the original.

## EXP-002: Guerrero-style trend smoothing and Hilbert market geometry

The project now contains [a complete second experiment](market_geometry/notes/002-guerrero_smoothing/README.md) built from the classical PLS smoothness representation in [Trend-Estimation](https://github.com/heri-espino/Trend-Estimation). It tests two mean-difference models ($\mu=0$ and Guerrero plug-in $\mu=\mathrm{mean}(D_2y)$), each with **fixed $S$**, **GCV-selected $S$**, and **chronological forecast-CV-selected $S$**. These are different choices: Guerrero's normalized smoothness index is a coordinate, not a universal hyperparameter selector.

An **actual run with 12 independently simulated panels** is archived in EXP-002 with per-seed CSV, smoothing trajectories, scores, immutable SHA-256 manifest and PNG/SVG plots. Pearson correlation, B-spline Sobolev, and an **essential signed lag-one cross-correlation network** are compared on the same panels. In the deliberately constructed AR-memory-switch case, lagged correlation and nearly all functional representations obtain nearly perfect regime AUROC. Therefore **no incremental advantage over lagged correlation is demonstrated**. Results are exploratory; the asset series are synthetic. See exact numeric results in the archived experiment notes.

Run the extra experiment:

```powershell
python market_geometry/code/002-guerrero_smoothing/run.py --quick
python market_geometry/code/002-guerrero_smoothing/run.py --publish --seeds 12
```

`--publish` creates a new, non-overwriting run. The repository's `run_all.ps1` / `run_all.sh` now cover all six experiments.


## EXP-003: any finite-difference order, nested LOOCV and Gaussian ML for lambda

[EXP-003 mathematical notes and archived figures](market_geometry/notes/003-order_lambda_loocv/README.md) implement the **same PLS family at d=1,2,3,4** with four smoothing-parameter selectors: fixed controlled smoothness S, inner LOOCV, inner GCV, and Gaussian profile maximum likelihood (ML). The last rule is **our derived comparator under Guerrero's Gaussian variance-ratio model, not a rule verified as being proposed in Guerrero (2007)**. Guerrero (2007) explicitly develops a smoothness index.

Outer held-out observations are excluded when choosing lambda; exact two-deletion PRESS identities allow **nested LOOCV**. This is observation *reconstruction*, not time-series forecasting. A second diagnostic computes RMSE against known simulated latent trends without using the true trend to select lambda. A separate descriptive figure shows Guerrero's observed-difference mean plug-in for d=1..4; it is not given the same exact nested-LOO claims as the zero-mean models.

The supplied ZIP includes a fresh 4-independent-seed, 8-scenario-per-seed run, 16 configurations, actual PNG/SVG figures, CSV/JSON and SHA-256 files. Each method's result is in the note and archive. Results are exploratory and do not establish an ordering for real equities.

```powershell
python market_geometry/code/003-order_lambda_loocv/run.py --quick
python market_geometry/code/003-order_lambda_loocv/run.py --publish --seeds 4 --n 48
```

## EXP-004: matched cross-family estimator comparison (executed 6- and 30-seed runs)

[Full mathematical protocol and embedded figures](market_geometry/notes/004-cross_family_methods/README.md) compare **49 prespecified configurations** across 12 families plus a combined selector: Guerrero/Whittaker PLS d=1..4, linear and cubic regression B-splines, three P-spline coefficient-penalty orders, natural cubic smoothing splines, RBF kernel ridge regression, and polynomial regression.

Each method's hyperparameters are selected by **inner** LOOCV on the n-1 observations retained in each **outer** held-out fold. The *combined all-family selector* is itself nested; simply choosing the minimum of the per-family outer RMSEs would be invalid as a fresh unbiased estimate. All methods in this first comparison are fixed-quadratic smoothers supporting exactly verified one-/two-deletion identities. Re-estimating Guerrero's observed difference mean, local regression, wavelets, and nonlinear trend filtering needs separate refitting machinery.

The archive contains genuine PNG/SVG figures, per-seed-and-scenario CSV predictions and scores, JSON summaries, and SHA-256 manifests. The larger completed **30-seed** synthetic run has mean outer-LOOCV RMSE around **0.2500** for d=1 PLS, **0.2545** for RBF kernel ridge, and **0.2549** for a natural cubic smoothing spline. This is *not* evidence about real equities, future-return forecasts or financial regimes.

```powershell
python market_geometry/code/004-cross_family_methods/run.py --quick
python market_geometry/code/004-cross_family_methods/run.py --publish --seeds 30 --n 48
```

**Future candidate families:** LOESS, l1 trend filtering, state-space/Kalman, wavelets, Gaussian-process uncertainty models, and robust spline estimators. The next high-priority study evaluates market-regime networks with lag/spectral controls and chronological forecast validation, not just in-window interpolation.

## Scientific warning: equivalence with correlation

For standardized length-\(W\) sampled return vectors \(z_i,z_j\), with population variance 1:

\[
\frac1W\|z_i-z_j\|_2^2=2(1-\widehat\rho_{ij}).
\]

Therefore **plain Euclidean \(L^2\) on standardized sampled returns cannot outperform Pearson correlation by containing different pairwise information**. Spline smoothing and positive Sobolev derivative terms change the representation, but any apparent advantage requires comparison with lag-aware, frequency-aware, and nonlinear baselines. See [mathematical derivation](market_geometry/notes/mathematics/001-hilbert_metrics.md).

## Next research decisions

Read [Research direction](paper/RESEARCH_DIRECTION.md), [verified literature starting points](literature/references/README.md), [EXP-001 notes](market_geometry/notes/001-temporal_regime_pilot/README.md), and [NEXT_STEPS](market_geometry/history/context/NEXT_STEPS.md). The current pilot deliberately constructs a regime where *population contemporaneous correlation is stable*, **but ordinary lagged correlation can also detect the change**—an essential comparator for honest publication assessment.


## EXP-005: Legendre polynomial compression of trends in Hilbert space

[Canonical mathematical note, exact numerical results, and generated figures](market_geometry/notes/005-orthogonal_polynomial_geometry/README.md). Rather than selecting polynomial regression as a raw-data smoother (EXP-004), EXP-005 takes a PLS-smoothed trend, then projects it onto a finite-degree **orthonormal Legendre polynomial basis**. Its continuous polynomial $L^2$ distance is the Euclidean coefficient distance; Sobolev $H^1$ adds an explicitly computed derivative Gram matrix.

Eight independently seeded synthetic panels per AR(1)/AR(2) case were compared across Pearson, signed lag-one and lag-two correlation features, B-spline/Sobolev, raw and PLS-filtered Legendre degrees 3 and 7. Fixed predeclared smoothness/degree choices avoid label-driven tuning. The results demonstrate recoverable regime information but **no advantage over the relevant lagged-correlation comparator**. All AUROCs are retrospective discrimination, **not forward forecasting, real-stock performance or publication claims**.

```powershell
python market_geometry/code/005-orthogonal_polynomial_geometry/run.py --quick --seeds 1
python market_geometry/code/005-orthogonal_polynomial_geometry/run.py --publish --seeds 8
```

## GitHub after creating an empty repository

Use the exact name `FunctionalMarket-Geometry` and **do not initialize it with a README**, because this ZIP already contains one. In the extracted `FunctionalMarket-Geometry` folder:

```powershell
git init
git add .
git commit -m "Initialize reproducible functional market geometry research"
git branch -M main
git remote add origin https://github.com/heri-espino/FunctionalMarket-Geometry.git
git push -u origin main
```

No push or remote repository creation was performed as part of generating the ZIP.


## EXP-006: Functional PCA and dynamic weighted proximity graph spectra

[Canonical mathematical notes with actual figure embeds and quantitative evidence](market_geometry/notes/006-functional_pca_graphs/README.md).
For each synthetic return window, first-order PLS trends are compressed to degree-7 Legendre
coefficients and weighted by the H1 Gram matrix. **Cross-sectional functional PCA**
then produces rank-2/3/5 representations (not dynamic long-run-covariance FPCA).
A fixed-calibration-bandwidth k=3 symmetric nearest-neighbor graph is monitored using
weighted edges and normalized Laplacian eigenvalues. Direct Pearson, lag-1 and lag-2
cross-correlation scores, their graph analogues, full-coefficient H1 geometry,
and an AR(1) no-change control are included.

**First 8-seed results:** fPCA PLS rank-3 graph-edge AUROC was **0.998** on an AR(1)
change and **0.980** on an AR(2) change; matched lag-correlation obtained **1.000**
in both. The rank-3 graph *spectrum* was weaker (0.879/0.649). Null-control
threshold exceedances were large: a simple reference-deviation metric must **not**
be deployed as an alarm. This experiment does **not** predict future returns.

```powershell
python market_geometry/code/006-functional_pca_graphs/run.py --quick --seeds 1
python market_geometry/code/006-functional_pca_graphs/run.py --publish --seeds 8
```

The complete archived run is linked from EXP-006 notes and includes CSV, JSON, NPZ,
PNG/SVG and a checked SHA-256 manifest. There is no empirical-market edge
or originality claim.

## EXP-007: simulation targets for functional similarity

[Experiment note](market_geometry/notes/007-similarity_targets/README.md) · [Executable runner](market_geometry/code/007-similarity_targets/run.py). Four controlled scenarios test price-scale invariance, noisy common factors, two-day lagged relationships, and volatility scaling. Compare raw/rebased prices, returns, Pearson, PLS-Sobolev, Legendre-Sobolev and explicit lag correlation. The 100-seed archive is generated from the committed code; the nearest pair is not a future-return forecast. The appropriate lagged-correlation baseline is expected to detect the constructed lag. Run:

~~~powershell
python market_geometry/code/007-similarity_targets/run.py --quick
python market_geometry/code/007-similarity_targets/run.py --publish --seeds 100
~~~