# 2026-10-18 — Bridging Trend-Estimation and FunctionalMarket-Geometry

The user requested a comparison with Guerrero's PLS smoothness methods from `heri-espino/Trend-Estimation`.

- Checked the project's `README.md`, `THEORETICAL_CONTRIBUTIONS.md`, and `notes/conceptos/06-Media-de-diferencias-Guerrero.md` / `07-Perfilamiento-de-mu-en-CV.md` through connected GitHub.
- Kept the classical mean-of-differences estimator separate from the normalized smoothness index and new proposed forecast-directed selector.
- Added standalone PLS spectral smoother in `src/market_geometry/guerrero.py`, with exact S=0/S=1 endpoint treatment and plug-in EDF-corrected GCV.
- Added EXP-002 comparing Pearson, lag-one signed correlation, spline-Sobolev and six PLS configurations under a matched synthetic AR-memory switch.
- Chronological inner forecast CV uses raw returns to avoid look-ahead from whole-window z-scoring; evaluation labels are still synthetic/oracle-defined.
- All figures/results must be produced by actual `--publish` execution, never pasted as unverified claims.
- No changes were made to Trend-Estimation or any remote repository. The project remains a local generated ZIP.

Failure modes/unfinished work: regime-detection metric tuned via forecast MSE may fail; GCV assumes conditions violated by serial dependence; positive PLS smoothness may obscure AR memory; no real data, alternative DGP validation or formal inferential significance. Need negative controls and independent external data.

## Completed matched pilot, not confirmatory

Actually published a 12-independent-seed (seeds 20261019 through 20261030) EXP-002 archive. Mean AUROC: Pearson 0.403; signed lag-one cross-correlation 1.000; spline-Sobolev 1.000; Guerrero fixed-S PLS 1.000; Guerrero GCV PLS approximately 1.000; Guerrero forecast-CV PLS 0.997. The lag baseline perfectly discriminates in all 12 simulations. There is no incremental information demonstrated by the functional model under this simulated AR change, and meaningful methodological discrimination requires harder synthetic mechanisms. Data and three plots were archived in same-ID result and figure directories; no user's private stock data was used.
