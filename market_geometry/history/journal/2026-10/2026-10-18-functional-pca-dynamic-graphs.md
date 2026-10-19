# 2026-10-18 — EXP-006 functional PCA and graph-spectral pilot

## Why

EXP-005 showed Hilbert polynomial compression can detect intentionally designed
memory shifts, but lagged correlations did at least as well. The user requested
fPCA followed by dynamic graph spectra, maintaining the same clean numbered
experiment structure.

## Changes made

- Added reusable cross-sectional PCA of finite Legendre Hilbert coordinates,
  exact Cholesky Gram realization, stable symmetrized weighted kNN and normalized
  Laplacian spectrum implementations in `src/market_geometry/functional_pca_graph.py`.
- Added a documented synthetic study with an AR(1) change, AR(2) change,
  and AR(1) no-coefficient-change control, calibration-only kernel bandwidth,
  Pearson/lag-1/lag-2 direct and graph baselines, complete actual figure archive.
- Recorded seed-level metrics and a parameter-sensitive reproducibility manifest.
- Added unit tests for metric preservation, contraction, graph symmetry,
  zero eigenvalues, time-locality, calibration-only reference and quick research run.

## Evidence and negative results

See `market_geometry/notes/006-functional_pca_graphs/README.md` and its actual
archived eight-seed run. AR1/AR2 switch average AUROC: FPCA-PLS rank3 graph
edge score 0.998/0.980 vs correctly matched direct lag correlation 1.000/1.000.
The FPCA graph spectrum 0.879/0.649 is worse than edge-level geometry.
The no-change control frequently exceeds naive 95%-calibration quantiles;
these are NOT statistically calibrated false alarms or independent windows.

## Decision

Retain EXP-006 as exploratory work and preserve prior experiment archives.
Do not claim novelty or a method advantage. Next work: chronological alert
calibration, tough nulls and real returns; avoid outcome-informed tuning.

## Reproduction

```
python market_geometry/code/006-functional_pca_graphs/run.py --publish --seeds 8
python -m pytest -q
python scripts/validate_research.py
```

`--publish` creates an additional immutable run; it does not overwrite evidence.
