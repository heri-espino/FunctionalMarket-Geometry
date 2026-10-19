# 2026-10-18 — Initialization

## Question

Explore whether time-varying Hilbert-space geometry of financial return curves provides incremental information about relationship networks and market regimes.

## Decisions

- Name `FunctionalMarket-Geometry`; short research folder `market_geometry/`.
- Preserve the structure and 4-fold numbered-experiment protocol of `ProximityGraphs-Isomap`; keep source package independent of presentation notes.
- Use aligned **returns** rather than absolute price levels; fit penalized cubic splines and compare correlation-chord distances with derivative-sensitive H1 distances.
- Avoid asserting novelty: sampled normalized L2 distances are Pearson-equivalent, and existing research covers financial functional time-series and time-varying correlation networks.
- EXP-001 deliberately controls the *population contemporaneous covariance* while changing a stationary factor's AR coefficient. Competing lagged estimators can detect this; not evidence that H1 is unique.

## Execution record

A runnable synthetic pilot was built. Its actual computed results and figures, if published, appear only in the linked [EXP-001 note](../../../notes/001-temporal_regime_pilot/README.md) and corresponding immutable directories. The journal alone is not the authority for numerical outputs.

## Follow-up

Confirm integrity of initial ZIP, execute repeated independent seeds, add lagged/spectral baselines and later assess actual finance data without look-ahead.
