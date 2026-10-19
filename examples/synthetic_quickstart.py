"""Minimal research-API example: compute two distance matrices on artificial data."""

from market_geometry import (
    SimulationSettings,
    SplineSettings,
    correlation_distance,
    simulate_market,
    spline_metric,
    standardize_windows,
)

returns, _, _ = simulate_market(SimulationSettings(n_steps=240, boundary=120))
z = standardize_windows(returns[:, :80])
corr = correlation_distance(z)
hilbert = spline_metric(z, SplineSettings(n_basis=36, derivative_weight=0.75))
print("Sample Pearson-chord asset 0 ↔ 4:", corr[0, 4])
print("Sample spline-H1 asset 0 ↔ 4:", hilbert[0, 4])
