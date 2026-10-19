"""Functional Market Geometry: exploratory Hilbert-space market representations."""

from .metrics import SplineSettings, correlation_distance, spline_metric, standardize_windows
from .rolling import RollingSettings, reference_deviation, rolling_distances
from .simulation import SimulationSettings, simulate_market

__all__ = [
    "RollingSettings",
    "SimulationSettings",
    "SplineSettings",
    "correlation_distance",
    "reference_deviation",
    "rolling_distances",
    "simulate_market",
    "spline_metric",
    "standardize_windows",
]
