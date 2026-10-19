"""Deterministically seedable financial-panel experiments with known regimes."""

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class SimulationSettings:
    """Synthetic stationary-variance panel with a temporal-memory regime shift.

    The common group-factor AR coefficient changes at the regime boundary,
    while each group's contemporaneous covariance is fixed in expectation.
    These are simulated standardized *returns*, not price observations.
    """

    seed: int = 20261018
    n_assets: int = 12
    n_groups: int = 3
    n_steps: int = 720
    boundary: int = 360
    group_strength: float = 0.70
    phi_before: float = 0.82
    phi_after: float = -0.65


def simulate_market(settings: SimulationSettings) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Simulate zero-mean group-correlated returns with a change in AR memory.

    Returns
    -------
    returns : ndarray of shape (n_assets, n_steps)
        Artificial returns, without market or real-stock observations.
    regimes : ndarray of shape (n_steps,)
        Ground-truth regime indicator, 0 before change and 1 after.
    groups : ndarray of shape (n_assets,)
        Synthetic asset-to-group labels (for diagnostics only).
    """
    c = settings
    if c.n_assets % c.n_groups or c.n_steps <= c.boundary or c.boundary < 50:
        raise ValueError("Use evenly divisible asset groups and a valid regime boundary")
    if not (0 < c.group_strength < 1 and abs(c.phi_before) < 1 and abs(c.phi_after) < 1):
        raise ValueError("Strength must be in (0,1) and AR coefficients must be stable")
    rng = np.random.default_rng(c.seed)
    factors = np.zeros((c.n_groups, c.n_steps))
    for start, stop, phi in (
        (0, c.boundary, c.phi_before),
        (c.boundary, c.n_steps, c.phi_after),
    ):
        # Restart at a stationary N(0,1) draw, avoiding initialization drift.
        factors[:, start] = rng.standard_normal(c.n_groups)
        for t in range(start + 1, stop):
            factors[:, t] = phi * factors[:, t - 1] + np.sqrt(1 - phi**2) * rng.standard_normal(c.n_groups)
    groups = np.repeat(np.arange(c.n_groups), c.n_assets // c.n_groups)
    idiosyncratic = rng.standard_normal((c.n_assets, c.n_steps))
    returns = np.sqrt(c.group_strength) * factors[groups] + np.sqrt(1 - c.group_strength) * idiosyncratic
    regimes = np.zeros(c.n_steps, dtype=int)
    regimes[c.boundary :] = 1
    return returns, regimes, groups
