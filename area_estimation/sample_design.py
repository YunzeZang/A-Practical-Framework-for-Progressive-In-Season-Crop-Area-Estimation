"""Sample size and allocation for a stratified design with a target confidence-interval half-width."""
from __future__ import annotations

import math

import numpy as np

ALLOCATIONS = ("neyman", "proportional", "equal")


def largest_remainder(n: int, shares) -> np.ndarray:
    """Integer n_h summing to n, as close as possible to n * shares (largest-remainder rounding)."""
    raw = n * np.asarray(shares, dtype=float)
    base = np.floor(raw).astype(int)
    order = np.argsort(-(raw - base))
    base[order[:n - base.sum()]] += 1
    return base


def allocation_shares(W, S, allocation="neyman"):
    """Share of the sample given to each stratum.

    neyman       n_h proportional to W_h S_h
    proportional n_h proportional to W_h
    equal        n_h = n / H
    """
    W, S = np.asarray(W, float), np.asarray(S, float)
    a = {"neyman": W * S, "proportional": W.copy(), "equal": np.ones_like(W)}[allocation]
    return a / a.sum()


def required_sample_size(W, S, z, d, allocation="neyman"):
    """Smallest n whose allocation (largest-remainder rounding) gives z * SE <= d.

    W : stratum weights, S : anticipated within-stratum standard deviations,
    z : normal quantile of the confidence level, d : target half-width of the proportion.
    Returns dict(n, n_h).
    """
    W, S = np.asarray(W, float), np.asarray(S, float)
    a = allocation_shares(W, S, allocation)
    n = math.ceil(float(np.sum(W ** 2 * S ** 2 / a)) / (d / z) ** 2)
    while True:
        n_h = largest_remainder(n, a)
        if z * math.sqrt(np.sum(W ** 2 * S ** 2 / n_h)) <= d + 1e-12:
            return dict(n=n, n_h=n_h)
        n += 1
