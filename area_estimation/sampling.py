"""Stratified random sampling without replacement from a population summarised by count tables.

The population is described by counts[h, c]: the number of pixels of design stratum h that
belong to category c (any cross-classification of the pixels, e.g. the pixel codes of
area_estimation.monte_carlo).  A sample of n_h pixels drawn without replacement from stratum h
is then fully described by its own count table, which follows a multivariate hypergeometric
distribution.
"""
from __future__ import annotations

import numpy as np


def draw_count_tables(counts, n_h, n_rep, rng):
    """n_rep stratified samples as count tables of shape (n_rep, H, C)."""
    counts = np.asarray(counts).astype(np.int64)
    return np.stack([rng.multivariate_hypergeometric(counts[s], int(n_h[s]), size=n_rep)
                     for s in range(counts.shape[0])], axis=1)


def draw_replicates(counts, n_h, seed, n_rep, n_base=None):
    """Count tables of the n_rep replicate samples of one design.

    Replicates 1 ... n_base are drawn with numpy.random.default_rng(seed) and replicates
    n_base + 1 ... n_rep with the independent stream default_rng([seed, 3]), so that the first
    n_base replicates do not depend on n_rep.  n_base = None draws all replicates with
    default_rng(seed).
    """
    base = n_rep if n_base is None else min(int(n_base), n_rep)
    tables = draw_count_tables(counts, n_h, base, np.random.default_rng(seed))
    if n_rep > base:
        extra = draw_count_tables(counts, n_h, n_rep - base, np.random.default_rng([seed, 3]))
        tables = np.concatenate([tables, extra], axis=0)
    return tables


def sort_population(strata, categories, n_categories):
    """Pixels of each stratum sorted by category, for drawing the pixels of a count table.

    Returns (members, offsets): members[h] holds the indices of the pixels of stratum h sorted
    by category; offsets[h][c] : offsets[h][c + 1] is the slice of category c.
    """
    members, offsets = [], []
    for s in np.unique(strata):
        idx = np.flatnonzero(strata == s)
        idx = idx[np.argsort(categories[idx], kind="stable")]
        members.append(idx)
        offsets.append(np.searchsorted(categories[idx], np.arange(n_categories + 1)))
    return members, offsets


def draw_sample_units(table, members, offsets, rng):
    """Pixels of one replicate: for every stratum and category, table[h, c] pixels drawn without
    replacement among the pixels of that stratum and category.  Returns (pixel index, stratum)."""
    pos, strata = [], []
    for s in range(table.shape[0]):
        for c in np.flatnonzero(table[s]):
            pool = members[s][offsets[s][c]:offsets[s][c + 1]]
            n = int(table[s, c])
            pos.append(pool[rng.choice(pool.size, size=n, replace=False)])
            strata.append(np.full(n, s, dtype=int))
    return np.concatenate(pos), np.concatenate(strata)
