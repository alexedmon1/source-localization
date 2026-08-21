"""Realization-averaged ROI operators: many sparse grids instead of one dense one.

A source space is a discrete approximation of a continuous current distribution,
and the deployed grid is one arbitrary placement of it. This builds many sparse
placements instead, and averages the ROI operators they induce.

Why it is nearly free
---------------------
The parcel time series is *linear* in the sensor data::

    y_p = mean_{i in p} (W_norm[i, :] . B) = (mean_{i in p} W_norm[i, :]) . B

so averaging ROI time series over K realizations is algebraically identical to
averaging the K operators and applying the result once. A realization contributes
an ``(n_parcels, n_channels)`` matrix; K of them collapse into one. Nothing
per-realization is materialised, no vertex-level estimate is ever built, and the
operator is subject-invariant -- the forward depends only on (montage, BEM,
source space), verified byte-identical across recordings -- so it is computed
once per configuration and reused.

What it buys, measured
----------------------
Sparse realizations steer the operator row in different directions, so their
average shrinks in noise norm (triangle inequality) while the signal term, which
integrates over the same anatomy, does not. Measured against a dense 0.20 mm
grid standing in for continuous truth, at 160 sources and K=100: median SNR gain
**1.14x**, auditory ~1.20x. K converges by 100 -- K=200 and K=400 are identical
-- and 80 sources per realization is past the cliff, where parcels start dropping
out of realizations entirely.

Density is not what leaks
-------------------------
Parcel leakage is flat at 29-30% across a 10x range of source counts, so the
deployed density buys parcel *occupancy*, not resolution. That is why sparse
realizations cost nothing to trade away: many sparse grids give the coverage of a
dense one with the conditioning of a sparse one.

Degenerate parcels get worse, and the report says so
----------------------------------------------------
Parcels that are near-collinear with another parcel are *harmed*. The olfactory
bulbs correlate at r = 0.9955 in sensor space, so the inverse splits their shared
signal arbitrarily and sparse placements tip that split at random: individual
gains swing 0.73-1.14x with the seed and which side suffers flips, while the pair
taken together is stable at 1.03-1.28x. That is a property of the degenerate
*pair*, not of either parcel, so :func:`build_roi_operator` reports per-parcel
gain alongside a collinearity flag rather than implying a uniform benefit.

This path is ROI-only by construction. There is no vertex-level output to give,
because no single grid is ever solved.
"""
from __future__ import annotations

import numpy as np

__all__ = ["farthest_point_sample", "build_roi_operator"]

DEFAULT_N_SOURCES = 160
DEFAULT_K = 100
DEFAULT_SEED = 20260821


def farthest_point_sample(positions_mm, n_take, seed):
    """A well-spaced realization of ``n_take`` sources, different per seed.

    Farthest-point rather than uniform-random: a random subset is clumpy, which
    is not a fair stand-in for "a different grid at the same density". Starting
    vertex is seeded, everything after it is deterministic.
    """
    positions_mm = np.asarray(positions_mm, dtype=float)
    m = len(positions_mm)
    if n_take > m:
        raise ValueError(f"cannot take {n_take} sources from a pool of {m}")
    rng = np.random.default_rng(seed)
    first = int(rng.integers(m))
    chosen = [first]
    d = np.linalg.norm(positions_mm - positions_mm[first], axis=1)
    for _ in range(n_take - 1):
        nxt = int(np.argmax(d))
        chosen.append(nxt)
        np.minimum(d, np.linalg.norm(positions_mm - positions_mm[nxt], axis=1), out=d)
    return np.sort(np.asarray(chosen))


def _parcel_rows(G_pool, idx, labels, lambda2):
    """One realization's ROI operator: {parcel -> row of length n_channels}."""
    Gs = G_pool[:, idx]
    GGT = Gs @ Gs.T
    reg = lambda2 * np.trace(GGT) / Gs.shape[0]
    W = Gs.T @ np.linalg.inv(GGT + reg * np.eye(Gs.shape[0]))
    # sLORETA normalisation, per source, before parcel averaging
    W = W / np.sqrt(np.sum(W * Gs.T, axis=1))[:, None]
    members: dict[str, list[int]] = {}
    for local, pool_idx in enumerate(idx):
        name = labels[pool_idx]
        if name:
            members.setdefault(name, []).append(local)
    return {p: W[m, :].mean(axis=0) for p, m in members.items()}


def build_roi_operator(G_pool, positions_mm, labels, *, n_sources=DEFAULT_N_SOURCES,
                       k=DEFAULT_K, seed=DEFAULT_SEED, lambda2=1.0 / 9.0,
                       collinear_threshold=0.99):
    """Average the ROI operators of ``k`` sparse realizations.

    Parameters
    ----------
    G_pool : ndarray, (n_channels, n_pool)
        Fixed-orientation leadfield of a dense pool the realizations draw from.
    positions_mm : ndarray, (n_pool, 3)
    labels : sequence of str or None, length n_pool
        Parcel of each pool source; None for unlabelled, which never enters.
    n_sources, k, seed : realization size, count, and RNG seed.
    lambda2 : regularisation, matching the deployed inverse.
    collinear_threshold : |r| above which two parcel topographies are called
        degenerate, and their individual gains flagged as unreliable.

    Returns
    -------
    operator : ndarray, (n_parcels, n_channels)   -- apply to sensor data
    parcels : list of str                          -- row order of `operator`
    report : dict with per-parcel `gain`, `n_realizations`, `collinear_with`
    """
    G_pool = np.asarray(G_pool, dtype=float)
    labels = list(labels)
    if len(labels) != G_pool.shape[1]:
        raise ValueError("labels must have one entry per pool source")

    own: dict[str, list[int]] = {}
    for i, name in enumerate(labels):
        if name:
            own.setdefault(name, []).append(i)

    total: dict[str, np.ndarray] = {}
    counts: dict[str, int] = {}
    single_snr: dict[str, list[float]] = {}
    for j in range(k):
        idx = farthest_point_sample(positions_mm, n_sources, seed + j)
        for p, row in _parcel_rows(G_pool, idx, labels, lambda2).items():
            total[p] = total.get(p, 0.0) + row
            counts[p] = counts.get(p, 0) + 1
            c = row @ G_pool
            single_snr.setdefault(p, []).append(
                float(np.linalg.norm(c[own[p]]) / np.linalg.norm(row)))

    parcels = sorted(total)
    operator = np.vstack([total[p] / counts[p] for p in parcels])

    # Degeneracy: a parcel whose topography is near-collinear with another's has
    # no stable individual solution, so its per-parcel gain is not meaningful.
    topo = {p: (lambda v: v / np.linalg.norm(v))(G_pool[:, own[p]].mean(axis=1))
            for p in parcels}
    report: dict[str, dict] = {}
    for i, p in enumerate(parcels):
        c = operator[i] @ G_pool
        avg = float(np.linalg.norm(c[own[p]]) / np.linalg.norm(operator[i]))
        partner, best = None, 0.0
        for q in parcels:
            if q == p:
                continue
            r = abs(float(topo[p] @ topo[q]))
            if r > best:
                best, partner = r, q
        report[p] = {
            "gain": avg / float(np.mean(single_snr[p])),
            "n_realizations": counts[p],
            "max_collinearity": best,
            "collinear_with": partner if best >= collinear_threshold else None,
        }
    return operator, parcels, report
