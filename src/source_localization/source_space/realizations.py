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

Draws can cancel each other, and alignment is opt-in
---------------------------------------------------
Each draw's parcel row carries an arbitrary sign: a fixed-orientation source's
row points along its normal, and a draw that happens to sample a parcel with one
or two sources inherits their sign. Summing the draws with no alignment then
cancels. Measured on the deployed allen26 surface at 48 sources per draw (surface-
evoked X54): Auditory_L/R get about one source per draw, 37% / 34% of their draws
point opposite to the final row, and the summed row keeps only 0.28 / 0.30 of the
mean single-draw norm (Cerebellum 0.68). Aligning each parcel's draws to their
first principal direction before summing raised how often a simulated auditory
source wins its own parcel from 0.19 to 0.44 (L) and 0.27 to 0.57 (R), at a cost
to Retrosplenial (0.72 to 0.59-0.63) and Motor.

``align_draws=True`` does that alignment; it is off by default so every existing
result reproduces. ``draw_coherence`` in the report -- the norm of the averaged
row over the mean single-draw norm, 1.0 when every draw agrees -- is written
either way, so the cancellation is visible whether or not it is corrected.

This path is ROI-only by construction. There is no vertex-level output to give,
because no single grid is ever solved.
"""
from __future__ import annotations

import numpy as np

from .roi_combine import DEFAULT_COMBINE_MODE, combine_sources

__all__ = ["farthest_point_sample", "build_roi_operator", "align_draw_signs"]

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


def _parcel_rows(G_pool, idx, labels, lambda2, combine=DEFAULT_COMBINE_MODE):
    """One realization's ROI operator: {parcel -> row of length n_channels}.

    ``combine`` selects how a parcel's member rows are merged; see
    :mod:`source_localization.source_space.roi_combine`.
    """
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
    return {p: combine_sources(W[m, :], combine) for p, m in members.items()}


def align_draw_signs(rows):
    """Flip each draw's row to agree with the draws' first principal direction.

    ``rows`` is ``(n_draws, n_channels)``, one parcel's row from each draw that
    sampled it. Returns the rows with signs flipped so they no longer cancel when
    summed. The result is anchored to the plain sum, so the aligned row never
    comes out globally negated relative to the unaligned one -- the same
    convention :mod:`source_localization.source_space.roi_combine` uses.
    """
    rows = np.asarray(rows, dtype=float)
    if len(rows) < 2:
        return rows
    u = np.linalg.svd(rows, full_matrices=False)[2][0]
    s = np.sign(rows @ u)
    s[s == 0] = 1.0
    aligned = rows * s[:, None]
    if aligned.sum(axis=0) @ rows.sum(axis=0) < 0:
        aligned = -aligned
    return aligned


def build_roi_operator(G_pool, positions_mm, labels, *, n_sources=DEFAULT_N_SOURCES,
                       k=DEFAULT_K, seed=DEFAULT_SEED, lambda2=1.0 / 9.0,
                       collinear_threshold=0.99,
                       combine=DEFAULT_COMBINE_MODE,
                       align_draws=False, strata=None):
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
    combine : how each realization's member rows are merged into a parcel row
        ('mean', 'mean_flip', 'pca_flip'); see
        :mod:`source_localization.source_space.roi_combine`.
    align_draws : flip each parcel's per-draw rows to a common sign before
        summing (:func:`align_draw_signs`). Off by default; see the module
        docstring for why draws cancel without it.
    strata : sequence of (pool_indices, n_take), optional
        Draw each stratum separately -- ``n_take`` farthest-point sources from
        ``pool_indices``, with the same seed per draw -- and solve them together.
        ``n_sources`` is then ignored. A hybrid source space uses this so its
        surface part draws exactly what a surface-only pool of the same
        positions would draw, and its volume part is added on top, rather than
        the two competing for one budget. None (default) draws from the whole
        pool at once, as before.

    Returns
    -------
    operator : ndarray, (n_parcels, n_channels)   -- apply to sensor data
    parcels : list of str                          -- row order of `operator`
    report : dict with per-parcel `gain`, `n_realizations`, `coverage`,
        `collinear_with`, `draw_coherence` and `draws_aligned`
    """
    G_pool = np.asarray(G_pool, dtype=float)
    labels = list(labels)
    if len(labels) != G_pool.shape[1]:
        raise ValueError("labels must have one entry per pool source")

    own: dict[str, list[int]] = {}
    for i, name in enumerate(labels):
        if name:
            own.setdefault(name, []).append(i)

    draws: dict[str, list[np.ndarray]] = {}
    single_snr: dict[str, list[float]] = {}
    positions_mm = np.asarray(positions_mm, dtype=float)
    if strata is not None:
        strata = [(np.asarray(ix, dtype=int), int(n)) for ix, n in strata]
        flat = np.concatenate([ix for ix, _ in strata])
        if len(np.unique(flat)) != len(flat):
            raise ValueError("strata overlap: a pool source is in more than one")
    for j in range(k):
        if strata is None:
            idx = farthest_point_sample(positions_mm, n_sources, seed + j)
        else:
            idx = np.sort(np.concatenate([
                ix[farthest_point_sample(positions_mm[ix], n, seed + j)] for ix, n in strata]))
        for p, row in _parcel_rows(G_pool, idx, labels, lambda2, combine).items():
            draws.setdefault(p, []).append(row)
            c = row @ G_pool
            single_snr.setdefault(p, []).append(
                float(np.linalg.norm(c[own[p]]) / np.linalg.norm(row)))

    counts = {p: len(r) for p, r in draws.items()}
    # How much the draws cancel: norm of the averaged row over the mean single-draw
    # norm, before any alignment. 1.0 when every draw agrees.
    coherence = {p: float(np.linalg.norm(np.sum(r, axis=0) / len(r))
                          / np.mean(np.linalg.norm(r, axis=1)))
                 for p, r in draws.items()}
    total = {p: (align_draw_signs(r) if align_draws else np.asarray(r)).sum(axis=0)
             for p, r in draws.items()}

    parcels = sorted(total)
    # Divide by k, not by counts[p].
    #
    # counts[p] is the number of draws *that parcel* was sampled in, and it is
    # not the same for every parcel: a small parcel can enter only a handful of
    # draws. Normalising each parcel by its own count leaves a parcel drawn once
    # as an unaveraged single-draw row, while a parcel drawn k times becomes an
    # average whose norm is reduced by cancellation across draws. Row scales are
    # then not comparable between parcels, and the rarest parcel carries the
    # largest row.
    #
    # Measured on the deployed allen32 chirp surface run (n=160, k=100):
    # Hippocampus_Ant_R appeared in 1 of 100 draws against 83-100 for every
    # other parcel, and its row norm was 2.6x the next largest -- enough to win
    # the argmax for nearly every simulated source.
    #
    # Dividing by k is the estimator the docstring describes: a mean over
    # placements, in which a placement that never sampled the parcel contributes
    # nothing to it. Parcels present in every draw are unchanged; rare parcels
    # shrink toward zero, so row scale now reflects how reliably the parcel was
    # sampled at all. `coverage` in the report makes that explicit.
    operator = np.vstack([total[p] / k for p in parcels])

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
            # Fraction of draws that sampled this parcel at all. Below ~1.0 the
            # parcel's row is an average over fewer placements than the others,
            # and its amplitude is scaled down to match; a parcel at 0.01 has
            # essentially no estimate and should not be read as a quiet source.
            "coverage": counts[p] / k,
            "max_collinearity": best,
            "collinear_with": partner if best >= collinear_threshold else None,
            # Below ~0.5 the draws largely cancel (surface-evoked X54); the
            # operator row is then much weaker than any one draw's.
            "draw_coherence": coherence[p],
            "draws_aligned": bool(align_draws),
        }

    sparse = [(p, report[p]["coverage"]) for p in parcels
              if report[p]["coverage"] < 0.5]
    if sparse:
        print(f"    ⚠️  {len(sparse)} parcel(s) sampled in under half the "
              f"draws; their rows are scaled down accordingly and should not "
              f"be compared as amplitudes:")
        for p, cov in sorted(sparse, key=lambda t: t[1]):
            print(f"          {p} ({cov:.0%} of draws)")

    return operator, parcels, report
