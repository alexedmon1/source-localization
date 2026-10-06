"""The parcel-subspace ROI operator: every parcel as its leadfield patterns, solved once.

What Monte Carlo sampling was approximating
-------------------------------------------
A parcel is a continuous sheet (or volume) of cortex, and the deployed grid is one arbitrary
placement of sources in it. Monte Carlo sampling (:mod:`.realizations`) integrates over
placement: it solves K sparse grids and averages their ROI operators. What it is estimating, in
effect, is the set of electrode patterns a parcel can produce.

That set can be taken directly. Over all of a parcel's pool sources, the leadfield columns
``G_p`` span its patterns, and their singular value decomposition orders them by how much of the
parcel's activity each carries. The top ``r`` patterns, scaled to a typical source's gain
(``S / sqrt(n_p)``), represent the parcel. Every parcel's patterns are stacked into one
leadfield, and a single minimum-norm inverse is solved over all of them at once.

Why it localizes better (surface-evoked troubleshooting log, sections 21-25)
-----------------------------------------------------------------------------
On a shared simulated truth, scored by MS1's peak displacement (expected mm between the true
parcel and the credited one) over 15 cortical parcels: the floor (exact Bayesian posterior) was
0.98 mm, ROI-based 1.85, the deployed Monte Carlo operator (48 x 100, sLORETA) 3.05, and this
operator with r = 3 and MNE 1.95 (level with ROI-based; better than it, 1.62 against 1.95, for
dipoles on the sheet along the normal). The draws each solve a different 48-source problem and
their average blurs them; here the parcels compete for the data in one solve, so a pattern two
parcels share is split between them instead of being added to both. The number of draws, their
seed and their sign alignment no longer exist.

Output per parcel
-----------------
``r`` component series per parcel, one per pattern. Which single series an evoked analysis
should use has **not** been tested. Provisionally the step emits the first (dominant) pattern's
output as the signed series and the root-sum-square over the ``r`` outputs as the magnitude, and
saves all components so the choice can be revisited. The dominant pattern's sign is anchored to
the parcel's summed leadfield, so a positive output means current along the parcel's mean
pattern, and the sign is stable across runs.

Methods: ``MNE`` (the measured best) and ``sLORETA``. Regularisation as everywhere in this
package: ``lambda2 * trace(GG') / n_channels``.
"""
from __future__ import annotations

import numpy as np

__all__ = ["DEFAULT_N_PATTERNS", "DEFAULT_METHOD", "SUBSPACE_METHODS", "parcel_patterns",
           "build_subspace_operator"]

DEFAULT_N_PATTERNS = 3
DEFAULT_METHOD = "MNE"
SUBSPACE_METHODS = ("MNE", "sLORETA")


def parcel_patterns(G_parcel: np.ndarray, n_patterns: int):
    """Top leadfield patterns of one parcel, scaled to a typical source's gain.

    Returns ``(patterns (n_channels, k), energy_captured, singular_values)``, with
    ``k = min(n_patterns, rank)``. ``energy_captured`` is the share of the parcel's total
    leadfield energy (sum of squared singular values) the ``k`` patterns carry.
    """
    G_parcel = np.asarray(G_parcel, dtype=float)
    U, S, _ = np.linalg.svd(G_parcel, full_matrices=False)
    k = int(min(n_patterns, int((S > S[0] * 1e-12).sum()) if len(S) else 0))
    if k == 0:
        raise ValueError("parcel has no leadfield energy")
    U = U[:, :k].copy()
    # Anchor the dominant pattern to the parcel's summed leadfield so its sign means
    # something and does not depend on the SVD routine.
    if U[:, 0] @ G_parcel.sum(axis=1) < 0:
        U[:, 0] = -U[:, 0]
    patterns = U * (S[:k] / np.sqrt(G_parcel.shape[1]))
    energy = float((S[:k] ** 2).sum() / (S ** 2).sum())
    return patterns, energy, S


def _solve(Gs, lambda2, method):
    n_ch = Gs.shape[0]
    GGT = Gs @ Gs.T
    W = Gs.T @ np.linalg.inv(GGT + lambda2 * np.trace(GGT) / n_ch * np.eye(n_ch))
    if method == "MNE":
        return W
    if method == "sLORETA":
        return W / np.sqrt(np.einsum("ij,ji->i", W, Gs))[:, None]
    raise ValueError(f"parcel_subspace method must be one of {SUBSPACE_METHODS}, got {method!r}")


def build_subspace_operator(G_pool, labels, *, n_patterns=DEFAULT_N_PATTERNS,
                            lambda2=1.0 / 9.0, method=DEFAULT_METHOD):
    """One inverse over every parcel's top leadfield patterns.

    Parameters
    ----------
    G_pool : ndarray, (n_channels, n_pool)
        One leadfield column per pool source (fixed orientation, or the dominant
        direction of a free source).
    labels : sequence of str or None, length n_pool
        Parcel of each pool source; None or '' for unlabelled, which never enters.
    n_patterns : int
        Patterns per parcel, ``r``. Fewer where a parcel's leadfield has lower rank.
    lambda2 : regularisation, matching the deployed inverse.
    method : 'MNE' or 'sLORETA'.

    Returns
    -------
    operator : ndarray, (n_components, n_channels)   -- every parcel's components, stacked
    parcels : list of str                             -- sorted parcel names
    components : dict  parcel -> list of operator row indices, dominant pattern first
    report : dict  parcel -> n_pool, n_patterns, energy_captured, singular_values (top 5)
    """
    G_pool = np.asarray(G_pool, dtype=float)
    labels = list(labels)
    if len(labels) != G_pool.shape[1]:
        raise ValueError("labels must have one entry per pool source")
    if method not in SUBSPACE_METHODS:
        raise ValueError(f"parcel_subspace method must be one of {SUBSPACE_METHODS}, got {method!r}")
    if int(n_patterns) < 1:
        raise ValueError("n_patterns must be at least 1")

    own: dict[str, list[int]] = {}
    for i, name in enumerate(labels):
        if name:
            own.setdefault(name, []).append(i)
    parcels = sorted(own)
    cols, components, report = [], {}, {}
    for p in parcels:
        pat, energy, S = parcel_patterns(G_pool[:, own[p]], int(n_patterns))
        start = sum(c.shape[1] for c in cols)
        components[p] = list(range(start, start + pat.shape[1]))
        cols.append(pat)
        report[p] = {"n_pool": len(own[p]), "n_patterns": int(pat.shape[1]),
                     "energy_captured": energy,
                     "singular_values": [float(s) for s in S[:5]]}
    operator = _solve(np.hstack(cols), float(lambda2), method)
    return operator, parcels, components, report
