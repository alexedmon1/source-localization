"""How the sources inside one ROI are combined into a single ROI estimate.

A parcel estimate is formed from several sources, and the obvious choice — average them —
is a *coherent* sum. Sources whose orientations differ then partially cancel, and the ROI
estimate keeps less signal than any one member carries. The standard remedy in the
MEG/EEG parcellation literature is to align the members before combining; MNE offers the
same choice as ``mean_flip`` and ``pca_flip`` on label extraction.

Three modes, selected by ``roi_extraction.combine`` in the study config:

``mean``
    Plain average. The historical behaviour and the default, so an existing config is
    unaffected by this module's existence.
``mean_flip``
    Flip each member to agree with the members' dominant direction, then average.
    Cancellation from sign disagreement is removed; everything else is unchanged.
``pca_flip``
    Take the members' dominant direction itself, rescaled to the plain average's norm.
    The most signal a linear combination can retain, at the cost of no longer being an
    average of the members.

Both flip modes are *sign-anchored to the plain average*, so the result never comes out
globally negated relative to ``mean``. Without that anchor the sign is arbitrary per call,
which matters when results are summed across Monte Carlo draws — the anchor is what keeps
those draws from cancelling each other.

The same three modes apply whether the members are time courses (fixed sampling, where
ROI extraction runs on the source estimate) or operator rows (Monte Carlo sampling, where
the parcel row is built before any data is touched). The algebra is identical: combine a
set of vectors into one.

One consequence is worth stating, because it makes ``pca_flip`` mean slightly different
things on the two paths. The dominant direction is taken from the members' own Gram matrix.
Under fixed sampling the members are time courses, so that Gram matrix *is* the data
covariance and the result is the best linear combination for that recording — the same
quantity MNE's ``pca_flip`` computes. Under Monte Carlo sampling the members are operator
rows, and the operator is deliberately built once and reused for every subject; the Gram
matrix then describes the operator's own geometry, not any recording. That keeps Monte
Carlo cheap, at the cost of not reaching the per-recording optimum: measured on a
30-channel montage, ``pca_flip`` retained ~0.53 of the available power where a
covariance-aware combination could have reached ~0.70 (plain averaging retained ~0.20).

A per-subject variant that rebuilds the parcel rows against each recording's covariance
would close that gap, at the cost of an operator per subject. Not implemented; worth
testing if the extra recovery is ever wanted.
"""
from __future__ import annotations

import numpy as np

__all__ = ["VALID_COMBINE_MODES", "DEFAULT_COMBINE_MODE", "combine_sources",
           "resolve_combine_mode"]

VALID_COMBINE_MODES = ("mean", "mean_flip", "pca_flip")
DEFAULT_COMBINE_MODE = "mean"


def resolve_combine_mode(config) -> str:
    """Read ``roi_extraction.combine`` from a study config, validating it."""
    mode = ((config or {}).get("roi_extraction") or {}).get(
        "combine", DEFAULT_COMBINE_MODE)
    if mode not in VALID_COMBINE_MODES:
        raise ValueError(
            f"Unknown roi_extraction.combine: {mode!r}. "
            f"Valid: {', '.join(VALID_COMBINE_MODES)}")
    return mode


def _dominant_direction(X):
    """Unit vector over members spanning their dominant common direction.

    Taken from the members' Gram matrix, which is (n_members x n_members) — so this stays
    cheap when the members are long time courses.
    """
    gram = X @ X.T
    w, v = np.linalg.eigh(gram)
    return v[:, int(np.argmax(w))]


def combine_sources(X, mode: str = DEFAULT_COMBINE_MODE):
    """Combine an ROI's member vectors into one.

    Parameters
    ----------
    X : ndarray, shape (n_members, n_features)
        One row per source. Features are time points, or channels for an operator row.
    mode : {'mean', 'mean_flip', 'pca_flip'}

    Returns
    -------
    ndarray, shape (n_features,)
    """
    X = np.asarray(X, dtype=float)
    if X.ndim != 2:
        raise ValueError(f"expected a 2-D (n_members, n_features) array, got {X.shape}")
    plain = X.mean(axis=0)
    if mode == "mean" or X.shape[0] < 2:
        return plain
    if mode not in VALID_COMBINE_MODES:
        raise ValueError(
            f"Unknown combine mode: {mode!r}. Valid: {', '.join(VALID_COMBINE_MODES)}")

    u = _dominant_direction(X)
    # Base convention: most members positive. Always well defined, and it is what decides
    # the sign when the plain average has itself cancelled to ~0 -- which is exactly the
    # case these modes exist for, so it cannot be used as the reference there.
    if float(u.sum()) < 0:
        u = -u

    if mode == "mean_flip":
        signs = np.sign(u)
        signs[signs == 0] = 1.0
        out = (signs[:, None] * X).mean(axis=0)
    else:
        # pca_flip: the dominant direction, scaled by 1/sqrt(n_members) so that members
        # which agree after alignment return their own amplitude rather than sqrt(n) times
        # it -- the same scale mean_flip gives. Scaling to the plain average's norm would
        # reintroduce the cancellation this mode removes.
        out = (u @ X) / np.sqrt(X.shape[0])

    # Prefer the plain average as the sign reference, so the result is never globally
    # negated relative to `mean`. Skipped when that average has cancelled to nothing, where
    # the base convention above already fixed the sign.
    if np.linalg.norm(plain) > 1e-9 * np.linalg.norm(X) and float(out @ plain) < 0:
        out = -out
    return out
