"""Which source samplings build a dense pool rather than a deployed grid.

Monte Carlo sampling and the parcel-subspace operator both start from the same thing: a
dense pool of candidate sources, from which an ROI operator is built without solving any one
grid. Every source space that builds a pool for one must build it for the other, so the
question is asked here, once, instead of as a string comparison in each source space.

Pool settings (``pool_spacing_mm``, ``pool_n_shells``, ``pool_points_per_shell``) are read
from the active sampling's own config section, ``source_space.monte_carlo`` or
``source_space.parcel_subspace``. A ``parcel_subspace`` section that omits a pool key falls
back to ``monte_carlo``'s, so a subspace run on a Monte Carlo config builds the identical pool.
"""
from __future__ import annotations

POOL_SAMPLINGS = ("monte_carlo", "parcel_subspace")


def sampling_of(config) -> str:
    return (config["source_space"].get("source_sampling", "fixed") or "fixed")


def is_pool_sampling(config) -> bool:
    """True when the source space is a pool for an ROI operator, not a deployed grid."""
    return sampling_of(config) in POOL_SAMPLINGS


def pool_config(config) -> dict:
    """Pool settings for the active sampling (empty under fixed sampling)."""
    ss = config["source_space"]
    mode = sampling_of(config)
    if mode == "monte_carlo":
        return dict(ss.get("monte_carlo") or {})
    if mode == "parcel_subspace":
        return {**(ss.get("monte_carlo") or {}), **(ss.get("parcel_subspace") or {})}
    return {}
