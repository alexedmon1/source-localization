"""The head-model prior, named ensembles drawn from it, and their leadfields.

A location posterior averages its likelihood over head models (:mod:`.ensemble_posterior`), so it widens only where
plausible head models disagree about where a pattern comes from. Which head models are plausible is this module's
prior. It is an assumption, stated so it can be argued with:

- **Array registration:** a rigid shift of the whole electrode array, each axis independently Normal(0, SD). Applied
  before MNE projects the electrodes onto the BEM scalp, which largely undoes the component normal to the scalp, so
  the effective error is mostly a sliding of the array.
- **Skull conductivity:** log-uniform between two multiples of the model's own value. Brain and scalp are unchanged.

Not in the prior: head or brain size, array rotation or stretch, per-electrode misplacement, BEM shape error, brain or
scalp conductivity error.

**Named ensembles.** An :class:`EnsembleSpec` fixes a prior, the number of draws and the random seed, so an ensemble is
identified by its catalogue id (``methods.LOCATION_PRIORS``) and reproduces exactly. :data:`ENSEMBLE_PA_PHASE2` is the
33-model ensemble every calibration in the catalogue was measured with.

Ported from the probability-atlas method workspace (``pa/ensemble.py``; 2026-10), merged with the copy the validation
regime used (``validation/head_models.py``, which now imports these classes from here). Draws are bit-identical to
both.
"""
from __future__ import annotations

import hashlib
import json
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Sequence, Tuple

import numpy as np

from ..bem.solution import SOLVER_TAG

__all__ = ["HeadModel", "HeadModelPrior", "EnsembleSpec", "ENSEMBLE_PA_PHASE2", "ENSEMBLES", "draw_ensemble",
           "ensemble_leadfields"]


@dataclass(frozen=True)
class HeadModel:
    """One head model, as a perturbation of the nominal one.

    Parameters
    ----------
    shift_mm : (3,) tuple
        Rigid shift of every electrode, in mm, in the head frame (x left-right, y posterior-anterior, z down-up).
    skull_factor : float
        Skull conductivity as a multiple of the nominal model's.
    """

    shift_mm: Tuple[float, float, float] = (0.0, 0.0, 0.0)
    skull_factor: float = 1.0

    def __post_init__(self):
        shift = tuple(float(v) for v in np.asarray(self.shift_mm, float).ravel())
        if len(shift) != 3:
            raise ValueError("shift_mm must have three components")
        if not self.skull_factor > 0:
            raise ValueError("skull_factor must be positive")
        object.__setattr__(self, "shift_mm", shift)
        object.__setattr__(self, "skull_factor", float(self.skull_factor))

    @property
    def is_nominal(self) -> bool:
        return self.shift_mm == (0.0, 0.0, 0.0) and self.skull_factor == 1.0


@dataclass(frozen=True)
class HeadModelPrior:
    """Prior over head models (module docstring).

    ``shift_sd_mm`` is one SD for all three axes, or a per-axis (x, y, z) tuple: ``(0.5, 0.5, 0.0)`` is "x and y
    only, 0.5 mm". ``skull_factor_range`` bounds the log-uniform skull-conductivity factor.
    """

    shift_sd_mm: object = 0.3
    skull_factor_range: Tuple[float, float] = (0.5, 2.0)

    def draw(self, rng: np.random.Generator) -> HeadModel:
        """One head model. Consumes 3 normals, then 1 uniform, from ``rng`` (the order is part of reproducibility)."""
        sd = np.broadcast_to(np.asarray(self.shift_sd_mm, float), (3,))
        shift = rng.normal(0.0, 1.0, 3) * sd
        lo, hi = self.skull_factor_range
        factor = float(np.exp(rng.uniform(np.log(lo), np.log(hi))))
        return HeadModel(tuple(shift), factor)

    def draw_many(self, n: int, seed) -> list:
        rng = np.random.default_rng(seed)
        return [self.draw(rng) for _ in range(n)]


def draw_ensemble(prior: HeadModelPrior, n_draws: int, rng: np.random.Generator,
                  include_nominal: bool = True) -> List[HeadModel]:
    """The nominal model (first, if ``include_nominal``) plus ``n_draws`` draws from ``prior``.

    The nominal model first matters: the ensemble posteriors take the moment prior's gain scale from their first
    model.
    """
    models = [HeadModel()] if include_nominal else []
    return models + [prior.draw(rng) for _ in range(n_draws)]


@dataclass(frozen=True)
class EnsembleSpec:
    """A named, reproducible head-model ensemble: its catalogue id, prior, number of draws and seed."""

    id: str
    prior: HeadModelPrior
    n_draws: int
    seed: Tuple[int, ...]
    include_nominal: bool = True

    def models(self) -> List[HeadModel]:
        return draw_ensemble(self.prior, self.n_draws, np.random.default_rng(list(self.seed)), self.include_nominal)

    @property
    def n_models(self) -> int:
        return self.n_draws + int(self.include_nominal)

    def provenance(self) -> dict:
        """What an output records about the ensemble it used."""
        from ..methods import catalogue_entries
        entry = catalogue_entries().get(self.id)
        return {"id": self.id, "version": entry.version if entry else None,
                "shift_sd_mm": np.broadcast_to(np.asarray(self.prior.shift_sd_mm, float), (3,)).tolist(),
                "skull_factor_range": list(self.prior.skull_factor_range), "n_draws": self.n_draws,
                "seed": list(self.seed), "include_nominal": self.include_nominal}


#: The ensemble of probability-atlas Phase 2 (``scripts/phase2_simulate.py``: ``VARIANTS["primary"]``, ``K_ENS = 32``,
#: ``rng_for(1, 0)`` with ``SEED = 20261005``), used by the field method and every catalogued calibration.
ENSEMBLE_PA_PHASE2 = EnsembleSpec("prior.head_ensemble_pa_phase2", HeadModelPrior(0.3, (0.5, 2.0)), 32,
                                  (20261005, 1, 0))

#: Every named ensemble, by catalogue id.
ENSEMBLES = {e.id: e for e in (ENSEMBLE_PA_PHASE2,)}


def _signature(geometry, models: Sequence[HeadModel], positions_mm: np.ndarray) -> str:
    # "bem_solver" is the solver the geometry's own leadfields use. geometry.signature cannot stand in for it: it is the
    # nominal operator's signature, which probability-atlas's Geometry (pa/geometry.py, inner-first solve) shares with
    # this package's. Before 2026-11-14 the key was exactly probability-atlas's, so an ensemble pa had solved in the
    # faulty order was served here under the corrected solver's key (bayes-sl, 2026-11-14).
    payload = {"geometry": geometry.signature, "models": [[m.shift_mm, m.skull_factor] for m in models],
               "positions": hashlib.sha1(np.ascontiguousarray(positions_mm, float).tobytes()).hexdigest(),
               "bem_solver": getattr(geometry, "bem_solver", None)}
    return hashlib.sha1(json.dumps(payload, sort_keys=True, default=str).encode()).hexdigest()[:16]


def ensemble_leadfields(geometry, models: Sequence[HeadModel], positions_mm: Optional[np.ndarray] = None,
                        cache_dir=None, verbose: bool = True) -> np.ndarray:
    """Leadfields of every model on the same positions, shape (K, n_positions, n_channels, 3).

    Unwhitened and unreferenced (as MNE returns them), channels in ``geometry.channel_names`` order. About 4 s per
    model on a 4,357-position grid. Cached in ``cache_dir`` as ``ensemble_<key>.npy``, the key built from the
    geometry's signature, its BEM solver (``geometry.bem_solver``), the models and the positions, so a changed prior,
    BEM or solver cannot be served a stale ensemble. A geometry that does not declare the current solver
    (``bem.solution.SOLVER_TAG``; probability-atlas's does not) is computed but never cached: its keys used to equal
    this package's while its leadfields were solved inner first.
    """
    positions_mm = geometry.positions_mm if positions_mm is None else np.asarray(positions_mm, float)
    path = None
    if cache_dir is not None and getattr(geometry, "bem_solver", None) != SOLVER_TAG:
        warnings.warn(f"{type(geometry).__module__}.{type(geometry).__name__} does not declare the current BEM solver "
                      f"({SOLVER_TAG!r}); its ensemble leadfields are neither read from nor written to {cache_dir}",
                      stacklevel=2)
        cache_dir = None
    if cache_dir is not None:
        path = Path(cache_dir) / f"ensemble_{_signature(geometry, models, positions_mm)}.npy"
        if path.exists():
            if verbose:
                print(f"  ensemble leadfields loaded from cache ({path.name})")
            return np.load(path)
    G = np.empty((len(models), len(positions_mm), geometry.n_channels, 3))
    for k, m in enumerate(models):
        G[k] = geometry.leadfield(m, positions_mm)
        if verbose and (k + 1) % 8 == 0:
            print(f"  ensemble: {k + 1}/{len(models)} head models")
    if path is not None:
        path.parent.mkdir(parents=True, exist_ok=True)
        np.save(path, G)
    return G
