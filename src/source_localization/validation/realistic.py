"""The ``realistic`` validation regime's truths and backgrounds, shared by every validation component.

:mod:`.regime` defines what the realistic regime assumes; this module supplies it, so the CLI runner,
``RobustnessTest`` and ``BatchValidationRunner`` simulate truths the same way:

- a pool of head models drawn once per run from :class:`.head_models.HeadModelPrior` (seeded);
- each trial picks one model at random from a key the caller gives (e.g. position and trial index), so a run is
  reproducible;
- the truth's leadfield at the **requested** position under that model (cached per position);
- a recorded background window (:class:`.noise.RecordedBackground`) when one is configured.
"""
from __future__ import annotations

from dataclasses import asdict
from typing import Any, Dict, Optional, Tuple

import numpy as np

from .head_models import HeadModelPrior, TruthForward
from .regime import Regime

__all__ = ["RealisticTruths"]


class RealisticTruths:
    """Truth leadfields and backgrounds for one realistic-regime run.

    Parameters
    ----------
    regime : Regime
        From :func:`.regime.resolve_regime`; must be the realistic regime.
    info : mne.Info
        The montage the simulation uses.
    bem : dict or ConductorModel
        The BEM the truths are simulated with (the pipeline's BEM step output, or a ``bem_cache`` pickle).
    background : RecordedBackground, optional
    """

    def __init__(self, regime: Regime, info, bem, background=None):
        if regime.is_legacy:
            raise ValueError("RealisticTruths is for the realistic regime; legacy simulates with the inversion's "
                             "own forward model")
        self.regime = regime
        self.forward = TruthForward(info, bem)
        self.models = HeadModelPrior(**regime.head_model_prior).draw_many(regime.n_head_models, regime.seed)
        self.background = background
        self._cache: Dict[Tuple[float, ...], np.ndarray] = {}
        self.n_fallback = 0

    @staticmethod
    def _key(position_mm) -> Tuple[float, ...]:
        return tuple(np.round(np.asarray(position_mm, float).ravel(), 6))

    def precompute(self, positions_mm) -> np.ndarray:
        """Compute and cache every model's leadfield at ``positions_mm`` (one forward per model). Returns the kept
        mask: False where MNE excluded a position (outside the inner skull)."""
        P = np.atleast_2d(np.asarray(positions_mm, float))
        todo = [i for i, p in enumerate(P) if self._key(p) not in self._cache]
        if todo:
            G, _ = self.forward.gains_for_models(self.models, P[todo])
            for j, i in enumerate(todo):
                self._cache[self._key(P[i])] = G[:, j]
        return np.array([not np.isnan(self._cache[self._key(p)]).any() for p in P])

    def gain(self, position_mm, k: int) -> Optional[np.ndarray]:
        """Model ``k``'s leadfield (n_channels, 3) at ``position_mm``; None if the position is outside the head
        (the caller then falls back to the simulation forward at the snapped source, and :attr:`n_fallback`
        counts it)."""
        key = self._key(position_mm)
        if key not in self._cache:
            self.precompute(np.asarray(position_mm, float)[None])
        g = self._cache[key][k]
        if np.isnan(g).any():
            self.n_fallback += 1
            return None
        return g

    def choose(self, *key: int):
        """(model index, rng) for one trial, reproducible from ``key`` and the regime seed."""
        rng = np.random.default_rng([self.regime.seed, *[int(v) for v in key]])
        return int(rng.integers(len(self.models))), rng

    def background_for(self, n_times: int, sfreq: float, rng, noise_type: Optional[str] = None):
        """A recorded background window, or None. A background replaces **white** noise only: an explicitly
        coloured ``noise_type`` is kept, because varying it is the point of the tests that set it."""
        if self.background is None or (noise_type not in (None, 'white')):
            return None
        return self.background.draw(n_times, rng, sfreq=sfreq)

    def meta(self) -> Dict[str, Any]:
        return {
            "prior": dict(self.regime.head_model_prior),
            "models": [asdict(m) for m in self.models],
            "n_positions": len(self._cache),
            # Positions outside the inner skull fall back to the simulation forward at the snapped source.
            "n_simulations_snapped_fallback": int(self.n_fallback),
            "background_files": list(self.background.files) if self.background is not None else None,
        }
