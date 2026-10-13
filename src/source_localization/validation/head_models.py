"""Head models for simulating truths: the prior, and leadfields under a perturbed head model.

Used by the ``realistic`` validation regime (:mod:`.regime`): each simulated truth goes through a head model drawn
from :class:`HeadModelPrior`, while the inverse keeps the nominal model, so the validation sees model error the way
real data do. Ported from the probability-atlas method workspace (``pa/ensemble.py``, ``pa/geometry.py``,
2026-10), where it was calibrated on Sentinel and FORGE MEA30 data.

The prior is an assumption, stated so it can be argued with:

- **Array registration:** a rigid shift of the whole electrode array, each axis independently N(0, 0.3 mm). It is
  applied before MNE projects the electrodes onto the BEM scalp, which largely undoes the component normal to the
  scalp. A z shift therefore acts about a third as strongly as the same shift in x or y.
- **Skull conductivity:** log-uniform between 0.5x and 2x the model's own value. Brain and scalp are unchanged.

Not in the prior: head or brain size, array rotation or stretch, per-electrode misplacement, BEM shape error, and
brain or scalp conductivity error.
"""
from __future__ import annotations

import copy
from collections import OrderedDict
from dataclasses import dataclass
from typing import Optional, Sequence, Tuple

import numpy as np

__all__ = ["HeadModel", "HeadModelPrior", "TruthForward"]


@dataclass(frozen=True)
class HeadModel:
    """One head model, as a perturbation of the nominal one.

    Parameters
    ----------
    shift_mm : (3,) tuple
        Rigid shift of every electrode, in mm, in the head frame.
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
        """One head model. Consumes 3 normals, then 1 uniform, from ``rng``."""
        sd = np.broadcast_to(np.asarray(self.shift_sd_mm, float), (3,))
        shift = rng.normal(0.0, 1.0, 3) * sd
        lo, hi = self.skull_factor_range
        factor = float(np.exp(rng.uniform(np.log(lo), np.log(hi))))
        return HeadModel(tuple(shift), factor)

    def draw_many(self, n: int, seed: int) -> list:
        rng = np.random.default_rng(seed)
        return [self.draw(rng) for _ in range(n)]


class TruthForward:
    """Free-orientation leadfields at arbitrary positions under a perturbed head model.

    Parameters
    ----------
    info : mne.Info
        The montage the simulation uses (channel order of every gain returned).
    bem : dict or mne.bem.ConductorModel
        The simulation's BEM, as the pipeline's BEM step returns it: a dict with ``surfs`` (ellipsoid, numerical
        sphere) or an analytical sphere model.
    mindist_mm : float
        Passed to MNE as ``mindist``. 0 keeps positions right at the inner skull.
    n_cached_solutions : int
        BEM solutions kept in memory (one per distinct skull factor).
    """

    def __init__(self, info, bem, mindist_mm: float = 0.0, n_cached_solutions: int = 4):
        self.info = info
        self.bem = bem
        self.mindist_mm = float(mindist_mm)
        self.n_channels = len(info["ch_names"])
        self._solutions: "OrderedDict[float, object]" = OrderedDict()
        self._n_cached = int(n_cached_solutions)

    # ------------------------------------------------------------------ BEM
    def _is_surface_bem(self) -> bool:
        return isinstance(self.bem, dict) and "surfs" in self.bem and not self.bem.get("is_sphere", False)

    def _solution(self, skull_factor: float):
        import mne
        key = round(float(skull_factor), 9)
        if key in self._solutions:
            self._solutions.move_to_end(key)
            return self._solutions[key]
        if self._is_surface_bem():
            surfs = copy.deepcopy(self.bem["surfs"])
            skull = int(np.argmin([s["sigma"] for s in surfs]))
            surfs[skull]["sigma"] = float(surfs[skull]["sigma"]) * skull_factor
            sol = mne.make_bem_solution(surfs, verbose=False)
        else:
            layers = self.bem["layers"]
            sig = [float(l["sigma"]) for l in layers]
            skull = int(np.argmin(sig))
            sig[skull] *= skull_factor
            sol = mne.make_sphere_model(r0=self.bem["r0"], head_radius=float(layers[-1]["rad"]),
                                        relative_radii=[float(l["rel_rad"]) for l in layers], sigmas=sig,
                                        verbose=False)
        self._solutions[key] = sol
        while len(self._solutions) > self._n_cached:
            self._solutions.popitem(last=False)
        return sol

    # ------------------------------------------------------------------ forward
    def gains(self, model: HeadModel, positions_mm: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Leadfields at ``positions_mm`` under ``model``.

        Returns
        -------
        gains : ndarray, shape (n_positions, n_channels, 3)
            Unreferenced, as MNE returns them. NaN for positions MNE excluded.
        kept : ndarray of bool, shape (n_positions,)
            False where MNE excluded the position (outside the inner skull, or closer than ``mindist_mm``).
        """
        import mne
        positions_mm = np.atleast_2d(np.asarray(positions_mm, float))
        info = copy.deepcopy(self.info)
        with info._unlock():
            for c in info["chs"]:
                c["loc"][:3] += np.asarray(model.shift_mm) / 1000.0
        n = len(positions_mm)
        src = mne.setup_volume_source_space(
            pos={"rr": positions_mm / 1000.0, "nn": np.tile([0.0, 0.0, 1.0], (n, 1))}, verbose=False)
        fwd = mne.make_forward_solution(info, trans=None, src=src, bem=self._solution(model.skull_factor),
                                        eeg=True, meg=False, mindist=self.mindist_mm, verbose=False)
        used = fwd["src"][0]["vertno"]
        out = np.full((n, self.n_channels, 3), np.nan)
        out[used] = fwd["sol"]["data"].reshape(self.n_channels, -1, 3).transpose(1, 0, 2)
        kept = np.zeros(n, bool)
        kept[used] = True
        return out, kept

    def gains_for_models(self, models: Sequence[HeadModel], positions_mm: np.ndarray):
        """Stacked :meth:`gains` for several models: (n_models, n_positions, n_channels, 3) and the kept mask
        (a position is kept only if every model kept it)."""
        G, keep = [], None
        for m in models:
            g, k = self.gains(m, positions_mm)
            G.append(g)
            keep = k if keep is None else (keep & k)
        return np.stack(G), keep
