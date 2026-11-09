"""The head geometry a location posterior runs on, and leadfields under a perturbed head model.

Ported unchanged in behaviour from probability-atlas ``pa/geometry.py`` (field-method plan, phase F1); its
``signature`` is the same, so ensemble caches written there are reused here.

Read from a completed source-localization pipeline run (read-only): its cached ellipsoid BEM
(``bem_cache/ellipsoid_3layer.pkl``), montage and registration (``data/step1_info.pkl``) and volume source-space
template (``data/step3_source_space.pkl``). The forward model depends only on these, not on the recording, so any run
of the same study and arm serves every recording. (The catalogued calibrations were measured on Sentinel's
``sentinel_40hz_cartesian_mc48`` run.)

The candidate positions are ``DipolePosterior``'s dense grid (default 0.5 mm spacing, inside the atlas brain mask and
inside 0.93 of the BEM's inner ellipsoid; 4,357 positions for Sentinel). The nominal model's leadfield is that
operator's; perturbed models are recomputed with MNE on the same positions, so their only difference is the
perturbation.
"""
from __future__ import annotations

import copy
import pickle
from collections import OrderedDict
from pathlib import Path

import mne
import numpy as np
from ..validation.posterior import DipolePosterior

from .head_ensemble import HeadModel

__all__ = ["Geometry"]


class Geometry:
    """Head geometry from a pipeline run directory (any study whose run has an ellipsoid BEM).

    Parameters
    ----------
    run_dir : path-like
        A completed pipeline run with ``bem_cache/`` and ``data/``. Read only.
    spacing_mm : float
        Dense grid spacing for the posterior (reference: 0.5).
    cache_dir : path-like, optional
        Where source-localization caches the nominal dense operator (``operator_<spacing>mm.npz``).
    """

    def __init__(self, run_dir, spacing_mm: float = 0.5, cache_dir=None, n_cached_solutions: int = 4):
        mne.set_log_level("ERROR")
        self.run_dir = Path(run_dir)
        self.nominal = DipolePosterior.from_pipeline_dir(self.run_dir, spacing_mm=spacing_mm,
                                                         cache_dir=cache_dir, verbose=False)
        self.positions_mm = self.nominal.positions_mm
        self.spacing_mm = self.nominal.spacing_mm
        self.channel_names = list(self.nominal.channel_names)
        self.electrode_pos_mm = np.asarray(self.nominal.electrode_pos_mm, float)
        self.n_channels = len(self.channel_names)

        self.bem = pickle.load(open(self.run_dir / "bem_cache/ellipsoid_3layer.pkl", "rb"))
        self.info = pickle.load(open(self.run_dir / "data/step1_info.pkl", "rb"))
        self.src_template = pickle.load(open(self.run_dir / "data/step3_source_space.pkl", "rb"))
        if list(self.info["ch_names"]) != self.channel_names:
            raise ValueError("step1_info channel order differs from the cached operator's")
        self.skull = int(np.argmin([s["sigma"] for s in self.bem["surfs"]]))
        self.skull_sigma = float(self.bem["surfs"][self.skull]["sigma"])
        self.signature = dict(self.nominal.provenance.get("signature", {}), skull_sigma=self.skull_sigma)
        self._solutions: OrderedDict = OrderedDict()
        self._n_cached = n_cached_solutions

    # ------------------------------------------------------------------ forward
    def _solution(self, skull_factor: float):
        key = round(float(skull_factor), 6)
        if key in self._solutions:
            self._solutions.move_to_end(key)
            return self._solutions[key]
        surfs = copy.deepcopy(self.bem["surfs"])
        surfs[self.skull]["sigma"] = self.skull_sigma * skull_factor
        sol = mne.make_bem_solution(surfs, verbose=False)
        self._solutions[key] = sol
        while len(self._solutions) > self._n_cached:
            self._solutions.popitem(last=False)
        return sol

    def leadfield(self, model: HeadModel, positions_mm: np.ndarray | None = None) -> np.ndarray:
        """Free-orientation leadfield under ``model``, shape (n_positions, n_channels, 3).

        Electrodes are shifted by ``model.shift_mm`` in the head frame before MNE projects them onto
        the scalp; the skull's conductivity is scaled by ``model.skull_factor``. Unreferenced (MNE's
        output); the whitener applies the average reference.
        """
        positions_mm = self.positions_mm if positions_mm is None else np.asarray(positions_mm, float)
        info = copy.deepcopy(self.info)
        with info._unlock():
            for c in info["chs"]:
                c["loc"][:3] += np.asarray(model.shift_mm) / 1000.0
        src = DipolePosterior._as_source_space(self.src_template, positions_mm / 1000.0)
        f = mne.make_forward_solution(info, trans=None, src=src, bem=self._solution(model.skull_factor),
                                      eeg=True, meg=False, verbose=False)
        if f["nsource"] != len(positions_mm):
            raise RuntimeError(f"MNE dropped {len(positions_mm) - f['nsource']} positions")
        return f["sol"]["data"].reshape(self.n_channels, -1, 3).transpose(1, 0, 2)

    def nominal_leadfield(self) -> np.ndarray:
        """The cached nominal operator's leadfield, (n_positions, n_channels, 3)."""
        return self.nominal._G
