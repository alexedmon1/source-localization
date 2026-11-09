"""Everything a location posterior needs about the head, in one object.

Before this module, each study script that located a field built the same block by hand: the geometry from a pipeline
run, the head-model ensemble and its leadfields, region labels for the grid, and the average-reference basis
(surface-evoked ``localize_fields_bayes.py``, ``readout_filters.py``, ``method_crossover.Space``; bayes-sl ``bm_fm.py``
and the ``tb*`` scripts). :class:`LocationSpace` builds it once, from named, catalogued parts, and says which ones it
used (:meth:`LocationSpace.provenance`), so a location result can name the ensemble and labelling it rests on.
"""
from __future__ import annotations

from pathlib import Path
from typing import Optional

import numpy as np

from .geometry import Geometry
from .head_ensemble import ENSEMBLES, EnsembleSpec, ensemble_leadfields
from .regions import labelling
from .whitening import average_reference_basis

__all__ = ["LocationSpace"]


class LocationSpace:
    """Grid, ensemble leadfields, region labels and reference basis for one study's head.

    Parameters
    ----------
    run_dir : path-like
        A completed pipeline run of the study (ellipsoid BEM), read only (:class:`~.geometry.Geometry`).
    ensemble : str or EnsembleSpec
        A catalogued ensemble id (``methods.LOCATION_PRIORS``) or a spec.
    spacing_mm : float
        Grid spacing of the candidate positions (the catalogued calibrations hold at 0.5 mm).
    labelling_kind, atlas, surface_run_dir
        Passed to :func:`~.regions.labelling`.
    cache_dir : path-like, optional
        Where the dense nominal operator and the ensemble leadfields are cached.

    Attributes
    ----------
    geometry, ensemble, models
        The geometry, the ensemble spec and its head models (nominal first).
    G_ens : (K, P, n_channels, 3)
        Unreferenced ensemble leadfields on the candidate positions.
    positions_mm, spacing_mm, channel_names, electrode_pos_mm
    labels : dict
        :func:`~.regions.labelling`'s output for the positions.
    B_ref : (n_channels, n_channels - 1)
        Orthonormal average-reference basis.
    """

    def __init__(self, run_dir, ensemble="prior.head_ensemble_pa_phase2", spacing_mm: float = 0.5,
                 labelling_kind: str = "volumetric", atlas: str = "allen26", surface_run_dir=None,
                 cache_dir=None, verbose: bool = False):
        self.run_dir = Path(run_dir)
        self.ensemble: EnsembleSpec = ENSEMBLES[ensemble] if isinstance(ensemble, str) else ensemble
        self.geometry = Geometry(self.run_dir, spacing_mm=spacing_mm,
                                 cache_dir=None if cache_dir is None else Path(cache_dir) / "posterior_cache")
        self.models = self.ensemble.models()
        self.labels = labelling(self.geometry, labelling_kind, atlas, surface_run_dir)
        self.positions_mm = self.labels["positions_mm"]
        self.G_ens = ensemble_leadfields(self.geometry, self.models, self.positions_mm, cache_dir=cache_dir,
                                         verbose=verbose)
        self.spacing_mm = self.geometry.spacing_mm
        self.channel_names = list(self.geometry.channel_names)
        self.electrode_pos_mm = self.geometry.electrode_pos_mm
        self.B_ref = average_reference_basis(len(self.channel_names))

    @property
    def region_names(self) -> list:
        return list(self.labels["parcel_names"])

    def referenced_leadfields(self) -> np.ndarray:
        """``G_ens`` average-referenced over channels (the reference the data are in before whitening)."""
        return self.G_ens - self.G_ens.mean(axis=2, keepdims=True)

    def provenance(self) -> dict:
        """What a location result records about the space it was computed in."""
        return {"ensemble": self.ensemble.provenance(),
                "geometry": {"run_dir": str(self.run_dir), "signature": self.geometry.signature,
                             "spacing_mm": self.spacing_mm, "n_positions": int(len(self.positions_mm)),
                             "channels": self.channel_names},
                "labelling": {"kind": self.labels["kind"], "atlas": self.labels["atlas"],
                              "n_regions": len(self.region_names)}}
