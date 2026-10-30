"""The head-model ensemble over a pipeline run's geometry, and the file that hands it to source-analytics.

:class:`EnsembleGeometry` reads a completed pipeline run (read-only): its cached ellipsoid BEM, montage and
volume source-space template. The forward model depends only on these, not on the recording, so any run of the same
arm serves every recording. The candidate positions are :class:`~.posterior.DipolePosterior`'s dense grid; the nominal
model's leadfield is that operator's, and perturbed models are recomputed with MNE on the same positions, so their only
difference is the perturbation. Ported from the probability-atlas method workspace (``pa/geometry.py``,
``pa/ensemble.py``, 2026-10, commit f49d292).

:data:`MEA30_ENSEMBLE` fixes the 33 head models and the 8 held-out truth models used on the MEA30 array. Its draws
reproduce the workspace's exactly.

:func:`export_head_ensemble` writes the ``.npz`` that source-analytics reads: it keeps the package boundary, where this
package builds the head-model side and source-analytics fits and scores, and they exchange files.
"""
from __future__ import annotations

import copy
import hashlib
import json
import pickle
from collections import OrderedDict
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Sequence

import numpy as np

from .head_models import HeadModel, HeadModelPrior, draw_ensemble

__all__ = ["EnsembleGeometry", "EnsembleSpec", "MEA30_ENSEMBLE", "ensemble_leadfields", "ccf_ids",
           "export_head_ensemble", "load_head_ensemble", "EXPORT_VERSION"]

EXPORT_VERSION = 1


@dataclass(frozen=True)
class EnsembleSpec:
    """A fixed head-model ensemble: the prior, its draw counts and the seeds that reproduce them."""

    prior: HeadModelPrior
    n_draws: int
    seed: tuple
    n_truth: int
    truth_seed: tuple

    def models(self, offset_mm: Sequence[float] = (0.0, 0.0, 0.0)) -> list:
        """The nominal model plus the draws, every one shifted by ``offset_mm`` (a registration error to test)."""
        return _offset(draw_ensemble(self.prior, self.n_draws, np.random.default_rng(list(self.seed))), offset_mm)

    def truth_models(self, offset_mm: Sequence[float] = (0.0, 0.0, 0.0)) -> list:
        """Models outside :meth:`models`, for simulating truths the fit has not seen."""
        rng = np.random.default_rng(list(self.truth_seed))
        return _offset(draw_ensemble(self.prior, self.n_truth, rng, include_nominal=False), offset_mm)


def _offset(models, offset_mm):
    off = np.asarray(offset_mm, float)
    return [HeadModel(tuple(np.asarray(m.shift_mm) + off), m.skull_factor) for m in models] if off.any() else models


MEA30_ENSEMBLE = EnsembleSpec(prior=HeadModelPrior(0.3, (0.5, 2.0)), n_draws=32, seed=(20261005, 1, 0),
                              n_truth=8, truth_seed=(20261005, 9, 0))


class EnsembleGeometry:
    """Head geometry from a pipeline run directory.

    Parameters
    ----------
    run_dir : path-like
        A completed pipeline run with ``bem_cache/`` and ``data/``. Read only.
    spacing_mm : float
        Dense grid spacing.
    cache_dir : path-like, optional
        Where the nominal dense operator is cached.
    """

    def __init__(self, run_dir, spacing_mm: float = 0.5, cache_dir=None, n_cached_solutions: int = 4):
        import mne
        from .posterior import DipolePosterior
        mne.set_log_level("ERROR")
        self.run_dir = Path(run_dir)
        self.nominal = DipolePosterior.from_pipeline_dir(self.run_dir, spacing_mm=spacing_mm, cache_dir=cache_dir,
                                                         verbose=False)
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

    def _solution(self, skull_factor: float):
        import mne
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

    def leadfield(self, model: HeadModel, positions_mm: Optional[np.ndarray] = None) -> np.ndarray:
        """Free-orientation leadfield under ``model``, (n_positions, n_channels, 3), unreferenced (MNE's output).

        Electrodes are shifted by ``model.shift_mm`` before MNE projects them onto the scalp, so the component of a shift
        normal to the scalp is largely undone; the skull's conductivity is scaled by ``model.skull_factor``.
        """
        import mne
        from .posterior import DipolePosterior
        positions_mm = self.positions_mm if positions_mm is None else np.asarray(positions_mm, float)
        info = copy.deepcopy(self.info)
        with info._unlock():
            for c in info["chs"]:
                c["loc"][:3] += np.asarray(model.shift_mm) / 1000.0
        src = DipolePosterior._as_source_space(self.src_template, positions_mm / 1000.0)
        f = mne.make_forward_solution(info, trans=None, src=src, bem=self._solution(model.skull_factor), eeg=True,
                                      meg=False, verbose=False)
        if f["nsource"] != len(positions_mm):
            raise RuntimeError(f"MNE dropped {len(positions_mm) - f['nsource']} positions")
        return f["sol"]["data"].reshape(self.n_channels, -1, 3).transpose(1, 0, 2)


def _signature(geometry, models, positions_mm) -> str:
    payload = {"geometry": geometry.signature, "models": [[m.shift_mm, m.skull_factor] for m in models],
               "positions": hashlib.sha1(np.ascontiguousarray(positions_mm, float).tobytes()).hexdigest()}
    return hashlib.sha1(json.dumps(payload, sort_keys=True, default=str).encode()).hexdigest()[:16]


def ensemble_leadfields(geometry, models, positions_mm: Optional[np.ndarray] = None, cache_dir=None,
                        verbose: bool = True) -> np.ndarray:
    """Leadfields of every model on the same positions, (K, n_positions, n_channels, 3), unreferenced.

    Cached under a key from the geometry's signature, the models and the positions, so a changed prior or BEM cannot be
    served a stale ensemble.
    """
    positions_mm = geometry.positions_mm if positions_mm is None else np.asarray(positions_mm, float)
    path = None
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


def ccf_ids(positions_mm: np.ndarray) -> np.ndarray:
    """CCFv3 structure id per position; a position on an unlabelled voxel takes the nearest labelled voxel's id."""
    import nibabel as nib
    from scipy.spatial import cKDTree
    from ..utils.atlas import get_true_affine
    nii = nib.load(str(Path(__file__).resolve().parents[1] / "data/atlas/allen/allen_annotation_in_antwerp.nii.gz"))
    vol = np.asarray(nii.dataobj).astype(np.int64)
    occ = np.argwhere(vol > 0)
    occ_mm = (get_true_affine(nii) @ np.c_[occ, np.ones(len(occ))].T).T[:, :3]
    _, i = cKDTree(occ_mm).query(np.asarray(positions_mm, float))
    return vol[tuple(occ[i].T)]


def _ontology() -> dict:
    """CCFv3 structures as ``{id: {acronym, name, parent}}``; ids in the volume but not here are invalid (known issue)."""
    g = json.load(open(Path(__file__).resolve().parents[1] / "data/atlas/allen/structure_graph_1.json"))
    out, stack = {}, [(g["msg"][0] if "msg" in g else g, None)]
    while stack:
        n, parent = stack.pop()
        out[int(n["id"])] = dict(acronym=n["acronym"], name=n["name"], parent=parent)
        stack.extend((c, int(n["id"])) for c in n.get("children", []))
    return out


def export_head_ensemble(run_dir, out_path, atlas: str = "allen26", spec: EnsembleSpec = MEA30_ENSEMBLE,
                         offset_mm: Sequence[float] = (0.0, 0.0, 0.0), spacing_mm: float = 0.5, cache_dir=None,
                         verbose: bool = True) -> Path:
    """Write the head-model file: the ensemble's leadfields and everything needed to read a location off them.

    Arrays: ``leadfields`` (K, n, channels, 3), unreferenced; ``truth_leadfields`` (the models outside the fit);
    ``positions_mm``; ``electrode_pos_mm``; ``parcel_label`` (the atlas label id per position); ``ccf_id``.
    ``meta_json`` carries the channel names, parcel names by id, CCFv3 ontology, models, frame offset, atlas and versions.
    """
    from .. import __version__
    from .roi_certainty import ParcelMap
    geo = EnsembleGeometry(run_dir, spacing_mm=spacing_mm, cache_dir=cache_dir)
    models, truth = spec.models(offset_mm), spec.truth_models(offset_mm)
    G = ensemble_leadfields(geo, models, cache_dir=cache_dir, verbose=verbose)
    Gt = ensemble_leadfields(geo, truth, cache_dir=cache_dir, verbose=verbose)
    pmap = ParcelMap(geo.positions_mm, atlas=atlas, electrode_pos_mm=geo.electrode_pos_mm)
    ids = np.unique(pmap.labels)
    meta = dict(export_version=EXPORT_VERSION, source_localization_version=__version__, atlas=atlas,
                spacing_mm=float(geo.spacing_mm), channel_names=geo.channel_names, offset_mm=[float(v) for v in offset_mm],
                frame="atlas-space mm", run_dir=str(run_dir), geometry_signature=geo.signature,
                parcel_names={int(i): pmap.parcel_names[pmap.index_of(i)] for i in ids},
                models=[[list(m.shift_mm), m.skull_factor] for m in models],
                truth_models=[[list(m.shift_mm), m.skull_factor] for m in truth],
                ccf_ontology={str(k): v for k, v in _ontology().items()})
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(out_path, leadfields=G, truth_leadfields=Gt, positions_mm=geo.positions_mm,
             electrode_pos_mm=geo.electrode_pos_mm, parcel_label=np.asarray(pmap.labels), ccf_id=ccf_ids(geo.positions_mm),
             meta_json=np.array(json.dumps(meta, default=str)))
    return out_path


def load_head_ensemble(path) -> dict:
    """Read a head-model file back; ``meta`` is the parsed ``meta_json``. Refuses a newer export version."""
    with np.load(path) as z:
        out = {k: z[k] for k in z.files if k != "meta_json"}
        out["meta"] = json.loads(str(z["meta_json"]))
    if out["meta"]["export_version"] > EXPORT_VERSION:
        raise ValueError(f"{path} has export version {out['meta']['export_version']}; this code reads {EXPORT_VERSION}")
    return out


def main(argv=None) -> int:
    import argparse
    ap = argparse.ArgumentParser(description="Export the head-model ensemble file source-analytics reads.")
    ap.add_argument("--run-dir", required=True, help="a completed pipeline run with bem_cache/ and data/")
    ap.add_argument("--out", required=True, help="the .npz to write")
    ap.add_argument("--atlas", default="allen26")
    ap.add_argument("--offset-mm", type=float, nargs=3, default=(0.0, 0.0, 0.0), metavar=("X", "Y", "Z"),
                    help="shift every head model by this registration error")
    ap.add_argument("--cache-dir", default=None)
    a = ap.parse_args(argv)
    print(export_head_ensemble(a.run_dir, a.out, atlas=a.atlas, offset_mm=a.offset_mm, cache_dir=a.cache_dir))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
