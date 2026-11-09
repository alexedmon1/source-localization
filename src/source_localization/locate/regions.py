"""From a position posterior to probabilities over regions, and credible sets.

Ported unchanged in behaviour from probability-atlas ``pa/regions.py`` (field-method plan, phase F1), with
:func:`labelling` from its ``scripts/phase2_simulate.py`` (``labelling_setup``), generalised to any run.

A region is a set of grid positions. Every position belongs to exactly one region, so aggregation
sums the position posterior within regions and conserves total probability. Built from an atlas
via source-localization's ``ParcelMap`` (every grid position force-assigned to its nearest labelled
voxel), or merged from finer regions (Phase 2's resolvable regions are merges of parcels).

Credible set at level L: the smallest set of regions, taken in order of decreasing probability,
whose probability reaches L (the rule of ``DipolePosterior.credible_mask``, applied to regions).
"""
from __future__ import annotations

from pathlib import Path
from typing import Mapping, Sequence

import numpy as np

__all__ = ["RegionMap", "credible_set", "restricted_parcel_labels", "labelling", "surface_positions", "SURFACE_CATEGORIES"]


def credible_set(probs: np.ndarray, level: float) -> np.ndarray:
    """Indices of the smallest highest-probability set holding at least ``level`` of the mass."""
    probs = np.asarray(probs, float)
    o = np.argsort(probs)[::-1]
    k = int(np.searchsorted(np.cumsum(probs[o]), level) + 1)
    return o[: min(k, len(o))]


class RegionMap:
    """Assignment of every grid position to one region.

    Parameters
    ----------
    labels : (P,) int array
        Region index (0..n_regions-1) of each position.
    names : sequence of str
        Region names, indexed by region index.
    """

    def __init__(self, labels: np.ndarray, names: Sequence[str]):
        self.labels = np.asarray(labels, int)
        self.names = list(names)
        if self.labels.min() < 0 or self.labels.max() >= len(self.names):
            raise ValueError("labels must index into names")

    @classmethod
    def from_parcel_map(cls, pmap) -> "RegionMap":
        """Regions = the atlas parcels of a source-localization ``ParcelMap`` (same assignment)."""
        return cls(np.searchsorted(pmap.parcel_ids, pmap.labels), pmap.parcel_names)

    @property
    def n_regions(self) -> int:
        return len(self.names)

    def aggregate(self, posterior: np.ndarray) -> np.ndarray:
        """Sum a position posterior (P,) or a batch (..., P) into regions (..., n_regions)."""
        posterior = np.asarray(posterior, float)
        if posterior.ndim == 1:
            return np.bincount(self.labels, weights=posterior, minlength=self.n_regions)
        onehot = np.zeros((len(self.labels), self.n_regions))
        onehot[np.arange(len(self.labels)), self.labels] = 1.0
        return posterior @ onehot

    def merge(self, groups: Mapping[str, Sequence[str]]) -> "RegionMap":
        """Merge regions: ``{new_name: [old names]}``; regions not mentioned keep their own name."""
        taken = {old for olds in groups.values() for old in olds}
        unknown = taken - set(self.names)
        if unknown:
            raise ValueError(f"unknown regions: {sorted(unknown)}")
        new_names, mapping = [], np.empty(self.n_regions, int)
        for name, olds in groups.items():
            for old in olds:
                mapping[self.names.index(old)] = len(new_names)
            new_names.append(name)
        for i, name in enumerate(self.names):
            if name not in taken:
                mapping[i] = len(new_names)
                new_names.append(name)
        return RegionMap(mapping[self.labels], new_names)


def restricted_parcel_labels(pmap, allowed_ids) -> np.ndarray:
    """Parcel id per position of a source-localization ``ParcelMap``, restricted to ``allowed_ids``.

    Positions whose nearest-label id (``pmap.labels``) is not allowed are re-assigned to the nearest
    labelled voxel among the allowed parcels. Used for the surface labelling, whose positions (the
    anatomical surface source space) cover only the cortical, cerebellum and olfactory categories,
    while the unrestricted nearest-label rule puts 1 of its 7,076 vertices in Brainstem_Tectum.
    """
    import nibabel as nib
    from scipy.spatial import cKDTree
    from ..config import atlas_input_paths
    from ..utils.atlas import get_true_affine

    allowed = np.isin(pmap.labels, list(allowed_ids))
    labels = pmap.labels.copy()
    if allowed.all():
        return labels
    pkg = Path(__import__("source_localization").__file__).parent
    nii = nib.load(str(pkg / atlas_input_paths(pmap.atlas)["brain_labels"]))
    vol = np.asarray(nii.get_fdata()).astype(int)
    occ = np.argwhere(np.isin(vol, list(allowed_ids)))
    occ_mm = (get_true_affine(nii) @ np.column_stack([occ, np.ones(len(occ))]).T).T[:, :3]
    _, idx = cKDTree(occ_mm).query(pmap.positions_mm[~allowed])
    labels[~allowed] = vol[occ[idx, 0], occ[idx, 1], occ[idx, 2]]
    return labels


#: Parcel categories the anatomical surface source space covers (allen26's ``roi_mapping_allen26.json``).
SURFACE_CATEGORIES = ("cortical", "cerebellum", "olfactory")


def surface_positions(run_dir):
    """Positions (mm) and triangles of a surface pipeline run's source space (``data/step3_*``), read only."""
    import pickle
    run_dir = Path(run_dir)
    src = pickle.load(open(run_dir / "data/step3_source_space.pkl", "rb"))
    xyz = np.load(run_dir / "data/step3_source_coords_mm.npy")
    tris, off = [], 0
    for s in src:
        if s["nuse"] != s["np"]:
            raise ValueError("surface source space must use every vertex")
        tris.append(np.asarray(s["use_tris"]) + off)
        off += s["np"]
    if off != len(xyz):
        raise ValueError("source-space vertex count differs from step3_source_coords_mm.npy")
    return xyz, np.concatenate(tris)


def labelling(geometry, kind: str = "volumetric", atlas: str = "allen26", surface_run_dir=None) -> dict:
    """Region labels for a location posterior's candidate positions.

    ``kind="volumetric"``: the geometry's dense grid, every position labelled by its nearest atlas voxel.
    ``kind="surface"``: the anatomical surface source space of ``surface_run_dir``, labelled within the surface
    categories only (:func:`restricted_parcel_labels`).

    Returns positions_mm, tris (surface only), parcel_ids, parcel_names, regions (:class:`RegionMap`), inside (the
    position's own nearest label is a labelled, in-brain voxel), volume (mm^3 per parcel for volumetric, vertex count
    for surface), labeler (positions -> (inside, region index)) and n_positions_per_parcel. As probability-atlas
    Phase 2's ``labelling_setup``.
    """
    import json

    from ..validation.roi_certainty import ParcelMap

    if kind == "volumetric":
        pos, tris = geometry.positions_mm, None
    elif kind == "surface":
        if surface_run_dir is None:
            raise ValueError("surface labelling needs surface_run_dir")
        pos, tris = surface_positions(surface_run_dir)
    else:
        raise ValueError(f"unknown labelling kind {kind!r}")
    pmap = ParcelMap(pos, atlas=atlas, electrode_pos_mm=geometry.electrode_pos_mm)
    if kind == "surface":
        mapping = json.loads((Path(__import__("source_localization").__file__).parent
                              / f"data/atlas/allen/roi_mapping_{atlas}.json").read_text())
        allowed = sorted({i for c in SURFACE_CATEGORIES for i in mapping["categories"][c]})
        labels = restricted_parcel_labels(pmap, allowed)
    else:
        labels = pmap.labels
    ids = np.unique(labels)
    names = [pmap.parcel_names[pmap.index_of(i)] for i in ids]
    region_idx = np.searchsorted(ids, labels)
    inside = pmap.inside & (labels == pmap.labels)

    def labeler(p):
        pm = ParcelMap(p, atlas=atlas)
        ok = np.isin(pm.labels, ids)
        return pm.inside & ok, np.where(ok, np.searchsorted(ids, pm.labels), -1)

    count = np.bincount(region_idx, minlength=len(ids))
    volume = count * geometry.spacing_mm ** 3 if kind == "volumetric" else count.astype(float)
    return {"positions_mm": pos, "tris": tris, "parcel_ids": ids, "parcel_names": names,
            "regions": RegionMap(region_idx, names), "inside": inside, "volume": volume, "labeler": labeler,
            "n_positions_per_parcel": count, "kind": kind, "atlas": atlas}
