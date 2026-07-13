"""Atlas bundles: a matched (template, labels, roi-mapping) triplet for verification.

A *bundle* is the single source of truth for a validation/verification run. Loading one
runs the consistency checks that make source-localization verification atlas-agnostic and
that would have caught the 2026-07 affine-convention collapse *up front* — instead of the
pipeline silently returning garbage ROI-classification accuracy.

The invariant a correct bundle must satisfy is simple: **the template, the labels, and the
source space share one voxel grid and one coordinate frame.** `load_bundle()` asserts it.

Usage
-----
>>> from source_localization.validation.atlas_bundle import load_bundle
>>> b = load_bundle(
...     template="data/atlas/Atlas_3DRois.nii",
...     labels="data/atlas/allen/allen_labels.nii.gz",
...     roi_mapping="data/atlas/allen/roi_mapping.json",
...     name="Allen-32",
... )
>>> b.n_rois
32
>>> b.centroids_mm()[9]           # Thalamus_L centroid in mm
array([...])

Adding a new atlas is then a *data* task: drop a matched triplet and load it. See
``docs/adding_an_atlas.md``.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Optional, Union

import numpy as np
import nibabel as nib

import source_localization
from .utils import corrected_atlas_affine

# A mouse brain is ~1 cm rostro-caudal; ROI centroids collapsed to the brain centre by a
# convention mismatch span << this. Bundles must clear it.
MIN_BRAIN_SPAN_MM = 6.0
# After convention correction every atlas volume in this pipeline is ~0.2 mm true-mm.
TRUE_MM_VOXEL_MAX = 1.0


class BundleConsistencyError(ValueError):
    """Raised when a bundle's template / labels / mapping are not mutually consistent."""


PathLike = Union[str, Path]


def _resolve(p: PathLike) -> Path:
    """Resolve a path, allowing paths relative to the packaged atlas data."""
    p = Path(p)
    if p.is_absolute() and p.exists():
        return p
    pkg = Path(source_localization.__file__).parent
    for base in (Path.cwd(), pkg):
        cand = base / p
        if cand.exists():
            return cand
    if p.exists():
        return p
    raise FileNotFoundError(f"Bundle file not found: {p}")


@dataclass
class AtlasBundle:
    """A validated, matched atlas triplet in one true-mm coordinate frame."""

    name: str
    template_path: Path
    labels_path: Path
    roi_mapping_path: Path
    labels_data: np.ndarray                 # integer ROI labels
    affine: np.ndarray                      # convention-corrected voxel->mm (shared frame)
    roi_names: Dict[int, str] = field(default_factory=dict)

    @property
    def roi_ids(self) -> np.ndarray:
        ids = np.unique(self.labels_data)
        return ids[ids > 0].astype(int)

    @property
    def n_rois(self) -> int:
        return int(self.roi_ids.size)

    def centroids_mm(self) -> Dict[int, np.ndarray]:
        """ROI id -> centroid in mm, in the bundle's shared frame."""
        out: Dict[int, np.ndarray] = {}
        for rid in self.roi_ids:
            vox = np.argwhere(self.labels_data == rid).mean(axis=0)
            out[int(rid)] = nib.affines.apply_affine(self.affine, vox)
        return out

    def summary(self) -> str:
        span = np.ptp(np.array(list(self.centroids_mm().values())), axis=0)
        vox = np.abs(np.diag(self.affine)[:3])
        return (
            f"AtlasBundle('{self.name}'): {self.n_rois} ROIs | "
            f"voxel {np.round(vox, 3)} mm | centroid span {np.round(span, 1)} mm"
        )


def load_bundle(
    template: PathLike,
    labels: PathLike,
    roi_mapping: Optional[PathLike] = None,
    name: str = "bundle",
    *,
    strict: bool = True,
) -> AtlasBundle:
    """Load a matched atlas triplet and assert its consistency.

    Parameters
    ----------
    template : path
        Anatomical volume defining the geometry and the working affine convention.
    labels : path
        Integer ROI parcellation on the SAME voxel grid as ``template``.
    roi_mapping : path, optional
        JSON mapping ``{"rois": {"<id>": {"name": ...}}}`` (id -> name).
    name : str
        Human label for the bundle.
    strict : bool
        If True (default), label/mapping coverage mismatches raise; else they warn.

    Returns
    -------
    AtlasBundle

    Raises
    ------
    BundleConsistencyError
        If template and labels are on different grids, resolve to different frames, the
        labels and mapping disagree, or the ROI centroids collapse to an implausibly small
        extent (the signature of a voxel-affine convention mismatch).
    """
    tpl_path, lab_path = _resolve(template), _resolve(labels)
    tpl = nib.load(str(tpl_path))
    lab = nib.load(str(lab_path))
    lab_data = np.asanyarray(lab.dataobj)
    lab_int = np.round(lab_data).astype(int)

    # (1) same voxel grid
    if tpl.shape[:3] != lab.shape[:3]:
        raise BundleConsistencyError(
            f"[{name}] template shape {tpl.shape[:3]} != labels shape {lab.shape[:3]}; "
            f"template and labels must share one voxel grid."
        )

    # (2) one coordinate frame after convention correction
    aff_tpl = corrected_atlas_affine(tpl)
    aff_lab = corrected_atlas_affine(lab)
    if not np.allclose(aff_tpl, aff_lab, atol=1e-3):
        raise BundleConsistencyError(
            f"[{name}] template and labels resolve to different affines after convention "
            f"correction (template voxel {np.round(np.abs(np.diag(aff_tpl)[:3]),3)} vs "
            f"labels {np.round(np.abs(np.diag(aff_lab)[:3]),3)} mm). One file is likely "
            f"stored in the wrong (10x vs true-mm) convention. See corrected_atlas_affine()."
        )
    if np.abs(np.diag(aff_lab)[:3]).max() > TRUE_MM_VOXEL_MAX:
        raise BundleConsistencyError(
            f"[{name}] corrected voxel size {np.round(np.abs(np.diag(aff_lab)[:3]),3)} mm "
            f"exceeds {TRUE_MM_VOXEL_MAX} mm — convention resolution failed."
        )

    # ROI ids present in the label volume
    label_ids = set(int(i) for i in np.unique(lab_int) if i > 0)
    if not label_ids:
        raise BundleConsistencyError(f"[{name}] labels volume has no non-zero ROIs.")

    # (3) label <-> mapping coverage
    roi_names: Dict[int, str] = {}
    if roi_mapping is not None:
        map_path = _resolve(roi_mapping)
        with open(map_path) as f:
            mp = json.load(f)
        rois = mp.get("rois", mp)
        map_ids = set()
        for k, v in rois.items():
            try:
                rid = int(k)
            except (ValueError, TypeError):
                continue
            map_ids.add(rid)
            roi_names[rid] = v.get("name", f"ROI_{rid}") if isinstance(v, dict) else str(v)
        orphans_lbl = label_ids - map_ids   # labels with no name
        orphans_map = map_ids - label_ids - {0}
        if orphans_lbl:
            msg = (f"[{name}] {len(orphans_lbl)} label id(s) have no roi_mapping entry: "
                   f"{sorted(orphans_lbl)[:8]}")
            if strict:
                raise BundleConsistencyError(msg)
            print("WARNING:", msg)
        if orphans_map:
            print(f"WARNING: [{name}] {len(orphans_map)} mapping id(s) absent from labels: "
                  f"{sorted(orphans_map)[:8]}")
    else:
        map_path = None
        roi_names = {rid: f"ROI_{rid}" for rid in label_ids}

    bundle = AtlasBundle(
        name=name,
        template_path=tpl_path,
        labels_path=lab_path,
        roi_mapping_path=map_path if roi_mapping is not None else None,
        labels_data=lab_int,
        affine=aff_lab,
        roi_names=roi_names,
    )

    # (4) centroids must span a plausible brain (the collapse guard)
    span = np.ptp(np.array(list(bundle.centroids_mm().values())), axis=0)
    if float(span.max()) < MIN_BRAIN_SPAN_MM:
        raise BundleConsistencyError(
            f"[{name}] ROI centroids span only {np.round(span,2)} mm (< {MIN_BRAIN_SPAN_MM} "
            f"mm) — they have collapsed toward the brain centre, the signature of a "
            f"voxel-size/affine convention mismatch. Check that template and labels share "
            f"the same convention."
        )
    return bundle


def verify_bundle(template: PathLike, labels: PathLike,
                  roi_mapping: Optional[PathLike] = None, name: str = "bundle") -> bool:
    """Load and print a one-line verification of a bundle. Returns True if consistent."""
    b = load_bundle(template, labels, roi_mapping, name=name)
    print("OK  " + b.summary())
    return True
