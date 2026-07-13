"""Consistency checks for atlas bundles (the standardized verification entry point).

These guard the invariant that a bundle's template, labels, and mapping share one voxel
grid and one coordinate frame — the invariant whose violation silently broke ROI accuracy
in 2026-07.
"""
from pathlib import Path

import numpy as np
import nibabel as nib
import pytest

import source_localization
from source_localization.validation.atlas_bundle import (
    load_bundle,
    BundleConsistencyError,
)

A = Path(source_localization.__file__).parent / "data" / "atlas"
TPL = A / "Atlas_3DRois.nii"
LAB = A / "allen" / "allen_labels.nii.gz"
MAP = A / "allen" / "roi_mapping.json"


def test_allen32_bundle_loads_and_is_consistent():
    b = load_bundle(str(TPL), str(LAB), str(MAP), name="Allen-32")
    assert b.n_rois == 32
    span = np.ptp(np.array(list(b.centroids_mm().values())), axis=0)
    assert span.max() > 6.0


def test_grid_mismatch_raises(tmp_path):
    # a labels volume on a different voxel grid must be rejected
    lab = nib.load(str(LAB))
    smaller = nib.Nifti1Image(np.asanyarray(lab.dataobj)[:32], lab.affine)
    p = tmp_path / "wrong_grid.nii.gz"
    nib.save(smaller, p)
    with pytest.raises(BundleConsistencyError):
        load_bundle(str(TPL), str(p), str(MAP), name="WrongGrid")


def test_convention_mismatch_raises(tmp_path):
    # labels re-saved in the wrong (double-divided) convention must be rejected
    lab = nib.load(str(LAB))
    bad = lab.affine.copy()
    bad[:3, :3] /= 10
    bad[:3, 3] /= 10
    p = tmp_path / "wrong_convention.nii.gz"
    nib.save(nib.Nifti1Image(np.asanyarray(lab.dataobj), bad), p)
    with pytest.raises(BundleConsistencyError):
        load_bundle(str(TPL), str(p), str(MAP), name="WrongConvention")
