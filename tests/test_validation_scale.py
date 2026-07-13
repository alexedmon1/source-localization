"""Regression guard for the atlas voxel-affine convention (see corrected_atlas_affine).

A too-broad `.gitignore` had excluded this whole module from version control, and a
convention mismatch (the Allen-32 label volume was re-saved in true-mm while the code
expected the 10x convention) silently collapsed every ROI centroid to the brain centre,
tanking ROI-classification accuracy from ~46% to ~5%. These fast, deterministic tests
fail loudly if that regresses.
"""
from pathlib import Path

import numpy as np
import nibabel as nib

import source_localization
from source_localization.validation.utils import (
    corrected_atlas_affine,
    load_atlas_roi_centroids,
)

ATLAS = Path(source_localization.__file__).parent / "data" / "atlas"


def _voxsize(affine):
    return np.abs(np.diag(affine)[:3])


def test_x10_convention_file_is_corrected():
    """A 10x-convention volume (~2 mm voxels) must be divided to true mm (~0.2 mm)."""
    nii = nib.load(str(ATLAS / "Atlas_3DRois.nii"))
    assert _voxsize(nii.affine).max() > 1.0, "fixture should be in the 10x convention"
    assert _voxsize(corrected_atlas_affine(nii)).max() < 0.5


def test_true_mm_file_is_not_double_corrected():
    """An already-true-mm label file (~0.2 mm) must be left unchanged (the bug's cause)."""
    nii = nib.load(str(ATLAS / "allen" / "allen_labels.nii.gz"))
    assert _voxsize(nii.affine).max() < 1.0, "fixture should already be in true mm"
    np.testing.assert_allclose(corrected_atlas_affine(nii), nii.affine)


def test_allen32_centroids_span_a_plausible_brain():
    """Allen-32 ROI centroids must span a real mouse brain, not collapse to the centre."""
    centroids, _ = load_atlas_roi_centroids(
        str(ATLAS / "allen" / "allen_labels.nii.gz"),
        str(ATLAS / "allen" / "roi_mapping.json"),
    )
    span = np.ptp(np.array(list(centroids.values())), axis=0)
    assert span.max() > 6.0, f"centroids collapsed (span={np.round(span, 2)} mm)"
