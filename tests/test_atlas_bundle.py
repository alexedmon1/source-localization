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


# ---------------------------------------------------------------------------
# Voxel-size convention detection
#
# The bundled atlas files do NOT share a header convention: most carry voxel
# sizes 10x too large, but Atlas_3DRoisLeftRight.Labels.nii is already in true
# units. Correcting an already-corrected file is silent — the affine stays
# well-formed and coordinates simply land 10x away — so these pin the detection.
# ---------------------------------------------------------------------------

def _atlas_dir():
    import source_localization as sl
    return Path(sl.__file__).parent / 'data' / 'atlas'


def test_inflated_and_true_files_are_distinguished():
    import nibabel as nib
    from source_localization.utils.atlas import header_is_inflated
    d = _atlas_dir()
    assert header_is_inflated(nib.load(d / 'Atlas_3DRois.nii')) is True
    assert header_is_inflated(nib.load(d / 'Atlas_3DRois_brain.nii.gz')) is True
    assert header_is_inflated(
        nib.load(d / 'Atlas_3DRoisLeftRight.Labels.nii')) is False


def test_both_conventions_yield_the_same_geometry():
    """The corrected brain volume and the native Labels affine must agree."""
    import numpy as np, nibabel as nib
    from source_localization.utils.atlas import get_true_affine
    d = _atlas_dir()
    a = get_true_affine(nib.load(d / 'Atlas_3DRois_brain.nii.gz'))
    b = get_true_affine(nib.load(d / 'Atlas_3DRoisLeftRight.Labels.nii'))
    assert np.allclose(a, b, atol=1e-3)


def test_get_true_affine_is_idempotent_on_true_units():
    """Calling it on an already-corrected file must not scale it again."""
    import numpy as np, nibabel as nib
    from source_localization.utils.atlas import get_true_affine
    nii = nib.load(_atlas_dir() / 'Atlas_3DRoisLeftRight.Labels.nii')
    assert np.allclose(get_true_affine(nii), nii.affine, atol=1e-9)


def test_translation_is_scaled_with_the_voxel_sizes():
    """Scaling only the 3x3 block leaves the origin 10x out."""
    import numpy as np, nibabel as nib
    from source_localization.utils.atlas import get_true_affine
    nii = nib.load(_atlas_dir() / 'Atlas_3DRois_brain.nii.gz')
    a = get_true_affine(nii)
    assert np.allclose(a[:3, 3], nii.affine[:3, 3] * 0.1, atol=1e-6)
