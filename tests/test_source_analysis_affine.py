"""source_analysis must read label volumes through utils.atlas, not scale by hand.

AtlasLookup, ROIVisualizer and CorticalSourceSpace used to divide
``affine[:3, :3]`` by 10 unconditionally. That left the translation at 10x
and shrank files that are already in true units (every current Allen label
file, Antwerp's Labels file) a further 10x. The failure is silent because the
affine stays well-formed; coordinates simply land outside the volume.
"""

from __future__ import annotations

import warnings
from pathlib import Path

import nibabel as nib
import numpy as np
import pytest

from source_localization.source_analysis.atlas_lookup import AtlasLookup
from source_localization.source_analysis.visualization_roi import ROIVisualizer
from source_localization.utils.atlas import get_true_affine

DATA = Path(__file__).resolve().parents[1] / 'src' / 'source_localization' / 'data' / 'atlas'
TRUE_UNITS = DATA / 'allen' / 'allen_labels.nii.gz'
INFLATED = DATA / 'coarse_parcellation' / 'coarse_22roi_atlas.nii'


@pytest.mark.parametrize('path', [TRUE_UNITS, INFLATED], ids=['true-units', 'inflated'])
@pytest.mark.parametrize('cls', [AtlasLookup, ROIVisualizer])
def test_affine_matches_detected_convention(cls, path):
    expected = get_true_affine(nib.load(path))
    got = cls(str(path)).affine
    np.testing.assert_allclose(got, expected)
    # Translation must be corrected together with the zooms.
    np.testing.assert_allclose(got[:3, 3], expected[:3, 3])


def test_true_unit_file_is_not_shrunk():
    nii = nib.load(TRUE_UNITS)
    got = AtlasLookup(str(TRUE_UNITS)).affine
    np.testing.assert_allclose(got, nii.affine)


def test_mm_lookup_lands_inside_the_volume():
    """A coordinate near the brain centre resolves to a labeled voxel."""
    lookup = AtlasLookup(str(TRUE_UNITS))
    nii = nib.load(TRUE_UNITS)
    data = np.asarray(nii.dataobj)
    centre_vox = np.array(np.nonzero(data)).mean(axis=1)
    centre_mm = nib.affines.apply_affine(get_true_affine(nii), centre_vox)
    vox = np.round(nib.affines.apply_affine(lookup.affine_inv, centre_mm)).astype(int)
    assert all(0 <= v < s for v, s in zip(vox, data.shape))


def test_legacy_kwarg_is_ignored_with_a_warning():
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter('always')
        lookup = AtlasLookup(str(TRUE_UNITS), apply_10x_correction=True)
    assert any(issubclass(x.category, DeprecationWarning) for x in w)
    np.testing.assert_allclose(lookup.affine, get_true_affine(nib.load(TRUE_UNITS)))
