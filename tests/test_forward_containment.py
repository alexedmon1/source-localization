"""Sources MNE drops from the forward must not shift downstream indexing.

``mne.make_forward_solution`` silently discards sources outside the inner
skull and records the survivors only in ``fwd['src'][i]['vertno']``. The
inverse export and ROI extraction used to reconcile the resulting length
mismatch by front-slicing ``source_coords_mm[:n]``, which pairs every STC row
after the first dropped source with the wrong coordinate and parcel. The
forward step now re-emits per-source arrays restricted to the kept sources,
and the downstream steps refuse a mismatch instead of hiding it.
"""

from __future__ import annotations

import numpy as np
import mne
import pytest

from source_localization.steps import forward_solution, roi_extraction
from source_localization.steps.forward_solution import kept_source_indices


def _info(names=('E1', 'E2', 'E3', 'E4')):
    info = mne.create_info(list(names), 100.0, 'eeg')
    pos = {c: np.array([0.1 * np.cos(i), 0.1 * np.sin(i), 0.02])
           for i, c in enumerate(names)}
    info.set_montage(mne.channels.make_dig_montage(ch_pos=pos, coord_frame='head'))
    return info


def _volume_src(rr_m):
    rr = np.asarray(rr_m, dtype=float)
    return mne.setup_volume_source_space(
        pos=dict(rr=rr, nn=np.tile([0, 0, 1.], (len(rr), 1))), verbose=False
    )


def test_mne_drops_out_of_skull_sources_without_warning():
    """The premise: MNE drops silently and only vertno records who survived."""
    sphere = mne.make_sphere_model(r0=(0, 0, 0), head_radius=0.09, verbose=False)
    src = _volume_src([[0, 0, 0.02], [0, 0, 0.05], [0, 0, 0.20], [0, 0, 0.03]])
    fwd = mne.make_forward_solution(_info(), None, src, sphere, meg=False,
                                    eeg=True, mindist=0, verbose=False)
    assert fwd['nsource'] == 3
    np.testing.assert_array_equal(fwd['src'][0]['vertno'], [0, 1, 3])


def test_kept_source_indices_maps_survivors_to_original_rows():
    sphere = mne.make_sphere_model(r0=(0, 0, 0), head_radius=0.09, verbose=False)
    src = _volume_src([[0, 0, 0.02], [0, 0, 0.05], [0, 0, 0.20], [0, 0, 0.03]])
    fwd = mne.make_forward_solution(_info(), None, src, sphere, meg=False,
                                    eeg=True, mindist=0, verbose=False)
    np.testing.assert_array_equal(kept_source_indices(src, fwd['src']), [0, 1, 3])


def test_kept_source_indices_offsets_across_hemispheres():
    """A two-entry source space concatenates lh then rh; offsets must follow."""
    lh = {'vertno': np.array([0, 1, 2])}
    rh = {'vertno': np.array([0, 1, 2, 3])}
    lh_after = {'vertno': np.array([0, 2])}
    rh_after = {'vertno': np.array([1, 3])}
    np.testing.assert_array_equal(
        kept_source_indices([lh, rh], [lh_after, rh_after]), [0, 2, 4, 6]
    )


def test_forward_step_reemits_per_source_arrays_restricted_to_kept(tmp_path):
    rr = np.array([[0, 0, 0.02], [0, 0, 0.05], [0, 0, 0.20], [0, 0, 0.03]])
    src = _volume_src(rr)
    src[0]['roi_assignments'] = np.array([11, 12, 13, 14])
    coords_mm = rr * 1000.0

    config = {
        'pipeline': {'bem_type': 'sphere', 'source_type': 'cartesian'},
        'forward': {'eeg': True, 'meg': False, 'mindist': 0.0},
        'outputs': {'save_intermediate': False, 'dir': str(tmp_path)},
    }
    bem = mne.make_sphere_model(r0=(0, 0, 0), head_radius=0.09, verbose=False)
    out = forward_solution.run(config, {
        'info': _info(), 'src': src, 'bem': bem,
        'source_coords_mm': coords_mm, 'n_sources': 4,
    })

    assert out['fwd']['nsource'] == 3
    assert out['n_sources'] == 3
    np.testing.assert_array_equal(out['kept_source_indices'], [0, 1, 3])
    # Row 2 of the output is the ORIGINAL row 3, not the original row 2.
    np.testing.assert_array_equal(out['source_coords_mm'], coords_mm[[0, 1, 3]])
    np.testing.assert_array_equal(out['roi_assignments'], [11, 12, 14])


def test_roi_extraction_refuses_length_mismatch(tmp_path):
    """Front-slicing is gone: a mismatch is an error, not a warning."""
    stc = mne.VolSourceEstimate(np.zeros((3, 5)), [np.arange(3)], 0.0, 0.01)
    config = {
        'inputs': {'brain_labels': 'data/atlas/allen/allen_labels.nii.gz',
                   'roi_mapping': 'data/atlas/allen/roi_mapping.json'},
        'roi': {'use_proximity': False},
        'outputs': {'save_intermediate': False, 'dir': str(tmp_path)},
    }
    with pytest.raises(ValueError, match="Source count mismatch"):
        roi_extraction.run(config, {
            'stc_magnitude': stc, 'stc_signed': stc,
            'source_coords_mm': np.zeros((4, 3)), 'n_sources': 4, 'src': None,
        })
