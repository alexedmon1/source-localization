"""EEG channel order must match the forward's row order before the inverse.

The forward is built from the electrode-registration ``info`` and its rows
follow that channel order. The inverse multiplies the operator by
``epochs.get_data()``, which follows the recording's channel order. Until the
EEG step reordered channels, a ``.set`` stored in a different order (the
eeg-preprocess output stores E23, E22, E30, ...) silently paired every
leadfield row with the wrong channel's data.
"""

from __future__ import annotations

import numpy as np
import mne
import pytest

from source_localization.steps.eeg_data import align_channels_to_info
from source_localization.steps.inverse_solution import _check_channel_alignment


def _info(names, sfreq=100.0):
    info = mne.create_info(list(names), sfreq, 'eeg')
    rng = np.random.default_rng(0)
    pos = {c: rng.normal(size=3) * 0.01 + [0, 0, 0.05] for c in names}
    info.set_montage(mne.channels.make_dig_montage(ch_pos=pos, coord_frame='head'))
    return info


def _epochs(names, sfreq=100.0, n_epochs=2, n_times=20, seed=1):
    rng = np.random.default_rng(seed)
    data = rng.normal(size=(n_epochs, len(names), n_times))
    return mne.EpochsArray(data, mne.create_info(list(names), sfreq, 'eeg'),
                           verbose=False)


REG = ['E1', 'E2', 'E3', 'E4', 'E5']


def test_shuffled_recording_is_reordered_to_registration_order():
    info = _info(REG)
    shuffled = ['E4', 'E1', 'E5', 'E2', 'E3']
    epochs = _epochs(shuffled)
    before = epochs.get_data().copy()

    align_channels_to_info(epochs, info)

    assert epochs.ch_names == REG
    # Each channel's data travels with its name.
    after = epochs.get_data()
    for new_idx, name in enumerate(REG):
        old_idx = shuffled.index(name)
        np.testing.assert_array_equal(after[:, new_idx], before[:, old_idx])


def test_already_aligned_recording_is_untouched():
    info = _info(REG)
    epochs = _epochs(REG)
    before = epochs.get_data().copy()
    align_channels_to_info(epochs, info)
    assert epochs.ch_names == REG
    np.testing.assert_array_equal(epochs.get_data(), before)


def test_missing_registered_electrode_raises():
    info = _info(REG)
    epochs = _epochs(['E1', 'E2', 'E3', 'E4'])
    with pytest.raises(ValueError, match="lacks 1 registered"):
        align_channels_to_info(epochs, info)


def test_unregistered_channels_are_dropped(capsys):
    info = _info(REG)
    epochs = _epochs(['EMG', 'E3', 'E1', 'E2', 'E5', 'E4'])
    align_channels_to_info(epochs, info)
    assert epochs.ch_names == REG
    assert 'Dropping 1 channel' in capsys.readouterr().out


def test_inverse_refuses_misordered_data():
    """Defence in depth: the inverse checks even if the EEG step was bypassed."""
    info = _info(REG)
    sphere = mne.make_sphere_model(r0=(0, 0, 0), head_radius=0.09, verbose=False)
    pos = dict(rr=np.array([[0, 0, 0.02], [0, 0, 0.04]]),
               nn=np.tile([0, 0, 1.], (2, 1)))
    src = mne.setup_volume_source_space(pos=pos, verbose=False)
    fwd = mne.make_forward_solution(info, None, src, sphere, meg=False, eeg=True,
                                    mindist=0, verbose=False)

    _check_channel_alignment(fwd, _epochs(REG))  # aligned: no error

    with pytest.raises(ValueError, match="different order"):
        _check_channel_alignment(fwd, _epochs(['E2', 'E1', 'E3', 'E4', 'E5']))
    with pytest.raises(ValueError, match="missing from data"):
        _check_channel_alignment(fwd, _epochs(['E1', 'E2', 'E3', 'E4', 'E9']))
