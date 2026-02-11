#!/usr/bin/env python3
"""
Export Source ROI Data to EEGLAB .set Format

Exports source-localized ROI time series as an EEGLAB-compatible .set file
where each ROI becomes a "channel" with the ROI name as the channel label.

Uses MNE's built-in export functionality for proper EEGLAB compatibility.

**Created:** 2025-12-05
**Last Updated:** 2025-12-05
"""

import numpy as np
import mne
from pathlib import Path


def export_roi_to_set(roi_data, sfreq, output_path, subject_id='source_localized'):
    """
    Export ROI time series to EEGLAB .set format.

    Each ROI becomes a channel with the ROI name as the channel label.
    This allows source-level data to be analyzed with standard EEG tools.

    Parameters
    ----------
    roi_data : dict
        Dictionary mapping ROI names to time series arrays.
        Each value should be shape (n_times,) for continuous or (n_epochs, n_times) for epoched
    sfreq : float
        Sampling frequency in Hz
    output_path : str or Path
        Output path for .set file
    subject_id : str
        Subject identifier for EEG.setname

    Returns
    -------
    Path
        Path to saved .set file
    """
    output_path = Path(output_path)
    if not output_path.suffix == '.set':
        output_path = output_path.with_suffix('.set')

    # Get ROI names and data
    roi_names = list(roi_data.keys())
    n_rois = len(roi_names)

    # Stack ROI data
    first_roi = roi_data[roi_names[0]]

    if first_roi.ndim == 1:
        # Continuous data: (n_rois, n_times)
        n_times = first_roi.shape[0]
        data = np.zeros((n_rois, n_times), dtype=np.float64)
        for i, name in enumerate(roi_names):
            data[i, :] = roi_data[name]
        is_epoched = False
    else:
        # Epoched data: (n_epochs, n_rois, n_times)
        n_epochs, n_times = first_roi.shape
        data = np.zeros((n_epochs, n_rois, n_times), dtype=np.float64)
        for i, name in enumerate(roi_names):
            data[:, i, :] = roi_data[name]
        is_epoched = True

    # Create MNE Info object
    # Use 'misc' channel type since these aren't real EEG channels
    info = mne.create_info(ch_names=roi_names, sfreq=sfreq, ch_types=['eeg'] * n_rois)

    if is_epoched:
        # Create Epochs object
        events = np.array([[i * n_times, 0, 1] for i in range(n_epochs)])
        event_id = {'epoch': 1}
        epochs = mne.EpochsArray(data, info, events=events, event_id=event_id, tmin=0.0)

        # Export to EEGLAB format
        epochs.export(str(output_path), fmt='eeglab', overwrite=True)
        print(f"  Exported {n_rois} ROIs × {n_epochs} epochs × {n_times} samples to: {output_path}")
    else:
        # Create Raw object
        raw = mne.io.RawArray(data, info, verbose=False)

        # Export to EEGLAB format
        raw.export(str(output_path), fmt='eeglab', overwrite=True)
        print(f"  Exported {n_rois} ROIs × {n_times} samples to: {output_path}")

    return output_path


def export_source_to_set(stc_data, source_coords, sfreq, output_path,
                         subject_id='source_localized', max_sources=None):
    """
    Export raw source estimates (all dipoles) to EEGLAB .set format.

    Each source dipole becomes a channel. Use with caution for large
    source spaces - consider using export_roi_to_set instead.

    Parameters
    ----------
    stc_data : ndarray
        Source time series, shape (n_sources, n_times)
    source_coords : ndarray
        Source coordinates in mm, shape (n_sources, 3)
    sfreq : float
        Sampling frequency in Hz
    output_path : str or Path
        Output path for .set file
    subject_id : str
        Subject identifier
    max_sources : int, optional
        Maximum number of sources to export (for memory management)

    Returns
    -------
    Path
        Path to saved .set file
    """
    output_path = Path(output_path)
    if not output_path.suffix == '.set':
        output_path = output_path.with_suffix('.set')

    n_sources, n_times = stc_data.shape

    if max_sources and n_sources > max_sources:
        print(f"  Warning: Limiting export to {max_sources} sources (of {n_sources})")
        # Select sources with highest RMS power
        rms = np.sqrt(np.mean(stc_data**2, axis=1))
        top_idx = np.argsort(rms)[-max_sources:]
        stc_data = stc_data[top_idx, :]
        source_coords = source_coords[top_idx, :]
        n_sources = max_sources

    # Create channel names
    ch_names = [f'src_{i+1:04d}' for i in range(n_sources)]

    # Create MNE Info object
    info = mne.create_info(ch_names=ch_names, sfreq=sfreq, ch_types=['eeg'] * n_sources)

    # Add source coordinates to channel locations
    for idx in range(n_sources):
        info['chs'][idx]['loc'][:3] = source_coords[idx] / 1000.0  # Convert mm to m

    # Create Raw object
    raw = mne.io.RawArray(stc_data.astype(np.float64), info, verbose=False)

    # Export to EEGLAB format
    raw.export(str(output_path), fmt='eeglab', overwrite=True)

    print(f"  Exported {n_sources} source dipoles × {n_times} samples to: {output_path}")

    return output_path
