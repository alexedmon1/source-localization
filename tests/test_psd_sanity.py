#!/usr/bin/env python
"""
PSD Sanity Check Test

Verifies that source localization preserves 1/f spectral characteristics.
Requires a real EEG .set file as input.

Usage:
    pytest tests/test_psd_sanity.py --eeg-file /path/to/file.set -v

Or run directly:
    python tests/test_psd_sanity.py /path/to/file.set
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy import signal
from pathlib import Path
import pickle
import sys

try:
    import pytest
    HAS_PYTEST = True
except ImportError:
    HAS_PYTEST = False


def compute_1f_slope(freqs, psd, f_low=2, f_high=30):
    """Compute 1/f slope (beta) from PSD."""
    mask = (freqs >= f_low) & (freqs <= f_high)
    if psd[mask].min() <= 0:
        return np.nan
    return -np.polyfit(np.log10(freqs[mask]), np.log10(psd[mask]), 1)[0]


def run_psd_sanity_check(eeg_file, results_dir=None, output_plot=None):
    """
    Run PSD sanity check on EEG file.

    Parameters
    ----------
    eeg_file : str or Path
        Path to EEG .set file
    results_dir : str or Path, optional
        Directory containing forward solution (default: test_results)
    output_plot : str or Path, optional
        Output path for sanity check plot

    Returns
    -------
    dict
        Results including slopes and pass/fail status
    """
    import mne

    eeg_file = Path(eeg_file)
    if results_dir is None:
        results_dir = Path(__file__).parent.parent / 'test_results'
    results_dir = Path(results_dir)

    if output_plot is None:
        output_plot = results_dir / 'psd_sanity_check.png'

    print(f"Loading EEG: {eeg_file}")
    print(f"Results dir: {results_dir}")

    # Load forward solution
    fwd_file = results_dir / 'data' / 'step4_forward.pkl'
    if not fwd_file.exists():
        raise FileNotFoundError(f"Forward solution not found: {fwd_file}")

    with open(fwd_file, 'rb') as f:
        fwd = pickle.load(f)

    # Load epochs
    epochs = mne.io.read_epochs_eeglab(str(eeg_file), verbose=False)
    eeg_data = epochs.get_data()  # (n_epochs, n_channels, n_times)
    sfreq = epochs.info['sfreq']
    n_epochs, n_channels, n_times = eeg_data.shape

    print(f"Epochs: {n_epochs} trials × {n_channels} channels × {n_times} samples")
    print(f"Sfreq: {sfreq} Hz")

    # Build inverse operator
    G = fwd['sol']['data']
    n_ch, n_dipoles = G.shape
    snr, lambda2 = 3.0, 1.0 / 9.0

    GGT = G @ G.T
    GGT_reg = GGT + lambda2 * np.trace(GGT) / n_ch * np.eye(n_ch)
    W = G.T @ np.linalg.inv(GGT_reg)

    # dSPM normalization factor
    resolution_diagonal = np.sum(W * G.T, axis=1)

    # PSD parameters
    nperseg = n_times
    noverlap = 0
    nfft = max(256, n_times)

    # Compute trial-by-trial PSDs
    ch_idx = n_channels // 2  # Middle channel

    sensor_psds = []
    source_psds = []

    for epoch_idx in range(n_epochs):
        # Sensor
        epoch_sensor = eeg_data[epoch_idx, ch_idx, :]
        freqs, psd = signal.welch(epoch_sensor, fs=sfreq, nperseg=nperseg,
                                   noverlap=noverlap, nfft=nfft)
        sensor_psds.append(psd)

        # Source (signed dSPM)
        epoch_eeg = eeg_data[epoch_idx, :, :]
        source_activity = W @ epoch_eeg
        source_signed = source_activity / (resolution_diagonal[:, np.newaxis] + 1e-10)
        dipole_ts = source_signed[0, :]

        _, psd = signal.welch(dipole_ts, fs=sfreq, nperseg=nperseg,
                               noverlap=noverlap, nfft=nfft)
        source_psds.append(psd)

    sensor_psds = np.array(sensor_psds)
    source_psds = np.array(source_psds)
    sensor_psd_mean = sensor_psds.mean(axis=0)
    source_psd_mean = source_psds.mean(axis=0)
    sensor_psd_sem = sensor_psds.std(axis=0) / np.sqrt(n_epochs)
    source_psd_sem = source_psds.std(axis=0) / np.sqrt(n_epochs)

    # Frequency mask
    freq_mask = (freqs >= 1) & (freqs <= 55)
    freqs_plot = freqs[freq_mask]

    # Compute slopes
    slope_sensor = compute_1f_slope(freqs_plot, sensor_psd_mean[freq_mask])
    slope_source = compute_1f_slope(freqs_plot, source_psd_mean[freq_mask])

    print(f"\n1/f Slopes (2-30 Hz):")
    print(f"  Sensor: β = {slope_sensor:.2f}")
    print(f"  Source: β = {slope_source:.2f}")

    # Sanity check criteria
    sensor_ok = 0.5 < slope_sensor < 3.0
    source_ok = 0.5 < slope_source < 3.0
    slopes_similar = abs(slope_sensor - slope_source) < 1.5

    passed = sensor_ok and source_ok and slopes_similar

    print(f"\nSanity Check:")
    print(f"  Sensor slope in range [0.5, 3.0]: {'PASS' if sensor_ok else 'FAIL'}")
    print(f"  Source slope in range [0.5, 3.0]: {'PASS' if source_ok else 'FAIL'}")
    print(f"  Slopes within 1.5 of each other: {'PASS' if slopes_similar else 'FAIL'}")
    print(f"  Overall: {'PASS' if passed else 'FAIL'}")

    # Get time series for plotting (concatenate a few epochs for ~10s strip)
    n_epochs_to_show = min(5, n_epochs)  # Show up to 5 epochs
    time_per_epoch = n_times / sfreq

    # Concatenate epochs for continuous display
    sensor_ts_concat = eeg_data[:n_epochs_to_show, ch_idx, :].flatten()

    # Compute source time series for display epochs
    source_ts_list = []
    for i in range(n_epochs_to_show):
        epoch_eeg = eeg_data[i, :, :]
        source_activity = W @ epoch_eeg
        source_signed = source_activity / (resolution_diagonal[:, np.newaxis] + 1e-10)
        source_ts_list.append(source_signed[0, :])
    source_ts_concat = np.concatenate(source_ts_list)

    # Time vector for concatenated data
    total_samples = n_epochs_to_show * n_times
    time_vec = np.arange(total_samples) / sfreq

    # Create plot
    plt.style.use('seaborn-v0_8-whitegrid')
    plt.rcParams.update({
        'font.size': 10,
        'axes.labelsize': 11,
        'axes.titlesize': 12,
        'axes.titleweight': 'bold',
        'figure.dpi': 150,
        'savefig.dpi': 300,
        'axes.spines.top': False,
        'axes.spines.right': False,
    })

    fig = plt.figure(figsize=(16, 10))
    gs = fig.add_gridspec(2, 2, height_ratios=[1, 1], hspace=0.3, wspace=0.25)

    # Color scheme
    sensor_color = '#1976D2'
    source_color = '#388E3C'

    # === TOP LEFT: Sensor Time Series ===
    ax1 = fig.add_subplot(gs[0, 0])
    ax1.plot(time_vec, sensor_ts_concat * 1e6, color=sensor_color, lw=0.5, alpha=0.8)
    ax1.set_xlabel('Time (s)')
    ax1.set_ylabel('Amplitude (µV)')
    ax1.set_title(f'Sensor Time Series (Ch {ch_idx+1})')
    ax1.set_xlim(0, time_vec[-1])

    # Add epoch markers
    for i in range(1, n_epochs_to_show):
        ax1.axvline(i * time_per_epoch, color='gray', linestyle='--', alpha=0.3, lw=0.5)

    # Stats box
    sensor_std = np.std(sensor_ts_concat) * 1e6
    ax1.text(0.02, 0.98, f'σ = {sensor_std:.2f} µV\n{n_epochs_to_show} epochs',
             transform=ax1.transAxes, va='top', ha='left', fontsize=9,
             bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

    # === TOP RIGHT: Source Time Series ===
    ax2 = fig.add_subplot(gs[0, 1])
    ax2.plot(time_vec, source_ts_concat, color=source_color, lw=0.5, alpha=0.8)
    ax2.set_xlabel('Time (s)')
    ax2.set_ylabel('Amplitude (a.u.)')
    ax2.set_title('Source Time Series (Signed dSPM)')
    ax2.set_xlim(0, time_vec[-1])

    # Add epoch markers
    for i in range(1, n_epochs_to_show):
        ax2.axvline(i * time_per_epoch, color='gray', linestyle='--', alpha=0.3, lw=0.5)

    # Stats box
    source_std = np.std(source_ts_concat)
    ax2.text(0.02, 0.98, f'σ = {source_std:.2e}\n{n_epochs_to_show} epochs',
             transform=ax2.transAxes, va='top', ha='left', fontsize=9,
             bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

    # === BOTTOM LEFT: Sensor PSD ===
    ax3 = fig.add_subplot(gs[1, 0])
    ax3.plot(freqs_plot, sensor_psd_mean[freq_mask], color=sensor_color, lw=2, label='Mean PSD')
    ax3.fill_between(freqs_plot,
                    sensor_psd_mean[freq_mask] - sensor_psd_sem[freq_mask],
                    sensor_psd_mean[freq_mask] + sensor_psd_sem[freq_mask],
                    alpha=0.3, color=sensor_color, label='±SEM')
    ax3.set_xlabel('Frequency (Hz)')
    ax3.set_ylabel('Power Spectral Density')
    ax3.set_title(f'Sensor PSD ({n_epochs} epochs) — 1/f slope β = {slope_sensor:.2f}')
    ax3.set_xlim(1, 55)
    ax3.legend(loc='upper right', fontsize=9)

    status_color = 'green' if sensor_ok else 'red'
    ax3.text(0.95, 0.05, 'PASS' if sensor_ok else 'FAIL',
            transform=ax3.transAxes, ha='right', va='bottom',
            fontsize=14, fontweight='bold', color=status_color,
            bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

    # === BOTTOM RIGHT: Source PSD ===
    ax4 = fig.add_subplot(gs[1, 1])
    ax4.plot(freqs_plot, source_psd_mean[freq_mask], color=source_color, lw=2, label='Mean PSD')
    ax4.fill_between(freqs_plot,
                    source_psd_mean[freq_mask] - source_psd_sem[freq_mask],
                    source_psd_mean[freq_mask] + source_psd_sem[freq_mask],
                    alpha=0.3, color=source_color, label='±SEM')
    ax4.set_xlabel('Frequency (Hz)')
    ax4.set_ylabel('Power (a.u.)')
    ax4.set_title(f'Source PSD ({n_epochs} epochs) — 1/f slope β = {slope_source:.2f}')
    ax4.set_xlim(1, 55)
    ax4.legend(loc='upper right', fontsize=9)

    status_color = 'green' if source_ok else 'red'
    ax4.text(0.95, 0.05, 'PASS' if source_ok else 'FAIL',
            transform=ax4.transAxes, ha='right', va='bottom',
            fontsize=14, fontweight='bold', color=status_color,
            bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

    # Overall title
    overall_status = 'PASS' if passed else 'FAIL'
    overall_color = '#388E3C' if passed else '#D32F2F'
    fig.suptitle(f'Source Localization Sanity Check: {overall_status}',
                 fontsize=14, fontweight='bold', y=0.98, color=overall_color)

    # Subtitle with details
    fig.text(0.5, 0.94, f'{n_epochs} epochs · {sfreq:.0f} Hz · {eeg_file.name}',
             ha='center', fontsize=10, color='gray')

    plt.savefig(output_plot, bbox_inches='tight', facecolor='white')
    print(f"\nSaved: {output_plot}")
    plt.close()

    return {
        'slope_sensor': slope_sensor,
        'slope_source': slope_source,
        'sensor_ok': sensor_ok,
        'source_ok': source_ok,
        'slopes_similar': slopes_similar,
        'passed': passed,
        'plot_path': str(output_plot)
    }


# Pytest fixtures and tests (only if pytest is available)
if HAS_PYTEST:
    def pytest_addoption(parser):
        parser.addoption("--eeg-file", action="store", help="Path to EEG .set file")

    @pytest.fixture
    def eeg_file(request):
        return request.config.getoption("--eeg-file")

    class TestPSDSanity:
        """Test suite for PSD sanity checks."""

        def test_1f_slope_preserved(self, eeg_file):
            """Test that 1/f spectral slope is preserved in source localization."""
            if eeg_file is None:
                pytest.skip("No EEG file provided. Use --eeg-file option.")

            results = run_psd_sanity_check(eeg_file)

            assert results['sensor_ok'], \
                f"Sensor slope {results['slope_sensor']:.2f} outside valid range [0.5, 3.0]"
            assert results['source_ok'], \
                f"Source slope {results['slope_source']:.2f} outside valid range [0.5, 3.0]"
            assert results['slopes_similar'], \
                f"Slopes too different: sensor={results['slope_sensor']:.2f}, source={results['slope_source']:.2f}"


if __name__ == '__main__':
    if len(sys.argv) < 2:
        print("Usage: python test_psd_sanity.py /path/to/eeg_file.set")
        sys.exit(1)

    eeg_file = sys.argv[1]
    results = run_psd_sanity_check(eeg_file)

    sys.exit(0 if results['passed'] else 1)
