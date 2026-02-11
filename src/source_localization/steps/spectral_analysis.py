"""Step 7: Spectral Analysis.

Compute power spectral features (band power) for each ROI.
"""

import numpy as np
from scipy import signal


def run(config, previous_outputs):
    """
    Compute spectral features for each ROI.

    Parameters
    ----------
    config : Config
        Pipeline configuration
    previous_outputs : dict
        Outputs from previous steps containing:
        - 'roi_stcs': dict - ROI time courses
        - 'roi_labels': list - ROI names
        - 'stc': mne.SourceEstimate - Source estimate (for sampling frequency)

    Returns
    -------
    outputs : dict
        Dictionary containing:
        - 'roi_band_power': dict - Band power for each ROI and frequency band
        - 'frequency_bands': dict - Frequency band definitions
        - 'primary_band': str - Primary frequency band of interest
    """
    frequency_bands = config['spectral']['frequency_bands']
    primary_band = config['spectral']['primary_band']

    print(f"  Computing spectral features:")
    print(f"    Frequency bands: {', '.join(frequency_bands.keys())}")
    print(f"    Primary band: {primary_band}")

    # Extract inputs
    roi_stcs = previous_outputs['roi_stcs']
    roi_labels = previous_outputs['roi_labels']
    stc = previous_outputs['stc']

    # Get sampling frequency from source estimate
    sfreq = 1.0 / stc.tstep  # Sampling frequency in Hz

    print(f"    Sampling frequency: {sfreq:.1f} Hz")

    # Compute band power for each ROI
    roi_band_power = {}

    for roi_name in roi_labels:
        roi_timecourse = roi_stcs[roi_name]

        # Compute power in each frequency band
        band_powers = {}
        for band_name, (fmin, fmax) in frequency_bands.items():
            # Compute band power using Welch's method
            power = compute_band_power(roi_timecourse, sfreq, fmin, fmax)
            band_powers[band_name] = power

        roi_band_power[roi_name] = band_powers

    # Report statistics for primary band
    primary_powers = [roi_band_power[roi][primary_band] for roi in roi_labels]
    print(f"    {primary_band.capitalize()} band power: "
          f"mean={np.mean(primary_powers):.2e}, "
          f"median={np.median(primary_powers):.2e}")

    # Save intermediate data and create visualizations
    if config['outputs'].get('save_intermediate', True):
        from ..utils.io_utils import save_pickle, get_data_dir, get_figures_dir
        from ..utils.step_visualizations import visualize_step7_band_power

        data_dir = get_data_dir(config)
        figures_dir = get_figures_dir(config)

        # Save band power data
        save_pickle(roi_band_power, data_dir / 'step7_band_power.pkl')
        print(f"    Saved: {data_dir / 'step7_band_power.pkl'}")

        # Create band power visualization
        fig = visualize_step7_band_power(roi_band_power, frequency_bands, figures_dir / 'step7_band_power.png')
        print(f"    Saved: {figures_dir / 'step7_band_power.png'}")
        import matplotlib.pyplot as plt
        plt.close(fig)

    return {
        'roi_band_power': roi_band_power,
        'frequency_bands': frequency_bands,
        'primary_band': primary_band
    }


def compute_band_power(timecourse, sfreq, fmin, fmax, method='welch'):
    """
    Compute power in a specific frequency band.

    Parameters
    ----------
    timecourse : ndarray, shape (n_times,)
        Time course data
    sfreq : float
        Sampling frequency in Hz
    fmin : float
        Lower frequency bound in Hz
    fmax : float
        Upper frequency bound in Hz
    method : str
        Method to compute power ('welch' or 'fft')

    Returns
    -------
    power : float
        Mean power in the frequency band
    """
    if method == 'welch':
        # Use Welch's method for robust power spectral density estimation
        # Window length: min of signal length or 2 seconds
        nperseg = min(len(timecourse), int(2.0 * sfreq))

        freqs, psd = signal.welch(
            timecourse,
            fs=sfreq,
            nperseg=nperseg,
            noverlap=nperseg // 2,
            scaling='density'
        )

        # Find frequencies in band
        freq_mask = (freqs >= fmin) & (freqs <= fmax)

        # Integrate power in band (trapezoidal integration)
        # Use trapezoid (numpy 2.0+) or trapz (numpy <2.0) for compatibility
        trapz_func = getattr(np, 'trapezoid', getattr(np, 'trapz', None))
        band_power = trapz_func(psd[freq_mask], freqs[freq_mask])

    elif method == 'fft':
        # Simple FFT-based power estimation
        fft_vals = np.fft.rfft(timecourse)
        fft_freqs = np.fft.rfftfreq(len(timecourse), d=1.0/sfreq)

        # Power spectral density
        psd = np.abs(fft_vals) ** 2 / len(timecourse)

        # Find frequencies in band
        freq_mask = (fft_freqs >= fmin) & (fft_freqs <= fmax)

        # Mean power in band
        band_power = psd[freq_mask].mean()

    else:
        raise ValueError(f"Unknown method: {method}")

    return band_power


def compute_relative_band_power(timecourse, sfreq, fmin, fmax, total_band=(1, 100)):
    """
    Compute relative power in a frequency band (band power / total power).

    Parameters
    ----------
    timecourse : ndarray, shape (n_times,)
        Time course data
    sfreq : float
        Sampling frequency in Hz
    fmin : float
        Lower frequency bound of target band in Hz
    fmax : float
        Upper frequency bound of target band in Hz
    total_band : tuple
        Frequency range for total power normalization (fmin, fmax)

    Returns
    -------
    relative_power : float
        Relative power in the frequency band (0-1)
    """
    # Compute power in target band
    band_power = compute_band_power(timecourse, sfreq, fmin, fmax)

    # Compute total power
    total_power = compute_band_power(timecourse, sfreq, total_band[0], total_band[1])

    # Relative power
    if total_power > 0:
        relative_power = band_power / total_power
    else:
        relative_power = 0.0

    return relative_power
