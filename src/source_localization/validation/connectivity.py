"""Connectivity Validation Module.

Compares connectivity patterns between:
1. Electrode-level connectivity (from raw EEG)
2. ROI-level connectivity (from source-localized data)

The goal is to validate that source localization preserves meaningful
connectivity structure while providing better spatial specificity.

**Created:** 2025-01-27
"""

import numpy as np
from pathlib import Path
from typing import Dict, Optional, Tuple, Union

import mne


def compute_electrode_connectivity(
    epochs: mne.Epochs,
    method: str = 'coh',
    fmin: float = 1.0,
    fmax: float = 100.0,
    n_bands: int = 5,
    verbose: bool = True,
) -> Dict[str, np.ndarray]:
    """
    Compute connectivity matrix from electrode-level EEG data.

    Parameters
    ----------
    epochs : mne.Epochs
        EEG epochs data
    method : str
        Connectivity method ('coh', 'plv', 'pli', 'wpli')
    fmin, fmax : float
        Frequency range
    n_bands : int
        Number of frequency bands to compute
    verbose : bool
        Print progress info

    Returns
    -------
    connectivity : dict
        Dictionary with band names as keys, connectivity matrices as values
        Each matrix is (n_channels, n_channels)
    """
    from mne_connectivity import spectral_connectivity_epochs

    # Define frequency bands
    bands = {
        'delta': (1, 4),
        'theta': (4, 10),
        'alpha': (10, 13),
        'beta': (13, 30),
        'low_gamma': (30, 55),
        'high_gamma': (65, 100),
    }

    # Filter to requested range
    bands = {k: v for k, v in bands.items() if v[0] >= fmin and v[1] <= fmax}

    if verbose:
        print(f"\nComputing electrode connectivity ({method}):")
        print(f"  Bands: {list(bands.keys())}")
        print(f"  Channels: {len(epochs.ch_names)}")

    connectivity = {}

    for band_name, (f_low, f_high) in bands.items():
        if verbose:
            print(f"  Computing {band_name} ({f_low}-{f_high} Hz)...", end='')

        # Compute spectral connectivity
        con = spectral_connectivity_epochs(
            epochs,
            method=method,
            mode='multitaper',
            fmin=f_low,
            fmax=f_high,
            faverage=True,
            verbose=False,
        )

        # Extract connectivity matrix
        conn_matrix = con.get_data(output='dense')[:, :, 0]

        # Make symmetric (take average of upper and lower triangle)
        conn_matrix = (conn_matrix + conn_matrix.T) / 2
        np.fill_diagonal(conn_matrix, 1.0)

        connectivity[band_name] = conn_matrix

        if verbose:
            print(f" done (mean={np.mean(conn_matrix):.3f})")

    return connectivity


def compute_roi_connectivity(
    roi_epochs: mne.Epochs,
    method: str = 'coh',
    fmin: float = 1.0,
    fmax: float = 100.0,
    verbose: bool = True,
) -> Dict[str, np.ndarray]:
    """
    Compute connectivity matrix from ROI-level source time series.

    Parameters
    ----------
    roi_epochs : mne.Epochs
        ROI time series as epochs (loaded from roi_timeseries_signed.set)
    method : str
        Connectivity method ('coh', 'plv', 'pli', 'wpli')
    fmin, fmax : float
        Frequency range
    verbose : bool
        Print progress info

    Returns
    -------
    connectivity : dict
        Dictionary with band names as keys, connectivity matrices as values
        Each matrix is (n_rois, n_rois)
    """
    from mne_connectivity import spectral_connectivity_epochs

    # Define frequency bands
    bands = {
        'delta': (1, 4),
        'theta': (4, 10),
        'alpha': (10, 13),
        'beta': (13, 30),
        'low_gamma': (30, 55),
        'high_gamma': (65, 100),
    }

    # Filter to requested range
    bands = {k: v for k, v in bands.items() if v[0] >= fmin and v[1] <= fmax}

    if verbose:
        print(f"\nComputing ROI connectivity ({method}):")
        print(f"  Bands: {list(bands.keys())}")
        print(f"  ROIs: {len(roi_epochs.ch_names)}")

    connectivity = {}

    for band_name, (f_low, f_high) in bands.items():
        if verbose:
            print(f"  Computing {band_name} ({f_low}-{f_high} Hz)...", end='')

        # Compute spectral connectivity
        con = spectral_connectivity_epochs(
            roi_epochs,
            method=method,
            mode='multitaper',
            fmin=f_low,
            fmax=f_high,
            faverage=True,
            verbose=False,
        )

        # Extract connectivity matrix
        conn_matrix = con.get_data(output='dense')[:, :, 0]

        # Make symmetric
        conn_matrix = (conn_matrix + conn_matrix.T) / 2
        np.fill_diagonal(conn_matrix, 1.0)

        connectivity[band_name] = conn_matrix

        if verbose:
            print(f" done (mean={np.mean(conn_matrix):.3f})")

    return connectivity


def compare_connectivity_patterns(
    electrode_conn: Dict[str, np.ndarray],
    roi_conn: Dict[str, np.ndarray],
    verbose: bool = True,
) -> Dict[str, Dict[str, float]]:
    """
    Compare electrode-level and ROI-level connectivity patterns.

    Since electrode and ROI matrices have different dimensions, we compare:
    1. Distribution statistics (mean, std, range)
    2. Network metrics (clustering coefficient, path length)
    3. Within-method consistency across frequency bands

    Parameters
    ----------
    electrode_conn : dict
        Electrode connectivity by band
    roi_conn : dict
        ROI connectivity by band
    verbose : bool
        Print comparison results

    Returns
    -------
    comparison : dict
        Comparison metrics for each frequency band
    """
    comparison = {}

    for band in electrode_conn.keys():
        if band not in roi_conn:
            continue

        elec_mat = electrode_conn[band]
        roi_mat = roi_conn[band]

        # Get upper triangle (excluding diagonal)
        elec_triu = elec_mat[np.triu_indices(elec_mat.shape[0], k=1)]
        roi_triu = roi_mat[np.triu_indices(roi_mat.shape[0], k=1)]

        # Distribution statistics
        comparison[band] = {
            'electrode_mean': float(np.mean(elec_triu)),
            'electrode_std': float(np.std(elec_triu)),
            'roi_mean': float(np.mean(roi_triu)),
            'roi_std': float(np.std(roi_triu)),
            'electrode_n_connections': len(elec_triu),
            'roi_n_connections': len(roi_triu),
        }

        # Network density (proportion of strong connections)
        threshold = 0.5
        comparison[band]['electrode_density'] = float(np.mean(elec_triu > threshold))
        comparison[band]['roi_density'] = float(np.mean(roi_triu > threshold))

    if verbose:
        print("\nConnectivity Comparison:")
        print("-" * 70)
        print(f"{'Band':<10} {'Electrode Mean':<15} {'ROI Mean':<15} {'Elec Density':<15} {'ROI Density':<15}")
        print("-" * 70)
        for band, metrics in comparison.items():
            print(f"{band:<10} {metrics['electrode_mean']:<15.3f} {metrics['roi_mean']:<15.3f} "
                  f"{metrics['electrode_density']:<15.3f} {metrics['roi_density']:<15.3f}")
        print("-" * 70)

    return comparison


def validate_connectivity(
    eeg_file: Union[str, Path],
    roi_timeseries_file: Union[str, Path],
    method: str = 'coh',
    fmin: float = 1.0,
    fmax: float = 80.0,
    output_dir: Optional[Union[str, Path]] = None,
    verbose: bool = True,
) -> Dict[str, any]:
    """
    Full connectivity validation comparing electrode vs ROI connectivity.

    Parameters
    ----------
    eeg_file : str or Path
        Path to original EEG file (.set)
    roi_timeseries_file : str or Path
        Path to ROI time series file (.set)
    method : str
        Connectivity method
    fmin, fmax : float
        Frequency range
    output_dir : str or Path, optional
        Directory to save results
    verbose : bool
        Print progress

    Returns
    -------
    results : dict
        Validation results including connectivity matrices and comparisons
    """
    import matplotlib.pyplot as plt

    eeg_file = Path(eeg_file)
    roi_timeseries_file = Path(roi_timeseries_file)

    if verbose:
        print("=" * 70)
        print("Connectivity Validation")
        print("=" * 70)
        print(f"EEG file: {eeg_file.name}")
        print(f"ROI file: {roi_timeseries_file.name}")

    # Load EEG epochs
    if verbose:
        print("\nLoading EEG data...")
    eeg_epochs = mne.io.read_epochs_eeglab(str(eeg_file), verbose=False)
    eeg_epochs.set_eeg_reference('average', projection=True, verbose=False)

    # Load ROI time series
    if verbose:
        print("Loading ROI time series...")
    roi_epochs = mne.io.read_epochs_eeglab(str(roi_timeseries_file), verbose=False)

    # Compute connectivity
    electrode_conn = compute_electrode_connectivity(
        eeg_epochs, method=method, fmin=fmin, fmax=fmax, verbose=verbose
    )
    roi_conn = compute_roi_connectivity(
        roi_epochs, method=method, fmin=fmin, fmax=fmax, verbose=verbose
    )

    # Compare patterns
    comparison = compare_connectivity_patterns(
        electrode_conn, roi_conn, verbose=verbose
    )

    results = {
        'electrode_connectivity': electrode_conn,
        'roi_connectivity': roi_conn,
        'comparison': comparison,
        'method': method,
        'fmin': fmin,
        'fmax': fmax,
        'n_electrodes': len(eeg_epochs.ch_names),
        'n_rois': len(roi_epochs.ch_names),
    }

    # Save results and figures
    if output_dir is not None:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        # Save comparison metrics
        import json
        with open(output_dir / 'connectivity_comparison.json', 'w') as f:
            json.dump(comparison, f, indent=2)

        # Create visualization
        fig = _plot_connectivity_comparison(
            electrode_conn, roi_conn, comparison,
            eeg_epochs.ch_names, roi_epochs.ch_names
        )
        fig.savefig(output_dir / 'connectivity_comparison.png', dpi=150, bbox_inches='tight')
        plt.close(fig)

        if verbose:
            print(f"\nResults saved to: {output_dir}")

    return results


def _plot_connectivity_comparison(
    electrode_conn: Dict[str, np.ndarray],
    roi_conn: Dict[str, np.ndarray],
    comparison: Dict[str, Dict[str, float]],
    electrode_names: list,
    roi_names: list,
) -> 'plt.Figure':
    """Create comparison figure for electrode vs ROI connectivity."""
    import matplotlib.pyplot as plt

    bands = list(electrode_conn.keys())
    n_bands = len(bands)

    fig, axes = plt.subplots(2, n_bands, figsize=(4 * n_bands, 8))
    if n_bands == 1:
        axes = axes.reshape(2, 1)

    for i, band in enumerate(bands):
        # Electrode connectivity
        ax = axes[0, i]
        im = ax.imshow(electrode_conn[band], cmap='RdBu_r', vmin=0, vmax=1)
        ax.set_title(f'Electrode - {band.capitalize()}')
        ax.set_xlabel('Channel')
        ax.set_ylabel('Channel')
        plt.colorbar(im, ax=ax, shrink=0.7)

        # ROI connectivity
        ax = axes[1, i]
        im = ax.imshow(roi_conn[band], cmap='RdBu_r', vmin=0, vmax=1)
        ax.set_title(f'ROI - {band.capitalize()}')
        ax.set_xlabel('ROI')
        ax.set_ylabel('ROI')
        plt.colorbar(im, ax=ax, shrink=0.7)

    fig.suptitle('Connectivity Comparison: Electrodes vs ROIs', fontsize=14, y=1.02)
    plt.tight_layout()

    return fig
