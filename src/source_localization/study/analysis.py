"""
Analysis Module - MNE Wrappers for Batch Analysis

Provides simple functions that wrap MNE-Python and MNE-Connectivity
to compute spectral and connectivity analyses, saving results to
the study folder hierarchy.

This module uses MNE's optimized implementations internally while
providing a simple interface that integrates with our pipeline outputs.

Usage:
    from source_localization.study import analyze_subject, analyze_study

    # Analyze single subject
    analyze_subject(subject_dir, bands=['low_gamma', 'theta'], connectivity=['coherence'])

    # Analyze all subjects in study
    analyze_study(study_config, n_jobs=4)

Author: Claude Code
Date: 2026-01-26
"""

import json
import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union, Any
from concurrent.futures import ProcessPoolExecutor, as_completed

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# Default frequency bands
DEFAULT_BANDS = {
    'delta': (1, 4),
    'theta': (4, 10),
    'alpha': (10, 13),
    'beta': (13, 30),
    'low_gamma': (30, 55),
    'high_gamma': (65, 100),
}

# Default connectivity methods
DEFAULT_CONNECTIVITY_METHODS = ['coherence']


def analyze_subject(
    subject_dir: Union[str, Path],
    bands: Optional[Dict[str, Tuple[float, float]]] = None,
    connectivity_methods: Optional[List[str]] = None,
    connectivity_bands: Optional[List[str]] = None,
    overwrite: bool = False,
    epoch_length: float = 2.0,
) -> Dict[str, Any]:
    """
    Run spectral and connectivity analysis on a single subject.

    Uses MNE-Python for spectral analysis and MNE-Connectivity for
    connectivity measures. Results are saved to subject_dir/analysis/.

    Args:
        subject_dir: Path to subject output directory (contains roi_timeseries/)
        bands: Dict of band_name -> (fmin, fmax). Default: delta, theta, alpha, beta, gamma
        connectivity_methods: List of methods: 'coherence', 'plv', 'wpli', 'imcoh'
        connectivity_bands: Which bands to compute connectivity for (default: all)
        overwrite: Overwrite existing analysis files
        epoch_length: For continuous data, epoch into segments of this length (seconds)
                      for connectivity analysis. Default: 2.0s

    Returns:
        Dict with 'band_power' and 'connectivity' results

    Outputs saved:
        analysis/band_power.csv - Power per ROI per band
        analysis/connectivity_{method}_{band}.csv - Connectivity matrices
    """
    import mne

    subject_dir = Path(subject_dir)
    roi_file = subject_dir / "roi_timeseries" / "roi_timeseries_signed.set"

    if not roi_file.exists():
        # Try pipeline subdirectory
        roi_file = subject_dir / "pipeline" / "data" / "roi_timeseries_signed.set"

    if not roi_file.exists():
        raise FileNotFoundError(f"ROI timeseries not found: {roi_file}")

    # Create analysis directory
    analysis_dir = subject_dir / "analysis"
    analysis_dir.mkdir(exist_ok=True)

    # Use defaults if not specified
    bands = bands or DEFAULT_BANDS
    connectivity_methods = connectivity_methods or DEFAULT_CONNECTIVITY_METHODS
    connectivity_bands = connectivity_bands or list(bands.keys())

    results = {}

    # Load data with MNE
    logger.info(f"Loading {roi_file}")
    raw = None
    epochs = None
    try:
        # Try loading as epochs first
        epochs = mne.io.read_epochs_eeglab(str(roi_file), verbose=False)
        data = epochs.get_data()  # (n_epochs, n_channels, n_times)
        sfreq = epochs.info['sfreq']
        ch_names = epochs.ch_names
        is_epoched = True
    except Exception:
        # Fall back to raw
        raw = mne.io.read_raw_eeglab(str(roi_file), preload=True, verbose=False)
        data = raw.get_data()  # (n_channels, n_times)
        sfreq = raw.info['sfreq']
        ch_names = raw.ch_names
        is_epoched = False

    # Compute band power
    band_power_file = analysis_dir / "band_power.csv"
    if overwrite or not band_power_file.exists():
        logger.info("Computing band power...")
        band_power_df = _compute_band_power(data, sfreq, ch_names, bands, is_epoched)
        band_power_df.to_csv(band_power_file, index=False)
        logger.info(f"Saved: {band_power_file}")
        results['band_power'] = band_power_df
    else:
        logger.info(f"Band power exists, skipping (use overwrite=True to recompute)")
        results['band_power'] = pd.read_csv(band_power_file)

    # For connectivity, create epochs from continuous data if needed
    if connectivity_methods and not is_epoched and raw is not None:
        logger.info(f"Creating {epoch_length}s epochs from continuous data for connectivity...")
        epochs = mne.make_fixed_length_epochs(
            raw, duration=epoch_length, preload=True, verbose=False
        )
        logger.info(f"Created {len(epochs)} epochs")
        is_epoched = True

    # Compute connectivity
    if connectivity_methods and is_epoched:
        results['connectivity'] = {}

        for method in connectivity_methods:
            for band_name in connectivity_bands:
                conn_file = analysis_dir / f"connectivity_{method}_{band_name}.csv"

                if overwrite or not conn_file.exists():
                    logger.info(f"Computing {method} connectivity for {band_name}...")
                    fmin, fmax = bands[band_name]

                    try:
                        conn_matrix = _compute_connectivity(
                            epochs, method, fmin, fmax
                        )
                        conn_df = pd.DataFrame(
                            conn_matrix,
                            index=ch_names,
                            columns=ch_names
                        )
                        conn_df.to_csv(conn_file)
                        logger.info(f"Saved: {conn_file}")
                        results['connectivity'][f"{method}_{band_name}"] = conn_matrix
                    except Exception as e:
                        logger.warning(f"Failed to compute {method} for {band_name}: {e}")
                else:
                    logger.info(f"Connectivity {method}_{band_name} exists, skipping")

    elif connectivity_methods and not is_epoched:
        logger.warning("Connectivity requires epoched data. Skipping connectivity analysis.")

    return results


def _compute_band_power(
    data: np.ndarray,
    sfreq: float,
    ch_names: List[str],
    bands: Dict[str, Tuple[float, float]],
    is_epoched: bool
) -> pd.DataFrame:
    """Compute band power using MNE's Welch method."""
    from mne.time_frequency import psd_array_welch

    # Average across epochs if needed
    if is_epoched and data.ndim == 3:
        # (n_epochs, n_channels, n_times) -> (n_channels, n_times)
        data_2d = data.mean(axis=0)
    else:
        data_2d = data

    # Compute PSD
    psds, freqs = psd_array_welch(
        data_2d,
        sfreq=sfreq,
        fmin=0.5,
        fmax=min(100, sfreq / 2 - 1),
        n_fft=min(256, data_2d.shape[-1]),
        verbose=False
    )

    # Extract band power
    rows = []
    for ch_idx, ch_name in enumerate(ch_names):
        for band_name, (fmin, fmax) in bands.items():
            freq_mask = (freqs >= fmin) & (freqs <= fmax)
            if not np.any(freq_mask):
                continue

            power = np.mean(psds[ch_idx, freq_mask])
            power_db = 10 * np.log10(power + 1e-20)

            rows.append({
                'roi': ch_name,
                'band': band_name,
                'fmin': fmin,
                'fmax': fmax,
                'power': power,
                'power_db': power_db,
            })

    return pd.DataFrame(rows)


def _compute_connectivity(
    epochs,
    method: str,
    fmin: float,
    fmax: float
) -> np.ndarray:
    """Compute connectivity using MNE-Connectivity."""
    try:
        from mne_connectivity import spectral_connectivity_epochs
    except ImportError:
        raise ImportError(
            "mne-connectivity is required for connectivity analysis. "
            "Install with: pip install mne-connectivity"
        )

    # Map method names to MNE-Connectivity names
    method_map = {
        'coherence': 'coh',
        'coh': 'coh',
        'plv': 'plv',
        'wpli': 'wpli',
        'imcoh': 'imcoh',
        'pli': 'pli',
    }

    mne_method = method_map.get(method.lower(), method)

    # Compute connectivity
    conn = spectral_connectivity_epochs(
        epochs,
        method=mne_method,
        mode='multitaper',
        fmin=fmin,
        fmax=fmax,
        faverage=True,  # Average over frequency band
        verbose=False
    )

    # Extract matrix
    # conn.get_data() returns (n_connections,) for faverage=True
    # We need to reshape to (n_channels, n_channels)
    n_channels = len(epochs.ch_names)
    data = conn.get_data(output='dense')

    # Handle different output shapes
    if data.ndim == 3:
        # (n_channels, n_channels, n_freqs)
        data = data.mean(axis=-1)
    elif data.ndim == 1:
        # Reshape from upper triangle to full matrix
        matrix = np.zeros((n_channels, n_channels))
        idx = np.triu_indices(n_channels, k=1)
        matrix[idx] = data
        matrix = matrix + matrix.T
        np.fill_diagonal(matrix, 1.0)
        data = matrix

    return data


def analyze_study(
    config: 'StudyConfig',
    bands: Optional[Dict[str, Tuple[float, float]]] = None,
    connectivity_methods: Optional[List[str]] = None,
    connectivity_bands: Optional[List[str]] = None,
    n_jobs: int = 1,
    overwrite: bool = False,
    subjects: Optional[List[str]] = None,
    epoch_length: float = 2.0,
) -> pd.DataFrame:
    """
    Run analysis on all subjects in a study.

    Args:
        config: StudyConfig object
        bands: Frequency bands to analyze
        connectivity_methods: Connectivity methods to compute
        connectivity_bands: Which bands to compute connectivity for
        n_jobs: Number of parallel jobs
        overwrite: Overwrite existing analysis
        subjects: Optional list of subject IDs (default: all)
        epoch_length: For continuous data, epoch into segments of this length (seconds)

    Returns:
        Combined DataFrame with all subjects' band power

    Outputs:
        - Per-subject: analysis/band_power.csv, analysis/connectivity_*.csv
        - Group: group/group_band_power.csv
    """
    from .config import StudyConfig

    bands = bands or DEFAULT_BANDS
    connectivity_methods = connectivity_methods or DEFAULT_CONNECTIVITY_METHODS

    # Get subjects to analyze
    if subjects:
        subjects_to_analyze = [s for s in config.subjects if s.subject_id in subjects]
    else:
        subjects_to_analyze = config.subjects

    logger.info(f"Analyzing {len(subjects_to_analyze)} subjects...")

    # Analyze subjects
    all_results = []

    if n_jobs == 1:
        for subject in subjects_to_analyze:
            output_dir = config.get_subject_output_dir(subject)
            try:
                result = analyze_subject(
                    output_dir,
                    bands=bands,
                    connectivity_methods=connectivity_methods,
                    connectivity_bands=connectivity_bands,
                    overwrite=overwrite,
                    epoch_length=epoch_length,
                )
                if 'band_power' in result:
                    df = result['band_power'].copy()
                    df['subject_id'] = subject.subject_id
                    df['group'] = subject.group
                    all_results.append(df)
                logger.info(f"Subject {subject.subject_id}: OK")
            except Exception as e:
                logger.error(f"Subject {subject.subject_id}: {e}")
    else:
        # Parallel processing
        with ProcessPoolExecutor(max_workers=n_jobs) as executor:
            futures = {}
            for subject in subjects_to_analyze:
                output_dir = config.get_subject_output_dir(subject)
                future = executor.submit(
                    analyze_subject,
                    output_dir,
                    bands=bands,
                    connectivity_methods=connectivity_methods,
                    connectivity_bands=connectivity_bands,
                    overwrite=overwrite,
                    epoch_length=epoch_length,
                )
                futures[future] = subject

            for future in as_completed(futures):
                subject = futures[future]
                try:
                    result = future.result()
                    if 'band_power' in result:
                        df = result['band_power'].copy()
                        df['subject_id'] = subject.subject_id
                        df['group'] = subject.group
                        all_results.append(df)
                    logger.info(f"Subject {subject.subject_id}: OK")
                except Exception as e:
                    logger.error(f"Subject {subject.subject_id}: {e}")

    # Combine and save group results
    if all_results:
        group_df = pd.concat(all_results, ignore_index=True)

        # Reorder columns
        cols = ['subject_id', 'group', 'roi', 'band', 'fmin', 'fmax', 'power', 'power_db']
        group_df = group_df[[c for c in cols if c in group_df.columns]]

        # Save
        config.group_dir.mkdir(exist_ok=True)
        group_file = config.group_dir / "group_band_power.csv"
        group_df.to_csv(group_file, index=False)
        logger.info(f"Saved group results: {group_file}")

        return group_df

    return pd.DataFrame()


def collect_connectivity_matrices(
    config: 'StudyConfig',
    method: str = 'coherence',
    band: str = 'low_gamma',
    subjects: Optional[List[str]] = None,
) -> Dict[str, np.ndarray]:
    """
    Collect connectivity matrices from all subjects.

    Args:
        config: StudyConfig object
        method: Connectivity method
        band: Frequency band
        subjects: Optional list of subject IDs

    Returns:
        Dict mapping subject_id -> connectivity matrix
    """
    if subjects:
        subjects_to_collect = [s for s in config.subjects if s.subject_id in subjects]
    else:
        subjects_to_collect = config.subjects

    matrices = {}

    for subject in subjects_to_collect:
        output_dir = config.get_subject_output_dir(subject)
        conn_file = output_dir / "analysis" / f"connectivity_{method}_{band}.csv"

        if conn_file.exists():
            df = pd.read_csv(conn_file, index_col=0)
            matrices[subject.subject_id] = df.values
        else:
            logger.warning(f"Missing connectivity for {subject.subject_id}")

    return matrices


def compute_group_connectivity(
    config: 'StudyConfig',
    method: str = 'coherence',
    band: str = 'low_gamma',
    subjects: Optional[List[str]] = None,
    by_group: bool = True,
) -> Union[np.ndarray, Dict[str, np.ndarray]]:
    """
    Compute average connectivity matrix across subjects.

    Args:
        config: StudyConfig object
        method: Connectivity method
        band: Frequency band
        subjects: Optional list of subject IDs
        by_group: If True, return dict of group -> average matrix

    Returns:
        Average connectivity matrix, or dict by group
    """
    matrices = collect_connectivity_matrices(config, method, band, subjects)

    if not matrices:
        raise ValueError("No connectivity matrices found")

    if by_group:
        # Group by experimental group
        group_matrices = {}
        for subject in config.subjects:
            if subject.subject_id in matrices:
                group = subject.group or 'unknown'
                if group not in group_matrices:
                    group_matrices[group] = []
                group_matrices[group].append(matrices[subject.subject_id])

        # Average within groups
        result = {}
        for group, mats in group_matrices.items():
            result[group] = np.mean(mats, axis=0)

        # Save
        config.group_dir.mkdir(exist_ok=True)
        for group, mat in result.items():
            # Get ROI names from first matrix
            sample_file = list(Path(config.derivatives_dir).glob(
                f"*/analysis/connectivity_{method}_{band}.csv"
            ))[0]
            roi_names = pd.read_csv(sample_file, index_col=0).index.tolist()

            df = pd.DataFrame(mat, index=roi_names, columns=roi_names)
            safe_group = group.replace('/', '_').replace(' ', '_')
            out_file = config.group_dir / f"connectivity_{method}_{band}_{safe_group}.csv"
            df.to_csv(out_file)
            logger.info(f"Saved: {out_file}")

        return result
    else:
        # Average across all subjects
        return np.mean(list(matrices.values()), axis=0)
