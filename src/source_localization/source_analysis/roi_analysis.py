"""
ROI-Level Time Series Extraction with Depth-Based Accuracy Weighting

This module extracts ROI-level signals from source localization results,
weighted by localization accuracy (depth-based). This is our unique contribution
based on validation results showing depth-dependent accuracy.

For connectivity and spectral analysis, use MNE-Python / MNE-Connectivity:

    import mne
    from mne_connectivity import spectral_connectivity_epochs

    # Load pipeline output
    epochs = mne.io.read_epochs_eeglab('roi_timeseries_signed.set')

    # Connectivity analysis
    conn = spectral_connectivity_epochs(epochs, method='coh', fmin=30, fmax=80)

    # Spectral analysis
    psds, freqs = mne.time_frequency.psd_array_welch(epochs.get_data(), sfreq=epochs.info['sfreq'])

Key design decisions:
1. ROI signals are computed by averaging ALL sources within ROI boundaries
   (not just centroid) - this provides more representative ROI activity
2. Sources are weighted by localization accuracy (depth-based) based on
   our validation results showing depth-dependent accuracy

Depth-Based Accuracy (from validation):
    - 0-1mm: 77% ROI accuracy
    - 1-2mm: 36% accuracy
    - 2-3mm: 4% accuracy
    - >3mm: ~0% accuracy

Author: Claude Code
Date: 2026-01-26
"""

import numpy as np
from typing import Dict, List, Optional, Union, Any
from dataclasses import dataclass
import logging

logger = logging.getLogger(__name__)


# =============================================================================
# Data Classes
# =============================================================================

@dataclass
class ROITimeSeries:
    """ROI-level time series with metadata."""
    roi_id: int
    roi_name: str
    data: np.ndarray  # (n_times,) or (n_epochs, n_times)
    n_sources: int
    source_indices: np.ndarray
    mean_depth_mm: float
    accuracy_weight: float  # Based on depth


# =============================================================================
# Depth-Based Accuracy Weighting
# =============================================================================

def compute_depth_weights(
    depths_mm: np.ndarray,
    method: str = 'empirical'
) -> np.ndarray:
    """
    Compute accuracy weights based on source depth from brain surface.

    Based on validation results:
    - 0-1mm: 77% ROI accuracy
    - 1-2mm: 36% accuracy
    - 2-3mm: 4% accuracy
    - >3mm: ~0% accuracy

    Args:
        depths_mm: (n_sources,) depth from brain surface in mm
        method: 'empirical' (validation-based), 'linear', or 'exponential'

    Returns:
        (n_sources,) weights in [0, 1]
    """
    depths_mm = np.asarray(depths_mm)
    weights = np.zeros_like(depths_mm)

    if method == 'empirical':
        # Piecewise linear based on validation results
        # Interpolate within bins for smooth weighting
        mask_01 = depths_mm <= 1.0
        mask_12 = (depths_mm > 1.0) & (depths_mm <= 2.0)
        mask_23 = (depths_mm > 2.0) & (depths_mm <= 3.0)

        weights[mask_01] = 0.77 - 0.41 * (depths_mm[mask_01] / 1.0)  # 0.77 -> 0.36
        weights[mask_12] = 0.36 - 0.32 * ((depths_mm[mask_12] - 1.0) / 1.0)  # 0.36 -> 0.04
        weights[mask_23] = 0.04 - 0.04 * ((depths_mm[mask_23] - 2.0) / 1.0)  # 0.04 -> 0.0
        # >3mm stays at 0

    elif method == 'linear':
        # Linear decay from 1.0 at surface to 0.0 at 3mm
        weights = np.maximum(0, 1.0 - depths_mm / 3.0)

    elif method == 'exponential':
        # Exponential decay with length constant of 1mm
        weights = np.exp(-depths_mm / 1.0)

    else:
        raise ValueError(f"Unknown weighting method: {method}")

    return np.clip(weights, 0, 1)


# =============================================================================
# ROI Time Series Extraction
# =============================================================================

class ROIAnalysis:
    """
    Depth-weighted ROI time series extraction from source localization.

    Extracts ROI signals by averaging all sources within ROI boundaries,
    weighted by localization accuracy (depth-based).

    For connectivity and spectral analysis, export ROI time series and use
    MNE-Python / MNE-Connectivity which have optimized, well-tested implementations.
    """

    def __init__(
        self,
        source_data: np.ndarray,
        source_coords_mm: np.ndarray,
        source_to_roi: Dict[int, int],
        roi_names: Dict[int, str],
        sfreq: float,
        source_depths_mm: Optional[np.ndarray] = None,
        depth_weighting: str = 'empirical'
    ):
        """
        Initialize ROI analysis.

        Args:
            source_data: (n_sources, n_times) or (n_sources, n_epochs, n_times)
            source_coords_mm: (n_sources, 3) source coordinates
            source_to_roi: Dict mapping source index to ROI ID
            roi_names: Dict mapping ROI ID to name
            sfreq: Sampling frequency in Hz
            source_depths_mm: Optional (n_sources,) depth from surface
            depth_weighting: 'empirical', 'linear', 'exponential', or 'none'
        """
        self.source_data = np.asarray(source_data)
        self.source_coords_mm = np.asarray(source_coords_mm)
        self.source_to_roi = source_to_roi
        self.roi_names = roi_names
        self.sfreq = sfreq
        self.source_depths_mm = source_depths_mm
        self.depth_weighting = depth_weighting

        # Handle data shape
        if self.source_data.ndim == 2:
            self.n_sources, self.n_times = self.source_data.shape
            self.n_epochs = 1
            self._data_3d = self.source_data[:, np.newaxis, :]
        else:
            self.n_sources, self.n_epochs, self.n_times = self.source_data.shape
            self._data_3d = self.source_data

        # Compute depth weights
        if source_depths_mm is not None and depth_weighting != 'none':
            self._depth_weights = compute_depth_weights(source_depths_mm, depth_weighting)
        else:
            self._depth_weights = np.ones(self.n_sources)

        # Build ROI -> source indices mapping
        self._roi_to_sources: Dict[int, List[int]] = {}
        for src_idx, roi_id in source_to_roi.items():
            if roi_id not in self._roi_to_sources:
                self._roi_to_sources[roi_id] = []
            self._roi_to_sources[roi_id].append(src_idx)

        # Get list of ROIs with sources
        self.roi_ids = sorted(self._roi_to_sources.keys())

        logger.info(f"ROIAnalysis initialized: {self.n_sources} sources, "
                   f"{len(self.roi_ids)} ROIs with sources, "
                   f"depth_weighting={depth_weighting}")

    def get_roi_timeseries(
        self,
        roi_id: int,
        method: str = 'weighted_mean'
    ) -> ROITimeSeries:
        """
        Extract time series for a single ROI.

        Args:
            roi_id: ROI identifier
            method: 'weighted_mean' (depth-weighted), 'mean', or 'pca'

        Returns:
            ROITimeSeries object
        """
        if roi_id not in self._roi_to_sources:
            raise ValueError(f"ROI {roi_id} has no sources")

        source_indices = np.array(self._roi_to_sources[roi_id])
        roi_data = self._data_3d[source_indices]  # (n_roi_sources, n_epochs, n_times)
        weights = self._depth_weights[source_indices]

        if method == 'weighted_mean':
            # Weighted average across sources
            weights_norm = weights / (weights.sum() + 1e-10)
            # Average: (n_roi_sources, n_epochs, n_times) -> (n_epochs, n_times)
            roi_signal = np.einsum('s,set->et', weights_norm, roi_data)

        elif method == 'mean':
            # Simple mean
            roi_signal = roi_data.mean(axis=0)

        elif method == 'pca':
            # First principal component
            flat_data = roi_data.reshape(len(source_indices), -1)
            flat_data = flat_data - flat_data.mean(axis=1, keepdims=True)
            U, S, Vt = np.linalg.svd(flat_data, full_matrices=False)
            roi_signal = Vt[0].reshape(self.n_epochs, self.n_times)

        else:
            raise ValueError(f"Unknown method: {method}")

        # Compute mean depth and accuracy for this ROI
        if self.source_depths_mm is not None:
            mean_depth = np.average(
                self.source_depths_mm[source_indices],
                weights=weights
            )
        else:
            mean_depth = 0.0

        accuracy_weight = np.mean(weights)

        # Squeeze if single epoch
        if self.n_epochs == 1:
            roi_signal = roi_signal.squeeze(0)

        return ROITimeSeries(
            roi_id=roi_id,
            roi_name=self.roi_names.get(roi_id, f'ROI_{roi_id}'),
            data=roi_signal,
            n_sources=len(source_indices),
            source_indices=source_indices,
            mean_depth_mm=mean_depth,
            accuracy_weight=accuracy_weight
        )

    def get_all_roi_timeseries(
        self,
        method: str = 'weighted_mean',
        min_sources: int = 1,
        min_accuracy: float = 0.0
    ) -> Dict[int, ROITimeSeries]:
        """
        Extract time series for all ROIs.

        Args:
            method: Aggregation method ('weighted_mean', 'mean', 'pca')
            min_sources: Minimum sources required per ROI
            min_accuracy: Minimum accuracy weight required

        Returns:
            Dict mapping ROI ID to ROITimeSeries
        """
        results = {}

        for roi_id in self.roi_ids:
            n_sources = len(self._roi_to_sources[roi_id])
            if n_sources < min_sources:
                continue

            roi_ts = self.get_roi_timeseries(roi_id, method)

            if roi_ts.accuracy_weight < min_accuracy:
                continue

            results[roi_id] = roi_ts

        logger.info(f"Extracted {len(results)} ROI time series "
                   f"(min_sources={min_sources}, min_accuracy={min_accuracy})")

        return results

    def get_roi_metadata(self) -> Dict[int, Dict[str, Any]]:
        """
        Get metadata for all ROIs (useful for analysis).

        Returns:
            Dict mapping ROI ID to metadata dict with:
            - name, n_sources, mean_depth_mm, accuracy_weight
        """
        metadata = {}
        for roi_id in self.roi_ids:
            source_indices = np.array(self._roi_to_sources[roi_id])
            weights = self._depth_weights[source_indices]

            if self.source_depths_mm is not None:
                mean_depth = np.average(
                    self.source_depths_mm[source_indices],
                    weights=weights
                )
            else:
                mean_depth = 0.0

            metadata[roi_id] = {
                'name': self.roi_names.get(roi_id, f'ROI_{roi_id}'),
                'n_sources': len(source_indices),
                'mean_depth_mm': mean_depth,
                'accuracy_weight': np.mean(weights),
            }

        return metadata


# =============================================================================
# Convenience Functions
# =============================================================================

def create_roi_analysis_from_pipeline(
    pipeline_results: Dict[str, Any],
    depth_weighting: str = 'empirical'
) -> Optional[ROIAnalysis]:
    """
    Create ROIAnalysis from pipeline output.

    Args:
        pipeline_results: Output from Pipeline.run()
        depth_weighting: Depth weighting method

    Returns:
        ROIAnalysis instance or None if required data is missing
    """
    try:
        # Extract needed data from pipeline results
        stc = pipeline_results['inverse_solution']['stc']
        source_coords = pipeline_results['source_space']['source_coords_mm']

        # Get sampling frequency from epochs or stc
        epochs = pipeline_results['eeg_data'].get('epochs')
        if epochs is not None and hasattr(epochs, 'info'):
            sfreq = epochs.info['sfreq']
        elif hasattr(stc, 'sfreq'):
            sfreq = stc.sfreq
        else:
            sfreq = 1000.0

        # Get ROI mapping - pipeline returns roi_source_mapping (ROI name → source indices)
        roi_source_mapping = pipeline_results['roi_extraction'].get('roi_source_mapping', {})
        roi_labels = pipeline_results['roi_extraction'].get('roi_labels', [])

        # Build ROI name → ID mapping (using index as ID)
        roi_name_to_id = {name: idx for idx, name in enumerate(roi_labels)}

        # Build roi_names: ID → name
        roi_names = {idx: name for idx, name in enumerate(roi_labels)}

        # Convert roi_source_mapping to source_to_roi
        source_to_roi = {}
        for roi_name, source_indices in roi_source_mapping.items():
            roi_id = roi_name_to_id.get(roi_name)
            if roi_id is not None:
                for src_idx in source_indices:
                    if src_idx not in source_to_roi:
                        source_to_roi[src_idx] = roi_id

        # Get depths if available
        source_depths = pipeline_results['source_space'].get('source_depths_mm')

        return ROIAnalysis(
            source_data=stc.data,
            source_coords_mm=source_coords,
            source_to_roi=source_to_roi,
            roi_names=roi_names,
            sfreq=sfreq,
            source_depths_mm=source_depths,
            depth_weighting=depth_weighting
        )
    except (KeyError, TypeError, AttributeError) as e:
        logger.warning(f"Could not create ROIAnalysis from pipeline: {e}")
        return None
