"""
Source-Level Statistical Analysis

Performs statistical analysis at each source location, following the
SPM-LORETA paradigm used in human EEG source imaging.

Key operations:
1. Compute metrics at each source (power, connectivity, etc.)
2. Generate statistical parametric maps
3. Find significant clusters/peaks
4. Support group-level statistics

This mirrors the workflow of:
- SPM (Statistical Parametric Mapping) for fMRI
- KEY Institute eLORETA software for human EEG
- FieldTrip beamformer "virtual electrodes"

Author: Claude Code
Date: 2026-01-26
"""

import numpy as np
from scipy import signal, stats
from scipy.integrate import trapezoid
from scipy.ndimage import label, gaussian_filter
from scipy.spatial import cKDTree
from typing import Tuple, Optional, Dict, List, Any, Union
from dataclasses import dataclass
import logging

logger = logging.getLogger(__name__)


@dataclass
class SourcePeak:
    """Represents a peak in source-level statistical map."""
    index: int
    coord_mm: np.ndarray
    value: float
    depth_mm: Optional[float] = None
    cluster_id: Optional[int] = None
    region_label: Optional[str] = None


@dataclass
class SourceCluster:
    """Represents a cluster of significant sources."""
    cluster_id: int
    indices: np.ndarray
    coords_mm: np.ndarray
    values: np.ndarray
    peak_index: int
    peak_coord_mm: np.ndarray
    peak_value: float
    size: int
    mass: float  # Sum of values in cluster
    centroid_mm: np.ndarray
    region_labels: Optional[List[str]] = None


class SourceStatistics:
    """
    Source-level statistical analysis for EEG source localization.

    Computes metrics at each source location and generates statistical
    parametric maps, following human EEG source imaging conventions.

    Attributes:
        source_data: (n_sources, n_times) source activity time series
        source_coords_mm: (n_sources, 3) source coordinates
        sfreq: Sampling frequency in Hz
        source_depths_mm: Optional depth information for weighting
    """

    def __init__(
        self,
        source_data: np.ndarray,
        source_coords_mm: np.ndarray,
        sfreq: float,
        source_depths_mm: Optional[np.ndarray] = None
    ):
        """
        Initialize source statistics.

        Args:
            source_data: (n_sources, n_times) or (n_sources, n_epochs, n_times)
            source_coords_mm: (n_sources, 3) coordinates
            sfreq: Sampling frequency in Hz
            source_depths_mm: Optional (n_sources,) depth from surface
        """
        self.source_data = np.asarray(source_data)
        self.source_coords_mm = np.asarray(source_coords_mm)
        self.sfreq = sfreq
        self.source_depths_mm = source_depths_mm

        # Handle different data shapes
        if self.source_data.ndim == 2:
            self.n_sources, self.n_times = self.source_data.shape
            self.n_epochs = 1
            self._data_3d = self.source_data[:, np.newaxis, :]
        elif self.source_data.ndim == 3:
            self.n_sources, self.n_epochs, self.n_times = self.source_data.shape
            self._data_3d = self.source_data
        else:
            raise ValueError(f"source_data must be 2D or 3D, got {self.source_data.ndim}D")

        if len(self.source_coords_mm) != self.n_sources:
            raise ValueError(
                f"Mismatch: {len(self.source_coords_mm)} coords vs {self.n_sources} sources"
            )

        # Build spatial index
        self._kdtree = cKDTree(self.source_coords_mm)

        logger.info(f"SourceStatistics initialized: {self.n_sources} sources, "
                   f"{self.n_epochs} epochs, {self.n_times} samples at {sfreq} Hz")

    # =========================================================================
    # SPECTRAL ANALYSIS
    # =========================================================================

    def compute_band_power(
        self,
        band: Union[str, Tuple[float, float]],
        method: str = 'welch',
        relative: bool = False,
        log_transform: bool = True
    ) -> np.ndarray:
        """
        Compute spectral power in frequency band at each source.

        Args:
            band: Frequency band name ('theta', 'alpha', 'beta', 'low_gamma', 'high_gamma')
                  or tuple (low_hz, high_hz)
            method: 'welch' or 'multitaper'
            relative: If True, compute relative power (band / total)
            log_transform: If True, return log10(power)

        Returns:
            (n_sources,) array of band power values
        """
        # Define standard bands
        BANDS = {
            'delta': (1, 4),
            'theta': (4, 10),
            'alpha': (10, 13),
            'beta': (13, 30),
            'low_gamma': (30, 55),
            'high_gamma': (65, 100),
        }

        if isinstance(band, str):
            if band.lower() not in BANDS:
                raise ValueError(f"Unknown band '{band}'. Choose from {list(BANDS.keys())}")
            fmin, fmax = BANDS[band.lower()]
        else:
            fmin, fmax = band

        logger.info(f"Computing {band} power ({fmin}-{fmax} Hz) using {method}")

        # Compute PSD at each source
        band_power = np.zeros(self.n_sources)
        total_power = np.zeros(self.n_sources) if relative else None

        nperseg = min(256, self.n_times)

        for i in range(self.n_sources):
            # Average across epochs
            data = self._data_3d[i].mean(axis=0)  # (n_times,)

            if method == 'welch':
                freqs, psd = signal.welch(data, fs=self.sfreq, nperseg=nperseg)
            elif method == 'multitaper':
                # Simple multitaper approximation using multiple windows
                freqs, psd = signal.welch(
                    data, fs=self.sfreq, nperseg=nperseg,
                    window='dpss', return_onesided=True
                )
            else:
                raise ValueError(f"Unknown method: {method}")

            # Integrate power in band
            band_mask = (freqs >= fmin) & (freqs <= fmax)
            band_power[i] = trapezoid(psd[band_mask], freqs[band_mask])

            if relative:
                total_power[i] = trapezoid(psd, freqs)

        if relative:
            band_power = band_power / (total_power + 1e-10)

        if log_transform:
            band_power = np.log10(band_power + 1e-10)

        return band_power

    def compute_all_bands(
        self,
        bands: Optional[Dict[str, Tuple[float, float]]] = None,
        **kwargs
    ) -> Dict[str, np.ndarray]:
        """
        Compute power for multiple frequency bands.

        Args:
            bands: Dict of band_name -> (fmin, fmax). Uses defaults if None.
            **kwargs: Additional arguments passed to compute_band_power

        Returns:
            Dict mapping band name to (n_sources,) power array
        """
        if bands is None:
            bands = {
                'delta': (1, 4),
                'theta': (4, 10),
                'alpha': (10, 13),
                'beta': (13, 30),
                'low_gamma': (30, 55),
                'high_gamma': (65, 100),
            }

        results = {}
        for name, (fmin, fmax) in bands.items():
            results[name] = self.compute_band_power((fmin, fmax), **kwargs)

        return results

    # =========================================================================
    # CONNECTIVITY ANALYSIS
    # =========================================================================

    def compute_seed_connectivity(
        self,
        seed_coord_mm: np.ndarray,
        method: str = 'correlation',
        band: Optional[Union[str, Tuple[float, float]]] = None
    ) -> np.ndarray:
        """
        Compute connectivity from a seed location to all other sources.

        Args:
            seed_coord_mm: (3,) seed coordinate
            method: 'correlation', 'coherence', or 'icoh'
            band: Optional frequency band for coherence methods

        Returns:
            (n_sources,) connectivity values
        """
        # Find nearest source to seed
        _, seed_idx = self._kdtree.query(seed_coord_mm, k=1)
        seed_data = self._data_3d[seed_idx].mean(axis=0)

        logger.info(f"Computing {method} connectivity from seed at {seed_coord_mm}")

        connectivity = np.zeros(self.n_sources)

        for i in range(self.n_sources):
            target_data = self._data_3d[i].mean(axis=0)

            if method == 'correlation':
                connectivity[i] = np.corrcoef(seed_data, target_data)[0, 1]

            elif method == 'coherence':
                f, coh = signal.coherence(seed_data, target_data, fs=self.sfreq)
                if band is not None:
                    fmin, fmax = self._parse_band(band)
                    mask = (f >= fmin) & (f <= fmax)
                    connectivity[i] = np.mean(coh[mask])
                else:
                    connectivity[i] = np.mean(coh)

            elif method == 'icoh':
                # Imaginary coherence
                f, cxy = signal.csd(seed_data, target_data, fs=self.sfreq)
                f, pxx = signal.welch(seed_data, fs=self.sfreq)
                f, pyy = signal.welch(target_data, fs=self.sfreq)
                coherency = cxy / np.sqrt(pxx * pyy + 1e-10)
                icoh = np.abs(np.imag(coherency))

                if band is not None:
                    fmin, fmax = self._parse_band(band)
                    mask = (f >= fmin) & (f <= fmax)
                    connectivity[i] = np.mean(icoh[mask])
                else:
                    connectivity[i] = np.mean(icoh)

        return connectivity

    def _parse_band(self, band: Union[str, Tuple[float, float]]) -> Tuple[float, float]:
        """Parse band specification to (fmin, fmax)."""
        BANDS = {
            'delta': (1, 4), 'theta': (4, 10), 'alpha': (10, 13),
            'beta': (13, 30), 'low_gamma': (30, 55), 'high_gamma': (65, 100),
        }
        if isinstance(band, str):
            return BANDS[band.lower()]
        return band

    # =========================================================================
    # STATISTICAL ANALYSIS
    # =========================================================================

    def compute_zscore_map(
        self,
        values: np.ndarray,
        baseline_values: Optional[np.ndarray] = None
    ) -> np.ndarray:
        """
        Convert values to z-scores for statistical mapping.

        Args:
            values: (n_sources,) values to normalize
            baseline_values: Optional baseline for z-scoring (uses values if None)

        Returns:
            (n_sources,) z-score map
        """
        if baseline_values is None:
            baseline_values = values

        mean = np.mean(baseline_values)
        std = np.std(baseline_values)

        if std < 1e-10:
            logger.warning("Near-zero standard deviation in z-score computation")
            return np.zeros_like(values)

        return (values - mean) / std

    def threshold_map(
        self,
        values: np.ndarray,
        threshold: float,
        threshold_type: str = 'absolute'
    ) -> np.ndarray:
        """
        Threshold statistical map.

        Args:
            values: (n_sources,) statistical values
            threshold: Threshold value
            threshold_type: 'absolute', 'percentile', or 'zscore'

        Returns:
            (n_sources,) boolean mask of suprathreshold sources
        """
        if threshold_type == 'absolute':
            thresh_value = threshold
        elif threshold_type == 'percentile':
            thresh_value = np.percentile(values, threshold)
        elif threshold_type == 'zscore':
            mean, std = np.mean(values), np.std(values)
            thresh_value = mean + threshold * std
        else:
            raise ValueError(f"Unknown threshold_type: {threshold_type}")

        return values >= thresh_value

    # =========================================================================
    # CLUSTER ANALYSIS
    # =========================================================================

    def find_clusters(
        self,
        values: np.ndarray,
        threshold: float,
        min_cluster_size: int = 3,
        cluster_radius_mm: float = 2.0
    ) -> List[SourceCluster]:
        """
        Find clusters of significant sources using spatial connectivity.

        Args:
            values: (n_sources,) statistical values
            threshold: Threshold for cluster forming
            min_cluster_size: Minimum sources per cluster
            cluster_radius_mm: Maximum distance between cluster members

        Returns:
            List of SourceCluster objects
        """
        logger.info(f"Finding clusters with threshold={threshold}, "
                   f"min_size={min_cluster_size}, radius={cluster_radius_mm}mm")

        # Get suprathreshold sources
        above_thresh = values >= threshold
        suprathresh_indices = np.where(above_thresh)[0]

        if len(suprathresh_indices) == 0:
            logger.info("No suprathreshold sources found")
            return []

        # Build connectivity graph based on spatial proximity
        suprathresh_coords = self.source_coords_mm[suprathresh_indices]
        tree = cKDTree(suprathresh_coords)

        # Find connected components using spatial proximity
        # Start with each source as its own cluster
        cluster_labels = np.arange(len(suprathresh_indices))

        # Union-find for connected components
        def find(x):
            if cluster_labels[x] != x:
                cluster_labels[x] = find(cluster_labels[x])
            return cluster_labels[x]

        def union(x, y):
            px, py = find(x), find(y)
            if px != py:
                cluster_labels[px] = py

        # Connect nearby sources
        pairs = tree.query_pairs(cluster_radius_mm)
        for i, j in pairs:
            union(i, j)

        # Relabel to consecutive integers
        unique_labels = {}
        for i in range(len(suprathresh_indices)):
            root = find(i)
            if root not in unique_labels:
                unique_labels[root] = len(unique_labels)
            cluster_labels[i] = unique_labels[root]

        # Build cluster objects
        clusters = []
        for cluster_id in range(len(unique_labels)):
            mask = cluster_labels == cluster_id
            if np.sum(mask) < min_cluster_size:
                continue

            indices = suprathresh_indices[mask]
            coords = self.source_coords_mm[indices]
            vals = values[indices]

            peak_local_idx = np.argmax(vals)
            peak_idx = indices[peak_local_idx]

            cluster = SourceCluster(
                cluster_id=cluster_id,
                indices=indices,
                coords_mm=coords,
                values=vals,
                peak_index=peak_idx,
                peak_coord_mm=coords[peak_local_idx],
                peak_value=vals[peak_local_idx],
                size=len(indices),
                mass=np.sum(vals),
                centroid_mm=np.mean(coords, axis=0)
            )
            clusters.append(cluster)

        # Sort by peak value
        clusters.sort(key=lambda c: c.peak_value, reverse=True)

        logger.info(f"Found {len(clusters)} clusters (min_size={min_cluster_size})")

        return clusters

    def find_peaks(
        self,
        values: np.ndarray,
        min_distance_mm: float = 3.0,
        threshold_percentile: float = 95,
        n_peaks: Optional[int] = None
    ) -> List[SourcePeak]:
        """
        Find local maxima (peaks) in the statistical map.

        Args:
            values: (n_sources,) statistical values
            min_distance_mm: Minimum distance between peaks
            threshold_percentile: Only consider sources above this percentile
            n_peaks: Maximum number of peaks to return

        Returns:
            List of SourcePeak objects, sorted by value
        """
        threshold = np.percentile(values, threshold_percentile)
        above_thresh = values >= threshold

        peaks = []
        remaining_mask = above_thresh.copy()

        while np.any(remaining_mask):
            # Find global maximum among remaining sources
            remaining_indices = np.where(remaining_mask)[0]
            remaining_values = values[remaining_indices]
            local_max_idx = np.argmax(remaining_values)
            peak_idx = remaining_indices[local_max_idx]

            # Create peak object
            peak = SourcePeak(
                index=peak_idx,
                coord_mm=self.source_coords_mm[peak_idx],
                value=values[peak_idx],
                depth_mm=self.source_depths_mm[peak_idx] if self.source_depths_mm is not None else None
            )
            peaks.append(peak)

            if n_peaks is not None and len(peaks) >= n_peaks:
                break

            # Exclude sources within min_distance of this peak
            distances = np.linalg.norm(
                self.source_coords_mm - peak.coord_mm,
                axis=1
            )
            remaining_mask &= (distances > min_distance_mm)

        return peaks

    # =========================================================================
    # SPATIAL SMOOTHING
    # =========================================================================

    def smooth_map(
        self,
        values: np.ndarray,
        fwhm_mm: float = 3.0
    ) -> np.ndarray:
        """
        Apply spatial smoothing to statistical map.

        Uses Gaussian kernel based on source distances.

        Args:
            values: (n_sources,) values to smooth
            fwhm_mm: Full width at half maximum of Gaussian kernel

        Returns:
            (n_sources,) smoothed values
        """
        # Convert FWHM to sigma
        sigma_mm = fwhm_mm / (2 * np.sqrt(2 * np.log(2)))

        # Compute smoothed values using distance-weighted averaging
        smoothed = np.zeros_like(values)

        for i in range(self.n_sources):
            distances = np.linalg.norm(
                self.source_coords_mm - self.source_coords_mm[i],
                axis=1
            )
            weights = np.exp(-distances**2 / (2 * sigma_mm**2))
            weights /= weights.sum()
            smoothed[i] = np.sum(weights * values)

        return smoothed

    # =========================================================================
    # GROUP ANALYSIS
    # =========================================================================

    @staticmethod
    def group_ttest(
        group1_maps: np.ndarray,
        group2_maps: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Perform source-level two-sample t-test.

        Args:
            group1_maps: (n_subjects1, n_sources) maps for group 1
            group2_maps: (n_subjects2, n_sources) maps for group 2

        Returns:
            Tuple of (t_values, p_values), each (n_sources,)
        """
        n_sources = group1_maps.shape[1]
        t_values = np.zeros(n_sources)
        p_values = np.zeros(n_sources)

        for i in range(n_sources):
            t, p = stats.ttest_ind(group1_maps[:, i], group2_maps[:, i])
            t_values[i] = t
            p_values[i] = p

        return t_values, p_values

    @staticmethod
    def group_correlation(
        maps: np.ndarray,
        covariate: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Correlate source-level maps with a covariate.

        Args:
            maps: (n_subjects, n_sources) individual maps
            covariate: (n_subjects,) covariate values

        Returns:
            Tuple of (r_values, p_values), each (n_sources,)
        """
        n_sources = maps.shape[1]
        r_values = np.zeros(n_sources)
        p_values = np.zeros(n_sources)

        for i in range(n_sources):
            r, p = stats.pearsonr(maps[:, i], covariate)
            r_values[i] = r
            p_values[i] = p

        return r_values, p_values

    def __repr__(self) -> str:
        return (
            f"SourceStatistics(n_sources={self.n_sources}, "
            f"n_epochs={self.n_epochs}, n_times={self.n_times}, "
            f"sfreq={self.sfreq}Hz)"
        )
