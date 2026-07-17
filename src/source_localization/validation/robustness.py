"""
Robustness testing for source localization validation.

This module provides tools for testing how localization accuracy varies with
signal and noise parameters, validating the fundamental physics of the pipeline.

Classes
-------
RobustnessTest
    Run robustness tests for SNR, noise level, and amplitude sensitivity.

Examples
--------
>>> from source_localization.validation.robustness import RobustnessTest
>>>
>>> # Initialize from pipeline directory
>>> test = RobustnessTest.from_pipeline_dir('/path/to/pipeline/results')
>>>
>>> # Run SNR sensitivity test
>>> snr_results = test.run_snr_test(snr_range=(-20, 40), n_trials=10)
>>>
>>> # Run noise level test
>>> noise_results = test.run_noise_test(noise_range=(0.01, 10000), n_trials=10)
>>>
>>> # Generate plots
>>> test.plot_results(output_dir='/path/to/figures')
>>>
>>> # Get summary statistics
>>> summary = test.get_summary()
"""

import numpy as np
import pickle
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union
from dataclasses import dataclass, field
from scipy import stats
from scipy.spatial import cKDTree
from scipy.spatial.distance import cdist
import json
from datetime import datetime
import warnings

import mne

from .noise import NOISE_TYPES
from .simulation import DipoleSimulator
from ..inverse.methods import apply_inverse_sLORETA, apply_inverse_MNE, apply_inverse_dSPM


__all__ = ['RobustnessTest', 'RobustnessResults']


@dataclass
class RobustnessResults:
    """Container for robustness test results."""

    test_type: str  # 'snr', 'noise', 'amplitude', 'noise_type'
    parameter_values: List[Union[float, str]]
    errors: Dict[Union[float, str], List[float]]  # parameter -> list of errors
    n_sources: int
    n_trials_per_position: int
    n_positions: int
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())

    @property
    def means(self) -> Dict[float, float]:
        """Mean error for each parameter value."""
        return {k: np.mean(v) for k, v in self.errors.items()}

    @property
    def stds(self) -> Dict[float, float]:
        """Standard deviation for each parameter value."""
        return {k: np.std(v) for k, v in self.errors.items()}

    @property
    def medians(self) -> Dict[float, float]:
        """Median error for each parameter value."""
        return {k: np.median(v) for k, v in self.errors.items()}

    @property
    def is_categorical(self) -> bool:
        """True when the swept parameter is a label (e.g. noise_type), not a number."""
        return any(isinstance(p, str) for p in self.parameter_values)

    @property
    def correlation(self) -> float:
        """Correlation between parameter and mean error.

        Undefined (NaN) for categorical sweeps such as noise_type, where the
        parameter has no ordering to correlate against.
        """
        if self.is_categorical:
            return np.nan

        params = sorted(self.errors.keys())
        means = [np.mean(self.errors[p]) for p in params]

        # For noise test, use log scale
        if self.test_type == 'noise':
            params = np.log10(params)

        if len(params) < 2:
            return np.nan
        return np.corrcoef(params, means)[0, 1]

    def to_dict(self) -> dict:
        """Convert to dictionary for JSON serialization."""
        # Convert numpy types to native Python types
        def to_native(x):
            if isinstance(x, (np.integer, np.floating)):
                return float(x)
            elif isinstance(x, np.ndarray):
                return x.tolist()
            elif isinstance(x, list):
                return [to_native(v) for v in x]
            return x

        return {
            'test_type': self.test_type,
            'parameter_values': [
                p if isinstance(p, str) else float(p) for p in self.parameter_values
            ],
            'errors': {str(k): to_native(v) for k, v in self.errors.items()},
            'means': {str(k): float(v) for k, v in self.means.items()},
            'stds': {str(k): float(v) for k, v in self.stds.items()},
            'medians': {str(k): float(v) for k, v in self.medians.items()},
            'correlation': float(self.correlation) if not np.isnan(self.correlation) else None,
            'n_sources': int(self.n_sources),
            'n_trials_per_position': int(self.n_trials_per_position),
            'n_positions': int(self.n_positions),
            'timestamp': self.timestamp
        }


class RobustnessTest:
    """
    Robustness testing for source localization pipelines.

    Tests how localization accuracy varies with signal and noise parameters
    to validate the fundamental physics of the forward/inverse pipeline.

    Parameters
    ----------
    fwd : mne.Forward
        Forward solution (leadfield matrix)
    src : mne.SourceSpaces
        Source space definition
    info : mne.Info
        EEG channel information
    inverse_method : str, default='sLORETA'
        Inverse method to use ('sLORETA', 'MNE', 'dSPM')
    inverse_snr : float, default=3.0
        SNR assumption for inverse solution regularization
    verbose : bool, default=True
        Print progress messages

    Attributes
    ----------
    simulator : DipoleSimulator
        Dipole simulator instance
    source_pos_mm : ndarray
        Source positions in mm
    n_sources : int
        Number of sources in source space
    results : dict
        Dictionary of RobustnessResults from completed tests

    Examples
    --------
    >>> test = RobustnessTest(fwd, src, info)
    >>> snr_results = test.run_snr_test()
    >>> print(f"SNR-Error correlation: {snr_results.correlation:.3f}")
    """

    def __init__(
        self,
        fwd: mne.Forward,
        src: mne.SourceSpaces,
        info: mne.Info,
        inverse_method: str = 'sLORETA',
        inverse_snr: float = 3.0,
        verbose: bool = True
    ):
        self.fwd = fwd
        self.src = src
        self.info = info
        self.inverse_method = inverse_method.lower()
        self.inverse_snr = inverse_snr
        self.verbose = verbose

        # Create simulator
        self.simulator = DipoleSimulator(fwd, info, src, verbose=False)

        # Get source positions
        self.source_pos_mm = src[0]['rr'] * 1000
        self.n_sources = len(self.source_pos_mm)

        # Get electrode positions for depth calculation
        self.electrode_pos_mm = np.array([
            ch['loc'][:3] for ch in info['chs'] if ch['kind'] == 2
        ]) * 1000

        # Compute source depths (distance to nearest electrode)
        distances = cdist(self.source_pos_mm, self.electrode_pos_mm)
        self.source_depths = distances.min(axis=1)

        # Storage for results
        self.results: Dict[str, RobustnessResults] = {}

        # Suppress MNE verbose output
        mne.set_log_level('ERROR')

    @classmethod
    def from_pipeline_dir(
        cls,
        pipeline_dir: Union[str, Path],
        inverse_method: str = 'sLORETA',
        inverse_snr: float = 3.0,
        verbose: bool = True
    ) -> 'RobustnessTest':
        """
        Create RobustnessTest from a pipeline output directory.

        Parameters
        ----------
        pipeline_dir : str or Path
            Path to pipeline output directory containing data/*.pkl files
        inverse_method : str, default='sLORETA'
            Inverse method to use
        inverse_snr : float, default=3.0
            SNR assumption for regularization
        verbose : bool, default=True
            Print progress messages

        Returns
        -------
        RobustnessTest
            Initialized robustness test instance
        """
        pipeline_dir = Path(pipeline_dir)
        data_dir = pipeline_dir / 'data'

        # Load forward solution
        fwd_files = list(data_dir.glob('step*_forward*.pkl'))
        if not fwd_files:
            raise FileNotFoundError(f"No forward solution found in {data_dir}")
        with open(fwd_files[0], 'rb') as f:
            fwd = pickle.load(f)

        # Load source space
        src_files = list(data_dir.glob('step*_source_space.pkl'))
        if not src_files:
            raise FileNotFoundError(f"No source space found in {data_dir}")
        with open(src_files[0], 'rb') as f:
            src = pickle.load(f)

        # Get info from forward model
        info = fwd['info']

        return cls(fwd, src, info, inverse_method, inverse_snr, verbose)

    def _select_test_positions(
        self,
        n_positions: int = 4,
        depth_percentiles: Optional[List[float]] = None
    ) -> np.ndarray:
        """Select test positions at different depths."""
        if depth_percentiles is None:
            # Default: spread across depth range
            depth_percentiles = np.linspace(20, 80, n_positions)

        test_indices = [
            np.argmin(np.abs(self.source_depths - np.percentile(self.source_depths, p)))
            for p in depth_percentiles
        ]
        return np.array(test_indices)

    def _apply_inverse(self, eeg_data: np.ndarray) -> np.ndarray:
        """Apply inverse solution and return source activity."""
        import io
        import sys

        # Create evoked object
        info_with_sfreq = mne.create_info(
            ch_names=self.info.ch_names,
            sfreq=256.0,
            ch_types=['eeg'] * len(self.info.ch_names)
        )
        evoked = mne.EvokedArray(eeg_data, info_with_sfreq, tmin=0)

        # Suppress output
        old_stdout = sys.stdout
        sys.stdout = io.StringIO()

        try:
            if self.inverse_method == 'sloreta':
                source_activity, _, _ = apply_inverse_sLORETA(
                    self.fwd, self.info, evoked=evoked, snr=self.inverse_snr
                )
            elif self.inverse_method == 'mne':
                source_activity, _, _ = apply_inverse_MNE(
                    self.fwd, self.info, evoked=evoked, snr=self.inverse_snr
                )
            elif self.inverse_method == 'dspm':
                source_activity, _, _ = apply_inverse_dSPM(
                    self.fwd, self.info, evoked=evoked, snr=self.inverse_snr
                )
            else:
                raise ValueError(f"Unknown inverse method: {self.inverse_method}")
        finally:
            sys.stdout = old_stdout

        return source_activity

    def _find_peak_and_error(
        self,
        source_activity: np.ndarray,
        true_position: np.ndarray
    ) -> float:
        """Find peak source and compute localization error."""
        n_orient = source_activity.shape[0] // self.n_sources

        if n_orient == 3:
            # Free orientation - take norm across orientations
            sa_3d = source_activity.reshape(self.n_sources, 3, -1)
            sa_norm = np.linalg.norm(sa_3d, axis=1).mean(axis=1)
        else:
            sa_norm = np.abs(source_activity).mean(axis=1)

        peak_idx = np.argmax(sa_norm)
        peak_pos = self.source_pos_mm[peak_idx]

        return np.linalg.norm(peak_pos - true_position)

    def _source_activity_norm(self, source_activity: np.ndarray) -> np.ndarray:
        """Collapse (oriented, time) source activity to one magnitude per source."""
        n_orient = source_activity.shape[0] // self.n_sources
        if n_orient == 3:
            sa_3d = source_activity.reshape(self.n_sources, 3, -1)
            return np.linalg.norm(sa_3d, axis=1).mean(axis=1)
        return np.abs(source_activity).mean(axis=1)

    def _find_two_peaks_and_errors(
        self,
        source_activity: np.ndarray,
        true_positions: np.ndarray,
        suppression_mm: float
    ) -> Tuple[float, float]:
        """
        Recover two source peaks and match them to two true positions.

        The strongest source is the first peak. Before taking the second, sources
        within ``suppression_mm`` of the first are masked out — otherwise the two
        largest values are usually adjacent grid points of a single blob, not two
        distinct sources. The two recovered peaks are then assigned to the two
        true positions by whichever pairing minimizes total distance (a 2x2
        assignment), since the inverse has no inherent source ordering.

        Parameters
        ----------
        source_activity : ndarray
            Inverse-solution source activity.
        true_positions : ndarray, shape (2, 3)
            The two true dipole positions in mm.
        suppression_mm : float
            Radius around the first peak to exclude when finding the second.

        Returns
        -------
        (error1_mm, error2_mm) : tuple of float
            Per-dipole localization errors, ordered to match ``true_positions``.
        """
        sa_norm = self._source_activity_norm(source_activity)

        peak1_idx = int(np.argmax(sa_norm))
        peak1_pos = self.source_pos_mm[peak1_idx]

        # Mask a neighborhood of the first peak, then take the next strongest.
        dist_to_peak1 = np.linalg.norm(self.source_pos_mm - peak1_pos, axis=1)
        masked = sa_norm.copy()
        masked[dist_to_peak1 <= suppression_mm] = -np.inf
        if not np.isfinite(masked).any():
            # Suppression swallowed every source (tiny grid); fall back to the
            # global second-best so we still return two distinct peaks.
            masked = sa_norm.copy()
            masked[peak1_idx] = -np.inf
        peak2_idx = int(np.argmax(masked))
        peak2_pos = self.source_pos_mm[peak2_idx]

        recovered = np.array([peak1_pos, peak2_pos])
        true = np.asarray(true_positions)

        # Two possible assignments; pick the one with smaller total error.
        straight = (np.linalg.norm(recovered[0] - true[0])
                    + np.linalg.norm(recovered[1] - true[1]))
        swapped = (np.linalg.norm(recovered[1] - true[0])
                   + np.linalg.norm(recovered[0] - true[1]))
        if swapped < straight:
            recovered = recovered[::-1]

        return (
            float(np.linalg.norm(recovered[0] - true[0])),
            float(np.linalg.norm(recovered[1] - true[1])),
        )

    def _get_source_kdtree(self) -> cKDTree:
        """KD-tree over source positions, built once and cached."""
        tree = getattr(self, '_source_kdtree_cache', None)
        if tree is None:
            tree = cKDTree(self.source_pos_mm)
            self._source_kdtree_cache = tree
        return tree

    def _median_grid_spacing(self) -> float:
        """Median nearest-neighbor spacing of the source grid (its resolution floor)."""
        val = getattr(self, '_median_spacing_cache', None)
        if val is None:
            tree = self._get_source_kdtree()
            d, _ = tree.query(self.source_pos_mm, k=2)  # col 1 = nearest neighbor
            val = float(np.median(d[:, 1]))
            self._median_spacing_cache = val
        return val

    def _local_maxima(self, values: np.ndarray, radius_mm: float) -> np.ndarray:
        """Indices of sources whose value is >= every neighbor within radius_mm."""
        tree = self._get_source_kdtree()
        neighbor_lists = tree.query_ball_point(self.source_pos_mm, r=radius_mm)
        maxima = []
        for i, nb in enumerate(neighbor_lists):
            others = [j for j in nb if j != i]
            if not others or values[i] >= values[others].max():
                maxima.append(i)
        return np.array(maxima, dtype=int)

    def _segment_trough(
        self,
        p1: np.ndarray,
        p2: np.ndarray,
        values: np.ndarray,
        n_samples: int = 21,
        interior: Tuple[float, float] = (0.15, 0.85)
    ) -> float:
        """
        Minimum reconstructed value along the segment between two peaks.

        Samples the interior of the p1->p2 line (endpoints excluded, since those
        are the peaks themselves) and takes the nearest source's value at each
        sample. The minimum is the saddle/trough between the two peaks — the
        quantity a Rayleigh-style resolution criterion tests.
        """
        tree = self._get_source_kdtree()
        ts = np.linspace(interior[0], interior[1], n_samples)
        pts = p1[None, :] + ts[:, None] * (p2 - p1)[None, :]
        _, idx = tree.query(pts)
        return float(values[idx].min())

    def _resolve_two_sources(
        self,
        source_activity: np.ndarray,
        true_positions: np.ndarray,
        neighbor_radius_mm: Optional[float] = None,
        exclude_radius_mm: Optional[float] = None,
        saddle_ratio: float = 0.8,
        prominence_frac: float = 0.5,
        max_match_mm: Optional[float] = None
    ) -> Dict[str, Any]:
        """
        Decide whether a reconstruction shows two distinct sources or one blob.

        This answers "can we tell there are two sources?" — the resolvability
        question — rather than "how accurately are they placed?" (localization).
        A pair counts as *resolved* when all three hold:

        1. **Two prominent maxima:** the global peak, plus a second local maximum
           farther than ``exclude_radius_mm`` from it AND at least
           ``prominence_frac`` x the global peak's height. The prominence gate is
           essential: without it, a small noise ripple counts as a "second
           source" and a single dipole is mislabeled as two ~40-50% of the time
           (measured). A real second source produces a comparable lobe.
        2. **A saddle between them:** the trough along the connecting segment
           falls to <= ``saddle_ratio`` x the weaker peak. This is the EEG
           analogue of the Rayleigh criterion: two merged blobs have no dip.
        3. **Correspondence (parameter-free):** the two maxima fall nearest to
           *different* true sources — one lobe per source (a Voronoi assignment).
           This rules out two spurious peaks on the same source without
           re-introducing a localization-accuracy tolerance. A fixed millimeter
           tolerance is deliberately avoided here: making it tighter than the
           localization error would reject genuine two-lobe detections and turn
           this back into a localization test.

        Parameters
        ----------
        source_activity : ndarray
            Inverse-solution source activity.
        true_positions : ndarray, shape (2, 3)
            The two true dipole positions in mm.
        neighbor_radius_mm : float, optional
            Radius defining "neighbor" for the local-maximum test. Defaults to
            1.5x the median grid spacing.
        exclude_radius_mm : float, optional
            A second peak must be at least this far from the first. Defaults to
            1.5x the median grid spacing (i.e. beyond the immediate neighborhood).
        saddle_ratio : float, default=0.8
            Trough-to-weaker-peak ratio below which a dip counts as a saddle.
            Rayleigh's criterion corresponds to ~0.81.
        prominence_frac : float, default=0.5
            The second peak must be at least this fraction of the global peak's
            height to count. Suppresses noise-induced spurious second peaks.
        max_match_mm : float, optional
            Optional loose sanity cap: if set, each peak must also be within this
            distance of its assigned true source. Off by default — correspondence
            is the parameter-free distinct-nearest rule. Set generously (e.g. one
            source separation) only to reject peaks in an entirely wrong region.

        Returns
        -------
        dict
            ``resolved`` (bool) plus diagnostics: ``n_candidate_peaks``,
            ``peak2_prominence``, ``saddle_ratio_observed``, ``saddle_ok``,
            ``distinct_sources``, ``match_max_mm``, ``match_ok``, ``peak1_mm``,
            ``peak2_mm``.
        """
        sa = self._source_activity_norm(source_activity)
        spacing = self._median_grid_spacing()
        if neighbor_radius_mm is None:
            neighbor_radius_mm = 1.5 * spacing
        if exclude_radius_mm is None:
            exclude_radius_mm = 1.5 * spacing

        peak1_idx = int(np.argmax(sa))
        val1 = float(sa[peak1_idx])
        pos1 = self.source_pos_mm[peak1_idx]

        result = {
            'resolved': False,
            'n_candidate_peaks': 0,
            'peak2_prominence': None,
            'saddle_ratio_observed': None,
            'saddle_ok': False,
            'distinct_sources': False,
            'match_max_mm': None,
            'match_ok': False,
            'peak1_mm': pos1.tolist(),
            'peak2_mm': None,
        }

        maxima = self._local_maxima(sa, neighbor_radius_mm)
        dist_from_peak1 = np.linalg.norm(self.source_pos_mm[maxima] - pos1, axis=1)
        # A second peak must be far enough from the first AND prominent enough
        # (a real lobe, not a noise ripple).
        far_enough = dist_from_peak1 > exclude_radius_mm
        prominent = sa[maxima] >= prominence_frac * val1
        candidates = maxima[far_enough & prominent]
        result['n_candidate_peaks'] = int(candidates.size)
        if candidates.size == 0:
            return result  # no prominent second lobe — one source, not resolved

        peak2_idx = int(candidates[np.argmax(sa[candidates])])
        val2 = float(sa[peak2_idx])
        pos2 = self.source_pos_mm[peak2_idx]
        result['peak2_mm'] = pos2.tolist()
        result['peak2_prominence'] = float(val2 / val1) if val1 > 0 else None

        trough = self._segment_trough(pos1, pos2, sa)
        weaker = min(val1, val2)
        observed_ratio = trough / weaker if weaker > 0 else np.inf
        result['saddle_ratio_observed'] = float(observed_ratio)
        saddle_ok = observed_ratio <= saddle_ratio
        result['saddle_ok'] = bool(saddle_ok)

        # Correspondence (parameter-free): each peak nearest a DIFFERENT true
        # source — one lobe per source. Distances kept only as diagnostics /
        # optional sanity cap; they are NOT a localization-accuracy gate.
        true = np.asarray(true_positions)
        d1 = np.linalg.norm(true - pos1, axis=1)  # peak1 to each true source
        d2 = np.linalg.norm(true - pos2, axis=1)  # peak2 to each true source
        nearest1, nearest2 = int(np.argmin(d1)), int(np.argmin(d2))
        distinct_sources = nearest1 != nearest2
        result['distinct_sources'] = bool(distinct_sources)
        # Assign each peak to its own nearest source (they differ when distinct).
        match_max = float(max(d1[nearest1], d2[nearest2]))
        result['match_max_mm'] = match_max
        within_cap = (max_match_mm is None) or (match_max <= max_match_mm)
        result['match_ok'] = bool(distinct_sources and within_cap)

        result['resolved'] = bool(saddle_ok and distinct_sources and within_cap)
        return result

    def _select_dipole_pairs(
        self,
        separations_mm: List[float],
        n_pairs: int,
        tolerance_mm: float = 1.0
    ) -> Dict[float, List[Tuple[int, int]]]:
        """
        Find source-index pairs at each requested separation.

        For each base position (spread across depth) and each target separation,
        pick the source whose distance is closest to the target; keep it only if
        within ``tolerance_mm``. Returns a dict mapping requested separation to a
        list of (idx1, idx2) pairs. Separations with no realizable pair on this
        grid map to an empty list (the caller logs and skips them).
        """
        base_indices = self._select_test_positions(n_pairs)
        pairs: Dict[float, List[Tuple[int, int]]] = {s: [] for s in separations_mm}

        for base_idx in base_indices:
            base_pos = self.source_pos_mm[base_idx]
            dists = np.linalg.norm(self.source_pos_mm - base_pos, axis=1)
            for sep in separations_mm:
                partner_idx = int(np.argmin(np.abs(dists - sep)))
                if partner_idx == base_idx:
                    continue
                if abs(dists[partner_idx] - sep) <= tolerance_mm:
                    pairs[sep].append((int(base_idx), partner_idx))

        return pairs

    def _select_depth_matched_pairs(
        self,
        separations_mm: List[float],
        depth_bins: List[Tuple[str, float, float]],
        n_pairs: int,
        separation_tolerance_mm: float = 1.0,
        depth_tolerance_mm: float = 1.0
    ) -> Dict[Tuple[str, float], List[Tuple[int, int]]]:
        """
        Find source pairs at each (depth bin, separation), both sources at matched depth.

        Localization resolution degrades with depth (distance to the nearest
        electrode): a deep source produces a broad, low-amplitude scalp pattern,
        so the inverse smears it and two close deep sources blur into one. To
        make "the pair's depth" a well-defined axis, both dipoles are required to
        sit in the same depth bin (their depths within ``depth_tolerance_mm``),
        rather than pairing a shallow source with a deep one.

        Parameters
        ----------
        separations_mm : list of float
            Target inter-dipole separations.
        depth_bins : list of (label, lo_pct, hi_pct)
            Depth bins as percentile ranges of the source-depth distribution.
        n_pairs : int
            Max base positions per (bin, separation) cell.
        separation_tolerance_mm : float, default=1.0
            Allowed deviation of a grid pair from the target separation.
        depth_tolerance_mm : float, default=1.0
            Allowed depth difference between the two sources of a pair.

        Returns
        -------
        dict of (depth_label, separation) -> list of (idx1, idx2)
            Cells with no realizable pair map to an empty list.
        """
        depth = self.source_depths
        pairs: Dict[Tuple[str, float], List[Tuple[int, int]]] = {
            (label, sep): [] for (label, _, _) in depth_bins for sep in separations_mm
        }

        for label, lo_pct, hi_pct in depth_bins:
            lo, hi = np.percentile(depth, lo_pct), np.percentile(depth, hi_pct)
            bin_indices = np.where((depth >= lo) & (depth <= hi))[0]

            for sep in separations_mm:
                count = 0
                for base_idx in bin_indices:
                    if count >= n_pairs:
                        break
                    dists = np.linalg.norm(
                        self.source_pos_mm - self.source_pos_mm[base_idx], axis=1
                    )
                    # Candidate partners: right separation AND matched depth.
                    near_sep = np.abs(dists - sep) <= separation_tolerance_mm
                    near_depth = np.abs(depth - depth[base_idx]) <= depth_tolerance_mm
                    candidates = np.where(near_sep & near_depth)[0]
                    candidates = candidates[candidates != base_idx]
                    if candidates.size == 0:
                        continue
                    partner_idx = int(candidates[np.argmin(np.abs(dists[candidates] - sep))])
                    pairs[(label, sep)].append((int(base_idx), partner_idx))
                    count += 1

        return pairs

    #: Default depth bins (label, lo_percentile, hi_percentile). A dedicated
    #: very-shallow bin isolates the sources closest to the electrodes, where
    #: even two-dipole resolution approaches the single-dipole floor (~1 mm).
    DEFAULT_DEPTH_BINS = [
        ('very_shallow', 0, 15),
        ('shallow', 15, 40),
        ('mid', 40, 70),
        ('deep', 70, 100),
    ]

    def run_two_dipole_test(
        self,
        separations_mm: Optional[List[float]] = None,
        noise_types: Optional[List[str]] = None,
        depth_bins: Optional[List[Tuple[str, float, float]]] = None,
        snr_db: float = 10.0,
        n_pairs: int = 4,
        n_trials: int = 10,
        amplitude_nAm: float = 50.0,
        spatial_scale_mm: float = 3.0,
        temporal_exponent: float = 1.0,
        suppression_mm: Optional[float] = None,
        separation_tolerance_mm: float = 1.0,
        depth_tolerance_mm: float = 1.0
    ) -> Dict[str, Any]:
        """
        Test per-dipole localization error for two simultaneous sources.

        A single DC dipole is the friendliest case for a min-norm inverse; under
        superposition the smooth solution smears the two sources together, and
        that smearing worsens both as they get closer AND as they get deeper
        (further from the electrodes). This sweep crosses three axes — depth,
        separation, and noise structure — so the realistic bound is reported as a
        grid rather than a single pooled number that hides the depth dependence.

        Both dipoles of a pair are placed at matched depth (see
        :meth:`_select_depth_matched_pairs`), so each pair has a well-defined
        depth. All noise types share positions and seeds within a cell, so the
        white-vs-colored comparison is paired.

        Parameters
        ----------
        separations_mm : list of float, optional
            Target inter-dipole separations. Default: [2, 4, 6, 8].
        noise_types : list of str, optional
            Noise structures. Default: all of
            :data:`~source_localization.validation.noise.NOISE_TYPES`.
        depth_bins : list of (label, lo_pct, hi_pct), optional
            Depth bins as percentile ranges of source depth. Default:
            :data:`DEFAULT_DEPTH_BINS` (very_shallow / shallow / mid / deep).
        snr_db : float, default=10.0
            Fixed combined-signal SNR.
        n_pairs : int, default=4
            Max base positions per (depth bin, separation) cell.
        n_trials : int, default=10
            Trials per pair.
        amplitude_nAm : float, default=50.0
            Amplitude of each dipole (equal DC sources).
        spatial_scale_mm, temporal_exponent : float
            Noise-shaping parameters, as in :meth:`run_noise_type_test`.
        suppression_mm : float, optional
            Radius for masking the first peak before finding the second. Defaults
            to half the smallest separation.
        separation_tolerance_mm : float, default=1.0
            Allowed deviation of a grid pair from the target separation.
        depth_tolerance_mm : float, default=1.0
            Allowed depth difference between the two sources of a pair.

        Returns
        -------
        dict
            ``{'records': [...], 'depth_bins': [...], 'separations_mm': [...],
            'noise_types': [...], 'depth_bin_edges_mm': {...}}``. Each record is
            ``{noise_type, depth_bin, mean_depth_mm, separation_mm, error_mm}``
            for one recovered dipole, so callers can pivot freely over the three
            axes.
        """
        if separations_mm is None:
            separations_mm = [2.0, 4.0, 6.0, 8.0]
        if noise_types is None:
            noise_types = list(NOISE_TYPES)
        if depth_bins is None:
            depth_bins = self.DEFAULT_DEPTH_BINS

        unknown = set(noise_types) - set(NOISE_TYPES)
        if unknown:
            raise ValueError(
                f"Unknown noise types {sorted(unknown)}. Expected {NOISE_TYPES}."
            )

        if suppression_mm is None:
            suppression_mm = 0.5 * min(separations_mm)

        pairs_by_cell = self._select_depth_matched_pairs(
            separations_mm, depth_bins, n_pairs,
            separation_tolerance_mm=separation_tolerance_mm,
            depth_tolerance_mm=depth_tolerance_mm
        )

        depth = self.source_depths
        depth_bin_edges = {
            label: (float(np.percentile(depth, lo)), float(np.percentile(depth, hi)))
            for (label, lo, hi) in depth_bins
        }

        if self.verbose:
            print(f"Running two-dipole test: {len(noise_types)} noise types x "
                  f"{len(depth_bins)} depth bins x {len(separations_mm)} separations, "
                  f"{n_pairs} pairs, {n_trials} trials each, SNR = {snr_db} dB")
            for (label, sep), plist in pairs_by_cell.items():
                if not plist:
                    print(f"  WARNING: no matched pair for depth={label}, "
                          f"sep={sep} mm — skipping this cell.")

        records: List[Dict[str, Any]] = []

        for noise_type in noise_types:
            if self.verbose:
                print(f"  Noise type = {noise_type}:")

            for (label, _lo, _hi) in depth_bins:
                for sep in separations_mm:
                    cell_errors = []
                    for pair_i, (idx1, idx2) in enumerate(pairs_by_cell[(label, sep)]):
                        pos1 = self.source_pos_mm[idx1]
                        pos2 = self.source_pos_mm[idx2]
                        mean_depth = float(0.5 * (depth[idx1] + depth[idx2]))

                        for trial in range(n_trials):
                            try:
                                eeg_data, meta = self.simulator.simulate_two_dipoles(
                                    position1_mm=pos1,
                                    position2_mm=pos2,
                                    amplitude1_nAm=amplitude_nAm,
                                    amplitude2_nAm=amplitude_nAm,
                                    snr_db=snr_db,
                                    noise_seed=trial * 1000 + pair_i,
                                    noise_type=noise_type,
                                    noise_spatial_scale_mm=spatial_scale_mm,
                                    noise_temporal_exponent=temporal_exponent,
                                    duration_s=0.5,
                                    sfreq=256.0
                                )

                                source_activity = self._apply_inverse(eeg_data)

                                true_pair = np.array([
                                    meta['dipole1']['actual_position_mm'],
                                    meta['dipole2']['actual_position_mm'],
                                ])
                                e1, e2 = self._find_two_peaks_and_errors(
                                    source_activity, true_pair, suppression_mm
                                )
                                for err in (e1, e2):
                                    records.append({
                                        'noise_type': noise_type,
                                        'depth_bin': label,
                                        'mean_depth_mm': mean_depth,
                                        'separation_mm': float(sep),
                                        'error_mm': float(err),
                                    })
                                    cell_errors.append(err)

                            except Exception as e:
                                if self.verbose:
                                    warnings.warn(f"Trial failed: {e}")

                    if self.verbose and cell_errors:
                        print(f"    {label:>12s}  sep {sep:>4.1f} mm: "
                              f"mean per-dipole error = {np.mean(cell_errors):.2f} mm "
                              f"(n={len(cell_errors)})")

        result = {
            'records': records,
            'depth_bins': [label for (label, _, _) in depth_bins],
            'depth_bin_edges_mm': depth_bin_edges,
            'separations_mm': list(separations_mm),
            'noise_types': list(noise_types),
        }
        self.results['two_dipole'] = result
        return result

    def run_resolvability_test(
        self,
        separations_mm: Optional[List[float]] = None,
        noise_types: Optional[List[str]] = None,
        depth_bins: Optional[List[Tuple[str, float, float]]] = None,
        snr_db: float = 10.0,
        n_pairs: int = 4,
        n_trials: int = 10,
        amplitude_nAm: float = 50.0,
        spatial_scale_mm: float = 3.0,
        temporal_exponent: float = 1.0,
        separation_tolerance_mm: float = 1.0,
        depth_tolerance_mm: float = 1.0,
        saddle_ratio: float = 0.8,
        prominence_frac: float = 0.5,
        max_match_mm: Optional[float] = None,
        neighbor_radius_mm: Optional[float] = None,
        exclude_radius_mm: Optional[float] = None
    ) -> Dict[str, Any]:
        """
        Sweep the probability of *resolving* two sources as two, not one.

        Where :meth:`run_two_dipole_test` asks "how accurately are two known
        sources placed?", this asks the prior question — "can we tell there are
        two at all?" — which is the meaningful limit for closely-spaced sources.
        For each (depth bin, separation, noise type) cell it reports the fraction
        of trials in which the reconstruction shows two distinct peaks with a
        saddle between them (see :meth:`_resolve_two_sources`). From the
        resolution-probability-vs-separation curve it extracts a **threshold
        separation**: the closest two sources can be while still being resolved.

        Per-dipole localization error is retained as a secondary field, so a
        single run yields both "can we separate them" and "given separation, how
        well are they placed".

        Parameters
        ----------
        separations_mm, noise_types, depth_bins, snr_db, n_pairs, n_trials,
        amplitude_nAm, spatial_scale_mm, temporal_exponent,
        separation_tolerance_mm, depth_tolerance_mm
            As in :meth:`run_two_dipole_test`.
        saddle_ratio : float, default=0.8
            Rayleigh-style dip threshold passed to the detector.
        max_match_mm : float, optional
            Optional loose sanity cap on peak-to-source distance. Off by default;
            correspondence is the parameter-free distinct-nearest-source rule.
        neighbor_radius_mm, exclude_radius_mm : float, optional
            Detector radii; default to 1.5x median grid spacing.

        Returns
        -------
        dict
            ``{'records': [...], 'resolution_probability': {...},
            'threshold_separation_mm': {...}, 'depth_bins', 'depth_bin_edges_mm',
            'separations_mm', 'noise_types', 'detector_params'}``. Each record is
            ``{noise_type, depth_bin, mean_depth_mm, separation_mm, resolved,
            loc_error_mm, saddle_ratio_observed}``. ``threshold_separation_mm``
            is keyed [noise_type][depth_bin] and gives the smallest separation
            with resolution probability >= 0.5 (None if never reached).
        """
        if separations_mm is None:
            separations_mm = [2.0, 4.0, 6.0, 8.0]
        if noise_types is None:
            noise_types = list(NOISE_TYPES)
        if depth_bins is None:
            depth_bins = self.DEFAULT_DEPTH_BINS

        unknown = set(noise_types) - set(NOISE_TYPES)
        if unknown:
            raise ValueError(
                f"Unknown noise types {sorted(unknown)}. Expected {NOISE_TYPES}."
            )

        pairs_by_cell = self._select_depth_matched_pairs(
            separations_mm, depth_bins, n_pairs,
            separation_tolerance_mm=separation_tolerance_mm,
            depth_tolerance_mm=depth_tolerance_mm
        )

        depth = self.source_depths
        depth_bin_edges = {
            label: (float(np.percentile(depth, lo)), float(np.percentile(depth, hi)))
            for (label, lo, hi) in depth_bins
        }

        if self.verbose:
            print(f"Running resolvability test: {len(noise_types)} noise types x "
                  f"{len(depth_bins)} depth bins x {len(separations_mm)} separations, "
                  f"{n_pairs} pairs, {n_trials} trials each, SNR = {snr_db} dB")
            cap = f"{max_match_mm} mm" if max_match_mm is not None else "off"
            print(f"  detector: saddle_ratio={saddle_ratio}, "
                  f"correspondence=distinct-nearest-source, sanity_cap={cap}")

        records: List[Dict[str, Any]] = []

        for noise_type in noise_types:
            if self.verbose:
                print(f"  Noise type = {noise_type}:")

            for (label, _lo, _hi) in depth_bins:
                for sep in separations_mm:
                    cell_resolved = []
                    for pair_i, (idx1, idx2) in enumerate(pairs_by_cell[(label, sep)]):
                        pos1 = self.source_pos_mm[idx1]
                        pos2 = self.source_pos_mm[idx2]
                        mean_depth = float(0.5 * (depth[idx1] + depth[idx2]))

                        for trial in range(n_trials):
                            try:
                                eeg_data, meta = self.simulator.simulate_two_dipoles(
                                    position1_mm=pos1,
                                    position2_mm=pos2,
                                    amplitude1_nAm=amplitude_nAm,
                                    amplitude2_nAm=amplitude_nAm,
                                    snr_db=snr_db,
                                    noise_seed=trial * 1000 + pair_i,
                                    noise_type=noise_type,
                                    noise_spatial_scale_mm=spatial_scale_mm,
                                    noise_temporal_exponent=temporal_exponent,
                                    duration_s=0.5,
                                    sfreq=256.0
                                )

                                source_activity = self._apply_inverse(eeg_data)
                                true_pair = np.array([
                                    meta['dipole1']['actual_position_mm'],
                                    meta['dipole2']['actual_position_mm'],
                                ])

                                res = self._resolve_two_sources(
                                    source_activity, true_pair,
                                    neighbor_radius_mm=neighbor_radius_mm,
                                    exclude_radius_mm=exclude_radius_mm,
                                    saddle_ratio=saddle_ratio,
                                    prominence_frac=prominence_frac,
                                    max_match_mm=max_match_mm
                                )
                                # Secondary: localization error (forced 2 peaks).
                                e1, e2 = self._find_two_peaks_and_errors(
                                    source_activity, true_pair,
                                    suppression_mm=0.5 * min(separations_mm)
                                )

                                records.append({
                                    'noise_type': noise_type,
                                    'depth_bin': label,
                                    'mean_depth_mm': mean_depth,
                                    'separation_mm': float(sep),
                                    'resolved': bool(res['resolved']),
                                    'loc_error_mm': float(0.5 * (e1 + e2)),
                                    'saddle_ratio_observed': res['saddle_ratio_observed'],
                                    'peak2_prominence': res['peak2_prominence'],
                                })
                                cell_resolved.append(res['resolved'])

                            except Exception as e:
                                if self.verbose:
                                    warnings.warn(f"Trial failed: {e}")

                    if self.verbose and cell_resolved:
                        frac = float(np.mean(cell_resolved))
                        print(f"    {label:>12s}  sep {sep:>4.1f} mm: "
                              f"resolved {frac*100:>5.1f}% (n={len(cell_resolved)})")

        # Resolution probability grid and threshold separation per (noise, depth).
        resolution_probability: Dict[str, Dict[str, Dict[str, float]]] = {}
        threshold_separation: Dict[str, Dict[str, Optional[float]]] = {}
        sorted_seps = sorted(separations_mm)
        for nt in noise_types:
            resolution_probability[nt] = {}
            threshold_separation[nt] = {}
            for (label, _, _) in depth_bins:
                probs = {}
                for sep in sorted_seps:
                    cell = [r['resolved'] for r in records
                            if r['noise_type'] == nt and r['depth_bin'] == label
                            and r['separation_mm'] == sep]
                    probs[f"{sep:.1f}"] = float(np.mean(cell)) if cell else None
                resolution_probability[nt][label] = probs
                # Smallest separation whose probability first reaches >= 0.5.
                threshold = None
                for sep in sorted_seps:
                    p = probs[f"{sep:.1f}"]
                    if p is not None and p >= 0.5:
                        threshold = float(sep)
                        break
                threshold_separation[nt][label] = threshold

        result = {
            'records': records,
            'resolution_probability': resolution_probability,
            'threshold_separation_mm': threshold_separation,
            'depth_bins': [label for (label, _, _) in depth_bins],
            'depth_bin_edges_mm': depth_bin_edges,
            'separations_mm': list(separations_mm),
            'noise_types': list(noise_types),
            'detector_params': {
                'saddle_ratio': saddle_ratio,
                'prominence_frac': prominence_frac,
                'correspondence': 'distinct_nearest_source',
                'max_match_mm': max_match_mm,
                'neighbor_radius_mm': neighbor_radius_mm,
                'exclude_radius_mm': exclude_radius_mm,
                'median_grid_spacing_mm': self._median_grid_spacing(),
            },
        }
        self.results['resolvability'] = result
        return result

    def run_snr_test(
        self,
        snr_range: Tuple[float, float] = (-20, 40),
        snr_step: float = 5,
        snr_values: Optional[List[float]] = None,
        n_positions: int = 4,
        n_trials: int = 10,
        amplitude_nAm: float = 50.0
    ) -> RobustnessResults:
        """
        Test localization error across SNR levels.

        Parameters
        ----------
        snr_range : tuple, default=(-20, 40)
            (min_snr_db, max_snr_db) range to test
        snr_step : float, default=5
            Step size in dB
        snr_values : list, optional
            Explicit list of SNR values to test (overrides range/step)
        n_positions : int, default=4
            Number of source positions to test
        n_trials : int, default=10
            Number of trials per position per SNR level
        amplitude_nAm : float, default=50.0
            Dipole amplitude in nAm

        Returns
        -------
        RobustnessResults
            Test results with errors for each SNR level
        """
        if snr_values is None:
            snr_values = list(np.arange(snr_range[0], snr_range[1] + snr_step, snr_step))

        test_indices = self._select_test_positions(n_positions)

        if self.verbose:
            print(f"Running SNR test: {len(snr_values)} levels, "
                  f"{n_positions} positions, {n_trials} trials each")

        results = {snr: [] for snr in snr_values}

        for snr_db in snr_values:
            if self.verbose:
                print(f"  SNR = {snr_db:+6.1f} dB: ", end="", flush=True)

            for test_idx in test_indices:
                test_pos = self.source_pos_mm[test_idx]

                for trial in range(n_trials):
                    try:
                        # Simulate dipole
                        eeg_data, metadata = self.simulator.simulate_dipole(
                            position_mm=test_pos,
                            amplitude_nAm=amplitude_nAm,
                            snr_db=snr_db,
                            duration_s=0.5,
                            sfreq=256.0
                        )

                        # Apply inverse
                        source_activity = self._apply_inverse(eeg_data)

                        # Compute error
                        true_pos = metadata['actual_position_mm']
                        error = self._find_peak_and_error(source_activity, true_pos)
                        results[snr_db].append(error)

                    except Exception as e:
                        if self.verbose:
                            warnings.warn(f"Trial failed: {e}")

            if self.verbose:
                mean_err = np.mean(results[snr_db]) if results[snr_db] else np.nan
                print(f"mean = {mean_err:.2f} mm")

        result = RobustnessResults(
            test_type='snr',
            parameter_values=snr_values,
            errors=results,
            n_sources=self.n_sources,
            n_trials_per_position=n_trials,
            n_positions=n_positions
        )

        self.results['snr'] = result
        return result

    def run_noise_test(
        self,
        noise_range: Tuple[float, float] = (0.01, 10000),
        n_levels: int = 8,
        noise_values: Optional[List[float]] = None,
        n_positions: int = 4,
        n_trials: int = 10,
        amplitude_nAm: float = 50.0
    ) -> RobustnessResults:
        """
        Test localization error across noise levels (fixed signal amplitude).

        Parameters
        ----------
        noise_range : tuple, default=(0.01, 10000)
            (min_noise, max_noise) in µV² (log-spaced)
        n_levels : int, default=8
            Number of noise levels to test
        noise_values : list, optional
            Explicit list of noise values in µV² (overrides range)
        n_positions : int, default=4
            Number of source positions to test
        n_trials : int, default=10
            Number of trials per position per noise level
        amplitude_nAm : float, default=50.0
            Fixed dipole amplitude in nAm

        Returns
        -------
        RobustnessResults
            Test results with errors for each noise level
        """
        if noise_values is None:
            noise_values = list(np.logspace(
                np.log10(noise_range[0]),
                np.log10(noise_range[1]),
                n_levels
            ))

        test_indices = self._select_test_positions(n_positions)

        if self.verbose:
            print(f"Running noise test: {len(noise_values)} levels, "
                  f"{n_positions} positions, {n_trials} trials each")

        results = {nv: [] for nv in noise_values}

        for noise_var in noise_values:
            if self.verbose:
                print(f"  Noise = {noise_var:10.3f} µV²: ", end="", flush=True)

            for test_idx in test_indices:
                test_pos = self.source_pos_mm[test_idx]

                for trial in range(n_trials):
                    try:
                        # Simulate dipole with fixed noise variance
                        eeg_data, metadata = self.simulator.simulate_dipole(
                            position_mm=test_pos,
                            amplitude_nAm=amplitude_nAm,
                            noise_mode="fixed_variance",
                            noise_variance_uV2=noise_var,
                            duration_s=0.5,
                            sfreq=256.0
                        )

                        # Apply inverse
                        source_activity = self._apply_inverse(eeg_data)

                        # Compute error
                        true_pos = metadata['actual_position_mm']
                        error = self._find_peak_and_error(source_activity, true_pos)
                        results[noise_var].append(error)

                    except Exception as e:
                        if self.verbose:
                            warnings.warn(f"Trial failed: {e}")

            if self.verbose:
                mean_err = np.mean(results[noise_var]) if results[noise_var] else np.nan
                print(f"mean = {mean_err:.2f} mm")

        result = RobustnessResults(
            test_type='noise',
            parameter_values=noise_values,
            errors=results,
            n_sources=self.n_sources,
            n_trials_per_position=n_trials,
            n_positions=n_positions
        )

        self.results['noise'] = result
        return result

    def run_noise_type_test(
        self,
        noise_types: Optional[List[str]] = None,
        snr_db: float = 10.0,
        n_positions: int = 4,
        n_trials: int = 10,
        amplitude_nAm: float = 50.0,
        spatial_scale_mm: float = 3.0,
        temporal_exponent: float = 1.0
    ) -> RobustnessResults:
        """
        Test localization error across noise *structures* at a fixed SNR.

        This is the sweep that produces a range of expectations rather than a
        single best-case number. ``run_noise_test`` varies noise *level*, which
        leaves the channel covariance proportional to identity at every level —
        exactly the assumption the scaled-identity inverse makes, so it cannot
        probe that assumption. This sweep instead holds the level fixed and
        varies the structure: 'white' satisfies C = I and gives the optimistic
        bound; 'colored' (spatially correlated + 1/f) violates it and gives the
        realistic bound.

        All noise types are tested at the same positions with the same seeds,
        so the comparison is paired — differences reflect noise structure, not
        which trials happened to be drawn.

        Parameters
        ----------
        noise_types : list of str, optional
            Noise types to sweep. Defaults to all of
            :data:`~source_localization.validation.noise.NOISE_TYPES`.
        snr_db : float, default=10.0
            Fixed SNR for every type, so only structure varies.
        n_positions : int, default=4
            Number of source positions to test.
        n_trials : int, default=10
            Number of trials per position per noise type.
        amplitude_nAm : float, default=50.0
            Fixed dipole amplitude in nAm.
        spatial_scale_mm : float, default=3.0
            Correlation length for spatially-correlated types.
        temporal_exponent : float, default=1.0
            Spectral exponent for 1/f-shaped types (1.0 = pink).

        Returns
        -------
        RobustnessResults
            Errors for each noise type. ``test_type='noise_type'``; the result
            is categorical, so ``correlation`` is NaN by design.

        Examples
        --------
        >>> test = RobustnessTest(fwd, src, info)
        >>> results = test.run_noise_type_test(n_trials=20)
        >>> means = results.means
        >>> print(f"white {means['white']:.2f} mm -> colored {means['colored']:.2f} mm")
        """
        if noise_types is None:
            noise_types = list(NOISE_TYPES)

        unknown = set(noise_types) - set(NOISE_TYPES)
        if unknown:
            raise ValueError(
                f"Unknown noise types {sorted(unknown)}. Expected {NOISE_TYPES}."
            )

        test_indices = self._select_test_positions(n_positions)

        if self.verbose:
            print(f"Running noise-type test: {len(noise_types)} types, "
                  f"{n_positions} positions, {n_trials} trials each, "
                  f"SNR = {snr_db} dB")

        results = {nt: [] for nt in noise_types}

        for noise_type in noise_types:
            if self.verbose:
                print(f"  Noise type = {noise_type:9s}: ", end="", flush=True)

            for pos_i, test_idx in enumerate(test_indices):
                test_pos = self.source_pos_mm[test_idx]

                for trial in range(n_trials):
                    try:
                        # Seed depends on position/trial but NOT on noise_type,
                        # so every type sees the same underlying draw.
                        eeg_data, metadata = self.simulator.simulate_dipole(
                            position_mm=test_pos,
                            amplitude_nAm=amplitude_nAm,
                            snr_db=snr_db,
                            noise_mode="snr",
                            noise_seed=trial * 1000 + pos_i,
                            noise_type=noise_type,
                            noise_spatial_scale_mm=spatial_scale_mm,
                            noise_temporal_exponent=temporal_exponent,
                            duration_s=0.5,
                            sfreq=256.0
                        )

                        source_activity = self._apply_inverse(eeg_data)

                        true_pos = metadata['actual_position_mm']
                        error = self._find_peak_and_error(source_activity, true_pos)
                        results[noise_type].append(error)

                    except Exception as e:
                        if self.verbose:
                            warnings.warn(f"Trial failed: {e}")

            if self.verbose:
                mean_err = np.mean(results[noise_type]) if results[noise_type] else np.nan
                print(f"mean = {mean_err:.2f} mm")

        result = RobustnessResults(
            test_type='noise_type',
            parameter_values=list(noise_types),
            errors=results,
            n_sources=self.n_sources,
            n_trials_per_position=n_trials,
            n_positions=n_positions
        )

        self.results['noise_type'] = result
        return result

    def run_amplitude_test(
        self,
        amplitude_range: Tuple[float, float] = (5, 500),
        n_levels: int = 8,
        amplitude_values: Optional[List[float]] = None,
        n_positions: int = 4,
        n_trials: int = 10,
        noise_variance_uV2: float = 1.0
    ) -> RobustnessResults:
        """
        Test localization error across dipole amplitudes (fixed noise).

        Parameters
        ----------
        amplitude_range : tuple, default=(5, 500)
            (min_amplitude, max_amplitude) in nAm (log-spaced)
        n_levels : int, default=8
            Number of amplitude levels to test
        amplitude_values : list, optional
            Explicit list of amplitude values in nAm (overrides range)
        n_positions : int, default=4
            Number of source positions to test
        n_trials : int, default=10
            Number of trials per position per amplitude
        noise_variance_uV2 : float, default=1.0
            Fixed noise variance in µV²

        Returns
        -------
        RobustnessResults
            Test results with errors for each amplitude level
        """
        if amplitude_values is None:
            amplitude_values = list(np.logspace(
                np.log10(amplitude_range[0]),
                np.log10(amplitude_range[1]),
                n_levels
            ))

        test_indices = self._select_test_positions(n_positions)

        if self.verbose:
            print(f"Running amplitude test: {len(amplitude_values)} levels, "
                  f"{n_positions} positions, {n_trials} trials each")

        results = {amp: [] for amp in amplitude_values}

        for amplitude in amplitude_values:
            if self.verbose:
                print(f"  Amplitude = {amplitude:8.1f} nAm: ", end="", flush=True)

            for test_idx in test_indices:
                test_pos = self.source_pos_mm[test_idx]

                for trial in range(n_trials):
                    try:
                        # Simulate dipole with fixed noise
                        eeg_data, metadata = self.simulator.simulate_dipole(
                            position_mm=test_pos,
                            amplitude_nAm=amplitude,
                            noise_mode="fixed_variance",
                            noise_variance_uV2=noise_variance_uV2,
                            duration_s=0.5,
                            sfreq=256.0
                        )

                        # Apply inverse
                        source_activity = self._apply_inverse(eeg_data)

                        # Compute error
                        true_pos = metadata['actual_position_mm']
                        error = self._find_peak_and_error(source_activity, true_pos)
                        results[amplitude].append(error)

                    except Exception as e:
                        if self.verbose:
                            warnings.warn(f"Trial failed: {e}")

            if self.verbose:
                mean_err = np.mean(results[amplitude]) if results[amplitude] else np.nan
                print(f"mean = {mean_err:.2f} mm")

        result = RobustnessResults(
            test_type='amplitude',
            parameter_values=amplitude_values,
            errors=results,
            n_sources=self.n_sources,
            n_trials_per_position=n_trials,
            n_positions=n_positions
        )

        self.results['amplitude'] = result
        return result

    def run_all_tests(
        self,
        snr_range: Tuple[float, float] = (-20, 40),
        noise_range: Tuple[float, float] = (0.01, 10000),
        amplitude_range: Tuple[float, float] = (5, 500),
        n_positions: int = 4,
        n_trials: int = 10
    ) -> Dict[str, RobustnessResults]:
        """
        Run all robustness tests.

        Parameters
        ----------
        snr_range : tuple
            SNR range in dB
        noise_range : tuple
            Noise range in µV²
        amplitude_range : tuple
            Amplitude range in nAm
        n_positions : int
            Number of test positions
        n_trials : int
            Trials per position

        Returns
        -------
        dict
            Dictionary with 'snr', 'noise', 'amplitude' keys
        """
        if self.verbose:
            print("=" * 60)
            print("RUNNING ALL ROBUSTNESS TESTS")
            print("=" * 60)

        self.run_snr_test(snr_range=snr_range, n_positions=n_positions, n_trials=n_trials)
        self.run_noise_test(noise_range=noise_range, n_positions=n_positions, n_trials=n_trials)
        self.run_amplitude_test(amplitude_range=amplitude_range, n_positions=n_positions, n_trials=n_trials)

        return self.results

    def get_summary(self) -> dict:
        """
        Get summary of all completed tests.

        Returns
        -------
        dict
            Summary statistics for each test
        """
        summary = {
            'n_sources': int(self.n_sources),
            'inverse_method': self.inverse_method,
            'inverse_snr': float(self.inverse_snr),
            'tests': {}
        }

        for test_name, result in self.results.items():
            # The two-dipole and resolvability tests store a records dict (not a
            # RobustnessResults): a flat list tagged by noise_type/depth/separation.
            if isinstance(result, dict) and 'records' in result:
                records = result['records']
                if records and 'resolved' in records[0]:
                    # Resolvability: probability grid + threshold already computed.
                    summary['tests'][test_name] = {
                        'resolution_probability': result.get('resolution_probability', {}),
                        'threshold_separation_mm': result.get('threshold_separation_mm', {}),
                        'n_trials_total': len(records),
                        'depth_bin_edges_mm': result.get('depth_bin_edges_mm', {}),
                        'detector_params': result.get('detector_params', {}),
                    }
                    continue

                # Localization: mean-error grid over (noise_type, depth, separation).
                grid: Dict[str, Dict[str, Dict[str, list]]] = {}
                for rec in records:
                    (grid.setdefault(rec['noise_type'], {})
                         .setdefault(rec['depth_bin'], {})
                         .setdefault(f"{rec['separation_mm']:.1f}", [])
                         .append(rec['error_mm']))
                summary['tests'][test_name] = {
                    'mean_error_mm': {
                        nt: {db: {sep: float(np.mean(errs)) for sep, errs in seps.items()}
                             for db, seps in bins.items()}
                        for nt, bins in grid.items()
                    },
                    'n_trials_total': len(records),
                    'depth_bin_edges_mm': result.get('depth_bin_edges_mm', {}),
                }
                continue

            params = sorted(result.errors.keys())
            means = [float(np.mean(result.errors[p])) for p in params]
            n_trials_total = int(sum(len(v) for v in result.errors.values()))

            if result.is_categorical:
                # A categorical sweep (noise_type) has no parameter ordering, so
                # correlation and the correlation-based physics check are both
                # undefined. The per-label errors ARE the result: they bound the
                # range of expectations from best case (white) to realistic
                # (colored).
                summary['tests'][test_name] = {
                    'correlation': None,
                    'parameter_range': None,
                    'error_by_parameter': {
                        str(p): float(np.mean(result.errors[p])) for p in params
                    },
                    'error_range': (float(min(means)), float(max(means))),
                    'n_trials_total': n_trials_total,
                    'physics_valid': None
                }
                continue

            corr = result.correlation
            corr_float = float(corr) if not np.isnan(corr) else 0.0

            summary['tests'][test_name] = {
                'correlation': corr_float,
                'parameter_range': (float(min(params)), float(max(params))),
                'error_range': (float(min(means)), float(max(means))),
                'n_trials_total': n_trials_total,
                'physics_valid': bool(
                    corr_float < -0.5 if test_name == 'snr'
                    else corr_float > 0.5 if test_name == 'noise'
                    else corr_float < -0.3  # amplitude
                )
            }

        return summary

    def plot_results(
        self,
        output_dir: Optional[Union[str, Path]] = None,
        figsize: Tuple[float, float] = (10, 6),
        show: bool = False,
        config_name: Optional[str] = None
    ) -> Optional[dict]:
        """
        Generate comprehensive plots for all completed tests.

        Generates multiple plot types:
        - Individual test plots with regression lines
        - Combined summary figure
        - Detailed plots with scatter points

        Parameters
        ----------
        output_dir : str or Path, optional
            Directory to save plots. If None, returns figure objects.
        figsize : tuple, default=(10, 6)
            Figure size in inches
        show : bool, default=False
            Whether to display plots
        config_name : str, optional
            Name to use in plot titles

        Returns
        -------
        dict or None
            Dictionary of figure objects if output_dir is None
        """
        try:
            import matplotlib.pyplot as plt
        except ImportError:
            raise ImportError("matplotlib required for plotting")

        plt.style.use('seaborn-v0_8-whitegrid')
        plt.rcParams.update({
            'font.size': 11,
            'axes.labelsize': 12,
            'axes.titlesize': 13,
            'figure.dpi': 150,
        })

        if output_dir:
            output_dir = Path(output_dir)
            output_dir.mkdir(parents=True, exist_ok=True)

        figures = {}
        title_suffix = f" ({config_name})" if config_name else f" (n={self.n_sources})"

        # Colors for different test types
        colors = {'snr': '#2ecc71', 'noise': '#e74c3c', 'amplitude': '#3498db'}

        # =====================================================================
        # Individual test plots with scatter and regression
        # =====================================================================
        for test_name, result in self.results.items():
            fig, ax = plt.subplots(figsize=figsize)

            params = sorted(result.errors.keys())
            means = [np.mean(result.errors[p]) for p in params]
            stds = [np.std(result.errors[p]) for p in params]
            sems = [std / np.sqrt(len(result.errors[p])) for std, p in zip(stds, params)]

            color = colors.get(test_name, '#333333')

            # Scatter all individual points
            for p in params:
                errors = result.errors[p]
                if test_name in ('noise', 'amplitude'):
                    jittered_p = p * np.random.uniform(0.9, 1.1, len(errors))
                else:
                    jittered_p = p + np.random.uniform(-0.5, 0.5, len(errors))
                ax.scatter(jittered_p, errors, alpha=0.3, s=20, color=color)

            # Mean line with error bars
            ax.errorbar(params, means, yerr=sems, fmt='o-', color=color,
                       markersize=10, capsize=4, capthick=1.5, linewidth=2.5,
                       label='Mean ± SEM', zorder=3)

            # Regression line and correlation
            if test_name in ('noise', 'amplitude'):
                log_params = np.log10(params)
                slope, intercept, r_value, p_value, _ = stats.linregress(log_params, means)
                ax.set_xscale('log')
                x_label = 'Noise Variance (µV²)' if test_name == 'noise' else 'Dipole Amplitude (nAm)'
            else:  # snr
                slope, intercept, r_value, p_value, _ = stats.linregress(params, means)
                x_fit = np.array([min(params), max(params)])
                y_fit = slope * x_fit + intercept
                ax.plot(x_fit, y_fit, '--', color='gray', linewidth=1.5, alpha=0.7,
                       label=f'Fit (r={r_value:.2f})')
                x_label = 'SNR (dB)'

            # Target accuracy zone
            ax.axhspan(0, 2, alpha=0.1, color='green', label='Target (<2mm)')

            # Zero line for SNR
            if test_name == 'snr':
                ax.axvline(x=0, color='gray', linestyle=':', alpha=0.5)

            ax.set_xlabel(x_label)
            ax.set_ylabel('Localization Error (mm)')
            ax.set_title(f'{test_name.upper()} Robustness Test{title_suffix}')
            ax.legend(loc='upper right' if test_name != 'snr' else 'upper left')

            # Add statistics annotation
            stats_text = f'r = {r_value:.3f}\nslope = {slope:.4f}'
            if test_name == 'snr':
                ax.text(0.95, 0.95, stats_text, transform=ax.transAxes, fontsize=10,
                       verticalalignment='top', horizontalalignment='right',
                       bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
            else:
                ax.text(0.05, 0.95, stats_text, transform=ax.transAxes, fontsize=10,
                       verticalalignment='top', horizontalalignment='left',
                       bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

            figures[f'{test_name}_detailed'] = fig

            if output_dir:
                fig.savefig(output_dir / f'robustness_{test_name}_detailed.png',
                           dpi=300, bbox_inches='tight')
                fig.savefig(output_dir / f'robustness_{test_name}_detailed.pdf',
                           bbox_inches='tight')

            if not show:
                plt.close(fig)

        # =====================================================================
        # Combined summary figure (if multiple tests completed)
        # =====================================================================
        if len(self.results) > 1:
            n_tests = len(self.results)
            fig, axes = plt.subplots(1, n_tests, figsize=(5 * n_tests, 5))
            if n_tests == 1:
                axes = [axes]

            for idx, (test_name, result) in enumerate(self.results.items()):
                ax = axes[idx]
                params = sorted(result.errors.keys())
                means = [np.mean(result.errors[p]) for p in params]
                color = colors.get(test_name, '#333333')

                ax.plot(params, means, 'o-', color=color, markersize=8, linewidth=2)

                if test_name in ('noise', 'amplitude'):
                    ax.set_xscale('log')
                    x_label = 'Noise (µV²)' if test_name == 'noise' else 'Amplitude (nAm)'
                else:
                    x_label = 'SNR (dB)'
                    ax.axvline(x=0, color='gray', linestyle=':', alpha=0.5)

                ax.axhspan(0, 2, alpha=0.1, color='green')
                ax.set_xlabel(x_label)
                ax.set_ylabel('Localization Error (mm)')
                ax.set_title(f'{test_name.upper()} (r={result.correlation:.2f})')

            plt.suptitle(f'Robustness Tests Summary{title_suffix}', fontsize=14, y=1.02)
            plt.tight_layout()

            figures['summary'] = fig

            if output_dir:
                fig.savefig(output_dir / 'robustness_summary.png',
                           dpi=300, bbox_inches='tight')
                fig.savefig(output_dir / 'robustness_summary.pdf',
                           bbox_inches='tight')

            if not show:
                plt.close(fig)

        # =====================================================================
        # SNR bar chart comparison (if SNR test completed)
        # =====================================================================
        if 'snr' in self.results:
            snr_result = self.results['snr']
            params = sorted(snr_result.errors.keys())

            # Find low, mid, high SNR values
            low_snr = min(params)
            high_snr = max(params)
            mid_snr = params[len(params) // 2]

            fig, ax = plt.subplots(figsize=(8, 5))

            x = np.arange(3)
            errors = [
                np.mean(snr_result.errors[low_snr]),
                np.mean(snr_result.errors[mid_snr]),
                np.mean(snr_result.errors[high_snr])
            ]
            stds = [
                np.std(snr_result.errors[low_snr]),
                np.std(snr_result.errors[mid_snr]),
                np.std(snr_result.errors[high_snr])
            ]

            bar_colors = ['#e74c3c', '#f39c12', '#2ecc71']
            bars = ax.bar(x, errors, yerr=stds, capsize=5, color=bar_colors, alpha=0.8)

            ax.set_ylabel('Localization Error (mm)')
            ax.set_title(f'Error at Different SNR Levels{title_suffix}')
            ax.set_xticks(x)
            ax.set_xticklabels([f'{low_snr:+.0f} dB', f'{mid_snr:+.0f} dB', f'{high_snr:+.0f} dB'])

            # Add improvement percentage
            improvement = (errors[0] - errors[2]) / errors[0] * 100
            ax.annotate(f'{improvement:.0f}% reduction',
                       xy=(1, max(errors) * 0.9), fontsize=11, ha='center',
                       bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

            ax.axhline(y=2, color='green', linestyle='--', alpha=0.5, label='Target (2mm)')
            ax.legend()

            figures['snr_comparison'] = fig

            if output_dir:
                fig.savefig(output_dir / 'robustness_snr_comparison.png',
                           dpi=300, bbox_inches='tight')
                fig.savefig(output_dir / 'robustness_snr_comparison.pdf',
                           bbox_inches='tight')

            if not show:
                plt.close(fig)

        if show:
            plt.show()

        if self.verbose and output_dir:
            print(f"Plots saved to: {output_dir}")

        return figures if not output_dir else None

    def save_results(self, output_path: Union[str, Path]):
        """
        Save results to JSON file.

        Parameters
        ----------
        output_path : str or Path
            Output file path
        """
        output_path = Path(output_path)

        def convert_numpy(obj):
            """Recursively convert numpy types to native Python types."""
            if isinstance(obj, dict):
                return {k: convert_numpy(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [convert_numpy(v) for v in obj]
            elif isinstance(obj, (np.integer,)):
                return int(obj)
            elif isinstance(obj, (np.floating,)):
                return float(obj)
            elif isinstance(obj, np.ndarray):
                return obj.tolist()
            elif isinstance(obj, (np.bool_,)):
                return bool(obj)
            return obj

        data = {
            'metadata': {
                'n_sources': int(self.n_sources),
                'inverse_method': self.inverse_method,
                'inverse_snr': float(self.inverse_snr),
                'timestamp': datetime.now().isoformat()
            },
            'results': {name: result.to_dict() for name, result in self.results.items()},
            'summary': convert_numpy(self.get_summary())
        }

        with open(output_path, 'w') as f:
            json.dump(data, f, indent=2)

        if self.verbose:
            print(f"Results saved to: {output_path}")

    @staticmethod
    def load_results(input_path: Union[str, Path]) -> dict:
        """
        Load results from JSON file.

        Parameters
        ----------
        input_path : str or Path
            Input file path

        Returns
        -------
        dict
            Loaded results
        """
        with open(input_path, 'r') as f:
            return json.load(f)

    @staticmethod
    def plot_multi_config_summary(
        config_results: Dict[str, dict],
        output_dir: Union[str, Path],
        colors: Optional[Dict[str, str]] = None,
        show: bool = False
    ) -> None:
        """
        Create a 2x2 summary plot comparing multiple configurations.

        This generates the same layout as robustness_summary_2x2.png from
        the original validation scripts.

        Parameters
        ----------
        config_results : dict
            Dictionary mapping config names to loaded results.
            Each value should be the output of RobustnessTest.load_results()
            or a dict with 'results' containing 'snr' and 'noise' keys.
        output_dir : str or Path
            Directory to save the plot
        colors : dict, optional
            Mapping of config names to colors. Defaults to a standard palette.
        show : bool, default=False
            Whether to display the plot

        Examples
        --------
        >>> results = {}
        >>> for name, path in [('ROI-based', 'roi/robustness_results.json'),
        ...                     ('Cartesian', 'cart/robustness_results.json')]:
        ...     results[name] = RobustnessTest.load_results(path)
        >>> RobustnessTest.plot_multi_config_summary(results, './figures')
        """
        try:
            import matplotlib.pyplot as plt
        except ImportError:
            raise ImportError("matplotlib required for plotting")

        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        # Default colors
        if colors is None:
            default_colors = ['#2ecc71', '#3498db', '#e74c3c', '#9b59b6', '#f39c12']
            colors = {name: default_colors[i % len(default_colors)]
                     for i, name in enumerate(config_results.keys())}

        plt.style.use('seaborn-v0_8-whitegrid')
        plt.rcParams.update({
            'font.size': 11,
            'axes.labelsize': 12,
            'axes.titlesize': 13,
            'figure.dpi': 150,
        })

        fig, axes = plt.subplots(2, 2, figsize=(12, 10))

        # Extract data
        snr_data = {}
        noise_data = {}

        for config_name, data in config_results.items():
            results = data.get('results', data)

            if 'snr' in results:
                snr_info = results['snr']
                errors = snr_info['errors']
                # Convert string keys back to floats
                snr_data[config_name] = {
                    'errors': {float(k): v for k, v in errors.items()},
                    'n_sources': snr_info.get('n_sources', data.get('metadata', {}).get('n_sources', '?'))
                }

            if 'noise' in results:
                noise_info = results['noise']
                errors = noise_info['errors']
                noise_data[config_name] = {
                    'errors': {float(k): v for k, v in errors.items()},
                    'n_sources': noise_info.get('n_sources', data.get('metadata', {}).get('n_sources', '?'))
                }

        # =====================================================================
        # Top left: SNR vs Error (all configs)
        # =====================================================================
        ax = axes[0, 0]
        for config_name, data in snr_data.items():
            errors = data['errors']
            snr_values = sorted(errors.keys())
            means = [np.mean(errors[snr]) for snr in snr_values]
            ax.plot(snr_values, means, 'o-', color=colors[config_name],
                   markersize=6, linewidth=2, label=config_name)

        ax.axvline(x=0, color='gray', linestyle='--', alpha=0.5)
        ax.axhspan(0, 2, alpha=0.1, color='green')
        ax.set_xlabel('SNR (dB)')
        ax.set_ylabel('Localization Error (mm)')
        ax.set_title('A) SNR vs Error (All Configs)')
        ax.legend(loc='upper right', fontsize=9)
        if snr_data:
            all_snr = [s for d in snr_data.values() for s in d['errors'].keys()]
            ax.set_xlim(min(all_snr) - 3, max(all_snr) + 3)
        ax.set_ylim(0, 7)

        # =====================================================================
        # Top right: Noise vs Error (all configs)
        # =====================================================================
        ax = axes[0, 1]
        for config_name, data in noise_data.items():
            errors = data['errors']
            noise_values = sorted(errors.keys())
            means = [np.mean(errors[nv]) for nv in noise_values]
            ax.plot(noise_values, means, 'o-', color=colors[config_name],
                   markersize=6, linewidth=2, label=config_name)

        ax.set_xscale('log')
        ax.axhspan(0, 2, alpha=0.1, color='green')
        ax.set_xlabel('Noise Variance (µV²)')
        ax.set_ylabel('Localization Error (mm)')
        ax.set_title('B) Noise Level vs Error (All Configs)')
        ax.legend(loc='upper left', fontsize=9)
        ax.set_ylim(0, 7)

        # =====================================================================
        # Bottom left: SNR-Error correlations bar chart
        # =====================================================================
        ax = axes[1, 0]
        configs = list(snr_data.keys())
        correlations = []

        for config_name in configs:
            errors = snr_data[config_name]['errors']
            snr_values = sorted(errors.keys())
            means = [np.mean(errors[snr]) for snr in snr_values]
            corr, _ = stats.pearsonr(snr_values, means)
            correlations.append(corr)

        x_pos = np.arange(len(configs))
        bars = ax.bar(x_pos, correlations, color=[colors[c] for c in configs], alpha=0.8)
        ax.axhline(y=0, color='black', linewidth=0.5)
        ax.axhline(y=-0.7, color='green', linestyle='--', alpha=0.5, label='Strong correlation')
        ax.set_ylabel('Correlation (r)')
        ax.set_title('C) SNR-Error Correlations')
        ax.set_ylim(-1, 0.2)
        ax.set_xticks(x_pos)
        ax.set_xticklabels(configs, rotation=15, ha='right')

        for bar, corr in zip(bars, correlations):
            ax.text(bar.get_x() + bar.get_width()/2, corr - 0.05, f'{corr:.2f}',
                   ha='center', va='top', fontsize=10, fontweight='bold')

        # =====================================================================
        # Bottom right: Error at different SNR levels (grouped bars)
        # =====================================================================
        ax = axes[1, 1]
        x = np.arange(len(configs))
        width = 0.25

        low_snr_vals = []
        mid_snr_vals = []
        high_snr_vals = []

        for config_name in configs:
            errors = snr_data[config_name]['errors']
            snr_values = sorted(errors.keys())
            low_snr = min(snr_values)
            high_snr = max(snr_values)
            mid_snr = snr_values[len(snr_values) // 2]

            low_snr_vals.append(np.mean(errors[low_snr]))
            mid_snr_vals.append(np.mean(errors[mid_snr]))
            high_snr_vals.append(np.mean(errors[high_snr]))

        # Get labels for legend
        if snr_data:
            first_config = list(snr_data.keys())[0]
            snr_values = sorted(snr_data[first_config]['errors'].keys())
            low_label = f'SNR = {min(snr_values):+.0f} dB'
            high_label = f'SNR = {max(snr_values):+.0f} dB'
            mid_label = f'SNR = {snr_values[len(snr_values)//2]:+.0f} dB'
        else:
            low_label, mid_label, high_label = 'Low', 'Mid', 'High'

        ax.bar(x - width, low_snr_vals, width, label=low_label, color='#e74c3c', alpha=0.8)
        ax.bar(x, mid_snr_vals, width, label=mid_label, color='#f39c12', alpha=0.8)
        ax.bar(x + width, high_snr_vals, width, label=high_label, color='#2ecc71', alpha=0.8)

        ax.set_ylabel('Localization Error (mm)')
        ax.set_title('D) Error at Different SNR Levels')
        ax.set_xticks(x)
        ax.set_xticklabels(configs, rotation=15, ha='right')
        ax.legend(loc='upper right', fontsize=9)
        ax.set_ylim(0, 8)

        plt.tight_layout()
        plt.savefig(output_dir / 'robustness_summary_2x2.png', dpi=300, bbox_inches='tight')
        plt.savefig(output_dir / 'robustness_summary_2x2.pdf', bbox_inches='tight')
        print(f'Saved: {output_dir}/robustness_summary_2x2.png/pdf')

        if show:
            plt.show()
        else:
            plt.close(fig)

    def generate_report(
        self,
        output_path: Optional[Union[str, Path]] = None,
        config_name: Optional[str] = None
    ) -> str:
        """
        Generate a text summary report of robustness test results.

        Parameters
        ----------
        output_path : str or Path, optional
            Path to save report. If None, only returns string.
        config_name : str, optional
            Name to use in report header

        Returns
        -------
        str
            Report text
        """
        lines = []
        lines.append("=" * 70)
        lines.append("ROBUSTNESS TEST REPORT")
        if config_name:
            lines.append(f"Configuration: {config_name}")
        lines.append(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        lines.append("=" * 70)

        lines.append(f"\nPipeline Info:")
        lines.append(f"  Sources: {self.n_sources}")
        lines.append(f"  Inverse method: {self.inverse_method}")
        lines.append(f"  Inverse SNR: {self.inverse_snr}")
        lines.append(f"  Depth range: {self.source_depths.min():.2f} - {self.source_depths.max():.2f} mm")

        # SNR Test Results
        if 'snr' in self.results:
            result = self.results['snr']
            params = sorted(result.errors.keys())
            lines.append("\n" + "-" * 70)
            lines.append("SNR SENSITIVITY TEST")
            lines.append("-" * 70)
            lines.append(f"  Range: {min(params):.0f} to {max(params):.0f} dB")
            lines.append(f"  Positions tested: {result.n_positions}")
            lines.append(f"  Trials per position: {result.n_trials_per_position}")
            lines.append(f"\n  {'SNR (dB)':>10} | {'Mean (mm)':>10} | {'Std (mm)':>10} | {'Median (mm)':>12}")
            lines.append("  " + "-" * 50)
            for p in params:
                mean = np.mean(result.errors[p])
                std = np.std(result.errors[p])
                median = np.median(result.errors[p])
                lines.append(f"  {p:>10.0f} | {mean:>10.2f} | {std:>10.2f} | {median:>12.2f}")
            lines.append(f"\n  Correlation: r = {result.correlation:.3f}")
            lines.append(f"  Physics valid (r < -0.5): {'YES' if result.correlation < -0.5 else 'NO'}")

        # Noise Test Results
        if 'noise' in self.results:
            result = self.results['noise']
            params = sorted(result.errors.keys())
            lines.append("\n" + "-" * 70)
            lines.append("NOISE LEVEL SENSITIVITY TEST")
            lines.append("-" * 70)
            lines.append(f"  Range: {min(params):.3f} to {max(params):.0f} µV²")
            lines.append(f"  Positions tested: {result.n_positions}")
            lines.append(f"  Trials per position: {result.n_trials_per_position}")
            lines.append(f"\n  {'Noise (µV²)':>12} | {'Mean (mm)':>10} | {'Std (mm)':>10} | {'Median (mm)':>12}")
            lines.append("  " + "-" * 52)
            for p in params:
                mean = np.mean(result.errors[p])
                std = np.std(result.errors[p])
                median = np.median(result.errors[p])
                lines.append(f"  {p:>12.3f} | {mean:>10.2f} | {std:>10.2f} | {median:>12.2f}")
            lines.append(f"\n  Correlation (log scale): r = {result.correlation:.3f}")
            lines.append(f"  Physics valid (r > 0.5): {'YES' if result.correlation > 0.5 else 'NO'}")

        # Amplitude Test Results
        if 'amplitude' in self.results:
            result = self.results['amplitude']
            params = sorted(result.errors.keys())
            lines.append("\n" + "-" * 70)
            lines.append("AMPLITUDE SENSITIVITY TEST")
            lines.append("-" * 70)
            lines.append(f"  Range: {min(params):.1f} to {max(params):.0f} nAm")
            lines.append(f"  Positions tested: {result.n_positions}")
            lines.append(f"  Trials per position: {result.n_trials_per_position}")
            lines.append(f"\n  {'Amplitude (nAm)':>15} | {'Mean (mm)':>10} | {'Std (mm)':>10} | {'Median (mm)':>12}")
            lines.append("  " + "-" * 55)
            for p in params:
                mean = np.mean(result.errors[p])
                std = np.std(result.errors[p])
                median = np.median(result.errors[p])
                lines.append(f"  {p:>15.1f} | {mean:>10.2f} | {std:>10.2f} | {median:>12.2f}")
            lines.append(f"\n  Correlation (log scale): r = {result.correlation:.3f}")
            lines.append(f"  Physics valid (r < -0.3): {'YES' if result.correlation < -0.3 else 'NO'}")

        # Summary
        lines.append("\n" + "=" * 70)
        lines.append("SUMMARY")
        lines.append("=" * 70)
        summary = self.get_summary()
        all_valid = all(t['physics_valid'] for t in summary['tests'].values())
        lines.append(f"\n  Tests completed: {len(self.results)}")
        for test_name, stats in summary['tests'].items():
            status = "PASSED" if stats['physics_valid'] else "FAILED"
            lines.append(f"  {test_name:12s}: r = {stats['correlation']:+.3f}  [{status}]")

        if all_valid:
            lines.append("\n  ✓ ALL PHYSICS TESTS PASSED")
            lines.append("    Source localization exhibits correct signal/noise behavior.")
        else:
            lines.append("\n  ⚠ SOME PHYSICS TESTS FAILED")
            lines.append("    Review results for potential issues.")

        lines.append("\n" + "=" * 70)

        report = "\n".join(lines)

        if output_path:
            output_path = Path(output_path)
            with open(output_path, 'w') as f:
                f.write(report)
            if self.verbose:
                print(f"Report saved to: {output_path}")

        return report
