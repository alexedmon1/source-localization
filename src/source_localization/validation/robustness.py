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
from typing import Dict, List, Optional, Tuple, Union
from dataclasses import dataclass, field
from scipy import stats
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
