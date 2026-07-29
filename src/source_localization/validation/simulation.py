"""
Dipole simulation for source localization validation.

This module creates synthetic EEG data from simulated dipole sources with known
ground truth positions and orientations. The synthetic data is used to validate
source localization accuracy by comparing recovered source positions against
known true positions.

Classes
-------
DipoleSimulator
    Simulate EEG data from dipole sources for validation testing.

Examples
--------
>>> from source_localization.validation import DipoleSimulator
>>> simulator = DipoleSimulator(forward_model, info, source_space)
>>> eeg_data, metadata = simulator.simulate_dipole(
...     position_mm=[0, 0, 5],
...     amplitude_nAm=50.0,
...     snr_db=10.0
... )
>>> print(f"Simulated dipole at {metadata['actual_position_mm']} mm")
"""

import numpy as np
import mne
from typing import Tuple, Optional, Dict

from .noise import generate_noise

__all__ = ['DipoleSimulator']


class DipoleSimulator:
    """
    Simulate EEG data from dipole sources for validation testing.

    This class generates synthetic EEG recordings by:
    1. Placing dipoles at known locations
    2. Computing scalp potentials using the forward model
    3. Adding realistic noise at specified SNR

    Parameters
    ----------
    forward_model : mne.Forward
        Forward solution (leadfield matrix mapping sources to electrodes)
    info : mne.Info
        EEG info object with channel information
    source_space : mne.SourceSpaces
        Source space definition (volumetric, surface, or discrete)
    verbose : bool, default=True
        Print progress and diagnostic messages

    Attributes
    ----------
    leadfield : ndarray, shape (n_channels, n_dipoles * 3)
        Forward model leadfield matrix
    source_positions : ndarray, shape (n_sources, 3)
        Source positions in mm [x, y, z]
    n_channels : int
        Number of EEG channels
    n_dipoles : int
        Number of source dipoles

    Examples
    --------
    >>> # Initialize simulator with forward model
    >>> simulator = DipoleSimulator(fwd, info, src, verbose=False)
    >>> print(f"Simulator ready: {simulator.n_channels} channels, {simulator.n_dipoles} sources")

    >>> # Simulate single dipole at specific location
    >>> eeg_data, meta = simulator.simulate_dipole(
    ...     position_mm=[0, 0, 5],
    ...     amplitude_nAm=50.0,
    ...     snr_db=10.0,
    ...     duration_s=1.0
    ... )
    >>> print(f"Localization error: {meta['position_error_mm']:.2f} mm")

    >>> # Simulate dipole at ROI centroid
    >>> roi_sources = np.array([10, 11, 12, 13])  # Source indices in ROI
    >>> eeg_data, meta = simulator.simulate_roi_dipole(
    ...     roi_sources=roi_sources,
    ...     roi_label="Primary Motor Cortex",
    ...     amplitude_nAm=50.0
    ... )
    """

    def __init__(
        self,
        forward_model: mne.Forward,
        info: mne.Info,
        source_space: mne.SourceSpaces,
        verbose: bool = True
    ):
        """Initialize dipole simulator."""
        self.fwd = forward_model
        self.info = info
        self.src = source_space
        self.verbose = verbose

        # Extract leadfield matrix (channels × dipoles × 3 orientations)
        self.leadfield = self.fwd['sol']['data']
        self.n_channels = self.leadfield.shape[0]
        self.n_dipoles = self.leadfield.shape[1] // 3  # 3 components per dipole

        # Get source positions
        self.source_positions = self._get_source_positions()

        # Electrode positions, for spatially-correlated noise generation
        self.electrode_positions_mm = self._get_electrode_positions()

        if self.verbose:
            print(f"DipoleSimulator initialized:")
            print(f"  Channels: {self.n_channels}")
            print(f"  Sources: {self.n_dipoles}")
            print(f"  Source positions shape: {self.source_positions.shape}")

    def _get_electrode_positions(self) -> Optional[np.ndarray]:
        """
        Extract electrode positions from info, in mm.

        Returns None if positions are unavailable or degenerate (all-zero or
        NaN ``loc`` fields), in which case spatially-correlated noise cannot be
        generated and requesting it raises.
        """
        positions_m = np.array([ch['loc'][:3] for ch in self.info['chs']])

        if positions_m.shape[0] != self.n_channels:
            return None
        if not np.isfinite(positions_m).all():
            return None
        if np.allclose(positions_m, 0):
            return None

        return positions_m * 1000  # m to mm

    def _get_source_positions(self) -> np.ndarray:
        """
        Extract source positions from forward solution.

        IMPORTANT: Uses fwd['source_rr'] (positions in the forward solution)
        rather than source space positions. The forward solution may filter
        out sources too close to the BEM surface, so source space can have
        more positions than the leadfield has columns.

        Returns
        -------
        positions : ndarray, shape (n_sources, 3)
            Source positions in mm [x, y, z]. Number of sources matches
            the leadfield matrix (n_dipoles).
        """
        # Use positions from forward solution to ensure alignment with leadfield
        # fwd['source_rr'] contains only sources that survived forward model filtering
        positions_m = self.fwd['source_rr']

        if self.verbose:
            print(f"\n=== Extracting source positions from forward solution ===")
            print(f"  Forward source_rr shape: {positions_m.shape}")
            print(f"  Leadfield n_dipoles: {self.n_dipoles}")
            if positions_m.shape[0] != self.n_dipoles:
                print(f"  WARNING: Position count mismatch!")
            print(f"===================================\n")

        return positions_m * 1000  # Convert m to mm

    def simulate_dipole(
        self,
        position_mm: np.ndarray,
        orientation: Optional[np.ndarray] = None,
        amplitude_nAm: float = 50.0,
        duration_s: float = 1.0,
        sfreq: float = 500.0,
        snr_db: float = 10.0,
        noise_seed: Optional[int] = None,
        noise_mode: str = "snr",
        noise_variance_uV2: float = 1.0,
        noise_type: str = "white",
        noise_spatial_scale_mm: float = 3.0,
        noise_temporal_exponent: float = 1.0,
        time_course: Optional[np.ndarray] = None
    ) -> Tuple[np.ndarray, Dict]:
        """
        Simulate EEG from a single dipole source.

        Parameters
        ----------
        position_mm : array-like, shape (3,)
            Target dipole position in mm [x, y, z]. The nearest source point
            will be used.
        orientation : array-like, shape (3,), optional
            Dipole orientation (normalized automatically). If None, a random
            orientation is used.
        amplitude_nAm : float, default=50.0
            Dipole amplitude in nanoAmpere-meters (nAm). Typical range: 10-100 nAm.
        duration_s : float, default=1.0
            Signal duration in seconds
        sfreq : float, default=500.0
            Sampling frequency in Hz
        snr_db : float, default=10.0
            Signal-to-noise ratio in dB (used when noise_mode="snr"):
            - 0 dB: Signal power equals noise power
            - 10 dB: Signal is 10× stronger than noise
            - 20 dB: Signal is 100× stronger than noise
        noise_seed : int, optional
            Random seed for reproducible noise
        noise_mode : str, default="snr"
            Noise scaling mode:
            - "snr": Scale noise to achieve target SNR (amplitude-independent)
            - "fixed_variance": Use fixed noise variance (amplitude-sensitive)
        noise_variance_uV2 : float, default=1.0
            Noise variance in µV² (used when noise_mode="fixed_variance").
            Typical EEG sensor noise: 0.1-10 µV²
        noise_type : {'white', 'spatial', 'temporal', 'colored'}, default='white'
            Noise structure. 'white' satisfies the scaled-identity noise
            assumption in the inverse and gives best-case performance;
            'colored' (spatially correlated + 1/f) violates it and gives a
            realistic lower bound. See :mod:`source_localization.validation.noise`.
            Independent of noise_mode, which controls noise *level*.
        noise_spatial_scale_mm : float, default=3.0
            Correlation length in mm for spatially-correlated noise types.
        noise_temporal_exponent : float, default=1.0
            Spectral exponent for 1/f-shaped noise types (1.0 = pink).

        Returns
        -------
        eeg_data : ndarray, shape (n_channels, n_times)
            Simulated EEG with added noise
        metadata : dict
            Simulation metadata including:
            - requested_position_mm : Original position requested (use for localization error)
            - snapped_source_position_mm : Nearest source position where dipole was placed
            - snapping_error_mm : Distance from requested to snapped position
            - dipole_position_mm : (legacy alias for requested_position_mm)
            - actual_position_mm : (legacy alias for snapped_source_position_mm)
            - position_error_mm : (legacy alias for snapping_error_mm)
            - source_index : Index of nearest source
            - orientation : Dipole orientation (normalized)
            - amplitude_nAm : Dipole amplitude
            - snr_db : Requested or achieved SNR
            - actual_snr_db : Achieved SNR
            - noise_mode : Noise scaling mode used

        Examples
        --------
        >>> # Simulate dipole with SNR-based noise (default)
        >>> eeg, meta = simulator.simulate_dipole(position_mm=[0, 0, 5])

        >>> # Simulate with fixed noise for amplitude sensitivity testing
        >>> eeg, meta = simulator.simulate_dipole(
        ...     position_mm=[2, -1, 6],
        ...     amplitude_nAm=75.0,
        ...     noise_mode="fixed_variance",
        ...     noise_variance_uV2=1.0,
        ...     noise_seed=42
        ... )
        >>> print(f"Achieved SNR: {meta['actual_snr_db']:.1f} dB")
        """
        # Find nearest source point
        position_m = np.asarray(position_mm) / 1000.0  # Convert to meters for MNE
        distances = np.linalg.norm(self.source_positions / 1000.0 - position_m, axis=1)
        source_idx = np.argmin(distances)
        actual_position_mm = self.source_positions[source_idx]

        # Set orientation
        if orientation is None:
            # Random orientation
            rng = np.random.RandomState(noise_seed)
            orientation = rng.randn(3)
        orientation = np.array(orientation) / np.linalg.norm(orientation)

        # Create dipole moment vector (3 components per source, all zeros except target)
        n_times = int(duration_s * sfreq)
        dipole_moment = np.zeros((self.n_dipoles * 3, n_times))

        # Set dipole amplitude (constant over time unless a waveform is given)
        # Convert nAm to Am (MNE uses SI units)
        amplitude_Am = amplitude_nAm * 1e-9

        if time_course is None:
            # Unchanged constant-moment path. Deliberately a separate branch
            # rather than multiplying by np.ones(n_times), so that every
            # existing call reproduces its previous output bit-for-bit.
            for i in range(3):
                dipole_moment[source_idx * 3 + i, :] = orientation[i] * amplitude_Am
        else:
            tc = np.asarray(time_course, dtype=float).ravel()
            if tc.shape[0] != n_times:
                raise ValueError(
                    f"time_course has {tc.shape[0]} samples but duration_s x sfreq "
                    f"implies {n_times}"
                )
            for i in range(3):
                dipole_moment[source_idx * 3 + i, :] = orientation[i] * amplitude_Am * tc

        # Generate clean EEG using forward model
        eeg_clean = self.leadfield @ dipole_moment

        # Add realistic noise (unit variance; scaled to target SNR below)
        rng = np.random.RandomState(noise_seed)
        noise = generate_noise(
            self.n_channels, n_times, rng,
            noise_type=noise_type,
            electrode_positions_mm=self.electrode_positions_mm,
            spatial_scale_mm=noise_spatial_scale_mm,
            temporal_exponent=noise_temporal_exponent
        )

        # Calculate signal power
        signal_power = np.mean(eeg_clean ** 2)
        noise_power_raw = np.mean(noise ** 2)

        if noise_mode == "fixed_variance":
            # Fixed noise variance mode: amplitude affects SNR
            # Convert µV² to V² (MNE uses SI units internally)
            target_noise_variance_V2 = noise_variance_uV2 * 1e-12
            noise_scale = np.sqrt(target_noise_variance_V2 / noise_power_raw)
            noise_power_added = target_noise_variance_V2
            # Calculate achieved SNR
            if noise_power_added > 0:
                actual_snr_db = float(10 * np.log10(signal_power / noise_power_added))
            else:
                actual_snr_db = np.inf
        else:
            # SNR-based mode (default): scale noise to achieve target SNR
            snr_linear = 10 ** (snr_db / 10)
            noise_scale = np.sqrt(signal_power / (snr_linear * noise_power_raw))
            noise_power_added = noise_scale ** 2 * noise_power_raw
            actual_snr_db = float(10 * np.log10(signal_power / noise_power_added))

        eeg_data = eeg_clean + noise_scale * noise

        # Metadata
        # Note on position naming:
        # - requested_position_mm: The position requested by validation (e.g., uniform grid point)
        # - snapped_source_position_mm: Nearest source space position where dipole was actually placed
        # - snapping_error_mm: Distance between requested and snapped positions
        # For localization error calculation, use requested_position_mm as reference (not snapped)
        metadata = {
            'requested_position_mm': position_mm,           # Original request (use for localization error)
            'snapped_source_position_mm': actual_position_mm,  # Where dipole was placed (snapped to source)
            'snapping_error_mm': np.linalg.norm(position_mm - actual_position_mm),  # Snapping distance
            # Legacy aliases for backward compatibility
            'dipole_position_mm': position_mm,
            'actual_position_mm': actual_position_mm,
            'position_error_mm': np.linalg.norm(position_mm - actual_position_mm),
            'source_index': source_idx,
            'orientation': orientation,
            'amplitude_nAm': amplitude_nAm,
            'snr_db': snr_db if noise_mode == "snr" else actual_snr_db,
            'duration_s': duration_s,
            'sfreq': sfreq,
            'n_times': n_times,
            'signal_power': float(signal_power),
            'noise_power_added': float(noise_power_added),
            'actual_snr_db': actual_snr_db,
            'noise_mode': noise_mode,
            'noise_variance_uV2': noise_variance_uV2 if noise_mode == "fixed_variance" else None,
            'noise_type': noise_type,
            'noise_spatial_scale_mm': (
                noise_spatial_scale_mm if noise_type in ('spatial', 'colored') else None
            ),
            'noise_temporal_exponent': (
                noise_temporal_exponent if noise_type in ('temporal', 'colored') else None
            )
        }

        if self.verbose:
            print(f"\nSimulated dipole:")
            print(f"  Requested position: {position_mm} mm")
            print(f"  Snapped to source: {actual_position_mm} mm")
            print(f"  Snapping error: {metadata['snapping_error_mm']:.3f} mm")
            print(f"  Source index: {source_idx}/{self.n_dipoles}")
            print(f"  Orientation: {orientation}")
            print(f"  Amplitude: {amplitude_nAm} nAm")
            if noise_mode == "fixed_variance":
                print(f"  Noise mode: fixed_variance ({noise_variance_uV2} µV²)")
                print(f"  Achieved SNR: {actual_snr_db:.2f} dB")
            else:
                print(f"  SNR: {snr_db} dB (actual: {actual_snr_db:.2f} dB)")
            print(f"  Noise type: {noise_type}")

        return eeg_data, metadata

    def simulate_roi_dipole(
        self,
        roi_sources: np.ndarray,
        roi_label: str = "Unknown",
        **kwargs
    ) -> Tuple[np.ndarray, Dict]:
        """
        Simulate dipole at the center of an ROI.

        Parameters
        ----------
        roi_sources : ndarray, shape (n_sources_in_roi,)
            Indices of sources belonging to the ROI
        roi_label : str, default="Unknown"
            ROI name for metadata
        **kwargs : dict
            Additional arguments passed to simulate_dipole():
            amplitude_nAm, duration_s, sfreq, snr_db, noise_seed

        Returns
        -------
        eeg_data : ndarray
            Simulated EEG
        metadata : dict
            Simulation metadata with additional ROI information:
            - roi_label : ROI name
            - roi_n_sources : Number of sources in ROI
            - roi_centroid_mm : ROI centroid position

        Examples
        --------
        >>> # Get sources for a specific ROI
        >>> roi_sources = np.where(roi_mapping == 15)[0]  # ROI ID 15
        >>> eeg, meta = simulator.simulate_roi_dipole(
        ...     roi_sources=roi_sources,
        ...     roi_label="Primary Visual Cortex",
        ...     amplitude_nAm=60.0,
        ...     snr_db=12.0
        ... )
        >>> print(f"Simulated {meta['roi_label']} with {meta['roi_n_sources']} sources")
        """
        # Compute ROI centroid
        roi_positions = self.source_positions[roi_sources]
        roi_centroid = np.mean(roi_positions, axis=0)

        # Simulate dipole at centroid
        eeg_data, metadata = self.simulate_dipole(
            position_mm=roi_centroid,
            **kwargs
        )

        # Add ROI metadata
        metadata['roi_label'] = roi_label
        metadata['roi_n_sources'] = len(roi_sources)
        metadata['roi_centroid_mm'] = roi_centroid

        if self.verbose:
            print(f"  ROI: {roi_label} ({len(roi_sources)} sources)")

        return eeg_data, metadata

    def simulate_two_dipoles(
        self,
        position1_mm: np.ndarray,
        position2_mm: np.ndarray,
        orientation1: Optional[np.ndarray] = None,
        orientation2: Optional[np.ndarray] = None,
        amplitude1_nAm: float = 50.0,
        amplitude2_nAm: float = 50.0,
        **kwargs
    ) -> Tuple[np.ndarray, Dict]:
        """
        Simulate EEG from two simultaneous dipoles.

        Useful for testing spatial resolution limits and ability to resolve
        multiple concurrent sources.

        Parameters
        ----------
        position1_mm, position2_mm : array-like, shape (3,)
            Dipole positions in mm
        orientation1, orientation2 : array-like, shape (3,), optional
            Dipole orientations
        amplitude1_nAm, amplitude2_nAm : float, default=50.0
            Dipole amplitudes in nAm
        **kwargs : dict
            Additional arguments: duration_s, sfreq, snr_db, noise_seed

        Returns
        -------
        eeg_data : ndarray
            Combined EEG from both dipoles with noise
        metadata : dict
            Simulation metadata for both dipoles:
            - dipole1 : dict with metadata for first dipole
            - dipole2 : dict with metadata for second dipole
            - separation_mm : Distance between actual dipole positions
            - snr_db : Requested SNR
            - signal_power : Combined signal power

        Examples
        --------
        >>> # Test ability to resolve two nearby sources
        >>> eeg, meta = simulator.simulate_two_dipoles(
        ...     position1_mm=[0, 0, 5],
        ...     position2_mm=[2, 0, 5],  # 2mm apart
        ...     amplitude1_nAm=50.0,
        ...     amplitude2_nAm=50.0,
        ...     snr_db=10.0
        ... )
        >>> print(f"Dipole separation: {meta['separation_mm']:.2f} mm")
        """
        # Simulate first dipole (no noise yet)
        kwargs_copy = kwargs.copy()
        kwargs_copy['snr_db'] = np.inf  # Infinite SNR = no noise
        eeg1, meta1 = self.simulate_dipole(
            position_mm=position1_mm,
            orientation=orientation1,
            amplitude_nAm=amplitude1_nAm,
            **kwargs_copy
        )

        # Simulate second dipole (no noise yet)
        eeg2, meta2 = self.simulate_dipole(
            position_mm=position2_mm,
            orientation=orientation2,
            amplitude_nAm=amplitude2_nAm,
            **kwargs_copy
        )

        # Combine signals
        eeg_clean = eeg1 + eeg2

        # Add noise to combined signal
        snr_db = kwargs.get('snr_db', 10.0)
        noise_seed = kwargs.get('noise_seed', None)
        duration_s = kwargs.get('duration_s', 1.0)
        sfreq = kwargs.get('sfreq', 500.0)

        noise_type = kwargs.get('noise_type', 'white')
        noise_spatial_scale_mm = kwargs.get('noise_spatial_scale_mm', 3.0)
        noise_temporal_exponent = kwargs.get('noise_temporal_exponent', 1.0)

        n_times = int(duration_s * sfreq)
        rng = np.random.RandomState(noise_seed)
        noise = generate_noise(
            self.n_channels, n_times, rng,
            noise_type=noise_type,
            electrode_positions_mm=self.electrode_positions_mm,
            spatial_scale_mm=noise_spatial_scale_mm,
            temporal_exponent=noise_temporal_exponent
        )

        signal_power = np.mean(eeg_clean ** 2)
        noise_power = np.mean(noise ** 2)
        snr_linear = 10 ** (snr_db / 10)
        noise_scale = np.sqrt(signal_power / (snr_linear * noise_power))

        eeg_data = eeg_clean + noise_scale * noise

        # Combined metadata.
        #
        # Two separations exist and they are NOT interchangeable:
        #   requested_separation_mm : distance between the positions the caller
        #       asked for. In a resolvability sweep this is the INDEPENDENT
        #       VARIABLE -- it is what the experiment varies.
        #   separation_mm           : distance between the SNAPPED source points
        #       the dipoles were actually placed on. This is what was physically
        #       simulated, and it can differ from the request by up to the local
        #       source spacing.
        # Reporting only the latter (the behaviour before this fix) silently
        # loses the design variable; reporting only the former would misstate
        # what was simulated. Both are recorded.
        separation_mm = np.linalg.norm(
            np.asarray(meta1['snapped_source_position_mm'])
            - np.asarray(meta2['snapped_source_position_mm'])
        )
        requested_separation_mm = np.linalg.norm(
            np.asarray(meta1['requested_position_mm'])
            - np.asarray(meta2['requested_position_mm'])
        )

        metadata = {
            'dipole1': meta1,
            'dipole2': meta2,
            # Achieved (snapped) separation -- what was actually simulated.
            'separation_mm': float(separation_mm),
            # Requested separation -- the design variable of a resolvability sweep.
            'requested_separation_mm': float(requested_separation_mm),
            # How far snapping moved the pair together/apart.
            'separation_snapping_error_mm': float(
                abs(separation_mm - requested_separation_mm)
            ),
            'snr_db': snr_db,
            'signal_power': float(signal_power),
            'noise_power_added': float(noise_scale ** 2 * noise_power),
            'noise_type': noise_type
        }

        if self.verbose:
            print(f"\nSimulated two dipoles:")
            print(f"  Separation: {separation_mm:.2f} mm")
            print(f"  Combined SNR: {snr_db} dB")

        return eeg_data, metadata

    def create_mne_raw(
        self,
        eeg_data: np.ndarray,
        sfreq: float = 500.0
    ) -> mne.io.RawArray:
        """
        Convert simulated EEG array to MNE Raw object.

        Parameters
        ----------
        eeg_data : ndarray, shape (n_channels, n_times)
            Simulated EEG data
        sfreq : float, default=500.0
            Sampling frequency in Hz

        Returns
        -------
        raw : mne.io.RawArray
            MNE Raw object ready for inverse solution

        Examples
        --------
        >>> eeg_data, _ = simulator.simulate_dipole(position_mm=[0, 0, 5])
        >>> raw = simulator.create_mne_raw(eeg_data, sfreq=500.0)
        >>> print(raw)
        """
        # Create info with correct sampling frequency
        info = mne.create_info(
            ch_names=self.info['ch_names'],
            sfreq=sfreq,
            ch_types='eeg'
        )

        # Copy channel locations from original info
        if 'chs' in self.info:
            for i, ch in enumerate(info['chs']):
                if i < len(self.info['chs']):
                    ch['loc'] = self.info['chs'][i]['loc'].copy()

        raw = mne.io.RawArray(eeg_data, info, verbose=False)

        return raw

    # ------------------------------------------------------------------
    # WP-4: multiple co-active sources and spatially extended patches.
    #
    # Added for the NIMG-26-1224 revision, answering R3-1b ("validation is
    # limited to single point-like dipoles and does not test scenarios with
    # multiple simultaneously active sources or spatially extended (patch-like)
    # activations") and the time-series-fidelity part of R3-2b.
    #
    # Both are new entry points. `simulate_dipole` and `simulate_two_dipoles`
    # are untouched apart from the additive `time_course` argument, whose
    # default preserves their previous output bit-for-bit.
    # ------------------------------------------------------------------

    def _assemble_moment(
        self,
        source_indices,
        orientations: np.ndarray,
        amplitudes_Am: np.ndarray,
        time_courses: np.ndarray,
    ) -> np.ndarray:
        """Build the (n_dipoles*3, n_times) moment matrix for several sources.

        Sources sharing a snapped index are summed, which is what physically
        happens when two requested positions land on the same grid point.
        """
        n_times = time_courses.shape[1]
        moment = np.zeros((self.n_dipoles * 3, n_times))
        for k, idx in enumerate(source_indices):
            for i in range(3):
                moment[idx * 3 + i, :] += (
                    orientations[k, i] * amplitudes_Am[k] * time_courses[k]
                )
        return moment

    def simulate_n_dipoles(
        self,
        positions_mm,
        orientations=None,
        amplitudes_nAm=None,
        time_courses=None,
        duration_s: float = 1.0,
        sfreq: float = 500.0,
        snr_db: float = 10.0,
        noise_seed: Optional[int] = None,
        noise_type: str = "white",
        noise_spatial_scale_mm: float = 3.0,
        noise_temporal_exponent: float = 1.0,
    ) -> Tuple[np.ndarray, Dict]:
        """
        Simulate EEG from N simultaneously active dipoles.

        Generalizes :meth:`simulate_two_dipoles` to any number of sources with
        independent waveforms, which is what makes *controllable source
        correlation* possible. Note that the pre-existing two-dipole path gives
        perfectly correlated (r = 1) sources, because each dipole carries a
        constant DC moment with no time course.

        Parameters
        ----------
        positions_mm : array-like, shape (N, 3)
            Requested dipole positions in mm. Each is snapped to the nearest
            source, exactly as in :meth:`simulate_dipole`.
        orientations : array-like, shape (N, 3), optional
            Dipole orientations; normalized internally. Default: isotropic
            random, drawn from a single RandomState seeded with ``noise_seed``.
        amplitudes_nAm : array-like, shape (N,), optional
            Per-source amplitudes. Default: 50 nAm each.
        time_courses : array-like, shape (N, n_times), optional
            Per-source waveforms. Default: constant (DC), i.e. fully correlated
            sources -- the behaviour of the existing two-dipole path.
        snr_db : float, default=10.0
            SNR of the *combined* signal, matching `simulate_two_dipoles`.

        Returns
        -------
        eeg_data : ndarray, shape (n_channels, n_times)
        metadata : dict
            ``dipoles`` (per-source metadata), ``requested_positions_mm``,
            ``snapped_positions_mm``, ``snapping_error_mm``,
            ``pairwise_requested_separation_mm``, ``pairwise_snapped_separation_mm``,
            ``source_correlation`` (empirical, from the time courses),
            ``collided`` (True when two sources snapped to the same grid point),
            plus the usual power/SNR fields.
        """
        positions_mm = np.atleast_2d(np.asarray(positions_mm, dtype=float))
        n_src = positions_mm.shape[0]
        n_times = int(duration_s * sfreq)

        if amplitudes_nAm is None:
            amplitudes_nAm = np.full(n_src, 50.0)
        amplitudes_nAm = np.asarray(amplitudes_nAm, dtype=float).ravel()

        rng = np.random.RandomState(noise_seed)
        if orientations is None:
            orientations = rng.randn(n_src, 3)
        orientations = np.atleast_2d(np.asarray(orientations, dtype=float))
        orientations = orientations / np.linalg.norm(orientations, axis=1, keepdims=True)

        if time_courses is None:
            time_courses = np.ones((n_src, n_times))
        time_courses = np.atleast_2d(np.asarray(time_courses, dtype=float))
        if time_courses.shape != (n_src, n_times):
            raise ValueError(
                f"time_courses must have shape ({n_src}, {n_times}), got "
                f"{time_courses.shape}"
            )

        # Snap each requested position to the nearest source point.
        src_pos_m = self.source_positions / 1000.0
        source_indices, snapped, snap_err = [], [], []
        for p in positions_mm:
            d = np.linalg.norm(src_pos_m - p / 1000.0, axis=1)
            idx = int(np.argmin(d))
            source_indices.append(idx)
            snapped.append(self.source_positions[idx])
            snap_err.append(float(np.linalg.norm(self.source_positions[idx] - p)))
        snapped = np.asarray(snapped)

        # Two requested positions can snap onto the same grid point. That is not
        # an error, but it means the simulation no longer contains N distinct
        # sources, so any resolvability claim about that trial is meaningless.
        collided = len(set(source_indices)) < n_src

        moment = self._assemble_moment(
            source_indices, orientations, amplitudes_nAm * 1e-9, time_courses)
        eeg_clean = self.leadfield @ moment

        noise = generate_noise(
            self.n_channels, n_times, np.random.RandomState(noise_seed),
            noise_type=noise_type,
            electrode_positions_mm=self.electrode_positions_mm,
            spatial_scale_mm=noise_spatial_scale_mm,
            temporal_exponent=noise_temporal_exponent,
        )

        signal_power = float(np.mean(eeg_clean ** 2))
        noise_power_raw = float(np.mean(noise ** 2))
        if np.isinf(snr_db):
            noise_scale = 0.0
        else:
            snr_linear = 10 ** (snr_db / 10)
            noise_scale = np.sqrt(signal_power / (snr_linear * noise_power_raw))
        eeg_data = eeg_clean + noise_scale * noise

        req_sep, snap_sep = {}, {}
        for a in range(n_src):
            for b in range(a + 1, n_src):
                req_sep[f"{a}-{b}"] = float(
                    np.linalg.norm(positions_mm[a] - positions_mm[b]))
                snap_sep[f"{a}-{b}"] = float(
                    np.linalg.norm(snapped[a] - snapped[b]))

        corr = (np.corrcoef(time_courses) if n_src > 1 and
                np.all(time_courses.std(axis=1) > 0) else None)

        metadata = {
            'n_sources': n_src,
            'dipoles': [
                {
                    'requested_position_mm': positions_mm[k],
                    'snapped_source_position_mm': snapped[k],
                    'snapping_error_mm': snap_err[k],
                    'source_index': source_indices[k],
                    'orientation': orientations[k],
                    'amplitude_nAm': float(amplitudes_nAm[k]),
                }
                for k in range(n_src)
            ],
            'requested_positions_mm': positions_mm,
            'snapped_positions_mm': snapped,
            'snapping_error_mm': snap_err,
            'pairwise_requested_separation_mm': req_sep,
            'pairwise_snapped_separation_mm': snap_sep,
            'collided': bool(collided),
            'source_correlation': corr,
            'signal_power': signal_power,
            'snr_db': snr_db,
            'noise_type': noise_type,
            'duration_s': duration_s,
            'sfreq': sfreq,
            'n_times': n_times,
        }
        return eeg_data, metadata

    def simulate_patch(
        self,
        center_mm,
        radius_mm: float = 1.0,
        amplitude_nAm: float = 50.0,
        orientation=None,
        time_course=None,
        coherent: bool = True,
        **kwargs
    ) -> Tuple[np.ndarray, Dict]:
        """
        Simulate a spatially extended (patch-like) activation.

        Every source within ``radius_mm`` of ``center_mm`` is activated. The
        total moment is held equal to ``amplitude_nAm`` regardless of how many
        sources the patch covers, so a patch and a point source of the same
        strength are comparable -- otherwise a larger patch would trivially win
        by carrying more current.

        Parameters
        ----------
        center_mm : array-like, shape (3,)
        radius_mm : float, default=1.0
            Patch radius. Sources within this distance of the centre are active.
        coherent : bool, default=True
            True: all sources share one waveform and one orientation (a
            synchronously active patch, the physiologically standard case).
            False: independent random waveforms per source.
        time_course : array-like, optional
            Waveform for the coherent case. Default: constant.

        Returns
        -------
        eeg_data, metadata
            ``metadata['n_patch_sources']`` and ``metadata['patch_extent_mm']``
            (RMS spread of the activated sources about their centroid) record
            what was actually simulated -- the requested radius is not
            achievable exactly on a discrete grid.
        """
        center_mm = np.asarray(center_mm, dtype=float)
        d = np.linalg.norm(self.source_positions - center_mm, axis=1)
        members = np.flatnonzero(d <= radius_mm)
        if members.size == 0:  # radius smaller than local grid spacing
            members = np.array([int(np.argmin(d))])

        n_mem = members.size
        positions = self.source_positions[members]

        duration_s = kwargs.get('duration_s', 1.0)
        sfreq = kwargs.get('sfreq', 500.0)
        n_times = int(duration_s * sfreq)
        rng = np.random.RandomState(kwargs.get('noise_seed'))

        if coherent:
            tc = (np.ones(n_times) if time_course is None
                  else np.asarray(time_course, dtype=float).ravel())
            time_courses = np.tile(tc, (n_mem, 1))
            orient = rng.randn(3) if orientation is None else np.asarray(orientation)
            orientations = np.tile(orient, (n_mem, 1))
        else:
            time_courses = rng.randn(n_mem, n_times)
            orientations = rng.randn(n_mem, 3)

        eeg, meta = self.simulate_n_dipoles(
            positions_mm=positions,
            orientations=orientations,
            amplitudes_nAm=np.full(n_mem, amplitude_nAm / n_mem),
            time_courses=time_courses,
            **kwargs
        )

        centroid = positions.mean(axis=0)
        meta.update({
            'patch_center_mm': center_mm,
            'patch_radius_mm': float(radius_mm),
            'n_patch_sources': int(n_mem),
            'patch_source_indices': members,
            'patch_centroid_mm': centroid,
            'patch_extent_mm': float(
                np.sqrt(((positions - centroid) ** 2).sum(axis=1).mean())),
            'patch_coherent': bool(coherent),
            'total_amplitude_nAm': float(amplitude_nAm),
        })
        return eeg, meta
