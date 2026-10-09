"""
Noise generation for source localization validation.

The custom minimum-norm inverse in ``steps/inverse_solution.py`` regularizes
with a scaled identity, which is the assumption that sensor noise is white and
homoscedastic (C = I). Validation that injects i.i.d. Gaussian noise satisfies
that assumption exactly, so it cannot expose the cost of the assumption being
wrong. This module generates noise that violates it, in the two ways real EEG
sensor noise does:

- **Spatially correlated:** nearby electrodes see shared signal (volume
  conduction, reference, common physiological artifact). Modeled as a
  non-identity channel covariance with exponential decay in inter-electrode
  distance, applied to white noise via its Cholesky factor.
- **Temporally colored:** real EEG noise has a 1/f spectrum rather than a flat
  one. Applied as FFT amplitude shaping.

All generators return **unit-variance** noise, so callers can scale to a target
SNR or variance with the same arithmetic they use for white noise.

Functions
---------
generate_noise
    Generate noise of a given type, shape (n_channels, n_times).
spatial_covariance
    Build an exponential-decay channel covariance from electrode positions.

Examples
--------
>>> rng = np.random.RandomState(42)
>>> noise = generate_noise(32, 500, rng, noise_type='white')
>>> noise.shape
(32, 500)
"""

import numpy as np
from typing import Optional

__all__ = ['generate_noise', 'spatial_covariance', 'NOISE_TYPES', 'RecordedBackground']

#: Supported noise types. 'colored' applies both spatial and temporal shaping.
NOISE_TYPES = ('white', 'spatial', 'temporal', 'colored')


def spatial_covariance(
    electrode_positions_mm: np.ndarray,
    spatial_scale_mm: float = 3.0,
    jitter: float = 1e-6
) -> np.ndarray:
    """
    Build an exponential-decay channel covariance matrix.

    Covariance between electrodes i and j is ``exp(-d_ij / spatial_scale_mm)``,
    where ``d_ij`` is their Euclidean separation. Diagonal is 1, so the matrix
    is a correlation matrix and the generated noise has unit per-channel
    variance before any global rescaling.

    Parameters
    ----------
    electrode_positions_mm : ndarray, shape (n_channels, 3)
        Electrode positions in mm.
    spatial_scale_mm : float, default=3.0
        Correlation length. Electrodes this far apart have correlation 1/e.
        Larger values mean more broadly shared noise. For the 30-channel mouse
        array (~1-2 mm spacing) 3.0 mm gives strong neighbor correlation.
    jitter : float, default=1e-6
        Added to the diagonal to keep the matrix positive-definite for the
        Cholesky factorization.

    Returns
    -------
    cov : ndarray, shape (n_channels, n_channels)
        Symmetric positive-definite covariance matrix.

    Examples
    --------
    >>> pos = np.array([[0., 0., 0.], [1., 0., 0.], [10., 0., 0.]])
    >>> cov = spatial_covariance(pos, spatial_scale_mm=1.0)
    >>> bool(cov[0, 1] > cov[0, 2])  # near pair more correlated than far pair
    True
    """
    positions = np.asarray(electrode_positions_mm, dtype=float)
    if positions.ndim != 2 or positions.shape[1] != 3:
        raise ValueError(
            f"electrode_positions_mm must have shape (n_channels, 3), "
            f"got {positions.shape}"
        )
    if spatial_scale_mm <= 0:
        raise ValueError(f"spatial_scale_mm must be > 0, got {spatial_scale_mm}")

    diff = positions[:, None, :] - positions[None, :, :]
    distances = np.linalg.norm(diff, axis=-1)
    cov = np.exp(-distances / spatial_scale_mm)
    cov[np.diag_indices_from(cov)] += jitter
    return cov


def _apply_temporal_shaping(
    noise: np.ndarray,
    temporal_exponent: float = 1.0
) -> np.ndarray:
    """
    Shape white noise to a 1/f^exponent power spectrum.

    Amplitude is scaled by ``f^(-exponent/2)``, so power scales by
    ``f^(-exponent)``. The DC bin is zeroed (noise is mean-free); with
    ``temporal_exponent=1.0`` this yields pink noise.

    Parameters
    ----------
    noise : ndarray, shape (n_channels, n_times)
        White noise to shape.
    temporal_exponent : float, default=1.0
        Spectral exponent. 0 leaves the noise white, 1 gives pink (1/f),
        2 gives brown (1/f²).

    Returns
    -------
    shaped : ndarray, shape (n_channels, n_times)
        Spectrally shaped noise (not renormalized).
    """
    n_times = noise.shape[1]
    spectrum = np.fft.rfft(noise, axis=1)
    freqs = np.fft.rfftfreq(n_times, d=1.0)  # normalized; shaping is scale-free

    scaling = np.zeros_like(freqs)
    nonzero = freqs > 0
    scaling[nonzero] = freqs[nonzero] ** (-temporal_exponent / 2.0)
    scaling[~nonzero] = 0.0  # drop DC — noise should be mean-free

    return np.fft.irfft(spectrum * scaling[None, :], n=n_times, axis=1)


def generate_noise(
    n_channels: int,
    n_times: int,
    rng: np.random.RandomState,
    noise_type: str = 'white',
    electrode_positions_mm: Optional[np.ndarray] = None,
    spatial_scale_mm: float = 3.0,
    temporal_exponent: float = 1.0
) -> np.ndarray:
    """
    Generate unit-variance noise of a given type.

    Parameters
    ----------
    n_channels : int
        Number of channels.
    n_times : int
        Number of time samples.
    rng : np.random.RandomState
        Seeded random state, for reproducibility.
    noise_type : {'white', 'spatial', 'temporal', 'colored'}, default='white'
        - 'white': i.i.d. Gaussian. Satisfies the inverse's C = I assumption.
        - 'spatial': non-identity channel covariance, flat spectrum.
        - 'temporal': 1/f spectrum, identity channel covariance.
        - 'colored': both — the most realistic, and the hardest case for the
          scaled-identity inverse.
    electrode_positions_mm : ndarray, shape (n_channels, 3), optional
        Required for 'spatial' and 'colored'.
    spatial_scale_mm : float, default=3.0
        Correlation length for the spatial covariance. See
        :func:`spatial_covariance`.
    temporal_exponent : float, default=1.0
        Spectral exponent for 1/f shaping. See :func:`_apply_temporal_shaping`.

    Returns
    -------
    noise : ndarray, shape (n_channels, n_times)
        Noise normalized to unit variance across all channels and samples.

    Raises
    ------
    ValueError
        If ``noise_type`` is unknown, or if electrode positions are missing or
        the wrong length for a spatially-correlated type.

    Examples
    --------
    >>> rng = np.random.RandomState(0)
    >>> pos = np.random.RandomState(1).randn(8, 3)
    >>> noise = generate_noise(8, 256, rng, noise_type='colored',
    ...                        electrode_positions_mm=pos)
    >>> bool(np.isclose(noise.std(), 1.0))
    True
    """
    if noise_type not in NOISE_TYPES:
        raise ValueError(
            f"Unknown noise_type {noise_type!r}. Expected one of {NOISE_TYPES}."
        )

    needs_positions = noise_type in ('spatial', 'colored')
    if needs_positions:
        if electrode_positions_mm is None:
            raise ValueError(
                f"noise_type={noise_type!r} requires electrode_positions_mm "
                f"to build the spatial covariance."
            )
        positions = np.asarray(electrode_positions_mm, dtype=float)
        if positions.shape[0] != n_channels:
            raise ValueError(
                f"electrode_positions_mm has {positions.shape[0]} channels but "
                f"n_channels={n_channels}."
            )

    noise = rng.randn(n_channels, n_times)

    # Temporal shaping first: spatial mixing is a static linear map across
    # channels, so it preserves each channel's spectrum. The reverse order
    # would not preserve the channel covariance.
    if noise_type in ('temporal', 'colored'):
        noise = _apply_temporal_shaping(noise, temporal_exponent)

    if needs_positions:
        cov = spatial_covariance(positions, spatial_scale_mm)
        chol = np.linalg.cholesky(cov)
        noise = chol @ noise

    std = noise.std()
    if std == 0:
        raise ValueError(
            "Generated noise has zero variance; check n_times and "
            "temporal_exponent."
        )
    return noise / std


class RecordedBackground:
    """
    Noise taken from real recordings: the ``realistic`` validation regime's background.

    Synthetic generators cannot reproduce the structure of real EEG backgrounds (non-stationary, spatially
    structured activity of real networks). This draws background segments from the user's own recordings
    instead. Each draw is a contiguous window from a randomly chosen epoch of a randomly chosen recording,
    average-referenced, resampled to the caller's rate if it differs from the recordings', and returned at
    **unit mean power**. Callers scale it to a target SNR with the same arithmetic as :func:`generate_noise`.

    Parameters
    ----------
    files : str or sequence of str
        EEGLAB ``.set`` files (epoched or continuous), or a glob pattern. Continuous files are cut into
        non-overlapping segments of ``segment_s``. All files must share one sampling rate.
    ch_names : sequence of str
        The montage's channel order. Every name must be present in every recording.
    segment_s : float, default=2.0
        Segment length for continuous recordings.

    Notes
    -----
    Real backgrounds contain real coupling and real sources. Any statistic that compares a planted condition
    with a null should therefore use the *same* background, unplanted, as the null; see
    :mod:`source_localization.validation.regime`.
    """

    def __init__(self, files, ch_names, segment_s: float = 2.0):
        import glob as _glob
        import mne
        if isinstance(files, str):
            paths = sorted(_glob.glob(files)) or [files]
        else:
            paths = [str(f) for f in files]
        if not paths:
            raise ValueError("RecordedBackground: no files")
        self.ch_names = list(ch_names)
        self.files = paths
        self.sfreq = None
        self._data = []
        for path in paths:
            try:
                ep = mne.read_epochs_eeglab(path, verbose='error')
                X = ep.get_data(picks=self._picks(ep.ch_names, path))
                fs = float(ep.info['sfreq'])
            except (ValueError, TypeError, KeyError):
                raw = mne.io.read_raw_eeglab(path, preload=True, verbose='error')
                fs = float(raw.info['sfreq'])
                Y = raw.get_data(picks=self._picks(raw.ch_names, path))
                n = int(round(segment_s * fs))
                k = Y.shape[1] // n
                if k == 0:
                    raise ValueError(f"{path}: shorter than one {segment_s} s segment")
                X = Y[:, :k * n].reshape(Y.shape[0], k, n).transpose(1, 0, 2)
            if self.sfreq is None:
                self.sfreq = fs
            elif abs(fs - self.sfreq) > 1e-6:
                raise ValueError(f"{path}: sampling rate {fs} Hz differs from the first file's {self.sfreq} Hz")
            X = X - X.mean(axis=1, keepdims=True)              # average reference
            self._data.append(X)

    def _picks(self, available, path):
        missing = [c for c in self.ch_names if c not in available]
        if missing:
            raise ValueError(f"{path}: channels missing from the recording: {missing}")
        return [available.index(c) for c in self.ch_names]

    @property
    def n_segments(self) -> int:
        return int(sum(len(x) for x in self._data))

    def draw(self, n_times: int, rng, sfreq: Optional[float] = None) -> np.ndarray:
        """
        One background window, shape (n_channels, n_times), average-referenced, unit mean power.

        ``rng`` is a ``numpy.random.Generator`` or ``RandomState``. ``sfreq`` is the caller's sampling rate
        (default: the recordings'); a window of the same duration is taken at the recordings' rate and resampled.
        """
        native = n_times if sfreq is None or abs(sfreq - self.sfreq) < 1e-9 else \
            int(round(n_times * self.sfreq / float(sfreq)))
        rec = self._data[_randint(rng, len(self._data))]
        seg = rec[_randint(rng, len(rec))]
        if seg.shape[1] < native:
            raise ValueError(f"background segments have {seg.shape[1]} samples; the simulation needs {native}")
        start = _randint(rng, seg.shape[1] - native + 1)
        x = seg[:, start:start + native].copy()
        if native != n_times:
            from scipy.signal import resample
            x = resample(x, n_times, axis=1)
        p = np.mean(x ** 2)
        if p <= 0:
            raise ValueError("background window has zero power")
        return x / np.sqrt(p)


def _randint(rng, n: int) -> int:
    """An integer in [0, n) from a ``numpy.random.Generator`` or a legacy ``RandomState``."""
    return int(rng.integers(n)) if hasattr(rng, 'integers') else int(rng.randint(n))
