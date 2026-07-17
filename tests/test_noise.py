"""Tests for the validation noise generator (see validation/noise.py).

The custom inverse regularizes with a scaled identity, i.e. it assumes white
sensor noise (C = I). Validation that only injects i.i.d. noise satisfies that
assumption exactly and so reports best-case numbers. These tests pin the
generator that violates the assumption on purpose: they check that requested
noise structure is actually present (spatial correlation matching the kernel,
1/f slope matching the exponent) and — critically — that the 'white' path is
still numerically identical to the pre-existing behaviour, so historical
validation results remain comparable.
"""
import numpy as np
import pytest

from source_localization.validation.noise import (
    NOISE_TYPES,
    generate_noise,
    spatial_covariance,
)

N_CHANNELS = 32
N_TIMES = 4000


@pytest.fixture
def electrode_positions():
    """32 electrodes on a ~6.4mm shell, roughly the mouse array's scale."""
    pos = np.random.RandomState(1).randn(N_CHANNELS, 3)
    return pos / np.linalg.norm(pos, axis=1, keepdims=True) * 6.4


@pytest.mark.parametrize("noise_type", NOISE_TYPES)
def test_noise_is_unit_variance(noise_type, electrode_positions):
    """Callers scale to a target SNR, so every type must arrive unit-variance."""
    noise = generate_noise(
        N_CHANNELS, N_TIMES, np.random.RandomState(0), noise_type=noise_type,
        electrode_positions_mm=electrode_positions
    )
    assert noise.shape == (N_CHANNELS, N_TIMES)
    np.testing.assert_allclose(noise.std(), 1.0, rtol=1e-9)


def test_generation_is_reproducible(electrode_positions):
    """Same seed, same noise — validation runs must be repeatable."""
    kwargs = dict(noise_type="colored", electrode_positions_mm=electrode_positions)
    a = generate_noise(N_CHANNELS, N_TIMES, np.random.RandomState(5), **kwargs)
    b = generate_noise(N_CHANNELS, N_TIMES, np.random.RandomState(5), **kwargs)
    np.testing.assert_array_equal(a, b)


def test_white_noise_has_no_spatial_correlation():
    """The C = I baseline: off-diagonal channel correlation must vanish."""
    corr = np.mean([
        np.corrcoef(generate_noise(N_CHANNELS, N_TIMES,
                                   np.random.RandomState(r), noise_type="white"))
        for r in range(40)
    ], axis=0)
    off_diag = ~np.eye(N_CHANNELS, dtype=bool)
    assert np.abs(corr[off_diag]).max() < 0.02


def test_spatial_noise_matches_requested_covariance(electrode_positions):
    """Empirical channel correlation must reproduce the exponential kernel."""
    corr = np.mean([
        np.corrcoef(generate_noise(
            N_CHANNELS, N_TIMES, np.random.RandomState(r), noise_type="spatial",
            electrode_positions_mm=electrode_positions, spatial_scale_mm=3.0
        ))
        for r in range(40)
    ], axis=0)
    target = spatial_covariance(electrode_positions, spatial_scale_mm=3.0)
    off_diag = ~np.eye(N_CHANNELS, dtype=bool)
    assert np.abs(corr - target)[off_diag].max() < 0.05


def test_spatial_covariance_decays_with_distance():
    """Nearby electrodes must be more correlated than distant ones."""
    pos = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [10.0, 0.0, 0.0]])
    cov = spatial_covariance(pos, spatial_scale_mm=1.0)
    assert cov[0, 1] > cov[0, 2]
    np.testing.assert_allclose(np.diag(cov), 1.0, atol=1e-5)
    # Must stay factorizable — the generator Cholesky-decomposes it.
    np.linalg.cholesky(cov)


def test_spatial_covariance_scale_controls_correlation_length():
    """A longer correlation length means more broadly shared noise."""
    pos = np.array([[0.0, 0.0, 0.0], [3.0, 0.0, 0.0]])
    tight = spatial_covariance(pos, spatial_scale_mm=1.0)
    broad = spatial_covariance(pos, spatial_scale_mm=10.0)
    assert broad[0, 1] > tight[0, 1]


@pytest.mark.parametrize("exponent", [0.0, 1.0, 2.0])
def test_temporal_noise_recovers_requested_slope(exponent):
    """A requested 1/f^a spectrum must show slope -a in the log-log PSD."""
    noise = generate_noise(
        N_CHANNELS, N_TIMES, np.random.RandomState(7), noise_type="temporal",
        temporal_exponent=exponent
    )
    psd = (np.abs(np.fft.rfft(noise, axis=1)) ** 2).mean(axis=0)
    freqs = np.fft.rfftfreq(N_TIMES, d=1.0)
    band = (freqs > 0.002) & (freqs < 0.4)
    slope = np.polyfit(np.log(freqs[band]), np.log(psd[band]), 1)[0]
    assert slope == pytest.approx(-exponent, abs=0.1)


def test_colored_noise_has_both_structures(electrode_positions):
    """'colored' is the realistic worst case: correlated AND 1/f at once."""
    noise = generate_noise(
        N_CHANNELS, N_TIMES, np.random.RandomState(3), noise_type="colored",
        electrode_positions_mm=electrode_positions, spatial_scale_mm=3.0,
        temporal_exponent=1.0
    )
    off_diag = ~np.eye(N_CHANNELS, dtype=bool)
    assert np.abs(np.corrcoef(noise)[off_diag]).mean() > 0.1

    psd = (np.abs(np.fft.rfft(noise, axis=1)) ** 2).mean(axis=0)
    freqs = np.fft.rfftfreq(N_TIMES, d=1.0)
    band = (freqs > 0.002) & (freqs < 0.4)
    slope = np.polyfit(np.log(freqs[band]), np.log(psd[band]), 1)[0]
    assert slope == pytest.approx(-1.0, abs=0.15)


def test_white_path_matches_legacy_scaling():
    """Unit-variance normalization must cancel out of the SNR scaling.

    Guards historical comparability: the pre-existing code scaled raw randn by
    sqrt(signal / (snr * mean(noise**2))). Because that scale is inversely
    proportional to the noise norm, normalizing first must leave the summed
    signal identical, or every previously-reported white-noise number shifts.
    """
    eeg_clean = np.random.RandomState(99).randn(N_CHANNELS, 500) * 1e-6
    signal_power = np.mean(eeg_clean ** 2)
    snr_linear = 10 ** (10 / 10)

    def scaled(noise):
        scale = np.sqrt(signal_power / (snr_linear * np.mean(noise ** 2)))
        return eeg_clean + scale * noise

    for seed in (0, 42, 7):
        legacy = scaled(np.random.RandomState(seed).randn(N_CHANNELS, 500))
        current = scaled(generate_noise(N_CHANNELS, 500,
                                        np.random.RandomState(seed),
                                        noise_type="white"))
        np.testing.assert_allclose(legacy, current, rtol=1e-12, atol=0)


def test_unknown_noise_type_raises():
    with pytest.raises(ValueError, match="Unknown noise_type"):
        generate_noise(N_CHANNELS, 100, np.random.RandomState(0),
                       noise_type="pink")


def test_spatial_without_positions_raises():
    """Silently falling back to white would fake a passing colored-noise run."""
    with pytest.raises(ValueError, match="requires electrode_positions_mm"):
        generate_noise(N_CHANNELS, 100, np.random.RandomState(0),
                       noise_type="spatial")


def test_position_channel_mismatch_raises(electrode_positions):
    with pytest.raises(ValueError, match="channels"):
        generate_noise(8, 100, np.random.RandomState(0), noise_type="spatial",
                       electrode_positions_mm=electrode_positions)


def test_bad_spatial_scale_raises(electrode_positions):
    with pytest.raises(ValueError, match="spatial_scale_mm"):
        spatial_covariance(electrode_positions, spatial_scale_mm=0.0)
