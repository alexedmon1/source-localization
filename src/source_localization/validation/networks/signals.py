"""Planted signals for network validation: band-limited carriers, coupling kinds, SNR scaling.

Ported unchanged from the probability-atlas method workspace (``pa/networks.py``, ``pa/directed.py``, 2026-10), so
a run here reproduces that workspace's Phase 9 and Phase 10 simulations at the same seeds.

Kinds (undirected, Phase 9): ``lagged`` (B follows A by a whole-sample lag), ``zero`` (B = A), ``envelope`` (shared
slow envelope, independent carriers), ``single`` (A alone), ``uncoupled`` (independent A and B). Directed
validation uses :func:`delayed_pair` (a fractional lag by an exact frequency-domain shift).
"""
from __future__ import annotations

import numpy as np
from scipy.signal import butter, sosfiltfilt

__all__ = ["KINDS", "avg_ref", "band_carrier", "plant_timecourses", "delayed_pair", "band_power", "scale_to_snr"]

KINDS = ("lagged", "zero", "envelope", "single", "uncoupled")


def avg_ref(M: np.ndarray, axis: int = 0) -> np.ndarray:
    """Average reference along the channel axis."""
    return M - M.mean(axis=axis, keepdims=True)


def band_carrier(rng: np.random.Generator, n_epochs: int, n_samples: int, sfreq: float, band, pad: int = 0):
    """(n_epochs, n_samples + pad) Gaussian noise band-passed to ``band`` (4th-order Butterworth, zero phase),
    unit variance per epoch."""
    sos = butter(4, band, btype="bandpass", fs=sfreq, output="sos")
    edge = int(sfreq)                                     # discard filter edges
    x = sosfiltfilt(sos, rng.standard_normal((n_epochs, n_samples + pad + 2 * edge)), axis=-1)[:, edge:-edge]
    return x / x.std(axis=-1, keepdims=True)


def _envelope(rng, n_epochs, n_samples, sfreq):
    sos = butter(2, 1.0, btype="lowpass", fs=sfreq, output="sos")
    edge = int(sfreq)
    z = sosfiltfilt(sos, rng.standard_normal((n_epochs, n_samples + 2 * edge)), axis=-1)[:, edge:-edge]
    z = z / z.std(axis=-1, keepdims=True)
    return np.exp(z)


def plant_timecourses(kind: str, rng: np.random.Generator, n_epochs: int, n_samples: int, sfreq: float, band,
                      lag_ms: float | None = None) -> np.ndarray:
    """(n_sources, n_epochs, n_samples) source time courses for ``kind`` (see :data:`KINDS`)."""
    if kind == "lagged":
        lag = int(round(lag_ms * sfreq / 1000.0))
        x = band_carrier(rng, n_epochs, n_samples, sfreq, band, pad=lag)
        return np.stack([x[:, lag:lag + n_samples], x[:, :n_samples]])          # B(t) = A(t - lag)
    if kind == "zero":
        a = band_carrier(rng, n_epochs, n_samples, sfreq, band)
        return np.stack([a, a.copy()])
    if kind == "envelope":
        e = _envelope(rng, n_epochs, n_samples, sfreq)
        a = band_carrier(rng, n_epochs, n_samples, sfreq, band)
        b = band_carrier(rng, n_epochs, n_samples, sfreq, band)
        return np.stack([e * a, e * b])
    if kind == "single":
        return band_carrier(rng, n_epochs, n_samples, sfreq, band)[None]
    if kind == "uncoupled":
        return np.stack([band_carrier(rng, n_epochs, n_samples, sfreq, band),
                         band_carrier(rng, n_epochs, n_samples, sfreq, band)])
    raise ValueError(f"kind must be one of {KINDS}")


def delayed_pair(rng: np.random.Generator, n_epochs: int, n_samples: int, sfreq: float, band, lag_ms: float):
    """(2, n_epochs, n_samples): a band-limited driver A and B(t) = A(t - lag), for a fractional lag."""
    tau = lag_ms / 1000.0
    pad = int(np.ceil(tau * sfreq)) + 16
    x = band_carrier(rng, n_epochs, n_samples + 2 * pad, sfreq, band)
    f = np.fft.rfftfreq(x.shape[-1], 1.0 / sfreq)
    y = np.fft.irfft(np.fft.rfft(x, axis=-1) * np.exp(-2j * np.pi * f * tau), n=x.shape[-1], axis=-1)
    return np.stack([x[:, pad:pad + n_samples], y[:, pad:pad + n_samples]])


def band_power(X: np.ndarray, sfreq: float, band) -> np.ndarray:
    """Per-channel band power of (n_epochs, n_channels, n_samples) data: variance after the band-pass."""
    sos = butter(4, band, btype="bandpass", fs=sfreq, output="sos")
    return sosfiltfilt(sos, X, axis=-1).var(axis=-1).mean(axis=0)


def scale_to_snr(g: np.ndarray, tc: np.ndarray, background_bp: np.ndarray, snr_db: float, sfreq: float,
                 band) -> float:
    """Factor k so that the source k * g * tc has band power 10^(snr/10) x the background's at its most
    sensitive electrode (average reference). ``g`` (n_channels,), ``tc`` (n_epochs, n_samples)."""
    ga = avg_ref(g)
    e = int(np.argmax(np.abs(ga)))
    sos = butter(4, band, btype="bandpass", fs=sfreq, output="sos")
    p = float(sosfiltfilt(sos, tc, axis=-1).var(axis=-1).mean()) * ga[e] ** 2
    return float(np.sqrt(10 ** (snr_db / 10.0) * background_bp[e] / p))
