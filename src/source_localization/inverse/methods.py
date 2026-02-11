#!/usr/bin/env python3
"""
Inverse Methods Module

**Created:** 2025-11-20
**Last Updated:** 2025-12-05

Implements inverse solution methods for EEG source localization:
- MNE (Minimum Norm Estimate)
- dSPM (dynamic Statistical Parametric Mapping)
- sLORETA (standardized Low Resolution Electromagnetic Tomography)

Based on production script: ../scripts/017_apply_source_localization.py
"""

import numpy as np
import mne


def apply_inverse_MNE(fwd, info, epochs=None, evoked=None, snr=3.0, lambda2=None, depth=0.8):
    """
    Apply MNE (Minimum Norm Estimate) inverse method with depth weighting.

    MNE finds the source distribution with minimum overall power that explains
    the measured data. Depth weighting compensates for the bias toward
    superficial sources.

    Parameters
    ----------
    fwd : mne.Forward
        Forward solution
    info : mne.Info
        Measurement info (for channel matching)
    epochs : mne.Epochs, optional
        Epoched data (will be averaged)
    evoked : mne.Evoked, optional
        Already-averaged evoked data
    snr : float
        Signal-to-noise ratio (default: 3.0)
    lambda2 : float, optional
        Regularization parameter (default: 1/snr^2)
    depth : float
        Depth weighting exponent (default: 0.8). Higher values give more
        compensation for deep sources. Set to 0 for no depth weighting.

    Returns
    -------
    source_activity : ndarray
        Source activity (n_dipoles, n_times)
    source_signed : ndarray
        Signed source activity from dominant orientation (n_sources, n_times)
        Uses x-orientation to preserve 1/f spectral characteristics
    inverse_operator : ndarray
        Inverse operator matrix W (n_dipoles, n_channels)
    """
    if lambda2 is None:
        lambda2 = 1.0 / snr ** 2

    # Get EEG data
    if evoked is not None:
        eeg_data = evoked.data
    elif epochs is not None:
        eeg_data = epochs.get_data().mean(axis=0)  # Average epochs
    else:
        raise ValueError("Must provide either epochs or evoked data")

    # Get leadfield
    G = fwd['sol']['data']  # Shape: (n_channels, n_dipoles)
    n_channels, n_dipoles = G.shape

    print(f"  Computing MNE inverse operator...")
    print(f"    SNR: {snr}")
    print(f"    Lambda2 (regularization): {lambda2:.6f}")
    print(f"    Depth weighting: {depth}")
    print(f"    Leadfield: {G.shape}")

    # Compute depth weights from leadfield column norms
    # Deeper sources have smaller leadfield norms, so we weight them up
    column_norms = np.linalg.norm(G, axis=0)  # Shape: (n_dipoles,)

    # Depth weighting: w_i = ||g_i||^(-depth/2)
    # The /2 is because we apply weights to both sides of the source covariance
    # This compensates for the 1/r^2 decay of the leadfield
    if depth > 0:
        # Avoid division by zero for very small norms
        depth_weights = np.power(column_norms + 1e-10, -depth / 2)
        # Normalize to preserve overall scale
        depth_weights /= np.mean(depth_weights)
        # Create weighted leadfield: G_weighted = G @ diag(depth_weights)
        G_weighted = G * depth_weights[np.newaxis, :]
    else:
        G_weighted = G
        depth_weights = np.ones(n_dipoles)

    # Sensor-space formulation with depth weighting:
    # This is equivalent to: W = R @ G.T @ (G @ R @ G.T + λI)^-1
    # where R = diag(depth_weights^2) is the source covariance prior
    GGT = G_weighted @ G_weighted.T
    GGT_reg = GGT + lambda2 * np.trace(GGT) / n_channels * np.eye(n_channels)
    GGT_inv = np.linalg.inv(GGT_reg)
    W = G_weighted.T @ GGT_inv  # Shape: (n_dipoles, n_channels)

    # Apply depth weights to transform back to source space
    W = W * depth_weights[:, np.newaxis]

    # Apply inverse to data
    source_activity = W @ eeg_data  # Shape: (n_dipoles, n_times)

    # No additional normalization for MNE (depth weighting handles bias)
    source_activity_norm = source_activity

    # Use signed values from dominant (first) orientation to preserve 1/f spectrum
    # Note: np.linalg.norm() destroys phase information and flattens spectral slope
    n_sources = n_dipoles // 3
    source_activity_reshaped = source_activity_norm.reshape(n_sources, 3, -1)
    source_power = source_activity_reshaped[:, 0, :]  # Use x-orientation (signed)

    print(f"    ✓ MNE complete: {source_power.shape}")

    return source_activity_norm, source_power, W


def apply_inverse_dSPM(fwd, info, epochs=None, evoked=None, snr=3.0, lambda2=None):
    """
    Apply dSPM (dynamic Statistical Parametric Mapping) inverse method.

    dSPM normalizes MNE by the noise sensitivity at each source, converting
    estimates into statistical maps (SNR units).

    Parameters
    ----------
    fwd : mne.Forward
        Forward solution
    info : mne.Info
        Measurement info (for channel matching)
    epochs : mne.Epochs, optional
        Epoched data (will be averaged)
    evoked : mne.Evoked, optional
        Already-averaged evoked data
    snr : float
        Signal-to-noise ratio (default: 3.0)
    lambda2 : float, optional
        Regularization parameter (default: 1/snr^2)

    Returns
    -------
    source_activity : ndarray
        Normalized source activity (n_dipoles, n_times)
    source_signed : ndarray
        Signed source activity from dominant orientation (n_sources, n_times)
        Uses x-orientation to preserve 1/f spectral characteristics
    inverse_operator : ndarray
        Inverse operator matrix W (n_dipoles, n_channels)
    """
    if lambda2 is None:
        lambda2 = 1.0 / snr ** 2

    # Get EEG data
    if evoked is not None:
        eeg_data = evoked.data
    elif epochs is not None:
        eeg_data = epochs.get_data().mean(axis=0)  # Average epochs
    else:
        raise ValueError("Must provide either epochs or evoked data")

    # Get leadfield
    G = fwd['sol']['data']  # Shape: (n_channels, n_dipoles)
    n_channels, n_dipoles = G.shape

    print(f"  Computing dSPM inverse operator...")
    print(f"    SNR: {snr}")
    print(f"    Lambda2 (regularization): {lambda2:.6f}")
    print(f"    Leadfield: {G.shape}")

    # Sensor-space formulation: W = G^T * (G*G^T + lambda*I)^-1
    GGT = G @ G.T
    GGT_reg = GGT + lambda2 * np.trace(GGT) / n_channels * np.eye(n_channels)
    GGT_inv = np.linalg.inv(GGT_reg)
    W = G.T @ GGT_inv  # Shape: (n_dipoles, n_channels)

    # Apply inverse to data
    source_activity = W @ eeg_data  # Shape: (n_dipoles, n_times)

    # dSPM normalization: divide by noise sensitivity (Dale et al., 2000)
    # For white noise (identity covariance): noise_norm = sqrt(diag(W @ W.T))
    print(f"    Computing dSPM normalization...")
    noise_norm = np.sqrt(np.sum(W ** 2, axis=1))  # sqrt(diag(W @ W.T))
    source_activity_norm = source_activity / (noise_norm[:, np.newaxis] + 1e-10)

    # Use signed values from dominant (first) orientation to preserve 1/f spectrum
    # Note: np.linalg.norm() destroys phase information and flattens spectral slope
    n_sources = n_dipoles // 3
    source_activity_reshaped = source_activity_norm.reshape(n_sources, 3, -1)
    source_power = source_activity_reshaped[:, 0, :]  # Use x-orientation (signed)

    print(f"    ✓ dSPM complete: {source_power.shape}")

    return source_activity_norm, source_power, W


def apply_inverse_sLORETA(fwd, info, epochs=None, evoked=None, snr=3.0, lambda2=None):
    """
    Apply sLORETA (standardized Low Resolution Electromagnetic Tomography).

    sLORETA normalizes by the resolution matrix to achieve zero localization
    error for point sources. Provides better spatial accuracy than MNE/dSPM.

    Parameters
    ----------
    fwd : mne.Forward
        Forward solution
    info : mne.Info
        Measurement info (for channel matching)
    epochs : mne.Epochs, optional
        Epoched data (will be averaged)
    evoked : mne.Evoked, optional
        Already-averaged evoked data
    snr : float
        Signal-to-noise ratio (default: 3.0)
    lambda2 : float, optional
        Regularization parameter (default: 1/snr^2)

    Returns
    -------
    source_activity : ndarray
        Normalized source activity (n_dipoles, n_times)
    source_signed : ndarray
        Signed source activity from dominant orientation (n_sources, n_times)
        Uses x-orientation to preserve 1/f spectral characteristics
    inverse_operator : ndarray
        Inverse operator matrix W (n_dipoles, n_channels)
    """
    if lambda2 is None:
        lambda2 = 1.0 / snr ** 2

    # Get EEG data
    if evoked is not None:
        eeg_data = evoked.data
    elif epochs is not None:
        eeg_data = epochs.get_data().mean(axis=0)  # Average epochs
    else:
        raise ValueError("Must provide either epochs or evoked data")

    # Get leadfield
    G = fwd['sol']['data']  # Shape: (n_channels, n_dipoles)
    n_channels, n_dipoles = G.shape

    print(f"  Computing sLORETA inverse operator...")
    print(f"    SNR: {snr}")
    print(f"    Lambda2 (regularization): {lambda2:.6f}")
    print(f"    Leadfield: {G.shape}")

    # Sensor-space formulation: W = G^T * (G*G^T + lambda*I)^-1
    GGT = G @ G.T
    GGT_reg = GGT + lambda2 * np.trace(GGT) / n_channels * np.eye(n_channels)
    GGT_inv = np.linalg.inv(GGT_reg)
    W = G.T @ GGT_inv  # Shape: (n_dipoles, n_channels)

    # Apply inverse to data
    source_activity = W @ eeg_data  # Shape: (n_dipoles, n_times)

    # sLORETA normalization: divide by sqrt of resolution matrix diagonal
    # (Pascual-Marqui, 2002) - achieves zero localization error for point sources
    print(f"    Computing sLORETA normalization...")
    resolution_matrix_diag = np.sum(W * G.T, axis=1)  # diag(W @ G)
    # Add epsilon before sqrt to avoid division by very small numbers
    source_activity_norm = source_activity / np.sqrt(np.maximum(resolution_matrix_diag[:, np.newaxis], 1e-20))

    # Use signed values from dominant (first) orientation to preserve 1/f spectrum
    # Note: np.linalg.norm() destroys phase information and flattens spectral slope
    n_sources = n_dipoles // 3
    source_activity_reshaped = source_activity_norm.reshape(n_sources, 3, -1)
    source_power = source_activity_reshaped[:, 0, :]  # Use x-orientation (signed)

    print(f"    ✓ sLORETA complete: {source_power.shape}")

    return source_activity_norm, source_power, W


def create_simulated_data(info, duration=1.0):
    """
    Create simulated EEG data for testing when real data is not available.

    Parameters
    ----------
    info : mne.Info
        Measurement info (defines channels and sampling rate)
    duration : float
        Duration in seconds (default: 1.0)

    Returns
    -------
    evoked : mne.Evoked
        Simulated evoked data with random noise
    """
    n_channels = len(info['ch_names'])
    sfreq = info['sfreq']
    n_times = int(duration * sfreq)

    # Create random data (simulating background noise + weak signal)
    data = np.random.randn(n_channels, n_times) * 1e-6
    times = np.arange(n_times) / sfreq

    # Create evoked object
    evoked = mne.EvokedArray(data, info, tmin=0.0)

    print(f"  Created simulated data: {n_channels} channels × {n_times} timepoints")

    return evoked
