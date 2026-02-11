#!/usr/bin/env python3
"""Test how regularization affects inverse solution."""

import numpy as np
import pickle
from pathlib import Path

# Load forward model
results_dir = Path("/home/metalexy/sandbox/AlexProjects/mouse-eeg-source-localization/source_localization/validation/results/original/V01_sphere_vol_dspm")
fwd_path = results_dir / "data" / "step4_forward.pkl"

with open(fwd_path, 'rb') as f:
    fwd = pickle.load(f)

G = fwd['sol']['data']
n_channels, n_dipoles = G.shape
n_sources = n_dipoles // 3

coords_path = results_dir / "data" / "step3_source_coords_mm.npy"
coords = np.load(coords_path)

brain_radius = 6.4
source_distances = np.linalg.norm(coords, axis=1)
source_depths = brain_radius - source_distances

# Test source
shallow_idx = np.where((source_depths >= 0) & (source_depths < 1))[0][0]
print(f"Testing with source {shallow_idx}, depth={source_depths[shallow_idx]:.2f}mm")
print(f"Position: {coords[shallow_idx]}")

# Simulate dipole
dipole_moment = np.zeros(n_dipoles)
dipole_moment[shallow_idx * 3] = 50e-9
sensor_data = (G @ dipole_moment).reshape(-1, 1)

print(f"\nSensor data range: {sensor_data.min():.6e} to {sensor_data.max():.6e}")

# Test different regularization levels
print("\n=== Effect of regularization on MNE localization ===\n")

for snr in [1, 2, 3, 5, 10, 100, 1000]:
    lambda2 = 1.0 / snr ** 2

    GGT = G @ G.T
    GGT_reg = GGT + lambda2 * np.trace(GGT) / n_channels * np.eye(n_channels)
    GGT_inv = np.linalg.inv(GGT_reg)
    W = G.T @ GGT_inv

    mne_source = W @ sensor_data
    mne_source_power = mne_source.reshape(n_sources, 3, -1)[:, 0, 0]
    mne_peak = np.argmax(np.abs(mne_source_power))
    mne_error = np.linalg.norm(coords[mne_peak] - coords[shallow_idx])

    # sLORETA
    resolution_diag = np.sum(W * G.T, axis=1)
    sloreta_source = mne_source / np.sqrt(np.maximum(resolution_diag[:, np.newaxis], 1e-20))
    sloreta_source_power = sloreta_source.reshape(n_sources, 3, -1)[:, 0, 0]
    sloreta_peak = np.argmax(np.abs(sloreta_source_power))
    sloreta_error = np.linalg.norm(coords[sloreta_peak] - coords[shallow_idx])

    # dSPM
    noise_norm = np.sqrt(np.sum(W ** 2, axis=1))
    dspm_source = mne_source / (noise_norm[:, np.newaxis] + 1e-20)
    dspm_source_power = dspm_source.reshape(n_sources, 3, -1)[:, 0, 0]
    dspm_peak = np.argmax(np.abs(dspm_source_power))
    dspm_error = np.linalg.norm(coords[dspm_peak] - coords[shallow_idx])

    print(f"SNR={snr:4d} (λ²={lambda2:.6f}): MNE={mne_error:.2f}mm (src {mne_peak}), dSPM={dspm_error:.2f}mm, sLORETA={sloreta_error:.2f}mm")

# Check the condition number of G @ G.T
print("\n=== Leadfield matrix properties ===")
GGT = G @ G.T
eigenvalues = np.linalg.eigvalsh(GGT)
print(f"G @ G.T eigenvalues: min={eigenvalues.min():.6e}, max={eigenvalues.max():.6e}")
print(f"Condition number: {eigenvalues.max() / eigenvalues.min():.2e}")
print(f"Trace: {np.trace(GGT):.6e}")
print(f"Regularization term (λ² * trace/n): {(1/9) * np.trace(GGT) / n_channels:.6e}")

# Check SVD of G
u, s, vh = np.linalg.svd(G, full_matrices=False)
print(f"\nLeadfield SVD singular values:")
print(f"  Max: {s.max():.6e}")
print(f"  Min: {s.min():.6e}")
print(f"  Ratio: {s.max()/s.min():.2e}")
print(f"  Top 5: {s[:5]}")
print(f"  Bottom 5: {s[-5:]}")
