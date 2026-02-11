#!/usr/bin/env python3
"""Test inverse method localization with a single dipole."""

import numpy as np
import pickle
from pathlib import Path

# Load forward model
results_dir = Path("/home/metalexy/sandbox/AlexProjects/mouse-eeg-source-localization/source_localization/validation/results/original/V01_sphere_vol_dspm")
fwd_path = results_dir / "data" / "step4_forward.pkl"

print("Loading forward model...")
with open(fwd_path, 'rb') as f:
    fwd = pickle.load(f)

G = fwd['sol']['data']  # (n_channels, n_dipoles)
n_channels, n_dipoles = G.shape
n_sources = n_dipoles // 3

# Load source coordinates
coords_path = results_dir / "data" / "step3_source_coords_mm.npy"
coords = np.load(coords_path)

# Compute inverse operator
snr = 3.0
lambda2 = 1.0 / snr ** 2

GGT = G @ G.T
GGT_reg = GGT + lambda2 * np.trace(GGT) / n_channels * np.eye(n_channels)
GGT_inv = np.linalg.inv(GGT_reg)
W = G.T @ GGT_inv

# Compute normalizations
noise_norm = np.sqrt(np.sum(W ** 2, axis=1))  # dSPM
resolution_diag = np.sum(W * G.T, axis=1)  # sLORETA

# Test with a superficial source (index 0 is near surface)
# Find a source at different depths
brain_radius = 6.4
source_distances = np.linalg.norm(coords, axis=1)
source_depths = brain_radius - source_distances

# Pick sources at different depths
test_sources = {
    'shallow (0-1mm)': np.where((source_depths >= 0) & (source_depths < 1))[0][0],
    'medium (2-3mm)': np.where((source_depths >= 2) & (source_depths < 3))[0][0],
    'deep (4+mm)': np.where((source_depths >= 4))[0][0] if np.any(source_depths >= 4) else None,
}

print("\n=== Testing inverse methods on simulated dipoles ===\n")

for depth_label, src_idx in test_sources.items():
    if src_idx is None:
        continue

    print(f"\n--- {depth_label}: source {src_idx}, depth={source_depths[src_idx]:.2f}mm ---")
    true_pos = coords[src_idx]
    print(f"True position: {true_pos}")

    # Simulate dipole in x-direction
    dipole_moment = np.zeros(n_dipoles)
    dipole_moment[src_idx * 3] = 50e-9  # 50 nAm

    # Generate sensor data (no noise for clarity)
    sensor_data = G @ dipole_moment
    sensor_data = sensor_data.reshape(-1, 1)  # (n_channels, 1)

    print(f"Sensor data range: {sensor_data.min():.6e} to {sensor_data.max():.6e}")

    # MNE
    mne_source = W @ sensor_data
    mne_source_power = mne_source.reshape(n_sources, 3, -1)[:, 0, 0]  # x-component
    mne_peak = np.argmax(np.abs(mne_source_power))
    mne_error = np.linalg.norm(coords[mne_peak] - true_pos)
    print(f"\nMNE:")
    print(f"  Source range: {mne_source.min():.6e} to {mne_source.max():.6e}")
    print(f"  Peak source: {mne_peak}, error: {mne_error:.2f}mm")
    print(f"  True source value: {mne_source_power[src_idx]:.6e}")
    print(f"  Peak source value: {mne_source_power[mne_peak]:.6e}")

    # dSPM
    dspm_source = mne_source / (noise_norm[:, np.newaxis] + 1e-20)
    dspm_source_power = dspm_source.reshape(n_sources, 3, -1)[:, 0, 0]
    dspm_peak = np.argmax(np.abs(dspm_source_power))
    dspm_error = np.linalg.norm(coords[dspm_peak] - true_pos)
    print(f"\ndSPM:")
    print(f"  Source range: {dspm_source.min():.6e} to {dspm_source.max():.6e}")
    print(f"  Peak source: {dspm_peak}, error: {dspm_error:.2f}mm")
    print(f"  True source value: {dspm_source_power[src_idx]:.6e}")
    print(f"  Peak source value: {dspm_source_power[dspm_peak]:.6e}")
    print(f"  True source norm factor: {noise_norm[src_idx*3]:.6e}")
    print(f"  Peak source norm factor: {noise_norm[dspm_peak*3]:.6e}")

    # sLORETA
    sloreta_source = mne_source / np.sqrt(np.maximum(resolution_diag[:, np.newaxis], 1e-20))
    sloreta_source_power = sloreta_source.reshape(n_sources, 3, -1)[:, 0, 0]
    sloreta_peak = np.argmax(np.abs(sloreta_source_power))
    sloreta_error = np.linalg.norm(coords[sloreta_peak] - true_pos)
    print(f"\nsLORETA:")
    print(f"  Source range: {sloreta_source.min():.6e} to {sloreta_source.max():.6e}")
    print(f"  Peak source: {sloreta_peak}, error: {sloreta_error:.2f}mm")
    print(f"  True source value: {sloreta_source_power[src_idx]:.6e}")
    print(f"  Peak source value: {sloreta_source_power[sloreta_peak]:.6e}")
    print(f"  True source norm factor: {np.sqrt(resolution_diag[src_idx*3]):.6e}")
    print(f"  Peak source norm factor: {np.sqrt(resolution_diag[sloreta_peak*3]):.6e}")

# Check which sources have the highest dSPM normalization
print("\n\n=== Sources with highest dSPM noise norm (lowest sensitivity) ===")
top_norm_sources = np.argsort(noise_norm[::3])[-10:]
for src_idx in top_norm_sources[::-1]:
    print(f"Source {src_idx}: depth={source_depths[src_idx]:.2f}mm, norm={noise_norm[src_idx*3]:.6e}, pos={coords[src_idx]}")

print("\n=== Sources with lowest dSPM noise norm (highest sensitivity) ===")
bottom_norm_sources = np.argsort(noise_norm[::3])[:10]
for src_idx in bottom_norm_sources:
    print(f"Source {src_idx}: depth={source_depths[src_idx]:.2f}mm, norm={noise_norm[src_idx*3]:.6e}, pos={coords[src_idx]}")
