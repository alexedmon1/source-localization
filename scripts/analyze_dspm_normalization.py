#!/usr/bin/env python3
"""Analyze dSPM vs sLORETA normalization factors."""

import numpy as np
import pickle
from pathlib import Path

# Load a forward model
results_dir = Path("/home/metalexy/sandbox/AlexProjects/mouse-eeg-source-localization/source_localization/validation/results/original/V01_sphere_vol_dspm")
fwd_path = results_dir / "data" / "step4_forward.pkl"

print("Loading forward model...")
with open(fwd_path, 'rb') as f:
    fwd = pickle.load(f)

G = fwd['sol']['data']  # (n_channels, n_dipoles)
n_channels, n_dipoles = G.shape
print(f"Leadfield shape: {G.shape}")

# Compute MNE inverse operator
snr = 3.0
lambda2 = 1.0 / snr ** 2

GGT = G @ G.T
GGT_reg = GGT + lambda2 * np.trace(GGT) / n_channels * np.eye(n_channels)
GGT_inv = np.linalg.inv(GGT_reg)
W = G.T @ GGT_inv  # Shape: (n_dipoles, n_channels)

print(f"Inverse operator W shape: {W.shape}")

# dSPM normalization: noise gain (corrected)
noise_norm = np.sqrt(np.sum(W ** 2, axis=1))

# sLORETA normalization: resolution diagonal
resolution_diag = np.sum(W * G.T, axis=1)

# OLD dSPM (incorrect): resolution diagonal
old_dspm_norm = resolution_diag  # This was the bug

print("\n=== Normalization Statistics ===")
print(f"\n1. dSPM noise normalization sqrt(diag(W @ W.T)):")
print(f"   Min: {noise_norm.min():.6f}")
print(f"   Max: {noise_norm.max():.6f}")
print(f"   Mean: {noise_norm.mean():.6f}")
print(f"   Std: {noise_norm.std():.6f}")
print(f"   Ratio max/min: {noise_norm.max() / (noise_norm.min() + 1e-10):.1f}x")

print(f"\n2. sLORETA resolution normalization diag(W @ G):")
print(f"   Min: {resolution_diag.min():.8f}")
print(f"   Max: {resolution_diag.max():.8f}")
print(f"   Mean: {resolution_diag.mean():.8f}")
print(f"   Std: {resolution_diag.std():.8f}")
print(f"   Ratio max/min: {resolution_diag.max() / (resolution_diag.min() + 1e-10):.1f}x")

# Check for near-zero values
near_zero_resolution = np.sum(resolution_diag < 1e-6)
near_zero_noise = np.sum(noise_norm < 1e-6)
print(f"\n   Near-zero resolution values (<1e-6): {near_zero_resolution}")
print(f"   Near-zero noise values (<1e-6): {near_zero_noise}")

# Leadfield analysis
print("\n=== Leadfield Analysis ===")
leadfield_norms = np.linalg.norm(G, axis=0)  # Norm per dipole
print(f"Leadfield norm per dipole:")
print(f"   Min: {leadfield_norms.min():.6e}")
print(f"   Max: {leadfield_norms.max():.6e}")
print(f"   Ratio: {leadfield_norms.max() / (leadfield_norms.min() + 1e-20):.1f}x")

# Group by source (3 orientations per source)
n_sources = n_dipoles // 3
source_leadfield_power = np.array([
    np.linalg.norm(G[:, i*3:(i+1)*3]) for i in range(n_sources)
])
print(f"\nLeadfield power per source (combining 3 orientations):")
print(f"   Min: {source_leadfield_power.min():.6e}")
print(f"   Max: {source_leadfield_power.max():.6e}")
print(f"   Ratio: {source_leadfield_power.max() / (source_leadfield_power.min() + 1e-20):.1f}x")

# Load source coordinates to correlate with depth
coords_path = results_dir / "data" / "step3_source_coords_mm.npy"
coords = np.load(coords_path)
print(f"\nSource coordinates shape: {coords.shape}")

# Estimate depth from surface (approximate as distance from center at radius 6.4mm)
brain_radius = 6.4
source_distances = np.linalg.norm(coords, axis=1)
source_depths = brain_radius - source_distances
print(f"Source depth range: {source_depths.min():.2f} to {source_depths.max():.2f} mm")

# Correlate leadfield power with depth
from scipy.stats import spearmanr
corr, p = spearmanr(source_depths, source_leadfield_power)
print(f"\nCorrelation between depth and leadfield power: r={corr:.3f}, p={p:.3e}")

# Group noise normalization by source
source_noise_norm = np.array([
    np.mean(noise_norm[i*3:(i+1)*3]) for i in range(n_sources)
])
corr_noise, p_noise = spearmanr(source_depths, source_noise_norm)
print(f"Correlation between depth and dSPM noise norm: r={corr_noise:.3f}, p={p_noise:.3e}")

source_res_norm = np.array([
    np.mean(resolution_diag[i*3:(i+1)*3]) for i in range(n_sources)
])
corr_res, p_res = spearmanr(source_depths, source_res_norm)
print(f"Correlation between depth and sLORETA resolution norm: r={corr_res:.3f}, p={p_res:.3e}")

# Print by depth bin
print("\n=== Normalization by Depth Bin ===")
depth_bins = [(0, 1), (1, 2), (2, 3), (3, 4), (4, 7)]
for dmin, dmax in depth_bins:
    mask = (source_depths >= dmin) & (source_depths < dmax)
    if np.sum(mask) > 0:
        print(f"\nDepth {dmin}-{dmax}mm ({np.sum(mask)} sources):")
        print(f"   dSPM noise norm: {source_noise_norm[mask].mean():.6f} ± {source_noise_norm[mask].std():.6f}")
        print(f"   sLORETA res norm: {source_res_norm[mask].mean():.8f} ± {source_res_norm[mask].std():.8f}")
        print(f"   Leadfield power: {source_leadfield_power[mask].mean():.6e} ± {source_leadfield_power[mask].std():.6e}")
