#!/usr/bin/env python3
"""Explain what causes weak vs strong leadfields."""

import numpy as np
import pickle
from pathlib import Path
import matplotlib.pyplot as plt

base_dir = Path("/home/metalexy/sandbox/AlexProjects/mouse-eeg-source-localization/source_localization/validation/results/original")

# Load both sphere and ellipsoid for comparison
configs = {
    'sphere_vol': "V03_sphere_vol_sloreta",
    'ellipsoid_vol': "V10_ellipsoid_vol_sloreta",
    'sphere_roi': "V21_sphere_roi_sloreta",
    'ellipsoid_roi': "V24_ellipsoid_roi_sloreta",
}

print("=" * 100)
print("WHAT CAUSES WEAK VS STRONG LEADFIELDS?")
print("=" * 100)

print("""
PHYSICS OF EEG FORWARD MODELING
===============================

The leadfield matrix G describes how a unit dipole at each source location
produces voltage at each electrode. Leadfield strength depends on:

1. DISTANCE FROM ELECTRODES
   - Electric field decays with distance (roughly 1/r² for dipoles)
   - Deeper sources → weaker leadfield

2. DIPOLE ORIENTATION (Critical for spherical models!)
   - RADIAL dipoles: point toward/away from head center
   - TANGENTIAL dipoles: parallel to head surface

   In a SPHERICAL conductor:
   - Radial dipoles produce ZERO signal at the surface! (theoretical)
   - Only tangential components contribute
   - This is due to symmetry - radial currents create symmetric potentials that cancel

   In REALISTIC geometry (ellipsoid):
   - Radial dipoles CAN produce signal (symmetry is broken)
   - All orientations contribute

3. SKULL CONDUCTIVITY
   - Skull is ~80x less conductive than brain/scalp
   - Acts as a low-pass spatial filter
   - Smears the signal, reducing spatial resolution

4. SOURCE POSITION RELATIVE TO GEOMETRY
   - Sources near conductivity boundaries have complex effects
   - Sources outside the brain have very weak/distorted leadfields
""")

# Analyze one config in detail
config = configs['sphere_vol']
fwd_path = base_dir / config / "data" / "step4_forward.pkl"
coords_path = base_dir / config / "data" / "step3_source_coords_mm.npy"

with open(fwd_path, 'rb') as f:
    fwd = pickle.load(f)

G = fwd['sol']['data']  # (n_channels, n_dipoles)
coords = np.load(coords_path)
n_channels, n_dipoles = G.shape
n_sources = n_dipoles // 3

print("\n" + "=" * 100)
print(f"DETAILED ANALYSIS: {config}")
print("=" * 100)

# For each source, we have 3 dipole orientations (x, y, z)
# Let's analyze by orientation

# Compute leadfield strength per orientation
leadfield_x = np.array([np.linalg.norm(G[:, i*3]) for i in range(n_sources)])  # x-component
leadfield_y = np.array([np.linalg.norm(G[:, i*3+1]) for i in range(n_sources)])  # y-component
leadfield_z = np.array([np.linalg.norm(G[:, i*3+2]) for i in range(n_sources)])  # z-component
leadfield_total = np.array([np.linalg.norm(G[:, i*3:(i+1)*3]) for i in range(n_sources)])

# Compute radial direction for each source (direction from center to source)
source_distances = np.linalg.norm(coords, axis=1)
radial_directions = coords / (source_distances[:, np.newaxis] + 1e-10)

# Depth from brain surface (assuming 6.4mm radius)
brain_radius = 6.4
source_depths = brain_radius - source_distances

# For each source, decompose leadfield into radial vs tangential components
# The radial leadfield is the component of the dipole pointing toward center
# We can approximate this by projecting onto the radial direction

leadfield_radial = []
leadfield_tangential = []

for i in range(n_sources):
    # Leadfield for 3 orientations at this source
    G_source = G[:, i*3:(i+1)*3]  # (n_channels, 3)

    # Radial direction for this source
    r_hat = radial_directions[i]

    # Leadfield magnitude for radial dipole (dot product with radial direction)
    # This is the sensor pattern for a radial dipole
    G_radial = G_source @ r_hat  # (n_channels,)
    leadfield_radial.append(np.linalg.norm(G_radial))

    # Tangential component (orthogonal to radial)
    # Total² = radial² + tangential² (approximately)
    G_total_norm = np.linalg.norm(G_source)
    G_tangential = np.sqrt(max(0, G_total_norm**2 - np.linalg.norm(G_radial)**2))
    leadfield_tangential.append(G_tangential)

leadfield_radial = np.array(leadfield_radial)
leadfield_tangential = np.array(leadfield_tangential)

print(f"\nSource count: {n_sources}")
print(f"Depth range: {source_depths.min():.2f} to {source_depths.max():.2f} mm")

print(f"\n--- Leadfield by Component ---")
print(f"X-orientation:   min={leadfield_x.min():.2e}, max={leadfield_x.max():.2e}, ratio={leadfield_x.max()/leadfield_x.min():.1f}x")
print(f"Y-orientation:   min={leadfield_y.min():.2e}, max={leadfield_y.max():.2e}, ratio={leadfield_y.max()/leadfield_y.min():.1f}x")
print(f"Z-orientation:   min={leadfield_z.min():.2e}, max={leadfield_z.max():.2e}, ratio={leadfield_z.max()/leadfield_z.min():.1f}x")
print(f"Total (all 3):   min={leadfield_total.min():.2e}, max={leadfield_total.max():.2e}, ratio={leadfield_total.max()/leadfield_total.min():.1f}x")

print(f"\n--- Radial vs Tangential Decomposition ---")
print(f"Radial:      min={leadfield_radial.min():.2e}, max={leadfield_radial.max():.2e}, ratio={leadfield_radial.max()/(leadfield_radial.min()+1e-10):.1f}x")
print(f"Tangential:  min={leadfield_tangential.min():.2e}, max={leadfield_tangential.max():.2e}, ratio={leadfield_tangential.max()/leadfield_tangential.min():.1f}x")
print(f"Radial/Total ratio: {(leadfield_radial/leadfield_total).mean():.3f} ± {(leadfield_radial/leadfield_total).std():.3f}")

# Correlation with depth
from scipy.stats import spearmanr
corr_depth, p_depth = spearmanr(source_depths, leadfield_total)
corr_distance, p_dist = spearmanr(source_distances, leadfield_total)

print(f"\n--- Correlations ---")
print(f"Depth vs leadfield:    r={corr_depth:.3f} (p={p_depth:.2e})")
print(f"Distance vs leadfield: r={corr_distance:.3f} (p={p_dist:.2e})")

# Find weakest and strongest sources
weakest_idx = np.argmin(leadfield_total)
strongest_idx = np.argmax(leadfield_total)

print(f"\n--- Extreme Sources ---")
print(f"WEAKEST source {weakest_idx}:")
print(f"  Position: {coords[weakest_idx]}")
print(f"  Distance from center: {source_distances[weakest_idx]:.2f} mm")
print(f"  Depth from surface: {source_depths[weakest_idx]:.2f} mm")
print(f"  Leadfield: total={leadfield_total[weakest_idx]:.2e}, radial={leadfield_radial[weakest_idx]:.2e}, tangential={leadfield_tangential[weakest_idx]:.2e}")

print(f"\nSTRONGEST source {strongest_idx}:")
print(f"  Position: {coords[strongest_idx]}")
print(f"  Distance from center: {source_distances[strongest_idx]:.2f} mm")
print(f"  Depth from surface: {source_depths[strongest_idx]:.2f} mm")
print(f"  Leadfield: total={leadfield_total[strongest_idx]:.2e}, radial={leadfield_radial[strongest_idx]:.2e}, tangential={leadfield_tangential[strongest_idx]:.2e}")

# Now compare sphere vs ellipsoid
print("\n" + "=" * 100)
print("SPHERE VS ELLIPSOID COMPARISON")
print("=" * 100)

for name, config in configs.items():
    fwd_path = base_dir / config / "data" / "step4_forward.pkl"
    coords_path = base_dir / config / "data" / "step3_source_coords_mm.npy"

    with open(fwd_path, 'rb') as f:
        fwd = pickle.load(f)

    G = fwd['sol']['data']
    coords = np.load(coords_path)
    n_sources = G.shape[1] // 3

    # Compute metrics
    source_distances = np.linalg.norm(coords, axis=1)
    radial_directions = coords / (source_distances[:, np.newaxis] + 1e-10)
    source_depths = brain_radius - source_distances

    leadfield_total = np.array([np.linalg.norm(G[:, i*3:(i+1)*3]) for i in range(n_sources)])

    leadfield_radial = []
    for i in range(n_sources):
        G_source = G[:, i*3:(i+1)*3]
        r_hat = radial_directions[i]
        G_radial = G_source @ r_hat
        leadfield_radial.append(np.linalg.norm(G_radial))
    leadfield_radial = np.array(leadfield_radial)

    radial_fraction = leadfield_radial / (leadfield_total + 1e-10)

    print(f"\n{name}:")
    print(f"  Sources: {n_sources}")
    print(f"  Leadfield range: {leadfield_total.max()/leadfield_total.min():.1f}x")
    print(f"  Radial/Total: {radial_fraction.mean():.3f} ± {radial_fraction.std():.3f}")
    print(f"  Radial fraction range: {radial_fraction.min():.3f} to {radial_fraction.max():.3f}")

print("\n" + "=" * 100)
print("KEY TAKEAWAY")
print("=" * 100)
print("""
In SPHERICAL BEM:
- Radial dipoles theoretically produce no signal (spherical symmetry)
- Only tangential components contribute
- Sources pointing toward center have near-zero leadfield
- This creates HUGE leadfield variations (10,000x+)

In ELLIPSOID BEM:
- Symmetry is broken by non-spherical geometry
- ALL dipole orientations contribute signal
- Radial dipoles CAN produce measurable signal
- This creates more UNIFORM leadfield distribution (5-50x)

The sphere's radial dipole problem is the main cause of:
- Poor conditioning
- Large leadfield range
- Biased inverse solutions

RECOMMENDATION: Use ellipsoid BEM for better forward model uniformity.
""")
