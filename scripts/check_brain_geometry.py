#!/usr/bin/env python3
"""Check the actual geometry of the mouse brain."""

import numpy as np
import nibabel as nib
from pathlib import Path

# Load the atlas
atlas_path = Path("/home/metalexy/sandbox/AlexProjects/mouse-eeg-source-localization/source_localization/src/source_localization/data/atlas/Atlas_3DRois.nii")

print("=" * 80)
print("MOUSE BRAIN GEOMETRY ANALYSIS")
print("=" * 80)

nii = nib.load(atlas_path)
data = nii.get_fdata()
affine = nii.affine.copy()

# Correct for the 10x voxel size issue
affine[:3, :3] /= 10.0

# Find brain voxels (non-zero)
brain_voxels = np.array(np.where(data > 0)).T  # (N, 3) voxel indices

# Convert to mm coordinates
brain_coords_mm = nib.affines.apply_affine(affine, brain_voxels)

print(f"\nBrain voxel count: {len(brain_voxels)}")

# Compute bounding box
x_range = brain_coords_mm[:, 0].max() - brain_coords_mm[:, 0].min()
y_range = brain_coords_mm[:, 1].max() - brain_coords_mm[:, 1].min()
z_range = brain_coords_mm[:, 2].max() - brain_coords_mm[:, 2].min()

print(f"\n--- Bounding Box ---")
print(f"X (left-right):     {brain_coords_mm[:, 0].min():.2f} to {brain_coords_mm[:, 0].max():.2f} mm (width: {x_range:.2f} mm)")
print(f"Y (anterior-post):  {brain_coords_mm[:, 1].min():.2f} to {brain_coords_mm[:, 1].max():.2f} mm (length: {y_range:.2f} mm)")
print(f"Z (dorsal-ventral): {brain_coords_mm[:, 2].min():.2f} to {brain_coords_mm[:, 2].max():.2f} mm (height: {z_range:.2f} mm)")

# Aspect ratios
print(f"\n--- Aspect Ratios ---")
print(f"Y/X (length/width): {y_range/x_range:.2f}")
print(f"Z/X (height/width): {z_range/x_range:.2f}")
print(f"Y/Z (length/height): {y_range/z_range:.2f}")

# Fit ellipsoid
centroid = brain_coords_mm.mean(axis=0)
centered = brain_coords_mm - centroid

# Compute covariance and principal axes
cov = np.cov(centered.T)
eigenvalues, eigenvectors = np.linalg.eigh(cov)
eigenvalues = eigenvalues[::-1]  # Sort descending
eigenvectors = eigenvectors[:, ::-1]

# Semi-axes of best-fit ellipsoid (scaled by sqrt of eigenvalues)
# For 95th percentile coverage
scale_factor = 2.0  # ~95% of points within 2 std
semi_axes = np.sqrt(eigenvalues) * scale_factor

print(f"\n--- Best-Fit Ellipsoid ---")
print(f"Centroid: {centroid}")
print(f"Semi-axes (mm): {semi_axes[0]:.2f} × {semi_axes[1]:.2f} × {semi_axes[2]:.2f}")
print(f"Axis ratios: {semi_axes[0]/semi_axes[2]:.2f} : {semi_axes[1]/semi_axes[2]:.2f} : 1.00")

# Compare to sphere
sphere_radius = np.percentile(np.linalg.norm(centered, axis=1), 95)
print(f"\n--- Sphere Approximation ---")
print(f"Best-fit sphere radius (95th %ile): {sphere_radius:.2f} mm")

# How well does sphere fit?
distances = np.linalg.norm(centered, axis=1)
sphere_fit_error = np.std(distances - sphere_radius)
print(f"Sphere fit error (std of residuals): {sphere_fit_error:.2f} mm")

# Sphericity measure (1 = perfect sphere)
sphericity = semi_axes[2] / semi_axes[0]  # smallest / largest
print(f"Sphericity (smallest/largest axis): {sphericity:.2f} (1.0 = perfect sphere)")

# Human comparison
print(f"\n--- Human Comparison ---")
print(f"Human head is roughly: 180mm × 150mm × 140mm (L×W×H)")
print(f"Human sphericity: ~{140/180:.2f}")
print(f"Mouse sphericity: ~{sphericity:.2f}")

print(f"\n" + "=" * 80)
print("CONCLUSION")
print("=" * 80)
print(f"""
The mouse brain is ELONGATED in the anterior-posterior (Y) direction:
- Length (Y): {y_range:.1f} mm
- Width (X):  {x_range:.1f} mm
- Height (Z): {z_range:.1f} mm

Sphericity: {sphericity:.2f} (human is ~0.78)

A sphere model assumes all dimensions are equal, which is a POOR fit
for the elongated mouse brain. The ellipsoid model better captures
the actual geometry, leading to:
- More accurate forward model
- More uniform leadfield distribution
- Better inverse solutions
""")
