#!/usr/bin/env python3
"""Debug script to check electrode coordinates from pipeline output."""

import sys
from pathlib import Path
import numpy as np

# Add package to path
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

from source_localization.utils.electrode_registration import load_electrodes_from_p100

# Load electrodes using the same parameters as pipeline
package_dir = Path(__file__).parent.parent / 'src' / 'source_localization'
electrodes_csv = package_dir / 'data' / 'electrodes' / 'mouse_array_coords.csv'
atlas_nii = package_dir / 'data' / 'atlas' / 'Atlas_3DRois.nii'

print("="*70)
print("Debug: Electrode Coordinate Loading")
print("="*70)
print(f"Electrodes CSV: {electrodes_csv}")
print(f"Atlas: {atlas_nii}")
print()

info, results = load_electrodes_from_p100(
    electrodes_csv=electrodes_csv,
    atlas_nii=atlas_nii,
    projection_method='intensity',
    skull_offset_mm=0.0,
    bregma_vox=(30, 149, 41),
    lambda_vox=(30, 97, 41),
    create_visualization=False,
    output_dir=None,
    sfreq=1000.0
)

print("\n" + "="*70)
print("Extracting Coordinates from MNE Info Object")
print("="*70)

# Extract exactly as pipeline step does
n_electrodes = len(info['ch_names'])
electrodes_mri_m = np.array([info['chs'][i]['loc'][:3] for i in range(n_electrodes)])
electrodes_mri_mm = electrodes_mri_m * 1000  # Convert to mm

print(f"\nNumber of electrodes: {n_electrodes}")
print(f"\nCoordinates in METERS (as stored in info object):")
print(f"  X: [{electrodes_mri_m[:, 0].min():.6f}, {electrodes_mri_m[:, 0].max():.6f}] m")
print(f"  Y: [{electrodes_mri_m[:, 1].min():.6f}, {electrodes_mri_m[:, 1].max():.6f}] m")
print(f"  Z: [{electrodes_mri_m[:, 2].min():.6f}, {electrodes_mri_m[:, 2].max():.6f}] m")

print(f"\nCoordinates in MILLIMETERS (×1000):")
print(f"  X: [{electrodes_mri_mm[:, 0].min():.2f}, {electrodes_mri_mm[:, 0].max():.2f}] mm")
print(f"  Y: [{electrodes_mri_mm[:, 1].min():.2f}, {electrodes_mri_mm[:, 1].max():.2f}] mm")
print(f"  Z: [{electrodes_mri_mm[:, 2].min():.2f}, {electrodes_mri_mm[:, 2].max():.2f}] mm")

print(f"\nFrom validation results (coords returned by function):")
coords_from_results = results['electrode_coords_mm']
print(f"  X: [{coords_from_results[:, 0].min():.2f}, {coords_from_results[:, 0].max():.2f}] mm")
print(f"  Y: [{coords_from_results[:, 1].min():.2f}, {coords_from_results[:, 1].max():.2f}] mm")
print(f"  Z: [{coords_from_results[:, 2].min():.2f}, {coords_from_results[:, 2].max():.2f}] mm")

print(f"\nSample electrodes (first 3):")
for i in range(min(3, n_electrodes)):
    print(f"  {info['ch_names'][i]}: "
          f"[{electrodes_mri_mm[i, 0]:.2f}, {electrodes_mri_mm[i, 1]:.2f}, {electrodes_mri_mm[i, 2]:.2f}] mm")

print("\n" + "="*70)
