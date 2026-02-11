#!/usr/bin/env python
"""
Demonstrate shell-to-MRI space mapping for parametric analysis.

Shows how to:
1. Create shell-based source space
2. Extract depth metadata for each source
3. Assign sources to atlas ROIs
4. Create parametric NIfTI volumes
5. Generate depth-stratified maps

Usage:
    python scripts/012_demo_shell_mri_mapping.py

Output:
    - Console output showing mapping workflow
    - NIfTI files demonstrating parametric mapping
"""

import sys
from pathlib import Path

# Add source to path
script_dir = Path(__file__).parent
src_dir = script_dir.parent / "src"
if src_dir.exists():
    sys.path.insert(0, str(src_dir))

import numpy as np
import nibabel as nib
import json
from source_localization.config import Config
from source_localization.steps import electrode_registration, bem_model, source_space
from source_localization.utils.shell_mapping import (
    get_shell_metadata,
    sources_to_voxels,
    assign_sources_to_rois,
    create_parametric_nifti,
    create_depth_stratified_maps
)


def main():
    print("="*70)
    print("SHELL-TO-MRI MAPPING DEMONSTRATION")
    print("="*70)

    # Output directory
    output_dir = Path('/tmp/shell_mri_demo')
    output_dir.mkdir(parents=True, exist_ok=True)

    # 1. Set up pipeline through source space
    print("\n1. Creating shell-based source space...")
    config = Config.from_preset('shell_based_ellipsoid')

    # Use 4-shell-dense for best conditioning
    config['source_space']['shell_based'].update({
        'n_shells': 4,
        'shell_scales': [0.3, 0.5, 0.7, 0.9],
        'min_points_per_shell': 40,
        'max_points_per_shell': 160,
        'scale_by_area': True,
    })
    config['outputs']['save_intermediate'] = False
    config['outputs']['dir'] = str(output_dir)

    elec_outputs = electrode_registration.run(config, {})
    bem_outputs = bem_model.run(config, elec_outputs)
    previous = {**elec_outputs, **bem_outputs}
    src_outputs = source_space.run(config, previous)

    source_coords_mm = src_outputs['source_coords_mm']
    n_sources = src_outputs['n_sources']
    bem_params = bem_outputs['bem_params']

    print(f"\n   Created {n_sources} sources on 4 concentric shells")

    # 2. Get shell/depth metadata
    print("\n2. Computing depth metadata...")
    shell_scales = config['source_space']['shell_based']['shell_scales']
    metadata = get_shell_metadata(source_coords_mm, bem_params, shell_scales)

    print(f"   Depth distribution:")
    for cat in ['deep', 'middle', 'superficial']:
        count = np.sum(metadata['depth_category'] == cat)
        pct = 100 * count / n_sources
        print(f"     {cat}: {count} sources ({pct:.1f}%)")

    print(f"\n   Shell assignment:")
    for i, scale in enumerate(shell_scales):
        count = np.sum(metadata['shell_index'] == i)
        mean_depth = np.mean(metadata['normalized_depth'][metadata['shell_index'] == i])
        print(f"     Shell {i} (scale={scale:.1f}): {count} sources, mean depth={mean_depth:.2f}")

    # 3. Map sources to voxels
    print("\n3. Mapping sources to MRI voxels...")
    package_dir = Path(__file__).parent.parent / "src/source_localization"
    atlas_path = package_dir / "data/atlas/Atlas_3DRois.nii"
    labels_path = package_dir / "data/atlas/Atlas_3DRoisLeftRight.Labels.nii"
    roi_mapping_path = package_dir / "data/atlas/roi_mapping.json"

    nii_atlas = nib.load(atlas_path)
    nii_labels = nib.load(labels_path)

    voxel_indices, valid_mask = sources_to_voxels(source_coords_mm, nii_atlas)
    print(f"   {np.sum(valid_mask)}/{n_sources} sources mapped to valid voxels")

    # 4. Assign sources to ROIs
    print("\n4. Assigning sources to atlas ROIs...")
    with open(roi_mapping_path) as f:
        roi_data = json.load(f)
    roi_mapping = roi_data.get('rois', roi_data)

    roi_ids, roi_names = assign_sources_to_rois(source_coords_mm, nii_labels, roi_mapping)

    unique_rois = np.unique(roi_ids[roi_ids > 0])
    print(f"   Sources assigned to {len(unique_rois)} unique ROIs")

    # Show top ROIs by source count
    roi_counts = {}
    for roi_id, name in zip(roi_ids, roi_names):
        if roi_id > 0:
            roi_counts[name] = roi_counts.get(name, 0) + 1

    print(f"\n   Top 10 ROIs by source count:")
    for name, count in sorted(roi_counts.items(), key=lambda x: -x[1])[:10]:
        print(f"     {name}: {count} sources")

    # 5. Create example parametric map
    print("\n5. Creating example parametric NIfTI...")

    # Simulate some statistical values (e.g., t-statistics)
    # Higher values for superficial sources, varying by position
    np.random.seed(42)
    simulated_tstat = (
        2.0 * metadata['normalized_depth'] +  # Depth effect
        0.5 * np.random.randn(n_sources)  # Noise
    )

    # Threshold to keep positive effects
    simulated_tstat = np.maximum(simulated_tstat, 0)

    output_nifti = output_dir / 'example_parametric_map.nii.gz'
    nii_out = create_parametric_nifti(
        source_coords_mm,
        simulated_tstat,
        nii_atlas,
        output_path=output_nifti,
        method='linear',
        smooth_fwhm_mm=1.0,
        description='Example t-statistic map from shell sources'
    )

    print(f"   Output shape: {nii_out.shape}")
    print(f"   Value range: [{nii_out.get_fdata().min():.2f}, {nii_out.get_fdata().max():.2f}]")

    # 6. Create depth-stratified maps
    print("\n6. Creating depth-stratified maps...")
    depth_maps = create_depth_stratified_maps(
        source_coords_mm,
        simulated_tstat,
        bem_params,
        nii_atlas,
        output_dir=output_dir / 'depth_stratified'
    )

    for depth_label, nii in depth_maps.items():
        data = nii.get_fdata()
        nonzero = data[data > 0]
        if len(nonzero) > 0:
            print(f"   {depth_label}: mean={nonzero.mean():.2f}, max={nonzero.max():.2f}")

    # Summary
    print("\n" + "="*70)
    print("SUMMARY: Shell-to-MRI Mapping Workflow")
    print("="*70)
    print("""
For parametric analysis with shell-based sources:

1. SOURCE SPACE
   - Create sources on concentric shells matching BEM geometry
   - Each source has (x, y, z) coordinates in mm (RAS space)

2. DEPTH METADATA
   - normalized_depth: 0 (center) to 1 (surface)
   - shell_index: which shell the source belongs to
   - depth_category: 'deep', 'middle', or 'superficial'
   - Use for depth-stratified statistical analysis

3. ROI ASSIGNMENT
   - Map source coordinates to atlas voxels
   - Look up ROI label for each source
   - Use for regional summary statistics

4. PARAMETRIC VISUALIZATION
   - Interpolate source values to voxel grid
   - Apply smoothing (FWHM ~1mm for mouse brain)
   - Save as NIfTI for overlay on anatomy

5. DEPTH-STRATIFIED ANALYSIS
   - Create separate maps for deep/middle/superficial sources
   - Compare effects across depth levels
   - Assess depth bias in localization
""")

    print(f"\nOutput files saved to: {output_dir}")
    print("  - example_parametric_map.nii.gz")
    print("  - depth_stratified/parametric_map_*.nii.gz")


if __name__ == '__main__':
    main()
