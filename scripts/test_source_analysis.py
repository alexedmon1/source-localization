#!/usr/bin/env python
"""
Test Source Analysis Module on Real Data

This script tests the new source_analysis module by:
1. Running the standard pipeline to get source estimates
2. Creating a cortical source space
3. Computing source-level statistics
4. Generating heatmap visualizations
5. Using atlas lookup for interpretation

Usage:
    python scripts/test_source_analysis.py /path/to/eeg_file.set --output ./test_output

Author: Claude Code
Date: 2026-01-26
"""

import argparse
import sys
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt

# Add package to path
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

from source_localization import Pipeline
from source_localization.source_analysis import (
    CorticalSourceSpace,
    SourceStatistics,
    AtlasLookup,
    SourceMapVisualizer
)


def main():
    parser = argparse.ArgumentParser(description='Test source analysis module')
    parser.add_argument('eeg_file', type=str, help='Path to EEG .set file')
    parser.add_argument('--output', type=str, default='./source_analysis_test',
                       help='Output directory')
    parser.add_argument('--preset', type=str, default='ellipsoid_volumetric',
                       help='Pipeline preset to use')
    parser.add_argument('--max-depth', type=float, default=2.0,
                       help='Maximum depth from surface in mm')
    parser.add_argument('--band', type=str, default='gamma',
                       choices=['theta', 'alpha', 'beta', 'gamma'],
                       help='Frequency band to analyze')
    args = parser.parse_args()

    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("SOURCE ANALYSIS MODULE TEST")
    print("=" * 60)
    print(f"EEG file: {args.eeg_file}")
    print(f"Output: {output_dir}")
    print(f"Preset: {args.preset}")
    print(f"Max depth: {args.max_depth} mm")
    print(f"Frequency band: {args.band}")
    print()

    # =========================================================================
    # Step 1: Run standard pipeline to get source estimates
    # =========================================================================
    print("Step 1: Running source localization pipeline...")

    pipeline = Pipeline.from_preset(args.preset)
    pipeline_output = output_dir / 'pipeline'

    results = pipeline.run(
        eeg_file=args.eeg_file,
        output_dir=str(pipeline_output)
    )

    print(f"  - Pipeline complete")

    # Load outputs from saved files
    import pickle

    # Load source time course (VolSourceEstimate)
    with open(pipeline_output / 'data/step5_stc_magnitude.pkl', 'rb') as f:
        stc = pickle.load(f)

    # Load source coordinates
    source_coords_mm = np.load(pipeline_output / 'data/step3_source_coords_mm.npy')

    # Get source activity data
    source_activity = stc.data  # (n_sources, n_times)
    sfreq = stc.sfreq

    print(f"  - Source activity shape: {source_activity.shape}")

    print(f"  - {len(source_coords_mm)} sources at {sfreq} Hz")

    # =========================================================================
    # Step 2: Create cortical (depth-restricted) source space
    # =========================================================================
    print("\nStep 2: Creating cortical source space...")

    # Get atlas path
    atlas_path = Path(__file__).parent.parent / 'src/source_localization/data/atlas/Atlas_3DRois_brain.nii.gz'
    roi_mapping_path = Path(__file__).parent.parent / 'src/source_localization/data/atlas/roi_mapping.json'

    cortical_src = CorticalSourceSpace.from_atlas(
        str(atlas_path),
        max_depth_mm=args.max_depth,
        spacing_mm=0.5,
        apply_10x_correction=True
    )

    print(f"  - Cortical sources: {cortical_src.n_sources}")
    print(f"  - Depth range: {cortical_src.source_depths_mm.min():.2f} - "
          f"{cortical_src.source_depths_mm.max():.2f} mm")

    # Map pipeline sources to cortical sources (find nearest)
    # This step is needed because the pipeline may use different source placement
    print("\nStep 3: Mapping pipeline sources to cortical space...")

    # Find which pipeline sources are within our cortical depth range
    from scipy.spatial import cKDTree
    from scipy.ndimage import distance_transform_edt
    import nibabel as nib

    # Load brain mask and compute depth for each pipeline source
    nii = nib.load(atlas_path)
    brain_mask = np.asarray(nii.dataobj) > 0
    affine = nii.affine.copy()
    affine[:3, :3] /= 10.0  # Apply 10x correction

    distance_voxels = distance_transform_edt(brain_mask)
    voxel_size = np.mean(np.abs(np.diag(affine)[:3]))
    distance_mm = distance_voxels * voxel_size

    # Get depth for each pipeline source
    affine_inv = np.linalg.inv(affine)
    source_voxels = nib.affines.apply_affine(affine_inv, source_coords_mm)
    source_voxels = np.round(source_voxels).astype(int)

    # Clip to valid range
    for i in range(3):
        source_voxels[:, i] = np.clip(source_voxels[:, i], 0, brain_mask.shape[i] - 1)

    source_depths = np.array([
        distance_mm[tuple(v)] for v in source_voxels
    ])

    # Filter to cortical sources
    cortical_mask = source_depths <= args.max_depth
    n_cortical = np.sum(cortical_mask)
    print(f"  - {n_cortical} of {len(source_coords_mm)} pipeline sources are cortical")

    cortical_source_activity = source_activity[cortical_mask]
    cortical_source_coords = source_coords_mm[cortical_mask]
    cortical_source_depths = source_depths[cortical_mask]

    # =========================================================================
    # Step 4: Compute source-level statistics
    # =========================================================================
    print("\nStep 4: Computing source-level statistics...")

    stats = SourceStatistics(
        source_data=cortical_source_activity,
        source_coords_mm=cortical_source_coords,
        sfreq=sfreq,
        source_depths_mm=cortical_source_depths
    )

    # Compute band power at each source
    band_power = stats.compute_band_power(args.band, log_transform=True)
    print(f"  - {args.band} power range: {band_power.min():.2f} to {band_power.max():.2f}")

    # Find clusters
    threshold = np.percentile(band_power, 90)
    clusters = stats.find_clusters(band_power, threshold=threshold, min_cluster_size=5)
    print(f"  - Found {len(clusters)} clusters above 90th percentile")

    # Find peaks
    peaks = stats.find_peaks(band_power, min_distance_mm=2.0, n_peaks=10)
    print(f"  - Found {len(peaks)} peaks")

    # =========================================================================
    # Step 5: Atlas lookup for interpretation
    # =========================================================================
    print("\nStep 5: Atlas lookup for peak interpretation...")

    atlas = AtlasLookup(
        str(atlas_path.parent / 'Atlas_3DRois.nii'),
        roi_mapping_path=str(roi_mapping_path),
        apply_10x_correction=True
    )

    # Label peaks
    atlas.label_peaks(peaks)

    print("\n  Top 10 Peaks:")
    print("  " + "-" * 60)
    print(f"  {'Rank':<6}{'Value':<10}{'Depth':<10}{'Region':<30}")
    print("  " + "-" * 60)
    for i, peak in enumerate(peaks[:10]):
        depth_str = f"{peak.depth_mm:.1f}mm" if peak.depth_mm else "N/A"
        region = peak.region_label or "Unknown"
        print(f"  {i+1:<6}{peak.value:<10.2f}{depth_str:<10}{region:<30}")

    # Summarize by region
    region_summary = atlas.summarize_sources_by_region(
        cortical_source_coords,
        band_power,
        search_radius_mm=2.0
    )

    print(f"\n  Sources mapped to {len(region_summary)} regions")

    # Top regions by mean power
    sorted_regions = sorted(region_summary.items(), key=lambda x: x[1]['mean'], reverse=True)
    print("\n  Top 5 Regions by Mean Power:")
    print("  " + "-" * 50)
    for name, summary in sorted_regions[:5]:
        print(f"  {name}: mean={summary['mean']:.2f}, n={summary['n_sources']}")

    # =========================================================================
    # Step 6: Generate visualizations
    # =========================================================================
    print("\nStep 6: Generating visualizations...")

    # Get brain surface for visualization
    surface_mask = (distance_mm > 0) & (distance_mm <= voxel_size)
    surface_voxels = np.array(np.where(surface_mask & brain_mask)).T
    brain_surface_coords = nib.affines.apply_affine(affine, surface_voxels)

    viz = SourceMapVisualizer(
        source_coords_mm=cortical_source_coords,
        brain_surface_coords=brain_surface_coords,
        source_depths_mm=cortical_source_depths
    )

    # 1. Multi-view heatmap
    print("  - Creating multi-view heatmap...")
    fig = viz.plot_multiview_heatmap(
        band_power,
        cmap='hot',
        threshold_percentile=75,
        title=f'{args.band.capitalize()} Power - Multi-View'
    )
    fig.savefig(output_dir / 'heatmap_multiview.png', dpi=150, bbox_inches='tight')
    plt.close(fig)

    # 2. MIP heatmap
    print("  - Creating MIP heatmap...")
    fig = viz.plot_mip_heatmap(
        band_power,
        smoothing_mm=1.5,
        threshold_percentile=85,
        title=f'{args.band.capitalize()} Power - Maximum Intensity Projection'
    )
    fig.savefig(output_dir / 'heatmap_mip.png', dpi=150, bbox_inches='tight')
    plt.close(fig)

    # 3. Slice montage
    print("  - Creating axial slice montage...")
    fig = viz.plot_slice_heatmap_montage(
        band_power,
        plane='axial',
        n_slices=6,
        smoothing_mm=1.0,
        threshold_percentile=80,
        title=f'{args.band.capitalize()} Power - Axial Slices'
    )
    fig.savefig(output_dir / 'heatmap_axial_slices.png', dpi=150, bbox_inches='tight')
    plt.close(fig)

    # 4. Coronal slice montage
    print("  - Creating coronal slice montage...")
    fig = viz.plot_slice_heatmap_montage(
        band_power,
        plane='coronal',
        n_slices=6,
        smoothing_mm=1.0,
        threshold_percentile=80,
        title=f'{args.band.capitalize()} Power - Coronal Slices'
    )
    fig.savefig(output_dir / 'heatmap_coronal_slices.png', dpi=150, bbox_inches='tight')
    plt.close(fig)

    # 5. Single slice heatmap (mid-brain)
    print("  - Creating single slice heatmap...")
    fig = viz.plot_slice_heatmap(
        band_power,
        plane='axial',
        smoothing_mm=1.5,
        threshold_percentile=75,
        title=f'{args.band.capitalize()} Power - Mid-Axial Slice'
    )
    fig.savefig(output_dir / 'heatmap_single_slice.png', dpi=150, bbox_inches='tight')
    plt.close(fig)

    # 6. Dorsal view surface heatmap
    print("  - Creating dorsal surface heatmap...")
    fig = viz.plot_surface_heatmap(
        band_power,
        view='dorsal',
        threshold_percentile=70,
        title=f'{args.band.capitalize()} Power - Dorsal View'
    )
    fig.savefig(output_dir / 'heatmap_dorsal.png', dpi=150, bbox_inches='tight')
    plt.close(fig)

    # 7. 3D scatter plot
    print("  - Creating 3D scatter plot...")
    fig = viz.plot_3d_scatter(
        band_power,
        threshold=np.percentile(band_power, 75),
        title=f'{args.band.capitalize()} Power - 3D View',
        size_by_value=True
    )
    fig.savefig(output_dir / 'scatter_3d.png', dpi=150, bbox_inches='tight')
    plt.close(fig)

    # 8. Depth histogram
    print("  - Creating depth histogram...")
    fig = viz.plot_depth_histogram(
        values=band_power,
        title='Source Power by Depth'
    )
    fig.savefig(output_dir / 'depth_histogram.png', dpi=150, bbox_inches='tight')
    plt.close(fig)

    # 9. Cluster visualization
    if clusters:
        print("  - Creating cluster visualization...")
        fig = viz.plot_clusters(
            clusters[:10],
            title=f'{args.band.capitalize()} Power Clusters'
        )
        fig.savefig(output_dir / 'clusters.png', dpi=150, bbox_inches='tight')
        plt.close(fig)

    # 10. Peak visualization
    if peaks:
        print("  - Creating peak visualization...")
        fig = viz.plot_peaks(
            peaks[:10],
            values=band_power,
            title=f'Top 10 {args.band.capitalize()} Power Peaks'
        )
        fig.savefig(output_dir / 'peaks.png', dpi=150, bbox_inches='tight')
        plt.close(fig)

    # =========================================================================
    # Summary
    # =========================================================================
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"Total pipeline sources: {len(source_coords_mm)}")
    print(f"Cortical sources (depth <= {args.max_depth}mm): {n_cortical}")
    print(f"Frequency band analyzed: {args.band}")
    print(f"Clusters found: {len(clusters)}")
    print(f"Peaks found: {len(peaks)}")
    print(f"\nOutput directory: {output_dir}")
    print("\nGenerated files:")
    for f in sorted(output_dir.glob('*.png')):
        print(f"  - {f.name}")
    print("\nDone!")


if __name__ == '__main__':
    main()
