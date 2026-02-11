#!/usr/bin/env python3
"""
Visualize ROI Accuracy as Mosaic Maps.

Creates mosaic views (coronal, axial, sagittal) of the mouse brain atlas
with ROI classification accuracy overlaid as a color map.

Usage:
    python visualize_roi_accuracy_mosaic.py <metrics_json> [--output-dir <dir>] [--atlas <full|coarse_22roi>]

Example:
    python visualize_roi_accuracy_mosaic.py validation/results/V10_ellipsoid_vol_sloreta/metrics.json
    python visualize_roi_accuracy_mosaic.py validation/results/V10_ellipsoid_vol_sloreta_coarse22/metrics.json --atlas coarse_22roi
"""

import argparse
import json
import numpy as np
import nibabel as nib
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from pathlib import Path


def load_roi_accuracy(metrics_path: Path) -> dict:
    """Load ROI accuracy from validation metrics JSON."""
    with open(metrics_path) as f:
        metrics = json.load(f)

    raw = metrics['snr_results']['10']['raw_data']
    true_roi_ids = np.array(raw['true_roi_ids'])
    roi_correct = np.array(raw['roi_correct'])

    # Compute accuracy per ROI
    roi_accuracy = {}
    for roi_id in np.unique(true_roi_ids):
        mask = true_roi_ids == roi_id
        acc = 100 * np.mean(roi_correct[mask])
        roi_accuracy[int(roi_id)] = acc

    return roi_accuracy, metrics


def create_accuracy_volume(atlas_data: np.ndarray, roi_accuracy: dict, all_roi_ids: set) -> np.ndarray:
    """Create a volume with accuracy values for each ROI."""
    accuracy_volume = np.zeros_like(atlas_data, dtype=float)
    accuracy_volume[:] = np.nan  # NaN for non-ROI areas

    tested_roi_ids = set(roi_accuracy.keys())

    # Fill tested ROIs with accuracy
    for roi_id, acc in roi_accuracy.items():
        roi_mask = atlas_data == roi_id
        accuracy_volume[roi_mask] = acc

    # Mark untested ROIs
    untested_roi_ids = all_roi_ids - tested_roi_ids - {0}  # Exclude exterior
    for roi_id in untested_roi_ids:
        roi_mask = atlas_data == roi_id
        accuracy_volume[roi_mask] = -1  # Mark as untested

    return accuracy_volume, untested_roi_ids


def create_mosaic(bg_data: np.ndarray, acc_volume: np.ndarray,
                  view: str, title: str, output_path: Path,
                  slice_range: tuple = None):
    """Create a mosaic visualization for a given view."""

    # Custom colormap: red (0%) -> yellow (50%) -> green (100%)
    colors = ['#d73027', '#fc8d59', '#fee08b', '#d9ef8b', '#91cf60', '#1a9850']
    cmap_acc = LinearSegmentedColormap.from_list('accuracy', colors, N=256)
    cmap_acc.set_bad(alpha=0)  # Transparent for NaN
    cmap_acc.set_under(color='gray', alpha=0.3)  # Gray for untested

    # Determine slice parameters based on view
    if view == 'coronal':
        n_rows, n_cols = 4, 6
        dim = 1  # Y axis
        default_range = (20, 235)
    elif view == 'axial':
        n_rows, n_cols = 3, 6
        dim = 2  # Z axis
        default_range = (5, 45)
    elif view == 'sagittal':
        n_rows, n_cols = 3, 6
        dim = 0  # X axis
        default_range = (10, 54)
    else:
        raise ValueError(f"Unknown view: {view}")

    if slice_range is None:
        slice_range = default_range

    n_slices = n_rows * n_cols
    slice_indices = np.linspace(slice_range[0], slice_range[1], n_slices, dtype=int)

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(18, n_rows * 3))
    fig.suptitle(title, fontsize=14, fontweight='bold')

    axis_labels = ['X', 'Y', 'Z']

    for idx, ax in enumerate(axes.flat):
        if idx >= len(slice_indices):
            ax.axis('off')
            continue

        slice_idx = slice_indices[idx]

        # Get slice based on view
        if dim == 0:
            bg_slice = bg_data[slice_idx, :, :]
            acc_slice = acc_volume[slice_idx, :, :]
        elif dim == 1:
            bg_slice = bg_data[:, slice_idx, :]
            acc_slice = acc_volume[:, slice_idx, :]
        else:
            bg_slice = bg_data[:, :, slice_idx]
            acc_slice = acc_volume[:, :, slice_idx]

        # Plot background (intensity)
        ax.imshow(bg_slice.T, cmap='gray', origin='lower', aspect='auto')

        # Overlay accuracy with transparency
        masked_acc = np.ma.masked_invalid(acc_slice.T)
        masked_acc = np.ma.masked_less(masked_acc, 0)

        im = ax.imshow(masked_acc, cmap=cmap_acc, origin='lower', aspect='auto',
                       alpha=0.7, vmin=0, vmax=100)

        # Show untested ROIs in gray
        untested_mask = acc_slice.T == -1
        if np.any(untested_mask):
            untested_overlay = np.zeros((*acc_slice.T.shape, 4))
            untested_overlay[untested_mask] = [0.5, 0.5, 0.5, 0.3]
            ax.imshow(untested_overlay, origin='lower', aspect='auto')

        ax.set_title(f'{axis_labels[dim]}={slice_idx}', fontsize=8)
        ax.axis('off')

    # Add colorbar
    cbar_ax = fig.add_axes([0.92, 0.15, 0.02, 0.7])
    cbar = fig.colorbar(im, cax=cbar_ax)
    cbar.set_label('ROI Accuracy (%)', fontsize=12)
    cbar.set_ticks([0, 25, 50, 75, 100])

    plt.tight_layout(rect=[0, 0, 0.9, 0.95])
    plt.savefig(output_path, dpi=150, bbox_inches='tight',
                facecolor='white', edgecolor='none')
    plt.close()
    print(f"Saved: {output_path}")


def main():
    parser = argparse.ArgumentParser(description='Create ROI accuracy mosaic visualizations')
    parser.add_argument('metrics_json', type=str, help='Path to validation metrics.json')
    parser.add_argument('--output-dir', type=str, default=None,
                        help='Output directory (default: same as metrics.json)')
    parser.add_argument('--atlas', type=str, choices=['full', 'coarse_22roi'], default='full',
                        help='Atlas type used in validation')
    args = parser.parse_args()

    metrics_path = Path(args.metrics_json)
    output_dir = Path(args.output_dir) if args.output_dir else metrics_path.parent

    # Determine paths based on atlas type
    package_dir = Path(__file__).parent.parent / 'src' / 'source_localization' / 'data'

    if args.atlas == 'coarse_22roi':
        atlas_path = package_dir / 'atlas' / 'coarse_parcellation' / 'coarse_22roi_atlas.nii.gz'
        roi_mapping_path = package_dir / 'atlas' / 'coarse_parcellation' / 'coarse_22roi_mapping.json'
    else:
        atlas_path = package_dir / 'atlas' / 'Atlas_3DRoisLeftRight.Labels.nii'
        roi_mapping_path = package_dir / 'atlas' / 'roi_mapping.json'

    orig_atlas_path = package_dir / 'atlas' / 'Atlas_3DRois.nii'

    print(f"Loading atlas: {atlas_path}")
    print(f"Loading ROI mapping: {roi_mapping_path}")
    print(f"Loading metrics: {metrics_path}")

    # Load data
    atlas_nii = nib.load(atlas_path)
    atlas_data = atlas_nii.get_fdata()

    orig_nii = nib.load(orig_atlas_path)
    orig_data = orig_nii.get_fdata()

    with open(roi_mapping_path) as f:
        roi_mapping = json.load(f)

    # Get all ROI IDs from mapping
    if 'rois' in roi_mapping:
        all_roi_ids = set(int(k) for k in roi_mapping['rois'].keys())
    else:
        all_roi_ids = set(int(k) for k in roi_mapping.keys() if k.isdigit())

    # Load accuracy data
    roi_accuracy, metrics = load_roi_accuracy(metrics_path)

    print(f"\nROI Accuracies ({len(roi_accuracy)} ROIs tested):")
    for roi_id, acc in sorted(roi_accuracy.items()):
        if 'rois' in roi_mapping:
            roi_name = roi_mapping['rois'].get(str(roi_id), {}).get('name', f'ROI_{roi_id}')
        else:
            roi_name = roi_mapping.get(str(roi_id), {}).get('name', f'ROI_{roi_id}')
        print(f"  {roi_id:>2}: {roi_name:<30} = {acc:.1f}%")

    # Create accuracy volume
    accuracy_volume, untested = create_accuracy_volume(atlas_data, roi_accuracy, all_roi_ids)
    if untested:
        print(f"\nUntested ROIs: {untested}")

    # Get config info for title
    config_name = metrics.get('config_name', 'Unknown')
    inverse_method = metrics.get('inverse_method', 'Unknown')
    atlas_name = 'Coarse 22-ROI' if args.atlas == 'coarse_22roi' else 'Full 47-ROI'

    # Create mosaics
    base_title = f'Mouse Brain ROI Accuracy Map ({inverse_method}, {atlas_name} Atlas)\nOverlay: ROI Classification Accuracy (%)'

    output_dir.mkdir(parents=True, exist_ok=True)
    prefix = config_name.replace(' ', '_')

    create_mosaic(orig_data, accuracy_volume, 'coronal',
                  base_title.replace('Map', 'Map - Coronal View'),
                  output_dir / f'{prefix}_accuracy_mosaic_coronal.png')

    create_mosaic(orig_data, accuracy_volume, 'axial',
                  base_title.replace('Map', 'Map - Axial View'),
                  output_dir / f'{prefix}_accuracy_mosaic_axial.png')

    create_mosaic(orig_data, accuracy_volume, 'sagittal',
                  base_title.replace('Map', 'Map - Sagittal View'),
                  output_dir / f'{prefix}_accuracy_mosaic_sagittal.png')

    print(f"\nDone! Created 3 mosaic views in {output_dir}")


if __name__ == '__main__':
    main()
