"""
Source coverage visualization with brain template overlay.

Creates figures showing source space coverage on the mouse brain atlas
with electrode positions for comparing different preset configurations.
"""

import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import nibabel as nib


def load_brain_outline(atlas_path: Path, slice_axis: int = 2) -> Tuple[np.ndarray, np.ndarray]:
    """
    Load brain atlas and extract outline for visualization.

    Parameters
    ----------
    atlas_path : Path
        Path to brain atlas NIfTI file
    slice_axis : int
        Axis to slice along (0=sagittal, 1=coronal, 2=axial)

    Returns
    -------
    brain_mask : ndarray
        3D binary brain mask
    affine : ndarray
        Corrected affine transformation (with 10x scaling fix)
    """
    nii = nib.load(atlas_path)
    data = nii.get_fdata()

    # Correct affine for 10x voxel size scaling issue
    affine = nii.affine.copy()
    affine[:3, :3] /= 10.0

    # Create binary brain mask (any non-zero label is brain)
    brain_mask = (data > 0).astype(np.float32)

    return brain_mask, affine


def get_brain_contour_coords(brain_mask: np.ndarray, affine: np.ndarray,
                              slice_idx: int, axis: int) -> Tuple[np.ndarray, np.ndarray]:
    """
    Get brain outline coordinates for a given slice.

    Parameters
    ----------
    brain_mask : ndarray
        3D binary brain mask
    affine : ndarray
        Affine transformation
    slice_idx : int
        Slice index
    axis : int
        Axis to slice along (0=X, 1=Y, 2=Z)

    Returns
    -------
    x_coords, y_coords : ndarray
        Coordinates of brain outline in mm
    """
    from scipy import ndimage

    # Get the 2D slice
    if axis == 0:
        slice_2d = brain_mask[slice_idx, :, :]
        voxel_axes = (1, 2)  # Y, Z
    elif axis == 1:
        slice_2d = brain_mask[:, slice_idx, :]
        voxel_axes = (0, 2)  # X, Z
    else:  # axis == 2
        slice_2d = brain_mask[:, :, slice_idx]
        voxel_axes = (0, 1)  # X, Y

    # Get edge/contour using gradient
    edges = ndimage.sobel(slice_2d.astype(float))
    edge_mask = np.abs(edges) > 0.1

    # Get coordinates
    voxel_coords = np.where(edge_mask)

    if len(voxel_coords[0]) == 0:
        return np.array([]), np.array([])

    # Convert to mm using affine
    # Create full 3D coordinates for transformation
    n_points = len(voxel_coords[0])
    full_coords = np.zeros((n_points, 3))
    full_coords[:, voxel_axes[0]] = voxel_coords[0]
    full_coords[:, voxel_axes[1]] = voxel_coords[1]
    full_coords[:, axis] = slice_idx

    # Apply affine
    coords_mm = nib.affines.apply_affine(affine, full_coords)

    return coords_mm[:, voxel_axes[0]], coords_mm[:, voxel_axes[1]]


def create_source_coverage_figure(
    source_coords_mm: np.ndarray,
    electrode_coords_mm: np.ndarray,
    preset_name: str,
    brain_atlas_path: Optional[Path] = None,
    output_path: Optional[Path] = None,
    figsize: Tuple[int, int] = (18, 6)
) -> plt.Figure:
    """
    Create source coverage visualization with brain template.

    Parameters
    ----------
    source_coords_mm : ndarray, shape (n_sources, 3)
        Source positions in mm
    electrode_coords_mm : ndarray, shape (n_electrodes, 3)
        Electrode positions in mm
    preset_name : str
        Name of the preset configuration
    brain_atlas_path : Path, optional
        Path to brain atlas NIfTI file
    output_path : Path, optional
        Path to save figure
    figsize : tuple
        Figure size (width, height)

    Returns
    -------
    fig : matplotlib.Figure
        Generated figure
    """
    fig, axes = plt.subplots(1, 3, figsize=figsize)

    n_sources = len(source_coords_mm)

    # Load brain outline if atlas provided
    brain_outline = None
    if brain_atlas_path is not None and Path(brain_atlas_path).exists():
        try:
            brain_mask, affine = load_brain_outline(brain_atlas_path)
            brain_outline = (brain_mask, affine)
        except Exception as e:
            print(f"Warning: Could not load brain atlas: {e}")

    # Color sources by depth (Z coordinate)
    z_coords = source_coords_mm[:, 2]
    z_norm = (z_coords - z_coords.min()) / (z_coords.max() - z_coords.min() + 1e-6)
    colors = plt.cm.viridis(z_norm)

    views = [
        (0, 1, 'Axial (Dorsal View)', 'X (mm)', 'Y (mm)', 2),
        (0, 2, 'Coronal (Anterior View)', 'X (mm)', 'Z (mm)', 1),
        (1, 2, 'Sagittal (Lateral View)', 'Y (mm)', 'Z (mm)', 0)
    ]

    for ax, (xi, yi, title, xlabel, ylabel, slice_axis) in zip(axes, views):
        # Draw brain outline if available
        if brain_outline is not None:
            brain_mask, affine = brain_outline
            # Get middle slice
            slice_idx = brain_mask.shape[slice_axis] // 2
            try:
                x_outline, y_outline = get_brain_contour_coords(
                    brain_mask, affine, slice_idx, slice_axis
                )
                if len(x_outline) > 0:
                    ax.scatter(x_outline, y_outline, s=1, c='lightgray', alpha=0.3, zorder=1)
            except Exception:
                pass

        # Plot sources colored by depth
        scatter = ax.scatter(
            source_coords_mm[:, xi], source_coords_mm[:, yi],
            s=15, c=colors, alpha=0.7, edgecolors='none', zorder=2
        )

        # Plot electrodes
        ax.scatter(
            electrode_coords_mm[:, xi], electrode_coords_mm[:, yi],
            s=80, c='red', marker='^', edgecolors='black', linewidth=1,
            label='Electrodes', zorder=3
        )

        ax.set_xlabel(xlabel, fontsize=11)
        ax.set_ylabel(ylabel, fontsize=11)
        ax.set_title(title, fontsize=12)
        ax.set_aspect('equal')
        ax.grid(True, alpha=0.3)

        # Set consistent axis limits based on electrode positions (with margin)
        x_range = electrode_coords_mm[:, xi].max() - electrode_coords_mm[:, xi].min()
        y_range = electrode_coords_mm[:, yi].max() - electrode_coords_mm[:, yi].min()
        margin = max(x_range, y_range) * 0.3

        ax.set_xlim(
            electrode_coords_mm[:, xi].min() - margin,
            electrode_coords_mm[:, xi].max() + margin
        )
        ax.set_ylim(
            min(source_coords_mm[:, yi].min(), electrode_coords_mm[:, yi].min()) - margin,
            max(source_coords_mm[:, yi].max(), electrode_coords_mm[:, yi].max()) + margin
        )

    # Add colorbar for depth
    cbar = fig.colorbar(
        plt.cm.ScalarMappable(cmap='viridis', norm=plt.Normalize(z_coords.min(), z_coords.max())),
        ax=axes, orientation='vertical', shrink=0.8, pad=0.02
    )
    cbar.set_label('Z depth (mm)', fontsize=10)

    # Add legend
    axes[0].legend(loc='upper left', fontsize=9)

    fig.suptitle(f'{preset_name}\n{n_sources} sources', fontsize=14, fontweight='bold')
    plt.tight_layout(rect=[0, 0, 0.95, 0.95])

    if output_path is not None:
        fig.savefig(output_path, dpi=150, bbox_inches='tight')

    return fig


def create_preset_comparison_figure(
    preset_data: Dict[str, Dict],
    output_path: Optional[Path] = None,
    brain_atlas_path: Optional[Path] = None
) -> plt.Figure:
    """
    Create a comparison figure showing all presets.

    Parameters
    ----------
    preset_data : dict
        Dictionary mapping preset names to dicts with 'source_coords_mm' and 'electrode_coords_mm'
    output_path : Path, optional
        Path to save figure
    brain_atlas_path : Path, optional
        Path to brain atlas for outline

    Returns
    -------
    fig : matplotlib.Figure
        Generated figure
    """
    n_presets = len(preset_data)
    n_cols = min(4, n_presets)
    n_rows = (n_presets + n_cols - 1) // n_cols

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(5 * n_cols, 5 * n_rows))
    if n_presets == 1:
        axes = np.array([[axes]])
    elif n_rows == 1:
        axes = axes.reshape(1, -1)

    # Load brain outline if available
    brain_outline = None
    if brain_atlas_path is not None and Path(brain_atlas_path).exists():
        try:
            brain_mask, affine = load_brain_outline(brain_atlas_path)
            brain_outline = (brain_mask, affine)
        except Exception:
            pass

    for idx, (preset_name, data) in enumerate(preset_data.items()):
        row = idx // n_cols
        col = idx % n_cols
        ax = axes[row, col]

        source_coords = data['source_coords_mm']
        elec_coords = data['electrode_coords_mm']
        n_sources = len(source_coords)

        # Color by depth
        z_coords = source_coords[:, 2]
        z_norm = (z_coords - z_coords.min()) / (z_coords.max() - z_coords.min() + 1e-6)
        colors = plt.cm.viridis(z_norm)

        # Draw brain outline (axial view, middle slice)
        if brain_outline is not None:
            brain_mask, affine = brain_outline
            slice_idx = brain_mask.shape[2] // 2
            try:
                x_outline, y_outline = get_brain_contour_coords(brain_mask, affine, slice_idx, 2)
                if len(x_outline) > 0:
                    ax.scatter(x_outline, y_outline, s=0.5, c='lightgray', alpha=0.3, zorder=1)
            except Exception:
                pass

        # Plot sources (axial view: X vs Y)
        ax.scatter(
            source_coords[:, 0], source_coords[:, 1],
            s=8, c=colors, alpha=0.7, edgecolors='none', zorder=2
        )

        # Plot electrodes
        ax.scatter(
            elec_coords[:, 0], elec_coords[:, 1],
            s=50, c='red', marker='^', edgecolors='black', linewidth=0.5, zorder=3
        )

        ax.set_title(f'{preset_name}\n({n_sources} sources)', fontsize=10)
        ax.set_aspect('equal')
        ax.grid(True, alpha=0.2)

        # Axis limits
        margin = 2.0
        ax.set_xlim(elec_coords[:, 0].min() - margin, elec_coords[:, 0].max() + margin)
        ax.set_ylim(
            min(source_coords[:, 1].min(), elec_coords[:, 1].min()) - margin,
            max(source_coords[:, 1].max(), elec_coords[:, 1].max()) + margin
        )

    # Hide unused axes
    for idx in range(n_presets, n_rows * n_cols):
        row = idx // n_cols
        col = idx % n_cols
        axes[row, col].axis('off')

    fig.suptitle('Source Coverage Comparison (Axial View)', fontsize=14, fontweight='bold')
    plt.tight_layout(rect=[0, 0, 1, 0.96])

    if output_path is not None:
        fig.savefig(output_path, dpi=150, bbox_inches='tight')

    return fig


def generate_coverage_figures_from_presets(
    presets: List[str],
    eeg_file: str,
    output_dir: Path,
    verbose: bool = True
) -> Dict[str, Path]:
    """
    Generate source coverage figures for multiple presets.

    Parameters
    ----------
    presets : list of str
        List of preset names
    eeg_file : str
        Path to EEG file (needed to get electrode positions)
    output_dir : Path
        Output directory for figures
    verbose : bool
        Print progress

    Returns
    -------
    figure_paths : dict
        Dictionary mapping preset names to figure paths
    """
    from source_localization import Pipeline
    import source_localization

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Get brain atlas path
    package_dir = Path(source_localization.__file__).parent
    brain_atlas_path = package_dir / 'data' / 'atlas' / 'Atlas_3DRois_brain.nii.gz'

    figure_paths = {}
    preset_data = {}

    for preset in presets:
        if verbose:
            print(f"Processing {preset}...")

        try:
            # Run pipeline to get source space and electrode positions
            pipeline = Pipeline.from_preset(preset)
            results = pipeline.run(
                eeg_file=eeg_file,
                output_dir=str(output_dir / f'_temp_{preset}')
            )

            source_coords_mm = results['source_space']['source_coords_mm']

            # Get electrode coordinates from info object
            info = results['electrode_registration']['info']
            n_electrodes = len(info['ch_names'])
            electrode_coords_mm = np.array([
                info['chs'][i]['loc'][:3] for i in range(n_electrodes)
            ]) * 1000  # Convert to mm

            # Store for comparison figure
            preset_data[preset] = {
                'source_coords_mm': source_coords_mm,
                'electrode_coords_mm': electrode_coords_mm
            }

            # Create individual figure
            fig_path = output_dir / f'{preset}_coverage.png'
            fig = create_source_coverage_figure(
                source_coords_mm, electrode_coords_mm, preset,
                brain_atlas_path=brain_atlas_path,
                output_path=fig_path
            )
            plt.close(fig)

            figure_paths[preset] = fig_path

            if verbose:
                print(f"  Saved: {fig_path}")
                print(f"  Sources: {len(source_coords_mm)}")

        except Exception as e:
            if verbose:
                print(f"  ERROR: {e}")
            continue

    # Create comparison figure
    if len(preset_data) > 1:
        comparison_path = output_dir / 'preset_comparison.png'
        fig = create_preset_comparison_figure(
            preset_data,
            output_path=comparison_path,
            brain_atlas_path=brain_atlas_path
        )
        plt.close(fig)
        figure_paths['_comparison'] = comparison_path

        if verbose:
            print(f"\nComparison figure saved: {comparison_path}")

    return figure_paths


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='Generate source coverage figures')
    parser.add_argument('--eeg', required=True, help='Path to EEG file')
    parser.add_argument('--output', '-o', default='./coverage_figures', help='Output directory')
    parser.add_argument('--presets', nargs='+', help='Presets to compare')
    parser.add_argument('--all-presets', action='store_true', help='Run all presets')

    args = parser.parse_args()

    if args.all_presets:
        presets = [
            'shell_ellipsoid', 'shell_sphere',
            'ellipsoid_surface', 'sphere_surface',
            'ellipsoid_cartesian', 'sphere_cartesian',
            'roi_based_ellipsoid', 'roi_based_sphere'
        ]
    elif args.presets:
        presets = args.presets
    else:
        presets = ['shell_ellipsoid', 'ellipsoid_surface', 'roi_based_ellipsoid']

    generate_coverage_figures_from_presets(
        presets=presets,
        eeg_file=args.eeg,
        output_dir=Path(args.output),
        verbose=True
    )
