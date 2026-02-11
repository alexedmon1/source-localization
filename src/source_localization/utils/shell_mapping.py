"""
Shell-to-MRI space mapping utilities for parametric analysis.

This module provides functions to map shell-based source estimates back to
MRI voxel space for visualization and atlas-based analysis.

Key functions:
- get_shell_metadata: Extract depth and shell index for each source
- sources_to_voxels: Map source coordinates to MRI voxel indices
- interpolate_to_volume: Interpolate source values to a 3D voxel grid
- create_parametric_nifti: Create a NIfTI image from source-level statistics

**Created:** 2026-01-28
"""

import numpy as np
import nibabel as nib
from scipy.interpolate import griddata
from scipy.ndimage import gaussian_filter
from pathlib import Path


def get_shell_metadata(source_coords_mm, bem_params, shell_scales=None):
    """
    Compute depth and shell assignment for each source.

    Parameters
    ----------
    source_coords_mm : ndarray, shape (n_sources, 3)
        Source coordinates in mm (RAS space)
    bem_params : dict
        BEM parameters with center_mm, semi_axes_mm (ellipsoid) or brain_radius_mm (sphere)
    shell_scales : list of float, optional
        Shell scale factors used during source space creation.
        If provided, assigns each source to the nearest shell.

    Returns
    -------
    metadata : dict
        - 'normalized_depth': ndarray (n_sources,) - Distance from center (0) to surface (1)
        - 'shell_index': ndarray (n_sources,) - Index of assigned shell (if shell_scales provided)
        - 'depth_mm': ndarray (n_sources,) - Absolute depth from surface in mm
        - 'depth_category': ndarray (n_sources,) - 'deep', 'middle', 'superficial'
    """
    center = np.array(bem_params['center_mm'])
    n_sources = len(source_coords_mm)

    # Center coordinates
    centered = source_coords_mm - center

    # Compute normalized distance (0=center, 1=surface)
    if 'semi_axes_mm' in bem_params:
        # Ellipsoid BEM
        semi_axes = np.array(bem_params['semi_axes_mm'])
        normalized_depth = np.sqrt(np.sum((centered / semi_axes) ** 2, axis=1))
        # Approximate depth from surface
        mean_radius = np.mean(semi_axes)
        depth_mm = mean_radius * (1 - normalized_depth)
    else:
        # Sphere BEM
        brain_radius = bem_params['brain_radius_mm']
        distances = np.linalg.norm(centered, axis=1)
        normalized_depth = distances / brain_radius
        depth_mm = brain_radius - distances

    # Assign to shells if shell_scales provided
    if shell_scales is not None:
        shell_scales = np.array(shell_scales)
        # Find closest shell for each source
        distances_to_shells = np.abs(normalized_depth[:, np.newaxis] - shell_scales[np.newaxis, :])
        shell_index = np.argmin(distances_to_shells, axis=1)
    else:
        shell_index = np.zeros(n_sources, dtype=int)

    # Categorize depth
    depth_category = np.empty(n_sources, dtype='U12')
    depth_category[normalized_depth < 0.4] = 'deep'
    depth_category[(normalized_depth >= 0.4) & (normalized_depth < 0.7)] = 'middle'
    depth_category[normalized_depth >= 0.7] = 'superficial'

    return {
        'normalized_depth': normalized_depth,
        'shell_index': shell_index,
        'depth_mm': depth_mm,
        'depth_category': depth_category,
    }


def sources_to_voxels(source_coords_mm, nii_img):
    """
    Map source coordinates to MRI voxel indices.

    Parameters
    ----------
    source_coords_mm : ndarray, shape (n_sources, 3)
        Source coordinates in mm (RAS space)
    nii_img : nibabel.Nifti1Image
        Reference NIfTI image (for affine transformation)

    Returns
    -------
    voxel_indices : ndarray, shape (n_sources, 3)
        Voxel indices (i, j, k) for each source
    valid_mask : ndarray, shape (n_sources,)
        Boolean mask indicating which sources fall within image bounds
    """
    from ..utils.atlas import get_true_affine

    # Get corrected affine (10× fix)
    affine = get_true_affine(nii_img)

    # Invert affine: mm → voxel
    affine_inv = np.linalg.inv(affine)

    # Apply inverse affine
    coords_homogeneous = np.column_stack([source_coords_mm, np.ones(len(source_coords_mm))])
    voxel_coords = (affine_inv @ coords_homogeneous.T).T[:, :3]

    # Round to nearest voxel
    voxel_indices = np.round(voxel_coords).astype(int)

    # Check bounds
    shape = nii_img.shape[:3]
    valid_mask = (
        (voxel_indices[:, 0] >= 0) & (voxel_indices[:, 0] < shape[0]) &
        (voxel_indices[:, 1] >= 0) & (voxel_indices[:, 1] < shape[1]) &
        (voxel_indices[:, 2] >= 0) & (voxel_indices[:, 2] < shape[2])
    )

    return voxel_indices, valid_mask


def interpolate_to_volume(
    source_coords_mm,
    source_values,
    nii_template,
    method='linear',
    fill_value=0.0,
    smooth_fwhm_mm=None
):
    """
    Interpolate source-level values to a 3D voxel grid.

    Parameters
    ----------
    source_coords_mm : ndarray, shape (n_sources, 3)
        Source coordinates in mm
    source_values : ndarray, shape (n_sources,)
        Values at each source (e.g., t-statistic, power)
    nii_template : nibabel.Nifti1Image
        Template NIfTI defining output grid
    method : str
        Interpolation method: 'nearest', 'linear', or 'cubic'
    fill_value : float
        Value for voxels outside source convex hull
    smooth_fwhm_mm : float, optional
        FWHM of Gaussian smoothing in mm (applied after interpolation)

    Returns
    -------
    volume : ndarray, shape (nx, ny, nz)
        3D volume with interpolated values
    """
    from ..utils.atlas import get_true_affine

    # Get output grid
    affine = get_true_affine(nii_template)
    shape = nii_template.shape[:3]

    # Create voxel coordinate grid
    i_range = np.arange(shape[0])
    j_range = np.arange(shape[1])
    k_range = np.arange(shape[2])
    ii, jj, kk = np.meshgrid(i_range, j_range, k_range, indexing='ij')

    # Convert voxels to mm
    voxel_coords = np.column_stack([ii.ravel(), jj.ravel(), kk.ravel()])
    mm_coords = nib.affines.apply_affine(affine, voxel_coords)

    # Interpolate
    interpolated = griddata(
        source_coords_mm,
        source_values,
        mm_coords,
        method=method,
        fill_value=fill_value
    )

    # Reshape to 3D
    volume = interpolated.reshape(shape)

    # Optional smoothing
    if smooth_fwhm_mm is not None and smooth_fwhm_mm > 0:
        voxel_size = np.abs(np.diag(affine[:3, :3]))
        sigma_voxels = (smooth_fwhm_mm / 2.355) / voxel_size  # FWHM to sigma
        volume = gaussian_filter(volume, sigma=sigma_voxels)

    return volume


def create_parametric_nifti(
    source_coords_mm,
    source_values,
    nii_template,
    output_path=None,
    method='linear',
    smooth_fwhm_mm=1.0,
    description='Parametric map from shell-based source localization'
):
    """
    Create a NIfTI image from source-level statistics.

    Parameters
    ----------
    source_coords_mm : ndarray, shape (n_sources, 3)
        Source coordinates in mm
    source_values : ndarray, shape (n_sources,) or (n_sources, n_timepoints)
        Statistical values at each source
    nii_template : nibabel.Nifti1Image or str or Path
        Template NIfTI for geometry
    output_path : str or Path, optional
        Path to save NIfTI file
    method : str
        Interpolation method
    smooth_fwhm_mm : float
        Smoothing FWHM in mm
    description : str
        Description for NIfTI header

    Returns
    -------
    nii_out : nibabel.Nifti1Image
        Parametric map as NIfTI
    """
    from ..utils.atlas import get_true_affine

    # Load template if path
    if isinstance(nii_template, (str, Path)):
        nii_template = nib.load(nii_template)

    # Handle multi-timepoint data
    if source_values.ndim == 1:
        volume = interpolate_to_volume(
            source_coords_mm, source_values, nii_template,
            method=method, smooth_fwhm_mm=smooth_fwhm_mm
        )
    else:
        # Stack timepoints
        n_timepoints = source_values.shape[1]
        shape = nii_template.shape[:3]
        volume = np.zeros((*shape, n_timepoints))
        for t in range(n_timepoints):
            volume[..., t] = interpolate_to_volume(
                source_coords_mm, source_values[:, t], nii_template,
                method=method, smooth_fwhm_mm=smooth_fwhm_mm
            )

    # Create NIfTI
    # Use the ORIGINAL affine (not corrected) so the output aligns with atlas
    affine_out = nii_template.affine.copy()

    nii_out = nib.Nifti1Image(volume.astype(np.float32), affine_out)
    nii_out.header['descrip'] = description[:80]  # Max 80 chars

    if output_path is not None:
        nib.save(nii_out, output_path)
        print(f"Saved parametric map: {output_path}")

    return nii_out


def assign_sources_to_rois(source_coords_mm, nii_labels, roi_mapping=None):
    """
    Assign each source to an atlas ROI based on voxel location.

    Parameters
    ----------
    source_coords_mm : ndarray, shape (n_sources, 3)
        Source coordinates in mm
    nii_labels : nibabel.Nifti1Image or str
        Atlas label volume
    roi_mapping : dict, optional
        Mapping from label ID to ROI name

    Returns
    -------
    roi_ids : ndarray, shape (n_sources,)
        ROI label ID for each source (0 = outside brain)
    roi_names : list
        ROI name for each source (if roi_mapping provided)
    """
    if isinstance(nii_labels, (str, Path)):
        nii_labels = nib.load(nii_labels)

    label_data = nii_labels.get_fdata()
    voxel_indices, valid_mask = sources_to_voxels(source_coords_mm, nii_labels)

    # Get label for each source
    roi_ids = np.zeros(len(source_coords_mm), dtype=int)
    roi_ids[valid_mask] = label_data[
        voxel_indices[valid_mask, 0],
        voxel_indices[valid_mask, 1],
        voxel_indices[valid_mask, 2]
    ].astype(int)

    # Map to names if provided
    if roi_mapping is not None:
        roi_names = []
        for roi_id in roi_ids:
            if str(roi_id) in roi_mapping:
                roi_names.append(roi_mapping[str(roi_id)].get('name', f'ROI_{roi_id}'))
            elif roi_id in roi_mapping:
                roi_names.append(roi_mapping[roi_id].get('name', f'ROI_{roi_id}'))
            else:
                roi_names.append(f'Unknown_{roi_id}')
        return roi_ids, roi_names

    return roi_ids, None


def create_depth_stratified_maps(
    source_coords_mm,
    source_values,
    bem_params,
    nii_template,
    output_dir=None,
    n_depth_bins=3
):
    """
    Create separate parametric maps for different depth levels.

    Parameters
    ----------
    source_coords_mm : ndarray, shape (n_sources, 3)
        Source coordinates
    source_values : ndarray, shape (n_sources,)
        Values at each source
    bem_params : dict
        BEM parameters for depth computation
    nii_template : nibabel.Nifti1Image
        Template for output geometry
    output_dir : str or Path, optional
        Directory to save depth-stratified maps
    n_depth_bins : int
        Number of depth levels (default: 3 = deep/middle/superficial)

    Returns
    -------
    depth_maps : dict
        Dictionary mapping depth labels to NIfTI images
    """
    # Get depth metadata
    metadata = get_shell_metadata(source_coords_mm, bem_params)
    normalized_depth = metadata['normalized_depth']

    # Create depth bins
    depth_bins = np.linspace(0, 1, n_depth_bins + 1)
    depth_labels = ['deep', 'middle', 'superficial'][:n_depth_bins]

    depth_maps = {}
    for i, label in enumerate(depth_labels):
        # Select sources in this depth range
        mask = (normalized_depth >= depth_bins[i]) & (normalized_depth < depth_bins[i + 1])
        if i == n_depth_bins - 1:  # Include upper bound for last bin
            mask = (normalized_depth >= depth_bins[i]) & (normalized_depth <= depth_bins[i + 1])

        if np.sum(mask) < 4:
            print(f"Warning: Only {np.sum(mask)} sources in {label} depth bin, skipping")
            continue

        # Create map for this depth level
        output_path = None
        if output_dir is not None:
            output_dir = Path(output_dir)
            output_dir.mkdir(parents=True, exist_ok=True)
            output_path = output_dir / f'parametric_map_{label}.nii.gz'

        nii = create_parametric_nifti(
            source_coords_mm[mask],
            source_values[mask],
            nii_template,
            output_path=output_path,
            description=f'{label.capitalize()} sources (depth {depth_bins[i]:.1f}-{depth_bins[i+1]:.1f})'
        )
        depth_maps[label] = nii

    return depth_maps
