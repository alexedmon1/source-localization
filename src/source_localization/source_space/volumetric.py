"""Volumetric source space implementation."""

import mne
import numpy as np
import nibabel as nib
from pathlib import Path


def create_source_space(config, previous_outputs):
    """
    Create volumetric source space constrained by BEM geometry and brain mask.

    Parameters
    ----------
    config : Config
        Pipeline configuration
    previous_outputs : dict
        Previous pipeline outputs containing 'bem_params' and 'info'

    Returns
    -------
    src : mne.SourceSpaces
        Volumetric source space
    source_coords_mm : ndarray, shape (n_sources, 3)
        Source coordinates in mm
    n_sources : int
        Number of sources
    """
    # Get BEM parameters and configuration
    bem_params = previous_outputs['bem_params']
    bem_type = config['pipeline']['bem_type']

    # Support both 'volumetric' and 'cartesian' config keys for backward compatibility
    # The source_type is 'cartesian' (modern name) but config key may be 'volumetric' (historical)
    source_config = config['source_space'].get('volumetric') or config['source_space'].get('cartesian') or {}

    spacing_mm = source_config.get('spacing_mm', 1.5)  # Default 1.5mm if not specified
    pos_mm = source_config.get('pos_mm', 0.0)
    use_brain_mask = source_config.get('use_brain_mask', True)
    apply_bem_constraint = source_config.get('apply_bem_constraint', True)
    # Inset factor: constrain sources to this fraction of brain radius/semi-axes
    # Default 0.80 matches surface/shell for consistent ~2mm electrode distance
    inset_factor = source_config.get('inset_factor', 0.80)

    # Handle auto-spacing calculation based on brain volume and channel count
    if spacing_mm == 'auto':
        info = previous_outputs.get('info')
        if info is None:
            raise ValueError("Cannot auto-calculate spacing: no electrode info available")

        # Get target sources/channel ratio from config (default: 7)
        target_sources_per_channel = source_config.get(
            'target_sources_per_channel', 7
        )
        max_sources_per_channel = source_config.get(
            'max_sources_per_channel', 15
        )

        spacing_mm = _calculate_spacing_from_brain_volume(
            config, info, target_sources_per_channel, max_sources_per_channel
        )
        print(f"  Auto-calculated spacing: {spacing_mm:.2f} mm")

    print(f"  Creating volumetric source space:")
    print(f"    Spacing: {spacing_mm} mm")
    print(f"    Inset factor: {inset_factor:.2f} (sources within {inset_factor*100:.0f}% of brain radius)")
    print(f"    Brain mask filtering: {'enabled' if use_brain_mask else 'disabled'}")
    print(f"    BEM constraint: {'enabled' if apply_bem_constraint else 'disabled (adv_test mode)'}")

    # Load brain volume and brain mask
    package_dir = Path(__file__).parent.parent
    brain_file = package_dir / config['inputs']['brain_volume']
    nii = nib.load(brain_file)
    brain_data = nii.get_fdata()

    # Apply 10× voxel size correction (see CLAUDE.md critical note)
    from ..utils.atlas import get_true_affine
    affine = get_true_affine(nii)

    # Create brain mask based on BEM type (matches adv_test approach)
    # Sphere: Use full atlas (Atlas_3DRois.nii) for cubic grid bounds
    # Ellipsoid: Use skull-stripped brain (Atlas_3DRois_brain.nii.gz) for ellipsoidal bounds
    if use_brain_mask:
        if bem_type == 'sphere':
            # Sphere BEM: use full atlas for more cubic grid bounds
            # Spherical constraint will cut this to a sphere
            brain_mask = brain_data > 0
            print(f"    Using full atlas for sphere (more cubic bounds): Atlas_3DRois.nii")
        else:
            # Ellipsoid/other BEM: use skull-stripped brain for ellipsoidal bounds
            brain_mask_file = package_dir / "data/atlas/Atlas_3DRois_brain.nii.gz"
            mask_nii = nib.load(brain_mask_file)
            brain_mask = mask_nii.get_fdata() > 0  # Binary brain mask
            print(f"    Using skull-stripped brain mask: Atlas_3DRois_brain.nii.gz")
    else:
        # If brain mask disabled, still need a mask for grid bounds
        brain_mask = brain_data > 0
        print(f"    Using full atlas as mask (includes skull/exterior)")

    # Get voxel size from affine
    voxel_size = np.abs(np.diag(affine[:3, :3]))
    print(f"    Voxel size: [{voxel_size[0]:.4f}, {voxel_size[1]:.4f}, {voxel_size[2]:.4f}] mm")

    # Calculate grid spacing in voxels (matches adv_test approach)
    grid_spacing_voxels = spacing_mm / voxel_size

    # Create grid in voxel coordinates
    shape = brain_mask.shape
    x_range = np.arange(0, shape[0], grid_spacing_voxels[0])
    y_range = np.arange(0, shape[1], grid_spacing_voxels[1])
    z_range = np.arange(0, shape[2], grid_spacing_voxels[2])

    xx, yy, zz = np.meshgrid(x_range, y_range, z_range, indexing='ij')
    grid_coords_voxel = np.vstack([xx.ravel(), yy.ravel(), zz.ravel()]).T

    print(f"    Initial grid: {len(grid_coords_voxel):,} points")

    # Filter grid to brain mask (both sphere and ellipsoid use brain mask)
    # Sphere: full atlas → spherical constraint cuts to sphere
    # Ellipsoid: skull-stripped brain → keeps entire brain
    grid_coords_int = np.round(grid_coords_voxel).astype(int)

    # Check bounds
    valid_mask = (
        (grid_coords_int[:, 0] >= 0) & (grid_coords_int[:, 0] < shape[0]) &
        (grid_coords_int[:, 1] >= 0) & (grid_coords_int[:, 1] < shape[1]) &
        (grid_coords_int[:, 2] >= 0) & (grid_coords_int[:, 2] < shape[2])
    )

    grid_coords_int_valid = grid_coords_int[valid_mask]

    # Check which are in brain mask
    brain_sources_mask = brain_mask[
        grid_coords_int_valid[:, 0],
        grid_coords_int_valid[:, 1],
        grid_coords_int_valid[:, 2]
    ]

    source_coords_voxel = grid_coords_voxel[valid_mask][brain_sources_mask]

    # Convert to RAS mm coordinates
    grid_coords_mm = nib.affines.apply_affine(affine, source_coords_voxel)

    print(f"    After brain mask filter: {len(grid_coords_mm):,} points")

    # Optionally apply BEM constraint (disabled in adv_test mode)
    if not apply_bem_constraint:
        # Skip BEM filtering - use brain mask only (matches adv_test)
        source_coords_mm = grid_coords_mm
        n_sources = len(source_coords_mm)
        print(f"    Skipping BEM constraint (adv_test mode)")
        print(f"    Final source count: {n_sources:,} sources")
    else:
        # Apply BEM constraint to keep only sources inside brain geometry
        center_mm = np.array(bem_params['center_mm'])

        if bem_type == 'sphere':
            # Sphere constraint: distance from center <= brain radius * inset_factor
            brain_radius_mm = bem_params['brain_radius_mm']
            effective_radius_mm = brain_radius_mm * inset_factor
            distances_from_center = np.linalg.norm(grid_coords_mm - center_mm, axis=1)
            bem_mask = distances_from_center <= effective_radius_mm

            print(f"    Sphere constraint: brain radius = {brain_radius_mm:.2f} mm, "
                  f"effective = {effective_radius_mm:.2f} mm (inset {inset_factor:.2f})")

        elif bem_type == 'ellipsoid':
            # Ellipsoid constraint: normalized distance <= inset_factor
            # CRITICAL: Use the ORIGINAL fitted semi-axes (which already include margin)
            # The fitted ellipsoid IS the brain boundary - do NOT scale by ratio!
            # Ratios are used only for BEM layer creation, not source constraint
            semi_axes_mm = np.array(bem_params['semi_axes_mm'])
            effective_semi_axes_mm = semi_axes_mm * inset_factor
            rotation = np.array(bem_params['rotation_matrix'])

            # Transform to ellipsoid coordinate system
            centered = grid_coords_mm - center_mm

            # Check if rotation is identity (axis-aligned) or requires transformation
            if np.allclose(rotation, np.eye(3)):
                # Axis-aligned ellipsoid - direct calculation
                ellipsoid_distances = np.sqrt(
                    (centered[:, 0] / semi_axes_mm[0])**2 +
                    (centered[:, 1] / semi_axes_mm[1])**2 +
                    (centered[:, 2] / semi_axes_mm[2])**2
                )
            else:
                # Rotated ellipsoid - transform to principal axes first
                rotated = centered @ rotation.T
                ellipsoid_distances = np.sqrt(
                    (rotated[:, 0] / semi_axes_mm[0])**2 +
                    (rotated[:, 1] / semi_axes_mm[1])**2 +
                    (rotated[:, 2] / semi_axes_mm[2])**2
                )

            # Apply inset: sources must be within inset_factor of brain boundary
            bem_mask = ellipsoid_distances <= inset_factor

            print(f"    Ellipsoid constraint: semi-axes = [{semi_axes_mm[0]:.2f}, "
                  f"{semi_axes_mm[1]:.2f}, {semi_axes_mm[2]:.2f}] mm")
            print(f"    Effective (inset {inset_factor:.2f}): [{effective_semi_axes_mm[0]:.2f}, "
                  f"{effective_semi_axes_mm[1]:.2f}, {effective_semi_axes_mm[2]:.2f}] mm")
            print(f"    Rotation: {'axis-aligned' if np.allclose(rotation, np.eye(3)) else 'rotated'}")

        else:
            raise ValueError(f"Unknown BEM type: {bem_type}")

        # Filter grid points to those inside BEM
        source_coords_mm = grid_coords_mm[bem_mask]
        n_sources = len(source_coords_mm)

        print(f"    After BEM constraint: {n_sources:,} sources")

    # Convert to meters for MNE (MNE uses meters internally)
    source_coords_m = source_coords_mm / 1000.0

    # Get brain volume shape for volumetric source space
    brain_shape = brain_data.shape

    # Create MNE SourceSpaces object
    # For volumetric sources, we create a single source space
    src = mne.SourceSpaces([{
        'rr': source_coords_m,  # Positions in meters
        'nn': np.zeros_like(source_coords_m),  # No normals for volumetric
        'inuse': np.ones(n_sources, dtype=int),  # All sources active
        'vertno': np.arange(n_sources),  # Vertex numbers
        'nuse': n_sources,
        'coord_frame': mne.io.constants.FIFF.FIFFV_COORD_MRI,
        'id': 1,  # Source space ID
        'type': 'vol',  # Volumetric source space
        'np': n_sources,  # Number of points
        'ntri': 0,  # No triangles for volumetric
        'tris': np.array([], dtype=np.int32).reshape(0, 3),  # Empty triangles
        'shape': brain_shape,  # Volume shape
        'mri_width': brain_shape[0],
        'mri_height': brain_shape[1],
        'mri_depth': brain_shape[2],
        'interpolator': None,
    }])

    return src, source_coords_mm, n_sources


def _calculate_spacing_from_brain_volume(config, info, target_sources_per_channel=7, max_sources_per_channel=15):
    """
    Calculate optimal source spacing based on brain volume and channel count.

    This auto-regulates source density to maintain a reasonable sources/channel
    ratio, reducing collinearity issues in the inverse solution.

    Parameters
    ----------
    config : dict
        Pipeline configuration (to access brain mask path)
    info : mne.Info
        MNE info object with electrode positions
    target_sources_per_channel : float
        Target ratio of sources to channels (default: 7)
    max_sources_per_channel : float
        Maximum allowed ratio (default: 15)

    Returns
    -------
    spacing_mm : float
        Optimal source spacing in mm

    Notes
    -----
    The relationship between spacing and source count is:
        n_sources ≈ brain_volume / spacing³

    Rearranging for spacing:
        spacing = (brain_volume / n_sources)^(1/3)

    For 30 channels and target_ratio=7:
        target_sources = 210
        For brain_volume ≈ 543 mm³:
        spacing = (543 / 210)^(1/3) ≈ 1.37 mm
    """
    from pathlib import Path
    from ..utils.atlas import get_true_affine

    # Get number of channels
    n_channels = len(info['ch_names'])

    # Calculate target source count
    target_sources = n_channels * target_sources_per_channel
    max_sources = n_channels * max_sources_per_channel

    # Load brain mask to calculate volume
    package_dir = Path(__file__).parent.parent
    brain_mask_file = package_dir / "data/atlas/Atlas_3DRois_brain.nii.gz"

    brain_img = nib.load(brain_mask_file)
    brain_data = brain_img.get_fdata()
    affine = get_true_affine(brain_img)

    # Calculate brain volume
    brain_mask = brain_data > 0
    n_brain_voxels = np.sum(brain_mask)
    voxel_size = np.abs(np.diag(affine[:3, :3]))
    brain_volume_mm3 = n_brain_voxels * np.prod(voxel_size)

    # Calculate optimal spacing: spacing = (volume / sources)^(1/3)
    optimal_spacing = (brain_volume_mm3 / target_sources) ** (1/3)

    # Apply bounds: min 1.0mm (too dense otherwise), max 3.0mm (too sparse)
    min_spacing = 1.0
    max_spacing = 3.0
    spacing_mm = np.clip(optimal_spacing, min_spacing, max_spacing)

    # Calculate expected sources with this spacing
    expected_sources = brain_volume_mm3 / (spacing_mm ** 3)
    actual_ratio = expected_sources / n_channels

    print(f"    Auto-spacing calculation:")
    print(f"      Brain volume: {brain_volume_mm3:.1f} mm³")
    print(f"      Channels: {n_channels}")
    print(f"      Target: {target_sources_per_channel} sources/channel ({target_sources} sources)")
    print(f"      Optimal spacing: {optimal_spacing:.2f} mm")
    if spacing_mm != optimal_spacing:
        print(f"      Clamped to: {spacing_mm:.2f} mm (bounds: {min_spacing}-{max_spacing} mm)")
    print(f"      Expected sources: ~{expected_sources:.0f} ({actual_ratio:.1f} sources/channel)")

    # Warn if we're above max ratio
    if actual_ratio > max_sources_per_channel:
        print(f"      ⚠️ Warning: {actual_ratio:.1f} sources/channel exceeds max ({max_sources_per_channel})")
        print(f"         Consider using surface or ROI-based source space for lower collinearity")

    return spacing_mm
