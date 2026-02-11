"""Shell-based source space implementation.

Creates sources on concentric shells that match the BEM geometry (spherical or
ellipsoidal). This approach addresses the ill-conditioning problem of Cartesian
volumetric grids by:

1. Placing sources at explicit depth levels (shells)
2. Ensuring uniform electrode distance within each shell
3. Reducing redundancy between sources (different shells = different depths)
4. Allowing adaptive density (more sources near surface where resolution is better)

**Created:** 2026-01-28
**Last Updated:** 2026-01-28
"""

import mne
import numpy as np


def create_source_space(config, previous_outputs):
    """
    Create shell-based source space with concentric shells matching BEM geometry.

    For sphere BEM: Creates concentric spherical shells
    For ellipsoid BEM: Creates concentric ellipsoidal shells

    Parameters
    ----------
    config : Config
        Pipeline configuration with source_space.shell_based section
    previous_outputs : dict
        Previous pipeline outputs containing 'bem_params'

    Returns
    -------
    src : mne.SourceSpaces
        Shell-based source space
    source_coords_mm : ndarray, shape (n_sources, 3)
        Source coordinates in mm
    n_sources : int
        Number of sources
    """
    # Get BEM parameters and configuration
    bem_params = previous_outputs['bem_params']
    bem_type = config['pipeline']['bem_type']

    # Get shell configuration (support both 'shell' and 'shell_based' for compatibility)
    shell_config = config['source_space'].get('shell', config['source_space'].get('shell_based', {}))
    n_shells = shell_config.get('n_shells', 3)
    shell_scales = shell_config.get('shell_scales', None)
    min_points = shell_config.get('min_points_per_shell', 20)
    max_points = shell_config.get('max_points_per_shell', 100)
    scale_by_area = shell_config.get('scale_by_area', True)
    distribution = shell_config.get('distribution', 'fibonacci')  # 'fibonacci' or 'latlon'

    print(f"  Creating shell-based source space:")
    print(f"    BEM type: {bem_type}")
    print(f"    Number of shells: {n_shells}")
    print(f"    Distribution: {distribution}")

    # Auto-compute shell scales if not provided (uniform spacing from 0.4 to 0.8)
    # Outer scale 0.8 matches surface inset_factor for consistent electrode distance
    if shell_scales is None:
        shell_scales = np.linspace(0.4, 0.8, n_shells).tolist()

    print(f"    Shell scales: {[f'{s:.2f}' for s in shell_scales]}")

    # Compute points per shell
    if scale_by_area:
        # Surface area scales with scale^2, allocate points proportionally
        max_scale_sq = max(shell_scales) ** 2
        points_per_shell = [
            int(min_points + (max_points - min_points) * (s ** 2 / max_scale_sq))
            for s in shell_scales
        ]
        print(f"    Points per shell (area-scaled): {points_per_shell}")
    else:
        points_per_shell = [max_points] * n_shells
        print(f"    Points per shell (uniform): {points_per_shell}")

    # Generate sources for each shell
    all_vertices = []
    all_normals = []

    # Offset angle for odd shells to prevent radial alignment
    # For fibonacci: rotate 30° around Z-axis
    # For latlon: offset longitude by 30°
    odd_shell_offset = np.pi / 6.0  # 30 degrees

    for shell_idx, (scale, n_pts) in enumerate(zip(shell_scales, points_per_shell)):
        # Generate unit sphere points using selected distribution method
        if distribution == 'latlon':
            # Latitude-longitude grid (onion pattern)
            phi_offset = odd_shell_offset if (shell_idx % 2 == 1) else 0.0
            unit_pts = latlon_sphere(n_pts, phi_offset=phi_offset)
        else:
            # Fibonacci spiral (default)
            unit_pts = fibonacci_sphere(n_pts)

            # For odd shells, rotate points around Z-axis to break radial alignment
            if shell_idx % 2 == 1:
                cos_r = np.cos(odd_shell_offset)
                sin_r = np.sin(odd_shell_offset)
                x_rot = unit_pts[:, 0] * cos_r - unit_pts[:, 2] * sin_r
                z_rot = unit_pts[:, 0] * sin_r + unit_pts[:, 2] * cos_r
                unit_pts = np.column_stack([x_rot, unit_pts[:, 1], z_rot])

        if bem_type == 'sphere':
            vertices_mm, normals = _sphere_shell(unit_pts, scale, bem_params)
        elif bem_type == 'ellipsoid':
            vertices_mm, normals = _ellipsoid_shell(unit_pts, scale, bem_params)
        else:
            raise ValueError(f"Unknown BEM type: {bem_type}")

        all_vertices.append(vertices_mm)
        all_normals.append(normals)

        offset_str = " (offset 30°)" if (shell_idx % 2 == 1) else ""
        print(f"    Shell {shell_idx + 1}: scale={scale:.2f}, "
              f"{len(vertices_mm)} sources{offset_str}")

    # Combine all shells
    vertices_mm = np.vstack(all_vertices)
    normals = np.vstack(all_normals)

    n_sources = len(vertices_mm)
    print(f"    Total sources: {n_sources}")

    # Optional: Filter to dorsal hemisphere only (Z > center)
    filter_dorsal = shell_config.get('filter_dorsal', False)

    if filter_dorsal:
        center_mm = np.array(bem_params['center_mm'])
        center_z = center_mm[2]
        dorsal_mask = vertices_mm[:, 2] > center_z

        n_before = len(vertices_mm)
        vertices_mm = vertices_mm[dorsal_mask]
        normals = normals[dorsal_mask]
        n_sources = len(vertices_mm)

        print(f"    Dorsal filtering (Z > {center_z:.3f} mm):")
        print(f"      Before: {n_before} sources")
        print(f"      After: {n_sources} sources ({100 * n_sources / n_before:.1f}%)")

    # Optional: Filter sources outside brain volume (using brain mask, not ROI labels)
    filter_exterior = shell_config.get('filter_exterior', True)  # Default True

    if filter_exterior:
        try:
            from pathlib import Path
            import nibabel as nib

            # Load brain mask (skull-stripped brain volume)
            package_dir = Path(__file__).parent.parent
            brain_mask_path = package_dir / 'data' / 'atlas' / 'Atlas_3DRois_brain.nii.gz'

            if brain_mask_path.exists():
                brain_img = nib.load(brain_mask_path)
                brain_data = brain_img.get_fdata()

                # Get corrected affine (10x scaling fix)
                affine = brain_img.affine.copy()
                affine[:3, :3] *= 0.1
                affine[:3, 3] *= 0.1

                # Inverse affine to convert mm to voxels
                inv_affine = np.linalg.inv(affine)

                # Map sources to voxels
                voxel_coords = nib.affines.apply_affine(inv_affine, vertices_mm)
                voxel_coords_int = np.round(voxel_coords).astype(int)

                # Check which sources are inside the brain mask
                n_before = len(vertices_mm)
                inside_brain_mask = np.zeros(n_before, dtype=bool)

                for i, vox in enumerate(voxel_coords_int):
                    # Check if voxel is within bounds and inside brain
                    if (0 <= vox[0] < brain_data.shape[0] and
                        0 <= vox[1] < brain_data.shape[1] and
                        0 <= vox[2] < brain_data.shape[2]):
                        if brain_data[vox[0], vox[1], vox[2]] > 0:
                            inside_brain_mask[i] = True

                # Filter to keep only sources inside brain
                vertices_mm = vertices_mm[inside_brain_mask]
                normals = normals[inside_brain_mask]
                n_sources = len(vertices_mm)

                n_exterior = n_before - n_sources
                if n_exterior > 0:
                    print(f"    Brain mask filtering:")
                    print(f"      Before: {n_before} sources")
                    print(f"      After: {n_sources} sources (removed {n_exterior} outside brain)")
        except Exception as e:
            print(f"    Warning: Could not filter exterior sources: {e}")

    # Convert to meters for MNE
    vertices_m = vertices_mm / 1000.0

    # Create MNE-compatible source space dictionary
    # Use 'vol' type since these are volumetric sources (not on a cortical surface)
    src_dict = {
        'rr': np.array(vertices_m, dtype=np.float64),  # Source positions (meters)
        'nn': np.array(normals, dtype=np.float64),  # Normal vectors (radial)
        'nuse': n_sources,
        'inuse': np.ones(n_sources, dtype=np.int32),
        'vertno': np.arange(n_sources, dtype=np.int32),
        'type': 'vol',  # Volumetric type (not surface)
        'coord_frame': mne.io.constants.FIFF.FIFFV_COORD_MRI,
        'id': 1,  # Source space ID
        'np': n_sources,
        'ntri': 0,  # No triangulation for shell-based
        'tris': np.array([], dtype=np.int32).reshape(0, 3),
        'nearest': None,
        'nearest_dist': None,
        'pinfo': None,
        'patch_inds': None,
        'dist': None,
        'dist_limit': None,
        'interpolator': None,
        # Required for MNE repr (volumetric source spaces need shape)
        'shape': (n_sources, 1, 1),  # Dummy shape for shell-based
        'mri_width': n_sources,
        'mri_height': 1,
        'mri_depth': 1,
    }

    # Wrap in MNE SourceSpaces object
    src = mne.SourceSpaces([src_dict])

    print(f"    ✓ Created shell-based source space with {n_sources:,} sources")

    return src, vertices_mm, n_sources


def latlon_sphere(n_points, phi_offset=0.0):
    """
    Generate points on a unit sphere using latitude-longitude grid.

    Creates an "onion layer" pattern with regular spacing in latitude and longitude.
    Points are placed at regular latitude intervals, with longitude points scaled
    by cos(latitude) to maintain roughly uniform spacing.

    Parameters
    ----------
    n_points : int
        Approximate number of points to generate
    phi_offset : float
        Angular offset in radians for longitude. Used to offset alternating
        shells so sources don't align radially.

    Returns
    -------
    points : ndarray, shape (n_points, 3)
        Points on the unit sphere
    """
    if n_points <= 0:
        return np.array([]).reshape(0, 3)

    if n_points == 1:
        return np.array([[0, 0, 1]])

    # Estimate number of latitude bands to get approximately n_points
    # For uniform spacing, n_lat * avg_n_lon ≈ n_points
    # avg_n_lon ≈ n_lat * pi/2 (average of cos over hemisphere)
    n_lat = int(np.sqrt(n_points * 2 / np.pi))
    n_lat = max(3, n_lat)  # At least 3 latitude bands

    points = []

    # Latitude from -90 to +90 degrees (poles to poles)
    for i in range(n_lat):
        # Latitude angle (avoid exact poles to prevent singularity)
        lat = np.pi * (i + 0.5) / n_lat - np.pi / 2  # -pi/2 to pi/2

        # Number of longitude points at this latitude (fewer near poles)
        n_lon = max(1, int(n_lat * np.cos(lat) * 2))

        for j in range(n_lon):
            # Longitude angle with offset
            lon = 2 * np.pi * j / n_lon + phi_offset

            # Convert to Cartesian
            x = np.cos(lat) * np.cos(lon)
            y = np.sin(lat)  # Y is "up" in our convention
            z = np.cos(lat) * np.sin(lon)

            points.append([x, y, z])

    return np.array(points)


def fibonacci_sphere(n_points, theta_offset=0.0):
    """
    Generate n_points approximately uniformly distributed on a unit sphere.

    Uses the Fibonacci spiral (golden angle) method, which produces a near-uniform
    distribution for any number of points.

    Parameters
    ----------
    n_points : int
        Number of points to generate
    theta_offset : float
        Angular offset in radians to rotate the spiral pattern. Used to offset
        alternating shells so that sources don't align radially.

    Returns
    -------
    points : ndarray, shape (n_points, 3)
        Points on the unit sphere

    References
    ----------
    Gonzalez (2010) "Measurement of Areas on a Sphere Using Fibonacci and
    Latitude-Longitude Lattices"
    """
    if n_points <= 0:
        return np.array([]).reshape(0, 3)

    if n_points == 1:
        return np.array([[0, 0, 1]])

    indices = np.arange(n_points)
    phi = np.pi * (3.0 - np.sqrt(5.0))  # Golden angle in radians (~2.4)

    # y goes from 1 to -1 (top to bottom of sphere)
    y = 1.0 - (indices / (n_points - 1)) * 2.0

    # Radius at each y level (distance from y-axis)
    radius = np.sqrt(1.0 - y ** 2)

    # Theta angle for each point (golden angle spiral) + offset
    theta = phi * indices + theta_offset

    # Convert to Cartesian coordinates
    x = np.cos(theta) * radius
    z = np.sin(theta) * radius

    return np.column_stack([x, y, z])


def _sphere_shell(unit_pts, scale, bem_params):
    """
    Transform unit sphere points to a spherical shell.

    Parameters
    ----------
    unit_pts : ndarray, shape (n_points, 3)
        Points on unit sphere
    scale : float
        Scale factor (0, 1] relative to brain radius
    bem_params : dict
        BEM parameters with 'center_mm' and 'brain_radius_mm'

    Returns
    -------
    vertices_mm : ndarray, shape (n_points, 3)
        Source positions in mm
    normals : ndarray, shape (n_points, 3)
        Normal vectors (pointing radially outward)
    """
    center_mm = np.array(bem_params['center_mm'])
    brain_radius_mm = bem_params['brain_radius_mm']

    # Scale to shell radius
    shell_radius_mm = brain_radius_mm * scale

    # Transform: scale by radius and translate to center
    vertices_mm = unit_pts * shell_radius_mm + center_mm

    # Normals are radial (same as unit sphere points)
    normals = unit_pts.copy()

    return vertices_mm, normals


def _ellipsoid_shell(unit_pts, scale, bem_params):
    """
    Transform unit sphere points to an ellipsoidal shell.

    Parameters
    ----------
    unit_pts : ndarray, shape (n_points, 3)
        Points on unit sphere
    scale : float
        Scale factor (0, 1] relative to ellipsoid semi-axes
    bem_params : dict
        BEM parameters with 'center_mm', 'semi_axes_mm', 'rotation_matrix'

    Returns
    -------
    vertices_mm : ndarray, shape (n_points, 3)
        Source positions in mm
    normals : ndarray, shape (n_points, 3)
        Normal vectors (perpendicular to ellipsoid surface)
    """
    center_mm = np.array(bem_params['center_mm'])
    semi_axes_mm = np.array(bem_params['semi_axes_mm'])
    rotation = np.array(bem_params['rotation_matrix'])

    # Scale semi-axes for this shell
    shell_semi_axes = semi_axes_mm * scale

    # Transform unit sphere to ellipsoid:
    # 1. Scale by semi-axes
    vertices_scaled = unit_pts * shell_semi_axes[np.newaxis, :]

    # 2. Rotate according to principal axes
    vertices_rotated = vertices_scaled @ rotation.T

    # 3. Translate to ellipsoid center
    vertices_mm = vertices_rotated + center_mm[np.newaxis, :]

    # Compute normals for ellipsoid surface
    # Normal at point on ellipsoid is gradient of F(x,y,z) = (x/a)^2 + (y/b)^2 + (z/c)^2 - 1
    # In local (unrotated) coordinates: n = [2x/a^2, 2y/b^2, 2z/c^2]
    # Simplified: n_local = point_local / semi_axes^2
    normals_local = unit_pts * shell_semi_axes[np.newaxis, :] / (shell_semi_axes[np.newaxis, :] ** 2)
    # Simplifies to: normals_local = unit_pts / shell_semi_axes

    # Normalize to unit length
    normals_local = normals_local / np.linalg.norm(normals_local, axis=1, keepdims=True)

    # Rotate normals to world coordinates
    normals = normals_local @ rotation.T

    return vertices_mm, normals
