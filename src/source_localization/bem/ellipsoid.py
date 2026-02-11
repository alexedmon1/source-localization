"""Ellipsoidal BEM model implementation."""

import mne
import numpy as np
import nibabel as nib
from pathlib import Path


def fit_ellipsoid_to_brain(brain_coords_mm, method='axis_aligned', margin=1.23,
                           center_offset_mm=None, axis_extensions_mm=None):
    """
    Fit ellipsoid to brain surface.

    Parameters
    ----------
    brain_coords_mm : ndarray, shape (n_voxels, 3)
        Brain voxel coordinates in mm
    method : str
        Fitting method: 'axis_aligned' (default) or 'pca'
        - 'axis_aligned': Ellipsoid aligned with RAS coordinates (avoids axis swaps)
        - 'pca': Principal Component Analysis (may swap axes)
    margin : float
        Expansion factor for semi-axes (default 1.23 = 23% margin, matches adv_test)
    center_offset_mm : array-like, shape (3,), optional
        Offset to apply to ellipsoid center [x, y, z] in mm.
        Positive Y shifts anteriorly (towards nose/olfactory bulb).
    axis_extensions_mm : array-like, shape (3,), optional
        Additional extension to add to each semi-axis [x, y, z] in mm.
        Applied AFTER margin scaling. Use to extend coverage in specific directions.

    Returns
    -------
    center : ndarray, shape (3,)
        Ellipsoid center in mm
    semi_axes : ndarray, shape (3,)
        Semi-axes lengths in mm
    rotation : ndarray, shape (3, 3)
        Rotation matrix (identity for axis_aligned, eigenvectors for pca)
    """
    if method == 'axis_aligned':
        center, semi_axes, rotation = _fit_ellipsoid_axis_aligned(brain_coords_mm, margin)
    elif method == 'pca':
        center, semi_axes, rotation = _fit_ellipsoid_pca(brain_coords_mm, margin)
    else:
        raise ValueError(f"Unknown method: {method}. Use 'axis_aligned' or 'pca'")

    # Apply center offset if provided
    if center_offset_mm is not None:
        center = center + np.asarray(center_offset_mm)

    # Apply axis extensions if provided
    if axis_extensions_mm is not None:
        semi_axes = semi_axes + np.asarray(axis_extensions_mm)

    return center, semi_axes, rotation


def _fit_ellipsoid_axis_aligned(brain_coords_mm, margin=1.23):
    """
    Fit AXIS-ALIGNED ellipsoid to point cloud (no rotation).

    This creates an ellipsoid aligned with RAS coordinates, which avoids
    the axis swap issue from PCA-based fitting.

    Parameters
    ----------
    brain_coords_mm : ndarray, shape (n_points, 3)
        3D coordinates in mm (RAS coordinate system)
    margin : float
        Expansion factor for semi-axes (default 1.23 = 23% margin)

    Returns
    -------
    center : ndarray, shape (3,)
        Ellipsoid center (x0, y0, z0)
    semi_axes : ndarray, shape (3,)
        Semi-axes lengths (a, b, c) in X, Y, Z order
    rotation : ndarray, shape (3, 3)
        Identity matrix (no rotation, axis-aligned)
    """
    # Center the data
    center = brain_coords_mm.mean(axis=0)
    centered = brain_coords_mm - center

    # Compute standard deviation along each RAS axis
    stds = centered.std(axis=0)

    # Semi-axes are 2× std dev (95% coverage) with margin
    semi_axes = margin * 2 * stds

    # No rotation - identity matrix
    rotation = np.eye(3)

    return center, semi_axes, rotation


def _fit_ellipsoid_pca(brain_coords_mm, margin=1.23):
    """
    Fit ellipsoid to brain surface using PCA.

    Uses Principal Component Analysis to find the ellipsoid that best fits
    the brain voxels. The ellipsoid is defined by its center, semi-axes
    lengths, and rotation matrix.

    WARNING: This can result in axis swaps that don't match brain geometry!
    Consider using _fit_ellipsoid_axis_aligned() instead.

    Parameters
    ----------
    brain_coords_mm : ndarray, shape (n_voxels, 3)
        Brain voxel coordinates in mm
    margin : float
        Expansion factor for semi-axes (default 1.23)

    Returns
    -------
    center : ndarray, shape (3,)
        Ellipsoid center in mm
    semi_axes : ndarray, shape (3,)
        Semi-axes lengths in mm (along principal components)
    rotation : ndarray, shape (3, 3)
        Rotation matrix (eigenvectors as columns)
    """
    # Center the data
    center = brain_coords_mm.mean(axis=0)
    centered = brain_coords_mm - center

    # PCA to find principal axes
    cov = np.cov(centered.T)
    eigenvalues, eigenvectors = np.linalg.eig(cov)

    # Sort by eigenvalue magnitude (largest to smallest)
    idx = eigenvalues.argsort()[::-1]
    eigenvalues = eigenvalues[idx]
    eigenvectors = eigenvectors[:, idx]

    # Ensure right-handed coordinate system
    if np.linalg.det(eigenvectors) < 0:
        eigenvectors[:, 2] *= -1

    # Compute standard deviation along each principal axis
    rotated = centered @ eigenvectors
    stds = rotated.std(axis=0)

    # Semi-axes are 2× std dev with margin
    semi_axes = margin * 2 * stds

    return center, semi_axes, eigenvectors


def _create_icosphere(subdivisions=3):
    """
    Create an icosphere (subdivided icosahedron) with unit radius.

    Parameters
    ----------
    subdivisions : int
        Number of subdivision iterations (0 = ico, 1 = 80 tri, 2 = 320 tri, 3 = 1280 tri)

    Returns
    -------
    vertices : ndarray, shape (n_vertices, 3)
        Vertex coordinates on unit sphere
    triangles : ndarray, shape (n_triangles, 3)
        Triangle vertex indices
    """
    # Icosahedron vertices (12 vertices)
    t = (1.0 + np.sqrt(5.0)) / 2.0  # Golden ratio

    vertices = np.array([
        [-1,  t,  0], [ 1,  t,  0], [-1, -t,  0], [ 1, -t,  0],
        [ 0, -1,  t], [ 0,  1,  t], [ 0, -1, -t], [ 0,  1, -t],
        [ t,  0, -1], [ t,  0,  1], [-t,  0, -1], [-t,  0,  1]
    ], dtype=float)

    # Normalize to unit sphere
    vertices = vertices / np.linalg.norm(vertices, axis=1, keepdims=True)

    # Icosahedron faces (20 triangles)
    triangles = np.array([
        [0, 11, 5], [0, 5, 1], [0, 1, 7], [0, 7, 10], [0, 10, 11],
        [1, 5, 9], [5, 11, 4], [11, 10, 2], [10, 7, 6], [7, 1, 8],
        [3, 9, 4], [3, 4, 2], [3, 2, 6], [3, 6, 8], [3, 8, 9],
        [4, 9, 5], [2, 4, 11], [6, 2, 10], [8, 6, 7], [9, 8, 1]
    ], dtype=int)

    # Subdivide
    for _ in range(subdivisions):
        vertices, triangles = _subdivide_mesh(vertices, triangles)

    return vertices, triangles


def _subdivide_mesh(vertices, triangles):
    """
    Subdivide each triangle into 4 smaller triangles.

    Parameters
    ----------
    vertices : ndarray, shape (n_vertices, 3)
        Current vertices
    triangles : ndarray, shape (n_triangles, 3)
        Current triangles

    Returns
    -------
    new_vertices : ndarray
        Vertices after subdivision
    new_triangles : ndarray
        Triangles after subdivision
    """
    vertex_list = list(vertices)
    vertex_to_idx = {tuple(v): i for i, v in enumerate(vertices)}

    new_tri_indices = []
    for tri in triangles:
        v1, v2, v3 = tri

        # Midpoints
        m1 = (vertices[v1] + vertices[v2]) / 2.0
        m1 = m1 / np.linalg.norm(m1)
        m2 = (vertices[v2] + vertices[v3]) / 2.0
        m2 = m2 / np.linalg.norm(m2)
        m3 = (vertices[v3] + vertices[v1]) / 2.0
        m3 = m3 / np.linalg.norm(m3)

        # Get/create indices
        for m in [m1, m2, m3]:
            m_tuple = tuple(m)
            if m_tuple not in vertex_to_idx:
                vertex_to_idx[m_tuple] = len(vertex_list)
                vertex_list.append(m)

        a_idx = vertex_to_idx[tuple(m1)]
        b_idx = vertex_to_idx[tuple(m2)]
        c_idx = vertex_to_idx[tuple(m3)]

        # Four new triangles
        new_tri_indices.append([v1, a_idx, c_idx])
        new_tri_indices.append([v2, b_idx, a_idx])
        new_tri_indices.append([v3, c_idx, b_idx])
        new_tri_indices.append([a_idx, b_idx, c_idx])

    return np.array(vertex_list), np.array(new_tri_indices)


def create_ellipsoid_surfaces_from_axes(center_mm, layer_semi_axes_list, rotation, conductivities, n_vertices=642):
    """
    Create triangulated ellipsoid surfaces for BEM from explicit semi-axes.

    Creates nested ellipsoid surfaces (brain, skull, scalp) with specified
    semi-axes for each layer.

    Parameters
    ----------
    center_mm : ndarray, shape (3,)
        Ellipsoid center in mm
    layer_semi_axes_list : list of ndarray
        List of semi-axes for each layer (inner to outer)
        Each element is shape (3,) array of semi-axes in mm
    rotation : ndarray, shape (3, 3)
        Rotation matrix
    conductivities : list of float
        Conductivity values for each layer in S/m
    n_vertices : int
        Number of vertices per surface (default: 642, ico-3 subdivision)

    Returns
    -------
    surfaces : list of dict
        List of surface dictionaries in MNE format
    """
    surfaces = []

    # Determine subdivision level based on target vertices
    # ico-3 = 642 vertices (default), ico-4 = 2562, ico-5 = 10242
    if n_vertices <= 642:
        n_subdivisions = 3
    elif n_vertices <= 2562:
        n_subdivisions = 4
    else:
        n_subdivisions = 5

    # Create unit sphere with icosahedron subdivision
    rr, tris = _create_icosphere(n_subdivisions)

    # MNE surface IDs: brain=4, skull=3, scalp=1 (inner to outer)
    surface_ids = [4, 3, 1]

    for layer_idx, layer_semi_axes in enumerate(layer_semi_axes_list):
        # Use specified semi-axes for this layer

        # Scale by semi-axes to create ellipsoid
        rr_scaled = rr * layer_semi_axes

        # Rotate to align with brain orientation
        rr_rotated = rr_scaled @ rotation.T

        # Translate to center
        rr_final = rr_rotated + center_mm

        # Convert to meters (MNE uses meters)
        rr_m = rr_final / 1000.0

        # Compute normals (pointing outward from ellipsoid)
        nn = np.zeros_like(rr_m)
        for i, vertex in enumerate(rr_m):
            # Normal = gradient of ellipsoid equation at this point
            # For ellipsoid: (x/a)^2 + (y/b)^2 + (z/c)^2 = 1
            # Gradient = [2x/a^2, 2y/b^2, 2z/c^2]

            # Transform to ellipsoid coordinate system
            centered = (vertex * 1000 - center_mm) @ rotation

            # Compute gradient in ellipsoid space
            grad = 2 * centered / (layer_semi_axes ** 2)

            # Transform back to original space and normalize
            grad_rotated = grad @ rotation.T
            nn[i] = grad_rotated / np.linalg.norm(grad_rotated)

        # Create surface dictionary (MNE format)
        surface = {
            'rr': rr_m,
            'tris': tris,
            'nn': nn,
            'sigma': conductivities[layer_idx],
            'coord_frame': mne.io.constants.FIFF.FIFFV_COORD_MRI,
            'id': surface_ids[layer_idx],
            'np': len(rr_m),
            'ntri': len(tris)
        }
        surfaces.append(surface)

    return surfaces


def create_bem(config, previous_outputs):
    """
    Create ellipsoidal BEM model from brain volume.

    Fits an ellipsoid to the brain surface using PCA, then creates
    multi-layer nested ellipsoid surfaces for the BEM model.

    Parameters
    ----------
    config : Config
        Pipeline configuration
    previous_outputs : dict
        Previous pipeline outputs (not used for ellipsoid BEM)

    Returns
    -------
    bem_model : dict
        Ellipsoidal BEM model containing:
        - 'surfs': List of triangulated surfaces
        - 'sigma': Conductivities
        - 'is_sphere': False
        - 'coord_frame': MRI coordinate frame
    bem_params : dict
        Dictionary containing:
        - 'center_mm': Ellipsoid center in mm
        - 'semi_axes_mm': Semi-axes lengths in mm
        - 'rotation_matrix': Rotation matrix (3x3)
        - 'radii_ratios': Ratios for each layer
        - 'conductivities': Conductivities for each layer
        - 'n_layers': Number of layers
    """
    from ..utils.atlas import get_true_affine

    # Get package directory to locate brain volume
    # CRITICAL: Must use skull-stripped brain mask for ellipsoid fitting
    # NOT the full atlas which includes skull/exterior
    package_dir = Path(__file__).parent.parent
    brain_mask_file = package_dir / "data/atlas/Atlas_3DRois_brain.nii.gz"

    print(f"  Loading skull-stripped brain: {brain_mask_file.name}")
    brain_img = nib.load(brain_mask_file)
    brain_data = brain_img.get_fdata()
    affine_corrected = get_true_affine(brain_img)

    # Extract brain surface voxels (skull-stripped brain only)
    brain_mask = brain_data > 0
    brain_voxels = np.argwhere(brain_mask)
    brain_coords_mm = nib.affines.apply_affine(affine_corrected, brain_voxels)

    print(f"  Fitting ellipsoid to {len(brain_coords_mm):,} skull-stripped brain voxels...")

    # Get BEM parameters from config
    conductivities = config['bem']['ellipsoid']['conductivities']
    fitting_method = config['bem']['ellipsoid'].get('ellipsoid_method', 'axis_aligned')
    fitting_margin = config['bem']['ellipsoid'].get('ellipsoid_margin', 1.23)

    # Optional center offset and axis extensions (for olfactory bulb coverage)
    center_offset_mm = config['bem']['ellipsoid'].get('center_offset_mm', None)
    axis_extensions_mm = config['bem']['ellipsoid'].get('axis_extensions_mm', None)

    # Tissue thicknesses (mm) - matches adv_test approach
    skull_thickness_mm = config['bem']['ellipsoid'].get('skull_thickness_mm', 0.5)
    scalp_thickness_mm = config['bem']['ellipsoid'].get('scalp_thickness_mm', 0.3)

    # Fit ellipsoid
    center_mm, semi_axes_mm, rotation = fit_ellipsoid_to_brain(
        brain_coords_mm,
        method=fitting_method,
        margin=fitting_margin,
        center_offset_mm=center_offset_mm,
        axis_extensions_mm=axis_extensions_mm,
    )

    if center_offset_mm is not None:
        print(f"  Center offset applied: {center_offset_mm} mm")
    if axis_extensions_mm is not None:
        print(f"  Axis extensions applied: {axis_extensions_mm} mm")

    # Apply scale factor if provided (for brain_size validation tests)
    scale_factor = config['bem']['ellipsoid'].get('scale_factor', 1.0)
    if scale_factor != 1.0:
        print(f"  Applying {scale_factor}x scale factor to BEM geometry...")
        center_mm = center_mm * scale_factor
        semi_axes_mm = semi_axes_mm * scale_factor
        skull_thickness_mm = skull_thickness_mm * scale_factor
        scalp_thickness_mm = scalp_thickness_mm * scale_factor

    print(f"  Ellipsoid center: [{center_mm[0]:.3f}, {center_mm[1]:.3f}, {center_mm[2]:.3f}] mm")
    print(f"  Ellipsoid semi-axes: [{semi_axes_mm[0]:.3f}, {semi_axes_mm[1]:.3f}, {semi_axes_mm[2]:.3f}] mm")
    print(f"  Fitting method: {fitting_method}, margin: {fitting_margin}")

    # Check if rotation is significant
    if np.allclose(rotation, np.eye(3), atol=1e-2):
        print(f"  Rotation: axis-aligned (identity)")
    else:
        print(f"  Rotation: non-axis-aligned")

    n_layers = len(conductivities)

    # CRITICAL: Fitted ellipsoid IS the brain layer
    # Add skull and scalp OUTSIDE by expanding with thickness
    # This matches adv_test/bem_ellipsoid approach
    brain_semi_axes_mm = semi_axes_mm.copy()

    # Compute average radius for thickness-to-expansion conversion
    avg_brain_radius = brain_semi_axes_mm.mean()

    # Skull expansion: add thickness to brain
    skull_expansion = 1.0 + (skull_thickness_mm / avg_brain_radius)
    skull_semi_axes_mm = brain_semi_axes_mm * skull_expansion

    # Scalp expansion: add thickness to skull
    avg_skull_radius = skull_semi_axes_mm.mean()
    scalp_expansion = 1.0 + (scalp_thickness_mm / avg_skull_radius)
    scalp_semi_axes_mm = skull_semi_axes_mm * scalp_expansion

    # Create list of semi-axes for each layer (inner to outer)
    layer_semi_axes = [brain_semi_axes_mm, skull_semi_axes_mm, scalp_semi_axes_mm]

    print(f"  Tissue thicknesses: skull={skull_thickness_mm}mm, scalp={scalp_thickness_mm}mm")
    print(f"  Creating {n_layers}-layer ellipsoidal BEM:")

    # Create triangulated surfaces for each layer
    surfaces = create_ellipsoid_surfaces_from_axes(
        center_mm, layer_semi_axes, rotation, conductivities
    )

    layer_names = ['brain', 'skull', 'scalp']
    for i, (surf, sigma, axes) in enumerate(zip(surfaces, conductivities, layer_semi_axes)):
        layer_name = layer_names[i] if i < 3 else f'layer{i+1}'
        print(f"    {layer_name}: {surf['np']} vertices, {surf['ntri']} triangles, "
              f"semi-axes=[{axes[0]:.2f}, {axes[1]:.2f}, {axes[2]:.2f}] mm, "
              f"σ={sigma} S/m")

    # Create BEM model structure
    bem_model = {
        'surfs': surfaces,
        'sigma': conductivities,
        'is_sphere': False,
        'coord_frame': mne.io.constants.FIFF.FIFFV_COORD_MRI
    }

    print(f"  ✓ Created {n_layers}-layer ellipsoidal BEM model")

    # Store parameters for caching and source space construction
    # CRITICAL: Store BRAIN semi-axes (not scalp) for source constraint
    bem_params = {
        'center_mm': center_mm.tolist(),
        'semi_axes_mm': brain_semi_axes_mm.tolist(),  # Brain layer, not scaled
        'rotation_matrix': rotation.tolist(),
        'skull_thickness_mm': skull_thickness_mm,
        'scalp_thickness_mm': scalp_thickness_mm,
        'conductivities': conductivities,
        'n_layers': n_layers,
        'scale_factor': scale_factor
    }

    return bem_model, bem_params
