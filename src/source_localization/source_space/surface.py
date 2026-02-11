"""Surface source space implementation.

Creates surface sources using icosahedral subdivision on BEM geometry.
Matches adv_test approach for both sphere and ellipsoid BEM types.

**Created:** 2025-11-28
**Last Updated:** 2025-11-28
"""

import mne
import numpy as np
from pathlib import Path


def create_source_space(config, previous_outputs):
    """
    Create surface source space using icosahedral subdivision.

    For sphere BEM: Creates icosphere at brain radius with inset
    For ellipsoid BEM: Creates icosphere scaled to ellipsoid shape with inset

    This matches the adv_test approach:
    - bem_sphere/scripts/101_surface_sources.py
    - bem_ellipsoid/scripts/101_surface_sources.py

    Parameters
    ----------
    config : Config
        Pipeline configuration
    previous_outputs : dict
        Previous pipeline outputs containing 'bem_params'

    Returns
    -------
    src : mne.SourceSpaces
        Surface source space
    source_coords_mm : ndarray, shape (n_sources, 3)
        Source coordinates in mm
    n_sources : int
        Number of sources
    """
    # Get BEM parameters and configuration
    bem_params = previous_outputs['bem_params']
    bem_type = config['pipeline']['bem_type']

    # Get ico level from config (default: 4 for ~2562 vertices)
    surface_config = config['source_space']['surface']
    ico_level = surface_config.get('ico_level', 4)

    print(f"  Creating surface source space:")
    print(f"    BEM type: {bem_type}")
    print(f"    Ico level: {ico_level}")

    # Create icosphere (unit sphere)
    vertices_unit, triangles = _create_icosphere(ico_level)

    print(f"    Generated {len(vertices_unit):,} vertices on unit sphere")

    if bem_type == 'sphere':
        # Sphere surface: scale icosphere to brain radius with inset
        center_mm = np.array(bem_params['center_mm'])
        brain_radius_mm = bem_params['brain_radius_mm']

        # Apply inset factor (sources at 85% of brain radius to ensure interior)
        # Matches adv_test/bem_sphere/scripts/101_surface_sources.py line 109
        inset_factor = config['source_space']['surface'].get('inset_factor', 0.85)
        radius_inset_mm = brain_radius_mm * inset_factor

        print(f"    Sphere center: [{center_mm[0]:.3f}, {center_mm[1]:.3f}, {center_mm[2]:.3f}] mm")
        print(f"    Brain radius: {brain_radius_mm:.3f} mm")
        print(f"    Inset factor: {inset_factor:.2f}")
        print(f"    Source radius: {radius_inset_mm:.3f} mm")

        # Scale unit sphere to inset radius
        vertices_mm = vertices_unit * radius_inset_mm

        # Translate to sphere center
        vertices_mm = vertices_mm + center_mm

        # Normals are radial for sphere (pointing outward)
        normals = vertices_unit.copy()

    elif bem_type == 'ellipsoid':
        # Ellipsoid surface: scale icosphere to ellipsoid shape with inset
        center_mm = np.array(bem_params['center_mm'])
        semi_axes_mm = np.array(bem_params['semi_axes_mm'])
        rotation = np.array(bem_params['rotation_matrix'])

        # Apply inset factor (sources at 80% of ellipsoid to ensure interior)
        # Matches adv_test/bem_ellipsoid/scripts/101_surface_sources.py line 120
        inset_factor = config['source_space']['surface'].get('inset_factor', 0.80)
        semi_axes_inset_mm = semi_axes_mm * inset_factor

        print(f"    Ellipsoid center: [{center_mm[0]:.3f}, {center_mm[1]:.3f}, {center_mm[2]:.3f}] mm")
        print(f"    Semi-axes: [{semi_axes_mm[0]:.3f}, {semi_axes_mm[1]:.3f}, {semi_axes_mm[2]:.3f}] mm")
        print(f"    Inset factor: {inset_factor:.2f}")
        print(f"    Source semi-axes: [{semi_axes_inset_mm[0]:.3f}, {semi_axes_inset_mm[1]:.3f}, {semi_axes_inset_mm[2]:.3f}] mm")

        # Transform unit sphere to ellipsoid
        # 1. Scale by semi-axes with inset
        vertices_scaled = vertices_unit * semi_axes_inset_mm[np.newaxis, :]

        # 2. Rotate according to principal axes
        vertices_rotated = vertices_scaled @ rotation.T

        # 3. Translate to ellipsoid center
        vertices_mm = vertices_rotated + center_mm[np.newaxis, :]

        # Compute normals for ellipsoid (perpendicular to surface)
        # Normal at point on ellipsoid is gradient of F(x,y,z) = (x/a)^2 + (y/b)^2 + (z/c)^2 - 1
        # In the rotated coordinate system
        vertices_local = (vertices_mm - center_mm[np.newaxis, :]) @ rotation
        normals_local = vertices_local / (semi_axes_inset_mm[np.newaxis, :]**2)
        normals_local = normals_local / np.linalg.norm(normals_local, axis=1, keepdims=True)
        normals = normals_local @ rotation.T

        # Check rotation
        if np.allclose(rotation, np.eye(3)):
            print(f"    Rotation: axis-aligned (identity)")
        else:
            print(f"    Rotation: rotated (non-identity)")

    else:
        raise ValueError(f"Unknown BEM type: {bem_type}")

    # Optional: Filter to dorsal hemisphere only (Z > center)
    # This reduces sources and improves condition number for top-electrode setups
    filter_dorsal = config['source_space']['surface'].get('filter_dorsal', False)

    if filter_dorsal:
        center_z = center_mm[2]
        dorsal_mask = vertices_mm[:, 2] > center_z

        n_before = len(vertices_mm)
        vertices_mm = vertices_mm[dorsal_mask]
        normals = normals[dorsal_mask]

        # Update triangles to only include those with all vertices in dorsal hemisphere
        if len(triangles) > 0:
            # Get new vertex indices after filtering
            old_to_new = np.full(n_before, -1, dtype=int)
            old_to_new[dorsal_mask] = np.arange(np.sum(dorsal_mask))

            # Filter triangles: keep only if all vertices are dorsal
            valid_tris = []
            for tri in triangles:
                if np.all(dorsal_mask[tri]):
                    new_tri = old_to_new[tri]
                    valid_tris.append(new_tri)

            triangles = np.array(valid_tris, dtype=np.int32) if valid_tris else np.array([], dtype=np.int32).reshape(0, 3)

        print(f"    Dorsal filtering (Z > {center_z:.3f} mm):")
        print(f"      Before: {n_before} sources")
        print(f"      After: {len(vertices_mm)} sources ({100*len(vertices_mm)/n_before:.1f}%)")
        print(f"      Triangles: {len(triangles)}")

    n_sources = len(vertices_mm)

    # Convert to meters for MNE
    vertices_m = vertices_mm / 1000.0

    # Create MNE-compatible source space dictionary
    src_dict = {
        'rr': np.array(vertices_m, dtype=np.float64),  # Source positions (meters)
        'nn': np.array(normals, dtype=np.float64),      # Normal vectors
        'nuse': n_sources,
        'inuse': np.ones(n_sources, dtype=np.int32),
        'vertno': np.arange(n_sources, dtype=np.int32),
        'type': 'surf',  # Surface type
        'coord_frame': mne.io.constants.FIFF.FIFFV_COORD_MRI,
        'id': 101,  # Source space ID (MNE convention: 101 for left hemi)
        'np': n_sources,
        'ntri': len(triangles),
        'nuse_tri': len(triangles),
        'tris': np.array(triangles, dtype=np.int32),
        'use_tris': np.array(triangles, dtype=np.int32),
        'nearest': None,
        'nearest_dist': None,
        'pinfo': None,
        'patch_inds': None,
        'dist': None,
        'dist_limit': None,
    }

    # Wrap in MNE SourceSpaces object
    src = mne.SourceSpaces([src_dict])

    print(f"    ✓ Created surface source space with {n_sources:,} sources")

    return src, vertices_mm, n_sources


def _create_icosphere(subdivisions=4):
    """
    Create an icosphere (subdivided icosahedron) with unit radius.

    This is a pure Python implementation that matches PyVista's icosphere.
    Adapted from adv_test approach and bem/ellipsoid.py.

    Parameters
    ----------
    subdivisions : int
        Number of subdivision iterations
        - 0 = icosahedron (12 vertices, 20 faces)
        - 1 = 42 vertices, 80 faces
        - 2 = 162 vertices, 320 faces
        - 3 = 642 vertices, 1280 faces
        - 4 = 2562 vertices, 5120 faces (default, matches adv_test)
        - 5 = 10242 vertices, 20480 faces

    Returns
    -------
    vertices : ndarray, shape (n_vertices, 3)
        Vertex coordinates on unit sphere
    triangles : ndarray, shape (n_triangles, 3)
        Triangle vertex indices
    """
    # Icosahedron vertices (12 vertices on unit sphere)
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

    Each edge is split at its midpoint, and the midpoints are projected
    onto the unit sphere to maintain spherical geometry.

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

        # Compute midpoints of each edge
        m1 = (vertices[v1] + vertices[v2]) / 2.0
        m1 = m1 / np.linalg.norm(m1)  # Project to unit sphere

        m2 = (vertices[v2] + vertices[v3]) / 2.0
        m2 = m2 / np.linalg.norm(m2)  # Project to unit sphere

        m3 = (vertices[v3] + vertices[v1]) / 2.0
        m3 = m3 / np.linalg.norm(m3)  # Project to unit sphere

        # Get or create indices for midpoints
        for m in [m1, m2, m3]:
            m_tuple = tuple(m)
            if m_tuple not in vertex_to_idx:
                vertex_to_idx[m_tuple] = len(vertex_list)
                vertex_list.append(m)

        a_idx = vertex_to_idx[tuple(m1)]
        b_idx = vertex_to_idx[tuple(m2)]
        c_idx = vertex_to_idx[tuple(m3)]

        # Create four new triangles from the subdivided triangle
        # Pattern:
        #       v1
        #      /  \
        #    m1    m3
        #   /  \  /  \
        # v2---m2---v3
        new_tri_indices.append([v1, a_idx, c_idx])    # Top corner
        new_tri_indices.append([v2, b_idx, a_idx])    # Bottom left
        new_tri_indices.append([v3, c_idx, b_idx])    # Bottom right
        new_tri_indices.append([a_idx, b_idx, c_idx]) # Center

    return np.array(vertex_list), np.array(new_tri_indices)
