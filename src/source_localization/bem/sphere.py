"""Spherical BEM model implementation.

Supports both:
- Analytical sphere (Berg-Scherg approximation) - fast, default
- Numerical sphere (BEM with tessellated surfaces) - accurate magnitude scaling
"""

import mne
import numpy as np
import nibabel as nib
from pathlib import Path
from mne.surface import complete_surface_info, _get_ico_surface


def create_numerical_sphere_surfaces(center_mm, radii_mm, conductivities, n_vertices=642):
    """
    Create triangulated sphere surfaces for numerical BEM.

    Parameters
    ----------
    center_mm : ndarray, shape (3,)
        Sphere center in mm
    radii_mm : list of float
        Radii for each layer in mm (inner to outer: brain, skull, scalp)
    conductivities : list of float
        Conductivity values for each layer in S/m
    n_vertices : int
        Number of vertices per surface (default: 642, ico-3 subdivision)

    Returns
    -------
    surfaces : list of dict
        List of surface dictionaries in MNE format
    """
    # Determine subdivision level
    if n_vertices <= 162:
        subdivision = 2
    elif n_vertices <= 642:
        subdivision = 3
    elif n_vertices <= 2562:
        subdivision = 4
    else:
        subdivision = 5

    # Get MNE's standard icosphere
    ico = _get_ico_surface(subdivision)

    # MNE surface IDs: brain=4, skull=3, scalp=1 (inner to outer)
    surface_ids = [4, 3, 1]

    surfaces = []
    for layer_idx, (radius_mm, sigma) in enumerate(zip(radii_mm, conductivities)):
        # Scale icosphere to desired radius and convert to meters
        rr_m = ico['rr'] * (radius_mm / 1000.0) + (center_mm / 1000.0)

        # Create surface dictionary (MNE format)
        surface = {
            'rr': rr_m,
            'tris': ico['tris'].copy(),
            'ntri': len(ico['tris']),
            'np': len(rr_m),
            'coord_frame': mne.io.constants.FIFF.FIFFV_COORD_MRI,
            'id': surface_ids[layer_idx],
            'sigma': sigma,
        }

        # Complete surface info (adds tri_area, tri_nn, neighbor_tri)
        complete_surface_info(surface, copy=False)
        surfaces.append(surface)

    return surfaces


def create_bem(config, previous_outputs):
    """
    Create spherical BEM model from brain volume.

    Fits a sphere to the brain surface extracted from the brain volume.
    The sphere center is computed as the centroid of brain voxels, and the
    radius is the maximum distance from center to any brain voxel.

    Parameters
    ----------
    config : Config
        Pipeline configuration
    previous_outputs : dict
        Previous pipeline outputs (not used for sphere BEM)

    Returns
    -------
    bem_model : mne.bem.ConductorModel
        Spherical BEM model
    bem_params : dict
        Dictionary containing:
        - 'center_mm': Sphere center in mm coordinates
        - 'brain_radius_mm': Brain (innermost) sphere radius in mm
        - 'radii_mm': List of radii for all layers (brain, skull, scalp)
        - 'radii_ratios': Ratios used to compute layer radii
        - 'conductivities': Conductivities for each layer
        - 'n_layers': Number of layers
    """
    from ..utils.atlas import get_true_affine

    # Get package directory to locate brain volume
    # CRITICAL: Must use skull-stripped brain mask for sphere fitting
    # NOT the full atlas which includes skull/exterior
    package_dir = Path(__file__).parent.parent
    brain_mask_file = package_dir / "data/atlas/Atlas_3DRois_brain.nii.gz"

    print(f"  Loading skull-stripped brain: {brain_mask_file.name}")
    brain_img = nib.load(brain_mask_file)
    brain_data = brain_img.get_fdata()
    affine_corrected = get_true_affine(brain_img)

    # Extract brain mask (all non-zero voxels)
    brain_mask = brain_data > 0
    brain_voxels = np.argwhere(brain_mask)

    print(f"  Brain voxels: {len(brain_voxels):,}")

    # Compute brain center in voxel coordinates
    center_vox = brain_voxels.mean(axis=0)

    # Convert to mm coordinates
    center_mm = nib.affines.apply_affine(affine_corrected, center_vox)

    # Check if electrode-centered sphere positioning is requested
    electrode_centered = config['bem']['sphere'].get('electrode_centered', False)

    if electrode_centered and 'info' in previous_outputs:
        # ELECTRODE-CENTERED SPHERE: Position sphere with top at brain surface,
        # centered under the electrode array
        # This ensures sources are INSIDE the brain, below the electrodes
        info = previous_outputs['info']
        n_electrodes = len(info['ch_names'])
        elec_coords_mm = np.array([info['chs'][i]['loc'][:3] for i in range(n_electrodes)]) * 1000

        # Get electrode array center (X, Y)
        elec_centroid = elec_coords_mm.mean(axis=0)

        # Get brain top Z (maximum Z of brain voxels)
        brain_coords_mm = nib.affines.apply_affine(affine_corrected, brain_voxels)
        brain_top_z = brain_coords_mm[:, 2].max()

        # Compute brain radius from config or auto
        brain_radius_mm_config = config['bem']['sphere'].get('brain_radius_mm', None)
        if brain_radius_mm_config is None:
            # Auto-compute radius to fit brain extent
            # Use the vertical extent of the brain as a guide
            brain_z_range = brain_coords_mm[:, 2].max() - brain_coords_mm[:, 2].min()
            brain_x_range = brain_coords_mm[:, 0].max() - brain_coords_mm[:, 0].min()
            brain_y_range = brain_coords_mm[:, 1].max() - brain_coords_mm[:, 1].min()

            # Use average of X and Y extents as diameter (brain is wider than tall)
            brain_radius_mm = (brain_x_range + brain_y_range) / 4.0  # half of average extent
        else:
            brain_radius_mm = float(brain_radius_mm_config)

        # Position sphere center so TOP is at brain_top_z
        # Center Z = brain_top_z - brain_radius
        center_mm = np.array([
            elec_centroid[0],  # X: under electrode center
            elec_centroid[1],  # Y: under electrode center
            brain_top_z - brain_radius_mm  # Z: so top of sphere is at brain top
        ])

        print(f"  Electrode-centered sphere fitting:")
        print(f"    Electrode centroid: [{elec_centroid[0]:.3f}, {elec_centroid[1]:.3f}, {elec_centroid[2]:.3f}] mm")
        print(f"    Brain top Z: {brain_top_z:.3f} mm")
        print(f"    Brain radius: {brain_radius_mm:.3f} mm")
        print(f"    Sphere center: [{center_mm[0]:.3f}, {center_mm[1]:.3f}, {center_mm[2]:.3f}] mm")
        print(f"    Sphere top Z: {center_mm[2] + brain_radius_mm:.3f} mm (= brain top)")

    else:
        # Brain-based fitting (default)
        # Determine brain radius computation method
        brain_radius_mm_config = config['bem']['sphere'].get('brain_radius_mm', None)

        if brain_radius_mm_config == 'auto_ellipsoid' or brain_radius_mm_config == 'ellipsoid':
            # Fit ellipsoid and convert to equivalent sphere radius
            print(f"  Fitting ellipsoid to brain surface...")
            brain_coords_mm = nib.affines.apply_affine(affine_corrected, brain_voxels)

            # Import ellipsoid fitting function
            from ..bem.ellipsoid import fit_ellipsoid_to_brain

            # Get fitting method and margin from config
            ellipsoid_method = config['bem']['sphere'].get('ellipsoid_method', 'axis_aligned')
            ellipsoid_margin = config['bem']['sphere'].get('ellipsoid_margin', 1.23)

            center_ellipsoid, semi_axes, rotation = fit_ellipsoid_to_brain(
                brain_coords_mm, method=ellipsoid_method, margin=ellipsoid_margin
            )

            # Update center to ellipsoid center
            center_mm = center_ellipsoid

            # Compute equivalent sphere radius (geometric mean of semi-axes)
            brain_radius_mm = np.prod(semi_axes) ** (1/3)

            print(f"  Ellipsoid semi-axes: [{semi_axes[0]:.2f}, {semi_axes[1]:.2f}, {semi_axes[2]:.2f}] mm")
            print(f"  Equivalent sphere radius: {brain_radius_mm:.3f} mm (geometric mean)")
            print(f"  Sphere center: [{center_mm[0]:.3f}, {center_mm[1]:.3f}, {center_mm[2]:.3f}] mm")

        elif brain_radius_mm_config is None:
            # Auto-compute: use percentile of distances from center to brain voxels
            brain_coords_mm = nib.affines.apply_affine(affine_corrected, brain_voxels)
            distances = np.linalg.norm(brain_coords_mm - center_mm, axis=1)

            # Get percentile from config (default 95.0 for robust sphere fitting)
            radius_percentile = config['bem']['sphere'].get('radius_percentile', 95.0)
            brain_radius_mm = np.percentile(distances, radius_percentile)

            print(f"  Sphere center: [{center_mm[0]:.3f}, {center_mm[1]:.3f}, {center_mm[2]:.3f}] mm")
            print(f"  Brain radius: {brain_radius_mm:.3f} mm ({radius_percentile}th percentile of {len(distances):,} brain voxels)")
            print(f"  Distance range: {distances.min():.2f} to {distances.max():.2f} mm")

        else:
            # Use specified numeric radius
            brain_radius_mm = float(brain_radius_mm_config)
            print(f"  Sphere center: [{center_mm[0]:.3f}, {center_mm[1]:.3f}, {center_mm[2]:.3f}] mm")
            print(f"  Brain radius: {brain_radius_mm:.3f} mm (from config)")

    # Get BEM parameters from config
    conductivities = config['bem']['sphere']['conductivities']
    n_layers = len(conductivities)

    # Compute layer radii using radii ratios
    # Default ratios for mouse: brain=0.87, skull=0.92, scalp=1.0
    radii_ratios = config['bem']['sphere'].get('radii_ratios', [0.87, 0.92, 1.0])

    # For 2-layer model, radii_ratios might be [0.90] (just brain ratio)
    # Append 1.0 for the outermost layer if not present
    if len(radii_ratios) < n_layers:
        radii_ratios = radii_ratios + [1.0]

    # Scale brain radius by ratios to get each layer's radius
    radii_mm = [brain_radius_mm / radii_ratios[0] * ratio for ratio in radii_ratios]

    # Head radius is the outermost layer
    head_radius_mm = radii_mm[-1]

    # Apply scale factor if provided (for brain_size validation tests)
    scale_factor = config['bem']['sphere'].get('scale_factor', 1.0)
    if scale_factor != 1.0:
        print(f"  Applying {scale_factor}x scale factor to BEM geometry...")
        center_mm = center_mm * scale_factor
        brain_radius_mm = brain_radius_mm * scale_factor
        radii_mm = [r * scale_factor for r in radii_mm]
        head_radius_mm = head_radius_mm * scale_factor

    print(f"  Creating {n_layers}-layer spherical BEM:")
    layer_names = ['brain', 'skull', 'scalp']
    for i, (r_mm, sigma) in enumerate(zip(radii_mm, conductivities)):
        layer_name = layer_names[i] if i < 3 else f'layer{i+1}'
        print(f"    {layer_name}: radius={r_mm:.3f} mm, σ={sigma} S/m")

    # Convert to meters for MNE
    center_m = center_mm / 1000.0
    head_radius_m = head_radius_mm / 1000.0

    # Compute relative radii for MNE (ALL layers as fraction of head radius, including outermost)
    # MNE expects relative_radii to have the same length as sigmas
    relative_radii_mne = [r / head_radius_mm for r in radii_mm]  # All layers

    # Debug: print what we're passing to MNE
    print(f"  DEBUG: sigmas = {conductivities} (len={len(conductivities)})")
    print(f"  DEBUG: relative_radii_mne = {relative_radii_mne} (len={len(relative_radii_mne)})")
    print(f"  DEBUG: head_radius_m = {head_radius_m}")

    # Check if numerical BEM is requested (DEFAULT: True for correct mouse brain scaling)
    # Numerical BEM uses tessellated surfaces and boundary integral method
    # This gives correct leadfield magnitude scaling for mouse brain
    # Set numerical: false to use analytical Berg-Scherg (human-scale, ~35x larger leadfield)
    use_numerical = config['bem']['sphere'].get('numerical', True)

    if use_numerical:
        # Create numerical sphere BEM using tessellated surfaces
        print(f"  Creating NUMERICAL {n_layers}-layer spherical BEM:")

        # Create sphere surfaces
        surfaces = create_numerical_sphere_surfaces(
            center_mm=center_mm,
            radii_mm=radii_mm,
            conductivities=conductivities,
            n_vertices=642  # ico-3 subdivision
        )

        layer_names = ['brain', 'skull', 'scalp']
        for i, (surf, r_mm, sigma) in enumerate(zip(surfaces, radii_mm, conductivities)):
            layer_name = layer_names[i] if i < 3 else f'layer{i+1}'
            print(f"    {layer_name}: {surf['np']} vertices, radius={r_mm:.3f} mm, σ={sigma} S/m")

        # Create BEM model structure (same format as ellipsoid)
        bem_model = {
            'surfs': surfaces,
            'sigma': conductivities,
            'is_sphere': False,  # Use numerical forward, not analytical
            'coord_frame': mne.io.constants.FIFF.FIFFV_COORD_MRI
        }

        print(f"  ✓ Created NUMERICAL {n_layers}-layer spherical BEM model")
        print(f"    (Uses boundary integral method for accurate leadfield magnitude)")

    else:
        # Create analytical spherical BEM model using MNE's Berg-Scherg approximation
        bem_model = mne.make_sphere_model(
            r0=center_m,
            head_radius=head_radius_m,  # Single float: outermost radius
            sigmas=conductivities,
            relative_radii=tuple(relative_radii_mne),  # Ratios for ALL layers (last should be 1.0)
            verbose=False
        )

        print(f"  ✓ Created ANALYTICAL {n_layers}-layer spherical BEM model")
        print(f"    (Uses Berg-Scherg equivalent dipole approximation)")

    # Store parameters for caching and source space construction
    bem_params = {
        'center_mm': center_mm.tolist(),
        'brain_radius_mm': float(brain_radius_mm),
        'radii_mm': [float(r) for r in radii_mm],
        'radii_ratios': radii_ratios,
        'conductivities': conductivities,
        'n_layers': n_layers,
        'scale_factor': scale_factor,
        'numerical': use_numerical
    }

    return bem_model, bem_params
