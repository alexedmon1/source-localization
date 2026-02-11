"""Step 3: Source Space Construction.

Create source space for source estimation.

Source types:
- cartesian: 3D Cartesian grid (formerly 'volumetric')
- surface: Icosphere surface mesh
- roi_based: Sources at ROI centroids
- shell: Concentric geometry-matched shells

All source types support a universal electrode proximity filter via:
  source_space.max_electrode_distance_mm: <float>
This removes any source farther than the specified distance from the nearest
electrode, ensuring consistent depth coverage across source types.
"""

import mne
import numpy as np
from scipy.spatial.distance import cdist


def run(config, previous_outputs):
    """
    Create source space.

    Parameters
    ----------
    config : Config
        Pipeline configuration
    previous_outputs : dict
        Outputs from previous steps containing 'bem_params'

    Returns
    -------
    outputs : dict
        Dictionary containing:
        - 'src': mne.SourceSpaces - Source space
        - 'source_type': str - Type of source space
        - 'source_coords_mm': ndarray - Source coordinates in mm
        - 'n_sources': int - Number of sources
    """
    source_type = config['pipeline']['source_type']
    print(f"Source space type: {source_type}")

    # Delegate to appropriate module
    # Support both old and new names for backward compatibility
    if source_type in ('cartesian', 'volumetric'):
        from ..source_space import volumetric
        src, source_coords_mm, n_sources = volumetric.create_source_space(config, previous_outputs)
    elif source_type == 'surface':
        from ..source_space import surface
        src, source_coords_mm, n_sources = surface.create_source_space(config, previous_outputs)
    elif source_type == 'roi_based':
        from ..source_space import roi_based
        src, source_coords_mm, n_sources = roi_based.create_source_space(config, previous_outputs)
    elif source_type in ('shell', 'shell_based'):
        from ..source_space import shell_based
        src, source_coords_mm, n_sources = shell_based.create_source_space(config, previous_outputs)
    else:
        raise ValueError(f"Unknown source type: {source_type}. "
                        f"Valid types: cartesian, surface, roi_based, shell")

    print(f"✓ Created {source_type} source space with {n_sources:,} sources")

    # Apply universal electrode proximity filter (if configured)
    max_electrode_distance_mm = config['source_space'].get('max_electrode_distance_mm', None)

    if max_electrode_distance_mm is not None:
        info = previous_outputs.get('info')
        if info is not None:
            src, source_coords_mm, n_sources = _filter_by_electrode_proximity(
                src, source_coords_mm, info, max_electrode_distance_mm
            )
        else:
            print(f"  Warning: Cannot apply electrode proximity filter - no electrode info available")

    # Save intermediate data and create visualizations
    if config['outputs'].get('save_intermediate', True):
        from ..utils.io_utils import save_pickle, save_numpy, get_data_dir, get_figures_dir
        from ..utils.step_visualizations import visualize_step3_source_space

        data_dir = get_data_dir(config)
        figures_dir = get_figures_dir(config)

        # Save source space
        save_pickle(src, data_dir / 'step3_source_space.pkl')
        save_numpy(source_coords_mm, data_dir / 'step3_source_coords_mm.npy')
        print(f"    Saved: {data_dir / 'step3_source_space.pkl'}")
        print(f"    Saved: {data_dir / 'step3_source_coords_mm.npy'}")

        # Create source space visualization (need electrode coords from previous step)
        info = previous_outputs.get('info')
        if info is not None:
            n_electrodes = len(info['ch_names'])
            elec_coords_mm = np.array([info['chs'][i]['loc'][:3] for i in range(n_electrodes)]) * 1000
            fig = visualize_step3_source_space(source_coords_mm, elec_coords_mm, figures_dir / 'step3_source_space.png')
            print(f"    Saved: {figures_dir / 'step3_source_space.png'}")
            import matplotlib.pyplot as plt
            plt.close(fig)

    return {
        'src': src,
        'source_type': source_type,
        'source_coords_mm': source_coords_mm,
        'n_sources': n_sources
    }


def _filter_by_electrode_proximity(src, source_coords_mm, info, max_distance_mm):
    """
    Filter sources to keep only those within max_distance_mm of nearest electrode.

    This provides consistent depth coverage across all source space types,
    removing deep sources that EEG cannot reliably localize.

    Parameters
    ----------
    src : mne.SourceSpaces
        Source space to filter
    source_coords_mm : ndarray, shape (n_sources, 3)
        Source coordinates in mm
    info : mne.Info
        MNE info object with electrode positions
    max_distance_mm : float
        Maximum allowed distance from source to nearest electrode

    Returns
    -------
    src_filtered : mne.SourceSpaces
        Filtered source space
    source_coords_filtered : ndarray
        Filtered source coordinates
    n_sources_filtered : int
        Number of sources after filtering
    """
    # Extract electrode coordinates from info (convert m to mm)
    n_electrodes = len(info['ch_names'])
    electrode_coords_mm = np.array([
        info['chs'][i]['loc'][:3] for i in range(n_electrodes)
    ]) * 1000

    # Compute distance from each source to nearest electrode
    distances = cdist(source_coords_mm, electrode_coords_mm)
    min_distances = distances.min(axis=1)

    # Create mask for sources within max distance
    keep_mask = min_distances <= max_distance_mm

    n_before = len(source_coords_mm)
    n_after = np.sum(keep_mask)
    n_removed = n_before - n_after

    print(f"  Electrode proximity filter (max {max_distance_mm:.1f} mm):")
    print(f"    Before: {n_before} sources")
    print(f"    After: {n_after} sources (removed {n_removed})")
    print(f"    Distance range (kept): [{min_distances[keep_mask].min():.2f}, "
          f"{min_distances[keep_mask].max():.2f}] mm")

    if n_after == 0:
        raise ValueError(f"All sources filtered out! max_electrode_distance_mm={max_distance_mm} is too restrictive.")

    # Filter source coordinates
    source_coords_filtered = source_coords_mm[keep_mask]

    # Rebuild source space with filtered sources
    src_filtered = _rebuild_source_space(src, keep_mask)

    return src_filtered, source_coords_filtered, n_after


def _rebuild_source_space(src, keep_mask):
    """
    Rebuild MNE SourceSpaces with only the sources indicated by keep_mask.

    Parameters
    ----------
    src : mne.SourceSpaces
        Original source space
    keep_mask : ndarray of bool
        Boolean mask indicating which sources to keep

    Returns
    -------
    src_new : mne.SourceSpaces
        New source space with only kept sources
    """
    # Get the original source space dict
    src_dict = src[0].copy()

    # Filter positions and normals
    src_dict['rr'] = src_dict['rr'][keep_mask]
    src_dict['nn'] = src_dict['nn'][keep_mask]

    # Update counts
    n_new = int(np.sum(keep_mask))
    src_dict['nuse'] = n_new
    src_dict['np'] = n_new
    src_dict['inuse'] = np.ones(n_new, dtype=np.int32)
    src_dict['vertno'] = np.arange(n_new, dtype=np.int32)

    # Handle ROI assignments if present
    if 'roi_assignments' in src_dict and src_dict['roi_assignments'] is not None:
        src_dict['roi_assignments'] = np.asarray(src_dict['roi_assignments'])[keep_mask]

    # Handle triangles for surface source spaces (remove invalid triangles)
    if src_dict.get('ntri', 0) > 0 and 'tris' in src_dict:
        # Build mapping from old to new indices
        old_to_new = np.full(len(keep_mask), -1, dtype=int)
        old_to_new[keep_mask] = np.arange(n_new)

        # Filter triangles to only those where all vertices are kept
        tris = src_dict['tris']
        valid_tris = []
        for tri in tris:
            if np.all(keep_mask[tri]):
                new_tri = old_to_new[tri]
                valid_tris.append(new_tri)

        if valid_tris:
            src_dict['tris'] = np.array(valid_tris, dtype=np.int32)
            src_dict['use_tris'] = src_dict['tris']
            src_dict['ntri'] = len(valid_tris)
            src_dict['nuse_tri'] = len(valid_tris)
        else:
            src_dict['tris'] = np.array([], dtype=np.int32).reshape(0, 3)
            src_dict['use_tris'] = src_dict['tris']
            src_dict['ntri'] = 0
            src_dict['nuse_tri'] = 0

    # Update shape for volumetric source spaces
    if src_dict.get('type') == 'vol':
        src_dict['shape'] = (n_new, 1, 1)
        src_dict['mri_width'] = n_new
        src_dict['mri_height'] = 1
        src_dict['mri_depth'] = 1

    # Create new SourceSpaces object
    src_new = mne.SourceSpaces([src_dict])

    return src_new
