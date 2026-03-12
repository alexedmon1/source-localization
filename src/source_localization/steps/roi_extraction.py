"""Step 6: ROI Extraction.

Extract source activity for each ROI defined in the atlas.

Produces both magnitude and signed ROI time series:
- Magnitude: Always positive, for power/spectral analysis
- Signed: Preserves sign (SVD-based), for connectivity/correlation analysis
"""

import numpy as np
import nibabel as nib
import json
from pathlib import Path
from scipy.spatial import cKDTree


def run(config, previous_outputs):
    """
    Extract ROI-level source estimates using spatial proximity mapping.

    Processes both magnitude and signed source time courses.

    Parameters
    ----------
    config : Config
        Pipeline configuration
    previous_outputs : dict
        Outputs from previous steps containing:
        - 'stc': mne.SourceEstimate - Source time courses (magnitude, backward compat)
        - 'stc_magnitude': mne.SourceEstimate - Magnitude source time courses
        - 'stc_signed': mne.SourceEstimate - Signed source time courses
        - 'source_coords_mm': ndarray - Source coordinates in mm
        - 'n_sources': int - Number of sources

    Returns
    -------
    outputs : dict
        Dictionary containing:
        - 'roi_stcs': dict - Magnitude ROI time series (backward compat)
        - 'roi_stcs_magnitude': dict - Magnitude ROI time series
        - 'roi_stcs_signed': dict - Signed ROI time series
        - 'roi_labels': list - ROI names
        - 'roi_source_mapping': dict - Sources assigned to each ROI
    """
    use_proximity = config['roi'].get('use_proximity', True)
    proximity_radius_mm = config['roi'].get('proximity_radius_mm', 2.0)

    print(f"  Extracting ROI-level source activity:")

    # Extract inputs - support both old and new format
    # New format has separate magnitude and signed STCs
    stc_magnitude = previous_outputs.get('stc_magnitude', previous_outputs.get('stc'))
    stc_signed = previous_outputs.get('stc_signed', stc_magnitude)  # Fall back to magnitude
    source_coords_mm = previous_outputs['source_coords_mm']
    n_sources = previous_outputs['n_sources']
    src = previous_outputs.get('src')

    # For backward compatibility, use stc_magnitude as the primary
    stc = stc_magnitude

    # Check if we have roi_assignments from ROI-based source space
    has_roi_assignments = False
    roi_assignments = None
    full_brain_coverage = config['inputs'].get('full_brain_coverage', False)

    if src is not None and len(src) > 0 and 'roi_assignments' in src[0]:
        roi_assignments = src[0]['roi_assignments']
        has_roi_assignments = True
        print(f"    Using direct ROI assignment (sources pre-assigned to ROIs)")
        use_proximity = False  # Override: use direct assignment
    elif not full_brain_coverage:
        print(f"    ⚠️  Skipping ROI extraction: atlas does not have full brain coverage")
        print(f"    Non-ROI source spaces (shell, cartesian) require a full-brain atlas for reliable ROI mapping.")
        print(f"    Use an ROI-based source space, or switch to an atlas with full_brain_coverage=True (e.g., Allen).")
        return {
            'roi_stcs': {},
            'roi_stcs_magnitude': {},
            'roi_stcs_signed': {},
            'roi_labels': [],
            'roi_source_mapping': {}
        }
    else:
        print(f"    Proximity mapping: {use_proximity} (atlas has full brain coverage)")
        if use_proximity:
            print(f"    Proximity radius: {proximity_radius_mm} mm")

    # Debug: Check for source count mismatch
    n_sources_stc = len(stc.data)
    n_sources_coords = len(source_coords_mm)
    print(f"    Source counts: stc.data={n_sources_stc}, coords={n_sources_coords}, n_sources={n_sources}")

    if n_sources_stc != n_sources_coords:
        print(f"    ⚠️  WARNING: Source count mismatch!")
        print(f"    Using only the first {n_sources_stc} source coordinates to match stc.data")
        source_coords_mm = source_coords_mm[:n_sources_stc]
        n_sources = n_sources_stc
        # Also truncate roi_assignments if present
        if roi_assignments is not None and len(roi_assignments) > n_sources_stc:
            roi_assignments = roi_assignments[:n_sources_stc]
            print(f"    Truncated ROI assignments from {n_sources_coords} to {n_sources_stc}")

    # Load brain labels and ROI mapping
    package_dir = Path(__file__).parent.parent
    brain_labels_file = package_dir / config['inputs']['brain_labels']
    roi_mapping_file = package_dir / config['inputs']['roi_mapping']

    # Load brain label volume
    nii_labels = nib.load(brain_labels_file)
    label_data = nii_labels.get_fdata()

    # Use the ORIGINAL NIfTI affine (not the 10× corrected one).
    # The pipeline's source coordinates are computed in the same coordinate
    # frame as the original atlas headers, so applying the 10× voxel size
    # correction here would create a 10× coordinate mismatch.
    affine = nii_labels.affine

    # Load ROI mapping JSON
    with open(roi_mapping_file, 'r') as f:
        roi_data = json.load(f)

    # Extract ROI definitions (JSON has top-level metadata + 'rois' dict)
    roi_mapping = roi_data.get('rois', roi_data)

    # Check for ROI category filtering (e.g., cortex-only)
    include_categories = config['roi'].get('include_categories', None)
    included_roi_ids = None
    if include_categories:
        categories = roi_data.get('categories', {})
        included_roi_ids = set()
        for category in include_categories:
            if category in categories:
                included_roi_ids.update(categories[category])
            else:
                print(f"    Warning: Unknown category '{category}', skipping")
        print(f"    ROI filtering: categories={include_categories}, {len(included_roi_ids)} ROIs included")

    # Create label -> ROI name mapping
    # In the UAnterwerpen atlas, the label ID in the NIfTI matches the ROI ID in JSON
    # Skip label 0 (Background) — it represents unlabeled tissue, not a brain ROI
    SKIP_LABEL_NAMES = {"Background", "Exterior"}
    label_to_roi = {}
    for roi_id_str, roi_info in roi_mapping.items():
        label_id = int(roi_id_str) if isinstance(roi_id_str, str) else roi_id_str
        roi_name = roi_info.get('name', roi_info.get('abbreviation', f'ROI_{label_id}'))
        # Skip non-brain labels
        if roi_name in SKIP_LABEL_NAMES:
            continue
        # Skip if filtering and this ROI is not in included categories
        if included_roi_ids is not None and label_id not in included_roi_ids:
            continue
        label_to_roi[label_id] = roi_name

    # Get unique ROI names
    roi_labels = sorted(set(label_to_roi.values()))
    print(f"    Found {len(roi_labels)} unique ROIs")

    # Map each source to ROI(s)
    roi_source_mapping = {roi_name: [] for roi_name in roi_labels}

    if has_roi_assignments:
        # Direct mapping: use roi_assignments from ROI-based source space
        print(f"    Mapping sources using pre-assigned ROI labels")
        for source_idx, roi_id in enumerate(roi_assignments):
            roi_id_int = int(roi_id)
            if roi_id_int in label_to_roi:
                roi_name = label_to_roi[roi_id_int]
                roi_source_mapping[roi_name].append(source_idx)
    elif use_proximity:
        # Proximity-based mapping: sources can belong to multiple ROIs
        roi_source_mapping = map_sources_to_rois_proximity(
            source_coords_mm,
            label_data,
            affine,
            label_to_roi,
            proximity_radius_mm
        )
    else:
        # Nearest neighbor mapping: each source belongs to exactly one ROI
        roi_source_mapping = map_sources_to_rois_nearest(
            source_coords_mm,
            label_data,
            affine,
            label_to_roi
        )

    # Extract ROI time courses by averaging sources within each ROI
    # Do this for both magnitude and signed time series
    # ONLY include ROIs that have sources assigned - skip untested ROIs
    roi_stcs_magnitude = {}
    roi_stcs_signed = {}
    rois_with_sources = []
    rois_without_sources = []

    for roi_name in roi_labels:
        source_indices = roi_source_mapping.get(roi_name, [])
        if len(source_indices) > 0:
            # Average source activity within ROI
            roi_stcs_magnitude[roi_name] = stc_magnitude.data[source_indices, :].mean(axis=0)
            roi_stcs_signed[roi_name] = stc_signed.data[source_indices, :].mean(axis=0)
            rois_with_sources.append(roi_name)
        else:
            # Skip ROIs with no sources - don't include zeros for untested regions
            rois_without_sources.append(roi_name)

    # Update roi_labels to only include ROIs that were actually tested
    roi_labels = rois_with_sources

    if rois_without_sources:
        print(f"    Excluded {len(rois_without_sources)} ROIs with no sources: {rois_without_sources[:5]}{'...' if len(rois_without_sources) > 5 else ''}")

    # For backward compatibility
    roi_stcs = roi_stcs_magnitude

    # Filter roi_source_mapping to only include tested ROIs
    roi_source_mapping_filtered = {roi: roi_source_mapping[roi] for roi in roi_labels if roi in roi_source_mapping}

    # Report statistics (only for tested ROIs)
    sources_per_roi = {roi: len(sources) for roi, sources in roi_source_mapping_filtered.items()}
    assigned_sources = sum(sources_per_roi.values())

    print(f"    Total sources assigned: {assigned_sources:,} / {n_sources:,}")
    print(f"    Tested ROIs: {len(roi_labels)}")
    print(f"    Sources per ROI: mean={np.mean(list(sources_per_roi.values())):.1f}, "
          f"median={np.median(list(sources_per_roi.values())):.0f}")

    # Report signed statistics
    signed_values = np.concatenate([v for v in roi_stcs_signed.values() if len(v) > 0])
    print(f"    Signed ROI time series - has negatives: {np.any(signed_values < 0)}")
    print(f"    Signed ROI time series - min: {signed_values.min():.2e}, max: {signed_values.max():.2e}")

    # Save intermediate data and create visualizations
    if config['outputs'].get('save_intermediate', True):
        from ..utils.io_utils import save_pickle, get_data_dir, get_figures_dir
        from ..utils.step_visualizations import visualize_step6_roi_extraction

        data_dir = get_data_dir(config)
        figures_dir = get_figures_dir(config)

        # Save both magnitude and signed ROI timeseries (pickle format)
        save_pickle(roi_stcs_magnitude, data_dir / 'step6_roi_timeseries_magnitude.pkl')
        save_pickle(roi_stcs_signed, data_dir / 'step6_roi_timeseries_signed.pkl')
        # Backward compatibility: also save as step6_roi_timeseries.pkl (magnitude)
        save_pickle(roi_stcs_magnitude, data_dir / 'step6_roi_timeseries.pkl')
        print(f"    Saved: {data_dir / 'step6_roi_timeseries_magnitude.pkl'}")
        print(f"    Saved: {data_dir / 'step6_roi_timeseries_signed.pkl'}")
        print(f"    Saved: {data_dir / 'step6_roi_timeseries.pkl'} (backward compat, same as magnitude)")

        # Export to EEGLAB .set format (MNE-compatible)
        from ..utils.export_set import export_roi_to_set
        sfreq = stc.sfreq if hasattr(stc, 'sfreq') else previous_outputs.get('sfreq', 500.0)

        # Export magnitude ROI time series
        export_roi_to_set(
            roi_stcs_magnitude,
            sfreq=sfreq,
            output_path=data_dir / 'roi_timeseries_magnitude.set',
            subject_id='source_localized_magnitude'
        )

        # Export signed ROI time series
        export_roi_to_set(
            roi_stcs_signed,
            sfreq=sfreq,
            output_path=data_dir / 'roi_timeseries_signed.set',
            subject_id='source_localized_signed'
        )

        # Create ROI extraction visualizations for both magnitude and signed time series
        import matplotlib.pyplot as plt

        # Magnitude visualization
        fig = visualize_step6_roi_extraction(roi_stcs_magnitude, roi_labels, stc, roi_source_mapping_filtered,
                                             figures_dir / 'step6_roi_extraction_magnitude.png',
                                             title_suffix=' (Magnitude)')
        print(f"    Saved: {figures_dir / 'step6_roi_extraction_magnitude.png'}")
        plt.close(fig)

        # Signed visualization
        fig = visualize_step6_roi_extraction(roi_stcs_signed, roi_labels, stc, roi_source_mapping_filtered,
                                             figures_dir / 'step6_roi_extraction_signed.png',
                                             title_suffix=' (Signed)')
        print(f"    Saved: {figures_dir / 'step6_roi_extraction_signed.png'}")
        plt.close(fig)

    return {
        'roi_stcs': roi_stcs_magnitude,  # Backward compatibility
        'roi_stcs_magnitude': roi_stcs_magnitude,
        'roi_stcs_signed': roi_stcs_signed,
        'roi_labels': roi_labels,
        'roi_source_mapping': roi_source_mapping_filtered  # Only includes tested ROIs
    }


def map_sources_to_rois_nearest(source_coords_mm, label_data, affine, label_to_roi):
    """
    Map sources to ROIs using nearest labeled voxel assignment.

    Each source is assigned to exactly one ROI. First checks the voxel at the
    source coordinate; if that voxel is unlabeled (e.g., inter-parcel gap),
    finds the nearest labeled voxel using a KD-tree.
    """
    roi_source_mapping = {}

    # Convert source coordinates to voxel indices
    affine_inv = np.linalg.inv(affine)
    source_coords_homogeneous = np.column_stack([source_coords_mm, np.ones(len(source_coords_mm))])
    source_voxels = (affine_inv @ source_coords_homogeneous.T).T[:, :3]
    source_voxels = np.round(source_voxels).astype(int)

    # Clip to volume bounds
    for i in range(3):
        source_voxels[:, i] = np.clip(source_voxels[:, i], 0, label_data.shape[i] - 1)

    # Build KD-tree of labeled voxel coordinates for fallback lookup
    labeled_voxel_indices = np.argwhere(label_data > 0)
    labeled_voxel_coords = np.column_stack([
        labeled_voxel_indices, np.ones(len(labeled_voxel_indices))
    ])
    labeled_mm = (affine @ labeled_voxel_coords.T).T[:, :3]
    labeled_labels = label_data[
        labeled_voxel_indices[:, 0],
        labeled_voxel_indices[:, 1],
        labeled_voxel_indices[:, 2]
    ].astype(int)
    tree = cKDTree(labeled_mm)

    n_fallback = 0
    # Assign each source to ROI based on label at that voxel
    for source_idx in range(len(source_coords_mm)):
        voxel = source_voxels[source_idx]
        label_id = int(label_data[voxel[0], voxel[1], voxel[2]])

        # If unlabeled, find nearest labeled voxel
        if label_id not in label_to_roi:
            _, nearest_idx = tree.query(source_coords_mm[source_idx])
            label_id = int(labeled_labels[nearest_idx])
            n_fallback += 1

        if label_id in label_to_roi:
            roi_name = label_to_roi[label_id]
            if roi_name not in roi_source_mapping:
                roi_source_mapping[roi_name] = []
            roi_source_mapping[roi_name].append(source_idx)

    if n_fallback > 0:
        print(f"    Nearest-labeled fallback used for {n_fallback}/{len(source_coords_mm)} sources")

    return roi_source_mapping


def map_sources_to_rois_proximity(source_coords_mm, label_data, affine, label_to_roi, radius_mm):
    """
    Map sources to ROIs using proximity-based assignment.

    Sources are assigned to all ROIs within a specified radius.
    This allows sources near ROI boundaries to contribute to multiple ROIs.
    """
    roi_source_mapping = {}

    # Get all atlas voxels and their labels
    label_voxels = np.argwhere(label_data > 0)
    label_ids = label_data[label_voxels[:, 0], label_voxels[:, 1], label_voxels[:, 2]].astype(int)

    # Convert voxel coordinates to mm
    label_voxels_homogeneous = np.column_stack([label_voxels, np.ones(len(label_voxels))])
    label_coords_mm = (affine @ label_voxels_homogeneous.T).T[:, :3]

    # Build KD-tree for fast spatial queries
    tree = cKDTree(label_coords_mm)

    # For each source, find all atlas voxels within radius
    for source_idx, source_coord in enumerate(source_coords_mm):
        # Query tree for neighbors within radius
        indices = tree.query_ball_point(source_coord, r=radius_mm)

        # Get ROIs of nearby voxels
        nearby_rois = set()
        for idx in indices:
            label_id = label_ids[idx]
            if label_id in label_to_roi:
                nearby_rois.add(label_to_roi[label_id])

        # Assign source to all nearby ROIs
        for roi_name in nearby_rois:
            if roi_name not in roi_source_mapping:
                roi_source_mapping[roi_name] = []
            roi_source_mapping[roi_name].append(source_idx)

    return roi_source_mapping
