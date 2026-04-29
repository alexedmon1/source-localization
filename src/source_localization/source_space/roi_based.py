"""ROI-Based Source Space Construction.

Places sources strategically within anatomical regions of interest (ROIs)
rather than using a uniform grid. This approach:

1. Reduces inter-ROI collinearity by ensuring spatial separation
2. Improves forward matrix conditioning (fewer sources)
3. Provides direct ROI-to-source mapping (simplified interpretation)
4. Maintains regional coverage with minimal redundancy

Placement Strategies:
- **centroid**: Single source at ROI center of mass
- **pca**: Sources spread throughout ROI volume using PCA axes
  - Uses all 3 principal components for 3D distribution
  - Spread scales with actual ROI dimensions (~80% of extent)
- **random**: Random sampling from ROI voxels
- **stratified**: K-means clustering to maximize spatial coverage
  - Best for connectivity analysis (sources spread throughout volume)

**Created:** 2025-11-26
**Last Updated:** 2025-01-27
"""

import json
import mne
import nibabel as nib
import numpy as np
from pathlib import Path
from scipy.ndimage import center_of_mass
from sklearn.decomposition import PCA


def _resolve_roi_mapping_path(config):
    """Resolve roi_mapping path (relative to package data dir if not absolute)."""
    package_dir = Path(__file__).parent.parent
    p = config['inputs'].get('roi_mapping')
    if p is None:
        return None
    p = Path(p)
    if not p.is_absolute():
        p = package_dir / p
    return p if p.exists() else None


def _get_lr_pairs(unique_rois, roi_mapping_path):
    """Return (pairs, unpaired) from a roi_mapping.json declaring `n_rois_per_hemisphere`.

    A "pair" is (L_id, R_id) where R_id == L_id + n_rois_per_hemisphere.
    Atlases without `n_rois_per_hemisphere` (or with no L/R structure) return
    ([], list(unique_rois)) and the caller falls back to per-ROI placement.
    """
    if roi_mapping_path is None:
        return [], [int(x) for x in unique_rois]
    try:
        with open(roi_mapping_path) as f:
            mapping = json.load(f)
    except Exception:
        return [], [int(x) for x in unique_rois]
    n_per_hemi = mapping.get('n_rois_per_hemisphere')
    if not n_per_hemi:
        return [], [int(x) for x in unique_rois]
    ids = set(int(x) for x in unique_rois)
    pairs = []
    paired = set()
    for left_id in range(1, n_per_hemi + 1):
        right_id = left_id + n_per_hemi
        if left_id in ids and right_id in ids:
            pairs.append((left_id, right_id))
            paired.add(left_id)
            paired.add(right_id)
    unpaired = sorted(int(x) for x in ids if int(x) not in paired)
    return pairs, unpaired


def _place_pca(roi_voxels, n_sources, spread_factor=0.8):
    """Compute PCA-based source placements for one ROI's voxel set.

    Returns list of voxel coordinates (np.ndarray, shape (3,)).
    Mirrors the original `pca` branch of `create_source_space` exactly so
    L-side placements are bitwise identical to the legacy code path.
    """
    if n_sources == 1 or len(roi_voxels) < 3:
        centroid_vox = roi_voxels.mean(axis=0)
        return [centroid_vox] * max(1, n_sources)

    pca = PCA(n_components=min(3, len(roi_voxels)))
    pca.fit(roi_voxels)
    centroid_vox = roi_voxels.mean(axis=0)
    roi_extents = np.std(pca.transform(roi_voxels), axis=0)

    if n_sources == 2:
        offset = spread_factor * roi_extents[0]
        return [
            centroid_vox - offset * pca.components_[0],
            centroid_vox + offset * pca.components_[0],
        ]
    if n_sources <= 4:
        n_per_axis = int(np.ceil(np.sqrt(n_sources)))
        out = []
        for i in range(n_per_axis):
            for j in range(n_per_axis):
                if len(out) >= n_sources:
                    break
                offset_1 = spread_factor * roi_extents[0] * (2 * i / (n_per_axis - 1) - 1) if n_per_axis > 1 else 0
                offset_2 = spread_factor * roi_extents[1] * (2 * j / (n_per_axis - 1) - 1) if n_per_axis > 1 else 0
                out.append(centroid_vox + offset_1 * pca.components_[0] + offset_2 * pca.components_[1])
        return out
    n_per_axis = int(np.ceil(n_sources ** (1 / 3)))
    out = []
    for i in range(n_per_axis):
        for j in range(n_per_axis):
            for k in range(n_per_axis):
                if len(out) >= n_sources:
                    break
                offset_1 = spread_factor * roi_extents[0] * (2 * i / (n_per_axis - 1) - 1) if n_per_axis > 1 else 0
                offset_2 = spread_factor * roi_extents[1] * (2 * j / (n_per_axis - 1) - 1) if n_per_axis > 1 else 0
                offset_3 = spread_factor * roi_extents[2] * (2 * k / (n_per_axis - 1) - 1) if n_per_axis > 1 and len(pca.components_) > 2 else 0
                v = centroid_vox + offset_1 * pca.components_[0] + offset_2 * pca.components_[1]
                if len(pca.components_) > 2:
                    v = v + offset_3 * pca.components_[2]
                out.append(v)
    return out


def _mirror_voxel_x(vox_coord, affine):
    """Mirror a voxel coordinate across the world X=0 plane.

    Converts voxel→mm via `affine`, negates X, converts back. Used to derive
    R-side dipole positions from L-side placements so L/R coords are exact
    mirrors regardless of small voxel-mask asymmetries.
    """
    mm = nib.affines.apply_affine(affine, np.asarray(vox_coord))
    mm_mirror = mm.copy()
    mm_mirror[0] = -mm_mirror[0]
    return nib.affines.apply_affine(np.linalg.inv(affine), mm_mirror)


def load_roi_categories(config):
    """
    Load ROI category definitions from roi_mapping.json.

    Returns dict mapping category names to lists of ROI IDs.
    """
    from pathlib import Path
    package_dir = Path(__file__).parent.parent

    roi_mapping_path = config['inputs'].get('roi_mapping', 'data/atlas/roi_mapping.json')
    if not Path(roi_mapping_path).is_absolute():
        roi_mapping_path = package_dir / roi_mapping_path

    with open(roi_mapping_path, 'r') as f:
        roi_data = json.load(f)

    return roi_data.get('categories', {})


def filter_rois_by_category(unique_rois, config):
    """
    Filter ROI list to include only specified categories.

    Parameters
    ----------
    unique_rois : ndarray
        All unique ROI IDs from atlas
    config : dict
        Config with optional 'source_space.roi_based.include_categories'

    Returns
    -------
    filtered_rois : ndarray
        ROI IDs that match the specified categories
    """
    include_categories = config['source_space']['roi_based'].get('include_categories', None)

    if include_categories is None:
        # No filtering, return all ROIs
        return unique_rois

    # Load category definitions
    categories = load_roi_categories(config)

    # Collect ROI IDs from specified categories
    included_roi_ids = set()
    for category in include_categories:
        if category in categories:
            included_roi_ids.update(categories[category])
        else:
            print(f"    Warning: Unknown category '{category}', skipping")

    # Filter unique_rois
    filtered_rois = np.array([roi for roi in unique_rois if roi in included_roi_ids])

    print(f"    ROI filtering: categories={include_categories}")
    print(f"    ROIs before filter: {len(unique_rois)}, after: {len(filtered_rois)}")

    return filtered_rois


def create_source_space(config, previous_outputs):
    """
    Create source space with sources placed strategically within ROIs.

    Supports two modes:

    1. **Adaptive allocation** (recommended): Sources allocated proportionally
       to ROI volume, with a cap on total sources (e.g., 200 max)
    2. **Fixed allocation**: Same number of sources per ROI

    Parameters
    ----------
    config : dict
        Must contain:
        - inputs['brain_volume']: Atlas NIfTI path
        - inputs['brain_labels']: ROI labels NIfTI path (optional, defaults to brain_volume)
        - source_space['roi_based']['placement_strategy']:
            'centroid', 'pca', or 'random'
        - source_space['roi_based']['adaptive_sources']: bool
            If True, use adaptive allocation; if False, use fixed
        - source_space['roi_based']['max_total_sources']: int
            Total source budget (only for adaptive mode)
        - source_space['roi_based']['sources_per_roi']: int
            Sources per ROI (only for fixed mode)
    previous_outputs : dict
        Not used for ROI-based source space

    Returns
    -------
    src : mne.SourceSpaces
        MNE source space object
    source_coords_mm : ndarray (n_sources, 3)
        Source coordinates in RAS mm
    n_sources : int
        Total number of sources created
    """
    print("\n" + "="*80)
    print("STEP 3: ROI-Based Source Space Construction")
    print("="*80)

    # Load atlas
    # Get package directory to resolve relative paths
    from pathlib import Path
    package_dir = Path(__file__).parent.parent

    atlas_path = config['inputs']['brain_volume']
    # Resolve to absolute path if relative
    if not Path(atlas_path).is_absolute():
        atlas_path = package_dir / atlas_path

    atlas_nii = nib.load(atlas_path)
    atlas_data = atlas_nii.get_fdata()

    # Apply 10× voxel correction (CRITICAL for mouse brain atlas)
    from ..utils.atlas import get_true_affine
    affine = get_true_affine(atlas_nii)

    # Get ROI labels (use brain_labels if specified, otherwise use brain_volume)
    roi_labels_path = config['inputs'].get('brain_labels', str(atlas_path))
    # Resolve to absolute path if relative
    if not Path(roi_labels_path).is_absolute():
        roi_labels_path = package_dir / roi_labels_path

    if str(roi_labels_path) != str(atlas_path):
        roi_labels_nii = nib.load(roi_labels_path)
        roi_labels_data = roi_labels_nii.get_fdata()
    else:
        roi_labels_data = atlas_data

    # Get unique ROIs (excluding background = 0)
    unique_rois = np.unique(roi_labels_data[roi_labels_data > 0])

    # Apply ROI category filtering if specified
    unique_rois = filter_rois_by_category(unique_rois, config)
    n_rois = len(unique_rois)

    if n_rois == 0:
        raise ValueError("No ROIs remaining after filtering. Check 'include_categories' config.")

    # Get placement parameters
    placement_strategy = config['source_space']['roi_based'].get(
        'placement_strategy', 'pca'
    )

    # Adaptive source allocation based on ROI size
    adaptive_sources = config['source_space']['roi_based'].get('adaptive_sources', False)
    max_total_sources = config['source_space']['roi_based'].get('max_total_sources', 200)

    if adaptive_sources:
        # Compute ROI sizes (number of voxels)
        roi_sizes = {}
        for roi_id in unique_rois:
            roi_mask = roi_labels_data == roi_id
            roi_sizes[roi_id] = np.sum(roi_mask)

        total_voxels = sum(roi_sizes.values())

        # Allocate sources proportionally to ROI size
        roi_source_counts = {}
        for roi_id, size in roi_sizes.items():
            # Proportional allocation (at least 1 source per ROI)
            proportion = size / total_voxels
            allocated = max(1, int(np.round(proportion * max_total_sources)))
            roi_source_counts[roi_id] = allocated

        # Adjust to hit max_total_sources exactly
        total_allocated = sum(roi_source_counts.values())
        if total_allocated > max_total_sources:
            # Scale down large ROIs to fit budget
            scale_factor = max_total_sources / total_allocated
            for roi_id in roi_source_counts:
                roi_source_counts[roi_id] = max(1, int(np.round(roi_source_counts[roi_id] * scale_factor)))

        print(f"\n  ROI-based source placement (ADAPTIVE):")
        print(f"    Number of ROIs: {n_rois}")
        print(f"    Max total sources: {max_total_sources}")
        print(f"    Placement strategy: {placement_strategy}")
        print(f"    Source allocation: proportional to ROI volume")
    else:
        # Fixed sources per ROI
        sources_per_roi = config['source_space']['roi_based']['sources_per_roi']
        roi_source_counts = {roi_id: sources_per_roi for roi_id in unique_rois}

        print(f"\n  ROI-based source placement (FIXED):")
        print(f"    Number of ROIs: {n_rois}")
        print(f"    Sources per ROI: {sources_per_roi}")
        print(f"    Placement strategy: {placement_strategy}")

    source_coords_voxel = []
    roi_assignments = []

    # ------------------------------------------------------------------
    # L/R symmetry enforcement (post-regfix bug discovered 2026-04-29):
    # Without this, the per-ROI PCA placement can produce L-vs-R dipole
    # counts that differ by 1-2 (rounding of proportional allocation when
    # L and R voxel masks are not bit-identical) and dipole positions
    # that are not mirror-symmetric (PCA axis sign ambiguity per side).
    # That bakes a hemispheric asymmetry into the source space and was
    # producing spurious KO-vs-WT lateralization signals at small
    # midline ROIs (e.g. Retrosplenial, Hippocampus_Ant).
    #
    # Fix: if the atlas roi_mapping declares `n_rois_per_hemisphere`
    # (Allen32 = 16), pair L (id 1..N) with R (id+N). For each pair,
    # equalize source counts (max of L,R) and compute the L-side
    # placement, then mirror across world X=0 to derive R-side coords.
    # Unpaired ROIs and atlases without this metadata fall back to the
    # original per-ROI logic.
    # ------------------------------------------------------------------
    enforce_lr_symmetry = config['source_space']['roi_based'].get(
        'enforce_lr_symmetry', True
    )
    if enforce_lr_symmetry:
        roi_mapping_path = _resolve_roi_mapping_path(config)
        lr_pairs, unpaired_ids = _get_lr_pairs(unique_rois, roi_mapping_path)
    else:
        lr_pairs, unpaired_ids = [], [int(x) for x in unique_rois]

    if lr_pairs:
        # Equalize per-pair source counts to max(n_L, n_R) so neither side
        # is undersampled relative to the other.
        for left_id, right_id in lr_pairs:
            n_pair = max(roi_source_counts[left_id], roi_source_counts[right_id])
            roi_source_counts[left_id] = n_pair
            roi_source_counts[right_id] = n_pair

        # PCA-mirror placement: compute L, mirror to R.
        if placement_strategy == 'pca':
            for left_id, right_id in lr_pairs:
                left_voxels = np.argwhere(roi_labels_data == left_id)
                if len(left_voxels) == 0:
                    print(f"    Warning: ROI {left_id} has no voxels, skipping pair ({left_id}, {right_id})")
                    continue
                n = roi_source_counts[left_id]
                left_placements = _place_pca(left_voxels, n)
                for vox_l in left_placements:
                    source_coords_voxel.append(vox_l)
                    roi_assignments.append(left_id)
                    source_coords_voxel.append(_mirror_voxel_x(vox_l, affine))
                    roi_assignments.append(right_id)
        elif placement_strategy == 'centroid':
            for left_id, right_id in lr_pairs:
                left_voxels = np.argwhere(roi_labels_data == left_id)
                if len(left_voxels) == 0:
                    continue
                centroid = left_voxels.mean(axis=0)
                source_coords_voxel.append(centroid)
                roi_assignments.append(left_id)
                source_coords_voxel.append(_mirror_voxel_x(centroid, affine))
                roi_assignments.append(right_id)
        else:
            print(f"    Note: enforce_lr_symmetry not implemented for placement_strategy='{placement_strategy}'; falling back to per-ROI logic")
            unpaired_ids = sorted(set(unpaired_ids) | {i for p in lr_pairs for i in p})
            lr_pairs = []

    # Place sources in each (remaining / unpaired) ROI using legacy logic
    paired_ids = {i for p in lr_pairs for i in p}
    for roi_id in unique_rois:
        if int(roi_id) in paired_ids:
            continue
        # Get voxel coordinates for this ROI
        roi_mask = roi_labels_data == roi_id
        roi_voxels = np.argwhere(roi_mask)

        if len(roi_voxels) == 0:
            print(f"    Warning: ROI {int(roi_id)} has no voxels, skipping")
            continue

        # Get number of sources for this ROI
        n_sources_this_roi = roi_source_counts[roi_id]

        if n_sources_this_roi == 1 or placement_strategy == 'centroid':
            # Place single source at centroid
            centroid_vox = roi_voxels.mean(axis=0)
            source_coords_voxel.append(centroid_vox)
            roi_assignments.append(roi_id)

        elif placement_strategy == 'pca':
            # Use PCA to spread sources throughout the ROI volume using all 3 axes
            if len(roi_voxels) < 3:
                # Too few voxels for PCA, use centroid repeated
                centroid_vox = roi_voxels.mean(axis=0)
                for i in range(n_sources_this_roi):
                    source_coords_voxel.append(centroid_vox)
                    roi_assignments.append(roi_id)
            else:
                # Fit PCA to get principal axes
                pca = PCA(n_components=min(3, len(roi_voxels)))
                pca.fit(roi_voxels)

                centroid_vox = roi_voxels.mean(axis=0)

                # Get ROI extent along each PCA axis (standard deviations)
                roi_projected = pca.transform(roi_voxels)
                roi_extents = np.std(roi_projected, axis=0)  # Spread in each direction

                # Scale spread to cover ~80% of ROI extent (±2 std covers ~95%)
                spread_factor = 0.8

                if n_sources_this_roi == 1:
                    source_coords_voxel.append(centroid_vox)
                    roi_assignments.append(roi_id)
                elif n_sources_this_roi == 2:
                    # Place along primary axis
                    offset = spread_factor * roi_extents[0]
                    source_coords_voxel.append(centroid_vox - offset * pca.components_[0])
                    source_coords_voxel.append(centroid_vox + offset * pca.components_[0])
                    roi_assignments.extend([roi_id, roi_id])
                elif n_sources_this_roi <= 4:
                    # Place in 2D plane (first 2 PCA components)
                    n_per_axis = int(np.ceil(np.sqrt(n_sources_this_roi)))
                    placed = 0
                    for i in range(n_per_axis):
                        for j in range(n_per_axis):
                            if placed >= n_sources_this_roi:
                                break
                            # Offset from center in 2D PCA space
                            offset_1 = spread_factor * roi_extents[0] * (2 * i / (n_per_axis - 1) - 1) if n_per_axis > 1 else 0
                            offset_2 = spread_factor * roi_extents[1] * (2 * j / (n_per_axis - 1) - 1) if n_per_axis > 1 else 0
                            source_vox = centroid_vox + offset_1 * pca.components_[0] + offset_2 * pca.components_[1]
                            source_coords_voxel.append(source_vox)
                            roi_assignments.append(roi_id)
                            placed += 1
                else:
                    # For more sources, use 3D grid in PCA space
                    n_per_axis = int(np.ceil(n_sources_this_roi ** (1/3)))
                    placed = 0
                    for i in range(n_per_axis):
                        for j in range(n_per_axis):
                            for k in range(n_per_axis):
                                if placed >= n_sources_this_roi:
                                    break
                                # Offset from center in 3D PCA space
                                offset_1 = spread_factor * roi_extents[0] * (2 * i / (n_per_axis - 1) - 1) if n_per_axis > 1 else 0
                                offset_2 = spread_factor * roi_extents[1] * (2 * j / (n_per_axis - 1) - 1) if n_per_axis > 1 else 0
                                offset_3 = spread_factor * roi_extents[2] * (2 * k / (n_per_axis - 1) - 1) if n_per_axis > 1 and len(pca.components_) > 2 else 0
                                source_vox = centroid_vox + offset_1 * pca.components_[0] + offset_2 * pca.components_[1]
                                if len(pca.components_) > 2:
                                    source_vox += offset_3 * pca.components_[2]
                                source_coords_voxel.append(source_vox)
                                roi_assignments.append(roi_id)
                                placed += 1

        elif placement_strategy == 'random':
            # Random sampling within ROI
            if len(roi_voxels) <= n_sources_this_roi:
                # Use all voxels
                selected = roi_voxels
            else:
                # Random sample
                indices = np.random.choice(
                    len(roi_voxels), n_sources_this_roi, replace=False
                )
                selected = roi_voxels[indices]

            for vox in selected:
                source_coords_voxel.append(vox)
                roi_assignments.append(roi_id)

        elif placement_strategy == 'stratified':
            # Use k-means clustering to find optimal source locations
            # that maximize spatial coverage within the ROI
            from sklearn.cluster import KMeans

            if len(roi_voxels) <= n_sources_this_roi:
                # Use all voxels as sources
                for vox in roi_voxels:
                    source_coords_voxel.append(vox)
                    roi_assignments.append(roi_id)
            else:
                # Run k-means to find cluster centers
                kmeans = KMeans(
                    n_clusters=n_sources_this_roi,
                    random_state=42,
                    n_init=10
                )
                kmeans.fit(roi_voxels)

                # Use cluster centers as source locations
                for center in kmeans.cluster_centers_:
                    source_coords_voxel.append(center)
                    roi_assignments.append(roi_id)

        else:
            raise ValueError(f"Unknown placement strategy: {placement_strategy}")

    source_coords_voxel = np.array(source_coords_voxel)
    roi_assignments = np.array(roi_assignments)

    # Convert to RAS mm coordinates
    source_coords_mm = nib.affines.apply_affine(affine, source_coords_voxel)

    n_sources = len(source_coords_mm)

    print(f"\n  Source placement summary:")
    print(f"    Total sources created: {n_sources}")
    print(f"    Sources/channel ratio: {n_sources/30:.2f}")
    print(f"    ROIs covered: {len(np.unique(roi_assignments))}/{n_rois}")

    # Create MNE SourceSpaces object
    source_coords_m = source_coords_mm / 1000.0  # Convert mm to m

    src = mne.SourceSpaces([{
        'rr': source_coords_m,
        'nn': np.zeros_like(source_coords_m),  # No surface normals for volumetric
        'inuse': np.ones(n_sources, dtype=int),
        'vertno': np.arange(n_sources),
        'nuse': n_sources,
        'coord_frame': mne.io.constants.FIFF.FIFFV_COORD_MRI,
        'id': 1,
        'type': 'vol',
        'np': n_sources,
        'ntri': 0,
        'tris': np.array([], dtype=np.int32).reshape(0, 3),
        'shape': atlas_data.shape,
        'mri_width': atlas_data.shape[0],
        'mri_height': atlas_data.shape[1],
        'mri_depth': atlas_data.shape[2],
        'interpolator': None,
        'roi_assignments': roi_assignments,  # Store ROI mapping (custom field)
    }])

    print(f"\n  ✓ ROI-based source space created successfully")
    print("="*80)

    return src, source_coords_mm, n_sources
