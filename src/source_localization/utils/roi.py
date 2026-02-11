#!/usr/bin/env python3
"""
ROI Utilities Module

**Created:** 2025-11-20
**Last Updated:** 2025-11-20

Utilities for ROI labeling and proximity-based mapping.
"""

import xml.etree.ElementTree as ET
import json
import numpy as np
import nibabel as nib
from pathlib import Path
from scipy.spatial import cKDTree


def parse_roi_json(json_path):
    """
    Parse ROI mapping JSON file to get ROI labels.

    Parameters
    ----------
    json_path : str or Path
        Path to roi_mapping.json file

    Returns
    -------
    roi_labels : dict
        Dictionary mapping ROI ID (int) -> ROI name (str)
    roi_abbrev : dict
        Dictionary mapping ROI ID (int) -> abbreviation (str)
    roi_colors : dict
        Dictionary mapping ROI ID (int) -> RGB color tuple
    """
    with open(json_path, 'r') as f:
        data = json.load(f)

    roi_labels = {}
    roi_abbrev = {}
    roi_colors = {}

    for roi_id_str, roi_info in data['rois'].items():
        roi_id = int(roi_id_str)
        roi_labels[roi_id] = roi_info['name']
        roi_abbrev[roi_id] = roi_info['abbreviation']
        roi_colors[roi_id] = tuple(roi_info['color_rgb'])

    return roi_labels, roi_abbrev, roi_colors


def parse_roi_xml(xml_path):
    """
    Parse Antwerp Atlas XML file to get ROI labels.

    Parameters
    ----------
    xml_path : str or Path
        Path to Atlas XML file

    Returns
    -------
    roi_labels : dict
        Dictionary mapping ROI ID (int) -> ROI name (str)
    roi_abbrev : dict
        Dictionary mapping ROI ID (int) -> abbreviation (str)
    roi_colors : dict
        Dictionary mapping ROI ID (int) -> RGB color tuple
    """
    tree = ET.parse(xml_path)
    root = tree.getroot()

    roi_labels = {}
    roi_abbrev = {}
    roi_colors = {}

    for area in root.findall('Area'):
        roi_id = int(area.get('value'))
        roi_name = area.get('name')
        abbrev = area.get('abbrev')

        # Get color
        color_elem = area.find('color')
        if color_elem is not None:
            r = int(color_elem.get('r', 0))
            g = int(color_elem.get('g', 0))
            b = int(color_elem.get('b', 0))
            roi_colors[roi_id] = (r, g, b)

        roi_labels[roi_id] = roi_name
        roi_abbrev[roi_id] = abbrev

    return roi_labels, roi_abbrev, roi_colors


def load_proximity_mapping(proximity_json_path):
    """
    Load pre-computed proximity-based ROI mapping.

    Parameters
    ----------
    proximity_json_path : str or Path
        Path to proximity mapping JSON file

    Returns
    -------
    roi_vertex_ids : dict
        Dictionary mapping ROI ID -> array of source indices
    roi_names : dict
        Dictionary mapping ROI ID -> ROI name
    radius_mm : float
        Radius used for proximity mapping
    """
    with open(proximity_json_path, 'r') as f:
        prox_map = json.load(f)

    source_roi_sets = prox_map['source_roi_sets']
    roi_names = prox_map.get('roi_names', {})
    radius_mm = prox_map.get('radius_mm', 1.0)

    # Convert source-centric to ROI-centric mapping
    roi_vertex_ids = {}
    for src_idx, roi_set in enumerate(source_roi_sets):
        if not roi_set:
            continue
        for roi_id in roi_set:
            if roi_id not in roi_vertex_ids:
                roi_vertex_ids[roi_id] = []
            roi_vertex_ids[roi_id].append(src_idx)

    # Convert lists to numpy arrays and ensure int keys
    roi_vertex_ids = {
        int(roi_id): np.array(verts, dtype=int)
        for roi_id, verts in roi_vertex_ids.items()
    }

    # Convert string keys to int for roi_names
    roi_names_map = {int(k): v for k, v in roi_names.items()}

    return roi_vertex_ids, roi_names_map, radius_mm


def compute_proximity_mapping(source_coords_mm, atlas_img, radius_mm=1.0):
    """
    Compute proximity-based ROI mapping (sources within radius_mm of any ROI voxel).

    This is more robust than exact mapping as it handles sources that fall
    slightly outside atlas voxels due to discretization.

    **IMPORTANT:** Applies 10× voxel correction for Antwerp Atlas.

    Parameters
    ----------
    source_coords_mm : ndarray
        Source coordinates in mm (n_sources, 3)
    atlas_img : nibabel.Nifti1Image
        Atlas image with ROI labels
    radius_mm : float
        Radius in mm for proximity search (default: 1.0)

    Returns
    -------
    roi_vertex_ids : dict
        Dictionary mapping ROI ID -> array of source indices
    coverage_pct : float
        Percentage of sources assigned to at least one ROI
    """
    atlas_data = atlas_img.get_fdata().astype(int)

    # Get all non-zero voxel coordinates and their labels
    roi_voxels = np.argwhere(atlas_data > 0)  # (n_roi_voxels, 3)
    roi_labels_flat = atlas_data[roi_voxels[:, 0], roi_voxels[:, 1], roi_voxels[:, 2]]

    # Convert voxel coordinates to mm with 10× correction for Antwerp Atlas
    # Import here to avoid circular dependency
    try:
        from atlas_utils import get_true_affine
        affine_corrected = get_true_affine(atlas_img)
    except ImportError:
        # Fallback: manual correction
        affine_corrected = atlas_img.affine.copy()
        affine_corrected[:3, :3] /= 10.0  # Scale
        affine_corrected[:3, 3] /= 10.0   # Translation

    roi_voxels_mm = nib.affines.apply_affine(affine_corrected, roi_voxels)

    # Build KD-tree for efficient proximity search
    tree = cKDTree(roi_voxels_mm)

    # For each source, find all ROI voxels within radius
    source_roi_sets = []
    for src_coord in source_coords_mm:
        indices = tree.query_ball_point(src_coord, r=radius_mm)
        if indices:
            # Get unique ROI labels within radius
            nearby_rois = set(roi_labels_flat[indices].tolist())
            nearby_rois.discard(0)  # Remove background label
            source_roi_sets.append(nearby_rois)
        else:
            source_roi_sets.append(set())

    # Convert to ROI-centric mapping
    roi_vertex_ids = {}
    for src_idx, roi_set in enumerate(source_roi_sets):
        for roi_id in roi_set:
            if roi_id not in roi_vertex_ids:
                roi_vertex_ids[roi_id] = []
            roi_vertex_ids[roi_id].append(src_idx)

    # Convert to numpy arrays
    roi_vertex_ids = {
        int(roi_id): np.array(verts, dtype=int)
        for roi_id, verts in roi_vertex_ids.items()
    }

    # Calculate coverage
    n_assigned = sum(1 for roi_set in source_roi_sets if roi_set)
    coverage_pct = 100.0 * n_assigned / len(source_coords_mm)

    return roi_vertex_ids, coverage_pct


def extract_roi_timeseries_proximity(source_power, roi_vertex_ids, method='mean'):
    """
    Extract ROI time series using proximity-based mapping.

    Sources may belong to multiple ROIs (within radius of multiple regions).

    Parameters
    ----------
    source_power : ndarray
        Source power time series (n_sources, n_times)
    roi_vertex_ids : dict
        Dictionary mapping ROI ID -> array of source indices
    method : str
        Aggregation method: 'mean' (default), 'median', 'max'

    Returns
    -------
    roi_timeseries : dict
        Dictionary mapping ROI ID -> timeseries (n_times,)
    roi_n_sources : dict
        Dictionary mapping ROI ID -> number of sources
    """
    roi_timeseries = {}
    roi_n_sources = {}

    for roi_id, source_indices in roi_vertex_ids.items():
        if len(source_indices) == 0:
            continue

        if method == 'mean':
            roi_timeseries[roi_id] = source_power[source_indices, :].mean(axis=0)
        elif method == 'median':
            roi_timeseries[roi_id] = np.median(source_power[source_indices, :], axis=0)
        elif method == 'max':
            roi_timeseries[roi_id] = source_power[source_indices, :].max(axis=0)
        else:
            raise ValueError(f"Unknown method: {method}")

        roi_n_sources[roi_id] = len(source_indices)

    return roi_timeseries, roi_n_sources
