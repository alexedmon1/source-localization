"""
Atlas Lookup - Coordinate-to-Region Mapping

Maps source coordinates to anatomical regions AFTER statistical analysis,
following the SPM-LORETA paradigm.

This is the opposite of the ROI-first approach:
- ROI-first: Assign sources to ROIs, then analyze
- Atlas lookup: Analyze sources, then interpret coordinates using atlas

The key insight is that we don't force sources into ROI boundaries before
analysis - we use the atlas as an interpretive tool after the fact.

Author: Claude Code
Date: 2026-01-26
"""

import numpy as np
import nibabel as nib
from scipy.spatial import cKDTree
from scipy.ndimage import distance_transform_edt
from typing import Tuple, Optional, Dict, List, Any, Union
from dataclasses import dataclass
from pathlib import Path
import json
import logging

logger = logging.getLogger(__name__)


@dataclass
class RegionInfo:
    """Information about an anatomical region."""
    label_id: int
    name: str
    abbreviation: Optional[str] = None
    hemisphere: Optional[str] = None  # 'L', 'R', or 'bilateral'
    category: Optional[str] = None  # 'cortical', 'subcortical', 'white_matter'
    centroid_mm: Optional[np.ndarray] = None
    volume_mm3: Optional[float] = None


@dataclass
class CoordinateLookup:
    """Result of looking up a coordinate in the atlas."""
    coord_mm: np.ndarray
    region: Optional[RegionInfo]
    distance_to_region_mm: float
    probability: float  # 1.0 if inside, decreases with distance
    nearby_regions: List[Tuple[RegionInfo, float]]  # (region, distance) pairs


class AtlasLookup:
    """
    Atlas-based coordinate lookup for source localization results.

    Provides coordinate-to-region mapping for interpreting source-level
    statistical results. Supports probabilistic assignment when coordinates
    fall between regions.

    Attributes:
        atlas_data: 3D array of region labels
        affine: Voxel-to-mm transformation
        regions: Dict mapping label_id to RegionInfo
    """

    def __init__(
        self,
        atlas_path: str,
        roi_mapping_path: Optional[str] = None,
        apply_10x_correction: bool = True
    ):
        """
        Initialize atlas lookup.

        Args:
            atlas_path: Path to atlas NIfTI file
            roi_mapping_path: Optional path to ROI mapping JSON file
            apply_10x_correction: Apply 10x voxel size correction for mouse atlas
        """
        logger.info(f"Loading atlas from {atlas_path}")

        # Load atlas
        nii = nib.load(atlas_path)
        self.atlas_data = np.asarray(nii.dataobj)
        self.affine = nii.affine.copy()

        # Apply 10x correction for mouse atlas
        if apply_10x_correction:
            logger.info("Applying 10x voxel size correction")
            self.affine[:3, :3] /= 10.0

        # Compute inverse affine for mm-to-voxel
        self.affine_inv = np.linalg.inv(self.affine)

        # Get voxel size
        self.voxel_size_mm = np.abs(np.diag(self.affine)[:3])

        # Load ROI mapping if provided
        self.regions: Dict[int, RegionInfo] = {}
        if roi_mapping_path is not None:
            self._load_roi_mapping(roi_mapping_path)
        else:
            # Create basic regions from unique labels
            self._create_basic_regions()

        # Precompute region centroids and build spatial index
        self._precompute_region_data()

        logger.info(f"Atlas loaded: {len(self.regions)} regions, "
                   f"shape {self.atlas_data.shape}")

    def _load_roi_mapping(self, path: str):
        """Load ROI mapping from JSON file."""
        with open(path, 'r') as f:
            mapping = json.load(f)

        # Handle both dict and list formats for ROI data
        rois_data = mapping.get('rois', mapping.get('regions', []))

        # If it's a dict (key -> entry), get the values
        if isinstance(rois_data, dict):
            entries = rois_data.values()
        else:
            entries = rois_data

        for entry in entries:
            label_id = entry.get('label_id', entry.get('id'))
            if label_id is None:
                continue

            self.regions[label_id] = RegionInfo(
                label_id=label_id,
                name=entry.get('name', f'Region_{label_id}'),
                abbreviation=entry.get('abbreviation'),
                hemisphere=entry.get('hemisphere'),
                category=entry.get('category')
            )

    def _create_basic_regions(self):
        """Create basic region info from atlas labels."""
        unique_labels = np.unique(self.atlas_data)
        for label_id in unique_labels:
            if label_id == 0:  # Skip background
                continue
            self.regions[int(label_id)] = RegionInfo(
                label_id=int(label_id),
                name=f'Region_{label_id}'
            )

    def _precompute_region_data(self):
        """Precompute centroids, volumes, and spatial index for regions."""
        self._region_centroids_mm = {}
        self._region_coords_mm = {}

        for label_id, region in self.regions.items():
            mask = self.atlas_data == label_id
            if not np.any(mask):
                continue

            # Get voxel coordinates
            voxel_coords = np.array(np.where(mask)).T  # (n_voxels, 3)
            coords_mm = nib.affines.apply_affine(self.affine, voxel_coords)

            # Compute centroid
            centroid_mm = np.mean(coords_mm, axis=0)
            region.centroid_mm = centroid_mm
            self._region_centroids_mm[label_id] = centroid_mm
            self._region_coords_mm[label_id] = coords_mm

            # Compute volume
            voxel_volume = np.prod(self.voxel_size_mm)
            region.volume_mm3 = len(voxel_coords) * voxel_volume

        # Build KD-tree of region centroids for fast lookup
        if self._region_centroids_mm:
            self._centroid_labels = list(self._region_centroids_mm.keys())
            centroids = np.array([self._region_centroids_mm[l] for l in self._centroid_labels])
            self._centroid_tree = cKDTree(centroids)
        else:
            self._centroid_labels = []
            self._centroid_tree = None

    def coord_to_voxel(self, coord_mm: np.ndarray) -> np.ndarray:
        """Convert mm coordinate to voxel indices."""
        voxel = nib.affines.apply_affine(self.affine_inv, coord_mm)
        return np.round(voxel).astype(int)

    def voxel_to_coord(self, voxel: np.ndarray) -> np.ndarray:
        """Convert voxel indices to mm coordinate."""
        return nib.affines.apply_affine(self.affine, voxel)

    def lookup_coordinate(
        self,
        coord_mm: np.ndarray,
        search_radius_mm: float = 5.0,
        return_nearby: bool = True
    ) -> CoordinateLookup:
        """
        Look up anatomical region for a coordinate.

        Args:
            coord_mm: (3,) coordinate in mm
            search_radius_mm: Radius to search for nearby regions
            return_nearby: Whether to return nearby regions

        Returns:
            CoordinateLookup with region information
        """
        coord_mm = np.asarray(coord_mm)
        voxel = self.coord_to_voxel(coord_mm)

        # Check if coordinate is within atlas bounds
        in_bounds = np.all((voxel >= 0) & (voxel < self.atlas_data.shape))

        region = None
        distance = 0.0
        probability = 0.0

        if in_bounds:
            label_id = int(self.atlas_data[tuple(voxel)])
            if label_id > 0 and label_id in self.regions:
                region = self.regions[label_id]
                probability = 1.0

        # If not in a region, find nearest region
        nearby_regions = []
        if region is None or return_nearby:
            nearby_regions = self._find_nearby_regions(coord_mm, search_radius_mm)

            if region is None and nearby_regions:
                # Use nearest region
                nearest_region, nearest_dist = nearby_regions[0]
                region = nearest_region
                distance = nearest_dist
                # Probability decreases with distance
                probability = max(0, 1.0 - (nearest_dist / search_radius_mm))

        return CoordinateLookup(
            coord_mm=coord_mm,
            region=region,
            distance_to_region_mm=distance,
            probability=probability,
            nearby_regions=nearby_regions
        )

    def _find_nearby_regions(
        self,
        coord_mm: np.ndarray,
        radius_mm: float
    ) -> List[Tuple[RegionInfo, float]]:
        """Find regions within radius of coordinate."""
        if self._centroid_tree is None:
            return []

        # Query nearby centroids
        indices = self._centroid_tree.query_ball_point(coord_mm, radius_mm * 2)

        nearby = []
        for idx in indices:
            label_id = self._centroid_labels[idx]
            region = self.regions[label_id]

            # Compute actual distance to nearest voxel in region
            if label_id in self._region_coords_mm:
                region_coords = self._region_coords_mm[label_id]
                distances = np.linalg.norm(region_coords - coord_mm, axis=1)
                min_dist = np.min(distances)
            else:
                min_dist = np.linalg.norm(region.centroid_mm - coord_mm)

            if min_dist <= radius_mm:
                nearby.append((region, min_dist))

        # Sort by distance
        nearby.sort(key=lambda x: x[1])
        return nearby

    def lookup_coordinates(
        self,
        coords_mm: np.ndarray,
        **kwargs
    ) -> List[CoordinateLookup]:
        """
        Look up regions for multiple coordinates.

        Args:
            coords_mm: (n, 3) array of coordinates
            **kwargs: Additional arguments passed to lookup_coordinate

        Returns:
            List of CoordinateLookup objects
        """
        return [self.lookup_coordinate(c, **kwargs) for c in coords_mm]

    def coords_to_region_labels(
        self,
        coords_mm: np.ndarray,
        search_radius_mm: float = 3.0
    ) -> List[Optional[str]]:
        """
        Get region labels for coordinates (simple interface).

        Args:
            coords_mm: (n, 3) array of coordinates
            search_radius_mm: Radius to search for nearby regions

        Returns:
            List of region names (None if not found)
        """
        lookups = self.lookup_coordinates(coords_mm, search_radius_mm=search_radius_mm)
        return [l.region.name if l.region else None for l in lookups]

    def get_region_mask(self, region_name: str) -> np.ndarray:
        """Get binary mask for a region by name."""
        for label_id, region in self.regions.items():
            if region.name == region_name or region.abbreviation == region_name:
                return self.atlas_data == label_id
        raise ValueError(f"Region '{region_name}' not found")

    def get_region_coordinates(self, region_name: str) -> np.ndarray:
        """Get all voxel coordinates (in mm) for a region."""
        mask = self.get_region_mask(region_name)
        voxels = np.array(np.where(mask)).T
        return nib.affines.apply_affine(self.affine, voxels)

    def summarize_sources_by_region(
        self,
        source_coords_mm: np.ndarray,
        source_values: np.ndarray,
        search_radius_mm: float = 2.0
    ) -> Dict[str, Dict[str, Any]]:
        """
        Summarize source values by anatomical region.

        This is the atlas interpretation step - given source-level results,
        summarize by brain region.

        Args:
            source_coords_mm: (n_sources, 3) coordinates
            source_values: (n_sources,) statistical values
            search_radius_mm: Radius for region assignment

        Returns:
            Dict mapping region_name to summary statistics
        """
        lookups = self.lookup_coordinates(source_coords_mm, search_radius_mm=search_radius_mm)

        # Group sources by region
        region_sources: Dict[str, List[Tuple[int, float]]] = {}

        for i, lookup in enumerate(lookups):
            if lookup.region is not None:
                name = lookup.region.name
                if name not in region_sources:
                    region_sources[name] = []
                region_sources[name].append((i, source_values[i]))

        # Compute summary statistics
        summary = {}
        for region_name, sources in region_sources.items():
            indices, values = zip(*sources)
            values = np.array(values)

            summary[region_name] = {
                'n_sources': len(values),
                'mean': float(np.mean(values)),
                'std': float(np.std(values)),
                'min': float(np.min(values)),
                'max': float(np.max(values)),
                'peak_value': float(np.max(values)),
                'peak_index': int(indices[np.argmax(values)]),
                'peak_coord_mm': source_coords_mm[indices[np.argmax(values)]].tolist(),
            }

        return summary

    def label_clusters(
        self,
        clusters: List[Any],  # List of SourceCluster objects
        search_radius_mm: float = 2.0
    ) -> List[Any]:
        """
        Add region labels to clusters.

        Args:
            clusters: List of SourceCluster objects
            search_radius_mm: Search radius for label assignment

        Returns:
            Same clusters with region_labels populated
        """
        for cluster in clusters:
            # Look up each source in the cluster
            labels = self.coords_to_region_labels(
                cluster.coords_mm,
                search_radius_mm=search_radius_mm
            )
            cluster.region_labels = labels

            # Also look up peak
            peak_lookup = self.lookup_coordinate(cluster.peak_coord_mm)
            if peak_lookup.region:
                cluster.peak_region = peak_lookup.region.name

        return clusters

    def label_peaks(
        self,
        peaks: List[Any],  # List of SourcePeak objects
        search_radius_mm: float = 2.0
    ) -> List[Any]:
        """
        Add region labels to peaks.

        Args:
            peaks: List of SourcePeak objects
            search_radius_mm: Search radius for label assignment

        Returns:
            Same peaks with region_label populated
        """
        for peak in peaks:
            lookup = self.lookup_coordinate(
                peak.coord_mm,
                search_radius_mm=search_radius_mm
            )
            if lookup.region:
                peak.region_label = lookup.region.name

        return peaks

    def get_cortical_regions(self) -> List[RegionInfo]:
        """Get list of cortical regions."""
        return [r for r in self.regions.values() if r.category == 'cortical']

    def get_subcortical_regions(self) -> List[RegionInfo]:
        """Get list of subcortical regions."""
        return [r for r in self.regions.values() if r.category == 'subcortical']

    def export_region_table(self) -> str:
        """Export region information as formatted table."""
        lines = ["| ID | Name | Hemisphere | Category | Volume (mm³) |"]
        lines.append("|---|---|---|---|---|")

        for label_id in sorted(self.regions.keys()):
            region = self.regions[label_id]
            vol = f"{region.volume_mm3:.2f}" if region.volume_mm3 else "N/A"
            lines.append(
                f"| {label_id} | {region.name} | {region.hemisphere or 'N/A'} | "
                f"{region.category or 'N/A'} | {vol} |"
            )

        return "\n".join(lines)

    def __repr__(self) -> str:
        return f"AtlasLookup(n_regions={len(self.regions)}, shape={self.atlas_data.shape})"
