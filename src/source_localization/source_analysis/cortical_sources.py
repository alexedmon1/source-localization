"""
Cortical Source Space - Depth-Restricted Source Generation

Creates source spaces restricted to cortical/superficial regions where
EEG source localization has meaningful accuracy.

Based on validation results:
- 0-1 mm depth: 76.9% ROI accuracy, 0.87 mm localization error
- 1-2 mm depth: 36.2% accuracy, 2.31 mm error
- 2-3 mm depth: 4.1% accuracy, 4.47 mm error
- >3 mm depth: ~0% accuracy (effectively undetectable)

This module restricts sources to a configurable depth from the brain surface,
ensuring that all source estimates have reasonable spatial precision.

Author: Claude Code
Date: 2026-01-26
"""

import numpy as np
from scipy.spatial import cKDTree
from scipy.ndimage import distance_transform_edt
import nibabel as nib
from typing import Tuple, Optional, Dict, Any
import logging

logger = logging.getLogger(__name__)


class CorticalSourceSpace:
    """
    Depth-restricted source space for cortical EEG source localization.

    Creates sources only within a specified depth from the brain surface,
    where EEG source localization has meaningful spatial accuracy.

    Attributes:
        source_coords_mm: (n_sources, 3) array of source coordinates in mm
        source_depths_mm: (n_sources,) array of depth from brain surface
        max_depth_mm: Maximum depth threshold used
        brain_surface_coords: Coordinates of brain surface voxels
    """

    def __init__(
        self,
        source_coords_mm: np.ndarray,
        source_depths_mm: np.ndarray,
        max_depth_mm: float,
        brain_surface_coords: Optional[np.ndarray] = None,
        metadata: Optional[Dict[str, Any]] = None
    ):
        """
        Initialize cortical source space.

        Args:
            source_coords_mm: (n_sources, 3) source coordinates in mm
            source_depths_mm: (n_sources,) depth from surface for each source
            max_depth_mm: Maximum depth threshold
            brain_surface_coords: Optional surface coordinates for reference
            metadata: Optional metadata dict
        """
        self.source_coords_mm = np.asarray(source_coords_mm)
        self.source_depths_mm = np.asarray(source_depths_mm)
        self.max_depth_mm = max_depth_mm
        self.brain_surface_coords = brain_surface_coords
        self.metadata = metadata or {}

        self.n_sources = len(self.source_coords_mm)

        # Build KD-tree for fast spatial queries
        self._kdtree = cKDTree(self.source_coords_mm)

    @classmethod
    def from_brain_mask(
        cls,
        brain_mask: np.ndarray,
        affine: np.ndarray,
        max_depth_mm: float = 2.0,
        spacing_mm: float = 0.5,
        min_depth_mm: float = 0.0
    ) -> 'CorticalSourceSpace':
        """
        Create cortical source space from a brain mask.

        Uses distance transform to compute depth from brain surface,
        then places sources on a regular grid within the depth range.

        Args:
            brain_mask: 3D binary brain mask array
            affine: 4x4 affine transformation (voxel to mm)
            max_depth_mm: Maximum depth from surface (default: 2.0 mm)
            spacing_mm: Source grid spacing (default: 0.5 mm)
            min_depth_mm: Minimum depth from surface (default: 0.0 mm)

        Returns:
            CorticalSourceSpace instance
        """
        logger.info(f"Creating cortical source space: depth {min_depth_mm}-{max_depth_mm} mm, "
                   f"spacing {spacing_mm} mm")

        # Compute distance transform (distance to nearest background voxel)
        # This gives us depth from the brain surface for interior voxels
        distance_voxels = distance_transform_edt(brain_mask)

        # Get voxel size from affine
        voxel_size_mm = np.abs(np.diag(affine)[:3])
        mean_voxel_size = np.mean(voxel_size_mm)

        # Convert distance to mm
        distance_mm = distance_voxels * mean_voxel_size

        # Create depth mask: voxels within our depth range
        depth_mask = (distance_mm >= min_depth_mm) & (distance_mm <= max_depth_mm)
        cortical_mask = brain_mask & depth_mask

        # Get voxel indices within the cortical shell
        cortical_voxels = np.array(np.where(cortical_mask)).T  # (n_voxels, 3)

        if len(cortical_voxels) == 0:
            raise ValueError(f"No voxels found within depth range {min_depth_mm}-{max_depth_mm} mm")

        logger.info(f"Found {len(cortical_voxels)} voxels in cortical shell")

        # Convert voxel coordinates to mm
        cortical_coords_mm = nib.affines.apply_affine(affine, cortical_voxels)

        # Subsample to achieve desired spacing
        source_coords_mm, source_depths_mm = cls._subsample_to_spacing(
            cortical_coords_mm,
            distance_mm[cortical_mask],
            spacing_mm
        )

        # Find brain surface coordinates (for reference/visualization)
        surface_mask = (distance_mm > 0) & (distance_mm <= mean_voxel_size)
        surface_voxels = np.array(np.where(surface_mask & brain_mask)).T
        surface_coords_mm = nib.affines.apply_affine(affine, surface_voxels)

        logger.info(f"Created {len(source_coords_mm)} cortical sources")

        metadata = {
            'creation_method': 'from_brain_mask',
            'max_depth_mm': max_depth_mm,
            'min_depth_mm': min_depth_mm,
            'spacing_mm': spacing_mm,
            'voxel_size_mm': voxel_size_mm.tolist(),
            'n_cortical_voxels': len(cortical_voxels),
            'n_surface_voxels': len(surface_voxels),
        }

        return cls(
            source_coords_mm=source_coords_mm,
            source_depths_mm=source_depths_mm,
            max_depth_mm=max_depth_mm,
            brain_surface_coords=surface_coords_mm,
            metadata=metadata
        )

    @classmethod
    def from_atlas(
        cls,
        atlas_path: str,
        max_depth_mm: float = 2.0,
        spacing_mm: float = 0.5,
        cortical_labels: Optional[list] = None,
        apply_10x_correction: bool = True
    ) -> 'CorticalSourceSpace':
        """
        Create cortical source space from atlas NIfTI file.

        Args:
            atlas_path: Path to atlas NIfTI file
            max_depth_mm: Maximum depth from surface
            spacing_mm: Source grid spacing
            cortical_labels: Optional list of label IDs to include (cortical ROIs)
            apply_10x_correction: Apply 10x voxel size correction for mouse atlas

        Returns:
            CorticalSourceSpace instance
        """
        logger.info(f"Loading atlas from {atlas_path}")

        nii = nib.load(atlas_path)
        atlas_data = np.asarray(nii.dataobj)
        affine = nii.affine.copy()

        # Apply 10x correction for mouse atlas (critical!)
        if apply_10x_correction:
            logger.info("Applying 10x voxel size correction for mouse atlas")
            affine[:3, :3] /= 10.0

        # Create brain mask
        if cortical_labels is not None:
            # Use only specified cortical labels
            brain_mask = np.isin(atlas_data, cortical_labels)
            logger.info(f"Using {len(cortical_labels)} cortical labels, "
                       f"{np.sum(brain_mask)} voxels")
        else:
            # Use all non-zero labels
            brain_mask = atlas_data > 0
            logger.info(f"Using all labels, {np.sum(brain_mask)} voxels")

        return cls.from_brain_mask(
            brain_mask=brain_mask,
            affine=affine,
            max_depth_mm=max_depth_mm,
            spacing_mm=spacing_mm
        )

    @staticmethod
    def _subsample_to_spacing(
        coords_mm: np.ndarray,
        depths_mm: np.ndarray,
        target_spacing_mm: float
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Subsample coordinates to achieve target spacing using grid-based approach.

        Args:
            coords_mm: (n, 3) coordinates in mm
            depths_mm: (n,) depths for each coordinate
            target_spacing_mm: Target spacing between sources

        Returns:
            Tuple of (subsampled_coords, subsampled_depths)
        """
        if len(coords_mm) == 0:
            return np.array([]).reshape(0, 3), np.array([])

        # Create a regular grid and find closest points
        min_coords = coords_mm.min(axis=0)
        max_coords = coords_mm.max(axis=0)

        # Generate grid points
        grid_points = []
        for x in np.arange(min_coords[0], max_coords[0] + target_spacing_mm, target_spacing_mm):
            for y in np.arange(min_coords[1], max_coords[1] + target_spacing_mm, target_spacing_mm):
                for z in np.arange(min_coords[2], max_coords[2] + target_spacing_mm, target_spacing_mm):
                    grid_points.append([x, y, z])

        grid_points = np.array(grid_points)

        if len(grid_points) == 0:
            return coords_mm, depths_mm

        # Build KD-tree of original coordinates
        tree = cKDTree(coords_mm)

        # Find closest original point to each grid point
        distances, indices = tree.query(grid_points, k=1)

        # Keep only grid points that have a nearby original point
        valid_mask = distances < target_spacing_mm
        valid_indices = np.unique(indices[valid_mask])

        return coords_mm[valid_indices], depths_mm[valid_indices]

    def get_sources_in_region(
        self,
        center_mm: np.ndarray,
        radius_mm: float
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Get sources within a spherical region.

        Args:
            center_mm: (3,) center coordinate
            radius_mm: Search radius

        Returns:
            Tuple of (indices, distances) for sources within region
        """
        indices = self._kdtree.query_ball_point(center_mm, radius_mm)
        if len(indices) == 0:
            return np.array([]), np.array([])

        distances = np.linalg.norm(
            self.source_coords_mm[indices] - center_mm,
            axis=1
        )
        return np.array(indices), distances

    def get_nearest_sources(
        self,
        coords_mm: np.ndarray,
        k: int = 1
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Find k nearest sources to given coordinates.

        Args:
            coords_mm: (n, 3) query coordinates
            k: Number of nearest neighbors

        Returns:
            Tuple of (distances, indices)
        """
        return self._kdtree.query(coords_mm, k=k)

    def depth_weight_mask(
        self,
        weight_by_depth: bool = True,
        depth_falloff: float = 1.0
    ) -> np.ndarray:
        """
        Generate depth-based weights for sources.

        Superficial sources get higher weight (more reliable),
        deeper sources get lower weight (less reliable).

        Args:
            weight_by_depth: Whether to apply depth weighting
            depth_falloff: Falloff rate (higher = faster falloff)

        Returns:
            (n_sources,) weight array
        """
        if not weight_by_depth:
            return np.ones(self.n_sources)

        # Linear falloff from 1.0 at surface to lower values at max depth
        # weights = 1.0 - (depth / max_depth) * falloff_factor
        weights = 1.0 - (self.source_depths_mm / self.max_depth_mm) * depth_falloff
        weights = np.clip(weights, 0.1, 1.0)  # Minimum weight of 0.1

        return weights

    def to_mne_source_space(self, subject: str = 'mouse') -> list:
        """
        Convert to MNE-Python SourceSpaces format for compatibility.

        Args:
            subject: Subject identifier

        Returns:
            MNE SourceSpaces-compatible list
        """
        import mne

        # Create a discrete source space
        src = mne.setup_volume_source_space(
            subject=subject,
            pos=self.source_coords_mm / 1000.0,  # Convert to meters
            mri=None,
            add_interpolator=False
        )

        # Store depth information as custom field
        src[0]['depths_mm'] = self.source_depths_mm
        src[0]['max_depth_mm'] = self.max_depth_mm

        return src

    def save(self, filepath: str):
        """Save cortical source space to numpy archive."""
        np.savez(
            filepath,
            source_coords_mm=self.source_coords_mm,
            source_depths_mm=self.source_depths_mm,
            max_depth_mm=self.max_depth_mm,
            brain_surface_coords=self.brain_surface_coords,
            metadata=self.metadata
        )
        logger.info(f"Saved cortical source space to {filepath}")

    @classmethod
    def load(cls, filepath: str) -> 'CorticalSourceSpace':
        """Load cortical source space from numpy archive."""
        data = np.load(filepath, allow_pickle=True)
        return cls(
            source_coords_mm=data['source_coords_mm'],
            source_depths_mm=data['source_depths_mm'],
            max_depth_mm=float(data['max_depth_mm']),
            brain_surface_coords=data.get('brain_surface_coords'),
            metadata=data.get('metadata', {}).item() if 'metadata' in data else {}
        )

    def __repr__(self) -> str:
        return (
            f"CorticalSourceSpace(n_sources={self.n_sources}, "
            f"max_depth={self.max_depth_mm}mm, "
            f"depth_range=[{self.source_depths_mm.min():.2f}, "
            f"{self.source_depths_mm.max():.2f}]mm)"
        )
