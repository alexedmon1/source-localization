"""
ROI Parcellation Visualization

Visualizes ROI-level results with anatomical parcellation boundaries,
following neuroimaging conventions for displaying regional results.

Features:
1. ROI maps with filled regions colored by value
2. ROI boundary overlays on existing plots
3. Bilateral hemisphere comparisons
4. ROI value bar charts by region

Author: Claude Code
Date: 2026-01-26
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize, LinearSegmentedColormap, to_rgba
from matplotlib.cm import ScalarMappable
from matplotlib.patches import Polygon
from matplotlib.collections import PatchCollection
import matplotlib.patches as mpatches
from scipy.interpolate import griddata
from scipy.ndimage import gaussian_filter
from typing import Optional, Dict, List, Tuple, Any, Union
from pathlib import Path
import json
import logging

try:
    from skimage.measure import find_contours
    HAS_SKIMAGE = True
except ImportError:
    HAS_SKIMAGE = False
    logging.warning("skimage not available. ROI boundary extraction will be limited.")

import nibabel as nib

logger = logging.getLogger(__name__)


class ROIVisualizer:
    """
    Visualizer for ROI-level results with parcellation overlays.

    Creates publication-quality figures showing ROI values with
    anatomical boundaries and region labels.

    Attributes:
        atlas_data: 3D array of ROI labels
        affine: Voxel-to-mm transformation
        roi_info: Dict mapping ROI ID to metadata (name, color, etc.)
    """

    def __init__(
        self,
        atlas_path: str,
        roi_mapping_path: Optional[str] = None,
        apply_10x_correction: bool = True
    ):
        """
        Initialize ROI visualizer.

        Args:
            atlas_path: Path to atlas NIfTI file
            roi_mapping_path: Path to ROI mapping JSON with names and colors
            apply_10x_correction: Apply 10x voxel size correction for mouse atlas
        """
        logger.info(f"Loading atlas from {atlas_path}")

        # Load atlas
        nii = nib.load(atlas_path)
        self.atlas_data = np.asarray(nii.dataobj)
        self.affine = nii.affine.copy()

        # Apply 10x correction for mouse atlas
        if apply_10x_correction:
            self.affine[:3, :3] /= 10.0

        # Compute inverse affine
        self.affine_inv = np.linalg.inv(self.affine)

        # Load ROI info
        self.roi_info: Dict[int, Dict] = {}
        self.roi_colors: Dict[int, Tuple[float, ...]] = {}
        self._load_roi_info(roi_mapping_path)

        # Precompute ROI centroids and coordinates
        self._roi_centroids_mm: Dict[int, np.ndarray] = {}
        self._roi_coords_mm: Dict[int, np.ndarray] = {}
        self._precompute_roi_data()

        # Cache for ROI boundary contours
        self._boundary_cache: Dict[str, Dict[int, List]] = {}

        logger.info(f"ROIVisualizer initialized with {len(self.roi_info)} ROIs")

    def _load_roi_info(self, roi_mapping_path: Optional[str]):
        """Load ROI names and colors from JSON file."""
        if roi_mapping_path is None:
            # Create basic info from atlas labels
            for label in np.unique(self.atlas_data):
                if label == 0:
                    continue
                self.roi_info[int(label)] = {
                    'id': int(label),
                    'name': f'ROI_{label}',
                    'abbreviation': f'R{label}'
                }
                # Generate a color
                hue = (label * 137.5) % 360  # Golden angle distribution
                self.roi_colors[int(label)] = plt.cm.hsv(hue / 360)[:3]
            return

        with open(roi_mapping_path, 'r') as f:
            mapping = json.load(f)

        rois_data = mapping.get('rois', {})
        if isinstance(rois_data, dict):
            for key, entry in rois_data.items():
                roi_id = int(key)
                self.roi_info[roi_id] = entry

                # Extract color
                if 'color_rgb' in entry:
                    rgb = entry['color_rgb']
                    self.roi_colors[roi_id] = (rgb[0]/255, rgb[1]/255, rgb[2]/255)
                elif 'color_hex' in entry:
                    hex_color = entry['color_hex']
                    self.roi_colors[roi_id] = tuple(
                        int(hex_color[i:i+2], 16)/255 for i in (0, 2, 4)
                    )
                else:
                    hue = (roi_id * 137.5) % 360
                    self.roi_colors[roi_id] = plt.cm.hsv(hue / 360)[:3]

    def _precompute_roi_data(self):
        """Precompute centroids and coordinates for each ROI."""
        for roi_id in self.roi_info.keys():
            if roi_id == 0:  # Skip exterior/background
                continue

            mask = self.atlas_data == roi_id
            if not np.any(mask):
                continue

            # Get voxel coordinates
            voxel_coords = np.array(np.where(mask)).T
            coords_mm = nib.affines.apply_affine(self.affine, voxel_coords)

            self._roi_centroids_mm[roi_id] = np.mean(coords_mm, axis=0)
            self._roi_coords_mm[roi_id] = coords_mm

    def _get_roi_boundaries_2d(
        self,
        view: str,
        slice_coord: Optional[float] = None,
        slice_thickness: float = 0.5
    ) -> Dict[int, List[np.ndarray]]:
        """
        Extract ROI boundary contours for a 2D projection.

        Args:
            view: View name ('dorsal', 'left', etc.)
            slice_coord: If provided, only show ROIs at this slice
            slice_thickness: Thickness of slice for selection

        Returns:
            Dict mapping ROI ID to list of contour arrays
        """
        if not HAS_SKIMAGE:
            logger.warning("skimage not available for contour extraction")
            return {}

        # View configurations: (x_axis, y_axis, proj_axis, flip_x, flip_y)
        view_configs = {
            'dorsal': (0, 1, 2, False, False),
            'ventral': (0, 1, 2, False, True),
            'left': (1, 2, 0, False, False),
            'right': (1, 2, 0, True, False),
            'anterior': (0, 2, 1, False, False),
            'posterior': (0, 2, 1, True, False),
        }

        if view.lower() not in view_configs:
            raise ValueError(f"Unknown view: {view}")

        x_idx, y_idx, proj_idx, flip_x, flip_y = view_configs[view.lower()]

        # Create cache key
        cache_key = f"{view}_{slice_coord}_{slice_thickness}"
        if cache_key in self._boundary_cache:
            return self._boundary_cache[cache_key]

        boundaries = {}

        for roi_id in self.roi_info.keys():
            if roi_id == 0:
                continue

            # Get ROI mask
            roi_mask = self.atlas_data == roi_id
            if not np.any(roi_mask):
                continue

            # Project to 2D (max projection along proj_axis)
            if slice_coord is not None:
                # Slice-specific boundaries
                # Convert slice_coord to voxel space
                test_point = np.zeros(3)
                test_point[proj_idx] = slice_coord
                voxel = nib.affines.apply_affine(self.affine_inv, test_point)
                slice_voxel = int(round(voxel[proj_idx]))

                # Get slice range
                voxel_thickness = int(np.ceil(slice_thickness / np.abs(self.affine[proj_idx, proj_idx])))
                start_slice = max(0, slice_voxel - voxel_thickness)
                end_slice = min(roi_mask.shape[proj_idx], slice_voxel + voxel_thickness + 1)

                # Extract slice
                slices = [slice(None)] * 3
                slices[proj_idx] = slice(start_slice, end_slice)
                roi_slice = roi_mask[tuple(slices)]

                # Max projection
                proj_2d = np.any(roi_slice, axis=proj_idx).astype(float)
            else:
                # Full projection
                proj_2d = np.any(roi_mask, axis=proj_idx).astype(float)

            if not np.any(proj_2d):
                continue

            # Find contours
            contours = find_contours(proj_2d, 0.5)

            if not contours:
                continue

            # Convert contours to mm coordinates
            roi_contours = []
            for contour in contours:
                # contour is in (row, col) = (y_pixel, x_pixel) order
                # Need to convert to mm
                contour_mm = np.zeros_like(contour)

                # Build 3D coordinates for each contour point
                for i, (row, col) in enumerate(contour):
                    voxel_3d = np.zeros(3)
                    # Note: contour row corresponds to first axis after projection removal
                    # Need to map back to original 3D axes
                    axes_2d = [a for a in range(3) if a != proj_idx]
                    voxel_3d[axes_2d[0]] = col  # x in 2D -> first remaining axis
                    voxel_3d[axes_2d[1]] = row  # y in 2D -> second remaining axis
                    voxel_3d[proj_idx] = self.atlas_data.shape[proj_idx] // 2  # middle

                    coord_mm = nib.affines.apply_affine(self.affine, voxel_3d)
                    contour_mm[i, 0] = coord_mm[x_idx]
                    contour_mm[i, 1] = coord_mm[y_idx]

                if flip_x:
                    contour_mm[:, 0] = -contour_mm[:, 0]
                if flip_y:
                    contour_mm[:, 1] = -contour_mm[:, 1]

                roi_contours.append(contour_mm)

            if roi_contours:
                boundaries[roi_id] = roi_contours

        self._boundary_cache[cache_key] = boundaries
        return boundaries

    def add_roi_boundaries_to_axis(
        self,
        ax,
        view: str = 'dorsal',
        roi_ids: Optional[List[int]] = None,
        linewidth: float = 1.0,
        alpha: float = 0.8,
        color: Optional[str] = None,
        use_roi_colors: bool = True
    ):
        """
        Add ROI boundary contours to an existing axis.

        Args:
            ax: Matplotlib axis
            view: View name
            roi_ids: Specific ROI IDs to show (None = all)
            linewidth: Contour line width
            alpha: Contour transparency
            color: Single color for all boundaries (overrides use_roi_colors)
            use_roi_colors: Use ROI-specific colors from atlas
        """
        boundaries = self._get_roi_boundaries_2d(view)

        for roi_id, contours in boundaries.items():
            if roi_ids is not None and roi_id not in roi_ids:
                continue

            if color is not None:
                line_color = color
            elif use_roi_colors and roi_id in self.roi_colors:
                line_color = self.roi_colors[roi_id]
            else:
                line_color = 'white'

            for contour in contours:
                ax.plot(
                    contour[:, 0], contour[:, 1],
                    color=line_color,
                    linewidth=linewidth,
                    alpha=alpha
                )

    def plot_roi_map(
        self,
        roi_values: Dict[int, float],
        view: str = 'dorsal',
        cmap: str = 'hot',
        vmin: Optional[float] = None,
        vmax: Optional[float] = None,
        show_boundaries: bool = True,
        show_labels: bool = False,
        boundary_color: str = 'black',
        boundary_width: float = 1.0,
        title: Optional[str] = None,
        figsize: Tuple[int, int] = (10, 8),
        background_color: str = '#2d2d2d'
    ) -> plt.Figure:
        """
        Create ROI map with filled regions colored by value.

        Args:
            roi_values: Dict mapping ROI ID to value
            view: 'dorsal', 'ventral', 'left', 'right', 'anterior', 'posterior'
            cmap: Colormap for values
            vmin, vmax: Colormap limits
            show_boundaries: Draw ROI boundary lines
            show_labels: Show ROI name labels
            boundary_color: Color for boundary lines
            boundary_width: Width for boundary lines
            title: Figure title
            figsize: Figure size
            background_color: Color for ROIs without values

        Returns:
            matplotlib Figure
        """
        # View configurations
        view_configs = {
            'dorsal': (0, 1, 2, False, False),
            'ventral': (0, 1, 2, False, True),
            'left': (1, 2, 0, False, False),
            'right': (1, 2, 0, True, False),
            'anterior': (0, 2, 1, False, False),
            'posterior': (0, 2, 1, True, False),
        }

        if view.lower() not in view_configs:
            raise ValueError(f"Unknown view: {view}")

        x_idx, y_idx, proj_idx, flip_x, flip_y = view_configs[view.lower()]

        # Set colormap limits
        values_array = np.array(list(roi_values.values()))
        if vmin is None:
            vmin = values_array.min()
        if vmax is None:
            vmax = values_array.max()

        norm = Normalize(vmin=vmin, vmax=vmax)
        cmap_obj = plt.cm.get_cmap(cmap)

        fig, ax = plt.subplots(figsize=figsize)
        ax.set_facecolor(background_color)

        # Draw filled ROI regions
        for roi_id, coords_mm in self._roi_coords_mm.items():
            if roi_id == 0:
                continue

            x = coords_mm[:, x_idx]
            y = coords_mm[:, y_idx]

            if flip_x:
                x = -x
            if flip_y:
                y = -y

            # Determine color
            if roi_id in roi_values:
                color = cmap_obj(norm(roi_values[roi_id]))
            else:
                color = background_color

            ax.scatter(x, y, c=[color], s=5, alpha=0.6, edgecolors='none')

        # Draw boundaries on top
        if show_boundaries:
            self.add_roi_boundaries_to_axis(
                ax, view,
                linewidth=boundary_width,
                color=boundary_color,
                use_roi_colors=False
            )

        # Add labels
        if show_labels:
            for roi_id, centroid in self._roi_centroids_mm.items():
                if roi_id == 0 or roi_id not in self.roi_info:
                    continue

                x = centroid[x_idx]
                y = centroid[y_idx]
                if flip_x:
                    x = -x
                if flip_y:
                    y = -y

                name = self.roi_info[roi_id].get('abbreviation', f'R{roi_id}')
                ax.annotate(
                    name,
                    (x, y),
                    fontsize=6,
                    ha='center',
                    va='center',
                    color='white',
                    weight='bold',
                    path_effects=[
                        plt.matplotlib.patheffects.withStroke(linewidth=2, foreground='black')
                    ]
                )

        ax.set_aspect('equal')
        ax.set_xlabel(['X', 'Y', 'Z'][x_idx] + ' (mm)')
        ax.set_ylabel(['X', 'Y', 'Z'][y_idx] + ' (mm)')

        if title is None:
            title = f'ROI Map - {view.capitalize()} View'
        ax.set_title(title)

        # Colorbar
        sm = ScalarMappable(cmap=cmap, norm=norm)
        sm.set_array([])
        plt.colorbar(sm, ax=ax, label='Value', shrink=0.8)

        plt.tight_layout()
        return fig

    def plot_roi_multiview(
        self,
        roi_values: Dict[int, float],
        views: List[str] = ['dorsal', 'left', 'right', 'anterior'],
        cmap: str = 'hot',
        vmin: Optional[float] = None,
        vmax: Optional[float] = None,
        show_boundaries: bool = True,
        title: str = 'ROI Values',
        figsize: Tuple[int, int] = (14, 10)
    ) -> plt.Figure:
        """
        Create multi-panel ROI map with multiple views.

        Args:
            roi_values: Dict mapping ROI ID to value
            views: List of view names to show
            cmap: Colormap
            vmin, vmax: Colormap limits
            show_boundaries: Show ROI boundaries
            title: Figure title
            figsize: Figure size

        Returns:
            matplotlib Figure
        """
        n_views = len(views)
        n_cols = min(n_views, 2)
        n_rows = int(np.ceil(n_views / n_cols))

        fig, axes = plt.subplots(n_rows, n_cols, figsize=figsize)
        axes = np.atleast_2d(axes).flatten()

        # Set global colormap limits
        values_array = np.array(list(roi_values.values()))
        if vmin is None:
            vmin = values_array.min()
        if vmax is None:
            vmax = values_array.max()

        norm = Normalize(vmin=vmin, vmax=vmax)
        cmap_obj = plt.cm.get_cmap(cmap)

        view_configs = {
            'dorsal': (0, 1, 2, False, False),
            'ventral': (0, 1, 2, False, True),
            'left': (1, 2, 0, False, False),
            'right': (1, 2, 0, True, False),
            'anterior': (0, 2, 1, False, False),
            'posterior': (0, 2, 1, True, False),
        }

        for i, (ax, view) in enumerate(zip(axes, views)):
            ax.set_facecolor('#2d2d2d')

            x_idx, y_idx, proj_idx, flip_x, flip_y = view_configs[view.lower()]

            # Draw filled ROI regions
            for roi_id, coords_mm in self._roi_coords_mm.items():
                if roi_id == 0:
                    continue

                x = coords_mm[:, x_idx]
                y = coords_mm[:, y_idx]

                if flip_x:
                    x = -x
                if flip_y:
                    y = -y

                if roi_id in roi_values:
                    color = cmap_obj(norm(roi_values[roi_id]))
                else:
                    color = '#2d2d2d'

                ax.scatter(x, y, c=[color], s=3, alpha=0.6, edgecolors='none')

            # Draw boundaries
            if show_boundaries:
                self.add_roi_boundaries_to_axis(
                    ax, view,
                    linewidth=0.8,
                    color='black',
                    use_roi_colors=False
                )

            ax.set_aspect('equal')
            ax.set_title(view.capitalize(), fontsize=11)
            ax.tick_params(labelsize=8)

        # Hide unused axes
        for ax in axes[n_views:]:
            ax.set_visible(False)

        # Global colorbar
        fig.subplots_adjust(right=0.88)
        cbar_ax = fig.add_axes([0.90, 0.15, 0.02, 0.7])
        sm = ScalarMappable(cmap=cmap, norm=norm)
        sm.set_array([])
        fig.colorbar(sm, cax=cbar_ax, label='Value')

        fig.suptitle(title, fontsize=14)
        plt.tight_layout(rect=[0, 0, 0.88, 0.96])

        return fig

    def plot_bilateral_comparison(
        self,
        roi_values: Dict[int, float],
        measure_name: str = 'Value',
        figsize: Tuple[int, int] = (12, 8),
        show_difference: bool = True
    ) -> plt.Figure:
        """
        Create bar chart comparing left vs right hemisphere ROI values.

        Args:
            roi_values: Dict mapping ROI ID to value
            measure_name: Name of the measure for axis label
            figsize: Figure size
            show_difference: Show L-R difference values

        Returns:
            matplotlib Figure
        """
        # Separate left and right ROIs
        left_rois = {}
        right_rois = {}

        for roi_id, info in self.roi_info.items():
            name = info.get('name', f'ROI_{roi_id}')
            if roi_id not in roi_values:
                continue

            if name.endswith('_L') or '_L_' in name:
                base_name = name.replace('_L', '').replace('L_', '')
                left_rois[base_name] = (roi_id, roi_values[roi_id])
            elif name.endswith('_R') or '_R_' in name:
                base_name = name.replace('_R', '').replace('R_', '')
                right_rois[base_name] = (roi_id, roi_values[roi_id])

        # Find matching pairs
        common_regions = sorted(set(left_rois.keys()) & set(right_rois.keys()))

        if not common_regions:
            logger.warning("No matching left/right ROI pairs found")
            fig, ax = plt.subplots(figsize=figsize)
            ax.text(0.5, 0.5, 'No L/R pairs found', ha='center', va='center')
            return fig

        left_vals = [left_rois[r][1] for r in common_regions]
        right_vals = [right_rois[r][1] for r in common_regions]

        x = np.arange(len(common_regions))
        width = 0.35

        fig, ax = plt.subplots(figsize=figsize)

        bars_left = ax.bar(x - width/2, left_vals, width, label='Left', color='#4a90d9')
        bars_right = ax.bar(x + width/2, right_vals, width, label='Right', color='#d94a4a')

        ax.set_xlabel('Brain Region')
        ax.set_ylabel(measure_name)
        ax.set_title(f'Bilateral Comparison: {measure_name}')
        ax.set_xticks(x)
        ax.set_xticklabels(common_regions, rotation=45, ha='right', fontsize=8)
        ax.legend()

        # Add difference annotation if requested
        if show_difference:
            for i, (l, r) in enumerate(zip(left_vals, right_vals)):
                diff = l - r
                y_pos = max(l, r) + 0.05 * (max(left_vals + right_vals))
                ax.annotate(
                    f'{diff:+.2f}',
                    (i, y_pos),
                    ha='center',
                    fontsize=7,
                    color='green' if diff > 0 else 'red'
                )

        ax.axhline(y=0, color='black', linestyle='-', linewidth=0.5)

        plt.tight_layout()
        return fig

    def plot_roi_bar_chart(
        self,
        roi_values: Dict[int, float],
        sort_by: str = 'value',
        n_top: Optional[int] = None,
        measure_name: str = 'Value',
        use_roi_colors: bool = True,
        figsize: Tuple[int, int] = (12, 8),
        horizontal: bool = True
    ) -> plt.Figure:
        """
        Create bar chart of ROI values.

        Args:
            roi_values: Dict mapping ROI ID to value
            sort_by: 'value' (descending), 'name', or 'id'
            n_top: Only show top N ROIs (None = all)
            measure_name: Name for axis label
            use_roi_colors: Color bars by ROI colors
            figsize: Figure size
            horizontal: Horizontal bars (True) or vertical (False)

        Returns:
            matplotlib Figure
        """
        # Build data
        data = []
        for roi_id, value in roi_values.items():
            if roi_id == 0:
                continue
            name = self.roi_info.get(roi_id, {}).get('name', f'ROI_{roi_id}')
            color = self.roi_colors.get(roi_id, (0.5, 0.5, 0.5))
            data.append((roi_id, name, value, color))

        # Sort
        if sort_by == 'value':
            data.sort(key=lambda x: x[2], reverse=True)
        elif sort_by == 'name':
            data.sort(key=lambda x: x[1])
        else:
            data.sort(key=lambda x: x[0])

        # Limit to top N
        if n_top is not None:
            data = data[:n_top]

        roi_names = [d[1] for d in data]
        values = [d[2] for d in data]
        colors = [d[3] for d in data] if use_roi_colors else 'steelblue'

        fig, ax = plt.subplots(figsize=figsize)

        if horizontal:
            y_pos = np.arange(len(roi_names))
            ax.barh(y_pos, values, color=colors, edgecolor='white', linewidth=0.5)
            ax.set_yticks(y_pos)
            ax.set_yticklabels(roi_names, fontsize=8)
            ax.set_xlabel(measure_name)
            ax.invert_yaxis()  # Highest on top
        else:
            x_pos = np.arange(len(roi_names))
            ax.bar(x_pos, values, color=colors, edgecolor='white', linewidth=0.5)
            ax.set_xticks(x_pos)
            ax.set_xticklabels(roi_names, rotation=45, ha='right', fontsize=8)
            ax.set_ylabel(measure_name)

        ax.set_title(f'ROI {measure_name}')
        plt.tight_layout()

        return fig

    def get_roi_centroids(self) -> Dict[int, np.ndarray]:
        """Get centroid coordinates for all ROIs."""
        return self._roi_centroids_mm.copy()

    def get_roi_names(self) -> Dict[int, str]:
        """Get ROI names."""
        return {
            roi_id: info.get('name', f'ROI_{roi_id}')
            for roi_id, info in self.roi_info.items()
        }

    def __repr__(self) -> str:
        return f"ROIVisualizer(n_rois={len(self.roi_info)})"
