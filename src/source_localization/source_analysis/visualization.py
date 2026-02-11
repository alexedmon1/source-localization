"""
Source-Level Visualization

Visualizes source-level statistical maps, clusters, and peaks.
Follows neuroimaging conventions for displaying spatial results.

Visualization types:
1. Slice montage (axial, coronal, sagittal)
2. 3D scatter plots with statistical coloring
3. Glass brain projections
4. Cluster/peak overlays on anatomy

Author: Claude Code
Date: 2026-01-26
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize, LinearSegmentedColormap
from matplotlib.cm import ScalarMappable
from mpl_toolkits.mplot3d import Axes3D
from scipy.interpolate import griddata
from scipy.ndimage import gaussian_filter
from typing import Optional, Dict, List, Tuple, Any, Union
import logging

logger = logging.getLogger(__name__)


# Custom colormaps for neuroimaging
def _create_hot_cold_cmap():
    """Create blue-white-red colormap for +/- values."""
    colors = ['#0000FF', '#4444FF', '#8888FF', '#FFFFFF',
              '#FF8888', '#FF4444', '#FF0000']
    return LinearSegmentedColormap.from_list('hot_cold', colors)

HOT_COLD_CMAP = _create_hot_cold_cmap()


class SourceMapVisualizer:
    """
    Visualizer for source-level statistical maps.

    Creates publication-quality figures showing source-level results
    with anatomical context.

    Attributes:
        source_coords_mm: (n_sources, 3) source coordinates
        brain_surface_coords: Optional brain surface for reference
        electrode_positions: Optional electrode coordinates for overlay
        electrode_labels: Optional electrode channel names
    """

    def __init__(
        self,
        source_coords_mm: np.ndarray,
        brain_surface_coords: Optional[np.ndarray] = None,
        source_depths_mm: Optional[np.ndarray] = None
    ):
        """
        Initialize visualizer.

        Args:
            source_coords_mm: (n_sources, 3) source coordinates in mm
            brain_surface_coords: Optional surface coordinates for outline
            source_depths_mm: Optional depth values for each source
        """
        self.source_coords_mm = np.asarray(source_coords_mm)
        self.brain_surface_coords = brain_surface_coords
        self.source_depths_mm = source_depths_mm

        self.n_sources = len(self.source_coords_mm)

        # Electrode overlay data
        self.electrode_positions: Optional[np.ndarray] = None
        self.electrode_labels: Optional[List[str]] = None

        # Compute coordinate bounds
        self.bounds = {
            'x': (self.source_coords_mm[:, 0].min(), self.source_coords_mm[:, 0].max()),
            'y': (self.source_coords_mm[:, 1].min(), self.source_coords_mm[:, 1].max()),
            'z': (self.source_coords_mm[:, 2].min(), self.source_coords_mm[:, 2].max()),
        }

    def set_electrode_positions(
        self,
        positions: np.ndarray,
        labels: Optional[List[str]] = None
    ):
        """
        Set electrode positions for overlay on visualizations.

        Args:
            positions: (n_electrodes, 3) electrode coordinates in mm
            labels: Optional list of channel names
        """
        self.electrode_positions = np.asarray(positions)
        self.electrode_labels = labels
        logger.info(f"Set {len(positions)} electrode positions for overlay")

    def add_electrodes_to_axis(
        self,
        ax,
        view: str = 'dorsal',
        marker: str = 'o',
        color: str = 'white',
        edgecolor: str = 'black',
        size: float = 50,
        show_labels: bool = False,
        label_fontsize: int = 6,
        alpha: float = 0.9,
        zorder: int = 10
    ):
        """
        Add electrode markers to an existing axis.

        Args:
            ax: Matplotlib axis (2D)
            view: View name for coordinate projection
            marker: Marker style
            color: Marker face color
            edgecolor: Marker edge color
            size: Marker size
            show_labels: Whether to show channel labels
            label_fontsize: Font size for labels
            alpha: Marker transparency
            zorder: Drawing order (higher = on top)
        """
        if self.electrode_positions is None:
            logger.warning("No electrode positions set. Call set_electrode_positions() first.")
            return

        # View configurations: (x_axis, y_axis, flip_x, flip_y)
        views = {
            'dorsal': (0, 1, False, False),
            'ventral': (0, 1, False, True),
            'left': (1, 2, False, False),
            'right': (1, 2, True, False),
            'anterior': (0, 2, False, False),
            'posterior': (0, 2, True, False),
        }

        if view.lower() not in views:
            logger.warning(f"Unknown view: {view}")
            return

        x_idx, y_idx, flip_x, flip_y = views[view.lower()]

        x_coords = self.electrode_positions[:, x_idx]
        y_coords = self.electrode_positions[:, y_idx]

        if flip_x:
            x_coords = -x_coords
        if flip_y:
            y_coords = -y_coords

        ax.scatter(
            x_coords, y_coords,
            marker=marker,
            c=color,
            edgecolors=edgecolor,
            s=size,
            alpha=alpha,
            zorder=zorder,
            linewidths=1
        )

        if show_labels and self.electrode_labels is not None:
            for i, label in enumerate(self.electrode_labels):
                ax.annotate(
                    label,
                    (x_coords[i], y_coords[i]),
                    fontsize=label_fontsize,
                    ha='center',
                    va='bottom',
                    xytext=(0, 3),
                    textcoords='offset points',
                    zorder=zorder + 1
                )

    def plot_3d_scatter(
        self,
        values: np.ndarray,
        cmap: str = 'hot',
        vmin: Optional[float] = None,
        vmax: Optional[float] = None,
        threshold: Optional[float] = None,
        title: str = 'Source-Level Map',
        size_by_value: bool = False,
        alpha: float = 0.7,
        figsize: Tuple[int, int] = (10, 8),
        view_angle: Tuple[float, float] = (30, -60),
        show_all_sources: bool = True,
        below_threshold_color: str = '#1a1a1a'
    ) -> plt.Figure:
        """
        Create 3D scatter plot of source values.

        Args:
            values: (n_sources,) values to plot
            cmap: Matplotlib colormap name
            vmin, vmax: Colormap limits (auto if None)
            threshold: Only show sources above this value (unless show_all_sources=True)
            title: Figure title
            size_by_value: Scale point size by value
            alpha: Point transparency
            figsize: Figure size
            view_angle: (elevation, azimuth) for 3D view
            show_all_sources: If True, show below-threshold sources in dark color
            below_threshold_color: Color for sources below threshold

        Returns:
            matplotlib Figure
        """
        fig = plt.figure(figsize=figsize)
        ax = fig.add_subplot(111, projection='3d')

        # Set colormap limits first (from all values)
        if vmin is None:
            vmin = values.min()
        if vmax is None:
            vmax = values.max()

        # Apply threshold if specified
        if threshold is not None:
            above_mask = values >= threshold
            below_mask = ~above_mask
        else:
            above_mask = np.ones(len(values), dtype=bool)
            below_mask = np.zeros(len(values), dtype=bool)

        # Plot below-threshold sources first (in background) if show_all_sources
        if show_all_sources and np.any(below_mask):
            below_coords = self.source_coords_mm[below_mask]
            ax.scatter(
                below_coords[:, 0], below_coords[:, 1], below_coords[:, 2],
                c=below_threshold_color, s=15, alpha=0.3, edgecolors='none',
                zorder=1
            )

        # Plot above-threshold sources
        if np.any(above_mask):
            coords = self.source_coords_mm[above_mask]
            vals = values[above_mask]

            # Point sizes
            if size_by_value:
                sizes = 20 + 80 * (vals - vmin) / (vmax - vmin + 1e-10)
            else:
                sizes = 30

            scatter = ax.scatter(
                coords[:, 0], coords[:, 1], coords[:, 2],
                c=vals, cmap=cmap, vmin=vmin, vmax=vmax,
                s=sizes, alpha=alpha, edgecolors='none',
                zorder=2
            )
            plt.colorbar(scatter, ax=ax, label='Value', shrink=0.6)
        elif not show_all_sources:
            ax.set_title(f"{title}\n(No sources above threshold)")
            return fig

        # Plot brain surface outline if available
        if self.brain_surface_coords is not None:
            # Subsample for performance
            n_surface = len(self.brain_surface_coords)
            stride = max(1, n_surface // 5000)
            surface_sub = self.brain_surface_coords[::stride]
            ax.scatter(
                surface_sub[:, 0], surface_sub[:, 1], surface_sub[:, 2],
                c='lightgray', s=1, alpha=0.1, zorder=0
            )

        # Labels and colorbar
        ax.set_xlabel('X (mm)')
        ax.set_ylabel('Y (mm)')
        ax.set_zlabel('Z (mm)')
        ax.set_title(title)
        ax.view_init(elev=view_angle[0], azim=view_angle[1])

        plt.tight_layout()
        return fig

    def plot_slice_montage(
        self,
        values: np.ndarray,
        n_slices: int = 5,
        plane: str = 'axial',
        cmap: str = 'hot',
        vmin: Optional[float] = None,
        vmax: Optional[float] = None,
        threshold: Optional[float] = None,
        title: str = 'Source Map',
        figsize: Optional[Tuple[int, int]] = None
    ) -> plt.Figure:
        """
        Create slice montage showing source values.

        Args:
            values: (n_sources,) values to plot
            n_slices: Number of slices
            plane: 'axial' (z), 'coronal' (y), or 'sagittal' (x)
            cmap: Colormap
            vmin, vmax: Colormap limits
            threshold: Only show sources above this value
            title: Figure title
            figsize: Figure size (auto if None)

        Returns:
            matplotlib Figure
        """
        # Determine slice axis
        axis_map = {'axial': 2, 'coronal': 1, 'sagittal': 0}
        axis = axis_map.get(plane.lower(), 2)
        axis_label = ['X', 'Y', 'Z'][axis]

        # Get slice positions
        coord_range = (self.source_coords_mm[:, axis].min(),
                      self.source_coords_mm[:, axis].max())
        slice_positions = np.linspace(coord_range[0], coord_range[1], n_slices)
        slice_thickness = (coord_range[1] - coord_range[0]) / (n_slices * 2)

        if figsize is None:
            figsize = (3 * n_slices, 3)

        fig, axes = plt.subplots(1, n_slices, figsize=figsize)
        if n_slices == 1:
            axes = [axes]

        # Get other two axes for plotting
        other_axes = [i for i in range(3) if i != axis]

        # Colormap setup
        if vmin is None:
            vmin = values.min() if threshold is None else threshold
        if vmax is None:
            vmax = values.max()

        for i, (ax, slice_pos) in enumerate(zip(axes, slice_positions)):
            # Find sources in this slice
            slice_mask = np.abs(self.source_coords_mm[:, axis] - slice_pos) < slice_thickness

            if threshold is not None:
                slice_mask &= values >= threshold

            slice_coords = self.source_coords_mm[slice_mask]
            slice_vals = values[slice_mask]

            # Plot brain outline if available
            if self.brain_surface_coords is not None:
                surface_mask = np.abs(self.brain_surface_coords[:, axis] - slice_pos) < slice_thickness
                surface_slice = self.brain_surface_coords[surface_mask]
                if len(surface_slice) > 0:
                    ax.scatter(
                        surface_slice[:, other_axes[0]],
                        surface_slice[:, other_axes[1]],
                        c='lightgray', s=1, alpha=0.3
                    )

            # Plot sources
            if len(slice_coords) > 0:
                scatter = ax.scatter(
                    slice_coords[:, other_axes[0]],
                    slice_coords[:, other_axes[1]],
                    c=slice_vals, cmap=cmap, vmin=vmin, vmax=vmax,
                    s=30, alpha=0.8
                )

            ax.set_title(f'{axis_label}={slice_pos:.1f}mm')
            ax.set_aspect('equal')
            ax.set_xlabel(['Y', 'X', 'X'][axis] + ' (mm)')
            ax.set_ylabel(['Z', 'Z', 'Y'][axis] + ' (mm)')

        fig.suptitle(title)
        plt.tight_layout()

        # Add colorbar
        if len(values) > 0:
            sm = ScalarMappable(cmap=cmap, norm=Normalize(vmin=vmin, vmax=vmax))
            sm.set_array([])
            fig.colorbar(sm, ax=axes, label='Value', shrink=0.8)

        return fig

    def plot_glass_brain(
        self,
        values: np.ndarray,
        cmap: str = 'hot',
        vmin: Optional[float] = None,
        vmax: Optional[float] = None,
        threshold: Optional[float] = None,
        title: str = 'Glass Brain',
        figsize: Tuple[int, int] = (12, 4),
        show_all_sources: bool = True,
        below_threshold_color: str = '#1a1a1a',
        show_electrodes: bool = False
    ) -> plt.Figure:
        """
        Create glass brain (maximum intensity projection) views.

        Shows three orthogonal projections: sagittal, coronal, axial.

        Args:
            values: (n_sources,) values to plot
            cmap: Colormap
            vmin, vmax: Colormap limits
            threshold: Only show sources above this value
            title: Figure title
            figsize: Figure size
            show_all_sources: If True, show below-threshold sources in dark color
            below_threshold_color: Color for sources below threshold
            show_electrodes: If True and electrodes set, overlay electrode positions

        Returns:
            matplotlib Figure
        """
        fig, axes = plt.subplots(1, 3, figsize=figsize)

        # Set colormap limits from all values
        if vmin is None:
            vmin = values.min()
        if vmax is None:
            vmax = values.max()

        # Apply threshold
        if threshold is not None:
            above_mask = values >= threshold
            below_mask = ~above_mask
        else:
            above_mask = np.ones(len(values), dtype=bool)
            below_mask = np.zeros(len(values), dtype=bool)

        projections = [
            ('Sagittal (YZ)', 1, 2, 0, 'left'),   # View from side, plot Y vs Z
            ('Coronal (XZ)', 0, 2, 1, 'anterior'),   # View from front, plot X vs Z
            ('Axial (XY)', 0, 1, 2, 'dorsal'),     # View from top, plot X vs Y
        ]

        for ax, (view_name, x_idx, y_idx, _, view_key) in zip(axes, projections):
            # Plot brain outline
            if self.brain_surface_coords is not None:
                ax.scatter(
                    self.brain_surface_coords[:, x_idx],
                    self.brain_surface_coords[:, y_idx],
                    c='lightgray', s=0.5, alpha=0.2, zorder=0
                )

            # Plot below-threshold sources first
            if show_all_sources and np.any(below_mask):
                below_coords = self.source_coords_mm[below_mask]
                ax.scatter(
                    below_coords[:, x_idx], below_coords[:, y_idx],
                    c=below_threshold_color, s=10, alpha=0.3, zorder=1
                )

            # Plot above-threshold sources
            if np.any(above_mask):
                coords = self.source_coords_mm[above_mask]
                vals = values[above_mask]
                scatter = ax.scatter(
                    coords[:, x_idx], coords[:, y_idx],
                    c=vals, cmap=cmap, vmin=vmin, vmax=vmax,
                    s=20, alpha=0.8, zorder=2
                )

            # Add electrode overlay
            if show_electrodes:
                self.add_electrodes_to_axis(ax, view=view_key, size=30, zorder=3)

            ax.set_title(view_name)
            ax.set_aspect('equal')
            ax.set_xlabel(['X', 'Y', 'Z'][x_idx] + ' (mm)')
            ax.set_ylabel(['X', 'Y', 'Z'][y_idx] + ' (mm)')

        fig.suptitle(title)
        plt.tight_layout()

        # Colorbar
        sm = ScalarMappable(cmap=cmap, norm=Normalize(vmin=vmin, vmax=vmax))
        sm.set_array([])
        fig.colorbar(sm, ax=axes.tolist(), label='Value', shrink=0.8)

        return fig

    def plot_clusters(
        self,
        clusters: List[Any],  # List of SourceCluster objects
        cmap: str = 'tab10',
        show_peaks: bool = True,
        title: str = 'Cluster Map',
        figsize: Tuple[int, int] = (10, 8)
    ) -> plt.Figure:
        """
        Plot clusters with different colors.

        Args:
            clusters: List of SourceCluster objects
            cmap: Colormap for clusters
            show_peaks: Mark cluster peaks
            title: Figure title
            figsize: Figure size

        Returns:
            matplotlib Figure
        """
        fig = plt.figure(figsize=figsize)
        ax = fig.add_subplot(111, projection='3d')

        colors = plt.cm.get_cmap(cmap)(np.linspace(0, 1, max(len(clusters), 1)))

        for i, cluster in enumerate(clusters):
            color = colors[i % len(colors)]

            # Plot cluster sources
            ax.scatter(
                cluster.coords_mm[:, 0],
                cluster.coords_mm[:, 1],
                cluster.coords_mm[:, 2],
                c=[color], s=20, alpha=0.6,
                label=f'Cluster {i+1} (n={cluster.size})'
            )

            # Mark peak
            if show_peaks:
                ax.scatter(
                    [cluster.peak_coord_mm[0]],
                    [cluster.peak_coord_mm[1]],
                    [cluster.peak_coord_mm[2]],
                    c=[color], s=100, marker='*', edgecolors='black'
                )

        ax.set_xlabel('X (mm)')
        ax.set_ylabel('Y (mm)')
        ax.set_zlabel('Z (mm)')
        ax.set_title(title)

        if len(clusters) <= 10:
            ax.legend(loc='upper left', fontsize=8)

        plt.tight_layout()
        return fig

    def plot_peaks(
        self,
        peaks: List[Any],  # List of SourcePeak objects
        values: Optional[np.ndarray] = None,
        background_cmap: str = 'gray',
        peak_color: str = 'red',
        title: str = 'Peak Locations',
        figsize: Tuple[int, int] = (10, 8)
    ) -> plt.Figure:
        """
        Plot peak locations on background map.

        Args:
            peaks: List of SourcePeak objects
            values: Optional background values to show
            background_cmap: Colormap for background
            peak_color: Color for peak markers
            title: Figure title
            figsize: Figure size

        Returns:
            matplotlib Figure
        """
        fig = plt.figure(figsize=figsize)
        ax = fig.add_subplot(111, projection='3d')

        # Background sources
        if values is not None:
            ax.scatter(
                self.source_coords_mm[:, 0],
                self.source_coords_mm[:, 1],
                self.source_coords_mm[:, 2],
                c=values, cmap=background_cmap, s=10, alpha=0.3
            )
        elif self.brain_surface_coords is not None:
            ax.scatter(
                self.brain_surface_coords[:, 0],
                self.brain_surface_coords[:, 1],
                self.brain_surface_coords[:, 2],
                c='lightgray', s=1, alpha=0.1
            )

        # Plot peaks
        if peaks:
            peak_coords = np.array([p.coord_mm for p in peaks])
            peak_values = np.array([p.value for p in peaks])

            scatter = ax.scatter(
                peak_coords[:, 0],
                peak_coords[:, 1],
                peak_coords[:, 2],
                c=peak_color, s=100, marker='*',
                edgecolors='black', linewidths=0.5
            )

            # Annotate with rank
            for i, peak in enumerate(peaks):
                ax.text(
                    peak.coord_mm[0], peak.coord_mm[1], peak.coord_mm[2],
                    f' {i+1}', fontsize=8
                )

        ax.set_xlabel('X (mm)')
        ax.set_ylabel('Y (mm)')
        ax.set_zlabel('Z (mm)')
        ax.set_title(title)

        plt.tight_layout()
        return fig

    def plot_depth_histogram(
        self,
        values: Optional[np.ndarray] = None,
        n_bins: int = 20,
        title: str = 'Source Depth Distribution',
        figsize: Tuple[int, int] = (8, 5)
    ) -> plt.Figure:
        """
        Plot histogram of source depths, optionally weighted by values.

        Args:
            values: Optional values for weighting
            n_bins: Number of histogram bins
            title: Figure title
            figsize: Figure size

        Returns:
            matplotlib Figure
        """
        if self.source_depths_mm is None:
            raise ValueError("No depth information available")

        fig, ax = plt.subplots(figsize=figsize)

        if values is not None:
            # Weighted histogram
            ax.hist(
                self.source_depths_mm, bins=n_bins,
                weights=np.abs(values), alpha=0.7,
                label='Value-weighted'
            )
            ax.hist(
                self.source_depths_mm, bins=n_bins,
                alpha=0.3, label='Unweighted'
            )
            ax.legend()
        else:
            ax.hist(self.source_depths_mm, bins=n_bins, alpha=0.7)

        ax.set_xlabel('Depth from surface (mm)')
        ax.set_ylabel('Count' if values is None else 'Sum of values')
        ax.set_title(title)

        # Add validation accuracy annotation
        ax.axvline(x=1, color='g', linestyle='--', alpha=0.5, label='77% accuracy')
        ax.axvline(x=2, color='y', linestyle='--', alpha=0.5, label='36% accuracy')
        ax.axvline(x=3, color='r', linestyle='--', alpha=0.5, label='<5% accuracy')

        plt.tight_layout()
        return fig

    def create_summary_figure(
        self,
        values: np.ndarray,
        title: str = 'Source Analysis Summary',
        threshold_percentile: float = 95,
        figsize: Tuple[int, int] = (15, 10)
    ) -> plt.Figure:
        """
        Create comprehensive summary figure.

        Includes: 3D view, glass brain, depth histogram.

        Args:
            values: (n_sources,) values to plot
            title: Figure title
            threshold_percentile: Threshold for display
            figsize: Figure size

        Returns:
            matplotlib Figure
        """
        fig = plt.figure(figsize=figsize)

        # 3D scatter (large, top left)
        ax1 = fig.add_subplot(2, 2, 1, projection='3d')
        threshold = np.percentile(values, threshold_percentile)
        mask = values >= threshold
        if np.any(mask):
            ax1.scatter(
                self.source_coords_mm[mask, 0],
                self.source_coords_mm[mask, 1],
                self.source_coords_mm[mask, 2],
                c=values[mask], cmap='hot', s=30, alpha=0.8
            )
        ax1.set_xlabel('X (mm)')
        ax1.set_ylabel('Y (mm)')
        ax1.set_zlabel('Z (mm)')
        ax1.set_title(f'3D View (>{threshold_percentile}th percentile)')

        # Glass brain views (top right and bottom left)
        projections = [
            (2, 2, 2, 'Axial (XY)', 0, 1),
            (2, 2, 3, 'Sagittal (YZ)', 1, 2),
        ]

        for subplot_idx, view_title, x_idx, y_idx in projections:
            ax = fig.add_subplot(*subplot_idx[:2], subplot_idx[2])
            if np.any(mask):
                ax.scatter(
                    self.source_coords_mm[mask, x_idx],
                    self.source_coords_mm[mask, y_idx],
                    c=values[mask], cmap='hot', s=20, alpha=0.7
                )
            ax.set_xlabel(['X', 'Y', 'Z'][x_idx] + ' (mm)')
            ax.set_ylabel(['X', 'Y', 'Z'][y_idx] + ' (mm)')
            ax.set_title(view_title)
            ax.set_aspect('equal')

        # Depth histogram (bottom right)
        if self.source_depths_mm is not None:
            ax4 = fig.add_subplot(2, 2, 4)
            ax4.hist(self.source_depths_mm, bins=20, alpha=0.7)
            ax4.axvline(x=2, color='r', linestyle='--', label='Accuracy cutoff')
            ax4.set_xlabel('Depth (mm)')
            ax4.set_ylabel('Count')
            ax4.set_title('Depth Distribution')
            ax4.legend()

        fig.suptitle(title, fontsize=14)
        plt.tight_layout()

        return fig

    def save_figure(
        self,
        fig: plt.Figure,
        filepath: str,
        dpi: int = 150
    ):
        """Save figure to file."""
        fig.savefig(filepath, dpi=dpi, bbox_inches='tight')
        logger.info(f"Saved figure to {filepath}")
        plt.close(fig)

    # =========================================================================
    # HEATMAP VISUALIZATIONS (Human EEG style)
    # =========================================================================

    def plot_slice_heatmap(
        self,
        values: np.ndarray,
        plane: str = 'axial',
        slice_coord: Optional[float] = None,
        grid_resolution: float = 0.2,
        smoothing_mm: float = 1.0,
        cmap: str = 'hot',
        vmin: Optional[float] = None,
        vmax: Optional[float] = None,
        threshold: Optional[float] = None,
        show_brain_outline: bool = True,
        title: Optional[str] = None,
        figsize: Tuple[int, int] = (8, 6)
    ) -> plt.Figure:
        """
        Create a 2D heatmap slice through the brain (fMRI style).

        Interpolates source values onto a regular grid to create smooth
        continuous colormaps like human neuroimaging visualizations.

        Args:
            values: (n_sources,) values to plot
            plane: 'axial' (z), 'coronal' (y), or 'sagittal' (x)
            slice_coord: Coordinate of slice in mm (uses center if None)
            grid_resolution: Resolution of interpolation grid in mm
            smoothing_mm: Gaussian smoothing sigma in mm
            cmap: Colormap
            vmin, vmax: Colormap limits
            threshold: Only show values above this
            show_brain_outline: Draw brain boundary
            title: Figure title
            figsize: Figure size

        Returns:
            matplotlib Figure
        """
        # Determine slice axis and other axes
        axis_map = {'axial': 2, 'coronal': 1, 'sagittal': 0}
        slice_axis = axis_map.get(plane.lower(), 2)
        other_axes = [i for i in range(3) if i != slice_axis]
        axis_labels = ['X', 'Y', 'Z']

        # Get slice coordinate
        if slice_coord is None:
            slice_coord = np.median(self.source_coords_mm[:, slice_axis])

        # Select sources near slice
        slice_thickness = max(grid_resolution * 3, 0.5)
        slice_mask = np.abs(self.source_coords_mm[:, slice_axis] - slice_coord) < slice_thickness

        if not np.any(slice_mask):
            logger.warning(f"No sources found near slice at {plane}={slice_coord}mm")
            fig, ax = plt.subplots(figsize=figsize)
            ax.text(0.5, 0.5, 'No data at this slice', ha='center', va='center')
            return fig

        slice_coords = self.source_coords_mm[slice_mask][:, other_axes]
        slice_values = values[slice_mask]

        # Create interpolation grid
        x_range = (slice_coords[:, 0].min() - 1, slice_coords[:, 0].max() + 1)
        y_range = (slice_coords[:, 1].min() - 1, slice_coords[:, 1].max() + 1)

        grid_x = np.arange(x_range[0], x_range[1], grid_resolution)
        grid_y = np.arange(y_range[0], y_range[1], grid_resolution)
        grid_xx, grid_yy = np.meshgrid(grid_x, grid_y)

        # Interpolate to grid
        grid_values = griddata(
            slice_coords, slice_values,
            (grid_xx, grid_yy),
            method='linear',
            fill_value=np.nan
        )

        # Apply Gaussian smoothing
        if smoothing_mm > 0:
            sigma_pixels = smoothing_mm / grid_resolution
            # Only smooth non-NaN values
            mask = ~np.isnan(grid_values)
            if np.any(mask):
                smoothed = gaussian_filter(np.nan_to_num(grid_values), sigma=sigma_pixels)
                # Restore NaN outside brain
                grid_values = np.where(mask, smoothed, np.nan)

        # Apply threshold
        if threshold is not None:
            grid_values = np.where(grid_values >= threshold, grid_values, np.nan)

        # Set colormap limits
        valid_values = grid_values[~np.isnan(grid_values)]
        if len(valid_values) > 0:
            if vmin is None:
                vmin = np.percentile(valid_values, 2)
            if vmax is None:
                vmax = np.percentile(valid_values, 98)

        # Create figure
        fig, ax = plt.subplots(figsize=figsize)

        # Plot heatmap
        im = ax.imshow(
            grid_values,
            extent=[x_range[0], x_range[1], y_range[0], y_range[1]],
            origin='lower',
            cmap=cmap,
            vmin=vmin,
            vmax=vmax,
            aspect='equal',
            interpolation='bilinear'
        )

        # Draw brain outline
        if show_brain_outline and self.brain_surface_coords is not None:
            surface_mask = np.abs(
                self.brain_surface_coords[:, slice_axis] - slice_coord
            ) < slice_thickness
            if np.any(surface_mask):
                surface_slice = self.brain_surface_coords[surface_mask][:, other_axes]
                ax.scatter(
                    surface_slice[:, 0], surface_slice[:, 1],
                    c='gray', s=0.5, alpha=0.3
                )

        # Labels
        ax.set_xlabel(f'{axis_labels[other_axes[0]]} (mm)')
        ax.set_ylabel(f'{axis_labels[other_axes[1]]} (mm)')

        if title is None:
            title = f'{plane.capitalize()} slice at {axis_labels[slice_axis]}={slice_coord:.1f}mm'
        ax.set_title(title)

        # Colorbar
        cbar = plt.colorbar(im, ax=ax, shrink=0.8)
        cbar.set_label('Value')

        plt.tight_layout()
        return fig

    def plot_slice_heatmap_montage(
        self,
        values: np.ndarray,
        plane: str = 'axial',
        n_slices: int = 6,
        grid_resolution: float = 0.2,
        smoothing_mm: float = 1.0,
        cmap: str = 'hot',
        vmin: Optional[float] = None,
        vmax: Optional[float] = None,
        threshold_percentile: Optional[float] = None,
        title: str = 'Source Activity',
        figsize: Optional[Tuple[int, int]] = None
    ) -> plt.Figure:
        """
        Create montage of heatmap slices (like fMRI activation maps).

        Args:
            values: (n_sources,) values to plot
            plane: 'axial', 'coronal', or 'sagittal'
            n_slices: Number of slices
            grid_resolution: Interpolation resolution in mm
            smoothing_mm: Gaussian smoothing sigma
            cmap: Colormap
            vmin, vmax: Colormap limits
            threshold_percentile: Only show values above this percentile
            title: Figure title
            figsize: Figure size

        Returns:
            matplotlib Figure
        """
        axis_map = {'axial': 2, 'coronal': 1, 'sagittal': 0}
        slice_axis = axis_map.get(plane.lower(), 2)
        other_axes = [i for i in range(3) if i != slice_axis]
        axis_labels = ['X', 'Y', 'Z']

        # Determine slice positions
        coord_min = self.source_coords_mm[:, slice_axis].min()
        coord_max = self.source_coords_mm[:, slice_axis].max()
        slice_coords = np.linspace(coord_min + 0.5, coord_max - 0.5, n_slices)

        # Apply threshold
        threshold = None
        if threshold_percentile is not None:
            threshold = np.percentile(values, threshold_percentile)

        # Set global colormap limits
        if vmin is None:
            vmin = np.percentile(values, 2)
        if vmax is None:
            vmax = np.percentile(values, 98)

        # Create figure
        n_cols = min(n_slices, 4)
        n_rows = int(np.ceil(n_slices / n_cols))
        if figsize is None:
            figsize = (4 * n_cols, 4 * n_rows)

        fig, axes = plt.subplots(n_rows, n_cols, figsize=figsize)
        axes = np.atleast_2d(axes).flatten()

        # Interpolation grid bounds (compute once)
        x_range = (
            self.source_coords_mm[:, other_axes[0]].min() - 1,
            self.source_coords_mm[:, other_axes[0]].max() + 1
        )
        y_range = (
            self.source_coords_mm[:, other_axes[1]].min() - 1,
            self.source_coords_mm[:, other_axes[1]].max() + 1
        )
        grid_x = np.arange(x_range[0], x_range[1], grid_resolution)
        grid_y = np.arange(y_range[0], y_range[1], grid_resolution)
        grid_xx, grid_yy = np.meshgrid(grid_x, grid_y)

        for i, (ax, slice_coord) in enumerate(zip(axes, slice_coords)):
            # Select sources near slice
            slice_thickness = max(grid_resolution * 3, 0.5)
            slice_mask = np.abs(
                self.source_coords_mm[:, slice_axis] - slice_coord
            ) < slice_thickness

            if np.any(slice_mask):
                slice_coords_2d = self.source_coords_mm[slice_mask][:, other_axes]
                slice_values = values[slice_mask]

                # Interpolate
                grid_values = griddata(
                    slice_coords_2d, slice_values,
                    (grid_xx, grid_yy),
                    method='linear',
                    fill_value=np.nan
                )

                # Smooth
                if smoothing_mm > 0:
                    sigma = smoothing_mm / grid_resolution
                    mask = ~np.isnan(grid_values)
                    if np.any(mask):
                        smoothed = gaussian_filter(np.nan_to_num(grid_values), sigma=sigma)
                        grid_values = np.where(mask, smoothed, np.nan)

                # Threshold
                if threshold is not None:
                    grid_values = np.where(grid_values >= threshold, grid_values, np.nan)

                # Plot
                im = ax.imshow(
                    grid_values,
                    extent=[x_range[0], x_range[1], y_range[0], y_range[1]],
                    origin='lower',
                    cmap=cmap,
                    vmin=vmin,
                    vmax=vmax,
                    aspect='equal',
                    interpolation='bilinear'
                )

            ax.set_title(f'{axis_labels[slice_axis]}={slice_coord:.1f}mm', fontsize=10)
            ax.set_xlabel(f'{axis_labels[other_axes[0]]} (mm)', fontsize=8)
            ax.set_ylabel(f'{axis_labels[other_axes[1]]} (mm)', fontsize=8)
            ax.tick_params(labelsize=7)

        # Hide unused axes
        for ax in axes[n_slices:]:
            ax.set_visible(False)

        # Add colorbar
        fig.subplots_adjust(right=0.9)
        cbar_ax = fig.add_axes([0.92, 0.15, 0.02, 0.7])
        sm = ScalarMappable(cmap=cmap, norm=Normalize(vmin=vmin, vmax=vmax))
        sm.set_array([])
        fig.colorbar(sm, cax=cbar_ax, label='Value')

        fig.suptitle(title, fontsize=14)
        plt.tight_layout(rect=[0, 0, 0.9, 0.96])

        return fig

    def plot_mip_heatmap(
        self,
        values: np.ndarray,
        grid_resolution: float = 0.2,
        smoothing_mm: float = 1.0,
        cmap: str = 'hot',
        vmin: Optional[float] = None,
        vmax: Optional[float] = None,
        threshold_percentile: Optional[float] = 90,
        title: str = 'Maximum Intensity Projection',
        figsize: Tuple[int, int] = (14, 5)
    ) -> plt.Figure:
        """
        Create Maximum Intensity Projection (MIP) heatmaps.

        Shows three orthogonal views with maximum value along each axis,
        similar to glass brain but with continuous colormaps.

        Args:
            values: (n_sources,) values to plot
            grid_resolution: Interpolation resolution
            smoothing_mm: Gaussian smoothing
            cmap: Colormap
            vmin, vmax: Colormap limits
            threshold_percentile: Only show values above this percentile
            title: Figure title
            figsize: Figure size

        Returns:
            matplotlib Figure
        """
        fig, axes = plt.subplots(1, 3, figsize=figsize)

        # Apply threshold
        plot_values = values.copy()
        if threshold_percentile is not None:
            threshold = np.percentile(values, threshold_percentile)
            plot_values = np.where(values >= threshold, values, 0)

        # Set colormap limits
        if vmin is None:
            vmin = np.percentile(plot_values[plot_values > 0], 5) if np.any(plot_values > 0) else 0
        if vmax is None:
            vmax = np.percentile(plot_values, 99)

        projections = [
            ('Sagittal (R→L)', 1, 2, 0),   # YZ plane, project along X
            ('Coronal (A→P)', 0, 2, 1),    # XZ plane, project along Y
            ('Axial (I→S)', 0, 1, 2),      # XY plane, project along Z
        ]

        for ax, (view_name, x_idx, y_idx, proj_idx) in zip(axes, projections):
            # Create 2D grid
            x_range = (
                self.source_coords_mm[:, x_idx].min() - 1,
                self.source_coords_mm[:, x_idx].max() + 1
            )
            y_range = (
                self.source_coords_mm[:, y_idx].min() - 1,
                self.source_coords_mm[:, y_idx].max() + 1
            )

            grid_x = np.arange(x_range[0], x_range[1], grid_resolution)
            grid_y = np.arange(y_range[0], y_range[1], grid_resolution)

            # Create MIP by taking max along projection axis
            mip = np.zeros((len(grid_y), len(grid_x)))

            for j, y in enumerate(grid_y):
                for k, x in enumerate(grid_x):
                    # Find sources near this (x, y) position
                    dist_x = np.abs(self.source_coords_mm[:, x_idx] - x)
                    dist_y = np.abs(self.source_coords_mm[:, y_idx] - y)
                    nearby = (dist_x < grid_resolution * 2) & (dist_y < grid_resolution * 2)

                    if np.any(nearby):
                        mip[j, k] = np.max(plot_values[nearby])

            # Smooth
            if smoothing_mm > 0:
                sigma = smoothing_mm / grid_resolution
                mip = gaussian_filter(mip, sigma=sigma)

            # Mask zeros
            mip_masked = np.ma.masked_where(mip <= 0, mip)

            # Plot
            im = ax.imshow(
                mip_masked,
                extent=[x_range[0], x_range[1], y_range[0], y_range[1]],
                origin='lower',
                cmap=cmap,
                vmin=vmin,
                vmax=vmax,
                aspect='equal',
                interpolation='bilinear'
            )

            # Brain outline
            if self.brain_surface_coords is not None:
                ax.scatter(
                    self.brain_surface_coords[:, x_idx],
                    self.brain_surface_coords[:, y_idx],
                    c='lightgray', s=0.3, alpha=0.2
                )

            ax.set_title(view_name, fontsize=11)
            ax.set_xlabel(['X', 'Y', 'Z'][x_idx] + ' (mm)')
            ax.set_ylabel(['X', 'Y', 'Z'][y_idx] + ' (mm)')

        # Colorbar
        fig.subplots_adjust(right=0.88)
        cbar_ax = fig.add_axes([0.90, 0.15, 0.02, 0.7])
        sm = ScalarMappable(cmap=cmap, norm=Normalize(vmin=vmin, vmax=vmax))
        sm.set_array([])
        fig.colorbar(sm, cax=cbar_ax, label='Value')

        fig.suptitle(title, fontsize=14)

        return fig

    def plot_surface_heatmap(
        self,
        values: np.ndarray,
        view: str = 'dorsal',
        cmap: str = 'hot',
        vmin: Optional[float] = None,
        vmax: Optional[float] = None,
        threshold_percentile: Optional[float] = None,
        title: Optional[str] = None,
        figsize: Tuple[int, int] = (10, 8),
        show_all_sources: bool = True,
        below_threshold_color: str = '#1a1a1a',
        show_electrodes: bool = False
    ) -> plt.Figure:
        """
        Create surface projection heatmap (like human EEG source maps).

        Projects source values onto a 2D view of the brain surface.

        Args:
            values: (n_sources,) values to plot
            view: 'dorsal', 'ventral', 'left', 'right', 'anterior', 'posterior'
            cmap: Colormap
            vmin, vmax: Colormap limits
            threshold_percentile: Only show values above this percentile
            title: Figure title
            figsize: Figure size
            show_all_sources: If True, show below-threshold sources in dark color
            below_threshold_color: Color for sources below threshold
            show_electrodes: If True and electrodes set, overlay electrode positions

        Returns:
            matplotlib Figure
        """
        # View configurations: (x_axis, y_axis, flip_x, flip_y)
        views = {
            'dorsal': (0, 1, False, False),      # X vs Y, looking down
            'ventral': (0, 1, False, True),      # X vs Y, looking up
            'left': (1, 2, False, False),        # Y vs Z, from left
            'right': (1, 2, True, False),        # Y vs Z, from right
            'anterior': (0, 2, False, False),    # X vs Z, from front
            'posterior': (0, 2, True, False),    # X vs Z, from back
        }

        if view.lower() not in views:
            raise ValueError(f"Unknown view: {view}. Choose from {list(views.keys())}")

        x_idx, y_idx, flip_x, flip_y = views[view.lower()]

        # Set colormap limits from all values first
        if vmin is None:
            vmin = np.percentile(values, 2)
        if vmax is None:
            vmax = np.percentile(values, 98)

        # Apply threshold
        if threshold_percentile is not None:
            threshold = np.percentile(values, threshold_percentile)
            above_mask = values >= threshold
            below_mask = ~above_mask
        else:
            above_mask = np.ones(len(values), dtype=bool)
            below_mask = np.zeros(len(values), dtype=bool)

        fig, ax = plt.subplots(figsize=figsize)

        # Get coordinates for this view
        x_coords = self.source_coords_mm[:, x_idx].copy()
        y_coords = self.source_coords_mm[:, y_idx].copy()

        if flip_x:
            x_coords = -x_coords
        if flip_y:
            y_coords = -y_coords

        # Draw brain outline first
        if self.brain_surface_coords is not None:
            surface_x = self.brain_surface_coords[:, x_idx]
            surface_y = self.brain_surface_coords[:, y_idx]
            if flip_x:
                surface_x = -surface_x
            if flip_y:
                surface_y = -surface_y
            ax.scatter(surface_x, surface_y, c='lightgray', s=1, alpha=0.2, zorder=0)

        # Plot below-threshold sources first (in background)
        if show_all_sources and np.any(below_mask):
            ax.scatter(
                x_coords[below_mask],
                y_coords[below_mask],
                c=below_threshold_color,
                s=25,
                alpha=0.4,
                edgecolors='none',
                zorder=1
            )

        # Plot above-threshold sources with heatmap coloring
        # Sort by value so highest values are on top
        if np.any(above_mask):
            above_vals = values[above_mask]
            above_x = x_coords[above_mask]
            above_y = y_coords[above_mask]
            sort_idx = np.argsort(above_vals)

            scatter = ax.scatter(
                above_x[sort_idx],
                above_y[sort_idx],
                c=above_vals[sort_idx],
                cmap=cmap,
                vmin=vmin,
                vmax=vmax,
                s=50,
                alpha=0.8,
                edgecolors='none',
                zorder=2
            )

        # Add colorbar
        sm = ScalarMappable(cmap=cmap, norm=Normalize(vmin=vmin, vmax=vmax))
        sm.set_array([])
        plt.colorbar(sm, ax=ax, label='Value', shrink=0.8)

        # Add electrode overlay
        if show_electrodes:
            self.add_electrodes_to_axis(ax, view=view, size=50, zorder=3)

        ax.set_aspect('equal')
        ax.set_xlabel(['X', 'Y', 'Z'][x_idx] + ' (mm)')
        ax.set_ylabel(['X', 'Y', 'Z'][y_idx] + ' (mm)')

        if title is None:
            title = f'{view.capitalize()} View'
        ax.set_title(title)

        plt.tight_layout()

        return fig

    def plot_surface_heatmap_smooth(
        self,
        values: np.ndarray,
        view: str = 'dorsal',
        cmap: str = 'hot',
        vmin: Optional[float] = None,
        vmax: Optional[float] = None,
        grid_resolution: float = 0.1,
        smoothing_mm: float = 0.5,
        title: Optional[str] = None,
        figsize: Tuple[int, int] = (10, 8),
        show_all_sources: bool = True,
        background_value: float = 0.0,
        show_brain_outline: bool = True,
        show_electrodes: bool = False,
        interpolation_method: str = 'linear'
    ) -> plt.Figure:
        """
        Create smooth interpolated surface projection heatmap (like fMRI activation maps).

        Uses 2D griddata interpolation + Gaussian smoothing to create continuous
        colormaps instead of discrete dots. This produces publication-quality
        visualizations matching human neuroimaging conventions.

        Args:
            values: (n_sources,) values to plot
            view: 'dorsal', 'ventral', 'left', 'right', 'anterior', 'posterior'
            cmap: Colormap (use 'hot' or custom HOT_BLACK_CMAP for neuroimaging style)
            vmin, vmax: Colormap limits
            grid_resolution: Resolution of interpolation grid in mm (smaller = smoother)
            smoothing_mm: Gaussian smoothing sigma in mm
            title: Figure title
            figsize: Figure size
            show_all_sources: If True, background shows low value (not transparent)
            background_value: Value used for areas outside source coverage
            show_brain_outline: Show brain surface outline
            show_electrodes: If True and electrodes set, overlay electrode positions
            interpolation_method: 'linear', 'nearest', or 'cubic'

        Returns:
            matplotlib Figure
        """
        # View configurations: (x_axis, y_axis, flip_x, flip_y)
        views = {
            'dorsal': (0, 1, False, False),
            'ventral': (0, 1, False, True),
            'left': (1, 2, False, False),
            'right': (1, 2, True, False),
            'anterior': (0, 2, False, False),
            'posterior': (0, 2, True, False),
        }

        if view.lower() not in views:
            raise ValueError(f"Unknown view: {view}. Choose from {list(views.keys())}")

        x_idx, y_idx, flip_x, flip_y = views[view.lower()]

        # Get 2D coordinates for this view
        x_coords = self.source_coords_mm[:, x_idx].copy()
        y_coords = self.source_coords_mm[:, y_idx].copy()

        if flip_x:
            x_coords = -x_coords
        if flip_y:
            y_coords = -y_coords

        # Set colormap limits
        if vmin is None:
            vmin = np.percentile(values, 2)
        if vmax is None:
            vmax = np.percentile(values, 98)

        # Create interpolation grid
        x_range = (x_coords.min() - 1, x_coords.max() + 1)
        y_range = (y_coords.min() - 1, y_coords.max() + 1)

        grid_x = np.arange(x_range[0], x_range[1], grid_resolution)
        grid_y = np.arange(y_range[0], y_range[1], grid_resolution)
        grid_xx, grid_yy = np.meshgrid(grid_x, grid_y)

        # Interpolate to grid
        points = np.column_stack([x_coords, y_coords])
        grid_values = griddata(
            points, values,
            (grid_xx, grid_yy),
            method=interpolation_method,
            fill_value=background_value if show_all_sources else np.nan
        )

        # Apply Gaussian smoothing
        if smoothing_mm > 0:
            sigma_pixels = smoothing_mm / grid_resolution
            # Handle NaN values carefully
            mask = np.isnan(grid_values)
            if np.any(mask):
                grid_values_filled = np.nan_to_num(grid_values, nan=background_value)
                smoothed = gaussian_filter(grid_values_filled, sigma=sigma_pixels)
                if not show_all_sources:
                    # Restore NaN outside brain
                    smoothed = np.where(mask, np.nan, smoothed)
                grid_values = smoothed
            else:
                grid_values = gaussian_filter(grid_values, sigma=sigma_pixels)

        # Create figure
        fig, ax = plt.subplots(figsize=figsize)

        # Plot the smoothed heatmap
        im = ax.imshow(
            grid_values,
            extent=[x_range[0], x_range[1], y_range[0], y_range[1]],
            origin='lower',
            cmap=cmap,
            vmin=vmin,
            vmax=vmax,
            aspect='equal',
            interpolation='bilinear'
        )

        # Draw brain outline
        if show_brain_outline and self.brain_surface_coords is not None:
            surface_x = self.brain_surface_coords[:, x_idx]
            surface_y = self.brain_surface_coords[:, y_idx]
            if flip_x:
                surface_x = -surface_x
            if flip_y:
                surface_y = -surface_y
            ax.scatter(surface_x, surface_y, c='white', s=0.5, alpha=0.3, zorder=1)

        # Add electrode overlay
        if show_electrodes:
            self.add_electrodes_to_axis(ax, view=view, size=50, zorder=3)

        # Labels
        axis_labels = ['X', 'Y', 'Z']
        ax.set_xlabel(f'{axis_labels[x_idx]} (mm)')
        ax.set_ylabel(f'{axis_labels[y_idx]} (mm)')

        if title is None:
            title = f'{view.capitalize()} View (Smoothed)'
        ax.set_title(title)

        # Colorbar
        cbar = plt.colorbar(im, ax=ax, shrink=0.8)
        cbar.set_label('Value')

        plt.tight_layout()
        return fig

    def plot_multiview_heatmap_smooth(
        self,
        values: np.ndarray,
        cmap: str = 'hot',
        vmin: Optional[float] = None,
        vmax: Optional[float] = None,
        grid_resolution: float = 0.1,
        smoothing_mm: float = 0.5,
        title: str = 'Source Activity',
        figsize: Tuple[int, int] = (15, 10),
        show_all_sources: bool = True,
        background_value: float = 0.0,
        show_electrodes: bool = False
    ) -> plt.Figure:
        """
        Create multi-view smooth heatmap figure (6 standard views).

        Uses interpolation + smoothing for publication-quality continuous colormaps.

        Args:
            values: (n_sources,) values to plot
            cmap: Colormap
            vmin, vmax: Colormap limits
            grid_resolution: Interpolation grid resolution in mm
            smoothing_mm: Gaussian smoothing sigma in mm
            title: Figure title
            figsize: Figure size
            show_all_sources: If True, background shows low value (not transparent)
            background_value: Value for areas outside source coverage
            show_electrodes: If True and electrodes set, overlay electrode positions

        Returns:
            matplotlib Figure
        """
        fig = plt.figure(figsize=figsize)

        views = ['dorsal', 'ventral', 'left', 'right', 'anterior', 'posterior']
        positions = [1, 4, 2, 3, 5, 6]

        # Global colormap limits
        if vmin is None:
            vmin = np.percentile(values, 2)
        if vmax is None:
            vmax = np.percentile(values, 98)

        # View configurations
        view_configs = {
            'dorsal': (0, 1, False, False),
            'ventral': (0, 1, False, True),
            'left': (1, 2, False, False),
            'right': (1, 2, True, False),
            'anterior': (0, 2, False, False),
            'posterior': (0, 2, True, False),
        }

        for view, pos in zip(views, positions):
            ax = fig.add_subplot(2, 3, pos)

            x_idx, y_idx, flip_x, flip_y = view_configs[view]

            x_coords = self.source_coords_mm[:, x_idx].copy()
            y_coords = self.source_coords_mm[:, y_idx].copy()

            if flip_x:
                x_coords = -x_coords
            if flip_y:
                y_coords = -y_coords

            # Create interpolation grid
            x_range = (x_coords.min() - 0.5, x_coords.max() + 0.5)
            y_range = (y_coords.min() - 0.5, y_coords.max() + 0.5)

            grid_x = np.arange(x_range[0], x_range[1], grid_resolution)
            grid_y = np.arange(y_range[0], y_range[1], grid_resolution)
            grid_xx, grid_yy = np.meshgrid(grid_x, grid_y)

            # Interpolate
            points = np.column_stack([x_coords, y_coords])
            grid_values = griddata(
                points, values,
                (grid_xx, grid_yy),
                method='linear',
                fill_value=background_value if show_all_sources else np.nan
            )

            # Smooth
            if smoothing_mm > 0:
                sigma = smoothing_mm / grid_resolution
                mask = np.isnan(grid_values)
                if np.any(mask):
                    filled = np.nan_to_num(grid_values, nan=background_value)
                    smoothed = gaussian_filter(filled, sigma=sigma)
                    if not show_all_sources:
                        smoothed = np.where(mask, np.nan, smoothed)
                    grid_values = smoothed
                else:
                    grid_values = gaussian_filter(grid_values, sigma=sigma)

            # Plot
            ax.imshow(
                grid_values,
                extent=[x_range[0], x_range[1], y_range[0], y_range[1]],
                origin='lower',
                cmap=cmap,
                vmin=vmin,
                vmax=vmax,
                aspect='equal',
                interpolation='bilinear'
            )

            # Electrode overlay
            if show_electrodes:
                self.add_electrodes_to_axis(ax, view=view, size=20, zorder=3)

            ax.set_title(view.capitalize(), fontsize=11)
            ax.tick_params(labelsize=7)

        # Global colorbar
        fig.subplots_adjust(right=0.88)
        cbar_ax = fig.add_axes([0.90, 0.15, 0.02, 0.7])
        sm = ScalarMappable(cmap=cmap, norm=Normalize(vmin=vmin, vmax=vmax))
        sm.set_array([])
        fig.colorbar(sm, cax=cbar_ax, label='Value')

        fig.suptitle(title, fontsize=14)
        plt.tight_layout(rect=[0, 0, 0.88, 0.96])

        return fig

    def plot_multiview_heatmap(
        self,
        values: np.ndarray,
        cmap: str = 'hot',
        vmin: Optional[float] = None,
        vmax: Optional[float] = None,
        threshold_percentile: Optional[float] = 75,
        title: str = 'Source Activity',
        figsize: Tuple[int, int] = (15, 10),
        show_all_sources: bool = True,
        below_threshold_color: str = '#1a1a1a',
        show_electrodes: bool = False
    ) -> plt.Figure:
        """
        Create multi-view heatmap figure (6 standard views).

        Shows dorsal, ventral, left, right, anterior, posterior views.

        Args:
            values: (n_sources,) values to plot
            cmap: Colormap
            vmin, vmax: Colormap limits
            threshold_percentile: Only show values above this percentile
            title: Figure title
            figsize: Figure size
            show_all_sources: If True, show below-threshold sources in dark color
            below_threshold_color: Color for sources below threshold
            show_electrodes: If True and electrodes set, overlay electrode positions

        Returns:
            matplotlib Figure
        """
        fig = plt.figure(figsize=figsize)

        views = ['dorsal', 'ventral', 'left', 'right', 'anterior', 'posterior']
        positions = [1, 4, 2, 3, 5, 6]  # Grid positions for 2x3 layout

        # Global colormap limits (from all values)
        if vmin is None:
            vmin = np.percentile(values, 2)
        if vmax is None:
            vmax = np.percentile(values, 98)

        # Apply threshold
        if threshold_percentile is not None:
            threshold = np.percentile(values, threshold_percentile)
            above_mask = values >= threshold
            below_mask = ~above_mask
        else:
            above_mask = np.ones(len(values), dtype=bool)
            below_mask = np.zeros(len(values), dtype=bool)

        for view, pos in zip(views, positions):
            ax = fig.add_subplot(2, 3, pos)

            # View configurations
            view_configs = {
                'dorsal': (0, 1, False, False),
                'ventral': (0, 1, False, True),
                'left': (1, 2, False, False),
                'right': (1, 2, True, False),
                'anterior': (0, 2, False, False),
                'posterior': (0, 2, True, False),
            }

            x_idx, y_idx, flip_x, flip_y = view_configs[view]

            x_coords = self.source_coords_mm[:, x_idx].copy()
            y_coords = self.source_coords_mm[:, y_idx].copy()

            if flip_x:
                x_coords = -x_coords
            if flip_y:
                y_coords = -y_coords

            # Brain outline
            if self.brain_surface_coords is not None:
                sx = self.brain_surface_coords[:, x_idx]
                sy = self.brain_surface_coords[:, y_idx]
                if flip_x:
                    sx = -sx
                if flip_y:
                    sy = -sy
                ax.scatter(sx, sy, c='lightgray', s=0.5, alpha=0.15, zorder=0)

            # Plot below-threshold sources first
            if show_all_sources and np.any(below_mask):
                ax.scatter(
                    x_coords[below_mask], y_coords[below_mask],
                    c=below_threshold_color, s=15, alpha=0.3, zorder=1
                )

            # Plot above-threshold sources
            if np.any(above_mask):
                above_vals = values[above_mask]
                above_x = x_coords[above_mask]
                above_y = y_coords[above_mask]
                sort_idx = np.argsort(above_vals)
                ax.scatter(
                    above_x[sort_idx], above_y[sort_idx],
                    c=above_vals[sort_idx], cmap=cmap,
                    vmin=vmin, vmax=vmax,
                    s=30, alpha=0.8, edgecolors='none', zorder=2
                )

            # Add electrode overlay
            if show_electrodes:
                self.add_electrodes_to_axis(ax, view=view, size=25, zorder=3)

            ax.set_aspect('equal')
            ax.set_title(view.capitalize(), fontsize=11)
            ax.tick_params(labelsize=8)

        # Global colorbar
        fig.subplots_adjust(right=0.88)
        cbar_ax = fig.add_axes([0.90, 0.15, 0.02, 0.7])
        sm = ScalarMappable(cmap=cmap, norm=Normalize(vmin=vmin, vmax=vmax))
        sm.set_array([])
        fig.colorbar(sm, cax=cbar_ax, label='Value')

        fig.suptitle(title, fontsize=14)
        plt.tight_layout(rect=[0, 0, 0.88, 0.96])

        return fig
