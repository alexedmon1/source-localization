"""
Connectivity Visualization

Visualizes functional connectivity matrices, networks, and graph metrics
following neuroimaging conventions.

Works with MNE-Connectivity outputs:

    from mne_connectivity import spectral_connectivity_epochs
    import mne

    # Load pipeline output
    epochs = mne.io.read_epochs_eeglab('roi_timeseries_signed.set')

    # Compute connectivity with MNE
    conn = spectral_connectivity_epochs(epochs, method='coh', fmin=30, fmax=80)

    # Visualize with our tools
    from source_localization.source_analysis import ConnectivityVisualizer

    viz = ConnectivityVisualizer(roi_labels=epochs.ch_names)
    fig = viz.plot_connectivity_matrix(conn.get_data('dense')[:, :, 0])  # First freq band

Features:
1. Connectivity matrices with hierarchical clustering
2. Chord diagrams with arc connections
3. Network graphs overlaid on brain anatomy
4. Graph metrics bar charts

Author: Claude Code
Date: 2026-01-26
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize, LinearSegmentedColormap
from matplotlib.cm import ScalarMappable
from matplotlib.patches import Arc, FancyArrowPatch, Circle
from matplotlib.collections import LineCollection
import matplotlib.patches as mpatches
from typing import Optional, Dict, List, Tuple, Any, Union
import logging

try:
    from scipy.cluster.hierarchy import dendrogram, linkage, leaves_list
    from scipy.spatial.distance import squareform
    HAS_SCIPY_CLUSTER = True
except ImportError:
    HAS_SCIPY_CLUSTER = False
    logging.warning("scipy.cluster not available. Hierarchical clustering will be disabled.")

try:
    import networkx as nx
    HAS_NETWORKX = True
except ImportError:
    HAS_NETWORKX = False
    logging.warning("networkx not available. Graph metrics will be limited.")

logger = logging.getLogger(__name__)


def extract_mne_connectivity(
    conn,
    freq_band: Optional[Tuple[float, float]] = None,
    method: str = 'mean'
) -> np.ndarray:
    """
    Extract connectivity matrix from MNE-Connectivity object.

    Works with outputs from mne_connectivity.spectral_connectivity_epochs().

    Args:
        conn: MNE-Connectivity Connectivity object or numpy array
        freq_band: (fmin, fmax) to average over, or None for all frequencies
        method: 'mean' to average over frequencies, 'max' for maximum

    Returns:
        (n_rois, n_rois) connectivity matrix

    Example:
        from mne_connectivity import spectral_connectivity_epochs

        conn = spectral_connectivity_epochs(epochs, method='coh')
        matrix = extract_mne_connectivity(conn, freq_band=(30, 55))
    """
    # If already a numpy array, return as-is
    if isinstance(conn, np.ndarray):
        if conn.ndim == 2:
            return conn
        elif conn.ndim == 3:
            # (n_connections, n_freqs, n_times) or (n_rois, n_rois, n_freqs)
            if method == 'mean':
                return conn.mean(axis=-1)
            else:
                return conn.max(axis=-1)
        else:
            raise ValueError(f"Unexpected array shape: {conn.shape}")

    # Try to get data from MNE-Connectivity object
    try:
        # Get data in dense format (n_rois, n_rois, n_freqs)
        data = conn.get_data(output='dense')
        freqs = conn.freqs

        if freq_band is not None and freqs is not None:
            # Select frequency band
            fmin, fmax = freq_band
            freq_mask = (freqs >= fmin) & (freqs <= fmax)
            data = data[:, :, freq_mask]

        # Average over frequencies
        if data.ndim == 3:
            if method == 'mean':
                data = data.mean(axis=-1)
            else:
                data = data.max(axis=-1)

        return data

    except AttributeError:
        raise TypeError(
            f"Expected MNE-Connectivity object or numpy array, got {type(conn)}. "
            "Use conn.get_data('dense') to extract the matrix first."
        )


class ConnectivityVisualizer:
    """
    Visualizer for functional connectivity results.

    Creates publication-quality figures showing connectivity matrices,
    chord diagrams, and network graphs.

    Attributes:
        roi_labels: List of ROI names
        roi_centroids: Dict mapping ROI index to (x, y, z) centroid in mm
        roi_colors: Dict mapping ROI index to color
    """

    def __init__(
        self,
        roi_labels: List[str],
        roi_centroids: Optional[Dict[int, np.ndarray]] = None,
        roi_colors: Optional[Dict[int, Tuple[float, ...]]] = None
    ):
        """
        Initialize connectivity visualizer.

        Args:
            roi_labels: List of ROI names (in order matching connectivity matrix)
            roi_centroids: Dict mapping ROI index to centroid coordinates
            roi_colors: Dict mapping ROI index to RGB color tuple
        """
        self.roi_labels = roi_labels
        self.n_rois = len(roi_labels)
        self.roi_centroids = roi_centroids or {}
        self.roi_colors = roi_colors or {}

        # Generate default colors if not provided
        if not self.roi_colors:
            for i in range(self.n_rois):
                hue = (i * 137.5) % 360
                self.roi_colors[i] = plt.cm.hsv(hue / 360)[:3]

        logger.info(f"ConnectivityVisualizer initialized with {self.n_rois} ROIs")

    def plot_connectivity_matrix(
        self,
        conn: np.ndarray,
        cmap: str = 'RdBu_r',
        vmin: Optional[float] = None,
        vmax: Optional[float] = None,
        symmetric_cmap: bool = True,
        cluster_order: bool = True,
        show_dendrogram: bool = False,
        title: str = 'Connectivity Matrix',
        figsize: Tuple[int, int] = (12, 10),
        show_labels: bool = True,
        label_fontsize: int = 7
    ) -> plt.Figure:
        """
        Plot connectivity matrix as heatmap with optional hierarchical clustering.

        Args:
            conn: (n_rois, n_rois) connectivity matrix
            cmap: Colormap (use 'RdBu_r' for diverging, 'hot' for positive-only)
            vmin, vmax: Colormap limits
            symmetric_cmap: Make colormap symmetric around 0
            cluster_order: Reorder rows/columns by hierarchical clustering
            show_dendrogram: Show dendrogram alongside matrix
            title: Figure title
            figsize: Figure size
            show_labels: Show ROI labels on axes
            label_fontsize: Font size for labels

        Returns:
            matplotlib Figure
        """
        conn = np.asarray(conn)

        if conn.shape != (self.n_rois, self.n_rois):
            raise ValueError(f"Expected ({self.n_rois}, {self.n_rois}) matrix, got {conn.shape}")

        # Determine ordering
        order = np.arange(self.n_rois)
        if cluster_order and HAS_SCIPY_CLUSTER:
            try:
                # Use correlation distance for clustering
                # Convert to distance matrix (1 - |correlation|)
                dist = 1 - np.abs(conn)
                np.fill_diagonal(dist, 0)

                # Ensure symmetry
                dist = (dist + dist.T) / 2

                # Cluster
                condensed = squareform(dist)
                linkage_matrix = linkage(condensed, method='average')
                order = leaves_list(linkage_matrix)
            except Exception as e:
                logger.warning(f"Clustering failed: {e}, using original order")

        # Reorder matrix
        conn_ordered = conn[np.ix_(order, order)]
        labels_ordered = [self.roi_labels[i] for i in order]

        # Set colormap limits
        if symmetric_cmap:
            max_abs = np.max(np.abs(conn_ordered))
            if vmin is None:
                vmin = -max_abs
            if vmax is None:
                vmax = max_abs
        else:
            if vmin is None:
                vmin = np.nanmin(conn_ordered)
            if vmax is None:
                vmax = np.nanmax(conn_ordered)

        # Create figure
        if show_dendrogram and HAS_SCIPY_CLUSTER:
            fig = plt.figure(figsize=(figsize[0] + 2, figsize[1]))
            gs = fig.add_gridspec(1, 2, width_ratios=[1, 5])
            ax_dendro = fig.add_subplot(gs[0])
            ax_matrix = fig.add_subplot(gs[1])

            # Plot dendrogram
            dendrogram(
                linkage_matrix,
                orientation='left',
                ax=ax_dendro,
                no_labels=True,
                color_threshold=0
            )
            ax_dendro.set_axis_off()
        else:
            fig, ax_matrix = plt.subplots(figsize=figsize)

        # Plot matrix
        im = ax_matrix.imshow(
            conn_ordered,
            cmap=cmap,
            vmin=vmin,
            vmax=vmax,
            aspect='equal'
        )

        # Add labels
        if show_labels:
            ax_matrix.set_xticks(np.arange(self.n_rois))
            ax_matrix.set_yticks(np.arange(self.n_rois))
            ax_matrix.set_xticklabels(labels_ordered, rotation=90, fontsize=label_fontsize)
            ax_matrix.set_yticklabels(labels_ordered, fontsize=label_fontsize)
        else:
            ax_matrix.set_xticks([])
            ax_matrix.set_yticks([])

        ax_matrix.set_title(title)

        # Colorbar
        cbar = plt.colorbar(im, ax=ax_matrix, shrink=0.8)
        cbar.set_label('Connectivity')

        plt.tight_layout()
        return fig

    def plot_chord_diagram(
        self,
        conn: np.ndarray,
        threshold: Optional[float] = None,
        threshold_percentile: Optional[float] = 90,
        cmap: str = 'RdBu_r',
        title: str = 'Connectivity Chord Diagram',
        figsize: Tuple[int, int] = (12, 12),
        linewidth_scale: float = 3.0,
        show_labels: bool = True,
        label_fontsize: int = 8
    ) -> plt.Figure:
        """
        Create circular chord diagram showing strongest connections.

        Args:
            conn: (n_rois, n_rois) connectivity matrix
            threshold: Absolute threshold for connections to show
            threshold_percentile: Percentile threshold (used if threshold is None)
            cmap: Colormap for connection strength
            title: Figure title
            figsize: Figure size
            linewidth_scale: Scale factor for line widths
            show_labels: Show ROI labels
            label_fontsize: Font size for labels

        Returns:
            matplotlib Figure
        """
        conn = np.asarray(conn)

        # Compute threshold
        if threshold is None:
            if threshold_percentile is not None:
                # Use upper triangle only
                upper_tri = conn[np.triu_indices(self.n_rois, k=1)]
                threshold = np.percentile(np.abs(upper_tri), threshold_percentile)
            else:
                threshold = 0

        fig, ax = plt.subplots(figsize=figsize, subplot_kw={'projection': 'polar'})

        # Position ROIs around circle
        angles = np.linspace(0, 2 * np.pi, self.n_rois, endpoint=False)

        # Draw ROI nodes
        for i, (angle, label) in enumerate(zip(angles, self.roi_labels)):
            color = self.roi_colors.get(i, (0.5, 0.5, 0.5))
            ax.scatter(angle, 1.0, s=100, c=[color], zorder=3, edgecolors='white')

            if show_labels:
                # Adjust label position based on angle
                ha = 'left' if angle < np.pi else 'right'
                rotation = np.degrees(angle) - 90
                if angle > np.pi / 2 and angle < 3 * np.pi / 2:
                    rotation += 180

                ax.text(
                    angle, 1.08, label,
                    fontsize=label_fontsize,
                    ha=ha,
                    va='center',
                    rotation=rotation,
                    rotation_mode='anchor'
                )

        # Get colormap
        cmap_obj = plt.cm.get_cmap(cmap)
        max_val = np.max(np.abs(conn))
        norm = Normalize(vmin=-max_val, vmax=max_val)

        # Draw connections
        for i in range(self.n_rois):
            for j in range(i + 1, self.n_rois):
                value = conn[i, j]
                if np.abs(value) < threshold:
                    continue

                # Draw curved line (bezier-like arc)
                angle1, angle2 = angles[i], angles[j]

                # Calculate control point for curve
                mid_angle = (angle1 + angle2) / 2
                # Pull curve inward
                curve_radius = 0.3 + 0.4 * (1 - np.abs(value) / max_val)

                # Draw arc using multiple line segments
                n_points = 50
                t = np.linspace(0, 1, n_points)

                # Quadratic bezier curve
                r1, r2 = 1.0, 1.0
                rm = curve_radius

                # Parametric curve
                r_curve = (1 - t)**2 * r1 + 2 * (1 - t) * t * rm + t**2 * r2
                angle_curve = (1 - t)**2 * angle1 + 2 * (1 - t) * t * mid_angle + t**2 * angle2

                color = cmap_obj(norm(value))
                linewidth = linewidth_scale * (np.abs(value) / max_val)

                ax.plot(
                    angle_curve, r_curve,
                    color=color,
                    linewidth=linewidth,
                    alpha=0.7,
                    zorder=1
                )

        ax.set_ylim(0, 1.2)
        ax.set_yticks([])
        ax.set_xticks([])
        ax.spines['polar'].set_visible(False)
        ax.set_title(title, y=1.08)

        # Add colorbar
        sm = ScalarMappable(cmap=cmap, norm=norm)
        sm.set_array([])
        cbar = plt.colorbar(sm, ax=ax, orientation='horizontal', shrink=0.5, pad=0.08)
        cbar.set_label('Connectivity')

        return fig

    def plot_network_on_brain(
        self,
        conn: np.ndarray,
        source_visualizer: Any,
        view: str = 'dorsal',
        threshold: Optional[float] = None,
        threshold_percentile: Optional[float] = 85,
        node_size_by: str = 'degree',
        edge_cmap: str = 'RdBu_r',
        title: str = 'Brain Network',
        figsize: Tuple[int, int] = (12, 10),
        node_scale: float = 200,
        edge_width_scale: float = 2.0,
        show_labels: bool = False
    ) -> plt.Figure:
        """
        Plot network graph overlaid on brain anatomy.

        Args:
            conn: (n_rois, n_rois) connectivity matrix
            source_visualizer: SourceMapVisualizer instance for brain outline
            view: View name
            threshold: Absolute threshold for edges
            threshold_percentile: Percentile threshold (used if threshold is None)
            node_size_by: 'degree', 'strength', or 'uniform'
            edge_cmap: Colormap for edge colors
            title: Figure title
            figsize: Figure size
            node_scale: Scale factor for node sizes
            edge_width_scale: Scale factor for edge widths
            show_labels: Show ROI labels

        Returns:
            matplotlib Figure
        """
        if not self.roi_centroids:
            raise ValueError("ROI centroids required for brain network plot")

        conn = np.asarray(conn)

        # Compute threshold
        if threshold is None:
            if threshold_percentile is not None:
                upper_tri = conn[np.triu_indices(self.n_rois, k=1)]
                threshold = np.percentile(np.abs(upper_tri), threshold_percentile)
            else:
                threshold = 0

        # View configurations
        view_configs = {
            'dorsal': (0, 1, False, False),
            'ventral': (0, 1, False, True),
            'left': (1, 2, False, False),
            'right': (1, 2, True, False),
            'anterior': (0, 2, False, False),
            'posterior': (0, 2, True, False),
        }

        if view.lower() not in view_configs:
            raise ValueError(f"Unknown view: {view}")

        x_idx, y_idx, flip_x, flip_y = view_configs[view.lower()]

        fig, ax = plt.subplots(figsize=figsize)

        # Draw brain outline
        if source_visualizer.brain_surface_coords is not None:
            sx = source_visualizer.brain_surface_coords[:, x_idx]
            sy = source_visualizer.brain_surface_coords[:, y_idx]
            if flip_x:
                sx = -sx
            if flip_y:
                sy = -sy
            ax.scatter(sx, sy, c='lightgray', s=0.5, alpha=0.15, zorder=0)

        # Get node positions
        node_x = []
        node_y = []
        valid_indices = []

        for i in range(self.n_rois):
            if i in self.roi_centroids:
                centroid = self.roi_centroids[i]
                x = centroid[x_idx]
                y = centroid[y_idx]
                if flip_x:
                    x = -x
                if flip_y:
                    y = -y
                node_x.append(x)
                node_y.append(y)
                valid_indices.append(i)

        node_x = np.array(node_x)
        node_y = np.array(node_y)

        # Compute node sizes
        if node_size_by == 'degree':
            degrees = np.sum(np.abs(conn) > threshold, axis=1)
            node_sizes = node_scale * (1 + degrees / np.max(degrees + 1))
        elif node_size_by == 'strength':
            strengths = np.sum(np.abs(conn), axis=1)
            node_sizes = node_scale * (1 + strengths / np.max(strengths + 1e-10))
        else:
            node_sizes = np.full(self.n_rois, node_scale)

        node_sizes = node_sizes[valid_indices]

        # Draw edges
        max_val = np.max(np.abs(conn))
        cmap_obj = plt.cm.get_cmap(edge_cmap)
        norm = Normalize(vmin=-max_val, vmax=max_val)

        for i, idx_i in enumerate(valid_indices):
            for j, idx_j in enumerate(valid_indices):
                if idx_j <= idx_i:  # Upper triangle only
                    continue

                value = conn[idx_i, idx_j]
                if np.abs(value) < threshold:
                    continue

                color = cmap_obj(norm(value))
                width = edge_width_scale * (np.abs(value) / max_val)

                ax.plot(
                    [node_x[i], node_x[j]],
                    [node_y[i], node_y[j]],
                    color=color,
                    linewidth=width,
                    alpha=0.6,
                    zorder=1
                )

        # Draw nodes
        node_colors = [self.roi_colors.get(i, (0.5, 0.5, 0.5)) for i in valid_indices]

        ax.scatter(
            node_x, node_y,
            s=node_sizes,
            c=node_colors,
            edgecolors='white',
            linewidths=1,
            zorder=2
        )

        # Add labels
        if show_labels:
            for i, idx in enumerate(valid_indices):
                ax.annotate(
                    self.roi_labels[idx],
                    (node_x[i], node_y[i]),
                    fontsize=6,
                    ha='center',
                    va='bottom',
                    xytext=(0, 5),
                    textcoords='offset points'
                )

        ax.set_aspect('equal')
        ax.set_xlabel(['X', 'Y', 'Z'][x_idx] + ' (mm)')
        ax.set_ylabel(['X', 'Y', 'Z'][y_idx] + ' (mm)')
        ax.set_title(title)

        # Colorbar for edges
        sm = ScalarMappable(cmap=edge_cmap, norm=norm)
        sm.set_array([])
        plt.colorbar(sm, ax=ax, label='Connectivity', shrink=0.8)

        plt.tight_layout()
        return fig

    def plot_graph_metrics(
        self,
        conn: np.ndarray,
        threshold: Optional[float] = None,
        threshold_percentile: Optional[float] = 50,
        metrics: List[str] = ['degree', 'strength', 'clustering'],
        n_top: int = 15,
        figsize: Tuple[int, int] = (15, 5)
    ) -> plt.Figure:
        """
        Plot bar charts of graph-theoretic metrics for each ROI.

        Args:
            conn: (n_rois, n_rois) connectivity matrix
            threshold: Threshold for binarizing (for degree/clustering)
            threshold_percentile: Percentile threshold
            metrics: List of metrics to plot ('degree', 'strength', 'clustering', 'betweenness')
            n_top: Number of top ROIs to show
            figsize: Figure size

        Returns:
            matplotlib Figure
        """
        conn = np.asarray(conn)

        # Compute threshold
        if threshold is None and threshold_percentile is not None:
            upper_tri = conn[np.triu_indices(self.n_rois, k=1)]
            threshold = np.percentile(np.abs(upper_tri), threshold_percentile)

        # Binarize for some metrics
        conn_binary = (np.abs(conn) > threshold).astype(float)
        np.fill_diagonal(conn_binary, 0)

        # Compute metrics
        metric_values = {}

        if 'degree' in metrics:
            metric_values['degree'] = np.sum(conn_binary, axis=1)

        if 'strength' in metrics:
            metric_values['strength'] = np.sum(np.abs(conn), axis=1)

        if 'clustering' in metrics:
            if HAS_NETWORKX:
                G = nx.from_numpy_array(conn_binary)
                clustering = nx.clustering(G)
                metric_values['clustering'] = np.array([clustering[i] for i in range(self.n_rois)])
            else:
                # Simple approximation
                clustering_coef = np.zeros(self.n_rois)
                for i in range(self.n_rois):
                    neighbors = np.where(conn_binary[i] > 0)[0]
                    k = len(neighbors)
                    if k < 2:
                        clustering_coef[i] = 0
                    else:
                        # Count triangles
                        triangles = 0
                        for j in neighbors:
                            for l in neighbors:
                                if j < l and conn_binary[j, l] > 0:
                                    triangles += 1
                        clustering_coef[i] = 2 * triangles / (k * (k - 1))
                metric_values['clustering'] = clustering_coef

        if 'betweenness' in metrics and HAS_NETWORKX:
            G = nx.from_numpy_array(conn_binary)
            betweenness = nx.betweenness_centrality(G)
            metric_values['betweenness'] = np.array([betweenness[i] for i in range(self.n_rois)])

        # Create figure
        n_metrics = len(metric_values)
        fig, axes = plt.subplots(1, n_metrics, figsize=figsize)
        if n_metrics == 1:
            axes = [axes]

        for ax, (metric_name, values) in zip(axes, metric_values.items()):
            # Sort by value and take top N
            sorted_idx = np.argsort(values)[::-1][:n_top]
            top_labels = [self.roi_labels[i] for i in sorted_idx]
            top_values = values[sorted_idx]
            top_colors = [self.roi_colors.get(i, (0.5, 0.5, 0.5)) for i in sorted_idx]

            y_pos = np.arange(len(top_labels))
            ax.barh(y_pos, top_values, color=top_colors, edgecolor='white', linewidth=0.5)
            ax.set_yticks(y_pos)
            ax.set_yticklabels(top_labels, fontsize=8)
            ax.set_xlabel(metric_name.capitalize())
            ax.set_title(f'Top {n_top} by {metric_name.capitalize()}')
            ax.invert_yaxis()

        plt.tight_layout()
        return fig

    def plot_connectivity_comparison(
        self,
        conn1: np.ndarray,
        conn2: np.ndarray,
        label1: str = 'Condition 1',
        label2: str = 'Condition 2',
        cmap: str = 'RdBu_r',
        title: str = 'Connectivity Comparison',
        figsize: Tuple[int, int] = (18, 6)
    ) -> plt.Figure:
        """
        Compare two connectivity matrices side by side with difference.

        Args:
            conn1: First connectivity matrix
            conn2: Second connectivity matrix
            label1: Label for first condition
            label2: Label for second condition
            cmap: Colormap
            title: Figure title
            figsize: Figure size

        Returns:
            matplotlib Figure
        """
        conn1 = np.asarray(conn1)
        conn2 = np.asarray(conn2)
        diff = conn2 - conn1

        fig, axes = plt.subplots(1, 3, figsize=figsize)

        # Common color limits
        max_val = max(np.max(np.abs(conn1)), np.max(np.abs(conn2)))
        diff_max = np.max(np.abs(diff))

        # Plot matrices
        for ax, data, subtitle, vlim in zip(
            axes,
            [conn1, conn2, diff],
            [label1, label2, f'{label2} - {label1}'],
            [max_val, max_val, diff_max]
        ):
            im = ax.imshow(
                data,
                cmap=cmap,
                vmin=-vlim,
                vmax=vlim,
                aspect='equal'
            )
            ax.set_title(subtitle)
            ax.set_xticks([])
            ax.set_yticks([])
            plt.colorbar(im, ax=ax, shrink=0.8)

        fig.suptitle(title, fontsize=14)
        plt.tight_layout()
        return fig

    def __repr__(self) -> str:
        return f"ConnectivityVisualizer(n_rois={self.n_rois})"
