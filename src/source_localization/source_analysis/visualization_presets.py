"""
Publication-Quality Visualization Presets

Provides standardized styles, colormaps, and export functions for
creating publication-ready neuroimaging figures.

Features:
1. PublicationStyle dataclass with font sizes, DPI, line widths
2. Preset configurations for publication, presentation, and poster
3. Custom colormaps for neuroimaging (hot-black, diverging)
4. Helper functions for saving figures in multiple formats

Author: Claude Code
Date: 2026-01-26
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap, Normalize
from matplotlib import rcParams
from dataclasses import dataclass, field
from typing import Optional, List, Dict, Tuple, Union
from pathlib import Path
import logging

logger = logging.getLogger(__name__)


# =============================================================================
# Custom Colormaps for Neuroimaging
# =============================================================================

def _create_hot_black_cmap() -> LinearSegmentedColormap:
    """
    Create hot colormap with black base (instead of white).

    This is ideal for neuroimaging where you want to show all sources
    with low values in black, transitioning to red/yellow/white for high values.
    """
    colors = [
        (0.0, (0.0, 0.0, 0.0)),      # Black
        (0.2, (0.3, 0.0, 0.0)),      # Very dark red
        (0.4, (0.7, 0.0, 0.0)),      # Dark red
        (0.6, (1.0, 0.3, 0.0)),      # Orange-red
        (0.8, (1.0, 0.7, 0.0)),      # Orange-yellow
        (1.0, (1.0, 1.0, 0.8)),      # Light yellow-white
    ]

    positions = [c[0] for c in colors]
    r = [c[1][0] for c in colors]
    g = [c[1][1] for c in colors]
    b = [c[1][2] for c in colors]

    cdict = {
        'red': list(zip(positions, r, r)),
        'green': list(zip(positions, g, g)),
        'blue': list(zip(positions, b, b)),
    }

    return LinearSegmentedColormap('hot_black', cdict, N=256)


def _create_hot_black_continuous_cmap() -> LinearSegmentedColormap:
    """
    Create continuous hot-black colormap (no threshold visible).

    Smooth gradient from black through red to yellow, ideal for
    showing all source values without artificial cutoffs.
    """
    colors = [
        '#000000',  # Black
        '#1a0000',  # Very dark red
        '#4d0000',  # Dark red
        '#800000',  # Maroon
        '#b30000',  # Red
        '#e60000',  # Bright red
        '#ff3300',  # Orange-red
        '#ff6600',  # Orange
        '#ff9900',  # Orange-yellow
        '#ffcc00',  # Yellow
        '#ffff66',  # Light yellow
    ]

    return LinearSegmentedColormap.from_list('hot_black_continuous', colors, N=256)


def _create_diverging_cmap() -> LinearSegmentedColormap:
    """
    Create blue-white-red diverging colormap for +/- values.

    Ideal for showing increases vs decreases or correlation coefficients.
    """
    colors = [
        (0.0, (0.0, 0.2, 0.6)),      # Dark blue
        (0.2, (0.2, 0.4, 0.8)),      # Blue
        (0.4, (0.6, 0.7, 0.9)),      # Light blue
        (0.5, (0.95, 0.95, 0.95)),   # Near white
        (0.6, (0.9, 0.7, 0.6)),      # Light red
        (0.8, (0.8, 0.4, 0.2)),      # Red
        (1.0, (0.6, 0.2, 0.0)),      # Dark red
    ]

    positions = [c[0] for c in colors]
    r = [c[1][0] for c in colors]
    g = [c[1][1] for c in colors]
    b = [c[1][2] for c in colors]

    cdict = {
        'red': list(zip(positions, r, r)),
        'green': list(zip(positions, g, g)),
        'blue': list(zip(positions, b, b)),
    }

    return LinearSegmentedColormap('diverging_bwr', cdict, N=256)


def _create_viridis_black_cmap() -> LinearSegmentedColormap:
    """
    Create viridis-style colormap with black base.

    Good for colorblind-friendly visualizations with dark background.
    """
    colors = [
        '#000000',  # Black
        '#0d0887',  # Dark purple
        '#46039f',  # Purple
        '#7201a8',  # Violet
        '#9c179e',  # Magenta
        '#bd3786',  # Pink
        '#d8576b',  # Salmon
        '#ed7953',  # Orange
        '#fb9f3a',  # Yellow-orange
        '#fdca26',  # Yellow
        '#f0f921',  # Bright yellow
    ]

    return LinearSegmentedColormap.from_list('viridis_black', colors, N=256)


# Create colormap instances
HOT_BLACK_CMAP = _create_hot_black_cmap()
HOT_BLACK_CONTINUOUS_CMAP = _create_hot_black_continuous_cmap()
DIVERGING_CMAP = _create_diverging_cmap()
VIRIDIS_BLACK_CMAP = _create_viridis_black_cmap()

# Register colormaps with matplotlib
try:
    plt.colormaps.register(cmap=HOT_BLACK_CMAP, name='hot_black')
    plt.colormaps.register(cmap=HOT_BLACK_CONTINUOUS_CMAP, name='hot_black_continuous')
    plt.colormaps.register(cmap=DIVERGING_CMAP, name='diverging_bwr')
    plt.colormaps.register(cmap=VIRIDIS_BLACK_CMAP, name='viridis_black')
except ValueError:
    # Already registered
    pass


# =============================================================================
# Publication Style Configuration
# =============================================================================

@dataclass
class PublicationStyle:
    """
    Publication-quality visualization style settings.

    Attributes:
        name: Style name
        dpi: Figure resolution
        font_family: Font family (sans-serif, serif, monospace)
        font_size: Base font size
        title_size: Title font size
        label_size: Axis label font size
        tick_size: Tick label font size
        legend_size: Legend font size
        line_width: Default line width
        marker_size: Default marker size
        axes_linewidth: Axis spine width
        figure_facecolor: Figure background color
        axes_facecolor: Axes background color
        grid_alpha: Grid transparency
        savefig_formats: Formats for saving
    """

    name: str
    dpi: int = 150
    font_family: str = 'sans-serif'
    font_size: float = 10
    title_size: float = 12
    label_size: float = 10
    tick_size: float = 8
    legend_size: float = 8
    line_width: float = 1.0
    marker_size: float = 6.0
    axes_linewidth: float = 1.0
    figure_facecolor: str = 'white'
    axes_facecolor: str = 'white'
    grid_alpha: float = 0.3
    savefig_formats: List[str] = field(default_factory=lambda: ['png'])

    def apply(self):
        """Apply this style to matplotlib's rcParams."""
        rcParams['figure.dpi'] = self.dpi
        rcParams['savefig.dpi'] = self.dpi
        rcParams['font.family'] = self.font_family
        rcParams['font.size'] = self.font_size
        rcParams['axes.titlesize'] = self.title_size
        rcParams['axes.labelsize'] = self.label_size
        rcParams['xtick.labelsize'] = self.tick_size
        rcParams['ytick.labelsize'] = self.tick_size
        rcParams['legend.fontsize'] = self.legend_size
        rcParams['lines.linewidth'] = self.line_width
        rcParams['lines.markersize'] = self.marker_size
        rcParams['axes.linewidth'] = self.axes_linewidth
        rcParams['figure.facecolor'] = self.figure_facecolor
        rcParams['axes.facecolor'] = self.axes_facecolor
        rcParams['grid.alpha'] = self.grid_alpha

        # Additional publication settings
        rcParams['axes.spines.top'] = False
        rcParams['axes.spines.right'] = False
        rcParams['legend.frameon'] = False

        logger.info(f"Applied '{self.name}' style")

    def get_figsize(self, aspect_ratio: float = 1.5, width: Optional[float] = None) -> Tuple[float, float]:
        """
        Get recommended figure size for this style.

        Args:
            aspect_ratio: Width/height ratio
            width: Explicit width (uses default if None)

        Returns:
            (width, height) in inches
        """
        default_widths = {
            'publication': 7.0,   # Single column
            'presentation': 10.0,
            'poster': 12.0,
        }

        if width is None:
            width = default_widths.get(self.name, 8.0)

        return (width, width / aspect_ratio)


# =============================================================================
# Preset Configurations
# =============================================================================

PRESETS: Dict[str, PublicationStyle] = {
    'publication': PublicationStyle(
        name='publication',
        dpi=300,
        font_family='sans-serif',
        font_size=8,
        title_size=10,
        label_size=8,
        tick_size=7,
        legend_size=7,
        line_width=0.8,
        marker_size=4.0,
        axes_linewidth=0.8,
        savefig_formats=['svg', 'png', 'pdf'],
    ),

    'presentation': PublicationStyle(
        name='presentation',
        dpi=150,
        font_family='sans-serif',
        font_size=12,
        title_size=16,
        label_size=12,
        tick_size=10,
        legend_size=10,
        line_width=1.5,
        marker_size=8.0,
        axes_linewidth=1.2,
        savefig_formats=['png'],
    ),

    'poster': PublicationStyle(
        name='poster',
        dpi=200,
        font_family='sans-serif',
        font_size=14,
        title_size=18,
        label_size=14,
        tick_size=12,
        legend_size=12,
        line_width=2.0,
        marker_size=10.0,
        axes_linewidth=1.5,
        savefig_formats=['png', 'pdf'],
    ),

    'dark': PublicationStyle(
        name='dark',
        dpi=150,
        font_family='sans-serif',
        font_size=10,
        title_size=12,
        label_size=10,
        tick_size=8,
        legend_size=8,
        line_width=1.0,
        marker_size=6.0,
        axes_linewidth=1.0,
        figure_facecolor='#1e1e1e',
        axes_facecolor='#2d2d2d',
        savefig_formats=['png'],
    ),

    'notebook': PublicationStyle(
        name='notebook',
        dpi=100,
        font_family='sans-serif',
        font_size=10,
        title_size=12,
        label_size=10,
        tick_size=8,
        legend_size=8,
        line_width=1.0,
        marker_size=6.0,
        axes_linewidth=1.0,
        savefig_formats=['png'],
    ),
}


# =============================================================================
# Helper Functions
# =============================================================================

def save_publication_figure(
    fig: plt.Figure,
    filepath: Union[str, Path],
    formats: Optional[List[str]] = None,
    dpi: Optional[int] = None,
    transparent: bool = False,
    bbox_inches: str = 'tight',
    pad_inches: float = 0.1
) -> List[Path]:
    """
    Save figure in multiple formats for publication.

    Args:
        fig: Matplotlib figure to save
        filepath: Base path (without extension)
        formats: List of formats ['svg', 'png', 'pdf']
        dpi: Resolution (uses figure's DPI if None)
        transparent: Transparent background
        bbox_inches: Bounding box mode
        pad_inches: Padding around figure

    Returns:
        List of saved file paths
    """
    if formats is None:
        formats = ['svg', 'png']

    filepath = Path(filepath)
    filepath.parent.mkdir(parents=True, exist_ok=True)

    saved_paths = []

    for fmt in formats:
        output_path = filepath.with_suffix(f'.{fmt}')
        fig.savefig(
            output_path,
            format=fmt,
            dpi=dpi,
            transparent=transparent,
            bbox_inches=bbox_inches,
            pad_inches=pad_inches
        )
        saved_paths.append(output_path)
        logger.info(f"Saved figure: {output_path}")

    return saved_paths


def apply_style(style_name: str = 'publication'):
    """
    Apply a preset style by name.

    Args:
        style_name: Name of preset ('publication', 'presentation', 'poster', 'dark', 'notebook')
    """
    if style_name not in PRESETS:
        raise ValueError(f"Unknown style: {style_name}. Available: {list(PRESETS.keys())}")

    PRESETS[style_name].apply()


def reset_style():
    """Reset matplotlib style to defaults."""
    plt.rcdefaults()
    logger.info("Reset to default matplotlib style")


def get_cmap(name: str) -> LinearSegmentedColormap:
    """
    Get a colormap by name, including custom neuroimaging colormaps.

    Args:
        name: Colormap name

    Returns:
        Colormap instance
    """
    custom_cmaps = {
        'hot_black': HOT_BLACK_CMAP,
        'hot_black_continuous': HOT_BLACK_CONTINUOUS_CMAP,
        'diverging': DIVERGING_CMAP,
        'diverging_bwr': DIVERGING_CMAP,
        'viridis_black': VIRIDIS_BLACK_CMAP,
    }

    if name in custom_cmaps:
        return custom_cmaps[name]

    return plt.cm.get_cmap(name)


def colorbar_only_figure(
    cmap: str = 'hot',
    vmin: float = 0,
    vmax: float = 1,
    label: str = '',
    orientation: str = 'vertical',
    figsize: Optional[Tuple[float, float]] = None
) -> plt.Figure:
    """
    Create a standalone colorbar figure (useful for multi-panel layouts).

    Args:
        cmap: Colormap name
        vmin, vmax: Color limits
        label: Colorbar label
        orientation: 'vertical' or 'horizontal'
        figsize: Figure size

    Returns:
        Figure with standalone colorbar
    """
    if figsize is None:
        figsize = (1.5, 6) if orientation == 'vertical' else (6, 1.5)

    fig, ax = plt.subplots(figsize=figsize)

    norm = Normalize(vmin=vmin, vmax=vmax)
    cmap_obj = get_cmap(cmap)

    cb = plt.colorbar(
        plt.cm.ScalarMappable(norm=norm, cmap=cmap_obj),
        cax=ax,
        orientation=orientation
    )
    cb.set_label(label)

    plt.tight_layout()
    return fig


def add_scalebar(
    ax,
    length_mm: float,
    location: str = 'lower right',
    color: str = 'black',
    fontsize: int = 8,
    pad: float = 0.5
):
    """
    Add a scale bar to an axis.

    Args:
        ax: Matplotlib axis
        length_mm: Scale bar length in mm
        location: Position ('lower right', 'lower left', etc.)
        color: Bar and text color
        fontsize: Text font size
        pad: Padding from edges
    """
    from mpl_toolkits.axes_grid1.anchored_artists import AnchoredSizeBar
    import matplotlib.font_manager as fm

    scalebar = AnchoredSizeBar(
        ax.transData,
        length_mm,
        f'{length_mm} mm',
        location,
        pad=pad,
        color=color,
        frameon=False,
        size_vertical=0.1,
        fontproperties=fm.FontProperties(size=fontsize)
    )
    ax.add_artist(scalebar)


# =============================================================================
# Convenience Functions for Common Figure Types
# =============================================================================

def setup_multiview_figure(
    n_views: int = 6,
    style: str = 'publication',
    figsize: Optional[Tuple[float, float]] = None
) -> Tuple[plt.Figure, np.ndarray]:
    """
    Set up a multi-view figure layout for brain visualizations.

    Args:
        n_views: Number of views (typically 6 for all standard views)
        style: Style preset name
        figsize: Figure size

    Returns:
        (figure, axes array)
    """
    apply_style(style)

    if figsize is None:
        figsize = PRESETS[style].get_figsize(aspect_ratio=1.5)

    n_cols = min(n_views, 3)
    n_rows = int(np.ceil(n_views / n_cols))

    fig, axes = plt.subplots(n_rows, n_cols, figsize=figsize)
    axes = np.atleast_2d(axes).flatten()

    return fig, axes


def annotate_subplot(
    ax,
    label: str,
    location: str = 'upper left',
    fontsize: Optional[int] = None,
    fontweight: str = 'bold',
    color: str = 'black'
):
    """
    Add a panel label (A, B, C...) to a subplot.

    Args:
        ax: Matplotlib axis
        label: Label text (e.g., 'A', 'B')
        location: Position
        fontsize: Font size (uses style default if None)
        fontweight: Font weight
        color: Text color
    """
    if fontsize is None:
        fontsize = rcParams['axes.titlesize']

    loc_map = {
        'upper left': (-0.1, 1.1),
        'upper right': (1.0, 1.1),
        'lower left': (-0.1, -0.1),
        'lower right': (1.0, -0.1),
    }

    x, y = loc_map.get(location, (-0.1, 1.1))

    ax.text(
        x, y, label,
        transform=ax.transAxes,
        fontsize=fontsize,
        fontweight=fontweight,
        color=color,
        va='top',
        ha='left'
    )
