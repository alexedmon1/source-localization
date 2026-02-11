"""
Source Analysis Module for Mouse EEG Source Localization

This module provides:
1. Depth-weighted ROI time series extraction (our unique contribution)
2. Visualization tools for source maps, ROI parcellations, and connectivity

For spectral and connectivity ANALYSIS, use MNE-Python / MNE-Connectivity,
which have optimized, well-tested implementations.

Recommended Workflow:
    # 1. Run source localization pipeline
    source-localization study run study_config.yaml

    # 2. Load ROI time series in MNE
    import mne
    from mne_connectivity import spectral_connectivity_epochs

    epochs = mne.io.read_epochs_eeglab('roi_timeseries_signed.set')

    # 3. Compute connectivity with MNE
    conn = spectral_connectivity_epochs(epochs, method='coh', fmin=30, fmax=80)

    # 4. Visualize with our tools
    from source_localization.source_analysis import (
        ConnectivityVisualizer, extract_mne_connectivity
    )

    matrix = extract_mne_connectivity(conn, freq_band=(30, 55))
    viz = ConnectivityVisualizer(roi_labels=epochs.ch_names)
    fig = viz.plot_connectivity_matrix(matrix, cluster_order=True)

Depth-Based Accuracy Weighting:
- EEG source localization accuracy degrades with depth
- Validation shows: 0-1mm: 77%, 1-2mm: 36%, 2-3mm: 4%, >3mm: ~0%
- ROI time series are weighted by this depth-based accuracy

Author: Claude Code
Date: 2026-01-26
"""

from .cortical_sources import CorticalSourceSpace
from .source_statistics import SourceStatistics
from .atlas_lookup import AtlasLookup
from .roi_analysis import (
    ROIAnalysis,
    ROITimeSeries,
    compute_depth_weights,
    create_roi_analysis_from_pipeline,
)
from .visualization import SourceMapVisualizer
from .visualization_roi import ROIVisualizer
from .visualization_connectivity import ConnectivityVisualizer, extract_mne_connectivity
from .visualization_presets import (
    PRESETS,
    PublicationStyle,
    HOT_BLACK_CMAP,
    HOT_BLACK_CONTINUOUS_CMAP,
    DIVERGING_CMAP,
    VIRIDIS_BLACK_CMAP,
    apply_style,
    reset_style,
    save_publication_figure,
    get_cmap,
)

__all__ = [
    # Core analysis classes
    'CorticalSourceSpace',
    'SourceStatistics',
    'AtlasLookup',

    # ROI-level extraction with depth weighting (our unique contribution)
    'ROIAnalysis',
    'ROITimeSeries',
    'compute_depth_weights',
    'create_roi_analysis_from_pipeline',

    # Visualization classes
    'SourceMapVisualizer',
    'ROIVisualizer',
    'ConnectivityVisualizer',
    'extract_mne_connectivity',

    # Publication presets and styles
    'PRESETS',
    'PublicationStyle',
    'apply_style',
    'reset_style',
    'save_publication_figure',
    'get_cmap',

    # Custom colormaps
    'HOT_BLACK_CMAP',
    'HOT_BLACK_CONTINUOUS_CMAP',
    'DIVERGING_CMAP',
    'VIRIDIS_BLACK_CMAP',
]

__version__ = '0.3.0'
