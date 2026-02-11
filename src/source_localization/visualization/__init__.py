"""
Visualization module for source localization.

Provides tools for creating localization error maps and other visualizations.
"""

from .localization_error_map import (
    generate_localization_error_map,
    generate_validated_error_map,
    create_localization_error_figure,
    create_error_colormap,
    estimate_localization_error,
)

__all__ = [
    'generate_localization_error_map',
    'generate_validated_error_map',
    'create_localization_error_figure',
    'create_error_colormap',
    'estimate_localization_error',
]
