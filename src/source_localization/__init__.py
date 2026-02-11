"""Mouse EEG Source Localization Package.

A self-contained package for performing source localization analysis on mouse EEG data.
Supports multiple BEM models (sphere, ellipsoid) and source spaces (volumetric, surface).
"""

from .pipeline import Pipeline
from .config import Config

# Lazy imports for optional submodules
def __getattr__(name):
    if name == 'study':
        from . import study
        return study
    if name == 'source_analysis':
        from . import source_analysis
        return source_analysis
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

__version__ = '0.4.0'
__all__ = ['Pipeline', 'Config', 'study', 'source_analysis']
