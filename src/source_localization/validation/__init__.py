"""
Validation module for source localization accuracy testing.

This module provides tools for validating source localization results by:
- Simulating EEG from known dipole sources
- Computing quantitative accuracy metrics
- Running validation test suites
- Generating comprehensive validation reports

Modules
-------
simulation
    DipoleSimulator for creating synthetic EEG data
metrics
    Quantitative validation metrics (localization error, PSF, CTF, etc.)
runner
    ValidationRunner for running validation test suites
utils
    Helper functions for data loading and report generation
cli
    Command-line interface for validation

Main Classes
------------
DipoleSimulator
    Simulate EEG from known dipole sources
ValidationRunner
    Run validation tests for a single configuration
ValidationConfigLoader
    Load and discover validation configuration files

Key Functions
-------------
run_validation
    High-level function to run validation on multiple configs
compute_localization_error
    Euclidean distance between true and estimated positions
compute_roi_classification_accuracy
    Binary ROI classification accuracy
compute_hierarchical_roi_accuracy
    Hierarchical ROI accuracy with spatial proximity
compute_point_spread_function
    PSF metrics (FWHM, spatial spread)
compute_crosstalk_function
    CTF metrics (power leakage between ROIs)
create_validation_report
    Generate comprehensive markdown report with figures
load_atlas_roi_centroids
    Extract ROI centroids from atlas
load_roi_source_mapping
    Map sources to ROIs

Examples
--------
>>> from source_localization.validation import DipoleSimulator
>>> from source_localization.validation import (
...     compute_localization_error,
...     compute_roi_classification_accuracy,
...     create_validation_report
... )

>>> # Initialize simulator
>>> simulator = DipoleSimulator(forward_model, info, source_space)

>>> # Simulate dipole at known location
>>> eeg_data, metadata = simulator.simulate_dipole(
...     position_mm=[0, 0, 5],
...     amplitude_nAm=50.0,
...     snr_db=10.0
... )

>>> # Compute metrics after running inverse solution
>>> error = compute_localization_error(
...     true_position_mm=metadata['actual_position_mm'],
...     estimated_position_mm=peak_position
... )
>>> print(f"Localization error: {error:.2f} mm")

>>> # Run validation tests programmatically
>>> from source_localization.validation import run_validation, ValidationRunner
>>> results = run_validation(test_name='original', config_names=['V01', 'V08'])

>>> # Or use ValidationRunner directly
>>> runner = ValidationRunner('path/to/config.yaml')
>>> runner.setup()
>>> metrics = runner.run(snr_levels=[10], n_trials=100)
>>> runner.save_results(metrics)
"""

# Import main classes
from .simulation import DipoleSimulator

# Import runner classes and functions
from .runner import (
    ValidationRunner,
    ValidationConfigLoader,
    run_validation,
    deep_merge
)

# Import batch runner
from .batch_runner import (
    BatchValidationRunner,
    ValidationOutputSchema,
    VALIDATION_OUTPUT_VERSION
)

# Import metrics functions
from .metrics import (
    compute_localization_error,
    compute_roi_classification_accuracy,
    compute_hierarchical_roi_accuracy,
    compute_point_spread_function,
    compute_crosstalk_function,
    compute_amplitude_recovery,
    compute_depth_bias,
    compute_two_dipole_resolution,
    summarize_validation_results,
    compute_source_depths,
    compute_depth_stratified_error,
    DEFAULT_DEPTH_BINS_MM,
    EXTERIOR_ROI_ID
)

# Import utility functions
from .utils import (
    load_atlas_roi_centroids,
    load_roi_source_mapping,
    get_roi_sources,
    create_validation_report,
    generate_uniform_test_grid
)

# Import connectivity validation
from .connectivity import (
    compute_electrode_connectivity,
    compute_roi_connectivity,
    compare_connectivity_patterns,
    validate_connectivity
)

# Import summarize module
from .summarize import summarize_validation_folders

# Import robustness testing
from .robustness import RobustnessTest, RobustnessResults

# Import visualization functions from visualization module
from ..visualization import (
    generate_localization_error_map,
    generate_validated_error_map,
    create_localization_error_figure,
    create_error_colormap
)

__all__ = [
    # Classes
    'DipoleSimulator',
    'ValidationRunner',
    'ValidationConfigLoader',
    'BatchValidationRunner',
    'ValidationOutputSchema',
    'RobustnessTest',
    'RobustnessResults',

    # Runner functions
    'run_validation',
    'deep_merge',
    'VALIDATION_OUTPUT_VERSION',

    # Metrics
    'compute_localization_error',
    'compute_roi_classification_accuracy',
    'compute_hierarchical_roi_accuracy',
    'compute_point_spread_function',
    'compute_crosstalk_function',
    'compute_amplitude_recovery',
    'compute_depth_bias',
    'compute_two_dipole_resolution',
    'summarize_validation_results',
    'compute_source_depths',
    'compute_depth_stratified_error',
    'DEFAULT_DEPTH_BINS_MM',
    'EXTERIOR_ROI_ID',

    # Utilities
    'load_atlas_roi_centroids',
    'load_roi_source_mapping',
    'get_roi_sources',
    'create_validation_report',
    'generate_uniform_test_grid',

    # Connectivity validation
    'compute_electrode_connectivity',
    'compute_roi_connectivity',
    'compare_connectivity_patterns',
    'validate_connectivity',

    # Summary
    'summarize_validation_folders',

    # Visualization
    'generate_localization_error_map',
    'generate_validated_error_map',
    'create_localization_error_figure',
    'create_error_colormap'
]

from .atlas_bundle import AtlasBundle, load_bundle, verify_bundle, BundleConsistencyError
