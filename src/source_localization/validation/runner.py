"""
Validation Runner for Mouse EEG Source Localization.

This module provides classes for running dipole simulation validation
across pipeline configurations and generating metrics for comparison.

Classes
-------
ValidationConfigLoader
    Handles discovery and loading of validation config files
ValidationRunner
    Runs validation for a single configuration

Functions
---------
run_validation
    High-level function to run validation on multiple configs
deep_merge
    Deep merge two dictionaries

Examples
--------
>>> from source_localization.validation.runner import (
...     ValidationRunner, ValidationConfigLoader, run_validation
... )

>>> # Run all original validation tests
>>> results = run_validation(test_name='original')

>>> # Run specific configs
>>> results = run_validation(
...     test_name='dipole_size',
...     config_names=['D01', 'D02']
... )

>>> # Use ValidationRunner directly
>>> loader = ValidationConfigLoader()
>>> config_path = loader.discover_configs('original', config_names=['V01'])[0]
>>> runner = ValidationRunner(config_path, output_dir='./results/V01')
>>> runner.setup()
>>> metrics = runner.run(snr_levels=[10], n_trials=100)
>>> runner.save_results(metrics)
"""

# ---------------------------------------------------------------------------
# Localization-error reference convention
# ---------------------------------------------------------------------------
# 'requested' : error is measured from the position the test protocol ASKED for.
#               The uniform test grid is generated independently of the source
#               space, so this includes the discretisation penalty of a coarse
#               source space -- a real property of that source space. This is
#               the definition the published MS1 benchmark reports, the one
#               stated in its Table 2 caption ("independent of the source space,
#               ensuring true spatial estimation rather than grid recovery"),
#               and the one simulate_dipole()'s metadata contract specifies.
#
# 'snapped'   : error is measured from the source point the dipole was snapped
#               onto. Isolates inverse-operator accuracy from source-space
#               density -- a legitimate but DIFFERENT question, and one that
#               flatters sparse source spaces on a grid-independent test.
#
# Both are always computed and persisted ('localization_errors' and
# 'localization_errors_snapped', plus 'snapping_errors' and the raw
# true/snapped/estimated positions) so that either can be reported, neither can
# be silently substituted for the other, and any past run can be re-derived
# under either convention without re-simulating.
#
# History: releases up to v0.3.0 computed ONLY the snapped error but reported it
# under the generic key 'localization_errors', contradicting both the docstring
# contract and the published benchmark. Fixed on this branch.
LOCALIZATION_ERROR_DEFINITION = 'requested'


import json
import sys
from collections import Counter
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import numpy as np
import yaml


def deep_merge(base: Dict, override: Dict) -> Dict:
    """
    Deep merge two dictionaries, with override taking precedence.

    Parameters
    ----------
    base : dict
        Base dictionary
    override : dict
        Override dictionary (values take precedence)

    Returns
    -------
    dict
        Merged dictionary
    """
    result = base.copy()
    for key, value in override.items():
        if key in result and isinstance(result[key], dict) and isinstance(value, dict):
            result[key] = deep_merge(result[key], value)
        else:
            result[key] = value
    return result


class ValidationConfigLoader:
    """
    Handles discovery and loading of validation configuration files.

    This class provides methods for finding bundled test configs,
    discovering configs in custom directories, and loading YAML
    configs with inheritance support.
    """

    @classmethod
    def get_bundled_test_dir(cls, test_name: str) -> Path:
        """
        Get the directory containing bundled test configs.

        Parameters
        ----------
        test_name : str
            Test name: 'original', 'dipole_size', 'conductivity_ratio', 'brain_size'

        Returns
        -------
        Path
            Path to the bundled test config directory

        Raises
        ------
        ValueError
            If test_name is not recognized
        """
        valid_tests = ['original', 'dipole_size', 'conductivity_ratio', 'brain_size']
        if test_name not in valid_tests:
            raise ValueError(f"Unknown test name: {test_name}. Valid: {valid_tests}")

        # Package config directory
        package_dir = Path(__file__).parent
        bundled_dir = package_dir / 'config' / 'default_tests' / test_name

        # Fallback to external validation/test directory (development mode)
        if not bundled_dir.exists():
            # Go up from src/source_localization/validation to root
            root_dir = package_dir.parent.parent.parent.parent.parent
            external_dir = root_dir / 'validation' / 'test' / test_name / 'configs'
            if external_dir.exists():
                return external_dir

        return bundled_dir

    @classmethod
    def discover_configs(
        cls,
        test_name: str,
        test_dir: Optional[Path] = None,
        config_names: Optional[List[str]] = None
    ) -> List[Path]:
        """
        Discover validation config files.

        Parameters
        ----------
        test_name : str
            Test name for bundled configs
        test_dir : Path, optional
            Custom test directory (overrides bundled)
        config_names : list of str, optional
            Filter to specific config names (e.g., ['V01', 'V08'])

        Returns
        -------
        list of Path
            Paths to discovered config files
        """
        if test_dir is not None:
            config_dir = Path(test_dir)
            if not config_dir.exists():
                config_dir = Path(test_dir) / 'configs'
        else:
            config_dir = cls.get_bundled_test_dir(test_name)

        if not config_dir.exists():
            return []

        # Find config files (V*, D*, C*, S* patterns, or *.yaml excluding _base*)
        config_files = []
        for pattern in ['[VDCS]*.yaml', '*.yaml']:
            found = sorted(config_dir.glob(pattern))
            for f in found:
                if not f.stem.startswith('_') and f not in config_files:
                    config_files.append(f)

        # Filter by requested config names
        if config_names:
            config_files = [
                f for f in config_files
                if any(name in f.stem for name in config_names)
            ]

        return config_files

    @classmethod
    def load_config(cls, config_path: Path) -> Dict[str, Any]:
        """
        Load a YAML config file, handling _base inheritance.

        Parameters
        ----------
        config_path : Path
            Path to the config file

        Returns
        -------
        dict
            Loaded and merged configuration
        """
        with open(config_path) as f:
            config = yaml.safe_load(f)

        # Handle base config inheritance
        if '_base' in config:
            base_path = config_path.parent / config['_base']
            if base_path.exists():
                with open(base_path) as f:
                    base_config = yaml.safe_load(f)
                # Merge: base values, then override with specific config
                merged = deep_merge(base_config, config)
                del merged['_base']
                return merged

        return config


class ValidationRunner:
    """
    Runs dipole simulation validation for a single configuration.

    This class encapsulates the pipeline setup, simulation loop,
    and results saving for validation testing.

    Parameters
    ----------
    config_path : Path or str
        Path to the validation config YAML file
    output_dir : Path or str, optional
        Output directory (default: derived from config)
    verbose : bool
        Print progress messages

    Attributes
    ----------
    config : dict
        Loaded configuration
    config_name : str
        Name of the configuration (filename stem)
    output_dir : Path
        Output directory for results
    pipeline_components : dict
        Built pipeline components after setup()
    """

    def __init__(
        self,
        config_path: Union[Path, str],
        output_dir: Optional[Union[Path, str]] = None,
        verbose: bool = True,
        atlas_overrides: Optional[Dict[str, str]] = None
    ):
        self.config_path = Path(config_path)
        self.config = ValidationConfigLoader.load_config(self.config_path)
        self.config_name = self.config_path.stem
        self.verbose = verbose

        # Apply atlas overrides if provided
        if atlas_overrides:
            if 'inputs' not in self.config:
                self.config['inputs'] = {}
            for key, value in atlas_overrides.items():
                self.config['inputs'][key] = value
            # Update config name to reflect atlas
            atlas_suffix = atlas_overrides.get('_atlas_name', 'custom')
            self.config_name = f"{self.config_name}_{atlas_suffix}"

        # Setup output directory
        if output_dir is not None:
            self.output_dir = Path(output_dir)
        else:
            # Use config's outputs.dir if specified, otherwise default to cwd/validation/results/{config_name}
            config_output_dir = self.config.get('outputs', {}).get('dir')
            if config_output_dir:
                # Config specifies output dir - use it relative to cwd
                self.output_dir = Path.cwd() / config_output_dir
            else:
                # No config output dir - use default
                self.output_dir = Path.cwd() / 'validation' / 'results' / self.config_name

        # Will be populated by setup()
        self.pipeline_components = {}
        self._setup_complete = False

    def setup(self) -> None:
        """
        Build pipeline components (electrodes, BEM, source space, forward).

        This must be called before run().
        """
        from source_localization.config import Config
        from source_localization.steps import (
            electrode_registration,
            bem_model,
            source_space,
            forward_solution
        )

        config = self.config

        # Get scale factor for human-scale testing
        scale_factor = config.get('validation', {}).get('scale_factor', 1.0)

        if self.verbose:
            print(f"\n{'='*60}")
            print(f"Running validation: {config['pipeline']['name']}")
            print(f"Config: {self.config_name}")
            if scale_factor != 1.0:
                print(f"Scale factor: {scale_factor}x (human-scale test)")
            print(f"{'='*60}")

        # Setup output directory structure
        self.output_dir.mkdir(parents=True, exist_ok=True)
        (self.output_dir / 'data').mkdir(exist_ok=True)
        (self.output_dir / 'figures').mkdir(exist_ok=True)

        # Update config with absolute output path
        config['outputs']['dir'] = str(self.output_dir)

        if self.verbose:
            print("\n  Building pipeline components...")

        # Wrap config dict in Config object
        pipeline_config = Config(config)

        # Step 1: Electrode registration
        if self.verbose:
            print("    Step 1: Electrode registration...")
        step1_outputs = electrode_registration.run(pipeline_config, {})

        # Apply scale factor to electrode positions if needed
        if scale_factor != 1.0:
            if self.verbose:
                print(f"    Applying {scale_factor}x scale to electrode positions...")
            info = step1_outputs['info']
            for ch in info['chs']:
                if ch['loc'] is not None:
                    ch['loc'][:3] *= scale_factor
            if info['dig'] is not None:
                for dig_point in info['dig']:
                    if dig_point['r'] is not None:
                        dig_point['r'] = dig_point['r'] * scale_factor
            step1_outputs['info'] = info
            if 'electrode_positions_mm' in step1_outputs:
                step1_outputs['electrode_positions_mm'] *= scale_factor

        # Step 2: BEM model
        if self.verbose:
            print("    Step 2: BEM model...")
        if scale_factor != 1.0:
            bem_type = config['pipeline']['bem_type']
            # Pass scale_factor to BEM config so geometry gets scaled
            config['bem'][bem_type]['scale_factor'] = scale_factor
            # Disable BEM cache for scaled tests (different geometry)
            config['bem'][bem_type]['use_cache'] = False
            pipeline_config = Config(config)
        step2_outputs = bem_model.run(pipeline_config, step1_outputs)

        # Step 3: Source space
        if self.verbose:
            print("    Step 3: Source space...")
        previous = {**step1_outputs, **step2_outputs}
        step3_outputs = source_space.run(pipeline_config, previous)

        # Apply scale factor to source space if needed
        if scale_factor != 1.0:
            if self.verbose:
                print(f"    Applying {scale_factor}x scale to source space coordinates...")

            # Scale source coordinates (mm)
            step3_outputs['source_coords_mm'] = step3_outputs['source_coords_mm'] * scale_factor

            # Scale MNE source space object (positions in meters)
            src = step3_outputs['src']
            for s in src:
                s['rr'] = s['rr'] * scale_factor
            step3_outputs['src'] = src

        # Step 4: Forward solution(s)
        if self.verbose:
            print("    Step 4: Forward solution...")
        previous = {**step1_outputs, **step2_outputs, **step3_outputs}

        # Check for forward model mismatch testing
        val_config = config.get('validation', {})
        forward_mismatch = val_config.get('forward_model_mismatch', False)
        ground_truth_conductivities = val_config.get('ground_truth_conductivities', None)

        # Get BEM type from pipeline config
        bem_type = config.get('pipeline', {}).get('bem_type', 'sphere')

        if forward_mismatch and ground_truth_conductivities:
            if self.verbose:
                print("    Forward model mismatch mode enabled!")
                print(f"    Ground truth conductivities: {ground_truth_conductivities}")
                print(f"    Test conductivities: {config['bem'][bem_type]['conductivities']}")

            # Build GROUND TRUTH forward model (for simulation)
            if self.verbose:
                print("    Building ground truth forward model (for simulation)...")
            ground_truth_config = config.copy()
            ground_truth_config['bem'] = config['bem'].copy()
            ground_truth_config['bem'][bem_type] = config['bem'][bem_type].copy()
            ground_truth_config['bem'][bem_type]['conductivities'] = ground_truth_conductivities
            ground_truth_config['bem'][bem_type]['use_cache'] = False
            ground_truth_pipeline_config = Config(ground_truth_config)

            step2_ground_truth = bem_model.run(ground_truth_pipeline_config, step1_outputs)
            previous_ground_truth = {**step1_outputs, **step2_ground_truth, **step3_outputs}
            step4_ground_truth = forward_solution.run(ground_truth_pipeline_config, previous_ground_truth)
            fwd_simulation = step4_ground_truth['fwd']

            # Build TEST forward model (for inversion)
            if self.verbose:
                print("    Building test forward model (for inversion)...")
            config['bem'][bem_type]['use_cache'] = False
            pipeline_config = Config(config)
            step2_test = bem_model.run(pipeline_config, step1_outputs)
            previous_test = {**step1_outputs, **step2_test, **step3_outputs}
            step4_test = forward_solution.run(pipeline_config, previous_test)
            fwd_inversion = step4_test['fwd']

            step4_outputs = step4_test
        else:
            # Standard mode: same forward model for simulation and inversion
            step4_outputs = forward_solution.run(pipeline_config, previous)
            fwd_simulation = step4_outputs['fwd']
            fwd_inversion = step4_outputs['fwd']

        # Extract source coordinates from forward solution
        source_coords_mm = fwd_inversion['source_rr'] * 1000
        n_sources_fwd = len(source_coords_mm)
        n_sources_step3 = len(step3_outputs['source_coords_mm'])
        if n_sources_fwd != n_sources_step3 and self.verbose:
            print(f"    Note: {n_sources_step3 - n_sources_fwd} sources filtered by forward solution")
            print(f"    Using {n_sources_fwd} sources from forward solution")

        # Store pipeline components
        self.pipeline_components = {
            'info': step1_outputs['info'],
            'src': step3_outputs['src'],
            'fwd_simulation': fwd_simulation,
            'fwd_inversion': fwd_inversion,
            'source_coords_mm': source_coords_mm,
            'scale_factor': scale_factor,
            'forward_mismatch': forward_mismatch,
            'ground_truth_conductivities': ground_truth_conductivities
        }

        self._setup_complete = True

    def run(
        self,
        snr_levels: Optional[List[float]] = None,
        n_trials: Optional[int] = None,
        roi_indices: Optional[List[int]] = None,
        test_mode: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Run the validation simulation loop.

        Parameters
        ----------
        snr_levels : list of float, optional
            Override SNR levels (default: from config)
        n_trials : int, optional
            Override number of trials per location (default: from config)
        roi_indices : list of int, optional
            Run only for specific ROI indices (default: all)
        test_mode : str, optional
            Override test mode:
            - "roi_centroids": Test at ROI centroid positions (legacy, best for ROI accuracy)
            - "uniform_grid": Test at uniform grid positions (best for localization error)
            - "combined": Run BOTH modes and report ROI accuracy from centroids,
              localization error from uniform grid (recommended for comprehensive validation)
            Default: from config, or "roi_centroids" if not specified.

        Returns
        -------
        dict
            Validation results with metrics for each SNR level
        """
        if not self._setup_complete:
            raise RuntimeError("Must call setup() before run()")

        from source_localization.validation import (
            DipoleSimulator,
            compute_localization_error,
            compute_roi_classification_accuracy,
            load_atlas_roi_centroids,
            load_roi_source_mapping,
            generate_uniform_test_grid
        )

        config = self.config
        val_config = config.get('validation', {})
        dipole_config = val_config.get('dipole', {})

        # Get parameters
        if snr_levels is None:
            snr_levels = val_config.get('snr_levels', [10])
        if n_trials is None:
            n_trials = val_config.get('n_trials', dipole_config.get('n_trials', 25))
        if test_mode is None:
            test_mode = val_config.get('test_mode', 'roi_centroids')

        # Handle combined mode: run both roi_centroids and uniform_grid, merge results
        if test_mode == 'combined':
            return self._run_combined_mode(snr_levels, n_trials, roi_indices)

        # Grid settings for uniform_grid mode
        grid_spacing_mm = val_config.get('grid_spacing_mm', 1.0)
        grid_margin_mm = val_config.get('grid_margin_mm', 0.2)

        # Extract pipeline components
        info = self.pipeline_components['info']
        src = self.pipeline_components['src']
        fwd_simulation = self.pipeline_components['fwd_simulation']
        fwd_inversion = self.pipeline_components['fwd_inversion']
        source_coords_mm = self.pipeline_components['source_coords_mm']
        scale_factor = self.pipeline_components['scale_factor']
        forward_mismatch = self.pipeline_components['forward_mismatch']
        ground_truth_conductivities = self.pipeline_components['ground_truth_conductivities']

        # Extract electrode coordinates for depth calculation
        electrode_coords_mm = np.array([
            ch['loc'][:3] * 1000 for ch in info['chs']
            if ch['kind'] == 2  # EEG channels
        ])

        # Initialize simulator
        if self.verbose:
            print("  Initializing dipole simulator...")
            if forward_mismatch:
                print("    Using ground truth forward model for simulation")
        simulator = DipoleSimulator(fwd_simulation, info, src, verbose=False)

        # Load ROI information
        import source_localization
        package_dir = Path(source_localization.__file__).parent

        roi_labels_file = package_dir / config['inputs']['brain_labels']
        roi_names_file = package_dir / config['inputs'].get('roi_mapping', 'data/atlas/roi_mapping.json')

        if self.verbose:
            print(f"  Loading ROIs from: {roi_labels_file.name}")
        roi_centroids, roi_names_map = load_atlas_roi_centroids(str(roi_labels_file), str(roi_names_file))

        # Apply scale factor to ROI centroids
        if scale_factor != 1.0:
            if self.verbose:
                print(f"    Scaling ROI centroids by {scale_factor}x...")
            roi_centroids = {k: v * scale_factor for k, v in roi_centroids.items()}

        # Get ROI-source mapping for peak ROI classification
        # IMPORTANT: Use pipeline's roi_assignments for consistency with how ROIs
        # are reported in downstream analysis. The pipeline assigns sources to ROIs
        # during source space creation (e.g., PCA placement within ROI boundaries).
        # Using KD-tree lookup can give different results when sources are near
        # ROI boundaries or placed using PCA spread.
        if 'roi_assignments' in src[0]:
            roi_mapping = np.array(src[0]['roi_assignments']).astype(int)
            if self.verbose:
                n_rois_with_sources = len(np.unique(roi_mapping[roi_mapping > 0]))
                print(f"  Using pipeline's ROI assignments ({n_rois_with_sources} ROIs with sources)")
        else:
            # Fall back to KD-tree lookup if pipeline doesn't have roi_assignments
            if self.verbose:
                print("  Computing ROI mapping via KD-tree (pipeline has no roi_assignments)")
            mapping_radius = 2.0 * scale_factor
            roi_mapping = load_roi_source_mapping(
                source_coords_mm, str(roi_labels_file),
                radius_mm=mapping_radius, scale_factor=scale_factor
            )

        # Determine test positions based on mode
        roi_placement_meta = None
        if test_mode == 'uniform_grid':
            if self.verbose:
                print(f"  Using UNIFORM GRID test mode (spacing={grid_spacing_mm}mm, margin={grid_margin_mm}mm)")

            # Generate uniform test grid (on mouse-scale atlas)
            # Note: spacing and margin are NOT scaled - grid is generated on actual atlas
            # Only the resulting positions are scaled for human-scale tests
            test_positions, test_roi_labels = generate_uniform_test_grid(
                str(roi_labels_file),
                spacing_mm=grid_spacing_mm,
                margin_mm=grid_margin_mm,
                verbose=self.verbose
            )

            # Apply scale factor
            if scale_factor != 1.0:
                test_positions = test_positions * scale_factor

            n_test_positions = len(test_positions)

            # Create position info list (position_mm, roi_label, position_name)
            test_points = [
                (test_positions[i], int(test_roi_labels[i]), f"Grid_{i:04d}")
                for i in range(n_test_positions)
            ]
        else:
            # ROI-based test mode. `roi_placement` decides WHERE inside each ROI
            # the test dipole goes:
            #   'medoid'  (default) nearest in-ROI voxel to the centroid
            #   'sample'  n_per_roi points spread within the ROI
            #   'centroid' legacy; the geometric centroid, which for non-convex
            #             parcels can lie OUTSIDE the ROI and then scores the
            #             trial against the wrong label (5/32 ROIs in Allen-32:
            #             Hippocampus_Post_L/R -> Thalamus, Lateral_Cortex_L/R
            #             and Cerebellum_L -> unlabeled). See
            #             load_atlas_roi_test_points().
            roi_placement = val_config.get('roi_placement', 'medoid')
            roi_n_per = int(val_config.get('roi_points_per_roi', 1))

            if roi_placement == 'centroid':
                if self.verbose:
                    print("  Using ROI_CENTROIDS test mode (LEGACY placement -- "
                          "non-convex ROIs may be tested outside themselves)")
                roi_ids = sorted([rid for rid in roi_centroids.keys() if rid != 0])
                if roi_indices is not None:
                    roi_ids = [roi_ids[i] for i in roi_indices if i < len(roi_ids)]
                test_points = [
                    (roi_centroids[roi_id], roi_id,
                     roi_names_map.get(roi_id, f"ROI_{roi_id}"))
                    for roi_id in roi_ids
                ]
                roi_placement_meta = {'placement': 'centroid', 'n_per_roi': 1}
            else:
                from .utils import load_atlas_roi_test_points
                if self.verbose:
                    print(f"  Using ROI test mode (placement={roi_placement}, "
                          f"n_per_roi={roi_n_per})")
                pts_by_roi, _, roi_placement_meta = load_atlas_roi_test_points(
                    str(roi_labels_file), str(roi_names_file),
                    placement=roi_placement, n_per_roi=roi_n_per,
                    sample_strategy=val_config.get('roi_sample_strategy', 'random'),
                    enforce_lr_symmetry=bool(
                        val_config.get('roi_enforce_lr_symmetry', True)),
                    seed=int(val_config.get('roi_sample_seed', 20260728)),
                )
                roi_ids = sorted([rid for rid in pts_by_roi.keys() if rid != 0])
                if roi_indices is not None:
                    roi_ids = [roi_ids[i] for i in roi_indices if i < len(roi_ids)]
                test_points = []
                for roi_id in roi_ids:
                    pts = np.atleast_2d(pts_by_roi[roi_id]) * scale_factor
                    base = roi_names_map.get(roi_id, f"ROI_{roi_id}")
                    for k, p in enumerate(pts):
                        name = base if len(pts) == 1 else f"{base}#{k+1}"
                        test_points.append((p, roi_id, name))

            n_test_positions = len(test_points)

        n_rois = len(set(p[1] for p in test_points if p[1] > 0))

        # Initialize results structure
        all_results = {
            'config_name': self.config_name,
            'pipeline_name': config['pipeline']['name'],
            'bem_type': config['pipeline']['bem_type'],
            'source_type': config['pipeline']['source_type'],
            'inverse_method': config['inverse']['method'],
            'n_sources': len(source_coords_mm),
            'n_rois': n_rois,
            'n_test_positions': len(test_points),
            'test_mode': test_mode,
            'grid_spacing_mm': grid_spacing_mm if test_mode == 'uniform_grid' else None,
            'grid_margin_mm': grid_margin_mm if test_mode == 'uniform_grid' else None,
            'scale_factor': scale_factor,
            'forward_model_mismatch': forward_mismatch,
            'ground_truth_conductivities': ground_truth_conductivities if forward_mismatch else None,
            'test_conductivities': config['bem'][config['pipeline']['bem_type']]['conductivities'] if forward_mismatch else None,
            'dipole_amplitude_nAm': dipole_config.get('amplitude_nAm', 50.0),
            'noise_mode': dipole_config.get('noise_mode', 'snr'),
            'noise_variance_uV2': dipole_config.get('noise_variance_uV2', 1.0) if dipole_config.get('noise_mode') == 'fixed_variance' else None,
            'timestamp': datetime.now().isoformat(),
            # Which reference position 'localization_errors' is measured FROM.
            # See LOCALIZATION_ERROR_DEFINITION. Recorded so every downstream
            # table can state its own definition without archaeology.
            'localization_error_definition': LOCALIZATION_ERROR_DEFINITION,
            # Where inside each ROI the test dipole was placed. Legacy
            # 'centroid' can fall outside non-convex parcels; recorded so an
            # ROI-accuracy table always states its own placement.
            'roi_placement': (roi_placement_meta
                              if test_mode != 'uniform_grid' else None),
            'snr_results': {}
        }

        # Run simulation loop
        for snr_db in snr_levels:
            if self.verbose:
                print(f"\n  Testing at SNR = {snr_db} dB...")

            snr_metrics = {
                # PRIMARY error: peak vs REQUESTED position (see
                # LOCALIZATION_ERROR_DEFINITION).
                'localization_errors': [],
                # DIAGNOSTIC error: peak vs SNAPPED source position.
                'localization_errors_snapped': [],
                # Distance requested -> snapped; explains the gap between the two.
                'snapping_errors': [],
                'roi_correct': [],
                'adjacent_roi_correct': [],
                'amplitude_ratios': [],
                'depths': [],
                'position_names': [],
                'true_roi_ids': [],
                'estimated_roi_ids': [],
                'estimated_roi_names': [],
                'true_positions_mm': [],
                'snapped_positions_mm': [],
                'estimated_positions_mm': [],
                'peak_in_background': []  # Track when absolute peak is in ROI 0
            }

            for i, (test_pos, true_roi_id, pos_name) in enumerate(test_points):
                if self.verbose and i % 10 == 0:
                    print(f"    Processing position {i+1}/{len(test_points)}: {pos_name}", flush=True)

                for trial in range(n_trials):
                    # Simulate dipole at test position
                    eeg_data, sim_meta = simulator.simulate_dipole(
                        position_mm=test_pos,
                        amplitude_nAm=dipole_config.get('amplitude_nAm', 50.0),
                        duration_s=dipole_config.get('duration_s', 1.0),
                        sfreq=dipole_config.get('sfreq', 500.0),
                        snr_db=snr_db,
                        noise_seed=trial * 1000 + i,
                        noise_mode=dipole_config.get('noise_mode', 'snr'),
                        noise_variance_uV2=dipole_config.get('noise_variance_uV2', 1.0)
                    )

                    # Create MNE Raw object
                    raw = simulator.create_mne_raw(eeg_data, sfreq=dipole_config.get('sfreq', 500.0))

                    # Apply inverse solution
                    stc = self._apply_inverse(raw, fwd_inversion, info, config['inverse'])

                    # Find peak source locations
                    # Sort sources by activation strength (descending)
                    source_activations = np.abs(stc.data).mean(axis=1)
                    sorted_indices = np.argsort(source_activations)[::-1]

                    # ABSOLUTE peak: highest activation regardless of ROI (for localization error)
                    absolute_peak_idx = sorted_indices[0]

                    # Validate index before access
                    if absolute_peak_idx >= len(source_coords_mm):
                        raise IndexError(
                            f"Peak index {absolute_peak_idx} out of bounds for {len(source_coords_mm)} sources. "
                            f"stc.data shape: {stc.data.shape}, fwd n_dipoles: {fwd_inversion['sol']['data'].shape[1]}"
                        )

                    absolute_peak_position = source_coords_mm[absolute_peak_idx]

                    # VALID ROI peak: highest activation with valid ROI (for ROI accuracy)
                    valid_roi_peak_idx = sorted_indices[0]
                    for idx in sorted_indices:
                        if roi_mapping[idx] != 0:
                            valid_roi_peak_idx = idx
                            break

                    # Localization error under BOTH reference conventions -- see
                    # LOCALIZATION_ERROR_DEFINITION in this module for the rationale.
                    requested_position_mm = np.array(sim_meta['requested_position_mm'])
                    snapped_position_mm = np.array(sim_meta['snapped_source_position_mm'])
                    loc_error = compute_localization_error(
                        requested_position_mm,
                        absolute_peak_position
                    ) / scale_factor
                    loc_error_snapped = compute_localization_error(
                        snapped_position_mm,
                        absolute_peak_position
                    ) / scale_factor
                    snapping_error = compute_localization_error(
                        requested_position_mm,
                        snapped_position_mm
                    ) / scale_factor

                    # Determine estimated ROI from VALID ROI peak (for ROI accuracy)
                    # ROI 0 is background - use valid peak for ROI classification
                    estimated_roi = roi_mapping[valid_roi_peak_idx]
                    estimated_roi_name = roi_names_map.get(estimated_roi, f"ROI_{estimated_roi}")
                    roi_correct = compute_roi_classification_accuracy(true_roi_id, estimated_roi)

                    # Track if absolute peak was in background (ROI 0)
                    peak_in_background = (roi_mapping[absolute_peak_idx] == 0)

                    # Get depth as distance to nearest electrode
                    depth = self._compute_source_depth(requested_position_mm, electrode_coords_mm)

                    # Store metrics
                    snr_metrics['localization_errors'].append(loc_error)
                    snr_metrics['localization_errors_snapped'].append(loc_error_snapped)
                    snr_metrics['snapping_errors'].append(snapping_error)
                    snr_metrics['roi_correct'].append(roi_correct)
                    snr_metrics['depths'].append(depth)
                    snr_metrics['position_names'].append(pos_name)
                    snr_metrics['true_roi_ids'].append(true_roi_id)
                    snr_metrics['estimated_roi_ids'].append(int(estimated_roi))
                    snr_metrics['estimated_roi_names'].append(estimated_roi_name)
                    snr_metrics['true_positions_mm'].append(requested_position_mm.tolist())
                    snr_metrics['snapped_positions_mm'].append(snapped_position_mm.tolist())
                    snr_metrics['estimated_positions_mm'].append(
                        np.asarray(absolute_peak_position).tolist()
                    )
                    snr_metrics['peak_in_background'].append(peak_in_background)

            # Compute per-position metrics (compatible with both modes)
            per_position_metrics = {}
            for i, pos_name in enumerate(snr_metrics['position_names']):
                if pos_name not in per_position_metrics:
                    per_position_metrics[pos_name] = {
                        'true_roi_id': snr_metrics['true_roi_ids'][i],
                        'localization_errors': [],
                        'roi_correct': [],
                        'estimated_rois': [],
                        'depth_mm': snr_metrics['depths'][i]  # Same for all trials at this position
                    }
                per_position_metrics[pos_name]['localization_errors'].append(snr_metrics['localization_errors'][i])
                per_position_metrics[pos_name]['roi_correct'].append(snr_metrics['roi_correct'][i])
                per_position_metrics[pos_name]['estimated_rois'].append(snr_metrics['estimated_roi_names'][i])

            # Summarize per-position
            per_position_summary = {}
            for pos_name, metrics in per_position_metrics.items():
                est_counts = Counter(metrics['estimated_rois'])
                most_common_est = est_counts.most_common(1)[0] if est_counts else (None, 0)

                per_position_summary[pos_name] = {
                    'true_roi_id': metrics['true_roi_id'],
                    'n_trials': len(metrics['localization_errors']),
                    'localization_error_mm': float(np.mean(metrics['localization_errors'])),
                    'source_depth_mm': float(metrics['depth_mm']),
                    'roi_correct': bool(np.all(metrics['roi_correct'])),
                    'roi_accuracy': float(np.mean(metrics['roi_correct'])),
                    'estimated_as': most_common_est[0],
                    'estimated_as_count': most_common_est[1]
                }

            # Compute depth-stratified metrics (depth = distance to nearest electrode)
            depth_bins = {
                '1-2mm': (1, 2),
                '2-3mm': (2, 3),
                '3-4mm': (3, 4),
                '4-5mm': (4, 5),
                '5+mm': (5, float('inf'))
            }
            depth_stratified = {}
            for bin_name, (low, high) in depth_bins.items():
                # Get all trials at positions in this depth range
                bin_loc_errors = []
                bin_roi_correct = []
                for i, depth in enumerate(snr_metrics['depths']):
                    if low <= depth < high:
                        bin_loc_errors.append(snr_metrics['localization_errors'][i])
                        bin_roi_correct.append(snr_metrics['roi_correct'][i])

                if bin_loc_errors:
                    depth_stratified[bin_name] = {
                        'n_trials': len(bin_loc_errors),
                        'localization_error_mean': float(np.mean(bin_loc_errors)),
                        'localization_error_std': float(np.std(bin_loc_errors)),
                        'roi_accuracy': float(np.mean(bin_roi_correct))
                    }
                else:
                    depth_stratified[bin_name] = None

            # Compute depth bias metrics
            from .metrics import compute_depth_bias
            depths_array = np.array(snr_metrics['depths'])
            errors_array = np.array(snr_metrics['localization_errors'])
            depth_bias = compute_depth_bias(errors_array, depths_array)

            # Compute summary statistics
            peak_in_bg_count = sum(snr_metrics['peak_in_background'])
            peak_in_bg_rate = peak_in_bg_count / len(snr_metrics['peak_in_background']) if snr_metrics['peak_in_background'] else 0.0

            snr_summary = {
                'snr_db': snr_db,
                'n_simulations': len(snr_metrics['localization_errors']),
                'n_test_positions': len(per_position_summary),
                'localization_error': {
                    'mean': float(np.mean(snr_metrics['localization_errors'])),
                    'std': float(np.std(snr_metrics['localization_errors'])),
                    'median': float(np.median(snr_metrics['localization_errors'])),
                    'min': float(np.min(snr_metrics['localization_errors'])),
                    'max': float(np.max(snr_metrics['localization_errors']))
                },
                'roi_accuracy': {
                    'exact': float(np.mean(snr_metrics['roi_correct'])),
                    'n_correct': int(np.sum(snr_metrics['roi_correct'])),
                    'n_total': len(snr_metrics['roi_correct'])
                },
                'peak_in_background': {
                    'count': peak_in_bg_count,
                    'rate': float(peak_in_bg_rate)
                },
                'depth_bias': {
                    'correlation': float(depth_bias['correlation']) if not np.isnan(depth_bias['correlation']) else None,
                    'slope_mm_per_mm': float(depth_bias['slope']) if not np.isnan(depth_bias['slope']) else None,
                    'superficial_error_mm': float(depth_bias['superficial_error_mm']) if not np.isnan(depth_bias['superficial_error_mm']) else None,
                    'deep_error_mm': float(depth_bias['deep_error_mm']) if not np.isnan(depth_bias['deep_error_mm']) else None,
                },
                'depth_stratified': depth_stratified,
                'per_position': per_position_summary,
                'raw_data': snr_metrics
            }

            all_results['snr_results'][snr_db] = snr_summary

            if self.verbose:
                print(f"    Mean localization error: {snr_summary['localization_error']['mean']:.2f} mm")
                print(f"    ROI accuracy: {snr_summary['roi_accuracy']['exact']*100:.1f}%")
                if test_mode == 'uniform_grid':
                    print(f"    (Tested at {len(per_position_summary)} uniform grid positions)")

        return all_results

    def _run_combined_mode(
        self,
        snr_levels: List[float],
        n_trials: int,
        roi_indices: Optional[List[int]] = None
    ) -> Dict[str, Any]:
        """
        Run combined validation mode: ROI centroids for accuracy, uniform grid for localization.

        This method runs two separate validation passes:
        1. ROI centroids: Used for ROI classification accuracy (tests at ROI centers)
        2. Uniform grid: Used for localization error (tests throughout brain volume)

        The results are merged to provide the most appropriate metric from each mode.

        Parameters
        ----------
        snr_levels : list of float
            SNR levels to test
        n_trials : int
            Number of trials per position
        roi_indices : list of int, optional
            ROI indices for centroid testing

        Returns
        -------
        dict
            Combined validation results
        """
        if self.verbose:
            print("\n  === COMBINED MODE: Running both test modes ===")
            print("  ROI centroids → ROI accuracy | Uniform grid → Localization error\n")

        # Run ROI centroids mode (for ROI accuracy)
        if self.verbose:
            print("  [1/2] Running ROI CENTROIDS mode (for ROI accuracy)...")
        centroids_results = self._run_single_mode(
            snr_levels, n_trials, roi_indices, test_mode='roi_centroids'
        )

        # Run uniform grid mode (for localization error)
        if self.verbose:
            print("\n  [2/2] Running UNIFORM GRID mode (for localization error)...")
        grid_results = self._run_single_mode(
            snr_levels, n_trials, roi_indices, test_mode='uniform_grid'
        )

        # Merge results: take ROI accuracy from centroids, localization error from grid
        combined_results = {
            'config_name': self.config_name,
            'pipeline_name': centroids_results['pipeline_name'],
            'bem_type': centroids_results['bem_type'],
            'source_type': centroids_results['source_type'],
            'inverse_method': centroids_results['inverse_method'],
            'n_sources': centroids_results['n_sources'],
            'n_rois': centroids_results['n_rois'],
            'test_mode': 'combined',
            'combined_mode_details': {
                'roi_accuracy_source': 'roi_centroids',
                'roi_centroids_n_positions': centroids_results['n_test_positions'],
                'localization_error_source': 'uniform_grid',
                'uniform_grid_n_positions': grid_results['n_test_positions'],
                'grid_spacing_mm': grid_results.get('grid_spacing_mm'),
                'grid_margin_mm': grid_results.get('grid_margin_mm'),
            },
            'n_test_positions': {
                'roi_centroids': centroids_results['n_test_positions'],
                'uniform_grid': grid_results['n_test_positions']
            },
            'scale_factor': centroids_results['scale_factor'],
            'forward_model_mismatch': centroids_results['forward_model_mismatch'],
            'ground_truth_conductivities': centroids_results.get('ground_truth_conductivities'),
            'test_conductivities': centroids_results.get('test_conductivities'),
            'dipole_amplitude_nAm': centroids_results['dipole_amplitude_nAm'],
            'noise_mode': centroids_results['noise_mode'],
            'noise_variance_uV2': centroids_results.get('noise_variance_uV2'),
            'timestamp': centroids_results['timestamp'],
            'snr_results': {}
        }

        # Merge SNR-level results
        for snr_db in snr_levels:
            snr_key = snr_db if snr_db in centroids_results['snr_results'] else str(snr_db)
            centroid_snr = centroids_results['snr_results'].get(snr_key, centroids_results['snr_results'].get(snr_db, {}))
            grid_snr = grid_results['snr_results'].get(snr_key, grid_results['snr_results'].get(snr_db, {}))

            combined_snr = {
                'snr_db': snr_db,
                # Total simulations from both modes
                'n_simulations': {
                    'roi_centroids': centroid_snr.get('n_simulations', 0),
                    'uniform_grid': grid_snr.get('n_simulations', 0),
                    'total': centroid_snr.get('n_simulations', 0) + grid_snr.get('n_simulations', 0)
                },
                'n_test_positions': {
                    'roi_centroids': centroid_snr.get('n_test_positions', 0),
                    'uniform_grid': grid_snr.get('n_test_positions', 0)
                },
                # ROI ACCURACY from centroids (best measure of ROI classification)
                'roi_accuracy': centroid_snr.get('roi_accuracy', {}),
                'roi_accuracy_source': 'roi_centroids',
                # LOCALIZATION ERROR from uniform grid (position-independent measure)
                'localization_error': grid_snr.get('localization_error', {}),
                'localization_error_source': 'uniform_grid',
                # Depth-stratified metrics - each from appropriate source
                # ROI accuracy by depth: from roi_centroids (tests at ROI centers)
                # Localization error by depth: from uniform_grid (position-independent)
                'depth_stratified_roi_accuracy': self._extract_depth_roi_accuracy(
                    centroid_snr.get('depth_stratified', {})
                ),
                'depth_stratified_localization_error': self._extract_depth_loc_error(
                    grid_snr.get('depth_stratified', {})
                ),
                # Depth bias metrics
                'depth_bias': {
                    'localization_error': grid_snr.get('depth_bias', {}),
                    'roi_accuracy': centroid_snr.get('depth_bias', {}),
                },
                # Peak in background from both (for comparison)
                'peak_in_background': {
                    'roi_centroids': centroid_snr.get('peak_in_background', {}),
                    'uniform_grid': grid_snr.get('peak_in_background', {})
                },
                # Per-position data from both modes
                'per_position': {
                    'roi_centroids': centroid_snr.get('per_position', {}),
                    'uniform_grid': grid_snr.get('per_position', {})
                },
                # Raw data from both modes
                'raw_data': {
                    'roi_centroids': centroid_snr.get('raw_data', {}),
                    'uniform_grid': grid_snr.get('raw_data', {})
                }
            }

            combined_results['snr_results'][snr_db] = combined_snr

        # Print combined summary with depth-stratified metrics
        if self.verbose:
            print(f"\n  === COMBINED MODE SUMMARY ===")
            for snr_db in snr_levels:
                snr_data = combined_results['snr_results'][snr_db]
                roi_acc = snr_data['roi_accuracy'].get('exact', 0)
                loc_err = snr_data['localization_error'].get('mean', 0)
                print(f"    SNR={snr_db}dB: ROI Accuracy={roi_acc*100:.1f}% (from centroids), "
                      f"Localization Error={loc_err:.2f}mm (from grid)")

                # Show depth-stratified summary
                depth_loc = snr_data.get('depth_stratified_localization_error', {})
                depth_roi = snr_data.get('depth_stratified_roi_accuracy', {})

                print(f"\n    Depth-Stratified Results (by electrode distance):")
                print(f"    {'Depth':<8} {'Loc Error (grid)':>18} {'ROI Acc (centroids)':>20}")
                print(f"    {'-'*50}")
                for bin_name in ['1-2mm', '2-3mm', '3-4mm', '4-5mm', '5+mm']:
                    loc_bin = depth_loc.get(bin_name)
                    roi_bin = depth_roi.get(bin_name)
                    loc_str = f"{loc_bin['localization_error_mean']:.2f}mm" if loc_bin else "N/A"
                    roi_str = f"{roi_bin['roi_accuracy']*100:.1f}%" if roi_bin else "N/A"
                    print(f"    {bin_name:<8} {loc_str:>18} {roi_str:>20}")

        return combined_results

    def _extract_depth_roi_accuracy(self, depth_stratified: Dict) -> Dict:
        """Extract only ROI accuracy metrics from depth-stratified data (from roi_centroids)."""
        result = {}
        for bin_name, bin_data in depth_stratified.items():
            if bin_data is not None:
                result[bin_name] = {
                    'n_trials': bin_data.get('n_trials', 0),
                    'roi_accuracy': bin_data.get('roi_accuracy', 0.0)
                }
            else:
                result[bin_name] = None
        return result

    def _extract_depth_loc_error(self, depth_stratified: Dict) -> Dict:
        """Extract only localization error metrics from depth-stratified data (from uniform_grid)."""
        result = {}
        for bin_name, bin_data in depth_stratified.items():
            if bin_data is not None:
                result[bin_name] = {
                    'n_trials': bin_data.get('n_trials', 0),
                    'localization_error_mean': bin_data.get('localization_error_mean', 0.0),
                    'localization_error_std': bin_data.get('localization_error_std', 0.0)
                }
            else:
                result[bin_name] = None
        return result

    def _run_single_mode(
        self,
        snr_levels: List[float],
        n_trials: int,
        roi_indices: Optional[List[int]],
        test_mode: str
    ) -> Dict[str, Any]:
        """
        Run validation in a single test mode (helper for combined mode).

        This is essentially the main run() logic but extracted to allow
        running multiple modes sequentially.
        """
        from source_localization.validation import (
            DipoleSimulator,
            compute_localization_error,
            compute_roi_classification_accuracy,
            load_atlas_roi_centroids,
            load_roi_source_mapping,
            generate_uniform_test_grid
        )

        config = self.config
        val_config = config.get('validation', {})
        dipole_config = val_config.get('dipole', {})

        # Grid settings for uniform_grid mode
        grid_spacing_mm = val_config.get('grid_spacing_mm', 1.0)
        grid_margin_mm = val_config.get('grid_margin_mm', 0.2)

        # Extract pipeline components
        info = self.pipeline_components['info']
        src = self.pipeline_components['src']
        fwd_simulation = self.pipeline_components['fwd_simulation']
        fwd_inversion = self.pipeline_components['fwd_inversion']
        source_coords_mm = self.pipeline_components['source_coords_mm']
        scale_factor = self.pipeline_components['scale_factor']
        forward_mismatch = self.pipeline_components['forward_mismatch']
        ground_truth_conductivities = self.pipeline_components['ground_truth_conductivities']

        # Extract electrode coordinates for depth calculation
        electrode_coords_mm = np.array([
            ch['loc'][:3] * 1000 for ch in info['chs']
            if ch['kind'] == 2  # EEG channels
        ])

        # Initialize simulator
        simulator = DipoleSimulator(fwd_simulation, info, src, verbose=False)

        # Load ROI information
        import source_localization
        package_dir = Path(source_localization.__file__).parent

        roi_labels_file = package_dir / config['inputs']['brain_labels']
        roi_names_file = package_dir / config['inputs'].get('roi_mapping', 'data/atlas/roi_mapping.json')

        roi_centroids, roi_names_map = load_atlas_roi_centroids(str(roi_labels_file), str(roi_names_file))

        # Apply scale factor to ROI centroids
        if scale_factor != 1.0:
            roi_centroids = {k: v * scale_factor for k, v in roi_centroids.items()}

        # Get ROI-source mapping (use pipeline's roi_assignments for consistency)
        if 'roi_assignments' in src[0]:
            roi_mapping = np.array(src[0]['roi_assignments']).astype(int)
        else:
            # Fall back to KD-tree lookup
            mapping_radius = 2.0 * scale_factor
            roi_mapping = load_roi_source_mapping(
                source_coords_mm, str(roi_labels_file),
                radius_mm=mapping_radius, scale_factor=scale_factor
            )

        # Determine test positions based on mode
        roi_placement_meta = None
        if test_mode == 'uniform_grid':
            test_positions, test_roi_labels = generate_uniform_test_grid(
                str(roi_labels_file),
                spacing_mm=grid_spacing_mm,
                margin_mm=grid_margin_mm,
                verbose=False  # Suppress verbose in combined mode
            )
            if scale_factor != 1.0:
                test_positions = test_positions * scale_factor
            n_test_positions = len(test_positions)
            test_points = [
                (test_positions[i], int(test_roi_labels[i]), f"Grid_{i:04d}")
                for i in range(n_test_positions)
            ]
        else:
            # ROI-based mode -- same placement contract as run(); see there and
            # load_atlas_roi_test_points() for why 'centroid' is not the default.
            roi_placement = val_config.get('roi_placement', 'medoid')
            roi_n_per = int(val_config.get('roi_points_per_roi', 1))

            if roi_placement == 'centroid':
                roi_ids = sorted([rid for rid in roi_centroids.keys() if rid != 0])
                if roi_indices is not None:
                    roi_ids = [roi_ids[i] for i in roi_indices if i < len(roi_ids)]
                test_points = [
                    (roi_centroids[roi_id], roi_id,
                     roi_names_map.get(roi_id, f"ROI_{roi_id}"))
                    for roi_id in roi_ids
                ]
                roi_placement_meta = {'placement': 'centroid', 'n_per_roi': 1}
            else:
                from .utils import load_atlas_roi_test_points
                pts_by_roi, _, roi_placement_meta = load_atlas_roi_test_points(
                    str(roi_labels_file), str(roi_names_file),
                    placement=roi_placement, n_per_roi=roi_n_per,
                    sample_strategy=val_config.get('roi_sample_strategy', 'random'),
                    enforce_lr_symmetry=bool(
                        val_config.get('roi_enforce_lr_symmetry', True)),
                    seed=int(val_config.get('roi_sample_seed', 20260728)),
                )
                roi_ids = sorted([rid for rid in pts_by_roi.keys() if rid != 0])
                if roi_indices is not None:
                    roi_ids = [roi_ids[i] for i in roi_indices if i < len(roi_ids)]
                test_points = []
                for roi_id in roi_ids:
                    pts = np.atleast_2d(pts_by_roi[roi_id]) * scale_factor
                    base = roi_names_map.get(roi_id, f"ROI_{roi_id}")
                    for k, p in enumerate(pts):
                        test_points.append(
                            (p, roi_id, base if len(pts) == 1 else f"{base}#{k+1}"))
            n_test_positions = len(test_points)

        n_rois = len(set(p[1] for p in test_points if p[1] > 0))

        # Initialize results
        all_results = {
            'config_name': self.config_name,
            'pipeline_name': config['pipeline']['name'],
            'bem_type': config['pipeline']['bem_type'],
            'source_type': config['pipeline']['source_type'],
            'inverse_method': config['inverse']['method'],
            'n_sources': len(source_coords_mm),
            'n_rois': n_rois,
            'n_test_positions': len(test_points),
            'test_mode': test_mode,
            'grid_spacing_mm': grid_spacing_mm if test_mode == 'uniform_grid' else None,
            'grid_margin_mm': grid_margin_mm if test_mode == 'uniform_grid' else None,
            'scale_factor': scale_factor,
            'forward_model_mismatch': forward_mismatch,
            'ground_truth_conductivities': ground_truth_conductivities if forward_mismatch else None,
            'test_conductivities': config['bem'][config['pipeline']['bem_type']]['conductivities'] if forward_mismatch else None,
            'dipole_amplitude_nAm': dipole_config.get('amplitude_nAm', 50.0),
            'noise_mode': dipole_config.get('noise_mode', 'snr'),
            'noise_variance_uV2': dipole_config.get('noise_variance_uV2', 1.0) if dipole_config.get('noise_mode') == 'fixed_variance' else None,
            'timestamp': datetime.now().isoformat(),
            # Which reference position 'localization_errors' is measured FROM.
            # See LOCALIZATION_ERROR_DEFINITION. Recorded so every downstream
            # table can state its own definition without archaeology.
            'localization_error_definition': LOCALIZATION_ERROR_DEFINITION,
            # Where inside each ROI the test dipole was placed. Legacy
            # 'centroid' can fall outside non-convex parcels; recorded so an
            # ROI-accuracy table always states its own placement.
            'roi_placement': (roi_placement_meta
                              if test_mode != 'uniform_grid' else None),
            'snr_results': {}
        }

        # Run simulation loop (same as main run() method)
        for snr_db in snr_levels:
            snr_metrics = {
                # PRIMARY error: peak vs REQUESTED position (see
                # LOCALIZATION_ERROR_DEFINITION).
                'localization_errors': [],
                # DIAGNOSTIC error: peak vs SNAPPED source position.
                'localization_errors_snapped': [],
                # Distance requested -> snapped; explains the gap between the two.
                'snapping_errors': [],
                'roi_correct': [],
                'adjacent_roi_correct': [],
                'amplitude_ratios': [],
                'depths': [],
                'position_names': [],
                'true_roi_ids': [],
                'estimated_roi_ids': [],
                'estimated_roi_names': [],
                'true_positions_mm': [],
                'snapped_positions_mm': [],
                'estimated_positions_mm': [],
                'peak_in_background': []
            }

            for i, (test_pos, true_roi_id, pos_name) in enumerate(test_points):
                for trial in range(n_trials):
                    # Simulate dipole
                    eeg_data, sim_meta = simulator.simulate_dipole(
                        position_mm=test_pos,
                        amplitude_nAm=dipole_config.get('amplitude_nAm', 50.0),
                        duration_s=dipole_config.get('duration_s', 1.0),
                        sfreq=dipole_config.get('sfreq', 500.0),
                        snr_db=snr_db,
                        noise_seed=trial * 1000 + i,
                        noise_mode=dipole_config.get('noise_mode', 'snr'),
                        noise_variance_uV2=dipole_config.get('noise_variance_uV2', 1.0)
                    )

                    raw = simulator.create_mne_raw(eeg_data, sfreq=dipole_config.get('sfreq', 500.0))
                    stc = self._apply_inverse(raw, fwd_inversion, info, config['inverse'])

                    # Find peak
                    source_activations = np.abs(stc.data).mean(axis=1)
                    sorted_indices = np.argsort(source_activations)[::-1]
                    absolute_peak_idx = sorted_indices[0]

                    if absolute_peak_idx >= len(source_coords_mm):
                        raise IndexError(f"Peak index {absolute_peak_idx} out of bounds")

                    absolute_peak_position = source_coords_mm[absolute_peak_idx]

                    # Valid ROI peak
                    valid_roi_peak_idx = sorted_indices[0]
                    for idx in sorted_indices:
                        if roi_mapping[idx] != 0:
                            valid_roi_peak_idx = idx
                            break

                    # Compute metrics -- both error conventions, see
                    # LOCALIZATION_ERROR_DEFINITION in this module.
                    requested_position_mm = np.array(sim_meta['requested_position_mm'])
                    snapped_position_mm = np.array(sim_meta['snapped_source_position_mm'])
                    loc_error = compute_localization_error(
                        requested_position_mm, absolute_peak_position) / scale_factor
                    loc_error_snapped = compute_localization_error(
                        snapped_position_mm, absolute_peak_position) / scale_factor
                    snapping_error = compute_localization_error(
                        requested_position_mm, snapped_position_mm) / scale_factor

                    estimated_roi = roi_mapping[valid_roi_peak_idx]
                    estimated_roi_name = roi_names_map.get(estimated_roi, f"ROI_{estimated_roi}")
                    roi_correct = compute_roi_classification_accuracy(true_roi_id, estimated_roi)
                    peak_in_background = (roi_mapping[absolute_peak_idx] == 0)
                    depth = self._compute_source_depth(requested_position_mm, electrode_coords_mm)

                    # Store
                    snr_metrics['localization_errors'].append(loc_error)
                    snr_metrics['localization_errors_snapped'].append(loc_error_snapped)
                    snr_metrics['snapping_errors'].append(snapping_error)
                    snr_metrics['roi_correct'].append(roi_correct)
                    snr_metrics['depths'].append(depth)
                    snr_metrics['position_names'].append(pos_name)
                    snr_metrics['true_roi_ids'].append(true_roi_id)
                    snr_metrics['estimated_roi_ids'].append(int(estimated_roi))
                    snr_metrics['estimated_roi_names'].append(estimated_roi_name)
                    snr_metrics['true_positions_mm'].append(requested_position_mm.tolist())
                    snr_metrics['snapped_positions_mm'].append(snapped_position_mm.tolist())
                    snr_metrics['estimated_positions_mm'].append(
                        np.asarray(absolute_peak_position).tolist()
                    )
                    snr_metrics['peak_in_background'].append(peak_in_background)

            # Compute summaries (same logic as main run method)
            per_position_metrics = {}
            for i, pos_name in enumerate(snr_metrics['position_names']):
                if pos_name not in per_position_metrics:
                    per_position_metrics[pos_name] = {
                        'true_roi_id': snr_metrics['true_roi_ids'][i],
                        'localization_errors': [],
                        'roi_correct': [],
                        'estimated_rois': [],
                        'depth_mm': snr_metrics['depths'][i]
                    }
                per_position_metrics[pos_name]['localization_errors'].append(snr_metrics['localization_errors'][i])
                per_position_metrics[pos_name]['roi_correct'].append(snr_metrics['roi_correct'][i])
                per_position_metrics[pos_name]['estimated_rois'].append(snr_metrics['estimated_roi_names'][i])

            per_position_summary = {}
            for pos_name, metrics in per_position_metrics.items():
                est_counts = Counter(metrics['estimated_rois'])
                most_common_est = est_counts.most_common(1)[0] if est_counts else (None, 0)
                per_position_summary[pos_name] = {
                    'true_roi_id': metrics['true_roi_id'],
                    'n_trials': len(metrics['localization_errors']),
                    'localization_error_mm': float(np.mean(metrics['localization_errors'])),
                    'source_depth_mm': float(metrics['depth_mm']),
                    'roi_correct': bool(np.all(metrics['roi_correct'])),
                    'roi_accuracy': float(np.mean(metrics['roi_correct'])),
                    'estimated_as': most_common_est[0],
                    'estimated_as_count': most_common_est[1]
                }

            # Depth-stratified
            depth_bins = {
                '1-2mm': (1, 2), '2-3mm': (2, 3), '3-4mm': (3, 4),
                '4-5mm': (4, 5), '5+mm': (5, float('inf'))
            }
            depth_stratified = {}
            for bin_name, (low, high) in depth_bins.items():
                bin_loc_errors = []
                bin_roi_correct = []
                for i, depth in enumerate(snr_metrics['depths']):
                    if low <= depth < high:
                        bin_loc_errors.append(snr_metrics['localization_errors'][i])
                        bin_roi_correct.append(snr_metrics['roi_correct'][i])
                if bin_loc_errors:
                    depth_stratified[bin_name] = {
                        'n_trials': len(bin_loc_errors),
                        'localization_error_mean': float(np.mean(bin_loc_errors)),
                        'localization_error_std': float(np.std(bin_loc_errors)),
                        'roi_accuracy': float(np.mean(bin_roi_correct))
                    }
                else:
                    depth_stratified[bin_name] = None

            # Depth bias
            from .metrics import compute_depth_bias
            depths_array = np.array(snr_metrics['depths'])
            errors_array = np.array(snr_metrics['localization_errors'])
            depth_bias = compute_depth_bias(errors_array, depths_array)

            peak_in_bg_count = sum(snr_metrics['peak_in_background'])
            peak_in_bg_rate = peak_in_bg_count / len(snr_metrics['peak_in_background']) if snr_metrics['peak_in_background'] else 0.0

            snr_summary = {
                'snr_db': snr_db,
                'n_simulations': len(snr_metrics['localization_errors']),
                'n_test_positions': len(per_position_summary),
                'localization_error': {
                    'mean': float(np.mean(snr_metrics['localization_errors'])),
                    'std': float(np.std(snr_metrics['localization_errors'])),
                    'median': float(np.median(snr_metrics['localization_errors'])),
                    'min': float(np.min(snr_metrics['localization_errors'])),
                    'max': float(np.max(snr_metrics['localization_errors']))
                },
                'roi_accuracy': {
                    'exact': float(np.mean(snr_metrics['roi_correct'])),
                    'n_correct': int(np.sum(snr_metrics['roi_correct'])),
                    'n_total': len(snr_metrics['roi_correct'])
                },
                'peak_in_background': {
                    'count': peak_in_bg_count,
                    'rate': float(peak_in_bg_rate)
                },
                'depth_bias': {
                    'correlation': float(depth_bias['correlation']) if not np.isnan(depth_bias['correlation']) else None,
                    'slope_mm_per_mm': float(depth_bias['slope']) if not np.isnan(depth_bias['slope']) else None,
                    'superficial_error_mm': float(depth_bias['superficial_error_mm']) if not np.isnan(depth_bias['superficial_error_mm']) else None,
                    'deep_error_mm': float(depth_bias['deep_error_mm']) if not np.isnan(depth_bias['deep_error_mm']) else None,
                },
                'depth_stratified': depth_stratified,
                'per_position': per_position_summary,
                'raw_data': snr_metrics
            }

            all_results['snr_results'][snr_db] = snr_summary

        return all_results

    def save_results(self, results: Dict[str, Any]) -> Path:
        """
        Save validation results to metrics.json, figures, and HTML report.

        Parameters
        ----------
        results : dict
            Validation results from run()

        Returns
        -------
        Path
            Path to saved metrics.json
        """
        results_path = self.output_dir / 'metrics.json'

        def json_serializer(obj):
            """Convert numpy types to JSON-serializable Python types."""
            if isinstance(obj, np.floating):
                return float(obj)
            elif isinstance(obj, np.integer):
                return int(obj)
            elif isinstance(obj, np.bool_):
                return bool(obj)
            elif isinstance(obj, np.ndarray):
                return obj.tolist()
            raise TypeError(f"Object of type {type(obj)} is not JSON serializable")

        with open(results_path, 'w') as f:
            json.dump(results, f, indent=2, default=json_serializer)

        if self.verbose:
            print(f"\n  Results saved to: {results_path}")

        # Generate validation figures
        figures_dir = self.output_dir / 'figures'
        self._generate_validation_figures(results, figures_dir)

        # Generate markdown report
        report_path = self.output_dir / 'report.md'
        self._generate_report(results, report_path)

        # Generate HTML report
        html_path = self.output_dir / 'validation_report.html'
        self._generate_html_report(results, figures_dir, html_path)

        # Generate standardized summary file
        summary_path = self.output_dir / 'validation_summary.txt'
        self._generate_summary_file(results, summary_path)

        return results_path

    def _generate_summary_file(self, results: Dict, summary_path: Path):
        """
        Generate a standardized human-readable validation summary.

        This summary provides key metrics at a glance for comparing configurations.
        ROI accuracy is always reported but is only meaningful for ROI-based configs.
        """
        lines = []
        lines.append("=" * 70)
        lines.append(f"VALIDATION SUMMARY: {results.get('config_name', 'Unknown')}")
        lines.append("=" * 70)
        lines.append("")

        # Configuration details
        lines.append("CONFIGURATION:")
        lines.append(f"  Source Space:    {results.get('source_type', 'unknown')} ({results.get('n_sources', 0)} sources)")
        lines.append(f"  BEM Model:       {results.get('bem_type', 'unknown')}")
        lines.append(f"  Inverse Method:  {results.get('inverse_method', 'unknown')}")
        lines.append(f"  N ROIs:          {results.get('n_rois', 0)}")
        lines.append(f"  Test Mode:       {results.get('test_mode', 'unknown')}")
        lines.append(f"  Scale Factor:    {results.get('scale_factor', 1.0)}x")
        if results.get('forward_model_mismatch'):
            lines.append(f"  Forward Mismatch: YES (conductivity mismatch test)")
        lines.append("")

        # Process each SNR level
        for snr_db, snr_data in results.get('snr_results', {}).items():
            lines.append("-" * 70)
            lines.append(f"RESULTS AT SNR = {snr_db} dB")
            lines.append("-" * 70)
            lines.append("")

            # Simulation counts
            n_sims = snr_data.get('n_simulations', {})
            if isinstance(n_sims, dict):
                total = n_sims.get('total', 0)
                lines.append(f"  Total Simulations: {total}")
            lines.append("")

            # PRIMARY METRICS
            lines.append("PRIMARY METRICS:")

            # Localization Error
            loc_err = snr_data.get('localization_error', {})
            if loc_err:
                mean = loc_err.get('mean', 0)
                std = loc_err.get('std', 0)
                median = loc_err.get('median', 0)
                min_err = loc_err.get('min', 0)
                max_err = loc_err.get('max', 0)
                lines.append(f"  Localization Error:  {mean:.2f} ± {std:.2f} mm")
                lines.append(f"                       Median: {median:.2f} mm, Range: [{min_err:.2f}, {max_err:.2f}] mm")
                lines.append(f"                       (Source: {snr_data.get('localization_error_source', 'N/A')})")

            # ROI Accuracy
            roi_acc = snr_data.get('roi_accuracy', {})
            if roi_acc:
                exact = roi_acc.get('exact', 0) * 100
                n_correct = roi_acc.get('n_correct', 0)
                n_total = roi_acc.get('n_total', 0)
                lines.append(f"  ROI Accuracy:        {exact:.1f}% ({n_correct}/{n_total} correct)")
                lines.append(f"                       (Source: {snr_data.get('roi_accuracy_source', 'N/A')})")

                # Add note about ROI accuracy interpretation
                source_type = results.get('source_type', '')
                if 'roi' not in source_type.lower():
                    lines.append(f"                       NOTE: ROI accuracy less meaningful for {source_type} sources")
            lines.append("")

            # Peak in Background
            peak_bg = snr_data.get('peak_in_background', {})
            if peak_bg:
                lines.append("PEAK DETECTION:")
                for source, data in peak_bg.items():
                    if isinstance(data, dict):
                        rate = data.get('rate', 0) * 100
                        count = data.get('count', 0)
                        total = data.get('total', 0)
                        lines.append(f"  Peak in background ({source}): {rate:.1f}% ({count}/{total})")
                lines.append("")

            # DEPTH-STRATIFIED METRICS
            lines.append("DEPTH-STRATIFIED LOCALIZATION ERROR (by electrode distance):")
            depth_loc = snr_data.get('depth_stratified_localization_error', {})
            if not depth_loc:
                depth_loc = snr_data.get('depth_stratified', {})

            if depth_loc:
                lines.append(f"  {'Depth':<10} {'Mean ± Std (mm)':<20} {'N trials':<10}")
                lines.append(f"  {'-'*10} {'-'*20} {'-'*10}")
                for bin_name in ['1-2mm', '2-3mm', '3-4mm', '4-5mm', '5+mm']:
                    bin_data = depth_loc.get(bin_name)
                    if bin_data:
                        mean = bin_data.get('localization_error_mean', 0)
                        std = bin_data.get('localization_error_std', 0)
                        n = bin_data.get('n_trials', 0)
                        lines.append(f"  {bin_name:<10} {mean:>6.2f} ± {std:<6.2f}       {n:<10}")
                    else:
                        lines.append(f"  {bin_name:<10} {'N/A':<20} {'N/A':<10}")
            lines.append("")

            # Depth-stratified ROI accuracy
            depth_roi = snr_data.get('depth_stratified_roi_accuracy', {})
            if depth_roi:
                lines.append("DEPTH-STRATIFIED ROI ACCURACY:")
                lines.append(f"  {'Depth':<10} {'Accuracy':<15} {'N trials':<10}")
                lines.append(f"  {'-'*10} {'-'*15} {'-'*10}")
                for bin_name in ['1-2mm', '2-3mm', '3-4mm', '4-5mm', '5+mm']:
                    bin_data = depth_roi.get(bin_name)
                    if bin_data:
                        acc = bin_data.get('roi_accuracy', 0) * 100
                        n = bin_data.get('n_trials', 0)
                        lines.append(f"  {bin_name:<10} {acc:>6.1f}%          {n:<10}")
                    else:
                        lines.append(f"  {bin_name:<10} {'N/A':<15} {'N/A':<10}")
                lines.append("")

            # DEPTH BIAS ANALYSIS
            depth_bias = snr_data.get('depth_bias', {})
            if depth_bias:
                lines.append("DEPTH BIAS ANALYSIS:")

                loc_bias = depth_bias.get('localization_error', {})
                if loc_bias:
                    corr = loc_bias.get('correlation', 0)
                    slope = loc_bias.get('slope_mm_per_mm', 0)
                    sup_err = loc_bias.get('superficial_error_mm', 0)
                    deep_err = loc_bias.get('deep_error_mm', 0)
                    lines.append(f"  Localization Error vs Depth:")
                    lines.append(f"    Correlation:     r = {corr:.3f}")
                    lines.append(f"    Slope:           {slope:.3f} mm error per mm depth")
                    lines.append(f"    Superficial:     {sup_err:.2f} mm error")
                    lines.append(f"    Deep:            {deep_err:.2f} mm error")

                roi_bias = depth_bias.get('roi_accuracy', {})
                if roi_bias:
                    corr = roi_bias.get('correlation', 0)
                    lines.append(f"  ROI Accuracy vs Depth:")
                    lines.append(f"    Correlation:     r = {corr:.3f}")
                lines.append("")

        # Footer
        lines.append("=" * 70)
        lines.append(f"Generated: {results.get('timestamp', 'N/A')}")
        lines.append("")
        lines.append("INTERPRETATION NOTES:")
        lines.append("- Localization error: Lower is better. <2mm is excellent for mouse brain.")
        lines.append("- ROI accuracy: Higher is better. Only meaningful for ROI-based source spaces.")
        lines.append("- Depth bias: Positive correlation means deeper sources have larger errors.")
        lines.append("- Peak in background: Indicates when peak activity falls outside brain ROIs.")
        lines.append("=" * 70)

        # Write to file
        with open(summary_path, 'w') as f:
            f.write('\n'.join(lines))

        if self.verbose:
            print(f"  Summary saved to: {summary_path}")

    def _generate_validation_figures(self, results: Dict, figures_dir: Path):
        """Generate validation-specific figures including localization error map."""
        import matplotlib.pyplot as plt

        figures_dir.mkdir(exist_ok=True)

        # Generate localization error map from validation data
        self._generate_error_map(results, figures_dir)

        for snr_db, snr_data in results.get('snr_results', {}).items():
            # Figure 1: Depth-stratified error bar chart
            depth_data = snr_data.get('depth_stratified', {})
            if depth_data:
                fig, ax = plt.subplots(figsize=(10, 6))

                bins = ['1-2mm', '2-3mm', '3-4mm', '4-5mm', '5+mm']
                means = []
                stds = []
                n_trials = []
                for bin_name in bins:
                    bin_data = depth_data.get(bin_name)
                    if bin_data is not None:
                        means.append(bin_data['localization_error_mean'])
                        stds.append(bin_data['localization_error_std'])
                        n_trials.append(bin_data['n_trials'])
                    else:
                        means.append(0)
                        stds.append(0)
                        n_trials.append(0)

                x = range(len(bins))
                bars = ax.bar(x, means, yerr=stds, capsize=5, color='steelblue', alpha=0.8)

                # Add N labels on bars
                for i, (bar, n) in enumerate(zip(bars, n_trials)):
                    if n > 0:
                        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + stds[i] + 0.1,
                               f'n={n}', ha='center', va='bottom', fontsize=9)

                ax.set_xlabel('Source Depth', fontsize=12)
                ax.set_ylabel('Localization Error (mm)', fontsize=12)
                ax.set_title(f"{results['config_name']}\nLocalization Error by Depth (SNR={snr_db}dB)", fontsize=14)
                ax.set_xticks(x)
                ax.set_xticklabels(bins)
                ax.grid(axis='y', alpha=0.3)

                plt.tight_layout()
                plt.savefig(figures_dir / f'depth_error_snr{snr_db}.png', dpi=150)
                plt.close()

            # Figure 2: ROI accuracy bar (single value)
            roi_acc = snr_data.get('roi_accuracy', {})
            if roi_acc:
                fig, ax = plt.subplots(figsize=(6, 4))

                acc = roi_acc.get('exact', 0) * 100
                ax.barh(['ROI Accuracy'], [acc], color='forestgreen' if acc > 50 else 'coral')
                ax.set_xlim(0, 100)
                ax.set_xlabel('Accuracy (%)')
                ax.set_title(f"{results['config_name']}\nROI Classification (SNR={snr_db}dB)")
                ax.axvline(50, color='gray', linestyle='--', alpha=0.5)

                # Add text label
                ax.text(acc + 2, 0, f'{acc:.1f}%', va='center', fontsize=12, fontweight='bold')

                plt.tight_layout()
                plt.savefig(figures_dir / f'roi_accuracy_snr{snr_db}.png', dpi=150)
                plt.close()

            # Figure 3: Error distribution histogram
            loc_error = snr_data.get('localization_error', {})
            if loc_error and 'mean' in loc_error:
                fig, ax = plt.subplots(figsize=(8, 5))

                # Create synthetic histogram data from summary stats
                mean_err = loc_error['mean']
                std_err = loc_error['std']
                median_err = loc_error['median']

                ax.axvline(mean_err, color='red', linestyle='-', linewidth=2, label=f'Mean: {mean_err:.2f} mm')
                ax.axvline(median_err, color='orange', linestyle='--', linewidth=2, label=f'Median: {median_err:.2f} mm')

                # Add text box with stats
                stats_text = f"Mean: {mean_err:.2f} mm\nMedian: {median_err:.2f} mm\nStd: {std_err:.2f} mm"
                ax.text(0.95, 0.95, stats_text, transform=ax.transAxes, fontsize=11,
                       verticalalignment='top', horizontalalignment='right',
                       bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

                ax.set_xlabel('Localization Error (mm)', fontsize=12)
                ax.set_ylabel('Density', fontsize=12)
                ax.set_title(f"{results['config_name']}\nLocalization Error Summary (SNR={snr_db}dB)", fontsize=14)
                ax.legend(loc='upper left')
                ax.set_xlim(0, max(mean_err + 3*std_err, 10))

                plt.tight_layout()
                plt.savefig(figures_dir / f'error_summary_snr{snr_db}.png', dpi=150)
                plt.close()

    def _generate_error_map(self, results: Dict, figures_dir: Path):
        """Generate spatial localization error map from validation results."""
        try:
            from source_localization.visualization.localization_error_map import (
                create_localization_error_figure
            )
        except ImportError:
            if self.verbose:
                print("  Warning: Could not import localization_error_map visualization")
            return

        # Get pipeline components (source coords, electrode coords)
        if not hasattr(self, 'pipeline_components') or not self.pipeline_components:
            if self.verbose:
                print("  Warning: Pipeline components not available for error map")
            return

        source_coords_mm = self.pipeline_components.get('source_coords_mm')
        info = self.pipeline_components.get('info')

        if source_coords_mm is None or info is None:
            return

        # Get electrode coordinates from info
        electrode_coords_mm = np.array([
            ch['loc'][:3] * 1000 for ch in info['chs']
            if ch['kind'] == 2  # EEG channels
        ])

        # For each SNR level, create error map using per-position results
        for snr_db, snr_data in results.get('snr_results', {}).items():
            per_position = snr_data.get('per_position_summary', [])

            if not per_position:
                # Use estimated error map based on depth
                if self.verbose:
                    print(f"  Generating estimated error map for SNR={snr_db}dB...")

                from source_localization.visualization.localization_error_map import (
                    estimate_localization_error
                )

                method = results.get('inverse_method', 'sLORETA')
                errors_mm, _ = estimate_localization_error(
                    source_coords_mm, electrode_coords_mm, method=method
                )

                output_path = figures_dir / f'localization_error_map_snr{snr_db}.png'
                create_localization_error_figure(
                    source_coords_mm, errors_mm, electrode_coords_mm,
                    config_name=results['config_name'],
                    method=method,
                    output_path=output_path,
                    show_roi_outlines=False,
                )
            else:
                # Use actual measured errors from validation
                if self.verbose:
                    print(f"  Generating validated error map for SNR={snr_db}dB...")

                # Extract test positions and errors
                test_positions = []
                test_errors = []

                for pos_data in per_position:
                    if 'true_position_mm' in pos_data and 'localization_error_mean' in pos_data:
                        test_positions.append(pos_data['true_position_mm'])
                        test_errors.append(pos_data['localization_error_mean'])

                if len(test_positions) < 10:
                    if self.verbose:
                        print(f"    Not enough test positions ({len(test_positions)}) for error map")
                    continue

                test_positions = np.array(test_positions)
                test_errors = np.array(test_errors)

                output_path = figures_dir / f'localization_error_map_snr{snr_db}.png'
                create_localization_error_figure(
                    test_positions, test_errors, electrode_coords_mm,
                    config_name=f"{results['config_name']} (Validated)",
                    method=results.get('inverse_method', 'sLORETA'),
                    output_path=output_path,
                    show_roi_outlines=False,
                )

    def _generate_html_report(self, results: Dict, figures_dir: Path, output_path: Path):
        """Generate HTML validation report with embedded figures."""
        import base64

        def img_to_base64(img_path):
            if img_path.exists():
                with open(img_path, 'rb') as f:
                    return base64.b64encode(f.read()).decode('utf-8')
            return None

        html = []
        html.append("""<!DOCTYPE html>
<html>
<head>
    <meta charset="UTF-8">
    <title>Validation Report</title>
    <style>
        body { font-family: Arial, sans-serif; margin: 40px; background: #f5f5f5; }
        .container { max-width: 1200px; margin: 0 auto; background: white; padding: 30px; box-shadow: 0 0 10px rgba(0,0,0,0.1); }
        h1 { color: #2c3e50; border-bottom: 3px solid #3498db; padding-bottom: 10px; }
        h2 { color: #34495e; margin-top: 30px; border-left: 4px solid #3498db; padding-left: 10px; }
        .metric { display: inline-block; margin: 10px 20px; padding: 15px; background: #ecf0f1; border-radius: 8px; text-align: center; min-width: 150px; }
        .metric-value { font-size: 28px; font-weight: bold; color: #2980b9; }
        .metric-label { font-size: 12px; color: #7f8c8d; }
        table { border-collapse: collapse; width: 100%; margin: 15px 0; }
        th, td { border: 1px solid #ddd; padding: 10px; text-align: left; }
        th { background: #3498db; color: white; }
        tr:nth-child(even) { background: #f9f9f9; }
        .figure { margin: 20px 0; text-align: center; }
        .figure img { max-width: 100%; border: 1px solid #ddd; border-radius: 4px; }
        .good { color: #27ae60; }
        .warning { color: #f39c12; }
        .bad { color: #e74c3c; }
    </style>
</head>
<body>
<div class="container">
""")

        # Header
        html.append(f"<h1>Validation Report: {results['config_name']}</h1>")
        html.append(f"<p><strong>Pipeline:</strong> {results['pipeline_name']}</p>")
        html.append(f"<p><strong>Generated:</strong> {results['timestamp']}</p>")

        # Configuration summary
        html.append("<h2>Configuration</h2>")
        html.append("<table>")
        html.append(f"<tr><th>Parameter</th><th>Value</th></tr>")
        html.append(f"<tr><td>BEM Type</td><td>{results['bem_type']}</td></tr>")
        html.append(f"<tr><td>Source Type</td><td>{results['source_type']}</td></tr>")
        html.append(f"<tr><td>Inverse Method</td><td>{results['inverse_method']}</td></tr>")
        html.append(f"<tr><td>Number of Sources</td><td>{results['n_sources']}</td></tr>")
        html.append(f"<tr><td>Test Mode</td><td>{results.get('test_mode', 'N/A')}</td></tr>")
        html.append("</table>")

        # Results by SNR
        for snr_db, snr_data in results.get('snr_results', {}).items():
            html.append(f"<h2>Results (SNR = {snr_db} dB)</h2>")

            # Key metrics
            loc_error = snr_data.get('localization_error', {})
            roi_acc = snr_data.get('roi_accuracy', {})
            mean_err = loc_error.get('mean', 0)
            acc = roi_acc.get('exact', 0) * 100

            err_class = 'good' if mean_err < 2 else ('warning' if mean_err < 3 else 'bad')
            acc_class = 'good' if acc > 60 else ('warning' if acc > 40 else 'bad')

            html.append("<div>")
            html.append(f'<div class="metric"><div class="metric-value {err_class}">{mean_err:.2f} mm</div><div class="metric-label">Mean Error</div></div>')
            html.append(f'<div class="metric"><div class="metric-value">{loc_error.get("median", 0):.2f} mm</div><div class="metric-label">Median Error</div></div>')
            html.append(f'<div class="metric"><div class="metric-value {acc_class}">{acc:.1f}%</div><div class="metric-label">ROI Accuracy</div></div>')
            html.append(f'<div class="metric"><div class="metric-value">{snr_data.get("n_simulations", 0)}</div><div class="metric-label">Simulations</div></div>')
            html.append("</div>")

            # Depth-stratified table
            depth_data = snr_data.get('depth_stratified', {})
            if depth_data:
                html.append("<h3>Depth-Stratified Results</h3>")
                html.append("<table>")
                html.append("<tr><th>Depth</th><th>N</th><th>Error (mm)</th><th>Std (mm)</th><th>ROI Acc</th></tr>")
                for bin_name in ['1-2mm', '2-3mm', '3-4mm', '4-5mm', '5+mm']:
                    d = depth_data.get(bin_name)
                    if d is not None:
                        html.append(f"<tr><td>{bin_name}</td><td>{d['n_trials']}</td><td>{d['localization_error_mean']:.2f}</td><td>{d['localization_error_std']:.2f}</td><td>{d.get('roi_accuracy', 0)*100:.1f}%</td></tr>")
                html.append("</table>")

            # Figures
            html.append("<h3>Figures</h3>")

            # Embed localization error map first (most important)
            error_map_path = figures_dir / f'localization_error_map_snr{snr_db}.png'
            b64 = img_to_base64(error_map_path)
            if b64:
                html.append('<h4>Localization Error Map</h4>')
                html.append(f'<div class="figure"><img src="data:image/png;base64,{b64}" alt="Localization Error Map"></div>')

            # Embed other validation figures
            for fig_name in ['depth_error', 'roi_accuracy', 'error_summary']:
                fig_path = figures_dir / f'{fig_name}_snr{snr_db}.png'
                b64 = img_to_base64(fig_path)
                if b64:
                    html.append(f'<div class="figure"><img src="data:image/png;base64,{b64}" alt="{fig_name}"></div>')

            # Also embed pipeline step figures
            for step_fig in ['step1_electrodes.png', 'step3_source_space.png', 'step4_forward.png']:
                fig_path = figures_dir / step_fig
                b64 = img_to_base64(fig_path)
                if b64:
                    html.append(f'<div class="figure"><img src="data:image/png;base64,{b64}" alt="{step_fig}"></div>')

        html.append("</div></body></html>")

        with open(output_path, 'w') as f:
            f.write('\n'.join(html))

        if self.verbose:
            print(f"  HTML report saved to: {output_path}")

    def _apply_inverse(self, raw, fwd, info, inverse_config):
        """Apply inverse method to simulated data."""
        import mne
        from source_localization.steps.inverse_solution import (
            apply_inverse_custom_dSPM,
            apply_inverse_custom_MNE,
            apply_inverse_custom_sLORETA
        )

        method = inverse_config['method'].upper()
        snr = inverse_config.get('snr', 3.0)
        lambda2 = inverse_config.get('lambda2', 1.0 / snr**2)

        # Create evoked from raw
        events = np.array([[0, 0, 1]])
        epochs = mne.Epochs(raw, events, tmin=0, tmax=raw.times[-1],
                           baseline=None, preload=True, verbose=False)
        evoked = epochs.average()

        # Apply inverse - extract data array from evoked object
        eeg_data = evoked.data
        if method == 'DSPM':
            magnitude, _ = apply_inverse_custom_dSPM(fwd, eeg_data, snr, lambda2, verbose=False)
            source_power = magnitude
        elif method == 'MNE':
            magnitude, _ = apply_inverse_custom_MNE(fwd, eeg_data, snr, lambda2, verbose=False)
            source_power = magnitude
        elif method == 'SLORETA':
            magnitude, _ = apply_inverse_custom_sLORETA(fwd, eeg_data, snr, lambda2, verbose=False)
            source_power = magnitude
        elif method == 'LCMV':
            source_power = _apply_lcmv_beamformer(epochs, fwd, inverse_config)
        elif method == 'DICS':
            source_power = _apply_dics_beamformer(epochs, fwd, inverse_config)
        else:
            raise ValueError(f"Unknown inverse method: {method}")

        # Create simple SourceEstimate-like object
        class SimpleStc:
            def __init__(self, data):
                self.data = data

        return SimpleStc(source_power)

    def _compute_source_depth(self, position_mm, electrode_coords_mm: np.ndarray):
        """Compute source depth as distance to nearest electrode.

        This metric represents how far a source is from the electrode array,
        which directly affects signal strength and localization accuracy.
        Sources closer to electrodes have stronger signals and better localization.

        Parameters
        ----------
        position_mm : array-like
            Position [x, y, z] in mm
        electrode_coords_mm : ndarray, shape (n_electrodes, 3)
            Electrode positions in mm

        Returns
        -------
        depth : float
            Distance to nearest electrode in mm
        """
        position_mm = np.asarray(position_mm)
        distances = np.linalg.norm(electrode_coords_mm - position_mm, axis=1)
        return float(np.min(distances))

    def _generate_report(self, results: Dict, output_path: Path):
        """Generate markdown validation report."""
        report = []
        report.append(f"# Validation Report: {results['config_name']}")
        report.append(f"\n**Pipeline:** {results['pipeline_name']}")
        report.append(f"**Generated:** {results['timestamp']}")
        report.append(f"\n## Configuration")
        report.append(f"- BEM Type: {results['bem_type']}")
        report.append(f"- Source Type: {results['source_type']}")
        report.append(f"- Inverse Method: {results['inverse_method']}")
        report.append(f"- Number of Sources: {results['n_sources']}")

        if results.get('forward_model_mismatch'):
            report.append(f"\n## Forward Model Mismatch")
            report.append(f"- Ground Truth: {results['ground_truth_conductivities']}")
            report.append(f"- Test: {results['test_conductivities']}")

        if results.get('noise_mode') == 'fixed_variance':
            report.append(f"\n## Fixed Noise Mode")
            report.append(f"- Dipole Amplitude: {results['dipole_amplitude_nAm']} nAm")
            report.append(f"- Noise Variance: {results['noise_variance_uV2']} uV^2")

        report.append(f"\n## Results by SNR")

        for snr_db, snr_data in results['snr_results'].items():
            report.append(f"\n### SNR = {snr_db} dB")
            report.append(f"- **Simulations:** {snr_data['n_simulations']}")
            report.append(f"- **Localization Error:**")
            report.append(f"  - Mean: {snr_data['localization_error']['mean']:.2f} mm")
            report.append(f"  - Std: {snr_data['localization_error']['std']:.2f} mm")
            report.append(f"  - Median: {snr_data['localization_error']['median']:.2f} mm")
            report.append(f"  - Range: [{snr_data['localization_error']['min']:.2f}, {snr_data['localization_error']['max']:.2f}] mm")
            report.append(f"- **ROI Accuracy:** {snr_data['roi_accuracy']['exact']*100:.1f}%")
            report.append(f"  - Correct: {snr_data['roi_accuracy']['n_correct']}/{snr_data['n_simulations']}")

            # Add depth bias summary
            depth_bias = snr_data.get('depth_bias', {})
            if depth_bias and depth_bias.get('correlation') is not None:
                report.append(f"\n#### Depth Bias Analysis")
                report.append(f"- **Correlation (r):** {depth_bias['correlation']:.3f}")
                report.append(f"- **Slope:** {depth_bias['slope_mm_per_mm']:.3f} mm error per mm depth")
                if depth_bias.get('superficial_error_mm') and depth_bias.get('deep_error_mm'):
                    report.append(f"- **Superficial sources:** {depth_bias['superficial_error_mm']:.2f} mm error")
                    report.append(f"- **Deep sources:** {depth_bias['deep_error_mm']:.2f} mm error")

            # Add depth-stratified results
            depth_stratified = snr_data.get('depth_stratified', {})
            if depth_stratified:
                report.append(f"\n#### Localization Error by Depth")
                report.append(f"| Depth | N | Error (mm) | Std (mm) |")
                report.append(f"|-------|---|------------|----------|")
                for bin_name in ['1-2mm', '2-3mm', '3-4mm', '4-5mm', '5+mm']:
                    bin_data = depth_stratified.get(bin_name)
                    if bin_data:
                        report.append(f"| {bin_name} | {bin_data['n_trials']} | {bin_data['localization_error_mean']:.2f} | {bin_data['localization_error_std']:.2f} |")
                    else:
                        report.append(f"| {bin_name} | 0 | - | - |")

        report.append(f"\n## Tags")
        report.append(f"#bem-{results['bem_type']} #source-{results['source_type']} #inverse-{results['inverse_method'].lower()}")

        with open(output_path, 'w') as f:
            f.write('\n'.join(report))


def _apply_lcmv_beamformer(epochs, fwd, inverse_config):
    """
    Apply LCMV beamformer to epoched data.

    LCMV (Linearly Constrained Minimum Variance) beamformer creates spatial
    filters that pass activity from a target location while minimizing
    contributions from other sources.
    """
    import mne
    from mne.beamformer import make_lcmv, apply_lcmv_epochs

    reg = inverse_config.get('beamformer_reg', inverse_config.get('reg', 0.1))

    # Set EEG average reference
    epochs.set_eeg_reference(projection=True, verbose=False)
    epochs.apply_proj(verbose=False)

    # Strategy configurations
    strategies = [
        ('shrunk', 'unit-noise-gain', 'matrix', True, 1.0),
        ('shrunk', 'unit-noise-gain', 'matrix', True, 5.0),
        ('shrunk', 'unit-noise-gain', 'matrix', True, 10.0),
        ('shrunk', 'unit-noise-gain', 'matrix', False, 1.0),
        ('shrunk', 'unit-noise-gain', 'matrix', False, 5.0),
        ('diagonal_fixed', 'unit-noise-gain', 'matrix', True, 1.0),
        ('diagonal_fixed', 'unit-noise-gain', 'matrix', True, 5.0),
        ('diagonal_fixed', 'unit-noise-gain', 'matrix', False, 5.0),
        ('shrunk', None, 'matrix', True, 1.0),
        ('diagonal_fixed', None, 'matrix', True, 5.0),
        ('diagonal_fixed', None, 'matrix', False, 10.0),
    ]

    last_error = None
    for cov_method, weight_norm, inversion, reduce_rank, reg_mult in strategies:
        try:
            reg_val = min(reg * reg_mult, 0.9)
            data_cov = mne.compute_covariance(epochs, method=cov_method, verbose=False)

            if weight_norm is None:
                noise_cov = None
            else:
                noise_cov = mne.compute_covariance(epochs, method=cov_method, verbose=False)

            filters = make_lcmv(
                epochs.info, fwd, data_cov, reg=reg_val, noise_cov=noise_cov,
                pick_ori='max-power', weight_norm=weight_norm,
                reduce_rank=reduce_rank, inversion=inversion, verbose=False
            )

            stcs = apply_lcmv_epochs(epochs, filters, verbose=False)

            if len(stcs) > 0:
                source_data = np.abs(stcs[0].data)
                for stc in stcs[1:]:
                    source_data += np.abs(stc.data)
                source_data /= len(stcs)
                return source_data
            else:
                raise ValueError("No source estimates returned from LCMV")

        except (np.linalg.LinAlgError, ValueError, RuntimeError) as e:
            last_error = e
            continue

    raise RuntimeError(f"LCMV failed with all strategies. Last error: {last_error}")


def _apply_dics_beamformer(epochs, fwd, inverse_config):
    """
    Apply DICS beamformer to epoched data.

    DICS (Dynamic Imaging of Coherent Sources) beamformer operates in the
    frequency domain using cross-spectral density matrices.
    """
    import mne
    from mne.beamformer import make_dics, apply_dics_csd
    from mne.time_frequency import csd_morlet

    reg = inverse_config.get('reg', 0.05)
    freq_min = inverse_config.get('freq_min', 8.0)
    freq_max = inverse_config.get('freq_max', 30.0)

    # Set EEG average reference
    epochs.set_eeg_reference(projection=True, verbose=False)
    epochs.apply_proj(verbose=False)

    # Define frequencies for CSD computation
    sfreq = epochs.info['sfreq']
    freqs = np.arange(freq_min, min(freq_max, sfreq/2 - 1), 2.0)

    if len(freqs) == 0:
        freqs = np.array([10.0])

    n_cycles = freqs / 2.0

    csd = csd_morlet(epochs, freqs, n_cycles=n_cycles, verbose=False)

    filters = make_dics(
        epochs.info, fwd, csd.mean(), reg=reg,
        pick_ori='max-power', reduce_rank=True, verbose=False
    )

    stc_power, freqs_out = apply_dics_csd(csd.mean(), filters, verbose=False)

    n_times = epochs.get_data().shape[-1]

    if stc_power.data.ndim == 1:
        source_power = np.tile(stc_power.data[:, np.newaxis], (1, n_times))
    else:
        source_power = np.tile(stc_power.data.mean(axis=1, keepdims=True), (1, n_times))

    return source_power


def run_validation(
    test_dir: Union[Path, str],
    config_files: List[Path],
    snr_levels: Optional[List[float]] = None,
    n_trials: Optional[int] = None,
    roi_indices: Optional[List[int]] = None,
    quick: bool = False,
    verbose: bool = True,
    atlas: str = 'full',
    test_mode: Optional[str] = None
) -> List[Dict[str, Any]]:
    """
    Run validation on multiple configurations.

    Parameters
    ----------
    test_dir : Path or str
        Validation directory (contains config/, results/)
    config_files : list of Path
        Config file paths to run
    snr_levels : list of float, optional
        Override SNR levels
    n_trials : int, optional
        Override number of trials per location
    roi_indices : list of int, optional
        Run only for specific ROI indices
    quick : bool
        Quick test mode (5 ROIs, 1 trial, SNR=10)
    verbose : bool
        Print progress
    atlas : str
        Atlas to use: 'full' (47 ROIs) or 'coarse_22roi' (22 ROIs)
    test_mode : str, optional
        Override test mode: 'roi_centroids' (legacy) or 'uniform_grid' (depth-stratified).
        Default: from config, or 'roi_centroids' if not specified.

    Returns
    -------
    list of dict
        Validation results for each configuration
    """
    test_dir = Path(test_dir)
    results_dir = test_dir / 'results'
    results_dir.mkdir(parents=True, exist_ok=True)

    # Quick mode overrides
    if quick:
        if verbose:
            print("Running in QUICK mode (5 well-covered ROIs, 1 trial, SNR=10)")
        roi_indices = [20, 7, 16, 30, 43]
        n_trials = 1
        snr_levels = [10]

    if not config_files:
        if verbose:
            print("No config files provided.")
        return []

    if verbose:
        print(f"\nValidation directory: {test_dir}")
        print(f"Results directory: {results_dir}")
        print(f"Running {len(config_files)} validation(s)...")

    # Atlas path overrides
    atlas_overrides = None
    if atlas != 'full':
        if atlas == 'coarse_22roi':
            atlas_overrides = {
                'brain_labels': 'data/atlas/coarse_parcellation/coarse_22roi_atlas.nii',
                'roi_mapping': 'data/atlas/coarse_parcellation/coarse_22roi_mapping.json',
                '_atlas_name': 'coarse22'
            }
            if verbose:
                print(f"Using COARSE 22-ROI atlas")
        elif atlas == 'allen32':
            atlas_overrides = {
                'brain_labels': 'data/atlas/allen/allen_labels.nii.gz',
                'roi_mapping': 'data/atlas/allen/roi_mapping.json',
                '_atlas_name': 'allen32'
            }
            if verbose:
                print(f"Using ALLEN32 atlas (32 Allen CCFv3 ROIs)")

    # Run validations
    all_results = []

    for config_path in config_files:
        try:
            # Set output directory to results/{config_name}
            output_dir = results_dir / config_path.stem

            runner = ValidationRunner(
                config_path,
                output_dir=output_dir,
                verbose=verbose,
                atlas_overrides=atlas_overrides
            )
            runner.setup()
            results = runner.run(
                snr_levels=snr_levels,
                n_trials=n_trials,
                roi_indices=roi_indices,
                test_mode=test_mode
            )
            runner.save_results(results)
            all_results.append(results)
        except Exception as e:
            if verbose:
                print(f"\n  ERROR running {config_path.stem}: {e}")
                import traceback
                traceback.print_exc()

    # Summary
    if verbose:
        print(f"\n{'='*60}")
        print("VALIDATION COMPLETE")
        print(f"{'='*60}")
        print(f"Completed: {len(all_results)}/{len(config_files)} configurations")
        print(f"Results saved to: {results_dir}")

    return all_results
