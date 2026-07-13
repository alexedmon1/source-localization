"""
Batch validation runner for multi-configuration testing.

This module provides tools for running validation across multiple pipeline
configurations (presets × methods) with standardized output format.

Classes
-------
BatchValidationRunner
    Run validation tests across multiple presets and inverse methods
ValidationOutputSchema
    Standardized output format for validation results

Examples
--------
>>> from source_localization.validation import BatchValidationRunner
>>> runner = BatchValidationRunner(
...     presets=['shell_ellipsoid', 'ellipsoid_cartesian'],
...     methods=['sLORETA', 'MNE', 'eLORETA'],
...     eeg_file='data.set',
...     output_dir='./validation_results'
... )
>>> results = runner.run_all(n_trials=10, snr_db=10.0)
>>> runner.generate_comparison_report()
"""

import json
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
from datetime import datetime
from dataclasses import dataclass, field, asdict
import pickle

__all__ = [
    'BatchValidationRunner',
    'ValidationOutputSchema',
    'VALIDATION_OUTPUT_VERSION'
]

# Version for output schema (increment when format changes)
VALIDATION_OUTPUT_VERSION = "1.0.0"


@dataclass
class ValidationOutputSchema:
    """
    Standardized output format for validation results.

    This schema ensures consistent output across all validation runs,
    making it easy to compare results and aggregate across configurations.

    Attributes
    ----------
    preset : str
        Pipeline preset name (e.g., 'shell_ellipsoid')
    method : str
        Inverse method (e.g., 'sLORETA', 'MNE', 'eLORETA')
    n_sources : int
        Number of sources in the source space
    n_simulations : int
        Total number of dipole simulations run
    snr_db : float
        Signal-to-noise ratio used for simulations
    n_trials : int
        Number of trials per source
    mean_error_mm : float
        Mean localization error in mm
    median_error_mm : float
        Median localization error in mm
    std_error_mm : float
        Standard deviation of localization error
    percentile_95_mm : float
        95th percentile of localization error
    roi_accuracy : float
        Fraction of correct ROI classifications
    roi_n_valid : int
        Number of valid ROI tests (excluding Exterior)
    error_by_depth : Dict[str, Dict[str, float]]
        Localization error stratified by depth bins
    error_by_roi : Optional[Dict[str, Dict[str, float]]]
        Localization error stratified by ROI (optional)
    timestamp : str
        ISO format timestamp of validation run
    duration_seconds : float
        Total runtime in seconds
    version : str
        Output schema version
    """
    preset: str
    method: str
    n_sources: int
    n_simulations: int
    snr_db: float
    n_trials: int
    mean_error_mm: float
    median_error_mm: float
    std_error_mm: float
    percentile_95_mm: float = 0.0
    roi_accuracy: float = 0.0
    roi_n_valid: int = 0
    error_by_depth: Dict[str, Dict[str, float]] = field(default_factory=dict)
    error_by_roi: Optional[Dict[str, Dict[str, float]]] = None
    timestamp: str = ""
    duration_seconds: float = 0.0
    version: str = VALIDATION_OUTPUT_VERSION

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return asdict(self)

    def to_json(self, path: Path) -> None:
        """Save to JSON file."""
        with open(path, 'w') as f:
            json.dump(self.to_dict(), f, indent=2, default=str)

    @classmethod
    def from_json(cls, path: Path) -> 'ValidationOutputSchema':
        """Load from JSON file, handling missing fields gracefully."""
        with open(path, 'r') as f:
            data = json.load(f)

        # Provide defaults for fields that may not exist in older formats
        defaults = {
            'snr_db': 10.0,
            'n_trials': 10,
            'percentile_95_mm': 0.0,
            'roi_accuracy': data.get('roi_accuracy', 0.0),
            'roi_n_valid': data.get('roi_n_valid', 0),
            'error_by_depth': data.get('error_by_depth', {}),
            'error_by_roi': None,
            'timestamp': '',
            'duration_seconds': 0.0,
            'version': '1.0.0'
        }

        # Merge defaults with loaded data
        for key, default_value in defaults.items():
            if key not in data:
                data[key] = default_value

        return cls(**data)


class BatchValidationRunner:
    """
    Run validation tests across multiple pipeline configurations.

    This class manages batch validation testing across multiple presets
    and inverse methods, generating standardized output and comparison
    reports.

    Parameters
    ----------
    presets : list of str
        List of preset names to test
    methods : list of str
        List of inverse methods to test
    eeg_file : str or Path
        Path to EEG data file
    output_dir : str or Path
        Base output directory for results
    n_trials : int, default=25
        Number of trials per source position
    snr_db : float, default=10.0
        Signal-to-noise ratio for simulations
    test_all_sources : bool, default=True
        If True, test all sources. If False, use n_test_sources.
    n_test_sources : int, optional
        Number of sources to test (if test_all_sources=False)

    Examples
    --------
    >>> runner = BatchValidationRunner(
    ...     presets=['shell_ellipsoid', 'ellipsoid_cartesian'],
    ...     methods=['sLORETA', 'MNE'],
    ...     eeg_file='data.set',
    ...     output_dir='./validation'
    ... )
    >>> results = runner.run_all()
    >>> runner.generate_comparison_report()
    """

    # Available presets (can be extended)
    AVAILABLE_PRESETS = [
        'shell_ellipsoid',
        'shell_sphere',
        'ellipsoid_surface',
        'sphere_surface',
        'ellipsoid_cartesian',
        'sphere_cartesian',
        'roi_based_ellipsoid',
        'roi_based_sphere'
    ]

    # Available inverse methods
    AVAILABLE_METHODS = ['MNE', 'sLORETA', 'eLORETA', 'dSPM', 'LCMV']

    def __init__(
        self,
        presets: List[str],
        methods: List[str],
        eeg_file: str,
        output_dir: str,
        n_trials: int = 25,
        snr_db: float = 10.0,
        test_all_sources: bool = True,
        n_test_sources: Optional[int] = None,
        atlas: str = 'full',
        test_mode: str = 'roi_centroids'
    ):
        self.presets = presets
        self.methods = methods
        self.eeg_file = Path(eeg_file)
        self.output_dir = Path(output_dir)
        self.n_trials = n_trials
        self.snr_db = snr_db
        self.test_all_sources = test_all_sources
        self.n_test_sources = n_test_sources
        self.atlas = atlas
        self.test_mode = test_mode  # 'roi_centroids', 'uniform_grid', or 'combined'

        # Results storage
        self.results: Dict[str, ValidationOutputSchema] = {}
        self._raw_results: Dict[str, List[Dict]] = {}  # Per-trial results

        # Create output directory
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def run_all(
        self,
        parallel: bool = False,
        verbose: bool = True
    ) -> Dict[str, ValidationOutputSchema]:
        """
        Run validation for all preset × method combinations.

        Parameters
        ----------
        parallel : bool, default=False
            If True, run configurations in parallel (not yet implemented)
        verbose : bool, default=True
            Print progress information

        Returns
        -------
        results : dict
            Dictionary mapping config names to ValidationOutputSchema
        """
        from source_localization import Pipeline
        from source_localization.steps.inverse_solution import (
            apply_inverse_custom_sLORETA,
            apply_inverse_custom_MNE,
            apply_inverse_custom_dSPM,
            apply_inverse_custom_eLORETA
        )
        from .simulation import DipoleSimulator
        from .metrics import (
            compute_localization_error,
            compute_roi_classification_accuracy,
            compute_source_depths,
            compute_depth_stratified_error
        )

        # Map method names to inverse functions
        inverse_functions = {
            'sLORETA': apply_inverse_custom_sLORETA,
            'SLORETA': apply_inverse_custom_sLORETA,
            'MNE': apply_inverse_custom_MNE,
            'dSPM': apply_inverse_custom_dSPM,
            'DSPM': apply_inverse_custom_dSPM,
            'eLORETA': apply_inverse_custom_eLORETA,
            'ELORETA': apply_inverse_custom_eLORETA,
        }

        total_configs = len(self.presets) * len(self.methods)
        config_num = 0

        if verbose:
            print(f"\n{'='*60}")
            print(f"BATCH VALIDATION: {total_configs} configurations")
            print(f"{'='*60}")
            print(f"Presets: {self.presets}")
            print(f"Methods: {self.methods}")
            print(f"Trials per source: {self.n_trials}")
            print(f"SNR: {self.snr_db} dB")
            print(f"{'='*60}\n")

        for preset in self.presets:
            for method in self.methods:
                config_num += 1
                config_name = f"{preset}_{method}"
                config_dir = self.output_dir / config_name
                config_dir.mkdir(parents=True, exist_ok=True)

                if verbose:
                    print(f"\n[{config_num}/{total_configs}] {config_name}")
                    print("-" * 40)

                start_time = datetime.now()

                try:
                    # Run pipeline to build forward model
                    atlas_arg = self.atlas if self.atlas != 'full' else None
                    pipeline = Pipeline.from_preset(
                        preset,
                        atlas=atlas_arg,
                        **{'inverse.method': method}
                    )

                    if verbose:
                        print(f"  Running pipeline for {preset}...")

                    pipeline_results = pipeline.run(
                        eeg_file=str(self.eeg_file),
                        output_dir=str(config_dir)
                    )

                    # Get forward model components
                    fwd = pipeline_results['forward_solution']['fwd']
                    info = pipeline_results['electrode_registration']['info']
                    src = pipeline_results['source_space']['src']
                    source_coords_mm = pipeline_results['source_space']['source_coords_mm']

                    # Extract ROI assignments if available
                    roi_assignments = None
                    if len(src) > 0 and 'roi_assignments' in src[0]:
                        roi_assignments = np.asarray(src[0]['roi_assignments']).astype(int)
                        if verbose:
                            n_rois = len(np.unique(roi_assignments[roi_assignments > 0]))
                            print(f"  ROI assignments available: {n_rois} ROIs")

                    # Extract electrode coordinates from info
                    n_electrodes = len(info['ch_names'])
                    electrode_coords_mm = np.array([
                        info['chs'][i]['loc'][:3] for i in range(n_electrodes)
                    ]) * 1000  # Convert to mm

                    # Create simulator
                    simulator = DipoleSimulator(
                        forward_model=fwd,
                        info=info,
                        source_space=src
                    )

                    # Determine test positions based on test_mode
                    n_sources = len(source_coords_mm)

                    if self.test_mode == 'uniform_grid':
                        # Generate uniform grid of test positions independent of source grid
                        # This forces real localization (not snap-to-grid recovery)
                        import nibabel as nib
                        if self.atlas == 'allen32':
                            labels_file = str(Path(__file__).parent.parent / 'data' / 'atlas' / 'allen' / 'allen_labels.nii.gz')
                        else:
                            labels_file = str(Path(__file__).parent.parent / 'data' / 'atlas' / 'Atlas_3DRoisLeftRight.Labels.nii')
                        atlas_nii = nib.load(labels_file)
                        atlas_data = atlas_nii.get_fdata().astype(int)
                        # Get brain bounding box in mm using native affine (no 10x correction)
                        brain_voxels = np.argwhere(atlas_data > 0)
                        min_v = brain_voxels.min(axis=0)
                        max_v = brain_voxels.max(axis=0)
                        min_mm = nib.affines.apply_affine(atlas_nii.affine, min_v)
                        max_mm = nib.affines.apply_affine(atlas_nii.affine, max_v)
                        for i in range(3):
                            if min_mm[i] > max_mm[i]:
                                min_mm[i], max_mm[i] = max_mm[i], min_mm[i]
                        # Generate uniform grid at 1.0 mm spacing
                        spacing = 1.0
                        x_c = np.arange(min_mm[0], max_mm[0] + spacing, spacing)
                        y_c = np.arange(min_mm[1], max_mm[1] + spacing, spacing)
                        z_c = np.arange(min_mm[2], max_mm[2] + spacing, spacing)
                        xx, yy, zz = np.meshgrid(x_c, y_c, z_c, indexing='ij')
                        grid_candidates = np.column_stack([xx.ravel(), yy.ravel(), zz.ravel()])
                        # Filter to points inside brain
                        affine_inv = np.linalg.inv(atlas_nii.affine)
                        grid_voxels = np.round(nib.affines.apply_affine(affine_inv, grid_candidates)).astype(int)
                        valid = np.ones(len(grid_candidates), dtype=bool)
                        for dim in range(3):
                            valid &= (grid_voxels[:, dim] >= 0) & (grid_voxels[:, dim] < atlas_data.shape[dim])
                        inside = np.zeros(len(grid_candidates), dtype=bool)
                        for i in range(len(grid_candidates)):
                            if valid[i]:
                                vi, vj, vk = grid_voxels[i]
                                inside[i] = atlas_data[vi, vj, vk] > 0
                        test_positions = list(grid_candidates[inside])
                        if verbose:
                            print(f"  Using uniform 1mm grid: {len(test_positions)} test positions")
                    else:
                        # Default: test source positions directly
                        if self.test_all_sources:
                            test_indices = list(range(n_sources))
                        else:
                            n_test = self.n_test_sources or min(50, n_sources)
                            test_indices = np.random.choice(n_sources, n_test, replace=False).tolist()
                        test_positions = [source_coords_mm[i] for i in test_indices]

                    if verbose:
                        print(f"  Testing {len(test_positions)} positions × {self.n_trials} trials")

                    # Compute source depths for source positions (used for ROI assignment)
                    depths = compute_source_depths(source_coords_mm, electrode_coords_mm)

                    # Compute test position depths
                    test_depths = compute_source_depths(np.array(test_positions), electrode_coords_mm)

                    # Run validation for each test position
                    trial_results = []

                    for pos_idx, src_pos in enumerate(test_positions):
                        src_pos = np.asarray(src_pos)
                        src_depth = test_depths[pos_idx]
                        # For ROI assignment: find nearest source to this test position
                        if self.test_mode == 'uniform_grid':
                            distances = np.linalg.norm(source_coords_mm - src_pos, axis=1)
                            src_idx = int(np.argmin(distances))
                        else:
                            src_idx = test_indices[pos_idx]

                        for trial in range(self.n_trials):
                            # Simulate dipole
                            eeg_data, meta = simulator.simulate_dipole(
                                position_mm=src_pos,
                                snr_db=self.snr_db
                            )

                            # Apply inverse using the appropriate method
                            inverse_func = inverse_functions.get(method.upper(), apply_inverse_custom_sLORETA)
                            source_magnitude, _ = inverse_func(fwd, eeg_data, snr=3.0, verbose=False)

                            # Find peak (source_magnitude is n_sources x n_times)
                            if source_magnitude.ndim == 2:
                                peak_idx = np.argmax(np.abs(source_magnitude).mean(axis=1))
                            else:
                                peak_idx = np.argmax(np.abs(source_magnitude))
                            peak_pos = source_coords_mm[peak_idx]

                            # Compute metrics
                            error_mm = compute_localization_error(src_pos, peak_pos)

                            # ROI classification (if available)
                            roi_correct = None
                            true_roi = None
                            estimated_roi = None
                            if roi_assignments is not None:
                                true_roi = int(roi_assignments[src_idx])
                                estimated_roi = int(roi_assignments[peak_idx])
                                roi_correct = compute_roi_classification_accuracy(
                                    true_roi, estimated_roi
                                )

                            trial_results.append({
                                'source_idx': src_idx,
                                'trial': trial,
                                'true_position': src_pos.tolist(),
                                'estimated_position': peak_pos.tolist(),
                                'localization_error': error_mm,
                                'depth': src_depth,
                                'true_roi': true_roi,
                                'estimated_roi': estimated_roi,
                                'roi_correct': roi_correct
                            })

                    # Compute aggregate statistics
                    errors = [r['localization_error'] for r in trial_results]
                    roi_results = [r['roi_correct'] for r in trial_results if r['roi_correct'] is not None]

                    # Depth-stratified error
                    error_by_depth = compute_depth_stratified_error(trial_results)

                    duration = (datetime.now() - start_time).total_seconds()

                    # Create output schema
                    output = ValidationOutputSchema(
                        preset=preset,
                        method=method,
                        n_sources=len(test_positions),
                        n_simulations=len(trial_results),
                        snr_db=self.snr_db,
                        n_trials=self.n_trials,
                        mean_error_mm=float(np.mean(errors)),
                        median_error_mm=float(np.median(errors)),
                        std_error_mm=float(np.std(errors)),
                        percentile_95_mm=float(np.percentile(errors, 95)),
                        roi_accuracy=float(np.mean(roi_results)) if roi_results else 0.0,
                        roi_n_valid=len(roi_results),
                        error_by_depth=error_by_depth,
                        timestamp=datetime.now().isoformat(),
                        duration_seconds=duration
                    )

                    # Save results
                    output.to_json(config_dir / 'validation_results.json')

                    # Save raw trial results
                    with open(config_dir / 'trial_results.pkl', 'wb') as f:
                        pickle.dump(trial_results, f)

                    self.results[config_name] = output
                    self._raw_results[config_name] = trial_results

                    if verbose:
                        print(f"  Mean error: {output.mean_error_mm:.2f} mm")
                        print(f"  Median error: {output.median_error_mm:.2f} mm")
                        print(f"  ROI accuracy: {output.roi_accuracy:.1%}")
                        print(f"  Duration: {duration:.1f}s")

                except Exception as e:
                    print(f"  ERROR: {e}")
                    import traceback
                    traceback.print_exc()

        # Generate summary report
        self.generate_comparison_report()

        return self.results

    def generate_comparison_report(self) -> Path:
        """
        Generate cross-configuration comparison report.

        Returns
        -------
        report_path : Path
            Path to generated markdown report
        """
        report_path = self.output_dir / 'validation_comparison.md'

        with open(report_path, 'w') as f:
            f.write("# Batch Validation Comparison Report\n\n")
            f.write(f"**Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
            f.write(f"**Configurations:** {len(self.results)}\n\n")
            f.write("---\n\n")

            # Summary table
            f.write("## Overall Results\n\n")
            f.write("| Configuration | Sources | Mean (mm) | Median (mm) | 95th (mm) | ROI Acc |\n")
            f.write("|---------------|---------|-----------|-------------|-----------|----------|\n")

            # Sort by median error
            sorted_configs = sorted(
                self.results.items(),
                key=lambda x: x[1].median_error_mm
            )

            for config_name, result in sorted_configs:
                f.write(f"| {config_name} | {result.n_sources} | "
                       f"{result.mean_error_mm:.2f} | {result.median_error_mm:.2f} | "
                       f"{result.percentile_95_mm:.2f} | {result.roi_accuracy:.1%} |\n")

            f.write("\n")

            # Depth-stratified comparison
            f.write("## Error by Depth\n\n")

            # Get all depth bins
            all_bins = set()
            for result in self.results.values():
                all_bins.update(result.error_by_depth.keys())
            bins_sorted = sorted(all_bins, key=lambda x: float(x.split('-')[0].replace('mm', '').replace('+', '99')))

            # Header
            header = "| Configuration |"
            for bin_name in bins_sorted:
                header += f" {bin_name} |"
            f.write(header + "\n")

            separator = "|---------------|"
            for _ in bins_sorted:
                separator += "---------|"
            f.write(separator + "\n")

            # Data rows
            for config_name, result in sorted_configs:
                row = f"| {config_name} |"
                for bin_name in bins_sorted:
                    if bin_name in result.error_by_depth:
                        val = result.error_by_depth[bin_name]['median']
                        if np.isnan(val):
                            row += " - |"
                        else:
                            row += f" {val:.2f} |"
                    else:
                        row += " - |"
                f.write(row + "\n")

            f.write("\n")

            # Best configuration
            f.write("## Recommendations\n\n")
            best = sorted_configs[0]
            f.write(f"**Best overall:** {best[0]} (median error: {best[1].median_error_mm:.2f} mm)\n\n")

            # Best by depth
            f.write("**Best by depth:**\n\n")
            for bin_name in bins_sorted:
                best_for_bin = min(
                    [(name, r.error_by_depth.get(bin_name, {}).get('median', np.inf))
                     for name, r in self.results.items()],
                    key=lambda x: x[1] if not np.isnan(x[1]) else np.inf
                )
                if not np.isnan(best_for_bin[1]) and best_for_bin[1] != np.inf:
                    f.write(f"- {bin_name}: {best_for_bin[0]} ({best_for_bin[1]:.2f} mm)\n")

        # Also save as JSON summary
        summary = {
            'timestamp': datetime.now().isoformat(),
            'n_configs': len(self.results),
            'results': {name: r.to_dict() for name, r in self.results.items()},
            'best_overall': sorted_configs[0][0] if sorted_configs else None
        }

        with open(self.output_dir / 'validation_summary.json', 'w') as f:
            json.dump(summary, f, indent=2, default=str)

        print(f"\nComparison report saved to: {report_path}")
        return report_path

    def load_existing_results(self) -> Dict[str, ValidationOutputSchema]:
        """
        Load existing validation results from output directory.

        Returns
        -------
        results : dict
            Dictionary mapping config names to ValidationOutputSchema
        """
        for config_dir in self.output_dir.iterdir():
            if config_dir.is_dir():
                json_file = config_dir / 'validation_results.json'
                if json_file.exists():
                    self.results[config_dir.name] = ValidationOutputSchema.from_json(json_file)

                    # Load raw results if available
                    pkl_file = config_dir / 'trial_results.pkl'
                    if pkl_file.exists():
                        with open(pkl_file, 'rb') as f:
                            self._raw_results[config_dir.name] = pickle.load(f)

        return self.results

    @classmethod
    def compare_results(
        cls,
        result_paths: List[str],
        output_path: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Compare validation results from multiple directories.

        Parameters
        ----------
        result_paths : list of str
            Paths to validation_results.json files or directories containing them
        output_path : str, optional
            Path to save comparison report

        Returns
        -------
        comparison : dict
            Comparison statistics
        """
        results = {}

        for path in result_paths:
            path = Path(path)
            if path.is_dir():
                json_file = path / 'validation_results.json'
            else:
                json_file = path

            if json_file.exists():
                result = ValidationOutputSchema.from_json(json_file)
                config_name = f"{result.preset}_{result.method}"
                results[config_name] = result

        # Generate comparison
        comparison = {
            'n_configs': len(results),
            'configs': list(results.keys()),
            'by_median_error': sorted(
                [(name, r.median_error_mm) for name, r in results.items()],
                key=lambda x: x[1]
            ),
            'by_roi_accuracy': sorted(
                [(name, r.roi_accuracy) for name, r in results.items()],
                key=lambda x: x[1],
                reverse=True
            )
        }

        if output_path:
            with open(output_path, 'w') as f:
                json.dump(comparison, f, indent=2)

        return comparison
