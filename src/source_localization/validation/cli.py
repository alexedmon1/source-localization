"""
Command-line interface for validation module.

This module provides CLI functionality for running source localization
validation tests both standalone and as a subcommand of the main CLI.

Usage:
    # Run validation from a validation directory
    source-localization validate --test-dir /path/to/validation --config config/ --all

    # Run specific configs
    source-localization validate --test-dir /path/to/validation --config config/P01_ellipsoid_sloreta.yaml

    # Quick test mode (5 ROIs, 1 trial)
    source-localization validate --test-dir /path/to/validation --config config/ --quick

    # Compare existing results
    source-localization validate --compare results/config1/ results/config2/

Directory Structure:
    validation_dir/
    ├── config/                    # Config files (--config points here)
    │   ├── _base_validation.yaml
    │   └── P01_preset_method.yaml
    └── results/                   # Results go here (auto-created)
        └── P01_preset_method/
"""

import argparse
import sys
from pathlib import Path
from typing import List, Optional

from ..config import ATLAS_DEFINITIONS, LEGACY_ATLAS_ALIASES


def _atlas_choices_help() -> str:
    """One-line summary per atlas, built from the registry.

    Generated rather than hand-written: the previous hardcoded string described
    'full (47 ROIs)' and 'allen32 (32 Allen CCFv3 ROIs)' with no way to notice
    when either drifted from the shipped files.
    """
    parts = []
    for name in sorted(ATLAS_DEFINITIONS):
        meta = ATLAS_DEFINITIONS[name].get('meta', {})
        if meta.get('alias_of'):
            parts.append(f"{name} (= {meta['alias_of']})")
        elif meta.get('parcels') is not None:
            parts.append(f"{name} ({meta['parcels']} parcels)")
        else:
            parts.append(name)
    legacy = ', '.join(f"{k} (= {v})"
                       for k, v in sorted(LEGACY_ATLAS_ALIASES.items()))
    return ', '.join(parts) + f". Legacy aliases: {legacy}"


def create_parser() -> argparse.ArgumentParser:
    """
    Create argument parser for validation CLI.

    Returns
    -------
    argparse.ArgumentParser
        Configured argument parser
    """
    parser = argparse.ArgumentParser(
        prog='source-localization validate',
        description='Run source localization validation tests',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run all configs from a validation directory
  source-localization validate --test-dir /mnt/d/research/eeg/validation-2026-01-29 --config config/ --all

  # Run specific config files
  source-localization validate --test-dir ./my_validation --config config/P01.yaml config/P02.yaml

  # Quick test mode (5 ROIs, 1 trial)
  source-localization validate --test-dir ./my_validation --config config/ --quick

  # Override SNR and trials
  source-localization validate --test-dir ./my_validation --config config/ --snr 5 10 20 --trials 50

  # List available configs
  source-localization validate --test-dir ./my_validation --config config/ --list

  # Compare existing results
  source-localization validate --compare results/P01/ results/P02/

Directory Structure:
  validation_dir/           # --test-dir points here
  ├── config/               # --config points here (folder or specific files)
  │   ├── _base_validation.yaml
  │   └── P01_preset_method.yaml
  └── results/              # Results saved here automatically
      └── P01_preset_method/
          └── validation_results.json
"""
    )

    # Validation directory and config selection
    parser.add_argument(
        '--test-dir',
        type=str,
        required=False,
        help='Validation directory (contains config/, results/). Results are saved to {test-dir}/results/'
    )
    parser.add_argument(
        '--config', '-c',
        nargs='+',
        help='Config folder or specific config file(s) relative to --test-dir (e.g., config/ or config/P01.yaml)'
    )

    # Run parameters
    parser.add_argument(
        '--snr',
        type=float,
        nargs='+',
        help='SNR levels to test (default: from config)'
    )
    parser.add_argument(
        '--trials', '-n',
        type=int,
        help='Number of trials per location (default: from config)'
    )
    parser.add_argument(
        '--rois',
        type=int,
        nargs='+',
        help='ROI indices to test (default: all)'
    )

    # Atlas selection. Choices come from the shared registry plus the legacy
    # names this CLI used before the registry existed ('full', 'coarse_22roi'),
    # which scripts/ still pass. Previously this list was hardcoded and diverged
    # from the localization CLI's, so 'antwerp', 'allen' and 'allen64' could not
    # be validated at all.
    parser.add_argument(
        '--atlas',
        type=str,
        choices=sorted(ATLAS_DEFINITIONS) + sorted(LEGACY_ATLAS_ALIASES),
        default='full',
        help='Atlas parcellation to validate (default: full, an alias for '
             'antwerp). Choices: ' + _atlas_choices_help()
    )

    # Test mode
    parser.add_argument(
        '--test-mode',
        type=str,
        choices=['roi_centroids', 'uniform_grid', 'combined'],
        default=None,
        help='Override test mode: roi_centroids (best for ROI accuracy), '
             'uniform_grid (best for localization error), or '
             'combined (recommended: runs both, reports ROI acc from centroids + loc error from grid)'
    )

    # Modes
    parser.add_argument(
        '--quick', '-q',
        action='store_true',
        help='Quick test mode (5 well-covered ROIs, 1 trial, SNR=10)'
    )
    parser.add_argument(
        '--all', '-a',
        action='store_true',
        help='Run all configurations (ignore --config filter)'
    )

    # Output
    parser.add_argument(
        '--verbose', '-v',
        action='store_true',
        default=True,
        help='Verbose output (default: True)'
    )
    parser.add_argument(
        '--quiet',
        action='store_true',
        help='Suppress verbose output'
    )

    # List available
    parser.add_argument(
        '--list',
        action='store_true',
        help='List available configs and exit'
    )

    # =========================================================================
    # Batch Validation Arguments
    # =========================================================================
    batch_group = parser.add_argument_group('Batch Validation',
        'Run validation across multiple presets and methods')

    batch_group.add_argument(
        '--batch',
        action='store_true',
        help='Enable batch validation mode'
    )
    batch_group.add_argument(
        '--all-presets',
        action='store_true',
        help='Test all available presets'
    )
    batch_group.add_argument(
        '--presets',
        nargs='+',
        help='Specific presets to test (e.g., shell_ellipsoid ellipsoid_cartesian)'
    )
    batch_group.add_argument(
        '--methods',
        nargs='+',
        default=['sLORETA'],
        help='Inverse methods to test (default: sLORETA). Options: MNE, sLORETA, eLORETA, dSPM'
    )
    batch_group.add_argument(
        '--eeg',
        type=str,
        help='Path to EEG data file for batch validation'
    )
    batch_group.add_argument(
        '--output', '-o',
        type=str,
        default='./validation_results',
        help='Output directory for batch validation results (default: ./validation_results)'
    )
    batch_group.add_argument(
        '--batch-trials',
        type=int,
        default=25,
        help='Number of trials per source in batch mode (default: 25)'
    )
    batch_group.add_argument(
        '--batch-snr',
        type=float,
        default=10.0,
        help='SNR (dB) for batch validation simulations (default: 10.0)'
    )
    batch_group.add_argument(
        '--test-all-sources',
        action='store_true',
        default=True,
        help='Test all sources (default: True)'
    )
    batch_group.add_argument(
        '--n-test-sources',
        type=int,
        help='Number of sources to test (overrides --test-all-sources)'
    )

    # Compare existing results
    batch_group.add_argument(
        '--compare',
        nargs='+',
        metavar='PATH',
        help='Compare existing validation results from multiple directories or JSON files'
    )

    # Summarize existing results
    batch_group.add_argument(
        '--summarize',
        type=str,
        metavar='RESULTS_DIR',
        help='Summarize validation results from existing results directory (includes depth analysis)'
    )

    return parser


def run_validation_cli(args: argparse.Namespace) -> int:
    """
    Execute validation based on CLI arguments.

    Parameters
    ----------
    args : argparse.Namespace
        Parsed command-line arguments

    Returns
    -------
    int
        Exit code (0 for success, non-zero for error)
    """
    from .runner import ValidationConfigLoader, run_validation

    # Handle verbose/quiet
    verbose = args.verbose and not args.quiet

    # Handle --summarize (summary mode)
    if args.summarize:
        return run_summarize_mode(args.summarize, verbose)

    # Handle --compare (comparison mode)
    if args.compare:
        return run_compare_mode(args.compare, verbose)

    # Handle --batch (batch validation mode)
    if args.batch:
        return run_batch_mode(args, verbose)

    # Require --test-dir for non-batch, non-compare modes
    if not args.test_dir:
        print("Error: --test-dir is required. Specify the validation directory.")
        print("Example: source-localization validate --test-dir /path/to/validation --config config/ --all")
        return 1

    test_dir = Path(args.test_dir)
    if not test_dir.exists():
        print(f"Error: Validation directory not found: {test_dir}")
        return 1

    # Resolve config paths relative to test_dir
    config_paths = []
    if args.config:
        for config_arg in args.config:
            config_path = test_dir / config_arg
            if config_path.is_dir():
                # It's a directory - discover all configs in it
                config_paths.extend(sorted(config_path.glob('[!_]*.yaml')))
            elif config_path.is_file():
                config_paths.append(config_path)
            else:
                print(f"Warning: Config path not found: {config_path}")
    else:
        # Default to config/ subdirectory
        default_config_dir = test_dir / 'config'
        if default_config_dir.exists():
            config_paths = sorted(default_config_dir.glob('[!_]*.yaml'))
        else:
            print(f"Error: No config directory found. Specify --config or create {default_config_dir}")
            return 1

    if not config_paths:
        print("Error: No config files found.")
        return 1

    # Handle --list
    if args.list:
        print(f"Configs in {test_dir}:")
        for config in config_paths:
            print(f"  {config.stem}")
        return 0

    # Filter configs if not --all
    if not args.all and args.config:
        # If specific files were given (not directories), use only those
        pass  # config_paths already contains only specified files
    elif not args.all:
        print("Error: Specify --all to run all configs, or provide specific config files.")
        print("Example: --config config/ --all  OR  --config config/P01.yaml config/P02.yaml")
        return 1

    # Get ROI indices
    roi_indices = args.rois

    try:
        results = run_validation(
            test_dir=str(test_dir),
            config_files=config_paths,
            snr_levels=args.snr,
            n_trials=args.trials,
            roi_indices=roi_indices,
            quick=args.quick,
            verbose=verbose,
            atlas=args.atlas,
            test_mode=args.test_mode
        )

        if not results:
            print("No validation results produced.")
            return 1

        return 0

    except Exception as e:
        print(f"Error running validation: {e}")
        import traceback
        traceback.print_exc()
        return 1


def run_batch_mode(args: argparse.Namespace, verbose: bool) -> int:
    """
    Run batch validation across multiple presets and methods.

    Parameters
    ----------
    args : argparse.Namespace
        Parsed command-line arguments
    verbose : bool
        Whether to print verbose output

    Returns
    -------
    int
        Exit code (0 for success, non-zero for error)
    """
    from .batch_runner import BatchValidationRunner

    # Validate required arguments
    if not args.eeg:
        print("Error: --eeg argument is required for batch validation")
        return 1

    eeg_path = Path(args.eeg)
    if not eeg_path.exists():
        print(f"Error: EEG file not found: {eeg_path}")
        return 1

    # Determine presets to test
    if args.all_presets:
        presets = BatchValidationRunner.AVAILABLE_PRESETS
    elif args.presets:
        presets = args.presets
    else:
        print("Error: Specify --all-presets or --presets for batch validation")
        return 1

    # Validate methods
    valid_methods = ['MNE', 'sLORETA', 'eLORETA', 'dSPM', 'LCMV']
    for method in args.methods:
        if method not in valid_methods:
            print(f"Error: Unknown method '{method}'. Valid methods: {valid_methods}")
            return 1

    # Determine test_all_sources
    test_all_sources = args.test_all_sources
    n_test_sources = None
    if args.n_test_sources:
        test_all_sources = False
        n_test_sources = args.n_test_sources

    try:
        if verbose:
            print(f"\nBatch Validation Configuration:")
            print(f"  EEG file: {eeg_path}")
            print(f"  Output: {args.output}")
            print(f"  Presets: {presets}")
            print(f"  Methods: {args.methods}")
            print(f"  Trials: {args.batch_trials}")
            print(f"  SNR: {args.batch_snr} dB")
            print(f"  Test all sources: {test_all_sources}")

        runner = BatchValidationRunner(
            presets=presets,
            methods=args.methods,
            eeg_file=str(eeg_path),
            output_dir=args.output,
            n_trials=args.batch_trials,
            snr_db=args.batch_snr,
            test_all_sources=test_all_sources,
            n_test_sources=n_test_sources,
            atlas=args.atlas,
            test_mode=args.test_mode
        )

        results = runner.run_all(verbose=verbose)

        if not results:
            print("No validation results produced.")
            return 1

        print(f"\nValidation complete. Results saved to: {args.output}")
        return 0

    except Exception as e:
        print(f"Error running batch validation: {e}")
        import traceback
        traceback.print_exc()
        return 1


def run_summarize_mode(results_dir: str, verbose: bool) -> int:
    """
    Summarize validation results from existing results directory.

    Parameters
    ----------
    results_dir : str
        Path to results directory containing validation result folders
    verbose : bool
        Whether to print verbose output

    Returns
    -------
    int
        Exit code (0 for success, non-zero for error)
    """
    from .summarize import summarize_validation_folders

    try:
        summary = summarize_validation_folders(
            results_dir=results_dir,
            verbose=verbose
        )
        return 0

    except Exception as e:
        print(f"Error summarizing results: {e}")
        import traceback
        traceback.print_exc()
        return 1


def run_compare_mode(result_paths: List[str], verbose: bool) -> int:
    """
    Compare existing validation results.

    Parameters
    ----------
    result_paths : list of str
        Paths to validation result directories or JSON files
    verbose : bool
        Whether to print verbose output

    Returns
    -------
    int
        Exit code (0 for success, non-zero for error)
    """
    from .batch_runner import BatchValidationRunner

    try:
        comparison = BatchValidationRunner.compare_results(result_paths)

        print("\n" + "=" * 60)
        print("VALIDATION COMPARISON")
        print("=" * 60)
        print(f"Configurations compared: {comparison['n_configs']}")
        print()

        print("Ranked by Median Localization Error:")
        print("-" * 40)
        for rank, (name, error) in enumerate(comparison['by_median_error'], 1):
            print(f"  {rank}. {name}: {error:.2f} mm")

        print()
        print("Ranked by ROI Accuracy:")
        print("-" * 40)
        for rank, (name, acc) in enumerate(comparison['by_roi_accuracy'], 1):
            print(f"  {rank}. {name}: {acc:.1%}")

        return 0

    except Exception as e:
        print(f"Error comparing results: {e}")
        import traceback
        traceback.print_exc()
        return 1


def main(args: Optional[List[str]] = None) -> int:
    """
    Main entry point for validation CLI.

    Parameters
    ----------
    args : list of str, optional
        Command-line arguments (default: sys.argv[1:])

    Returns
    -------
    int
        Exit code
    """
    parser = create_parser()
    parsed_args = parser.parse_args(args)
    return run_validation_cli(parsed_args)


if __name__ == '__main__':
    sys.exit(main())
