"""Command-line interface for source localization."""

import argparse
import logging
import sys
from pathlib import Path

from .pipeline import Pipeline
from .config import Config

logger = logging.getLogger(__name__)


def _get_available_presets():
    """Dynamically discover available presets from config/presets directory."""
    presets_dir = Path(__file__).parent / 'config' / 'presets'
    if presets_dir.exists():
        return sorted([p.stem for p in presets_dir.glob('*.yaml')])
    return []


def _create_run_parser(subparsers=None):
    """Create parser for the 'run' command (main pipeline execution)."""
    if subparsers is not None:
        parser = subparsers.add_parser(
            'run',
            help='Run the source localization pipeline',
            formatter_class=argparse.RawDescriptionHelpFormatter,
            epilog='''
Examples:
  source-localization run --preset sphere_volumetric --eeg data.set
  source-localization run --bem sphere --source volumetric --eeg data.set
  source-localization run --config my_config.yaml --eeg data.set
'''
        )
    else:
        parser = argparse.ArgumentParser(
            description='Mouse EEG Source Localization Pipeline',
            formatter_class=argparse.RawDescriptionHelpFormatter,
            epilog='''
Examples:
  %(prog)s --preset sphere_volumetric --eeg data.set
  %(prog)s --bem sphere --source volumetric --eeg data.set
  %(prog)s --config my_config.yaml --eeg data.set
  %(prog)s --preset ellipsoid_surface --eeg data.set --snr 5.0 --method dSPM
'''
        )

    # Configuration source (mutually exclusive)
    config_group = parser.add_mutually_exclusive_group(required=True)
    config_group.add_argument(
        '--preset', '-p',
        choices=_get_available_presets(),
        help='Use preset configuration'
    )
    config_group.add_argument(
        '--config', '-c',
        help='Path to custom config YAML file'
    )
    config_group.add_argument(
        '--bem',
        help='BEM type (use with --source)'
    )

    # Source type (used with --bem)
    parser.add_argument(
        '--source',
        help='Source space type (use with --bem)'
    )

    # Required inputs
    parser.add_argument(
        '--eeg',
        required=True,
        help='Path to EEG data file (.set format)'
    )

    # Optional overrides
    parser.add_argument('--output', '-o', help='Output directory')
    parser.add_argument('--snr', type=float, help='SNR for inverse solution')
    parser.add_argument('--method', choices=['MNE', 'dSPM', 'sLORETA'], help='Inverse method')
    parser.add_argument('--spacing', type=float, help='Source spacing in mm')

    # Optional post-processing flags
    parser.add_argument('--spectral', action='store_true',
                        help='Include spectral analysis (optional post-processing)')
    parser.add_argument('--visualize', action='store_true',
                        help='Include visualization (optional post-processing)')

    # Atlas selection
    parser.add_argument('--atlas',
                        choices=['antwerp', 'allen'],
                        default=None,
                        help='Atlas to use: antwerp (47 ROIs, default) or allen (49 ROIs)')

    # Other flags
    parser.add_argument('--verbose', '-v', action='store_true', help='Verbose output')

    # BEM caching control
    parser.add_argument('--recreate-bem', action='store_true',
                        help='Force BEM model recreation even if cached')
    parser.add_argument('--no-bem-cache', action='store_true',
                        help='Disable BEM caching (always create fresh)')

    return parser


def _run_pipeline(args):
    """Execute the source localization pipeline."""
    # Validate --bem and --source must be used together
    if args.bem and not args.source:
        print("Error: --bem requires --source", file=sys.stderr)
        return 1
    if args.source and not args.bem:
        print("Error: --source requires --bem", file=sys.stderr)
        return 1

    # Build overrides dict
    overrides = {}
    if args.snr:
        overrides['inverse.snr'] = args.snr
    if args.method:
        overrides['inverse.method'] = args.method
    if args.spacing:
        overrides['source_space.spacing_mm'] = args.spacing

    # BEM caching overrides
    if args.recreate_bem:
        overrides['bem.sphere.force_recreate'] = True
        overrides['bem.ellipsoid.force_recreate'] = True
    if args.no_bem_cache:
        overrides['bem.sphere.use_cache'] = False
        overrides['bem.ellipsoid.use_cache'] = False

    # Get optional flags
    include_spectral = getattr(args, 'spectral', False)
    include_visualization = getattr(args, 'visualize', False)

    # Get atlas selection
    atlas = getattr(args, 'atlas', None)

    # Create and run pipeline
    try:
        if args.preset:
            pipeline = Pipeline.from_preset(args.preset, atlas=atlas, **overrides)
            results = pipeline.run(eeg_file=args.eeg, output_dir=args.output,
                                   include_spectral=include_spectral,
                                   include_visualization=include_visualization)

        elif args.bem and args.source:
            pipeline = Pipeline.from_bem_source(args.bem, args.source, **overrides)
            if atlas:
                pipeline.config.apply_atlas(atlas)
            results = pipeline.run(eeg_file=args.eeg, output_dir=args.output,
                                   include_spectral=include_spectral,
                                   include_visualization=include_visualization)

        elif args.config:
            pipeline = Pipeline.from_file(args.config)
            if atlas:
                pipeline.config.apply_atlas(atlas)
            results = pipeline.run(eeg_file=args.eeg, output_dir=args.output,
                                   include_spectral=include_spectral,
                                   include_visualization=include_visualization)

        return 0

    except Exception as e:
        print(f"Pipeline failed: {e}", file=sys.stderr)
        if args.verbose:
            import traceback
            traceback.print_exc()
        return 1


def _create_validate_parser(subparsers):
    """Create parser for the 'validate' command."""
    try:
        from .validation.cli import create_parser as create_validation_parser
    except (ImportError, ModuleNotFoundError):
        return  # validation module not available

    # Get the validation parser
    val_parser = create_validation_parser()

    # Add it as a subcommand
    parser = subparsers.add_parser(
        'validate',
        help='Run source localization validation tests',
        parents=[val_parser],
        add_help=False,  # Parent already has help
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=val_parser.epilog
    )

    return parser


def _create_study_parser(subparsers):
    """Create parser for the 'study' command."""
    parser = subparsers.add_parser(
        'study',
        help='Process multi-subject studies',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog='''
Examples:
  # Initialize study from folder (auto-discover .set files)
  source-localization study init /path/to/data --name "MyStudy"

  # Process all subjects in a study
  source-localization study run /path/to/study_config.yaml

  # Process specific subjects
  source-localization study run study_config.yaml --subjects 901 902 903

  # Process with parallel workers
  source-localization study run study_config.yaml --jobs 4

  # Collect group results
  source-localization study collect study_config.yaml
'''
    )

    study_subparsers = parser.add_subparsers(dest='study_command', title='study commands')

    # Init subcommand
    init_parser = study_subparsers.add_parser(
        'init',
        help='Initialize study configuration from a folder',
    )
    init_parser.add_argument('folder', help='Folder containing EEG data')
    init_parser.add_argument('--name', help='Study name (default: folder name)')
    init_parser.add_argument('--participants', help='CSV file with participant metadata')
    init_parser.add_argument('--preset', default='roi_based_ellipsoid',
                             choices=_get_available_presets(),
                             help='Pipeline preset (default: roi_based_ellipsoid)')

    # Run subcommand
    run_parser = study_subparsers.add_parser(
        'run',
        help='Run source localization on all study subjects',
    )
    run_parser.add_argument('config', help='Path to study_config.yaml')
    run_parser.add_argument('--subjects', nargs='+', help='Specific subject IDs to process')
    run_parser.add_argument('--jobs', '-j', type=int, default=1,
                            help='Number of parallel jobs (default: 1)')
    run_parser.add_argument('--force', action='store_true',
                            help='Re-process subjects even if output exists')
    run_parser.add_argument('--verbose', '-v', action='store_true', help='Verbose output')
    run_parser.add_argument('--no-qc', action='store_true',
                            help='Skip automatic QC after processing')

    # Collect subcommand
    collect_parser = study_subparsers.add_parser(
        'collect',
        help='Collect group-level results',
    )
    collect_parser.add_argument('config', help='Path to study_config.yaml')

    # Status subcommand
    status_parser = study_subparsers.add_parser(
        'status',
        help='Show study processing status',
    )
    status_parser.add_argument('config', help='Path to study_config.yaml')

    # Analyze subcommand
    analyze_parser = study_subparsers.add_parser(
        'analyze',
        help='Run spectral and connectivity analysis (uses MNE)',
    )
    analyze_parser.add_argument('config', help='Path to study_config.yaml')
    analyze_parser.add_argument('--subjects', nargs='+', help='Specific subject IDs to analyze')
    analyze_parser.add_argument('--jobs', '-j', type=int, default=1,
                                help='Number of parallel jobs (default: 1)')
    analyze_parser.add_argument('--bands', nargs='+', default=['delta', 'theta', 'alpha', 'beta', 'low_gamma', 'high_gamma'],
                                help='Frequency bands to analyze')
    analyze_parser.add_argument('--connectivity', nargs='+', default=['coherence'],
                                help='Connectivity methods: coherence, plv, wpli, imcoh')
    analyze_parser.add_argument('--epoch-length', type=float, default=2.0,
                                help='Epoch length in seconds for connectivity (default: 2.0)')
    analyze_parser.add_argument('--force', action='store_true',
                                help='Overwrite existing analysis')
    analyze_parser.add_argument('--verbose', '-v', action='store_true', help='Verbose output')

    # QC subcommand
    qc_parser = study_subparsers.add_parser(
        'qc',
        help='Run quality control on processed subjects',
    )
    qc_parser.add_argument('config', help='Path to study_config.yaml')
    qc_parser.add_argument('--threshold', type=float, default=2.0,
                           help='Z-score threshold for outlier detection (default: 2.0)')
    qc_parser.add_argument('--output', help='Override QC output directory')
    qc_parser.add_argument('--verbose', '-v', action='store_true', help='Verbose output')

    return parser


def _run_study_command(args):
    """Execute study subcommands."""
    from .study import StudyConfig, process_study, create_study_from_folder, collect_group_results

    if args.study_command == 'init':
        # Initialize study from folder
        config = create_study_from_folder(
            folder=args.folder,
            name=args.name,
            participants_file=args.participants,
            preset=args.preset,
            save_config=True,
        )
        print(config.summary())
        print(f"\nStudy configuration saved to: {config.root_dir / 'study_config.yaml'}")
        return 0

    elif args.study_command == 'run':
        # Load configuration
        config = StudyConfig.from_yaml(args.config)

        # Validate
        errors = config.validate()
        if errors:
            print("Configuration errors:", file=sys.stderr)
            for error in errors:
                print(f"  - {error}", file=sys.stderr)
            return 1

        print(config.summary())
        print()

        # Set up logging
        if args.verbose:
            logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
        else:
            logging.basicConfig(level=logging.WARNING, format='%(levelname)s: %(message)s')

        # Progress callback
        def progress_callback(completed, total, subject_id, success):
            status = "OK" if success else "FAILED"
            print(f"[{completed}/{total}] {subject_id}: {status}")

        # Run processing
        result = process_study(
            config=config,
            n_jobs=args.jobs,
            skip_existing=not args.force,
            subjects=args.subjects,
            progress_callback=progress_callback,
        )

        print()
        print(result.summary())

        # Auto-run QC unless --no-qc
        if not args.no_qc:
            from .study import run_qc
            print("\nRunning QC...")
            try:
                qc_result = run_qc(config, study_result=result, verbose=args.verbose)
                if qc_result.report_path:
                    print(f"QC report: {qc_result.report_path}")
            except Exception as e:
                print(f"QC failed (non-fatal): {e}", file=sys.stderr)

        return 0 if result.n_failed == 0 else 1

    elif args.study_command == 'collect':
        # Collect group results
        config = StudyConfig.from_yaml(args.config)
        df = collect_group_results(config)
        print(f"Collected {len(df)} rows from {df['subject_id'].nunique()} subjects")
        print(f"Output saved to: {config.group_dir / 'group_band_power.csv'}")
        return 0

    elif args.study_command == 'status':
        # Show study status
        config = StudyConfig.from_yaml(args.config)
        print(config.summary())
        print()

        # Check processing status
        processed = 0
        analyzed = 0
        pending = 0
        for subject in config.subjects:
            output_dir = config.get_subject_output_dir(subject)
            if (output_dir / "roi_timeseries" / "roi_timeseries_signed.set").exists():
                processed += 1
                if (output_dir / "analysis" / "band_power.csv").exists():
                    analyzed += 1
            else:
                pending += 1

        print(f"Processing status:")
        print(f"  Processed: {processed}")
        print(f"  Analyzed: {analyzed}")
        print(f"  Pending: {pending}")
        return 0

    elif args.study_command == 'qc':
        # Run standalone QC
        from .study import run_qc

        config = StudyConfig.from_yaml(args.config)

        if args.verbose:
            logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
        else:
            logging.basicConfig(level=logging.WARNING, format='%(levelname)s: %(message)s')

        print(f"Running QC on {len(config.subjects)} subjects...")
        qc_result = run_qc(
            config,
            outlier_threshold=args.threshold,
            output_dir=args.output,
            verbose=args.verbose,
        )
        if qc_result.report_path:
            print(f"\nQC report: {qc_result.report_path}")
        return 0

    elif args.study_command == 'analyze':
        # Run analysis on processed subjects
        from .study import analyze_study, DEFAULT_BANDS

        config = StudyConfig.from_yaml(args.config)

        # Set up logging
        if args.verbose:
            logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
        else:
            logging.basicConfig(level=logging.WARNING, format='%(levelname)s: %(message)s')

        # Parse bands
        bands = {b: DEFAULT_BANDS[b] for b in args.bands if b in DEFAULT_BANDS}
        if not bands:
            print(f"No valid bands specified. Available: {list(DEFAULT_BANDS.keys())}")
            return 1

        print(f"Analyzing {len(config.subjects)} subjects...")
        print(f"Bands: {list(bands.keys())}")
        print(f"Connectivity: {args.connectivity}")
        print(f"Epoch length: {args.epoch_length}s")
        print()

        # Run analysis
        df = analyze_study(
            config=config,
            bands=bands,
            connectivity_methods=args.connectivity,
            connectivity_bands=list(bands.keys()),
            n_jobs=args.jobs,
            overwrite=args.force,
            subjects=args.subjects,
            epoch_length=args.epoch_length,
        )

        print(f"\nAnalysis complete!")
        print(f"Band power: {len(df)} rows, {df['subject_id'].nunique()} subjects")
        print(f"Results saved to: {config.group_dir}")
        return 0

    else:
        print("Use 'source-localization study --help' for usage", file=sys.stderr)
        return 1


def main():
    """Main CLI entry point."""
    # Check if called with old-style arguments (no subcommand)
    # For backwards compatibility, if --preset, --config, --bem, or --eeg are given,
    # treat as run command
    if len(sys.argv) > 1:
        first_arg = sys.argv[1]
        old_style_args = ['--preset', '-p', '--config', '-c', '--bem', '--eeg']
        if first_arg in old_style_args or first_arg.startswith('--preset='):
            # Old style: run without subcommand
            parser = _create_run_parser()
            args = parser.parse_args()
            return _run_pipeline(args)

    # New style with subcommands
    parser = argparse.ArgumentParser(
        prog='source-localization',
        description='Mouse EEG Source Localization Pipeline',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog='''
Commands:
  run         Run the source localization pipeline on EEG data
  validate    Run validation tests (dipole simulation)
  study       Process multi-subject studies

Examples:
  source-localization run --preset ellipsoid_surface --eeg data.set
  source-localization validate --test original --config V01 V08
  source-localization study init /path/to/data --name "MyStudy"
  source-localization study run study_config.yaml --jobs 4

For backwards compatibility, you can also run:
  source-localization --preset ellipsoid_surface --eeg data.set
'''
    )

    subparsers = parser.add_subparsers(dest='command', title='commands')

    # Add subcommands
    _create_run_parser(subparsers)
    _create_validate_parser(subparsers)
    _create_study_parser(subparsers)

    # Parse and execute
    args = parser.parse_args()

    if args.command is None:
        parser.print_help()
        return 0

    if args.command == 'run':
        return _run_pipeline(args)

    elif args.command == 'validate':
        from .validation.cli import run_validation_cli
        return run_validation_cli(args)

    elif args.command == 'study':
        return _run_study_command(args)

    return 0


if __name__ == '__main__':
    sys.exit(main())
