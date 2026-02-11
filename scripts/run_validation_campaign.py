#!/usr/bin/env python
"""
Comprehensive Validation Campaign Runner.

Runs all validation configurations with coarse 22-ROI atlas and uniform_grid test mode.
Organizes results into proper directory hierarchy.

Usage:
    python scripts/run_validation_campaign.py --phase original --trials 100
    python scripts/run_validation_campaign.py --phase all --trials 100
"""

import argparse
import json
import subprocess
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import shutil

# Base paths
SCRIPT_DIR = Path(__file__).parent
PROJECT_ROOT = SCRIPT_DIR.parent
RESULTS_BASE = PROJECT_ROOT / 'validation' / 'results'


def run_single_config(
    test_name: str,
    config_name: str,
    output_dir: Path,
    n_trials: int = 100,
    atlas: str = 'coarse_22roi',
    test_mode: str = 'uniform_grid'
) -> Tuple[str, bool, str]:
    """
    Run a single validation configuration.

    Returns tuple of (config_name, success, message)
    """
    cmd = [
        sys.executable, '-m', 'source_localization.validation',
        '--test', test_name,
        '--config', config_name,
        '--trials', str(n_trials),
        '--atlas', atlas,
        '--test-mode', test_mode
    ]

    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            cwd=str(PROJECT_ROOT),
            timeout=3600  # 1 hour timeout per config
        )

        if result.returncode == 0:
            # Move results to proper directory
            src_dir = PROJECT_ROOT / 'validation' / 'results' / config_name
            dst_dir = output_dir / config_name

            if src_dir.exists():
                if dst_dir.exists():
                    shutil.rmtree(dst_dir)
                shutil.move(str(src_dir), str(dst_dir))
                return (config_name, True, f"Success: {dst_dir}")
            else:
                # Check if it was placed in test-specific dir already
                return (config_name, True, f"Success (already in place)")
        else:
            return (config_name, False, f"Error: {result.stderr[:500]}")

    except subprocess.TimeoutExpired:
        return (config_name, False, "Timeout (>1 hour)")
    except Exception as e:
        return (config_name, False, f"Exception: {str(e)}")


def get_configs_for_test(test_name: str) -> List[str]:
    """Get list of config names for a test category."""
    config_dir = PROJECT_ROOT / 'src' / 'source_localization' / 'validation' / 'config' / 'default_tests' / test_name

    configs = []
    for f in sorted(config_dir.glob('[VDCS]*.yaml')):
        if not f.stem.startswith('_'):
            configs.append(f.stem)

    return configs


def run_test_category(
    test_name: str,
    n_trials: int = 100,
    max_workers: int = 4,
    config_filter: Optional[List[str]] = None,
    verbose: bool = True
) -> Dict[str, bool]:
    """
    Run all configs for a test category.

    Parameters
    ----------
    test_name : str
        Test category: 'original', 'dipole_size', 'conductivity_ratio', 'brain_size'
    n_trials : int
        Number of trials per position
    max_workers : int
        Maximum parallel processes
    config_filter : list of str, optional
        If provided, only run configs containing these substrings
    verbose : bool
        Print progress

    Returns
    -------
    dict
        Mapping of config_name -> success status
    """
    output_dir = RESULTS_BASE / test_name
    output_dir.mkdir(parents=True, exist_ok=True)

    configs = get_configs_for_test(test_name)

    if config_filter:
        configs = [c for c in configs if any(f in c for f in config_filter)]

    if verbose:
        print(f"\n{'='*60}")
        print(f"Running {test_name} tests ({len(configs)} configs)")
        print(f"Output: {output_dir}")
        print(f"Trials: {n_trials}, Workers: {max_workers}")
        print(f"{'='*60}\n")

    results = {}

    # Run in parallel
    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        futures = {
            executor.submit(
                run_single_config,
                test_name,
                config,
                output_dir,
                n_trials
            ): config
            for config in configs
        }

        for future in as_completed(futures):
            config = futures[future]
            try:
                config_name, success, message = future.result()
                results[config_name] = success

                if verbose:
                    status = "✓" if success else "✗"
                    print(f"  [{status}] {config_name}: {message}")

            except Exception as e:
                results[config] = False
                if verbose:
                    print(f"  [✗] {config}: Exception - {e}")

    # Summary
    n_success = sum(1 for v in results.values() if v)
    n_failed = len(results) - n_success

    if verbose:
        print(f"\n{test_name} Summary: {n_success}/{len(results)} succeeded")
        if n_failed > 0:
            failed = [k for k, v in results.items() if not v]
            print(f"  Failed: {failed}")

    return results


def find_best_original_config(results_dir: Path) -> Optional[str]:
    """
    Find the best-performing original config based on ROI accuracy.

    Returns config name (e.g., 'V10_ellipsoid_vol_sloreta')
    """
    original_dir = results_dir / 'original'
    if not original_dir.exists():
        return None

    best_config = None
    best_accuracy = -1

    for config_dir in original_dir.iterdir():
        if not config_dir.is_dir():
            continue

        metrics_file = config_dir / 'metrics.json'
        if not metrics_file.exists():
            continue

        try:
            with open(metrics_file) as f:
                metrics = json.load(f)

            # Get ROI accuracy for SNR=10 (standard test SNR)
            for snr_key, snr_data in metrics.get('snr_results', {}).items():
                accuracy = snr_data.get('roi_accuracy', {}).get('exact', 0)
                if accuracy > best_accuracy:
                    best_accuracy = accuracy
                    best_config = config_dir.name

        except Exception:
            continue

    return best_config


def get_best_config_type(config_name: str) -> Tuple[str, str, str]:
    """
    Extract BEM type, source type, and inverse method from config name.

    E.g., 'V10_ellipsoid_vol_sloreta' -> ('ellipsoid', 'vol', 'sloreta')
    """
    parts = config_name.split('_')
    # Format: V##_bem_source_method
    if len(parts) >= 4:
        return parts[1], parts[2], parts[3]
    return None, None, None


def main():
    parser = argparse.ArgumentParser(description='Run validation campaign')
    parser.add_argument(
        '--phase',
        choices=['original', 'specialized', 'all'],
        default='all',
        help='Which phase to run'
    )
    parser.add_argument(
        '--trials', '-n',
        type=int,
        default=100,
        help='Trials per position'
    )
    parser.add_argument(
        '--workers', '-w',
        type=int,
        default=4,
        help='Max parallel workers'
    )
    parser.add_argument(
        '--best-config',
        type=str,
        help='Override best config for specialized tests (e.g., V10_ellipsoid_vol_sloreta)'
    )

    args = parser.parse_args()

    print(f"\n{'#'*60}")
    print(f"# VALIDATION CAMPAIGN")
    print(f"# Started: {datetime.now().isoformat()}")
    print(f"# Trials: {args.trials}, Workers: {args.workers}")
    print(f"{'#'*60}\n")

    all_results = {}

    # Phase 1: Original tests
    if args.phase in ['original', 'all']:
        all_results['original'] = run_test_category(
            'original',
            n_trials=args.trials,
            max_workers=args.workers
        )

    # Determine best config
    if args.best_config:
        best_config = args.best_config
    else:
        best_config = find_best_original_config(RESULTS_BASE)
        if not best_config and args.phase != 'original':
            # Default to V10 (sLORETA ellipsoid volumetric - known good performer)
            best_config = 'V10_ellipsoid_vol_sloreta'

    print(f"\nBest config for specialized tests: {best_config}")

    if best_config:
        bem_type, source_type, method = get_best_config_type(best_config)
        print(f"  BEM: {bem_type}, Source: {source_type}, Method: {method}")

    # Phase 2: Specialized tests (only matching config type)
    if args.phase in ['specialized', 'all'] and best_config:
        bem_type, source_type, method = get_best_config_type(best_config)

        # Dipole size tests - filter to matching source type
        dipole_filter = [f'_{source_type}_']
        print(f"\nDipole size filter: {dipole_filter}")
        all_results['dipole_size'] = run_test_category(
            'dipole_size',
            n_trials=args.trials,
            max_workers=args.workers,
            config_filter=dipole_filter
        )

        # Conductivity ratio tests - filter to matching source type
        cond_filter = [f'_{source_type}_']
        print(f"\nConductivity ratio filter: {cond_filter}")
        all_results['conductivity_ratio'] = run_test_category(
            'conductivity_ratio',
            n_trials=args.trials,
            max_workers=args.workers,
            config_filter=cond_filter
        )

        # Brain size tests - filter to matching source type
        brain_filter = [f'_{source_type}_']
        print(f"\nBrain size filter: {brain_filter}")
        all_results['brain_size'] = run_test_category(
            'brain_size',
            n_trials=args.trials,
            max_workers=args.workers,
            config_filter=brain_filter
        )

    # Final summary
    print(f"\n{'='*60}")
    print("CAMPAIGN COMPLETE")
    print(f"{'='*60}")

    total_success = 0
    total_configs = 0

    for category, results in all_results.items():
        n_success = sum(1 for v in results.values() if v)
        n_total = len(results)
        total_success += n_success
        total_configs += n_total
        print(f"  {category}: {n_success}/{n_total}")

    print(f"\nTotal: {total_success}/{total_configs} configs succeeded")
    print(f"Finished: {datetime.now().isoformat()}")

    return 0 if total_success == total_configs else 1


if __name__ == '__main__':
    sys.exit(main())
