#!/usr/bin/env python
"""
Parallel Validation Runner.

Runs validation configs in parallel using multiprocessing.
Results are organized into test category subdirectories.
"""

import argparse
import shutil
import subprocess
import sys
import os
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent
RESULTS_BASE = PROJECT_ROOT / 'validation' / 'results'


def run_config(config_name: str, test_name: str, n_trials: int, atlas: str, test_mode: str) -> tuple:
    """Run a single config and return (config_name, success, error_msg)."""
    cmd = [
        sys.executable, '-m', 'source_localization.validation',
        '--test', test_name,
        '--config', config_name,
        '--trials', str(n_trials),
        '--atlas', atlas,
        '--test-mode', test_mode,
        '--quiet'
    ]

    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            cwd=str(PROJECT_ROOT),
            timeout=7200,  # 2 hour timeout
            env={**os.environ, 'MNE_LOGGING_LEVEL': 'ERROR'}
        )

        if result.returncode == 0:
            # Move results to test category subdirectory
            # Config name includes atlas suffix (e.g., V01_sphere_vol_dspm_coarse22)
            src_dir = RESULTS_BASE / config_name
            dst_dir = RESULTS_BASE / test_name / config_name

            if src_dir.exists():
                dst_dir.parent.mkdir(parents=True, exist_ok=True)
                if dst_dir.exists():
                    shutil.rmtree(dst_dir)
                shutil.move(str(src_dir), str(dst_dir))

            return (config_name, True, "")
        else:
            # Get last 200 chars of stderr for error msg
            err = result.stderr[-200:] if result.stderr else "Unknown error"
            return (config_name, False, err)

    except subprocess.TimeoutExpired:
        return (config_name, False, "TIMEOUT")
    except Exception as e:
        return (config_name, False, str(e))


def get_configs(test_name: str) -> list:
    """Get list of config names for a test."""
    config_dir = PROJECT_ROOT / 'src' / 'source_localization' / 'validation' / 'config' / 'default_tests' / test_name
    configs = []
    for f in sorted(config_dir.glob('[VDCS]*.yaml')):
        if not f.stem.startswith('_'):
            configs.append(f.stem)
    return configs


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--test', default='original')
    parser.add_argument('--trials', type=int, default=25)
    parser.add_argument('--workers', type=int, default=6)
    parser.add_argument('--atlas', default='coarse_22roi')
    parser.add_argument('--test-mode', default='uniform_grid')
    parser.add_argument('--configs', nargs='+', help='Specific configs to run')
    args = parser.parse_args()

    configs = args.configs if args.configs else get_configs(args.test)

    output_dir = RESULTS_BASE / args.test
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Running {len(configs)} configs in parallel (workers={args.workers})")
    print(f"Test: {args.test}, Trials: {args.trials}, Atlas: {args.atlas}")
    print(f"Output: {output_dir}")
    print(f"Started: {datetime.now().strftime('%H:%M:%S')}")
    print("-" * 60)

    completed = 0
    failed = []

    with ProcessPoolExecutor(max_workers=args.workers) as executor:
        futures = {
            executor.submit(run_config, cfg, args.test, args.trials, args.atlas, args.test_mode): cfg
            for cfg in configs
        }

        for future in as_completed(futures):
            cfg = futures[future]
            config_name, success, err = future.result()
            completed += 1

            if success:
                print(f"[{completed}/{len(configs)}] ✓ {config_name}")
            else:
                print(f"[{completed}/{len(configs)}] ✗ {config_name}: {err[:50]}")
                failed.append(config_name)

    print("-" * 60)
    print(f"Finished: {datetime.now().strftime('%H:%M:%S')}")
    print(f"Success: {len(configs) - len(failed)}/{len(configs)}")

    if failed:
        print(f"Failed: {failed}")
        return 1
    return 0


if __name__ == '__main__':
    sys.exit(main())
