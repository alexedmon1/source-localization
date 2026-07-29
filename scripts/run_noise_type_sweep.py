#!/usr/bin/env python
"""
Sweep localization error across noise structures to bound the range of expectations.

The custom min-norm inverse assumes white sensor noise (C = I). Validation that
only injects i.i.d. noise satisfies that assumption exactly, so its numbers are a
ceiling. This sweep holds SNR fixed and varies noise *structure* instead of level:

    white     satisfies C = I           -> optimistic bound
    spatial   correlated channels       -> isolates the covariance violation
    temporal  1/f spectrum              -> isolates the spectral violation
    colored   both                      -> realistic bound

All types run at the same positions with the same seeds, so the comparison is
paired: differences reflect noise structure, not trial luck.

Usage
-----
python scripts/run_noise_type_sweep.py \
    --pipeline-dir /path/to/pipeline/results \
    --output-dir /path/to/validation-tests/sweep_noise_type \
    --n-positions 6 --n-trials 20
"""
import argparse
import json
import platform
import subprocess
from datetime import datetime
from pathlib import Path

import numpy as np

from source_localization.validation.noise import NOISE_TYPES
from source_localization.validation.robustness import RobustnessTest


def git_revision() -> str:
    """Record the code version that produced these numbers."""
    try:
        return subprocess.check_output(
            ['git', 'rev-parse', 'HEAD'],
            cwd=Path(__file__).resolve().parent.parent,
            stderr=subprocess.DEVNULL
        ).decode().strip()
    except Exception:
        return 'unknown'


def summarize(errors: list) -> dict:
    """Distribution summary for one noise type's error samples."""
    arr = np.asarray(errors, dtype=float)
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return {'n': 0}
    return {
        'n': int(arr.size),
        'mean_mm': float(arr.mean()),
        'std_mm': float(arr.std()),
        'median_mm': float(np.median(arr)),
        'p25_mm': float(np.percentile(arr, 25)),
        'p75_mm': float(np.percentile(arr, 75)),
        'p95_mm': float(np.percentile(arr, 95)),
        'min_mm': float(arr.min()),
        'max_mm': float(arr.max()),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--pipeline-dir', required=True,
                        help='Pipeline output dir containing data/step*_forward*.pkl')
    parser.add_argument('--output-dir', required=True,
                        help='Where to write results (created if absent)')
    parser.add_argument('--snr-db', type=float, default=10.0,
                        help='Fixed SNR; only noise structure varies (default: 10)')
    parser.add_argument('--n-positions', type=int, default=6)
    parser.add_argument('--n-trials', type=int, default=20)
    parser.add_argument('--amplitude-nAm', type=float, default=50.0)
    parser.add_argument('--spatial-scale-mm', type=float, default=3.0)
    parser.add_argument('--temporal-exponent', type=float, default=1.0)
    parser.add_argument('--inverse-method', default='sLORETA')
    parser.add_argument('--inverse-snr', type=float, default=3.0)
    parser.add_argument('--noise-types', nargs='+', default=list(NOISE_TYPES),
                        choices=list(NOISE_TYPES))
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading pipeline from {args.pipeline_dir}")
    test = RobustnessTest.from_pipeline_dir(
        args.pipeline_dir,
        inverse_method=args.inverse_method,
        inverse_snr=args.inverse_snr,
        verbose=True
    )

    started = datetime.now()
    result = test.run_noise_type_test(
        noise_types=args.noise_types,
        snr_db=args.snr_db,
        n_positions=args.n_positions,
        n_trials=args.n_trials,
        amplitude_nAm=args.amplitude_nAm,
        spatial_scale_mm=args.spatial_scale_mm,
        temporal_exponent=args.temporal_exponent,
    )
    elapsed_s = (datetime.now() - started).total_seconds()

    per_type = {nt: summarize(result.errors[nt]) for nt in args.noise_types}

    # The headline: how much worse does realistic noise make things?
    degradation = {}
    baseline = per_type.get('white', {}).get('mean_mm')
    for nt, stats in per_type.items():
        if baseline and stats.get('mean_mm') is not None:
            degradation[nt] = {
                'delta_mm': float(stats['mean_mm'] - baseline),
                'ratio_vs_white': float(stats['mean_mm'] / baseline),
            }

    payload = {
        'sweep': 'noise_type',
        'purpose': 'Bound localization error between white (best case) and '
                   'colored (realistic) noise structure.',
        'provenance': {
            'timestamp': started.isoformat(),
            'elapsed_s': elapsed_s,
            'git_revision': git_revision(),
            'pipeline_dir': str(Path(args.pipeline_dir).resolve()),
            'hostname': platform.node(),
        },
        'parameters': {
            'snr_db': args.snr_db,
            'n_positions': args.n_positions,
            'n_trials': args.n_trials,
            'amplitude_nAm': args.amplitude_nAm,
            'spatial_scale_mm': args.spatial_scale_mm,
            'temporal_exponent': args.temporal_exponent,
            'inverse_method': args.inverse_method,
            'inverse_snr': args.inverse_snr,
            'n_sources': int(test.n_sources),
        },
        'error_mm_by_noise_type': per_type,
        'degradation_vs_white': degradation,
        'raw_errors_mm': {nt: [float(e) for e in result.errors[nt]]
                          for nt in args.noise_types},
    }

    results_path = output_dir / 'noise_type_sweep.json'
    with open(results_path, 'w') as f:
        json.dump(payload, f, indent=2)

    print("\n" + "=" * 68)
    print("NOISE-TYPE SWEEP — range of expectations")
    print("=" * 68)
    print(f"{'noise_type':<10} {'n':>5} {'mean':>8} {'median':>8} {'p95':>8}  vs white")
    print("-" * 68)
    for nt in args.noise_types:
        s = per_type[nt]
        if not s.get('n'):
            print(f"{nt:<10} {'0':>5}  (all trials failed)")
            continue
        ratio = degradation.get(nt, {}).get('ratio_vs_white')
        ratio_s = f"{ratio:.2f}x" if ratio is not None else "-"
        print(f"{nt:<10} {s['n']:>5} {s['mean_mm']:>7.2f}m {s['median_mm']:>7.2f}m "
              f"{s['p95_mm']:>7.2f}m  {ratio_s:>7}")
    print("-" * 68)
    if 'white' in per_type and 'colored' in per_type:
        w, c = per_type['white'].get('mean_mm'), per_type['colored'].get('mean_mm')
        if w and c:
            print(f"Expected range: {w:.2f} mm (best case, white/C=I) "
                  f"-> {c:.2f} mm (realistic, colored)")
    print(f"\nSaved: {results_path}")


if __name__ == '__main__':
    main()
