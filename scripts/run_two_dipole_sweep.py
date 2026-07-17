#!/usr/bin/env python
"""
Sweep per-dipole localization error for two simultaneous sources.

A single DC dipole is the friendliest case for a min-norm inverse. Under
superposition the smooth solution smears two sources together, worse as they
approach — and crossing that with noise structure gives the realistic lower
bound of the expectation range. This sweep varies both:

    separation:  how far apart the two sources are (smearing axis)
    noise_type:  white (C = I, optimistic) ... colored (realistic)

Per-dipole error is the localization error of each recovered peak after a 2x2
assignment to the two true positions.

Usage
-----
python scripts/run_two_dipole_sweep.py \
    --pipeline-dir /path/to/pipeline/results \
    --output-dir /path/to/validation-tests/sweep_two_dipole \
    --separations 2 4 6 8 --n-pairs 4 --n-trials 20
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
    try:
        return subprocess.check_output(
            ['git', 'rev-parse', 'HEAD'],
            cwd=Path(__file__).resolve().parent.parent,
            stderr=subprocess.DEVNULL
        ).decode().strip()
    except Exception:
        return 'unknown'


def summarize(errors: list) -> dict:
    arr = np.asarray(errors, dtype=float)
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return {'n': 0}
    return {
        'n': int(arr.size),
        'mean_mm': float(arr.mean()),
        'std_mm': float(arr.std()),
        'median_mm': float(np.median(arr)),
        'p95_mm': float(np.percentile(arr, 95)),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--pipeline-dir', required=True)
    parser.add_argument('--output-dir', required=True)
    parser.add_argument('--snr-db', type=float, default=10.0)
    parser.add_argument('--separations', type=float, nargs='+',
                        default=[2.0, 4.0, 6.0, 8.0])
    parser.add_argument('--n-pairs', type=int, default=4)
    parser.add_argument('--n-trials', type=int, default=20)
    parser.add_argument('--amplitude-nAm', type=float, default=50.0)
    parser.add_argument('--spatial-scale-mm', type=float, default=3.0)
    parser.add_argument('--temporal-exponent', type=float, default=1.0)
    parser.add_argument('--separation-tolerance-mm', type=float, default=1.0)
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
    results_by_type = test.run_two_dipole_test(
        separations_mm=args.separations,
        noise_types=args.noise_types,
        snr_db=args.snr_db,
        n_pairs=args.n_pairs,
        n_trials=args.n_trials,
        amplitude_nAm=args.amplitude_nAm,
        spatial_scale_mm=args.spatial_scale_mm,
        temporal_exponent=args.temporal_exponent,
        separation_tolerance_mm=args.separation_tolerance_mm,
    )
    elapsed_s = (datetime.now() - started).total_seconds()

    # Reshape to error[noise_type][separation] = distribution summary
    per_type = {}
    for nt, result in results_by_type.items():
        per_type[nt] = {
            str(sep): summarize(result.errors[sep])
            for sep in result.parameter_values
        }

    payload = {
        'sweep': 'two_dipole',
        'purpose': 'Realistic lower bound of the expectation range: two '
                   'correlated sources under varying separation and noise '
                   'structure.',
        'provenance': {
            'timestamp': started.isoformat(),
            'elapsed_s': elapsed_s,
            'git_revision': git_revision(),
            'pipeline_dir': str(Path(args.pipeline_dir).resolve()),
            'hostname': platform.node(),
        },
        'parameters': {
            'snr_db': args.snr_db,
            'separations_mm': args.separations,
            'n_pairs': args.n_pairs,
            'n_trials': args.n_trials,
            'amplitude_nAm': args.amplitude_nAm,
            'spatial_scale_mm': args.spatial_scale_mm,
            'temporal_exponent': args.temporal_exponent,
            'inverse_method': args.inverse_method,
            'inverse_snr': args.inverse_snr,
            'n_sources': int(test.n_sources),
        },
        'error_mm_by_noise_type_and_separation': per_type,
        'raw_errors_mm': {
            nt: {str(sep): [float(e) for e in result.errors[sep]]
                 for sep in result.parameter_values}
            for nt, result in results_by_type.items()
        },
    }

    results_path = output_dir / 'two_dipole_sweep.json'
    with open(results_path, 'w') as f:
        json.dump(payload, f, indent=2)

    print("\n" + "=" * 72)
    print("TWO-DIPOLE SWEEP — per-dipole error (mm) by separation and noise")
    print("=" * 72)
    realized = sorted(
        {float(s) for nt in per_type for s in per_type[nt]},
    )
    header = "noise_type  " + "".join(f"{s:>10.1f}mm" for s in realized)
    print(header)
    print("-" * len(header))
    for nt in args.noise_types:
        if nt not in per_type:
            continue
        row = f"{nt:<11}"
        for sep in realized:
            s = per_type[nt].get(str(sep), {})
            row += f"{s['mean_mm']:>10.2f}  " if s.get('n') else f"{'-':>10}  "
        print(row)
    print("-" * len(header))
    print("Reading: rightward = closer sources smear more; "
          "downward = noise color adds on top.")
    print(f"\nSaved: {results_path}")


if __name__ == '__main__':
    main()
