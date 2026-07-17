#!/usr/bin/env python
"""
Sweep per-dipole localization error for two simultaneous sources.

A single DC dipole is the friendliest case for a min-norm inverse. Under
superposition the smooth solution smears two sources together — worse as they
approach, and worse as they get deeper (further from the electrodes). Crossing
those with noise structure gives the realistic lower bound of the expectation
range. This sweep varies three axes:

    depth:       distance to nearest electrode (very_shallow ... deep)
    separation:  how far apart the two sources are (smearing axis)
    noise_type:  white (C = I, optimistic) ... colored (realistic)

Per-dipole error is the localization error of each recovered peak after a 2x2
assignment to the two true positions. Both dipoles of a pair sit at matched
depth, so each pair has a well-defined depth.

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
    parser.add_argument('--depth-tolerance-mm', type=float, default=1.0,
                        help='Max depth difference between the two dipoles of a pair')
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
    result = test.run_two_dipole_test(
        separations_mm=args.separations,
        noise_types=args.noise_types,
        snr_db=args.snr_db,
        n_pairs=args.n_pairs,
        n_trials=args.n_trials,
        amplitude_nAm=args.amplitude_nAm,
        spatial_scale_mm=args.spatial_scale_mm,
        temporal_exponent=args.temporal_exponent,
        separation_tolerance_mm=args.separation_tolerance_mm,
        depth_tolerance_mm=args.depth_tolerance_mm,
    )
    elapsed_s = (datetime.now() - started).total_seconds()

    records = result['records']
    depth_bins = result['depth_bins']
    separations = result['separations_mm']

    def cell(nt, db, sep):
        return [r['error_mm'] for r in records
                if r['noise_type'] == nt and r['depth_bin'] == db
                and r['separation_mm'] == sep]

    # error[noise_type][depth_bin][separation] = distribution summary
    grid = {
        nt: {db: {f"{sep:.1f}": summarize(cell(nt, db, sep)) for sep in separations}
             for db in depth_bins}
        for nt in args.noise_types
    }

    payload = {
        'sweep': 'two_dipole',
        'purpose': 'Realistic lower bound of the expectation range: two sources '
                   'under varying depth, separation, and noise structure.',
        'provenance': {
            'timestamp': started.isoformat(),
            'elapsed_s': elapsed_s,
            'git_revision': git_revision(),
            'pipeline_dir': str(Path(args.pipeline_dir).resolve()),
            'hostname': platform.node(),
        },
        'parameters': {
            'snr_db': args.snr_db,
            'separations_mm': separations,
            'depth_bins': depth_bins,
            'depth_bin_edges_mm': result['depth_bin_edges_mm'],
            'n_pairs': args.n_pairs,
            'n_trials': args.n_trials,
            'amplitude_nAm': args.amplitude_nAm,
            'spatial_scale_mm': args.spatial_scale_mm,
            'temporal_exponent': args.temporal_exponent,
            'depth_tolerance_mm': args.depth_tolerance_mm,
            'inverse_method': args.inverse_method,
            'inverse_snr': args.inverse_snr,
            'n_sources': int(test.n_sources),
        },
        'error_mm_by_noise_depth_separation': grid,
        'records': records,
    }

    results_path = output_dir / 'two_dipole_sweep.json'
    with open(results_path, 'w') as f:
        json.dump(payload, f, indent=2)

    edges = result['depth_bin_edges_mm']
    print("\n" + "=" * 78)
    print("TWO-DIPOLE SWEEP — per-dipole error (mm) by depth x separation x noise")
    print("=" * 78)
    for nt in args.noise_types:
        if nt not in grid:
            continue
        print(f"\nnoise = {nt}")
        header = f"  {'depth bin':<14}" + "".join(f"{s:>8.1f}mm" for s in separations)
        print(header)
        print("  " + "-" * (len(header) - 2))
        for db in depth_bins:
            lo, hi = edges.get(db, (float('nan'), float('nan')))
            row = f"  {db:<14}"
            for sep in separations:
                s = grid[nt][db][f"{sep:.1f}"]
                row += f"{s['mean_mm']:>8.2f}  " if s.get('n') else f"{'-':>8}  "
            print(f"{row}   [{lo:.1f}-{hi:.1f} mm deep]")
    print("\n" + "-" * 78)
    print("Reading: down = deeper (further from electrodes) resolves worse; "
          "right = closer sources smear more.")
    print(f"\nSaved: {results_path}")


if __name__ == '__main__':
    main()
