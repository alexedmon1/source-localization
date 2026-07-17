#!/usr/bin/env python
"""
Sweep the probability of RESOLVING two sources as two (not one).

The two-dipole localization sweep asks "how accurately are two known sources
placed?". This asks the prior, and for close sources more meaningful, question:
"can we tell there are two at all?" A pair is *resolved* when the reconstruction
shows two distinct peaks with a real dip between them (a Rayleigh-style saddle),
both matching the true sources. The headline output is a **threshold
separation** per depth: the closest two sources can be while still resolvable.

    depth:       distance to nearest electrode (very_shallow ... deep)
    separation:  how far apart the two sources are
    noise_type:  white (optimistic) ... colored (realistic)

Localization error is retained as a secondary field.

Usage
-----
python scripts/run_resolvability_sweep.py \
    --pipeline-dir /path/to/pipeline/results \
    --output-dir /path/to/validation-tests/sweep_resolvability \
    --separations 1 2 3 4 5 6 8 --n-pairs 4 --n-trials 20
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


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--pipeline-dir', required=True)
    parser.add_argument('--output-dir', required=True)
    parser.add_argument('--snr-db', type=float, default=10.0)
    parser.add_argument('--separations', type=float, nargs='+',
                        default=[1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 8.0])
    parser.add_argument('--n-pairs', type=int, default=4)
    parser.add_argument('--n-trials', type=int, default=20)
    parser.add_argument('--amplitude-nAm', type=float, default=50.0)
    parser.add_argument('--spatial-scale-mm', type=float, default=3.0)
    parser.add_argument('--temporal-exponent', type=float, default=1.0)
    parser.add_argument('--separation-tolerance-mm', type=float, default=0.75)
    parser.add_argument('--depth-tolerance-mm', type=float, default=1.0)
    parser.add_argument('--saddle-ratio', type=float, default=0.8,
                        help='Rayleigh-style dip threshold (~0.81 classic)')
    parser.add_argument('--prominence-frac', type=float, default=0.5,
                        help='2nd peak must be >= this fraction of the 1st peak '
                             '(suppresses noise-induced false positives)')
    parser.add_argument('--max-match-mm', type=float, default=None,
                        help='Optional loose sanity cap on peak-to-source distance '
                             '(off by default; correspondence is distinct-nearest-source)')
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
    result = test.run_resolvability_test(
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
        saddle_ratio=args.saddle_ratio,
        prominence_frac=args.prominence_frac,
        max_match_mm=args.max_match_mm,
    )
    elapsed_s = (datetime.now() - started).total_seconds()

    payload = {
        'sweep': 'resolvability',
        'purpose': 'Can we tell two sources apart? Threshold separation for '
                   'resolving two sources as two, by depth and noise structure.',
        'provenance': {
            'timestamp': started.isoformat(),
            'elapsed_s': elapsed_s,
            'git_revision': git_revision(),
            'pipeline_dir': str(Path(args.pipeline_dir).resolve()),
            'hostname': platform.node(),
        },
        'parameters': {
            'snr_db': args.snr_db,
            'separations_mm': result['separations_mm'],
            'depth_bins': result['depth_bins'],
            'depth_bin_edges_mm': result['depth_bin_edges_mm'],
            'n_pairs': args.n_pairs,
            'n_trials': args.n_trials,
            'inverse_method': args.inverse_method,
            'inverse_snr': args.inverse_snr,
            'n_sources': int(test.n_sources),
        },
        'detector_params': result['detector_params'],
        'resolution_probability': result['resolution_probability'],
        'threshold_separation_mm': result['threshold_separation_mm'],
        'records': result['records'],
    }

    results_path = output_dir / 'resolvability_sweep.json'
    with open(results_path, 'w') as f:
        json.dump(payload, f, indent=2)

    seps = result['separations_mm']
    depth_bins = result['depth_bins']
    edges = result['depth_bin_edges_mm']
    prob = result['resolution_probability']
    thresh = result['threshold_separation_mm']

    print("\n" + "=" * 80)
    print("RESOLVABILITY SWEEP — % of trials resolving two sources as two")
    print("=" * 80)
    for nt in args.noise_types:
        if nt not in prob:
            continue
        print(f"\nnoise = {nt}")
        header = f"  {'depth bin':<14}" + "".join(f"{s:>6.1f}mm" for s in seps) + "   threshold"
        print(header)
        print("  " + "-" * (len(header) - 2))
        for db in depth_bins:
            lo, hi = edges.get(db, (float('nan'), float('nan')))
            row = f"  {db:<14}"
            for s in seps:
                p = prob[nt][db].get(f"{s:.1f}")
                row += f"{p*100:>5.0f}% " if p is not None else f"{'-':>6}"
            t = thresh[nt][db]
            tstr = f"{t:.1f} mm" if t is not None else ">max"
            print(f"{row}   {tstr:>8}  [{lo:.1f}-{hi:.1f} mm]")
    print("\n" + "-" * 80)
    print("threshold = smallest separation resolved in >=50% of trials. "
          "Deeper sources need wider separation.")
    print(f"\nSaved: {results_path}")


if __name__ == '__main__':
    main()
