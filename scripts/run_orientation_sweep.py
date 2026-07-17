#!/usr/bin/env python
"""
Null-corrected resolvability by pair ORIENTATION and separation, with a figure.

Answers: can we tell two sources apart, and does it depend on their orientation
relative to the (dorsal) array? For each anatomical axis (front-back, lateral,
top-bottom, 45deg diagonals) and separation, measures how often two sources
produce two-lobe structure, minus a position-matched single-source null (how
often ONE source alone already looks like two — the false-alarm floor of the
smooth, ill-posed inverse). The reported signal is that excess.

Produces:
  orientation_sweep.json   full grid + records + null
  orientation_resolvability.png   two-panel figure

Usage
-----
python scripts/run_orientation_sweep.py \
    --pipeline-dir /path/to/pipeline/results \
    --output-dir /path/to/validation-tests/sweep_orientation \
    --separations 2 3 4 6 8 10 --n-pairs 5 --n-trials 12
"""
import argparse
import json
import platform
import subprocess
from datetime import datetime
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from source_localization.validation.noise import NOISE_TYPES
from source_localization.validation.robustness import RobustnessTest

# Okabe-Ito colorblind-safe categorical palette, assigned to axes in fixed order.
AXIS_STYLE = {
    'A-P (front-back)':        ('#0072B2', '-',  'o'),   # blue
    'L-R (lateral)':           ('#E69F00', '-',  's'),   # orange
    'D-V (top-bottom)':        ('#D55E00', '-',  '^'),   # vermillion
    '45deg sagittal (AP-DV)':  ('#009E73', '--', 'D'),   # bluish green
    '45deg coronal (LR-DV)':   ('#CC79A7', '--', 'v'),   # reddish purple
    'any (isotropic)':         ('#555555', ':',  'x'),   # neutral
}


def git_revision() -> str:
    try:
        return subprocess.check_output(
            ['git', 'rev-parse', 'HEAD'],
            cwd=Path(__file__).resolve().parent.parent,
            stderr=subprocess.DEVNULL).decode().strip()
    except Exception:
        return 'unknown'


def make_figure(result, noise_type, out_path, snr_db):
    """Two panels: (A) two-lobe rate vs null floor, (B) excess over null."""
    seps = result['separations_mm']
    grid = result['grid'][noise_type]
    # Plot the anatomically meaningful axes (skip isotropic control for clarity).
    axes_to_plot = [a for a in result['axes'] if a != 'any (isotropic)']

    plt.rcParams.update({
        'font.size': 11, 'axes.spines.top': False, 'axes.spines.right': False,
        'axes.edgecolor': '#888888', 'axes.linewidth': 0.8,
        'figure.facecolor': 'white', 'axes.facecolor': 'white',
    })
    fig, (axA, axB) = plt.subplots(1, 2, figsize=(13, 5.4))

    def series(axis, key):
        return [grid[axis][f"{s:.1f}"][key] for s in seps]

    # --- Panel A: raw two-lobe detection vs its null floor ---
    for axis in axes_to_plot:
        color, ls, mk = AXIS_STYLE[axis]
        p_two = [100 * v if v is not None else np.nan for v in series(axis, 'p_two')]
        p_null = [100 * v if v is not None else np.nan for v in series(axis, 'p_null')]
        axA.plot(seps, p_two, color=color, ls=ls, marker=mk, lw=2, ms=7,
                 label=axis)
        axA.plot(seps, p_null, color=color, ls=':', lw=1.3, alpha=0.55)
    axA.set_title('A  Two-lobe detection vs single-source null floor',
                  loc='left', fontweight='bold', fontsize=12)
    axA.set_xlabel('Source separation (mm)')
    axA.set_ylabel('Trials showing two lobes (%)')
    axA.set_ylim(0, 100)
    axA.text(0.98, 0.02,
             'solid = two sources    dotted = one-source null (false-alarm floor)',
             transform=axA.transAxes, ha='right', va='bottom', fontsize=9,
             color='#555555')
    axA.grid(True, color='#eeeeee', lw=0.8)

    # --- Panel B: excess over null (the real signal) ---
    for axis in axes_to_plot:
        color, ls, mk = AXIS_STYLE[axis]
        excess = [100 * v if v is not None else np.nan
                  for v in series(axis, 'excess')]
        axB.plot(seps, excess, color=color, ls=ls, marker=mk, lw=2.2, ms=7,
                 label=axis)
        # direct label at the right end
        last = next((i for i in range(len(seps) - 1, -1, -1)
                     if not np.isnan(excess[i])), None)
        if last is not None:
            axB.annotate(axis.split(' (')[0], (seps[last], excess[last]),
                         xytext=(5, 0), textcoords='offset points',
                         color=color, fontsize=9, va='center', fontweight='bold')
    axB.axhline(0, color='#888888', lw=1)
    thr = result['params']['excess_threshold'] * 100
    axB.axhline(thr, color='#888888', ls='--', lw=1)
    axB.text(seps[0], thr + 1.5, f'resolvable threshold (+{thr:.0f}%)',
             fontsize=9, color='#555555')
    axB.set_title('B  Excess over null  =  real resolvability signal',
                  loc='left', fontweight='bold', fontsize=12)
    axB.set_xlabel('Source separation (mm)')
    axB.set_ylabel('P(two lobes | two) − P(two lobes | one)  (%)')
    axB.grid(True, color='#eeeeee', lw=0.8)

    axA.legend(frameon=False, fontsize=9, loc='upper left', ncol=1)

    fig.suptitle(
        f'Two-source resolvability by orientation  —  {noise_type} noise, '
        f'SNR {snr_db:.0f} dB, 32-ch mouse EEG',
        fontweight='bold', fontsize=13, y=1.00)
    fig.text(0.5, -0.02,
             'Dorsal array: front-back (A-P) has the most room (~14 mm) and stays '
             'near the array; top-bottom (D-V) is shortest (~6 mm) and points away. '
             'Positive excess = two sources look different from one.',
             ha='center', fontsize=9, color='#555555', wrap=True)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close(fig)


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--pipeline-dir', required=True)
    p.add_argument('--output-dir', required=True)
    p.add_argument('--snr-db', type=float, default=10.0)
    p.add_argument('--separations', type=float, nargs='+',
                   default=[2.0, 3.0, 4.0, 6.0, 8.0, 10.0])
    p.add_argument('--n-pairs', type=int, default=5)
    p.add_argument('--n-trials', type=int, default=12)
    p.add_argument('--n-null-trials', type=int, default=None)
    p.add_argument('--angular-tolerance-deg', type=float, default=20.0)
    p.add_argument('--saddle-ratio', type=float, default=0.8)
    p.add_argument('--prominence-frac', type=float, default=0.5)
    p.add_argument('--inverse-method', default='sLORETA')
    p.add_argument('--inverse-snr', type=float, default=3.0)
    p.add_argument('--noise-types', nargs='+', default=['white', 'colored'],
                   choices=list(NOISE_TYPES))
    p.add_argument('--figure-noise', default='colored', choices=list(NOISE_TYPES))
    args = p.parse_args()

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading pipeline from {args.pipeline_dir}")
    test = RobustnessTest.from_pipeline_dir(
        args.pipeline_dir, inverse_method=args.inverse_method,
        inverse_snr=args.inverse_snr, verbose=True)

    started = datetime.now()
    result = test.run_orientation_resolvability_test(
        separations_mm=args.separations,
        noise_types=args.noise_types,
        snr_db=args.snr_db,
        n_pairs=args.n_pairs,
        n_trials=args.n_trials,
        n_null_trials=args.n_null_trials,
        angular_tolerance_deg=args.angular_tolerance_deg,
        saddle_ratio=args.saddle_ratio,
        prominence_frac=args.prominence_frac,
    )
    elapsed_s = (datetime.now() - started).total_seconds()

    payload = {
        'sweep': 'orientation_resolvability',
        'provenance': {
            'timestamp': started.isoformat(), 'elapsed_s': elapsed_s,
            'git_revision': git_revision(),
            'pipeline_dir': str(Path(args.pipeline_dir).resolve()),
            'hostname': platform.node(),
        },
        'params': {**result['params'], 'inverse_method': args.inverse_method,
                   'inverse_snr': args.inverse_snr, 'n_sources': int(test.n_sources)},
        'axes': result['axes'], 'separations_mm': result['separations_mm'],
        'noise_types': result['noise_types'],
        'grid': result['grid'],
        'threshold_separation_mm': result['threshold_separation_mm'],
        'null_rate': result['null_rate'],
        'records': result['records'],
    }
    json_path = out_dir / 'orientation_sweep.json'
    with open(json_path, 'w') as f:
        json.dump(payload, f, indent=2)

    fig_noise = args.figure_noise if args.figure_noise in result['noise_types'] \
        else result['noise_types'][-1]
    fig_path = out_dir / 'orientation_resolvability.png'
    make_figure(result, fig_noise, fig_path, args.snr_db)

    # Console summary of thresholds
    print("\n" + "=" * 74)
    print(f"ORIENTATION RESOLVABILITY — threshold separation (excess >= "
          f"{result['params']['excess_threshold']*100:.0f}%)")
    print("=" * 74)
    for nt in result['noise_types']:
        print(f"\nnoise = {nt}")
        for axis in result['axes']:
            t = result['threshold_separation_mm'][nt][axis]
            tstr = f"{t:.1f} mm" if t is not None else ">max (never reliably resolvable)"
            # peak excess for context
            excesses = [result['grid'][nt][axis][f"{s:.1f}"]['excess']
                        for s in result['separations_mm']]
            excesses = [e for e in excesses if e is not None]
            peak = max(excesses) * 100 if excesses else float('nan')
            print(f"  {axis:<26} threshold {tstr:<32} peak excess {peak:+.0f}%")
    print(f"\nSaved: {json_path}")
    print(f"Saved: {fig_path}")


if __name__ == '__main__':
    main()
