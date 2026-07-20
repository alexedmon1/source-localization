#!/usr/bin/env python
"""
Phase 2: how resolution degrades with SNR, on top of the Phase-1 geometric map.

For every source, at each SNR level, recomputes the two Phase-1 metrics:
  - peak localization error (PLE): median + IQR over noise realizations
  - spatial dispersion (SD): blur of the trial-averaged reconstruction
The noise-free (snr=inf) rung is the Phase-1 geometric anchor, so the figure
reads as "here is the geometry, and here is what noise costs you on top."

Usage
-----
python scripts/run_resolution_vs_snr.py --pipeline-dir /path/to/results \
    --output-dir /path/to/validation-tests/resolution_vs_snr
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

from source_localization.validation.resolution import ResolutionAnalysis

DEPTH_LABELS = ['very_shallow', 'shallow', 'mid', 'deep']
DEPTH_COLORS = {'very_shallow': '#0072B2', 'shallow': '#009E73',
                'mid': '#E69F00', 'deep': '#D55E00'}


def git_revision() -> str:
    try:
        return subprocess.check_output(
            ['git', 'rev-parse', 'HEAD'],
            cwd=Path(__file__).resolve().parent.parent,
            stderr=subprocess.DEVNULL).decode().strip()
    except Exception:
        return 'unknown'


def make_figure(sweep, summ, out_path, method, noise_type):
    snrs = sweep['snr_values']
    finite = np.isfinite(snrs)
    x = snrs[finite]
    # Plot inf as one step beyond the highest finite SNR, tick-labeled "noise-free".
    x_inf = x.max() + 10 if finite.any() else 0
    x_all = np.append(x, x_inf)

    plt.rcParams.update({'font.size': 11, 'axes.spines.top': False,
                         'axes.spines.right': False, 'axes.edgecolor': '#888888',
                         'figure.facecolor': 'white', 'axes.facecolor': 'white'})
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))

    for label in DEPTH_LABELS:
        if label not in summ:
            continue
        by_snr = summ[label]['by_snr']
        c = DEPTH_COLORS[label]
        ple = [by_snr[str(s)]['median_ple_mm'] for s in snrs]
        sd = [by_snr[str(s)]['median_sd_mm'] for s in snrs]
        rng = summ[label]['depth_range_mm']
        leg = f"{label} ({rng[0]:.1f}-{rng[1]:.1f} mm)"
        ax1.plot(x_all, ple, color=c, marker='o', ms=5, lw=2, label=leg)
        ax2.plot(x_all, sd, color=c, marker='s', ms=5, lw=2, label=leg)

    for ax, ylab, title in (
            (ax1, 'median PLE (mm)', 'A  Localization error vs SNR'),
            (ax2, 'median SD (mm)', 'B  Blur (spatial dispersion) vs SNR')):
        ax.axvline(x_inf - 5, color='#bbbbbb', ls=':', lw=1)
        ax.set_xticks(list(x_all))
        ax.set_xticklabels([f"{int(v)}" for v in x] + ['noise-\nfree'])
        ax.set_xlabel('SNR (dB)')
        ax.set_ylabel(ylab)
        ax.set_title(title, loc='left', fontweight='bold', fontsize=12)
        ax.grid(True, color='#eee')
        ax.legend(frameon=False, fontsize=9, title='depth bin',
                  title_fontsize=9)

    fig.suptitle(f'Resolution vs SNR ({method}, {noise_type} noise, 30-ch mouse EEG): '
                 f'geometry sets the floor, noise adds on top',
                 fontweight='bold', fontsize=13)
    # The deep bin's PSF is nearly flat (peak contrast ~1.3), so its argmax --
    # and hence its PLE -- is close to arbitrary. Say so on the figure rather
    # than let the non-monotonic deep curve read as a real SNR effect.
    fig.text(0.5, -0.02,
             'Deep sources have a nearly flat PSF (peak contrast ~1.3), so their '
             'peak location — and the PLE curve above — is largely arbitrary; '
             'read the deep bin as "unlocalizable", not as an SNR trend.',
             ha='center', fontsize=9, style='italic', color='#555555')
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--pipeline-dir', required=True)
    ap.add_argument('--output-dir', required=True)
    ap.add_argument('--inverse-method', default='sLORETA')
    ap.add_argument('--inverse-snr', type=float, default=3.0)
    ap.add_argument('--n-trials', type=int, default=20,
                    help='noise realizations per source per finite SNR')
    ap.add_argument('--noise-type', default='colored',
                    choices=['white', 'spatial', 'temporal', 'colored'],
                    help="'colored' is the realistic bound; 'white' matches the "
                         "inverse's C=I assumption")
    ap.add_argument('--snr-values', type=float, nargs='+', default=None,
                    help='finite SNR levels in dB (noise-free anchor always added)')
    ap.add_argument('--n-sources', type=int, default=None,
                    help='subsample this many sources (default: all)')
    args = ap.parse_args()

    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)

    print(f"Loading pipeline from {args.pipeline_dir}")
    ra = ResolutionAnalysis.from_pipeline_dir(
        args.pipeline_dir, inverse_method=args.inverse_method,
        inverse_snr=args.inverse_snr, verbose=True)

    snr_values = (list(args.snr_values) + [np.inf] if args.snr_values
                  else list(ra.DEFAULT_SNR_VALUES))
    source_indices = None
    if args.n_sources is not None and args.n_sources < ra.n_sources:
        source_indices = np.linspace(0, ra.n_sources - 1, args.n_sources
                                     ).astype(int).tolist()

    started = datetime.now()
    sweep = ra.resolution_vs_snr(
        snr_values=snr_values, source_indices=source_indices,
        n_trials=args.n_trials, noise_type=args.noise_type)
    summ = ra.summarize_vs_snr_by_depth(sweep)
    elapsed = (datetime.now() - started).total_seconds()

    payload = {
        'analysis': 'resolution_vs_snr',
        'provenance': {'timestamp': started.isoformat(), 'elapsed_s': elapsed,
                       'git_revision': git_revision(),
                       'pipeline_dir': str(Path(args.pipeline_dir).resolve()),
                       'hostname': platform.node()},
        'method': args.inverse_method, 'inverse_snr': args.inverse_snr,
        'noise_type': args.noise_type, 'n_trials': args.n_trials,
        'snr_values': [str(s) for s in sweep['snr_values']],
        'n_sources': int(len(sweep['source_idx'])),
        'by_depth': summ,
        'per_source': {
            'positions_mm': sweep['positions_mm'].tolist(),
            'depth_mm': sweep['depth_mm'].tolist(),
            'ple_median_mm': sweep['ple_median_mm'].tolist(),
            'ple_iqr_mm': sweep['ple_iqr_mm'].tolist(),
            'spatial_dispersion_mm': sweep['spatial_dispersion_mm'].tolist(),
            'peak_contrast': sweep['peak_contrast'].tolist(),
        },
    }
    json_path = out / 'resolution_vs_snr.json'
    with open(json_path, 'w') as f:
        json.dump(payload, f, indent=2)
    fig_path = out / 'resolution_vs_snr.png'
    make_figure(sweep, summ, fig_path, args.inverse_method, args.noise_type)

    print("\n" + "=" * 72)
    print(f"RESOLUTION vs SNR ({args.inverse_method}, {args.noise_type} noise, "
          f"{args.n_trials} trials)")
    print("=" * 72)
    hdr = ''.join(f"{('inf' if not np.isfinite(s) else f'{s:+.0f}dB'):>12}"
                  for s in sweep['snr_values'])
    print(f"{'depth bin':>14}{hdr}")
    for metric, name, unit in (('median_ple_mm', 'PLE', 'mm'),
                               ('median_sd_mm', 'SD', 'mm'),
                               ('median_peak_contrast', 'peak contrast',
                                'x over background; ~1 = flat PSF, peak arbitrary')):
        print(f"  -- median {name} ({unit}) --")
        for lab in DEPTH_LABELS:
            if lab not in summ:
                continue
            row = ''.join(f"{summ[lab]['by_snr'][str(s)][metric]:>12.2f}"
                          for s in sweep['snr_values'])
            print(f"{lab:>14}{row}")
    print(f"\nelapsed {elapsed:.1f}s")
    print(f"Saved: {json_path}\nSaved: {fig_path}")


if __name__ == '__main__':
    main()
