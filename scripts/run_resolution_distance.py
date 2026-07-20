#!/usr/bin/env python
"""
Phase 3: minimum resolvable separation between two sources — the limitation map.

For every source, places a second dipole at a range of separations and asks
whether the reconstruction shows two prominent lobes, scoring the identical
event on a position-matched single-source null. The reported resolution distance
is the smallest separation where the null-corrected excess
    P(two lobes | two sources) - P(two lobes | one source)
reaches 0.5 and stays there.

Noise-free by default: this is the *geometric* limit, the ceiling that noise can
only lower. Sources whose PSF is too flat for a peak to mean anything (see
ResolutionAnalysis.MIN_PEAK_CONTRAST) are reported separately rather than folded
into the statistics.

Usage
-----
python scripts/run_resolution_distance.py --pipeline-dir /path/to/results \
    --output-dir /path/to/validation-tests/resolution_distance
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


def make_figure(dmap, summ, out_path, method, ra):
    seps = dmap['separations_mm']
    depth = dmap['depth_mm']

    plt.rcParams.update({'font.size': 11, 'axes.spines.top': False,
                         'axes.spines.right': False, 'axes.edgecolor': '#888888',
                         'figure.facecolor': 'white', 'axes.facecolor': 'white'})
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(16, 4.8))

    # Panel A: detection vs separation, with the single-source null alongside.
    # If the two curves sit on top of each other, separation carries no signal.
    for label in DEPTH_LABELS:
        if label not in summ:
            continue
        lo, hi = summ[label]['depth_range_mm']
        m = (depth >= lo) & (depth <= hi)
        c = DEPTH_COLORS[label]
        ax1.plot(seps, np.nanmean(dmap['detect_fraction'][m], axis=0),
                 color=c, marker='o', ms=4, lw=2, label=f'{label} (two sources)')
        ax1.plot(seps, np.nanmean(dmap['null_fraction'][m], axis=0),
                 color=c, ls='--', lw=1.5, alpha=0.8)
    ax1.axhline(0.5, color='#bbbbbb', ls=':', lw=1)
    ax1.set_xlabel('separation between the two sources (mm)')
    ax1.set_ylabel('P(two lobes detected)')
    ax1.set_ylim(0, 1)
    ax1.set_title('A  Detection vs separation\n(dashed = one-source null)',
                  loc='left', fontweight='bold', fontsize=12)
    ax1.legend(frameon=False, fontsize=8)
    ax1.grid(True, color='#eee')

    # Panel B: the null itself — how often ONE source already looks like two.
    sc = ax2.scatter(dmap['positions_mm'][:, 0], dmap['positions_mm'][:, 1],
                     c=np.nanmean(dmap['null_fraction'], axis=1), s=55,
                     cmap='inferno', vmin=0, vmax=1, edgecolors='none')
    ax2.set_aspect('equal')
    ax2.set_xlabel('L-R (mm)'); ax2.set_ylabel('A-P / front-back (mm)')
    ax2.set_title('B  False-alarm rate:\none source that LOOKS like two',
                  loc='left', fontweight='bold', fontsize=12)
    fig.colorbar(sc, ax=ax2, label='P(two lobes | one source)', shrink=0.85)

    # Panel C: excess vs separation, pooled — the null-corrected signal.
    excess = dmap['detect_fraction'] - dmap['null_fraction']
    for label in DEPTH_LABELS:
        if label not in summ:
            continue
        lo, hi = summ[label]['depth_range_mm']
        m = (depth >= lo) & (depth <= hi)
        ax3.plot(seps, np.nanmean(excess[m], axis=0), color=DEPTH_COLORS[label],
                 marker='s', ms=4, lw=2, label=label)
    ax3.axhline(0.5, color='#cc0000', ls=':', lw=1.5)
    ax3.text(seps[0], 0.52, 'threshold for "resolved"', color='#cc0000', fontsize=8)
    ax3.axhline(0.0, color='#bbbbbb', lw=1)
    ax3.set_xlabel('separation (mm)')
    ax3.set_ylabel('excess = P(two|two) - P(two|one)')
    ax3.set_ylim(-0.4, 1.0)
    ax3.set_title('C  Null-corrected excess', loc='left',
                  fontweight='bold', fontsize=12)
    ax3.legend(frameon=False, fontsize=9)
    ax3.grid(True, color='#eee')

    n_res = int(np.sum(~np.isnan(dmap['resolution_distance_mm'])
                       & dmap['meaningful']))
    n_tot = len(dmap['resolution_distance_mm'])
    fig.suptitle(f'Two-source resolution limit ({method}, noise-free, 30-ch mouse EEG): '
                 f'only {n_res}/{n_tot} sources reach the threshold at ANY separation',
                 fontweight='bold', fontsize=13)
    fig.text(0.5, -0.03,
             'Noise-free — this is the geometric ceiling; noise can only lower it. '
             'Panel B is the practical warning: across much of the brain a SINGLE '
             'source already reconstructs as two lobes, so two blobs in a '
             'reconstruction are not evidence of two sources.',
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
    ap.add_argument('--snr-db', type=float, default=float('inf'),
                    help='simulation SNR; default inf (geometric limit)')
    ap.add_argument('--noise-type', default='white',
                    choices=['white', 'spatial', 'temporal', 'colored'])
    ap.add_argument('--n-trials', type=int, default=1,
                    help='noise realizations per pair (ignored when noise-free)')
    ap.add_argument('--n-partners', type=int, default=2)
    ap.add_argument('--n-sources', type=int, default=None,
                    help='subsample this many sources (default: all)')
    args = ap.parse_args()

    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)

    print(f"Loading pipeline from {args.pipeline_dir}")
    ra = ResolutionAnalysis.from_pipeline_dir(
        args.pipeline_dir, inverse_method=args.inverse_method,
        inverse_snr=args.inverse_snr, verbose=True)

    source_indices = None
    if args.n_sources is not None and args.n_sources < ra.n_sources:
        source_indices = np.linspace(0, ra.n_sources - 1, args.n_sources
                                     ).astype(int).tolist()

    started = datetime.now()
    dmap = ra.resolution_distance_map(
        source_indices=source_indices, snr_db=args.snr_db,
        noise_type=args.noise_type, n_trials=args.n_trials,
        n_partners=args.n_partners)
    summ = ra.summarize_distance_by_depth(dmap)
    elapsed = (datetime.now() - started).total_seconds()

    rd = dmap['resolution_distance_mm']
    ok = dmap['meaningful']
    payload = {
        'analysis': 'resolution_distance',
        'provenance': {'timestamp': started.isoformat(), 'elapsed_s': elapsed,
                       'git_revision': git_revision(),
                       'pipeline_dir': str(Path(args.pipeline_dir).resolve()),
                       'hostname': platform.node()},
        'method': args.inverse_method, 'inverse_snr': args.inverse_snr,
        'sim_snr_db': str(args.snr_db), 'noise_type': args.noise_type,
        'min_peak_contrast_gate': ra.MIN_PEAK_CONTRAST,
        'n_sources': int(len(rd)),
        'overall': {
            'n_resolved_at_any_separation': int(np.sum(~np.isnan(rd) & ok)),
            'n_flat_psf_excluded': int(np.sum(~ok)),
            'median_resolution_distance_mm': (
                float(np.median(rd[~np.isnan(rd) & ok]))
                if np.any(~np.isnan(rd) & ok) else None),
            'mean_null_fraction': float(np.nanmean(dmap['null_fraction'])),
        },
        'separations_mm': dmap['separations_mm'].tolist(),
        'by_depth': summ,
        'per_source': {
            'positions_mm': dmap['positions_mm'].tolist(),
            'depth_mm': dmap['depth_mm'].tolist(),
            'peak_contrast': dmap['peak_contrast'].tolist(),
            'meaningful': dmap['meaningful'].tolist(),
            'resolution_distance_mm': [None if np.isnan(v) else float(v) for v in rd],
            'detect_fraction': dmap['detect_fraction'].tolist(),
            'null_fraction': dmap['null_fraction'].tolist(),
        },
    }
    json_path = out / 'resolution_distance.json'
    with open(json_path, 'w') as f:
        json.dump(payload, f, indent=2)
    fig_path = out / 'resolution_distance.png'
    make_figure(dmap, summ, fig_path, args.inverse_method, ra)

    print("\n" + "=" * 74)
    print(f"TWO-SOURCE RESOLUTION DISTANCE ({args.inverse_method}, "
          f"snr={args.snr_db})")
    print("=" * 74)
    print(f"{'depth bin':>14}{'n':>5}{'flat PSF':>10}{'never res.':>12}"
          f"{'median RD':>12}{'max null':>10}")
    for lab in DEPTH_LABELS:
        if lab not in summ:
            continue
        s = summ[lab]
        med = ('--' if s['median_resolution_distance_mm'] is None
               else f"{s['median_resolution_distance_mm']:.1f}mm")
        mn = '--' if s['max_null_fraction'] is None else f"{s['max_null_fraction']:.2f}"
        print(f"{lab:>14}{s['n']:>5}{s['n_unreliable_flat_psf']:>10}"
              f"{s['n_never_resolved']:>12}{med:>12}{mn:>10}")
    o = payload['overall']
    print(f"\nResolved at ANY separation: {o['n_resolved_at_any_separation']}"
          f"/{payload['n_sources']} sources")
    print(f"Mean single-source false-alarm rate: {o['mean_null_fraction']:.2f}")
    print(f"elapsed {elapsed:.1f}s")
    print(f"Saved: {json_path}\nSaved: {fig_path}")


if __name__ == '__main__':
    main()
