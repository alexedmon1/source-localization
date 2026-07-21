#!/usr/bin/env python
"""
Threshold-based separability: at what threshold do two sources read as two blobs?

Takes everything above X% of the reconstruction's peak and counts connected
blobs — the way a source map is actually read. Reports, per depth:

  - blob radius vs threshold (the pedestal-free replacement for spatial
    dispersion, which is pinned near its own ceiling and barely varies);
  - P(separated) over the (separation x threshold) plane;
  - and the reason separation fails, which is not always "too close":
      merged    = both sources visible, one blob (a true resolution limit)
      dominated = the weaker source fell below threshold and is invisible
                  (raising the threshold makes this worse, not better)

The threshold is a fraction of the peak of a normalized activation map — a
relative plausibility, not a calibrated probability.

Usage
-----
python scripts/run_separability.py --pipeline-dir /path/to/results \
    --output-dir /path/to/validation-tests/separability
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

from source_localization.validation.separability import SeparabilityAnalysis

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


def make_figure(blob, sweep, summ, out_path, method):
    thr = sweep['thresholds']
    seps = sweep['separations_mm']
    depth_b = blob['depth_mm']

    plt.rcParams.update({'font.size': 10, 'axes.spines.top': False,
                         'axes.spines.right': False, 'axes.edgecolor': '#888888',
                         'figure.facecolor': 'white', 'axes.facecolor': 'white'})
    fig = plt.figure(figsize=(17, 8.2))
    gs = fig.add_gridspec(2, 4, height_ratios=[1, 1], hspace=0.42, wspace=0.3)

    # --- Panel A: blob radius vs threshold, per depth -----------------------
    axA = fig.add_subplot(gs[0, 0])
    for label in DEPTH_LABELS:
        if label not in summ:
            continue
        lo, hi = summ[label]['depth_range_mm']
        m = (depth_b >= lo) & (depth_b <= hi)
        axA.plot(thr * 100, np.nanmedian(blob['rms_radius_mm'][m], axis=0),
                 color=DEPTH_COLORS[label], marker='o', ms=4, lw=2, label=label)
    axA.set_xlabel('threshold (% of peak)')
    axA.set_ylabel('blob RMS radius (mm)')
    axA.set_title('A  Blob size vs threshold\n(replaces spatial dispersion)',
                  loc='left', fontweight='bold', fontsize=11)
    axA.legend(frameon=False, fontsize=8)
    axA.grid(True, color='#eee')

    # --- Panel B: implied minimum separation (sum of two blob radii) --------
    axB = fig.add_subplot(gs[0, 1])
    for label in DEPTH_LABELS:
        if label not in summ:
            continue
        lo, hi = summ[label]['depth_range_mm']
        m = (depth_b >= lo) & (depth_b <= hi)
        axB.plot(thr * 100, 2 * np.nanmedian(blob['rms_radius_mm'][m], axis=0),
                 color=DEPTH_COLORS[label], marker='s', ms=4, lw=2, label=label)
    axB.axhline(14.3, color='#cc0000', ls=':', lw=1.5)
    axB.text(16, 14.8, 'brain extent (A-P)', color='#cc0000', fontsize=8)
    axB.set_xlabel('threshold (% of peak)')
    axB.set_ylabel('2 x blob radius (mm)')
    axB.set_title('B  Implied minimum separation\n(blobs stop overlapping)',
                  loc='left', fontweight='bold', fontsize=11)
    axB.grid(True, color='#eee')

    # --- Panel C: failure-mode decomposition, pooled -----------------------
    axC = fig.add_subplot(gs[0, 2])
    sep_i = min(range(len(seps)), key=lambda k: abs(seps[k] - 6.0))
    for key, color, name in (('p_separated', '#009E73', 'separated'),
                             ('p_merged', '#E69F00', 'merged (one blob)'),
                             ('p_dominated', '#D55E00', 'dominated (weaker lost)')):
        axC.plot(thr * 100, np.nanmean(sweep[key][:, sep_i, :], axis=0),
                 color=color, marker='o', ms=4, lw=2, label=name)
    axC.set_xlabel('threshold (% of peak)')
    axC.set_ylabel('fraction of pairs')
    axC.set_ylim(0, 1)
    axC.set_title(f'C  Why separation fails\n(at {seps[sep_i]:.0f} mm, all depths)',
                  loc='left', fontweight='bold', fontsize=11)
    axC.legend(frameon=False, fontsize=8)
    axC.grid(True, color='#eee')

    # --- Panel D: best achievable separability over threshold --------------
    axD = fig.add_subplot(gs[0, 3])
    for label in DEPTH_LABELS:
        if label not in summ:
            continue
        best = np.nanmax(np.asarray(summ[label]['p_separated']), axis=1)
        axD.plot(seps, best, color=DEPTH_COLORS[label], marker='o', ms=4,
                 lw=2, label=label)
    axD.axhline(0.5, color='#bbbbbb', ls=':', lw=1)
    axD.set_xlabel('separation (mm)')
    axD.set_ylabel('P(separated) at best threshold')
    axD.set_ylim(0, 1)
    axD.set_title('D  Best case over all\nthresholds', loc='left',
                  fontweight='bold', fontsize=11)
    axD.legend(frameon=False, fontsize=8)
    axD.grid(True, color='#eee')

    # --- Bottom row: P(separated) heatmap per depth ------------------------
    for k, label in enumerate(DEPTH_LABELS):
        ax = fig.add_subplot(gs[1, k])
        if label not in summ:
            continue
        grid = np.asarray(summ[label]['p_separated'])   # (S, T)
        im = ax.imshow(grid.T, origin='lower', aspect='auto', cmap='viridis',
                       vmin=0, vmax=1,
                       extent=[seps[0], seps[-1], thr[0] * 100, thr[-1] * 100])
        ax.set_xlabel('separation (mm)')
        if k == 0:
            ax.set_ylabel('threshold (% of peak)')
        ax.set_title(f'{label}', loc='left', fontweight='bold', fontsize=11)
        if k == len(DEPTH_LABELS) - 1:
            fig.colorbar(im, ax=ax, label='P(separated)', shrink=0.9)

    fig.suptitle(f'Threshold-based separability ({method}, noise-free, 30-ch mouse EEG): '
                 f'thresholding removes the leakage pedestal that spatial dispersion could not',
                 fontweight='bold', fontsize=13)
    fig.text(0.5, 0.005,
             'Threshold is a fraction of the map peak — a relative plausibility, '
             'not a calibrated probability. "Dominated" means the weaker source '
             'fell below threshold: for those pairs a HIGHER threshold hurts.',
             ha='center', fontsize=9, style='italic', color='#555555')
    fig.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--pipeline-dir', required=True)
    ap.add_argument('--output-dir', required=True)
    ap.add_argument('--inverse-method', default='sLORETA')
    ap.add_argument('--inverse-snr', type=float, default=3.0)
    ap.add_argument('--snr-db', type=float, default=float('inf'))
    ap.add_argument('--noise-type', default='white',
                    choices=['white', 'spatial', 'temporal', 'colored'])
    ap.add_argument('--n-trials', type=int, default=1)
    ap.add_argument('--n-sources', type=int, default=None)
    ap.add_argument('--thresholds', type=float, nargs='+', default=None)
    ap.add_argument('--depth-match-mm', type=float, default=None,
                    help='restrict pairs to comparable depth (e.g. 0.4). Without '
                         'it most pairs fail as "dominated" from gain imbalance '
                         'rather than proximity')
    args = ap.parse_args()

    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)

    print(f"Loading pipeline from {args.pipeline_dir}")
    sa = SeparabilityAnalysis.from_pipeline_dir(
        args.pipeline_dir, inverse_method=args.inverse_method,
        inverse_snr=args.inverse_snr, verbose=True)

    idx = None
    if args.n_sources is not None and args.n_sources < sa.n_sources:
        idx = np.linspace(0, sa.n_sources - 1, args.n_sources).astype(int).tolist()
    thresholds = args.thresholds or list(sa.DEFAULT_THRESHOLDS)

    started = datetime.now()
    print("Computing single-source blob extents...")
    blob = sa.blob_extent_map(thresholds=thresholds, source_indices=idx)
    print("Sweeping two-source separability...")
    sweep = sa.separability_sweep(
        source_indices=idx, thresholds=thresholds, snr_db=args.snr_db,
        noise_type=args.noise_type, n_trials=args.n_trials,
        depth_match_mm=args.depth_match_mm)
    summ = sa.summarize_separability_by_depth(sweep)
    elapsed = (datetime.now() - started).total_seconds()

    payload = {
        'analysis': 'threshold_separability',
        'provenance': {'timestamp': started.isoformat(), 'elapsed_s': elapsed,
                       'git_revision': git_revision(),
                       'pipeline_dir': str(Path(args.pipeline_dir).resolve()),
                       'hostname': platform.node()},
        'method': args.inverse_method, 'sim_snr_db': str(args.snr_db),
        'noise_type': args.noise_type,
        'depth_match_mm': args.depth_match_mm,
        'thresholds': [float(t) for t in thresholds],
        'separations_mm': sweep['separations_mm'].tolist(),
        'n_sources': int(len(sweep['source_idx'])),
        'by_depth': summ,
        'blob_extent': {
            'depth_mm': blob['depth_mm'].tolist(),
            'positions_mm': blob['positions_mm'].tolist(),
            'rms_radius_mm': blob['rms_radius_mm'].tolist(),
            'max_radius_mm': blob['max_radius_mm'].tolist(),
            'volume_mm3': blob['volume_mm3'].tolist(),
        },
    }
    json_path = out / 'separability.json'
    with open(json_path, 'w') as f:
        json.dump(payload, f, indent=2)
    fig_path = out / 'separability.png'
    make_figure(blob, sweep, summ, fig_path, args.inverse_method)

    depth_b = blob['depth_mm']
    print("\n" + "=" * 78)
    print("BLOB RMS RADIUS (mm) — pedestal-free 'dispersion'")
    print("=" * 78)
    print(f"{'depth bin':>14}" + ''.join(f"{f'T={t:.0%}':>9}" for t in thresholds))
    for lab in DEPTH_LABELS:
        if lab not in summ:
            continue
        lo, hi = summ[lab]['depth_range_mm']
        m = (depth_b >= lo) & (depth_b <= hi)
        print(f"{lab:>14}" + ''.join(
            f"{v:>9.2f}" for v in np.nanmedian(blob['rms_radius_mm'][m], axis=0)))

    print("\n" + "=" * 78)
    print("P(SEPARATED) at the best threshold, by separation")
    print("=" * 78)
    print(f"{'depth bin':>14}" + ''.join(
        f"{f'{s:.0f}mm':>8}" for s in sweep['separations_mm']))
    for lab in DEPTH_LABELS:
        if lab not in summ:
            continue
        best = np.nanmax(np.asarray(summ[lab]['p_separated']), axis=1)
        print(f"{lab:>14}" + ''.join(f"{v:>8.2f}" for v in best))
    print(f"\nelapsed {elapsed:.1f}s")
    print(f"Saved: {json_path}\nSaved: {fig_path}")


if __name__ == '__main__':
    main()
