#!/usr/bin/env python
"""
Phase 1: brain map of geometric resolution (noise-free), + figure.

Computes, for every source, the point-spread-function metrics:
  - peak localization error (PLE): is the reconstructed peak in the right place?
  - spatial dispersion (SD): how blurred is the reconstruction around it?
Writes JSON and a two-panel figure: PLE and SD vs depth, plus a top-down map.

Usage
-----
python scripts/run_resolution_map.py --pipeline-dir /path/to/results \
    --output-dir /path/to/validation-tests/resolution_map
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


def git_revision() -> str:
    try:
        return subprocess.check_output(
            ['git', 'rev-parse', 'HEAD'],
            cwd=Path(__file__).resolve().parent.parent,
            stderr=subprocess.DEVNULL).decode().strip()
    except Exception:
        return 'unknown'


def make_figure(m, summ, out_path, method):
    depth = m['depth_mm']
    ple = m['peak_localization_error_mm']
    sd = m['spatial_dispersion_mm']
    pos = m['positions_mm']

    plt.rcParams.update({'font.size': 11, 'axes.spines.top': False,
                         'axes.spines.right': False, 'axes.edgecolor': '#888888',
                         'figure.facecolor': 'white', 'axes.facecolor': 'white'})
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(16, 4.8))

    # Panel A: PLE and SD vs depth (the reliability story)
    order = np.argsort(depth)
    ax1.plot(depth[order], ple[order], color='#0072B2', lw=0, marker='o', ms=4,
             alpha=0.5, label='localization error (PLE)')
    ax1.plot(depth[order], sd[order], color='#D55E00', lw=0, marker='s', ms=4,
             alpha=0.5, label='spatial dispersion (SD)')
    # median lines per depth bin
    labels = list(summ.keys())
    centers = [np.mean(summ[l]['depth_range_mm']) for l in labels]
    ax1.plot(centers, [summ[l]['median_ple_mm'] for l in labels], color='#0072B2', lw=2.5)
    ax1.plot(centers, [summ[l]['median_sd_mm'] for l in labels], color='#D55E00', lw=2.5)
    ax1.set_xlabel('source depth (mm to nearest electrode)')
    ax1.set_ylabel('mm')
    ax1.set_title('A  Localization error vs blur, by depth', loc='left',
                  fontweight='bold', fontsize=12)
    ax1.legend(frameon=False, fontsize=9)
    ax1.grid(True, color='#eee')

    # Panel B: top-down map colored by SD (the blur footprint)
    sc = ax2.scatter(pos[:, 0], pos[:, 1], c=sd, s=55, cmap='viridis',
                     edgecolors='none')
    ax2.set_aspect('equal')
    ax2.set_xlabel('L-R (mm)'); ax2.set_ylabel('A-P / front-back (mm)')
    ax2.set_title('B  Spatial dispersion map (top-down)', loc='left',
                  fontweight='bold', fontsize=12)
    fig.colorbar(sc, ax=ax2, label='SD (mm)', shrink=0.85)

    # Panel C: top-down map colored by PLE
    sc2 = ax3.scatter(pos[:, 0], pos[:, 1], c=ple, s=55, cmap='magma',
                      edgecolors='none')
    ax3.set_aspect('equal')
    ax3.set_xlabel('L-R (mm)'); ax3.set_ylabel('A-P / front-back (mm)')
    ax3.set_title('C  Peak localization error map', loc='left',
                  fontweight='bold', fontsize=12)
    fig.colorbar(sc2, ax=ax3, label='PLE (mm)', shrink=0.85)

    fig.suptitle(f'Geometric resolution ({method}, noise-free, 30-ch mouse EEG): '
                 f'accurate peak (PLE~0), ~5 mm blur (SD)',
                 fontweight='bold', fontsize=13)
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
    args = ap.parse_args()

    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)

    print(f"Loading pipeline from {args.pipeline_dir}")
    ra = ResolutionAnalysis.from_pipeline_dir(
        args.pipeline_dir, inverse_method=args.inverse_method,
        inverse_snr=args.inverse_snr, verbose=True)

    started = datetime.now()
    m = ra.resolution_map()
    summ = ra.summarize_by_depth(m)
    elapsed = (datetime.now() - started).total_seconds()

    payload = {
        'analysis': 'resolution_map_noise_free',
        'provenance': {'timestamp': started.isoformat(), 'elapsed_s': elapsed,
                       'git_revision': git_revision(),
                       'pipeline_dir': str(Path(args.pipeline_dir).resolve()),
                       'hostname': platform.node()},
        'method': args.inverse_method, 'inverse_snr': args.inverse_snr,
        'n_sources': int(ra.n_sources),
        'overall': {
            'median_ple_mm': float(np.nanmedian(m['peak_localization_error_mm'])),
            'median_sd_mm': float(np.nanmedian(m['spatial_dispersion_mm'])),
        },
        'by_depth': summ,
        'per_source': {
            'positions_mm': m['positions_mm'].tolist(),
            'depth_mm': m['depth_mm'].tolist(),
            'peak_localization_error_mm': m['peak_localization_error_mm'].tolist(),
            'spatial_dispersion_mm': m['spatial_dispersion_mm'].tolist(),
        },
    }
    json_path = out / 'resolution_map.json'
    with open(json_path, 'w') as f:
        json.dump(payload, f, indent=2)
    fig_path = out / 'resolution_map.png'
    make_figure(m, summ, fig_path, args.inverse_method)

    print("\n" + "=" * 64)
    print(f"RESOLUTION MAP ({args.inverse_method}, noise-free)")
    print("=" * 64)
    print(f"{'depth bin':>14} {'n':>4} {'median PLE':>11} {'median SD':>10}")
    for lab in ['very_shallow', 'shallow', 'mid', 'deep']:
        if lab in summ:
            s = summ[lab]
            print(f"{lab:>14} {s['n']:>4} {s['median_ple_mm']:>8.2f}mm "
                  f"{s['median_sd_mm']:>8.2f}mm")
    print(f"\noverall: PLE {payload['overall']['median_ple_mm']:.2f}mm  "
          f"SD {payload['overall']['median_sd_mm']:.2f}mm")
    print(f"Saved: {json_path}\nSaved: {fig_path}")


if __name__ == '__main__':
    main()
