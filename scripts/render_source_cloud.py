#!/usr/bin/env python
"""
Render the reconstruction as a 3D cloud: where does the inverse think the source is?

Simulates one or two dipoles through the real pipeline operator and volume-renders
the reconstructed activation, with the true dipole position(s) marked in cyan and
the reconstruction peak marked with a green cross. Two nearby sources produce the
characteristic picture: two fuzzy balls joined by a faint bridge.

The cloud is activation normalized to its peak — a relative plausibility map, not
a calibrated probability.

Usage
-----
# two shallow sources 6 mm apart (the "two fuzzy balls" case)
python scripts/render_source_cloud.py --pipeline-dir /path/to/results \
    --output-dir /path/to/figs --mode two --separation 6 --depth shallow

# single source, for the blur-only picture
python scripts/render_source_cloud.py --pipeline-dir /path/to/results \
    --output-dir /path/to/figs --mode one --depth very_shallow

# a separation series showing merge -> resolve
python scripts/render_source_cloud.py --pipeline-dir /path/to/results \
    --output-dir /path/to/figs --mode series --depth shallow
"""
import argparse
from pathlib import Path

import numpy as np

from source_localization.validation.resolution import ResolutionAnalysis
from source_localization.validation.viz3d import render_activation_cloud

DEPTH_BINS = {'very_shallow': (0, 15), 'shallow': (15, 40),
              'mid': (40, 70), 'deep': (70, 100)}


def pick_source(ra, depth_label):
    """The best-behaved source in a depth bin: sharpest PSF, so clearest picture."""
    lo, hi = DEPTH_BINS[depth_label]
    d = ra.source_depths
    idxs = np.where((d >= np.percentile(d, lo)) & (d <= np.percentile(d, hi)))[0]
    contrast = np.array([ra.psf_metrics(int(i))['peak_contrast'] for i in idxs])
    return int(idxs[int(np.argmax(contrast))])


def reconstruct_one(ra, idx, orientation, snr_db, noise_type, seed):
    act = ra._reconstruct(idx, orientation, snr_db=snr_db,
                          noise_seed=seed, noise_type=noise_type)
    return act


def reconstruct_two(ra, a, b, o1, o2, snr_db, noise_type, seed):
    """Returns (magnitude per source, raw activity) — the detector needs the raw."""
    eeg, _ = ra.simulator.simulate_two_dipoles(
        position1_mm=ra.source_pos_mm[a], position2_mm=ra.source_pos_mm[b],
        orientation1=o1, orientation2=o2,
        amplitude1_nAm=50.0, amplitude2_nAm=50.0,
        snr_db=snr_db, noise_seed=seed, noise_type=noise_type,
        duration_s=0.2, sfreq=256.0)
    raw = ra._apply_inverse(eeg)
    return ra._source_activity_norm(raw), raw


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--pipeline-dir', required=True)
    ap.add_argument('--output-dir', required=True)
    ap.add_argument('--mode', default='two', choices=['one', 'two', 'series'])
    ap.add_argument('--depth', default='shallow', choices=list(DEPTH_BINS))
    ap.add_argument('--separation', type=float, default=6.0)
    ap.add_argument('--series', type=float, nargs='+', default=[3.0, 5.0, 7.0, 9.0],
                    help='separations for --mode series')
    ap.add_argument('--snr-db', type=float, default=float('inf'))
    ap.add_argument('--noise-type', default='white',
                    choices=['white', 'spatial', 'temporal', 'colored'])
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--inverse-method', default='sLORETA')
    ap.add_argument('--inverse-snr', type=float, default=3.0)
    ap.add_argument('--source-idx', type=int, default=None,
                    help='explicit source index (overrides --depth selection)')
    ap.add_argument('--pick-orientation', action='store_true',
                    help='search the orientation pairs for one that resolves as '
                         'two lobes, i.e. render the best case rather than an '
                         'arbitrary one')
    ap.add_argument('--resolution', type=int, default=72)
    ap.add_argument('--views', nargs='+', default=['iso', 'top', 'side'])
    args = ap.parse_args()

    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)

    print(f"Loading pipeline from {args.pipeline_dir}")
    ra = ResolutionAnalysis.from_pipeline_dir(
        args.pipeline_dir, inverse_method=args.inverse_method,
        inverse_snr=args.inverse_snr, verbose=False)

    a = args.source_idx if args.source_idx is not None else pick_source(ra, args.depth)
    o1, o2 = np.eye(3)[0], np.eye(3)[0]
    sp = ra.source_pos_mm
    snr_tag = 'noise-free' if not np.isfinite(args.snr_db) else f'SNR {args.snr_db:g} dB'

    def peak_of(act):
        return sp[int(np.argmax(act))]

    if args.mode == 'one':
        act = reconstruct_one(ra, a, o1, args.snr_db, args.noise_type, args.seed)
        err = float(np.linalg.norm(peak_of(act) - sp[a]))
        path = out / f'cloud_one_{args.depth}.png'
        render_activation_cloud(
            sp, act, path, true_positions_mm=[sp[a]], peak_position_mm=peak_of(act),
            title=(f'Single source, {args.depth} (depth {ra.source_depths[a]:.1f} mm), '
                   f'{snr_tag}\ncyan = true position, green cross = reconstruction peak '
                   f'(error {err:.2f} mm)'),
            resolution=args.resolution, views=args.views)
        print(f"peak error {err:.2f} mm -> {path}")
        return

    seps = args.series if args.mode == 'series' else [args.separation]
    for sep in seps:
        partners = ra._partners_at(a, sep, n_partners=1, tol_mm=0.8)
        if not partners:
            print(f"  no partner at {sep} mm, skipping")
            continue
        b = partners[0]
        actual = float(np.linalg.norm(sp[a] - sp[b]))
        pairs = ([(x, y) for x in np.eye(3) for y in np.eye(3)]
                 if args.pick_orientation else [(o1, o2)])
        for u1, u2 in pairs:
            act, raw = reconstruct_two(ra, a, b, u1, u2, args.snr_db,
                                       args.noise_type, args.seed)
            det = ra._detect_two_lobes(raw)['detected']
            if det:
                break   # found a resolving orientation; render that one
        verdict = 'TWO lobes resolved' if det else 'MERGED into one lobe'
        path = out / f'cloud_two_{args.depth}_{actual:.0f}mm.png'
        render_activation_cloud(
            sp, act, path, true_positions_mm=[sp[a], sp[b]],
            title=(f'Two sources {actual:.1f} mm apart, {args.depth}, {snr_tag}\n'
                   f'cyan = true positions — {verdict}'),
            resolution=args.resolution, views=args.views)
        print(f"  {actual:5.1f} mm: {verdict:22} -> {path}")


if __name__ == '__main__':
    main()
