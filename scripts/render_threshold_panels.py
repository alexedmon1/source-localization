#!/usr/bin/env python
"""
Visual audit of the threshold percentages: what does "above X% of peak" look like?

Two figures, both cross-sections through the actual reconstruction:

--mode depth   One source per depth bin, thresholded at a ladder of levels. This
               is the check on "a less precise source keeps a larger region at a
               high threshold": the supra-threshold patch should shrink to a dot
               for a shallow source and stay broad for a deep one.

--mode pair    One source pair at a fixed separation, thresholded at the same
               ladder, annotated with each source's activation as a % of peak and
               the separability verdict. This is the check on the merged ->
               separated -> dominated squeeze: you can read off exactly which
               threshold loses the weaker source.

The plane is chosen to contain both sources (pair mode) or the source and the
depth direction (depth mode), so nothing is hidden by projection.

Usage
-----
python scripts/render_threshold_panels.py --pipeline-dir DIR --output-dir DIR --mode depth
python scripts/render_threshold_panels.py --pipeline-dir DIR --output-dir DIR --mode pair --separation 6
"""
import argparse
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap

from source_localization.validation.separability import SeparabilityAnalysis
from source_localization.validation.viz3d import sample_on_plane

DEPTH_BINS = [('very_shallow', 0, 15), ('shallow', 15, 40),
              ('mid', 40, 70), ('deep', 70, 100)]
THRESHOLDS = [0.15, 0.25, 0.35, 0.50, 0.65, 0.80]


def pick_representative(sa, lo, hi):
    """Median-contrast source in the bin — typical, not cherry-picked."""
    d = sa.source_depths
    idxs = np.where((d >= np.percentile(d, lo)) & (d <= np.percentile(d, hi)))[0]
    c = np.array([sa.psf_metrics(int(i))['peak_contrast'] for i in idxs])
    return int(idxs[np.argsort(c)[len(c) // 2]])


def plane_basis(sa, a, b=None):
    """In-plane axes: separation axis (pair) or a lateral axis, plus depth (Z)."""
    if b is not None:
        e1 = sa.source_pos_mm[b] - sa.source_pos_mm[a]
        center = 0.5 * (sa.source_pos_mm[a] + sa.source_pos_mm[b])
    else:
        e1 = np.array([0.0, 1.0, 0.0])       # A-P, the long axis
        center = sa.source_pos_mm[a].copy()
    return center, e1, np.array([0.0, 0.0, 1.0])


BLOB_COLORS = ['#D55E00', '#0072B2', '#009E73', '#CC79A7', '#E69F00']


def project(sa, center, e1, e2):
    """Source coordinates in the plane frame, plus out-of-plane distance."""
    e1 = np.asarray(e1, float) / np.linalg.norm(e1)
    e2 = np.asarray(e2, float)
    e2 = e2 - np.dot(e2, e1) * e1
    e2 = e2 / np.linalg.norm(e2)
    rel = sa.source_pos_mm - np.asarray(center, float)
    u, v = rel @ e1, rel @ e2
    out = np.linalg.norm(rel - np.outer(u, e1) - np.outer(v, e2), axis=1)
    return u, v, out


def draw_panel(ax, sa, act, thr, uvo, marks, title, sub=None, slab_mm=1.6,
               half=8.0):
    """
    Plot the SOURCES themselves — one dot per source, coloured by blob.

    Deliberately not a smoothed image. The metrics operate on the discrete
    source grid, so a smoothed field can disagree with them: interpolation
    averages a sharp peak down against its neighbours, and a contour drawn on it
    is then not the threshold the metric applied. Here each dot is a source, its
    colour is its own supra-threshold blob id, and grey means below threshold —
    so "merged vs separated" is read off the picture exactly as scored.
    """
    u, v, out = uvo
    near = out <= slab_mm

    labels = sa.suprathreshold_labels(act, thr)
    supra = labels >= 0

    sub_mask = near & ~supra
    ax.scatter(u[sub_mask], v[sub_mask], s=13, c='#d9d9d9', edgecolors='none',
               zorder=1)

    ids = [b for b in np.unique(labels[near & supra]) if b >= 0]
    # Largest blob first, so colours are stable across thresholds.
    ids.sort(key=lambda b: -np.sum(labels == b))
    for k, b in enumerate(ids):
        m = near & (labels == b)
        ax.scatter(u[m], v[m], s=34, c=BLOB_COLORS[k % len(BLOB_COLORS)],
                   edgecolors='#333333', linewidths=0.4, zorder=2)

    for (uu, vv, lab, col) in marks:
        ax.plot(uu, vv, marker='+', ms=14, mew=2.6, color=col, zorder=4)
        if lab:
            ax.annotate(lab, (uu, vv), textcoords='offset points', xytext=(7, 6),
                        fontsize=8, color='#006D77', fontweight='bold', zorder=4)

    ax.set_title(title, fontsize=10, fontweight='bold', loc='left')
    n_blobs = len(ids)
    ax.text(0.03, 0.03, (sub if sub else '') + f'  [{n_blobs} blob'
            + ('s' if n_blobs != 1 else '') + ']',
            transform=ax.transAxes, fontsize=8, color='#111111',
            bbox=dict(fc='white', ec='none', alpha=0.8, pad=1.6))
    ax.set_xlim(-half, half); ax.set_ylim(-half, half)
    ax.set_aspect('equal')
    ax.set_xticks([]); ax.set_yticks([])


def fig_depth(sa, out_path, half=8.0):
    fig, axes = plt.subplots(len(DEPTH_BINS), len(THRESHOLDS),
                             figsize=(2.15 * len(THRESHOLDS), 2.3 * len(DEPTH_BINS)))
    for r, (label, lo, hi) in enumerate(DEPTH_BINS):
        a = pick_representative(sa, lo, hi)
        psf = sa.compute_psf(a)
        center, e1, e2 = plane_basis(sa, a)
        uvo = project(sa, center, e1, e2)
        for c, thr in enumerate(THRESHOLDS):
            ext = sa.blob_extent(psf, thr, source_idx=a)
            rad = ('--' if not ext['contains_source']
                   else f"r={ext['rms_radius_mm']:.1f}mm")
            draw_panel(axes[r, c], sa, psf, thr, uvo,
                       [(0.0, 0.0, 'true', '#00E5FF')],
                       f'{label}  T={thr:.0%}' if c == 0 else f'T={thr:.0%}',
                       sub=rad, half=half)
        axes[r, 0].set_ylabel(f'{label}\ndepth {sa.source_depths[a]:.1f} mm',
                              fontsize=9, fontweight='bold')
    fig.suptitle('What "above X% of peak" actually looks like — ONE source, by depth\n'
                 'each dot is a source: grey = below threshold, colour = supra-threshold blob. '
                 'The less precise (deeper) the source, the larger the region that survives a high threshold.',
                 fontweight='bold', fontsize=12)
    fig.tight_layout(rect=[0, 0, 1, 0.93])
    fig.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close(fig)


def fig_pair(sa, out_path, separation, depth_label, depth_match, half=9.0):
    lo, hi = dict((l, (a, b)) for l, a, b in DEPTH_BINS)[depth_label]
    d = sa.source_depths
    idxs = np.where((d >= np.percentile(d, lo)) & (d <= np.percentile(d, hi)))[0]

    chosen = None
    for a in idxs:
        p = sa._partners_at(int(a), separation, 1, 0.8, depth_match_mm=depth_match)
        if p:
            chosen = (int(a), int(p[0]))
            break
    if chosen is None:
        raise SystemExit(f'no depth-matched pair at {separation} mm in {depth_label}')
    a, b = chosen

    triad = np.eye(3)
    best = None
    for o1 in triad:                       # pick the most balanced orientation pair
        for o2 in triad:
            eeg, _ = sa.simulator.simulate_two_dipoles(
                position1_mm=sa.source_pos_mm[a], position2_mm=sa.source_pos_mm[b],
                orientation1=o1, orientation2=o2, amplitude1_nAm=50.0,
                amplitude2_nAm=50.0, snr_db=np.inf, noise_seed=0,
                noise_type='white', duration_s=0.2, sfreq=256.0)
            act = sa._source_activity_norm(sa._apply_inverse(eeg))
            ratio = min(act[a], act[b]) / max(act[a], act[b])
            if best is None or ratio > best[0]:
                best = (ratio, act)
    ratio, act = best

    center, e1, e2 = plane_basis(sa, a, b)
    uvo = project(sa, center, e1, e2)
    half_sep = 0.5 * float(np.linalg.norm(sa.source_pos_mm[b] - sa.source_pos_mm[a]))
    peak = act.max()
    pa, pb = act[a] / peak, act[b] / peak

    fig, axes = plt.subplots(1, len(THRESHOLDS),
                             figsize=(2.6 * len(THRESHOLDS), 3.4))
    for c, thr in enumerate(THRESHOLDS):
        verdict = sa.classify_pair(act, a, b, thr)
        colour = {'separated': '#00A000', 'merged': '#B07800',
                  'dominated': '#C03000'}[verdict]
        draw_panel(axes[c], sa, act, thr, uvo,
                   [(-half_sep, 0.0, f'A {pa:.0%}', '#00E5FF'),
                    (half_sep, 0.0, f'B {pb:.0%}', '#00E5FF')],
                   f'T={thr:.0%}', sub=verdict.upper(), half=half)
        axes[c].set_title(f'T={thr:.0%}', fontsize=10, fontweight='bold', loc='left',
                          color=colour)
    matched = 'depth-matched' if depth_match is not None else 'NOT depth-matched'
    fig.suptitle(
        f'Two sources {2*half_sep:.1f} mm apart ({depth_label}, {matched}, noise-free, '
        f'best of 9 orientation pairs) — A reconstructs at {pa:.0%} of peak, '
        f'B at {pb:.0%} (balance {ratio:.2f})\n'
        f'each dot is a source; grey = below threshold, colour = blob id.  '
        f'MERGED = one blob · DOMINATED = weaker source dropped below the threshold',
        fontweight='bold', fontsize=11)
    fig.tight_layout(rect=[0, 0, 1, 0.86])
    fig.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--pipeline-dir', required=True)
    ap.add_argument('--output-dir', required=True)
    ap.add_argument('--mode', default='depth', choices=['depth', 'pair', 'both'])
    ap.add_argument('--separation', type=float, default=6.0)
    ap.add_argument('--depth-label', default='very_shallow',
                    choices=[b[0] for b in DEPTH_BINS])
    ap.add_argument('--depth-match-mm', type=float, default=0.4,
                    help='set to a negative value to disable depth matching, '
                         'which surfaces the "dominated" failure mode')
    ap.add_argument('--inverse-method', default='sLORETA')
    ap.add_argument('--inverse-snr', type=float, default=3.0)
    args = ap.parse_args()

    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    sa = SeparabilityAnalysis.from_pipeline_dir(
        args.pipeline_dir, inverse_method=args.inverse_method,
        inverse_snr=args.inverse_snr, verbose=False)

    if args.mode in ('depth', 'both'):
        p = out / 'threshold_panels_depth.png'
        fig_depth(sa, p)
        print(f'Saved: {p}')
    if args.mode in ('pair', 'both'):
        dm = None if args.depth_match_mm < 0 else args.depth_match_mm
        p = out / (f'threshold_panels_pair_{args.separation:.0f}mm'
                   f'{"" if dm is not None else "_unmatched"}.png')
        fig_pair(sa, p, args.separation, args.depth_label, dm)
        print(f'Saved: {p}')


if __name__ == '__main__':
    main()
