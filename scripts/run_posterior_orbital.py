#!/usr/bin/env python
"""
Calibrated source-location probability clouds ("orbitals"), and the proof they mean something.

Produces two figures:

  posterior_calibration.png  Does a p% credible region contain the true source p%
                             of the time? Compares profiling the dipole moment
                             out against marginalizing over it, across SNR. This
                             is what licenses reading the percentages literally.

  posterior_orbital.png      The clouds themselves: axial slices of p(location |
                             data) at several SNRs, with the 50% and 95%
                             highest-density contours and the true source marked.

Unlike a thresholded reconstruction image, these regions are normalized
probability and are checked against ground truth, so "95%" is a testable claim
rather than a display setting.

Usage
-----
python scripts/run_posterior_orbital.py --pipeline-dir DIR --output-dir DIR
"""
import argparse
import json
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from source_localization.validation.posterior import DipolePosterior

LEVELS = (0.5, 0.68, 0.9, 0.95)
SNRS = (20.0, 10.0, 5.0, 0.0, -5.0)


def fig_calibration(dp, out_path, n_trials, seed):
    prof, marg = {}, {}
    for snr in SNRS:
        prof[snr] = dp.coverage_test(levels=LEVELS, snr_db=snr,
                                     n_trials=n_trials, seed=seed)
        marg[snr] = dp.coverage_test(levels=LEVELS, snr_db=snr,
                                     n_trials=n_trials, seed=seed,
                                     moment_std=1.0)

    brain_mm3 = dp.n_positions * dp.spacing_mm ** 3
    plt.rcParams.update({'font.size': 10, 'axes.spines.top': False,
                         'axes.spines.right': False, 'axes.edgecolor': '#888888',
                         'figure.facecolor': 'white', 'axes.facecolor': 'white'})
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(15, 4.6))
    colors = plt.cm.viridis(np.linspace(0, 0.85, len(SNRS)))

    for ax, res, name in ((ax1, prof, 'profile (maximize moment)'),
                          (ax2, marg, 'marginal (integrate moment)')):
        ax.plot([0, 1], [0, 1], color='#999999', ls='--', lw=1.5,
                label='perfect calibration')
        for c, snr in zip(colors, SNRS):
            ax.plot(res[snr]['levels'], res[snr]['coverage'], marker='o', ms=5,
                    lw=2, color=c, label=f'{snr:+.0f} dB')
        ax.set_xlabel('nominal credible level')
        ax.set_ylabel('empirical coverage')
        ax.set_xlim(0.4, 1.0); ax.set_ylim(0.2, 1.02)
        ax.set_title(name, loc='left', fontweight='bold', fontsize=11)
        ax.grid(True, color='#eee')
        ax.legend(frameon=False, fontsize=8)

    for c, snr in zip(colors, SNRS):
        pct = 100 * np.array(marg[snr]['median_volume_mm3']) / brain_mm3
        ax3.plot(marg[snr]['levels'], pct, marker='s', ms=5, lw=2, color=c,
                 label=f'{snr:+.0f} dB')
    ax3.set_xlabel('credible level')
    ax3.set_ylabel('region size (% of brain volume)')
    ax3.set_yscale('log')
    ax3.set_title('How big is the region?  (marginal)', loc='left',
                  fontweight='bold', fontsize=11)
    ax3.grid(True, color='#eee', which='both')
    ax3.legend(frameon=False, fontsize=8)

    fig.suptitle('Is the probability real? Coverage of credible regions vs ground truth '
                 f'({n_trials} trials/point, truth placed off-grid)',
                 fontweight='bold', fontsize=12)
    fig.text(0.5, -0.03,
             'Points on the dashed line mean the number is honest. Profiling the dipole moment out '
             'is over-confident once noise is comparable to signal; marginalizing stays calibrated, '
             'erring slightly conservative. The residual over-coverage at +20 dB is grid '
             f'discretization — the region is only a few {dp.spacing_mm} mm cells.',
             ha='center', fontsize=9, style='italic', color='#555555')
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    return {'profile': {str(k): v for k, v in prof.items()},
            'marginal': {str(k): v for k, v in marg.items()},
            'brain_volume_mm3': brain_mm3}


def _slice_image(dp, post, z0):
    """
    Exact 2D image of one axial layer.

    The grid is regular, so the layer maps onto an array without interpolation;
    cells outside the brain stay NaN.
    """
    pos = dp.positions_mm
    sel = np.abs(pos[:, 2] - z0) < dp.spacing_mm * 0.51
    if not sel.any():
        return None, None, None
    xs = np.unique(np.round(pos[sel, 0], 6))
    ys = np.unique(np.round(pos[sel, 1], 6))
    img = np.full((len(ys), len(xs)), np.nan)
    xi = np.searchsorted(xs, np.round(pos[sel, 0], 6))
    yi = np.searchsorted(ys, np.round(pos[sel, 1], 6))
    img[yi, xi] = post[sel]
    extent = [xs[0], xs[-1], ys[0], ys[-1]]
    return img, extent, sel


def _hdr_level(post, level):
    """Density value bounding the highest-density region of a given mass."""
    order = np.argsort(post)[::-1]
    c = np.cumsum(post[order])
    k = int(np.searchsorted(c, level))
    return float(post[order[min(k, len(order) - 1)]])


def fig_orbital(dp, out_path, snrs=(20.0, 10.0, 0.0), seed=3):
    """Axial slices of the posterior through the true source, per SNR and depth."""
    from matplotlib.colors import LogNorm

    pos = dp.positions_mm
    # Dorsal and ventral test locations, taken INSET from the boundary and near
    # the mid-sagittal plane: an edge source gives a truncated slice with the
    # truth stuck in a corner, which reads as an artifact.
    z_hi, z_lo = np.percentile(pos[:, 2], [88, 12])
    def _pick(z_target):
        cost = (np.abs(pos[:, 2] - z_target) + 0.5 * np.abs(pos[:, 0])
                + 0.5 * np.abs(pos[:, 1]))
        return int(np.argmin(cost))
    picks = [('shallow (dorsal)', _pick(z_hi)),
             ('deep (ventral)', _pick(z_lo))]

    brain = dp.n_positions * dp.spacing_mm ** 3
    fig, axes = plt.subplots(len(picks), len(snrs),
                             figsize=(4.3 * len(snrs), 4.2 * len(picks)))
    axes = np.atleast_2d(axes)

    for r, (label, idx) in enumerate(picks):
        truth = pos[idx]
        for c, snr in enumerate(snrs):
            data, noise_std = dp.simulate(truth, np.array([1.0, 0.0, 0.0]), snr,
                                          np.random.default_rng(seed + c))
            post = dp.posterior(data, noise_std, moment_std=1.0)
            img, extent, _ = _slice_image(dp, post, truth[2])
            ax = axes[r, c]

            # Linear, as a percentage of the peak: a log density axis spans
            # orders of magnitude whose numbers mean nothing to a reader, and
            # it exaggerates the far tail. The probability statements are the
            # credible contours below, not the colour.
            peak = np.nanmax(img)
            ax.imshow(100.0 * img / peak, origin='lower', extent=extent,
                      cmap='magma', vmin=0, vmax=100,
                      interpolation='bilinear', aspect='equal')
            # Contour the credible regions at their true density levels, so the
            # outline is the actual boundary of the 50% / 95% mass.
            for lv, col in ((0.95, '#00E5FF'), (0.5, '#39FF14')):
                lvl = _hdr_level(post, lv)
                if np.isfinite(lvl) and lvl > 0:
                    # Mask, don't zero-fill: filling NaN with 0 draws a spurious
                    # contour around the whole brain boundary.
                    ax.contour(np.linspace(extent[0], extent[1], img.shape[1]),
                               np.linspace(extent[2], extent[3], img.shape[0]),
                               np.ma.masked_invalid(img), levels=[lvl],
                               colors=col, linewidths=1.8)
            ax.plot(truth[0], truth[1], marker='+', ms=16, mew=3.0,
                    color='#FFFFFF', zorder=5)
            ax.plot(truth[0], truth[1], marker='+', ms=16, mew=1.2,
                    color='#000000', zorder=6)

            v50 = dp.credible_volume_mm3(post, 0.5)
            v95 = dp.credible_volume_mm3(post, 0.95)
            ax.set_title(f'{label}, SNR {snr:+.0f} dB', loc='left',
                         fontweight='bold', fontsize=11)
            ax.text(0.03, 0.03,
                    f'50%: {v50:.1f} mm³ ({100*v50/brain:.1f}% of brain)\n'
                    f'95%: {v95:.1f} mm³ ({100*v95/brain:.1f}%)',
                    transform=ax.transAxes, fontsize=8, color='#111111',
                    bbox=dict(fc='white', ec='none', alpha=0.82, pad=2))
            ax.set_xticks([]); ax.set_yticks([])
            if c == 0:
                ax.set_ylabel(f'{label}\nA-P (mm)', fontweight='bold', fontsize=10)

    fig.suptitle('p(true source location | data) — axial slice through the source\n'
                 'colour = posterior density, % of peak (100% = most probable location).  '
                 'green = 50% credible region, cyan = 95%, cross = true source',
                 fontweight='bold', fontsize=12)
    fig.tight_layout(rect=[0, 0, 1, 0.93])
    fig.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--pipeline-dir', required=True)
    ap.add_argument('--output-dir', required=True)
    ap.add_argument('--spacing-mm', type=float, default=0.5)
    ap.add_argument('--orbital-spacing-mm', type=float, default=0.3,
                    help='finer grid for the cloud figure; the calibration test '
                         'stays on --spacing-mm since it needs many trials')
    ap.add_argument('--n-trials', type=int, default=200)
    ap.add_argument('--seed', type=int, default=1)
    args = ap.parse_args()

    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)

    dp = DipolePosterior.from_pipeline_dir(args.pipeline_dir,
                                           spacing_mm=args.spacing_mm)
    print(f"  grid: {dp.n_positions} positions, "
          f"brain volume {dp.n_positions * dp.spacing_mm**3:.0f} mm3")

    cal = fig_calibration(dp, out / 'posterior_calibration.png',
                          args.n_trials, args.seed)
    print(f"Saved: {out / 'posterior_calibration.png'}")
    dpo = dp
    if abs(args.orbital_spacing_mm - args.spacing_mm) > 1e-9:
        print(f"  building finer grid for the cloud figure...")
        dpo = DipolePosterior.from_pipeline_dir(
            args.pipeline_dir, spacing_mm=args.orbital_spacing_mm)
    fig_orbital(dpo, out / 'posterior_orbital.png')
    print(f"Saved: {out / 'posterior_orbital.png'}")

    with open(out / 'posterior_calibration.json', 'w') as f:
        json.dump(cal, f, indent=2)

    print("\nCALIBRATED LOCALIZATION PRECISION (marginal posterior)")
    print(f"{'SNR':>7}{'50% region':>14}{'95% region':>14}{'95% radius':>13}"
          f"{'% of brain':>12}")
    brain = cal['brain_volume_mm3']
    for snr in SNRS:
        m = cal['marginal'][str(snr)]
        v50 = m['median_volume_mm3'][0]
        v95 = m['median_volume_mm3'][3]
        r95 = m['median_radius_mm'][3]
        print(f"{snr:>+6.0f}dB{v50:>11.1f}mm³{v95:>11.1f}mm³{r95:>10.2f}mm"
              f"{100*v95/brain:>11.1f}%")


if __name__ == '__main__':
    main()
