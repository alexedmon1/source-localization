#!/usr/bin/env python
"""
Two sources: can we tell there are two, and how far apart are they?

Everything here is a probability from the two-dipole posterior, scored against a
matched one-source null, rather than a threshold verdict on a reconstruction.

Figure panels:
  A  log10 Bayes factor (two sources vs one) — null distribution and each true
     separation. Separation of the null from the signal is the detectability.
  B  Detection rate vs true separation, per SNR, calling a detection when the
     Bayes factor exceeds the null's 90th percentile (a 10% false-alarm rate by
     construction, so the number is honest).
  C  Posterior separation estimate vs truth — detecting that there are two
     sources and measuring how far apart they are are different problems, and
     the second is much harder.
  D  The joint posterior's symmetrized marginal: p(a source is here), which
     shows both lobes and the ambiguity bridge between them.

Usage
-----
python scripts/run_two_source_posterior.py --pipeline-dir DIR --output-dir DIR
"""
import argparse
import json
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm

from source_localization.validation.posterior import DipolePosterior
from source_localization.validation.two_source import TwoSourcePosterior

SNRS = (20.0, 10.0, 0.0)
SEPARATIONS = (2.0, 3.0, 4.0, 6.0, 8.0)
COLORS = {20.0: '#0072B2', 10.0: '#E69F00', 0.0: '#D55E00'}


def run_condition(ts, dp, idx, snr, seps, n_trials, seed):
    pos = dp.positions_mm
    rng = np.random.default_rng(seed)

    null = []
    for _ in range(n_trials):
        a = int(rng.choice(idx))
        data, sig = dp.simulate(pos[a], rng.normal(size=3), snr, rng)
        null.append(ts.analyze(data, sig)['log10_bayes_factor'])
    null = np.array(null)
    cut = float(np.percentile(null, 90))

    rows = []
    for sep in seps:
        bf, est, true = [], [], []
        for _ in range(n_trials):
            a = int(rng.choice(idx))
            d = np.linalg.norm(pos[idx] - pos[a], axis=1)
            cand = idx[np.abs(d - sep) <= 0.6]
            cand = cand[cand != a]
            if not len(cand):
                continue
            b = int(rng.choice(cand))
            data, sig = ts.simulate_pair(pos[a], pos[b], rng.normal(size=3),
                                         rng.normal(size=3), snr, rng)
            r = ts.analyze(data, sig)
            bf.append(r['log10_bayes_factor'])
            est.append(r['median_separation_mm'])
            true.append(float(np.linalg.norm(pos[a] - pos[b])))
        if not bf:
            continue
        bf = np.array(bf)
        rows.append({
            'nominal_sep_mm': float(sep),
            'true_sep_mm': float(np.mean(true)),
            'log10_bf': bf.tolist(),
            'median_log10_bf': float(np.median(bf)),
            'detection_rate': float(np.mean(bf > cut)),
            'median_posterior_sep_mm': float(np.nanmedian(est)),
            'iqr_posterior_sep_mm': float(np.subtract(
                *np.nanpercentile(est, [75, 25]))),
        })
    return {'null_log10_bf': null.tolist(), 'null_cut_90': cut, 'rows': rows}


def fig_main(results, ts, dp, idx, out_path, seed):
    plt.rcParams.update({'font.size': 10, 'axes.spines.top': False,
                         'axes.spines.right': False, 'axes.edgecolor': '#888888',
                         'figure.facecolor': 'white', 'axes.facecolor': 'white'})
    fig = plt.figure(figsize=(16.5, 8.4))
    gs = fig.add_gridspec(2, 3, hspace=0.36, wspace=0.28)

    # A: Bayes factor distributions at the middle SNR.
    axA = fig.add_subplot(gs[0, 0])
    snr_ref = 20.0
    res = results[snr_ref]
    data = [res['null_log10_bf']] + [r['log10_bf'] for r in res['rows']]
    labels = ['null\n(1 src)'] + [f"{r['true_sep_mm']:.0f}mm" for r in res['rows']]
    bp = axA.boxplot(data, labels=labels, patch_artist=True, widths=0.6,
                     showfliers=False)
    for k, patch in enumerate(bp['boxes']):
        patch.set_facecolor('#cccccc' if k == 0 else '#0072B2')
        patch.set_alpha(0.75)
    axA.axhline(res['null_cut_90'], color='#cc0000', ls=':', lw=1.5)
    axA.text(0.5, res['null_cut_90'], ' null 90th pct', color='#cc0000',
             fontsize=8, va='bottom')
    axA.set_ylabel('log10 Bayes factor (2 vs 1)')
    axA.set_title(f'A  Evidence for two sources, SNR {snr_ref:+.0f} dB',
                  loc='left', fontweight='bold', fontsize=11)
    axA.grid(True, color='#eee', axis='y')

    # B: detection rate vs separation.
    axB = fig.add_subplot(gs[0, 1])
    for snr in SNRS:
        rows = results[snr]['rows']
        axB.plot([r['true_sep_mm'] for r in rows],
                 [r['detection_rate'] for r in rows], marker='o', ms=5, lw=2,
                 color=COLORS[snr], label=f'{snr:+.0f} dB')
    axB.axhline(0.1, color='#999999', ls=':', lw=1.2)
    axB.text(2.0, 0.12, 'false-alarm rate (10%)', fontsize=8, color='#777777')
    axB.set_xlabel('true separation (mm)')
    axB.set_ylabel('P(detect two sources)')
    axB.set_ylim(0, 1.02)
    axB.set_title('B  Detectability vs separation', loc='left',
                  fontweight='bold', fontsize=11)
    axB.legend(frameon=False, fontsize=9)
    axB.grid(True, color='#eee')

    # C: separation estimate vs truth.
    axC = fig.add_subplot(gs[0, 2])
    lim = max(SEPARATIONS) + 2
    axC.plot([0, lim], [0, lim], color='#999999', ls='--', lw=1.5,
             label='perfect estimate')
    for snr in SNRS:
        rows = results[snr]['rows']
        axC.errorbar([r['true_sep_mm'] for r in rows],
                     [r['median_posterior_sep_mm'] for r in rows],
                     yerr=[r['iqr_posterior_sep_mm'] / 2 for r in rows],
                     marker='s', ms=5, lw=2, capsize=3, color=COLORS[snr],
                     label=f'{snr:+.0f} dB')
    axC.set_xlabel('true separation (mm)')
    axC.set_ylabel('posterior median separation (mm)')
    axC.set_title('C  Estimating HOW FAR apart', loc='left',
                  fontweight='bold', fontsize=11)
    axC.legend(frameon=False, fontsize=8)
    axC.grid(True, color='#eee')

    # D-F: the marginal density itself, at three SNRs, for one 4 mm pair.
    # Pick a pair whose evidence is TYPICAL (median log10 BF at the reference
    # SNR) rather than the first one drawn: an edge pair with BF ~ 0 would
    # contradict panel B and read as a broken figure.
    pos = dp.positions_mm
    rng = np.random.default_rng(seed)
    cands = []
    for _ in range(9):
        a = int(rng.choice(idx))
        d = np.linalg.norm(pos[idx] - pos[a], axis=1)
        c = idx[np.abs(d - 4.0) <= 0.6]
        c = c[c != a]
        if len(c):
            cands.append((a, int(rng.choice(c))))
    scored = []
    for (a, b) in cands:
        data, sig = ts.simulate_pair(pos[a], pos[b], np.array([1.0, 0, 0]),
                                     np.array([0, 1.0, 0]), snr_ref,
                                     np.random.default_rng(seed))
        scored.append((ts.analyze(data, sig)['log10_bayes_factor'], a, b))
    scored.sort()
    _, a, b = scored[len(scored) // 2]

    for k, snr in enumerate(SNRS):
        ax = fig.add_subplot(gs[1, k])
        data, sig = ts.simulate_pair(pos[a], pos[b], np.array([1.0, 0, 0]),
                                     np.array([0, 1.0, 0]), snr,
                                     np.random.default_rng(seed + k))
        r = ts.analyze(data, sig)
        marg = ts.marginal_density(r['joint']) / 2.0

        # Project along the dorsal-ventral axis rather than slicing: the density
        # is 3D, and a single slice hides most of the mass out of plane.
        xs = np.unique(np.round(pos[:, 0], 6))
        ys = np.unique(np.round(pos[:, 1], 6))
        img = np.zeros((len(ys), len(xs)))
        xi = np.searchsorted(xs, np.round(pos[:, 0], 6))
        yi = np.searchsorted(ys, np.round(pos[:, 1], 6))
        np.add.at(img, (yi, xi), marg)
        img[img <= 0] = np.nan

        peak = np.nanmax(img)
        im = ax.imshow(img, origin='lower', cmap='magma',
                       extent=[xs[0], xs[-1], ys[0], ys[-1]],
                       norm=LogNorm(vmin=peak * 1e-3, vmax=peak),
                       interpolation='bilinear', aspect='equal')
        for p_, lab in ((pos[a], 'A'), (pos[b], 'B')):
            ax.plot(p_[0], p_[1], marker='+', ms=15, mew=2.6, color='#00E5FF')
            ax.annotate(lab, (p_[0], p_[1]), textcoords='offset points',
                        xytext=(7, 5), color='#00E5FF', fontweight='bold',
                        fontsize=9)
        ax.set_xticks([]); ax.set_yticks([])
        ax.set_title(f'{"DEF"[k]}  p(a source is here), SNR {snr:+.0f} dB',
                     loc='left', fontweight='bold', fontsize=11)
        # Top-left: the sources can sit low in the projection, and a bottom
        # annotation covered marker A.
        ax.text(0.03, 0.97,
                f"log10 BF = {r['log10_bayes_factor']:+.1f}\n"
                f"post. sep = {r['median_separation_mm']:.1f} mm "
                f"(true {np.linalg.norm(pos[a]-pos[b]):.1f})",
                transform=ax.transAxes, fontsize=8, va='top',
                bbox=dict(fc='white', ec='none', alpha=0.8, pad=2))
        if k == 0:
            fig.colorbar(im, ax=ax, label='probability per cell (projected)',
                         shrink=0.85)

    fig.suptitle('Two sources under the calibrated posterior — detection, separation, '
                 'and the joint density\n'
                 f'(fundamental single/two-dipole limit, {dp.n_positions} positions '
                 f'at {dp.spacing_mm} mm, dorsal half of the brain)',
                 fontweight='bold', fontsize=12)
    fig.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--pipeline-dir', required=True)
    ap.add_argument('--output-dir', required=True)
    ap.add_argument('--spacing-mm', type=float, default=1.0,
                    help='pair scan is quadratic in position count; 1.0 mm '
                         'gives ~760 positions / ~288k pairs')
    ap.add_argument('--n-trials', type=int, default=30)
    ap.add_argument('--seed', type=int, default=4)
    ap.add_argument('--figure-only', action='store_true',
                    help='re-render from a saved JSON without re-running')
    ap.add_argument('--depth-percentile', type=float, default=60.0,
                    help='restrict sources to the dorsal fraction above this '
                         'z-percentile, so depth (and therefore gain) is '
                         'roughly matched within a pair')
    args = ap.parse_args()

    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)

    dp = DipolePosterior.from_pipeline_dir(args.pipeline_dir,
                                           spacing_mm=args.spacing_mm)
    ts = TwoSourcePosterior(dp)
    print(f"  {dp.n_positions} positions, {ts.n_pairs} pairs")

    pos = dp.positions_mm
    idx = np.where(pos[:, 2] >= np.percentile(pos[:, 2],
                                              args.depth_percentile))[0]
    print(f"  restricted to {len(idx)} dorsal positions")

    results = {}
    if args.figure_only:
        with open(out / 'two_source_posterior.json') as f:
            results = {float(k): v for k, v in json.load(f).items()}
    for snr in (() if args.figure_only else SNRS):
        print(f"  SNR {snr:+.0f} dB ...", flush=True)
        results[snr] = run_condition(ts, dp, idx, snr, SEPARATIONS,
                                     args.n_trials, args.seed)

    fig_main(results, ts, dp, idx, out / 'two_source_posterior.png', args.seed)
    if not args.figure_only:
        with open(out / 'two_source_posterior.json', 'w') as f:
            json.dump({str(k): v for k, v in results.items()}, f, indent=2)

    print("\n" + "=" * 76)
    print("CAN WE TELL THERE ARE TWO SOURCES?  (detection at a 10% false-alarm rate)")
    print("=" * 76)
    print(f"{'SNR':>7}{'null median':>14}" +
          ''.join(f"{f'{s:.0f}mm':>9}" for s in SEPARATIONS))
    for snr in SNRS:
        r = results[snr]
        rates = ''.join(f"{row['detection_rate']:>9.0%}" for row in r['rows'])
        print(f"{snr:>+6.0f}dB{np.median(r['null_log10_bf']):>14.2f}{rates}")

    print("\nHOW FAR APART? (posterior median separation, mm)")
    print(f"{'SNR':>7}" + ''.join(f"{f'true {s:.0f}':>10}" for s in SEPARATIONS))
    for snr in SNRS:
        est = ''.join(f"{row['median_posterior_sep_mm']:>10.1f}"
                      for row in results[snr]['rows'])
        print(f"{snr:>+6.0f}dB{est}")
    print(f"\nSaved: {out / 'two_source_posterior.png'}")


if __name__ == '__main__':
    main()
