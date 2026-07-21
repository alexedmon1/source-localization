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


def _credible_level_map(density):
    """
    For each voxel, the smallest credible level whose region contains it.

    Runs 0-100% and is a genuine probability statement — a voxel at 50% is
    exactly on the boundary of the 50% highest-density region. This is what the
    colour axis shows, instead of "% of peak" (an amplitude, not a probability)
    or a log density (unreadable, and its numbers mean nothing to a reader).
    """
    import numpy as np
    order = np.argsort(density)[::-1]
    lvl = np.empty_like(density)
    lvl[order] = np.cumsum(density[order]) / max(density.sum(), 1e-300)
    return lvl


def _island_labels(dp, mask, radius_mm):
    """Connected-component label per voxel for a boolean mask; -1 outside."""
    import numpy as np
    from scipy.sparse import coo_matrix
    from scipy.sparse.csgraph import connected_components
    from scipy.spatial import cKDTree
    lab = np.full(mask.shape, -1, dtype=int)
    idx = np.where(mask)[0]
    if idx.size == 0:
        return lab
    pr = np.array(sorted(cKDTree(dp.positions_mm[idx]).query_pairs(radius_mm)),
                  dtype=int)
    if len(pr) == 0:
        lab[idx] = np.arange(idx.size)
        return lab
    g = coo_matrix((np.ones(len(pr)), (pr[:, 0], pr[:, 1])),
                   shape=(idx.size, idx.size))
    _, comp = connected_components(g + g.T, directed=False)
    lab[idx] = comp
    return lab


def _max_separating_level(dp, lvl, a, b, radius_mm,
                          levels=np.arange(0.05, 0.99, 0.05)):
    """
    Largest credible level at which the two sources still sit in separate islands.

    A single fixed level cannot answer this. The marginal density holds TWO
    units of mass (one per source), so its 50% region contains roughly ONE lobe
    by construction — at high SNR that region is a couple of voxels and the
    other source falls outside it, which looks like failure but is an artifact
    of the level, not a property of the data. Scanning instead reports how much
    of the probability mass stays resolved into two distinct blobs: higher is
    better, and "no level works" is a genuine merge.

    Returns (best_level or None, verdict).
    """
    best = None
    both_in_one = False
    for L in levels:
        mask = lvl <= L
        if not (mask[a] and mask[b]):
            continue
        lab = _island_labels(dp, mask, radius_mm)
        if lab[a] != lab[b]:
            best = float(L)
        else:
            both_in_one = True
    if best is not None:
        return best, f'SEPARATE to {best:.0%}'
    return None, ('MERGED' if both_in_one else 'ONE SOURCE LOST')


def _draw_head(ax, dp, z_mm, electrodes=True):
    """Overlay the BEM layers (and nearby electrodes) at this slice height."""
    import numpy as np
    styles = [('#111111', 1.4, 1.0), ('#777777', 1.0, 0.8), ('#777777', 1.0, 0.5)]
    for layer, (col, lw, alpha) in enumerate(styles):
        ring = dp.layer_outline_mm(layer, z_mm)
        if ring is not None:
            ax.plot(ring[:, 0], ring[:, 1], color=col, lw=lw, alpha=alpha, zorder=4)
    if electrodes and dp.electrode_pos_mm is not None:
        e = dp.electrode_pos_mm
        # The array is dorsal, so most electrodes sit above any source slice;
        # only draw those close to this plane rather than implying they lie in it.
        near = np.abs(e[:, 2] - z_mm) < 4.0
        if near.any():
            ax.scatter(e[near, 0], e[near, 1], s=13, marker='o', facecolors='none',
                       edgecolors='#0072B2', linewidths=0.9, alpha=0.85, zorder=5)


def fig_geometry(dp, out_path):
    """The head model the posterior is computed in: BEM layers, electrodes, grid."""
    import numpy as np
    pos, e = dp.positions_mm, dp.electrode_pos_mm
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.9))
    views = [(0, 1, 'X  left-right (mm)', 'Y  anterior-posterior (mm)', 'A  Axial (from above)'),
             (1, 2, 'Y  anterior-posterior (mm)', 'Z  dorsal-ventral (mm)', 'B  Sagittal (from the side)'),
             (0, 2, 'X  left-right (mm)', 'Z  dorsal-ventral (mm)', 'C  Coronal (from the front)')]
    for ax, (i, j, xl, yl, title) in zip(axes, views):
        for layer, (col, lw, alpha, name) in enumerate(
                [('#111111', 1.6, 1.0, 'brain (inner BEM)'),
                 ('#777777', 1.1, 0.85, 'skull'), ('#bbbbbb', 1.1, 0.85, 'scalp')]):
            rr = dp.bem_surfaces_mm[layer]
            centre = 0.5 * (rr.max(axis=0) + rr.min(axis=0))
            semi = 0.5 * (rr.max(axis=0) - rr.min(axis=0))
            ang = np.linspace(0, 2 * np.pi, 300)
            ax.plot(centre[i] + semi[i] * np.cos(ang), centre[j] + semi[j] * np.sin(ang),
                    color=col, lw=lw, alpha=alpha, label=name if ax is axes[0] else None)
        ax.scatter(pos[:, i], pos[:, j], s=3, c='#E69F00', alpha=0.45, edgecolors='none',
                   label=f'source grid ({len(pos)} pts)' if ax is axes[0] else None)
        ax.scatter(e[:, i], e[:, j], s=34, marker='o', facecolors='none',
                   edgecolors='#0072B2', linewidths=1.4,
                   label=f'electrodes ({len(e)})' if ax is axes[0] else None)
        ax.set_xlabel(xl); ax.set_ylabel(yl)
        ax.set_title(title, loc='left', fontweight='bold', fontsize=11)
        ax.set_aspect('equal'); ax.grid(True, color='#eee')
    axes[0].legend(frameon=False, fontsize=8, loc='upper center',
                   bbox_to_anchor=(0.5, -0.16), ncol=2)
    sigma = ', '.join(f'{c:g}' for c in dp.bem_conductivities)
    fig.suptitle('The head model everything is computed in — 3-layer ellipsoid BEM, '
                 f'{len(e)}-channel dorsal array\n'
                 f'conductivities (S/m): {sigma}   ·   posterior grid {dp.spacing_mm} mm '
                 f'({len(pos)} positions inside the brain)', fontweight='bold', fontsize=12)
    fig.tight_layout(rect=[0, 0, 1, 0.90])
    fig.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close(fig)


def fig_series(ts, dp, idx_pool, out_path, seed, level=0.5,
               separations=(2.0, 3.0, 4.0, 6.0, 8.0), snrs=(20.0, 10.0, 0.0)):
    """Separation x SNR grid of the two-source posterior, with a non-overlap test."""
    import numpy as np
    pos = dp.positions_mm
    rng = np.random.default_rng(seed)
    radius = 1.6 * dp.spacing_mm

    # One pair per separation, held fixed down the SNR column, so a column shows
    # the effect of noise alone rather than the effect of a different pair.
    pairs = []
    for sep in separations:
        chosen = None
        for _ in range(400):
            a = int(rng.choice(idx_pool))
            d = np.linalg.norm(pos[idx_pool] - pos[a], axis=1)
            cand = idx_pool[np.abs(d - sep) <= 0.5]
            cand = cand[cand != a]
            # depth-matched: gain falls steeply with depth, and an unmatched
            # pair fails because the weaker source vanishes, not because the two
            # are too close to tell apart.
            cand = cand[np.abs(pos[cand, 2] - pos[a, 2]) <= 0.5]
            if len(cand):
                chosen = (a, int(cand[0]))
                break
        pairs.append(chosen)

    fig, axes = plt.subplots(len(snrs), len(separations),
                             figsize=(3.05 * len(separations), 3.35 * len(snrs)))
    axes = np.atleast_2d(axes)
    im = None

    for r, snr in enumerate(snrs):
        for c, (sep, pr) in enumerate(zip(separations, pairs)):
            ax = axes[r, c]
            ax.set_xticks([]); ax.set_yticks([])
            if pr is None:
                ax.text(0.5, 0.5, 'no depth-matched\npair at this separation',
                        ha='center', va='center', fontsize=8, color='#888888',
                        transform=ax.transAxes)
                continue
            a, b = pr
            # Pick the orientation pair giving the most balanced sensor
            # contribution. With fixed orientations one dipole can dominate
            # purely through gain, and the panel then shows amplitude imbalance
            # rather than whether the two are resolvable.
            triad = np.eye(3)
            best_bal, o1b, o2b = -1.0, triad[0], triad[1]
            g1s, g2s = dp.leadfield_at(pos[a]), dp.leadfield_at(pos[b])
            for o1 in triad:
                for o2 in triad:
                    n1 = np.linalg.norm(g1s @ o1); n2 = np.linalg.norm(g2s @ o2)
                    bal = min(n1, n2) / max(n1, n2, 1e-30)
                    if bal > best_bal:
                        best_bal, o1b, o2b = bal, o1, o2
            data, sig = ts.simulate_pair(pos[a], pos[b], o1b, o2b, snr,
                                         np.random.default_rng(seed + 17 * r))
            res = ts.analyze(data, sig)
            marg = ts.marginal_density(res['joint']) / 2.0
            lvl = _credible_level_map(marg)

            # Non-overlap test: scan levels rather than fixing one (see
            # _max_separating_level for why a fixed level is misleading here).
            best_L, verdict = _max_separating_level(dp, lvl, a, b, radius)
            sep_ok = best_L is not None

            z0 = 0.5 * (pos[a, 2] + pos[b, 2])
            # Maximum-intensity projection along the dorsal-ventral axis. A
            # single slice hides mass that sits just out of plane, which made
            # panels look empty while the 3D verdict said otherwise. The two
            # sources are depth-matched, so projecting along z does not merge
            # them.
            sl = np.ones(len(pos), bool)
            # Colour = posterior density relative to its own peak, so certainty
            # is highest AT the source and decays outward, which is how a
            # certainty map is read. (The credible-LEVEL map is the cumulative
            # inverse of this — 0% at the source, 100% far away — and reads
            # backwards no matter how it is labelled.) The probability
            # statements are the contours below, not the colour.
            rel = 100.0 * marg / max(marg.max(), 1e-300)
            key = np.round(pos[:, :2], 6)
            uniq, inv = np.unique(key, axis=0, return_inverse=True)
            proj = np.zeros(len(uniq))
            np.maximum.at(proj, inv, rel)
            im = ax.scatter(uniq[:, 0], uniq[:, 1], c=proj, s=64, marker='s',
                            cmap='inferno', vmin=0, vmax=100, edgecolors='none')
            _draw_head(ax, dp, z0)
            for p_ in (pos[a], pos[b]):
                ax.plot(p_[0], p_[1], marker='+', ms=13, mew=2.4, color='#FF2D95',
                        zorder=6)
            ax.set_aspect('equal')
            def _proj_mask(mask):
                out = np.zeros(len(uniq), bool)
                np.logical_or.at(out, inv, mask)
                return out

            m95 = _proj_mask(lvl <= 0.95)
            if m95.any():
                ax.scatter(uniq[m95, 0], uniq[m95, 1], s=64, marker='s',
                           facecolors='none', edgecolors='#00E5FF',
                           linewidths=0.35, alpha=0.5, zorder=2)
            if sep_ok:
                lab = _island_labels(dp, lvl <= best_L, radius)
                for comp in (lab[a], lab[b]):
                    m2 = _proj_mask(lab == comp)
                    if m2.any():
                        ax.scatter(uniq[m2, 0], uniq[m2, 1], s=64, marker='s',
                                   facecolors='none', edgecolors='#39FF14',
                                   linewidths=0.7, zorder=3)
            ax.set_title(f'{np.linalg.norm(pos[a]-pos[b]):.1f} mm  |  {verdict}',
                         fontsize=9, loc='left',
                         color=('#00790f' if sep_ok else '#b03000'),
                         fontweight='bold')
            if c == 0:
                ax.set_ylabel(f'SNR {snr:+.0f} dB', fontweight='bold', fontsize=11)

    cb = fig.colorbar(im, ax=axes, fraction=0.018, pad=0.015)
    cb.set_label('posterior probability density, % of peak\n'
                 '100% = most probable location for a source   ·   0% = ruled out',
                 fontsize=9)
    fig.suptitle(
        'Two sources: up to what credible level do they stay separate blobs?\n'
        'columns = true separation, rows = SNR.  pink + = true sources, '
        'green = the two still-distinct blobs, cyan = 95% credible region',
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
    fig_geometry(dp, out / 'bem_setup.png')
    print(f"Saved: {out / 'bem_setup.png'}")
    fig_series(ts, dp, idx, out / 'two_source_series.png', args.seed)
    print(f"Saved: {out / 'two_source_series.png'}")
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
