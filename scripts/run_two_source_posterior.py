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
    """Delegates to DipolePosterior.credible_level_map — single definition."""
    return DipolePosterior.credible_level_map(density)


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
                          levels=np.arange(0.05, 1.0, 0.05)):
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


def _draw_head(ax, dp, z_mm, electrodes=True, on_dark=False):
    """Overlay the BEM layers (and nearby electrodes) at this slice height."""
    import numpy as np
    styles = ([('#dddddd', 1.5, 1.0), ('#999999', 1.0, 0.7), ('#999999', 1.0, 0.45)]
              if on_dark else
              [('#111111', 1.4, 1.0), ('#777777', 1.0, 0.8), ('#777777', 1.0, 0.5)])
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
                       edgecolors=('#7fd4ff' if on_dark else '#0072B2'),
                       linewidths=0.9, alpha=0.85, zorder=5)


def _brain_silhouette(i, j):
    """
    Exact projected outline of the ACTUAL brain mask in the (i, j) plane.

    Projects the mask along the remaining voxel axis rather than binning the
    voxel centres: the atlas voxels are strongly anisotropic (0.203 x 0.080 x
    0.200 mm), so a 2D histogram of centres renders as stripes and ragged edges
    that look like a defect in the geometry.
    """
    import numpy as np, nibabel as nib
    from pathlib import Path as _P
    import source_localization as _sl
    from source_localization.utils.atlas import get_true_affine
    f = _P(_sl.__file__).parent / 'data' / 'atlas' / 'Atlas_3DRois_brain.nii.gz'
    nii = nib.load(f)
    mask = np.asarray(nii.get_fdata()) > 0
    aff = get_true_affine(nii)
    k = ({0, 1, 2} - {i, j}).pop()
    occ = mask.any(axis=k)                       # (n_i, n_j) in voxel order
    if i > j:
        occ = occ.T
    ax_i = aff[i, i] * np.arange(mask.shape[i]) + aff[i, 3]
    ax_j = aff[j, j] * np.arange(mask.shape[j]) + aff[j, 3]
    return ax_i, ax_j, occ.T.astype(float) if i < j else occ.astype(float)


def fig_geometry(dp, out_path):
    """
    The head model everything is computed in.

    Draws the ACTUAL brain (atlas mask) as well as the BEM layers, because the
    two are not the same and the difference is what makes this figure
    confusing otherwise: the BEM's innermost layer is an ellipsoid fitted to the
    brain and then inflated by `ellipsoid_margin` (1.23), so it extends above
    the electrode plane. Electrodes therefore sit *inside the inflated
    ellipsoid* while still being correctly outside the real brain. Showing only
    the ellipsoids makes that look like a registration error, which is what it
    was first mistaken for.
    """
    import numpy as np
    pos, e = dp.positions_mm, dp.electrode_pos_mm
    fig, axes = plt.subplots(1, 3, figsize=(16, 5.4))
    views = [(0, 1, 'X  left-right (mm)', 'Y  anterior-posterior (mm)',
              'A  Axial (from above)'),
             (1, 2, 'Y  anterior-posterior (mm)', 'Z  dorsal-ventral (mm)',
              'B  Sagittal (from the side)'),
             (0, 2, 'X  left-right (mm)', 'Z  dorsal-ventral (mm)',
              'C  Coronal (from the front)')]

    for ax, (i, j, xl, yl, title) in zip(axes, views):
        first = ax is axes[0]
        # Actual brain, filled — the thing sources can occupy.
        xc, yc, occ = _brain_silhouette(i, j)
        ax.contourf(xc, yc, occ, levels=[0.5, 1.5], colors=['#dceaf7'],
                    alpha=0.75, zorder=0)
        ax.contour(xc, yc, occ, levels=[0.5], colors=['#2a6ea6'],
                   linewidths=1.8, zorder=1)
        # BEM layers, dashed to mark them as the idealized conductor.
        for layer, (col, name) in enumerate(
                [('#111111', 'BEM inner layer ("brain", inflated x1.23)'),
                 ('#888888', 'BEM skull'), ('#bbbbbb', 'BEM scalp')]):
            rr = dp.bem_surfaces_mm[layer]
            centre = 0.5 * (rr.max(axis=0) + rr.min(axis=0))
            semi = 0.5 * (rr.max(axis=0) - rr.min(axis=0))
            ang = np.linspace(0, 2 * np.pi, 300)
            ax.plot(centre[i] + semi[i] * np.cos(ang),
                    centre[j] + semi[j] * np.sin(ang), color=col, lw=1.4,
                    ls='--', zorder=2, label=name if first else None)
        ax.scatter(pos[:, i], pos[:, j], s=3.0, c='#D55E00', alpha=0.55,
                   edgecolors='none', zorder=3,
                   label=f'source grid ({len(pos)} pts, in brain)' if first else None)
        ax.scatter(e[:, i], e[:, j], s=42, marker='o', facecolors='none',
                   edgecolors='#0072B2', linewidths=1.6, zorder=4,
                   label=f'electrodes ({len(e)})' if first else None)
        if j == 2:   # a view that shows depth: mark the array plane
            ax.axhline(e[:, 2].max(), color='#0072B2', ls=':', lw=1.1, zorder=2)
            ax.text(ax.get_xlim()[0], e[:, 2].max(), ' highest electrode',
                    color='#0072B2', fontsize=8, va='bottom')
        ax.set_xlabel(xl); ax.set_ylabel(yl)
        ax.set_title(title, loc='left', fontweight='bold', fontsize=11)
        ax.set_aspect('equal'); ax.grid(True, color='#eee', zorder=0)
        # Same span in every panel, so the views are directly comparable
        # instead of one appearing smaller because its extent is smaller.
        ax.set_xlim(-11, 11); ax.set_ylim(-11, 11)

    axes[0].legend(frameon=False, fontsize=8, loc='upper left',
                   bbox_to_anchor=(0.0, -0.13), ncol=2)
    sigma = ', '.join(f'{c:g}' for c in dp.bem_conductivities)
    vol = len(pos) * dp.spacing_mm ** 3
    fig.suptitle(
        'Head model: 3-layer ellipsoid BEM (dashed) vs the actual brain (shaded)\n'
        f'{len(e)} dorsal electrodes  ·  conductivities {sigma} S/m  ·  '
        f'source grid {dp.spacing_mm} mm, {len(pos)} positions, {vol:.0f} mm³',
        fontweight='bold', fontsize=12)
    fig.text(0.5, -0.13,
             'The BEM inner layer is the brain ellipsoid inflated by '
             'ellipsoid_margin = 1.23, so it rises above the electrodes. That is '
             'expected and harmless — sources are confined to the shaded brain, '
             'which lies entirely below the array.\n'
             'The grid does not quite fill the shaded brain: 2% of brain volume '
             'falls outside the BEM ellipsoid (it is not ellipsoidal), and a 0.93 '
             'margin keeps sources off the boundary where the BEM is '
             'ill-conditioned. Coverage is 532 of 564 mm³ (94.5%).',
             ha='center', fontsize=9, style='italic', color='#555555')
    fig.tight_layout(rect=[0, 0, 1, 0.90])
    fig.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close(fig)


def series_data(ts, dp, idx_pool, seed, out_npz,
                separations=(2.0, 3.0, 4.0, 6.0, 8.0), snrs=(20.0, 10.0, 0.0)):
    """
    Compute the series panels once and persist them.

    The pair scan behind each panel is the expensive part (~2 min for the grid),
    and it was being repeated for every cosmetic change to the figure. Everything
    a panel needs — the marginal density, the pair, the verdict inputs — is saved
    here so redrawing is instant and reproducible.
    """
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

    marg = np.zeros((len(snrs), len(separations), dp.n_positions))
    dens_a = np.zeros_like(marg)
    dens_b = np.zeros_like(marg)
    meta = np.full((len(snrs), len(separations), 5), np.nan)  # a,b,bf,postsep,truesep

    for r, snr in enumerate(snrs):
        for c, pr in enumerate(pairs):
            if pr is None:
                continue
            a, b = pr
            # Most balanced orientation pair: with fixed orientations one dipole
            # can dominate through gain alone, and the panel then shows amplitude
            # imbalance rather than whether the two are resolvable.
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
            marg[r, c] = ts.marginal_density(res['joint']) / 2.0
            # Split the joint mass between the two sources. The pooled marginal
            # cannot answer "how sure are we about THIS source" — its regions
            # are shared, so one source can own the entire innermost band. For
            # each candidate pair, assign its two locations to the two true
            # sources by whichever pairing is closer overall, then accumulate.
            # This uses the known truth, which is legitimate for validation and
            # is what makes a per-source statement possible at all.
            j = res['joint']
            di_a = np.linalg.norm(pos[ts.pair_i] - pos[a], axis=1)
            di_b = np.linalg.norm(pos[ts.pair_i] - pos[b], axis=1)
            dj_a = np.linalg.norm(pos[ts.pair_j] - pos[a], axis=1)
            dj_b = np.linalg.norm(pos[ts.pair_j] - pos[b], axis=1)
            straight = (di_a + dj_b) <= (di_b + dj_a)
            idx_a = np.where(straight, ts.pair_i, ts.pair_j)
            idx_b = np.where(straight, ts.pair_j, ts.pair_i)
            pa = np.zeros(dp.n_positions); pb = np.zeros(dp.n_positions)
            np.add.at(pa, idx_a, j)
            np.add.at(pb, idx_b, j)
            dens_a[r, c] = pa / max(pa.sum(), 1e-300)
            dens_b[r, c] = pb / max(pb.sum(), 1e-300)
            meta[r, c] = [a, b, res['log10_bayes_factor'],
                          res['median_separation_mm'],
                          float(np.linalg.norm(pos[a] - pos[b]))]
            print(f"    SNR {snr:+.0f} dB, {separations[c]:.0f} mm done", flush=True)

    out_npz = Path(out_npz)
    out_npz.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out_npz, marginal=marg, meta=meta,
                        density_a=dens_a, density_b=dens_b,
                        snrs=np.asarray(snrs, float),
                        separations=np.asarray(separations, float),
                        positions_mm=dp.positions_mm,
                        spacing_mm=np.array(dp.spacing_mm))
    return out_npz


def _shell_probabilities(density, positions, centre, edges):
    """
    Probability that the source lies in each distance shell around ``centre``.

    Shells, not cumulative containment. Containment necessarily rises with
    radius — a bigger region holds more mass — so the outermost circle always
    carries the largest number, which reads backwards against the intuition that
    being closer to the source means a higher chance of finding it there. Shell
    mass falls with distance for a well-localized source, so it says what a
    reader expects it to say.
    """
    import numpy as np
    d = np.linalg.norm(positions - centre, axis=1)
    total = max(density.sum(), 1e-300)
    return [float(density[(d >= lo) & (d < hi)].sum() / total)
            for lo, hi in zip(edges[:-1], edges[1:])]


def _neighbourhood_probability(density, positions, radius_mm, tree=None):
    """
    Per voxel, the probability the source lies within ``radius_mm`` of it.

    This is the quantity the colour encodes. A raw density cannot be used: at
    high SNR the posterior collapses onto one voxel, so a linear colour scale
    renders an almost empty panel and a log scale renders numbers that mean
    nothing. Integrating over a small ball keeps a genuine probability (0-100%),
    peaks at the source, decays with distance, and — the point of the figure —
    merges into a single bright region when two sources are too close to be told
    apart.
    """
    import numpy as np
    from scipy.spatial import cKDTree
    from scipy.sparse import coo_matrix, identity

    n = len(positions)
    tree = tree if tree is not None else cKDTree(positions)
    pairs = tree.query_pairs(radius_mm, output_type='ndarray')
    if len(pairs):
        a = coo_matrix((np.ones(len(pairs)), (pairs[:, 0], pairs[:, 1])),
                       shape=(n, n))
        a = a + a.T + identity(n)
    else:
        a = identity(n)
    return np.asarray(a @ density).ravel()


#: Colour scale for the probability field. 'hot' runs black -> red -> yellow ->
#: white, so brightness increases monotonically with probability and the eye
#: reads the peak without needing the colorbar. Swap for 'cool' if the figure is
#: going somewhere that needs a light background.
CMAP = 'hot'


def fig_series(dp, npz_path, out_path, radius_mm=1.0,
               edges=(0.0, 0.5, 1.0, 2.0, 4.0, np.inf)):
    """
    Colour-coded probability of finding each source, and whether they separate.

    Brighter means the source is more likely to be there. Where the two bright
    regions stay apart the sources are resolvable; where they merge into one
    they are not, which the eye reads immediately without any verdict label.
    """
    import numpy as np
    from scipy.spatial import cKDTree

    z = np.load(npz_path)
    meta, snrs, seps, pos = z['meta'], z['snrs'], z['separations'], z['positions_mm']
    dens = (z['density_a'], z['density_b'])
    spacing = float(z['spacing_mm'])
    edges = list(edges)
    tree = cKDTree(pos)

    fig, axes = plt.subplots(len(snrs), len(seps),
                             figsize=(3.25 * len(seps), 3.5 * len(snrs)))
    axes = np.atleast_2d(axes)
    sc = None

    def _label(lo, hi):
        return f'>{lo:g}' if not np.isfinite(hi) else f'{lo:g}-{hi:g}'

    for r in range(len(snrs)):
        for c in range(len(seps)):
            ax = axes[r, c]
            ax.set_xticks([]); ax.set_yticks([])
            if not np.isfinite(meta[r, c, 0]):
                ax.text(0.5, 0.5, 'no depth-matched\npair', ha='center',
                        va='center', fontsize=8, color='#888888',
                        transform=ax.transAxes)
                continue
            idx = (int(meta[r, c, 0]), int(meta[r, c, 1]))

            # Each source's own field; show whichever is larger at each voxel,
            # so both sources reach 100% at their own peak and the merge or
            # separation between them stays visible.
            fields = [_neighbourhood_probability(dens[k][r, c], pos, radius_mm,
                                                 tree) for k in (0, 1)]
            field = np.maximum(fields[0], fields[1])

            key = np.round(pos[:, :2], 6)
            uniq, inv = np.unique(key, axis=0, return_inverse=True)
            proj = np.zeros(len(uniq))
            np.maximum.at(proj, inv, field)
            # Marker area must match the grid pitch or the voxels tile with
            # gaps and the panel reads as a checkerboard rather than a field.
            pts_per_mm = ax.get_window_extent().width / 16.0
            marker_s = (spacing * pts_per_mm * 72.0 / fig.dpi) ** 2 * 1.25
            sc = ax.scatter(uniq[:, 0], uniq[:, 1], c=100 * proj, s=marker_s,
                            marker='s', cmap=CMAP, vmin=0, vmax=100,
                            edgecolors='none', zorder=1)
            _draw_head(ax, dp, 0.5 * (pos[idx[0], 2] + pos[idx[1], 2]),
                       on_dark=True)

            lines = []
            for k in (0, 1):
                probs = _shell_probabilities(dens[k][r, c], pos, pos[idx[k]],
                                             edges)
                ax.plot(pos[idx[k], 0], pos[idx[k], 1], marker='+', ms=12,
                        mew=2.4, color='#00E5FF', zorder=20)
                lines.append(f"{'AB'[k]} " + ' '.join(
                    f'{_label(edges[i], edges[i+1])}:{100*probs[i]:.0f}%'
                    for i in range(len(probs))))

            ax.set_aspect('equal')
            ax.set_xlim(-8, 8); ax.set_ylim(-9, 9)
            ax.set_title(f'{meta[r, c, 4]:.1f} mm apart', fontsize=9.5,
                         loc='left', fontweight='bold')
            ax.text(0.02, 0.02, '\n'.join(lines), transform=ax.transAxes,
                    fontsize=6.4, va='bottom', family='monospace',
                    bbox=dict(fc='white', ec='none', alpha=0.85, pad=1.4))
            if c == 0:
                ax.set_ylabel(f'SNR {snrs[r]:+.0f} dB', fontweight='bold',
                              fontsize=11)

    cb = fig.colorbar(sc, ax=axes, fraction=0.015, pad=0.012)
    cb.set_label(f'probability the source is within {radius_mm:g} mm of this '
                 f'point (%)\nbrighter = more likely', fontsize=9)
    fig.suptitle(
        'Where is each source, and can the two be told apart?\n'
        'Brighter = the source is more likely to be there. Two separate bright '
        'regions = resolvable; one merged region = not.  '
        f'cyan + = true positions.  Grid {spacing:g} mm.',
        fontweight='bold', fontsize=12)
    fig.text(0.5, 0.012,
             'Text gives the probability the source lies in each distance shell '
             'from its TRUE position, in mm — exclusive bands summing to 100% '
             'per source, so probability falls off with distance rather than '
             'accumulating.',
             ha='center', fontsize=8.5, style='italic', color='#555555')
    fig.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close(fig)


def canonical_series_data(ts, dp, idx_pool, seed, out_npz,
                          separations=(2.0, 3.0, 4.0, 6.0, 8.0),
                          snrs=(20.0, 10.0, 0.0), n_pairs=40, n_draws=2,
                          pitch_mm=0.2, half_extent_mm=(8.0, 7.0), tol_mm=0.6):
    """
    The two-source density averaged over many pairs in a common (canonical) frame.

    A single random pair with a single noise draw is idiosyncratic — depth,
    dipole orientation and gain balance make two panels at the same separation
    disagree completely (one merged blob vs two crisp dots). That scatter is
    real pair-to-pair variability, not estimator noise, and it made the per-panel
    figure read as random.

    This removes it by *marginalizing over the pair's placement*. For every
    trial the two sources are assigned to source A and source B (by whichever
    pairing is closer, using the known truth — legitimate for validation), then
    the whole posterior is rotated in the axial plane so the A->B direction lies
    along +x and the pair midpoint sits at the origin. A therefore always lands
    near ``(-d/2, 0)`` and B near ``(+d/2, 0)``. Averaging the rotated densities
    over many pairs x noise draws gives the *typical* shape of the two-source
    posterior at each separation — a smooth, systematic picture of how the two
    blobs merge as ``d`` shrinks.

    The cost is the anatomy: this is displacement space (mm from the pair
    midpoint, along and across the separation axis), not a brain. Depth is
    integrated out by the top-down projection, exactly as the per-panel figure
    already did.

    The accumulation is bilinear onto a fine grid (``pitch_mm``); no smoothing is
    applied to the stored density — the display's within-radius convolution and
    the ring integrals are what make it smooth, so the stored object stays a
    faithful average.
    """
    import numpy as np
    pos = dp.positions_mm
    rng = np.random.default_rng(seed)

    hx, hy = half_extent_mm
    xe = np.arange(-hx, hx + pitch_mm, pitch_mm)
    ye = np.arange(-hy, hy + pitch_mm, pitch_mm)
    xc = 0.5 * (xe[:-1] + xe[1:])
    yc = 0.5 * (ye[:-1] + ye[1:])
    nx, ny = len(xc), len(yc)

    def _deposit(grid, cx, cy, w):
        """Bilinear-add masses ``w`` at continuous coords (cx, cy)."""
        fx = (cx - xc[0]) / pitch_mm
        fy = (cy - yc[0]) / pitch_mm
        i0 = np.floor(fx).astype(int); j0 = np.floor(fy).astype(int)
        dx = fx - i0; dy = fy - j0
        for di, wx in ((0, 1 - dx), (1, dx)):
            ii = i0 + di
            for dj, wy in ((0, 1 - dy), (1, dy)):
                jj = j0 + dj
                ok = (ii >= 0) & (ii < nx) & (jj >= 0) & (jj < ny)
                np.add.at(grid, (jj[ok], ii[ok]), (w * wx * wy)[ok])

    acc_a = np.zeros((len(snrs), len(separations), ny, nx))
    acc_b = np.zeros_like(acc_a)
    n_used = np.zeros((len(snrs), len(separations)), int)
    sep_xy = np.zeros_like(n_used, float)      # mean projected separation
    sep_true = np.zeros_like(n_used, float)    # mean true 3-D separation
    bf_med = np.full((len(snrs), len(separations)), np.nan)

    triad = np.eye(3)
    for c, sep in enumerate(separations):
        # Draw the pair set once and reuse it across SNRs and draws, so a column
        # differs only by noise/SNR, not by which pairs happened to be drawn.
        pairs = []
        for _ in range(n_pairs):
            for _try in range(200):
                a = int(rng.choice(idx_pool))
                d = np.linalg.norm(pos[idx_pool] - pos[a], axis=1)
                cand = idx_pool[np.abs(d - sep) <= tol_mm]
                cand = cand[cand != a]
                # depth-match: gain falls steeply with depth, so an unmatched
                # pair fails because the weaker source vanishes, not because the
                # two are too close to tell apart.
                cand = cand[np.abs(pos[cand, 2] - pos[a, 2]) <= 0.5]
                if len(cand):
                    pairs.append((a, int(rng.choice(cand))))
                    break
        if not pairs:
            continue

        for r, snr in enumerate(snrs):
            bfs = []
            for (a, b) in pairs:
                # Most balanced orientation pair: with fixed orientations one
                # dipole can dominate through gain alone, and the panel then
                # shows amplitude imbalance rather than resolvability.
                g1s, g2s = dp.leadfield_at(pos[a]), dp.leadfield_at(pos[b])
                best_bal, o1b, o2b = -1.0, triad[0], triad[1]
                for o1 in triad:
                    for o2 in triad:
                        n1 = np.linalg.norm(g1s @ o1)
                        n2 = np.linalg.norm(g2s @ o2)
                        bal = min(n1, n2) / max(n1, n2, 1e-30)
                        if bal > best_bal:
                            best_bal, o1b, o2b = bal, o1, o2

                # Canonical rotation: A->B along +x, midpoint at origin (2-D,
                # axial). Depth-matched pairs are nearly horizontal, so the
                # projected separation is ~the true one.
                v = (pos[b] - pos[a])[:2]
                vn = np.linalg.norm(v)
                if vn < 1e-9:
                    continue
                cth, sth = v[0] / vn, v[1] / vn        # rotate by -theta
                mid = 0.5 * (pos[a] + pos[b])[:2]
                rel = pos[:, :2] - mid
                cx = cth * rel[:, 0] + sth * rel[:, 1]
                cy = -sth * rel[:, 0] + cth * rel[:, 1]

                for k in range(n_draws):
                    data, sig = ts.simulate_pair(
                        pos[a], pos[b], o1b, o2b, snr,
                        np.random.default_rng(seed + 991 * r + 7 * k
                                              + 13 * (a + b)))
                    res = ts.analyze(data, sig)
                    bfs.append(res['log10_bayes_factor'])

                    # Split the joint mass between the two true sources, then
                    # accumulate each into the canonical frame. The pooled
                    # marginal cannot say "how sure about THIS source" — its
                    # regions are shared — so assign each candidate pair's two
                    # locations to A and B by the closer pairing first.
                    j = res['joint']
                    di_a = np.linalg.norm(pos[ts.pair_i] - pos[a], axis=1)
                    di_b = np.linalg.norm(pos[ts.pair_i] - pos[b], axis=1)
                    dj_a = np.linalg.norm(pos[ts.pair_j] - pos[a], axis=1)
                    dj_b = np.linalg.norm(pos[ts.pair_j] - pos[b], axis=1)
                    straight = (di_a + dj_b) <= (di_b + dj_a)
                    idx_a = np.where(straight, ts.pair_i, ts.pair_j)
                    idx_b = np.where(straight, ts.pair_j, ts.pair_i)
                    pa = np.zeros(dp.n_positions); pb = np.zeros(dp.n_positions)
                    np.add.at(pa, idx_a, j)
                    np.add.at(pb, idx_b, j)
                    pa /= max(pa.sum(), 1e-300)
                    pb /= max(pb.sum(), 1e-300)
                    _deposit(acc_a[r, c], cx, cy, pa)
                    _deposit(acc_b[r, c], cx, cy, pb)

                sep_xy[r, c] += vn
                sep_true[r, c] += float(np.linalg.norm(pos[a] - pos[b]))
                n_used[r, c] += 1
            if bfs:
                bf_med[r, c] = float(np.median(bfs))
            print(f"    SNR {snr:+.0f} dB, {sep:.0f} mm: {len(pairs)} pairs x "
                  f"{n_draws} draws done", flush=True)

    # Normalize each averaged density back to unit mass per source.
    for r in range(len(snrs)):
        for c in range(len(separations)):
            if n_used[r, c]:
                acc_a[r, c] /= max(acc_a[r, c].sum(), 1e-300)
                acc_b[r, c] /= max(acc_b[r, c].sum(), 1e-300)
                sep_xy[r, c] /= n_used[r, c]
                sep_true[r, c] /= n_used[r, c]

    out_npz = Path(out_npz)
    out_npz.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out_npz, acc_a=acc_a, acc_b=acc_b,
                        x_centers=xc, y_centers=yc, pitch_mm=np.array(pitch_mm),
                        snrs=np.asarray(snrs, float),
                        separations=np.asarray(separations, float),
                        sep_xy=sep_xy, sep_true=sep_true, bf_med=bf_med,
                        n_used=n_used, spacing_mm=np.array(dp.spacing_mm))
    return out_npz


def _disk_within(density, pitch_mm, radius_mm):
    """P(source within ``radius_mm``) per cell, by convolving with a disk."""
    import numpy as np
    from scipy.ndimage import convolve
    rr = int(np.ceil(radius_mm / pitch_mm))
    yy, xx = np.mgrid[-rr:rr + 1, -rr:rr + 1]
    kernel = ((xx ** 2 + yy ** 2) * pitch_mm ** 2 <= radius_mm ** 2).astype(float)
    return convolve(density, kernel, mode='constant', cval=0.0)


def _ring_probabilities(density, xc, yc, centre_xy, edges):
    """Probability mass of ``density`` in each distance shell around a point."""
    import numpy as np
    X, Y = np.meshgrid(xc, yc)
    d = np.hypot(X - centre_xy[0], Y - centre_xy[1])
    total = max(density.sum(), 1e-300)
    return [float(density[(d >= lo) & (d < hi)].sum() / total)
            for lo, hi in zip(edges[:-1], edges[1:])]


def fig_canonical_series(npz_path, out_path, radius_mm=1.0,
                         edges=(0.0, 0.5, 1.0, 2.0, 4.0, np.inf)):
    """
    Two-source posterior, averaged over many pairs into a canonical frame.

    Each panel is the *typical* posterior at a given true separation: every
    trial was rotated so source A sits at ``(-d/2, 0)`` and B at ``(+d/2, 0)``,
    then averaged over pairs x noise draws (see :func:`canonical_series_data`).
    Idiosyncratic pairs wash out, leaving a smooth statement of how the two
    blobs merge as separation shrinks. This is displacement space, in mm from
    the pair midpoint — not a brain.

    Colour is ``P(a source is within radius_mm of this point)`` (the hard-won
    encoding: bounded, peaked at the source, and it visibly merges when the two
    are unresolvable). Text gives per-shell probabilities around each true
    location. Brighter = more likely.
    """
    import numpy as np
    z = np.load(npz_path)
    acc_a, acc_b = z['acc_a'], z['acc_b']
    xc, yc = z['x_centers'], z['y_centers']
    snrs, seps = z['snrs'], z['separations']
    sep_xy, sep_true = z['sep_xy'], z['sep_true']
    bf_med, n_used = z['bf_med'], z['n_used']
    pitch = float(z['pitch_mm'])
    edges = list(edges)
    extent = [xc[0], xc[-1], yc[0], yc[-1]]

    fig, axes = plt.subplots(len(snrs), len(seps),
                             figsize=(3.25 * len(seps), 3.5 * len(snrs)))
    axes = np.atleast_2d(axes)
    im = None

    def _label(lo, hi):
        return f'>{lo:g}' if not np.isfinite(hi) else f'{lo:g}-{hi:g}'

    for r in range(len(snrs)):
        for c in range(len(seps)):
            ax = axes[r, c]
            ax.set_xticks([]); ax.set_yticks([])
            if not n_used[r, c]:
                ax.text(0.5, 0.5, 'no depth-matched\npair', ha='center',
                        va='center', fontsize=8, color='#888888',
                        transform=ax.transAxes)
                continue

            fa = _disk_within(acc_a[r, c], pitch, radius_mm)
            fb = _disk_within(acc_b[r, c], pitch, radius_mm)
            field = np.maximum(fa, fb)
            im = ax.imshow(100 * field, origin='lower', extent=extent,
                           cmap=CMAP, vmin=0, vmax=100, aspect='equal',
                           interpolation='bilinear')

            half = 0.5 * sep_xy[r, c]
            centres = ((-half, 0.0), (half, 0.0))
            dens = (acc_a[r, c], acc_b[r, c])
            lines = []
            for k in (0, 1):
                cx, cy = centres[k]
                ax.plot(cx, cy, marker='+', ms=13, mew=2.4, color='#00E5FF',
                        zorder=20)
                probs = _ring_probabilities(dens[k], xc, yc, centres[k], edges)
                lines.append(f"{'AB'[k]} " + ' '.join(
                    f'{_label(edges[i], edges[i+1])}:{100*probs[i]:.0f}%'
                    for i in range(len(probs))))

            ax.set_xlim(xc[0], xc[-1]); ax.set_ylim(yc[0], yc[-1])
            ax.axhline(0, color='#2a2a2a', lw=0.6, zorder=2)
            ax.set_title(f'{sep_true[r, c]:.1f} mm apart', fontsize=9.5,
                         loc='left', fontweight='bold')
            ax.text(0.5, 0.985, f'log10 BF {bf_med[r, c]:+.1f}',
                    transform=ax.transAxes, ha='center', va='top', fontsize=7,
                    color='#dddddd')
            ax.text(0.02, 0.02, '\n'.join(lines), transform=ax.transAxes,
                    fontsize=6.4, va='bottom', family='monospace',
                    bbox=dict(fc='white', ec='none', alpha=0.85, pad=1.4))
            if c == 0:
                ax.set_ylabel(f'SNR {snrs[r]:+.0f} dB', fontweight='bold',
                              fontsize=11)

    # A shared scale bar, upper-right of the bottom-right panel — clear of the
    # centred Bayes-factor label above and the shell-probability box below.
    axsb = axes[-1, -1]
    y0 = yc[0] + 0.80 * (yc[-1] - yc[0])
    x1 = xc[-1] - 0.12 * (xc[-1] - xc[0])
    axsb.plot([x1 - 2.0, x1], [y0, y0], color='#ffffff', lw=2.5, zorder=25)
    axsb.text(x1 - 1.0, y0 + 0.25, '2 mm', color='#ffffff', ha='center',
              va='bottom', fontsize=7.5, zorder=25)

    cb = fig.colorbar(im, ax=axes, fraction=0.015, pad=0.012)
    cb.set_label(f'P(a source within {radius_mm:g} mm of this point) (%)\n'
                 f'brighter = more likely', fontsize=9)
    fig.suptitle(
        'Typical two-source posterior vs separation — averaged over many pairs '
        'in a canonical frame\n'
        'Every trial rotated so A is at (-d/2, 0) and B at (+d/2, 0), then '
        'averaged over pairs x noise draws.  Two bright regions = resolvable; '
        'one merged region = not.  cyan + = true (canonical) positions.',
        fontweight='bold', fontsize=12)
    fig.text(0.5, 0.012,
             'Displacement space (mm from the pair midpoint; +x is the '
             'separation axis), NOT anatomy — depth is projected out and every '
             "pair's orientation is rotated into a common frame, so idiosyncratic "
             'pairs average away. Text: per-shell probability around each true '
             'position (exclusive bands summing to 100% per source).',
             ha='center', fontsize=8.5, style='italic', color='#555555')
    fig.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close(fig)


# ------------------------------------------------------------ depth-stratified
#: Depth bands (distance to the nearest electrode, mm), from the resolution work.
#: Depth — not SNR — is the binding constraint on localization: gain falls
#: steeply with it and the point-spread flattens, so deep sources reconstruct
#: almost uniformly regardless of noise. Stratifying by depth is what turns "on
#: average, two blobs" into "here is where in the brain the two are resolvable".
DEPTH_BANDS = (('very shallow', 1.2, 2.3), ('shallow', 2.3, 3.5),
               ('mid', 3.5, 4.7), ('deep', 4.7, 7.6))


def _source_depth_mm(dp):
    """Distance from each source to the nearest electrode (the depth axis)."""
    import numpy as np
    e = dp.electrode_pos_mm
    return np.min(np.linalg.norm(dp.positions_mm[:, None, :] - e[None, :, :],
                                 axis=2), axis=1)


def _canonical_cell(ts, dp, pairs, snr, n_draws, noise_base, deposit_a,
                    deposit_b):
    """
    Average one canonical-frame cell over its pairs x noise draws.

    Factored out of :func:`canonical_series_data` so the depth-stratified sweep
    shares exactly the same per-cell computation. ``deposit_a``/``deposit_b`` are
    bound to the (already zeroed) accumulator for this cell.

    Returns ``(bfs, sep_xy_sum, sep_true_sum, n_used)``.
    """
    import numpy as np
    pos = dp.positions_mm
    triad = np.eye(3)
    bfs = []
    sep_xy_sum = sep_true_sum = 0.0
    n_used = 0
    for (a, b) in pairs:
        # Most balanced orientation pair, so a panel shows resolvability rather
        # than one dipole dominating through gain alone.
        g1s, g2s = dp.leadfield_at(pos[a]), dp.leadfield_at(pos[b])
        best_bal, o1b, o2b = -1.0, triad[0], triad[1]
        for o1 in triad:
            for o2 in triad:
                n1 = np.linalg.norm(g1s @ o1); n2 = np.linalg.norm(g2s @ o2)
                bal = min(n1, n2) / max(n1, n2, 1e-30)
                if bal > best_bal:
                    best_bal, o1b, o2b = bal, o1, o2

        v = (pos[b] - pos[a])[:2]
        vn = float(np.linalg.norm(v))
        if vn < 1e-9:
            continue
        cth, sth = v[0] / vn, v[1] / vn        # rotate by -theta -> A->B on +x
        mid = 0.5 * (pos[a] + pos[b])[:2]
        rel = pos[:, :2] - mid
        cx = cth * rel[:, 0] + sth * rel[:, 1]
        cy = -sth * rel[:, 0] + cth * rel[:, 1]

        for k in range(n_draws):
            data, sig = ts.simulate_pair(
                pos[a], pos[b], o1b, o2b, snr,
                np.random.default_rng(noise_base + 7 * k + 13 * (a + b)))
            res = ts.analyze(data, sig)
            bfs.append(res['log10_bayes_factor'])

            j = res['joint']
            di_a = np.linalg.norm(pos[ts.pair_i] - pos[a], axis=1)
            di_b = np.linalg.norm(pos[ts.pair_i] - pos[b], axis=1)
            dj_a = np.linalg.norm(pos[ts.pair_j] - pos[a], axis=1)
            dj_b = np.linalg.norm(pos[ts.pair_j] - pos[b], axis=1)
            straight = (di_a + dj_b) <= (di_b + dj_a)
            idx_a = np.where(straight, ts.pair_i, ts.pair_j)
            idx_b = np.where(straight, ts.pair_j, ts.pair_i)
            pa = np.zeros(dp.n_positions); pb = np.zeros(dp.n_positions)
            np.add.at(pa, idx_a, j)
            np.add.at(pb, idx_b, j)
            pa /= max(pa.sum(), 1e-300)
            pb /= max(pb.sum(), 1e-300)
            deposit_a(cx, cy, pa)
            deposit_b(cx, cy, pb)

        sep_xy_sum += vn
        sep_true_sum += float(np.linalg.norm(pos[a] - pos[b]))
        n_used += 1
    return bfs, sep_xy_sum, sep_true_sum, n_used


def _draw_matched_pairs(rng, pos, idx_pool, sep, n_pairs, tol_mm=0.6,
                        zmatch_mm=0.5):
    """``n_pairs`` depth-matched pairs from ``idx_pool`` at ~``sep`` apart."""
    import numpy as np
    pairs = []
    for _ in range(n_pairs):
        for _try in range(200):
            a = int(rng.choice(idx_pool))
            d = np.linalg.norm(pos[idx_pool] - pos[a], axis=1)
            cand = idx_pool[np.abs(d - sep) <= tol_mm]
            cand = cand[cand != a]
            cand = cand[np.abs(pos[cand, 2] - pos[a, 2]) <= zmatch_mm]
            if len(cand):
                pairs.append((a, int(rng.choice(cand))))
                break
    return pairs


def canonical_depth_data(ts, dp, seed, out_npz,
                         depth_bands=DEPTH_BANDS,
                         separations=(2.0, 3.0, 4.0, 6.0, 8.0),
                         snrs=(20.0, 10.0, 0.0), n_pairs=40, n_draws=2,
                         pitch_mm=0.2, half_extent_mm=(8.0, 7.0)):
    """
    Canonical-frame average stratified by source depth.

    Same construction as :func:`canonical_series_data`, but the pair pool is a
    band of source depth (distance to the nearest electrode) rather than the
    dorsal half of the brain. The result is indexed
    ``[depth_band, snr, separation]`` so one figure can be drawn per SNR showing
    how resolvability falls off with depth.

    The pair set for a given (band, separation) is drawn once and reused across
    SNRs, so a column differs only by noise — and the same band/separation cell
    is the same pairs in every SNR figure, making the three figures directly
    comparable.
    """
    import numpy as np
    pos = dp.positions_mm
    depth = _source_depth_mm(dp)

    hx, hy = half_extent_mm
    xe = np.arange(-hx, hx + pitch_mm, pitch_mm)
    ye = np.arange(-hy, hy + pitch_mm, pitch_mm)
    xc = 0.5 * (xe[:-1] + xe[1:]); yc = 0.5 * (ye[:-1] + ye[1:])
    nx, ny = len(xc), len(yc)

    def _make_deposit(grid):
        def _dep(cx, cy, w):
            fx = (cx - xc[0]) / pitch_mm; fy = (cy - yc[0]) / pitch_mm
            i0 = np.floor(fx).astype(int); j0 = np.floor(fy).astype(int)
            dxf = fx - i0; dyf = fy - j0
            for di, wx in ((0, 1 - dxf), (1, dxf)):
                ii = i0 + di
                for dj, wy in ((0, 1 - dyf), (1, dyf)):
                    jj = j0 + dj
                    ok = (ii >= 0) & (ii < nx) & (jj >= 0) & (jj < ny)
                    np.add.at(grid, (jj[ok], ii[ok]), (w * wx * wy)[ok])
        return _dep

    nb, ns, nc = len(depth_bands), len(snrs), len(separations)
    acc_a = np.zeros((nb, ns, nc, ny, nx)); acc_b = np.zeros_like(acc_a)
    n_used = np.zeros((nb, ns, nc), int)
    sep_xy = np.zeros((nb, ns, nc)); sep_true = np.zeros((nb, ns, nc))
    bf_med = np.full((nb, ns, nc), np.nan)
    depth_med = np.full(nb, np.nan)

    rng = np.random.default_rng(seed)
    for bi, (bname, lo, hi) in enumerate(depth_bands):
        idx_pool = np.where((depth >= lo) & (depth < hi))[0]
        if len(idx_pool):
            depth_med[bi] = float(np.median(depth[idx_pool]))
        for c, sep in enumerate(separations):
            pairs = _draw_matched_pairs(rng, pos, idx_pool, sep, n_pairs)
            if not pairs:
                print(f"    {bname:12s} {sep:.0f} mm: no pair", flush=True)
                continue
            for r, snr in enumerate(snrs):
                base = seed + 991 * r + 100003 * bi
                bfs, sxy, strue, nu = _canonical_cell(
                    ts, dp, pairs, snr, n_draws, base,
                    _make_deposit(acc_a[bi, r, c]),
                    _make_deposit(acc_b[bi, r, c]))
                if nu:
                    acc_a[bi, r, c] /= max(acc_a[bi, r, c].sum(), 1e-300)
                    acc_b[bi, r, c] /= max(acc_b[bi, r, c].sum(), 1e-300)
                    sep_xy[bi, r, c] = sxy / nu
                    sep_true[bi, r, c] = strue / nu
                    n_used[bi, r, c] = nu
                if bfs:
                    bf_med[bi, r, c] = float(np.median(bfs))
            print(f"    {bname:12s} {sep:.0f} mm: {len(pairs)} pairs done",
                  flush=True)

    out_npz = Path(out_npz)
    out_npz.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        out_npz, acc_a=acc_a, acc_b=acc_b, x_centers=xc, y_centers=yc,
        pitch_mm=np.array(pitch_mm), snrs=np.asarray(snrs, float),
        separations=np.asarray(separations, float), sep_xy=sep_xy,
        sep_true=sep_true, bf_med=bf_med, n_used=n_used, depth_med=depth_med,
        depth_names=np.array([b[0] for b in depth_bands]),
        depth_lo=np.array([b[1] for b in depth_bands]),
        depth_hi=np.array([b[2] for b in depth_bands]),
        spacing_mm=np.array(dp.spacing_mm))
    return out_npz


def fig_canonical_depth(npz_path, snr_index, out_path, radius_mm=1.0,
                        edges=(0.0, 0.5, 1.0, 2.0, 4.0, np.inf)):
    """
    One SNR's depth x separation grid of canonical-frame averages.

    Rows are depth bands (shallowest on top, where localization is best); columns
    are true separation. Reads how resolvability collapses with depth: shallow
    rows hold two crisp blobs, deep rows smear into one regardless of separation,
    because gain falls and the point-spread flattens with depth.
    """
    import numpy as np
    z = np.load(npz_path)
    acc_a, acc_b = z['acc_a'], z['acc_b']
    xc, yc = z['x_centers'], z['y_centers']
    snrs, seps = z['snrs'], z['separations']
    sep_xy, sep_true = z['sep_xy'], z['sep_true']
    bf_med, n_used = z['bf_med'], z['n_used']
    dnames, dlo, dhi = z['depth_names'], z['depth_lo'], z['depth_hi']
    dmed = z['depth_med']
    pitch = float(z['pitch_mm'])
    edges = list(edges)
    extent = [xc[0], xc[-1], yc[0], yc[-1]]
    si = int(snr_index)
    nb, nc = len(dnames), len(seps)

    fig, axes = plt.subplots(nb, nc, figsize=(3.25 * nc, 3.4 * nb))
    axes = np.atleast_2d(axes)
    im = None

    def _label(lo, hi):
        return f'>{lo:g}' if not np.isfinite(hi) else f'{lo:g}-{hi:g}'

    for bi in range(nb):
        for c in range(nc):
            ax = axes[bi, c]
            ax.set_xticks([]); ax.set_yticks([])
            if not n_used[bi, si, c]:
                ax.text(0.5, 0.5, 'no depth-matched\npair', ha='center',
                        va='center', fontsize=8, color='#888888',
                        transform=ax.transAxes)
                continue
            fa = _disk_within(acc_a[bi, si, c], pitch, radius_mm)
            fb = _disk_within(acc_b[bi, si, c], pitch, radius_mm)
            im = ax.imshow(100 * np.maximum(fa, fb), origin='lower',
                           extent=extent, cmap=CMAP, vmin=0, vmax=100,
                           aspect='equal', interpolation='bilinear')

            half = 0.5 * sep_xy[bi, si, c]
            centres = ((-half, 0.0), (half, 0.0))
            dens = (acc_a[bi, si, c], acc_b[bi, si, c])
            lines = []
            for k in (0, 1):
                ax.plot(centres[k][0], centres[k][1], marker='+', ms=13,
                        mew=2.4, color='#00E5FF', zorder=20)
                probs = _ring_probabilities(dens[k], xc, yc, centres[k], edges)
                lines.append(f"{'AB'[k]} " + ' '.join(
                    f'{_label(edges[i], edges[i+1])}:{100*probs[i]:.0f}%'
                    for i in range(len(probs))))
            ax.set_xlim(xc[0], xc[-1]); ax.set_ylim(yc[0], yc[-1])
            ax.axhline(0, color='#2a2a2a', lw=0.6, zorder=2)
            if bi == 0:
                ax.set_title(f'{sep_true[bi, si, c]:.1f} mm apart', fontsize=10,
                             loc='center', fontweight='bold')
            ax.text(0.5, 0.985, f'log10 BF {bf_med[bi, si, c]:+.1f}',
                    transform=ax.transAxes, ha='center', va='top', fontsize=7,
                    color='#dddddd')
            ax.text(0.02, 0.02, '\n'.join(lines), transform=ax.transAxes,
                    fontsize=6.4, va='bottom', family='monospace',
                    bbox=dict(fc='white', ec='none', alpha=0.85, pad=1.4))
            if c == 0:
                ax.set_ylabel(f'{dnames[bi]}\n{dlo[bi]:g}-{dhi[bi]:g} mm deep'
                              f'\n(median {dmed[bi]:.1f})', fontweight='bold',
                              fontsize=10)

    axsb = axes[-1, -1]
    if n_used[-1, si, -1]:
        y0 = yc[0] + 0.80 * (yc[-1] - yc[0])
        x1 = xc[-1] - 0.12 * (xc[-1] - xc[0])
        axsb.plot([x1 - 2.0, x1], [y0, y0], color='#ffffff', lw=2.5, zorder=25)
        axsb.text(x1 - 1.0, y0 + 0.25, '2 mm', color='#ffffff', ha='center',
                  va='bottom', fontsize=7.5, zorder=25)

    cb = fig.colorbar(im, ax=axes, fraction=0.015, pad=0.012)
    cb.set_label(f'P(a source within {radius_mm:g} mm of this point) (%)\n'
                 f'brighter = more likely', fontsize=9)
    fig.suptitle(
        f'Two-source resolvability by depth — SNR {snrs[si]:+.0f} dB\n'
        'Rows: source depth (distance to nearest electrode), shallowest on top.  '
        'Columns: true separation.  Two bright regions = resolvable; one merged '
        'region = not.  cyan + = true (canonical) positions.',
        fontweight='bold', fontsize=12)
    fig.text(0.5, 0.01,
             'Each cell is the canonical-frame average over many depth-matched '
             'pairs at that depth and separation (A rotated to (-d/2,0), B to '
             '(+d/2,0)). Displacement space, not anatomy. Depth — not SNR — is '
             'the binding constraint: gain falls and the point-spread flattens '
             'with depth, so deep rows smear together even when well separated.',
             ha='center', fontsize=8.5, style='italic', color='#555555')
    fig.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close(fig)


# ---------------------------------------------------------- resolvability map
# "Resolvable" is two questions, not one, and they need separate thresholds:
#   detection  — are there two sources or one? -> Bayes factor -> P(two | B)
#   resolution — given two, are they at distinguishable locations? -> whether
#                the two per-source 95% credible regions are DISJOINT
# A pair is "resolved" on a trial when BOTH hold: P(two|B) >= 0.9 (strong
# evidence, ~10:1, and close to the calibrated 10%-false-alarm operating point)
# AND the two 95% credible regions do not overlap. The map reports, per depth
# band x separation x SNR, the median P(two|B) (detection confidence) and the
# resolution RATE (fraction of trials meeting the full conjunction).
#
# The spatial test is per trial, not on the pair-averaged density: averaging
# over many pairs inflates each source's 95% region (it folds in pair-to-pair
# scatter), which would make "disjoint" fail even for cleanly resolved pairs.


def _credible_mask_1d(density, level=0.95):
    """Smallest set of positions holding ``level`` of the mass (HPD region)."""
    import numpy as np
    o = np.argsort(density)[::-1]
    cs = np.cumsum(density[o])
    tot = cs[-1] if cs[-1] > 0 else 1e-300
    k = int(np.searchsorted(cs, level * tot)) + 1
    m = np.zeros(len(density), bool)
    m[o[:k]] = True
    return m


def resolvability_map_data(ts, dp, seed, out_npz, depth_bands=DEPTH_BANDS,
                           separations=(2.0, 3.0, 4.0, 6.0, 8.0),
                           snrs=(20.0, 10.0, 0.0), n_pairs=40, n_draws=1,
                           p_threshold=0.9, cred_level=0.95):
    """
    Per (depth band, SNR, separation): detection confidence and resolution rate.

    For each depth-matched pair and noise draw the trial's two-source posterior
    gives (a) log₁₀ Bayes factor → ``P(two | B)`` and (b) the two per-source 95%
    credible regions. The trial is *detected* when ``P(two|B) >= p_threshold``,
    *resolved* when it is detected AND the two regions are disjoint. The cell
    stores median ``P(two|B)`` and the detection/resolution rates over trials.
    """
    import numpy as np
    pos = dp.positions_mm
    depth = _source_depth_mm(dp)
    bf_cut = float(np.log10(p_threshold / (1.0 - p_threshold)))
    triad = np.eye(3)

    nb, ns, nc = len(depth_bands), len(snrs), len(separations)
    det_rate = np.full((nb, ns, nc), np.nan)
    res_rate = np.full((nb, ns, nc), np.nan)
    med_p = np.full((nb, ns, nc), np.nan)
    sep_true = np.zeros((nb, ns, nc))
    n_used = np.zeros((nb, ns, nc), int)
    depth_med = np.full(nb, np.nan)

    rng = np.random.default_rng(seed)
    for bi, (bname, lo, hi) in enumerate(depth_bands):
        idx_pool = np.where((depth >= lo) & (depth < hi))[0]
        if len(idx_pool):
            depth_med[bi] = float(np.median(depth[idx_pool]))
        for c, sep in enumerate(separations):
            pairs = _draw_matched_pairs(rng, pos, idx_pool, sep, n_pairs)
            if not pairs:
                print(f"    {bname:12s} {sep:.0f} mm: no pair", flush=True)
                continue
            for si, snr in enumerate(snrs):
                base = seed + 911 * si + 100003 * bi
                dcount = rcount = ntot = 0
                ps, strue = [], 0.0
                for (a, b) in pairs:
                    g1, g2 = dp.leadfield_at(pos[a]), dp.leadfield_at(pos[b])
                    bb, o1b, o2b = -1.0, triad[0], triad[1]
                    for o1 in triad:
                        for o2 in triad:
                            n1 = np.linalg.norm(g1 @ o1)
                            n2 = np.linalg.norm(g2 @ o2)
                            bal = min(n1, n2) / max(n1, n2, 1e-30)
                            if bal > bb:
                                bb, o1b, o2b = bal, o1, o2
                    for k in range(n_draws):
                        data, sig = ts.simulate_pair(
                            pos[a], pos[b], o1b, o2b, snr,
                            np.random.default_rng(base + 7 * k + 13 * (a + b)))
                        r = ts.analyze(data, sig)
                        lbf = r['log10_bayes_factor']
                        # P(two|B) = 1/(1+10^-lbf); clamp so huge Bayes factors
                        # (log10 > 300 for shallow high-SNR pairs) don't overflow.
                        lc = float(np.clip(lbf, -300.0, 300.0))
                        ps.append(1.0 / (1.0 + 10.0 ** (-lc)))
                        j = r['joint']
                        di_a = np.linalg.norm(pos[ts.pair_i] - pos[a], axis=1)
                        di_b = np.linalg.norm(pos[ts.pair_i] - pos[b], axis=1)
                        dj_a = np.linalg.norm(pos[ts.pair_j] - pos[a], axis=1)
                        dj_b = np.linalg.norm(pos[ts.pair_j] - pos[b], axis=1)
                        st = (di_a + dj_b) <= (di_b + dj_a)
                        ia = np.where(st, ts.pair_i, ts.pair_j)
                        ib = np.where(st, ts.pair_j, ts.pair_i)
                        pa = np.zeros(dp.n_positions)
                        pb = np.zeros(dp.n_positions)
                        np.add.at(pa, ia, j)
                        np.add.at(pb, ib, j)
                        detected = lbf >= bf_cut
                        disjoint = not (_credible_mask_1d(pa, cred_level)
                                        & _credible_mask_1d(pb, cred_level)).any()
                        dcount += int(detected)
                        rcount += int(detected and disjoint)
                        ntot += 1
                    strue += float(np.linalg.norm(pos[a] - pos[b]))
                det_rate[bi, si, c] = dcount / ntot
                res_rate[bi, si, c] = rcount / ntot
                med_p[bi, si, c] = float(np.median(ps))
                sep_true[bi, si, c] = strue / len(pairs)
                n_used[bi, si, c] = ntot
            print(f"    {bname:12s} {sep:.0f} mm: {len(pairs)} pairs done",
                  flush=True)

    out_npz = Path(out_npz)
    out_npz.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        out_npz, det_rate=det_rate, res_rate=res_rate, med_p=med_p,
        sep_true=sep_true, n_used=n_used, depth_med=depth_med,
        snrs=np.asarray(snrs, float),
        separations=np.asarray(separations, float),
        depth_names=np.array([b[0] for b in depth_bands]),
        depth_lo=np.array([b[1] for b in depth_bands]),
        depth_hi=np.array([b[2] for b in depth_bands]),
        p_threshold=np.array(p_threshold), cred_level=np.array(cred_level))
    return out_npz


def fig_resolvability_map(npz_path, out_path, res_verdict=0.5):
    """
    Summary map: where are two sources resolvable, by depth, separation, and SNR?

    One panel per SNR. Cell colour is the median ``P(two | B)`` (detection
    confidence); the dashed contour is the detection threshold. Cells where the
    full conjunction (detected AND spatially disjoint) holds on at least
    ``res_verdict`` of trials are outlined and marked "R" — these are the
    genuinely *resolved* cells. The resolution rate is printed in each cell.
    """
    import numpy as np
    from matplotlib.patches import Rectangle

    z = np.load(npz_path)
    det, res, medp = z['det_rate'], z['res_rate'], z['med_p']
    seps, snrs = z['separations'], z['snrs']
    sep_true = z['sep_true']
    dnames, dlo, dhi, dmed = (z['depth_names'], z['depth_lo'], z['depth_hi'],
                              z['depth_med'])
    p_thr = float(z['p_threshold'])
    nb, ns, nc = medp.shape

    fig, axes = plt.subplots(1, ns, figsize=(5.2 * ns, 1.05 * nb + 1.6),
                             squeeze=False)
    axes = axes[0]
    im = None
    for si in range(ns):
        ax = axes[si]
        M = medp[:, si, :]
        im = ax.imshow(M, origin='upper', cmap='magma', vmin=0, vmax=1,
                       aspect='auto', extent=[-0.5, nc - 0.5, nb - 0.5, -0.5])
        # Detection-threshold contour on the (interpolated) P field.
        try:
            ax.contour(np.arange(nc), np.arange(nb), M, levels=[p_thr],
                       colors='#00E5FF', linewidths=1.8, linestyles='--')
        except Exception:
            pass
        for bi in range(nb):
            for c in range(nc):
                if not np.isfinite(M[bi, c]):
                    continue
                resolved = res[bi, si, c] >= res_verdict
                txt = f"P={M[bi, c]:.2f}\nres {100*res[bi, si, c]:.0f}%"
                ax.text(c, bi, txt, ha='center', va='center', fontsize=7.5,
                        color='white' if M[bi, c] < 0.6 else 'black',
                        fontweight='bold' if resolved else 'normal')
                if resolved:
                    ax.add_patch(Rectangle((c - 0.5, bi - 0.5), 1, 1, fill=False,
                                           edgecolor='#39FF14', lw=2.5, zorder=5))
        ax.set_xticks(range(nc))
        ax.set_xticklabels([f'{sep_true[0, si, c]:.1f}' for c in range(nc)])
        ax.set_yticks(range(nb))
        if si == 0:
            ax.set_yticklabels([f'{dnames[bi]}\n{dmed[bi]:.1f} mm'
                                for bi in range(nb)], fontsize=8)
        else:
            ax.set_yticklabels([])
        ax.set_xlabel('true separation (mm)')
        ax.set_title(f'SNR {snrs[si]:+.0f} dB', fontweight='bold', fontsize=11)

    cb = fig.colorbar(im, ax=axes, fraction=0.02, pad=0.02)
    cb.set_label('median P(two sources | data)', fontsize=9)
    fig.suptitle(
        'Where are two sources resolvable?  Detection (colour) and full '
        'resolution (green outline) by depth, separation, and SNR\n'
        'Colour = P(two|B); cyan dashed = detection threshold P≥%.2f.  Green '
        'outline = RESOLVED on ≥%.0f%% of trials (detected AND the two 95%% '
        'credible regions disjoint).' % (p_thr, 100 * res_verdict),
        fontweight='bold', fontsize=11, y=1.08)
    fig.text(0.5, -0.02,
             'Depth is distance to the nearest electrode (rows, shallowest on '
             'top). "res %" is the resolution rate: the fraction of trials that '
             'are both detected as two and spatially separated into disjoint '
             'credible regions. Detection is common; full resolution is confined '
             'to shallow sources at good SNR.',
             ha='center', fontsize=8.5, style='italic', color='#555555')
    fig.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--pipeline-dir', required=True)
    ap.add_argument('--output-dir', required=True)
    ap.add_argument('--canonical-pairs', type=int, default=40,
                    help='pairs averaged into each canonical-frame series panel; '
                         'more washes out pair-to-pair idiosyncrasy (cost is '
                         'linear: pairs x draws x cells pair scans)')
    ap.add_argument('--canonical-draws', type=int, default=2,
                    help='noise draws per pair in the canonical-frame average')
    ap.add_argument('--series-spacing-mm', type=float, default=0.5,
                    help='DEPRECATED — the per-panel series figure it fed is '
                         'superseded by the canonical-frame average, which runs '
                         'on --spacing-mm. Kept so old invocations still parse.')
    ap.add_argument('--spacing-mm', type=float, default=1.0,
                    help='pair scan is quadratic in position count; 1.0 mm '
                         'gives ~760 positions / ~288k pairs')
    ap.add_argument('--n-trials', type=int, default=30)
    ap.add_argument('--seed', type=int, default=4)
    ap.add_argument('--replot', action='store_true',
                    help='redraw every figure from cached data — no forward '
                         'solve, no pair scan, no simulation')
    ap.add_argument('--figure-only', action='store_true',
                    help='alias for --replot (kept for existing invocations)')
    ap.add_argument('--depth-percentile', type=float, default=60.0,
                    help='restrict sources to the dorsal fraction above this '
                         'z-percentile, so depth (and therefore gain) is '
                         'roughly matched within a pair')
    ap.add_argument('--depth-series', action='store_true',
                    help='instead of the pooled figures, produce one '
                         'depth x separation figure per SNR (resolvability '
                         'stratified by source depth)')
    ap.add_argument('--resolvability-map', action='store_true',
                    help='produce the resolvability summary map: detection '
                         'confidence P(two|B) and the resolution rate (detected '
                         'AND credible regions disjoint) by depth x separation, '
                         'one panel per SNR')
    args = ap.parse_args()

    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)

    replot = args.replot or args.figure_only
    dp = DipolePosterior.from_pipeline_dir(args.pipeline_dir,
                                           spacing_mm=args.spacing_mm,
                                           cache_dir=out / 'cache')
    ts = TwoSourcePosterior(dp)
    print(f"  {dp.n_positions} positions, {ts.n_pairs} pairs")

    if args.depth_series:
        sp = args.spacing_mm
        npz = (out / 'cache' /
               f'canonical_depth_{sp:g}mm_p{args.canonical_pairs}'
               f'_d{args.canonical_draws}.npz')
        if not replot or not npz.exists():
            print(f"Computing depth-stratified canonical average at {sp:g} mm "
                  f"({len(DEPTH_BANDS)} bands x {len(SNRS)} SNR x "
                  f"{len(SEPARATIONS)} sep, {args.canonical_pairs} pairs x "
                  f"{args.canonical_draws} draws)...", flush=True)
            canonical_depth_data(ts, dp, args.seed, npz,
                                 separations=SEPARATIONS, snrs=SNRS,
                                 n_pairs=args.canonical_pairs,
                                 n_draws=args.canonical_draws)
        for si, snr in enumerate(SNRS):
            tag = f'{snr:+.0f}dB'.replace('+', 'p').replace('-', 'm')
            fp = out / f'two_source_series_depth_{tag}.png'
            fig_canonical_depth(npz, si, fp)
            print(f"Saved: {fp}")
        return

    if args.resolvability_map:
        sp = args.spacing_mm
        npz = (out / 'cache' /
               f'resolvability_map_{sp:g}mm_p{args.canonical_pairs}'
               f'_d{args.canonical_draws}.npz')
        if not replot or not npz.exists():
            print(f"Computing resolvability map at {sp:g} mm "
                  f"({len(DEPTH_BANDS)} bands x {len(SNRS)} SNR x "
                  f"{len(SEPARATIONS)} sep, {args.canonical_pairs} pairs x "
                  f"{args.canonical_draws} draws)...", flush=True)
            resolvability_map_data(ts, dp, args.seed, npz,
                                   separations=SEPARATIONS, snrs=SNRS,
                                   n_pairs=args.canonical_pairs,
                                   n_draws=args.canonical_draws)
        fp = out / 'two_source_resolvability_map.png'
        fig_resolvability_map(npz, fp)
        print(f"Saved: {fp}")
        return

    pos = dp.positions_mm
    idx = np.where(pos[:, 2] >= np.percentile(pos[:, 2],
                                              args.depth_percentile))[0]
    print(f"  restricted to {len(idx)} dorsal positions")

    results = {}
    if replot:
        with open(out / 'two_source_posterior.json') as f:
            results = {float(k): v for k, v in json.load(f).items()}
    for snr in (() if replot else SNRS):
        print(f"  SNR {snr:+.0f} dB ...", flush=True)
        results[snr] = run_condition(ts, dp, idx, snr, SEPARATIONS,
                                     args.n_trials, args.seed)

    fig_main(results, ts, dp, idx, out / 'two_source_posterior.png', args.seed)
    fig_geometry(dp, out / 'bem_setup.png')
    print(f"Saved: {out / 'bem_setup.png'}")

    # The series is the canonical-frame average (see canonical_series_data): the
    # typical two-source posterior at each separation, with pair-to-pair
    # idiosyncrasy averaged out. It runs on the same coarse grid as the pair
    # scan above — the average needs many pairs x draws, so a fine grid (whose
    # scan is quadratic) is not affordable here; smoothness comes from averaging,
    # not from grid resolution.
    sp = args.spacing_mm
    series_npz = (out / 'cache' /
                  f'canonical_series_{sp:g}mm_p{args.canonical_pairs}'
                  f'_d{args.canonical_draws}.npz')
    if not replot or not series_npz.exists():
        print(f"Computing canonical-frame average at {sp:g} mm "
              f"({args.canonical_pairs} pairs x {args.canonical_draws} draws "
              f"per cell)...", flush=True)
        canonical_series_data(ts, dp, idx, args.seed, series_npz,
                              separations=SEPARATIONS, snrs=SNRS,
                              n_pairs=args.canonical_pairs,
                              n_draws=args.canonical_draws)
    fig_canonical_series(series_npz, out / 'two_source_series.png')
    print(f"Saved: {out / 'two_source_series.png'}")
    if not replot:
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
