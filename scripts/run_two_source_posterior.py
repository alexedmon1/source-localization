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


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--pipeline-dir', required=True)
    ap.add_argument('--output-dir', required=True)
    ap.add_argument('--series-spacing-mm', type=float, default=0.5,
                    help='finer grid used for the per-source series figure; '
                         'the pair scan is quadratic, so this is the expensive '
                         'knob (0.5 mm ~ 4250 positions ~ 9M pairs)')
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
    args = ap.parse_args()

    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)

    replot = args.replot or args.figure_only
    dp = DipolePosterior.from_pipeline_dir(args.pipeline_dir,
                                           spacing_mm=args.spacing_mm,
                                           cache_dir=out / 'cache')
    ts = TwoSourcePosterior(dp)
    print(f"  {dp.n_positions} positions, {ts.n_pairs} pairs")

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

    # The series runs on its own finer grid: it is a demonstration of how
    # precisely a source can be pinned down, and a 1 mm grid floors that at 1 mm.
    sp = args.series_spacing_mm
    series_npz = out / 'cache' / f'series_panels_{sp:g}mm.npz'
    if not replot or not series_npz.exists():
        dp_s = DipolePosterior.from_pipeline_dir(
            args.pipeline_dir, spacing_mm=sp, cache_dir=out / 'cache')
        ts_s = TwoSourcePosterior(dp_s)
        pos_s = dp_s.positions_mm
        idx_s = np.where(pos_s[:, 2] >= np.percentile(pos_s[:, 2],
                                                      args.depth_percentile))[0]
        print(f"Computing series panels at {sp:g} mm "
              f"({dp_s.n_positions} positions, {ts_s.n_pairs} pairs)...",
              flush=True)
        series_data(ts_s, dp_s, idx_s, args.seed, series_npz)
    else:
        dp_s = DipolePosterior.from_pipeline_dir(
            args.pipeline_dir, spacing_mm=sp, cache_dir=out / 'cache',
            verbose=False)
    fig_series(dp_s, series_npz, out / 'two_source_series.png')
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
