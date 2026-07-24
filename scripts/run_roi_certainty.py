"""
ROI-level identification certainty: if the pipeline attributes activity to an
atlas parcel, how much should you believe it — and what does that cost relative
to the fundamental limit?

Two arms on the same simulated data (see :mod:`validation.roi_certainty`):

- **ceiling** — the calibrated posterior on the dense grid. Its headline is not
  an accuracy but a probability: ``P(true parcel | B)``, the honest, calibrated
  "chance the signal is from parcel k", validated by the reliability diagram.
- **deployed** — sLORETA on the 215-point shell, the estimator a user actually
  runs. A point estimate, so it gets a confusion matrix but no probability.

The gap between them is the *total cost of deployment* (source space +
attribution rule + operator), NOT a bare "estimator gap"; see the module.

Metrics reported:

- **exact-match accuracy** — attributed parcel is exactly the true one. Harsh and
  anatomically blind (a contralateral homologue scores the same as the far side
  of the brain), and — as the size/depth probe showed — dominated by *depth*,
  with a latent parcel-size effect only visible once depth is controlled. So it
  is stratified by depth and read next to:
- **distance-tolerance accuracy** ``A(r)`` — attributed parcel within *r* mm
  (centroid-to-centroid) of the true parcel. Grades "how anatomically wrong" a
  miss is; forgives a 1.2 mm midline homologue but not an 8.6 mm lateral one.

Everything is stratified by depth (distance to nearest electrode), the dominant
driver. allen32 only — it tiles grey matter, so nearest-parcel truth is a
sub-voxel nudge (see CLAUDE.md / roi_certainty).

Usage
-----
    python scripts/run_roi_certainty.py \
        --pipeline-dir /path/to/pipeline_shell_ellipsoid \
        --output-dir   /path/to/validation-tests/roi_certainty \
        --n-per-parcel 30

    # redraw all figures from cache, no forward computation:
    python scripts/run_roi_certainty.py --pipeline-dir ... --output-dir ... --replot
"""
import argparse
import json
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from source_localization.validation.posterior import DipolePosterior
import source_localization.validation.roi_certainty as rc

# SNR sweep and depth bands mirror run_two_source_posterior so the ROI results
# sit on the same axes as the resolvability work.
SNRS = (20.0, 10.0, 0.0)
DEPTH_BANDS = (('very shallow', 1.2, 2.3), ('shallow', 2.3, 3.5),
               ('mid', 3.5, 4.7), ('deep', 4.7, 7.6))
MOMENT_STD = 1.0            # calibrated marginal likelihood (not profiled)
DEPLOYED_INVERSE_SNR = 3.0  # sLORETA regularization: the pipeline's fixed default,
                            # used regardless of true data SNR (a user cannot know it)

BAND_COLORS = {'very shallow': '#2166ac', 'shallow': '#4393c3',
               'mid': '#d6604d', 'deep': '#b2182b'}


# --------------------------------------------------------- parcel geometry

def parcel_geometry(atlas, parcel_ids):
    """Centroid-to-centroid distance matrix (mm) and volume (mm^3) per parcel."""
    import nibabel as nib
    from source_localization.config import atlas_input_paths
    from source_localization.utils.atlas import get_true_affine, get_true_voxel_sizes

    pkg = Path(rc.__file__).resolve().parents[1]
    paths = atlas_input_paths(atlas)
    nii = nib.load(str(pkg / paths['brain_labels']))
    vol = np.asarray(nii.get_fdata()).astype(int)
    aff = get_true_affine(nii)
    voxvol = float(np.prod(get_true_voxel_sizes(nii)))

    cent = np.zeros((len(parcel_ids), 3))
    size = np.zeros(len(parcel_ids))
    for i, pid in enumerate(parcel_ids):
        vox = np.argwhere(vol == pid)
        mm = (aff @ np.column_stack([vox, np.ones(len(vox))]).T).T[:, :3]
        cent[i] = mm.mean(0)
        size[i] = len(vox) * voxvol
    D = np.linalg.norm(cent[:, None, :] - cent[None, :, :], axis=2)
    return D, size


# --------------------------------------------------------- data computation

def compute_data(dp, pmap, pipeline_dir, atlas, out_npz, n_per_parcel, seed):
    """
    One pass per (SNR, trial): simulate once, run both arms on that same data.

    Stores per-trial arrays so every figure can be rebuilt from cache: the true
    parcel, each arm's attributed parcel, the posterior mass on the true parcel,
    and the trial depth. Also the static parcel geometry.
    """
    sloreta = rc.sloreta_attributor(pipeline_dir, atlas, snr=DEPLOYED_INVERSE_SNR)
    D, size = parcel_geometry(atlas, pmap.parcel_ids)
    parcel_depth = np.array([pmap.depth_mm[pmap.inside & (pmap.labels == pid)].mean()
                             for pid in pmap.parcel_ids])

    store = {'D': D, 'size': size, 'parcel_depth': parcel_depth,
             'names': np.array(pmap.parcel_names), 'snrs': np.array(SNRS)}

    for snr in SNRS:
        rng = np.random.default_rng(seed)
        truth = rc.sample_truth_positions(pmap, n_per_parcel, rng)
        idx, true_p, depth = truth['idx'], truth['parcel'], truth['depth_mm']
        n = len(idx)
        ceil = np.empty(n, int)
        depl = np.empty(n, int)
        mass = np.empty(n, float)
        probs_all = np.empty((n, pmap.n_parcels))  # full P(parcel|B), for calibration
        for t, gi in enumerate(idx):
            o = rng.normal(size=3)
            data, noise_std = dp.simulate(dp.positions_mm[gi], o, snr, rng,
                                          leadfield=dp._G[gi])
            probs = pmap.aggregate(dp.posterior(data, noise_std,
                                                moment_std=MOMENT_STD))
            probs_all[t] = probs
            ceil[t] = int(probs.argmax())
            mass[t] = probs[true_p[t]]
            depl[t] = sloreta(gi, data, noise_std, rng)
        tag = _tag(snr)
        store[f'true_{tag}'] = true_p
        store[f'ceil_{tag}'] = ceil
        store[f'depl_{tag}'] = depl
        store[f'mass_{tag}'] = mass
        store[f'depth_{tag}'] = depth
        store[f'probs_{tag}'] = probs_all
        print(f"  SNR {snr:+.0f} dB: {n} trials  "
              f"ceiling {(ceil == true_p).mean():.2f}  "
              f"deployed {(depl == true_p).mean():.2f}", flush=True)

    out_npz.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out_npz, **store)
    return out_npz


def _tag(snr):
    return f'{snr:+.0f}dB'.replace('+', 'p').replace('-', 'm')


def _band_of(depth, bands=DEPTH_BANDS):
    for name, lo, hi in bands:
        if lo <= depth < hi:
            return name
    return None


# --------------------------------------------------------- figures

def fig_accuracy_vs_snr(z, out_path):
    """Exact-match and within-2mm accuracy vs SNR, per depth band, both arms."""
    snrs = z['snrs']
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.8))
    for ax, (title, tol) in zip(axes, [('exact match (r = 0)', 0.0),
                                        ('within 2 mm', 2.0)]):
        for band, _, _ in DEPTH_BANDS:
            c = BAND_COLORS[band]
            ceil_acc, depl_acc = [], []
            for snr in snrs:
                tag = _tag(snr)
                true = z[f'true_{tag}']; depth = z[f'depth_{tag}']
                D = z['D']
                m = np.array([_band_of(d) == band for d in depth])
                if m.sum() == 0:
                    ceil_acc.append(np.nan); depl_acc.append(np.nan); continue
                dc = D[true[m], z[f'ceil_{tag}'][m]]
                dd = D[true[m], z[f'depl_{tag}'][m]]
                ceil_acc.append((dc <= tol).mean())
                depl_acc.append((dd <= tol).mean())
            ax.plot(snrs, ceil_acc, '-o', color=c, label=f'{band} · ceiling')
            ax.plot(snrs, depl_acc, '--s', color=c, mfc='white',
                    label=f'{band} · deployed')
        ax.set_xlabel('SNR (dB)'); ax.set_ylabel('accuracy')
        ax.set_title(title); ax.set_ylim(-0.03, 1.03)
        ax.set_xticks(snrs); ax.grid(alpha=0.3)
    axes[0].legend(fontsize=7, ncol=2, loc='upper left')
    fig.suptitle('ROI identification vs SNR — solid = ceiling, dashed = deployed '
                 '(sLORETA); depth is the ceiling', y=1.02)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close(fig)


def fig_tolerance(z, snr, out_path):
    """Distance-tolerance accuracy A(r) at one SNR, both arms, pooled + by band."""
    tag = _tag(snr)
    true = z[f'true_{tag}']; depth = z[f'depth_{tag}']; D = z['D']
    rr = np.linspace(0, 8, 81)

    def curve(attr, mask=None):
        d = D[true, attr]
        if mask is not None:
            d = d[mask]
        return np.array([(d <= r).mean() for r in rr])

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.6))
    axes[0].plot(rr, curve(z[f'ceil_{tag}']), '-', color='#1a1a1a', lw=2,
                 label='ceiling')
    axes[0].plot(rr, curve(z[f'depl_{tag}']), '--', color='#b2182b', lw=2,
                 label='deployed')
    axes[0].set_title(f'pooled  ({snr:+.0f} dB)')
    axes[0].legend()
    for band, _, _ in DEPTH_BANDS:
        m = np.array([_band_of(d) == band for d in depth])
        if m.sum() == 0:
            continue
        axes[1].plot(rr, curve(z[f'depl_{tag}'], m), '-',
                     color=BAND_COLORS[band], label=band)
    axes[1].set_title(f'deployed, by depth  ({snr:+.0f} dB)')
    axes[1].legend(fontsize=8)
    for ax in axes:
        ax.set_xlabel('tolerance r (mm, centroid-to-centroid)')
        ax.set_ylabel('accuracy  A(r)')
        ax.set_ylim(-0.03, 1.03); ax.grid(alpha=0.3)
        ax.axvline(0, color='grey', lw=0.6)
    fig.suptitle('Distance-tolerance accuracy: A(0) = exact match, A(∞) = 1', y=1.01)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close(fig)


def fig_reliability(z, out_path, n_bins=10):
    """
    Ceiling-arm calibration of the parcel probabilities themselves.

    Proper multiclass reliability: over *every* (trial, parcel) pair, bin by the
    predicted ``P(parcel | B)`` and plot the fraction of that bin where the parcel
    really is the truth. On the diagonal, "the pipeline said P(k)=p" means "the
    source is in k with probability p" — the claim that lets a user act on the
    number. (Binning the true parcel's mass vs argmax-win rate would mix two
    different events and is *not* a calibration test.)
    """
    snrs = z['snrs']
    edges = np.linspace(0, 1, n_bins + 1)
    ctr = 0.5 * (edges[:-1] + edges[1:])
    fig, (ax, axh) = plt.subplots(2, 1, figsize=(5.8, 6.8),
                                  gridspec_kw={'height_ratios': [3, 1]}, sharex=True)
    ax.plot([0, 1], [0, 1], 'k--', lw=1, alpha=0.6, label='ideal')
    for snr in snrs:
        tag = _tag(snr)
        probs = z[f'probs_{tag}']                       # (n_trials, n_parcels)
        true = z[f'true_{tag}']
        onehot = np.zeros_like(probs, bool)
        onehot[np.arange(len(true)), true] = True
        p = probs.ravel(); y = onehot.ravel().astype(float)
        b = np.clip((p * n_bins).astype(int), 0, n_bins - 1)
        pred = np.array([p[b == i].mean() if (b == i).any() else np.nan
                         for i in range(n_bins)])
        emp = np.array([y[b == i].mean() if (b == i).any() else np.nan
                        for i in range(n_bins)])
        line, = ax.plot(pred, emp, '-o', label=f'{snr:+.0f} dB')
        cnt = np.array([(b == i).sum() for i in range(n_bins)])
        axh.plot(ctr, cnt, '-', color=line.get_color(), alpha=0.7)
    ax.set_ylabel('empirical  P(parcel is true)')
    ax.set_title('Ceiling calibration — is P(parcel | B) an honest probability?')
    ax.set_xlim(0, 1); ax.set_ylim(0, 1)
    ax.legend(); ax.grid(alpha=0.3)
    axh.set_yscale('log'); axh.set_ylabel('(trial,parcel)\ncount')
    axh.set_xlabel('predicted  P(parcel | B)'); axh.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close(fig)


def fig_mass_by_parcel(z, snr, out_path):
    """The headline deliverable: calibrated 'believe it X%' per parcel, by depth."""
    tag = _tag(snr)
    true = z[f'true_{tag}']; mass = z[f'mass_{tag}']
    names = z['names']; pdepth = z['parcel_depth']
    nP = len(names)
    mean_mass = np.array([mass[true == k].mean() if (true == k).any() else np.nan
                          for k in range(nP)])
    recall = np.array([(z[f'ceil_{tag}'][true == k] == k).mean()
                       if (true == k).any() else np.nan for k in range(nP)])
    order = np.argsort(pdepth)
    y = np.arange(nP)
    fig, ax = plt.subplots(figsize=(7.2, 8.6))
    ax.barh(y, mean_mass[order], color=[
        BAND_COLORS.get(_band_of(pdepth[k]), '#888') for k in order],
        alpha=0.85, label='P(true parcel | B)  (calibrated belief)')
    ax.plot(recall[order], y, 'k.', ms=7, label='exact-match recall (argmax)')
    ax.set_yticks(y)
    ax.set_yticklabels([f'{names[k]}  ({pdepth[k]:.1f} mm)' for k in order],
                       fontsize=7)
    ax.set_xlabel('score')
    ax.set_title(f'Per-parcel ROI belief, ceiling arm  ({snr:+.0f} dB)\n'
                 f'sorted shallow → deep; colour = depth band')
    ax.set_xlim(0, 1); ax.legend(loc='lower right', fontsize=8)
    ax.grid(alpha=0.3, axis='x')
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close(fig)


def fig_confusion(z, snr, out_path):
    """Recall and precision confusion matrices, both arms, parcels depth-ordered."""
    tag = _tag(snr)
    names = z['names']; nP = len(names)
    order = np.argsort(z['parcel_depth'])

    def counts(attr, true):
        c = np.zeros((nP, nP), int)
        np.add.at(c, (true, attr), 1)
        return c[np.ix_(order, order)]

    def norm(c, axis):
        s = c.sum(axis=axis, keepdims=True)
        return np.divide(c, s, out=np.zeros_like(c, float), where=s > 0)

    true = z[f'true_{tag}']
    arms = {'ceiling': counts(z[f'ceil_{tag}'], true),
            'deployed': counts(z[f'depl_{tag}'], true)}
    fig, axes = plt.subplots(2, 2, figsize=(12, 11.5))
    for i, (arm, c) in enumerate(arms.items()):
        for j, (kind, ax_norm) in enumerate([('recall', 1), ('precision', 0)]):
            ax = axes[i, j]
            im = ax.imshow(norm(c, ax_norm), vmin=0, vmax=1, cmap='magma')
            ax.set_title(f'{arm} — {kind}  ({snr:+.0f} dB)')
            ax.set_xticks(range(nP))
            ax.set_yticks(range(nP))
            lbls = [names[k] for k in order]
            ax.set_xticklabels(lbls, rotation=90, fontsize=5)
            ax.set_yticklabels(lbls, fontsize=5)
            ax.set_xlabel('attributed'); ax.set_ylabel('true')
            fig.colorbar(im, ax=ax, fraction=0.046)
    fig.suptitle('Parcels ordered shallow → deep; diagonal = correct', y=1.0)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close(fig)


# --------------------------------------------------------- driver

def print_summary(z):
    print("\n" + "=" * 70)
    print("ROI IDENTIFICATION — exact-match accuracy (ceiling / deployed)")
    print("=" * 70)
    hdr = f"{'depth band':>14}" + ''.join(f"{s:+.0f}dB".rjust(14) for s in z['snrs'])
    print(hdr)
    for band, _, _ in DEPTH_BANDS:
        cells = []
        for snr in z['snrs']:
            tag = _tag(snr)
            true = z[f'true_{tag}']; depth = z[f'depth_{tag}']
            m = np.array([_band_of(d) == band for d in depth])
            if m.sum() == 0:
                cells.append('   -  /   -  '); continue
            c = (z[f'ceil_{tag}'][m] == true[m]).mean()
            d = (z[f'depl_{tag}'][m] == true[m]).mean()
            cells.append(f'{c:.2f} / {d:.2f}')
        print(f"{band:>14}" + ''.join(s.rjust(14) for s in cells))
    for snr in z['snrs']:
        tag = _tag(snr)
        true = z[f'true_{tag}']
        c = (z[f'ceil_{tag}'] == true).mean()
        d = (z[f'depl_{tag}'] == true).mean()
        print(f"  overall {snr:+.0f} dB: ceiling {c:.2f}  deployed {d:.2f}")


def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--pipeline-dir', required=True,
                    help='shell pipeline run (provides the deployed sLORETA forward)')
    ap.add_argument('--output-dir', required=True)
    ap.add_argument('--atlas', default='allen32')
    ap.add_argument('--spacing-mm', type=float, default=0.5)
    ap.add_argument('--n-per-parcel', type=int, default=30)
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--headline-snr', type=float, default=10.0,
                    help='SNR for the single-condition figures (confusion, '
                         'tolerance, per-parcel belief)')
    ap.add_argument('--replot', action='store_true',
                    help='redraw all figures from cached npz; no forward computation')
    args = ap.parse_args()

    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    npz_path = (out / 'cache' /
                f'roi_certainty_{args.atlas}_{args.spacing_mm:g}mm'
                f'_n{args.n_per_parcel}.npz')

    if not args.replot or not npz_path.exists():
        dp = DipolePosterior.from_pipeline_dir(
            args.pipeline_dir, spacing_mm=args.spacing_mm, cache_dir=out / 'cache')
        pmap = rc.ParcelMap(dp.positions_mm, atlas=args.atlas,
                            electrode_pos_mm=dp.electrode_pos_mm)
        print(f"grid={dp.n_positions}  parcels={pmap.n_parcels}  "
              f"atlas={args.atlas}  n/parcel={args.n_per_parcel}")
        compute_data(dp, pmap, args.pipeline_dir, args.atlas, npz_path,
                     args.n_per_parcel, args.seed)

    z = dict(np.load(npz_path, allow_pickle=False))
    hs = args.headline_snr

    fig_accuracy_vs_snr(z, out / 'roi_accuracy_vs_snr.png')
    fig_reliability(z, out / 'roi_reliability.png')
    fig_tolerance(z, hs, out / 'roi_tolerance.png')
    fig_mass_by_parcel(z, hs, out / 'roi_belief_by_parcel.png')
    fig_confusion(z, hs, out / 'roi_confusion.png')
    for f in ['roi_accuracy_vs_snr', 'roi_reliability', 'roi_tolerance',
              'roi_belief_by_parcel', 'roi_confusion']:
        print(f"Saved: {out / (f + '.png')}")

    print_summary(z)


if __name__ == '__main__':
    main()
