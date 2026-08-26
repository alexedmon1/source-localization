"""Monte Carlo ROI extraction: many sparse grids instead of one arbitrary grid.

Replaces `inverse_solution` + `roi_extraction` when
``source_space.source_sampling: monte_carlo``.

A source space is a discrete approximation of a continuous current
distribution, and a deployed grid is one arbitrary placement of it. Its phase is
not anatomical -- the Cartesian grid is anchored to voxel 0 of the volume array,
the corner of a bounding box -- and which small parcels get sampled depends on
it. Sweeping the Cartesian offset across a single cell moves coverage between 29
and 32 parcels, with `Auditory_L` sampled in 9 of 10 offsets and `Auditory_R` in
9 of 10, but not the same nine.

This integrates over placement instead of picking one: draw K sparse
configurations from a dense pool, and average the ROI operators they induce.

It is nearly free, because the parcel time series is linear in the sensor data::

    y_p = mean_{i in p} (W_norm[i, :] . B) = (mean_{i in p} W_norm[i, :]) . B

so averaging ROI time series over K draws is algebraically identical to
averaging the K operators and applying the result once. Nothing per-draw is
materialised, no vertex-level estimate is built, and the operator depends only
on (montage, BEM, source pool) -- not on the recording -- so it is computed once
and reused for every subject.

Measured against a dense 0.20 mm pool standing in for continuous truth, at 160
sources and K=100: median SNR gain 1.14x, auditory ~1.20x. K converges by 100;
K=200 and K=400 are identical. 80 sources per draw is past a cliff where parcels
start dropping out of draws entirely.

**ROI-only by construction.** No single grid is ever solved, so there is no
vertex set for a vertex-level analysis to refer to. That is a feature rather
than a gap: a vertex-level result was always a statement about one arbitrary
discretisation, and this makes that impossible to ignore rather than impossible
to see. Requesting vertex output alongside `monte_carlo` raises.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np

from ..source_space.realizations import build_roi_operator


def _labels_for_pool(config, previous_outputs, n_pool):
    """Parcel name per pool source, by the same rule roi_extraction uses."""
    import json

    import nibabel as nib

    from ..utils.atlas import get_true_affine
    from .roi_extraction import map_sources_to_rois_nearest

    pkg = Path(__file__).resolve().parent.parent
    mapping = json.loads((pkg / config['inputs']['roi_mapping']).read_text())
    rois = mapping.get('rois', mapping)
    skip = {'Background', 'Exterior'}
    label_to_roi = {
        int(k): v.get('name', f'ROI_{k}')
        for k, v in rois.items()
        if v.get('name') not in skip
    }

    src = previous_outputs['src']
    if src is not None and len(src) > 0 and 'roi_assignments' in src[0]:
        assign = np.asarray(src[0]['roi_assignments'])[:n_pool]
        return [label_to_roi.get(int(a)) for a in assign]

    coords = previous_outputs['source_coords_mm'][:n_pool]
    nii = nib.load(pkg / config['inputs']['brain_labels'])
    mapping_by_roi = map_sources_to_rois_nearest(
        coords, nii.get_fdata(), get_true_affine(nii), label_to_roi)
    labels = [None] * n_pool
    for roi_name, idxs in mapping_by_roi.items():
        for i in idxs:
            if i < n_pool:
                labels[i] = roi_name
    return labels


def run(config, previous_outputs):
    """Build the Monte Carlo ROI operator and apply it to the recording."""
    print("Monte Carlo ROI Extraction")

    mc = (config['source_space'].get('monte_carlo') or {})
    n_sources = int(mc.get('n_sources', 160))
    n_draws = int(mc.get('n_draws', 100))
    seed = int(mc.get('seed', 20260821))

    inv_cfg = config.get('inverse', {})
    lambda2 = inv_cfg.get('lambda2') or 1.0 / float(inv_cfg.get('snr', 3.0)) ** 2

    fwd = previous_outputs['fwd']
    epochs = previous_outputs['epochs']

    # Fixed orientation where the source space carries meaningful normals, so a
    # pool source is one leadfield column. Anything else stays free and is
    # collapsed to its dominant orientation, because the operator average is
    # defined over one column per source.
    from .inverse_solution import apply_orientation_constraint
    orientation = inv_cfg.get('orientation')
    if orientation is None:
        surf = (config.get('source_space') or {}).get('surface') or {}
        anatomical = (config.get('pipeline', {}).get('source_type') == 'surface'
                      and surf.get('method', 'anatomical') == 'anatomical')
        orientation = 'fixed' if anatomical else 'free'

    fwd_c, n_comp, _ = apply_orientation_constraint(
        fwd, orientation=orientation, loose=float(inv_cfg.get('loose', 0.2)),
        verbose=True)
    G = fwd_c['sol']['data']
    if n_comp == 3:
        n_pool = G.shape[1] // 3
        cols = []
        for i in range(n_pool):
            blk = G[:, 3 * i:3 * i + 3]
            u, s, _ = np.linalg.svd(blk, full_matrices=False)
            cols.append(u[:, 0] * s[0])
        G = np.column_stack(cols)
        print(f"    Free orientation collapsed to dominant direction "
              f"({n_pool} sources)")
    n_pool = G.shape[1]

    coords = previous_outputs['source_coords_mm'][:n_pool]
    labels = _labels_for_pool(config, previous_outputs, n_pool)
    n_labelled = sum(1 for x in labels if x)
    print(f"    Pool: {n_pool:,} sources, {n_labelled:,} labelled")
    print(f"    Drawing {n_draws} configurations of {n_sources} sources")

    operator, parcels, report = build_roi_operator(
        G, coords, labels, n_sources=n_sources, k=n_draws, seed=seed,
        lambda2=lambda2)
    print(f"    Operator: {operator.shape[0]} parcels x {operator.shape[1]} channels")

    # Per-parcel gain, and the collinearity that makes some of it unreliable.
    gains = [report[p]['gain'] for p in parcels]
    print(f"    SNR gain vs a single draw: median {np.median(gains):.2f}x, "
          f"range {min(gains):.2f}-{max(gains):.2f}x")
    degenerate = [p for p in parcels if report[p]['collinear_with']]
    if degenerate:
        print(f"    ⚠️  {len(degenerate)} parcel(s) near-collinear with another; "
              f"their individual gain is not meaningful:")
        for p in degenerate:
            print(f"          {p} ~ {report[p]['collinear_with']} "
                  f"(|cos| {report[p]['max_collinearity']:.3f})")

    data = epochs.get_data()                       # (n_epochs, n_chan, n_times)
    roi = operator @ data.transpose(1, 0, 2).reshape(data.shape[1], -1)
    roi = roi.reshape(len(parcels), data.shape[0], data.shape[2])
    roi_stcs_signed = {p: roi[i].reshape(-1) for i, p in enumerate(parcels)}

    if config['outputs'].get('save_intermediate', True):
        import json

        from ..utils.export_set import export_roi_to_set
        from ..utils.io_utils import get_data_dir, get_output_variants, save_pickle

        data_dir = get_data_dir(config)
        variants = get_output_variants(config)
        save_pickle(roi_stcs_signed, data_dir / 'step6_roi_timeseries_signed.pkl')
        (data_dir / 'monte_carlo_report.json').write_text(json.dumps(
            {'n_sources': n_sources, 'n_draws': n_draws, 'seed': seed,
             'pool_sources': int(n_pool), 'orientation': orientation,
             'parcels': parcels, 'per_parcel': report}, indent=2, default=float))
        if 'signed' in variants:
            export_roi_to_set(roi_stcs_signed, sfreq=epochs.info['sfreq'],
                              output_path=data_dir / 'roi_timeseries_signed.set',
                              subject_id='source_localized_signed')
        print(f"    Saved: {data_dir / 'roi_timeseries_signed.set'}")

    return {
        'roi_stcs': roi_stcs_signed,
        'roi_stcs_magnitude': roi_stcs_signed,
        'roi_stcs_signed': roi_stcs_signed,
        'roi_labels': parcels,
        'roi_source_mapping': {p: [] for p in parcels},
        'monte_carlo_report': report,
        'monte_carlo_operator': operator,
    }
