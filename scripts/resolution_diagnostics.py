#!/usr/bin/env python
"""
Noise-free resolution diagnostics.

.. warning::
   **Partially retracted.** The merge distances this script originally reported
   (6 / 7 / 10 mm by depth) came from a two-lobe test with no false-positive
   control and are withdrawn — see :func:`merge_distance`. The detector has been
   fixed, but for new work prefer ``validation/two_source.py``, which answers
   the same question with a Bayes factor against a matched one-source null.
   The single-source localization errors below were never affected.

The Monte-Carlo resolvability sweeps (run_resolvability_sweep etc.) mix two very
different things: the intrinsic geometric resolution of the forward+inverse
operator, and the degradation from noise at a chosen SNR. Reported together at a
single SNR they misled us into "can't resolve anything." This script isolates
the *geometric* part — feed each source through the inverse with NO noise — which
is exact and separates the operator's spatial resolution from noise effects.

Reports, by depth:
  - single-source localization error (how well ONE source is placed)
  - point-spread profile (activation vs distance from the true source)
  - two-source merge distance (how far apart two sources must be to show a dip)
and compares localization error to the electrode spacing (does source
localization add spatial information over the raw sensors?).

Usage
-----
python scripts/resolution_diagnostics.py --pipeline-dir /path/to/results \
    [--output-dir DIR]
"""
import argparse
import json
from pathlib import Path

import numpy as np
from scipy.spatial.distance import cdist

from source_localization.validation.robustness import RobustnessTest

INF = float('inf')


def noise_free_recon(test, position_mm):
    eeg, _ = test.simulator.simulate_dipole(
        position_mm=position_mm, amplitude_nAm=50.0, snr_db=INF,
        noise_seed=0, noise_type='white', duration_s=0.2, sfreq=256.0)
    return test._source_activity_norm(test._apply_inverse(eeg))


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--pipeline-dir', required=True)
    ap.add_argument('--output-dir', default=None)
    ap.add_argument('--inverse-method', default='sLORETA')
    ap.add_argument('--inverse-snr', type=float, default=3.0)
    args = ap.parse_args()

    t = RobustnessTest.from_pipeline_dir(
        args.pipeline_dir, inverse_method=args.inverse_method,
        inverse_snr=args.inverse_snr, verbose=False)
    sp, d = t.source_pos_mm, t.source_depths

    # --- single-source localization error, noise-free, per source ---
    errs = np.array([
        np.linalg.norm(sp[int(np.argmax(noise_free_recon(t, sp[i])))] - sp[i])
        for i in range(t.n_sources)
    ])
    el = t.electrode_pos_mm
    D = cdist(el, el)
    np.fill_diagonal(D, np.inf)
    elec_spacing = float(np.median(D.min(1)))

    depth_bins = [('very_shallow', 0, 15), ('shallow', 15, 40),
                  ('mid', 40, 70), ('deep', 70, 100)]
    loc_by_depth = {}
    for label, lo, hi in depth_bins:
        m = (d >= np.percentile(d, lo)) & (d <= np.percentile(d, hi))
        loc_by_depth[label] = {
            'median_error_mm': float(np.median(errs[m])),
            'mean_error_mm': float(errs[m].mean()),
            'depth_range_mm': [float(np.percentile(d, lo)), float(np.percentile(d, hi))],
        }

    # --- point-spread profile + two-source merge distance, per depth ---
    def merge_distance(bin_lo, bin_hi):
        """
        Smallest separation whose noise-free two-source recon shows two lobes.

        NOTE (fix): this originally scored a dip as
        ``activation[midpoint] < 0.8 * min(activation[a], activation[b])``,
        which never required the second location to *be* a lobe. With a flat or
        mislocalized point-spread function that test fires with only ONE source
        present — up to 67% of pairs, noise-free — so it had no false-positive
        control, and the merge distances it produced (6 / 7 / 10 mm by depth)
        were wrong and have been retracted.

        It now uses the prominence-gated ``_detect_two_lobes`` (two prominent
        local maxima plus a saddle), which is the same event used elsewhere and
        can be scored identically on a one-source null.

        Superseded by the two-source posterior
        (``validation/two_source.py``), which replaces this detector with a
        Bayes factor against a matched null. Prefer that for new work.
        """
        idxs = np.where((d >= np.percentile(d, bin_lo)) & (d <= np.percentile(d, bin_hi)))[0]
        for sep in [3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 10.0]:
            dips = []
            for a in idxs:
                cand = [b for b in idxs if b > a
                        and abs(np.linalg.norm(sp[a] - sp[b]) - sep) < 0.7]
                for b in cand[:2]:
                    eeg, meta = t.simulator.simulate_two_dipoles(
                        position1_mm=sp[a], position2_mm=sp[b],
                        amplitude1_nAm=50, amplitude2_nAm=50, snr_db=INF,
                        noise_seed=0, noise_type='white', duration_s=0.2, sfreq=256.0)
                    raw = t._apply_inverse(eeg)
                    dips.append(t._detect_two_lobes(raw)['detected'])
            if dips and np.mean(dips) >= 0.5:
                return float(sep)
        return None

    merge_by_depth = {label: merge_distance(lo, hi) for label, lo, hi in depth_bins}

    report = {
        'pipeline_dir': str(Path(args.pipeline_dir).resolve()),
        'n_sources': int(t.n_sources),
        'n_electrodes': int(el.shape[0]),
        'electrode_spacing_mm': elec_spacing,
        'single_source_localization_noise_free': {
            'overall_median_mm': float(np.median(errs)),
            'overall_mean_mm': float(errs.mean()),
            'by_depth': loc_by_depth,
        },
        'two_source_merge_distance_noise_free_mm': merge_by_depth,
        'interpretation': {
            'localization_beats_electrodes': bool(np.median(errs) <= elec_spacing),
            'note': 'Single-source localization is the validated capability. '
                    'Two-source merge distance is the noise-free geometric '
                    'resolution limit; noise raises it further.',
        },
    }

    print("=" * 70)
    print("NOISE-FREE RESOLUTION DIAGNOSTICS")
    print("=" * 70)
    print(f"sources={t.n_sources}  electrodes={el.shape[0]}  "
          f"electrode spacing={elec_spacing:.2f} mm")
    print(f"\nSingle-source localization (noise-free): "
          f"median {np.median(errs):.2f} mm  mean {errs.mean():.2f} mm")
    print(f"  {'depth bin':<14}{'loc error':>12}{'2-src merge':>13}")
    for label, _, _ in depth_bins:
        loc = loc_by_depth[label]['median_error_mm']
        mg = merge_by_depth[label]
        mgs = f"{mg:.0f} mm" if mg is not None else ">10 mm"
        print(f"  {label:<14}{loc:>9.2f} mm{mgs:>13}")
    verdict = "YES" if np.median(errs) <= elec_spacing else "NO"
    print(f"\nLocalization finer than electrode spacing? {verdict} "
          f"({np.median(errs):.2f} vs {elec_spacing:.2f} mm)")

    if args.output_dir:
        out = Path(args.output_dir)
        out.mkdir(parents=True, exist_ok=True)
        path = out / 'resolution_diagnostics.json'
        with open(path, 'w') as f:
            json.dump(report, f, indent=2)
        print(f"\nSaved: {path}")


if __name__ == '__main__':
    main()
