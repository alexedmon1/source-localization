"""Parcel-subspace ROI extraction: every parcel as its leadfield patterns, one inverse.

Replaces `inverse_solution` + `roi_extraction` when
``source_space.source_sampling: parcel_subspace``. Uses the same dense pool as Monte Carlo
sampling (:func:`.monte_carlo_roi.pool_leadfield`), but instead of averaging many sparse draws it
represents each parcel by the top ``n_patterns`` patterns of its pool leadfield and solves one
inverse over all of them (:mod:`source_localization.source_space.parcel_subspace`, which carries
the rationale and the measurements).

Config (``source_space.parcel_subspace``; pool keys fall back to ``monte_carlo``'s)::

    parcel_subspace:
      n_patterns: 3        # patterns per parcel
      method: MNE          # MNE | sLORETA
      pool_spacing_mm: 0.2

Output. Each parcel has up to ``n_patterns`` component series. Provisional, untested for evoked
measures: the signed series is the dominant pattern's output, the magnitude series the
root-sum-square over the components. All components are saved
(``step6_roi_components.pkl``) and so is the operator (``parcel_subspace_operator.npz``), so the
choice can be revisited without re-running. ROI-only, like Monte Carlo: no grid is solved.
"""
from __future__ import annotations

import numpy as np

from ..source_space.parcel_subspace import (DEFAULT_METHOD, DEFAULT_N_PATTERNS,
                                            build_subspace_operator)
from ..source_space.pool import pool_config
from .monte_carlo_roi import pool_leadfield


def run(config, previous_outputs):
    """Build the parcel-subspace ROI operator and apply it to the recording."""
    print("Parcel-Subspace ROI Extraction")
    cfg = pool_config(config)
    n_patterns = int(cfg.get("n_patterns", DEFAULT_N_PATTERNS))
    method = str(cfg.get("method", DEFAULT_METHOD))
    inv_cfg = config.get("inverse", {})
    lambda2 = inv_cfg.get("lambda2") or 1.0 / float(inv_cfg.get("snr", 3.0)) ** 2

    epochs = previous_outputs["epochs"]
    pool = pool_leadfield(config, previous_outputs)
    G, labels = pool["G"], pool["labels"]
    n_pool = G.shape[1]
    print(f"    Pool: {n_pool:,} sources, {sum(1 for x in labels if x):,} labelled")
    print(f"    {n_patterns} pattern(s) per parcel, one {method} inverse over all of them")

    operator, parcels, components, report = build_subspace_operator(
        G, labels, n_patterns=n_patterns, lambda2=lambda2, method=method)
    print(f"    Operator: {operator.shape[0]} components ({len(parcels)} parcels) "
          f"x {operator.shape[1]} channels")
    low = sorted((report[p]["energy_captured"], p) for p in parcels if report[p]["energy_captured"] < 0.9)
    if low:
        print(f"    {len(low)} parcel(s) whose {n_patterns} pattern(s) carry under 90% of their leadfield energy:")
        for e, p in low:
            print(f"          {p} ({e:.0%})")

    data = epochs.get_data()                                      # (n_epochs, n_chan, n_times)
    Y = operator @ data.transpose(1, 0, 2).reshape(data.shape[1], -1)
    comps = {p: Y[components[p]] for p in parcels}                # (k, n_samples) per parcel
    roi_stcs_signed = {p: comps[p][0] for p in parcels}
    roi_stcs_magnitude = {p: np.sqrt((comps[p] ** 2).sum(axis=0)) for p in parcels}

    if config["outputs"].get("save_intermediate", True):
        import json

        from ..utils.export_set import export_roi_to_set
        from ..utils.io_utils import get_data_dir, get_output_variants, save_pickle

        data_dir = get_data_dir(config)
        variants = get_output_variants(config)
        save_pickle(roi_stcs_signed, data_dir / "step6_roi_timeseries_signed.pkl")
        save_pickle(comps, data_dir / "step6_roi_components.pkl")
        np.savez(data_dir / "parcel_subspace_operator.npz", operator=operator,
                 parcels=np.array(parcels), channels=np.array(epochs.ch_names),
                 component_parcel=np.array([p for p in parcels for _ in components[p]]))
        (data_dir / "parcel_subspace_report.json").write_text(json.dumps(
            {"n_patterns": n_patterns, "method": method, "lambda2": lambda2,
             "pool_sources": int(n_pool), "orientation": pool["orientation"],
             "parcels": parcels, "per_parcel": report}, indent=2, default=float))
        if "signed" in variants:
            export_roi_to_set(roi_stcs_signed, sfreq=epochs.info["sfreq"],
                              output_path=data_dir / "roi_timeseries_signed.set",
                              subject_id="source_localized_signed")
        if "magnitude" in variants:
            export_roi_to_set(roi_stcs_magnitude, sfreq=epochs.info["sfreq"],
                              output_path=data_dir / "roi_timeseries_magnitude.set",
                              subject_id="source_localized_magnitude")
        print(f"    Saved: {data_dir / 'roi_timeseries_signed.set'}")

    return {
        "roi_stcs": roi_stcs_signed,
        "roi_stcs_magnitude": roi_stcs_magnitude,
        "roi_stcs_signed": roi_stcs_signed,
        "roi_labels": parcels,
        "roi_source_mapping": {p: [] for p in parcels},
        "parcel_subspace_report": report,
        "parcel_subspace_operator": operator,
        "parcel_subspace_components": components,
    }
