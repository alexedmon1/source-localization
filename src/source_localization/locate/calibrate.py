"""Calibrating location maps for a location space: the procedure behind ``calibration.volumetric`` and ``.surface``.

A raw ensemble posterior is overconfident: its credible regions hold the truth less often than they claim. A map
calibration (temper by T, blur by sigma mm; :class:`~.ensemble_posterior.MapCalibration`) widens it. It holds only in
the domain it was measured in, so a new array, ensemble, grid or kind of positions needs its own; the field locator
refuses to run without one (the Conductor, 2026-11-09). The procedure (bayes-sl ``tb11_calibrate_maps.py``,
``tb12_select_calibration.py``, 2026-10-13):

1. **Simulate** single-source truths in the space: positions uniform over the candidate positions, random
   orientations, leadfields from *fresh* head models drawn from the ensemble's prior but not in the ensemble (so the
   truth's head model is never one the posterior averages over), at several signal-to-noise ratios, with real
   held-out noise. Two independent sets: one to choose on (*calibration*), one to check on (*validation*), with fresh
   truths, fresh head models and the other half of the noise.
2. **Score** each map on a grid of (T, sigma): ``mass_above`` the truth (the posterior mass strictly more probable
   than the truth's position; the truth is inside the alpha credible region iff it is below alpha, so one number per
   map gives coverage at every level).
3. **Select** on the calibration set the (T, sigma) with the smallest worst |coverage - alpha| over alpha in
   {0.5, 0.8, 0.9, 0.95} and over the strata (SNR x depth class): a setting right at one SNR or depth and wrong at
   another is not calibrated.
4. **Check** it on the validation set, and record the domain.

:func:`select_setting` reproduces tb12's choice of ``calibration.volumetric`` (1.5, 0.25 mm) and
``calibration.surface`` (2.5, 0) from tb11's stored scores (bayes-sl ``scripts/f4c_check_calibration.py``).
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, Iterable, Optional, Sequence, Tuple

import numpy as np

from .ensemble_posterior import CalibrationDomain, EnsemblePosterior, MapCalibration, mass_above

__all__ = ["LEVELS", "coverage", "worst_error", "select_setting", "score_maps", "simulate_single_sources",
           "calibrate", "write_array_reference"]

LEVELS = (0.5, 0.8, 0.9, 0.95)


def coverage(mass_above_truth: np.ndarray, levels: Sequence[float] = LEVELS) -> Dict[float, float]:
    """Share of maps whose truth lies in the alpha credible region (mass above the truth < alpha), per level."""
    m = np.asarray(mass_above_truth, float)
    return {a: float(np.mean(m < a)) for a in levels}


def worst_error(mass_above_truth: np.ndarray, strata: np.ndarray, levels: Sequence[float] = LEVELS) -> float:
    """max over strata and levels of |coverage - level| (tb12's loss)."""
    m, s = np.asarray(mass_above_truth, float), np.asarray(strata)
    return max(abs(c - a) for k in np.unique(s) for a, c in coverage(m[s == k], levels).items())


def select_setting(scores: Dict[Tuple[float, float], np.ndarray], strata: np.ndarray,
                   levels: Sequence[float] = LEVELS) -> Tuple[Tuple[float, float], Dict[Tuple[float, float], float]]:
    """The (T, sigma) in ``scores`` (setting -> mass above the truth per map, all on the same maps) with the smallest
    :func:`worst_error`; ties go to the first setting in order of (T, sigma). Returns the setting and every loss."""
    losses = {k: worst_error(v, strata, levels) for k, v in sorted(scores.items())}
    return min(losses, key=losses.get), losses


def score_maps(log_posterior: np.ndarray, truth_index: Sequence[int], positions_mm: np.ndarray,
               settings: Iterable[Tuple[float, float]]) -> Dict[Tuple[float, float], np.ndarray]:
    """Mass above the truth for each map (rows of ``log_posterior``, normalised or not) under each (T, sigma)."""
    lp = np.atleast_2d(log_posterior)
    idx = np.asarray(truth_index, int)
    out = {}
    for T, s in settings:
        p = MapCalibration(T, s).apply(lp, positions_mm)
        out[(float(T), float(s))] = np.array([float(mass_above(p[i], idx[i])) for i in range(len(idx))])
    return out


def simulate_single_sources(G_ens_whitened: np.ndarray, positions_mm: np.ndarray, spacing_mm: float,
                            G_truth_whitened: np.ndarray, noise_columns: np.ndarray, snrs: Sequence[float],
                            n_per_snr: int, rng: np.random.Generator) -> dict:
    """Single-source truths and their raw log posteriors.

    ``G_truth_whitened`` (K_fresh, P, m, 3): leadfields of fresh head models (not in the ensemble), whitened like the
    ensemble's. ``noise_columns`` (n, m): whitened real noise columns (held out from the whitener's fit). For each SNR
    (whitened signal power per channel) ``n_per_snr`` truths: a position uniform over the candidates, a random
    orientation, a random fresh head model, a random noise column. Returns log posteriors (n, P), truth indices, the
    SNR of each map, and the head model used.
    """
    post = EnsemblePosterior(G_ens_whitened, positions_mm, spacing_mm)
    m = G_ens_whitened.shape[2]
    Y, idx, snr_of, model = [], [], [], []
    for snr in snrs:
        for _ in range(n_per_snr):
            i = int(rng.integers(len(positions_mm)))
            k = int(rng.integers(len(G_truth_whitened)))
            o = rng.normal(size=3)
            s = G_truth_whitened[k, i] @ (o / np.linalg.norm(o))
            s *= np.sqrt(snr * m) / np.linalg.norm(s)
            Y.append(s + noise_columns[int(rng.integers(len(noise_columns)))])
            idx.append(i), snr_of.append(snr), model.append(k)
    lp = post.log_posterior(np.array(Y)[:, :, None])
    return {"log_posterior": lp, "truth_index": np.array(idx), "snr": np.array(snr_of), "model": np.array(model)}


def write_array_reference(space, path, name: str, tolerance_mm: float = 0.1) -> Path:
    """Write the space's array (channels in order, electrode positions) as an array reference file for a domain."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({"name": name, "tolerance_mm": tolerance_mm, "channels": list(space.channel_names),
                                "electrodes_mm": np.asarray(space.electrode_pos_mm, float).round(4).tolist(),
                                "description": "array reference written by locate.calibrate"}, indent=1))
    return path


def calibrate(space, whitener, noise_calib: np.ndarray, noise_valid: np.ndarray, fresh_models_calib: Sequence,
              fresh_models_valid: Sequence, snrs: Sequence[float], n_per_snr: int, out_dir,
              temperatures: Sequence[float] = (1.0, 1.25, 1.5, 2.0, 2.5, 3.0),
              blurs_mm: Sequence[float] = (0.0, 0.25, 0.5, 1.0), depth_of=None, seed: int = 0,
              name: str = "calibration.custom") -> Tuple[MapCalibration, dict]:
    """Measure a map calibration for ``space`` (steps 1-4 of the module docstring).

    ``whitener``: a fitted :class:`~.whitening.NoiseWhitener` (fitted on noise not in ``noise_calib`` or
    ``noise_valid``); ``noise_*``: whitened noise columns (n, m) for each set; ``fresh_models_*``: head models from the
    ensemble's prior, none in the ensemble and none shared between the sets; ``depth_of``: optional function position
    index -> depth class for the strata. Writes the array reference and ``<name>.json`` (setting, domain, coverage per
    set and stratum, losses) to ``out_dir`` and returns the calibration, whose domain matches ``space``.
    """
    out_dir = Path(out_dir)
    Gw = whitener.whiten_leadfield(space.G_ens - space.G_ens.mean(2, keepdims=True))
    pos = space.positions_mm
    rng = np.random.default_rng(seed)
    settings = [(float(t), float(b)) for t in temperatures for b in blurs_mm]
    report = {"settings": settings, "sets": {}}
    scored = {}
    for set_name, models, noise in (("calib", fresh_models_calib, noise_calib), ("valid", fresh_models_valid,
                                                                                noise_valid)):
        Gt = np.stack([space.geometry.leadfield(mdl, pos) for mdl in models])
        Gt = whitener.whiten_leadfield(Gt - Gt.mean(2, keepdims=True))
        sim = simulate_single_sources(Gw, pos, space.spacing_mm, Gt, noise, snrs, n_per_snr, rng)
        depth = np.array([depth_of(i) if depth_of else "all" for i in sim["truth_index"]])
        strata = np.array([f"{s}|{d}" for s, d in zip(sim["snr"], depth)])
        scored[set_name] = (score_maps(sim["log_posterior"], sim["truth_index"], pos, settings), strata)
    best, losses = select_setting(*scored["calib"])
    for set_name, (scores, strata) in scored.items():
        report["sets"][set_name] = {k: coverage(scores[best][strata == k]) for k in np.unique(strata)}
    report["chosen"] = {"temperature": best[0], "blur_mm": best[1], "worst_abs_error_calib": losses[best]}
    report["losses"] = {f"{t},{b}": v for (t, b), v in losses.items()}
    ref = write_array_reference(space, out_dir / f"{name}_array.json", name)
    domain = CalibrationDomain(str(ref), space.ensemble.id, float(space.spacing_mm), space.labels["kind"])
    report["domain"] = {"array": str(ref), "ensemble": domain.ensemble, "spacing_mm": domain.spacing_mm,
                        "positions": domain.positions}
    (out_dir / f"{name}.json").write_text(json.dumps(report, indent=1))
    return MapCalibration(best[0], best[1], domain=domain, id=name), report
