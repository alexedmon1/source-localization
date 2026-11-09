"""Locating an electrode field: calibrated single-source and mirror-pair posteriors over a head-model ensemble.

The field method's location step (field-method plan, phase F4). A field's *column* is its electrode pattern at the
size the data give it (for an evoked field, the units' responses along the field's time course; for a steady-state
field, the phase-aligned pattern), with *noise columns* of the same construction (sign-flip averages). The locator:

1. fits a noise whitener on the units' noise columns (``whiten.oas`` by default: average reference, pooled shape,
   per-unit scale) and whitens the ensemble's leadfields with it;
2. builds ``locate.single_source`` and ``locate.mirror_pair`` posteriors over the whitened ensemble;
3. reads a whitened column: both calibrated maps, both log evidences and P(pair) (``prior.model_odds_equal``), the
   chosen location (the pair's two members when P(pair) >= the pair rule, else the single peak).

It refuses a location space outside the calibration's measured domain (array, ensemble, grid, positions) unless an
uncalibrated diagnostic map is asked for explicitly (the Conductor, 2026-11-09). Every result carries
:meth:`FieldLocator.provenance`: the catalogue ids and versions of the posteriors, priors, calibration and whitener,
and the space's ensemble and geometry.

Ported from surface-evoked ``localize_fields_bayes.py`` and ``method_crossover.FieldModel`` (setup, locate). The
report helpers (:func:`summarize`, :func:`report_response_svd_field`) reproduce ``localize_fields_bayes.json``.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Callable, Dict, Optional, Sequence

import numpy as np

from .ensemble_posterior import (VOLUMETRIC, EnsemblePosterior, MirrorPairPosterior, mass_above,
                                 model_probabilities)
from .whitening import NoiseWhitener

__all__ = ["FieldLocator", "summarize", "hpd_size", "parcel_mass", "report_response_svd_field", "KINDS", "Units",
           "inner_draws", "lead", "FieldModel", "model_filter_pair", "mean_referenced_leadfield", "region_fields",
           "region_power_db", "region_value_table", "write_region_values"]


def hpd_size(p: np.ndarray, level: float = 0.9) -> np.ndarray:
    """Number of positions in the smallest set holding ``level`` of the mass, per row."""
    q = -np.sort(-np.atleast_2d(p), axis=1)
    return (np.cumsum(q, axis=1) < level).sum(1) + 1


def parcel_mass(p: np.ndarray, labels: np.ndarray, names: Sequence[str], top: int = 6) -> Dict[str, float]:
    """The ``top`` regions by mass, rounded to 4 decimals (positions with label < 0 are unlabelled)."""
    m = np.bincount(labels[labels >= 0], weights=p[labels >= 0], minlength=len(names))
    order = np.argsort(-m)[:top]
    return {names[i]: round(float(m[i]), 4) for i in order if m[i] > 0}


class FieldLocator:
    """Calibrated location of field columns in one :class:`~.space.LocationSpace`.

    Parameters
    ----------
    space : LocationSpace (or an object with ``G_ens``, ``positions_mm``, ``spacing_mm``, ``B_ref`` and
        ``check_calibration``)
    calibration : MapCalibration
        Default ``calibration.volumetric``. Checked against the space's domain.
    whitener : str
        Estimator id suffix (``whiten.<whitener>``): sample, ledoit_wolf, oas (default), diagonal.
    pair_rule : float
        P(pair) at or above which :meth:`locate` returns the pair (default 0.5).
    allow_uncalibrated : bool
        Permit ``calibration.none`` (raw, overconfident maps) for diagnostics only.
    """

    def __init__(self, space, calibration=VOLUMETRIC, whitener: str = "oas", pair_rule: float = 0.5,
                 allow_uncalibrated: bool = False):
        space.check_calibration(calibration, allow_uncalibrated=allow_uncalibrated)
        self.space, self.cal, self.whitener_name, self.pair_rule = space, calibration, whitener, float(pair_rule)
        self.wh = self.one = self.two = None
        self._singles = None

    # -------------------------------------------------------------- fitting
    def fit(self, noise_columns: Sequence[np.ndarray]) -> "FieldLocator":
        """Fit the whitener on one noise block (channels, n_columns) per unit and build both posteriors."""
        self.wh = NoiseWhitener.fit([np.asarray(z, float) for z in noise_columns], estimator=self.whitener_name)
        Gw = self.wh.whiten_leadfield(self.space.G_ens)
        self.one = EnsemblePosterior(Gw, self.space.positions_mm, self.space.spacing_mm)
        self.two = MirrorPairPosterior(Gw, self.space.positions_mm, self.space.spacing_mm)
        self._Gw = Gw
        self._singles = None
        return self

    @property
    def n_whitened(self) -> int:
        return self.wh.n_whitened

    def single_model(self, k: int) -> EnsemblePosterior:
        """``locate.single_source`` under head model k alone (for a deterministic maximum-likelihood fit)."""
        if self._singles is None:
            self._singles = {}
        if k not in self._singles:
            self._singles[k] = EnsemblePosterior(self._Gw[[k]], self.space.positions_mm, self.space.spacing_mm)
        return self._singles[k]

    # -------------------------------------------------------------- reading
    def whiten(self, y: np.ndarray, noise: Optional[np.ndarray] = None, scale: Optional[float] = None) -> np.ndarray:
        """Whitened column: W y divided by ``scale``, or by the RMS of ``noise`` (channels, k) in the referenced space."""
        if scale is None:
            if noise is None:
                raise ValueError("give the column's noise or its scale")
            scale = NoiseWhitener.scale_of(noise, self.space.B_ref)
        return self.wh.W @ np.asarray(y, float) / scale

    def readout(self, Yw: np.ndarray) -> dict:
        """For whitened patterns (n, m) or (n, m, c): calibrated single and pair maps, log evidences, P(pair)."""
        Yw = np.asarray(Yw, float)
        if Yw.ndim == 2:
            Yw = Yw[:, :, None]
        ps, pp = self.one.posterior(Yw, self.cal), self.two.posterior(Yw, self.cal)
        ev1, ev2 = self.one.log_evidence(Yw), self.two.log_evidence(Yw)
        return {"single": ps, "pair": pp, "log_ev_single": ev1, "log_ev_pair": ev2,
                "p_pair": model_probabilities({"one": ev1, "pair": ev2})["pair"]}

    def locate(self, yw: np.ndarray):
        """The chosen location of one whitened column: ([right, left] pair indices if P(pair) >= pair_rule, else
        [single peak]), and P(pair). As ``method_crossover.FieldModel.locate``."""
        r = self.readout(np.asarray(yw, float)[None, :, None])
        ppair = float(r["p_pair"][0])
        if ppair >= self.pair_rule:
            c = self.two.configs[int(r["pair"][0].argmax())]
            return [int(c[0]), int(c[1])], ppair
        return [int(r["single"][0].argmax())], ppair

    def provenance(self) -> dict:
        from ..methods import catalogue_entries
        E = catalogue_entries()
        ids = ["locate.single_source", "locate.mirror_pair", "locate.field_choice", "prior.position_uniform",
               "prior.moment_empirical_bayes", "prior.model_odds_equal", self.space.ensemble.id, self.cal.id,
               f"whiten.{self.whitener_name}"]
        return {"models": {i: E[i].version for i in ids if i in E}, "pair_rule": self.pair_rule,
                "calibration": {"id": self.cal.id, "temperature": self.cal.temperature, "blur_mm": self.cal.blur_mm},
                "space": self.space.provenance() if hasattr(self.space, "provenance") else None}


def summarize(loc: FieldLocator, yw: np.ndarray, labels: np.ndarray, names: Sequence[str],
              level: float = 0.9, region_groups: Optional[Dict[str, Sequence[str]]] = None) -> dict:
    """Location summary of one whitened column: SNR per whitened channel; single source (peak, peak probability,
    ``level`` region volume, region masses); mirror pair (log Bayes factor against one source, P(pair), right member's
    peak, both members' regions, region volume per side, region masses). ``region_groups``: extra named masses (for
    example {"auditory": [Auditory_L, Auditory_R]})."""
    pos, vox = loc.space.positions_mm, loc.space.spacing_mm ** 3
    r = loc.readout(np.asarray(yw, float)[None, :, None])
    ps, pp = r["single"][0], r["pair"][0]
    top1, top2 = int(ps.argmax()), int(pp.argmax())
    pmap = loc.two.position_map(pp) / 2
    plab = np.array([names[i] if i >= 0 else "none" for i in labels])

    def groups(p):
        return {g: round(float(p[np.isin(labels, [names.index(n) for n in members])].sum()), 4)
                for g, members in (region_groups or {}).items()}

    return {"snr_per_whitened_channel": round(float(yw @ yw / loc.n_whitened - 1), 2),
            "single": {"peak_mm": pos[top1].round(2).tolist(), "peak_region": str(plab[top1]),
                       "peak_prob": round(float(ps[top1]), 4),
                       f"hpd{int(level * 100)}_mm3": round(float(hpd_size(ps, level)[0] * vox), 2),
                       "region_mass": parcel_mass(ps, labels, names), **groups(ps)},
            "pair": {"log_bf_pair_vs_one": round(float(r["log_ev_pair"][0] - r["log_ev_single"][0]), 2),
                     "p_pair": round(float(r["p_pair"][0]), 4),
                     "peak_right_mm": loc.two.config_positions_mm[top2].round(2).tolist(),
                     "peak_regions": [str(plab[loc.two.configs[top2, 0]]), str(plab[loc.two.configs[top2, 1]])],
                     f"hpd{int(level * 100)}_mm3_per_side": round(float(hpd_size(pp, level)[0] * vox), 2),
                     "region_mass": parcel_mass(pmap, labels, names), **groups(pmap)},
            "_maps": {"single": ps, "pair": pp, "top_single": top1, "top_pair": top2}}


def _q(v, ps=(5, 50, 95)):
    return [round(float(x), 3) for x in np.percentile(v, ps)]


def _shares(labels):
    u, c = np.unique(labels, return_counts=True)
    o = np.argsort(-c)
    return {str(u[i]): round(float(c[i] / len(labels)), 3) for i in o}


def report_response_svd_field(loc_factory: Callable[[Sequence[np.ndarray]], FieldLocator], Y: np.ndarray,
                              N: np.ndarray, span: np.ndarray, field: np.ndarray, component: int,
                              unit_subject: np.ndarray, labels: np.ndarray, names: Sequence[str],
                              rng: np.random.Generator, n_boot: int = 200, level: float = 0.9,
                              region_groups: Optional[Dict[str, Sequence[str]]] = None,
                              field_maps: Optional[Sequence[np.ndarray]] = None) -> dict:
    """Location report of one ``field.response_svd`` field from its defining data.

    ``Y`` (units, channels, times) and ``N`` (units, k, channels, times): each unit's response and noise, scaled as
    the definition does (:func:`~source_localization.field.define.scale_units`). ``field_maps``: the template's maps
    of components 0..``component`` (to align signs); ``field`` is component ``component``'s. Each unit's column is its
    response along the component's time course; the whitener is fitted on the units' noise columns
    (``loc_factory(noise_blocks)`` returns a fitted :class:`FieldLocator`).

    Reports the grand mean (:func:`summarize`), each unit, and a bootstrap over subjects (``unit_subject``) that
    re-derives the component per replicate: map cosine with the template, a maximum-likelihood fit under one head
    model drawn at random, the calibrated posterior peak, the pair peak and P(pair), and whether each falls in the
    grand mean's ``level`` region. The random draws (subjects, then the head model) follow
    surface-evoked ``localize_fields_bayes.py`` exactly, so its output reproduces.
    """
    field_maps = list(field_maps) if field_maps is not None else [field]

    def components(idx):
        U, s, Vt = np.linalg.svd(Y[idx].mean(0)[:, span], full_matrices=False)
        out = []
        for k, mp in enumerate(field_maps):
            u, v = U[:, k], Vt[k]
            if u @ mp < 0:
                u, v = -u, -v
            out.append((u, s[k], v))
        return out

    n = len(Y)
    u, s, v = components(np.arange(n))[component]
    yc = Y[:, :, span] @ v
    nc = N[:, :, :, span] @ v
    loc = loc_factory([z.T for z in nc])
    pos = loc.space.positions_mm
    plab = np.array([names[i] if i >= 0 else "none" for i in labels])

    def column(idx):
        u_, s_, v_ = components(idx)[component]
        noise = (N[idx][:, :, :, span] @ v_).mean(0).T
        return loc.whiten(s_ * u_, noise), u_

    ygm, _ = column(np.arange(n))
    gm = summarize(loc, ygm, labels, names, level, region_groups)
    ps, top1 = gm["_maps"]["single"], gm["_maps"]["top_single"]
    Ymd = np.stack([loc.wh.W @ yc[i] / loc.wh.scales[i] for i in range(n)])[:, :, None]
    r = loc.readout(Ymd)
    t_md = r["single"].argmax(1)
    md = {"snr_per_whitened_channel": _q(np.sum(Ymd[:, :, 0] ** 2, 1) / loc.n_whitened - 1),
          "single_peak_regions": _shares(plab[t_md]),
          "single_peak_distance_to_grand_mean_peak_mm": _q(np.linalg.norm(pos[t_md] - pos[top1], axis=1)),
          f"single_hpd{int(level * 100)}_mm3": _q(hpd_size(r["single"], level) * loc.space.spacing_mm ** 3),
          "grand_mean_peak_in_unit_region": round(float(np.mean(mass_above(r["single"], top1) < level)), 3),
          "p_pair": _q(r["p_pair"]), "p_pair_above_0.95": round(float(np.mean(r["p_pair"] > 0.95)), 3),
          "pair_peak_regions": _shares(plab[loc.two.configs[r["pair"].argmax(1), 0]])}
    subjects = np.array(sorted(set(unit_subject)))
    rows = {k: [] for k in ("cos", "ml", "post", "pair", "p_pair", "in_ml", "in_post", "gm_in_rep")}
    for _ in range(n_boot):
        pick = rng.choice(subjects, len(subjects), replace=True)
        idx = np.concatenate([np.flatnonzero(unit_subject == mo) for mo in pick])
        yb, ub = column(idx)
        Yb = yb[None, :, None]
        j = int(rng.integers(loc.space.G_ens.shape[0]))
        ml = int(loc.single_model(j).log_posterior(Yb)[0].argmax())
        rb = loc.readout(Yb)
        pk = int(rb["single"][0].argmax())
        rows["cos"].append(abs(float(ub @ field)))
        rows["ml"].append(ml)
        rows["post"].append(pk)
        rows["pair"].append(int(loc.two.configs[rb["pair"][0].argmax(), 0]))
        rows["p_pair"].append(float(rb["p_pair"][0]))
        rows["in_ml"].append(float(mass_above(ps, ml)) < level)
        rows["in_post"].append(float(mass_above(ps, pk)) < level)
        rows["gm_in_rep"].append(float(mass_above(rb["single"][0], top1)) < level)
    ml, pk, pr = (np.array(rows[x]) for x in ("ml", "post", "pair"))
    boot = {"n": n_boot, "map_cos_to_template": _q(rows["cos"]),
            "ml_fit": {"peak_regions": _shares(plab[ml]),
                       "distance_to_grand_mean_peak_mm": _q(np.linalg.norm(pos[ml] - pos[top1], axis=1)),
                       "spread_mm_rms": round(float(np.sqrt(np.mean(np.sum((pos[ml] - pos[ml].mean(0)) ** 2, 1)))),
                                              2),
                       "inside_grand_mean_region": round(float(np.mean(rows["in_ml"])), 3)},
            "posterior_peak": {"peak_regions": _shares(plab[pk]),
                               "distance_to_grand_mean_peak_mm": _q(np.linalg.norm(pos[pk] - pos[top1], axis=1)),
                               "inside_grand_mean_region": round(float(np.mean(rows["in_post"])), 3),
                               "grand_mean_peak_inside_replicate_region": round(float(np.mean(rows["gm_in_rep"])), 3)},
            "pair": {"p_pair": _q(rows["p_pair"]), "peak_regions": _shares(plab[pr]),
                     "peak_abs_x_mm": _q(np.abs(pos[pr, 0]))}}
    gm.pop("_maps")
    return {"grand_mean": gm, "units": md, "bootstrap": boot, "provenance": loc.provenance()}


# ====================================================================== the field model over a sample of subjects

#: The data kinds the field model reads, and each record's keys (one record per recording):
#:   evoked   xbar (channels, times) trial average; x_flip (k, channels, times) sign-flip averages
#:   steady   z (channels,) complex coefficient at the driving rate; z_flip (k, channels) sign-flip coefficients
#:   ongoing  csd (channels, channels) band cross-spectral matrix (real part); csd_q (k, channels, channels) on k
#:            disjoint parts of the epochs
KINDS = ("evoked", "steady", "ongoing")


class Units:
    """Which unit (subject) each record belongs to; unit weights become record weights w_unit / n_records_of_unit."""

    def __init__(self, record_unit: Sequence):
        self.ids, self.idx = np.unique(np.asarray(record_unit), return_inverse=True)
        self.U = len(self.ids)
        self.nrec = np.bincount(self.idx, minlength=self.U)

    def rec_weights(self, wm: np.ndarray) -> np.ndarray:
        return np.asarray(wm, float)[self.idx] / self.nrec[self.idx]


def inner_draws(wm: np.ndarray, b_in: int, rng: np.random.Generator) -> list:
    """``b_in`` count vectors, each drawing sum(wm) units with replacement from the multiset ``wm``."""
    pool = np.repeat(np.arange(len(wm)), np.asarray(wm).astype(int))
    return [np.bincount(rng.choice(pool, len(pool)), minlength=len(wm)).astype(float) for _ in range(b_in)]


def lead(C: np.ndarray, u: np.ndarray) -> np.ndarray:
    """Leading eigenvector of C scaled by sqrt(its share of the trace), signed to agree with u."""
    w, V = np.linalg.eigh(C)
    v = V[:, -1] * np.sqrt(w[-1] / np.trace(C))
    return v if v @ u >= 0 else -v


class FieldModel:
    """``locate.field_model_map``: locate a sample's field, and its spread over resamples of the sample's subjects.

    For record weights (a resample of subjects), the field is re-derived from the weighted records (``decompose``:
    evoked, the first time course of the mean unit-normalised response; steady, ``field.phase_free``; ongoing,
    ``field.csd_leading``), the weighted grand-mean column and its noise are formed along it (``columns``), and the
    column is located (``locate.field_choice``). ``map`` accumulates the located positions over draws (each draw puts
    mass 1/len(points) on its points) and the share of draws called a pair. The whitener is fitted once per sample on
    one noise block per unit, all through the whole sample's decomposition (``setup``).

    Ported from surface-evoked ``method_crossover.FieldModel`` (H039 arm C; bayes-sl L2/L3, benchmark arm FM).
    """

    def __init__(self, space, kind: str, calibration=VOLUMETRIC, whitener: str = "oas", pair_rule: float = 0.5,
                 allow_uncalibrated: bool = False):
        if kind not in KINDS:
            raise ValueError(f"kind must be one of {KINDS}")
        self.space, self.kind = space, kind
        self._loc_args = dict(calibration=calibration, whitener=whitener, pair_rule=pair_rule,
                              allow_uncalibrated=allow_uncalibrated)
        space.check_calibration(calibration, allow_uncalibrated=allow_uncalibrated)

    def decompose(self, recs: Sequence[dict], rw: np.ndarray) -> np.ndarray:
        idx = np.flatnonzero(rw > 0)
        w = rw[idx] / rw[idx].sum()
        R = [recs[i] for i in idx]
        if self.kind == "evoked":
            M = np.tensordot(w, np.stack([r["xbar"] / np.linalg.norm(r["xbar"]) for r in R]), axes=1)
            return np.linalg.svd(M, full_matrices=False)[2][0]
        if self.kind == "steady":
            Z = np.stack([r["z"] / np.linalg.norm(r["z"]) for r in R])
            return np.linalg.eigh((Z.real * w[:, None]).T @ Z.real + (Z.imag * w[:, None]).T @ Z.imag)[1][:, -1]
        return np.linalg.eigh(np.tensordot(w, np.stack([r["csd"] / np.trace(r["csd"]) for r in R]), axes=1))[1][:, -1]

    def columns(self, recs: Sequence[dict], rw: np.ndarray, ref: Optional[np.ndarray] = None):
        """Weighted grand-mean column (channels,) and its noise (channels, k) along ``ref`` (default: this
        weighting's own decomposition)."""
        ref = self.decompose(recs, rw) if ref is None else ref
        idx = np.flatnonzero(rw > 0)
        w = rw[idx] / rw[idx].sum()
        R = [recs[i] for i in idx]
        if self.kind == "evoked":
            nr = np.array([np.linalg.norm(r["xbar"]) for r in R])
            M = np.tensordot(w / nr, np.stack([r["xbar"] for r in R]), axes=1)
            N = np.tensordot(w / nr, np.stack([r["x_flip"] for r in R]), axes=1)
            return M @ ref, (N @ ref).T
        if self.kind == "steady":
            Z = np.stack([r["z"] / np.linalg.norm(r["z"]) for r in R])
            Nz = np.stack([r["z_flip"] / np.linalg.norm(r["z"]) for r in R])
            ph = np.exp(-1j * np.angle(Z @ ref))
            return w @ (Z * ph[:, None]).real, np.tensordot(w, (Nz * ph[:, None, None]).real, axes=1).T
        vs, ns = [], []
        for r in R:
            v = lead(r["csd"], ref)
            vs.append(v)
            ns.append(np.stack([(lead(q, ref) - v) / 2 for q in r["csd_q"]]))
        return w @ np.array(vs), np.tensordot(w, np.array(ns), axes=1).T

    def setup(self, recs: Sequence[dict], units: Units) -> FieldLocator:
        """The sample's locator: whitener fitted on one noise block per unit through the whole sample's field."""
        ref = self.decompose(recs, units.rec_weights(np.ones(units.U)))
        noise = [self.columns(recs, units.rec_weights(np.eye(units.U)[u]), ref)[1] for u in range(units.U)]
        return FieldLocator(self.space, **self._loc_args).fit(noise)

    def locate(self, loc: FieldLocator, y: np.ndarray, noise: np.ndarray):
        return loc.locate(loc.whiten(y, noise))

    def map(self, recs: Sequence[dict], units: Units, draws: Sequence[np.ndarray], loc: FieldLocator):
        """Mean located mass over the draws (count vectors over units), and the share of draws called a pair."""
        m = np.zeros(len(self.space.positions_mm))
        pp = []
        for wm in draws:
            y, n = self.columns(recs, units.rec_weights(wm))
            pts, p = self.locate(loc, y, n)
            np.add.at(m, pts, 1.0 / len(pts))
            pp.append(p)
        return m / m.sum(), float(np.mean(np.array(pp) >= loc.pair_rule))


# ====================================================================== the model filter of a located pair

MIRROR = np.diag([-1.0, 1.0, 1.0])


def model_filter_pair(space, right_mm: Sequence[float], target: np.ndarray, midline_x_mm: float = 0.0) -> dict:
    """``field.model_filter_pair``: the electrode field of a mirrored source pair, as a spatial filter.

    The right member is the grid position nearest ``right_mm``, the left member the position nearest its mirror image.
    Under each head model (average-referenced leadfields), one mirrored moment (the left member's moment is the right
    one's mirror image) is fitted to ``target`` by least squares; each model's pattern is normalised and signed to
    agree with ``target``, and the filter is their mean, normalised. Reads no recording.
    Returns the filter, both members' positions, and how well each model's pattern fits the target.
    """
    G = space.G_ens - space.G_ens.mean(2, keepdims=True)
    pos = space.positions_mm
    right = np.asarray(right_mm, float)
    left = right * [-1, 1, 1] + [2 * midline_x_mm, 0, 0]
    ir = int(np.argmin(np.linalg.norm(pos - right, axis=1)))
    il = int(np.argmin(np.linalg.norm(pos - left, axis=1)))
    D = np.asarray(target, float)
    pats, fits = [], []
    for h in range(len(G)):
        A = G[h, ir] + G[h, il] @ MIRROR
        p = A @ np.linalg.lstsq(A, D, rcond=None)[0]
        p /= np.linalg.norm(p)
        pats.append(p if p @ D > 0 else -p)
        fits.append(float((p @ D) ** 2))
    w = np.mean(pats, 0)
    w /= np.linalg.norm(w)
    pats = np.array(pats)
    return {"filter": w, "right_mm": pos[ir], "left_mm": pos[il], "right_index": ir, "left_index": il,
            "per_model_cos2_with_target": np.array(fits), "per_model_cos_with_filter": pats @ w,
            "head_models": len(G)}


# ====================================================================== each located region's own field (bayes-sl L3)

def mean_referenced_leadfield(space) -> np.ndarray:
    """The ensemble's leadfields average-referenced over channels and averaged over head models, (P, channels, 3)."""
    n = space.G_ens.shape[2]
    H = np.eye(n) - 1.0 / n
    return np.einsum("ij,kpjo->pio", H, space.G_ens) / space.G_ens.shape[0]


def region_fields(space, u: np.ndarray, located_index: Sequence[int], mass: Sequence[float],
                  region_of_position: Sequence[str], Gm: Optional[np.ndarray] = None) -> Dict[str, tuple]:
    """``field.region_field``: each located region's own electrode field.

    For each located position p (``located_index``, with its located ``mass``): the field u projected on the span of
    p's free-orientation leadfield (mean over head models, average-referenced), normalised and signed to agree with
    u. A region's field is the mass-weighted sum of its positions' fields, normalised. Returns region -> (field,
    located mass). As bayes-sl ``scripts/l3_fields.py`` (``_fields``).
    """
    Gm = mean_referenced_leadfield(space) if Gm is None else Gm
    u = np.asarray(u, float)
    acc: Dict[str, list] = {}
    for p, w in zip(located_index, mass):
        A = Gm[int(p)]
        v = A @ np.linalg.lstsq(A, u, rcond=None)[0]
        v = v / np.linalg.norm(v)
        v = v if v @ u >= 0 else -v
        a = acc.setdefault(region_of_position[int(p)], [np.zeros(len(u)), 0.0])
        a[0] += w * v
        a[1] += w
    return {r: (a / np.linalg.norm(a), w) for r, (a, w) in acc.items()}


def region_power_db(field: np.ndarray, csd: np.ndarray) -> float:
    """``field.region_power_db``: 10 log10(g' C g), a subject's band power along a field g (C average-referenced)."""
    return float(10 * np.log10(field @ csd @ field))


def region_value_table(fields: Dict[str, tuple], csd_by_subject: Dict[str, np.ndarray]) -> list:
    """One row per region and subject: region, p_source (the region's located mass), subject, y_db."""
    return [{"region": r, "p_source": w, "subject": s, "y_db": region_power_db(g, C)}
            for r, (g, w) in fields.items() for s, C in csd_by_subject.items()]


def write_region_values(path, rows: list, provenance: dict) -> Path:
    """Write the region-value table (CSV; columns region, p_source, subject, y_db and any extra keys) and its record
    ``<path>.json``: the catalogue ids and versions used and the space's provenance. source-analytics reads the CSV
    as a per-subject table with one outcome per region (no ROI in the engine's sense)."""
    import csv
    from pathlib import Path as _P

    from ..methods import catalogue_entries
    path = _P(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    keys = list(rows[0]) if rows else ["region", "p_source", "subject", "y_db"]
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        w.writerows(rows)
    E = catalogue_entries()
    rec = {"readout_mode": "field", "step": "region_values", "outcome": "y_db (dB, band power along each region's field)",
           "models": {i: E[i].version for i in ("field.region_field", "field.region_power_db",
                                                "locate.field_model_map") if i in E},
           **provenance}
    _P(str(path) + ".json").write_text(json.dumps(rec, indent=1, default=str))
    return path
