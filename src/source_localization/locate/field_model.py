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

from typing import Callable, Dict, Optional, Sequence

import numpy as np

from .ensemble_posterior import (VOLUMETRIC, EnsemblePosterior, MirrorPairPosterior, mass_above,
                                 model_probabilities)
from .whitening import NoiseWhitener

__all__ = ["FieldLocator", "summarize", "hpd_size", "parcel_mass", "report_response_svd_field"]


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
