"""Re-measure the map calibrations (calibration.volumetric, calibration.surface) under the corrected BEM solve.

bayes-sl ``scripts/tb11_calibrate_maps.py`` (a0dc8b3, 2026-10-13) measured them with leadfields from
probability-atlas ``pa/geometry.py``, which hands MNE's solver the BEM surfaces inner first (fixed here in b5de19f,
v0.9.0). This is tb11's design with every leadfield and posterior taken from this package instead:
``locate.geometry.Geometry`` (corrected solve), ``locate.head_ensemble`` (ENSEMBLE_PA_PHASE2),
``locate.ensemble_posterior.EnsemblePosterior`` and ``locate.whitening.NoiseWhitener``. Only data loading still comes
from probability-atlas ``scripts/phase2_simulate.py`` (Sentinel recordings and noise, whitener spec, SNRs, held-out
halves, surface positions, allen26 labelling); none of it solves a head model.

Same seeds and draw order as tb11, so the same truths, head models, orientations, profiles and noise columns are
drawn. Changes from tb11, fixed before the run (validation-tests/map_calibration_bemfix/PLAN.md): relative model error
(eps) dropped (tb12 did not choose it; eps 0 is the readout), and the (T, sigma) grid widened.

``--old-order`` swaps the package's solve for MNE's on the stored inner-first surfaces (what ``pa/geometry.py``
does); with tb11's grid it must reproduce tb11's stored scores (``--compare``). That is the control. It declares a
non-current solver, so its ensemble is never cached (the first control run, before 6b5bb4e, cached it under the
corrected key: map_calibration_bemfix/cache_old_order_FAULTY_inner_first/).

    uv run --no-sync python -I scripts/recalibrate_maps.py --measure 40hz --out DIR [--old-order]
    uv run --no-sync python -I scripts/recalibrate_maps.py --compare NEW.csv.gz REF.csv.gz
    uv run --no-sync python -I scripts/recalibrate_maps.py --select DIR      # tb12's rule, pooled over the measures
"""
import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import sparse
from scipy.spatial import cKDTree
from scipy.special import logsumexp

PA = Path("/mnt/e/research/EEG/probability-atlas")
STAGE = Path("/mnt/e/research/EEG/validation-tests/map_calibration_bemfix")
MEASURES = ("40hz", "80hz", "chirp_16ms", "chirp_36ms", "chirp_92ms", "chirp_240ms")
N = 400
LABS = ("volumetric", "surface")
TS = (1.0, 1.25, 1.5, 2.0, 2.5, 3.0)
SIGMAS = (0.0, 0.25, 0.5, 0.75, 1.0, 1.5)
MODES = (("half", 0.5), ("primary", 1.0), ("double", 2.0))
DEEP = ("Thalamus", "Hypothalamus", "Brainstem", "Midbrain", "Basal_Ganglia", "Hippocampus", "Amygdalar", "Cerebellum")


def blur_kernels(pos):
    """sigma -> sparse (P, P) row-stochastic Gaussian kernel K[y, x] (None for sigma 0). As tb11."""
    tree = cKDTree(pos)
    out = {}
    for sg in SIGMAS:
        if sg == 0:
            out[sg] = None
            continue
        D = tree.sparse_distance_matrix(tree, 3 * sg, output_type="coo_matrix")
        K = sparse.csr_matrix((np.exp(-D.data ** 2 / (2 * sg ** 2)), (D.row, D.col)), shape=D.shape)
        out[sg] = sparse.diags(1 / np.asarray(K.sum(1)).ravel()) @ K
    return out


def metrics(lp, pos, onehot, kernels, truth_idx=None, truth_par=None):
    """Per map, per (T, sigma): dict of arrays. truth_idx: list of (n,) index arrays (one per source). As tb11."""
    out = {}
    for T, sg in [(T, sg) for T in TS for sg in SIGMAS]:
        lq = lp / T
        p = np.exp(lq - logsumexp(lq, axis=1, keepdims=True))
        if kernels[sg] is not None:
            p = np.asarray((kernels[sg].T @ p.T).T)
        ps = -np.sort(-p, axis=1)
        k = (np.cumsum(ps, axis=1) < 0.9).sum(1) + 1
        mu = p @ pos
        rms = np.sqrt(np.maximum(p @ np.sum(pos ** 2, 1) - np.sum(mu ** 2, 1), 0))
        r = {"hpd90_n": k, "rms_mm": rms}
        if truth_idx is not None:
            for j, ti in enumerate(truth_idx):
                pt = p[np.arange(len(p)), ti]
                r[f"mass_above_{j}"] = np.sum(np.where(p > pt[:, None], p, 0), axis=1)
            q = p @ onehot
            qt = q[np.arange(len(q)), truth_par]
            r["mass_above_parcel"] = np.sum(np.where(q > qt[:, None], q, 0), axis=1)
        out[(T, sg)] = r
    return out


def run(meas, out_dir, old_order):
    sys.path.insert(0, str(PA))
    sys.path.insert(0, str(PA / "scripts"))
    import phase2_simulate as P                                                          # data loading only
    import mne
    from source_localization.locate import geometry as geometry_mod
    from source_localization.locate.ensemble_posterior import EnsemblePosterior
    from source_localization.locate.head_ensemble import ENSEMBLE_PA_PHASE2, draw_ensemble, ensemble_leadfields
    from source_localization.locate.whitening import NoiseWhitener, average_reference_basis

    if old_order:   # the control: MNE's solver on the surfaces as stored (inner first), as pa/geometry.py does
        geometry_mod._solve_bem = lambda surfs, verbose=False: mne.make_bem_solution(surfs, verbose=verbose)
    cache = out_dir.parent / ("cache_old_order" if old_order else "cache")
    out_dir.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    rng = np.random.default_rng(20261014)
    geo = geometry_mod.Geometry(P.RUN, spacing_mm=0.5, cache_dir=cache / "posterior_cache")
    if old_order:   # not the current solver: ensemble_leadfields neither reads nor writes a cache for it (6b5bb4e)
        geo.bem_solver = "surfaces inner-first (control)"
    surf_pos, _ = P.surface_space()
    prior = ENSEMBLE_PA_PHASE2.prior
    models = draw_ensemble(prior, ENSEMBLE_PA_PHASE2.n_draws, np.random.default_rng(list(ENSEMBLE_PA_PHASE2.seed)))
    G_all = ensemble_leadfields(geo, models, np.concatenate([geo.positions_mm, surf_pos]), cache_dir=cache,
                                verbose=False)
    nv_ = len(geo.positions_mm)
    G_ens = {"volumetric": G_all[:, :nv_], "surface": G_all[:, nv_:]}
    print(f"ensemble leadfields {G_all.shape} ({time.time() - t0:.0f} s)", flush=True)
    L = {lab: P.labelling_setup(geo, lab) for lab in LABS}
    onehot, kern = {}, {lab: blur_kernels(L[lab]["positions_mm"]) for lab in LABS}
    for lab in LABS:
        lb = L[lab]["regions"].labels
        onehot[lab] = np.zeros((len(lb), lb.max() + 1))
        onehot[lab][np.arange(len(lb)), lb] = 1

    # truths: tb11's draws in tb11's order
    sets = {}
    for sname in ("calib", "valid"):
        fresh = [prior.draw(rng) for _ in range(P.K_FRESH)]
        for lab in LABS:
            pos, inside = L[lab]["positions_mm"], L[lab]["inside"]
            idx = rng.choice(np.flatnonzero(inside), N)
            t = pos[idx] + (rng.uniform(-0.25, 0.25, (N, 3)) if lab == "volumetric" else 0)
            sets[f"{sname}|{lab}"] = {"lab": lab, "truths": [t], "fresh": fresh, "assign": rng.integers(0, P.K_FRESH, N)}
    pos, lab_v = L["volumetric"]["positions_mm"], L["volumetric"]["regions"].labels
    names_v = list(L["volumetric"]["parcel_names"])
    ins = L["volumetric"]["inside"]
    iL = np.flatnonzero((lab_v == names_v.index("Auditory_L")) & ins)
    iR = np.flatnonzero((lab_v == names_v.index("Auditory_R")) & ins)
    tL = pos[rng.choice(iL, N)] + rng.uniform(-0.25, 0.25, (N, 3))
    q = tL * np.array([-1, 1, 1])
    tR = pos[iR[np.argmin(np.linalg.norm(pos[iR][None] - q[:, None], axis=2), axis=1)]]
    sets["bilat|volumetric"] = {"lab": "volumetric", "truths": [tL, tR], "fresh": sets["calib|volumetric"]["fresh"],
                                "assign": rng.integers(0, P.K_FRESH, N)}
    for s in sets.values():
        s["G"] = [np.empty((N, geo.n_channels, 3)) for _ in s["truths"]]
    for fid in {id(s["fresh"]) for s in sets.values()}:
        group = [s for s in sets.values() if id(s["fresh"]) == fid]
        for k, model in enumerate(group[0]["fresh"]):
            parts = [(s, j, np.flatnonzero(s["assign"] == k)) for s in group for j in range(len(s["truths"]))]
            Pk = np.concatenate([s["truths"][j][i] for s, j, i in parts])
            G = geo.leadfield(model, Pk)
            o = 0
            for s, j, i in parts:
                s["G"][j][i] = G[o:o + len(i)]
                o += len(i)
    print(f"truth leadfields done ({time.time() - t0:.0f} s)", flush=True)

    M, ids, ext, spec, sel = P.measure_inputs()
    snr = P.snrs(M, ext, spec)
    snr_primary = float(np.median(list(snr.values())))
    halves = P.heldout_halves(ids, P.rng_for(5))
    B = average_reference_basis(geo.n_channels)
    full, noise = M[meas]
    cols = full[0].shape[1]

    def fit_whitener(idx):
        e = ext[meas]
        return NoiseWhitener.fit([noise[i] for i in idx], estimator=spec[meas]["estimator"],
                                 shape_samples=None if e is None else [e[i] for i in idx])

    srng = np.random.default_rng([20261014, list(M).index(meas)])
    rows = []
    for rnd, snames in ((0, ("calib|volumetric", "calib|surface", "bilat|volumetric")),
                        (1, ("valid|volumetric", "valid|surface"))):
        fit_idx, test_idx = halves[meas] if rnd == 0 else halves[meas][::-1]
        wh = fit_whitener(fit_idx)
        Nw = [wh.W @ noise[i] / NoiseWhitener.scale_of(noise[i], B) for i in test_idx]
        for lab in LABS:
            lpos = L[lab]["positions_mm"]
            ens = EnsemblePosterior(wh.whiten_leadfield(G_ens[lab]), lpos, geo.spacing_mm)
            for sname in snames:
                s = sets[sname]
                if s["lab"] != lab:
                    continue
                prof = srng.standard_normal((N, cols))
                S_ = 0
                for G in s["G"]:
                    ori = srng.standard_normal((N, 3))
                    ori /= np.linalg.norm(ori, axis=1, keepdims=True)
                    x = np.einsum("tak,tk->ta", wh.whiten_leadfield(G), ori)[:, :, None] * prof[:, None, :]
                    S_ = S_ + x / np.sqrt(np.sum(x ** 2, axis=(1, 2), keepdims=True) / cols)
                S_ = S_ / np.sqrt(np.sum(S_ ** 2, axis=(1, 2), keepdims=True) / cols)
                Nst = np.stack([Nw[k] for k in srng.integers(0, len(Nw), N)])
                tidx = [np.argmin(np.linalg.norm(lpos[None] - t[:, None], axis=2), axis=1) for t in s["truths"]]
                tpar = L[lab]["regions"].labels[tidx[0]]
                pnames = np.array(L[lab]["parcel_names"])[tpar]
                for mode, f in MODES:
                    Y = S_ * np.sqrt(f * snr_primary * wh.n_whitened) + Nst
                    lp = ens.log_posterior(Y)
                    for (T, sg), r in metrics(lp, lpos, onehot[lab], kern[lab], tidx, tpar).items():
                        rows.append(pd.DataFrame({"set": sname.split("|")[0], "labelling": lab, "measure": meas,
                                                  "mode": mode, "map": np.arange(N), "eps": 0.0, "T": T, "sigma": sg,
                                                  "truth_parcel": pnames,
                                                  "deep": pd.Series(pnames).str.startswith(DEEP).values, **r}))
                print(f"{meas} {sname} done ({time.time() - t0:.0f} s)", flush=True)
            del ens
    # real recordings and the grand mean (in-sample whitener), as tb11
    wh = fit_whitener(range(len(noise)))
    units = np.stack([full[i] / NoiseWhitener.scale_of(noise[i], B) for i in range(len(full))])
    Y = np.einsum("ac,nck->nak", wh.W, np.concatenate([units, units.mean(0)[None] * np.sqrt(len(units))]))
    gm = {}
    for lab in LABS:
        lpos = L[lab]["positions_mm"]
        ens = EnsemblePosterior(wh.whiten_leadfield(G_ens[lab]), lpos, geo.spacing_mm)
        lp = ens.log_posterior(Y)
        gm[f"{lab}|0.0"] = lp[-1]
        for (T, sg), r in metrics(lp, lpos, onehot[lab], kern[lab]).items():
            rows.append(pd.DataFrame({"set": "real", "labelling": lab, "measure": meas,
                                      "mode": ["recording"] * len(units) + ["grand_mean"], "map": np.arange(len(Y)),
                                      "eps": 0.0, "T": T, "sigma": sg, **r}))
        del ens
    df = pd.concat(rows, ignore_index=True)
    df.to_csv(out_dir / f"{meas}.csv.gz", index=False, float_format="%.5g")
    np.savez_compressed(out_dir / f"{meas}_grandmean.npz", **gm)
    (out_dir / f"{meas}_run.json").write_text(json.dumps({
        "measure": meas, "old_order": old_order, "snr_primary": snr_primary, "seconds": time.time() - t0,
        "geometry_signature": geo.signature, "ensemble": ENSEMBLE_PA_PHASE2.provenance(),
        "source_localization": __import__("source_localization").__version__}, indent=1, default=str))
    print(f"{meas} ALL DONE ({time.time() - t0:.0f} s)", flush=True)


def compare(new_path, ref_path):
    """The control: every row of the new scores against tb11's at the same (set, labelling, mode, map, T, sigma), eps 0.
    Values were written with 5 significant digits, so equality is to that precision."""
    new, ref = pd.read_csv(new_path), pd.read_csv(ref_path)
    ref = ref[ref.eps == 0]
    key = ["set", "labelling", "measure", "mode", "map", "T", "sigma"]
    m = new.merge(ref, on=key, suffixes=("", "_ref"))
    print(f"matched rows: {len(m)} (new {len(new)}, tb11 eps 0 {len(ref)})")
    ok = True
    for c in ("mass_above_0", "mass_above_1", "mass_above_parcel", "hpd90_n", "rms_mm"):
        if c not in m:
            continue
        a, b = m[c].to_numpy(float), m[f"{c}_ref"].to_numpy(float)
        both = ~(np.isnan(a) & np.isnan(b))
        close = np.isclose(a[both], b[both], rtol=2e-5, atol=1e-8)
        print(f"  {c}: {close.mean():.6f} of {both.sum()} equal to 5 significant digits; "
              f"max |diff| {np.nanmax(np.abs(a - b)):.3g}")
        ok &= bool(close.all())
    print("CONTROL PASSED" if ok and len(m) else "CONTROL FAILED")
    return ok


def select(score_dir):
    """tb12's rule per labelling, pooled over the six measures (``locate.calibrate.select_setting``)."""
    from source_localization.locate.calibrate import coverage, select_setting
    df = pd.concat([pd.read_csv(Path(score_dir) / f"{m}.csv.gz") for m in MEASURES], ignore_index=True)
    out = {}
    for lab in LABS:
        cal = df[(df.set == "calib") & (df.labelling == lab)]
        scores, strata = {}, None
        for (T, s), g in cal.groupby(["T", "sigma"]):
            g = g.sort_values(["measure", "mode", "map"])
            scores[(float(T), float(s))] = g.mass_above_0.to_numpy()
            st = (g["mode"] + "|" + g.deep.astype(str)).to_numpy()
            assert strata is None or (st == strata).all()
            strata = st
        best, losses = select_setting(scores, strata)
        res = {"chosen": {"temperature": best[0], "blur_mm": best[1], "worst_abs_error_calib": losses[best]},
               "on_grid_edge": best[0] in (min(TS), max(TS)) or best[1] == max(SIGMAS),
               "losses": {f"{t},{b}": v for (t, b), v in sorted(losses.items())}, "coverage": {}}
        for sname in ("calib", "valid"):
            v = df[(df.set == sname) & (df.labelling == lab) & (df["T"] == best[0]) & (df.sigma == best[1])]
            res["coverage"][sname] = {f"{md}|{'deep' if dp else 'cortex'}": coverage(g.mass_above_0)
                                      for (md, dp), g in v.groupby(["mode", "deep"])}
        out[lab] = res
        print(f"{lab}: chosen T {best[0]}, blur {best[1]} mm, worst calib error {losses[best]:.3f}"
              f"{'  (ON GRID EDGE)' if res['on_grid_edge'] else ''}")
    (Path(score_dir).parent / "selection.json").write_text(json.dumps(out, indent=1))
    return out


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--measure", choices=MEASURES)
    ap.add_argument("--out", type=Path)
    ap.add_argument("--old-order", action="store_true")
    ap.add_argument("--compare", nargs=2, type=Path, metavar=("NEW", "REF"))
    ap.add_argument("--select", type=Path, metavar="SCORE_DIR")
    a = ap.parse_args()
    if a.compare:
        sys.exit(0 if compare(*a.compare) else 1)
    elif a.select:
        select(a.select)
    else:
        run(a.measure, a.out or STAGE / ("control" if a.old_order else "scores"), a.old_order)
