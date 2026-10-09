"""Directed planted-network validation: can this montage tell which side of a coupling leads?

Optional second stage of :class:`.harness.NetworkValidation`, reusing its pool, readouts, truths (positions and head
models) and backgrounds. Ported from the probability-atlas method workspace (Phase 10, 2026-10), whose findings on
MEA30 were: direction not readable on the cortical axes; dPLI's sign set by source polarity; DTF biased toward the
stronger source.

Conditions per pair, band, SNR and truth (orientation 0: the pair's first node drives / is stronger / is alone):

========================  =============================================================  ==========
kind                      what                                                           role
========================  =============================================================  ==========
lagged (o, lag range)     B(t) = A(t - lag), fractional lag, both orientations           direction
weak (o)                  as lagged (long lags) with the driver ``weak_db`` below        robustness
zero_eq                   B = A, equal SNR                                               false dir.
zero_uneq (o)             B = A, one source ``weak_db`` below the other                  false dir.
single (o)                one source alone                                               false dir.
uncoupled                 independent A and B                                            false dir.
none                      background alone (once per band and draw)                      z reference
========================  =============================================================  ==========

Every measure is reduced to a net antisymmetric node matrix N (N[a, b] > 0: a leads b) and z-scored per node pair
against the no-plant draws (threshold: 97.5th percentile of |leave-one-out z|, two-sided 5%). Each simulation is
scored at the **exact** pair and on a **coarse** axis (group means of N, groups given in the spec) as correct /
wrong / none. **Readable** at an SNR: correct >= 0.80 and wrong <= 0.05 in each planted orientation (lagged), and
false direction <= 0.10 under every control, at that SNR and every higher one.
"""
from __future__ import annotations

import time
from pathlib import Path
from typing import Any, Dict

import numpy as np

from . import directed as dr
from . import signals as sg

__all__ = ["CONDS", "DirectedNetworkValidation"]

CONDS = [("lagged", 0, "short"), ("lagged", 1, "short"), ("lagged", 0, "long"), ("lagged", 1, "long"),
         ("weak", 0, "long"), ("weak", 1, "long"), ("zero_eq", 0, None), ("zero_uneq", 0, None),
         ("zero_uneq", 1, None), ("single", 0, None), ("single", 1, None), ("uncoupled", 0, None)]
CONTROLS = [("zero_eq", 0), ("zero_uneq", 0), ("zero_uneq", 1), ("single", 0), ("single", 1), ("uncoupled", 0)]
D_: Dict[str, Any] = {}


def _plant(cond, rng, band_range, lag_ranges, band, n_epochs, n_t, sfreq, weak_db):
    kind, o, lr = cond
    if kind in ("lagged", "weak"):
        lo, hi = lag_ranges[lr][band] if isinstance(lag_ranges[lr], dict) else lag_ranges[lr]
        lag = float(rng.uniform(lo, hi))
        drv, rcv = sg.delayed_pair(rng, n_epochs, n_t, sfreq, band_range, lag)
        tc = np.stack([drv, rcv]) if o == 0 else np.stack([rcv, drv])
        off = [0.0, 0.0]
        if kind == "weak":
            off[o] = -weak_db
        return tc, off, lag
    if kind in ("zero_eq", "zero_uneq"):
        a = sg.band_carrier(rng, n_epochs, n_t, sfreq, band_range)
        off = [0.0, 0.0]
        if kind == "zero_uneq":
            off[1 - o] = -weak_db
        return np.stack([a, a.copy()]), off, None
    if kind == "single":
        a = sg.band_carrier(rng, n_epochs, n_t, sfreq, band_range)
        off = [None, None]
        off[o] = 0.0
        return np.stack([a, a]), off, None
    if kind == "uncoupled":
        return (np.stack([sg.band_carrier(rng, n_epochs, n_t, sfreq, band_range),
                          sg.band_carrier(rng, n_epochs, n_t, sfreq, band_range)]), [0.0, 0.0], None)
    raise ValueError(kind)


class DirectedNetworkValidation:
    """Directed stage on top of a set-up :class:`.harness.NetworkValidation` (see the module docstring).

    ``spec`` (the ``directed:`` block of the network spec)::

        seed: 20261008
        snrs_db: [-5, 0, 5]
        lag_ranges: {short: [2, 5], long: {theta: [5, 25], beta: [5, 16], low_gamma: [5, 8.89]}}
        weak_db: 10
        gc_order: 15
        builtin: [psi, gc, trgc]                  # reference measures in .directed
        measures:                                 # injected; net: antisym (M - M.T) or net (already antisymmetric)
          dpli: {callable: ".../connectivity.py:compute_connectivity_matrix", key: dpli, net: antisym,
                 kwargs: {metrics: [dpli]}}
          te_band: {callable: ".../transfer_entropy.py:compute_transfer_entropy", key: net_te, net: net}
        axes:                                     # truth nodes per group; a merged node takes its members' group
          AP: {anterior: [Frontal_Anterior, Motor_L, Motor_R, Olfactory_Bulb], posterior: [...]}
          LR: {left: [Motor_L, ...], right: [Motor_R, ...]}
          CD: {Deep: [Deep], cortex: [...]}
        readability: {correct: 0.80, wrong: 0.05, false: 0.10}

    Pairs scored are the network spec's pairs that carry an ``axis`` (AP, LR or CD).
    """

    def __init__(self, base, spec: Dict[str, Any]):
        from .harness import _resolve_callable
        self.base = base
        self.spec = spec
        self.builtin = list(spec.get("builtin", ["psi", "gc", "trgc"]))
        self.injected = {}
        for name, m in (spec.get("measures") or {}).items():
            self.injected[name] = dict(fn=_resolve_callable(m["callable"]), key=m["key"], net=m.get("net", "antisym"),
                                       kwargs=dict(m.get("kwargs") or {}))
        self.measures = list(self.injected) + self.builtin
        self.pairs = [(p, pr) for p, pr in enumerate(base.spec.pairs) if pr.get("axis")]
        if not self.pairs:
            raise ValueError("no pair carries an 'axis'; directed validation scores pairs along AP/LR/CD axes")

    # ------------------------------------------------------------------ simulation
    def _nodes(self):
        st = self.base.state
        out = {}
        for name, r in st["readouts"].items():
            if r["type"] == "electrodes":
                out[name] = sorted(set(r["unit_to_node"].values()))
            else:
                out[name] = list(r["nodes"])
        return out

    def cells(self) -> list:
        sp, bs = self.spec, self.base.spec
        bands = list(bs.bands)
        snrs = [float(s) for s in sp.get("snrs_db", [-5.0, 0.0, 5.0])]
        cells = [(-1, b, None, -1, i) for b in bands for i in range(bs.n_truths)]
        cells += [(p, b, s, c, i) for p, _ in self.pairs for b in bands for s in snrs for c in range(len(CONDS))
                  for i in range(bs.n_truths)]
        return [(k, *c) for k, c in enumerate(cells)]

    @staticmethod
    def _unit_nets(X, readout, band):
        r = D_["readouts"][readout]
        if r["type"] == "electrodes":
            units, Y = list(D_["pool"].channels), X
        else:
            units = list(r["nodes"])
            Y = np.einsum("nc,ect->ent", r["op"], X)
        ts = {u: Y[:, k, :].ravel() for k, u in enumerate(units)}
        b = {band: D_["bands"][band]}
        out = {}
        for name, m in D_["injected"].items():
            res, names = m["fn"](ts, D_["sfreq"], b, **m["kwargs"])
            M = res[band][m["key"]]
            M = M - M.T if m["net"] == "antisym" else M
            idx = [list(names).index(u) for u in units]
            out[name] = M[np.ix_(idx, idx)]
        if D_["builtin"]:
            out.update(dr.directed_net(Y, D_["sfreq"], D_["bands"][band], D_["gc_order"]))
        return units, out

    @classmethod
    def _one(cls, cell):
        k, p, band, snr, c, i = cell
        X = D_["bg"][i].copy()
        lag = None
        if c >= 0:
            rng = np.random.default_rng([D_["seed"], c + 1, p, D_["band_list"].index(band),
                                         D_["snrs"].index(snr), i])
            tc, off, lag = _plant(CONDS[c], rng, D_["bands"][band], D_["lag_ranges"], band, X.shape[0], X.shape[-1],
                                  D_["sfreq"], D_["weak_db"])
            for s in range(2):
                if off[s] is None:
                    continue
                g = D_["g"][p, i, s]
                a = sg.scale_to_snr(g, tc[s], D_["bgp"][(i, band)], snr + off[s], D_["sfreq"], D_["bands"][band])
                X += sg.avg_ref(a * g)[None, :, None] * tc[s][:, None, :]
        vec = {}
        for r in D_["readouts"]:
            units, nets = cls._unit_nets(X, r, band)
            rd = D_["readouts"][r]
            node_of = rd["unit_to_node"] if rd["type"] == "electrodes" else {u: u for u in units}
            nodes = D_["nodes"][r]
            iu, ju = np.triu_indices(len(nodes), 1)
            vec[r] = np.stack([dr.node_net(nets[m], units, node_of, nodes)[iu, ju]
                               for m in D_["measures"]]).astype(np.float32)
        return k, lag, vec

    def simulate(self, cells=None, workers=None):
        """Node-level net matrices for every cell: (meta DataFrame, {readout: (n_cells, n_measures, n_upper)})."""
        import pandas as pd
        bs, st = self.base.spec, self.base.state
        cells = self.cells() if cells is None else list(cells)
        nodes = self._nodes()
        D_.update(readouts=st["readouts"], pool=st["pool"], bands=st["bands"], band_list=list(st["bands"]),
                  sfreq=self.base.sfreq, bg=st["bg"], bgp=st["bgp"], g=st["g"], nodes=nodes,
                  injected=self.injected, builtin=self.builtin, measures=self.measures,
                  gc_order=int(self.spec.get("gc_order", 15)), seed=int(self.spec.get("seed", 20261008)),
                  snrs=[float(s) for s in self.spec.get("snrs_db", [-5.0, 0.0, 5.0])],
                  lag_ranges=self.spec.get("lag_ranges", {"short": [2.0, 5.0], "long": [5.0, 25.0]}),
                  weak_db=float(self.spec.get("weak_db", 10.0)))
        nup = {r: len(n) * (len(n) - 1) // 2 for r, n in nodes.items()}
        arr = {r: np.full((len(cells), len(self.measures), nup[r]), np.nan, np.float32) for r in nodes}
        lags = np.full(len(cells), np.nan)
        workers = int(workers or bs.workers or 1)
        t0 = time.time()
        index = {c[0]: n for n, c in enumerate(cells)}
        if workers <= 1:
            from threadpoolctl import threadpool_limits
            with threadpool_limits(1):
                results = (self._one(c) for c in cells)
                self._collect(results, arr, lags, index)
        else:
            from multiprocessing import get_context
            from .harness import _worker_init
            with get_context("fork").Pool(workers, initializer=_worker_init) as mp:
                self._collect(mp.imap_unordered(DirectedNetworkValidation._one, cells, chunksize=2), arr, lags, index)
        meta = pd.DataFrame([dict(k=k, pair=p, band=b, snr=s, cond=c, truth=i,
                                  kind="none" if c < 0 else CONDS[c][0],
                                  orientation=None if c < 0 else CONDS[c][1],
                                  lag_range=None if c < 0 else CONDS[c][2]) for k, p, b, s, c, i in cells])
        meta["lag_ms"] = lags
        self.base._log(f"directed: {len(cells)} cells in {(time.time() - t0) / 60:.1f} min")
        return meta, arr, nodes

    @staticmethod
    def _collect(results, arr, lags, index):
        for k, lag, vec in results:
            n = index[k]
            lags[n] = np.nan if lag is None else lag
            for r in arr:
                arr[r][n] = vec[r]

    # ------------------------------------------------------------------ scoring
    def _group(self, readout, node, axis):
        groups = self.spec["axes"][axis]
        members = self.base.state["readouts"][readout].get("members", {}).get(node, {node})
        found = {g for g, lst in groups.items() for m in members if m in lst}
        return next(iter(found)) if len(found) == 1 else None

    def score(self, meta, arr, nodes):
        """Per simulation: correct / wrong / none at the exact pair and on the pair's axis."""
        import pandas as pd
        rows = []
        st = self.base.state
        for r, A in arr.items():
            nd_ = nodes[r]
            n = len(nd_)
            t2n = st["readouts"][r].get("truth_to_node")
            name = (lambda x: t2n[x]) if t2n else (lambda x: x)
            iu, ju = np.triu_indices(n, 1)
            for mi, m in enumerate(self.measures):
                for band in meta.band.unique():
                    cb = meta[meta.band == band]
                    none = cb[cb.kind == "none"].k.to_numpy()
                    planted = cb[cb.kind != "none"]
                    V = A[planted.k.to_numpy(), mi].astype(float)
                    V0 = A[none, mi].astype(float)
                    ze, te = _zscore(V, V0)
                    M, M0 = _full(V, n), _full(V0, n)
                    top = np.argmax(np.abs(ze), axis=1)
                    for p, pr in self.pairs:
                        sel = (planted.pair == p).to_numpy()
                        if not sel.any():
                            continue
                        sub = planted[sel]
                        axis = pr["axis"]
                        na, nb = name(pr["a"]), name(pr["b"])
                        ex_ok = na in nd_ and nb in nd_ and na != nb
                        if ex_ok:
                            ia, ib = nd_.index(na), nd_.index(nb)
                            col = np.flatnonzero((iu == min(ia, ib)) & (ju == max(ia, ib)))[0]
                            z_ex = ze[sel, col] * (1 if ia < ib else -1)
                            top_hit = top[sel] == col
                        ga, gb = self._group(r, na, axis), self._group(r, nb, axis)
                        gi = [k for k, x in enumerate(nd_) if self._group(r, x, axis) == ga]
                        gj = [k for k, x in enumerate(nd_) if self._group(r, x, axis) == gb]
                        co_ok = ga is not None and gb is not None and ga != gb and len(gi) and len(gj)
                        if co_ok:
                            F = M[sel][:, gi][:, :, gj].mean(axis=(1, 2))
                            F0 = M0[:, gi][:, :, gj].mean(axis=(1, 2))
                            z_co, t_co = _zscore(F[:, None], F0[:, None])
                            z_co = z_co[:, 0]
                        for level, ok in (("coarse", co_ok), ("exact", ex_ok)):
                            if not ok:
                                continue
                            z = z_co if level == "coarse" else z_ex
                            thr = t_co if level == "coarse" else te
                            sign = np.where(sub.orientation.fillna(0).to_numpy() == 0, 1.0, -1.0)
                            out = np.where(np.abs(z) <= thr, "none", np.where(np.sign(z) == sign, "correct", "wrong"))
                            df = sub[["kind", "orientation", "lag_range", "snr", "truth"]].copy()
                            df["z"], df["outcome"] = z, out
                            df["top_hit"] = (top_hit & (out == "correct")) if level == "exact" else np.nan
                            rows.append(df.assign(readout=r, measure=m, band=band, pair=p, node_a=pr["a"],
                                                  node_b=pr["b"], axis=axis, level=level, thr=thr))
        return pd.concat(rows, ignore_index=True)

    def readability(self, outcomes):
        """The directed-readability table (lowest readable SNR or ``never``) per pair, readout, measure, band,
        lag range and level."""
        import pandas as pd
        crit = {"correct": 0.80, "wrong": 0.05, "false": 0.10, **(self.spec.get("readability") or {})}
        keys = ["pair", "node_a", "node_b", "axis", "readout", "measure", "band", "level"]
        rows = []
        for k, g in outcomes.groupby(keys):
            ctrl = g[~g.kind.isin(["lagged", "weak"])]
            for lr in ("short", "long"):
                h = pd.concat([g[(g.kind == "lagged") & (g.lag_range == lr)], ctrl])
                rows.append(dict(zip(keys, k), lag_range=lr, readable_from_db=_readable_snr(h, crit)))
        return pd.DataFrame(rows)

    def run(self, out_dir) -> dict:
        out = Path(out_dir)
        out.mkdir(parents=True, exist_ok=True)
        meta, arr, nodes = self.simulate()
        meta.to_csv(out / "directed_cells.csv", index=False)
        for r, a in arr.items():
            np.save(out / f"directed_net_{r}.npy", a)
        oc = self.score(meta, arr, nodes)
        oc.to_csv(out / "directed_outcomes.csv.gz", index=False)
        tab = self.readability(oc)
        tab.to_csv(out / "directed_readability.csv", index=False)
        n = int((tab.readable_from_db != "never").sum())
        self.base._log(f"directed: {n} of {len(tab)} cells readable; outputs in {out}")
        return {"n_cells_readable": n, "n_cells": int(len(tab)), "measures": self.measures}


def _full(V, n):
    iu, ju = np.triu_indices(n, 1)
    M = np.zeros((V.shape[0], n, n))
    M[:, iu, ju], M[:, ju, iu] = V, -V
    return M


def _zscore(v, ref):
    """z of v (cells, k) against ref (draws, k); threshold = 97.5th pct of |LOO z| of ref, pooled over k."""
    mu, sd = ref.mean(0), ref.std(0, ddof=1)
    sd = np.where(sd > 0, sd, np.inf)
    loo = []
    for i in range(ref.shape[0]):
        rest = np.delete(ref, i, axis=0)
        s = rest.std(0, ddof=1)
        loo.append((ref[i] - rest.mean(0)) / np.where(s > 0, s, np.inf))
    return (v - mu) / sd, float(np.nanpercentile(np.abs(np.array(loo)), 97.5))


def _readable_snr(g, crit):
    ok = {}
    for snr, h in g.groupby("snr"):
        lag = h[h.kind == "lagged"]
        good = len(lag) > 0
        for o in (0, 1):
            x = lag[lag.orientation == o].outcome
            good &= len(x) > 0 and (x == "correct").mean() >= crit["correct"] and (x == "wrong").mean() <= crit["wrong"]
        for kind, o in CONTROLS:
            x = h[(h.kind == kind) & (h.orientation == o)].outcome
            good &= len(x) > 0 and (x != "none").mean() <= crit["false"]
        ok[snr] = bool(good)
    best = None
    for s in sorted(ok, reverse=True):
        if not ok[s]:
            break
        best = s
    return "never" if best is None else best
