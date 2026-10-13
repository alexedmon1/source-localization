"""Planted-network validation: which networks can this montage, source model and atlas resolve?

For a declared set of node pairs, coupled sources are planted at known positions, simulated through freshly drawn
head models (:mod:`..head_models`) into **real recorded backgrounds**, and read out at the electrodes and at
source nodes. Every edge is z-scored against the **same background with nothing planted**, which removes the real
coupling the backgrounds contain. The output is the resolvable-network table: for every pair, readout, metric,
band and coupling kind, the lowest SNR at which the pair is named correctly (top-1 >= 0.70) while a single
uncoupled source creates few false edges (<= 0.10), at that SNR and every higher one.

The connectivity metrics are **injected** (``metrics.callable``): any function with source-analytics'
``compute_connectivity_matrix`` contract, ``f(node_ts: dict[str, ndarray], sfreq, {band: (lo, hi)}) ->
(results[band][metric] -> (n, n) ndarray, names)``. The validated code is then the code an analysis runs, and
neither package depends on the other.

Ported from the probability-atlas method workspace (Phases 9-11, 2026-10). With the same inputs and seeds it
reproduces that workspace's Phase 9 simulations (see ``tests/test_network_validation.py`` and the acceptance check in
``docs/validation/README.md``).

Run it with ``source-localization validate networks --spec spec.yaml`` (see :func:`load_spec` for the spec).
"""
from __future__ import annotations

import glob
import importlib
import importlib.util
import json
import pickle
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence

import numpy as np

from ..head_models import HeadModel, HeadModelPrior, TruthForward
from . import nodes as nd
from . import signals as sg

__all__ = ["NetworkSpec", "load_spec", "NetworkValidation"]

G_: Dict[str, Any] = {}                      # worker globals (fork)


# ---------------------------------------------------------------------------------------------------- the spec
@dataclass
class NetworkSpec:
    """What to validate. Load from YAML with :func:`load_spec`; field docs are in the YAML example there."""

    pipeline_dir: str
    backgrounds: List[Dict[str, Any]]
    pairs: List[Dict[str, str]]
    metrics: Dict[str, Any]
    # Default readouts (probability-atlas Phase 11, MEA30): electrodes attribute cortical pairs best; volume parcels
    # collapsed into Deep are slightly better for cortical pairs; the full parcels (scored at the truth level) are
    # better for cortex-Deep pairs. Merged regions are opt-in: they hide coupling inside a merged region and lost the
    # frontal pairs there.
    readouts: List[Dict[str, Any]] = field(default_factory=lambda: [
        {"name": "R-el", "type": "electrodes"},
        {"name": "R-parcel", "type": "nodes", "collapse_volume": True},
        {"name": "R-parcel-full", "type": "nodes", "collapse_volume": False}])
    truth_labels: Dict[str, Any] = field(default_factory=lambda: {"collapse_volume": True})
    n_epochs: int = 30
    n_truths: int = 50
    seed: int = 20261007
    bands: Dict[str, List[float]] = field(default_factory=lambda: {
        "theta": [6.0, 8.0], "beta": [15.0, 25.0], "low_gamma": [35.0, 45.0]})
    snrs_db: List[float] = field(default_factory=lambda: [-10.0, -5.0, 0.0, 5.0])
    kinds: List[str] = field(default_factory=lambda: list(sg.KINDS))
    lag_ms: List[float] = field(default_factory=lambda: [5.0, 25.0])
    head_model_prior: Dict[str, Any] = field(default_factory=lambda: {"shift_sd_mm": 0.3,
                                                                       "skull_factor_range": [0.5, 2.0]})
    resolvable: Dict[str, float] = field(default_factory=lambda: {"top1": 0.70, "false_edges": 0.10})
    workers: int = 1
    undirected: bool = True                       # the Phase 9-style resolvable-network table
    directed: Optional[Dict[str, Any]] = None     # optional directed stage (see .directed_validation)


def load_spec(path) -> NetworkSpec:
    """A :class:`NetworkSpec` from YAML::

        pipeline_dir: /path/to/run          # a completed pipeline run of the montage (hybrid recommended)
        backgrounds:                        # real resting EEG of the same montage; draws alternate across groups
          - {group: KO, files: [a.set, b.set]}       # files in this order, or a glob (sorted)
          - {group: WT, files: "/data/wt/*.set"}
        n_epochs: 30                        # epochs per simulated recording (all files: same epoch length)
        n_truths: 50                        # truths per pair x band x SNR x kind (truth i: head model i, draw i)
        seed: 20261007
        bands: {theta: [6, 8], beta: [15, 25], low_gamma: [35, 45]}
        snrs_db: [-10, -5, 0, 5]            # planted band power at the most sensitive electrode / background's
        kinds: [lagged, zero, envelope, single, uncoupled]
        lag_ms: [5, 25]                     # lagged coupling: uniform lag range
        head_model_prior: {shift_sd_mm: 0.3, skull_factor_range: [0.5, 2.0]}
        truth_labels: {collapse_volume: true}         # the node names pairs refer to (volume parcels -> Deep)
        readouts:                           # default: R-el, R-parcel and R-parcel-full (below, without R-region)
          - {name: R-el, type: electrodes}            # 30 electrodes, each assigned to a truth node
          - {name: R-parcel, type: nodes, collapse_volume: true}
          - {name: R-parcel-full, type: nodes, collapse_volume: false}   # volume parcels kept; scored at truth level
          - {name: R-region, type: nodes, collapse_volume: true, merge_map: regions.json}   # opt-in; {parcel: region}
        pairs:
          - {class: dorsal_far, a: Frontal_Anterior, b: Retrosplenial_L}
        metrics:
          callable: source_analytics.spectral.connectivity:compute_connectivity_matrix   # or /path/file.py:func
          names: [coherence, imag_coherence, pli, wpli, dwpli, aec, partial_corr]
          abs: [partial_corr]                         # scored as |value|
        resolvable: {top1: 0.70, false_edges: 0.10}
        workers: 9
        undirected: true                    # set false to run only the directed stage
        directed:                           # optional; see validation.networks.directed_validation
          axes: {AP: {anterior: [...], posterior: [...]}, LR: {left: [...], right: [...]}}
          measures: {...}                   # injected directed measures; builtin: [psi, gc, trgc]
        # pairs scored by the directed stage carry an axis:  - {class: dorsal_far, a: ..., b: ..., axis: AP}
    """
    import yaml
    d = yaml.safe_load(Path(path).read_text())
    spec = NetworkSpec(**d)
    base = Path(path).parent
    for r in spec.readouts:
        mm = r.get("merge_map")
        if isinstance(mm, str):
            r["merge_map"] = json.loads((base / mm).read_text() if not Path(mm).is_absolute()
                                        else Path(mm).read_text())
    return spec


def _resolve_callable(ref: str) -> Callable:
    """``module:function`` (an importable module) or ``/path/to/file.py:function`` (loaded from the file, so the
    metric package need not be installed in this environment)."""
    mod, _, fn = ref.rpartition(":")
    if not mod or not fn:
        raise ValueError(f"metrics.callable must be 'module:function' or 'file.py:function'; got {ref!r}")
    if mod.endswith(".py"):
        spec = importlib.util.spec_from_file_location(Path(mod).stem + "_metrics", mod)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return getattr(module, fn)
    return getattr(importlib.import_module(mod), fn)


def _files(entry) -> List[str]:
    f = entry["files"]
    return sorted(glob.glob(f)) if isinstance(f, str) else [str(x) for x in f]


# ---------------------------------------------------------------------------------------------- the validation
class NetworkValidation:
    """One planted-network validation run (see the module docstring).

    ``setup()`` builds the pool, readouts, truths and backgrounds; ``control()`` the no-plant reference;
    ``simulate()`` the planted cells; ``summarise()`` the tables. ``run(out_dir)`` does all four and writes
    ``sims.csv.gz``, ``resolvable_networks.csv``, ``detection.csv``, ``electrode_assignment.csv`` and
    ``summary.json``.
    """

    def __init__(self, spec: NetworkSpec, verbose: bool = True):
        self.spec = spec
        self.verbose = verbose
        self.metric_fn = _resolve_callable(spec.metrics["callable"])
        self.metric_names = list(spec.metrics["names"])
        self.metric_abs = set(spec.metrics.get("abs", []))
        self.state: Dict[str, Any] = {}

    def _log(self, msg):
        if self.verbose:
            print(msg, flush=True)

    # ------------------------------------------------------------------ setup
    def setup(self) -> Dict[str, Any]:
        t0 = time.time()
        sp = self.spec
        run = Path(sp.pipeline_dir)
        pool = nd.Pool.from_run(run)
        lab_truth = nd.node_labels(pool, collapse_volume=sp.truth_labels.get("collapse_volume", True),
                                   merge_map=sp.truth_labels.get("merge_map"))
        truth_nodes = sorted({x for x in lab_truth if x is not None})
        for pr in sp.pairs:
            for n in (pr["a"], pr["b"]):
                if n not in truth_nodes:
                    raise ValueError(f"pair node {n!r} is not a truth node; truth nodes: {truth_nodes}")
        readouts = {}
        for r in sp.readouts:
            name = r["name"]
            if r["type"] == "electrodes":
                op_t, nodes_t = nd.node_operator(pool, lab_truth)      # nodes over which electrodes are assigned
                assign = nd.assign_electrodes(pool, lab_truth, nodes_t)
                readouts[name] = {"type": "electrodes", "unit_to_node": assign, "truth_to_node": None,
                                  "members": {n: {n} for n in set(assign.values())}}
            else:
                labs = nd.node_labels(pool, collapse_volume=r.get("collapse_volume", True),
                                      merge_map=r.get("merge_map"))
                op, nodes = nd.node_operator(pool, labs)
                # scoring nodes: a readout node maps onto the truth label space through its own sources, so finer
                # readouts (no collapse) are scored at the truth level; merged readouts map truth -> merged node
                u2n, t2n, members = {}, {}, {}
                for n in nodes:
                    tl = {lab_truth[k] for k in np.flatnonzero(labs == n)} - {None}
                    members[n] = tl
                    u2n[n] = n if (len(tl) != 1 or r.get("merge_map")) else next(iter(tl))
                for t in truth_nodes:
                    rl = {labs[k] for k in np.flatnonzero(lab_truth == t)} - {None}
                    t2n[t] = next(iter(rl)) if len(rl) == 1 else t
                readouts[name] = {"type": "nodes", "op": op, "nodes": nodes, "unit_to_node": u2n,
                                  "truth_to_node": t2n if r.get("merge_map") else None, "members": members}
        src = self._truth_sources(lab_truth)
        g = self._truth_leadfields(pool, src, run)
        bg, meta = self._background_draws(pool.channels)
        bands = {b: tuple(v) for b, v in sp.bands.items()}
        bgp = {(i, b): sg.band_power(bg[i], self.sfreq, bands[b]) for i in range(sp.n_truths) for b in bands}
        self.state = dict(pool=pool, lab_truth=lab_truth, readouts=readouts, src=src, g=g, bg=bg, meta=meta,
                          bands=bands, bgp=bgp)
        self._log(f"setup {time.time() - t0:.0f} s: readouts {list(readouts)}; truth nodes {len(truth_nodes)}")
        return self.state

    def _truth_sources(self, lab_truth) -> np.ndarray:
        """(pairs, truths, 2) pool indices of the planted sources (positions uniform within each node)."""
        sp = self.spec
        idx = {n: np.flatnonzero(lab_truth == n) for n in set(x for x in lab_truth if x is not None)}
        out = np.zeros((len(sp.pairs), sp.n_truths, 2), int)
        for p, pr in enumerate(sp.pairs):
            for i in range(sp.n_truths):
                rng = np.random.default_rng([sp.seed, 2, p, i])
                sa = rng.choice(idx[pr["a"]])
                sb = rng.choice(idx[pr["b"]][idx[pr["b"]] != sa])
                out[p, i] = (sa, sb)
        return out

    def _truth_leadfields(self, pool, src, run) -> np.ndarray:
        """(pairs, truths, 2, n_channels): each truth's sources under its fresh head model (truth i: model i),
        at the pool's own orientation."""
        sp = self.spec
        info = pickle.load(open(run / "data" / "step1_info.pkl", "rb"))
        bem_files = sorted((run / "bem_cache").glob("*.pkl"))
        if not bem_files:
            raise FileNotFoundError(f"{run}/bem_cache has no BEM pickle")
        bem = pickle.load(open(bem_files[0], "rb"))
        if list(info["ch_names"]) != list(pool.channels):
            raise ValueError("step1_info channel order differs from the forward's")
        tf = TruthForward(info, bem)
        flat = np.unique(src.ravel())
        L0, kept = tf.gains(HeadModel(), pool.xyz[flat])
        if not kept.all():
            raise RuntimeError("MNE excluded pool positions under the nominal model")
        u = {}
        for k, s in enumerate(flat):
            v = np.linalg.lstsq(L0[k], pool.G[:, s], rcond=None)[0]
            if np.corrcoef(L0[k] @ v, pool.G[:, s])[0, 1] < 0.9999:
                raise RuntimeError(f"pool source {s}: nominal leadfield does not reproduce the pool column")
            u[s] = v
        prior = HeadModelPrior(**{**sp.head_model_prior,
                                  "skull_factor_range": tuple(sp.head_model_prior["skull_factor_range"])})
        g = np.zeros(src.shape + (len(pool.channels),))
        for i in range(sp.n_truths):
            model = prior.draw(np.random.default_rng([sp.seed, 3, i]))
            pos = src[:, i, :].ravel()
            L, kept = tf.gains(model, pool.xyz[pos])
            if not kept.all():
                raise RuntimeError(f"truth {i}: MNE excluded a planted position under its head model")
            for k, s in enumerate(pos):
                g[k // 2, i, k % 2] = L[k] @ u[s]
        return g

    def _background_draws(self, channels):
        """``n_truths`` draws of ``n_epochs`` epochs: groups alternate, members cycle in the given order, epochs are
        sampled without replacement; average-referenced."""
        import mne
        sp = self.spec
        groups = [(e.get("group", f"g{j}"), _files(e)) for j, e in enumerate(sp.backgrounds)]
        cache, bg, meta = {}, [], []
        self.sfreq = None
        n_samples = None
        for i in range(sp.n_truths):
            gname, files = groups[i % len(groups)]
            path = files[(i // len(groups)) % len(files)]
            if path not in cache:
                ep = mne.read_epochs_eeglab(path, verbose="error")
                missing = [c for c in channels if c not in ep.ch_names]
                if missing:
                    raise ValueError(f"{path}: channels missing: {missing}")
                cache[path] = ep.get_data(picks=list(channels))
                fs = float(ep.info["sfreq"])
                if self.sfreq is None:
                    self.sfreq = fs
                elif fs != self.sfreq:
                    raise ValueError(f"{path}: {fs} Hz; the first background is {self.sfreq} Hz")
            X = cache[path]
            if n_samples is None:
                n_samples = X.shape[-1]
            if X.shape[-1] != n_samples or X.shape[0] < sp.n_epochs:
                raise ValueError(f"{path}: {X.shape}; need >= {sp.n_epochs} epochs of {n_samples} samples")
            sel = np.sort(np.random.default_rng([sp.seed, 1, i]).choice(X.shape[0], sp.n_epochs, replace=False))
            bg.append(sg.avg_ref(X[sel], axis=1))
            meta.append({"draw": i, "group": gname, "file": path})
        self.n_samples = n_samples
        return np.stack(bg), meta

    # ------------------------------------------------------------------ connectivity and scoring
    @staticmethod
    def _connectivity(X, readout, band):
        r = G_["readouts"][readout]
        if r["type"] == "electrodes":
            ts = {ch: X[:, e, :].ravel() for e, ch in enumerate(G_["pool"].channels)}
        else:
            Y = np.einsum("nc,ect->ent", r["op"], X)
            ts = {n: Y[:, k, :].ravel() for k, n in enumerate(r["nodes"])}
        res, names = G_["metric_fn"](ts, G_["sfreq"], {band: G_["bands"][band]})
        m = res[band]
        return {k: (np.abs(m[k]) if k in G_["metric_abs"] else m[k]) for k in G_["metric_names"]}, list(names)

    @staticmethod
    def _edge_index(names, readout):
        u2n = G_["readouts"][readout]["unit_to_node"]
        iu, ju = np.triu_indices(len(names), 1)
        return iu, ju, [nd.edge_pair(names[a], names[b], u2n) for a, b in zip(iu, ju)]

    @staticmethod
    def _planted_pair(p, readout):
        pr = G_["pairs"][p]
        a, b = pr["a"], pr["b"]
        t2n = G_["readouts"][readout]["truth_to_node"]
        if t2n is not None:
            a, b = t2n[a], t2n[b]
        return None if a == b else frozenset((a, b))

    def control(self) -> Dict[tuple, dict]:
        """No-plant reference per readout, band and metric: edge mean and SD over the draws, and the 95th
        percentile of the leave-one-out z of the no-plant draws over between-node edges (the false-edge
        threshold)."""
        t0 = time.time()
        self._init_globals()
        C = {}
        for readout in G_["readouts"]:
            for band in G_["bands"]:
                mats, names = [], None
                for i in range(self.spec.n_truths):
                    m, names = NetworkValidation._connectivity(G_["bg"][i], readout, band)
                    mats.append(m)
                iu, ju, pairs = self._edge_index(names, readout)
                between = np.array([q is not None for q in pairs])
                for metric in G_["metric_names"]:
                    V = np.array([mm[metric][iu, ju] for mm in mats])
                    mu, sd = V.mean(0), V.std(0, ddof=1)
                    loo = []
                    for i in range(len(V)):
                        rest = np.delete(V, i, axis=0)
                        loo.append((V[i] - rest.mean(0)) / rest.std(0, ddof=1))
                    loo = np.array(loo)
                    C[(readout, band, metric)] = dict(mu=mu, sd=np.where(sd > 0, sd, np.inf),
                                                      q95=float(np.nanpercentile(loo[:, between], 95)))
        G_["C1"] = C
        self.state["C1"] = C
        self._log(f"control (no plant): {time.time() - t0:.0f} s")
        return C

    @classmethod
    def _score(cls, mats, names, readout, band, p):
        iu, ju, pairs = cls._edge_index(names, readout)
        target = cls._planted_pair(p, readout)
        rows = []
        for metric in G_["metric_names"]:
            c = G_["C1"][(readout, band, metric)]
            z = (mats[metric][iu, ju] - c["mu"]) / c["sd"]
            between = np.array([q is not None for q in pairs])
            is_t = np.array([q == target for q in pairs]) if target is not None else np.zeros(len(pairs), bool)
            zb = np.where(between, z, -np.inf)
            top = pairs[int(np.argmax(zb))]
            others = between & ~is_t
            rows.append(dict(readout=readout, metric=metric, top_pair="|".join(sorted(top)),
                             top_hit=bool(target is not None and top == target),
                             top_deep=bool(G_["deep_name"] in top),
                             z_target=float(z[is_t].max()) if is_t.any() else np.nan,
                             false_frac=float((z[others] > c["q95"]).mean()) if others.any() else np.nan,
                             attributable=target is not None and bool(is_t.any())))
        return rows

    @classmethod
    def _one(cls, cell):
        p, band, snr, kind, i = cell
        sp = G_["spec"]
        bands = list(G_["bands"])
        rng = np.random.default_rng([sp.seed, 4, p, bands.index(band), list(sp.snrs_db).index(snr),
                                     list(sg.KINDS).index(kind), i])
        lag = float(rng.uniform(*sp.lag_ms)) if kind == "lagged" else None
        n_ep, n_t = G_["bg"][i].shape[0], G_["bg"][i].shape[-1]
        tc = sg.plant_timecourses(kind, rng, n_ep, n_t, G_["sfreq"], G_["bands"][band], lag_ms=lag)
        X = G_["bg"][i].copy()
        bgp = G_["bgp"][(i, band)]
        for k in range(tc.shape[0]):
            gk = G_["g"][p, i, k]
            a = sg.scale_to_snr(gk, tc[k], bgp, snr, G_["sfreq"], G_["bands"][band])
            X += sg.avg_ref(a * gk)[None, :, None] * tc[k][:, None, :]
        out = []
        pr = sp.pairs[p]
        for readout in G_["readouts"]:
            mats, names = cls._connectivity(X, readout, band)
            for r in cls._score(mats, names, readout, band, p):
                out.append(dict(pair=p, pair_class=pr.get("class", ""), node_a=pr["a"], node_b=pr["b"], band=band,
                                snr=snr, kind=kind, truth=i, lag_ms=lag, **r))
        return out

    def _init_globals(self):
        st = self.state
        G_.update(spec=self.spec, pool=st["pool"], readouts=st["readouts"], g=st["g"], bg=st["bg"],
                  bgp=st["bgp"], bands=st["bands"], sfreq=self.sfreq, pairs=self.spec.pairs,
                  metric_fn=self.metric_fn, metric_names=self.metric_names, metric_abs=self.metric_abs,
                  deep_name=nd.DEEP)
        if "C1" in st:
            G_["C1"] = st["C1"]

    def cells(self) -> list:
        sp = self.spec
        return [(p, b, float(s), k, i) for p in range(len(sp.pairs)) for b in sp.bands for s in sp.snrs_db
                for k in sp.kinds for i in range(sp.n_truths)]

    def simulate(self, cells: Optional[Sequence] = None, workers: Optional[int] = None):
        """Run the planted cells (default: every pair x band x SNR x kind x truth); returns a DataFrame."""
        import pandas as pd
        cells = self.cells() if cells is None else list(cells)
        workers = int(workers or self.spec.workers or 1)
        self._init_globals()
        rows, t0 = [], time.time()
        if workers <= 1:
            from threadpoolctl import threadpool_limits
            with threadpool_limits(1):
                for n, c in enumerate(cells):
                    rows.extend(self._one(c))
        else:
            from multiprocessing import get_context
            with get_context("fork").Pool(workers, initializer=_worker_init) as mp:
                for n, out in enumerate(mp.imap_unordered(NetworkValidation._one, cells, chunksize=4)):
                    rows.extend(out)
                    if self.verbose and (n + 1) % 1000 == 0:
                        el = time.time() - t0
                        print(f"  {n + 1}/{len(cells)}  {el / 60:.1f} min, "
                              f"eta {el / (n + 1) * (len(cells) - n - 1) / 60:.0f} min", flush=True)
        self._log(f"simulated {len(cells)} cells in {(time.time() - t0) / 60:.1f} min")
        return pd.DataFrame(rows)

    # ------------------------------------------------------------------ summaries and the whole run
    def summarise(self, sims) -> dict:
        from .scoring import detection_table, resolvable_table
        return {"resolvable": resolvable_table(sims, **{"top1": self.spec.resolvable["top1"],
                                                          "false_edges": self.spec.resolvable["false_edges"]}),
                "detection": detection_table(sims)}

    def run(self, out_dir) -> dict:
        import pandas as pd
        out = Path(out_dir)
        out.mkdir(parents=True, exist_ok=True)
        self.setup()
        directed = None
        if self.spec.directed:
            from .directed_validation import DirectedNetworkValidation
            directed = DirectedNetworkValidation(self, self.spec.directed).run(out)
        if not self.spec.undirected:
            summary = {"directed": directed, "spec": self.spec.__dict__}
            (out / "summary.json").write_text(json.dumps(summary, indent=1, default=str))
            return summary
        self.control()
        sims = self.simulate()
        sims.to_csv(out / "sims.csv.gz", index=False)
        tabs = self.summarise(sims)
        tabs["resolvable"].to_csv(out / "resolvable_networks.csv", index=False)
        tabs["detection"].to_csv(out / "detection.csv", index=False)
        for name, r in self.state["readouts"].items():
            if r["type"] == "electrodes":
                pd.DataFrame({"electrode": list(r["unit_to_node"]), "node": list(r["unit_to_node"].values())}).to_csv(
                    out / f"electrode_assignment_{name}.csv", index=False)
        pd.DataFrame(self.state["meta"]).to_csv(out / "background_draws.csv", index=False)
        n_res = int((tabs["resolvable"].resolvable_from_db != "never").sum())
        summary = {"n_simulations": int(len(self.cells())), "n_cells_resolvable": n_res,
                   "n_cells": int(len(tabs["resolvable"])), "directed": directed, "spec": self.spec.__dict__}
        (out / "summary.json").write_text(json.dumps(summary, indent=1, default=str))
        self._log(f"done: {n_res} of {len(tabs['resolvable'])} cells resolvable; outputs in {out}")
        return summary


def _worker_init():
    from threadpoolctl import threadpool_limits
    threadpool_limits(1)
