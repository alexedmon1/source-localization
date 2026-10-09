"""Nodes for network validation: a run's source pool, node labels, node operators, the electrode assignment.

Ported from the probability-atlas method workspace (``pa/networks.py``, 2026-10).

- **Pool:** one fixed-orientation leadfield column per source of a pipeline run, with its position, whether it is
  a surface source, and its parcel (by the pipeline's own rule, ``steps.monte_carlo_roi._labels_for_pool``).
- **Node labels:** parcels as they are, or with every volume source collapsed into one ``Deep`` node, or merged
  through a map (parcel -> region). Merging decides what a readout can name: coupling inside one node is
  invisible to it by construction.
- **Node operator:** the Monte Carlo ROI operator built exactly as the pipeline builds parcels
  (``source_space.realizations.build_roi_operator``), over the chosen labels.
- **Electrode assignment:** each electrode belongs to the node whose sources it sees most strongly, so electrode
  edges can be scored on the same nodes.
"""
from __future__ import annotations

import pickle
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping, Optional, Sequence

import numpy as np

from .signals import avg_ref

__all__ = ["DEEP", "Pool", "node_labels", "node_operator", "assign_electrodes", "edge_pair"]

DEEP = "Deep"

#: Monte Carlo operator settings used when a run's config has none (the pipeline's hybrid defaults).
DEFAULT_MONTE_CARLO = {"n_sources": 48, "n_sources_volume": 36, "n_draws": 100, "seed": 20260821,
                       "align_draws": False}


@dataclass
class Pool:
    """A run's source pool: fixed-orientation leadfield (unreferenced), positions, surface flag, parcel labels."""

    G: np.ndarray                # (n_channels, n_pool)
    xyz: np.ndarray              # (n_pool, 3) mm
    is_surf: np.ndarray          # (n_pool,) bool
    labels: list                 # parcel per pool source, None if unlabelled
    channels: list
    cfg: dict

    @classmethod
    def from_run(cls, run_dir) -> "Pool":
        """From a completed pipeline run (``data/config_resolved.yaml``, ``step3``, ``step4``), read only.

        Hybrid runs use :func:`source_space.hybrid.pool_columns` (surface normals when the inverse orientation is
        fixed). Other source types collapse each source's three columns to their dominant direction.
        """
        import yaml
        from ...steps.monte_carlo_roi import _labels_for_pool
        D = Path(run_dir) / "data"
        cfg = yaml.safe_load((D / "config_resolved.yaml").read_text())["config"]
        fwd = pickle.load(open(D / "step4_forward.pkl", "rb"))
        src = pickle.load(open(D / "step3_source_space.pkl", "rb"))
        xyz = np.load(D / "step3_source_coords_mm.npy")
        if cfg["pipeline"]["source_type"] == "hybrid":
            from ...source_space.hybrid import pool_columns
            G, is_surf = pool_columns(fwd, cfg["inverse"].get("orientation", "fixed"))
        else:
            G3 = fwd["sol"]["data"]
            n = G3.shape[1] // 3
            blocks = G3.reshape(G3.shape[0], n, 3)
            G = np.empty((G3.shape[0], n))
            for i in range(n):
                u, s, _ = np.linalg.svd(blocks[:, i, :], full_matrices=False)
                G[:, i] = u[:, 0] * s[0]
            is_surf = np.zeros(n, bool)
        n = G.shape[1]
        labels = list(_labels_for_pool(cfg, {"src": src, "source_coords_mm": xyz}, n))
        return cls(G=np.asarray(G, float), xyz=np.asarray(xyz[:n], float), is_surf=np.asarray(is_surf, bool),
                   labels=labels, channels=list(fwd["sol"]["row_names"]), cfg=cfg)


def node_labels(pool: Pool, collapse_volume: bool = True, merge_map: Optional[Mapping[str, str]] = None,
                deep_name: str = DEEP) -> np.ndarray:
    """Node per pool source.

    ``collapse_volume``: every volume source becomes ``deep_name`` (one compartment, never sub-labelled).
    ``merge_map``: surface parcels mapped to merged regions (parcels absent from the map keep their name).
    Unlabelled sources are None and never enter a node.
    """
    out = np.empty(len(pool.labels), dtype=object)
    for k, (lab, surf) in enumerate(zip(pool.labels, pool.is_surf)):
        if collapse_volume and not surf:
            out[k] = deep_name
        elif lab is None:
            out[k] = None
        else:
            out[k] = merge_map.get(lab, lab) if merge_map is not None else lab
    return out


def node_operator(pool: Pool, labels: Sequence) -> tuple:
    """(n_nodes, n_channels) Monte Carlo operator over ``labels`` exactly as the pipeline builds parcels."""
    from ...source_space.realizations import build_roi_operator
    mc = {**DEFAULT_MONTE_CARLO, **(pool.cfg.get("source_space", {}).get("monte_carlo") or {})}
    inv = pool.cfg["inverse"]
    lambda2 = inv.get("lambda2") or 1.0 / float(inv.get("snr", 3.0)) ** 2
    strata = [(idx, int(n)) for idx, n in ((np.flatnonzero(pool.is_surf), mc["n_sources"]),
                                            (np.flatnonzero(~pool.is_surf), mc["n_sources_volume"])) if len(idx)]
    op, nodes, _ = build_roi_operator(pool.G, pool.xyz, list(labels), n_sources=int(mc["n_sources"]),
                                      k=int(mc["n_draws"]), seed=int(mc["seed"]), lambda2=lambda2,
                                      align_draws=bool(mc.get("align_draws", False)), strata=strata)
    return np.asarray(op, float), list(nodes)


def assign_electrodes(pool: Pool, labels: Sequence, nodes: Sequence) -> dict:
    """Electrode -> node with the largest mean squared average-referenced gain over the node's pool sources."""
    Ga = avg_ref(pool.G, axis=0)
    labels = np.asarray(labels, dtype=object)
    gain = np.array([(Ga[:, labels == n] ** 2).mean(1) for n in nodes])        # (n_nodes, n_channels)
    return {ch: nodes[int(np.argmax(gain[:, e]))] for e, ch in enumerate(pool.channels)}


def edge_pair(a: str, b: str, node_of: Optional[dict] = None):
    """Unordered node pair of an edge between units ``a`` and ``b``; None if both lie in one node."""
    na, nb = (node_of[a], node_of[b]) if node_of is not None else (a, b)
    return None if na == nb else frozenset((na, nb))
