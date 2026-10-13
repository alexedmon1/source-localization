"""Scoring planted-network simulations: the resolvable-network table and detection.

Ported from the probability-atlas method workspace (``scripts/phase9_summarise.py``, 2026-10).

- **Resolvable:** a node pair is resolvable for a readout, metric, band and coupling kind at an SNR if top-1
  attribution >= ``top1`` (default 0.70) and the false-edge rate under a single uncoupled source (control 2) is
  <= ``false_edges`` (default 0.10), and both hold at that SNR and every higher one. The table gives the lowest
  such SNR, or ``never``.
- **Detection:** the AUC between the planted pair's edge z under coupling and under a single source (control 2),
  and under two uncoupled sources (control 3).
"""
from __future__ import annotations

import numpy as np

__all__ = ["COUPLED", "auc", "resolvable_snr", "resolvable_table", "detection_table"]

COUPLED = ("lagged", "envelope", "zero")


def auc(pos, neg) -> float:
    """Mann-Whitney AUC of ``pos`` over ``neg`` (NaN-safe)."""
    import pandas as pd
    pos, neg = np.asarray(pos, float), np.asarray(neg, float)
    pos, neg = pos[np.isfinite(pos)], neg[np.isfinite(neg)]
    if not len(pos) or not len(neg):
        return np.nan
    ranks = pd.Series(np.concatenate([pos, neg])).rank().to_numpy()
    return float((ranks[:len(pos)].sum() - len(pos) * (len(pos) + 1) / 2) / (len(pos) * len(neg)))


def resolvable_snr(top_by_snr, false_by_snr, top1: float = 0.70, false_edges: float = 0.10):
    """Lowest SNR from which both criteria hold at it and at every higher SNR (None if never)."""
    snrs = sorted(top_by_snr.index)
    ok = [(top_by_snr.get(s, 0) >= top1) and (false_by_snr.get(s, 1) <= false_edges) for s in snrs]
    best = None
    for s, good in zip(reversed(snrs), reversed(ok)):
        if not good:
            break
        best = s
    return best


def resolvable_table(sims, top1: float = 0.70, false_edges: float = 0.10):
    """One row per pair x readout x metric x band x coupled kind, with ``resolvable_from_db`` and the per-SNR
    top-1 and false-edge rates."""
    import pandas as pd
    keys = ["pair", "pair_class", "node_a", "node_b", "readout", "metric", "band"]
    agg = sims.groupby(keys + ["kind", "snr"]).agg(
        top=("top_hit", "mean"), false=("false_frac", "mean"), attributable=("attributable", "max")).reset_index()
    rows = []
    for k, g in agg.groupby(keys):
        single = g[g.kind == "single"].set_index("snr")["false"]
        for kind in COUPLED:
            gk = g[g.kind == kind].set_index("snr")
            if gk.empty:
                continue
            att = bool(gk["attributable"].any())
            r = resolvable_snr(gk["top"], single, top1, false_edges) if att else None
            rows.append(dict(zip(keys, k), kind=kind, attributable=att,
                             resolvable_from_db=("never" if r is None else r),
                             **{f"top1_{int(s):+d}dB": round(v, 2) for s, v in gk["top"].items()},
                             **{f"false_single_{int(s):+d}dB": round(v, 3) for s, v in single.items()}))
    return pd.DataFrame(rows)


def detection_table(sims):
    """Detection AUC of the planted pair's z (coupled vs control 2 and vs control 3) per readout,
    metric, band, SNR, pair and coupled kind."""
    import pandas as pd
    rows = []
    for k, g in sims.groupby(["readout", "metric", "band", "snr", "pair"]):
        for kind in COUPLED:
            pos = g[g.kind == kind].z_target
            if pos.empty:
                continue
            rows.append(dict(zip(["readout", "metric", "band", "snr", "pair"], k), kind=kind,
                             auc_vs_single=auc(pos, g[g.kind == "single"].z_target),
                             auc_vs_uncoupled=auc(pos, g[g.kind == "uncoupled"].z_target)))
    return pd.DataFrame(rows)
