"""
How far off, in millimetres, is a parcel attribution?

A confusion matrix scores attribution categorically: a source credited to the
neighbouring parcel counts as wrong exactly as one credited to the other
hemisphere does. Peak displacement asks the physical question instead. For each
true parcel *i*, with ``P(estimated = j | true = i)`` the row-normalised
confusion matrix,

    displacement_i = sum_j P(estimated = j | true = i) * ||centroid_j - centroid_i||

the expected distance between the parcel a source is in and the parcel it is
credited to. Correct attributions contribute zero.

This is the statistic of the published resting-state paper (MS1, revision 1,
``scripts/wp9_displacement_mm.py``), which also smooths its attribution maps by
it. It is ported here unchanged, so that any operator — ROI-based, Monte Carlo,
hybrid — can be scored in the same units, and the published table is the
reproduction check (``surface-evoked/analysis/displacement_ms1_reproduce.py``).

Alongside displacement, per true parcel:

- ``displacement_given_error_mm`` — the same expectation over misattributions
  only;
- ``null_displacement_mm`` — the expected distance if the label were assigned
  uniformly at random over the parcels that have a centroid (the parcel itself
  included, at distance zero), so a displacement has a reference;
- ``information_gain`` — ``1 - displacement / null``; 1 is perfect, 0 is
  chance, negative is worse than chance;
- ``displacement_in_radii`` — displacement in units of the parcel's RMS radius.

Centroids matter. MS1 took them from its analysis source space. A comparison
between operators is matched only if every operator is scored against the
**same** centroids, so :func:`parcel_geometry` takes whatever points the caller
decides parcels are made of, and the choice belongs in the caller's report.
"""
from __future__ import annotations

from typing import Callable, Dict, Mapping, Optional, Sequence

import numpy as np
import pandas as pd

__all__ = ['parcel_geometry', 'displacement_table', 'operator_attributor']


def parcel_geometry(coords_mm: np.ndarray, labels: Sequence) -> pd.DataFrame:
    """
    Centroid, RMS radius and point count of each parcel.

    Parameters
    ----------
    coords_mm : ndarray, shape (n_points, 3)
    labels : sequence, length n_points
        Parcel of each point. Points labelled ``None``, ``''`` or ``0`` are
        unassigned and ignored.

    Returns
    -------
    DataFrame indexed by parcel, columns ``cx, cy, cz, radius_mm, n_points``.
    """
    coords = np.asarray(coords_mm, float)
    lab = np.asarray(labels, dtype=object)
    if len(lab) != len(coords):
        raise ValueError(f'{len(lab)} labels for {len(coords)} points')
    keep = np.array([x is not None and x != '' and x != 0 for x in lab])
    rows = {}
    for p in pd.unique(lab[keep]):
        pts = coords[lab == p]
        c = pts.mean(axis=0)
        rows[p] = dict(cx=c[0], cy=c[1], cz=c[2],
                       radius_mm=float(np.sqrt(((pts - c) ** 2).sum(axis=1).mean())),
                       n_points=int(len(pts)))
    return pd.DataFrame.from_dict(rows, orient='index')


def displacement_table(confusion: pd.DataFrame, geometry: pd.DataFrame) -> pd.DataFrame:
    """
    Expected peak displacement per true parcel, from a confusion matrix.

    Parameters
    ----------
    confusion : DataFrame
        Rows are true parcels, columns estimated parcels; entries are
        attribution weights (counts or row-normalised probabilities). Displacement
        normalises each row over the columns that have a centroid, skipping any
        catch-all column such as ``<outside>``; recall is over the whole row.
        Both are MS1's conventions. Rows absent from ``geometry`` are skipped.
    geometry : DataFrame
        From :func:`parcel_geometry`.

    Returns
    -------
    DataFrame, one row per scored true parcel, sorted by displacement.
    """
    cent = {p: geometry.loc[p, ['cx', 'cy', 'cz']].to_numpy(float) for p in geometry.index}
    names = list(cent)
    rows = []
    for true, row in confusion.iterrows():
        if true not in cent:
            continue
        ci = cent[true]
        w_all = d_all = w_err = d_err = 0.0
        for est, w in row.items():
            if est not in cent:
                continue
            w = float(w)
            if not w > 0:
                continue
            d = float(np.linalg.norm(cent[est] - ci))
            w_all += w
            d_all += w * d
            if est != true:
                w_err += w
                d_err += w * d
        if w_all <= 0:
            continue
        disp = d_all / w_all
        null = float(np.mean([np.linalg.norm(cent[j] - ci) for j in names]))
        radius = float(geometry.loc[true, 'radius_mm'])
        rows.append({
            'roi': true,
            'n_points': int(geometry.loc[true, 'n_points']),
            'parcel_radius_mm': radius,
            'centroid_r_mm': float(np.linalg.norm(ci)),
            # over the whole row, catch-all columns included, as MS1 reports it
            'recall': float(row.get(true, 0.0)) / float(row.sum()),
            'displacement_mm': disp,
            'displacement_given_error_mm': d_err / w_err if w_err > 0 else np.nan,
            'null_displacement_mm': null,
            'information_gain': 1.0 - disp / null if null > 0 else np.nan,
            'displacement_in_radii': disp / radius if radius > 0 else np.nan,
        })
    return pd.DataFrame(rows).sort_values('displacement_mm').reset_index(drop=True)


def operator_attributor(
    operator: np.ndarray,
    operator_parcels: Sequence[str],
    parcel_names: Sequence[str],
    normalize: bool = True,
) -> Callable[[int, np.ndarray, float, np.random.Generator], int]:
    """
    Attributor for :func:`.roi_certainty.confusion_matrices` from a linear
    parcel operator: the parcel whose output carries the most power.

    Every source space that ends in parcel time series by a fixed linear map —
    the Monte Carlo ROI operator, the hybrid's — is scored this way, so the
    readout is the same across them.

    Parameters
    ----------
    operator : ndarray, shape (n_operator_parcels, n_channels)
        Maps sensor data to parcel time series.
    operator_parcels : sequence of str
        Parcel name of each operator row.
    parcel_names : sequence of str
        The confusion matrix's parcel order (``ParcelMap.parcel_names``). Every
        operator parcel must be among them.
    normalize : bool, default=True
        Divide each row by its norm before reading out, so a parcel does not win
        by having a larger gain. This is the pattern-level readout used in
        surface-evoked; ``False`` reads the operator's raw output.
    """
    W = np.asarray(operator, float)
    if normalize:
        W = W / np.linalg.norm(W, axis=1, keepdims=True)
    index: Mapping[str, int] = {n: i for i, n in enumerate(parcel_names)}
    missing = [p for p in operator_parcels if p not in index]
    if missing:
        raise ValueError(f'operator parcels not in the confusion matrix: {missing}')
    col = np.array([index[p] for p in operator_parcels])

    def attribute(gi, data, noise_std, rng):
        y = W @ np.atleast_2d(np.asarray(data, float).reshape(W.shape[1], -1))
        return int(col[int((y ** 2).sum(axis=1).argmax())])
    return attribute
