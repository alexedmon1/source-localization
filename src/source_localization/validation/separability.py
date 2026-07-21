"""
Threshold-based separability: can you *see* two sources as two blobs?

The resolution metrics elsewhere in this package summarize a whole
reconstruction with one number (PLE, spatial dispersion, a saddle test). All of
them are contaminated by the same thing: the reconstruction sits on a broad
leakage pedestal that never decays to zero (0.25 of peak for the shallowest
sources, 0.69 for the deepest). Spatial dispersion in particular is dominated by
that pedestal and lands within 78-96% of the value a *completely flat* map would
score, so it has almost no dynamic range.

Thresholding removes the pedestal. This module asks the question the way someone
actually reads a source map: **take everything above X% of the peak and count the
blobs.** Two sources are separable at threshold X if both survive it and they
fall in different connected blobs.

That framing also gives a pedestal-free replacement for spatial dispersion — the
*extent of the supra-threshold blob* around a source (:func:`blob_extent`) — and
makes the discrimination rule intuitive: two sources are distinguishable roughly
when their separation exceeds the sum of their blob radii.

Failure to separate is decomposed into two genuinely different causes, because
they call for different responses:

- ``merged``     — both sources are above threshold but sit in one blob. The
                   classic resolution limit; a higher threshold may split them.
- ``dominated``  — the weaker source falls below the threshold entirely and is
                   invisible. Raising the threshold makes this *worse*. Common
                   here because equal-strength dipoles at different depths
                   reconstruct with very unequal amplitude.

Note on "probability": the field being thresholded is activation normalized to
its peak. It is a relative plausibility map, NOT a calibrated posterior — a 25%
threshold does not mean 25% probability. The threshold is a display/decision
knob, and the metrics below are all defined relative to the peak.

Classes
-------
SeparabilityAnalysis
    Supra-threshold blob analysis for a pipeline's forward+inverse operator.
"""
from typing import Dict, List, Optional, Sequence

import numpy as np

from .resolution import ResolutionAnalysis

__all__ = ['SeparabilityAnalysis']


class SeparabilityAnalysis(ResolutionAnalysis):
    """
    Supra-threshold blob analysis: what a thresholded reconstruction shows.

    Examples
    --------
    >>> sa = SeparabilityAnalysis.from_pipeline_dir('/path/to/results')
    >>> sa.blob_extent(sa.compute_psf(10), threshold_frac=0.5, source_idx=10)
    {'n_sources': ..., 'rms_radius_mm': ..., 'max_radius_mm': ..., ...}
    """

    #: Thresholds (fraction of the map's peak) swept by default.
    DEFAULT_THRESHOLDS = [0.15, 0.25, 0.35, 0.50, 0.65, 0.80]

    #: Neighbour radius for the source adjacency graph, in units of the median
    #: grid spacing. 1.5x links each source to its immediate neighbours without
    #: bridging across gaps, so "connected" means spatially contiguous.
    ADJACENCY_SPACING_FACTOR = 1.5

    def _adjacency_pairs(self) -> np.ndarray:
        """Edges of the source-neighbour graph (cached)."""
        if getattr(self, '_adj_cache', None) is None:
            radius = self.ADJACENCY_SPACING_FACTOR * self._median_grid_spacing()
            pairs = self._get_source_kdtree().query_pairs(radius)
            self._adj_cache = (np.array(sorted(pairs), dtype=int)
                               if pairs else np.empty((0, 2), dtype=int))
        return self._adj_cache

    def suprathreshold_labels(
        self,
        activation: np.ndarray,
        threshold_frac: float,
    ) -> np.ndarray:
        """
        Label connected supra-threshold blobs.

        Parameters
        ----------
        activation : ndarray, shape (n_sources,)
            Per-source magnitude (not necessarily normalized).
        threshold_frac : float
            Threshold as a fraction of the map's peak.

        Returns
        -------
        labels : ndarray of int, shape (n_sources,)
            Blob id per source; ``-1`` for sub-threshold sources.
        """
        from scipy.sparse import coo_matrix
        from scipy.sparse.csgraph import connected_components

        act = np.asarray(activation, float)
        peak = act.max()
        labels = np.full(self.n_sources, -1, dtype=int)
        if peak <= 0:
            return labels

        mask = act >= threshold_frac * peak
        if not mask.any():
            return labels

        pairs = self._adjacency_pairs()
        if len(pairs):
            keep = pairs[mask[pairs[:, 0]] & mask[pairs[:, 1]]]
        else:
            keep = pairs
        graph = coo_matrix(
            (np.ones(len(keep)), (keep[:, 0], keep[:, 1])),
            shape=(self.n_sources, self.n_sources))
        _, comp = connected_components(graph + graph.T, directed=False)
        labels[mask] = comp[mask]
        return labels

    def classify_pair(
        self,
        activation: np.ndarray,
        idx_a: int,
        idx_b: int,
        threshold_frac: float,
    ) -> str:
        """
        Can the two sources be told apart at this threshold?

        Returns
        -------
        str
            ``'separated'`` — both above threshold, in different blobs.
            ``'merged'`` — both above threshold, same blob.
            ``'dominated'`` — at least one source fell below the threshold, so it
            is invisible rather than merged. Raising the threshold worsens this.
        """
        act = np.asarray(activation, float)
        peak = act.max()
        if peak <= 0:
            return 'dominated'
        cut = threshold_frac * peak
        if act[idx_a] < cut or act[idx_b] < cut:
            return 'dominated'

        labels = self.suprathreshold_labels(act, threshold_frac)
        return 'separated' if labels[idx_a] != labels[idx_b] else 'merged'

    def blob_extent(
        self,
        activation: np.ndarray,
        threshold_frac: float,
        source_idx: Optional[int] = None,
    ) -> Dict[str, float]:
        """
        Size of the supra-threshold blob — the pedestal-free "dispersion".

        Measured on the blob containing ``source_idx`` (or the peak if not
        given). Unlike spatial dispersion this ignores everything below the
        threshold, so the leakage pedestal cannot inflate it.

        Returns
        -------
        dict
            ``n_sources`` in the blob, ``volume_mm3`` (blob size x the per-source
            cell volume), ``rms_radius_mm`` and ``max_radius_mm`` measured from
            the blob's centroid, and ``contains_source`` — False when the source
            itself is below threshold.
        """
        act = np.asarray(activation, float)
        labels = self.suprathreshold_labels(act, threshold_frac)

        anchor = int(np.argmax(act)) if source_idx is None else int(source_idx)
        if labels[anchor] < 0:
            return {'n_sources': 0, 'volume_mm3': 0.0, 'rms_radius_mm': np.nan,
                    'max_radius_mm': np.nan, 'contains_source': False}

        members = np.where(labels == labels[anchor])[0]
        pts = self.source_pos_mm[members]
        centroid = pts.mean(axis=0)
        d = np.linalg.norm(pts - centroid, axis=1)
        cell = self._median_grid_spacing() ** 3
        return {
            'n_sources': int(members.size),
            'volume_mm3': float(members.size * cell),
            'rms_radius_mm': float(np.sqrt((d ** 2).mean())),
            'max_radius_mm': float(d.max()),
            'contains_source': True,
        }

    def blob_extent_map(
        self,
        thresholds: Optional[Sequence[float]] = None,
        source_indices: Optional[List[int]] = None,
    ) -> Dict[str, np.ndarray]:
        """
        Per-source supra-threshold blob size, across thresholds (noise-free PSFs).

        This is the map to read instead of spatial dispersion: "at threshold T, a
        source here reconstructs as a blob of radius R".

        Returns
        -------
        dict of ndarray
            ``thresholds`` (T,), ``source_idx`` / ``depth_mm`` / ``positions_mm``
            (N,...), and (N, T) grids ``rms_radius_mm``, ``max_radius_mm``,
            ``volume_mm3``.
        """
        if thresholds is None:
            thresholds = list(self.DEFAULT_THRESHOLDS)
        if source_indices is None:
            source_indices = list(range(self.n_sources))

        n, t = len(source_indices), len(thresholds)
        rms = np.full((n, t), np.nan)
        mx = np.full((n, t), np.nan)
        vol = np.full((n, t), np.nan)

        for i, idx in enumerate(source_indices):
            psf = self.compute_psf(int(idx))
            for j, thr in enumerate(thresholds):
                e = self.blob_extent(psf, thr, source_idx=int(idx))
                rms[i, j] = e['rms_radius_mm']
                mx[i, j] = e['max_radius_mm']
                vol[i, j] = e['volume_mm3']
            if self.verbose and i % 25 == 0:
                print(f"  blob extent: {i+1}/{n}", flush=True)

        return {
            'thresholds': np.asarray(thresholds, dtype=float),
            'source_idx': np.asarray(source_indices),
            'positions_mm': self.source_pos_mm[source_indices],
            'depth_mm': self.source_depths[source_indices],
            'rms_radius_mm': rms,
            'max_radius_mm': mx,
            'volume_mm3': vol,
        }

    def separability_sweep(
        self,
        source_indices: Optional[List[int]] = None,
        separations: Optional[List[float]] = None,
        thresholds: Optional[Sequence[float]] = None,
        n_partners: int = 2,
        snr_db: float = np.inf,
        noise_type: str = 'white',
        n_trials: int = 1,
        tol_mm: float = 0.7,
        depth_match_mm: Optional[float] = None,
        seed: int = 0,
    ) -> Dict[str, np.ndarray]:
        """
        Separability over the (separation x threshold) plane.

        For every source, partner, orientation pair and threshold, classifies the
        two-source reconstruction as separated / merged / dominated.

        ``depth_match_mm`` restricts each pair to comparable depth. Strongly
        recommended when the question is "how far apart must two sources be":
        without it, most pairs fail as ``dominated`` because of the depth-driven
        gain imbalance rather than because of proximity (P(separated) at 6 mm,
        very-shallow, best threshold: 0.04 unrestricted vs 0.45 matched).

        Returns
        -------
        dict of ndarray
            ``separations_mm`` (S,), ``thresholds`` (T,), ``depth_mm`` (N,),
            ``source_idx`` (N,), and (N, S, T) fraction grids ``p_separated``,
            ``p_merged``, ``p_dominated``.
        """
        if separations is None:
            separations = list(self.DEFAULT_SEPARATIONS)
        if thresholds is None:
            thresholds = list(self.DEFAULT_THRESHOLDS)
        if source_indices is None:
            source_indices = list(range(self.n_sources))

        triad = np.eye(3)
        orient_pairs = [(o1, o2) for o1 in triad for o2 in triad]
        trials = 1 if not np.isfinite(snr_db) else n_trials

        n, S, T = len(source_indices), len(separations), len(thresholds)
        p_sep = np.full((n, S, T), np.nan)
        p_mrg = np.full((n, S, T), np.nan)
        p_dom = np.full((n, S, T), np.nan)

        for i, a in enumerate(source_indices):
            for si, sep in enumerate(separations):
                partners = self._partners_at(int(a), sep, n_partners, tol_mm,
                                             depth_match_mm=depth_match_mm)
                if not partners:
                    continue
                counts = np.zeros((T, 3))     # separated / merged / dominated
                for pi, b in enumerate(partners):
                    for oi, (o1, o2) in enumerate(orient_pairs):
                        for tr in range(trials):
                            ns = (seed if not np.isfinite(snr_db)
                                  else seed + 7919 * si + 104729 * pi + 397 * oi + tr)
                            eeg, _ = self.simulator.simulate_two_dipoles(
                                position1_mm=self.source_pos_mm[a],
                                position2_mm=self.source_pos_mm[b],
                                orientation1=o1, orientation2=o2,
                                amplitude1_nAm=50.0, amplitude2_nAm=50.0,
                                snr_db=snr_db, noise_seed=ns,
                                noise_type=noise_type, duration_s=0.2, sfreq=256.0)
                            act = self._source_activity_norm(self._apply_inverse(eeg))
                            for j, thr in enumerate(thresholds):
                                verdict = self.classify_pair(act, int(a), int(b), thr)
                                counts[j, {'separated': 0, 'merged': 1,
                                           'dominated': 2}[verdict]] += 1
                total = counts.sum(axis=1)
                ok = total > 0
                p_sep[i, si, ok] = counts[ok, 0] / total[ok]
                p_mrg[i, si, ok] = counts[ok, 1] / total[ok]
                p_dom[i, si, ok] = counts[ok, 2] / total[ok]
            if self.verbose and i % 10 == 0:
                print(f"  separability: {i+1}/{n} sources", flush=True)

        return {
            'separations_mm': np.asarray(separations, dtype=float),
            'thresholds': np.asarray(thresholds, dtype=float),
            'source_idx': np.asarray(source_indices),
            'depth_mm': self.source_depths[source_indices],
            'positions_mm': self.source_pos_mm[source_indices],
            'p_separated': p_sep,
            'p_merged': p_mrg,
            'p_dominated': p_dom,
        }

    def summarize_separability_by_depth(
        self,
        sweep: Dict[str, np.ndarray],
        depth_bins: Optional[List] = None,
    ) -> Dict[str, Dict]:
        """Aggregate a separability sweep into depth bins (mean over sources)."""
        if depth_bins is None:
            depth_bins = self.DEFAULT_DEPTH_BINS
        depth = sweep['depth_mm']
        out = {}
        for label, lo, hi in depth_bins:
            m = (depth >= np.percentile(depth, lo)) & (depth <= np.percentile(depth, hi))
            if not m.any():
                continue
            with np.errstate(invalid='ignore'):
                out[label] = {
                    'n': int(m.sum()),
                    'depth_range_mm': [float(np.percentile(depth, lo)),
                                       float(np.percentile(depth, hi))],
                    'p_separated': np.nanmean(sweep['p_separated'][m], axis=0).tolist(),
                    'p_merged': np.nanmean(sweep['p_merged'][m], axis=0).tolist(),
                    'p_dominated': np.nanmean(sweep['p_dominated'][m], axis=0).tolist(),
                }
        return out
