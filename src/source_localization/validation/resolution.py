"""
Resolution analysis for source localization — the geometric foundation.

Instead of Monte-Carlo two-lobe detection at a single SNR (which conflates the
operator's intrinsic resolution with noise), this characterizes the *geometric*
resolution of the forward+inverse operator directly, noise-free. For each source
location it computes the **point-spread function** (PSF): what the reconstruction
looks like for a unit source there, with no noise. From the PSF come the standard
literature metrics (Molins et al. 2008; Hauk et al. 2011):

- **Peak localization error (PLE):** distance from the true source to the PSF's
  peak — how far off the reconstructed maximum is.
- **Spatial dispersion (SD):** RMS spread of the PSF around the true location —
  how blurred the reconstruction is (mm). This is the resolution-relevant number.

These are exact (one forward+inverse solve per source, no noise, no sampling
variance), so the whole brain is characterized with 215 solves.

Noise/SNR degradation is layered on *top* of that geometric baseline by
:meth:`ResolutionAnalysis.resolution_vs_snr`, which repeats the same two metrics
at finite SNR. Keeping the two separate is deliberate: conflating them at a
single SNR is what produced the earlier, misleading "can't resolve anything"
result.

Classes
-------
ResolutionAnalysis
    Compute PSFs and per-source resolution metrics for a pipeline's operator,
    noise-free and across SNR.
"""
import numpy as np
from typing import Dict, List, Optional

from .robustness import RobustnessTest

__all__ = ['ResolutionAnalysis']


def _orthonormal_triad() -> np.ndarray:
    """Three orthogonal unit orientations, for orientation-averaged PSFs."""
    return np.eye(3)


def _first_stable_separation(
    excess: List[float],
    separations: List[float],
    threshold: float = 0.5,
    min_support: int = 2,
) -> float:
    """
    Smallest separation that clears ``threshold`` and stays clear beyond it.

    Resolution should improve monotonically with separation, so a shell only
    counts if every larger tested shell also clears the bar — otherwise one lucky
    shell would set the threshold. Untested shells (NaN, no partners at that
    distance) are skipped rather than treated as failures.

    ``min_support`` shells must clear the bar, which matters at the top of the
    range: the largest shell has no larger shell to contradict it, so it would
    otherwise always pass vacuously and report a resolution distance on the
    strength of a single measurement. Returns NaN if no separation qualifies.
    """
    e = np.asarray(excess, dtype=float)
    tested = ~np.isnan(e)
    passes = e >= threshold
    for k in range(len(separations)):
        if not (tested[k] and passes[k]):
            continue
        beyond = passes[k:][tested[k:]]
        if beyond.all() and beyond.size >= min_support:
            return float(separations[k])
    return np.nan


class ResolutionAnalysis(RobustnessTest):
    """
    Noise-free resolution characterization for a pipeline's forward+inverse.

    Subclasses :class:`RobustnessTest` to reuse its forward model, inverse
    application, source geometry, and KD-tree, adding point-spread-function
    computation and the standard resolution metrics.

    Examples
    --------
    >>> ra = ResolutionAnalysis.from_pipeline_dir('/path/to/results')
    >>> m = ra.resolution_map()
    >>> m['peak_localization_error_mm'].shape  # one value per source
    (215,)
    """

    def compute_psf(
        self,
        source_idx: int,
        orientations: Optional[np.ndarray] = None
    ) -> np.ndarray:
        """
        Noise-free point-spread function for a source at ``source_idx``.

        Places a unit dipole at the source (no noise), reconstructs, and returns
        the per-source reconstructed magnitude. Averaged over orthogonal
        orientations so the PSF reflects location, not a particular dipole
        direction. Normalized to a peak of 1.

        Parameters
        ----------
        source_idx : int
            Index of the source to probe.
        orientations : ndarray, shape (k, 3), optional
            Orientations to average over. Default: three orthogonal axes.

        Returns
        -------
        psf : ndarray, shape (n_sources,)
            Reconstructed magnitude per source, peak-normalized to 1.
        """
        if orientations is None:
            orientations = _orthonormal_triad()

        accum = np.zeros(self.n_sources)
        for orient in orientations:
            accum += self._reconstruct(source_idx, orient, snr_db=np.inf)

        peak = accum.max()
        return accum / peak if peak > 0 else accum

    def _reconstruct(
        self,
        source_idx: int,
        orientation: np.ndarray,
        snr_db: float = np.inf,
        noise_seed: int = 0,
        noise_type: str = 'white',
    ) -> np.ndarray:
        """
        One forward+inverse solve: reconstructed magnitude per source.

        Shared by the noise-free PSF and the SNR sweep so both characterize the
        operator through exactly the same path. Not peak-normalized.
        """
        return self._source_activity_norm(
            self._reconstruct_raw(source_idx, orientation, snr_db,
                                  noise_seed, noise_type))

    def _reconstruct_raw(
        self,
        source_idx: int,
        orientation: np.ndarray,
        snr_db: float = np.inf,
        noise_seed: int = 0,
        noise_type: str = 'white',
    ) -> np.ndarray:
        """
        As :meth:`_reconstruct`, but the raw (oriented, time) source activity.

        Detectors such as ``_detect_two_lobes`` collapse orientation/time
        themselves, so they need the uncollapsed array.
        """
        eeg, _ = self.simulator.simulate_dipole(
            position_mm=self.source_pos_mm[source_idx],
            orientation=orientation,
            amplitude_nAm=50.0,
            snr_db=snr_db,
            noise_seed=noise_seed,
            noise_type=noise_type,
            duration_s=0.2,
            sfreq=256.0,
        )
        return self._apply_inverse(eeg)

    def psf_metrics(
        self,
        source_idx: int,
        psf: Optional[np.ndarray] = None
    ) -> Dict[str, float]:
        """
        Standard resolution metrics for one source's PSF.

        Parameters
        ----------
        source_idx : int
            The true source index.
        psf : ndarray, optional
            Precomputed PSF (peak-normalized). If None, computed here.

        Returns
        -------
        dict
            ``peak_localization_error_mm`` — distance from the true source to the
            PSF peak. ``spatial_dispersion_mm`` — RMS spread of the PSF around the
            true source (activation-weighted). ``peak_contrast`` — PSF peak over
            its median, i.e. how far the peak stands above the background; near 1
            means a flat PSF whose argmax (and therefore PLE) is close to
            arbitrary. ``depth_mm`` — source depth.
        """
        if psf is None:
            psf = self.compute_psf(source_idx)

        true_pos = self.source_pos_mm[source_idx]
        dists = np.linalg.norm(self.source_pos_mm - true_pos, axis=1)

        peak_idx = int(np.argmax(psf))
        ple = float(np.linalg.norm(self.source_pos_mm[peak_idx] - true_pos))

        # Spatial dispersion: sqrt( sum d^2 psf^2 / sum psf^2 ) — RMS spread of
        # the point-spread around the TRUE location (Molins/Hauk convention).
        w = psf ** 2
        denom = w.sum()
        sd = float(np.sqrt((dists ** 2 * w).sum() / denom)) if denom > 0 else np.nan

        med = np.median(psf)
        contrast = float(psf.max() / med) if med > 0 else np.inf

        return {
            'peak_localization_error_mm': ple,
            'spatial_dispersion_mm': sd,
            'peak_contrast': contrast,
            'depth_mm': float(self.source_depths[source_idx]),
        }

    def resolution_map(
        self,
        source_indices: Optional[List[int]] = None
    ) -> Dict[str, np.ndarray]:
        """
        Per-source resolution metrics across the brain.

        Parameters
        ----------
        source_indices : list of int, optional
            Which sources to characterize. Default: all.

        Returns
        -------
        dict of ndarray
            ``source_idx``, ``positions_mm`` (N,3), ``depth_mm``,
            ``peak_localization_error_mm``, ``spatial_dispersion_mm`` — each of
            length N, aligned.
        """
        if source_indices is None:
            source_indices = list(range(self.n_sources))

        ple = np.full(len(source_indices), np.nan)
        sd = np.full(len(source_indices), np.nan)
        contrast = np.full(len(source_indices), np.nan)
        depth = np.full(len(source_indices), np.nan)
        for k, idx in enumerate(source_indices):
            m = self.psf_metrics(idx)
            ple[k] = m['peak_localization_error_mm']
            sd[k] = m['spatial_dispersion_mm']
            contrast[k] = m['peak_contrast']
            depth[k] = m['depth_mm']
            if self.verbose and k % 25 == 0:
                print(f"  resolution map: {k+1}/{len(source_indices)}", flush=True)

        return {
            'source_idx': np.asarray(source_indices),
            'positions_mm': self.source_pos_mm[source_indices],
            'depth_mm': depth,
            'peak_localization_error_mm': ple,
            'spatial_dispersion_mm': sd,
            'peak_contrast': contrast,
        }

    #: Default SNR ladder for :meth:`resolution_vs_snr`. ``inf`` is the
    #: noise-free Phase-1 geometric anchor; the finite rungs bracket the
    #: 10 dB regime the earlier sweeps used as their single operating point.
    DEFAULT_SNR_VALUES = [-5.0, 0.0, 5.0, 10.0, 20.0, np.inf]

    def resolution_vs_snr(
        self,
        snr_values: Optional[List[float]] = None,
        source_indices: Optional[List[int]] = None,
        n_trials: int = 10,
        noise_type: str = 'colored',
        orientations: Optional[np.ndarray] = None,
        seed: int = 0,
    ) -> Dict[str, np.ndarray]:
        """
        How PLE and spatial dispersion degrade with SNR, per source.

        Repeats the Phase-1 metrics at finite SNR. The two metrics are
        aggregated differently on purpose:

        - **PLE** is computed per trial (the peak moves from trial to trial),
          then reported as a median and IQR over trials — a distribution.
        - **SD** is computed from the *trial-averaged* reconstruction. Spatial
          dispersion off a single noisy map is dominated by noise speckle far
          from the source and is not a stable blur estimate; averaging the maps
          first recovers the noise-degraded point-spread.

        Orientations are cycled across trials so the sweep stays
        orientation-averaged, like :meth:`compute_psf`.

        Parameters
        ----------
        snr_values : list of float, optional
            SNR levels in dB. Default :data:`DEFAULT_SNR_VALUES`. ``np.inf``
            is the noise-free anchor and is run once (it is deterministic).
        source_indices : list of int, optional
            Sources to characterize. Default: all.
        n_trials : int, default=10
            Noise realizations per source per finite SNR.
        noise_type : str, default='colored'
            Noise structure (see :mod:`validation.noise`). ``'colored'`` is the
            realistic bound; ``'white'`` matches the C=I inverse assumption.
        orientations : ndarray, shape (k, 3), optional
            Orientations to cycle over. Default: three orthogonal axes.
        seed : int, default=0
            Base seed; each (source, SNR, trial) gets a distinct derived seed.

        Returns
        -------
        dict of ndarray
            ``snr_values`` (S,), ``source_idx`` (N,), ``depth_mm`` (N,),
            ``positions_mm`` (N,3), and four (N, S) arrays: ``ple_median_mm``,
            ``ple_iqr_mm``, ``spatial_dispersion_mm``, ``peak_contrast``.
        """
        if snr_values is None:
            snr_values = list(self.DEFAULT_SNR_VALUES)
        if source_indices is None:
            source_indices = list(range(self.n_sources))
        if orientations is None:
            orientations = _orthonormal_triad()

        n_src, n_snr = len(source_indices), len(snr_values)
        ple_med = np.full((n_src, n_snr), np.nan)
        ple_iqr = np.full((n_src, n_snr), np.nan)
        sd = np.full((n_src, n_snr), np.nan)
        contrast = np.full((n_src, n_snr), np.nan)

        for i, idx in enumerate(source_indices):
            true_pos = self.source_pos_mm[idx]
            dists = np.linalg.norm(self.source_pos_mm - true_pos, axis=1)

            for j, snr_db in enumerate(snr_values):
                # Noise-free is deterministic — one solve per orientation.
                trials = 1 if not np.isfinite(snr_db) else n_trials

                ples = []
                accum = np.zeros(self.n_sources)
                for t in range(trials):
                    orient = orientations[t % len(orientations)]
                    recon = self._reconstruct(
                        idx, orient, snr_db=snr_db,
                        noise_seed=seed + 1000003 * i + 1009 * j + t,
                        noise_type=noise_type,
                    )
                    peak = recon.max()
                    recon = recon / peak if peak > 0 else recon
                    ples.append(float(dists[int(np.argmax(recon))]))
                    accum += recon

                ple_med[i, j] = float(np.median(ples))
                ple_iqr[i, j] = float(np.subtract(*np.percentile(ples, [75, 25])))
                avg = self.psf_metrics(idx, psf=accum / trials)
                sd[i, j] = avg['spatial_dispersion_mm']
                contrast[i, j] = avg['peak_contrast']

            if self.verbose and i % 10 == 0:
                print(f"  resolution vs SNR: {i+1}/{n_src} sources", flush=True)

        return {
            'snr_values': np.asarray(snr_values, dtype=float),
            'source_idx': np.asarray(source_indices),
            'positions_mm': self.source_pos_mm[source_indices],
            'depth_mm': self.source_depths[source_indices],
            'ple_median_mm': ple_med,
            'ple_iqr_mm': ple_iqr,
            'spatial_dispersion_mm': sd,
            'peak_contrast': contrast,
        }

    #: Separation shells (mm) probed by :meth:`resolution_distance`, ascending.
    DEFAULT_SEPARATIONS = [2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 10.0, 12.0]

    #: Minimum noise-free PSF peak contrast for a location's resolution distance
    #: to be meaningful. Below this the PSF is flat enough that its peak — and so
    #: any two-lobe test built on it — is arbitrary. Calibrated from the Phase-1
    #: map: noise-free PLE stays at 0.00 mm down to contrast ~1.3 and only becomes
    #: nonzero below it (22/215 sources, median depth 5.8 mm).
    MIN_PEAK_CONTRAST = 1.3

    def _partners_at(
        self,
        source_idx: int,
        separation_mm: float,
        n_partners: int,
        tol_mm: float,
        depth_match_mm: Optional[float] = None,
    ) -> List[int]:
        """
        Sources sitting ``separation_mm`` away, closest-to-exact first.

        ``depth_match_mm`` restricts partners to comparable depth. This matters
        more than it looks: gain falls off steeply with depth, so two
        equal-strength dipoles at different depths reconstruct with very unequal
        amplitude (mean ratio 0.38 for unrestricted partners vs 0.71 when matched
        to +-0.4 mm). Unmatched pairs therefore fail a threshold test because the
        weaker source drops out, which is a gain effect masquerading as a
        resolution limit. Pass a tolerance to isolate the resolution question.
        """
        d = np.linalg.norm(self.source_pos_mm - self.source_pos_mm[source_idx], axis=1)
        cand = np.where(np.abs(d - separation_mm) <= tol_mm)[0]
        cand = cand[cand != source_idx]
        if depth_match_mm is not None:
            depth_gap = np.abs(self.source_depths[cand]
                               - self.source_depths[source_idx])
            cand = cand[depth_gap <= depth_match_mm]
        return cand[np.argsort(np.abs(d[cand] - separation_mm))][:n_partners].tolist()

    def resolution_distance(
        self,
        source_idx: int,
        separations: Optional[List[float]] = None,
        n_partners: int = 2,
        snr_db: float = np.inf,
        noise_type: str = 'white',
        n_trials: int = 1,
        dip_ratio: float = 0.8,
        tol_mm: float = 0.7,
        seed: int = 0,
    ) -> Dict:
        """
        Minimum separation at which a second source is distinguishable.

        For each separation shell, places a second dipole that far from
        ``source_idx`` and asks whether the reconstruction shows two lobes (a
        saddle between them). The reported distance is the smallest shell where a
        majority of pairs show the saddle **and** every larger shell does too —
        the monotonicity requirement stops a single lucky shell from setting the
        threshold.

        Pairs are simulated directly rather than predicted by summing the two
        PSFs: because the PSF is an orientation-averaged *magnitude*, superposing
        PSFs only reproduces the direct simulation's saddle verdict ~72-76% of the
        time at 5-8 mm — precisely the separations that set the threshold.

        Scoring uses :meth:`~.robustness.RobustnessTest._detect_two_lobes` — two
        *prominent* local maxima with a saddle between them. A bare
        "midpoint below both endpoints" test is not usable here: it never
        requires the second location to be a lobe at all, so a flat or
        mislocalized PSF trips it with a single source present (observed
        noise-free at up to 67% of pairs). Because the detector is
        correspondence-free, the identical event scores the
        position-matched single-source null, and the threshold is set on the
        **excess** ``P(two lobes | two sources) - P(two lobes | one source)``.

        Parameters
        ----------
        source_idx : int
            The source whose resolution distance is being measured.
        separations : list of float, optional
            Separation shells in mm, ascending. Default
            :data:`DEFAULT_SEPARATIONS`.
        n_partners : int, default=2
            Partner sources per shell (closest to the exact separation).
        snr_db : float, default=inf
            Noise-free by default — this is a geometric limit. Finite values
            layer the Phase-2 noise story on top.
        noise_type : str, default='white'
            Noise structure when ``snr_db`` is finite.
        n_trials : int, default=1
            Noise realizations per pair per orientation. Forced to 1 when
            noise-free (deterministic).
        dip_ratio : float, default=0.8
            Saddle depth required, as a fraction of the weaker lobe.
        tol_mm : float, default=0.7
            How far a partner may deviate from the nominal separation.
        seed : int, default=0
            Base seed for noise realizations.

        Returns
        -------
        dict
            ``resolution_distance_mm`` (nan if never resolved within the tested
            shells), ``separations_mm``, ``detect_fraction`` / ``null_fraction`` /
            ``excess_fraction`` per shell, ``n_pairs`` per shell,
            ``peak_contrast`` and ``depth_mm``
            of the source, and ``meaningful`` — False when the PSF is too flat
            (see :data:`MIN_PEAK_CONTRAST`) for the result to be trusted.
        """
        if separations is None:
            separations = list(self.DEFAULT_SEPARATIONS)
        orient_pairs = [(a, b) for a in _orthonormal_triad()
                        for b in _orthonormal_triad()]
        trials = 1 if not np.isfinite(snr_db) else n_trials

        contrast = self.psf_metrics(source_idx)['peak_contrast']

        # The null depends only on (source, orientation of dipole 1, noise draw) —
        # not on the partner or the shell — so cache it instead of re-solving it
        # once per pair. Noise-free it collapses to one solve per orientation.
        null_cache: Dict = {}

        def null_detect(o1, oi, ns):
            key = (oi, ns)
            if key not in null_cache:
                solo = self._reconstruct_raw(source_idx, o1, snr_db=snr_db,
                                             noise_seed=ns, noise_type=noise_type)
                null_cache[key] = self._detect_two_lobes(
                    solo, saddle_ratio=dip_ratio)['detected']
            return null_cache[key]

        detect_frac, null_frac, n_pairs = [], [], []
        for si, sep in enumerate(separations):
            partners = self._partners_at(source_idx, sep, n_partners, tol_mm)
            hits, nulls = [], []
            for pi, b in enumerate(partners):
                for oi, (o1, o2) in enumerate(orient_pairs):
                    for t in range(trials):
                        # Noise-free is deterministic, so the seed is irrelevant;
                        # holding it fixed lets the null cache collapse.
                        ns = (seed if not np.isfinite(snr_db)
                              else seed + 7919 * si + 104729 * pi + 397 * oi + t)
                        eeg, _ = self.simulator.simulate_two_dipoles(
                            position1_mm=self.source_pos_mm[source_idx],
                            position2_mm=self.source_pos_mm[b],
                            orientation1=o1, orientation2=o2,
                            amplitude1_nAm=50.0, amplitude2_nAm=50.0,
                            snr_db=snr_db, noise_seed=ns, noise_type=noise_type,
                            duration_s=0.2, sfreq=256.0)
                        act = self._apply_inverse(eeg)
                        hits.append(self._detect_two_lobes(
                            act, saddle_ratio=dip_ratio)['detected'])

                        # Position-matched null: dipole 1 only, same geometry,
                        # same seed, scored by the identical event.
                        # orient_pairs is outer(o1) x inner(o2), so o1's index is oi // 3.
                        nulls.append(null_detect(o1, oi // 3, ns))
            detect_frac.append(float(np.mean(hits)) if hits else np.nan)
            null_frac.append(float(np.mean(nulls)) if nulls else np.nan)
            n_pairs.append(len(partners))

        # Threshold on the null-corrected excess, not the raw detection rate.
        excess = [d - n for d, n in zip(detect_frac, null_frac)]
        rd = _first_stable_separation(excess, separations)

        return {
            'source_idx': int(source_idx),
            'resolution_distance_mm': rd,
            'separations_mm': list(separations),
            'detect_fraction': detect_frac,
            'null_fraction': null_frac,
            'excess_fraction': excess,
            'n_pairs': n_pairs,
            'peak_contrast': contrast,
            'depth_mm': float(self.source_depths[source_idx]),
            'meaningful': bool(contrast >= self.MIN_PEAK_CONTRAST),
        }

    def resolution_distance_map(
        self,
        source_indices: Optional[List[int]] = None,
        **kwargs
    ) -> Dict[str, np.ndarray]:
        """
        Resolution distance for many sources — the "how far apart is far enough" map.

        Parameters
        ----------
        source_indices : list of int, optional
            Sources to characterize. Default: all.
        **kwargs
            Passed through to :meth:`resolution_distance`.

        Returns
        -------
        dict of ndarray
            ``source_idx``, ``positions_mm``, ``depth_mm``, ``peak_contrast``,
            ``meaningful`` (bool), ``resolution_distance_mm``, plus
            ``separations_mm`` (S,) and the (N, S) ``detect_fraction`` /
            ``null_fraction`` / ``excess_fraction`` grids.
        """
        if source_indices is None:
            source_indices = list(range(self.n_sources))

        rows = []
        for k, idx in enumerate(source_indices):
            rows.append(self.resolution_distance(idx, **kwargs))
            if self.verbose and k % 10 == 0:
                print(f"  resolution distance: {k+1}/{len(source_indices)}", flush=True)

        return {
            'source_idx': np.asarray(source_indices),
            'positions_mm': self.source_pos_mm[source_indices],
            'depth_mm': np.array([r['depth_mm'] for r in rows]),
            'peak_contrast': np.array([r['peak_contrast'] for r in rows]),
            'meaningful': np.array([r['meaningful'] for r in rows]),
            'resolution_distance_mm': np.array([r['resolution_distance_mm'] for r in rows]),
            'separations_mm': np.asarray(rows[0]['separations_mm'], dtype=float),
            'detect_fraction': np.array([r['detect_fraction'] for r in rows]),
            'null_fraction': np.array([r['null_fraction'] for r in rows]),
            'excess_fraction': np.array([r['excess_fraction'] for r in rows]),
        }

    def summarize_distance_by_depth(
        self,
        dmap: Dict[str, np.ndarray],
        depth_bins: Optional[List] = None
    ) -> Dict[str, Dict]:
        """
        Aggregate a resolution-distance map into depth bins.

        Only sources with a meaningful (peaked enough) PSF contribute to the
        distance statistics; the rest are counted in ``n_unreliable``, because
        averaging in locations whose peak is arbitrary would manufacture a
        resolution number where there is none.
        """
        if depth_bins is None:
            depth_bins = self.DEFAULT_DEPTH_BINS
        depth = dmap['depth_mm']
        out = {}
        for label, lo, hi in depth_bins:
            m = (depth >= np.percentile(depth, lo)) & (depth <= np.percentile(depth, hi))
            if not m.any():
                continue
            ok = m & dmap['meaningful']
            rd = dmap['resolution_distance_mm'][ok]
            never = int(np.isnan(rd).sum())
            finite = rd[~np.isnan(rd)]
            out[label] = {
                'n': int(m.sum()),
                'n_unreliable_flat_psf': int((m & ~dmap['meaningful']).sum()),
                'n_never_resolved': never,
                'depth_range_mm': [float(np.percentile(depth, lo)),
                                   float(np.percentile(depth, hi))],
                'median_resolution_distance_mm': (float(np.median(finite))
                                                  if finite.size else None),
                'max_null_fraction': float(np.nanmax(dmap['null_fraction'][ok]))
                                     if ok.any() else None,
            }
        return out

    def summarize_vs_snr_by_depth(
        self,
        sweep: Dict[str, np.ndarray],
        depth_bins: Optional[List] = None
    ) -> Dict[str, Dict[str, Dict[str, float]]]:
        """
        Aggregate a :meth:`resolution_vs_snr` sweep into depth bins.

        Returns ``{depth_label: {snr_str: {n, median_ple_mm, median_sd_mm}}}``.
        SNR keys are strings (``'10.0'``, ``'inf'``) so the result is
        JSON-serializable.
        """
        if depth_bins is None:
            depth_bins = self.DEFAULT_DEPTH_BINS
        depth = sweep['depth_mm']
        out = {}
        for label, lo, hi in depth_bins:
            m = (depth >= np.percentile(depth, lo)) & (depth <= np.percentile(depth, hi))
            if not m.any():
                continue
            per_snr = {}
            for j, snr_db in enumerate(sweep['snr_values']):
                per_snr[str(snr_db)] = {
                    'n': int(m.sum()),
                    'median_ple_mm': float(np.nanmedian(sweep['ple_median_mm'][m, j])),
                    'median_ple_iqr_mm': float(np.nanmedian(sweep['ple_iqr_mm'][m, j])),
                    'median_sd_mm': float(np.nanmedian(sweep['spatial_dispersion_mm'][m, j])),
                    'median_peak_contrast': float(np.nanmedian(sweep['peak_contrast'][m, j])),
                }
            out[label] = {
                'depth_range_mm': [float(np.percentile(depth, lo)),
                                   float(np.percentile(depth, hi))],
                'by_snr': per_snr,
            }
        return out

    def summarize_by_depth(
        self,
        resolution_map: Dict[str, np.ndarray],
        depth_bins: Optional[List] = None
    ) -> Dict[str, Dict[str, float]]:
        """Aggregate a resolution map into depth bins (median PLE and SD)."""
        if depth_bins is None:
            depth_bins = self.DEFAULT_DEPTH_BINS
        depth = resolution_map['depth_mm']
        out = {}
        for label, lo, hi in depth_bins:
            m = (depth >= np.percentile(depth, lo)) & (depth <= np.percentile(depth, hi))
            if not m.any():
                continue
            out[label] = {
                'n': int(m.sum()),
                'depth_range_mm': [float(np.percentile(depth, lo)),
                                   float(np.percentile(depth, hi))],
                'median_ple_mm': float(np.median(resolution_map['peak_localization_error_mm'][m])),
                'median_sd_mm': float(np.median(resolution_map['spatial_dispersion_mm'][m])),
            }
            if 'peak_contrast' in resolution_map:
                out[label]['median_peak_contrast'] = float(
                    np.median(resolution_map['peak_contrast'][m]))
        return out
