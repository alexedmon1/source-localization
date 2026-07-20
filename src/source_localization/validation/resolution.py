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
        return self._source_activity_norm(self._apply_inverse(eeg))

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
