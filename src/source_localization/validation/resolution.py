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
variance), so the whole brain is characterized with 215 solves. Noise/SNR
degradation is layered on separately (see robustness.py).

Classes
-------
ResolutionAnalysis
    Compute PSFs and per-source resolution metrics for a pipeline's operator.
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
            eeg, _ = self.simulator.simulate_dipole(
                position_mm=self.source_pos_mm[source_idx],
                orientation=orient,
                amplitude_nAm=50.0,
                snr_db=np.inf,          # noise-free: pure geometric response
                noise_seed=0,
                noise_type='white',
                duration_s=0.2,
                sfreq=256.0,
            )
            accum += self._source_activity_norm(self._apply_inverse(eeg))

        peak = accum.max()
        return accum / peak if peak > 0 else accum

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
            true source (activation-weighted). ``depth_mm`` — source depth.
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

        return {
            'peak_localization_error_mm': ple,
            'spatial_dispersion_mm': sd,
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
        depth = np.full(len(source_indices), np.nan)
        for k, idx in enumerate(source_indices):
            m = self.psf_metrics(idx)
            ple[k] = m['peak_localization_error_mm']
            sd[k] = m['spatial_dispersion_mm']
            depth[k] = m['depth_mm']
            if self.verbose and k % 25 == 0:
                print(f"  resolution map: {k+1}/{len(source_indices)}", flush=True)

        return {
            'source_idx': np.asarray(source_indices),
            'positions_mm': self.source_pos_mm[source_indices],
            'depth_mm': depth,
            'peak_localization_error_mm': ple,
            'spatial_dispersion_mm': sd,
        }

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
        return out
