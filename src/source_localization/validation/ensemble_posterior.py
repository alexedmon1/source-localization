"""
Where is the source, when the head model itself is uncertain? A head-model-ensemble
location posterior, and the calibration that makes its credible regions honest.

:class:`~source_localization.validation.posterior.DipolePosterior` assumes the
forward model is exact. On real data it is not: electrode registration and skull
conductivity are uncertain. :class:`EnsemblePosterior` averages the single-dipole
marginal likelihood over K head models (the nominal model plus draws from
:class:`~source_localization.validation.head_models.HeadModelPrior`)::

    p(y | x) = (1/K) sum_k p(y | x, model k)
    p(x | y) proportional to p(y | x)            (uniform prior over positions)

Each term is ``DipolePosterior``'s marginal likelihood with the moment
marginalized, ``y_j | x, k ~ N(0, s I + tau^2 G_xk G_xk^T)`` independently over
columns j, on **whitened** leadfields and data (whitening is the caller's job, so
s = 1). Averaging likelihoods rather than normalized posteriors is what
marginalizes over the head model. The moment scale is an empirical-Bayes estimate
per pattern, the same as the reference implementation (probability-atlas
``pa.posterior``)::

    tau^2 = max(||y||^2 / c - s m, 1e-3 m) / median_x ||G_x,nominal||_F^2

**Calibration** (:class:`MapCalibration`). The averaged map is still
overconfident. bayes-sl measured this on 2026-10-13 (``reports/TB_map_focality.md``,
``reports/TB_map_calibration.md``). The setup was MEA30, 33 models, a truth from a
fresh head model, and real held-out noise. Under it the 90% region held the truth
only about 55-65% of the time. At high SNR the 33-model average becomes 33 sharp
peaks, and the truth's own head model lies between them. Two corrections are
applied to the normalized log posterior:

1. temper by T: ``p proportional to exp(log p / T)``, which widens the core; and
2. blur with an isotropic Gaussian of ``blur_mm`` over the candidate positions
   (mass-conserving). This is a resolution floor that does not shrink with SNR.

The validated settings were chosen on one simulated set and checked on another
(fresh truths, fresh head models, the other half of the mice for noise). They
apply only to the MEA30 geometry, the 33-model ensemble and the 0.5 mm grid:

``VOLUMETRIC = MapCalibration(1.5, 0.25)``
    The 90% region holds the truth 88-89% of the time at half, primary and double
    SNR, for both cortical and deep truths (80%: 79-81%; 95%: 92-94%). The median
    90% region is 3-8 mm3 at primary SNR.

**What calibration cannot fix.** Coverage is calibrated for single-source truths.
A response with more than one source is still read as one confident point between
the sources. For mirrored bilateral auditory cortex, the calibrated 90% region
holds either source in only 29-57% of maps, and both in at most 5%. Read real
auditory maps with that in mind.

Functions
---------
blur_kernel
    Row-stochastic Gaussian kernel over a set of positions (sparse).
mass_above
    Posterior mass strictly more probable than a given position. The position lies
    in the alpha credible region iff this is below alpha, so one number per map
    scores coverage at every level.
coverage
    Empirical coverage per level from ``mass_above`` values.
"""
from dataclasses import dataclass
from functools import lru_cache
from typing import Dict, Optional, Sequence

import numpy as np
from scipy import sparse
from scipy.spatial import cKDTree
from scipy.special import logsumexp

from .posterior import DipolePosterior

__all__ = ['EnsemblePosterior', 'MapCalibration', 'UNCALIBRATED', 'VOLUMETRIC',
           'blur_kernel', 'mass_above', 'coverage']


class EnsemblePosterior:
    """
    Single-dipole location posterior averaged over an ensemble of head models.

    Parameters
    ----------
    leadfields : ndarray, shape (K, n_positions, n_channels, 3)
        Whitened free-orientation leadfields of the K head models on the same
        positions. The first is the nominal model; it sets the moment prior's gain
        scale.
    positions_mm : ndarray, shape (n_positions, 3)
    spacing_mm : float
    """

    def __init__(self, leadfields: np.ndarray, positions_mm: np.ndarray, spacing_mm: float):
        leadfields = np.asarray(leadfields, float)
        if leadfields.ndim == 3:
            leadfields = leadfields[None]
        K, P, m, _ = leadfields.shape
        self.positions_mm = np.asarray(positions_mm, float)
        self.spacing_mm = float(spacing_mm)
        self.n_positions = P
        self.models = [DipolePosterior(G.transpose(1, 0, 2).reshape(m, 3 * P), self.positions_mm, spacing_mm)
                       for G in leadfields]
        self.gain2 = float(np.median(np.sum(self.models[0]._S ** 2, axis=1)))

    def log_posterior(self, data: np.ndarray, noise_var: float = 1.0, moment_var: Optional[float] = None,
                      chunk: int = 64) -> np.ndarray:
        """
        Normalized log posterior over positions.

        Parameters
        ----------
        data : ndarray, shape (n_channels,), (n_channels, n_columns) or
            (n_patterns, n_channels, n_columns)
            Whitened pattern(s).
        noise_var : float
            Noise variance of the whitened data (1 if the whitener is right).
        moment_var : float, optional
            Fixed moment prior variance tau^2. Default: the per-pattern empirical-Bayes estimate.

        Returns
        -------
        ndarray, shape (n_patterns, n_positions), or (n_positions,) for one pattern
        """
        Y = np.asarray(data, float)
        single = Y.ndim < 3
        if Y.ndim == 1:
            Y = Y[:, None]
        if Y.ndim == 2:
            Y = Y[None]
        n_all, m, c = Y.shape
        out = np.empty((n_all, self.n_positions))
        for s in range(0, n_all, chunk):
            Yb = Y[s:s + chunk]
            n = len(Yb)
            total = np.sum(Yb ** 2, axis=(1, 2))                                   # (n,)
            tau2 = (np.maximum(total / c - noise_var * m, 1e-3 * m) / self.gain2 if moment_var is None
                    else np.full(n, float(moment_var)))                              # (n,)
            Y2 = Yb.transpose(1, 2, 0).reshape(m, c * n)
            lps = []
            for dp in self.models:
                P = dp._U.shape[0]
                proj = (dp._U.transpose(0, 2, 1).reshape(P * 3, m) @ Y2).reshape(P, 3, c, n)
                pe = np.sum(proj ** 2, axis=2)                                      # (P, 3, n)
                s2 = dp._S ** 2
                denom = noise_var + tau2[None, None, :] * s2[:, :, None]
                quad = (total[None, :] - np.sum(tau2[None, None, :] * s2[:, :, None] / denom * pe, axis=1)) / noise_var
                log_det = np.sum(np.log(denom), axis=1) - 3.0 * np.log(noise_var)
                lps.append(-0.5 * quad - 0.5 * c * log_det)                         # (P, n)
            lp = logsumexp(np.array(lps), axis=0)
            out[s:s + n] = (lp - logsumexp(lp, axis=0, keepdims=True)).T
        return out[0] if single else out

    def posterior(self, data: np.ndarray, calibration: Optional['MapCalibration'] = None,
                  noise_var: float = 1.0, moment_var: Optional[float] = None) -> np.ndarray:
        """Posterior probability per position (sums to 1 per pattern); calibrated if ``calibration`` is given."""
        lp = self.log_posterior(data, noise_var, moment_var)
        cal = calibration or UNCALIBRATED
        return cal.apply(lp, self.positions_mm)


@dataclass(frozen=True)
class MapCalibration:
    """Temper the normalized log posterior by ``temperature``, then blur by a Gaussian of ``blur_mm``."""
    temperature: float = 1.0
    blur_mm: float = 0.0

    def apply(self, log_posterior: np.ndarray, positions_mm: np.ndarray) -> np.ndarray:
        lp = np.atleast_2d(np.asarray(log_posterior, float)) / self.temperature
        p = np.exp(lp - logsumexp(lp, axis=1, keepdims=True))
        if self.blur_mm > 0:
            K = blur_kernel(positions_mm, self.blur_mm)
            p = np.asarray((K.T @ p.T).T)
        return p[0] if np.ndim(log_posterior) == 1 else p


UNCALIBRATED = MapCalibration()
VOLUMETRIC = MapCalibration(temperature=1.5, blur_mm=0.25)


def blur_kernel(positions_mm: np.ndarray, sigma_mm: float) -> sparse.csr_matrix:
    """
    Row-stochastic Gaussian kernel ``K[y, x]`` over the positions, truncated at 3 sigma.

    ``p @ K`` moves the mass at each position y onto its neighbours x, so a
    blurred map still sums to 1.
    """
    pos = np.ascontiguousarray(positions_mm, float)
    return _blur_kernel(pos.tobytes(), pos.shape, float(sigma_mm))


@lru_cache(maxsize=8)
def _blur_kernel(pos_bytes: bytes, shape, sigma_mm: float) -> sparse.csr_matrix:
    pos = np.frombuffer(pos_bytes).reshape(shape)
    tree = cKDTree(pos)
    D = tree.sparse_distance_matrix(tree, 3 * sigma_mm, output_type='coo_matrix')   # keeps d = 0
    K = sparse.csr_matrix((np.exp(-D.data ** 2 / (2 * sigma_mm ** 2)), (D.row, D.col)), shape=D.shape)
    return sparse.diags(1.0 / np.asarray(K.sum(axis=1)).ravel()) @ K


def mass_above(posterior: np.ndarray, index) -> np.ndarray:
    """
    Mass of positions strictly more probable than position ``index``, per map.

    Position ``index`` lies in the alpha highest-density region iff
    ``mass_above < alpha``.
    """
    p = np.atleast_2d(posterior)
    idx = np.broadcast_to(np.asarray(index), (len(p),))
    pt = p[np.arange(len(p)), idx]
    out = np.sum(np.where(p > pt[:, None], p, 0.0), axis=1)
    return out[0] if np.ndim(posterior) == 1 else out


def coverage(mass_above_values: Sequence[float], levels: Sequence[float] = (0.5, 0.8, 0.9, 0.95)) -> Dict[float, float]:
    """Empirical coverage per nominal level: the share of maps whose truth lies in that region."""
    m = np.asarray(mass_above_values, float)
    return {float(a): float(np.mean(m < a)) for a in levels}
