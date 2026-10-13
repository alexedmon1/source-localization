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

``SURFACE = MapCalibration(2.5, 0.0)``
    For the cortical/cerebellar/olfactory surface readout, the best single setting
    on the grid. It is approximately calibrated: the 90% region holds the truth
    88-91% of the time at half and primary SNR, but only 83-85% at double SNR.
    Blurring made it worse.

**What calibration cannot fix.** Coverage is calibrated for single-source truths.
A response with more than one source is still read as one confident point between
the sources. For mirrored bilateral auditory cortex, the calibrated 90% region
holds either source in only 29-57% of maps, and both in at most 5%. Read real
auditory maps with that in mind.

**More than one source.** :class:`SourceSetPosterior` is the same likelihood for a
configuration of several dipoles (a stacked leadfield block per configuration).
:class:`EnsemblePosterior` is its one-dipole case. :class:`MirrorPairPosterior`
places dipoles at a position and its mirror image across the midline (a bilateral
response). :meth:`SourceSetPosterior.log_evidence` averages the likelihood over a
set's configurations, so :func:`model_probabilities` can compare "one source"
with "a mirrored pair" on the same data.

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
mirror_pairs
    (right, left) index pairs of mirror-image positions.
model_probabilities
    Posterior probabilities of source sets from their log evidences.
"""
from dataclasses import dataclass
from functools import lru_cache
from typing import Dict, Optional, Sequence

import numpy as np
from scipy import sparse
from scipy.spatial import cKDTree
from scipy.special import logsumexp

__all__ = ['SourceSetPosterior', 'EnsemblePosterior', 'MirrorPairPosterior', 'MapCalibration', 'UNCALIBRATED',
           'VOLUMETRIC', 'SURFACE', 'blur_kernel', 'mass_above', 'coverage', 'mirror_pairs', 'model_probabilities']


class SourceSetPosterior:
    """
    Posterior over configurations of a fixed set of dipoles, averaged over head models.

    A configuration q places d / 3 free-orientation dipoles; its whitened leadfield
    block G_q (n_channels x d) stacks theirs. Per column j,
    ``y_j | q, k ~ N(0, s I + tau^2 G_qk G_qk^T)`` (each dipole's moment ~ N(0, tau^2 I3),
    independent), averaged over the K head models. The moment scale is the
    empirical-Bayes estimate per pattern: the pattern's power above the noise
    floor divided by the median of ||G_q||_F^2 under the nominal model. With
    several sources, tau^2 is therefore a per-source variance.

    Up to a constant common to every configuration, model and set size, the log
    likelihood is ``-1/2 y' C^-1 y - 1/2 log det C`` summed over columns. Its
    average over configurations, :meth:`log_evidence`, is comparable across source
    sets (a uniform prior over each set's configurations).

    Parameters
    ----------
    blocks : ndarray, shape (K, n_configs, n_channels, d)
        Whitened leadfield blocks of each configuration under each head model; the
        first model is the nominal one.
    """

    def __init__(self, blocks: np.ndarray):
        blocks = np.asarray(blocks, float)
        if blocks.ndim == 3:
            blocks = blocks[None]
        self.n_models, self.n_configs, self.n_channels, self.d = blocks.shape
        self._svd = []
        for G in blocks:
            U, S, _ = np.linalg.svd(G, full_matrices=False)                          # (Q, m, d), (Q, d)
            self._svd.append((U, S))
        self.gain2 = float(np.median(np.sum(self._svd[0][1] ** 2, axis=1)))

    @staticmethod
    def _as_patterns(data):
        Y = np.asarray(data, float)
        single = Y.ndim < 3
        if Y.ndim == 1:
            Y = Y[:, None]
        if Y.ndim == 2:
            Y = Y[None]
        return Y, single

    def moment_variance(self, Y: np.ndarray, noise_var: float = 1.0) -> np.ndarray:
        """Per-pattern empirical-Bayes tau^2 for patterns (n, m, c)."""
        n, m, c = Y.shape
        return np.maximum(np.sum(Y ** 2, axis=(1, 2)) / c - noise_var * m, 1e-3 * m) / self.gain2

    def log_likelihood(self, data: np.ndarray, noise_var: float = 1.0, moment_var: Optional[float] = None,
                       chunk: int = 64) -> np.ndarray:
        """
        Unnormalized log p(y | configuration), averaged over head models.

        Returns (n_patterns, n_configs), or (n_configs,) for one pattern.
        """
        Y, single = self._as_patterns(data)
        n_all, m, c = Y.shape
        d = self.d
        out = np.empty((n_all, self.n_configs))
        for s in range(0, n_all, chunk):
            Yb = Y[s:s + chunk]
            n = len(Yb)
            total = np.sum(Yb ** 2, axis=(1, 2))                                   # (n,)
            tau2 = self.moment_variance(Yb, noise_var) if moment_var is None else np.full(n, float(moment_var))
            Y2 = Yb.transpose(1, 2, 0).reshape(m, c * n)
            lps = []
            for U, S in self._svd:
                Q = U.shape[0]
                proj = (U.transpose(0, 2, 1).reshape(Q * d, m) @ Y2).reshape(Q, d, c, n)
                pe = np.sum(proj ** 2, axis=2)                                      # (Q, d, n)
                s2 = S ** 2
                denom = noise_var + tau2[None, None, :] * s2[:, :, None]
                quad = (total[None, :] - np.sum(tau2[None, None, :] * s2[:, :, None] / denom * pe, axis=1)) / noise_var
                log_det = np.sum(np.log(denom), axis=1) - d * np.log(noise_var)
                lps.append(-0.5 * quad - 0.5 * c * log_det)                         # (Q, n)
            out[s:s + n] = (logsumexp(np.array(lps), axis=0) - np.log(len(lps))).T
        return out[0] if single else out

    def log_evidence(self, data: np.ndarray, noise_var: float = 1.0, moment_var: Optional[float] = None
                     ) -> np.ndarray:
        """log p(y | this source set): the likelihood averaged over configurations (uniform prior)."""
        ll = np.atleast_2d(self.log_likelihood(data, noise_var, moment_var))
        ev = logsumexp(ll, axis=1) - np.log(self.n_configs)
        return ev[0] if np.ndim(data) < 3 else ev

    def log_posterior(self, data: np.ndarray, noise_var: float = 1.0, moment_var: Optional[float] = None,
                      chunk: int = 64) -> np.ndarray:
        """
        Normalized log posterior over configurations.

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
        ndarray, shape (n_patterns, n_configs), or (n_configs,) for one pattern
        """
        ll = self.log_likelihood(data, noise_var, moment_var, chunk)
        return ll - logsumexp(ll, axis=-1, keepdims=True)


class EnsemblePosterior(SourceSetPosterior):
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
        super().__init__(leadfields)
        self.positions_mm = np.asarray(positions_mm, float)
        self.spacing_mm = float(spacing_mm)
        self.n_positions = self.n_configs

    def posterior(self, data: np.ndarray, calibration: Optional['MapCalibration'] = None,
                  noise_var: float = 1.0, moment_var: Optional[float] = None) -> np.ndarray:
        """Posterior probability per position (sums to 1 per pattern); calibrated if ``calibration`` is given."""
        lp = self.log_posterior(data, noise_var, moment_var)
        cal = calibration or UNCALIBRATED
        return cal.apply(lp, self.positions_mm)


class MirrorPairPosterior(SourceSetPosterior):
    """
    Two dipoles at mirror-image positions across the midline (x -> 2 x0 - x).

    Auditory and other sensory responses are generated bilaterally. A single-dipole
    readout fitted to two mirrored sources puts one confident source between them
    (``EnsemblePosterior``'s known failure). This model places one free-orientation
    dipole at each of a position x on the right (x > x0 + ``min_offset_mm``) and its
    mirror image, snapped to the nearest grid position. The moments are independent,
    so coherent and incoherent pairs are both covered.

    ``configs`` holds (right index, left index) into ``positions_mm``.
    ``config_positions_mm`` is the right member, used to blur or map the posterior.
    """

    def __init__(self, leadfields: np.ndarray, positions_mm: np.ndarray, spacing_mm: float,
                 midline_x_mm: float = 0.0, min_offset_mm: float = 0.3):
        leadfields = np.asarray(leadfields, float)
        if leadfields.ndim == 3:
            leadfields = leadfields[None]
        self.positions_mm = np.asarray(positions_mm, float)
        self.spacing_mm = float(spacing_mm)
        self.midline_x_mm = float(midline_x_mm)
        self.configs = mirror_pairs(self.positions_mm, midline_x_mm, min_offset_mm)
        i, j = self.configs.T
        super().__init__(np.concatenate([leadfields[:, i], leadfields[:, j]], axis=-1))
        self.config_positions_mm = self.positions_mm[i]

    def posterior(self, data: np.ndarray, calibration: Optional['MapCalibration'] = None,
                  noise_var: float = 1.0, moment_var: Optional[float] = None) -> np.ndarray:
        """Posterior per configuration (sums to 1 per pattern); calibrated if ``calibration`` is given."""
        lp = self.log_posterior(data, noise_var, moment_var)
        cal = calibration or UNCALIBRATED
        return cal.apply(lp, self.config_positions_mm)

    def position_map(self, posterior: np.ndarray) -> np.ndarray:
        """Spread a configuration posterior onto positions: each pair puts its mass on both members (sums to 2)."""
        p = np.atleast_2d(posterior)
        out = np.zeros((len(p), len(self.positions_mm)))
        for col in (0, 1):
            np.add.at(out.T, self.configs[:, col], p.T)
        return out[0] if np.ndim(posterior) == 1 else out


def mirror_pairs(positions_mm: np.ndarray, midline_x_mm: float = 0.0, min_offset_mm: float = 0.3) -> np.ndarray:
    """(n_pairs, 2) indices (right, left): each position with x > x0 + offset and its nearest mirror image."""
    pos = np.asarray(positions_mm, float)
    right = np.flatnonzero(pos[:, 0] > midline_x_mm + min_offset_mm)
    q = pos[right].copy()
    q[:, 0] = 2 * midline_x_mm - q[:, 0]
    _, left = cKDTree(pos).query(q)
    return np.stack([right, left], axis=1)


def model_probabilities(log_evidences: Dict[str, np.ndarray], prior: Optional[Dict[str, float]] = None
                        ) -> Dict[str, np.ndarray]:
    """Posterior probability of each source set from their log evidences (equal prior odds by default)."""
    names = list(log_evidences)
    E = np.stack([np.asarray(log_evidences[k], float) for k in names])
    if prior is not None:
        E = E + np.log([prior[k] for k in names]).reshape(-1, *[1] * (E.ndim - 1))
    P = np.exp(E - logsumexp(E, axis=0, keepdims=True))
    return {k: P[i] for i, k in enumerate(names)}


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
SURFACE = MapCalibration(temperature=2.5, blur_mm=0.0)


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
