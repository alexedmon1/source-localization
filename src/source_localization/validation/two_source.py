"""
Two sources: where are they, how far apart, and can we tell there are two at all?

Extends the calibrated single-source posterior (:mod:`.posterior`) to a
two-dipole model. Three questions, each with a probability rather than a
threshold verdict:

1. **Where are they?** The joint posterior over the pair ``p(x1, x2 | B)``,
   summarized as the symmetrized marginal ``p(a source is at x | B)``. Because
   the two sources are exchangeable, that marginal naturally has two lobes — the
   picture of two clouds joined by a low-probability bridge, where the bridge is
   genuine positional ambiguity rather than a rendering artifact.

2. **How far apart are they?** The posterior over separation
   ``p(|x1 - x2| | B)``, from which ``P(separation > d)`` follows directly. This
   replaces "did a saddle appear in the reconstruction" with a calibrated
   statement about distance.

3. **Are there even two?** The Bayes factor between the two-source and
   one-source models. Both models' marginal likelihoods are computed on the same
   absolute scale (moments integrated out, full log-determinants retained), so
   their ratio is meaningful. This is the principled replacement for the
   prominence-gated two-lobe detector, which needed a hand-tuned null.

Implementation note. The joint posterior lives on pairs of grid points, so cost
grows as the square of the grid: a 0.5 mm grid (6k points) would need 18M pairs.
The pair scan therefore runs on a coarser grid (1 mm by default). The residual
for a pair is computed without ever forming the 30x6 stacked leadfield, using
per-position quantities cached by :class:`~.posterior.DipolePosterior`::

    K = [U_i S_i , U_j S_j]        (30 x 6, implicit)
    K^T K  from  C_ij = U_i^T U_j  (3 x 3 per pair)
    K^T B  from  U_i^T B, U_j^T B  (3 per position)

so only 3x3 blocks per pair are ever materialized.

Classes
-------
TwoSourcePosterior
    Joint/marginal posteriors, separation posterior, and model comparison.
"""
from typing import Dict, Optional, Tuple

import numpy as np

from .posterior import DipolePosterior

__all__ = ['TwoSourcePosterior']


class TwoSourcePosterior:
    """
    Two-dipole posterior built on a :class:`~.posterior.DipolePosterior` grid.

    Parameters
    ----------
    single : DipolePosterior
        Supplies the grid, leadfield SVDs, and the one-source model used as the
        comparison baseline. Keep it coarse (~1 mm); the pair scan is quadratic.
    chunk : int, default=50000
        Pairs processed per batch, to bound memory.
    """

    def __init__(self, single: DipolePosterior, chunk: int = 50000):
        self.single = single
        self.chunk = int(chunk)
        n = single.n_positions
        self.pair_i, self.pair_j = np.triu_indices(n, k=1)
        self.n_pairs = len(self.pair_i)
        self.separation_mm = np.linalg.norm(
            single.positions_mm[self.pair_i] - single.positions_mm[self.pair_j],
            axis=1)

    # ------------------------------------------------------------ likelihood
    def pair_log_likelihood(
        self,
        data: np.ndarray,
        noise_std: float,
        moment_std: float = 1.0,
    ) -> np.ndarray:
        """
        Log marginal likelihood of a two-dipole model, per pair of positions.

        Moments are integrated out under a Gaussian prior, giving
        ``B | x1,x2 ~ N(0, sigma^2 I + tau^2 K K^T)``. The log-determinant is
        retained in full so this is on the same absolute scale as
        :meth:`one_source_log_likelihood`, which is what makes the Bayes factor
        valid.

        Returns
        -------
        ndarray, shape (n_pairs,)
        """
        b = np.asarray(data, float)
        if b.ndim == 1:
            b = b[:, None]
        n_times = b.shape[1]
        var = noise_std ** 2
        tau2 = moment_std ** 2
        n_chan = b.shape[0]
        total = float(np.sum(b ** 2))

        U, S = self.single._U, self.single._S           # (n_pos,nchan,3), (n_pos,3)
        # U_i^T B scaled by singular values -> K^T B blocks, per position.
        ub = np.einsum('pck,ct->pkt', U, b)             # (n_pos, 3, T)
        sub = S[:, :, None] * ub                        # (n_pos, 3, T)

        out = np.empty(self.n_pairs)
        eye6 = np.eye(6)
        for start in range(0, self.n_pairs, self.chunk):
            stop = min(start + self.chunk, self.n_pairs)
            i = self.pair_i[start:stop]
            j = self.pair_j[start:stop]
            m = stop - start

            cross = np.einsum('mck,mcl->mkl', U[i], U[j])       # (m,3,3)
            si, sj = S[i], S[j]

            ktk = np.empty((m, 6, 6))
            ktk[:, :3, :3] = np.einsum('mk,kl,ml->mkl', si,
                                       np.eye(3), si)           # diag(si^2)
            ktk[:, 3:, 3:] = np.einsum('mk,kl,ml->mkl', sj,
                                       np.eye(3), sj)
            block = si[:, :, None] * cross * sj[:, None, :]     # Si C Sj
            ktk[:, :3, 3:] = block
            ktk[:, 3:, :3] = np.transpose(block, (0, 2, 1))

            ktb = np.concatenate([sub[i], sub[j]], axis=1)      # (m,6,T)

            a = var * eye6[None] + tau2 * ktk
            # Ridge guards pairs whose leadfields are nearly collinear (very
            # close positions), where A is numerically singular.
            a[:, np.arange(6), np.arange(6)] += 1e-12 * var
            sol = np.linalg.solve(a, ktb)                       # (m,6,T)
            quad = (total - tau2 * np.einsum('mkt,mkt->m', ktb, sol)) / var

            sign, logdet6 = np.linalg.slogdet(a)
            log_det = logdet6 - 6.0 * np.log(var)
            out[start:stop] = -0.5 * quad - 0.5 * n_times * log_det

        # Constant shared with the one-source model (drops out of the posterior
        # but must be kept for the Bayes factor).
        out -= 0.5 * n_times * n_chan * np.log(2.0 * np.pi * var)
        return out

    def one_source_log_likelihood(
        self,
        data: np.ndarray,
        noise_std: float,
        moment_std: float = 1.0,
    ) -> np.ndarray:
        """Same quantity for the one-dipole model, per grid position."""
        b = np.asarray(data, float)
        if b.ndim == 1:
            b = b[:, None]
        n_times = b.shape[1]
        var = noise_std ** 2
        tau2 = moment_std ** 2
        n_chan = b.shape[0]
        total = float(np.sum(b ** 2))

        U, S = self.single._U, self.single._S
        proj = np.einsum('pck,ct->pkt', U, b)
        energy = np.sum(proj ** 2, axis=2)
        s2 = S ** 2
        denom = var + tau2 * s2
        quad = (total - np.sum(tau2 * s2 / denom * energy, axis=1)) / var
        log_det = np.sum(np.log(denom), axis=1) - 3.0 * np.log(var)
        return (-0.5 * quad - 0.5 * n_times * log_det
                - 0.5 * n_times * n_chan * np.log(2.0 * np.pi * var))

    # ------------------------------------------------------------- posteriors
    def joint_posterior(self, data, noise_std, moment_std: float = 1.0):
        """Normalized posterior over pairs; uniform prior over unordered pairs."""
        ll = self.pair_log_likelihood(data, noise_std, moment_std)
        ll = ll - ll.max()
        p = np.exp(ll)
        s = p.sum()
        return p / s if s > 0 else p

    def marginal_density(self, joint: np.ndarray) -> np.ndarray:
        """
        ``p(a source is at x | B)`` — the symmetrized marginal, per position.

        Because the two sources are exchangeable there is no "source 1" cloud
        and "source 2" cloud to report separately; the honest object is the
        density of *a* source, which shows both lobes at once. Sums to 2 (there
        are two sources), so divide by 2 for a probability per source.
        """
        m = np.zeros(self.single.n_positions)
        np.add.at(m, self.pair_i, joint)
        np.add.at(m, self.pair_j, joint)
        return m

    def separation_posterior(
        self,
        joint: np.ndarray,
        edges: Optional[np.ndarray] = None,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Posterior over the distance between the two sources.

        Returns
        -------
        (centers_mm, mass) : the binned posterior over ``|x1 - x2|``.
        """
        if edges is None:
            edges = np.arange(0.0, self.separation_mm.max() + 0.5, 0.5)
        mass, _ = np.histogram(self.separation_mm, bins=edges, weights=joint)
        return 0.5 * (edges[:-1] + edges[1:]), mass

    def p_separated_beyond(self, joint: np.ndarray, d_mm: float) -> float:
        """``P(|x1 - x2| > d | B)`` — the calibrated 'are they distinct' number."""
        return float(joint[self.separation_mm > d_mm].sum())

    def analyze(
        self,
        data: np.ndarray,
        noise_std: float,
        moment_std: float = 1.0,
    ) -> Dict:
        """
        Everything from a single pair scan: posterior, separation, Bayes factor.

        The pair scan dominates runtime, so callers running many trials should
        use this rather than calling :meth:`joint_posterior` and
        :meth:`log_bayes_factor` separately, which would scan twice.
        """
        from scipy.special import logsumexp
        ll2 = self.pair_log_likelihood(data, noise_std, moment_std)
        ll1 = self.one_source_log_likelihood(data, noise_std, moment_std)

        shifted = ll2 - ll2.max()
        joint = np.exp(shifted)
        joint /= joint.sum()

        centers, mass = self.separation_posterior(joint)
        cdf = np.cumsum(mass)
        median_sep = (float(centers[np.searchsorted(cdf, 0.5)])
                      if cdf[-1] > 0 else np.nan)
        log10_bf = float((logsumexp(ll2) - np.log(len(ll2))
                          - (logsumexp(ll1) - np.log(len(ll1)))) / np.log(10.0))
        return {'joint': joint, 'median_separation_mm': median_sep,
                'log10_bayes_factor': log10_bf,
                'separation_centers_mm': centers, 'separation_mass': mass}

    def log_bayes_factor(
        self,
        data: np.ndarray,
        noise_std: float,
        moment_std: float = 1.0,
    ) -> float:
        """
        log10 Bayes factor for two sources versus one.

        Each model's evidence is its likelihood averaged over its own uniform
        prior (positions, or unordered pairs), so the comparison already carries
        an Occam penalty for the larger model's bigger parameter space. Positive
        favours two sources; by the usual reading, >0.5 is substantial and >1
        (a factor of 10) is strong.
        """
        from scipy.special import logsumexp
        ll2 = self.pair_log_likelihood(data, noise_std, moment_std)
        ll1 = self.one_source_log_likelihood(data, noise_std, moment_std)
        ev2 = logsumexp(ll2) - np.log(len(ll2))
        ev1 = logsumexp(ll1) - np.log(len(ll1))
        return float((ev2 - ev1) / np.log(10.0))

    # ------------------------------------------------------------- simulation
    def simulate_pair(
        self,
        pos1_mm: np.ndarray,
        pos2_mm: np.ndarray,
        orient1: np.ndarray,
        orient2: np.ndarray,
        snr_db: float,
        rng: np.random.Generator,
        amplitude1: float = 1.0,
        amplitude2: float = 1.0,
    ) -> Tuple[np.ndarray, float]:
        """Sensor data from two dipoles with white noise at a target SNR."""
        g1 = self.single.leadfield_at(pos1_mm)
        g2 = self.single.leadfield_at(pos2_mm)
        o1 = np.asarray(orient1, float)
        o2 = np.asarray(orient2, float)
        signal = ((g1 @ (o1 / np.linalg.norm(o1))) * amplitude1
                  + (g2 @ (o2 / np.linalg.norm(o2))) * amplitude2)[:, None]
        noise_power = float(np.mean(signal ** 2)) / (10.0 ** (snr_db / 10.0))
        noise_std = float(np.sqrt(noise_power))
        return signal + rng.normal(0.0, noise_std, size=signal.shape), noise_std

    # ------------------------------------------------------------ experiment
    def separation_sweep(
        self,
        separations_mm=(1.0, 2.0, 3.0, 4.0, 6.0, 8.0),
        snr_db: float = 10.0,
        n_trials: int = 20,
        moment_std: float = 1.0,
        seed: int = 0,
        depth_band: Optional[Tuple[float, float]] = None,
        tol_mm: float = 0.6,
        verbose: bool = False,
    ) -> Dict:
        """
        How well two sources separate, as a function of their true separation.

        For each nominal separation, draws pairs at that distance, simulates, and
        records the posterior's own verdicts: the median ``P(separation > half
        the true separation)``, the median posterior separation, and the median
        log10 Bayes factor for two sources over one.

        ``depth_band`` optionally restricts both sources to a band of the
        dorsal-ventral axis (``z`` in mm), so depth can be held roughly fixed —
        it matters because gain falls steeply with depth.

        Returns
        -------
        dict
            Per-separation lists of the summary statistics above.
        """
        rng = np.random.default_rng(seed)
        pos = self.single.positions_mm
        ok = np.ones(len(pos), bool)
        if depth_band is not None:
            ok = (pos[:, 2] >= depth_band[0]) & (pos[:, 2] <= depth_band[1])
        ok_idx = np.where(ok)[0]

        results = {'separation_mm': [], 'p_resolved': [], 'median_post_sep_mm': [],
                   'log10_bayes_factor': [], 'n_used': []}

        for sep in separations_mm:
            pr, ps, bf = [], [], []
            for _ in range(n_trials):
                a = int(rng.choice(ok_idx))
                d = np.linalg.norm(pos[ok_idx] - pos[a], axis=1)
                cand = ok_idx[np.abs(d - sep) <= tol_mm]
                cand = cand[cand != a]
                if not len(cand):
                    continue
                b = int(rng.choice(cand))
                true_sep = float(np.linalg.norm(pos[a] - pos[b]))

                data, sigma = self.simulate_pair(
                    pos[a], pos[b], rng.normal(size=3), rng.normal(size=3),
                    snr_db, rng)
                joint = self.joint_posterior(data, sigma, moment_std)

                pr.append(self.p_separated_beyond(joint, 0.5 * true_sep))
                centers, mass = self.separation_posterior(joint)
                cdf = np.cumsum(mass)
                ps.append(float(centers[np.searchsorted(cdf, 0.5)])
                          if cdf[-1] > 0 else np.nan)
                bf.append(self.log_bayes_factor(data, sigma, moment_std))

            if not pr:
                continue
            results['separation_mm'].append(float(sep))
            results['p_resolved'].append(float(np.median(pr)))
            results['median_post_sep_mm'].append(float(np.median(ps)))
            results['log10_bayes_factor'].append(float(np.median(bf)))
            results['n_used'].append(len(pr))
            if verbose:
                print(f"    sep {sep:4.1f} mm: P(>half)={np.median(pr):.2f}  "
                      f"log10BF={np.median(bf):+.2f}", flush=True)
        return results
