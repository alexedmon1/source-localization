"""
Where is the source *really*? A calibrated spatial probability density.

Every other metric in this package is forward-looking: given a source at x, what
does the reconstruction look like (the PSF)? This module answers the inverse and
more useful question — given the measured data, what is the probability density
over the source's true location? The result is an "orbital": a normalized cloud
whose 50% region contains the true source half the time, by construction and by
test.

Method (standard dipole scanning / likelihood mapping). For a candidate location
x the leadfield ``G_x`` is 30x3 (three free orientations). Profiling out the
unknown dipole moment leaves a residual that depends only on position::

    r(x) = || B - Q_x Q_x^T B ||_F^2

where ``Q_x`` is an orthonormal basis for the column space of ``G_x``. Under
Gaussian sensor noise of standard deviation sigma, ``log p(B | x) = -r(x) /
(2 sigma^2)`` up to a constant, so with a uniform prior over the volume::

    p(x | B) proportional to exp( -r(x) / (2 sigma^2) )

This is a genuine probability density — it normalizes to 1 over the volume, and
its credible regions can be checked against simulated ground truth
(:meth:`DipolePosterior.coverage_test`). That check is the point: a threshold on
a reconstruction image has no such guarantee, and the "25%" in such a threshold
does not mean 25% of anything.

Not tied to the pipeline's 215-point lattice: the posterior is evaluated on its
own dense grid (default 0.5 mm), rebuilt from the same cached BEM, and truth can
be placed at arbitrary continuous positions.

Two honest caveats:

1. This is the **fundamental limit** for a single-dipole model — it assumes the
   forward model is exact and the noise is white with known sigma. It therefore
   describes how well *any* method could localize, not how well the pipeline's
   sLORETA estimator does. The gap between the two is the estimator's cost.
2. A uniform prior over the brain volume is assumed. Because gain falls steeply
   with depth, the posterior for deep sources is broad; that is a real statement
   about the data, not a prior artifact.

Classes
-------
DipolePosterior
    Dense leadfield, likelihood map, credible regions, and calibration test.
"""
import json
from pathlib import Path
from typing import Dict, Optional, Tuple

import numpy as np

__all__ = ['DipolePosterior']


class DipolePosterior:
    """
    Posterior density over a single dipole's location, on a dense grid.

    Parameters
    ----------
    leadfield : ndarray, shape (n_channels, 3 * n_positions)
        Free-orientation leadfield, column-blocked by position.
    positions_mm : ndarray, shape (n_positions, 3)
    spacing_mm : float
        Grid spacing, used for volume elements.
    """

    def __init__(self, leadfield: np.ndarray, positions_mm: np.ndarray,
                 spacing_mm: float):
        self.positions_mm = np.asarray(positions_mm, float)
        self.n_positions = self.positions_mm.shape[0]
        self.spacing_mm = float(spacing_mm)
        self.leadfield = np.asarray(leadfield, float)

        # (n_pos, n_chan, 3), then an SVD per position: the left singular
        # vectors give the projection for the profile likelihood, and the
        # singular values carry the position's gain, which the marginal
        # likelihood needs for its Occam factor.
        g = self.leadfield.reshape(self.leadfield.shape[0], self.n_positions, 3)
        self._G = np.transpose(g, (1, 0, 2))
        self._U, self._S, _ = np.linalg.svd(self._G, full_matrices=False)
        self._Q = self._U

    # ------------------------------------------------------------------ build
    @classmethod
    def from_pipeline_dir(
        cls,
        pipeline_dir,
        spacing_mm: float = 0.5,
        inside_frac: float = 0.93,
        verbose: bool = True,
        cache_dir=None,
    ) -> 'DipolePosterior':
        """
        Rebuild a dense forward model from a pipeline run's cached BEM.

        The pipeline's own source space is a 215-point shell grid, far too coarse
        to represent a density. This recomputes the leadfield on a regular grid
        at ``spacing_mm`` inside the innermost BEM surface, using the same BEM
        and electrode montage, so the physics is identical and only the sampling
        changes.

        Parameters
        ----------
        pipeline_dir : path-like
            A completed pipeline run (needs ``bem_cache/`` and ``data/``).
        spacing_mm : float, default=0.5
        inside_frac : float, default=0.93
            Keep grid points inside this fraction of the inner ellipsoid, so no
            source sits on the boundary where the BEM is ill-conditioned.
        """
        import pickle
        import mne
        mne.set_log_level('ERROR')

        signature = cls._cache_signature(pipeline_dir, spacing_mm, inside_frac)
        cache_path = None
        if cache_dir is not None:
            cache_path = (Path(cache_dir)
                          / f'operator_{spacing_mm:g}mm.npz')
            if cache_path.exists():
                obj = cls.load(cache_path)
                if obj.provenance.get('signature') == signature:
                    if verbose:
                        print(f"  operator loaded from cache "
                              f"({obj.n_positions} positions, {cache_path.name})")
                    return obj
                if verbose:
                    print(f"  cache {cache_path.name} is stale "
                          f"(inputs changed) — rebuilding")

        p = Path(pipeline_dir)
        bem = pickle.load(open(p / 'bem_cache' / 'ellipsoid_3layer.pkl', 'rb'))
        src = pickle.load(open(p / 'data' / 'step3_source_space.pkl', 'rb'))
        info = pickle.load(open(p / 'data' / 'step1_info.pkl', 'rb'))

        positions_m = cls._grid_inside(bem, spacing_mm, inside_frac)
        if verbose:
            print(f"  dense grid: {len(positions_m)} positions at "
                  f"{spacing_mm} mm spacing")

        dense_src = cls._as_source_space(src, positions_m)
        solution = mne.make_bem_solution(bem['surfs'], verbose=False)
        fwd = mne.make_forward_solution(info, trans=None, src=dense_src,
                                        bem=solution, eeg=True, meg=False,
                                        verbose=False)
        fwd = mne.convert_forward_solution(fwd, force_fixed=False,
                                           verbose=False)

        # make_forward_solution may drop positions it deems invalid; trust the
        # source space it returns rather than the grid we asked for.
        used = fwd['src'][0]['rr'][fwd['src'][0]['inuse'].astype(bool)]
        obj = cls(fwd['sol']['data'], used * 1000.0, spacing_mm)

        # Keep the geometry the leadfield was computed from, so figures can be
        # drawn against the actual head model rather than in a bare voxel space.
        obj.bem_surfaces_mm = [s['rr'] * 1000.0 for s in bem['surfs']]
        obj.bem_conductivities = list(bem.get('sigma', []))
        obj.electrode_pos_mm = np.array(
            [ch['loc'][:3] for ch in info['chs']], dtype=float) * 1000.0
        obj.channel_names = list(info['ch_names'])
        obj.provenance = {'signature': signature}
        obj._check_geometry(verbose=verbose)
        if cache_path is not None:
            obj.save(cache_path)
            if verbose:
                print(f"  operator cached -> {cache_path}")
        return obj

    # ------------------------------------------------------------- persistence
    def save(self, path) -> Path:
        """
        Write the operator and its geometry to a single ``.npz``.

        Rebuilding costs a BEM solution plus a forward solve (seconds to tens of
        seconds, and it grows with grid resolution), which is pure waste when
        the only thing being changed is a figure. Everything a plot or a further
        analysis needs is stored, so nothing has to be recomputed to redraw.
        """
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            'leadfield': self.leadfield,
            'positions_mm': self.positions_mm,
            'spacing_mm': np.array(self.spacing_mm),
            'provenance': np.array(json.dumps(self.provenance)),
        }
        if self.electrode_pos_mm is not None:
            payload['electrode_pos_mm'] = self.electrode_pos_mm
            payload['channel_names'] = np.array(list(self.channel_names))
        if self.bem_surfaces_mm is not None:
            payload['n_bem_surfaces'] = np.array(len(self.bem_surfaces_mm))
            payload['bem_conductivities'] = np.array(self.bem_conductivities)
            for i, s in enumerate(self.bem_surfaces_mm):
                payload[f'bem_surface_{i}'] = s
        np.savez_compressed(path, **payload)
        return path

    @classmethod
    def load(cls, path) -> 'DipolePosterior':
        """Reload an operator written by :meth:`save`."""
        z = np.load(Path(path), allow_pickle=False)
        obj = cls(z['leadfield'], z['positions_mm'], float(z['spacing_mm']))
        obj.provenance = json.loads(str(z['provenance']))
        if 'electrode_pos_mm' in z:
            obj.electrode_pos_mm = z['electrode_pos_mm']
            obj.channel_names = [str(c) for c in z['channel_names']]
        if 'n_bem_surfaces' in z:
            obj.bem_surfaces_mm = [z[f'bem_surface_{i}']
                                   for i in range(int(z['n_bem_surfaces']))]
            obj.bem_conductivities = list(z['bem_conductivities'])
        return obj

    @staticmethod
    def _cache_signature(pipeline_dir, spacing_mm: float,
                         inside_frac: float) -> Dict:
        """
        What the cached operator depends on.

        Includes the BEM file's size and mtime: a cache keyed only on spacing
        would silently serve a stale leadfield after the head model is rebuilt,
        which is exactly the kind of error that is invisible in a figure.
        """
        bem = Path(pipeline_dir) / 'bem_cache' / 'ellipsoid_3layer.pkl'
        st = bem.stat() if bem.exists() else None
        return {
            'pipeline_dir': str(Path(pipeline_dir).resolve()),
            'spacing_mm': float(spacing_mm),
            'inside_frac': float(inside_frac),
            'bem_bytes': (st.st_size if st else None),
            'bem_mtime': (int(st.st_mtime) if st else None),
        }

    def _check_geometry(self, verbose: bool = True) -> None:
        """
        Refuse a candidate space that contains impossible source locations.

        A source cannot sit above the electrode array that records it. This
        exact failure went unnoticed once — the grid was built from the inflated
        BEM surface and 28 positions ended up above the highest electrode — so
        it is now checked rather than assumed.
        """
        if self.electrode_pos_mm is None:
            return
        z_max = float(self.electrode_pos_mm[:, 2].max())
        above = int((self.positions_mm[:, 2] > z_max).sum())
        if above:
            raise ValueError(
                f"{above} of {self.n_positions} candidate positions lie above "
                f"the highest electrode (z = {z_max:.2f} mm). The source grid "
                f"is not confined to the brain — check the mask constraint in "
                f"_grid_inside.")
        if verbose:
            print(f"  geometry OK: all {self.n_positions} positions below the "
                  f"array (electrode z max {z_max:.2f} mm); "
                  f"volume {self.n_positions * self.spacing_mm ** 3:.0f} mm^3")

    #: Geometry attached by :meth:`from_pipeline_dir` (absent otherwise).
    provenance: Dict = {}
    bem_surfaces_mm = None
    bem_conductivities = ()
    electrode_pos_mm = None
    channel_names = ()

    def layer_outline_mm(self, layer: int, z_mm: float, n: int = 240):
        """
        Cross-section of a BEM layer at height ``z_mm``, as an (n, 2) polygon.

        The BEM layers are axis-aligned ellipsoids (preset
        ``ellipsoid_method: axis_aligned``), so the slice is an ellipse obtained
        analytically from the layer's centre and semi-axes. Returns ``None``
        when the plane misses the layer.
        """
        if self.bem_surfaces_mm is None:
            return None
        rr = self.bem_surfaces_mm[layer]
        centre = 0.5 * (rr.max(axis=0) + rr.min(axis=0))
        semi = 0.5 * (rr.max(axis=0) - rr.min(axis=0))
        t = (z_mm - centre[2]) / semi[2]
        if abs(t) >= 1.0:
            return None
        k = np.sqrt(1.0 - t ** 2)
        ang = np.linspace(0, 2 * np.pi, n)
        return np.column_stack([centre[0] + semi[0] * k * np.cos(ang),
                                centre[1] + semi[1] * k * np.sin(ang)])

    @staticmethod
    def _grid_inside(bem: Dict, spacing_mm: float, frac: float,
                     brain_mask_path=None) -> np.ndarray:
        """
        Regular grid (metres) of candidate source positions inside the brain.

        Constrained by the **atlas brain mask**, not by the BEM's innermost
        surface. That surface is an ellipsoid fitted to the brain voxels and
        then inflated by ``ellipsoid_margin`` (1.23 by default), so it is ~23%
        larger on every axis — about 1.9x the volume. Using it as the candidate
        space put 49% of the grid outside real tissue and 28 points *above the
        highest electrode*, which are impossible source locations, and inflated
        the "brain volume" from ~390 mm^3 to ~750 mm^3, halving every
        "percent of brain" figure. The pipeline's own source space never had
        this problem because it places sources at 0.25-0.95 of the brain radius.

        The mask is read with :func:`utils.atlas.get_true_affine` — the atlas
        header's voxel sizes are 10x too large, and the **translation must be
        scaled too**, otherwise the mask lands nowhere near the source
        coordinates.
        """
        import nibabel as nib
        from ..utils.atlas import get_true_affine

        if brain_mask_path is None:
            brain_mask_path = (Path(__file__).resolve().parents[1]
                               / 'data' / 'atlas' / 'Atlas_3DRois_brain.nii.gz')
        nii = nib.load(str(brain_mask_path))
        mask = np.asarray(nii.get_fdata()) > 0
        affine = get_true_affine(nii)

        occupied = np.argwhere(mask)
        mm = (affine @ np.column_stack(
            [occupied, np.ones(len(occupied))]).T).T[:, :3]
        lo, hi = mm.min(axis=0), mm.max(axis=0)

        axes = [np.arange(lo[i], hi[i] + spacing_mm / 2, spacing_mm)
                for i in range(3)]
        grid_mm = np.array(np.meshgrid(*axes, indexing='ij')).reshape(3, -1).T

        inv = np.linalg.inv(affine)
        vox = np.rint((inv @ np.column_stack(
            [grid_mm, np.ones(len(grid_mm))]).T).T[:, :3]).astype(int)
        in_bounds = np.all((vox >= 0) & (vox < np.array(mask.shape)), axis=1)
        keep = np.zeros(len(grid_mm), bool)
        v = vox[in_bounds]
        keep[in_bounds] = mask[v[:, 0], v[:, 1], v[:, 2]]

        # Also require containment in the BEM conductor, so no source sits on or
        # outside the innermost surface where the solution is ill-conditioned.
        inner = bem['surfs'][0]['rr'] * 1000.0
        centre = 0.5 * (inner.max(axis=0) + inner.min(axis=0))
        semi = 0.5 * (inner.max(axis=0) - inner.min(axis=0))
        keep &= np.sum(((grid_mm - centre) / (semi * frac)) ** 2, axis=1) <= 1.0

        return grid_mm[keep] / 1000.0

    @staticmethod
    def _as_source_space(template, positions_m: np.ndarray):
        """Clone a volumetric SourceSpaces with a new set of points."""
        import copy
        ss = copy.deepcopy(template)
        s = ss[0]
        n = len(positions_m)
        s['rr'] = positions_m
        s['nn'] = np.tile([0.0, 0.0, 1.0], (n, 1))
        s['np'] = n
        s['inuse'] = np.ones(n, dtype=int)
        s['vertno'] = np.arange(n)
        s['nuse'] = n
        return ss

    # ------------------------------------------------------------- likelihood
    def leadfield_at(self, position_mm: np.ndarray) -> np.ndarray:
        """Leadfield (n_chan, 3) at the nearest grid position."""
        idx = int(np.argmin(np.linalg.norm(self.positions_mm - position_mm, axis=1)))
        return self._G[idx]

    def residuals(self, data: np.ndarray) -> np.ndarray:
        """
        Squared residual of the best-fitting dipole at every grid position.

        Parameters
        ----------
        data : ndarray, shape (n_channels,) or (n_channels, n_times)

        Returns
        -------
        ndarray, shape (n_positions,)
        """
        b = np.asarray(data, float)
        if b.ndim == 1:
            b = b[:, None]
        total = float(np.sum(b ** 2))
        # ||Q^T B||^2 summed over the 3 basis directions and time.
        proj = np.einsum('pck,ct->pkt', self._Q, b)
        return total - np.sum(proj ** 2, axis=(1, 2))

    def posterior(
        self,
        data: np.ndarray,
        noise_std: float,
        moment_std: Optional[float] = None,
    ) -> np.ndarray:
        """
        Normalized posterior over location, uniform spatial prior.

        Parameters
        ----------
        data : ndarray, shape (n_channels,) or (n_channels, n_times)
        noise_std : float
            Sensor noise standard deviation, same units as ``data``.
        moment_std : float, optional
            Standard deviation of the Gaussian prior on the dipole moment. When
            given, the moment is **marginalized** rather than profiled out:

                B | x  ~  N(0,  sigma^2 I + tau^2 G_x G_x^T)

            which adds a ``log det`` (Occam) term that profiling drops. That term
            matters because it penalizes high-gain positions for being able to
            explain too many datasets, and gain varies by orders of magnitude
            with depth here. Profiling (``None``) is over-confident once the
            noise is comparable to the signal — at 0 dB its 68% region covered
            only 53% of the time, while the marginal version stays calibrated.

        Returns
        -------
        ndarray, shape (n_positions,)
            Probability mass per grid cell; sums to 1.
        """
        b = np.asarray(data, float)
        if b.ndim == 1:
            b = b[:, None]
        n_times = b.shape[1]
        var = noise_std ** 2

        # Energy of the data along each position's 3 signal directions.
        proj = np.einsum('pck,ct->pkt', self._U, b)
        proj_energy = np.sum(proj ** 2, axis=2)          # (n_pos, 3)
        total = float(np.sum(b ** 2))

        if moment_std is None:
            log_p = -(total - proj_energy.sum(axis=1)) / (2.0 * var)
        else:
            s2 = self._S ** 2                             # (n_pos, 3)
            denom = var + (moment_std ** 2) * s2
            shrink = (moment_std ** 2) * s2 / denom       # 0..1 per direction
            quad = (total - np.sum(shrink * proj_energy, axis=1)) / var
            log_det = np.sum(np.log(denom), axis=1) - 3.0 * np.log(var)
            log_p = -0.5 * quad - 0.5 * n_times * log_det

        log_p -= log_p.max()               # stabilize before exponentiating
        p = np.exp(log_p)
        s = p.sum()
        return p / s if s > 0 else p

    # --------------------------------------------------------- credible sets
    @staticmethod
    def credible_mask(posterior: np.ndarray, level: float = 0.5) -> np.ndarray:
        """
        Highest-density region holding ``level`` of the probability mass.

        Returns a boolean mask over grid positions. Highest-density (not
        symmetric-interval) so the region is the *smallest* set containing that
        much probability, which is what makes its volume meaningful.
        """
        order = np.argsort(posterior)[::-1]
        cumulative = np.cumsum(posterior[order])
        k = int(np.searchsorted(cumulative, level) + 1)
        mask = np.zeros(posterior.shape, dtype=bool)
        mask[order[:k]] = True
        return mask

    def credible_volume_mm3(self, posterior: np.ndarray,
                            level: float = 0.5) -> float:
        """Volume of the highest-density region at ``level``."""
        return float(self.credible_mask(posterior, level).sum()
                     * self.spacing_mm ** 3)

    def credible_radius_mm(self, posterior: np.ndarray,
                           level: float = 0.5) -> float:
        """
        Radius of a sphere with the same volume as the credible region.

        A single interpretable number: "the source is within about R mm of the
        estimate, with ``level`` probability" — provided the region is compact,
        which :meth:`credible_mask` does not guarantee, so check the volume too.
        """
        v = self.credible_volume_mm3(posterior, level)
        return float((3.0 * v / (4.0 * np.pi)) ** (1.0 / 3.0))

    # ------------------------------------------------------------ simulation
    def simulate(
        self,
        position_mm: np.ndarray,
        orientation: np.ndarray,
        snr_db: float,
        rng: np.random.Generator,
        n_times: int = 1,
        amplitude: float = 1.0,
        leadfield: Optional[np.ndarray] = None,
    ) -> Tuple[np.ndarray, float]:
        """
        Sensor data from a dipole, with white noise at a target SNR.

        Generated here rather than via ``validation.simulation`` so the noise
        exactly matches the likelihood's assumption — appropriate for a
        fundamental-limit calculation, where the question is what is achievable
        with a correct model.

        ``leadfield`` may be supplied to place truth at an arbitrary continuous
        position (off the grid); otherwise the nearest grid column is used.

        Returns
        -------
        (data, noise_std)
        """
        g = self.leadfield_at(position_mm) if leadfield is None else leadfield
        o = np.asarray(orientation, float)
        o = o / np.linalg.norm(o)
        signal = (g @ o)[:, None] * amplitude
        if n_times > 1:
            signal = np.repeat(signal, n_times, axis=1)

        signal_power = float(np.mean(signal ** 2))
        noise_power = signal_power / (10.0 ** (snr_db / 10.0))
        noise_std = float(np.sqrt(noise_power))
        data = signal + rng.normal(0.0, noise_std, size=signal.shape)
        return data, noise_std

    # ----------------------------------------------------------- calibration
    def coverage_test(
        self,
        levels: Tuple[float, ...] = (0.5, 0.68, 0.9, 0.95),
        snr_db: float = 10.0,
        n_trials: int = 200,
        seed: int = 0,
        off_grid: bool = True,
        truth_leadfields: Optional[Dict] = None,
        moment_std: Optional[float] = None,
    ) -> Dict:
        """
        Does the p% credible region contain the truth p% of the time?

        The test that makes the percentages mean something. Draws random true
        locations and orientations, simulates, computes the posterior, and checks
        whether the true location falls inside each credible region. Empirical
        coverage should track the nominal level; systematically low coverage
        means the regions are too small (over-confident).

        Parameters
        ----------
        levels : tuple of float
            Nominal credible levels to check.
        snr_db : float
        n_trials : int
        seed : int
        off_grid : bool, default=True
            Place truth at continuous positions between grid points, so the test
            also probes the discretization. Truth is then scored against the
            nearest grid cell.
        truth_leadfields : dict, optional
            Precomputed ``{'positions_mm', 'G'}`` for off-grid truth. Computing
            these needs a forward solve, so a driver can build them in one batch.

        Returns
        -------
        dict
            ``levels``, ``coverage`` (empirical fraction per level),
            ``median_volume_mm3`` and ``median_radius_mm`` per level,
            ``n_trials``, ``snr_db``.
        """
        rng = np.random.default_rng(seed)
        levels = tuple(levels)
        hits = {lv: 0 for lv in levels}
        vols = {lv: [] for lv in levels}

        if truth_leadfields is not None:
            truth_pos = truth_leadfields['positions_mm']
            truth_G = truth_leadfields['G']
            n_trials = min(n_trials, len(truth_pos))
        else:
            pick = rng.integers(0, self.n_positions, size=n_trials)
            truth_pos = self.positions_mm[pick]
            truth_G = self._G[pick]

        for t in range(n_trials):
            o = rng.normal(size=3)
            data, noise_std = self.simulate(
                truth_pos[t], o, snr_db, rng, leadfield=truth_G[t])
            post = self.posterior(data, noise_std, moment_std=moment_std)

            # Score truth by the grid cell it falls in.
            truth_idx = int(np.argmin(
                np.linalg.norm(self.positions_mm - truth_pos[t], axis=1)))
            for lv in levels:
                mask = self.credible_mask(post, lv)
                if mask[truth_idx]:
                    hits[lv] += 1
                vols[lv].append(mask.sum() * self.spacing_mm ** 3)

        return {
            'levels': list(levels),
            'coverage': [hits[lv] / n_trials for lv in levels],
            'median_volume_mm3': [float(np.median(vols[lv])) for lv in levels],
            'median_radius_mm': [float((3 * np.median(vols[lv]) / (4 * np.pi))
                                       ** (1 / 3)) for lv in levels],
            'n_trials': int(n_trials),
            'snr_db': float(snr_db),
            'off_grid': bool(off_grid),
        }
