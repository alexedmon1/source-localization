"""
If activity is attributed to a parcel, how much should you believe it?

The calibrated single-dipole posterior (:mod:`.posterior`) answers *where is the
source* as a probability density over location. This module integrates that
density over an atlas parcellation to answer the practical question a user asks
of the pipeline's ROI output: given the data, what is the probability the source
lies in parcel *k*, and when the pipeline names a parcel, how often is it right?

Two views of the same joint distribution over (true parcel, attributed parcel):

- **Recall** ``P(attributed = k | true = k)`` — a source really in *k*, how often
  is it found there. The row-normalized confusion matrix.
- **Precision** ``P(true = k | attributed = k)`` — the pipeline named *k*, how
  often is that correct. The column-normalized confusion matrix. These differ
  sharply for a small parcel beside a large one, and precision depends on the
  assumed prior over source location (here uniform over labeled tissue).

Two estimators, so the cost of the deployed method is visible:

- **posterior** — the fundamental limit. ``P(parcel | B)`` is the posterior mass
  falling in each parcel; the point attribution is its argmax. Because the
  density is calibrated, these parcel probabilities are auditable: when it says
  0.7 it should be right 70% of the time (:func:`reliability`).
- **sLORETA** — the pipeline's actual estimator. The attributed parcel is the one
  containing the peak of the sLORETA reconstruction on the shell source space.
  A point estimate only, so it has a confusion matrix but no reliability curve.

Everything is stratified by the true source's distance to the nearest electrode
(:data:`.run_two_source_posterior.DEPTH_BANDS`), because localization on this
array is depth-limited and a whole-brain average hides that.

Atlas note. Parcels are assigned by nearest labeled voxel, exactly as
``steps/roi_extraction.py`` does, so the numbers describe the deployed
convention. This is a sub-voxel nudge for allen32 (which tiles grey matter) and
a much larger reach for Antwerp; prefer allen32 here. Simulated truth is drawn
only from positions strictly inside a label, so "the true parcel" is always
defined.
"""
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np

from .posterior import DipolePosterior

__all__ = ['ParcelMap', 'posterior_parcel_probs', 'ConfusionResult',
           'confusion_matrices', 'reliability', 'sample_truth_positions',
           'posterior_attributor', 'sloreta_attributor']


class ParcelMap:
    """
    Maps posterior grid positions to atlas parcels.

    Built once for a given (posterior grid, atlas) pair and reused across trials.
    Holds the per-position parcel assignment, which positions are strictly inside
    a label (valid truth locations), and each position's depth.

    Parameters
    ----------
    positions_mm : ndarray, shape (n_positions, 3)
        The posterior grid — normally ``DipolePosterior.positions_mm``.
    atlas : str, default='allen32'
        Registry name of the parcellation.
    electrode_pos_mm : ndarray, shape (n_electrodes, 3), optional
        Used to compute per-position depth (distance to nearest electrode). If
        omitted, :attr:`depth_mm` is all-NaN.
    """

    def __init__(self, positions_mm: np.ndarray, atlas: str = 'allen32',
                 electrode_pos_mm: Optional[np.ndarray] = None):
        import nibabel as nib
        from ..config import atlas_input_paths
        from ..utils.atlas import get_true_affine

        self.positions_mm = np.asarray(positions_mm, float)
        self.n_positions = len(self.positions_mm)
        self.atlas = atlas

        pkg = Path(__file__).resolve().parents[1]
        paths = atlas_input_paths(atlas)
        nii = nib.load(str(pkg / paths['brain_labels']))
        label_vol = np.asarray(nii.get_fdata()).astype(int)
        affine = get_true_affine(nii)

        self.parcel_ids, self.parcel_names = self._load_names(
            pkg / paths['roi_mapping'])

        # Voxel index of each grid position, and its label there (0 if unlabeled).
        inv = np.linalg.inv(affine)
        vox = np.rint((inv @ np.column_stack(
            [self.positions_mm, np.ones(self.n_positions)]).T).T[:, :3]).astype(int)
        in_bounds = np.all((vox >= 0) & (vox < np.array(label_vol.shape)), axis=1)
        direct = np.zeros(self.n_positions, int)
        vb = vox[in_bounds]
        direct[in_bounds] = label_vol[vb[:, 0], vb[:, 1], vb[:, 2]]
        # A position is a valid truth location only if it sits strictly inside a
        # labeled voxel, so the true parcel is unambiguous.
        self.inside = direct > 0

        # Force-assign every position (including unlabeled ones) to the nearest
        # labeled voxel — the roi_extraction.py convention.
        self.labels = self._nearest_label(label_vol, affine)

        if electrode_pos_mm is not None:
            e = np.asarray(electrode_pos_mm, float)
            self.depth_mm = np.min(np.linalg.norm(
                self.positions_mm[:, None, :] - e[None, :, :], axis=2), axis=1)
        else:
            self.depth_mm = np.full(self.n_positions, np.nan)

    @staticmethod
    def _load_names(mapping_path) -> Tuple[np.ndarray, List[str]]:
        import json
        rois = json.load(open(mapping_path)).get('rois', {})
        ids = sorted(int(k) for k in rois if int(k) > 0)
        names = [rois[str(i)]['name'] for i in ids]
        return np.array(ids), names

    def _nearest_label(self, label_vol: np.ndarray,
                       affine: np.ndarray) -> np.ndarray:
        """Parcel id of the nearest labeled voxel to each grid position."""
        from scipy.spatial import cKDTree
        occ = np.argwhere(label_vol > 0)
        occ_mm = (affine @ np.column_stack(
            [occ, np.ones(len(occ))]).T).T[:, :3]
        occ_lab = label_vol[occ[:, 0], occ[:, 1], occ[:, 2]]
        _, idx = cKDTree(occ_mm).query(self.positions_mm)
        return occ_lab[idx]

    @property
    def n_parcels(self) -> int:
        return len(self.parcel_ids)

    def index_of(self, parcel_id: int) -> int:
        """Row/column index (0-based) of a parcel id in the confusion matrix."""
        return int(np.searchsorted(self.parcel_ids, parcel_id))

    def aggregate(self, posterior: np.ndarray) -> np.ndarray:
        """
        Sum a location posterior into per-parcel probability mass.

        Every grid cell is charged to its assigned parcel, so the result sums to
        the same total as ``posterior`` (1.0 for a normalized posterior). Returns
        a vector aligned with :attr:`parcel_ids`.
        """
        out = np.zeros(self.n_parcels, float)
        # labels -> contiguous parcel index via searchsorted on the sorted ids.
        col = np.searchsorted(self.parcel_ids, self.labels)
        np.add.at(out, col, posterior)
        return out


def posterior_parcel_probs(
    dp: DipolePosterior,
    data: np.ndarray,
    noise_std: float,
    pmap: ParcelMap,
    moment_std: Optional[float] = 1.0,
) -> np.ndarray:
    """
    ``P(source in parcel_k | B)`` for every parcel.

    Convenience wrapper: computes the location posterior with the calibrated
    marginal likelihood (``moment_std`` given) and aggregates it over parcels.
    """
    post = dp.posterior(data, noise_std, moment_std=moment_std)
    return pmap.aggregate(post)


# --------------------------------------------------------------- truth sampling

def sample_truth_positions(
    pmap: ParcelMap,
    n_per_parcel: int,
    rng: np.random.Generator,
) -> Dict[str, np.ndarray]:
    """
    Draw truth grid indices, balanced across parcels.

    Sampling per parcel rather than uniformly over the grid keeps small parcels
    from vanishing out of the confusion matrix — a uniform draw would give
    Auditory (32 positions) a hundredth the trials of Brainstem (359). Only
    positions strictly inside a label are eligible, so every drawn truth has a
    well-defined parcel. Draws are with replacement when a parcel has fewer
    inside positions than ``n_per_parcel``.

    Returns
    -------
    dict
        ``idx`` (grid indices into ``pmap.positions_mm``), ``parcel`` (true
        parcel index per trial), ``depth_mm`` (per trial).
    """
    inside_idx = np.where(pmap.inside)[0]
    inside_parcel = np.searchsorted(pmap.parcel_ids, pmap.labels[inside_idx])

    picks = []
    for k in range(pmap.n_parcels):
        pool = inside_idx[inside_parcel == k]
        if len(pool) == 0:
            continue
        replace = len(pool) < n_per_parcel
        picks.append(rng.choice(pool, size=n_per_parcel, replace=replace))
    idx = np.concatenate(picks)
    rng.shuffle(idx)
    return {
        'idx': idx,
        'parcel': np.searchsorted(pmap.parcel_ids, pmap.labels[idx]),
        'depth_mm': pmap.depth_mm[idx],
    }


# ------------------------------------------------------------ confusion matrix

@dataclass
class ConfusionResult:
    """(true parcel, attributed parcel) counts and the two normalized views."""
    counts: np.ndarray               # (n_parcels, n_parcels): rows=true, cols=attr
    parcel_names: List[str]
    estimator: str
    n_trials: int
    depth_band: Optional[str] = None

    @property
    def recall(self) -> np.ndarray:
        """Row-normalized: P(attributed = k | true = k). Diagonal is per-parcel recall."""
        row = self.counts.sum(axis=1, keepdims=True)
        return np.divide(self.counts, row, out=np.zeros_like(self.counts, float),
                         where=row > 0)

    @property
    def precision(self) -> np.ndarray:
        """Column-normalized: P(true = k | attributed = k). Diagonal is per-parcel precision."""
        col = self.counts.sum(axis=0, keepdims=True)
        return np.divide(self.counts, col, out=np.zeros_like(self.counts, float),
                         where=col > 0)

    @property
    def overall_accuracy(self) -> float:
        n = self.counts.sum()
        return float(np.trace(self.counts) / n) if n else 0.0

    def diagonal(self, kind: str) -> np.ndarray:
        m = self.recall if kind == 'recall' else self.precision
        return np.diag(m)


def confusion_matrices(
    attribute: Callable[[int, np.ndarray, float, np.random.Generator], int],
    dp: DipolePosterior,
    pmap: ParcelMap,
    truth: Dict[str, np.ndarray],
    snr_db: float,
    estimator: str,
    rng: np.random.Generator,
    depth_bands: Optional[Sequence[Tuple[str, float, float]]] = None,
) -> Tuple[ConfusionResult, Dict[str, ConfusionResult]]:
    """
    Build the (true, attributed) confusion matrix over a set of truth trials.

    ``attribute`` is the estimator: given a truth grid index, the simulated
    sensor data, the noise std and an rng, it returns the attributed parcel
    index. Keeping it a callback lets the posterior and sLORETA arms share this
    loop and see identical truth and noise.

    Returns the pooled result and, if ``depth_bands`` is given, one result per
    band (keyed by band name).
    """
    n = pmap.n_parcels
    pooled = np.zeros((n, n), int)
    banded = None
    if depth_bands is not None:
        banded = {name: np.zeros((n, n), int) for name, _, _ in depth_bands}

    idx = truth['idx']
    true_parcel = truth['parcel']
    depth = truth['depth_mm']

    for t, gi in enumerate(idx):
        o = rng.normal(size=3)
        data, noise_std = dp.simulate(
            dp.positions_mm[gi], o, snr_db, rng, leadfield=dp._G[gi])
        attr = attribute(gi, data, noise_std, rng)
        pooled[true_parcel[t], attr] += 1
        if banded is not None:
            b = _band_of(depth[t], depth_bands)
            if b is not None:
                banded[b][true_parcel[t], attr] += 1

    pooled_res = ConfusionResult(pooled, pmap.parcel_names, estimator, len(idx))
    band_res = {}
    if banded is not None:
        for name, mat in banded.items():
            band_res[name] = ConfusionResult(
                mat, pmap.parcel_names, estimator, int(mat.sum()), depth_band=name)
    return pooled_res, band_res


def _band_of(depth_mm: float,
             bands: Sequence[Tuple[str, float, float]]) -> Optional[str]:
    for name, lo, hi in bands:
        if lo <= depth_mm < hi:
            return name
    return None


# ------------------------------------------------------------ estimator arms

def posterior_attributor(
    dp: DipolePosterior,
    pmap: ParcelMap,
    moment_std: Optional[float] = 1.0,
) -> Callable[[int, np.ndarray, float, np.random.Generator], int]:
    """
    Attributor for the fundamental-limit arm: argmax of ``P(parcel | B)``.

    Point attribution for the confusion matrix. The full probability vector,
    used by the reliability diagram, comes from :func:`posterior_parcel_probs`.
    """
    def attribute(gi, data, noise_std, rng):
        probs = posterior_parcel_probs(dp, data, noise_std, pmap, moment_std)
        return int(probs.argmax())
    return attribute


def sloreta_attributor(
    pipeline_dir,
    atlas: str,
    snr: float = 3.0,
) -> Callable[[int, np.ndarray, float, np.random.Generator], int]:
    """
    Attributor for the deployed arm: the parcel holding the sLORETA peak.

    Loads the pipeline run's shell forward once and applies the package's own
    ``apply_inverse_custom_sLORETA`` per trial, so this is the estimator a user
    actually gets — not the fundamental limit. The attributed parcel is the one
    containing the reconstruction's peak source on the 215-point shell.

    The shell forward and the posterior's dense grid come from the same BEM and
    montage, so feeding both arms the same simulated sensor vector is a fair
    comparison; only the estimator differs.
    """
    import pickle
    from ..steps.inverse_solution import apply_inverse_custom_sLORETA

    p = Path(pipeline_dir)
    fwd = pickle.load(open(p / 'data' / 'step4_forward.pkl', 'rb'))
    shell_coords = np.load(p / 'data' / 'step3_source_coords_mm.npy')

    # Assign each shell source to a parcel, aligned with the confusion matrix's
    # parcel indexing (pmap over the dense grid uses the same sorted ids).
    shell_map = ParcelMap(shell_coords, atlas)
    shell_parcel_col = np.searchsorted(shell_map.parcel_ids, shell_map.labels)

    def attribute(gi, data, noise_std, rng):
        mag, _ = apply_inverse_custom_sLORETA(fwd, data, snr=snr, verbose=False)
        # mag is (n_sources, n_times); collapse time to a per-source strength.
        strength = mag.mean(axis=1) if mag.ndim == 2 else mag
        peak = int(strength.argmax())
        return int(shell_parcel_col[peak])
    return attribute


# ---------------------------------------------------------------- reliability

def reliability(
    dp: DipolePosterior,
    pmap: ParcelMap,
    truth: Dict[str, np.ndarray],
    snr_db: float,
    rng: np.random.Generator,
    n_bins: int = 10,
    moment_std: Optional[float] = 1.0,
) -> Dict[str, np.ndarray]:
    """
    Reliability of the posterior parcel probabilities (posterior arm only).

    For each trial, bin the posterior probability assigned to the *true* parcel,
    and within each probability bin record how often the true parcel really was
    the argmax. If the probabilities are honest, a bin centred at 0.7 should
    contain trials whose true parcel wins ~70% of the time. This is the parcel
    analogue of :meth:`DipolePosterior.coverage_test`.

    Returns bin centres, mean predicted probability, empirical hit rate, and
    per-bin counts.
    """
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    pred_sum = np.zeros(n_bins)
    hit_sum = np.zeros(n_bins)
    count = np.zeros(n_bins, int)

    for t, gi in enumerate(truth['idx']):
        o = rng.normal(size=3)
        data, noise_std = dp.simulate(
            dp.positions_mm[gi], o, snr_db, rng, leadfield=dp._G[gi])
        probs = posterior_parcel_probs(dp, data, noise_std, pmap, moment_std)
        true_k = truth['parcel'][t]
        p_true = probs[true_k]
        won = int(probs.argmax() == true_k)

        b = min(int(p_true * n_bins), n_bins - 1)
        pred_sum[b] += p_true
        hit_sum[b] += won
        count[b] += 1

    with np.errstate(invalid='ignore', divide='ignore'):
        mean_pred = np.where(count > 0, pred_sum / count, np.nan)
        hit_rate = np.where(count > 0, hit_sum / count, np.nan)
    return {
        'bin_centres': 0.5 * (edges[:-1] + edges[1:]),
        'mean_predicted': mean_pred,
        'empirical': hit_rate,
        'count': count,
        'snr_db': float(snr_db),
    }
