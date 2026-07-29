"""Tests for the resolution metrics (validation/resolution.py).

The PSF computation needs a forward model, but the metrics that ride on top —
peak localization error and spatial dispersion — are pure geometry over the
source grid and a PSF vector, so they're tested here without one. These pin the
behavior the resolution map depends on: a sharp PSF disperses less than a broad
one, a shifted peak reports localization error, and depth aggregation is correct.
"""
import json

import numpy as np
import pytest

from source_localization.validation.resolution import (
    ResolutionAnalysis, _first_stable_separation)


class _StubResolution:
    """ResolutionAnalysis metric methods over a hand-placed source grid."""
    psf_metrics = ResolutionAnalysis.psf_metrics
    summarize_by_depth = ResolutionAnalysis.summarize_by_depth
    summarize_distance_by_depth = ResolutionAnalysis.summarize_distance_by_depth
    _partners_at = ResolutionAnalysis._partners_at
    resolution_vs_snr = ResolutionAnalysis.resolution_vs_snr
    summarize_vs_snr_by_depth = ResolutionAnalysis.summarize_vs_snr_by_depth
    DEFAULT_DEPTH_BINS = ResolutionAnalysis.DEFAULT_DEPTH_BINS
    DEFAULT_SNR_VALUES = ResolutionAnalysis.DEFAULT_SNR_VALUES

    def __init__(self, source_pos_mm, source_depths):
        self.source_pos_mm = np.asarray(source_pos_mm, float)
        self.n_sources = self.source_pos_mm.shape[0]
        self.source_depths = np.asarray(source_depths, float)
        self.verbose = False
        self.calls = []  # (source_idx, snr_db, noise_seed) per _reconstruct

    def _reconstruct(self, source_idx, orientation, snr_db=np.inf,
                     noise_seed=0, noise_type='white'):
        """Stand-in operator: a clean peak plus a trial-dependent far speckle."""
        self.calls.append((source_idx, snr_db, noise_seed))
        psf = _gauss_psf(self, self.source_pos_mm[source_idx, 0], 0.8)
        if np.isfinite(snr_db):
            # Speckle hops around with the seed, as sensor noise would.
            psf = psf.copy()
            psf[noise_seed % self.n_sources] += 0.9
        return psf


@pytest.fixture
def line():
    """21 sources on a 0.5 mm line, depth = |x| (surface at x=0)."""
    x = np.arange(0, 10.5, 0.5)
    pos = np.column_stack([x, np.zeros_like(x), np.zeros_like(x)])
    return _StubResolution(pos, np.abs(x))


def _gauss_psf(stub, center, sigma):
    x = stub.source_pos_mm[:, 0]
    v = np.exp(-((x - center) ** 2) / (2 * sigma ** 2))
    return v / v.max()


def test_sharp_psf_has_low_dispersion(line):
    true_idx = 10  # x = 5.0
    sharp = line.psf_metrics(true_idx, psf=_gauss_psf(line, 5.0, 0.5))
    broad = line.psf_metrics(true_idx, psf=_gauss_psf(line, 5.0, 3.0))
    assert sharp['spatial_dispersion_mm'] < broad['spatial_dispersion_mm']
    # For a Gaussian PSF, SD = sigma/sqrt(2): ~0.35mm for sigma=0.5, ~2.1mm for
    # sigma=3 (slightly less here, truncated by the 10mm grid).
    assert sharp['spatial_dispersion_mm'] < 0.6
    assert broad['spatial_dispersion_mm'] > 1.8
    assert broad['spatial_dispersion_mm'] > 3 * sharp['spatial_dispersion_mm']


def test_centered_psf_has_zero_localization_error(line):
    true_idx = 10  # x = 5.0
    m = line.psf_metrics(true_idx, psf=_gauss_psf(line, 5.0, 0.8))
    assert m['peak_localization_error_mm'] == pytest.approx(0.0)


def test_shifted_peak_reports_localization_error(line):
    true_idx = 10  # x = 5.0, but PSF peaks at x = 8.0
    m = line.psf_metrics(true_idx, psf=_gauss_psf(line, 8.0, 0.8))
    assert m['peak_localization_error_mm'] == pytest.approx(3.0, abs=0.25)


def test_dispersion_measured_from_true_not_peak(line):
    """SD is spread about the TRUE source, so an off-peak PSF disperses more."""
    true_idx = 10  # x = 5.0
    centered = line.psf_metrics(true_idx, psf=_gauss_psf(line, 5.0, 1.0))
    offset = line.psf_metrics(true_idx, psf=_gauss_psf(line, 8.0, 1.0))
    assert offset['spatial_dispersion_mm'] > centered['spatial_dispersion_mm']


def test_peak_contrast_flags_a_flat_psf(line):
    """A flat PSF's argmax is arbitrary; contrast ~1 is what says so."""
    sharp = line.psf_metrics(10, psf=_gauss_psf(line, 5.0, 0.5))
    flat = line.psf_metrics(10, psf=np.ones(line.n_sources))
    assert flat['peak_contrast'] == pytest.approx(1.0)
    assert sharp['peak_contrast'] > 5.0


def test_noise_free_snr_runs_a_single_trial(line):
    """snr=inf is deterministic, so spending n_trials solves on it is waste."""
    line.resolution_vs_snr(snr_values=[10.0, np.inf], source_indices=[10],
                           n_trials=6)
    finite = [c for c in line.calls if np.isfinite(c[1])]
    noise_free = [c for c in line.calls if not np.isfinite(c[1])]
    assert len(finite) == 6
    assert len(noise_free) == 1


def test_each_trial_gets_a_distinct_seed(line):
    line.resolution_vs_snr(snr_values=[0.0, 10.0], source_indices=[8, 12],
                           n_trials=5)
    seeds = [c[2] for c in line.calls]
    assert len(set(seeds)) == len(seeds)  # no accidental seed collisions


def test_dispersion_uses_trial_averaged_map(line):
    """Averaging maps before SD suppresses noise speckle; per-trial SD doesn't."""
    true_idx = 10
    sweep = line.resolution_vs_snr(snr_values=[10.0], source_indices=[true_idx],
                                   n_trials=20)
    averaged_sd = sweep['spatial_dispersion_mm'][0, 0]

    # SD of one noisy trial, for comparison — inflated by the far speckle.
    single = line._reconstruct(true_idx, np.array([1.0, 0.0, 0.0]), snr_db=10.0,
                               noise_seed=0)
    single_sd = line.psf_metrics(true_idx, psf=single / single.max())[
        'spatial_dispersion_mm']
    clean_sd = line.psf_metrics(true_idx, psf=_gauss_psf(line, 5.0, 0.8))[
        'spatial_dispersion_mm']

    assert averaged_sd < single_sd
    assert averaged_sd == pytest.approx(clean_sd, abs=0.2)


def test_ple_iqr_zero_when_peak_is_stable(line):
    """The stub's speckle never outgrows the true peak, so PLE shouldn't wander."""
    sweep = line.resolution_vs_snr(snr_values=[10.0], source_indices=[10],
                                   n_trials=8)
    assert sweep['ple_median_mm'][0, 0] == pytest.approx(0.0)
    assert sweep['ple_iqr_mm'][0, 0] == pytest.approx(0.0)


def test_vs_snr_summary_is_json_friendly(line):
    sweep = line.resolution_vs_snr(snr_values=[10.0, np.inf], n_trials=2)
    summ = line.summarize_vs_snr_by_depth(sweep)
    assert set(summ['deep']['by_snr']) == {'10.0', 'inf'}
    json.dumps(summ)  # raises if numpy scalars leaked through


def test_resolution_distance_needs_a_stable_threshold():
    """One lucky shell must not set the threshold; larger shells must hold too."""
    seps = [2.0, 3.0, 4.0, 5.0]
    # Clears at 3 but collapses at 4 -> not a real threshold.
    assert np.isnan(_first_stable_separation([0.1, 0.6, 0.2, 0.7], seps))
    # Clears at 3 and stays clear.
    assert _first_stable_separation([0.1, 0.6, 0.7, 0.8], seps) == 3.0
    # Never clears.
    assert np.isnan(_first_stable_separation([0.1, 0.2, 0.3, 0.4], seps))


def test_resolution_distance_skips_untested_shells():
    """A shell with no partners is unknown, not a failure."""
    seps = [2.0, 3.0, 4.0, 5.0]
    assert _first_stable_separation([0.1, 0.6, np.nan, 0.7], seps) == 3.0


def test_partners_are_selected_at_the_requested_separation(line):
    # Grid is 0.5 mm spaced along x; source 10 sits at x=5.0.
    partners = line._partners_at(10, separation_mm=3.0, n_partners=2, tol_mm=0.3)
    xs = sorted(line.source_pos_mm[p, 0] for p in partners)
    assert xs == [2.0, 8.0]  # both directions, exactly 3 mm away
    assert 10 not in partners  # never itself


def test_flat_psf_sources_excluded_from_distance_summary(line):
    """Averaging in locations whose peak is arbitrary would invent a number."""
    depth = line.source_depths
    n = len(depth)
    dmap = {
        'depth_mm': depth,
        # Everything "resolves" at 4 mm, but only half the sources are meaningful.
        'resolution_distance_mm': np.full(n, 4.0),
        'meaningful': np.arange(n) % 2 == 0,
        'null_fraction': np.zeros((n, 3)),
    }
    summ = line.summarize_distance_by_depth(dmap)
    for lab, s in summ.items():
        assert s['median_resolution_distance_mm'] == 4.0
        assert s['n_unreliable_flat_psf'] > 0
        assert s['n'] > s['n_unreliable_flat_psf']


def test_summarize_by_depth_orders_bins(line):
    # Build a resolution map where SD grows with depth, then check aggregation.
    depth = line.source_depths
    rmap = {
        'depth_mm': depth,
        'peak_localization_error_mm': np.zeros_like(depth),
        'spatial_dispersion_mm': depth.copy(),  # SD == depth by construction
    }
    summ = line.summarize_by_depth(rmap)
    labels = ['very_shallow', 'shallow', 'mid', 'deep']
    sds = [summ[l]['median_sd_mm'] for l in labels if l in summ]
    assert sds == sorted(sds)  # deeper bins have larger SD
