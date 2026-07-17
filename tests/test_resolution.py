"""Tests for the resolution metrics (validation/resolution.py).

The PSF computation needs a forward model, but the metrics that ride on top —
peak localization error and spatial dispersion — are pure geometry over the
source grid and a PSF vector, so they're tested here without one. These pin the
behavior the resolution map depends on: a sharp PSF disperses less than a broad
one, a shifted peak reports localization error, and depth aggregation is correct.
"""
import numpy as np
import pytest

from source_localization.validation.resolution import ResolutionAnalysis


class _StubResolution:
    """ResolutionAnalysis metric methods over a hand-placed source grid."""
    psf_metrics = ResolutionAnalysis.psf_metrics
    summarize_by_depth = ResolutionAnalysis.summarize_by_depth
    DEFAULT_DEPTH_BINS = ResolutionAnalysis.DEFAULT_DEPTH_BINS

    def __init__(self, source_pos_mm, source_depths):
        self.source_pos_mm = np.asarray(source_pos_mm, float)
        self.n_sources = self.source_pos_mm.shape[0]
        self.source_depths = np.asarray(source_depths, float)


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
