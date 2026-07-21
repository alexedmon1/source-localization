"""Tests for threshold-based separability (validation/separability.py).

The blob logic is pure geometry over a source grid and an activation vector, so
it is tested here without a forward model. These pin the behaviour the
separability claims rest on: thresholding splits spatially disjoint activation,
the three verdicts mean what they say, and blob size shrinks as the threshold
rises (the property spatial dispersion lacked).
"""
import numpy as np
import pytest
from scipy.spatial import cKDTree

from source_localization.validation.separability import SeparabilityAnalysis


class _StubSeparability:
    """SeparabilityAnalysis blob methods over a hand-placed source grid."""
    suprathreshold_labels = SeparabilityAnalysis.suprathreshold_labels
    classify_pair = SeparabilityAnalysis.classify_pair
    blob_extent = SeparabilityAnalysis.blob_extent
    _adjacency_pairs = SeparabilityAnalysis._adjacency_pairs
    ADJACENCY_SPACING_FACTOR = SeparabilityAnalysis.ADJACENCY_SPACING_FACTOR

    def __init__(self, source_pos_mm, spacing):
        self.source_pos_mm = np.asarray(source_pos_mm, float)
        self.n_sources = self.source_pos_mm.shape[0]
        self._spacing = spacing
        self._adj_cache = None

    def _median_grid_spacing(self):
        return self._spacing

    def _get_source_kdtree(self):
        return cKDTree(self.source_pos_mm)


@pytest.fixture
def line():
    """41 sources on a 0.5 mm line spanning 0-20 mm."""
    x = np.arange(0, 20.01, 0.5)
    pos = np.column_stack([x, np.zeros_like(x), np.zeros_like(x)])
    return _StubSeparability(pos, spacing=0.5)


def _two_bumps(stub, c1, c2, sigma, amp2=1.0):
    x = stub.source_pos_mm[:, 0]
    return (np.exp(-((x - c1) ** 2) / (2 * sigma ** 2))
            + amp2 * np.exp(-((x - c2) ** 2) / (2 * sigma ** 2)))


def test_disjoint_activation_splits_into_two_blobs(line):
    act = _two_bumps(line, 5.0, 15.0, sigma=0.8)
    labels = line.suprathreshold_labels(act, 0.5)
    assert len(np.unique(labels[labels >= 0])) == 2


def test_overlapping_activation_stays_one_blob(line):
    act = _two_bumps(line, 9.5, 10.5, sigma=1.5)
    labels = line.suprathreshold_labels(act, 0.5)
    assert len(np.unique(labels[labels >= 0])) == 1


def test_raising_the_threshold_can_split_a_blob(line):
    """The whole premise: cutting lower activation apart reveals two lobes."""
    act = _two_bumps(line, 8.0, 12.0, sigma=1.4)
    low = line.suprathreshold_labels(act, 0.2)
    high = line.suprathreshold_labels(act, 0.9)
    assert len(np.unique(low[low >= 0])) == 1
    assert len(np.unique(high[high >= 0])) == 2


def test_classify_pair_reports_separated_merged_and_dominated(line):
    i1 = int(np.argmin(np.abs(line.source_pos_mm[:, 0] - 5.0)))
    i2 = int(np.argmin(np.abs(line.source_pos_mm[:, 0] - 15.0)))

    assert line.classify_pair(_two_bumps(line, 5.0, 15.0, 0.8), i1, i2, 0.5) == 'separated'

    close1 = int(np.argmin(np.abs(line.source_pos_mm[:, 0] - 9.5)))
    close2 = int(np.argmin(np.abs(line.source_pos_mm[:, 0] - 10.5)))
    assert line.classify_pair(_two_bumps(line, 9.5, 10.5, 1.5),
                              close1, close2, 0.5) == 'merged'

    # Second source at 20% amplitude: well separated, but below a 50% threshold.
    faint = _two_bumps(line, 5.0, 15.0, 0.8, amp2=0.2)
    assert line.classify_pair(faint, i1, i2, 0.5) == 'dominated'
    # ...and dropping the threshold below it recovers the separation, which is
    # the asymmetry that makes 'dominated' worth distinguishing from 'merged'.
    assert line.classify_pair(faint, i1, i2, 0.1) == 'separated'


def test_blob_shrinks_as_threshold_rises(line):
    """Spatial dispersion could not do this — it sat near its ceiling always."""
    act = np.exp(-((line.source_pos_mm[:, 0] - 10.0) ** 2) / (2 * 2.0 ** 2))
    radii = [line.blob_extent(act, t)['rms_radius_mm'] for t in (0.2, 0.5, 0.8)]
    assert radii[0] > radii[1] > radii[2]


def test_blob_extent_flags_a_sub_threshold_source(line):
    act = _two_bumps(line, 5.0, 15.0, 0.8, amp2=0.2)
    i2 = int(np.argmin(np.abs(line.source_pos_mm[:, 0] - 15.0)))
    e = line.blob_extent(act, 0.5, source_idx=i2)
    assert e['contains_source'] is False
    assert e['n_sources'] == 0
