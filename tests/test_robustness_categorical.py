"""Tests for categorical (noise_type) robustness sweeps.

Every pre-existing robustness sweep varied a *numeric* parameter (SNR, noise
level, amplitude), so RobustnessResults assumed floats: `correlation` fed the
keys to np.corrcoef and `to_dict`/`get_summary` called float() on them. The
noise_type sweep is the first categorical one — its parameter is a label with
no ordering — so these tests pin the behaviour that keeps numeric sweeps
working while making categorical ones report honestly (no fabricated
correlation, no bogus physics_valid verdict).
"""
import numpy as np
import pytest

from source_localization.validation.robustness import RobustnessResults


def _categorical(errors=None):
    errors = errors or {
        'white': [1.0, 1.2, 1.1],
        'spatial': [1.6, 1.8, 1.7],
        'colored': [2.0, 2.4, 2.2],
    }
    return RobustnessResults(
        test_type='noise_type',
        parameter_values=list(errors.keys()),
        errors=errors,
        n_sources=200,
        n_trials_per_position=3,
        n_positions=1,
    )


def _numeric():
    return RobustnessResults(
        test_type='noise',
        parameter_values=[0.1, 1.0, 10.0],
        errors={0.1: [1.0, 1.1], 1.0: [1.5, 1.6], 10.0: [2.5, 2.6]},
        n_sources=200,
        n_trials_per_position=2,
        n_positions=1,
    )


def test_categorical_is_detected():
    assert _categorical().is_categorical is True
    assert _numeric().is_categorical is False


def test_categorical_correlation_is_nan_not_a_crash():
    """String labels have no ordering; correlating them would be meaningless."""
    assert np.isnan(_categorical().correlation)


def test_numeric_correlation_still_computed():
    """Guard the pre-existing sweeps: noise level still correlates with error."""
    corr = _numeric().correlation
    assert not np.isnan(corr)
    assert corr > 0.5


def test_categorical_to_dict_preserves_labels():
    """float() on a label would raise; labels must survive serialization."""
    d = _categorical().to_dict()
    assert d['parameter_values'] == ['white', 'spatial', 'colored']
    assert d['correlation'] is None
    assert set(d['means']) == {'white', 'spatial', 'colored'}


def test_numeric_to_dict_still_floats():
    d = _numeric().to_dict()
    assert d['parameter_values'] == [0.1, 1.0, 10.0]
    assert d['correlation'] is not None


def test_categorical_means_track_degradation():
    """The band itself: white is the optimistic end, colored the realistic end."""
    means = _categorical().means
    assert means['white'] < means['spatial'] < means['colored']


class _FakeTest:
    """Minimal stand-in exposing just what get_summary() touches."""

    from source_localization.validation.robustness import RobustnessTest
    get_summary = RobustnessTest.get_summary

    def __init__(self, results):
        self.results = results
        self.n_sources = 200
        self.inverse_method = 'sloreta'
        self.inverse_snr = 3.0


def test_get_summary_handles_categorical():
    """float(min(params)) on labels used to be a TypeError waiting to happen."""
    summary = _FakeTest({'noise_type': _categorical()}).get_summary()
    entry = summary['tests']['noise_type']

    assert entry['correlation'] is None
    assert entry['parameter_range'] is None
    # physics_valid is correlation-based, so it must abstain rather than say False
    assert entry['physics_valid'] is None
    assert entry['error_by_parameter'] == {
        'white': pytest.approx(1.1), 'spatial': pytest.approx(1.7),
        'colored': pytest.approx(2.2),
    }
    assert entry['error_range'] == (pytest.approx(1.1), pytest.approx(2.2))
    assert entry['n_trials_total'] == 9


def test_get_summary_numeric_unchanged():
    """The noise sweep must still get a real correlation and verdict."""
    entry = _FakeTest({'noise': _numeric()}).get_summary()['tests']['noise']
    assert entry['correlation'] > 0.5
    assert entry['physics_valid'] is True
    assert entry['parameter_range'] == (0.1, 10.0)


def test_get_summary_mixed_sweeps_coexist():
    """A session that ran both kinds must summarize both."""
    summary = _FakeTest({
        'noise': _numeric(), 'noise_type': _categorical()
    }).get_summary()
    assert summary['tests']['noise']['physics_valid'] is True
    assert summary['tests']['noise_type']['physics_valid'] is None
