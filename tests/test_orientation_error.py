"""Angular orientation error.

The metric the validation harness never had: ground-truth orientation was
generated and stored by every simulation and read back by nothing.
"""

from __future__ import annotations

import numpy as np
import pytest

from source_localization.validation.metrics import (
    RANDOM_AXIS_ANGLE_DEG,
    compute_orientation_error,
    extract_dipole_orientation,
)


def test_identical_orientations_give_zero():
    assert compute_orientation_error([1, 0, 0], [1, 0, 0]) == pytest.approx(0.0)
    assert compute_orientation_error([0.3, -0.5, 0.8],
                                     [0.3, -0.5, 0.8]) == pytest.approx(0.0)


def test_polarity_is_ignored():
    """A flipped dipole with a flipped time course is the same source."""
    assert compute_orientation_error([1, 0, 0], [-1, 0, 0]) == pytest.approx(0.0)
    assert compute_orientation_error([0.3, -0.5, 0.8],
                                     [-0.3, 0.5, -0.8]) == pytest.approx(0.0)


def test_orthogonal_is_ninety_and_that_is_the_maximum():
    assert compute_orientation_error([1, 0, 0], [0, 1, 0]) == pytest.approx(90.0)
    rng = np.random.default_rng(0)
    for _ in range(200):
        a, b = rng.standard_normal(3), rng.standard_normal(3)
        assert 0.0 <= compute_orientation_error(a, b) <= 90.0 + 1e-9


def test_magnitude_does_not_matter():
    assert compute_orientation_error([2, 0, 0], [0.001, 0, 0]) == pytest.approx(0.0)


def test_radians_option():
    assert compute_orientation_error([1, 0, 0], [0, 1, 0], degrees=False) == \
        pytest.approx(np.pi / 2)


def test_zero_vector_is_nan_not_an_exception():
    """Source spaces that write nn = zeros must not crash the metric."""
    assert np.isnan(compute_orientation_error([0, 0, 0], [1, 0, 0]))
    assert np.isnan(compute_orientation_error([1, 0, 0], [0, 0, 0]))


def test_chance_level_is_one_radian():
    """For random axes |cos| is uniform on [0,1], so E[angle] = 1 rad exactly.

    This is the number an orientation result has to beat. Beating 90 degrees
    means nothing, because 90 is the maximum, not the chance level.
    """
    rng = np.random.default_rng(42)
    a = rng.standard_normal((20000, 3))
    b = rng.standard_normal((20000, 3))
    angles = [compute_orientation_error(x, y) for x, y in zip(a, b)]
    assert np.mean(angles) == pytest.approx(np.degrees(1.0), abs=0.5)
    assert RANDOM_AXIS_ANGLE_DEG == pytest.approx(np.degrees(1.0), abs=1e-3)


def test_extract_orientation_recovers_a_known_direction():
    direction = np.array([0.2, -0.9, 0.4])
    direction /= np.linalg.norm(direction)
    time_course = np.sin(np.linspace(0, 6 * np.pi, 300))
    activity = np.outer(direction, time_course)

    recovered = extract_dipole_orientation(activity)
    assert compute_orientation_error(direction, recovered) == pytest.approx(
        0.0, abs=1e-6
    )


def test_extract_orientation_is_undefined_under_fixed_orientation():
    """With one component the orientation is imposed, not measured."""
    out = extract_dipole_orientation(np.ones((1, 10)), n_comp=1)
    assert out.shape == (3,) and np.all(np.isnan(out))


def test_extract_orientation_handles_silence():
    assert np.all(np.isnan(extract_dipole_orientation(np.zeros((3, 50)))))


def test_extract_orientation_rejects_wrong_shape():
    with pytest.raises(ValueError, match="expected 3 components"):
        extract_dipole_orientation(np.zeros((2, 10)))
