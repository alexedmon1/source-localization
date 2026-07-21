"""Tests for the 3D activation-cloud interpolation (validation/viz3d.py).

Rendering needs a GPU context, but the field the renderer draws is plain
numerics: a Gaussian splat of scattered source values onto a regular grid. These
pin the properties the picture depends on — the cloud peaks where the activation
is, it never rings or overshoots (the failure that made a thin-plate-spline
unusable here), and two separated sources stay two lobes.
"""
import numpy as np
import pytest

pytest.importorskip('pyvista')

from source_localization.validation.viz3d import interpolate_to_grid


@pytest.fixture
def cloud():
    """
    A filled 10 mm cube of sources at near-uniform density.

    Jittered lattice rather than uniform-random on purpose: the splat is a
    weighted sum and so is mildly density-weighted, and clumps in a purely
    random cloud pull the peak toward themselves. The real source grid is
    near-uniform (median spacing ~1.25 mm), so this is the representative case.
    """
    rng = np.random.default_rng(0)
    g = np.arange(-5.0, 5.01, 1.25)
    lattice = np.array(np.meshgrid(g, g, g, indexing='ij')).reshape(3, -1).T
    return lattice + rng.normal(0, 0.15, lattice.shape)


def _value_at(grid, point):
    idx = int(np.argmin(np.linalg.norm(grid.points - np.asarray(point), axis=1)))
    return float(grid.point_data['activation'][idx])


def test_cloud_peaks_at_the_active_source(cloud):
    values = np.exp(-np.sum((cloud - np.array([2.0, 1.0, 0.0])) ** 2, axis=1) / 2)
    grid = interpolate_to_grid(cloud, values, resolution=40)
    field = grid.point_data['activation']
    brightest = grid.points[int(np.argmax(field))]
    assert np.linalg.norm(brightest - [2.0, 1.0, 0.0]) < 1.0


def test_field_is_normalized_and_non_negative(cloud):
    values = np.exp(-np.sum(cloud ** 2, axis=1) / 4)
    field = interpolate_to_grid(cloud, values, resolution=40).point_data['activation']
    assert field.min() >= 0.0
    assert field.max() == pytest.approx(1.0)


def test_splat_never_overshoots_the_data(cloud):
    """The RBF this replaced rang badly; a splat cannot exceed its inputs' shape."""
    values = np.zeros(len(cloud))
    values[np.argmin(np.linalg.norm(cloud, axis=1))] = 1.0
    grid = interpolate_to_grid(cloud, values, resolution=40)
    field = grid.point_data['activation']
    # A single positive source must produce a single decaying bump: no negative
    # lobes, and nothing bright far away from the one active source.
    assert field.min() >= 0.0
    far = np.linalg.norm(grid.points, axis=1) > 6.0
    assert field[far].max() < 0.05


def test_two_separated_sources_stay_two_lobes(cloud):
    a, b = np.array([-3.5, 0.0, 0.0]), np.array([3.5, 0.0, 0.0])
    values = (np.exp(-np.sum((cloud - a) ** 2, axis=1) / 1.5)
              + np.exp(-np.sum((cloud - b) ** 2, axis=1) / 1.5))
    grid = interpolate_to_grid(cloud, values, resolution=48)
    va, vb = _value_at(grid, a), _value_at(grid, b)
    vmid = _value_at(grid, [0.0, 0.0, 0.0])
    assert vmid < 0.6 * min(va, vb)   # a real dip between them


def test_zero_activation_gives_an_empty_cloud(cloud):
    field = interpolate_to_grid(cloud, np.zeros(len(cloud)),
                                resolution=32).point_data['activation']
    assert np.all(field == 0.0)
