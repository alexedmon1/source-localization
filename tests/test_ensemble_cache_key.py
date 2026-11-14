"""Ensemble leadfield caches are keyed on the BEM solver that computed them.

On 2026-11-13 probability-atlas's Geometry (pa/geometry.py, which still solves the BEM inner first) wrote an ensemble
cache under a key equal to this package's: the key came from the nominal operator's signature, which both share and
which carries the corrected solver's tag. locate.Geometry pointed at that cache directory would have served the faulty
leadfields as corrected ones (found by bayes-sl, 2026-11-14). The key now carries ``geometry.bem_solver``, and a
geometry that does not declare the current solver is never cached.
"""
import numpy as np
import pytest

from source_localization.bem.solution import SOLVER_TAG
from source_localization.locate import head_ensemble as lhe
from source_localization.locate.geometry import Geometry
from source_localization.locate.head_ensemble import HeadModel, ensemble_leadfields
from source_localization.validation import head_ensemble as vhe


class FakeGeometry:
    """Only what ensemble_leadfields reads. ``value`` fills the leadfields so the caller can tell which were served."""
    signature = {"bem_solver": SOLVER_TAG, "spacing_mm": 0.5}     # what the nominal operator's provenance carries
    n_channels = 4

    def __init__(self, value, bem_solver=SOLVER_TAG):
        self.value = value
        self.positions_mm = np.zeros((3, 3))
        if bem_solver is not None:
            self.bem_solver = bem_solver

    def leadfield(self, model, positions_mm):
        return np.full((len(positions_mm), self.n_channels, 3), float(self.value))


MODELS = [HeadModel(), HeadModel((0.1, 0.0, 0.0), 1.5)]


def test_geometries_declare_the_current_solver():
    assert Geometry.bem_solver == SOLVER_TAG
    assert vhe.EnsembleGeometry.bem_solver == SOLVER_TAG
    assert vhe.ensemble_leadfields is lhe.ensemble_leadfields        # one implementation, one key


def test_key_depends_on_the_solver():
    pos = np.zeros((3, 3))
    assert (lhe._signature(FakeGeometry(1), MODELS, pos)
            != lhe._signature(FakeGeometry(1, bem_solver="surfaces inner-first"), MODELS, pos))
    assert lhe._signature(FakeGeometry(1), MODELS, pos) != lhe._signature(FakeGeometry(1, None), MODELS, pos)


def test_undeclared_solver_neither_reads_nor_writes_the_cache(tmp_path):
    pa_like = FakeGeometry(7.0, bem_solver=None)                  # probability-atlas's Geometry: no bem_solver
    with pytest.warns(UserWarning, match="does not declare the current BEM solver"):
        G = ensemble_leadfields(pa_like, MODELS, cache_dir=tmp_path, verbose=False)
    assert np.all(G == 7.0)
    assert list(tmp_path.iterdir()) == []

    # a cache file planted under the key such a geometry would have had is not served to it either
    np.save(tmp_path / f"ensemble_{lhe._signature(pa_like, MODELS, pa_like.positions_mm)}.npy",
            np.full((2, 3, 4, 3), -1.0))
    with pytest.warns(UserWarning):
        assert np.all(ensemble_leadfields(pa_like, MODELS, cache_dir=tmp_path, verbose=False) == 7.0)


def test_declared_solver_round_trips_and_does_not_see_undeclared_caches(tmp_path):
    pa_like = FakeGeometry(7.0, bem_solver=None)
    np.save(tmp_path / f"ensemble_{lhe._signature(pa_like, MODELS, pa_like.positions_mm)}.npy",
            np.full((2, 3, 4, 3), -1.0))
    fixed = FakeGeometry(3.0)
    assert np.all(ensemble_leadfields(fixed, MODELS, cache_dir=tmp_path, verbose=False) == 3.0)
    assert len(list(tmp_path.glob("ensemble_*.npy"))) == 2
    fixed.value = 99.0                                            # served from its own cache, not recomputed
    assert np.all(ensemble_leadfields(fixed, MODELS, cache_dir=tmp_path, verbose=False) == 3.0)
